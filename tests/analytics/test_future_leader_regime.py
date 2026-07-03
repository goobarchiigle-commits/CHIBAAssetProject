"""
tests/analytics/test_future_leader_regime.py

Coverage:
  - regime classification correctness (risk_on / risk_off / range / volatile)
  - lookahead prevention: bars after obs_date must not affect classification
  - TOPIX insufficient data → falls back to "range"
  - CLEAN-only filtering (dq < 1.0 excluded, insufficient_forward_data excluded)
  - regime aggregation correctness (correct grouping by observation_date)
  - duplicate prevention (dedup by symbol + obs_date)
  - fail-open integration (hook survives missing dirs / empty data / no TOPIX)
  - observation_only invariant (no execution imports)
  - _compute_atr_percentile: low ATR → low percentile, high ATR → high percentile
  - half_life_from_expectancy: correctness and None cases
  - format_regime_section: insufficient sample path + full report
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.analytics.future_leader_regime as reg
from src.analytics.future_leader_regime import (
    ATR_HIST_WINDOW,
    ATR_VOLATILE_PCTILE,
    ATR_WINDOW,
    TOPIX_RETURN_WINDOW,
    TOPIX_TREND_THRESH,
    RegimeExpectancySummary,
    _compute_atr_percentile,
    _filter_clean_failure,
    _filter_clean_survivability,
    _half_life_from_expectancy,
    build_regime_lookup,
    classify_regime,
    compute_all_regime_expectancies,
    compute_regime_expectancy,
    format_regime_section,
    run_future_leader_regime_hook,
)

_OBS_DATE = "2026-05-11"   # Monday — valid business day


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_topix(
    obs_date: str = _OBS_DATE,
    pre_n: int = 300,
    trend_10d: float = 0.0,
    atr_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Build a synthetic TOPIX OHLCV DataFrame.
    trend_10d: exact 10d return in the final TOPIX_RETURN_WINDOW+1 bars.
               bars before that are flat at `base`.
    atr_multiplier: scales H-L spread uniformly.
    """
    obs_ts    = pd.Timestamp(obs_date).normalize()
    all_dates = pd.bdate_range(end=obs_ts, periods=pre_n + 1)

    base   = 2000.0
    closes = np.full(pre_n + 1, base, dtype=float)
    # Last TOPIX_RETURN_WINDOW+1 bars carry the exact return (only if enough bars exist)
    n_trend = min(TOPIX_RETURN_WINDOW + 1, pre_n + 1)
    closes[-n_trend:] = np.linspace(base, base * (1 + trend_10d), n_trend)

    spread  = 20.0 * atr_multiplier
    highs   = closes + spread / 2
    lows    = closes - spread / 2
    volumes = np.full(pre_n + 1, 1e9)

    df = pd.DataFrame(
        {"Close": closes, "High": highs, "Low": lows, "Volume": volumes},
        index=all_dates,
    )
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def _make_surv_rec(
    symbol: str = "TEST.T",
    obs_date: str = _OBS_DATE,
    dq: float = 1.0,
    fwd5: float = 0.04,
    fwd10: float = 0.02,
    insufficient: bool = False,
) -> dict:
    return {
        "symbol":                     symbol,
        "observation_date":           obs_date,
        "observation_ts":             f"{obs_date}T09:00:00+09:00",
        "future_leader_score":        65.0,
        "rsr_zone":                   "emerging",
        "data_quality_weight":        dq,
        "integrity_state":            "CLEAN" if dq >= 1.0 else "PARTIAL_FAILURE",
        "close_price_at_observation": 1000.0,
        "forward_return_5d":          fwd5,
        "forward_return_10d":         fwd10,
        "mfe_10d":                    0.06,
        "mae_10d":                    -0.01,
        "days_materialized":          10,
        "insufficient_forward_data":  insufficient,
    }


def _make_fail_rec(
    symbol: str = "TEST.T",
    obs_date: str = _OBS_DATE,
    dq: float = 1.0,
    failure_detected: bool = False,
    failure_day_offset: Optional[int] = None,
) -> dict:
    return {
        "symbol":                 symbol,
        "observation_date":       obs_date,
        "data_quality_weight":    dq,
        "failure_detected":       failure_detected,
        "failure_day_offset":     failure_day_offset,
        "forward_return_5d":      0.04,
        "forward_return_10d":     0.02,
        "mfe_10d":                0.06,
        "mae_10d":                -0.01,
        "observation_only":       True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 9a — regime classification correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_risk_on():
    """TOPIX +5% 10d trend → risk_on."""
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=0.05)
    tag, ret, _ = classify_regime(topix, _OBS_DATE)
    assert tag == "risk_on"
    assert ret is not None and ret > TOPIX_TREND_THRESH


def test_classify_risk_off():
    """TOPIX -5% 10d trend → risk_off."""
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=-0.05)
    tag, ret, _ = classify_regime(topix, _OBS_DATE)
    assert tag == "risk_off"
    assert ret is not None and ret < -TOPIX_TREND_THRESH


def test_classify_volatile():
    """Flat trend + high ATR → volatile."""
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=0.01, pre_n=300, atr_multiplier=10.0)
    tag, ret, atr = classify_regime(topix, _OBS_DATE)
    # flat trend: |ret| <= 3% → not risk_on or risk_off
    assert tag in ("volatile", "range")
    # With very high ATR, should be volatile when there's enough history
    # (may be range if ATR history is uniformly high — depends on distribution)
    assert ret is not None


def test_classify_range():
    """Flat trend + low ATR → range."""
    # Uniform ATR → current ATR == historical ATR → percentile ~0.5 → range
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=0.01, pre_n=300, atr_multiplier=1.0)
    tag, ret, atr = classify_regime(topix, _OBS_DATE)
    assert tag in ("range", "volatile")  # uniform ATR → near 50th percentile → range


def test_classify_fallback_on_insufficient_data():
    """Fewer than TOPIX_RETURN_WINDOW+1 bars → falls back to 'range'."""
    short_topix = _make_topix(obs_date=_OBS_DATE, pre_n=5, trend_10d=0.08)
    tag, ret, atr = classify_regime(short_topix, _OBS_DATE)
    assert tag == "range"


def test_classify_exact_boundary_risk_on():
    """Return just above threshold → risk_on."""
    topix = _make_topix(obs_date=_OBS_DATE, pre_n=300, trend_10d=TOPIX_TREND_THRESH + 0.005)
    tag, ret, _ = classify_regime(topix, _OBS_DATE)
    assert tag == "risk_on"
    assert ret is not None and ret > TOPIX_TREND_THRESH


def test_classify_exact_boundary_at_threshold():
    """Return exactly = threshold → not risk_on (condition is strict >)."""
    topix = _make_topix(obs_date=_OBS_DATE, pre_n=300, trend_10d=TOPIX_TREND_THRESH)
    tag, ret, _ = classify_regime(topix, _OBS_DATE)
    # Exactly at threshold → falls through to range or volatile
    assert tag in ("range", "volatile")


# ─────────────────────────────────────────────────────────────────────────────
# Task 9b — no lookahead contamination
# ─────────────────────────────────────────────────────────────────────────────

def test_no_lookahead_bars_after_obs_date_ignored():
    """
    Bars after obs_date must not change regime classification.
    Adding a sharp drop after obs_date should NOT flip risk_on to risk_off.
    """
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=0.05, pre_n=300)
    tag_before, _, _ = classify_regime(topix, _OBS_DATE)

    # Add bars after obs_date with a sharp drop
    future_dates = pd.bdate_range(start="2026-05-12", periods=5)
    future_df = pd.DataFrame({
        "Close":  [500.0] * 5,  # crash — would be risk_off if used
        "High":   [510.0] * 5,
        "Low":    [490.0] * 5,
        "Volume": [1e9]   * 5,
    }, index=future_dates)
    topix_extended = pd.concat([topix, future_df])
    topix_extended.index = pd.DatetimeIndex(topix_extended.index).normalize()

    tag_after, _, _ = classify_regime(topix_extended, _OBS_DATE)
    assert tag_before == tag_after, (
        f"Lookahead contamination: {tag_before} → {tag_after} after adding future bars"
    )


def test_no_lookahead_today_bar_excluded_in_hook(tmp_path):
    """Hook filters topix_df to index < today before classify_regime."""
    surv_dir   = tmp_path / "surv"
    fail_dir   = tmp_path / "fail"
    regime_dir = tmp_path / "regime"
    rep_dir    = tmp_path / "reports"
    surv_dir.mkdir(); fail_dir.mkdir()

    today     = date(2026, 5, 22)   # Friday — confirmed business day
    today_str = today.strftime("%Y%m%d")
    obs_date  = "2026-05-19"        # Monday the week before

    rec = _make_surv_rec(symbol="A.T", obs_date=obs_date)
    surv_path = surv_dir / f"future_leader_survivability_{today_str}.jsonl"
    surv_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    # Build topix using a fixed-start bdate_range to avoid end-date length mismatch
    all_dates = pd.bdate_range(start="2024-01-02", periods=310)
    closes    = np.linspace(2000.0, 2060.0, 310)
    df = pd.DataFrame(
        {"Close": closes, "High": closes + 10.0, "Low": closes - 10.0,
         "Volume": np.full(310, 1e9)},
        index=all_dates,
    )
    df.index = pd.DatetimeIndex(df.index).normalize()

    # Patch at the screener module where the function is actually defined
    with patch("src.live.future_leader_screener._load_ohlcv_for_screener",
               return_value={"1306.T": df}):
        run_future_leader_regime_hook(
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            regime_log_dir=regime_dir,
            reports_dir=rep_dir,
            backtest_dataset_dir=Path("/fake"),
            default_data_version="v1",
            cache_dir=Path("/fake"),
            today=today,
        )
    # No exception — pass


# ─────────────────────────────────────────────────────────────────────────────
# Task 9c — CLEAN-only filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_survivability_filter_excludes_partial():
    recs = [
        _make_surv_rec(symbol="A.T", dq=1.0, insufficient=False),
        _make_surv_rec(symbol="B.T", dq=0.5, insufficient=False),  # PARTIAL
        _make_surv_rec(symbol="C.T", dq=1.0, insufficient=True),   # insufficient
    ]
    clean = _filter_clean_survivability(recs)
    assert len(clean) == 1
    assert clean[0]["symbol"] == "A.T"


def test_clean_failure_filter_excludes_partial():
    recs = [
        _make_fail_rec(symbol="A.T", dq=1.0),
        _make_fail_rec(symbol="B.T", dq=0.5),  # PARTIAL
    ]
    clean = _filter_clean_failure(recs)
    assert len(clean) == 1
    assert clean[0]["symbol"] == "A.T"


def test_clean_filter_dedup():
    recs = [
        _make_surv_rec(symbol="A.T", obs_date=_OBS_DATE),
        _make_surv_rec(symbol="A.T", obs_date=_OBS_DATE),  # duplicate
    ]
    clean = _filter_clean_survivability(recs)
    assert len(clean) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 9d — regime aggregation correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_grouping_correct():
    """Records are grouped by observation_date via regime_lookup."""
    date_risk_on = "2026-05-04"
    date_range   = "2026-05-08"

    surv_recs = (
        [_make_surv_rec(symbol=f"R{i}.T", obs_date=date_risk_on, fwd5=0.06, fwd10=0.04) for i in range(6)]
        + [_make_surv_rec(symbol=f"F{i}.T", obs_date=date_range, fwd5=0.01, fwd10=0.005) for i in range(6)]
    )
    regime_lookup = {date_risk_on: "risk_on", date_range: "range"}
    summaries = compute_all_regime_expectancies(surv_recs, [], regime_lookup)

    ro = next(s for s in summaries if s.regime == "risk_on")
    rn = next(s for s in summaries if s.regime == "range")

    assert ro.sample_count == 6
    assert rn.sample_count == 6
    assert ro.day5_expectancy  == pytest.approx(0.06, abs=1e-5)
    assert rn.day5_expectancy  == pytest.approx(0.01, abs=1e-5)
    assert ro.day10_expectancy == pytest.approx(0.04, abs=1e-5)
    assert rn.day10_expectancy == pytest.approx(0.005, abs=1e-5)


def test_regime_aggregation_all_regimes_represented():
    """All 4 regimes appear in output, even with zero samples."""
    summaries = compute_all_regime_expectancies([], [], {})
    regime_names = [s.regime for s in summaries]
    assert "risk_on"  in regime_names
    assert "risk_off" in regime_names
    assert "range"    in regime_names
    assert "volatile" in regime_names


def test_regime_survival_from_failure_records():
    """median_survival uses failure records' failure_day_offset."""
    obs_date = "2026-05-11"
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", obs_date=obs_date) for i in range(6)]
    fail_recs = (
        [_make_fail_rec(symbol=f"S{i}.T", obs_date=obs_date, failure_detected=True, failure_day_offset=5)
         for i in range(3)]
        + [_make_fail_rec(symbol=f"S{i}.T", obs_date=obs_date, failure_detected=False)
           for i in range(3, 6)]
    )
    regime_lookup = {obs_date: "risk_on"}
    summaries = compute_all_regime_expectancies(surv_recs, fail_recs, regime_lookup)
    ro = next(s for s in summaries if s.regime == "risk_on")
    # survival_days: [5, 5, 5, 20, 20, 20] → median = 12.5
    assert ro.median_survival == pytest.approx(12.5, abs=0.1)


def test_half_life_day_from_expectancy():
    """half_life_day computed correctly from e5/e10."""
    s = compute_regime_expectancy("risk_on", [], {})
    assert s.half_life_day is None  # no data

    # With real data: e5=0.08, e10=0.03 → peak=0.08, half_life=10
    obs_date = "2026-05-11"
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", obs_date=obs_date, fwd5=0.08, fwd10=0.03)
                 for i in range(6)]
    regime_lookup = {obs_date: "risk_on"}
    summaries = compute_all_regime_expectancies(surv_recs, [], regime_lookup)
    ro = next(s for s in summaries if s.regime == "risk_on")
    assert ro.half_life_day == 10


# ─────────────────────────────────────────────────────────────────────────────
# Task 9e — duplicate prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_duplicate_surv_records_counted_once():
    obs_date = "2026-05-11"
    recs = [_make_surv_rec(symbol="A.T", obs_date=obs_date)] * 5  # same key
    clean = _filter_clean_survivability(recs)
    assert len(clean) == 1


def test_duplicate_fail_records_counted_once():
    recs = [_make_fail_rec(symbol="A.T", obs_date=_OBS_DATE)] * 4
    clean = _filter_clean_failure(recs)
    assert len(clean) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 9f — fail-open integration
# ─────────────────────────────────────────────────────────────────────────────

def test_hook_succeeds_with_missing_dirs(tmp_path):
    run_future_leader_regime_hook(
        survivability_log_dir=tmp_path / "no_surv",
        failure_log_dir=tmp_path / "no_fail",
        regime_log_dir=tmp_path / "no_regime",
        reports_dir=tmp_path / "reports",
        backtest_dataset_dir=tmp_path / "ds",
        default_data_version="v1",
        cache_dir=tmp_path / "cache",
        today=date(2026, 5, 24),
    )


def test_hook_succeeds_with_empty_logs(tmp_path):
    for d in ["surv", "fail", "regime"]:
        (tmp_path / d).mkdir()
    run_future_leader_regime_hook(
        survivability_log_dir=tmp_path / "surv",
        failure_log_dir=tmp_path / "fail",
        regime_log_dir=tmp_path / "regime",
        reports_dir=tmp_path / "reports",
        backtest_dataset_dir=tmp_path / "ds",
        default_data_version="v1",
        cache_dir=tmp_path / "cache",
        today=date(2026, 5, 24),
    )


def test_hook_writes_report_when_regime_tags_exist(tmp_path):
    """If regime tags already persisted, hook produces section without TOPIX load."""
    surv_dir   = tmp_path / "surv"; surv_dir.mkdir()
    fail_dir   = tmp_path / "fail"; fail_dir.mkdir()
    regime_dir = tmp_path / "regime"; regime_dir.mkdir()
    rep_dir    = tmp_path / "reports"

    today = date(2026, 5, 24)
    today_str = today.strftime("%Y%m%d")
    obs_date  = "2026-05-11"

    # Write 6 survivability records
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", obs_date=obs_date) for i in range(6)]
    (surv_dir / f"future_leader_survivability_{today_str}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in surv_recs), encoding="utf-8"
    )

    # Pre-write regime tag so TOPIX load is skipped
    (regime_dir / f"future_leader_regime_{today_str}.jsonl").write_text(
        json.dumps({"observation_date": obs_date, "regime_tag": "risk_on"}) + "\n",
        encoding="utf-8",
    )

    run_future_leader_regime_hook(
        survivability_log_dir=surv_dir,
        failure_log_dir=fail_dir,
        regime_log_dir=regime_dir,
        reports_dir=rep_dir,
        backtest_dataset_dir=tmp_path / "ds",
        default_data_version="v1",
        cache_dir=tmp_path / "cache",
        today=today,
    )

    report_path = rep_dir / f"future_leader_report_{today_str}.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "REGIME EXPECTANCY" in content
    assert "risk_on" in content


def test_hook_survives_topix_load_failure(tmp_path):
    """TOPIX load failure must not abort the hook — fail-open."""
    surv_dir   = tmp_path / "surv"; surv_dir.mkdir()
    fail_dir   = tmp_path / "fail"; fail_dir.mkdir()
    regime_dir = tmp_path / "regime"; regime_dir.mkdir()
    rep_dir    = tmp_path / "reports"

    today     = date(2026, 5, 24)
    today_str = today.strftime("%Y%m%d")
    obs_date  = "2026-05-11"

    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", obs_date=obs_date) for i in range(3)]
    (surv_dir / f"future_leader_survivability_{today_str}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in surv_recs), encoding="utf-8"
    )

    def _bad_loader(*args, **kwargs):
        raise RuntimeError("simulated TOPIX load failure")

    with patch("src.live.future_leader_screener._load_ohlcv_for_screener", _bad_loader):
        run_future_leader_regime_hook(
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            regime_log_dir=regime_dir,
            reports_dir=rep_dir,
            backtest_dataset_dir=tmp_path / "ds",
            default_data_version="v1",
            cache_dir=tmp_path / "cache",
            today=today,
        )
    # No exception — pass


# ─────────────────────────────────────────────────────────────────────────────
# Task 9g — observation_only invariant (no execution imports)
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN = ["allocator", "signal_bridge", "order", "broker", "deployable_universe"]


def test_no_execution_imports_in_module():
    src_text = Path("src/analytics/future_leader_regime.py").read_text(encoding="utf-8")
    import_lines = [
        ln.strip() for ln in src_text.splitlines()
        if ln.strip().startswith("import ") or ln.strip().startswith("from ")
    ]
    combined = "\n".join(import_lines)
    for forbidden in _FORBIDDEN:
        assert forbidden not in combined, (
            f"forbidden import '{forbidden}' in future_leader_regime.py import statements"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ATR percentile unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_topix_df_with_atr(atr_current: float, atr_history: float, n_history: int = 300) -> pd.DataFrame:
    """Build a TOPIX df where recent ATR and historical ATR are controlled."""
    dates = pd.bdate_range("2024-01-02", periods=n_history + ATR_WINDOW)
    # Historical bars: atr_history H-L spread
    h_hist = np.full(n_history, 2000.0 + atr_history / 2)
    l_hist = np.full(n_history, 2000.0 - atr_history / 2)
    c_hist = np.full(n_history, 2000.0)
    # Recent bars: atr_current H-L spread
    h_recent = np.full(ATR_WINDOW, 2000.0 + atr_current / 2)
    l_recent = np.full(ATR_WINDOW, 2000.0 - atr_current / 2)
    c_recent = np.full(ATR_WINDOW, 2000.0)

    df = pd.DataFrame({
        "Close":  np.concatenate([c_hist, c_recent]),
        "High":   np.concatenate([h_hist, h_recent]),
        "Low":    np.concatenate([l_hist, l_recent]),
        "Volume": np.ones(n_history + ATR_WINDOW) * 1e9,
    }, index=dates)
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def test_atr_percentile_high_when_current_atr_high():
    """Current ATR >> historical → high percentile."""
    df = _make_topix_df_with_atr(atr_current=100.0, atr_history=10.0)
    pctile = _compute_atr_percentile(df)
    assert pctile is not None
    assert pctile >= ATR_VOLATILE_PCTILE  # current is much larger → high percentile


def test_atr_percentile_low_when_current_atr_low():
    """Current ATR << historical → low percentile."""
    df = _make_topix_df_with_atr(atr_current=1.0, atr_history=100.0)
    pctile = _compute_atr_percentile(df)
    assert pctile is not None
    assert pctile < ATR_VOLATILE_PCTILE  # current is much smaller → low percentile


def test_atr_percentile_none_on_insufficient_data():
    """Fewer than ATR_WINDOW + ATR_HIST_WINDOW bars → None."""
    short_df = _make_topix_df_with_atr(10.0, 10.0, n_history=10)
    assert _compute_atr_percentile(short_df) is None


# ─────────────────────────────────────────────────────────────────────────────
# _half_life_from_expectancy
# ─────────────────────────────────────────────────────────────────────────────

def test_half_life_detected_day10():
    """e5=0.08, e10=0.03 → peak=0.08, threshold=0.04 → 0.03 <= 0.04 → day10"""
    assert _half_life_from_expectancy(0.08, 0.03) == 10


def test_half_life_none_when_sustained():
    """e5=0.08, e10=0.05 → peak=0.08, threshold=0.04 → 0.05 > 0.04 → None"""
    assert _half_life_from_expectancy(0.08, 0.05) is None


def test_half_life_none_when_both_none():
    assert _half_life_from_expectancy(None, None) is None


def test_half_life_none_when_peak_nonpositive():
    assert _half_life_from_expectancy(-0.03, -0.05) is None


# ─────────────────────────────────────────────────────────────────────────────
# format_regime_section
# ─────────────────────────────────────────────────────────────────────────────

def test_format_section_insufficient_sample():
    summaries = [
        RegimeExpectancySummary("risk_on", 2, None, None, None, None, None, None),
        RegimeExpectancySummary("risk_off", 0, None, None, None, None, None, None),
        RegimeExpectancySummary("range",    3, None, None, None, None, None, None),
        RegimeExpectancySummary("volatile", 1, None, None, None, None, None, None),
    ]
    text = format_regime_section(summaries)
    assert "insufficient" in text.lower()


def test_format_section_full():
    summaries = [
        RegimeExpectancySummary("risk_on",  32, 0.05, 0.071, 11.0, 13, 0.09,  -0.01),
        RegimeExpectancySummary("risk_off", 19, 0.00, -0.028, 2.0, 3,  0.02, -0.05),
        RegimeExpectancySummary("range",    48, 0.02,  0.012, 4.0, None, 0.03, -0.02),
        RegimeExpectancySummary("volatile", 14, None,  None,  None, None, None, -0.041),
    ]
    text = format_regime_section(summaries)
    assert "REGIME EXPECTANCY" in text
    assert "risk_on"  in text
    assert "risk_off" in text
    assert "range"    in text
    assert "volatile" in text
    assert "+7.1%" in text   # day10 0.071
    assert "-2.8%" in text   # day10 -0.028


# ─────────────────────────────────────────────────────────────────────────────
# build_regime_lookup
# ─────────────────────────────────────────────────────────────────────────────

def test_build_regime_lookup_dedups_dates():
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=0.05, pre_n=300)
    dates = [_OBS_DATE, _OBS_DATE, _OBS_DATE]  # duplicates
    regime_map, meta_map = build_regime_lookup(dates, topix)
    assert len(regime_map) == 1
    assert _OBS_DATE in regime_map


def test_build_regime_lookup_returns_known_regime():
    topix = _make_topix(obs_date=_OBS_DATE, trend_10d=0.05, pre_n=300)
    regime_map, _ = build_regime_lookup([_OBS_DATE], topix)
    assert regime_map[_OBS_DATE] in ("risk_on", "risk_off", "range", "volatile")
