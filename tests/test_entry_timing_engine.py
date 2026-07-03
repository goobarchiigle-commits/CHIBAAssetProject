"""
tests/test_entry_timing_engine.py — Entry Timing Engine

Coverage:
  - Weight constants sum to 1.0
  - _score_breakout_quality: pre-computed, proxy fallback, phase modifier
  - _score_pullback_quality: ideal / extended / deep / below MA20
  - _score_trend_persistence: rising / flat / declining RSR, SEPA levels
  - _score_market_context: cluster levels, breadth variations
  - _compute_bonuses_penalties: bonuses, penalties, combinations
  - compute_entry_timing: golden path, FAIL_OPEN on exception, clamped output
  - Confidence/action classification thresholds
  - build_entry_timing_input_from_df: normal, short df
  - compute_entry_timing_for_candidates: feature flag disabled, per-symbol
  - apply_entry_timing_boost: range, disabled, None result
  - append_et_telemetry: writes valid JSONL
  - materialize_et_returns: materializes old records, skips recent
  - compute_et_kpis: decile stats, confidence stats, monotonicity
  - format_et_report_section: normal, empty, KPI section
  - _score_to_decile: boundary mapping
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.entry.entry_timing_engine import (
    ACTION_IMMEDIATE,
    ACTION_NORMAL,
    ACTION_WATCH,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    W_BREAKOUT,
    W_PULLBACK,
    W_TREND,
    W_MARKET,
    EntryTimingInput,
    EntryTimingResult,
    _clamp,
    _compute_atr20,
    _compute_ma,
    _compute_volume_ratio_5d,
    _score_breakout_quality,
    _score_pullback_quality,
    _score_trend_persistence,
    _score_market_context,
    _compute_bonuses_penalties,
    _score_to_decile,
    compute_entry_timing,
    build_entry_timing_input_from_df,
    compute_entry_timing_for_candidates,
    apply_entry_timing_boost,
    append_et_telemetry,
    materialize_et_returns,
    compute_et_kpis,
    format_et_report_section,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_closes(n: int = 40, base: float = 1000.0, trend: float = 2.0) -> List[float]:
    return [base + i * trend for i in range(n)]


def _make_ohlcv(n: int = 40, base: float = 1000.0) -> dict:
    closes  = _make_closes(n, base)
    highs   = [c * 1.01 for c in closes]
    lows    = [c * 0.99 for c in closes]
    volumes = [1_000_000.0] * n
    return {"closes": closes, "highs": highs, "lows": lows, "volumes": volumes}


def _make_input(
    symbol: str = "9501",
    rsr: float = 80.0,
    rsr_momentum: float = 3.0,
    sepa_score: int = 6,
    n: int = 40,
    base: float = 1000.0,
    trend_cluster_level: int = 0,
    rsr_series: Optional[List[float]] = None,
    bq_score: Optional[float] = None,
    bq_phase: Optional[str] = None,
) -> EntryTimingInput:
    ohlcv = _make_ohlcv(n, base)
    return EntryTimingInput(
        symbol               = symbol,
        rsr                  = rsr,
        rsr_momentum         = rsr_momentum,
        sepa_score           = sepa_score,
        closes               = ohlcv["closes"],
        highs                = ohlcv["highs"],
        lows                 = ohlcv["lows"],
        volumes              = ohlcv["volumes"],
        rsr_series           = rsr_series or [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],
        trend_cluster_level  = trend_cluster_level,
        universe_rsr_values  = [75.0] * 30 + [60.0] * 12,
        breakout_quality_score = bq_score,
        breakout_phase         = bq_phase,
    )


def _make_df(n: int = 40, base: float = 1000.0) -> pd.DataFrame:
    closes  = [base + i * 2.0 for i in range(n)]
    df = pd.DataFrame({
        "Close":  closes,
        "High":   [c * 1.01 for c in closes],
        "Low":    [c * 0.99 for c in closes],
        "Volume": [1_000_000] * n,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    total = W_BREAKOUT + W_PULLBACK + W_TREND + W_MARKET
    assert abs(total - 1.0) < 1e-9


def test_confidence_thresholds_ordered():
    assert CONFIDENCE_HIGH_THRESHOLD > CONFIDENCE_MEDIUM_THRESHOLD >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_clamp_bounds():
    assert _clamp(-10.0) == 0.0
    assert _clamp(110.0) == 100.0
    assert _clamp(50.0)  == 50.0


def test_compute_atr20_basic():
    highs  = [105.0] * 25
    lows   = [95.0]  * 25
    closes = [100.0] * 25
    atr = _compute_atr20(highs, lows, closes)
    assert 0 < atr < 20.0


def test_compute_atr20_short_fallback():
    closes = [100.0]
    atr = _compute_atr20([100.0], [99.0], closes)
    assert atr == pytest.approx(2.0, rel=0.01)


def test_compute_ma_basic():
    series = list(range(1, 21))   # 1..20, mean of last 5 = 18
    assert _compute_ma(series, 5) == pytest.approx(18.0)


def test_compute_ma_insufficient():
    assert _compute_ma([1.0, 2.0], 5) is None


def test_compute_volume_ratio_5d_normal():
    vols = [1_000_000.0] * 20 + [2_000_000.0]
    r = _compute_volume_ratio_5d(vols)
    assert r == pytest.approx(2.0, rel=0.05)


def test_score_to_decile_bounds():
    assert _score_to_decile(0.0)   == 1
    assert _score_to_decile(99.9)  == 10
    assert _score_to_decile(100.0) == 10
    assert _score_to_decile(50.0)  == 6


# ─────────────────────────────────────────────────────────────────────────────
# 3. Breakout quality component
# ─────────────────────────────────────────────────────────────────────────────

def test_bq_component_uses_precomputed():
    inp = _make_input(bq_score=85.0, bq_phase="healthy_breakout")
    score, phase = _score_breakout_quality(inp)
    assert score == pytest.approx(85.0)
    assert phase == "healthy_breakout"


def test_bq_component_proxy_high_cpr():
    # Close near top of range → high CPR → high score
    # CPR = (close - low) / (high - low) → close=108, low=90, high=110 → CPR=0.90
    # Needs >=2 closes for proxy computation
    inp = _make_input()
    inp.closes = [105.0, 108.0]   # 2 closes required
    inp.highs  = [106.0, 110.0]
    inp.lows   = [103.0, 90.0]
    inp.volumes = [1_200_000.0] * 21  # vol ratio ~1.2
    score, _ = _score_breakout_quality(inp)
    assert score > 50.0   # close near top of range → high CPR


def test_bq_component_failed_phase_passed_through():
    inp = _make_input(bq_score=40.0, bq_phase="failed_breakout")
    score, phase = _score_breakout_quality(inp)
    assert phase == "failed_breakout"
    assert score == pytest.approx(40.0)


def test_bq_component_fallback_on_empty():
    inp = _make_input()
    inp.closes = []
    inp.highs  = []
    inp.lows   = []
    score, phase = _score_breakout_quality(inp)
    assert score == pytest.approx(50.0)
    assert phase == "weak_breakout"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pullback quality component
# ─────────────────────────────────────────────────────────────────────────────

def test_pullback_ideal_depth():
    # close at 1 ATR below 20d high → ideal
    ohlcv = _make_ohlcv(25, base=1000.0)
    # Set 20d high to 1100, close to 1000 (100 = 1 ATR of ~10)
    ohlcv["closes"][-1] = 990.0
    ohlcv["highs"][-1]  = 1000.0
    # Set a spike high in the window
    ohlcv["highs"][-10] = 1010.0
    inp = _make_input()
    inp.closes = ohlcv["closes"]
    inp.highs  = ohlcv["highs"]
    inp.lows   = ohlcv["lows"]
    score = _score_pullback_quality(inp)
    assert 0 <= score <= 100


def test_pullback_below_ma20_penalized():
    closes = [1000.0] * 19 + [900.0]   # drops below 20d MA
    inp = _make_input()
    inp.closes = closes
    inp.highs  = [c * 1.01 for c in closes]
    inp.lows   = [c * 0.99 for c in closes]
    score = _score_pullback_quality(inp)
    # MA20 ≈ 995; close=900 is well below → ema_score=20
    assert score < 60.0


def test_pullback_insufficient_data():
    inp = _make_input()
    inp.closes = [100.0] * 5  # only 5 bars
    inp.highs  = [101.0] * 5
    score = _score_pullback_quality(inp)
    assert score == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Trend persistence component
# ─────────────────────────────────────────────────────────────────────────────

def test_trend_all_rising_rsr():
    inp = _make_input(rsr=82.0, sepa_score=7,
                      rsr_series=[70, 72, 74, 76, 78, 80, 82])
    score = _score_trend_persistence(inp)
    assert score >= 70.0   # high RSR + all rising + good SEPA


def test_trend_declining_rsr_penalized():
    inp = _make_input(rsr=77.0, sepa_score=5,
                      rsr_series=[85, 82, 80, 77, 75, 73])
    score = _score_trend_persistence(inp)
    # Declining RSR series → cont_score = 20
    assert score < 65.0


def test_trend_high_rsr_level():
    inp = _make_input(rsr=90.0, sepa_score=8,
                      rsr_series=[88, 89, 90, 91, 90, 90])
    score = _score_trend_persistence(inp)
    assert score >= 75.0


def test_trend_sepa_zero():
    inp = _make_input(rsr=75.0, sepa_score=0)
    score = _score_trend_persistence(inp)
    # sepa_score 0 → 0pts for sepa component → lower overall
    assert score < 75.0


def test_trend_no_rsr_series_neutral():
    inp = _make_input(rsr=76.0, sepa_score=5)
    inp.rsr_series = None
    score = _score_trend_persistence(inp)
    assert 40.0 <= score <= 90.0   # neutral cont_score=60


# ─────────────────────────────────────────────────────────────────────────────
# 6. Market context component
# ─────────────────────────────────────────────────────────────────────────────

def test_market_strong_cluster():
    inp = _make_input(trend_cluster_level=2)
    score = _score_market_context(inp)
    assert score >= 70.0


def test_market_neutral_cluster():
    inp = _make_input(trend_cluster_level=0)
    score = _score_market_context(inp)
    # regime_score=60, breadth will depend on universe_rsr_values
    assert 40.0 <= score <= 80.0


def test_market_low_breadth():
    inp = _make_input(trend_cluster_level=0)
    inp.universe_rsr_values = [50.0] * 42   # all below 65
    score = _score_market_context(inp)
    # breadth=0 → breadth_score=30
    assert score < 60.0


def test_market_no_universe_rsr():
    inp = _make_input(trend_cluster_level=1)
    inp.universe_rsr_values = None
    score = _score_market_context(inp)
    # Falls back to neutral breadth_score=60
    assert score > 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bonuses and penalties
# ─────────────────────────────────────────────────────────────────────────────

def test_bonus_healthy_plus_volume():
    inp = _make_input(bq_score=85.0, bq_phase="healthy_breakout",
                      rsr=82.0)
    # vol ratio > 1.5
    inp.volumes = [1_000_000.0] * 20 + [2_000_000.0]
    delta, bonuses, _ = _compute_bonuses_penalties(
        inp, "healthy_breakout", 2.0, False
    )
    assert any("healthy_breakout+volume" in b for b in bonuses)
    assert delta >= 10.0


def test_bonus_rsr_high_rising():
    inp = _make_input(rsr=88.0)
    delta, bonuses, _ = _compute_bonuses_penalties(
        inp, "weak_breakout", 1.0, True   # all_rising=True
    )
    assert any("rsr_high_rising" in b for b in bonuses)
    assert delta >= 7.0


def test_penalty_failed_breakout():
    inp = _make_input()
    delta, _, penalties = _compute_bonuses_penalties(
        inp, "failed_breakout", 1.0, False
    )
    assert any("failed_breakout" in p for p in penalties)
    assert delta <= -15.0


def test_penalty_overextended():
    # close >> MA20
    closes = [1000.0] * 19 + [1250.0]   # 25% above 20d MA
    inp = _make_input()
    inp.closes = closes
    inp.highs  = [c * 1.01 for c in closes]
    inp.lows   = [c * 0.99 for c in closes]
    delta, _, penalties = _compute_bonuses_penalties(
        inp, "weak_breakout", 1.0, False
    )
    assert any("overextended" in p for p in penalties)


def test_penalty_low_volume():
    inp = _make_input()
    delta, _, penalties = _compute_bonuses_penalties(
        inp, "weak_breakout", 0.3, False   # vol_ratio < 0.5
    )
    assert any("low_volume" in p for p in penalties)


# ─────────────────────────────────────────────────────────────────────────────
# 8. compute_entry_timing — integration
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_entry_timing_golden_path():
    inp = _make_input(rsr=82.0, sepa_score=7,
                      bq_score=80.0, bq_phase="healthy_breakout",
                      trend_cluster_level=1)
    result = compute_entry_timing(inp)
    assert isinstance(result, EntryTimingResult)
    assert 0.0 <= result.score <= 100.0
    assert result.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)
    assert result.action in (ACTION_IMMEDIATE, ACTION_NORMAL, ACTION_WATCH)
    assert result.symbol == "9501"


def test_compute_entry_timing_score_bounds():
    for _ in range(5):
        inp = _make_input(rsr=75.0 + _ * 3, sepa_score=_)
        result = compute_entry_timing(inp)
        assert 0.0 <= result.score <= 100.0


def test_compute_entry_timing_high_confidence():
    inp = _make_input(
        rsr=90.0, sepa_score=8,
        bq_score=90.0, bq_phase="healthy_breakout",
        trend_cluster_level=2,
        rsr_series=[80, 82, 84, 86, 88, 90],
    )
    inp.volumes = [1_000_000.0] * 20 + [2_000_000.0]
    result = compute_entry_timing(inp)
    # Strong entry signals should yield HIGH or at minimum MEDIUM
    assert result.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)


def test_compute_entry_timing_low_confidence():
    inp = _make_input(
        rsr=75.0, sepa_score=2,
        bq_score=30.0, bq_phase="failed_breakout",
        trend_cluster_level=0,
        rsr_series=[85, 82, 80, 77, 75, 73],  # declining
    )
    inp.universe_rsr_values = [50.0] * 42
    result = compute_entry_timing(inp)
    assert result.confidence in (CONFIDENCE_MEDIUM, CONFIDENCE_LOW)


def test_compute_entry_timing_fail_open():
    # Pass a broken input — should return neutral, never raise
    broken = EntryTimingInput(
        symbol="ERR",
        rsr=float("nan"),
        rsr_momentum=float("inf"),
        sepa_score=-1,
        closes=[],
        highs=[],
        lows=[],
        volumes=[],
    )
    result = compute_entry_timing(broken)
    assert isinstance(result, EntryTimingResult)
    assert 0.0 <= result.score <= 100.0


def test_confidence_action_consistency():
    for conf, expected_action in [
        (CONFIDENCE_HIGH,   ACTION_IMMEDIATE),
        (CONFIDENCE_MEDIUM, ACTION_NORMAL),
        (CONFIDENCE_LOW,    ACTION_WATCH),
    ]:
        inp = _make_input()
        result = compute_entry_timing(inp)
        # Re-check the mapping without asserting specific confidence
        # (just validate action matches confidence on the returned result)
        action_map = {
            CONFIDENCE_HIGH:   ACTION_IMMEDIATE,
            CONFIDENCE_MEDIUM: ACTION_NORMAL,
            CONFIDENCE_LOW:    ACTION_WATCH,
        }
        assert result.action == action_map[result.confidence]


# ─────────────────────────────────────────────────────────────────────────────
# 9. build_entry_timing_input_from_df
# ─────────────────────────────────────────────────────────────────────────────

def test_build_input_from_df_normal():
    df = _make_df(40)
    inp = build_entry_timing_input_from_df(
        symbol="1234",
        df=df,
        rsr=78.0,
        rsr_momentum=2.5,
        sepa_score=5,
    )
    assert inp.symbol == "1234"
    assert inp.rsr == pytest.approx(78.0)
    assert len(inp.closes) > 0
    assert len(inp.highs) == len(inp.closes)


def test_build_input_from_df_short():
    df = _make_df(5)
    inp = build_entry_timing_input_from_df("9999", df, 75.0, 1.0, 4)
    # Should not raise; closes will be short
    assert len(inp.closes) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 10. compute_entry_timing_for_candidates
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_signal(symbol: str, rsr: float = 78.0) -> MagicMock:
    sig = MagicMock()
    sig.symbol      = symbol
    sig.rsr         = rsr
    sig.rsr_mom     = 2.0
    sig.sepa_score  = 6
    return sig


def test_for_candidates_feature_flag_disabled():
    results = compute_entry_timing_for_candidates(
        buy_eligible=[(80.0, "1234"), (78.0, "5678")],
        universe_raw={},
        rsr_universe=pd.DataFrame(),
        signals_map={},
        enabled=False,
    )
    assert len(results) == 2
    for r in results.values():
        assert r.score == pytest.approx(50.0)
        assert r.confidence == CONFIDENCE_MEDIUM


def test_for_candidates_missing_universe_raw():
    results = compute_entry_timing_for_candidates(
        buy_eligible=[(80.0, "1234")],
        universe_raw={},            # symbol not present
        rsr_universe=pd.DataFrame(),
        signals_map={"1234": _make_mock_signal("1234")},
        enabled=True,
    )
    assert "1234" in results
    assert results["1234"].score == pytest.approx(50.0)  # neutral fallback


def test_for_candidates_with_data():
    df = _make_df(40)
    rsr_df = pd.DataFrame({"1234": [70.0 + i * 0.5 for i in range(30)]})
    results = compute_entry_timing_for_candidates(
        buy_eligible=[(80.0, "1234")],
        universe_raw={"1234": {"df": df}},
        rsr_universe=rsr_df,
        signals_map={"1234": _make_mock_signal("1234", rsr=82.0)},
        trend_cluster_level=1,
        enabled=True,
    )
    assert "1234" in results
    r = results["1234"]
    assert 0.0 <= r.score <= 100.0
    assert isinstance(r.confidence, str)


# ─────────────────────────────────────────────────────────────────────────────
# 11. apply_entry_timing_boost
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(score: float, conf: str = CONFIDENCE_MEDIUM) -> EntryTimingResult:
    return EntryTimingResult(
        symbol="T", score=score, confidence=conf, action=ACTION_NORMAL,
        breakout_component=50.0, pullback_component=50.0,
        trend_component=50.0, market_component=60.0,
    )


def test_apply_boost_neutral_at_50():
    base = 85.0
    r = _make_result(50.0)
    assert apply_entry_timing_boost(base, r, 0.06) == pytest.approx(base)


def test_apply_boost_positive_above_50():
    base = 85.0
    r = _make_result(100.0)
    boosted = apply_entry_timing_boost(base, r, 0.06)
    assert boosted > base
    assert boosted == pytest.approx(base + 3.0)


def test_apply_boost_negative_below_50():
    base = 85.0
    r = _make_result(0.0)
    boosted = apply_entry_timing_boost(base, r, 0.06)
    assert boosted < base
    assert boosted == pytest.approx(base - 3.0)


def test_apply_boost_disabled():
    base = 85.0
    r = _make_result(100.0)
    assert apply_entry_timing_boost(base, r, 0.06, enabled=False) == pytest.approx(base)


def test_apply_boost_none_result():
    assert apply_entry_timing_boost(80.0, None, 0.06) == pytest.approx(80.0)


# ─────────────────────────────────────────────────────────────────────────────
# 12. append_et_telemetry
# ─────────────────────────────────────────────────────────────────────────────

def test_append_et_telemetry_writes_jsonl(tmp_path):
    path = tmp_path / "et.jsonl"
    r = _make_result(72.0, CONFIDENCE_HIGH)
    r.symbol = "9501"
    r.action = ACTION_IMMEDIATE
    r.phase  = "breakout"
    append_et_telemetry(r, "ENTERED", path, "2026-01-15")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["symbol"] == "9501"
    assert rec["entry_timing_score"] == pytest.approx(72.0)
    assert rec["confidence"] == CONFIDENCE_HIGH
    assert rec["action_taken"] == "ENTERED"
    assert rec["materialized_3d"] is False
    assert rec["score_decile"] >= 7   # score=72 → decile 8


def test_append_et_telemetry_multiple_entries(tmp_path):
    path = tmp_path / "et.jsonl"
    for i in range(3):
        r = _make_result(50.0 + i * 10)
        r.symbol = f"000{i}"
        r.action = ACTION_NORMAL
        r.phase  = "normal"
        append_et_telemetry(r, "WATCHED", path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 13. materialize_et_returns
# ─────────────────────────────────────────────────────────────────────────────

def _write_telemetry(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_materialize_old_record(tmp_path):
    path = tmp_path / "et.jsonl"
    old_date = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    _write_telemetry(path, [{
        "symbol": "1234", "date": old_date,
        "entry_timing_score": 72.0, "confidence": CONFIDENCE_HIGH,
        "action": ACTION_IMMEDIATE, "action_taken": "ENTERED",
        "phase": "breakout", "breakout_component": 80.0, "pullback_component": 70.0,
        "trend_component": 75.0, "market_component": 65.0,
        "bonuses": [], "penalties": [], "score_decile": 8,
        "subsequent_3d_return": None, "subsequent_5d_return": None,
        "win_3d": None, "materialized_3d": False, "materialized_5d": False,
        "schema_version": "v1",
    }])

    close_series = [100.0 + i for i in range(20)]
    idx = pd.date_range(start=old_date, periods=20, freq="B")
    df_ohlcv = pd.DataFrame({"Close": close_series}, index=idx)

    stats = materialize_et_returns(path, lambda s: df_ohlcv)
    assert stats.get("n_materialized_3d", 0) >= 1


def test_materialize_skips_recent(tmp_path):
    path = tmp_path / "et.jsonl"
    today_str = date.today().strftime("%Y-%m-%d")
    _write_telemetry(path, [{
        "symbol": "1234", "date": today_str,
        "entry_timing_score": 72.0, "confidence": CONFIDENCE_HIGH,
        "action": ACTION_IMMEDIATE, "action_taken": "ENTERED",
        "phase": "normal", "breakout_component": 50.0, "pullback_component": 50.0,
        "trend_component": 50.0, "market_component": 60.0,
        "bonuses": [], "penalties": [], "score_decile": 8,
        "subsequent_3d_return": None, "subsequent_5d_return": None,
        "win_3d": None, "materialized_3d": False, "materialized_5d": False,
        "schema_version": "v1",
    }])
    stats = materialize_et_returns(path, lambda s: None)
    assert stats.get("n_materialized_3d", 0) == 0


def test_materialize_empty_file(tmp_path):
    path = tmp_path / "et.jsonl"
    path.write_text("")
    stats = materialize_et_returns(path, lambda s: None)
    assert stats == {}


def test_materialize_nonexistent_file(tmp_path):
    path = tmp_path / "no_file.jsonl"
    stats = materialize_et_returns(path, lambda s: None)
    assert stats == {}


# ─────────────────────────────────────────────────────────────────────────────
# 14. compute_et_kpis
# ─────────────────────────────────────────────────────────────────────────────

def _make_kpi_telemetry(tmp_path: Path, n: int = 20) -> Path:
    path = tmp_path / "et.jsonl"
    records = []
    for i in range(n):
        score = 30.0 + i * 3.5   # 30→97.5
        conf  = CONFIDENCE_HIGH if score >= 75 else (CONFIDENCE_MEDIUM if score >= 50 else CONFIDENCE_LOW)
        ret3d = 0.01 if score > 60 else -0.01
        records.append({
            "date": "2026-01-01", "symbol": f"SYM{i:04d}",
            "entry_timing_score": score, "confidence": conf,
            "action": ACTION_NORMAL, "action_taken": "ENTERED",
            "phase": "normal", "breakout_component": 50.0, "pullback_component": 50.0,
            "trend_component": 50.0, "market_component": 60.0,
            "bonuses": [], "penalties": [], "score_decile": _score_to_decile(score),
            "subsequent_3d_return": ret3d, "subsequent_5d_return": ret3d,
            "win_3d": ret3d > 0, "materialized_3d": True, "materialized_5d": True,
            "schema_version": "v1",
        })
    _write_telemetry(path, records)
    return path


def test_compute_kpis_returns_structure(tmp_path):
    path = _make_kpi_telemetry(tmp_path, n=20)
    kpis = compute_et_kpis(path)
    assert "total_records" in kpis
    assert kpis["total_records"] == 20
    assert "decile_stats_3d" in kpis
    assert "confidence_stats" in kpis


def test_compute_kpis_decile_keys(tmp_path):
    path = _make_kpi_telemetry(tmp_path, n=20)
    kpis = compute_et_kpis(path)
    assert set(kpis["decile_stats_3d"].keys()) == {str(d) for d in range(1, 11)}


def test_compute_kpis_empty_file(tmp_path):
    path = tmp_path / "et.jsonl"
    path.write_text("")
    assert compute_et_kpis(path) == {}


def test_compute_kpis_nonexistent(tmp_path):
    path = tmp_path / "no.jsonl"
    assert compute_et_kpis(path) == {}


def test_compute_kpis_monotonicity_computed(tmp_path):
    path = _make_kpi_telemetry(tmp_path, n=20)
    kpis = compute_et_kpis(path)
    mono = kpis.get("monotonicity_score")
    # May be None if fewer than 5 deciles have data; otherwise 0-1
    if mono is not None:
        assert 0.0 <= mono <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 15. format_et_report_section
# ─────────────────────────────────────────────────────────────────────────────

def test_format_report_empty():
    assert format_et_report_section({}) == ""


def test_format_report_has_header():
    scores = {"9501": _make_result(80.0, CONFIDENCE_HIGH)}
    scores["9501"].symbol = "9501"
    scores["9501"].action = ACTION_IMMEDIATE
    scores["9501"].phase  = "breakout"
    out = format_et_report_section(scores)
    assert "ENTRY TIMING" in out
    assert "9501" in out
    assert "HIGH" in out


def test_format_report_confidence_sections(tmp_path):
    scores = {
        "A": _make_result(80.0, CONFIDENCE_HIGH),
        "B": _make_result(60.0, CONFIDENCE_MEDIUM),
        "C": _make_result(40.0, CONFIDENCE_LOW),
    }
    for sym, r in scores.items():
        r.symbol = sym
        r.action = ACTION_WATCH
        r.phase  = "normal"
    out = format_et_report_section(scores)
    assert "HIGH" in out
    assert "MEDIUM" in out
    assert "LOW" in out


def test_format_report_with_kpi(tmp_path):
    path = _make_kpi_telemetry(tmp_path, n=15)
    scores = {"9501": _make_result(75.0, CONFIDENCE_HIGH)}
    scores["9501"].symbol = "9501"
    scores["9501"].action = ACTION_IMMEDIATE
    scores["9501"].phase  = "breakout"
    out = format_et_report_section(scores, telemetry_path=path)
    # Either KPI section or basic section
    assert "ENTRY TIMING" in out
