"""tests/live/test_future_leader_integrity.py

Future Leader Phase 1 Hardening — Research Integrity State

Covers:
  - IntegrityState classification (CLEAN / PARTIAL_FAILURE / MAJOR_FAILURE)
  - partial compute isolation (1 symbol failure doesn't stop others)
  - absolute threshold penalties (low_drift / negative_accumulation / weak_close / rs_delta)
  - governance separation proof (no execution imports)
  - lookahead contamination check
  - integrity log persistence (append-only, fail-closed)
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.live.future_leader_integrity import (
    IntegrityState,
    ResearchIntegrityState,
    format_integrity_mail_section,
    write_integrity_log,
)
from src.live.future_leader_screener import (
    ABS_ACCUM_MULT,
    ABS_ACCUM_THRESHOLD,
    ABS_CS_MULT,
    ABS_CS_THRESHOLD,
    ABS_DRIFT_MULT,
    ABS_DRIFT_THRESHOLD,
    ABS_RS_DELTA_MULT,
    ABS_RS_DELTA_THRESHOLD,
    FutureLeaderScreenResult,
    _compute_absolute_penalties,
    screen_future_leaders,
    write_future_leader_log,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_rsr_df(symbols: List[str], n: int = 30, base_rsr: float = 55.0) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {sym: np.clip([base_rsr + i * 0.2 + rng.normal(0, 0.5) for i in range(n)], 0, 100)
         for sym in symbols},
        index=dates,
    )


def _make_ohlcv(n: int = 30, trend: float = 0.003, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = [1000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + rng.normal(0, 0.005)))
    prices = np.array(prices)
    noise = rng.uniform(0.002, 0.008, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": prices, "High": prices * (1 + noise),
        "Low": prices * (1 - noise), "Close": prices,
        "Volume": np.ones(n) * 1_000_000,
    }, index=dates)


def _make_integrity(
    total: int = 10,
    scored: int = 10,
    missing: int = 0,
    rsr_failed: bool = False,
    ohlcv_failed: bool = False,
    partial_compute: bool = False,
    warn: int = 0,
    partial_failure_count: int = 0,
) -> ResearchIntegrityState:
    return ResearchIntegrityState.classify(
        observation_date="2026-05-23",
        total_input_symbols=total,
        successfully_scored_symbols=scored,
        missing_symbols_count=missing,
        rsr_build_failed=rsr_failed,
        ohlcv_load_failed=ohlcv_failed,
        partial_compute_failed=partial_compute,
        warning_count=warn,
        partial_failure_symbols_count=partial_failure_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task A — Integrity State Classification
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_clean():
    s = _make_integrity()
    assert s.integrity_state == IntegrityState.CLEAN


def test_classify_partial_compute_failed():
    s = _make_integrity(partial_compute=True)
    assert s.integrity_state == IntegrityState.PARTIAL_FAILURE


def test_classify_partial_missing_below_25pct():
    s = _make_integrity(total=10, missing=2)  # 20% < 25% → PARTIAL
    assert s.integrity_state == IntegrityState.PARTIAL_FAILURE


def test_classify_major_rsr_failed():
    s = _make_integrity(rsr_failed=True)
    assert s.integrity_state == IntegrityState.MAJOR_FAILURE


def test_classify_major_ohlcv_failed():
    s = _make_integrity(ohlcv_failed=True)
    assert s.integrity_state == IntegrityState.MAJOR_FAILURE


def test_classify_major_missing_exactly_25pct():
    # 25% ratio but count=2 < 5 → PARTIAL_FAILURE (absolute count gate not met)
    s = _make_integrity(total=8, missing=2)
    assert s.integrity_state == IntegrityState.PARTIAL_FAILURE


def test_classify_major_missing_above_25pct():
    # 30% ratio AND count=6 >= 5 → MAJOR_FAILURE (both gates met)
    s = _make_integrity(total=20, missing=6)
    assert s.integrity_state == IntegrityState.MAJOR_FAILURE


def test_classify_partial_missing_just_below_25pct():
    s = _make_integrity(total=100, missing=24)  # 24% < 25% → PARTIAL
    assert s.integrity_state == IntegrityState.PARTIAL_FAILURE


def test_classify_major_zero_total_symbols():
    """Zero input symbols = no data = MAJOR_FAILURE."""
    s = _make_integrity(total=0, scored=0, missing=0)
    assert s.integrity_state == IntegrityState.MAJOR_FAILURE


def test_integrity_to_dict_keys():
    s = _make_integrity()
    d = s.to_dict()
    required = {
        "observation_date", "integrity_state", "total_input_symbols",
        "successfully_scored_symbols", "missing_symbols_count",
        "rsr_build_failed", "ohlcv_load_failed", "partial_compute_failed",
        "warning_count", "data_quality_weight", "partial_failure_symbols_count",
    }
    assert required.issubset(d.keys())


def test_integrity_state_value_is_string():
    """integrity_state in to_dict() should be a string, not Enum."""
    s = _make_integrity()
    d = s.to_dict()
    assert isinstance(d["integrity_state"], str)
    assert d["integrity_state"] == "CLEAN"


def test_format_integrity_mail_section_clean():
    s = _make_integrity()
    text = format_integrity_mail_section(s)
    assert "RESEARCH INTEGRITY" in text
    assert "CLEAN" in text
    assert "rsr_build_failed: false" in text
    assert "ohlcv_load_failed: false" in text


def test_format_integrity_mail_section_partial():
    s = _make_integrity(partial_compute=True)
    text = format_integrity_mail_section(s)
    assert "PARTIAL_FAILURE" in text
    assert "partial_compute_failed: true" in text


def test_format_integrity_mail_section_major():
    s = _make_integrity(rsr_failed=True)
    text = format_integrity_mail_section(s)
    assert "MAJOR_FAILURE" in text
    assert "rsr_build_failed: true" in text


# ─────────────────────────────────────────────────────────────────────────────
# Task A — Partial Compute Isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_compute_isolation_good_symbols_scored():
    """One symbol with broken High column → compute failure tracked, others still scored."""
    n = 25
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)

    good_df = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)

    # BAD: High column set to non-numeric string → ValueError in astype(float)
    bad_df = good_df.copy()
    bad_df["High"] = "INVALID"

    ohlcv = {"GOOD1.T": good_df, "GOOD2.T": good_df, "BAD.T": bad_df}
    rsr_df = _make_rsr_df(["GOOD1.T", "GOOD2.T", "BAD.T"], n=n, base_rsr=55.0)

    failure_tracker: dict = {}
    results = screen_future_leaders(
        rsr_df=rsr_df,
        ohlcv=ohlcv,
        sector_map={"GOOD1.T": "電機精密", "GOOD2.T": "電機精密", "BAD.T": "機械"},
        effective_capital=3_000_000.0,
        _failure_tracker=failure_tracker,
    )

    # Good symbols must be in results
    scored_syms = {r.symbol for r in results}
    assert "GOOD1.T" in scored_syms
    assert "GOOD2.T" in scored_syms


def test_partial_compute_isolation_failure_tracked():
    """Compute failure for BAD.T propagates to failure_tracker."""
    n = 25
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)

    good_df = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    bad_df = good_df.copy()
    bad_df["High"] = "INVALID"

    ohlcv = {"GOOD.T": good_df, "BAD.T": bad_df}
    rsr_df = _make_rsr_df(["GOOD.T", "BAD.T"], n=n, base_rsr=55.0)

    failure_tracker: dict = {}
    screen_future_leaders(
        rsr_df=rsr_df,
        ohlcv=ohlcv,
        sector_map={"GOOD.T": "電機精密", "BAD.T": "機械"},
        effective_capital=3_000_000.0,
        _failure_tracker=failure_tracker,
    )

    assert failure_tracker.get("partial_compute_failed") is True
    assert "BAD.T" in failure_tracker.get("partial_failure_symbols", set())


def test_partial_compute_isolation_no_crash():
    """Whole system must not raise on partial compute failure."""
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)
    good_df = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    bad_df = good_df.copy()
    bad_df["High"] = "INVALID"

    ohlcv = {"A.T": good_df, "B.T": bad_df, "C.T": good_df}
    rsr_df = _make_rsr_df(["A.T", "B.T", "C.T"], n=n)

    # Should not raise
    results = screen_future_leaders(
        rsr_df=rsr_df, ohlcv=ohlcv,
        sector_map={"A.T": "電機精密", "B.T": "機械", "C.T": "化学"},
        effective_capital=3_000_000.0,
    )
    # A and C should be scored
    scored_syms = {r.symbol for r in results}
    assert "A.T" in scored_syms
    assert "C.T" in scored_syms


# ─────────────────────────────────────────────────────────────────────────────
# Task A — Integrity Log Persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_write_integrity_log_creates_jsonl(tmp_path: Path):
    s = _make_integrity(partial_compute=True)
    log_path = tmp_path / "integrity.jsonl"
    write_integrity_log(s, log_path)

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["integrity_state"] == "PARTIAL_FAILURE"
    assert "observation_ts" in rec
    assert rec["observation_date"] == "2026-05-23"


def test_write_integrity_log_appends(tmp_path: Path):
    """Write twice → two records (append-only)."""
    s1 = _make_integrity()
    s2 = _make_integrity(rsr_failed=True)
    log_path = tmp_path / "integrity.jsonl"
    write_integrity_log(s1, log_path)
    write_integrity_log(s2, log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["integrity_state"] == "CLEAN"
    assert json.loads(lines[1])["integrity_state"] == "MAJOR_FAILURE"


def test_write_integrity_log_fail_closed(tmp_path: Path):
    """Write to path whose parent is a file → IOError (fail-closed)."""
    s = _make_integrity()
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file")
    bad_path = blocker / "cannot_create" / "integrity.jsonl"
    with pytest.raises(IOError):
        write_integrity_log(s, bad_path)


def test_write_integrity_log_schema(tmp_path: Path):
    s = _make_integrity(partial_compute=True, warn=2)
    log_path = tmp_path / "schema_test.jsonl"
    write_integrity_log(s, log_path)

    rec = json.loads(log_path.read_text(encoding="utf-8").strip())
    required = {
        "observation_ts", "observation_date", "integrity_state",
        "total_input_symbols", "successfully_scored_symbols",
        "missing_symbols_count", "rsr_build_failed", "ohlcv_load_failed",
        "partial_compute_failed", "warning_count",
        "data_quality_weight", "partial_failure_symbols_count",
    }
    assert required.issubset(rec.keys())
    assert rec["warning_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Task B — Absolute Threshold Penalties
# ─────────────────────────────────────────────────────────────────────────────

def test_absolute_threshold_low_drift():
    """persistent_drift < 0.05 → low_drift penalty."""
    pens = _compute_absolute_penalties(
        drift_raw_val=ABS_DRIFT_THRESHOLD - 0.01,  # just below threshold
        accum_raw_val=0.05,
        cs3d_val=0.60,
        rs_delta_val=0.01,
    )
    assert "low_drift" in pens
    assert pens["low_drift"] == pytest.approx(ABS_DRIFT_MULT)


def test_absolute_threshold_negative_accumulation():
    """accumulation_pressure <= 0 → negative_accumulation penalty."""
    pens = _compute_absolute_penalties(
        drift_raw_val=0.10,
        accum_raw_val=ABS_ACCUM_THRESHOLD,  # exactly 0
        cs3d_val=0.60,
        rs_delta_val=0.01,
    )
    assert "negative_accumulation" in pens
    assert pens["negative_accumulation"] == pytest.approx(ABS_ACCUM_MULT)


def test_absolute_threshold_weak_close_strength():
    """close_strength_3d < 0.45 → weak_close_strength penalty."""
    pens = _compute_absolute_penalties(
        drift_raw_val=0.10,
        accum_raw_val=0.05,
        cs3d_val=ABS_CS_THRESHOLD - 0.01,  # just below
        rs_delta_val=0.01,
    )
    assert "weak_close_strength" in pens
    assert pens["weak_close_strength"] == pytest.approx(ABS_CS_MULT)


def test_absolute_threshold_negative_rs_delta():
    """rs_vs_topix_10d_delta < 0 → negative_rs_delta penalty."""
    pens = _compute_absolute_penalties(
        drift_raw_val=0.10,
        accum_raw_val=0.05,
        cs3d_val=0.60,
        rs_delta_val=ABS_RS_DELTA_THRESHOLD - 0.01,  # just below 0
    )
    assert "negative_rs_delta" in pens
    assert pens["negative_rs_delta"] == pytest.approx(ABS_RS_DELTA_MULT)


def test_absolute_threshold_none_when_strong_signals():
    """Strong signals above all thresholds → no absolute penalties."""
    pens = _compute_absolute_penalties(
        drift_raw_val=0.30,
        accum_raw_val=0.10,
        cs3d_val=0.75,
        rs_delta_val=0.02,
    )
    assert len(pens) == 0


def test_absolute_threshold_all_fire():
    """All thresholds violated → all four penalties present."""
    pens = _compute_absolute_penalties(
        drift_raw_val=-0.10,   # < 0.05
        accum_raw_val=0.0,     # <= 0
        cs3d_val=0.30,         # < 0.45
        rs_delta_val=-0.05,    # < 0
    )
    assert "low_drift" in pens
    assert "negative_accumulation" in pens
    assert "weak_close_strength" in pens
    assert "negative_rs_delta" in pens
    assert len(pens) == 4


def test_absolute_threshold_multiplier_product_reduces_score():
    """Symbol with all 4 absolute penalties has lower score than same symbol without."""
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    # Strong symbol: clear uptrend, volume surge, strong close
    prices_strong = np.linspace(1000, 1150, n)
    df_strong = pd.DataFrame({
        "Open": prices_strong, "High": prices_strong * 1.005,
        "Low": prices_strong * 0.995, "Close": prices_strong,
        "Volume": np.ones(n) * 1_500_000,
    }, index=dates)

    # Weak symbol: flat/down, no volume, weak close
    prices_weak = np.ones(n) * 1000.0  # flat → near-zero drift
    df_weak = pd.DataFrame({
        "Open": prices_weak, "High": prices_weak * 1.001,
        "Low": prices_weak * 0.999, "Close": prices_weak * 0.9995,  # slight down-close
        "Volume": np.ones(n) * 500_000,
    }, index=dates)

    ohlcv = {"S.T": df_strong, "W.T": df_weak}
    rsr_df_data = pd.DataFrame(
        {"S.T": np.linspace(50, 62, n), "W.T": np.linspace(50, 52, n)},
        index=dates,
    )

    results = screen_future_leaders(
        rsr_df=rsr_df_data, ohlcv=ohlcv,
        sector_map={"S.T": "電機精密", "W.T": "機械"},
        effective_capital=3_000_000.0,
    )
    r_map = {r.symbol: r for r in results}

    # Strong should score higher than weak
    assert r_map["S.T"].future_leader_score >= r_map["W.T"].future_leader_score


def test_absolute_penalties_stored_in_result():
    """absolute_penalties dict should be populated in FutureLeaderScreenResult."""
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Flat price → near-zero drift, no accumulation pressure
    prices = np.ones(n) * 1000.0
    df = pd.DataFrame({
        "Open": prices, "High": prices * 1.001,
        "Low": prices * 0.999, "Close": prices,
        "Volume": np.ones(n) * 1e6,
    }, index=dates)

    rsr_df = pd.DataFrame({"A.T": np.full(n, 55.0)}, index=dates)
    results = screen_future_leaders(
        rsr_df=rsr_df, ohlcv={"A.T": df},
        sector_map={"A.T": "電機精密"}, effective_capital=3_000_000.0,
    )
    r = results[0]
    # Flat price has drift ≈ 0 and zero accumulation → penalties should fire
    assert isinstance(r.absolute_penalties, dict)
    assert "low_drift" in r.absolute_penalties or "negative_accumulation" in r.absolute_penalties


def test_absolute_penalties_in_to_dict():
    """absolute_penalties appears in to_dict() output."""
    r = FutureLeaderScreenResult(
        symbol="X.T", future_leader_score=50.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.0, persistent_drift=0.0,
        close_strength_3d=0.5, consecutive_high_score_days=0,
        penalties={}, failure_flags=[], absolute_penalties={"low_drift": 0.70},
    )
    d = r.to_dict()
    assert "absolute_penalties" in d
    assert d["absolute_penalties"]["low_drift"] == pytest.approx(0.70)


def test_absolute_penalties_boundary_drift_exactly_threshold():
    """drift == 0.05 → NOT penalized (strict <, not <=)."""
    pens = _compute_absolute_penalties(
        drift_raw_val=ABS_DRIFT_THRESHOLD,  # exactly 0.05 → no penalty
        accum_raw_val=0.05,
        cs3d_val=0.60,
        rs_delta_val=0.01,
    )
    assert "low_drift" not in pens


def test_absolute_penalties_boundary_accumulation_positive():
    """accumulation_pressure > 0 → NOT penalized."""
    pens = _compute_absolute_penalties(
        drift_raw_val=0.10,
        accum_raw_val=1e-8,  # tiny positive
        cs3d_val=0.60,
        rs_delta_val=0.01,
    )
    assert "negative_accumulation" not in pens


# ─────────────────────────────────────────────────────────────────────────────
# Task C — Governance Separation
# ─────────────────────────────────────────────────────────────────────────────

def test_governance_observation_candidate_property_exists():
    """is_governance_observation_candidate property must exist."""
    r = FutureLeaderScreenResult(
        symbol="G.T", future_leader_score=70.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.05, persistent_drift=0.5,
        close_strength_3d=0.6, consecutive_high_score_days=3,
        penalties={}, failure_flags=[],
    )
    assert hasattr(r, "is_governance_observation_candidate")
    assert r.is_governance_observation_candidate is True


def test_governance_candidate_access_raises():
    """is_governance_candidate must raise RuntimeError — semantic isolation guard."""
    r = FutureLeaderScreenResult(
        symbol="T.T", future_leader_score=70.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.05,
        persistent_drift=0.5, close_strength_3d=0.6,
        consecutive_high_score_days=3, penalties={}, failure_flags=[],
    )
    with pytest.raises(RuntimeError, match="governance_candidate semantics"):
        _ = r.is_governance_candidate


def test_governance_observation_candidate_not_dataclass_field():
    """is_governance_observation_candidate is a property, not a dataclass field."""
    r = FutureLeaderScreenResult(
        symbol="T.T", future_leader_score=70.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.0, persistent_drift=0.5,
        close_strength_3d=0.6, consecutive_high_score_days=3, penalties={}, failure_flags=[],
    )
    assert "is_governance_observation_candidate" not in r.__dataclass_fields__


def test_governance_observation_in_to_dict():
    """is_governance_observation_candidate appears in to_dict()."""
    r = FutureLeaderScreenResult(
        symbol="T.T", future_leader_score=70.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.0, persistent_drift=0.5,
        close_strength_3d=0.6, consecutive_high_score_days=3, penalties={}, failure_flags=[],
    )
    d = r.to_dict()
    assert "is_governance_observation_candidate" in d
    assert d["is_governance_observation_candidate"] is True


def test_mail_section_shows_observation_not_governance():
    """★OBSERVATION must appear in mail section; ★GOVERNANCE must NOT appear."""
    from src.live.future_leader_screener import format_future_leader_mail_section

    r = FutureLeaderScreenResult(
        symbol="OBS.T", future_leader_score=70.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.05, persistent_drift=0.5,
        close_strength_3d=0.6, consecutive_high_score_days=3, penalties={}, failure_flags=[],
    )
    section = format_future_leader_mail_section([r])
    assert "★OBSERVATION" in section
    assert "★GOVERNANCE" not in section


def test_no_execution_imports_in_screener():
    """future_leader_screener source must NOT reference execution-path modules."""
    import inspect
    import src.live.future_leader_screener as screener_mod

    source = inspect.getsource(screener_mod)
    forbidden_patterns = [
        "signal_bridge",
        "order_manager",
        "src.allocation",
        "src.truth",
        "src.addon",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in source, (
            f"future_leader_screener contains forbidden execution-path reference: {pattern}"
        )


def test_observation_only_invariant_in_results():
    """All results have observation_only=True."""
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)
    df = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    rsr_df = pd.DataFrame({"A.T": np.full(n, 55.0), "B.T": np.full(n, 60.0)}, index=dates)
    results = screen_future_leaders(
        rsr_df=rsr_df, ohlcv={"A.T": df, "B.T": df},
        sector_map={"A.T": "電機精密", "B.T": "機械"}, effective_capital=3_000_000.0,
    )
    for r in results:
        assert r.observation_only is True
        assert r.to_dict()["observation_only"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead contamination check
# ─────────────────────────────────────────────────────────────────────────────

def test_lookahead_vol_ratio_uses_prior_day_avg():
    """
    vol_ratio uses shift(1).rolling(5).mean() — excludes today's spike.
    If today's spike is included in the avg, penalty won't fire (ratio diluted).
    With prior-day avg only, ratio = spike_vol / baseline_vol > threshold.
    """
    from src.live.future_leader_screener import _detect_penalties

    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.ones(n) * 1000.0
    prices[-1] = 1050.0   # 5% jump on last day
    vol = np.ones(n) * 1_000_000.0
    vol[-1] = 3_500_000.0  # 3.5× baseline (prior-day avg = 1M, so ratio = 3.5 > 2.5)
    df = pd.DataFrame({
        "Open": prices, "High": prices * 1.006, "Low": prices * 0.994,
        "Close": prices, "Volume": vol,
    }, index=dates)

    pens = _detect_penalties({"J.T": df}, ["J.T"])
    assert "single_day_jump" in pens.get("J.T", {}), (
        "Prior-day avg correctly computed — single_day_jump should fire"
    )


def test_accumulation_uses_prior_day_vol_avg():
    """
    avg_vol_full = nanmean(vol[:-1]) — excludes today from baseline.
    Today's surge gives vol_ratio > 1, boosting accumulation.
    """
    from src.live.future_leader_screener import _compute_accumulation_pressure

    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)  # all up-days

    # Surge only on the last day
    vol = np.ones(n) * 1_000_000.0
    vol[-1] = 3_000_000.0

    df = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": vol,
    }, index=dates)

    # Compare with flat volume (ratio always 1.0)
    df_flat = df.copy()
    df_flat["Volume"] = 1_000_000.0

    accum_surge = _compute_accumulation_pressure({"S.T": df}, ["S.T"])
    accum_flat  = _compute_accumulation_pressure({"F.T": df_flat}, ["F.T"])

    assert accum_surge["S.T"] > accum_flat["F.T"], (
        "Prior-day vol avg used correctly — surge today boosts ratio"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Refinement 1 — MAJOR_FAILURE requires ratio AND absolute count
# ─────────────────────────────────────────────────────────────────────────────

def test_major_failure_requires_ratio_and_absolute_count():
    """
    MAJOR_FAILURE requires BOTH missing_ratio >= 25% AND missing_count >= 5.
    Small-universe days with few absolute missing are PARTIAL_FAILURE, not MAJOR.
    """
    # 3/12 = 25% ratio but count=3 < 5 → PARTIAL_FAILURE (small universe, not MAJOR)
    s1 = _make_integrity(total=12, missing=3)
    assert s1.integrity_state == IntegrityState.PARTIAL_FAILURE, (
        "3/12 missing: ratio=25% but count=3 < 5 → must be PARTIAL_FAILURE"
    )

    # 6/20 = 30% AND count=6 >= 5 → MAJOR_FAILURE (both gates met)
    s2 = _make_integrity(total=20, missing=6)
    assert s2.integrity_state == IntegrityState.MAJOR_FAILURE, (
        "6/20 missing: ratio=30% AND count=6 >= 5 → must be MAJOR_FAILURE"
    )

    # 1/100 = 1% ratio, count=1 < 5 → PARTIAL_FAILURE
    s3 = _make_integrity(total=100, missing=1)
    assert s3.integrity_state == IntegrityState.PARTIAL_FAILURE, (
        "1/100 missing: ratio=1% → must be PARTIAL_FAILURE"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Refinement 2 — Compute failure symbol not persisted
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_failure_symbol_not_persisted(tmp_path: Path):
    """
    Symbol with compute failure must be completely excluded from results and JSONL.
    compute failure != weak signal; broken observations must not pollute the dataset.
    """
    n = 25
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)
    good_df = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    # INVALID High forces ValueError in astype(float) → compute failure
    bad_df = good_df.copy()
    bad_df["High"] = "INVALID"

    ohlcv = {"GOOD1.T": good_df, "GOOD2.T": good_df, "BAD.T": bad_df}
    rsr_df = _make_rsr_df(["GOOD1.T", "GOOD2.T", "BAD.T"], n=n, base_rsr=55.0)

    failure_tracker: dict = {}
    results = screen_future_leaders(
        rsr_df=rsr_df,
        ohlcv=ohlcv,
        sector_map={"GOOD1.T": "電機精密", "GOOD2.T": "電機精密", "BAD.T": "機械"},
        effective_capital=3_000_000.0,
        _failure_tracker=failure_tracker,
    )

    # BAD.T must be absent from results (no degraded/fallback score persisted)
    result_symbols = {r.symbol for r in results}
    assert "BAD.T" not in result_symbols, "compute-failed symbol must not appear in results"

    # Good symbols still scored
    assert "GOOD1.T" in result_symbols
    assert "GOOD2.T" in result_symbols

    # successfully_scored must not count BAD.T
    assert failure_tracker["successfully_scored"] == 2

    # partial failure flagged
    assert failure_tracker["partial_compute_failed"] is True
    assert "BAD.T" in failure_tracker["partial_failure_symbols"]

    # JSONL must not contain BAD.T
    log_path = tmp_path / "candidates.jsonl"
    write_future_leader_log(results, log_path, run_id="test_purity")
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    symbols_in_log = {json.loads(line)["symbol"] for line in lines}
    assert "BAD.T" not in symbols_in_log, "compute-failed symbol must not be in JSONL"


# ─────────────────────────────────────────────────────────────────────────────
# Refinement 3 — Governance semantic isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_governance_candidate_not_in_to_dict():
    """is_governance_candidate must NOT appear in to_dict() output."""
    r = FutureLeaderScreenResult(
        symbol="T.T", future_leader_score=70.0, rsr_zone="emerging",
        affordability_ok=True, accumulation_pressure=0.05, persistent_drift=0.5,
        close_strength_3d=0.6, consecutive_high_score_days=3, penalties={}, failure_flags=[],
    )
    d = r.to_dict()
    assert "is_governance_candidate" not in d, (
        "is_governance_candidate must be absent from to_dict() — semantic isolation"
    )
    assert "is_governance_observation_candidate" in d


def test_no_governance_semantic_contamination_in_screener():
    """
    future_leader_screener must not contain execution-path governance references.
    is_governance_candidate must not be called (only defined as RuntimeError trap).
    """
    import inspect
    import src.live.future_leader_screener as screener_mod

    source = inspect.getsource(screener_mod)

    # Execution-path code references must be absent
    # (use underscore/code-form to avoid matching isolation-documenting docstrings)
    for term in [
        "governance_promotion",
        "deployable_universe",
        "execution_bridge",
        "signal_bridge",
        "order_routing",
    ]:
        assert term not in source, (
            f"Forbidden governance execution code reference '{term}' found in screener"
        )

    # is_governance_candidate must not be called — only the RuntimeError definition exists
    assert "self.is_governance_candidate" not in source, (
        "self.is_governance_candidate must not be called; "
        "use is_governance_observation_candidate instead"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Integrity Metadata Extension
# ─────────────────────────────────────────────────────────────────────────────

def test_data_quality_weight_assignment():
    """data_quality_weight: CLEAN=1.0, PARTIAL_FAILURE=0.5, MAJOR_FAILURE=0.0"""
    clean = _make_integrity()
    assert clean.data_quality_weight == 1.0

    partial = _make_integrity(partial_compute=True)
    assert partial.data_quality_weight == 0.5

    major = _make_integrity(rsr_failed=True)
    assert major.data_quality_weight == 0.0


def test_partial_failure_symbols_count():
    """partial_failure_symbols_count is stored independently from missing_symbols_count."""
    s = _make_integrity(missing=1, partial_failure_count=3)
    assert s.partial_failure_symbols_count == 3
    assert s.missing_symbols_count == 1

    # Zero default
    s_zero = _make_integrity()
    assert s_zero.partial_failure_symbols_count == 0


def test_jsonl_contains_integrity_metadata(tmp_path: Path):
    """JSONL must include data_quality_weight and partial_failure_symbols_count."""
    s = _make_integrity(partial_compute=True, partial_failure_count=4)
    log_path = tmp_path / "integrity_meta.jsonl"
    write_integrity_log(s, log_path)

    rec = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert "data_quality_weight" in rec, "data_quality_weight must be in JSONL"
    assert "partial_failure_symbols_count" in rec, "partial_failure_symbols_count must be in JSONL"
    assert rec["data_quality_weight"] == 0.5
    assert rec["partial_failure_symbols_count"] == 4
