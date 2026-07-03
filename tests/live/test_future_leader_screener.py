"""tests/live/test_future_leader_screener.py

Future Leader Screener — unit tests

Covers:
  - observation_only invariant (shadow-only guarantee)
  - score stability (deterministic)
  - no NaN propagation
  - affordability gate
  - RSR zone classification
  - accumulation pressure (up-day only logic)
  - persistent drift ratio
  - close_strength_3d penalty
  - single_day_jump penalty
  - gap_exhaustion penalty
  - mature zone penalty
  - weak zone penalty
  - logging schema validation
  - fail-closed log write
  - consecutive tracking (prior day state)
  - governance candidate conditions
  - write_future_leader_log persistence
  - empty universe
  - single-symbol universe
  - weights sum to 1.0
  - format_future_leader_mail_section
  - governance candidate property
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.live.future_leader_screener import (
    CLOSE_STRENGTH_PENALTY_MULT,
    CLOSE_STRENGTH_WARN_THRESHOLD,
    GAP_EXHAUSTION_MULT,
    GOVERNANCE_CONSECUTIVE_MIN,
    GOVERNANCE_SCORE_MIN,
    MATURE_ZONE_PENALTY,
    RSR_EMERGING_MAX,
    RSR_EMERGING_MIN,
    SINGLE_DAY_JUMP_MULT,
    WEAK_ZONE_PENALTY,
    WEIGHTS,
    FutureLeaderScreenResult,
    _classify_rsr_zone,
    _compute_accumulation_pressure,
    _compute_affordability,
    _compute_close_strength_3d,
    _compute_persistent_drift,
    _compute_rs_topix_delta,
    _compute_rsr_velocity,
    _cs_rank_high,
    _detect_penalties,
    _load_prior_day_state,
    format_future_leader_mail_section,
    screen_future_leaders,
    write_future_leader_log,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 30,
    base_price: float = 2000.0,
    trend: float = 0.0,      # daily drift
    noise: float = 0.005,    # daily noise amplitude
    up_vol_mult: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    prices = [base_price]
    for _ in range(n - 1):
        r = trend + rng.normal(0, noise)
        prices.append(prices[-1] * (1 + r))
    prices = np.array(prices)
    noise_arr = rng.uniform(0.002, 0.008, n)
    high  = prices * (1 + noise_arr)
    low   = prices * (1 - noise_arr)
    close = prices
    vol_base = 1_000_000.0
    vol = np.where(
        close > np.roll(close, 1),
        vol_base * up_vol_mult,
        vol_base * 0.7,
    )
    vol[0] = vol_base
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": prices, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _make_rsr_df(
    symbols: List[str],
    n: int = 30,
    base_rsr: float = 55.0,
    vel_profile: Dict[str, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Create synthetic RSR DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    data = {}
    for sym in symbols:
        vel = (vel_profile or {}).get(sym, 0.0)
        vals = [base_rsr + i * vel + rng.normal(0, 0.5) for i in range(n)]
        data[sym] = np.clip(vals, 0, 100)
    return pd.DataFrame(data, index=dates)


def _sector_map(symbols: List[str]) -> Dict[str, str]:
    sectors = ["電機精密", "機械", "商社", "鉄鋼", "化学"]
    return {sym: sectors[i % len(sectors)] for i, sym in enumerate(symbols)}


SYMS = ["1111.T", "2222.T", "3333.T", "4444.T", "5555.T"]
OHLCV_5 = {sym: _make_ohlcv(n=35, base_price=2000 + i * 500, seed=i) for i, sym in enumerate(SYMS)}
RSR_5    = _make_rsr_df(SYMS, n=35, base_rsr=55.0, vel_profile={sym: float(i) * 0.3 for i, sym in enumerate(SYMS)})
SECTOR_5 = _sector_map(SYMS)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: weights sum to 1.0
# ─────────────────────────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: observation_only is always True (invariant)
# ─────────────────────────────────────────────────────────────────────────────

def test_observation_only_invariant():
    results = screen_future_leaders(
        rsr_df=RSR_5,
        ohlcv=OHLCV_5,
        sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    assert len(results) > 0
    for r in results:
        assert r.observation_only is True


def test_observation_only_in_to_dict():
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    for r in results:
        d = r.to_dict()
        assert d["observation_only"] is True


def test_observation_only_not_a_settable_field():
    """observation_only is a property — cannot be set to False via __init__."""
    r = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )[0]
    # Reading it always returns True
    assert r.observation_only is True
    # It is a property, not a dataclass field
    assert "observation_only" not in r.__dataclass_fields__


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: score stability (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def test_score_stability():
    r1 = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    r2 = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    assert [r.symbol for r in r1] == [r.symbol for r in r2]
    for a, b in zip(r1, r2):
        assert a.future_leader_score == b.future_leader_score


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: no NaN propagation
# ─────────────────────────────────────────────────────────────────────────────

def test_no_nan_propagation():
    # Inject NaN into some cells
    ohlcv_nan = {}
    for sym, df in OHLCV_5.items():
        df2 = df.copy()
        df2.iloc[5:8, df2.columns.get_loc("Volume")] = np.nan
        df2.iloc[10, df2.columns.get_loc("Close")]   = np.nan
        ohlcv_nan[sym] = df2

    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=ohlcv_nan, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    for r in results:
        assert math.isfinite(r.future_leader_score), f"{r.symbol} score is not finite"
        assert math.isfinite(r.accumulation_pressure)
        assert math.isfinite(r.persistent_drift)
        assert math.isfinite(r.close_strength_3d)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: affordability gate
# ─────────────────────────────────────────────────────────────────────────────

def test_affordability_gate_expensive_symbol():
    """Symbol with price > capital × 30% / 100shares → affordability_ok=False."""
    capital = 1_000_000.0
    threshold = capital * 0.30  # 300,000 JPY per lot
    max_price = threshold / 100  # 3,000 JPY per share → 300,000 per lot
    high_price = max_price * 1.5   # 4,500 JPY → 450,000 per lot → over threshold

    ohlcv_high = {"EXPR.T": _make_ohlcv(n=30, base_price=high_price, seed=99)}
    rsr_high   = _make_rsr_df(["EXPR.T"], n=30, base_rsr=58.0)
    sector_high = {"EXPR.T": "電機精密"}

    results = screen_future_leaders(
        rsr_df=rsr_high, ohlcv=ohlcv_high, sector_map=sector_high,
        effective_capital=capital, lot_size=100, max_lot_cost_ratio=0.30,
    )
    assert len(results) == 1
    assert results[0].affordability_ok is False
    assert "affordability_fail" in results[0].failure_flags


def test_affordability_gate_cheap_symbol():
    """Symbol with price << capital × 30% → affordability_ok=True."""
    capital = 3_000_000.0
    cheap_price = 500.0  # 50,000 JPY per lot << 900,000 threshold

    ohlcv_cheap = {"CHEP.T": _make_ohlcv(n=30, base_price=cheap_price, seed=77)}
    rsr_cheap   = _make_rsr_df(["CHEP.T"], n=30, base_rsr=58.0)
    sector_cheap = {"CHEP.T": "機械"}

    results = screen_future_leaders(
        rsr_df=rsr_cheap, ohlcv=ohlcv_cheap, sector_map=sector_cheap,
        effective_capital=capital,
    )
    assert len(results) == 1
    assert results[0].affordability_ok is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: RSR zone classification
# ─────────────────────────────────────────────────────────────────────────────

def test_rsr_zone_weak():
    assert _classify_rsr_zone(0.0)   == "weak"
    assert _classify_rsr_zone(30.0)  == "weak"
    assert _classify_rsr_zone(RSR_EMERGING_MIN - 0.1) == "weak"


def test_rsr_zone_emerging():
    assert _classify_rsr_zone(RSR_EMERGING_MIN)     == "emerging"
    assert _classify_rsr_zone(60.0)                  == "emerging"
    assert _classify_rsr_zone(RSR_EMERGING_MAX)      == "emerging"


def test_rsr_zone_mature():
    assert _classify_rsr_zone(RSR_EMERGING_MAX + 0.1) == "mature"
    assert _classify_rsr_zone(80.0)                    == "mature"
    assert _classify_rsr_zone(100.0)                   == "mature"


def test_rsr_zone_in_screen_result():
    """RSR 55 → emerging zone."""
    rsr_emerging = _make_rsr_df(["A.T"], n=30, base_rsr=55.0)
    ohlcv_e = {"A.T": _make_ohlcv(n=30, seed=1)}
    results = screen_future_leaders(
        rsr_df=rsr_emerging, ohlcv=ohlcv_e, sector_map={"A.T": "電機精密"},
        effective_capital=3_000_000.0,
    )
    assert results[0].rsr_zone == "emerging"
    assert results[0].current_rsr == pytest.approx(55.0, abs=5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: accumulation pressure — up-day only
# ─────────────────────────────────────────────────────────────────────────────

def test_accumulation_pressure_up_days_only():
    """Symbol with all up-days should have higher accum than all down-days."""
    n = 20
    # All up-days
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices_up = np.linspace(1000, 1100, n)
    df_up = pd.DataFrame({
        "Open": prices_up, "High": prices_up * 1.005, "Low": prices_up * 0.995,
        "Close": prices_up, "Volume": np.full(n, 1_500_000),
    }, index=dates)
    # All down-days
    prices_dn = np.linspace(1100, 1000, n)
    df_dn = pd.DataFrame({
        "Open": prices_dn, "High": prices_dn * 1.005, "Low": prices_dn * 0.995,
        "Close": prices_dn, "Volume": np.full(n, 1_500_000),
    }, index=dates)

    accum_up = _compute_accumulation_pressure({"U.T": df_up}, ["U.T"])
    accum_dn = _compute_accumulation_pressure({"D.T": df_dn}, ["D.T"])

    assert accum_up["U.T"] > 0, "up-day should yield positive accumulation"
    assert accum_dn["D.T"] == 0.0, "down-day should yield zero accumulation"


def test_accumulation_pressure_volume_weighted():
    """Volume surge on most recent up-day → higher accumulation than flat volume."""
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.linspace(1000, 1100, n)  # all up-days

    # H.T: baseline 1M throughout, but TODAY (last row) has 3M surge.
    # avg_vol_full = mean of prior-day rows in tail = 1M → vol_ratio_today = 3.0
    vol_h = np.ones(n) * 1_000_000.0
    vol_h[-1] = 3_000_000.0  # today's surge only

    df_h = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": vol_h,
    }, index=dates)

    # L.T: flat 1M throughout → vol_ratio = 1.0 on every up-day
    vol_l = np.ones(n) * 1_000_000.0

    df_l = pd.DataFrame({
        "Open": prices, "High": prices * 1.005, "Low": prices * 0.995,
        "Close": prices, "Volume": vol_l,
    }, index=dates)

    high_vol_accum = _compute_accumulation_pressure({"H.T": df_h}, ["H.T"])
    low_vol_accum  = _compute_accumulation_pressure({"L.T": df_l}, ["L.T"])

    # H.T: 9 up-days at log1p(1.0) + 1 up-day at log1p(3.0) = 7.624 × factor
    # L.T: 10 up-days at log1p(1.0) = 6.931 × factor  → H.T > L.T
    assert high_vol_accum["H.T"] > low_vol_accum["L.T"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: persistent drift
# ─────────────────────────────────────────────────────────────────────────────

def test_persistent_drift_positive_for_smooth_uptrend():
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Smooth uptrend (low noise)
    prices = np.linspace(1000, 1100, n)
    df_smooth = pd.DataFrame({
        "Open": prices, "High": prices * 1.003, "Low": prices * 0.997,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    drift = _compute_persistent_drift({"S.T": df_smooth}, ["S.T"])
    assert drift["S.T"] > 0, "smooth uptrend should yield positive drift"


def test_persistent_drift_low_for_choppy():
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Choppy: alternating up/down with large noise
    prices = 1000 + np.cumsum(np.where(np.arange(n) % 2 == 0, 30, -28))
    df_chop = pd.DataFrame({
        "Open": prices, "High": prices * 1.020, "Low": prices * 0.980,
        "Close": prices, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    drift = _compute_persistent_drift({"C.T": df_chop}, ["C.T"])
    # Smooth uptrend should have higher drift than choppy
    prices_s = np.linspace(1000, 1020, n)
    df_smooth = pd.DataFrame({
        "Open": prices_s, "High": prices_s * 1.003, "Low": prices_s * 0.997,
        "Close": prices_s, "Volume": np.ones(n) * 1e6,
    }, index=dates)
    drift_s = _compute_persistent_drift({"SM.T": df_smooth}, ["SM.T"])
    assert drift_s["SM.T"] > drift["C.T"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: close_strength_3d and penalty
# ─────────────────────────────────────────────────────────────────────────────

def test_close_strength_3d_strong_close():
    """Close at high → close_strength ≈ 1.0"""
    n = 10
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": np.ones(n) * 1000,
        "High": np.ones(n) * 1050,
        "Low":  np.ones(n) * 950,
        "Close": np.ones(n) * 1050,   # close at high
        "Volume": np.ones(n) * 1e6,
    }, index=dates)
    cs = _compute_close_strength_3d({"X.T": df}, ["X.T"])
    assert cs["X.T"] == pytest.approx(1.0, abs=0.01)


def test_close_strength_3d_weak_close():
    """Close at low → close_strength ≈ 0.0"""
    n = 10
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": np.ones(n) * 1000,
        "High": np.ones(n) * 1050,
        "Low":  np.ones(n) * 950,
        "Close": np.ones(n) * 950,   # close at low
        "Volume": np.ones(n) * 1e6,
    }, index=dates)
    cs = _compute_close_strength_3d({"X.T": df}, ["X.T"])
    assert cs["X.T"] == pytest.approx(0.0, abs=0.01)


def test_close_strength_penalty_applied():
    """Symbols with cs3d < 0.40 should have long_upper_shadow in penalties."""
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Weak close (cs ≈ 0.1 → clearly below 0.40)
    df_weak = pd.DataFrame({
        "Open": np.ones(n) * 1000, "High": np.ones(n) * 1200,
        "Low":  np.ones(n) * 900,  "Close": np.ones(n) * 920,
        "Volume": np.ones(n) * 1e6,
    }, index=dates)
    rsr_df = _make_rsr_df(["W.T"], n=n, base_rsr=55.0)
    results = screen_future_leaders(
        rsr_df=rsr_df, ohlcv={"W.T": df_weak},
        sector_map={"W.T": "電機精密"}, effective_capital=3_000_000.0,
    )
    assert len(results) == 1
    assert "long_upper_shadow" in results[0].penalties
    assert results[0].penalties["long_upper_shadow"] == pytest.approx(CLOSE_STRENGTH_PENALTY_MULT)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: single_day_jump penalty
# ─────────────────────────────────────────────────────────────────────────────

def test_single_day_jump_penalty():
    """Large single-day return with volume spike → single_day_jump penalty."""
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.ones(n) * 1000.0
    prices[-1] = 1050.0  # 5% jump on last day (> 4%)
    vol = np.ones(n) * 1_000_000.0
    vol[-1] = 3_500_000.0  # 3.5× average (> 2.5)
    df = pd.DataFrame({
        "Open": prices, "High": prices * 1.006, "Low": prices * 0.994,
        "Close": prices, "Volume": vol,
    }, index=dates)
    pens = _detect_penalties({"J.T": df}, ["J.T"])
    assert "single_day_jump" in pens["J.T"]
    assert pens["J.T"]["single_day_jump"] == pytest.approx(SINGLE_DAY_JUMP_MULT)


def test_single_day_jump_penalty_not_applied_without_volume():
    """Large return without volume spike → no penalty."""
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.ones(n) * 1000.0
    prices[-1] = 1050.0  # 5% jump
    vol = np.ones(n) * 1_000_000.0
    vol[-1] = 1_200_000.0  # only 1.2× — below 2.5× threshold
    df = pd.DataFrame({
        "Open": prices, "High": prices * 1.006, "Low": prices * 0.994,
        "Close": prices, "Volume": vol,
    }, index=dates)
    pens = _detect_penalties({"J2.T": df}, ["J2.T"])
    assert "single_day_jump" not in pens.get("J2.T", {})


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: gap_exhaustion penalty
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_exhaustion_penalty():
    """Gap up > 2% but close_strength < 0.35 → gap_exhaustion penalty."""
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.ones(n) * 1000.0
    prev_close = 1000.0
    gap_open   = 1030.0  # 3% gap (> 2%)
    close_today = 990.0  # weak close after gap (cs = (990-950)/(1100-950) = 40/150 = 0.267 < 0.35)
    prices[-1] = close_today
    prices[-2] = prev_close

    df = pd.DataFrame({
        "Open":  np.where(np.arange(n) == n - 1, gap_open, prices),
        "High":  np.where(np.arange(n) == n - 1, 1100.0, prices * 1.005),
        "Low":   np.where(np.arange(n) == n - 1, 950.0,  prices * 0.995),
        "Close": prices,
        "Volume": np.ones(n) * 1e6,
    }, index=dates)

    pens = _detect_penalties({"G.T": df}, ["G.T"])
    assert "gap_exhaustion" in pens.get("G.T", {}), "gap_exhaustion should be detected"
    assert pens["G.T"]["gap_exhaustion"] == pytest.approx(GAP_EXHAUSTION_MULT)


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: zone penalties reduce scores
# ─────────────────────────────────────────────────────────────────────────────

def test_mature_zone_score_reduced():
    """RSR 80 (mature) → score must be lower than RSR 60 (emerging) ceteris paribus."""
    sym = "Z.T"
    ohlcv_same = {sym: _make_ohlcv(n=35, base_price=1500.0, seed=10)}

    rsr_mature   = _make_rsr_df([sym], n=35, base_rsr=80.0)   # mature
    rsr_emerging = _make_rsr_df([sym], n=35, base_rsr=60.0)   # emerging

    r_mature   = screen_future_leaders(rsr_mature,   ohlcv_same, {sym: "電機精密"}, effective_capital=3_000_000.0)
    r_emerging = screen_future_leaders(rsr_emerging, ohlcv_same, {sym: "電機精密"}, effective_capital=3_000_000.0)

    # emerging zone should score higher (mature penalty = 0.70)
    assert r_emerging[0].future_leader_score >= r_mature[0].future_leader_score


def test_weak_zone_penalty_less_than_mature():
    """Weak zone penalty (0.80) > mature zone penalty (0.70) → weak scores higher than mature."""
    assert WEAK_ZONE_PENALTY > MATURE_ZONE_PENALTY


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: logging schema validation (to_dict keys)
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "symbol", "future_leader_score", "rsr_zone", "affordability_ok",
    "accumulation_pressure", "persistent_drift", "close_strength_3d",
    "consecutive_high_score_days", "penalties", "failure_flags",
    "observation_only",
    "is_governance_observation_candidate",
    "rsr_velocity_score", "accumulation_score", "persistent_drift_score",
    "close_strength_score", "rs_topix_delta_score", "affordability_score",
    "current_rsr", "rs_vs_topix_10d_delta", "lot_cost_jpy",
    "absolute_penalties",
}

def test_logging_schema_completeness():
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    for r in results:
        d = r.to_dict()
        missing = _REQUIRED_KEYS - set(d.keys())
        assert not missing, f"Missing keys in to_dict(): {missing}"


def test_to_dict_observation_only_is_true():
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    for r in results:
        assert r.to_dict()["observation_only"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: fail-closed log write
# ─────────────────────────────────────────────────────────────────────────────

def test_write_future_leader_log_writes_jsonl(tmp_path: Path):
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    log_path = tmp_path / "candidates.jsonl"
    write_future_leader_log(results, log_path, run_id="test_run")

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(results)

    for line in lines:
        rec = json.loads(line)
        assert "observation_ts" in rec
        assert "observation_date" in rec
        assert rec.get("observation_only") is True
        assert rec.get("run_id") == "test_run"


def test_write_future_leader_log_fail_closed(tmp_path: Path):
    """Write to non-writable path → IOError (fail-closed)."""
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    # Use a path whose parent is a file (cannot create dir)
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file")
    bad_path = blocker / "cannot_create_dir" / "log.jsonl"

    with pytest.raises(IOError):
        write_future_leader_log(results, bad_path, run_id="test")


def test_write_future_leader_log_empty_no_error(tmp_path: Path):
    """Empty results → no write, no error."""
    log_path = tmp_path / "candidates.jsonl"
    write_future_leader_log([], log_path, run_id="test")
    assert not log_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: consecutive tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_consecutive_tracking_increments(tmp_path: Path):
    """Prior score above threshold → consecutive_high_score_days increments."""
    sym = "CONSEC.T"
    ohlcv_c = {sym: _make_ohlcv(n=35, base_price=1000.0, seed=5)}
    rsr_c   = _make_rsr_df([sym], n=35, base_rsr=60.0, vel_profile={sym: 0.5})
    sector_c = {sym: "電機精密"}

    # Simulate yesterday: score was 75 (above 60 threshold), consecutive=1
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    prior_state = {sym: (75.0, 1)}

    results = screen_future_leaders(
        rsr_df=rsr_c, ohlcv=ohlcv_c, sector_map=sector_c,
        effective_capital=3_000_000.0,
        prior_day_state=prior_state,
    )
    r = results[0]
    if r.future_leader_score >= 60.0:
        assert r.consecutive_high_score_days == 2  # yesterday(1) + today
    else:
        assert r.consecutive_high_score_days == 0  # reset


def test_consecutive_reset_when_below_threshold():
    """If today's score < threshold → consecutive = 0."""
    sym = "RESET.T"
    # Use very short RSR history → velocity = 0 → low score
    ohlcv_r = {sym: _make_ohlcv(n=10, base_price=800.0, seed=7)}
    rsr_r   = _make_rsr_df([sym], n=10, base_rsr=20.0)  # RSR 20 → weak zone
    sector_r = {sym: "化学"}

    prior_state = {sym: (80.0, 3)}  # was high yesterday

    results = screen_future_leaders(
        rsr_df=rsr_r, ohlcv=ohlcv_r, sector_map=sector_r,
        effective_capital=3_000_000.0,
        prior_day_state=prior_state,
    )
    # With weak RSR and penalties, score should be low enough to reset
    r = results[0]
    assert r.rsr_zone == "weak"  # RSR 20 → weak
    # Weak zone penalty reduces score; may reset consecutive
    if r.future_leader_score < 60.0:
        assert r.consecutive_high_score_days == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: governance candidate conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_governance_observation_candidate_true_when_all_conditions_met():
    r = FutureLeaderScreenResult(
        symbol="GOOD.T",
        future_leader_score=70.0,        # ≥ 65
        rsr_zone="emerging",              # must be emerging
        affordability_ok=True,            # must be affordable
        accumulation_pressure=0.05,
        persistent_drift=0.5,
        close_strength_3d=0.6,
        consecutive_high_score_days=3,    # ≥ 2
        penalties={},
        failure_flags=[],
    )
    assert r.is_governance_observation_candidate is True


def test_governance_observation_candidate_false_score_too_low():
    r = FutureLeaderScreenResult(
        symbol="LOW.T",
        future_leader_score=50.0,         # < 65
        rsr_zone="emerging",
        affordability_ok=True,
        accumulation_pressure=0.05,
        persistent_drift=0.5,
        close_strength_3d=0.6,
        consecutive_high_score_days=3,
        penalties={},
        failure_flags=[],
    )
    assert r.is_governance_observation_candidate is False


def test_governance_observation_candidate_false_mature_zone():
    r = FutureLeaderScreenResult(
        symbol="LATE.T",
        future_leader_score=80.0,
        rsr_zone="mature",                # must be emerging
        affordability_ok=True,
        accumulation_pressure=0.05,
        persistent_drift=0.5,
        close_strength_3d=0.6,
        consecutive_high_score_days=5,
        penalties={},
        failure_flags=[],
    )
    assert r.is_governance_observation_candidate is False


def test_governance_observation_candidate_false_not_affordable():
    r = FutureLeaderScreenResult(
        symbol="EXP.T",
        future_leader_score=80.0,
        rsr_zone="emerging",
        affordability_ok=False,           # not affordable
        accumulation_pressure=0.05,
        persistent_drift=0.5,
        close_strength_3d=0.6,
        consecutive_high_score_days=5,
        penalties={},
        failure_flags=["affordability_fail"],
    )
    assert r.is_governance_observation_candidate is False


def test_governance_observation_candidate_false_low_consecutive():
    r = FutureLeaderScreenResult(
        symbol="FRESH.T",
        future_leader_score=80.0,
        rsr_zone="emerging",
        affordability_ok=True,
        accumulation_pressure=0.05,
        persistent_drift=0.5,
        close_strength_3d=0.6,
        consecutive_high_score_days=1,    # < 2
        penalties={},
        failure_flags=[],
    )
    assert r.is_governance_observation_candidate is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: empty universe returns empty list
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_universe_returns_empty():
    results = screen_future_leaders(
        rsr_df=pd.DataFrame(), ohlcv={}, sector_map={},
        effective_capital=3_000_000.0,
    )
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: single-symbol universe
# ─────────────────────────────────────────────────────────────────────────────

def test_single_symbol_universe():
    sym = "SOLO.T"
    ohlcv_s = {sym: _make_ohlcv(n=30, base_price=1800.0, seed=3)}
    rsr_s   = _make_rsr_df([sym], n=30, base_rsr=58.0)
    results = screen_future_leaders(
        rsr_df=rsr_s, ohlcv=ohlcv_s, sector_map={sym: "電機精密"},
        effective_capital=3_000_000.0,
    )
    assert len(results) == 1
    assert results[0].symbol == sym
    assert math.isfinite(results[0].future_leader_score)


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: format_future_leader_mail_section
# ─────────────────────────────────────────────────────────────────────────────

def test_format_future_leader_mail_section_contains_shadow_guarantee():
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    section = format_future_leader_mail_section(results, top_k=3)
    assert "observation_only=True" in section
    assert "FUTURE LEADER WATCH" in section


def test_format_future_leader_mail_section_empty():
    section = format_future_leader_mail_section([], top_k=5)
    assert "observation_only=True" in section
    assert "候補なし" in section


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: _load_prior_day_state
# ─────────────────────────────────────────────────────────────────────────────

def test_load_prior_day_state_returns_correct_scores(tmp_path: Path):
    log_path = tmp_path / "candidates.jsonl"
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today     = date.today().isoformat()

    records = [
        json.dumps({"observation_date": yesterday, "symbol": "A.T",
                    "future_leader_score": 72.0, "consecutive_high_score_days": 1}),
        json.dumps({"observation_date": yesterday, "symbol": "B.T",
                    "future_leader_score": 45.0, "consecutive_high_score_days": 0}),
        json.dumps({"observation_date": today, "symbol": "A.T",
                    "future_leader_score": 80.0, "consecutive_high_score_days": 2}),
    ]
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    state = _load_prior_day_state(log_path, today)
    assert "A.T" in state
    assert state["A.T"][0] == pytest.approx(72.0)  # yesterday's score, not today's
    assert state["A.T"][1] == 1                      # yesterday's consecutive
    assert "B.T" in state
    assert state["B.T"][0] == pytest.approx(45.0)


def test_load_prior_day_state_missing_file(tmp_path: Path):
    state = _load_prior_day_state(tmp_path / "nonexistent.jsonl", "2025-01-01")
    assert state == {}


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: results sorted by score descending
# ─────────────────────────────────────────────────────────────────────────────

def test_results_sorted_by_score():
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    scores = [r.future_leader_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 22: all scores in [0, 100]
# ─────────────────────────────────────────────────────────────────────────────

def test_all_scores_in_valid_range():
    results = screen_future_leaders(
        rsr_df=RSR_5, ohlcv=OHLCV_5, sector_map=SECTOR_5,
        effective_capital=3_000_000.0,
    )
    for r in results:
        assert 0.0 <= r.future_leader_score <= 100.0, (
            f"{r.symbol} score {r.future_leader_score} out of range"
        )
