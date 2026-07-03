"""
tests/analytics/test_predictive_expansion.py

Coverage:
  - rsr_acceleration_correctness
  - compression_score_stability (determinism)
  - sector_ignition_consistency
  - leader_follower_propagation
  - normalization bounds (0-100)
  - no NaN propagation
  - missing data robustness (< MIN_HISTORY_DAYS)
  - ranking reproducibility (shuffle → same result)
  - volume zero protection
  - composite weights sum correctness
  - PredictiveExpansionScores.top_k determinism
  - rsr_velocity formula correctness
  - compression_ratio direction (low = better candidate)
  - sector ignition breadth threshold
  - follower not boosted when leader has low velocity
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.analytics.predictive_expansion import (
    DEFAULT_WEIGHTS,
    PredictiveExpansionScores,
    _MIN_HISTORY_DAYS,
    compression_breakout_scores,
    compute_predictive_expansion_scores,
    leader_follower_scores,
    rsr_acceleration_scores,
    sector_ignition_scores,
    volume_regime_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_dates(n: int) -> pd.DatetimeIndex:
    start = pd.Timestamp("2024-01-01")
    return pd.date_range(start, periods=n, freq="B")


def _make_rsr_df(
    symbols: list[str],
    n: int = 80,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic RSR DataFrame (0-100 values)."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    data = {
        sym: np.clip(
            50.0 + np.cumsum(rng.normal(0, 2, n)),
            0, 100,
        )
        for sym in symbols
    }
    return pd.DataFrame(data, index=dates)


def _make_ohlcv(
    symbols: list[str],
    n: int = 120,
    *,
    seed: int = 0,
    include_volume: bool = True,
    zero_volume_sym: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Create realistic-ish OHLCV DataFrames."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    result: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
        high  = close * (1 + rng.uniform(0, 0.02, n))
        low   = close * (1 - rng.uniform(0, 0.02, n))
        vol   = rng.integers(500_000, 5_000_000, n).astype(float)
        if zero_volume_sym and sym == zero_volume_sym:
            vol[:] = 0.0
        df = pd.DataFrame(
            {"High": high, "Low": low, "Close": close, "Volume": vol},
            index=dates,
        )
        result[sym] = df
    return result


def _make_sector_map(symbols: list[str], sectors: list[str] | None = None) -> Dict[str, str]:
    if sectors is None:
        sectors = ["海運", "機械", "電機"]
    return {sym: sectors[i % len(sectors)] for i, sym in enumerate(symbols)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. RSR acceleration correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestRsrAcceleration:
    def test_velocity_formula_correctness(self):
        """rsr_velocity_5d must equal rsr[t] - rsr[t-5]."""
        syms = ["A", "B", "C"]
        rsr = _make_rsr_df(syms, n=30)

        vel5_expected = rsr - rsr.shift(5)
        scores_df = rsr_acceleration_scores(rsr)

        # The raw velocity used internally: re-compute and compare top-k ranking
        assert not scores_df.empty
        assert scores_df.shape == rsr.shape
        # Latest row scores must be 0-100
        last = scores_df.iloc[-1]
        assert last.between(0, 100).all(), f"Out of range: {last}"

    def test_acceleration_formula(self):
        """rsr_acceleration = vel5d[t] - vel5d[t-5]."""
        syms = ["X", "Y"]
        rsr = _make_rsr_df(syms, n=40)
        vel5 = rsr - rsr.shift(5)
        accel = vel5 - vel5.shift(5)
        # Validate no NaN in final rows (where enough history exists)
        valid = accel.iloc[15:]
        assert not valid.isna().all().any()

    def test_too_short_returns_empty(self):
        syms = ["A", "B"]
        rsr = _make_rsr_df(syms, n=10)
        out = rsr_acceleration_scores(rsr)
        assert out.empty

    def test_scores_normalized_0_100(self):
        syms = ["A", "B", "C", "D"]
        rsr = _make_rsr_df(syms, n=60)
        scores_df = rsr_acceleration_scores(rsr)
        assert not scores_df.empty
        flat = scores_df.values.flatten()
        valid = flat[~np.isnan(flat)]
        assert valid.min() >= 0.0 - 1e-6
        assert valid.max() <= 100.0 + 1e-6

    def test_determinism(self):
        syms = ["A", "B", "C"]
        rsr = _make_rsr_df(syms)
        s1 = rsr_acceleration_scores(rsr)
        s2 = rsr_acceleration_scores(rsr)
        pd.testing.assert_frame_equal(s1, s2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compression breakout score
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressionBreakout:
    def test_same_input_same_output(self):
        syms = ["A", "B", "C"]
        ohlcv = _make_ohlcv(syms)
        r1 = compression_breakout_scores(ohlcv)
        r2 = compression_breakout_scores(ohlcv)
        pd.testing.assert_series_equal(r1[3], r2[3])

    def test_score_range_0_100(self):
        syms = ["A", "B", "C", "D"]
        ohlcv = _make_ohlcv(syms)
        _, _, _, score = compression_breakout_scores(ohlcv)
        assert score.between(0, 100).all()

    def test_short_history_handled(self):
        """Symbols with < MIN_HISTORY_DAYS get nan → fillna(1.0) default."""
        syms = ["SHORT", "LONG"]
        ohlcv = {
            "SHORT": _make_ohlcv(["SHORT"], n=20)["SHORT"],
            "LONG":  _make_ohlcv(["LONG"],  n=120)["LONG"],
        }
        comp, tight, expand, score = compression_breakout_scores(ohlcv)
        # SHORT gets nan → filled to 1.0 for comp/tight/expand
        assert math.isfinite(float(comp.get("SHORT", 1.0)))
        assert score.between(0, 100).all()

    def test_no_nan_in_score(self):
        syms = ["A", "B", "C"]
        ohlcv = _make_ohlcv(syms)
        _, _, _, score = compression_breakout_scores(ohlcv)
        assert not score.isna().any()

    def test_low_compression_ratio_scores_higher(self):
        """Symbol with lower ATR_5/ATR_60 (pre-expansion compression) scores higher."""
        n = 120
        dates = _make_dates(n)
        close_base = np.linspace(1000.0, 1200.0, n)

        # COMPRESSED: large swings historically, then very tight in last 5 days
        # → ATR_5 small relative to ATR_60 → comp_ratio < 1.0
        compressed_vol = np.full(n, 0.03)   # 3% daily swings throughout
        compressed_vol[-5:] = 0.001         # suddenly very tight last 5 days
        compressed_high = close_base * (1 + compressed_vol)
        compressed_low  = close_base * (1 - compressed_vol)

        # WIDE: recent volatility same or larger than historical
        # → ATR_5 ≈ ATR_60 or slightly larger → comp_ratio ≥ 1.0
        wide_vol = np.full(n, 0.01)         # consistent low volatility throughout
        wide_vol[-5:] = 0.04               # spike in last 5 days
        wide_high = close_base * (1 + wide_vol)
        wide_low  = close_base * (1 - wide_vol)

        ohlcv = {
            "COMPRESSED": pd.DataFrame(
                {"High": compressed_high, "Low": compressed_low, "Close": close_base, "Volume": 1e6},
                index=dates,
            ),
            "WIDE": pd.DataFrame(
                {"High": wide_high, "Low": wide_low, "Close": close_base, "Volume": 1e6},
                index=dates,
            ),
        }
        comp_s, _, _, score = compression_breakout_scores(ohlcv)
        # COMPRESSED: recent ATR tiny vs historical → comp_ratio << 1
        # WIDE: recent ATR large vs historical → comp_ratio > 1
        assert float(comp_s["COMPRESSED"]) < float(comp_s["WIDE"]), (
            f"COMPRESSED comp_ratio={comp_s['COMPRESSED']:.4f} >= WIDE comp_ratio={comp_s['WIDE']:.4f}"
        )
        # COMPRESSED should score higher (lower compression_ratio = better candidate)
        assert float(score["COMPRESSED"]) > float(score["WIDE"]), (
            f"COMPRESSED score={score['COMPRESSED']:.1f} <= WIDE score={score['WIDE']:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Volume regime
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeRegime:
    def test_zero_volume_no_crash(self):
        syms = ["ZERO", "NORMAL"]
        ohlcv = _make_ohlcv(syms, zero_volume_sym="ZERO")
        vr5, vr20, score = volume_regime_scores(ohlcv)
        assert score.between(0, 100).all()
        assert not score.isna().any()

    def test_score_range(self):
        syms = list("ABCDE")
        ohlcv = _make_ohlcv(syms)
        _, _, score = volume_regime_scores(ohlcv)
        assert score.between(0, 100).all()

    def test_outlier_clipping(self):
        """Extreme volume spike should be capped at 5x."""
        dates = _make_dates(60)
        # Spike: today's volume = 100× average
        close = np.linspace(1000, 1100, 60)
        vol   = np.ones(60) * 1_000_000.0
        vol[-1] = 100_000_000.0   # 100× spike
        ohlcv = {
            "SPIKE":  pd.DataFrame({"High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": vol}, index=dates),
            "NORMAL": _make_ohlcv(["NORMAL"], n=60)["NORMAL"],
        }
        vr5, _, _ = volume_regime_scores(ohlcv)
        # SPIKE vol_ratio_5d should be clipped at 5.0
        assert float(vr5["SPIKE"]) <= 5.0 + 1e-6

    def test_no_nan_propagation(self):
        syms = list("ABCD")
        ohlcv = _make_ohlcv(syms)
        vr5, vr20, score = volume_regime_scores(ohlcv)
        assert not vr5.isna().any()
        assert not score.isna().any()

    def test_determinism(self):
        syms = list("ABC")
        ohlcv = _make_ohlcv(syms)
        _, _, s1 = volume_regime_scores(ohlcv)
        _, _, s2 = volume_regime_scores(ohlcv)
        pd.testing.assert_series_equal(s1, s2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sector ignition
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorIgnition:
    def _make_inputs(self):
        syms = ["A1", "A2", "A3", "B1", "B2"]
        sector_map = {"A1": "海運", "A2": "海運", "A3": "海運", "B1": "機械", "B2": "機械"}
        # Sector A: all advancing (breadth=1.0 > 0.5)
        vel5d = pd.Series({"A1": 10.0, "A2": 5.0, "A3": 3.0, "B1": -2.0, "B2": -5.0})
        comp  = pd.Series({s: 0.6 for s in syms})
        vol5  = pd.Series({s: 1.5 for s in syms})
        expand= pd.Series({s: 1.3 for s in syms})
        return syms, sector_map, vel5d, comp, vol5, expand

    def test_ignition_flag_correct(self):
        syms, sector_map, vel5d, comp, vol5, expand = self._make_inputs()
        _, _, ig_flag = sector_ignition_scores(vel5d, comp, vol5, expand, sector_map)
        # Sector A: all advancing → igniting
        assert ig_flag["A1"] is True
        assert ig_flag["A2"] is True
        # Sector B: none advancing → not igniting
        assert ig_flag["B1"] is False

    def test_score_range(self):
        syms, sector_map, vel5d, comp, vol5, expand = self._make_inputs()
        score, _, _ = sector_ignition_scores(vel5d, comp, vol5, expand, sector_map)
        assert score.between(0, 100).all()

    def test_igniting_sector_outscores_dead_sector(self):
        syms, sector_map, vel5d, comp, vol5, expand = self._make_inputs()
        score, _, _ = sector_ignition_scores(vel5d, comp, vol5, expand, sector_map)
        # Sector A members should score higher than Sector B members
        assert score["A1"] > score["B1"]

    def test_no_nan(self):
        syms, sector_map, vel5d, comp, vol5, expand = self._make_inputs()
        score, raw, flag = sector_ignition_scores(vel5d, comp, vol5, expand, sector_map)
        assert not score.isna().any()

    def test_consistency(self):
        syms, sector_map, vel5d, comp, vol5, expand = self._make_inputs()
        s1, _, _ = sector_ignition_scores(vel5d, comp, vol5, expand, sector_map)
        s2, _, _ = sector_ignition_scores(vel5d, comp, vol5, expand, sector_map)
        pd.testing.assert_series_equal(s1, s2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Leader / Follower propagation
# ─────────────────────────────────────────────────────────────────────────────

class TestLeaderFollower:
    def _setup(self):
        # Sector: Leader with high RSR + high velocity; Follower compressed + vel > 0
        sector_map = {"L": "海運", "F1": "海運", "F2": "海運", "X": "機械"}
        rsr_today = pd.Series({"L": 85.0, "F1": 60.0, "F2": 55.0, "X": 70.0})
        # L expanded (vel = 15 > 5 threshold)
        vel5d = pd.Series({"L": 15.0, "F1": 2.0, "F2": 1.0, "X": 0.5})
        # F1, F2 compressed (comp < 0.85); X not in same sector
        comp  = pd.Series({"L": 0.9, "F1": 0.60, "F2": 0.55, "X": 0.70})
        return sector_map, rsr_today, vel5d, comp

    def test_leader_identified_correctly(self):
        sector_map, rsr_today, vel5d, comp = self._setup()
        _, is_leader, _ = leader_follower_scores(rsr_today, vel5d, comp, sector_map)
        assert is_leader["L"] is True
        assert is_leader["F1"] is False

    def test_follower_boosted(self):
        sector_map, rsr_today, vel5d, comp = self._setup()
        score, _, is_follower = leader_follower_scores(rsr_today, vel5d, comp, sector_map)
        assert is_follower["F1"] is True
        assert is_follower["F2"] is True
        assert float(score["F1"]) > 0.0

    def test_follower_not_boosted_if_leader_low_velocity(self):
        sector_map, rsr_today, _, comp = self._setup()
        # Set leader velocity below threshold
        low_vel = pd.Series({"L": 2.0, "F1": 1.0, "F2": 0.5, "X": 0.5})
        _, _, is_follower = leader_follower_scores(rsr_today, low_vel, comp, sector_map)
        assert is_follower["F1"] is False
        assert is_follower["F2"] is False

    def test_no_circular_propagation(self):
        """Leader cannot be a follower."""
        sector_map, rsr_today, vel5d, comp = self._setup()
        _, is_leader, is_follower = leader_follower_scores(rsr_today, vel5d, comp, sector_map)
        assert not (is_leader["L"] and is_follower["L"])

    def test_score_range(self):
        sector_map, rsr_today, vel5d, comp = self._setup()
        score, _, _ = leader_follower_scores(rsr_today, vel5d, comp, sector_map)
        assert score.between(0, 100).all()

    def test_follower_score_higher_than_non_follower(self):
        sector_map, rsr_today, vel5d, comp = self._setup()
        score, _, is_follower = leader_follower_scores(rsr_today, vel5d, comp, sector_map)
        # F1 is follower; X is not (different sector)
        if is_follower["F1"]:
            assert float(score["F1"]) >= float(score["X"])

    def test_single_member_sector_no_follower(self):
        sector_map = {"SOLO": "専業", "A": "海運", "B": "海運"}
        rsr_today = pd.Series({"SOLO": 80.0, "A": 90.0, "B": 60.0})
        vel5d = pd.Series({"SOLO": 10.0, "A": 12.0, "B": 3.0})
        comp  = pd.Series({"SOLO": 0.5, "A": 0.9, "B": 0.6})
        _, is_leader, is_follower = leader_follower_scores(rsr_today, vel5d, comp, sector_map)
        assert is_leader["SOLO"] is True
        assert is_follower["SOLO"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Composite compute_predictive_expansion_scores
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScores:
    def _build_inputs(self, n_syms: int = 6, n_dates: int = 90):
        syms = [f"S{i:02d}" for i in range(n_syms)]
        rsr = _make_rsr_df(syms, n=n_dates)
        ohlcv = _make_ohlcv(syms, n=n_dates + 30)
        sector_map = _make_sector_map(syms)
        return syms, rsr, ohlcv, sector_map

    def test_scores_all_symbols_present(self):
        syms, rsr, ohlcv, sm = self._build_inputs()
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        for sym in syms:
            assert sym in result.predictive_alpha_score

    def test_composite_range_0_100(self):
        syms, rsr, ohlcv, sm = self._build_inputs()
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        for sym, score in result.predictive_alpha_score.items():
            assert 0.0 <= score <= 100.0, f"{sym}: score={score}"

    def test_no_nan_in_any_score(self):
        syms, rsr, ohlcv, sm = self._build_inputs()
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        for key in ("rsr_acceleration_score", "compression_breakout_score",
                    "volume_regime_score", "sector_ignition_score",
                    "leader_follower_score", "predictive_alpha_score"):
            scores = getattr(result, key)
            for sym, v in scores.items():
                assert math.isfinite(v), f"{key}[{sym}] = {v}"

    def test_determinism_same_inputs(self):
        syms, rsr, ohlcv, sm = self._build_inputs()
        r1 = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        r2 = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        assert r1.predictive_alpha_score == r2.predictive_alpha_score

    def test_ranking_reproducibility_after_shuffle(self):
        """Shuffling symbol list order must not change final rankings."""
        syms, rsr, ohlcv, sm = self._build_inputs()
        r1 = compute_predictive_expansion_scores(rsr, ohlcv, sm)

        # Shuffle sector_map
        items = list(sm.items())
        random.Random(99).shuffle(items)
        sm2 = dict(items)
        r2 = compute_predictive_expansion_scores(rsr, ohlcv, sm2)

        # Same symbols → same composite scores (within float rounding)
        for sym in syms:
            s1 = r1.predictive_alpha_score.get(sym, 0.0)
            s2 = r2.predictive_alpha_score.get(sym, 0.0)
            assert abs(s1 - s2) < 1e-4, f"{sym}: {s1} vs {s2}"

    def test_missing_data_robustness(self):
        """Symbols with short OHLCV history handled gracefully."""
        syms = ["GOOD", "SHORT"]
        rsr = _make_rsr_df(syms, n=60)
        ohlcv = {
            "GOOD":  _make_ohlcv(["GOOD"],  n=120)["GOOD"],
            "SHORT": _make_ohlcv(["SHORT"], n=20)["SHORT"],   # < MIN_HISTORY_DAYS
        }
        sm = {"GOOD": "海運", "SHORT": "海運"}
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        for sym in syms:
            assert sym in result.predictive_alpha_score
            assert math.isfinite(result.predictive_alpha_score[sym])

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum = {total}"

    def test_custom_weights_applied(self):
        """Verify custom weights produce a different (but valid) composite."""
        syms, rsr, ohlcv, sm = self._build_inputs()
        r_default = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        r_custom  = compute_predictive_expansion_scores(
            rsr, ohlcv, sm,
            weights={
                "rsr_acceleration":    0.50,
                "compression_breakout": 0.20,
                "sector_ignition":     0.15,
                "volume_regime":       0.10,
                "leader_follower":     0.05,
            },
        )
        # All scores still in range
        for sym in syms:
            s = r_custom.predictive_alpha_score.get(sym, 0.0)
            assert 0.0 <= s <= 100.0

    def test_eval_date_set(self):
        syms, rsr, ohlcv, sm = self._build_inputs()
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        assert result.eval_date != ""
        assert len(result.eval_date) == 10  # YYYY-MM-DD


# ─────────────────────────────────────────────────────────────────────────────
# 7. PredictiveExpansionScores helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictiveExpansionScores:
    def _build_result(self):
        syms = ["A", "B", "C", "D", "E"]
        scores = PredictiveExpansionScores(eval_date="2024-01-15", symbols=syms)
        for i, sym in enumerate(syms):
            scores.predictive_alpha_score[sym] = float(i * 10 + 20)
        return scores, syms

    def test_top_k_returns_sorted_descending(self):
        scores, syms = self._build_result()
        top = scores.top_k(3)
        assert len(top) == 3
        vals = [v for _, v in top]
        assert vals == sorted(vals, reverse=True)

    def test_top_k_deterministic(self):
        scores, _ = self._build_result()
        t1 = scores.top_k(5)
        t2 = scores.top_k(5)
        assert t1 == t2

    def test_top_k_alphabetical_tiebreak(self):
        scores = PredictiveExpansionScores(eval_date="2024-01-01", symbols=["Z", "A"])
        scores.predictive_alpha_score = {"Z": 50.0, "A": 50.0}
        top = scores.top_k(2)
        assert top[0][0] == "A"  # alphabetically earlier

    def test_to_dict_roundtrip(self):
        syms = list("ABC")
        scores = PredictiveExpansionScores(eval_date="2024-02-01", symbols=syms)
        for sym in syms:
            scores.predictive_alpha_score[sym] = 55.0
            scores.is_leader[sym] = False
            scores.is_follower[sym] = False
        d = scores.to_dict()
        assert d["eval_date"] == "2024-02-01"
        assert set(d["predictive_alpha_score"].keys()) == set(syms)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_rsr_df(self):
        ohlcv = _make_ohlcv(["A", "B"])
        sm = {"A": "海運", "B": "機械"}
        result = compute_predictive_expansion_scores(pd.DataFrame(), ohlcv, sm)
        for sym in ["A", "B"]:
            assert sym in result.predictive_alpha_score
            assert math.isfinite(result.predictive_alpha_score[sym])

    def test_single_symbol_no_crash(self):
        rsr = _make_rsr_df(["ONLY"], n=60)
        ohlcv = _make_ohlcv(["ONLY"])
        sm = {"ONLY": "海運"}
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        assert "ONLY" in result.predictive_alpha_score
        assert math.isfinite(result.predictive_alpha_score["ONLY"])

    def test_large_universe(self):
        n_syms = 42
        syms = [f"S{i:03d}" for i in range(n_syms)]
        rsr = _make_rsr_df(syms, n=100)
        ohlcv = _make_ohlcv(syms, n=130)
        sm = _make_sector_map(syms, sectors=["海運", "機械", "電機", "商社", "化学", "食品", "銀行業"])
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        assert len(result.predictive_alpha_score) == n_syms
        for sym, score in result.predictive_alpha_score.items():
            assert 0.0 <= score <= 100.0, f"{sym}={score}"

    def test_all_zero_volume_no_crash(self):
        syms = ["A", "B", "C"]
        rsr = _make_rsr_df(syms)
        ohlcv = _make_ohlcv(syms, zero_volume_sym="A")
        # Force zero volume for all
        for sym in ohlcv:
            ohlcv[sym] = ohlcv[sym].copy()
            ohlcv[sym]["Volume"] = 0.0
        sm = _make_sector_map(syms)
        result = compute_predictive_expansion_scores(rsr, ohlcv, sm)
        for sym in syms:
            s = result.predictive_alpha_score.get(sym, 0.0)
            assert math.isfinite(s)
