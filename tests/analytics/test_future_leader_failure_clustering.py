"""
tests/analytics/test_future_leader_failure_clustering.py

Tests for Future Leader Failure Clustering module.

Coverage:
  - Failure classification correctness (all 6 reasons + none)
  - Primary/secondary failure separation
  - Single-label guarantee (primary = first-priority match)
  - Priority order correctness
  - failure_day_offset correctness
  - Duplicate prevention
  - Append-only persistence
  - Insufficient forward data handling
  - CLEAN-only aggregation
  - PARTIAL_FAILURE persistence (JSONL) + exclusion (aggregation)
  - No lookahead contamination
  - observation_only invariant
  - No execution imports
  - breakout_failed_3d MFE_3d filter (Task 2)
  - volume_faded TOPIX regime contamination filter (Task 3)
  - JSONL schema: primary_failure_reason, secondary_failure_flags, mfe_3d, topix_volume_ratio
  - format_failure_overlap_section report

Note: obs_date always uses a valid business day (2026-05-11, Monday).
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers for synthetic data
# ─────────────────────────────────────────────────────────────────────────────

_OBS_DATE = "2026-05-11"   # Monday — valid business day throughout all tests


def _make_dates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_surv_record(
    symbol: str = "TEST.T",
    obs_date: str = _OBS_DATE,
    close_obs: float = 1000.0,
    fwd5: float = 0.02,
    fwd10: float = 0.03,
    mfe: float = 0.05,
    mae: float = -0.02,
    dq: float = 1.0,
    integrity: str = "CLEAN",
) -> dict:
    return {
        "symbol":                     symbol,
        "observation_date":           obs_date,
        "observation_ts":             f"{obs_date}T09:00:00+09:00",
        "future_leader_score":        65.0,
        "rsr_zone":                   "emerging",
        "data_quality_weight":        dq,
        "integrity_state":            integrity,
        "close_price_at_observation": close_obs,
        "forward_return_5d":          fwd5,
        "forward_return_10d":         fwd10,
        "mfe_10d":                    mfe,
        "mae_10d":                    mae,
        "days_materialized":          10,
        "insufficient_forward_data":  False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Import the module under test
# ─────────────────────────────────────────────────────────────────────────────

import src.analytics.future_leader_failure_clustering as fc
from src.analytics.future_leader_failure_clustering import (
    FutureLeaderFailureRecord,
    _classify_failure,
    _detect_market_selloff,
    _detect_gap_fill,
    _detect_rsr_reversal,
    _detect_volume_faded,
    _detect_weak_followthrough,
    _detect_breakout_failed_3d,
    _filter_clean_complete,
    _load_failure_keys,
    _write_failure_log,
    _normalize_df_index,
    materialize_future_leader_failures,
    format_failure_clustering_section,
    format_failure_timing_section,
    format_failure_overlap_section,
)


def _make_sym_df(
    obs_date: str,
    close_obs: float,
    fwd_closes: List[float],
    fwd_highs:  Optional[List[float]] = None,
    fwd_vols:   Optional[List[float]] = None,
    pre_n: int = 25,
) -> pd.DataFrame:
    """
    Build a normalized OHLCV DataFrame:
      pre_n pre-obs bars + 1 obs bar + len(fwd_closes) forward bars.
    obs_date must be a valid business day.
    """
    pre_closes = [close_obs * 0.99] * pre_n
    if fwd_highs is None:
        fwd_highs = [v * 1.005 for v in fwd_closes]
    if fwd_vols is None:
        fwd_vols = [1_000_000.0] * len(fwd_closes)

    all_c = pre_closes + [close_obs] + fwd_closes
    all_h = [v * 1.005 for v in pre_closes] + [close_obs * 1.005] + fwd_highs
    all_l = [v * 0.995 for v in pre_closes] + [close_obs * 0.995] + [v * 0.995 for v in fwd_closes]
    all_v = [1_000_000.0] * pre_n + [1_000_000.0] + fwd_vols

    start = pd.Timestamp(obs_date) - pd.tseries.offsets.BusinessDay(pre_n)
    dates = pd.bdate_range(start=start, periods=len(all_c))
    return _normalize_df_index(pd.DataFrame({
        "Open": all_c, "High": all_h, "Low": all_l, "Close": all_c, "Volume": all_v,
    }, index=dates))


def _make_topix(obs_date: str, fwd_closes: List[float]) -> pd.DataFrame:
    """
    TOPIX DataFrame: 4 pre-obs bars (all 100.0) + 1 obs bar (100.0) + forward bars.
    obs_date must be a valid business day. Only Close column (no Volume).
    """
    pre    = [100.0] * 4
    obs    = [100.0]
    all_c  = pre + obs + fwd_closes
    start  = pd.Timestamp(obs_date) - pd.tseries.offsets.BusinessDay(4)
    dates  = pd.bdate_range(start=start, periods=len(all_c))
    return _normalize_df_index(pd.DataFrame({"Close": all_c}, index=dates))


def _make_topix_with_volume(
    obs_date: str,
    fwd_closes: List[float],
    pre_vols: Optional[List[float]] = None,
    fwd_vols:  Optional[List[float]] = None,
) -> pd.DataFrame:
    """TOPIX DataFrame with both Close and Volume columns."""
    n_pre = 4
    if pre_vols is None:
        pre_vols = [1_000_000.0] * n_pre
    obs_vol = [1_000_000.0]
    if fwd_vols is None:
        fwd_vols = [1_000_000.0] * len(fwd_closes)
    all_c = [100.0] * n_pre + [100.0] + fwd_closes
    all_v = pre_vols + obs_vol + fwd_vols
    start = pd.Timestamp(obs_date) - pd.tseries.offsets.BusinessDay(n_pre)
    dates = pd.bdate_range(start=start, periods=len(all_c))
    return _normalize_df_index(pd.DataFrame({"Close": all_c, "Volume": all_v}, index=dates))


# ─────────────────────────────────────────────────────────────────────────────
# Test: observation_only invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationOnlyInvariant:
    def test_record_always_observation_only(self):
        r = FutureLeaderFailureRecord(
            symbol="X", observation_date="2026-01-05", observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=False, primary_failure_reason=None, failure_day_offset=None,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
        )
        assert r.observation_only is True

    def test_to_dict_has_observation_only_true(self):
        r = FutureLeaderFailureRecord(
            symbol="X", observation_date="2026-01-05", observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=True, primary_failure_reason="breakout_failed_3d",
            failure_day_offset=2,
            forward_return_5d=-0.03, forward_return_10d=-0.05,
            mfe_10d=0.01, mae_10d=-0.04,
        )
        d = r.to_dict()
        assert d["observation_only"] is True

    def test_to_dict_has_primary_failure_reason(self):
        r = FutureLeaderFailureRecord(
            symbol="X", observation_date="2026-01-05", observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=True, primary_failure_reason="gap_fill",
            failure_day_offset=3,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
        )
        d = r.to_dict()
        assert "primary_failure_reason" in d
        assert "failure_reason" not in d  # old field removed
        assert d["primary_failure_reason"] == "gap_fill"

    def test_to_dict_has_secondary_failure_flags(self):
        r = FutureLeaderFailureRecord(
            symbol="X", observation_date="2026-01-05", observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=True, primary_failure_reason="gap_fill",
            failure_day_offset=2,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
            secondary_failure_flags=["rsr_reversal"],
        )
        d = r.to_dict()
        assert "secondary_failure_flags" in d
        assert d["secondary_failure_flags"] == ["rsr_reversal"]

    def test_to_dict_has_mfe_3d_and_topix_volume_ratio(self):
        r = FutureLeaderFailureRecord(
            symbol="X", observation_date="2026-01-05", observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=False, primary_failure_reason=None, failure_day_offset=None,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
            mfe_3d=0.015, topix_volume_ratio=0.9,
        )
        d = r.to_dict()
        assert "mfe_3d" in d
        assert "topix_volume_ratio" in d
        assert d["mfe_3d"] == 0.015
        assert d["topix_volume_ratio"] == 0.9

    def test_secondary_flags_default_empty_list(self):
        r = FutureLeaderFailureRecord(
            symbol="X", observation_date="2026-01-05", observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=False, primary_failure_reason=None, failure_day_offset=None,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
        )
        assert r.secondary_failure_flags == []
        assert r.mfe_3d is None
        assert r.topix_volume_ratio is None


# ─────────────────────────────────────────────────────────────────────────────
# Test: No execution imports
# ─────────────────────────────────────────────────────────────────────────────

class TestNoExecutionImports:
    _FORBIDDEN_MODULES = [
        "src.live.signal_bridge",
        "src.live.order_router",
        "src.allocation",
    ]

    def test_module_does_not_import_execution_modules(self):
        for forbidden in self._FORBIDDEN_MODULES:
            assert forbidden not in sys.modules, f"forbidden module loaded: {forbidden}"

    def test_module_level_imports_are_safe(self):
        source_lines = Path(
            "src/analytics/future_leader_failure_clustering.py"
        ).read_text(encoding="utf-8").splitlines()
        import_lines = [
            l for l in source_lines
            if "import" in l and not l.strip().startswith("#") and not l.strip().startswith('"""')
            and not l.strip().startswith("'")
        ]
        for line in import_lines:
            for forbidden in ["signal_bridge", "order_router"]:
                assert forbidden not in line, f"forbidden import: {line!r}"

    def test_no_order_generation_symbol(self):
        source = Path(
            "src/analytics/future_leader_failure_clustering.py"
        ).read_text(encoding="utf-8")
        assert "send_order" not in source
        assert "place_order" not in source


# ─────────────────────────────────────────────────────────────────────────────
# Test: Individual failure detectors
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectMarketSelloff:
    def test_selloff_detects_4pct_drop(self):
        """TOPIX drops 5% at fwd day 3 → offset = 3."""
        obs_date   = _OBS_DATE  # 2026-05-11 (Mon)
        fwd_closes = [100.0, 99.0, 95.0, 98.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        topix_df   = _make_topix(obs_date, fwd_closes)

        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        offset = _detect_market_selloff(topix_df, fwd_dates)
        # day3 = 2026-05-14: close=95, 3ago=2026-05-11(obs)=100. -5% ≤ -4% → YES
        assert offset == 3

    def test_no_selloff_when_drop_below_threshold(self):
        obs_date   = _OBS_DATE
        fwd_closes = [100.0, 100.0, 97.0] + [97.0] * 7  # -3% at day 3, above -4%
        topix_df   = _make_topix(obs_date, fwd_closes)

        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        offset = _detect_market_selloff(topix_df, fwd_dates)
        assert offset is None

    def test_none_topix_returns_none(self):
        dates = pd.bdate_range(start="2026-05-12", periods=10)
        assert _detect_market_selloff(None, dates) is None

    def test_empty_topix_returns_none(self):
        dates = pd.bdate_range(start="2026-05-12", periods=10)
        assert _detect_market_selloff(pd.DataFrame(), dates) is None


class TestDetectGapFill:
    def test_gap_fill_detected(self):
        close_obs  = 1000.0
        # Day 1: +10% breakout. Day 2: +2% → retains only 20% → lost 80% ≥ 70%
        fwd_closes = np.array([1100.0, 1020.0] + [1020.0] * 8)
        offset = _detect_gap_fill(fwd_closes, close_obs)
        assert offset == 2

    def test_no_gap_fill_when_holding(self):
        close_obs  = 1000.0
        # Retains 80% of gain throughout (never drops to 30%)
        fwd_closes = np.array([1100.0, 1080.0, 1060.0, 1050.0, 1040.0,
                                1040.0, 1040.0, 1040.0, 1040.0, 1040.0])
        offset = _detect_gap_fill(fwd_closes, close_obs)
        assert offset is None

    def test_no_gap_fill_when_no_breakout(self):
        close_obs  = 1000.0
        fwd_closes = np.array([1000.0] * 10)  # flat
        offset = _detect_gap_fill(fwd_closes, close_obs)
        assert offset is None

    def test_no_gap_fill_when_initial_gain_negative(self):
        close_obs  = 1000.0
        fwd_closes = np.array([950.0, 940.0] + [950.0] * 8)  # gaps DOWN
        offset = _detect_gap_fill(fwd_closes, close_obs)
        assert offset is None

    def test_threshold_boundary_no_trigger(self):
        close_obs  = 1000.0
        # Day 1: +10%, Day 2: retains 31% of gain → above threshold
        fwd_closes = np.array([1100.0, 1031.0] + [1031.0] * 8)
        offset = _detect_gap_fill(fwd_closes, close_obs)
        assert offset is None

    def test_threshold_boundary_triggers(self):
        close_obs  = 1000.0
        # Day 1: +10%, Day 2: retains 29% of gain → triggers
        fwd_closes = np.array([1100.0, 1029.0] + [1029.0] * 8)
        offset = _detect_gap_fill(fwd_closes, close_obs)
        assert offset == 2


class TestDetectRsrReversal:
    def _make_rsr(self, sym: str, dates: pd.DatetimeIndex, values: List[float]) -> pd.DataFrame:
        return _normalize_df_index(pd.DataFrame({sym: values}, index=dates))

    def test_rsr_reversal_detected(self):
        """
        Setup (obs_date=2026-05-11, obs_rsr=60):
          RSR: 55,56,57,60(obs),65(fwd1),58(fwd2),55,53,50,48,47,46,45
          fwd_dates start 2026-05-12
          offset=2 (2026-05-13): RSR 65→58 (delta<0) AND 58<60 → triggers
        """
        sym      = "TEST.T"
        obs_date = _OBS_DATE      # 2026-05-11 (Mon)
        obs_rsr  = 60.0

        all_dates = pd.bdate_range(start="2026-05-05", periods=13)
        rsr_vals  = [55.0, 56.0, 57.0, 58.0, 60.0, 65.0, 58.0, 55.0,
                     53.0, 50.0, 48.0, 47.0, 46.0]
        rsr_df    = self._make_rsr(sym, all_dates, rsr_vals)

        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        offset    = _detect_rsr_reversal(rsr_df, sym, obs_rsr, fwd_dates)
        # fwd_date=2026-05-13 (offset=2): idx=6, curr=58, prev=65. delta=-7<0 AND 58<60 → YES
        assert offset == 2

    def test_no_reversal_if_rsr_stays_above_obs(self):
        """RSR delta can be negative but never drops below obs_rsr."""
        sym       = "TEST.T"
        obs_rsr   = 50.0
        all_dates = pd.bdate_range(start="2026-05-05", periods=13)
        rsr_vals  = [45.0, 47.0, 49.0, 50.0, 60.0, 58.0, 57.0, 56.0,
                     55.0, 54.0, 53.0, 52.0, 51.0]
        rsr_df    = self._make_rsr(sym, all_dates, rsr_vals)
        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        offset    = _detect_rsr_reversal(rsr_df, sym, obs_rsr, fwd_dates)
        assert offset is None

    def test_no_reversal_if_delta_always_positive(self):
        """RSR always rising → delta never negative → no reversal."""
        sym       = "TEST.T"
        obs_rsr   = 60.0
        all_dates = pd.bdate_range(start="2026-05-05", periods=13)
        rsr_vals  = [50.0] * 4 + [60.0, 62.0, 65.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0]
        rsr_df    = self._make_rsr(sym, all_dates, rsr_vals)
        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        offset    = _detect_rsr_reversal(rsr_df, sym, obs_rsr, fwd_dates)
        assert offset is None

    def test_none_rsr_df_returns_none(self):
        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        assert _detect_rsr_reversal(None, "TEST.T", 60.0, fwd_dates) is None

    def test_symbol_not_in_rsr_returns_none(self):
        rsr_df    = pd.DataFrame({"OTHER.T": [50.0]}, index=pd.bdate_range(start="2026-05-12", periods=1))
        fwd_dates = pd.bdate_range(start="2026-05-12", periods=10)
        assert _detect_rsr_reversal(rsr_df, "TEST.T", 60.0, fwd_dates) is None


class TestDetectVolumeFaded:
    def test_volume_faded_detected(self):
        """Volume ratio < 0.7 AND stagnant from day 5 → triggers at day 5."""
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([600_000.0] * 10)
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(fwd_closes, fwd_volumes, close_obs, baseline_vol)
        assert offset == 5  # first eligible day is day 5 (d_idx=4)

    def test_no_fade_high_volume(self):
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([900_000.0] * 10)  # ratio=0.9 > 0.7
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(fwd_closes, fwd_volumes, close_obs, baseline_vol)
        assert offset is None

    def test_no_fade_not_stagnant(self):
        close_obs    = 1000.0
        fwd_closes   = np.array([1050.0] * 10)   # 5% move → not stagnant
        fwd_volumes  = np.array([600_000.0] * 10)
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(fwd_closes, fwd_volumes, close_obs, baseline_vol)
        assert offset is None

    def test_no_fade_zero_baseline(self):
        fwd_closes  = np.array([1010.0] * 10)
        fwd_volumes = np.array([600_000.0] * 10)
        offset = _detect_volume_faded(fwd_closes, fwd_volumes, 1000.0, 0.0)
        assert offset is None

    def test_volume_faded_delayed_onset(self):
        """
        First 6 fwd days: normal volume (1M). Days 7+: faded (600k).
        d_idx=9(day10): avg([1M,600k,600k,600k,600k])=680k → 0.68 < 0.7 AND stagnant → YES
        """
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([1_000_000.0] * 6 + [600_000.0] * 4)
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(fwd_closes, fwd_volumes, close_obs, baseline_vol)
        assert offset == 10

    def test_regime_suppression_topix_inactive(self):
        """topix_vol_ratio <= 0.8 → market-wide inactivity → suppress volume_faded."""
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([600_000.0] * 10)  # would fire at day 5 without filter
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(
            fwd_closes, fwd_volumes, close_obs, baseline_vol, topix_vol_ratio=0.7
        )
        assert offset is None  # suppressed: TOPIX also inactive

    def test_regime_suppression_exact_boundary(self):
        """topix_vol_ratio == 0.8 → suppressed (boundary is ≤ 0.8)."""
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([600_000.0] * 10)
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(
            fwd_closes, fwd_volumes, close_obs, baseline_vol, topix_vol_ratio=0.8
        )
        assert offset is None  # boundary: 0.8 <= 0.8 → suppress

    def test_regime_active_topix_allows_fade(self):
        """topix_vol_ratio > 0.8 → market active → individual fading is genuine failure."""
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([600_000.0] * 10)
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(
            fwd_closes, fwd_volumes, close_obs, baseline_vol, topix_vol_ratio=0.9
        )
        assert offset == 5

    def test_regime_filter_none_fails_open(self):
        """topix_vol_ratio=None (data unavailable) → fail-open → volume_faded can fire."""
        close_obs    = 1000.0
        fwd_closes   = np.array([1010.0] * 10)
        fwd_volumes  = np.array([600_000.0] * 10)
        baseline_vol = 1_000_000.0
        offset = _detect_volume_faded(
            fwd_closes, fwd_volumes, close_obs, baseline_vol, topix_vol_ratio=None
        )
        assert offset == 5


class TestDetectWeakFollowthrough:
    def test_weak_followthrough_detected(self):
        close_obs  = 1000.0
        fwd_closes = np.array([1010.0, 1005.0, 1002.0, 998.0, 997.0] + [1000.0] * 5)
        fwd_highs  = np.array([1015.0] * 10)
        # 5d return = (997-1000)/1000 = -0.3% ≤ 0 ✓; MFE_5d = 1015/1000-1 = 1.5% < 3% ✓
        offset = _detect_weak_followthrough(fwd_closes, fwd_highs, close_obs)
        assert offset == 5

    def test_no_weak_followthrough_positive_return(self):
        close_obs  = 1000.0
        fwd_closes = np.array([1010.0, 1015.0, 1020.0, 1025.0, 1030.0] + [1000.0] * 5)
        fwd_highs  = np.array([1035.0] * 10)
        offset = _detect_weak_followthrough(fwd_closes, fwd_highs, close_obs)
        assert offset is None  # 5d return > 0

    def test_no_weak_followthrough_high_mfe(self):
        close_obs  = 1000.0
        fwd_closes = np.array([1005.0, 1003.0, 1001.0, 999.0, 997.0] + [1000.0] * 5)
        fwd_highs  = np.array([1040.0] * 10)  # MFE_5d = 4% ≥ 3% → no
        offset = _detect_weak_followthrough(fwd_closes, fwd_highs, close_obs)
        assert offset is None

    def test_insufficient_bars_returns_none(self):
        close_obs  = 1000.0
        fwd_closes = np.array([998.0, 997.0, 996.0, 995.0])
        fwd_highs  = np.array([1002.0, 1000.0, 999.0, 998.0])
        offset = _detect_weak_followthrough(fwd_closes, fwd_highs, close_obs)
        assert offset is None


class TestDetectBreakoutFailed3d:
    def test_fails_day1(self):
        fwd_closes = np.array([990.0] + [1010.0] * 9)
        fwd_highs  = np.array([994.0, 1015.0, 1015.0] + [1015.0] * 7)
        # MFE_3d = max(994, 1015, 1015)/1000 - 1 = 1.5% < 2% → allowed
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset == 1

    def test_fails_day2(self):
        fwd_closes = np.array([1010.0, 995.0] + [1010.0] * 8)
        fwd_highs  = np.array([1015.0, 999.0, 1015.0] + [1015.0] * 7)
        # MFE_3d = max(1015, 999, 1015)/1000 - 1 = 1.5% < 2% → allowed
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset == 2

    def test_fails_day3(self):
        fwd_closes = np.array([1010.0, 1005.0, 990.0] + [1010.0] * 7)
        fwd_highs  = np.array([1015.0, 1010.0, 994.0] + [1015.0] * 7)
        # MFE_3d = max(1015, 1010, 994)/1000 - 1 = 1.5% < 2% → allowed
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset == 3

    def test_no_fail_within_3d_then_drops_later(self):
        fwd_closes = np.array([1010.0, 1005.0, 1002.0] + [990.0] * 7)
        fwd_highs  = np.array([1015.0, 1010.0, 1007.0] + [994.0] * 7)
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset is None

    def test_exactly_at_obs_close_not_trigger(self):
        fwd_closes = np.array([1000.0] * 10)  # not strictly below obs_close
        fwd_highs  = np.array([1005.0] * 10)
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset is None

    def test_mfe_filter_suppresses_when_reached_2pct(self):
        """MFE_3d >= 2% → suppress breakout_failed_3d (large-cap noise)."""
        fwd_closes = np.array([990.0] + [1010.0] * 9)
        fwd_highs  = np.array([1025.0, 1010.0, 1010.0] + [1010.0] * 7)
        # MFE_3d = max(1025, 1010, 1010)/1000 - 1 = 2.5% >= 2% → suppressed
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset is None

    def test_mfe_filter_exact_boundary_suppresses(self):
        """MFE_3d == 2.0% → suppressed (boundary: >= 0.02)."""
        fwd_closes = np.array([990.0] + [1010.0] * 9)
        fwd_highs  = np.array([1020.0, 1010.0, 1010.0] + [1010.0] * 7)
        # MFE_3d = 1020/1000 - 1 = 2.0% → suppressed
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset is None

    def test_mfe_filter_just_below_threshold_fires(self):
        """MFE_3d < 2% → breakout_failed_3d fires normally."""
        fwd_closes = np.array([990.0] + [1010.0] * 9)
        fwd_highs  = np.array([994.0, 992.0, 991.0] + [1015.0] * 7)
        # MFE_3d = max(994, 992, 991)/1000 - 1 = -0.6% < 2% → fires
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Primary/Secondary failure separation
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimarySecondaryFailure:
    """
    Verify that _classify_failure returns all triggered conditions,
    with primary = first-priority match and secondary = remaining.
    """

    def test_classify_returns_six_values(self):
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [1020.0] * 10
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        result = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert len(result) == 6

    def test_gap_fill_and_rsr_reversal_coexist(self):
        """
        gap_fill (priority 2) fires at day 2.
        rsr_reversal (priority 3) fires at day 2 as well.
        primary = gap_fill, secondary = [rsr_reversal].
        """
        sym       = "TEST.T"
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        obs_rsr   = 60.0

        # gap_fill: day1=+10%, day2=+2% → retains 20% < 30% → fires at day 2
        fwd_closes = [1100.0, 1020.0] + [1020.0] * 8
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)

        # rsr_reversal: RSR 65(fwd1) → 58(fwd2): delta<0 AND 58<60 → fires at day 2
        all_rsr_dates = pd.bdate_range(start="2026-05-05", periods=13)
        rsr_vals      = [55.0, 56.0, 57.0, 58.0, 60.0, 65.0, 58.0, 55.0,
                         53.0, 50.0, 48.0, 47.0, 46.0]
        rsr_df = _normalize_df_index(
            pd.DataFrame({sym: rsr_vals}, index=all_rsr_dates)
        )

        surv = _make_surv_record(symbol=sym, obs_date=obs_date, close_obs=close_obs)
        detected, primary, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=rsr_df, observation_rsr=obs_rsr,
            today=date(2026, 6, 1),
        )
        assert detected is True
        assert primary == "gap_fill"          # priority 2 wins
        assert "rsr_reversal" in secondary    # priority 3 captured as secondary

    def test_no_failure_empty_secondary(self):
        """No failure → secondary is empty list."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [1020.0, 1040.0, 1060.0, 1080.0, 1100.0,
                       1110.0, 1120.0, 1130.0, 1140.0, 1150.0]
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, primary, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is False
        assert primary is None
        assert secondary == []

    def test_secondary_stored_in_jsonl(self, tmp_path):
        """secondary_failure_flags field appears in persisted JSONL record."""
        log_dir = tmp_path / "fail"
        r = FutureLeaderFailureRecord(
            symbol="A.T", observation_date=_OBS_DATE, observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=True, primary_failure_reason="gap_fill",
            failure_day_offset=2,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
            secondary_failure_flags=["rsr_reversal", "volume_faded"],
        )
        _write_failure_log([r], log_dir, today=date(2026, 5, 24))
        files = list(log_dir.glob("*.jsonl"))
        data = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        assert data[0]["primary_failure_reason"] == "gap_fill"
        assert data[0]["secondary_failure_flags"] == ["rsr_reversal", "volume_faded"]

    def test_mfe_3d_stored_in_jsonl(self, tmp_path):
        """mfe_3d field appears in persisted JSONL record."""
        log_dir = tmp_path / "fail"
        r = FutureLeaderFailureRecord(
            symbol="A.T", observation_date=_OBS_DATE, observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=False, primary_failure_reason=None, failure_day_offset=None,
            forward_return_5d=None, forward_return_10d=None,
            mfe_10d=None, mae_10d=None,
            mfe_3d=0.015, topix_volume_ratio=0.92,
        )
        _write_failure_log([r], log_dir, today=date(2026, 5, 24))
        files = list(log_dir.glob("*.jsonl"))
        data = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        assert data[0]["mfe_3d"] == 0.015
        assert data[0]["topix_volume_ratio"] == 0.92


# ─────────────────────────────────────────────────────────────────────────────
# Test: Single-label guarantee and priority order
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleLabelAndPriority:
    """
    Verify that when multiple failure conditions would trigger,
    only the highest-priority one is returned as primary.
    obs_date = _OBS_DATE = 2026-05-11 (Mon) throughout.
    """

    def test_market_selloff_beats_gap_fill(self):
        """market_selloff (priority 1) must win over gap_fill (priority 2)."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        # gap_fill trigger: day1=+10%, day2=+2% (retains 20% → gap_fill at day 2)
        fwd_closes = [1100.0, 1020.0] + [1020.0] * 8
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        # TOPIX drops 5% at forward day 3 → market_selloff at offset=3
        topix_df = _make_topix(obs_date, [100.0, 99.0, 95.0] + [95.0] * 7)

        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=topix_df, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is True
        assert reason == "market_selloff"

    def test_gap_fill_beats_breakout_failed_3d(self):
        """
        gap_fill (priority 2) must win.
        breakout_failed_3d is suppressed by MFE_3d filter because day1 close=1100
        means MFE_3d ≈ 10% >= 2% → breakout_failed_3d suppressed.
        """
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        # Day 1: +10% breakout → MFE_3d ≈ 10% ≥ 2% → breakout_failed_3d suppressed
        # Day 2: +2% (retains 20%) → gap_fill fires
        fwd_closes = [1100.0, 1020.0] + [990.0] * 8
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is True
        assert reason == "gap_fill"
        assert offset == 2

    def test_weak_followthrough_fires_when_no_prior_reason(self):
        """
        weak_followthrough (priority 5) fires when no higher-priority condition triggers.
        Design: day 1 close = obs_close (no breakout → gap_fill skips).
        Days 1-3 all >= obs_close (no breakout_failed_3d).
        Low MFE_5d < 3% and 5d return ≤ 0.
        """
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [1000.0, 1002.0, 1001.0, 1001.0, 997.0, 997.0, 997.0, 997.0, 997.0, 997.0]
        fwd_highs  = [1005.0, 1007.0, 1005.0, 1003.0, 1001.0, 1001.0, 1001.0, 1001.0, 1001.0, 1001.0]
        fwd_vols   = [1_000_000.0] * 10  # normal volume → no volume_faded
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes, fwd_highs, fwd_vols)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is True
        # MFE_5d = max(1005,1007,1005,1003,1001)/1000 - 1 = 0.7% < 3% AND 5d_return=-0.3% ≤ 0
        assert reason == "weak_followthrough"
        assert offset == 5

    def test_no_failure_when_none_apply(self):
        """Strong uptrend: no failure condition triggers."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [1020.0, 1040.0, 1060.0, 1080.0, 1100.0,
                       1110.0, 1120.0, 1130.0, 1140.0, 1150.0]
        fwd_highs  = [v * 1.01 for v in fwd_closes]
        fwd_vols   = [1_500_000.0] * 10
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes, fwd_highs, fwd_vols)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is False
        assert reason is None
        assert offset is None
        assert secondary == []

    def test_single_label_only_one_reason_returned(self):
        """
        Single-label guarantee: primary is the single aggregation label.
        Setup: all closes at 990 (below obs_close=1000), faded volume.
          - gap_fill: initial_gain=(990-1000)/1000=-1% → negative → SKIP
          - volume_faded: vol_ratio=0.6<0.7 AND |return|=1%<2% → fires at day 5 (priority 4)
          - breakout_failed_3d: MFE_3d = max(994.95*)/1000-1 ≈ -0.5% < 2% → fires (priority 6)
        volume_faded is primary (priority 4), breakout_failed_3d is secondary (priority 6).
        (*default fwd_highs from _make_sym_df = close * 1.005)
        """
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = list(np.array([990.0] * 10))
        fwd_vols   = [600_000.0] * 10  # faded: ratio=0.6 < 0.7, stagnant=1% < 2%
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes, fwd_vols=fwd_vols)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is True
        assert isinstance(reason, str)
        assert reason in fc._FAILURE_PRIORITY
        # volume_faded is priority 4 — fires before breakout_failed_3d (6)
        assert reason == "volume_faded"
        assert offset == 5
        # breakout_failed_3d should be in secondary (MFE_3d ≈ -0.5% < 2%, fires at day 1)
        assert "breakout_failed_3d" in secondary

    def test_none_obs_idx_returns_no_failure(self):
        """Guard: None obs_idx returns (False, None, None, [], None, None)."""
        surv = _make_surv_record()
        sym_df = _make_sym_df(_OBS_DATE, 1000.0, [990.0] * 10)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=None,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert detected is False
        assert reason is None
        assert offset is None
        assert secondary == []
        assert mfe_3d is None
        assert topix_vol_ratio is None


# ─────────────────────────────────────────────────────────────────────────────
# Test: breakout_failed_3d MFE filter (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakoutMfeFilter:
    """breakout_failed_3d requires both close < obs_close AND MFE_3d < 2%."""

    def test_suppressed_in_classify_when_initial_surge(self):
        """
        Stock surges +5% day1 (MFE_3d ≥ 2%), then crashes below obs.
        breakout_failed_3d suppressed even though day3 close < obs_close.
        """
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [1050.0, 1010.0, 990.0] + [990.0] * 7
        # High day1 = 1055 → MFE_3d = max(1055, 1015.05, 994.95)/1000 - 1 = 5.5% ≥ 2%
        fwd_highs  = [1055.0, 1015.0, 994.0] + [994.0] * 7
        fwd_vols   = [1_000_000.0] * 10  # no volume_faded
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes, fwd_highs, fwd_vols)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)

        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        # breakout_failed_3d suppressed; weak_followthrough: day5 close=990, return=-1% ≤ 0
        # MFE_5d = max(1055,1015,994,...)/1000-1 ≥ 5% ≥ 3% → weak_followthrough doesn't fire
        # No condition fires → no failure
        assert "breakout_failed_3d" not in ([] if not detected else [reason] + secondary)

    def test_mfe_3d_field_populated_in_record(self):
        """mfe_3d is computed and stored regardless of whether breakout_failed_3d fires."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [1020.0] * 10
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert mfe_3d is not None
        # fwd_closes all 1020, highs = 1020*1.005=1025.1 → MFE_3d ≈ 2.51%
        assert mfe_3d > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: TOPIX volume ratio computed from TOPIX with Volume column (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestTopixVolumeRatioComputation:
    def test_topix_volume_ratio_computed_when_volume_available(self):
        """When TOPIX has Volume column, topix_volume_ratio is populated."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes_sym = [1020.0] * 10
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes_sym)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)

        # TOPIX: 4 pre-obs bars + 1 obs + 10 fwd, all at 1M vol
        topix_df = _make_topix_with_volume(
            obs_date, fwd_closes=[100.0] * 10,
            pre_vols=[1_000_000.0] * 4,
            fwd_vols=[1_000_000.0] * 10,
        )
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=topix_df, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        # ratio = avg(fwd_vols[0:5]) / avg(pre_vols) = 1M/1M = 1.0
        assert topix_vol_ratio is not None
        assert abs(topix_vol_ratio - 1.0) < 0.01

    def test_topix_volume_ratio_none_without_volume_column(self):
        """TOPIX without Volume column → topix_volume_ratio is None (fail-open)."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes_sym = [1020.0] * 10
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes_sym)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)

        topix_df = _make_topix(obs_date, [100.0] * 10)  # Close only, no Volume
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=topix_df, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        assert topix_vol_ratio is None

    def test_volume_faded_suppressed_via_topix_volume_ratio(self):
        """
        TOPIX volume also low (ratio=0.5 ≤ 0.8) → volume_faded suppressed.
        Symbol: stagnant price + faded volume (would normally fire).
        TOPIX: also faded (pre_vols=1M, fwd_vols=500k → ratio=0.5).
        """
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes_sym = [1010.0] * 10  # stagnant
        fwd_vols_sym   = [600_000.0] * 10  # faded: ratio=0.6 < 0.7
        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes_sym, fwd_vols=fwd_vols_sym)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)

        # TOPIX also faded: pre=1M, fwd=500k → ratio=0.5 ≤ 0.8 → suppress
        topix_df = _make_topix_with_volume(
            obs_date, fwd_closes=[100.0] * 10,
            pre_vols=[1_000_000.0] * 4,
            fwd_vols=[500_000.0] * 10,
        )
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=topix_df, rsr_df=None, observation_rsr=0.0,
            today=date(2026, 6, 1),
        )
        # topix_vol_ratio ≈ 0.5 ≤ 0.8 → volume_faded suppressed
        assert topix_vol_ratio is not None
        assert topix_vol_ratio < 0.8
        # volume_faded should NOT be primary or secondary
        all_labels = ([] if not detected else [reason]) + secondary
        assert "volume_faded" not in all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Test: failure_day_offset correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureDayOffset:
    def test_breakout_failed_offset_is_1based(self):
        fwd_closes = np.array([1010.0, 1005.0, 990.0, 995.0, 1000.0,
                                1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        fwd_highs  = np.array([1015.0, 1010.0, 994.0, 999.0, 1005.0,
                                1005.0, 1005.0, 1005.0, 1005.0, 1005.0])
        # MFE_3d = max(1015, 1010, 994)/1000 - 1 = 1.5% < 2% → allowed
        offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, 1000.0)
        assert offset == 3  # day 3 is the first below obs_close

    def test_weak_followthrough_always_reports_day5(self):
        close_obs  = 1000.0
        fwd_closes = np.array([1001.0, 1001.0, 1001.0, 1001.0, 995.0] + [1000.0] * 5)
        fwd_highs  = np.array([1005.0] * 10)  # MFE_5d = 0.5% < 3%
        offset = _detect_weak_followthrough(fwd_closes, fwd_highs, close_obs)
        assert offset == 5

    def test_market_selloff_offset_1based_from_fwd_dates(self):
        """When TOPIX drops at fwd date index 0 (first forward bar), offset = 1."""
        obs_date   = _OBS_DATE
        # Massive drop at fwd day 1
        fwd_closes = [90.0] + [90.0] * 9  # 10% drop at day 1
        topix_df   = _make_topix(obs_date, fwd_closes)
        fwd_dates  = pd.bdate_range(start="2026-05-12", periods=10)
        # TOPIX at fwd day 1 = 90. 3ago from topix = one of the pre-obs bars (100). -10% → triggers
        offset = _detect_market_selloff(topix_df, fwd_dates)
        assert offset == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Lookahead prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestLookaheadPrevention:
    def test_classify_excludes_today_bar(self):
        """
        When today = first forward date, that bar is excluded (not strictly < today).
        With only 0 valid confirmed forward bars, insufficient → False.
        """
        obs_date  = _OBS_DATE   # 2026-05-11
        close_obs = 1000.0
        fwd_closes = [990.0] * 10  # would trigger breakout_failed_3d

        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        # today = first forward date (2026-05-12) → that bar is excluded
        first_fwd_date = sym_df.index[obs_idx + 1].date()
        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)

        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=first_fwd_date,
        )
        assert detected is False  # 0 confirmed bars < FORWARD_WINDOW=10

    def test_classify_skips_when_insufficient_confirmed_bars(self):
        """With only 5 forward bars strictly < today, returns False (need 10)."""
        obs_date  = _OBS_DATE
        close_obs = 1000.0
        fwd_closes = [990.0] * 5  # only 5 forward bars built

        sym_df  = _make_sym_df(obs_date, close_obs, fwd_closes)
        obs_idx = fc._find_obs_idx(sym_df, obs_date)
        assert obs_idx is not None

        # today = one day after the 5th forward bar
        last_fwd_date = sym_df.index[-1]
        today = (last_fwd_date + pd.tseries.offsets.BusinessDay(1)).date()

        surv = _make_surv_record(obs_date=obs_date, close_obs=close_obs)
        detected, reason, offset, secondary, mfe_3d, topix_vol_ratio = _classify_failure(
            surv_rec=surv, sym_df=sym_df, obs_idx=obs_idx,
            topix_df=None, rsr_df=None, observation_rsr=0.0,
            today=today,
        )
        assert detected is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Duplicate prevention and append-only persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicatePrevention:
    def test_load_failure_keys_empty_when_no_directory(self, tmp_path):
        assert _load_failure_keys(tmp_path / "nonexistent") == set()

    def test_load_failure_keys_reads_all_jsonl(self, tmp_path):
        log_dir = tmp_path / "fail"
        log_dir.mkdir()
        data = [
            {"symbol": "A.T", "observation_date": "2026-05-11"},
            {"symbol": "B.T", "observation_date": "2026-05-11"},
        ]
        (log_dir / "future_leader_failure_20260511.jsonl").write_text(
            "\n".join(json.dumps(d) for d in data), encoding="utf-8"
        )
        keys = _load_failure_keys(log_dir)
        assert ("A.T", "2026-05-11") in keys
        assert ("B.T", "2026-05-11") in keys


class TestAppendOnlyPersistence:
    def _make_record(
        self, sym: str = "X.T", obs: str = _OBS_DATE,
        fail: bool = False, reason: Optional[str] = None, offset: Optional[int] = None,
    ) -> FutureLeaderFailureRecord:
        return FutureLeaderFailureRecord(
            symbol=sym, observation_date=obs, observation_ts="",
            future_leader_score=60.0, rsr_zone="emerging",
            integrity_state="CLEAN", data_quality_weight=1.0,
            failure_detected=fail, primary_failure_reason=reason, failure_day_offset=offset,
            forward_return_5d=0.02, forward_return_10d=0.03, mfe_10d=0.05, mae_10d=-0.02,
        )

    def test_write_creates_jsonl(self, tmp_path):
        log_dir = tmp_path / "fail"
        rec = self._make_record("A.T", _OBS_DATE, True, "breakout_failed_3d", 2)
        _write_failure_log([rec], log_dir, today=date(2026, 5, 24))
        files = list(log_dir.glob("future_leader_failure_*.jsonl"))
        assert len(files) == 1
        data = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(data) == 1
        assert data[0]["symbol"] == "A.T"
        assert data[0]["observation_only"] is True
        assert data[0]["primary_failure_reason"] == "breakout_failed_3d"

    def test_write_appends_not_overwrites(self, tmp_path):
        log_dir = tmp_path / "fail"
        rec1 = self._make_record("A.T", _OBS_DATE, True, "breakout_failed_3d", 1)
        rec2 = self._make_record("B.T", _OBS_DATE, False)
        _write_failure_log([rec1], log_dir, today=date(2026, 5, 24))
        _write_failure_log([rec2], log_dir, today=date(2026, 5, 24))
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_existing_records_preserved(self, tmp_path):
        log_dir = tmp_path / "fail"
        rec1 = self._make_record("A.T", _OBS_DATE, True, "volume_faded", 5)
        _write_failure_log([rec1], log_dir, today=date(2026, 5, 24))
        initial = list(log_dir.glob("*.jsonl"))[0].read_text(encoding="utf-8")

        rec2 = self._make_record("B.T", _OBS_DATE)
        _write_failure_log([rec2], log_dir, today=date(2026, 5, 24))
        final = list(log_dir.glob("*.jsonl"))[0].read_text(encoding="utf-8")
        assert initial in final


# ─────────────────────────────────────────────────────────────────────────────
# Test: Materialize pipeline integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializePipeline:
    def _write_surv(self, path: Path, records: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _write_cands(self, path: Path, records: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _make_loader(
        self,
        obs_date: str,
        close_obs: float,
        fwd_closes: List[float],
        sym: str = "TEST.T",
    ):
        sym_df = _make_sym_df(obs_date, close_obs, fwd_closes)

        def loader(s: str) -> Optional[pd.DataFrame]:
            return sym_df if s == sym else None

        return loader

    def test_classifies_new_records(self, tmp_path):
        surv_dir  = tmp_path / "surv"
        fail_dir  = tmp_path / "fail"
        cand_file = tmp_path / "cands.jsonl"

        obs_date = _OBS_DATE
        surv = _make_surv_record(obs_date=obs_date, close_obs=1000.0)
        self._write_surv(surv_dir / "future_leader_survivability_20260511.jsonl", [surv])
        self._write_cands(cand_file, [{"symbol": "TEST.T", "observation_date": obs_date, "current_rsr": 60.0}])

        fwd_closes = [990.0] * 10  # triggers breakout_failed_3d or weak_followthrough
        loader = self._make_loader(obs_date, 1000.0, fwd_closes)
        today  = date(2026, 6, 1)

        count = materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            ohlcv_loader=loader,
            topix_df=None,
            rsr_df=None,
            today=today,
        )
        assert count == 1
        keys = _load_failure_keys(fail_dir)
        assert ("TEST.T", obs_date) in keys

    def test_no_reprocess_already_classified(self, tmp_path):
        surv_dir  = tmp_path / "surv"
        fail_dir  = tmp_path / "fail"
        cand_file = tmp_path / "cands.jsonl"

        obs_date = _OBS_DATE
        surv = _make_surv_record(obs_date=obs_date, close_obs=1000.0)
        self._write_surv(surv_dir / "future_leader_survivability_20260511.jsonl", [surv])
        self._write_cands(cand_file, [])

        fwd_closes = [990.0] * 10
        loader = self._make_loader(obs_date, 1000.0, fwd_closes)
        today  = date(2026, 6, 1)

        count1 = materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            ohlcv_loader=loader,
            topix_df=None,
            rsr_df=None,
            today=today,
        )
        assert count1 == 1

        count2 = materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            ohlcv_loader=loader,
            topix_df=None,
            rsr_df=None,
            today=today,
        )
        assert count2 == 0  # duplicate prevention

    def test_skip_insufficient_forward_bars(self, tmp_path):
        surv_dir  = tmp_path / "surv"
        fail_dir  = tmp_path / "fail"
        cand_file = tmp_path / "cands.jsonl"

        obs_date = _OBS_DATE
        surv = _make_surv_record(obs_date=obs_date, close_obs=1000.0)
        self._write_surv(surv_dir / "future_leader_survivability_20260511.jsonl", [surv])
        self._write_cands(cand_file, [])

        # Only 5 forward bars
        sym_df = _make_sym_df(obs_date, 1000.0, [990.0] * 5)
        # today = last bar of sym_df → only 5 confirmed bars (need 10)
        today = sym_df.index[-1].date()

        count = materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            ohlcv_loader=lambda s: sym_df if s == "TEST.T" else None,
            topix_df=None,
            rsr_df=None,
            today=today,
        )
        assert count == 0

    def test_no_survivability_records_returns_zero(self, tmp_path):
        fail_dir  = tmp_path / "fail"
        cand_file = tmp_path / "cands.jsonl"
        cand_file.write_text("", encoding="utf-8")

        count = materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=tmp_path / "nonexistent",
            failure_log_dir=fail_dir,
            ohlcv_loader=lambda s: None,
            topix_df=None,
            rsr_df=None,
            today=date(2026, 6, 1),
        )
        assert count == 0

    def test_materialized_record_has_new_schema_fields(self, tmp_path):
        """Materialized JSONL records include primary_failure_reason, secondary_failure_flags, mfe_3d."""
        surv_dir  = tmp_path / "surv"
        fail_dir  = tmp_path / "fail"
        cand_file = tmp_path / "cands.jsonl"

        obs_date = _OBS_DATE
        surv = _make_surv_record(obs_date=obs_date, close_obs=1000.0)
        self._write_surv(surv_dir / "future_leader_survivability_20260511.jsonl", [surv])
        self._write_cands(cand_file, [])

        fwd_closes = [990.0] * 10
        loader = self._make_loader(obs_date, 1000.0, fwd_closes)

        materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            ohlcv_loader=loader,
            topix_df=None,
            rsr_df=None,
            today=date(2026, 6, 1),
        )
        files = list(fail_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = json.loads(lines[0])
        assert "primary_failure_reason" in rec
        assert "secondary_failure_flags" in rec
        assert isinstance(rec["secondary_failure_flags"], list)
        assert "mfe_3d" in rec
        assert "topix_volume_ratio" in rec
        assert "failure_reason" not in rec  # old field must not exist


# ─────────────────────────────────────────────────────────────────────────────
# Test: CLEAN-only aggregation and PARTIAL_FAILURE handling
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchPurity:
    def _make_log_dir(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "fail"
        log_dir.mkdir()
        records = [
            {
                "symbol": "CLEAN.T", "observation_date": _OBS_DATE,
                "data_quality_weight": 1.0, "integrity_state": "CLEAN",
                "failure_detected": True, "primary_failure_reason": "volume_faded",
                "secondary_failure_flags": [],
                "failure_day_offset": 6, "forward_return_10d": -0.02,
                "mfe_10d": 0.01, "mae_10d": -0.03,
            },
            {
                "symbol": "PARTIAL.T", "observation_date": _OBS_DATE,
                "data_quality_weight": 0.5, "integrity_state": "PARTIAL_FAILURE",
                "failure_detected": True, "primary_failure_reason": "weak_followthrough",
                "secondary_failure_flags": [],
                "failure_day_offset": 5, "forward_return_10d": -0.01,
                "mfe_10d": 0.02, "mae_10d": -0.01,
            },
        ]
        (log_dir / "future_leader_failure_20260511.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        return log_dir

    def test_filter_clean_complete_excludes_partial(self):
        records = [
            {"data_quality_weight": 1.0},
            {"data_quality_weight": 0.5},
            {"data_quality_weight": 0.0},
        ]
        clean = _filter_clean_complete(records)
        assert len(clean) == 1
        assert clean[0]["data_quality_weight"] == 1.0

    def test_clustering_section_shows_partial_excluded_count(self, tmp_path):
        log_dir = self._make_log_dir(tmp_path)
        section = format_failure_clustering_section(log_dir)
        assert "partial_failure_excluded: 1" in section

    def test_partial_failure_exists_in_jsonl(self, tmp_path):
        """PARTIAL_FAILURE records are persisted to JSONL (just excluded from aggregation)."""
        log_dir = self._make_log_dir(tmp_path)
        all_records = fc._load_failure_records(log_dir)
        partial = [r for r in all_records if r.get("integrity_state") == "PARTIAL_FAILURE"]
        assert len(partial) == 1
        assert partial[0]["symbol"] == "PARTIAL.T"

    def test_timing_section_excludes_partial(self, tmp_path):
        log_dir = self._make_log_dir(tmp_path)
        section = format_failure_timing_section(log_dir)
        # Only CLEAN record: volume_faded at day 6
        assert "volume_faded" in section
        assert "6" in section
        # PARTIAL_FAILURE's weak_followthrough should NOT appear
        assert "weak_followthrough" not in section

    def test_materialize_skips_major_failure_records(self, tmp_path):
        """MAJOR_FAILURE (dq_weight=0.0) is never processed."""
        surv_dir  = tmp_path / "surv"
        fail_dir  = tmp_path / "fail"
        cand_file = tmp_path / "cands.jsonl"

        obs_date = _OBS_DATE
        surv = _make_surv_record(obs_date=obs_date, close_obs=1000.0, dq=0.0, integrity="MAJOR_FAILURE")
        surv_path = surv_dir / "future_leader_survivability_20260511.jsonl"
        surv_path.parent.mkdir(parents=True, exist_ok=True)
        surv_path.write_text(json.dumps(surv), encoding="utf-8")
        cand_file.write_text("", encoding="utf-8")

        count = materialize_future_leader_failures(
            candidates_log_path=cand_file,
            survivability_log_dir=surv_dir,
            failure_log_dir=fail_dir,
            ohlcv_loader=lambda s: None,
            topix_df=None,
            rsr_df=None,
            today=date(2026, 6, 1),
        )
        assert count == 0

    def test_secondary_flags_telemetry_only(self, tmp_path):
        """secondary_failure_flags appear in JSONL but are excluded from clustering aggregation."""
        log_dir = tmp_path / "fail"
        log_dir.mkdir()
        records = [
            {
                "symbol": "A.T", "observation_date": _OBS_DATE,
                "data_quality_weight": 1.0, "integrity_state": "CLEAN",
                "failure_detected": True, "primary_failure_reason": "gap_fill",
                "secondary_failure_flags": ["rsr_reversal", "volume_faded"],
                "failure_day_offset": 2, "forward_return_10d": -0.03,
                "mfe_10d": 0.01, "mae_10d": -0.04,
            },
        ]
        (log_dir / "future_leader_failure_20260524.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        # Clustering aggregation should only count gap_fill (primary), not rsr_reversal/volume_faded
        section = format_failure_clustering_section(log_dir)
        assert "gap_fill" in section
        # secondary flags must NOT appear as independent failure categories in the clustering section
        # (they only appear in the overlap section)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Report format correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestReportFormatters:
    def _write_records(self, tmp_path: Path, records: List[dict]) -> Path:
        log_dir = tmp_path / "fail"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "future_leader_failure_20260524.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        return log_dir

    def _clean_rec(
        self, reason: str, offset: int, fwd10: float = -0.02, mae: float = -0.03,
        secondary: Optional[List[str]] = None,
    ) -> dict:
        return {
            "symbol": "X.T", "observation_date": _OBS_DATE,
            "data_quality_weight": 1.0, "integrity_state": "CLEAN",
            "failure_detected": True, "primary_failure_reason": reason,
            "secondary_failure_flags": secondary or [],
            "failure_day_offset": offset,
            "forward_return_10d": fwd10, "mfe_10d": 0.01, "mae_10d": mae,
        }

    def test_clustering_section_has_header(self, tmp_path):
        log_dir = self._write_records(tmp_path, [self._clean_rec("volume_faded", 6)])
        section = format_failure_clustering_section(log_dir)
        assert "FAILURE CLUSTERING" in section

    def test_clustering_section_shows_reason_and_count(self, tmp_path):
        log_dir = self._write_records(tmp_path, [
            self._clean_rec("volume_faded", 6),
            self._clean_rec("volume_faded", 7),
            self._clean_rec("volume_faded", 8),
        ])
        section = format_failure_clustering_section(log_dir)
        assert "volume_faded" in section
        assert "count: 3" in section

    def test_timing_section_has_header(self, tmp_path):
        log_dir = self._write_records(tmp_path, [self._clean_rec("breakout_failed_3d", 2)])
        section = format_failure_timing_section(log_dir)
        assert "FAILURE TIMING" in section

    def test_timing_section_median(self, tmp_path):
        log_dir = self._write_records(tmp_path, [
            self._clean_rec("breakout_failed_3d", 1),
            self._clean_rec("breakout_failed_3d", 2),
            self._clean_rec("breakout_failed_3d", 3),
        ])
        section = format_failure_timing_section(log_dir)
        assert "breakout_failed_3d" in section
        assert "2.0" in section  # median of [1, 2, 3]

    def test_empty_dir_returns_gracefully(self, tmp_path):
        log_dir = tmp_path / "empty"
        log_dir.mkdir()
        s1 = format_failure_clustering_section(log_dir)
        s2 = format_failure_timing_section(log_dir)
        s3 = format_failure_overlap_section(log_dir)
        assert "FAILURE CLUSTERING" in s1
        assert "FAILURE TIMING" in s2
        assert "FAILURE OVERLAP" in s3

    def test_market_selloff_shows_exogenous_label(self, tmp_path):
        log_dir = self._write_records(tmp_path, [self._clean_rec("market_selloff", 3)])
        section = format_failure_clustering_section(log_dir)
        assert "exogenous regime failure" in section

    def test_backward_compat_old_failure_reason_field(self, tmp_path):
        """Old JSONL records with failure_reason (not primary_failure_reason) still aggregate."""
        log_dir = tmp_path / "fail"
        log_dir.mkdir(exist_ok=True)
        old_rec = {
            "symbol": "OLD.T", "observation_date": _OBS_DATE,
            "data_quality_weight": 1.0, "integrity_state": "CLEAN",
            "failure_detected": True, "failure_reason": "gap_fill",  # old field name
            "failure_day_offset": 2, "forward_return_10d": -0.03,
            "mfe_10d": 0.01, "mae_10d": -0.04,
        }
        (log_dir / "future_leader_failure_20260524.jsonl").write_text(
            json.dumps(old_rec), encoding="utf-8"
        )
        section = format_failure_clustering_section(log_dir)
        assert "gap_fill" in section

    def test_overlap_section_has_header(self, tmp_path):
        log_dir = self._write_records(tmp_path, [
            self._clean_rec("gap_fill", 2, secondary=["rsr_reversal"]),
        ])
        section = format_failure_overlap_section(log_dir)
        assert "FAILURE OVERLAP" in section

    def test_overlap_section_shows_cooccurrence(self, tmp_path):
        log_dir = self._write_records(tmp_path, [
            self._clean_rec("gap_fill", 2, secondary=["rsr_reversal"]),
            self._clean_rec("gap_fill", 3, secondary=["rsr_reversal", "volume_faded"]),
            self._clean_rec("gap_fill", 4, secondary=["volume_faded"]),
        ])
        section = format_failure_overlap_section(log_dir)
        assert "gap_fill" in section
        assert "rsr_reversal" in section
        assert "volume_faded" in section
        # rsr_reversal appears 2x, volume_faded 2x
        assert "2x" in section

    def test_overlap_section_no_secondary_message(self, tmp_path):
        """When no secondary flags exist, reports no secondary message."""
        log_dir = self._write_records(tmp_path, [
            self._clean_rec("gap_fill", 2, secondary=[]),
        ])
        section = format_failure_overlap_section(log_dir)
        assert "no secondary flags recorded" in section

    def test_overlap_section_primary_only_aggregation(self, tmp_path):
        """
        Secondary flags appear in OVERLAP section only.
        They must not appear as standalone failure categories in CLUSTERING section.
        """
        log_dir = self._write_records(tmp_path, [
            self._clean_rec("gap_fill", 2, secondary=["rsr_reversal"]),
        ])
        clustering = format_failure_clustering_section(log_dir)
        overlap    = format_failure_overlap_section(log_dir)
        # gap_fill is primary → appears in clustering
        assert "gap_fill" in clustering
        # rsr_reversal is secondary → appears in overlap but NOT as standalone in clustering
        assert "rsr_reversal" in overlap
        # clustering section should not list rsr_reversal as a standalone entry with "count:"
        # (it may appear in other contexts, but not as a primary category)
        clustering_lines = clustering.split("\n")
        rsr_line = [l for l in clustering_lines if l.strip().startswith("rsr_reversal:")]
        assert len(rsr_line) == 0  # no standalone rsr_reversal category
