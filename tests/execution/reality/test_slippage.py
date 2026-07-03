"""Tests: slippage.py — SlippageTracker, SlippageObservation."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.slippage import SlippageTracker, SlippageObservation
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, TS3, make_slippage_tracker


class TestSlippageObservation:
    def test_frozen(self):
        tracker = make_slippage_tracker()
        obs = tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        with pytest.raises(Exception):
            obs.slippage_bps = 0.0  # type: ignore

    def test_obs_id_prefix(self):
        tracker = make_slippage_tracker()
        obs = tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        assert obs.obs_id.startswith("slip_")

    def test_to_dict_from_dict_roundtrip(self):
        tracker = make_slippage_tracker()
        obs = tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        restored = SlippageObservation.from_dict(obs.to_dict())
        assert restored == obs


class TestSlippageTracker:
    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SlippageTracker(drift_threshold_bps=-1.0)

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            SlippageTracker(tail_percentile=1.1)

    def test_slippage_bps_buy(self):
        tracker = make_slippage_tracker()
        obs = tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        assert obs.slippage_bps == pytest.approx(10.0, abs=1e-4)

    def test_slippage_bps_sell(self):
        tracker = make_slippage_tracker()
        obs = tracker.record(SESSION, SYM_A, "SELL", 1000.0, 999.0, TS)
        assert obs.slippage_bps == pytest.approx(10.0, abs=1e-4)

    def test_slippage_zero_price_no_crash(self):
        tracker = make_slippage_tracker()
        obs = tracker.record(SESSION, SYM_A, "BUY", 0.0, 1001.0, TS)
        assert obs.slippage_bps == 0.0

    def test_mean_slippage_single(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        assert tracker.mean_slippage() == pytest.approx(10.0, abs=1e-4)

    def test_mean_slippage_multiple(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1010.0, TS)   # 100 bps
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1005.0, TS2)  # 50 bps
        assert tracker.mean_slippage() == pytest.approx(75.0, abs=1e-4)

    def test_tail_slippage(self):
        tracker = SlippageTracker(tail_percentile=0.95)
        for i in range(10):
            tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1000.0 + i, TS)
        assert tracker.tail_slippage() > tracker.mean_slippage()

    def test_slippage_volatility_zero_single(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        assert tracker.slippage_volatility() == 0.0

    def test_slippage_volatility_positive(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1010.0, TS2)
        assert tracker.slippage_volatility() > 0.0

    def test_has_structural_drift_false(self):
        tracker = SlippageTracker(drift_threshold_bps=100.0)
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        assert not tracker.has_structural_drift()

    def test_has_structural_drift_true(self):
        tracker = SlippageTracker(drift_threshold_bps=5.0)
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1010.0, TS)  # 100 bps > 5 threshold
        assert tracker.has_structural_drift()

    def test_observation_count(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1002.0, TS2)
        assert tracker.observation_count() == 2

    def test_observation_count_by_symbol(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        tracker.record(SESSION, SYM_B, "BUY", 500.0, 501.0, TS2)
        assert tracker.observation_count(SYM_A) == 1
        assert tracker.observation_count(SYM_B) == 1

    def test_all_observations_sorted(self):
        tracker = make_slippage_tracker()
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS2)
        tracker.record(SESSION, SYM_A, "BUY", 1000.0, 1001.0, TS)
        obs = tracker.all_observations()
        assert [o.observed_at for o in obs] == sorted([o.observed_at for o in obs])

    def test_mean_slippage_empty(self):
        tracker = make_slippage_tracker()
        assert tracker.mean_slippage() == 0.0

    def test_to_summary_keys(self):
        tracker = make_slippage_tracker()
        s = tracker.to_summary()
        assert "mean_slippage_bps" in s and "has_structural_drift" in s
