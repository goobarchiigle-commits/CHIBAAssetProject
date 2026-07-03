"""Tests: fill_quality.py — FillQualityAnalyzer, FillObservation."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.fill_quality import FillQualityAnalyzer, FillObservation
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, TS3, make_fill_analyzer


class TestFillObservation:
    def test_frozen(self):
        analyzer = make_fill_analyzer()
        obs = analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        with pytest.raises(Exception):
            obs.fill_ratio = 0.0  # type: ignore

    def test_obs_id_prefix(self):
        analyzer = make_fill_analyzer()
        obs = analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        assert obs.obs_id.startswith("fill_")

    def test_to_dict_from_dict_roundtrip(self):
        analyzer = make_fill_analyzer()
        obs = analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        restored = FillObservation.from_dict(obs.to_dict())
        assert restored == obs


class TestFillQualityAnalyzer:
    def test_invalid_burst_window_raises(self):
        with pytest.raises(ValueError):
            FillQualityAnalyzer(reject_burst_window=0)

    def test_invalid_burst_threshold_raises(self):
        with pytest.raises(ValueError):
            FillQualityAnalyzer(reject_burst_threshold=0.0)

    def test_full_fill_ratio_1(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        assert analyzer.fill_ratio() == pytest.approx(1.0, abs=1e-9)

    def test_partial_fill_ratio(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 50, False, TS)
        assert analyzer.fill_ratio() == pytest.approx(0.5, abs=1e-6)

    def test_rejected_fill_ratio_zero(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 0, True, TS)
        assert analyzer.fill_ratio() == pytest.approx(0.0, abs=1e-9)

    def test_partial_flag_set(self):
        analyzer = make_fill_analyzer()
        obs = analyzer.record(SESSION, SYM_A, "BUY", 100, 60, False, TS)
        assert obs.partial is True

    def test_partial_flag_not_set_full(self):
        analyzer = make_fill_analyzer()
        obs = analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        assert obs.partial is False

    def test_partial_fill_rate(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 50, False, TS)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS2)
        assert analyzer.partial_fill_rate() == pytest.approx(0.5, abs=1e-6)

    def test_reject_rate(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 0, True, TS)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS2)
        assert analyzer.reject_rate() == pytest.approx(0.5, abs=1e-6)

    def test_no_reject_burst_below_threshold(self):
        analyzer = FillQualityAnalyzer(reject_burst_window=5, reject_burst_threshold=0.50)
        for _ in range(4):
            analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        assert not analyzer.has_reject_burst()

    def test_has_reject_burst(self):
        analyzer = FillQualityAnalyzer(reject_burst_window=4, reject_burst_threshold=0.50)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 0, True, TS)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 0, True, TS2)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS3)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 0, True, TS)
        assert analyzer.has_reject_burst()

    def test_fill_ratio_empty_returns_one(self):
        analyzer = make_fill_analyzer()
        assert analyzer.fill_ratio() == 1.0

    def test_reject_rate_empty_returns_zero(self):
        analyzer = make_fill_analyzer()
        assert analyzer.reject_rate() == 0.0

    def test_observation_count(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS2)
        assert analyzer.observation_count() == 2

    def test_observation_count_by_symbol(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        analyzer.record(SESSION, SYM_B, "SELL", 50, 50, False, TS2)
        assert analyzer.observation_count(SYM_A) == 1

    def test_all_observations_sorted(self):
        analyzer = make_fill_analyzer()
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS2)
        analyzer.record(SESSION, SYM_A, "BUY", 100, 100, False, TS)
        obs = analyzer.all_observations()
        ts_list = [o.observed_at for o in obs]
        assert ts_list == sorted(ts_list)

    def test_to_summary_keys(self):
        analyzer = make_fill_analyzer()
        s = analyzer.to_summary()
        assert "fill_ratio" in s and "reject_rate" in s and "has_reject_burst" in s
