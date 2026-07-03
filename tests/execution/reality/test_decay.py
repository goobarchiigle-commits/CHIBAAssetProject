"""Tests: decay.py — AlphaDecayTracker, AlphaDecayObservation."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.decay import (
    AlphaDecayTracker, AlphaDecayObservation,
    DECAY_RESEARCH_ALPHA, DECAY_EXECUTION_INDUCED, ALL_DECAY_TYPES,
)
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, TS3, make_decay_tracker


def record_obs(tracker, sym=SYM_A, dtype=DECAY_RESEARCH_ALPHA,
               research=100.0, realized=80.0, ts=TS):
    return tracker.record(SESSION, sym, dtype, research, realized, ts)


class TestAlphaDecayObservation:
    def test_frozen(self):
        t = make_decay_tracker()
        obs = record_obs(t)
        with pytest.raises(Exception):
            obs.decay_bps = 0.0  # type: ignore

    def test_obs_id_prefix(self):
        t = make_decay_tracker()
        obs = record_obs(t)
        assert obs.obs_id.startswith("adcy_")

    def test_to_dict_from_dict_roundtrip(self):
        t = make_decay_tracker()
        obs = record_obs(t)
        restored = AlphaDecayObservation.from_dict(obs.to_dict())
        assert restored == obs


class TestAlphaDecayTracker:
    def test_all_decay_types(self):
        assert len(ALL_DECAY_TYPES) == 2

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            AlphaDecayTracker(decay_alert_threshold=1.1)

    def test_unknown_decay_type_raises(self):
        t = make_decay_tracker()
        with pytest.raises(ValueError):
            t.record(SESSION, SYM_A, "BAD_TYPE", 100.0, 80.0, TS)

    def test_decay_bps_computed(self):
        t = make_decay_tracker()
        obs = record_obs(t, research=100.0, realized=70.0)
        assert obs.decay_bps == pytest.approx(30.0, abs=1e-9)

    def test_decay_fraction_computed(self):
        t = make_decay_tracker()
        obs = record_obs(t, research=100.0, realized=60.0)
        assert obs.decay_fraction == pytest.approx(0.4, abs=1e-6)

    def test_decay_fraction_zero_research(self):
        t = make_decay_tracker()
        obs = record_obs(t, research=0.0, realized=0.0)
        assert obs.decay_fraction == 0.0

    def test_decay_fraction_capped_at_one(self):
        t = make_decay_tracker()
        obs = record_obs(t, research=10.0, realized=-1000.0)
        assert obs.decay_fraction <= 1.0

    def test_decay_fraction_capped_at_zero(self):
        t = make_decay_tracker()
        obs = record_obs(t, research=100.0, realized=200.0)
        assert obs.decay_fraction >= 0.0

    def test_mean_decay_bps(self):
        t = make_decay_tracker()
        record_obs(t, research=100.0, realized=80.0, ts=TS)  # 20 bps decay
        record_obs(t, research=100.0, realized=60.0, ts=TS2)  # 40 bps decay
        assert t.mean_decay_bps() == pytest.approx(30.0, abs=1e-6)

    def test_mean_decay_bps_empty(self):
        t = make_decay_tracker()
        assert t.mean_decay_bps() == 0.0

    def test_mean_decay_fraction(self):
        t = make_decay_tracker()
        record_obs(t, research=100.0, realized=80.0, ts=TS)  # 0.2 fraction
        record_obs(t, research=100.0, realized=60.0, ts=TS2)  # 0.4 fraction
        assert t.mean_decay_fraction() == pytest.approx(0.3, abs=1e-6)

    def test_attribute_decay_by_type(self):
        t = make_decay_tracker()
        record_obs(t, dtype=DECAY_RESEARCH_ALPHA, research=100.0, realized=80.0, ts=TS)
        record_obs(t, dtype=DECAY_EXECUTION_INDUCED, research=100.0, realized=90.0, ts=TS2)
        attr = t.attribute_decay()
        assert DECAY_RESEARCH_ALPHA in attr
        assert DECAY_EXECUTION_INDUCED in attr
        assert attr[DECAY_RESEARCH_ALPHA] == pytest.approx(20.0, abs=1e-6)
        assert attr[DECAY_EXECUTION_INDUCED] == pytest.approx(10.0, abs=1e-6)

    def test_has_significant_decay_false(self):
        t = AlphaDecayTracker(decay_alert_threshold=0.80)
        record_obs(t, research=100.0, realized=90.0)
        assert not t.has_significant_decay()

    def test_has_significant_decay_true(self):
        t = AlphaDecayTracker(decay_alert_threshold=0.30)
        record_obs(t, research=100.0, realized=50.0)
        assert t.has_significant_decay()

    def test_observation_count(self):
        t = make_decay_tracker()
        record_obs(t, ts=TS)
        record_obs(t, ts=TS2)
        assert t.observation_count() == 2

    def test_observation_count_by_symbol(self):
        t = make_decay_tracker()
        record_obs(t, sym=SYM_A, ts=TS)
        record_obs(t, sym=SYM_B, ts=TS2)
        assert t.observation_count(SYM_A) == 1

    def test_all_observations_sorted(self):
        t = make_decay_tracker()
        record_obs(t, ts=TS2)
        record_obs(t, ts=TS)
        obs = t.all_observations()
        ts_list = [o.observed_at for o in obs]
        assert ts_list == sorted(ts_list)

    def test_to_summary_keys(self):
        t = make_decay_tracker()
        s = t.to_summary()
        assert "mean_decay_bps" in s and "has_significant_decay" in s and "decay_attribution" in s
