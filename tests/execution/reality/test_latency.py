"""Tests: latency.py — LatencyDriftMonitor, LatencyObservation."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.latency import LatencyDriftMonitor, LatencyObservation
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, TS3, make_latency_monitor


def record_obs(monitor, sym=SYM_A, s2o=10_000_000, o2f=10_000_000, broker=8_000_000,
               expected_total=20_000_000, ts=TS):
    return monitor.record(SESSION, sym, s2o, o2f, broker, expected_total, ts)


class TestLatencyObservation:
    def test_frozen(self):
        m = make_latency_monitor()
        obs = record_obs(m)
        with pytest.raises(Exception):
            obs.total_latency_ns = 0  # type: ignore

    def test_obs_id_prefix(self):
        m = make_latency_monitor()
        obs = record_obs(m)
        assert obs.obs_id.startswith("lat_")

    def test_to_dict_from_dict_roundtrip(self):
        m = make_latency_monitor()
        obs = record_obs(m)
        restored = LatencyObservation.from_dict(obs.to_dict())
        assert restored == obs


class TestLatencyDriftMonitor:
    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            LatencyDriftMonitor(baseline_latency_ns=0)

    def test_invalid_drift_factor_raises(self):
        with pytest.raises(ValueError):
            LatencyDriftMonitor(drift_factor=1.0)

    def test_total_latency_computed(self):
        m = make_latency_monitor()
        obs = record_obs(m, s2o=10_000_000, o2f=15_000_000)
        assert obs.total_latency_ns == 25_000_000

    def test_jitter_computed(self):
        m = make_latency_monitor()
        obs = record_obs(m, s2o=10_000_000, o2f=15_000_000, expected_total=20_000_000)
        assert obs.jitter_ns == 5_000_000  # |25M - 20M|

    def test_mean_latency_ns(self):
        m = make_latency_monitor()
        record_obs(m, s2o=10_000_000, o2f=10_000_000, ts=TS)  # 20M
        record_obs(m, s2o=20_000_000, o2f=20_000_000, ts=TS2)  # 40M
        assert m.mean_latency_ns() == pytest.approx(30_000_000, abs=1)

    def test_mean_latency_empty(self):
        m = make_latency_monitor()
        assert m.mean_latency_ns() == 0.0

    def test_mean_jitter_ns(self):
        m = make_latency_monitor()
        record_obs(m, s2o=10_000_000, o2f=10_000_000, expected_total=20_000_000, ts=TS)
        assert m.mean_jitter_ns() == 0.0

    def test_latency_volatility_zero_single(self):
        m = make_latency_monitor()
        record_obs(m)
        assert m.latency_volatility_ns() == 0.0

    def test_latency_volatility_positive(self):
        m = make_latency_monitor()
        record_obs(m, s2o=10_000_000, o2f=10_000_000, ts=TS)
        record_obs(m, s2o=50_000_000, o2f=50_000_000, ts=TS2)
        assert m.latency_volatility_ns() > 0.0

    def test_no_latency_drift(self):
        m = LatencyDriftMonitor(baseline_latency_ns=50_000_000, drift_factor=2.0)
        record_obs(m, s2o=20_000_000, o2f=20_000_000)
        assert not m.has_latency_drift()

    def test_has_latency_drift(self):
        m = LatencyDriftMonitor(baseline_latency_ns=10_000_000, drift_factor=2.0)
        record_obs(m, s2o=30_000_000, o2f=30_000_000)  # 60M > 20M threshold
        assert m.has_latency_drift()

    def test_p99_latency(self):
        m = make_latency_monitor()
        for i in range(10):
            record_obs(m, s2o=i * 1_000_000, o2f=i * 1_000_000, ts=TS)
        p99 = m.p99_latency_ns()
        assert p99 >= m.mean_latency_ns()

    def test_p99_empty(self):
        m = make_latency_monitor()
        assert m.p99_latency_ns() == 0.0

    def test_observation_count(self):
        m = make_latency_monitor()
        record_obs(m, ts=TS)
        record_obs(m, ts=TS2)
        assert m.observation_count() == 2

    def test_observation_count_by_symbol(self):
        m = make_latency_monitor()
        record_obs(m, sym=SYM_A, ts=TS)
        record_obs(m, sym=SYM_B, ts=TS2)
        assert m.observation_count(SYM_A) == 1

    def test_all_observations_sorted(self):
        m = make_latency_monitor()
        record_obs(m, ts=TS2)
        record_obs(m, ts=TS)
        obs = m.all_observations()
        ts_list = [o.observed_at for o in obs]
        assert ts_list == sorted(ts_list)

    def test_to_summary_keys(self):
        m = make_latency_monitor()
        s = m.to_summary()
        assert "mean_latency_ns" in s and "has_latency_drift" in s
