"""Tests: liquidity.py — LiquidityRealityMonitor, LiquidityObservation."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.liquidity import LiquidityRealityMonitor, LiquidityObservation
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, TS3, make_liquidity_monitor


def record_obs(monitor, sym=SYM_A, pr=0.10, adv=0.02, spread=10.0, collapse=False, ts=TS):
    return monitor.record(SESSION, sym, pr, adv, spread, collapse, ts)


class TestLiquidityObservation:
    def test_frozen(self):
        m = make_liquidity_monitor()
        obs = record_obs(m)
        with pytest.raises(Exception):
            obs.spread_bps = 0.0  # type: ignore

    def test_obs_id_prefix(self):
        m = make_liquidity_monitor()
        obs = record_obs(m)
        assert obs.obs_id.startswith("liq_")

    def test_to_dict_from_dict_roundtrip(self):
        m = make_liquidity_monitor()
        obs = record_obs(m)
        restored = LiquidityObservation.from_dict(obs.to_dict())
        assert restored == obs


class TestLiquidityRealityMonitor:
    def test_invalid_participation_raises(self):
        with pytest.raises(ValueError):
            LiquidityRealityMonitor(max_participation_rate=0.0)

    def test_invalid_adv_raises(self):
        with pytest.raises(ValueError):
            LiquidityRealityMonitor(max_adv_usage=0.0)

    def test_invalid_collapse_threshold_raises(self):
        with pytest.raises(ValueError):
            LiquidityRealityMonitor(collapse_threshold=0.0)

    def test_mean_participation_rate(self):
        m = make_liquidity_monitor()
        record_obs(m, pr=0.10, ts=TS)
        record_obs(m, pr=0.20, ts=TS2)
        assert m.mean_participation_rate() == pytest.approx(0.15, abs=1e-6)

    def test_mean_adv_usage(self):
        m = make_liquidity_monitor()
        record_obs(m, adv=0.02, ts=TS)
        record_obs(m, adv=0.04, ts=TS2)
        assert m.mean_adv_usage() == pytest.approx(0.03, abs=1e-6)

    def test_mean_spread_bps(self):
        m = make_liquidity_monitor()
        record_obs(m, spread=10.0, ts=TS)
        record_obs(m, spread=20.0, ts=TS2)
        assert m.mean_spread_bps() == pytest.approx(15.0, abs=1e-6)

    def test_fill_collapse_rate(self):
        m = make_liquidity_monitor()
        record_obs(m, collapse=True, ts=TS)
        record_obs(m, collapse=False, ts=TS2)
        assert m.fill_collapse_rate() == pytest.approx(0.5, abs=1e-6)

    def test_is_over_participating_false(self):
        m = LiquidityRealityMonitor(max_participation_rate=0.25)
        record_obs(m, pr=0.10)
        assert not m.is_over_participating()

    def test_is_over_participating_true(self):
        m = LiquidityRealityMonitor(max_participation_rate=0.25)
        record_obs(m, pr=0.30)
        assert m.is_over_participating()

    def test_is_over_adv_false(self):
        m = LiquidityRealityMonitor(max_adv_usage=0.05)
        record_obs(m, adv=0.02)
        assert not m.is_over_adv()

    def test_is_over_adv_true(self):
        m = LiquidityRealityMonitor(max_adv_usage=0.05)
        record_obs(m, adv=0.10)
        assert m.is_over_adv()

    def test_has_liquidity_collapse_false(self):
        m = LiquidityRealityMonitor(collapse_threshold=0.50)
        record_obs(m, collapse=False, ts=TS)
        assert not m.has_liquidity_collapse()

    def test_has_liquidity_collapse_true(self):
        m = LiquidityRealityMonitor(collapse_threshold=0.50)
        record_obs(m, collapse=True, ts=TS)
        record_obs(m, collapse=True, ts=TS2)
        assert m.has_liquidity_collapse()

    def test_mean_participation_empty(self):
        m = make_liquidity_monitor()
        assert m.mean_participation_rate() == 0.0

    def test_observation_count(self):
        m = make_liquidity_monitor()
        record_obs(m, ts=TS)
        record_obs(m, ts=TS2)
        assert m.observation_count() == 2

    def test_observation_count_by_symbol(self):
        m = make_liquidity_monitor()
        record_obs(m, sym=SYM_A, ts=TS)
        record_obs(m, sym=SYM_B, ts=TS2)
        assert m.observation_count(SYM_A) == 1

    def test_all_observations_sorted(self):
        m = make_liquidity_monitor()
        record_obs(m, ts=TS2)
        record_obs(m, ts=TS)
        obs = m.all_observations()
        ts_list = [o.observed_at for o in obs]
        assert ts_list == sorted(ts_list)

    def test_to_summary_keys(self):
        m = make_liquidity_monitor()
        s = m.to_summary()
        assert "mean_participation_rate" in s and "has_liquidity_collapse" in s
