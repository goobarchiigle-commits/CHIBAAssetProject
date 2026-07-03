"""Tests: replay.py — CounterfactualReplay, ReplayScenario, ReplayResult."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.replay import (
    CounterfactualReplay, ReplayScenario, ReplayResult, build_scenario,
    SCENARIO_REDUCED_PARTICIPATION, SCENARIO_DELAYED_ENTRY, SCENARIO_AUCTION_AVOIDANCE,
    SCENARIO_ALTERNATE_THROTTLING, SCENARIO_ALTERNATE_LIQUIDITY_CAP, ALL_SCENARIOS,
)
from tests.execution.reality.conftest import SESSION, TS, TS2, TS3, make_replay


def make_scenario(scenario_type=SCENARIO_REDUCED_PARTICIPATION, params=None, desc="test"):
    return build_scenario(scenario_type, params or {}, desc)


def do_replay(cr, scenario, slip=10.0, fill=1.0, lat=20_000_000, ts=TS):
    return cr.replay(SESSION, scenario, slip, fill, lat, ts)


class TestReplayScenario:
    def test_frozen(self):
        sc = make_scenario()
        with pytest.raises(Exception):
            sc.description = "CHANGED"  # type: ignore

    def test_scenario_id_prefix(self):
        sc = make_scenario()
        assert sc.scenario_id.startswith("scn_")

    def test_scenario_id_deterministic(self):
        s1 = make_scenario()
        s2 = make_scenario()
        assert s1.scenario_id == s2.scenario_id

    def test_scenario_id_differs_by_type(self):
        s1 = make_scenario(SCENARIO_REDUCED_PARTICIPATION)
        s2 = make_scenario(SCENARIO_DELAYED_ENTRY)
        assert s1.scenario_id != s2.scenario_id

    def test_to_dict_from_dict_roundtrip(self):
        sc = make_scenario()
        restored = ReplayScenario.from_dict(sc.to_dict())
        assert restored.scenario_id == sc.scenario_id

    def test_unknown_scenario_type_raises(self):
        with pytest.raises(ValueError):
            build_scenario("BAD_TYPE", {}, "desc")

    def test_parameters_sorted(self):
        sc = build_scenario(SCENARIO_REDUCED_PARTICIPATION, {"b": "2", "a": "1"}, "desc")
        keys = [p[0] for p in sc.parameters]
        assert keys == sorted(keys)


class TestReplayResult:
    def test_frozen(self):
        cr = make_replay()
        sc = make_scenario()
        result = do_replay(cr, sc)
        with pytest.raises(Exception):
            result.slippage_improvement_bps = 0.0  # type: ignore

    def test_result_id_prefix(self):
        cr = make_replay()
        sc = make_scenario()
        result = do_replay(cr, sc)
        assert result.result_id.startswith("rply_")

    def test_checksum_16_chars(self):
        cr = make_replay()
        sc = make_scenario()
        result = do_replay(cr, sc)
        assert len(result.checksum) == 16

    def test_to_dict_from_dict_roundtrip(self):
        cr = make_replay()
        sc = make_scenario()
        result = do_replay(cr, sc)
        restored = ReplayResult.from_dict(result.to_dict())
        assert restored.result_id == result.result_id


class TestCounterfactualReplay:
    def test_all_scenarios(self):
        assert len(ALL_SCENARIOS) == 5

    def test_reduced_participation_reduces_slippage(self):
        cr = make_replay()
        sc = make_scenario(SCENARIO_REDUCED_PARTICIPATION)
        result = do_replay(cr, sc, slip=10.0)
        assert result.counterfactual_slippage_bps < result.baseline_slippage_bps

    def test_delayed_entry_increases_slippage(self):
        cr = make_replay()
        sc = make_scenario(SCENARIO_DELAYED_ENTRY)
        result = do_replay(cr, sc, slip=10.0)
        assert result.counterfactual_slippage_bps > result.baseline_slippage_bps

    def test_fill_ratio_capped_at_one(self):
        cr = make_replay()
        sc = make_scenario(SCENARIO_REDUCED_PARTICIPATION)
        result = do_replay(cr, sc, fill=1.0)
        assert result.counterfactual_fill_ratio <= 1.0

    def test_slippage_improvement_positive_for_reduced_participation(self):
        cr = make_replay()
        sc = make_scenario(SCENARIO_REDUCED_PARTICIPATION)
        result = do_replay(cr, sc, slip=10.0)
        assert result.slippage_improvement_bps > 0.0

    def test_is_improvement_reduced_participation(self):
        cr = make_replay()
        sc = make_scenario(SCENARIO_REDUCED_PARTICIPATION)
        result = do_replay(cr, sc)
        assert result.is_improvement()

    def test_result_count(self):
        cr = make_replay()
        sc = make_scenario()
        do_replay(cr, sc, ts=TS)
        do_replay(cr, sc, ts=TS2)
        assert cr.result_count() == 2

    def test_all_results_sorted(self):
        cr = make_replay()
        sc = make_scenario()
        do_replay(cr, sc, ts=TS2)
        do_replay(cr, sc, ts=TS)
        results = cr.all_results()
        ts_list = [r.replayed_at for r in results]
        assert ts_list == sorted(ts_list)

    def test_best_scenario_nonempty(self):
        cr = make_replay()
        for stype in [SCENARIO_REDUCED_PARTICIPATION, SCENARIO_AUCTION_AVOIDANCE]:
            sc = make_scenario(stype)
            do_replay(cr, sc, ts=TS if stype == SCENARIO_REDUCED_PARTICIPATION else TS2)
        assert cr.best_scenario() is not None

    def test_best_scenario_none_when_empty(self):
        cr = make_replay()
        assert cr.best_scenario() is None

    def test_all_scenario_types_replay(self):
        cr = make_replay()
        for i, stype in enumerate(ALL_SCENARIOS):
            sc = make_scenario(stype)
            result = cr.replay(SESSION, sc, 10.0, 1.0, 20_000_000,
                               f"2026-01-01T00:00:0{i}.000000Z")
            assert result.schema_version == "1.0.0"

    def test_deterministic_replay(self):
        sc = make_scenario()
        cr1 = make_replay()
        cr2 = make_replay()
        r1 = do_replay(cr1, sc)
        r2 = do_replay(cr2, sc)
        assert r1.counterfactual_slippage_bps == r2.counterfactual_slippage_bps
        assert r1.slippage_improvement_bps == r2.slippage_improvement_bps

    def test_to_summary_keys(self):
        cr = make_replay()
        s = cr.to_summary()
        assert "result_count" in s and "best_scenario" in s
