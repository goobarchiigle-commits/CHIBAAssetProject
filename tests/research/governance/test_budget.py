"""Tests: budget.py — ExplorationBudgetController."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.budget import (
    ExplorationBudgetController, BudgetAllocation,
    PRIORITY_PERSISTENT_CORE, PRIORITY_UNDEREXPLORED, PRIORITY_PROMISING,
    PRIORITY_STABLE_PLATEAU, PRIORITY_SUPPRESSED,
)
from src.research.governance.negative import FAILURE_DEAD_TOPOLOGY
from src.research.governance.topology import TOPOLOGY_CLIFF_EDGE
from tests.research.governance.conftest import (
    make_ledger, make_negative, make_topology, make_persistence,
    FAM_A, FAM_B, FAM_C, GH_A, GH_B, HYP_A, HYP_B, TMPL, PARAM_H, TAX, TS, TS2, TS3,
)


def _ctrl(**kwargs) -> ExplorationBudgetController:
    return ExplorationBudgetController(**kwargs)


def _default_map() -> dict:
    return {FAM_A: GH_A, FAM_B: GH_B}


class TestBudgetAllocation:
    def test_frozen(self):
        alloc = BudgetAllocation("bud_test", FAM_A, PRIORITY_PROMISING, 10, "ok", TS)
        with pytest.raises(Exception):
            alloc.allocated_budget = 99  # type: ignore

    def test_allocation_id_prefix(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({FAM_A: GH_A}, ledger, neg, topo, pt, timestamp=TS)
        assert allocs[0].allocation_id.startswith("bud_")

    def test_to_dict_from_dict_roundtrip(self):
        alloc = BudgetAllocation("bud_xyz", FAM_A, PRIORITY_PROMISING, 10, "ok", TS)
        restored = BudgetAllocation.from_dict(alloc.to_dict())
        assert restored.family_id == alloc.family_id
        assert restored.allocated_budget == alloc.allocated_budget


class TestExplorationBudgetController:
    def test_invalid_total_budget_raises(self):
        with pytest.raises(ValueError):
            ExplorationBudgetController(total_budget=-1)

    def test_invalid_max_per_family_raises(self):
        with pytest.raises(ValueError):
            ExplorationBudgetController(max_per_family=-1)

    def test_invalid_suppression_threshold_raises(self):
        with pytest.raises(ValueError):
            ExplorationBudgetController(suppression_threshold=0)

    # ── Classification ─────────────────────────────────────────────────────────

    def test_classify_suppressed_by_failure_count(self):
        ctrl = _ctrl(suppression_threshold=3)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        for _ in range(3):
            neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        priority, reason = ctrl.classify_family(FAM_A, GH_A, ledger, neg, topo, pt)
        assert priority == PRIORITY_SUPPRESSED

    def test_classify_suppressed_by_topology_fragility(self):
        ctrl = _ctrl(topology_fragility_cap=0.70)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        priority, reason = ctrl.classify_family(FAM_A, GH_A, ledger, neg, topo, pt)
        assert priority == PRIORITY_SUPPRESSED

    def test_classify_persistent_core(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        for _ in range(4):
            pt.record_observation(FAM_A, survived=True, mutation_id="m", timestamp=TS)
        priority, reason = ctrl.classify_family(FAM_A, GH_A, ledger, neg, topo, pt)
        assert priority == PRIORITY_PERSISTENT_CORE

    def test_classify_underexplored(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        priority, reason = ctrl.classify_family(FAM_A, GH_A, ledger, neg, topo, pt)
        assert priority == PRIORITY_UNDEREXPLORED

    def test_classify_stable_plateau(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        # Some exploration, but no topology failures
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        priority, reason = ctrl.classify_family(FAM_A, GH_A, ledger, neg, topo, pt)
        assert priority == PRIORITY_STABLE_PLATEAU

    def test_classify_promising(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.5, timestamp=TS)
        priority, reason = ctrl.classify_family(FAM_A, GH_A, ledger, neg, topo, pt)
        assert priority == PRIORITY_PROMISING

    # ── Allocate ───────────────────────────────────────────────────────────────

    def test_allocate_returns_all_families(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger, neg, topo, pt, timestamp=TS)
        assert len(allocs) == 2

    def test_allocate_sorted_by_family_id(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({FAM_B: GH_B, FAM_A: GH_A}, ledger, neg, topo, pt, timestamp=TS)
        fids = [a.family_id for a in allocs]
        assert fids == sorted(fids)

    def test_suppressed_family_gets_zero_budget(self):
        ctrl = _ctrl(suppression_threshold=1)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        allocs = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger, neg, topo, pt, timestamp=TS)
        alloc_a = next(a for a in allocs if a.family_id == FAM_A)
        assert alloc_a.allocated_budget == 0

    def test_total_allocated_within_budget(self):
        ctrl = _ctrl(total_budget=100)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger, neg, topo, pt, timestamp=TS)
        assert ctrl.total_allocated(allocs) <= 100

    def test_max_per_family_cap(self):
        ctrl = _ctrl(total_budget=1000, max_per_family=20)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger, neg, topo, pt, timestamp=TS)
        for a in allocs:
            assert a.allocated_budget <= 20

    def test_suppressed_families_list(self):
        ctrl = _ctrl(suppression_threshold=1)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        allocs = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger, neg, topo, pt, timestamp=TS)
        suppressed = ctrl.suppressed_families(allocs)
        assert FAM_A in suppressed
        assert FAM_B not in suppressed

    def test_allocation_deterministic(self):
        ctrl = _ctrl(total_budget=100)
        ledger1 = make_ledger()
        ledger2 = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        a1 = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger1, neg, topo, pt, timestamp=TS)
        a2 = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger2, neg, topo, pt, timestamp=TS)
        for x, y in zip(a1, a2):
            assert x.family_id == y.family_id
            assert x.priority_class == y.priority_class
            assert x.allocated_budget == y.allocated_budget

    def test_to_report_keys(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({FAM_A: GH_A}, ledger, neg, topo, pt, timestamp=TS)
        report = ctrl.to_report(allocs)
        assert "total_budget" in report
        assert "total_allocated" in report
        assert "suppressed_count" in report
        assert "allocations" in report

    def test_to_dict(self):
        ctrl = _ctrl()
        d = ctrl.to_dict()
        assert "total_budget" in d
        assert "max_per_family" in d
        assert "suppression_threshold" in d

    def test_empty_family_map_returns_empty(self):
        ctrl = _ctrl()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        allocs = ctrl.allocate({}, ledger, neg, topo, pt, timestamp=TS)
        assert allocs == []

    def test_persistent_core_gets_more_than_underexplored_not_guaranteed_but_priority_higher(self):
        ctrl = _ctrl(total_budget=100, max_per_family=100)
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        pt = make_persistence()
        # FAM_A = persistent core
        for _ in range(4):
            pt.record_observation(FAM_A, survived=True, mutation_id="m", timestamp=TS)
        # FAM_B = underexplored (no entries)
        allocs = ctrl.allocate({FAM_A: GH_A, FAM_B: GH_B}, ledger, neg, topo, pt, timestamp=TS)
        a = {x.family_id: x for x in allocs}
        # PERSISTENT_CORE score=5, UNDEREXPLORED score=4
        assert a[FAM_A].priority_class == PRIORITY_PERSISTENT_CORE
        assert a[FAM_B].priority_class == PRIORITY_UNDEREXPLORED
        assert a[FAM_A].allocated_budget >= a[FAM_B].allocated_budget
