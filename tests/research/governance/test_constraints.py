"""Tests: constraints.py — ResearchConstraintEngine."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.constraints import (
    ResearchConstraintEngine, GovernanceConstraints,
)
from src.research.governance.negative import FAILURE_DEAD_TOPOLOGY
from src.research.governance.topology import TOPOLOGY_CLIFF_EDGE
from tests.research.governance.conftest import (
    make_ledger, make_negative, make_topology,
    FAM_A, FAM_B, GH_A, GH_B, HYP_A, HYP_B, TMPL, PARAM_H, TAX, TS, TS2,
)


class TestGovernanceConstraints:
    def test_defaults(self):
        c = GovernanceConstraints()
        assert c.max_exploration_density == 0.90
        assert c.max_family_branching == 10
        assert c.max_failure_recurrence == 5
        assert c.max_topology_mean_fragility == 0.80

    def test_frozen(self):
        c = GovernanceConstraints()
        with pytest.raises(Exception):
            c.max_failure_recurrence = 99  # type: ignore

    def test_to_dict_keys(self):
        d = GovernanceConstraints().to_dict()
        assert "max_exploration_density" in d
        assert "max_family_branching" in d
        assert "max_failure_recurrence" in d


class TestResearchConstraintEngine:
    def test_default_factory(self):
        engine = ResearchConstraintEngine.default()
        assert isinstance(engine.constraints, GovernanceConstraints)

    # ── exploration density ────────────────────────────────────────────────────

    def test_check_density_clean_family(self):
        engine = ResearchConstraintEngine()
        ledger = make_ledger()
        valid, violations = engine.check_exploration_density(FAM_A, ledger)
        assert valid
        assert violations == []

    def test_check_density_oversaturated(self):
        engine = ResearchConstraintEngine(GovernanceConstraints(
            max_exploration_density=0.10,
            exploration_density_max_entries=10,
        ))
        ledger = make_ledger()
        for _ in range(5):  # 5/10 = 0.50 >= 0.10
            ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        valid, violations = engine.check_exploration_density(FAM_A, ledger)
        assert not valid
        assert any("exploration_density" in v for v in violations)

    # ── topology stability ─────────────────────────────────────────────────────

    def test_check_topology_stable(self):
        engine = ResearchConstraintEngine()
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.5, timestamp=TS)
        valid, violations = engine.check_topology_stability(GH_A, topo)
        assert valid  # 0.5 <= 0.80

    def test_check_topology_fragile(self):
        engine = ResearchConstraintEngine()
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.95, timestamp=TS)
        valid, violations = engine.check_topology_stability(GH_A, topo)
        assert not valid
        assert any("mean_fragility" in v for v in violations)

    def test_check_topology_no_failures(self):
        engine = ResearchConstraintEngine()
        topo = make_topology()
        # No failures recorded → mean_fragility = 0.0 → valid
        valid, _ = engine.check_topology_stability(GH_A, topo)
        assert valid

    # ── failure recurrence ─────────────────────────────────────────────────────

    def test_check_failure_recurrence_below_threshold(self):
        engine = ResearchConstraintEngine()
        neg = make_negative()
        for _ in range(3):
            neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        valid, violations = engine.check_failure_recurrence(FAM_A, neg)
        assert valid  # 3 < 5

    def test_check_failure_recurrence_at_threshold(self):
        engine = ResearchConstraintEngine()
        neg = make_negative()
        for _ in range(5):
            neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        valid, violations = engine.check_failure_recurrence(FAM_A, neg)
        assert not valid
        assert any("failure_count" in v for v in violations)

    def test_check_failure_recurrence_zero(self):
        engine = ResearchConstraintEngine()
        neg = make_negative()
        valid, _ = engine.check_failure_recurrence(FAM_A, neg)
        assert valid

    # ── family branching ───────────────────────────────────────────────────────

    def test_check_branching_ok(self):
        engine = ResearchConstraintEngine()
        valid, _ = engine.check_family_branching(FAM_A, 5)
        assert valid

    def test_check_branching_exceeded(self):
        engine = ResearchConstraintEngine(GovernanceConstraints(max_family_branching=3))
        valid, violations = engine.check_family_branching(FAM_A, 4)
        assert not valid
        assert any("branch_count" in v for v in violations)

    # ── unstable fraction ──────────────────────────────────────────────────────

    def test_check_unstable_fraction_ok(self):
        engine = ResearchConstraintEngine()
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_unstable(FAM_A, HYP_B, TMPL, PARAM_H, GH_A, TAX, timestamp=TS2)
        # 1 unstable / 2 total = 0.50 == max_unstable_fraction default
        valid, _ = engine.check_unstable_fraction(FAM_A, ledger)
        assert valid  # 0.50 is not > 0.50 (strictly greater)

    def test_check_unstable_fraction_exceeded(self):
        engine = ResearchConstraintEngine(GovernanceConstraints(max_unstable_fraction=0.30))
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_unstable(FAM_A, HYP_B, TMPL, PARAM_H, GH_A, TAX, timestamp=TS2)
        # 1/2 = 0.50 > 0.30
        valid, violations = engine.check_unstable_fraction(FAM_A, ledger)
        assert not valid

    def test_check_unstable_fraction_empty_family(self):
        engine = ResearchConstraintEngine()
        ledger = make_ledger()
        valid, _ = engine.check_unstable_fraction(FAM_A, ledger)
        assert valid  # no entries → no violation

    # ── check_all composite ────────────────────────────────────────────────────

    def test_check_all_clean_passes(self):
        engine = ResearchConstraintEngine()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        valid, violations = engine.check_all(FAM_A, GH_A, ledger, neg, topo)
        assert valid
        assert violations == []

    def test_check_all_aggregates_violations(self):
        engine = ResearchConstraintEngine(GovernanceConstraints(
            max_failure_recurrence=1,
            max_topology_mean_fragility=0.20,
        ))
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS2)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        valid, violations = engine.check_all(FAM_A, GH_A, ledger, neg, topo)
        assert not valid
        assert len(violations) >= 2

    def test_should_suppress_true_on_violation(self):
        engine = ResearchConstraintEngine(GovernanceConstraints(max_failure_recurrence=1))
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS2)
        assert engine.should_suppress(FAM_A, GH_A, ledger, neg, topo)

    def test_should_suppress_false_clean(self):
        engine = ResearchConstraintEngine()
        ledger = make_ledger()
        neg = make_negative()
        topo = make_topology()
        assert not engine.should_suppress(FAM_A, GH_A, ledger, neg, topo)

    def test_to_dict(self):
        d = ResearchConstraintEngine().to_dict()
        assert "constraints" in d
