"""
tests/bootstrap/test_state_coherence_validator.py

Coverage:
  - partial initialization detected (mixed tier)
  - no partial init when all initialized
  - None critical dep detected (phase INITIALIZED but state is None)
  - stale runtime positions: runtime has qty > 0, broker has 0
  - no stale when in sync
  - missing capital state after phase initialized
  - broken allocation deps: allocator or cap_state None
  - inconsistent inflight registry state
  - inconsistent reconciliation state
  - inconsistent runtime_supervisor state
  - validate_all aggregation
  - has_critical_violations
"""
import pytest

from src.bootstrap.initialization_graph import (
    ALL_PHASES,
    PHASE_ALLOCATION,
    PHASE_BROKER_INTEGRATION,
    PHASE_CAPITAL_STATE,
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
    RuntimeInitializationGraph,
)
from src.bootstrap.state_coherence_validator import StateCoherenceValidator


class _FakeCapState:
    actual_equity        = 3_000_000.0
    effective_deployable = 2_900_000.0


def _all_init(g: RuntimeInitializationGraph) -> None:
    for p in ALL_PHASES:
        g.mark_initialized(p)


def _full_state():
    return {
        "config":                object(),
        "governance":            object(),
        "capital_state":         _FakeCapState(),
        "allocator":             object(),
        "broker_client":         object(),
        "reconciliation_engine": object(),
        "inflight_registry":     object(),
        "runtime_supervisor":    object(),
        "analytics_registry":    object(),
    }


# ── check_partial_initialization ─────────────────────────────────────────────

def test_partial_init_detected_when_sibling_fails():
    """governance and broker_integration are siblings (same depth=1). If one fails → partial."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_GOVERNANCE)
    g.mark_failed(PHASE_BROKER_INTEGRATION)
    v = StateCoherenceValidator()
    violations = v.check_partial_initialization(g)
    assert any("[COHERENCE:WARN]" in x and "partial" in x for x in violations)


def test_no_partial_when_all_initialized():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    v = StateCoherenceValidator()
    violations = v.check_partial_initialization(g)
    assert violations == []


def test_no_partial_when_all_pending():
    g = RuntimeInitializationGraph.build_default()
    v = StateCoherenceValidator()
    # All PENDING — not "partial" (none initialized at all)
    violations = v.check_partial_initialization(g)
    assert violations == []


# ── check_none_critical_deps ──────────────────────────────────────────────────

def test_none_critical_dep_detected_for_initialized_phase():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_GOVERNANCE)
    g.mark_initialized(PHASE_CAPITAL_STATE)
    # capital_state phase is INITIALIZED but state dict has None
    state = {"config": object(), "governance": object(), "capital_state": None}
    v = StateCoherenceValidator()
    violations = v.check_none_critical_deps(state, g)
    assert any(":CRITICAL]" in x and "capital_state" in x for x in violations)


def test_no_none_violation_for_pending_phase():
    """PENDING phase with None state should NOT trigger a violation."""
    g = RuntimeInitializationGraph.build_default()
    # capital_state is PENDING
    state = {"capital_state": None}
    v = StateCoherenceValidator()
    violations = v.check_none_critical_deps(state, g)
    assert all("capital_state" not in x for x in violations)


def test_no_none_violation_on_full_state():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    v = StateCoherenceValidator()
    violations = v.check_none_critical_deps(_full_state(), g)
    assert not any(":CRITICAL]" in x for x in violations)


# ── check_stale_runtime_positions ─────────────────────────────────────────────

def test_stale_runtime_position_detected():
    v = StateCoherenceValidator()
    broker  = {"1234": 0}
    runtime = {"1234": 100}
    violations = v.check_stale_runtime_positions(broker, runtime)
    assert any("stale" in x and "1234" in x for x in violations)


def test_no_stale_when_positions_match():
    v = StateCoherenceValidator()
    broker  = {"1234": 100}
    runtime = {"1234": 100}
    assert v.check_stale_runtime_positions(broker, runtime) == []


def test_no_stale_when_runtime_zero():
    v = StateCoherenceValidator()
    broker  = {"1234": 0}
    runtime = {"1234": 0}
    assert v.check_stale_runtime_positions(broker, runtime) == []


def test_no_stale_when_positions_none():
    v = StateCoherenceValidator()
    assert v.check_stale_runtime_positions(None, None) == []


def test_stale_multiple_symbols():
    v = StateCoherenceValidator()
    broker  = {"A": 0, "B": 100}
    runtime = {"A": 50, "B": 100}
    violations = v.check_stale_runtime_positions(broker, runtime)
    assert len(violations) == 1
    assert "A" in violations[0]


# ── check_missing_capital_state ───────────────────────────────────────────────

def test_missing_capital_state_critical():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CAPITAL_STATE)
    state = {"capital_state": None}
    v = StateCoherenceValidator()
    violations = v.check_missing_capital_state(state, g)
    assert any(":CRITICAL]" in x for x in violations)


def test_missing_attr_on_capital_state():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CAPITAL_STATE)

    class PartialCap:
        actual_equity = 1_000_000.0
        # missing effective_deployable

    state = {"capital_state": PartialCap()}
    v = StateCoherenceValidator()
    violations = v.check_missing_capital_state(state, g)
    assert any("effective_deployable" in x for x in violations)


def test_no_capital_violation_when_phase_pending():
    g = RuntimeInitializationGraph.build_default()
    # capital_state phase is PENDING
    v = StateCoherenceValidator()
    violations = v.check_missing_capital_state({"capital_state": None}, g)
    assert violations == []


# ── check_broken_allocation_deps ─────────────────────────────────────────────

def test_broken_allocation_when_allocator_none():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_ALLOCATION)
    state = {"allocator": None, "capital_state": _FakeCapState()}
    v = StateCoherenceValidator()
    violations = v.check_broken_allocation_deps(state, g)
    assert any(":CRITICAL]" in x and "allocator" in x for x in violations)


def test_broken_allocation_when_cap_state_none():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_ALLOCATION)
    state = {"allocator": object(), "capital_state": None}
    v = StateCoherenceValidator()
    violations = v.check_broken_allocation_deps(state, g)
    assert any(":CRITICAL]" in x and "capital_state" in x for x in violations)


def test_no_allocation_violation_when_phase_pending():
    g = RuntimeInitializationGraph.build_default()
    v = StateCoherenceValidator()
    violations = v.check_broken_allocation_deps({"allocator": None}, g)
    assert violations == []


# ── check_inconsistent_registry_states ───────────────────────────────────────

def test_inflight_registry_initialized_but_none():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_INFLIGHT_REGISTRY)
    state = {"inflight_registry": None}
    v = StateCoherenceValidator()
    violations = v.check_inconsistent_registry_states(state, g)
    assert any(":CRITICAL]" in x and "inflight_registry" in x for x in violations)


def test_reconciliation_missing_broker_client():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_RECONCILIATION)
    state = {
        "reconciliation_engine": object(),
        "broker_client":  None,
        "capital_state":  _FakeCapState(),
    }
    v = StateCoherenceValidator()
    violations = v.check_inconsistent_registry_states(state, g)
    assert any(":CRITICAL]" in x and "broker_client" in x for x in violations)


def test_runtime_supervisor_missing_deps():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_RUNTIME_SUPERVISOR)
    state = {
        "runtime_supervisor":    object(),
        "reconciliation_engine": None,
        "inflight_registry":     None,
    }
    v = StateCoherenceValidator()
    violations = v.check_inconsistent_registry_states(state, g)
    assert any("reconciliation_engine" in x for x in violations)
    assert any("inflight_registry" in x for x in violations)


# ── validate_all ──────────────────────────────────────────────────────────────

def test_validate_all_no_violations_on_clean_state():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    v = StateCoherenceValidator()
    violations = v.validate_all(g, state=_full_state())
    assert not any(":CRITICAL]" in x for x in violations)


def test_validate_all_aggregates_multiple_issues():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_GOVERNANCE)
    g.mark_failed(PHASE_BROKER_INTEGRATION)
    g.mark_initialized(PHASE_CAPITAL_STATE)
    # stale runtime positions too
    v = StateCoherenceValidator()
    violations = v.validate_all(
        g,
        state={"config": object(), "governance": object(), "capital_state": _FakeCapState()},
        broker_positions={"X": 0},
        runtime_positions={"X": 10},
    )
    assert len(violations) >= 2  # partial + stale


# ── has_critical_violations ───────────────────────────────────────────────────

def test_has_critical_violations_true():
    assert StateCoherenceValidator.has_critical_violations(
        ["[COHERENCE:CRITICAL] something broke"]
    ) is True


def test_has_critical_violations_false_on_warn_only():
    assert StateCoherenceValidator.has_critical_violations(
        ["[COHERENCE:WARN] degraded"]
    ) is False


def test_has_critical_violations_false_on_empty():
    assert StateCoherenceValidator.has_critical_violations([]) is False
