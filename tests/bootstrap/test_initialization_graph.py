"""
tests/bootstrap/test_initialization_graph.py

Coverage:
  - default graph construction (9 phases)
  - topological ordering correctness
  - mark_initialized / mark_failed state transitions
  - failure propagation to direct and transitive dependents
  - root failure identification
  - already-initialized phases not overridden by propagation
  - all_critical_initialized / can_execute / has_critical_failures
  - dump() output structure
"""
import pytest

from src.bootstrap.initialization_graph import (
    ALL_PHASES,
    CRITICAL_PHASES,
    DEFAULT_PHASE_DEPENDENCIES,
    NON_CRITICAL_PHASES,
    PHASE_ALLOCATION,
    PHASE_ANALYTICS_REGISTRY,
    PHASE_BROKER_INTEGRATION,
    PHASE_CAPITAL_STATE,
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
    PhaseStatus,
    RuntimeInitializationGraph,
)


def _all_initialized(g: RuntimeInitializationGraph) -> None:
    """Helper: mark all phases as initialized."""
    for name in ALL_PHASES:
        g.mark_initialized(name)


# ── Construction ──────────────────────────────────────────────────────────────

def test_build_default_creates_all_nine_phases():
    g = RuntimeInitializationGraph.build_default()
    for phase in ALL_PHASES:
        assert g.get_phase(phase) is not None, f"Missing phase: {phase}"


def test_build_default_critical_phases_correct():
    g = RuntimeInitializationGraph.build_default()
    for phase in CRITICAL_PHASES:
        assert g.get_phase(phase).is_critical is True
    for phase in NON_CRITICAL_PHASES:
        assert g.get_phase(phase).is_critical is False


def test_build_default_dependencies_match_spec():
    g = RuntimeInitializationGraph.build_default()
    for phase, deps in DEFAULT_PHASE_DEPENDENCIES.items():
        record = g.get_phase(phase)
        assert sorted(record.dependencies) == sorted(deps), (
            f"{phase}: expected deps {deps}, got {record.dependencies}"
        )


def test_initial_status_is_pending():
    g = RuntimeInitializationGraph.build_default()
    for phase in ALL_PHASES:
        assert g.get_phase(phase).status == PhaseStatus.PENDING


# ── Topological ordering ──────────────────────────────────────────────────────

def test_topological_order_has_all_phases():
    g = RuntimeInitializationGraph.build_default()
    order = g.get_topological_order()
    assert sorted(order) == sorted(ALL_PHASES)


def test_topological_order_configuration_first():
    g = RuntimeInitializationGraph.build_default()
    order = g.get_topological_order()
    assert order[0] == PHASE_CONFIGURATION


def test_topological_order_runtime_supervisor_last():
    """runtime_supervisor has the deepest dependency chain."""
    g = RuntimeInitializationGraph.build_default()
    order = g.get_topological_order()
    assert order[-1] == PHASE_RUNTIME_SUPERVISOR


def test_topological_order_respects_dep_precedence():
    """Every phase appears after all its dependencies."""
    g = RuntimeInitializationGraph.build_default()
    order = g.get_topological_order()
    index = {name: i for i, name in enumerate(order)}
    for phase, deps in DEFAULT_PHASE_DEPENDENCIES.items():
        for dep in deps:
            assert index[dep] < index[phase], (
                f"{dep} must precede {phase} in topological order"
            )


def test_topological_order_deterministic():
    """Same graph produces identical order across calls."""
    g = RuntimeInitializationGraph.build_default()
    assert g.get_topological_order() == g.get_topological_order()


# ── State transitions ─────────────────────────────────────────────────────────

def test_mark_initialized_sets_status():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    assert g.get_phase(PHASE_CONFIGURATION).status == PhaseStatus.INITIALIZED


def test_mark_initialized_sets_timestamp():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    assert g.get_phase(PHASE_CONFIGURATION).initialized_at is not None


def test_mark_failed_sets_status():
    g = RuntimeInitializationGraph.build_default()
    g.mark_failed(PHASE_CONFIGURATION, "test error")
    assert g.get_phase(PHASE_CONFIGURATION).status == PhaseStatus.FAILED
    assert g.get_phase(PHASE_CONFIGURATION).error_msg == "test error"


# ── Failure propagation ───────────────────────────────────────────────────────

def test_mark_failed_governance_blocks_capital_state():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE, "gov error")
    assert g.get_phase(PHASE_CAPITAL_STATE).status == PhaseStatus.BLOCKED
    assert PHASE_GOVERNANCE in g.get_phase(PHASE_CAPITAL_STATE).blocked_by


def test_mark_failed_governance_transitively_blocks_allocation():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE, "gov error")
    # allocation depends on capital_state which is blocked by governance
    assert g.get_phase(PHASE_ALLOCATION).status == PhaseStatus.BLOCKED


def test_mark_failed_governance_blocks_runtime_supervisor():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_BROKER_INTEGRATION)
    g.mark_failed(PHASE_GOVERNANCE)
    # capital_state → blocked
    # reconciliation depends on capital_state + broker_integration; capital_state blocked → reconciliation blocked
    # runtime_supervisor depends on reconciliation → also blocked
    assert g.get_phase(PHASE_RUNTIME_SUPERVISOR).status == PhaseStatus.BLOCKED


def test_mark_failed_does_not_block_broker_integration_on_governance_fail():
    """broker_integration only depends on configuration, not governance."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    # broker_integration should remain PENDING (not blocked)
    assert g.get_phase(PHASE_BROKER_INTEGRATION).status == PhaseStatus.PENDING


def test_mark_failed_does_not_block_analytics_registry_on_capital_fail():
    """analytics_registry depends only on configuration."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_GOVERNANCE)
    g.mark_failed(PHASE_CAPITAL_STATE)
    assert g.get_phase(PHASE_ANALYTICS_REGISTRY).status == PhaseStatus.PENDING


def test_initialized_phase_not_overridden_by_propagation():
    """An already-INITIALIZED phase must not be downgraded to BLOCKED."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_GOVERNANCE)
    g.mark_initialized(PHASE_CAPITAL_STATE)
    g.mark_initialized(PHASE_ALLOCATION)
    # Now fail governance retroactively
    g.mark_failed(PHASE_GOVERNANCE)
    # allocation was already initialized — must stay INITIALIZED
    assert g.get_phase(PHASE_ALLOCATION).status == PhaseStatus.INITIALIZED


# ── Root failure identification ───────────────────────────────────────────────

def test_get_root_failures_single():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    roots = g.get_root_failures()
    assert roots == [PHASE_GOVERNANCE]


def test_get_root_failures_excludes_blocked_cascades():
    """capital_state is BLOCKED (not a root failure) when governance fails."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    roots = g.get_root_failures()
    assert PHASE_CAPITAL_STATE not in roots
    assert PHASE_ALLOCATION not in roots


def test_get_root_failures_multiple_independent():
    """Two independent root failures."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    g.mark_failed(PHASE_BROKER_INTEGRATION)
    roots = g.get_root_failures()
    assert PHASE_GOVERNANCE in roots
    assert PHASE_BROKER_INTEGRATION in roots


# ── Execution gate ────────────────────────────────────────────────────────────

def test_can_execute_false_when_not_all_initialized():
    g = RuntimeInitializationGraph.build_default()
    assert g.can_execute() is False


def test_can_execute_true_when_all_critical_initialized():
    g = RuntimeInitializationGraph.build_default()
    _all_initialized(g)
    assert g.can_execute() is True


def test_can_execute_false_when_critical_blocked():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    assert g.can_execute() is False


def test_has_critical_failures_false_when_only_noncritical_fails():
    g = RuntimeInitializationGraph.build_default()
    for phase in ALL_PHASES:
        if phase != PHASE_ANALYTICS_REGISTRY:
            g.mark_initialized(phase)
    g.mark_failed(PHASE_ANALYTICS_REGISTRY)
    assert g.has_critical_failures() is False


def test_has_critical_failures_true_on_critical_fail():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    assert g.has_critical_failures() is True


# ── Dump ──────────────────────────────────────────────────────────────────────

def test_dump_contains_all_phases():
    g = RuntimeInitializationGraph.build_default()
    d = g.dump()
    assert sorted(d["phases"].keys()) == sorted(ALL_PHASES)


def test_dump_topology_order_valid():
    g = RuntimeInitializationGraph.build_default()
    d = g.dump()
    assert d["topology_order"][0] == PHASE_CONFIGURATION
    assert d["topology_order"][-1] == PHASE_RUNTIME_SUPERVISOR
