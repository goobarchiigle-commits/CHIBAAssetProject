"""
tests/bootstrap/test_dependency_validator.py

Coverage:
  - validate_order: passes when deps initialized
  - validate_order: raises when dep not yet initialized
  - validate_complete: raises on critical FAILED
  - validate_complete: raises on critical BLOCKED
  - validate_complete: passes when all critical initialized
  - validate_complete: passes when only non-critical fails
  - violations_only: never raises, returns strings
  - fail propagation detection in completeness check
"""
import pytest

from src.bootstrap.dependency_validator import (
    BootstrapDependencyValidator,
    BootstrapIntegrityError,
)
from src.bootstrap.initialization_graph import (
    ALL_PHASES,
    PHASE_ANALYTICS_REGISTRY,
    PHASE_BROKER_INTEGRATION,
    PHASE_CAPITAL_STATE,
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
    RuntimeInitializationGraph,
)


def _fully_initialized_graph() -> RuntimeInitializationGraph:
    g = RuntimeInitializationGraph.build_default()
    for phase in ALL_PHASES:
        g.mark_initialized(phase)
    return g


# ── validate_order ────────────────────────────────────────────────────────────

def test_validate_order_passes_when_deps_initialized():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    v = BootstrapDependencyValidator(g)
    # governance depends on configuration (already initialized) — should not raise
    v.validate_order(PHASE_GOVERNANCE)


def test_validate_order_raises_when_dep_pending():
    g = RuntimeInitializationGraph.build_default()
    # configuration not initialized yet
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError) as exc_info:
        v.validate_order(PHASE_GOVERNANCE)
    assert PHASE_GOVERNANCE in str(exc_info.value)


def test_validate_order_raises_when_dep_failed():
    g = RuntimeInitializationGraph.build_default()
    g.mark_failed(PHASE_CONFIGURATION)
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError):
        v.validate_order(PHASE_GOVERNANCE)


def test_validate_order_raises_when_dep_blocked():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    # capital_state is BLOCKED; trying to initialize reconciliation (dep: capital_state) should fail
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError):
        v.validate_order(PHASE_RECONCILIATION)


def test_validate_order_no_deps_always_passes():
    g = RuntimeInitializationGraph.build_default()
    v = BootstrapDependencyValidator(g)
    # configuration has no deps — always safe
    v.validate_order(PHASE_CONFIGURATION)


def test_validate_order_error_includes_failed_phase_name():
    g = RuntimeInitializationGraph.build_default()
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError) as exc_info:
        v.validate_order(PHASE_GOVERNANCE)
    assert exc_info.value.failed_phases == [PHASE_GOVERNANCE]


# ── validate_complete ─────────────────────────────────────────────────────────

def test_validate_complete_passes_all_critical_initialized():
    g = _fully_initialized_graph()
    v = BootstrapDependencyValidator(g)
    violations = v.validate_complete()
    assert violations == []


def test_validate_complete_raises_on_critical_failed():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE, "config load error")
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError) as exc_info:
        v.validate_complete()
    assert PHASE_GOVERNANCE in exc_info.value.failed_phases


def test_validate_complete_raises_on_critical_blocked():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError) as exc_info:
        v.validate_complete()
    # capital_state, allocation, reconciliation, runtime_supervisor should all be in failed_phases
    assert len(exc_info.value.failed_phases) > 1


def test_validate_complete_raises_on_pending_critical():
    g = RuntimeInitializationGraph.build_default()
    # Only initialize configuration; leave everything else PENDING
    g.mark_initialized(PHASE_CONFIGURATION)
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError):
        v.validate_complete()


def test_validate_complete_passes_with_noncritical_failure():
    g = RuntimeInitializationGraph.build_default()
    for phase in ALL_PHASES:
        if phase != PHASE_ANALYTICS_REGISTRY:
            g.mark_initialized(phase)
    g.mark_failed(PHASE_ANALYTICS_REGISTRY)
    v = BootstrapDependencyValidator(g)
    # Should not raise — analytics_registry is non-critical
    violations = v.validate_complete()
    assert violations == []


def test_validate_complete_error_includes_all_failed_phases():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    g.mark_failed(PHASE_BROKER_INTEGRATION)
    v = BootstrapDependencyValidator(g)
    with pytest.raises(BootstrapIntegrityError) as exc_info:
        v.validate_complete()
    # Both root failures and their blocked phases should be reported
    assert len(exc_info.value.failed_phases) >= 2


# ── violations_only ───────────────────────────────────────────────────────────

def test_violations_only_never_raises():
    g = RuntimeInitializationGraph.build_default()
    g.mark_failed(PHASE_CONFIGURATION)
    v = BootstrapDependencyValidator(g)
    # Must not raise — only returns strings
    violations = v.violations_only()
    assert isinstance(violations, list)
    assert len(violations) > 0


def test_violations_only_empty_on_full_success():
    g = _fully_initialized_graph()
    v = BootstrapDependencyValidator(g)
    assert v.violations_only() == []


def test_violations_only_skips_noncritical():
    g = RuntimeInitializationGraph.build_default()
    for phase in ALL_PHASES:
        if phase != PHASE_ANALYTICS_REGISTRY:
            g.mark_initialized(phase)
    g.mark_failed(PHASE_ANALYTICS_REGISTRY)
    v = BootstrapDependencyValidator(g)
    assert v.violations_only() == []
