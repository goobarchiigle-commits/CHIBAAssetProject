"""
tests/bootstrap/test_root_cause_compressor.py

Coverage:
  - no failures → empty string
  - single root failure, no downstream
  - single root failure blocks immediate downstream
  - single root failure with transitive chain
  - multiple independent root failures
  - determinism: same graph → same output
  - compress_chain format
  - operator_summary keys and content
  - critical failure → ABORT safe action
  - no failures → continue safe action
"""
import pytest

from src.bootstrap.initialization_graph import (
    ALL_PHASES,
    PHASE_ALLOCATION,
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
from src.bootstrap.root_cause_compressor import RootCauseCompressor


def _all_init(g: RuntimeInitializationGraph) -> None:
    for p in ALL_PHASES:
        g.mark_initialized(p)


# ── compress: no failures ─────────────────────────────────────────────────────

def test_compress_no_failures_returns_empty():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    c = RootCauseCompressor()
    assert c.compress(g) == ""


# ── compress: single root, no downstream ─────────────────────────────────────

def test_compress_single_root_no_downstream():
    g = RuntimeInitializationGraph.build_default()
    # Mark all init, then retroactively fail analytics_registry (no dependents)
    _all_init(g)
    g.mark_failed(PHASE_ANALYTICS_REGISTRY)
    c = RootCauseCompressor()
    result = c.compress(g)
    assert PHASE_ANALYTICS_REGISTRY in result
    assert "failed" in result.lower()


# ── compress: single root blocks immediate ─────────────────────────────────────

def test_compress_governance_blocks_capital_state():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_BROKER_INTEGRATION)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    result = c.compress(g)
    assert PHASE_GOVERNANCE in result
    assert PHASE_CAPITAL_STATE in result


# ── compress: transitive chain ────────────────────────────────────────────────

def test_compress_governance_failure_transitive_summary():
    """governance fail → capital_state blocked → allocation, reconciliation blocked → runtime_supervisor blocked."""
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_initialized(PHASE_BROKER_INTEGRATION)
    g.mark_initialized(PHASE_INFLIGHT_REGISTRY)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    result = c.compress(g)
    # Should mention governance as root and runtime_supervisor as transitive
    assert PHASE_GOVERNANCE in result
    assert PHASE_RUNTIME_SUPERVISOR in result


def test_compress_configuration_failure_blocks_everything():
    g = RuntimeInitializationGraph.build_default()
    g.mark_failed(PHASE_CONFIGURATION)
    c = RootCauseCompressor()
    result = c.compress(g)
    assert PHASE_CONFIGURATION in result
    # All others should be mentioned as blocked
    assert PHASE_GOVERNANCE in result or PHASE_BROKER_INTEGRATION in result


# ── compress: multiple root failures ─────────────────────────────────────────

def test_compress_multiple_roots():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    g.mark_failed(PHASE_BROKER_INTEGRATION)
    c = RootCauseCompressor()
    result = c.compress(g)
    assert PHASE_GOVERNANCE in result
    assert PHASE_BROKER_INTEGRATION in result


# ── determinism ───────────────────────────────────────────────────────────────

def test_compress_deterministic():
    g1 = RuntimeInitializationGraph.build_default()
    g1.mark_initialized(PHASE_CONFIGURATION)
    g1.mark_failed(PHASE_GOVERNANCE)

    g2 = RuntimeInitializationGraph.build_default()
    g2.mark_initialized(PHASE_CONFIGURATION)
    g2.mark_failed(PHASE_GOVERNANCE)

    c = RootCauseCompressor()
    assert c.compress(g1) == c.compress(g2)


def test_compress_same_graph_twice_returns_same():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    assert c.compress(g) == c.compress(g)


# ── compress_chain ────────────────────────────────────────────────────────────

def test_compress_chain_empty_on_no_failures():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    c = RootCauseCompressor()
    assert c.compress_chain(g) == []


def test_compress_chain_contains_root_prefix():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE, "governance DB unavailable")
    c = RootCauseCompressor()
    chains = c.compress_chain(g)
    assert len(chains) == 1
    assert chains[0].startswith("ROOT: governance")


def test_compress_chain_contains_immediate_blocked():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    chains = c.compress_chain(g)
    assert any("IMMEDIATE_BLOCKED" in ch for ch in chains)
    assert any(PHASE_CAPITAL_STATE in ch for ch in chains)


# ── operator_summary ──────────────────────────────────────────────────────────

def test_operator_summary_has_required_keys():
    g = RuntimeInitializationGraph.build_default()
    c = RootCauseCompressor()
    op = c.operator_summary(g)
    assert "ROOT_CAUSE" in op
    assert "BLOCKED_SYSTEMS" in op
    assert "SAFE_ACTION" in op
    assert "REQUIRED_FIX" in op


def test_operator_summary_abort_on_critical_failure():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    op = c.operator_summary(g)
    assert "Abort" in op["SAFE_ACTION"] or "abort" in op["SAFE_ACTION"].lower()


def test_operator_summary_blocked_systems_not_empty_on_failure():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    op = c.operator_summary(g)
    assert len(op["BLOCKED_SYSTEMS"]) > 0


def test_operator_summary_no_failures_safe_action_continue():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    c = RootCauseCompressor()
    op = c.operator_summary(g)
    assert "No action required" in op["REQUIRED_FIX"]


def test_operator_summary_required_fix_mentions_root():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    c = RootCauseCompressor()
    op = c.operator_summary(g)
    assert PHASE_GOVERNANCE in op["REQUIRED_FIX"]
