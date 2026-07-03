"""
tests/bootstrap/test_contract_verifier.py

Coverage:
  - configuration contract: passes with config object, fails without
  - capital_state contract: None, missing attr, valid
  - allocation contract: None allocator, None cap_state
  - broker_integration contract
  - reconciliation contract: missing dependencies
  - analytics_registry: only WARN, not CRITICAL
  - verify_all: returns dict of violations per phase
  - verify_all determinism
  - custom contract override
  - has_critical_violations
"""
import pytest

from src.bootstrap.contract_verifier import InitializationContractVerifier
from src.bootstrap.initialization_graph import (
    PHASE_ALLOCATION,
    PHASE_ANALYTICS_REGISTRY,
    PHASE_BROKER_INTEGRATION,
    PHASE_CAPITAL_STATE,
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
)


class _FakeCapState:
    actual_equity      = 3_000_000.0
    effective_deployable = 2_900_000.0


class _FakeCapStateNoAttr:
    pass


class _FakeAllocator:
    def return_buffers(self):
        return []


def _full_state():
    return {
        "config":               object(),
        "governance":           object(),
        "capital_state":        _FakeCapState(),
        "allocator":            _FakeAllocator(),
        "broker_client":        object(),
        "reconciliation_engine": object(),
        "inflight_registry":    object(),
        "runtime_supervisor":   object(),
        "analytics_registry":   object(),
    }


# ── configuration ─────────────────────────────────────────────────────────────

def test_configuration_passes_with_config():
    v = InitializationContractVerifier()
    assert v.verify(PHASE_CONFIGURATION, {"config": object()}) == []


def test_configuration_fails_without_config():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_CONFIGURATION, {})
    assert any(":CRITICAL]" in x for x in viol)


# ── governance ────────────────────────────────────────────────────────────────

def test_governance_passes_with_object():
    v = InitializationContractVerifier()
    assert v.verify(PHASE_GOVERNANCE, {"governance": object()}) == []


def test_governance_fails_when_none():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_GOVERNANCE, {"governance": None})
    assert any(":CRITICAL]" in x for x in viol)


# ── capital_state ─────────────────────────────────────────────────────────────

def test_capital_state_passes_with_valid_object():
    v = InitializationContractVerifier()
    assert v.verify(PHASE_CAPITAL_STATE, {"capital_state": _FakeCapState()}) == []


def test_capital_state_fails_when_none():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_CAPITAL_STATE, {"capital_state": None})
    assert any(":CRITICAL]" in x for x in viol)


def test_capital_state_fails_missing_actual_equity():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_CAPITAL_STATE, {"capital_state": _FakeCapStateNoAttr()})
    assert any("actual_equity" in x for x in viol)
    assert any(":CRITICAL]" in x for x in viol)


def test_capital_state_warn_on_missing_effective_deployable():
    v = InitializationContractVerifier()

    class PartialCap:
        actual_equity = 1_000_000.0

    viol = v.verify(PHASE_CAPITAL_STATE, {"capital_state": PartialCap()})
    assert any(":WARN]" in x and "effective_deployable" in x for x in viol)


# ── allocation ────────────────────────────────────────────────────────────────

def test_allocation_passes_with_full_state():
    v = InitializationContractVerifier()
    state = {"allocator": _FakeAllocator(), "capital_state": _FakeCapState()}
    assert v.verify(PHASE_ALLOCATION, state) == []


def test_allocation_fails_when_allocator_none():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_ALLOCATION, {"allocator": None, "capital_state": _FakeCapState()})
    assert any(":CRITICAL]" in x and "allocator" in x for x in viol)


def test_allocation_fails_when_capital_state_none():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_ALLOCATION, {"allocator": _FakeAllocator(), "capital_state": None})
    assert any(":CRITICAL]" in x and "capital_state" in x for x in viol)


# ── broker_integration ────────────────────────────────────────────────────────

def test_broker_passes_with_client():
    v = InitializationContractVerifier()
    assert v.verify(PHASE_BROKER_INTEGRATION, {"broker_client": object()}) == []


def test_broker_fails_without_client():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_BROKER_INTEGRATION, {})
    assert any(":CRITICAL]" in x for x in viol)


# ── reconciliation ────────────────────────────────────────────────────────────

def test_reconciliation_fails_without_engine():
    v = InitializationContractVerifier()
    state = {
        "broker_client":  object(),
        "capital_state":  _FakeCapState(),
    }
    viol = v.verify(PHASE_RECONCILIATION, state)
    assert any("reconciliation_engine" in x and ":CRITICAL]" in x for x in viol)


def test_reconciliation_fails_without_broker():
    v = InitializationContractVerifier()
    state = {
        "reconciliation_engine": object(),
        "capital_state":  _FakeCapState(),
    }
    viol = v.verify(PHASE_RECONCILIATION, state)
    assert any("broker_client" in x and ":CRITICAL]" in x for x in viol)


# ── analytics_registry ────────────────────────────────────────────────────────

def test_analytics_registry_only_warns_when_none():
    v = InitializationContractVerifier()
    viol = v.verify(PHASE_ANALYTICS_REGISTRY, {})
    assert viol  # has violations
    assert all(":WARN]" in x for x in viol)  # but no CRITICAL


# ── verify_all ────────────────────────────────────────────────────────────────

def test_verify_all_no_violations_on_full_state():
    v = InitializationContractVerifier()
    result = v.verify_all(_full_state())
    # analytics_registry may emit a WARN, but no CRITICAL
    critical_viol = [
        x for vlist in result.values() for x in vlist if ":CRITICAL]" in x
    ]
    assert critical_viol == []


def test_verify_all_returns_violations_per_phase():
    v = InitializationContractVerifier()
    result = v.verify_all({})   # empty state → all phases fail
    assert PHASE_CONFIGURATION in result
    assert PHASE_CAPITAL_STATE in result
    assert PHASE_BROKER_INTEGRATION in result


def test_verify_all_deterministic():
    v = InitializationContractVerifier()
    s = {}
    assert v.verify_all(s) == v.verify_all(s)


# ── custom contract override ──────────────────────────────────────────────────

def test_custom_contract_override():
    v = InitializationContractVerifier()
    v.register_contract(PHASE_CONFIGURATION, lambda state: ["[CONTRACT:CRITICAL] custom"])
    viol = v.verify(PHASE_CONFIGURATION, {"config": object()})
    assert viol == ["[CONTRACT:CRITICAL] custom"]


# ── has_critical_violations ───────────────────────────────────────────────────

def test_has_critical_violations_true_on_empty_state():
    v = InitializationContractVerifier()
    assert v.has_critical_violations({}) is True


def test_has_critical_violations_false_on_full_state():
    v = InitializationContractVerifier()
    assert v.has_critical_violations(_full_state()) is False
