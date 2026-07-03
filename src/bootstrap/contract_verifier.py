"""
contract_verifier.py — Per-phase initialization contract verification

Each phase declares a contract: a function that inspects the runtime state
dict and returns a list of violation strings.

State dict convention:
  config              → configuration phase object
  governance          → governance object
  capital_state       → CapitalTruth (or compatible)
  allocator           → allocation engine
  broker_client       → broker API client
  reconciliation_engine → reconciliation engine
  inflight_registry   → InflightRegistry
  runtime_supervisor  → StagedSupervisor / RuntimeSupervisor
  analytics_registry  → analytics registry
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

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
)

ContractFn = Callable[[Dict[str, Any]], List[str]]


# ── Default per-phase contracts ───────────────────────────────────────────────

def _check_configuration(state: Dict[str, Any]) -> List[str]:
    if state.get("config") is None:
        return ["[CONTRACT:CRITICAL] configuration: config object is None"]
    return []


def _check_governance(state: Dict[str, Any]) -> List[str]:
    if state.get("governance") is None:
        return ["[CONTRACT:CRITICAL] governance: governance object is None"]
    return []


def _check_capital_state(state: Dict[str, Any]) -> List[str]:
    v: List[str] = []
    cap = state.get("capital_state")
    if cap is None:
        v.append("[CONTRACT:CRITICAL] capital_state: cap_state object is None")
        return v
    if not hasattr(cap, "actual_equity"):
        v.append("[CONTRACT:CRITICAL] capital_state: missing 'actual_equity' attribute")
    elif getattr(cap, "actual_equity", None) is None:
        v.append("[CONTRACT:CRITICAL] capital_state: actual_equity is None")
    if not hasattr(cap, "effective_deployable"):
        v.append("[CONTRACT:WARN] capital_state: missing 'effective_deployable' attribute")
    return v


def _check_allocation(state: Dict[str, Any]) -> List[str]:
    v: List[str] = []
    if state.get("allocator") is None:
        v.append("[CONTRACT:CRITICAL] allocation: allocator is None")
    if state.get("capital_state") is None:
        v.append("[CONTRACT:CRITICAL] allocation: capital_state dependency is None")
    alloc = state.get("allocator")
    if alloc is not None and not hasattr(alloc, "return_buffers"):
        v.append("[CONTRACT:WARN] allocation: allocator missing 'return_buffers' interface")
    return v


def _check_broker_integration(state: Dict[str, Any]) -> List[str]:
    if state.get("broker_client") is None:
        return ["[CONTRACT:CRITICAL] broker_integration: broker_client is None"]
    return []


def _check_reconciliation(state: Dict[str, Any]) -> List[str]:
    v: List[str] = []
    if state.get("reconciliation_engine") is None:
        v.append("[CONTRACT:CRITICAL] reconciliation: reconciliation_engine is None")
    if state.get("broker_client") is None:
        v.append("[CONTRACT:CRITICAL] reconciliation: broker_client dependency is None")
    if state.get("capital_state") is None:
        v.append("[CONTRACT:CRITICAL] reconciliation: capital_state dependency is None")
    return v


def _check_inflight_registry(state: Dict[str, Any]) -> List[str]:
    v: List[str] = []
    if state.get("inflight_registry") is None:
        v.append("[CONTRACT:CRITICAL] inflight_registry: inflight_registry object is None")
    if state.get("broker_client") is None:
        v.append("[CONTRACT:CRITICAL] inflight_registry: broker_client dependency is None")
    return v


def _check_runtime_supervisor(state: Dict[str, Any]) -> List[str]:
    v: List[str] = []
    if state.get("runtime_supervisor") is None:
        v.append("[CONTRACT:CRITICAL] runtime_supervisor: runtime_supervisor object is None")
    if state.get("reconciliation_engine") is None:
        v.append(
            "[CONTRACT:CRITICAL] runtime_supervisor: reconciliation_engine dependency is None"
        )
    if state.get("inflight_registry") is None:
        v.append(
            "[CONTRACT:CRITICAL] runtime_supervisor: inflight_registry dependency is None"
        )
    return v


def _check_analytics_registry(state: Dict[str, Any]) -> List[str]:
    if state.get("analytics_registry") is None:
        return ["[CONTRACT:WARN] analytics_registry: analytics_registry is None (non-critical)"]
    return []


_DEFAULT_CONTRACTS: Dict[str, ContractFn] = {
    PHASE_CONFIGURATION:       _check_configuration,
    PHASE_GOVERNANCE:          _check_governance,
    PHASE_CAPITAL_STATE:       _check_capital_state,
    PHASE_ALLOCATION:          _check_allocation,
    PHASE_BROKER_INTEGRATION:  _check_broker_integration,
    PHASE_RECONCILIATION:      _check_reconciliation,
    PHASE_INFLIGHT_REGISTRY:   _check_inflight_registry,
    PHASE_RUNTIME_SUPERVISOR:  _check_runtime_supervisor,
    PHASE_ANALYTICS_REGISTRY:  _check_analytics_registry,
}


class InitializationContractVerifier:
    """
    Verifies per-phase initialization contracts against runtime state dict.

    Violations:
      [CONTRACT:CRITICAL] → execution must not proceed
      [CONTRACT:WARN]     → degraded mode, non-blocking
    """

    def __init__(self) -> None:
        self._contracts: Dict[str, ContractFn] = dict(_DEFAULT_CONTRACTS)

    def register_contract(self, phase_name: str, fn: ContractFn) -> None:
        """Override or add a contract function for a phase."""
        self._contracts[phase_name] = fn

    def verify(self, phase_name: str, state: Dict[str, Any]) -> List[str]:
        """Run contract for one phase. Returns violation strings."""
        fn = self._contracts.get(phase_name)
        if fn is None:
            return []
        try:
            return fn(state)
        except Exception as exc:
            return [f"[CONTRACT:ERROR] {phase_name}: contract check raised {exc}"]

    def verify_all(self, state: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Run all registered contracts.
        Returns {phase_name: [violations]} for phases with violations only.
        """
        result: Dict[str, List[str]] = {}
        for phase_name in ALL_PHASES:
            violations = self.verify(phase_name, state)
            if violations:
                result[phase_name] = violations
        return result

    def all_violations(self, state: Dict[str, Any]) -> List[str]:
        """Flat list of all violations across all phases (deterministic order)."""
        viol: List[str] = []
        for phase_name in ALL_PHASES:
            viol.extend(self.verify(phase_name, state))
        return viol

    def has_critical_violations(self, state: Dict[str, Any]) -> bool:
        return any(":CRITICAL]" in v for v in self.all_violations(state))
