"""
state_coherence_validator.py — Cross-phase state coherence validation

Detects:
  - partial initialization (mixed INITIALIZED/FAILED in the same depth tier)
  - None-state in phases marked INITIALIZED (contract violation)
  - stale runtime positions vs broker truth
  - missing capital state after capital_state phase initialized
  - broken allocation dependencies (allocator or cap_state is None post-init)
  - inconsistent inflight/reconciliation/supervisor registry states
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.bootstrap.initialization_graph import (
    PHASE_ALLOCATION,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
    PHASE_CAPITAL_STATE,
    PhaseStatus,
    RuntimeInitializationGraph,
)

# Map phase name → runtime state dict key
_PHASE_STATE_KEY: Dict[str, str] = {
    "configuration":       "config",
    "governance":          "governance",
    "capital_state":       "capital_state",
    "allocation":          "allocator",
    "broker_integration":  "broker_client",
    "reconciliation":      "reconciliation_engine",
    "inflight_registry":   "inflight_registry",
    "runtime_supervisor":  "runtime_supervisor",
    "analytics_registry":  "analytics_registry",
}


class StateCoherenceValidator:
    """
    Cross-phase coherence checks after bootstrap.

    All checks return lists of violation strings.
    Violations tagged [COHERENCE:CRITICAL] → execution must not proceed.
    Violations tagged [COHERENCE:WARN] → degraded mode, non-blocking.

    Use has_critical_violations(violations) to gate execution.
    """

    def check_partial_initialization(
        self, graph: RuntimeInitializationGraph
    ) -> List[str]:
        """
        Detect partial initialization: any tier (dependency depth) where some phases
        are INITIALIZED while others are FAILED or BLOCKED.
        """
        violations: List[str] = []
        order = graph.get_topological_order()

        # Assign depth: max(dep depths) + 1
        depth: Dict[str, int] = {}
        for name in order:
            record = graph.get_phase(name)
            if record is None or not record.dependencies:
                depth[name] = 0
            else:
                depth[name] = 1 + max(depth.get(d, 0) for d in record.dependencies)

        tiers: Dict[int, List[str]] = defaultdict(list)
        for name, d in depth.items():
            tiers[d].append(name)

        for tier_depth, phases in tiers.items():
            statuses = {
                name: (
                    graph.get_phase(name).status
                    if graph.get_phase(name) else PhaseStatus.PENDING
                )
                for name in phases
            }
            has_init   = any(s == PhaseStatus.INITIALIZED for s in statuses.values())
            has_failed = any(
                s in (PhaseStatus.FAILED, PhaseStatus.BLOCKED)
                for s in statuses.values()
            )
            if has_init and has_failed:
                failed_in_tier = sorted(
                    n for n, s in statuses.items()
                    if s in (PhaseStatus.FAILED, PhaseStatus.BLOCKED)
                )
                violations.append(
                    f"[COHERENCE:WARN] partial initialization at depth {tier_depth}: "
                    f"phases {failed_in_tier} failed/blocked in same tier as initialized phases"
                )

        return violations

    def check_none_critical_deps(
        self,
        state: Dict[str, Any],
        graph: RuntimeInitializationGraph,
    ) -> List[str]:
        """
        For each INITIALIZED phase, verify its required state object is not None.
        A phase marked INITIALIZED but with a None state object is a contract violation.
        """
        violations: List[str] = []
        for phase_name, state_key in _PHASE_STATE_KEY.items():
            record = graph.get_phase(phase_name)
            if record is None or record.status != PhaseStatus.INITIALIZED:
                continue
            if state.get(state_key) is None:
                severity = "CRITICAL" if record.is_critical else "WARN"
                violations.append(
                    f"[COHERENCE:{severity}] phase '{phase_name}' is INITIALIZED "
                    f"but state key '{state_key}' is None"
                )
        return violations

    def check_stale_runtime_positions(
        self,
        broker_positions:  Optional[Dict[str, int]],
        runtime_positions: Optional[Dict[str, int]],
    ) -> List[str]:
        """
        Stale runtime position: runtime holds qty > 0 for a symbol the broker
        shows as 0. Broker truth is authoritative.
        """
        violations: List[str] = []
        if broker_positions is None or runtime_positions is None:
            return violations

        for symbol, rq in sorted(runtime_positions.items()):
            if rq <= 0:
                continue
            bq = broker_positions.get(symbol, 0)
            if bq == 0:
                violations.append(
                    f"[COHERENCE:WARN] stale runtime position: "
                    f"'{symbol}' runtime_qty={rq} but broker_qty=0"
                )
        return violations

    def check_missing_capital_state(
        self,
        state: Dict[str, Any],
        graph: RuntimeInitializationGraph,
    ) -> List[str]:
        """
        capital_state phase INITIALIZED → cap_state object must not be None
        and must expose 'actual_equity' and 'effective_deployable'.
        """
        violations: List[str] = []
        record = graph.get_phase(PHASE_CAPITAL_STATE)
        if record is None or record.status != PhaseStatus.INITIALIZED:
            return violations

        cap = state.get("capital_state")
        if cap is None:
            violations.append(
                "[COHERENCE:CRITICAL] capital_state phase INITIALIZED "
                "but 'capital_state' object is None — missing capital state"
            )
            return violations

        for attr in ("actual_equity", "effective_deployable"):
            if not hasattr(cap, attr):
                violations.append(
                    f"[COHERENCE:CRITICAL] capital_state object missing required attribute '{attr}'"
                )

        return violations

    def check_broken_allocation_deps(
        self,
        state: Dict[str, Any],
        graph: RuntimeInitializationGraph,
    ) -> List[str]:
        """
        allocation phase INITIALIZED → allocator and capital_state must both be non-None.
        """
        violations: List[str] = []
        record = graph.get_phase(PHASE_ALLOCATION)
        if record is None or record.status != PhaseStatus.INITIALIZED:
            return violations

        if state.get("capital_state") is None:
            violations.append(
                "[COHERENCE:CRITICAL] allocation INITIALIZED "
                "but capital_state dependency is None"
            )
        if state.get("allocator") is None:
            violations.append(
                "[COHERENCE:CRITICAL] allocation INITIALIZED but allocator object is None"
            )
        return violations

    def check_inconsistent_registry_states(
        self,
        state: Dict[str, Any],
        graph: RuntimeInitializationGraph,
    ) -> List[str]:
        """
        Detect contradictory registry states:
          - inflight_registry INITIALIZED but object is None
          - reconciliation INITIALIZED without broker_client or capital_state
          - runtime_supervisor INITIALIZED without reconciliation_engine or inflight_registry
        """
        violations: List[str] = []

        inf_record = graph.get_phase(PHASE_INFLIGHT_REGISTRY)
        if inf_record and inf_record.status == PhaseStatus.INITIALIZED:
            if state.get("inflight_registry") is None:
                violations.append(
                    "[COHERENCE:CRITICAL] inflight_registry INITIALIZED but object is None"
                )

        rec_record = graph.get_phase(PHASE_RECONCILIATION)
        if rec_record and rec_record.status == PhaseStatus.INITIALIZED:
            if state.get("broker_client") is None:
                violations.append(
                    "[COHERENCE:CRITICAL] reconciliation INITIALIZED "
                    "but broker_client dependency is None"
                )
            if state.get("capital_state") is None:
                violations.append(
                    "[COHERENCE:CRITICAL] reconciliation INITIALIZED "
                    "but capital_state dependency is None"
                )

        sup_record = graph.get_phase(PHASE_RUNTIME_SUPERVISOR)
        if sup_record and sup_record.status == PhaseStatus.INITIALIZED:
            if state.get("reconciliation_engine") is None:
                violations.append(
                    "[COHERENCE:CRITICAL] runtime_supervisor INITIALIZED "
                    "but reconciliation_engine is None"
                )
            if state.get("inflight_registry") is None:
                violations.append(
                    "[COHERENCE:CRITICAL] runtime_supervisor INITIALIZED "
                    "but inflight_registry is None"
                )

        return violations

    def validate_all(
        self,
        graph:             RuntimeInitializationGraph,
        state:             Optional[Dict[str, Any]] = None,
        broker_positions:  Optional[Dict[str, int]] = None,
        runtime_positions: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Aggregate all coherence checks into a flat violation list."""
        s = state or {}
        violations: List[str] = []
        violations.extend(self.check_partial_initialization(graph))
        violations.extend(self.check_none_critical_deps(s, graph))
        violations.extend(self.check_stale_runtime_positions(broker_positions, runtime_positions))
        violations.extend(self.check_missing_capital_state(s, graph))
        violations.extend(self.check_broken_allocation_deps(s, graph))
        violations.extend(self.check_inconsistent_registry_states(s, graph))
        return violations

    @staticmethod
    def has_critical_violations(violations: List[str]) -> bool:
        return any(":CRITICAL]" in v for v in violations)
