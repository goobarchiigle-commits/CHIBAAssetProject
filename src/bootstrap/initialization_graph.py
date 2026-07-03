"""
initialization_graph.py — Runtime bootstrap phase tracking

Tracks 9 initialization phases with their dependency relationships.
Deterministic failure propagation: mark_failed(X) blocks all downstream
phases that (transitively) depend on X.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set

# ── Phase name constants ──────────────────────────────────────────────────────

PHASE_CONFIGURATION       = "configuration"
PHASE_GOVERNANCE          = "governance"
PHASE_CAPITAL_STATE       = "capital_state"
PHASE_ALLOCATION          = "allocation"
PHASE_BROKER_INTEGRATION  = "broker_integration"
PHASE_RECONCILIATION      = "reconciliation"
PHASE_INFLIGHT_REGISTRY   = "inflight_registry"
PHASE_RUNTIME_SUPERVISOR  = "runtime_supervisor"
PHASE_ANALYTICS_REGISTRY  = "analytics_registry"

ALL_PHASES: List[str] = [
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_CAPITAL_STATE,
    PHASE_ALLOCATION,
    PHASE_BROKER_INTEGRATION,
    PHASE_RECONCILIATION,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RUNTIME_SUPERVISOR,
    PHASE_ANALYTICS_REGISTRY,
]

DEFAULT_PHASE_DEPENDENCIES: Dict[str, List[str]] = {
    PHASE_CONFIGURATION:       [],
    PHASE_GOVERNANCE:          [PHASE_CONFIGURATION],
    PHASE_CAPITAL_STATE:       [PHASE_GOVERNANCE],
    PHASE_ALLOCATION:          [PHASE_CAPITAL_STATE],
    PHASE_BROKER_INTEGRATION:  [PHASE_CONFIGURATION],
    PHASE_RECONCILIATION:      [PHASE_BROKER_INTEGRATION, PHASE_CAPITAL_STATE],
    PHASE_INFLIGHT_REGISTRY:   [PHASE_BROKER_INTEGRATION],
    PHASE_RUNTIME_SUPERVISOR:  [PHASE_RECONCILIATION, PHASE_INFLIGHT_REGISTRY],
    PHASE_ANALYTICS_REGISTRY:  [PHASE_CONFIGURATION],
}

# Execution-critical phases — fail-closed on failure
CRITICAL_PHASES: FrozenSet[str] = frozenset({
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_CAPITAL_STATE,
    PHASE_ALLOCATION,
    PHASE_BROKER_INTEGRATION,
    PHASE_RECONCILIATION,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RUNTIME_SUPERVISOR,
})

# Non-critical phases — fail-open (analytics degradation only)
NON_CRITICAL_PHASES: FrozenSet[str] = frozenset({
    PHASE_ANALYTICS_REGISTRY,
})


class PhaseStatus(str, Enum):
    PENDING     = "pending"       # not yet attempted
    INITIALIZED = "initialized"   # successfully completed
    FAILED      = "failed"        # direct initialization failure
    BLOCKED     = "blocked"       # upstream failure propagated here


@dataclass
class PhaseRecord:
    name:           str
    dependencies:   List[str]
    is_critical:    bool
    status:         PhaseStatus   = PhaseStatus.PENDING
    error_msg:      Optional[str] = None
    initialized_at: Optional[str] = None
    blocked_by:     List[str]     = field(default_factory=list)


class RuntimeInitializationGraph:
    """
    Directed acyclic graph of bootstrap initialization phases.

    Phase transitions:
      PENDING → INITIALIZED  (via mark_initialized)
      PENDING → FAILED       (via mark_failed)
      PENDING → BLOCKED      (propagated when upstream fails)

    Failure propagation is deterministic: BFS from failed node over
    reverse-dependency edges. Already-INITIALIZED phases are not overridden.
    """

    def __init__(self) -> None:
        self._phases: Dict[str, PhaseRecord] = {}

    # ── Construction ──────────────────────────────────────────────────────────

    def declare_phase(
        self,
        name:         str,
        dependencies: Optional[List[str]] = None,
        is_critical:  bool                = True,
    ) -> None:
        deps = list(dependencies or [])
        if name in self._phases:
            self._phases[name].dependencies = deps
            self._phases[name].is_critical  = is_critical
        else:
            self._phases[name] = PhaseRecord(
                name=name, dependencies=deps, is_critical=is_critical,
            )

    @classmethod
    def build_default(cls) -> "RuntimeInitializationGraph":
        """Build the canonical 9-phase bootstrap graph."""
        g = cls()
        for phase in ALL_PHASES:
            g.declare_phase(
                name=phase,
                dependencies=DEFAULT_PHASE_DEPENDENCIES[phase],
                is_critical=(phase in CRITICAL_PHASES),
            )
        return g

    # ── State transitions ─────────────────────────────────────────────────────

    def mark_initialized(self, name: str, ts: Optional[str] = None) -> None:
        """Record phase as successfully initialized."""
        if name not in self._phases:
            self._phases[name] = PhaseRecord(name=name, dependencies=[], is_critical=True)
        record                = self._phases[name]
        record.status         = PhaseStatus.INITIALIZED
        record.initialized_at = ts or datetime.now(timezone.utc).isoformat()
        record.error_msg      = None

    def mark_failed(self, name: str, error_msg: str = "") -> None:
        """
        Record phase as directly failed.
        Propagates BLOCKED status to all transitively dependent phases.
        """
        if name not in self._phases:
            self._phases[name] = PhaseRecord(name=name, dependencies=[], is_critical=True)
        record           = self._phases[name]
        record.status    = PhaseStatus.FAILED
        record.error_msg = error_msg or f"{name} initialization failed"
        self._propagate_blocked(name)

    def _propagate_blocked(self, failed_name: str) -> None:
        """BFS from failed_name over reverse-dependency edges."""
        # reverse_adj[dep] = list of phases that depend on dep
        reverse_adj: Dict[str, List[str]] = {n: [] for n in self._phases}
        for n, record in self._phases.items():
            for dep in record.dependencies:
                if dep in reverse_adj:
                    reverse_adj[dep].append(n)

        queue:   List[str] = list(reverse_adj.get(failed_name, []))
        visited: Set[str]  = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            rec = self._phases.get(current)
            if rec is None or rec.status == PhaseStatus.FAILED:
                # Don't override explicit failures with BLOCKED
                queue.extend(
                    d for d in reverse_adj.get(current, []) if d not in visited
                )
                continue

            # Identify which of this phase's deps are now failed/blocked
            blocking = [
                dep for dep in rec.dependencies
                if dep in self._phases
                and self._phases[dep].status in (PhaseStatus.FAILED, PhaseStatus.BLOCKED)
            ]
            if blocking and rec.status != PhaseStatus.INITIALIZED:
                rec.status     = PhaseStatus.BLOCKED
                rec.blocked_by = blocking

            queue.extend(
                d for d in reverse_adj.get(current, []) if d not in visited
            )

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_phase(self, name: str) -> Optional[PhaseRecord]:
        return self._phases.get(name)

    def get_topological_order(self) -> List[str]:
        """BFS topological sort — deterministic (alphabetical tie-breaking)."""
        in_degree: Dict[str, int] = {n: 0 for n in self._phases}
        for n, record in self._phases.items():
            for dep in record.dependencies:
                if dep in in_degree:
                    in_degree[n] += 1

        # Forward adjacency: dep → list of dependents
        adj: Dict[str, List[str]] = {n: [] for n in self._phases}
        for n, record in self._phases.items():
            for dep in record.dependencies:
                if dep in adj:
                    adj[dep].append(n)

        queue  = sorted(n for n, d in in_degree.items() if d == 0)
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append any remaining nodes (cycle guard — should not occur)
        remaining = sorted(set(self._phases) - set(result))
        result.extend(remaining)
        return result

    def get_failed_phases(self) -> List[str]:
        """Phases with direct FAILED status (true root failures)."""
        return sorted(
            n for n, r in self._phases.items() if r.status == PhaseStatus.FAILED
        )

    def get_blocked_phases(self) -> List[str]:
        """Phases blocked by upstream failure."""
        return sorted(
            n for n, r in self._phases.items() if r.status == PhaseStatus.BLOCKED
        )

    def get_initialized_phases(self) -> List[str]:
        return sorted(
            n for n, r in self._phases.items() if r.status == PhaseStatus.INITIALIZED
        )

    def get_pending_phases(self) -> List[str]:
        return sorted(
            n for n, r in self._phases.items() if r.status == PhaseStatus.PENDING
        )

    def get_root_failures(self) -> List[str]:
        """
        Failed phases whose dependencies are all not failed/blocked.
        These are true root causes (not cascades from upstream failure).
        """
        roots = []
        for n, r in self._phases.items():
            if r.status != PhaseStatus.FAILED:
                continue
            upstream_clean = all(
                dep not in self._phases
                or self._phases[dep].status
                not in (PhaseStatus.FAILED, PhaseStatus.BLOCKED)
                for dep in r.dependencies
            )
            if upstream_clean:
                roots.append(n)
        return sorted(roots)

    def all_critical_initialized(self) -> bool:
        for _, record in self._phases.items():
            if record.is_critical and record.status != PhaseStatus.INITIALIZED:
                return False
        return True

    def can_execute(self) -> bool:
        """True only when all critical phases are INITIALIZED."""
        return self.all_critical_initialized()

    def has_critical_failures(self) -> bool:
        for _, record in self._phases.items():
            if record.is_critical and record.status in (
                PhaseStatus.FAILED, PhaseStatus.BLOCKED
            ):
                return True
        return False

    def dump(self) -> Dict:
        order = self.get_topological_order()
        return {
            "phase_count":      len(self._phases),
            "topology_order":   order,
            "phases": {
                name: {
                    "dependencies":   self._phases[name].dependencies,
                    "is_critical":    self._phases[name].is_critical,
                    "status":         self._phases[name].status.value,
                    "error_msg":      self._phases[name].error_msg,
                    "initialized_at": self._phases[name].initialized_at,
                    "blocked_by":     self._phases[name].blocked_by,
                }
                for name in order
                if name in self._phases
            },
        }
