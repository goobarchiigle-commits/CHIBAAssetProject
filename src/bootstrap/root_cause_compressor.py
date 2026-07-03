"""
root_cause_compressor.py — Bootstrap failure root-cause compression

Produces deterministic, human-readable summaries of initialization failure chains.

Examples:
  "governance initialization failure caused capital_state corruption, blocking
   allocation, reconciliation, runtime_supervisor."

  "configuration failure blocked analytics_registry, broker_integration, governance."

Outputs are always deterministic: phase names sorted alphabetically.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.bootstrap.initialization_graph import PhaseStatus, RuntimeInitializationGraph


class RootCauseCompressor:
    """
    Analyzes a RuntimeInitializationGraph after failures have been propagated
    and compresses the failure chain into human-readable summaries.
    """

    def compress(self, graph: RuntimeInitializationGraph) -> str:
        """
        Single compressed root cause summary string.
        Returns "" if no failures exist.
        """
        root_failures = graph.get_root_failures()
        blocked       = graph.get_blocked_phases()

        if not root_failures and not blocked:
            failed = graph.get_failed_phases()
            if not failed:
                return ""
            return f"Initialization failures (non-root): {', '.join(sorted(failed))}."

        if not root_failures:
            return (
                f"Upstream failures blocked: {', '.join(sorted(blocked))}. "
                "Root cause could not be determined."
            )

        parts: List[str] = []
        for root in sorted(root_failures):
            immediate  = sorted(self._immediate_blocked_by(graph, root))
            transitive = sorted(
                self._transitive_blocked_by(graph, root) - set(immediate)
            )

            if not immediate and not transitive:
                parts.append(f"{root} initialization failed.")
            elif not transitive:
                parts.append(
                    f"{root} initialization failure blocked "
                    f"{', '.join(immediate)}."
                )
            else:
                parts.append(
                    f"{root} initialization failure caused "
                    f"{', '.join(immediate)} corruption, "
                    f"blocking {', '.join(transitive)}."
                )

        return " ".join(parts)

    def compress_chain(self, graph: RuntimeInitializationGraph) -> List[str]:
        """
        Detailed root cause chain — one entry per root failure.
        Format: "ROOT: X [→ IMMEDIATE_BLOCKED: A, B] [→ TRANSITIVE_BLOCKED: C]"
        """
        root_failures = graph.get_root_failures()
        if not root_failures:
            return []

        chains: List[str] = []
        for root in sorted(root_failures):
            record    = graph.get_phase(root)
            err_part  = f": {record.error_msg}" if (record and record.error_msg) else ""
            immediate = sorted(self._immediate_blocked_by(graph, root))
            transitive = sorted(
                self._transitive_blocked_by(graph, root) - set(immediate)
            )

            parts = [f"ROOT: {root}{err_part}"]
            if immediate:
                parts.append(f"IMMEDIATE_BLOCKED: {', '.join(immediate)}")
            if transitive:
                parts.append(f"TRANSITIVE_BLOCKED: {', '.join(transitive)}")
            chains.append(" → ".join(parts))

        return chains

    def operator_summary(
        self,
        graph:         RuntimeInitializationGraph,
        safe_action:   Optional[str] = None,
        required_fix:  Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Concise operator-facing summary.

        Keys:
          ROOT_CAUSE      — compressed failure description
          BLOCKED_SYSTEMS — sorted list of all failed + blocked phases
          SAFE_ACTION     — what the operator should do now
          REQUIRED_FIX    — what must be fixed to resume
        """
        root_failures   = graph.get_root_failures()
        blocked         = graph.get_blocked_phases()
        failed          = graph.get_failed_phases()
        root_cause_str  = self.compress(graph) or "No failures detected."
        blocked_systems = sorted(set(blocked) | set(failed))

        if safe_action is None:
            if graph.has_critical_failures():
                safe_action = "Abort live execution. Do not place orders."
            else:
                safe_action = (
                    "Non-critical failures only. Proceed with caution; "
                    "monitor analytics degradation."
                )

        if required_fix is None:
            if root_failures:
                required_fix = (
                    f"Resolve initialization of: {', '.join(sorted(root_failures))}. "
                    "Re-run bootstrap after fix."
                )
            else:
                required_fix = "No action required."

        return {
            "ROOT_CAUSE":      root_cause_str,
            "BLOCKED_SYSTEMS": blocked_systems,
            "SAFE_ACTION":     safe_action,
            "REQUIRED_FIX":    required_fix,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _immediate_blocked_by(
        self, graph: RuntimeInitializationGraph, failed_name: str
    ) -> List[str]:
        """Phases directly blocked by failed_name (single hop via blocked_by list)."""
        return sorted(
            name
            for name, record in graph._phases.items()
            if record.status == PhaseStatus.BLOCKED
            and failed_name in record.blocked_by
        )

    def _transitive_blocked_by(
        self, graph: RuntimeInitializationGraph, root: str
    ) -> Set[str]:
        """All phases transitively blocked because of root (BFS)."""
        blocked: Set[str] = set()
        queue: List[str]  = list(self._immediate_blocked_by(graph, root))

        while queue:
            current = queue.pop(0)
            if current in blocked:
                continue
            blocked.add(current)
            for name, record in graph._phases.items():
                if (
                    record.status == PhaseStatus.BLOCKED
                    and current in record.blocked_by
                    and name not in blocked
                ):
                    queue.append(name)

        return blocked
