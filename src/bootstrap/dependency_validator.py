"""
dependency_validator.py — Bootstrap dependency ordering + completeness enforcement

Fail-closed: raises BootstrapIntegrityError on any critical phase not initialized.
"""
from __future__ import annotations

from typing import List, Optional

from src.bootstrap.initialization_graph import PhaseStatus, RuntimeInitializationGraph


class BootstrapIntegrityError(RuntimeError):
    """Critical bootstrap invariant violated. Execution must not continue."""

    def __init__(self, message: str, failed_phases: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.failed_phases = list(failed_phases or [])


class BootstrapDependencyValidator:
    """
    Enforces initialization ordering and post-bootstrap completeness.

    validate_order(phase):
      Call before marking a phase initialized.
      Raises if any declared dependency is not yet INITIALIZED.

    validate_complete():
      Call after all initialization attempts.
      Raises BootstrapIntegrityError if any critical phase is not INITIALIZED.
    """

    def __init__(self, graph: RuntimeInitializationGraph) -> None:
        self._graph = graph

    def validate_order(self, phase_name: str) -> None:
        """
        Raise BootstrapIntegrityError if any dependency of phase_name is
        not yet INITIALIZED. Protects against out-of-order initialization.
        """
        record = self._graph.get_phase(phase_name)
        if record is None:
            return

        unresolved: List[str] = []
        for dep in record.dependencies:
            dep_record = self._graph.get_phase(dep)
            if dep_record is None:
                unresolved.append(f"{dep} (undeclared)")
            elif dep_record.status != PhaseStatus.INITIALIZED:
                unresolved.append(f"{dep} (status={dep_record.status.value})")

        if unresolved:
            raise BootstrapIntegrityError(
                f"Phase '{phase_name}' initialized before dependencies resolved: "
                f"{unresolved}",
                failed_phases=[phase_name],
            )

    def validate_complete(self) -> List[str]:
        """
        Verify all critical phases are INITIALIZED after bootstrap.
        Returns the violation list AND raises BootstrapIntegrityError if any
        critical phase has FAILED or BLOCKED status.
        """
        violations: List[str] = []
        hard_failures: List[str] = []

        for name, record in self._graph._phases.items():
            if not record.is_critical:
                continue
            if record.status == PhaseStatus.INITIALIZED:
                continue

            msg = (
                f"[BOOTSTRAP:CRITICAL] phase='{name}' "
                f"status={record.status.value}"
            )
            if record.blocked_by:
                msg += f" blocked_by={record.blocked_by}"
            if record.error_msg:
                msg += f" error='{record.error_msg}'"
            violations.append(msg)

            # PENDING at bootstrap completion = never attempted = integrity violation
            if record.status != PhaseStatus.INITIALIZED:
                hard_failures.append(name)

        if hard_failures:
            raise BootstrapIntegrityError(
                f"Bootstrap integrity violated: {len(hard_failures)} critical phase(s) "
                f"not initialized: {sorted(hard_failures)}",
                failed_phases=sorted(hard_failures),
            )

        return violations

    def violations_only(self) -> List[str]:
        """
        Same logic as validate_complete but never raises.
        Returns violation strings for all non-INITIALIZED critical phases.
        """
        violations: List[str] = []
        for name, record in self._graph._phases.items():
            if not record.is_critical:
                continue
            if record.status == PhaseStatus.INITIALIZED:
                continue
            violations.append(
                f"[BOOTSTRAP:CRITICAL] phase='{name}' status={record.status.value}"
                + (f" blocked_by={record.blocked_by}" if record.blocked_by else "")
            )
        return violations
