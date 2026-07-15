"""
src/runtime/severity_classifier.py
Invariant-aware severity escalation: GovernanceSummary → List[AnomalyEvent].

Rules are explicit and exhaustive.  Adding a new rule requires adding a new
method here — no implicit logic, no dynamic dispatch.

Severity ladder:
  INFO     — informational, no action required
  WARN     — operator should review; does not block execution
  CRITICAL — requires immediate operator attention; may block next run
"""
from __future__ import annotations

from typing import List

from src.runtime.runtime_summary import (
    CRITICAL, INFO, WARN,
    AnomalyEvent,
    GovernanceSummary,
)


class SeverityClassifier:
    """
    Stateless classifier.  classify() is pure and deterministic.
    Same GovernanceSummary always produces the same anomaly list.
    Output order is deterministic (rule declaration order).
    """

    def classify(self, summary: GovernanceSummary) -> List[AnomalyEvent]:
        anomalies: List[AnomalyEvent] = []
        anomalies.extend(self._lock_rules(summary))
        anomalies.extend(self._governance_rules(summary))
        anomalies.extend(self._execution_rules(summary))
        anomalies.extend(self._allocation_rules(summary))
        anomalies.extend(self._risk_rules(summary))
        anomalies.extend(self._data_health_rules(summary))
        anomalies.extend(self._run_rules(summary))
        return anomalies

    # ── lock rules ────────────────────────────────────────────────────────────

    def _lock_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out = []
        lk = s.lock

        if lk.stale_detected and lk.reclaimed:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "STALE_LOCK_RECLAIMED",
                description = (
                    f"Stale execution lock was auto-reclaimed "
                    f"(reason={lk.reclaim_reason or 'unknown'})."
                ),
                section     = "lock",
            ))

        if lk.stale_detected and not lk.reclaimed:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "STALE_LOCK_NOT_RECLAIMED",
                description = "Stale lock detected but reclaim did not complete.",
                section     = "lock",
            ))

        if lk.orphan_lock_detected:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "ORPHAN_LOCK_DETECTED",
                description = "Lock held by a PID with no corresponding run record (orphan).",
                section     = "lock",
            ))

        if lk.concurrent_blocked:
            out.append(AnomalyEvent(
                severity    = INFO,
                code        = "CONCURRENT_EXECUTION_BLOCKED",
                description = "A concurrent acquisition attempt was detected and blocked.",
                section     = "lock",
            ))

        if lk.acquisition == "unknown":
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "LOCK_STATE_UNKNOWN",
                description = "Lock acquisition state could not be determined from telemetry.",
                section     = "lock",
            ))

        # Heartbeat health
        if lk.heartbeat_gap_sec is not None and lk.heartbeat_gap_sec > 90:
            sev = CRITICAL if lk.heartbeat_gap_sec > 120 else WARN
            out.append(AnomalyEvent(
                severity    = sev,
                code        = "HEARTBEAT_STALE",
                description = (
                    f"Lock heartbeat gap={lk.heartbeat_gap_sec:.0f}s "
                    f"({'expired' if sev == CRITICAL else 'near expiry'})."
                ),
                section     = "lock",
            ))

        return out

    # ── governance rules ──────────────────────────────────────────────────────

    def _governance_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out = []
        gov = s.governance

        if not gov.replay_match:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "REPLAY_DIVERGENCE",
                description = (
                    f"Replay divergence detected "
                    f"(divergence_score={gov.divergence_score:.3f}). "
                    "Execution outputs may not be reproducible."
                ),
                section     = "governance",
            ))

        if not gov.broker_authoritative:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "BROKER_AUTHORITY_MISMATCH",
                description = "Broker state is not authoritative. Position sync required.",
                section     = "governance",
            ))

        if not gov.fail_closed:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "FAIL_OPEN_DETECTED",
                description = "Circuit breaker tripped — system did not fail closed.",
                section     = "governance",
            ))

        if gov.quarantine_events > 0:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "QUARANTINE_EVENTS",
                description = f"{gov.quarantine_events} artifact(s) quarantined this run.",
                section     = "governance",
            ))

        if gov.reconciliation_required and not gov.reconciliation_success:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "RECONCILIATION_FAILED",
                description = "Reconciliation was required but did not succeed.",
                section     = "governance",
            ))

        if gov.divergence_score > 0.5:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "HIGH_DIVERGENCE_SCORE",
                description = f"Divergence score {gov.divergence_score:.2f} exceeds 0.5 threshold.",
                section     = "governance",
            ))

        return out

    # ── execution rules ───────────────────────────────────────────────────────

    def _execution_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out = []
        ex  = s.execution

        if not ex.idempotency_pass:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "IDEMPOTENCY_VIOLATION",
                description = "Idempotency invariant violated — duplicate execution risk.",
                section     = "execution",
            ))

        if ex.rejected_orders > 0:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "ORDERS_REJECTED",
                description = f"{ex.rejected_orders} order(s) rejected during execution.",
                section     = "execution",
            ))

        if ex.retry_count > 3:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "EXCESSIVE_RETRIES",
                description = f"Execution retry count={ex.retry_count} exceeds threshold.",
                section     = "execution",
            ))

        return out

    # ── allocation rules ──────────────────────────────────────────────────────

    def _allocation_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out  = []
        alloc = s.allocation

        if alloc.missed_entries > 5:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "EXCESSIVE_MISSED_ENTRIES",
                description = (
                    f"missed_entries={alloc.missed_entries} "
                    "indicates persistent idle capital. Review deployability ranking."
                ),
                section     = "allocation",
            ))

        if alloc.capital_efficiency < 0.5 and alloc.missed_entries > 0:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "LOW_CAPITAL_EFFICIENCY",
                description = (
                    f"capital_efficiency={alloc.capital_efficiency:.3f} "
                    "with missed entries — idle capital accumulation."
                ),
                section     = "allocation",
            ))

        if alloc.concentration_score > 0.4:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "HIGH_CONCENTRATION",
                description = (
                    f"Max sector concentration={alloc.concentration_score:.3f} "
                    "exceeds 40% — review sector cap."
                ),
                section     = "allocation",
            ))

        return out

    # ── risk rules ────────────────────────────────────────────────────────────

    def _risk_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out  = []
        risk = s.risk

        if risk.drawdown >= 0.15:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "DRAWDOWN_LIMIT_BREACH",
                description = (
                    f"Portfolio drawdown={risk.drawdown:.1%} "
                    "at or beyond -15% circuit-breaker threshold."
                ),
                section     = "risk",
            ))
        elif risk.drawdown >= 0.10:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "DRAWDOWN_APPROACHING_LIMIT",
                description = (
                    f"Portfolio drawdown={risk.drawdown:.1%} "
                    "approaching -15% circuit-breaker threshold."
                ),
                section     = "risk",
            ))

        for flag in risk.emergency_flags:
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = f"EMERGENCY_FLAG_{flag.upper()}",
                description = f"Emergency flag active: {flag}",
                section     = "risk",
            ))

        return out

    # ── data health rules ─────────────────────────────────────────────────────

    def _data_health_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out = []
        dh  = s.data_health

        if dh.freshness_score < 0.7:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "LOW_DATA_FRESHNESS",
                description = (
                    f"Data freshness_score={dh.freshness_score:.2f} "
                    f"({dh.stale_snapshots} stale snapshots)."
                ),
                section     = "data_health",
            ))

        if dh.missing_data_symbols:
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "MISSING_DATA_SYMBOLS",
                description = (
                    f"{len(dh.missing_data_symbols)} symbol(s) missing data: "
                    f"{dh.missing_data_symbols[:5]}"
                    + ("..." if len(dh.missing_data_symbols) > 5 else "")
                ),
                section     = "data_health",
            ))

        return out

    # ── run rules ─────────────────────────────────────────────────────────────

    def _run_rules(self, s: GovernanceSummary) -> List[AnomalyEvent]:
        out = []

        if s.run.run_status == "failed":
            out.append(AnomalyEvent(
                severity    = CRITICAL,
                code        = "RUN_FAILED",
                description = f"Run {s.run.run_id} terminated with status=failed.",
                section     = "run",
            ))

        if s.run.run_status == "partial":
            out.append(AnomalyEvent(
                severity    = WARN,
                code        = "RUN_INCOMPLETE",
                description = (
                    f"Run {s.run.run_id} has no completion record — "
                    "may still be running or crashed without cleanup."
                ),
                section     = "run",
            ))

        return out
