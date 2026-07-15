"""
src/runtime/runtime_score.py
Deterministic runtime score engine: 0-100.

Score is computed from GovernanceSummary + classified anomalies.
All arithmetic is explicit and auditable.
Invariant caps are applied AFTER deductions, in declared order.
No randomness. No wall-clock dependency. Replay-stable.

Score table
-----------
Base score:               100

Deductions:
  stale lock reclaimed    -15
  replay divergence       -30
  missed entries > 5      -10
  run failed              -20
  run incomplete (partial) -10
  idempotency violation   -25
  broker mismatch         -20
  fail open (CB tripped)  -20
  orphan lock             -10
  heartbeat expired       -10
  excessive retries        -5
  quarantine events (each) -5  (max -20)
  drawdown >= 15%         -15
  drawdown >= 10%          -5

Bonuses:
  fail-closed success      +5

Invariant caps (applied in order, most severe first):
  replay divergence        → max 40
  broker_authoritative=F   → max 25
  run failed               → max 30
  concurrent_blocked=T AND
    idempotency violation  → max 10
"""
from __future__ import annotations

from typing import List

from src.runtime.runtime_summary import (
    CRITICAL,
    AnomalyEvent,
    GovernanceSummary,
)

_BASE_SCORE = 100


class RuntimeScoreEngine:
    """
    Stateless score engine.  compute() is pure and deterministic.
    Returns int in [0, 100].
    """

    def compute(
        self,
        summary:   GovernanceSummary,
        anomalies: List[AnomalyEvent],
    ) -> int:
        score = _BASE_SCORE
        score = self._apply_deductions(score, summary, anomalies)
        score = self._apply_bonuses(score, summary)
        score = self._apply_caps(score, summary, anomalies)
        return max(0, min(100, score))

    # ── deductions ────────────────────────────────────────────────────────────

    def _apply_deductions(
        self,
        score:     int,
        summary:   GovernanceSummary,
        anomalies: List[AnomalyEvent],
    ) -> int:
        codes = {a.code for a in anomalies}

        if "STALE_LOCK_RECLAIMED" in codes:
            score -= 15
        if "REPLAY_DIVERGENCE" in codes:
            score -= 30
        if summary.allocation.missed_entries > 5:
            score -= 10
        if summary.run.run_status == "failed":
            score -= 20
        if summary.run.run_status == "partial":
            score -= 10
        if "IDEMPOTENCY_VIOLATION" in codes:
            score -= 25
        if "BROKER_AUTHORITY_MISMATCH" in codes:
            score -= 20
        if "FAIL_OPEN_DETECTED" in codes:
            score -= 20
        if "ORPHAN_LOCK_DETECTED" in codes:
            score -= 10
        if "HEARTBEAT_STALE" in codes:
            critical_hb = any(
                a.code == "HEARTBEAT_STALE" and a.severity == CRITICAL
                for a in anomalies
            )
            score -= 10 if critical_hb else 5
        if "EXCESSIVE_RETRIES" in codes:
            score -= 5
        if "DRAWDOWN_LIMIT_BREACH" in codes:
            score -= 15
        elif "DRAWDOWN_APPROACHING_LIMIT" in codes:
            score -= 5
        if "STALE_LOCK_NOT_RECLAIMED" in codes:
            score -= 20  # worse than reclaimed
        if "RUN_FAILED" in codes:
            # Already counted via run_status == "failed" above
            pass

        # Quarantine: -5 each, capped at -20
        q_count = summary.governance.quarantine_events
        if q_count > 0:
            score -= min(20, q_count * 5)

        return score

    # ── bonuses ───────────────────────────────────────────────────────────────

    def _apply_bonuses(self, score: int, summary: GovernanceSummary) -> int:
        if summary.governance.fail_closed:
            score += 5
        return score

    # ── invariant caps ────────────────────────────────────────────────────────

    def _apply_caps(
        self,
        score:     int,
        summary:   GovernanceSummary,
        anomalies: List[AnomalyEvent],
    ) -> int:
        codes = {a.code for a in anomalies}

        # Most severe cap first
        if "REPLAY_DIVERGENCE" in codes:
            score = min(score, 40)

        if not summary.governance.broker_authoritative:
            score = min(score, 25)

        if summary.run.run_status == "failed":
            score = min(score, 30)

        # Concurrent block failure + idempotency violation → catastrophic
        if "IDEMPOTENCY_VIOLATION" in codes:
            score = min(score, 10)

        return score


# ─────────────────────────────────────────────────────────────────────────────
# Score band labels for display
# ─────────────────────────────────────────────────────────────────────────────

def score_band(score: int) -> str:
    if score >= 85:
        return "HEALTHY"
    if score >= 65:
        return "DEGRADED"
    if score >= 40:
        return "IMPAIRED"
    return "CRITICAL"
