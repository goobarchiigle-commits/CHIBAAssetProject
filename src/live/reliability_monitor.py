"""
src/live/reliability_monitor.py — Study17 Production Reliability Monitor

Tracks PAPER_ACK reliability over 30 trading days.
Scores: reliability_score, authority_integrity_score,
        execution_integrity_score, state_consistency_score
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RELIABILITY_TELEMETRY_FILE = Path("logs/reliability_telemetry.jsonl")

# Stop conditions — any triggers ABORT immediately
STOP_CONDITION_CHECKS = {
    "unexpected_position":     lambda m: m.unexpected_position >= 1,
    "cash_negative":           lambda m: m.cash_negative >= 1,
    "position_truth_gap":      lambda m: m.position_truth_gap != 0,
    "cash_truth_gap":          lambda m: m.cash_truth_gap != 0,
    "orphan_order_unresolved": lambda m: m.orphan_order_count >= 1,
    "duplicate_submit":        lambda m: m.duplicate_submit_count >= 1,
    "chain_break":             lambda m: m.chain_break_count >= 1,
    "manual_override":         lambda m: m.manual_override_count >= 1,
}

# Authority integrity criteria keys (mirror Study16)
AUTHORITY_INTEGRITY_KEYS = [
    "chain_break_count",
    "duplicate_submit_count",
    "orphan_order_count",
    "manual_override_count",
    "cash_truth_gap",
    "position_truth_gap",
    "reconciliation_error_count",
    "intent_execution_divergence",
]


@dataclass
class ReliabilityMetrics:
    # Duration
    n_trading_days: int = 0
    n_signal_days: int = 0

    # API / connectivity
    api_available_days: int = 0
    broker_reconnect_count: int = 0
    api_timeout_count: int = 0

    # Latency samples (raw, for percentile computation)
    order_ack_latency_samples: List[float] = field(default_factory=list)
    cancel_ack_latency_samples: List[float] = field(default_factory=list)

    # State integrity
    state_restore_attempts: int = 0
    state_restore_success: int = 0
    session_recovery_attempts: int = 0
    session_recovery_success: int = 0

    # Order flow
    total_signals: int = 0
    approved_signals: int = 0
    cancelled_signals: int = 0
    partial_fill_count: int = 0
    partial_fill_resolved: int = 0

    # Failure counters — all must remain 0 for full score
    intent_execution_divergence: int = 0
    chain_break_count: int = 0
    manual_override_count: int = 0
    cash_truth_gap: int = 0
    position_truth_gap: int = 0
    duplicate_submit_count: int = 0
    orphan_order_count: int = 0
    reconciliation_error_count: int = 0
    unexpected_position: int = 0
    cash_negative: int = 0

    stop_condition_fired: str = ""

    # ── Computed properties ────────────────────────────────────────────

    @property
    def uptime_pct(self) -> float:
        if self.n_trading_days == 0:
            return 100.0
        return round(self.api_available_days / self.n_trading_days * 100, 2)

    @property
    def order_ack_p50(self) -> float:
        return _percentile(self.order_ack_latency_samples, 50)

    @property
    def order_ack_p95(self) -> float:
        return _percentile(self.order_ack_latency_samples, 95)

    @property
    def order_ack_p99(self) -> float:
        return _percentile(self.order_ack_latency_samples, 99)

    @property
    def cancel_ack_p50(self) -> float:
        return _percentile(self.cancel_ack_latency_samples, 50)

    @property
    def cancel_ack_p95(self) -> float:
        return _percentile(self.cancel_ack_latency_samples, 95)

    @property
    def cancel_ack_p99(self) -> float:
        return _percentile(self.cancel_ack_latency_samples, 99)

    @property
    def state_restore_success_rate(self) -> float:
        if self.state_restore_attempts == 0:
            return 100.0
        return round(self.state_restore_success / self.state_restore_attempts * 100, 2)

    @property
    def session_recovery_rate(self) -> float:
        if self.session_recovery_attempts == 0:
            return 100.0
        return round(self.session_recovery_success / self.session_recovery_attempts * 100, 2)

    @property
    def authority_precision(self) -> float:
        if self.total_signals == 0:
            return 100.0
        return round(self.approved_signals / self.total_signals * 100, 2)

    @property
    def cancel_completeness(self) -> float:
        if self.approved_signals == 0:
            return 100.0
        return round(self.cancelled_signals / self.approved_signals * 100, 2)


def _percentile(samples: List[float], p: int) -> float:
    if not samples:
        return 0.0
    return round(float(np.percentile(np.array(samples), p)), 1)


# ──────────────────────────────────────────────────────────────────────
#  Stop Condition Check
# ──────────────────────────────────────────────────────────────────────

def check_stop_conditions(metrics: ReliabilityMetrics) -> Optional[str]:
    """Returns first triggered stop condition name, or None."""
    for cond, fn in STOP_CONDITION_CHECKS.items():
        try:
            if fn(metrics):
                return cond
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────
#  Score Computation
# ──────────────────────────────────────────────────────────────────────

def compute_reliability_score(m: ReliabilityMetrics) -> float:
    """
    Composite reliability score 0–100.
      uptime           30pt   uptime_pct / 100 × 30
      reconnect        15pt   max(0, 15 – reconnect_count)
      state_restore    20pt   restore_rate / 100 × 20
      session_recovery 15pt   recovery_rate / 100 × 15
      critical_zero    20pt   –2.5 per violated critical counter
    """
    uptime_score    = (m.uptime_pct / 100.0) * 30.0
    reconnect_score = max(0.0, 15.0 - float(m.broker_reconnect_count))
    restore_score   = (m.state_restore_success_rate / 100.0) * 20.0
    recovery_score  = (m.session_recovery_rate / 100.0) * 15.0

    critical_violations = sum(
        1 for k in AUTHORITY_INTEGRITY_KEYS
        if getattr(m, k, 0) > 0
    )
    critical_score = (8 - critical_violations) * 2.5

    return round(
        uptime_score + reconnect_score + restore_score + recovery_score + critical_score,
        2,
    )


def compute_authority_integrity_score(m: ReliabilityMetrics) -> float:
    """8 criteria × 12.5pt each. Any violation deducts."""
    score = 100.0
    for k in AUTHORITY_INTEGRITY_KEYS:
        if getattr(m, k, 0) > 0:
            score -= 12.5
    return round(max(0.0, score), 2)


def compute_execution_integrity_score(m: ReliabilityMetrics) -> float:
    """
    PAPER_ACK execution integrity 0–100.
      cancel_completeness  50pt   cancelled / approved
      partial_fill_zero    25pt   partial_fill_count == 0
      divergence_zero      25pt   intent_execution_divergence == 0
    """
    if m.approved_signals == 0:
        return 100.0
    cancel_rate  = m.cancelled_signals / m.approved_signals
    cancel_score = cancel_rate * 50.0
    pf_score     = 25.0 if m.partial_fill_count == 0 else 0.0
    div_score    = 25.0 if m.intent_execution_divergence == 0 else 0.0
    return round(cancel_score + pf_score + div_score, 2)


def compute_state_consistency_score(m: ReliabilityMetrics) -> float:
    """
    State consistency 0–100.
      cash_truth_zero       25pt
      position_truth_zero   25pt
      reconciliation_zero   25pt
      restore_rate          25pt
    """
    s = 0.0
    s += 25.0 if m.cash_truth_gap == 0 else 0.0
    s += 25.0 if m.position_truth_gap == 0 else 0.0
    s += 25.0 if m.reconciliation_error_count == 0 else 0.0
    s += (m.state_restore_success_rate / 100.0) * 25.0
    return round(s, 2)


# ──────────────────────────────────────────────────────────────────────
#  Production Readiness Determination
# ──────────────────────────────────────────────────────────────────────

def determine_production_readiness(
    reliability_score: float,
    authority_integrity_score: float,
    execution_integrity_score: float,
    state_consistency_score: float,
    stop_fired: str,
    n_days: int,
) -> tuple[str, str]:
    """Returns (readiness_label, recommended_next_authority)."""
    if stop_fired:
        return "ABORTED", "SHADOW"
    if n_days < 30:
        return "INSUFFICIENT_DURATION", "PAPER_ACK"
    all_pass = (
        reliability_score >= 95.0
        and authority_integrity_score >= 100.0
        and execution_integrity_score >= 90.0
        and state_consistency_score >= 95.0
    )
    if all_pass:
        return "GO_LIMITED_LIVE", "LIMITED_LIVE"
    if reliability_score >= 90.0 and authority_integrity_score >= 100.0:
        return "CONTINUE_PAPER_ACK", "PAPER_ACK"
    if authority_integrity_score < 100.0:
        return "NO_GO_INTEGRITY_FAILURE", "BROKER_DRY_RUN"
    return "NO_GO", "PAPER_ACK"


# ──────────────────────────────────────────────────────────────────────
#  Telemetry
# ──────────────────────────────────────────────────────────────────────

def append_daily_telemetry(
    record: dict,
    path: Path = RELIABILITY_TELEMETRY_FILE,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[RELIABILITY] telemetry write failed: %s", e)
