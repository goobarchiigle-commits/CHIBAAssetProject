"""
src/live/live_sandbox_gate.py — Study18 Live Sandbox Exposure Gate

Minimum-exposure live broker validation framework.
Tracks all operational, broker-truth, recovery and advanced metrics
required to gate progression from PAPER_ACK → LIMITED_LIVE.

No profit optimization. Operational verification only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Exposure Constants ───────────────────────────────────────────────
CAPITAL              = 1_800_000.0
MAX_NOTIONAL         = min(0.20 * CAPITAL, 400_000.0)   # ¥360,000
MAX_OPEN_POSITIONS   = 1
MAX_ORDER_PER_DAY    = 1
MAX_CANCEL_RETRY     = 1
P95_DD_STUDY15B      = 140_000.0
RISK_GUARD_THRESHOLD = 0.5 * P95_DD_STUDY15B            # ¥70,000
COMPLETION_DAYS      = 30
COMPLETION_FILLS     = 10

SANDBOX_STATE_FILE   = Path("runtime/sandbox_state.json")
SANDBOX_AUDIT_LOG    = Path("logs/live_sandbox_audit.jsonl")
SANDBOX_TELEMETRY    = Path("logs/sandbox_telemetry.jsonl")


# ──────────────────────────────────────────────────────────────────────
#  Sandbox Metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SandboxMetrics:
    # Duration
    n_trading_days: int = 0
    n_signal_days: int = 0
    api_available_days: int = 0

    # Signal / authority / execution counts
    total_signals: int = 0
    matched_signals: int = 0      # hash-valid signals
    approved_signals: int = 0     # passed authority check
    executed_signals: int = 0     # submitted to broker
    filled_orders: int = 0        # confirmed fills
    failed_fills: int = 0         # timeout / rejected fills
    partial_fill_count: int = 0
    cancelled_orders: int = 0

    # Exposure
    total_notional_submitted: float = 0.0
    total_notional_filled: float = 0.0
    exposure_utilization_pct_samples: List[float] = field(default_factory=list)
    orders_per_day_max: int = 0
    exposure_exceeded_count: int = 0     # orders rejected by exposure cap

    # Latency samples (ms)
    order_roundtrip_ms_samples: List[float] = field(default_factory=list)
    intent_to_fill_ms_samples: List[float] = field(default_factory=list)

    # Integrity
    reconciliation_error: int = 0
    partial_fill_gap_total: int = 0
    cash_truth_gap: int = 0
    position_truth_gap: int = 0
    cancel_integrity_failures: int = 0

    # Broker truth incidents
    broker_cash_diff_incidents: int = 0
    broker_position_diff_incidents: int = 0
    broker_order_diff_incidents: int = 0

    # Connectivity
    broker_disconnect_count: int = 0

    # Recovery
    restart_recovery_count: int = 0
    restart_recovery_success: int = 0
    restart_recovery_time_s_samples: List[float] = field(default_factory=list)

    # Advanced monitoring
    silent_failure_count: int = 0
    ghost_position_count: int = 0
    manual_intervention: int = 0
    intent_queue_depth_max: int = 0
    state_recovery_time_max_s: float = 0.0
    clock_skew_sec_max: float = 0.0
    unmatched_execution: int = 0

    # P&L / Risk guard
    realized_pnl: float = 0.0
    realized_dd: float = 0.0
    peak_equity: float = CAPITAL
    risk_guard_fired: bool = False

    # Stop condition emergency counters
    unexpected_position: int = 0
    cash_negative: int = 0
    duplicate_submit: int = 0
    orphan_order: int = 0
    manual_override: int = 0
    broker_state_unknown: int = 0

    stop_condition_fired: str = ""

    # ── Computed properties ──────────────────────────────────────────

    @property
    def live_fill_rate(self) -> float:
        denom = self.filled_orders + self.failed_fills
        return 1.0 if denom == 0 else round(self.filled_orders / denom, 4)

    @property
    def signal_match(self) -> float:
        return 1.0 if self.total_signals == 0 else round(
            self.matched_signals / self.total_signals, 4)

    @property
    def authority_match(self) -> float:
        return 1.0 if self.total_signals == 0 else round(
            self.approved_signals / self.total_signals, 4)

    @property
    def execution_match(self) -> float:
        return 1.0 if self.executed_signals == 0 else round(
            self.filled_orders / self.executed_signals, 4)

    @property
    def cancel_integrity_rate(self) -> float:
        if self.cancelled_orders == 0:
            return 1.0
        return round(
            (self.cancelled_orders - self.cancel_integrity_failures)
            / self.cancelled_orders, 4)

    @property
    def api_uptime_pct(self) -> float:
        return 100.0 if self.n_trading_days == 0 else round(
            self.api_available_days / self.n_trading_days * 100.0, 2)

    @property
    def restart_recovery_rate(self) -> float:
        return 1.0 if self.restart_recovery_count == 0 else round(
            self.restart_recovery_success / self.restart_recovery_count, 4)

    @property
    def restart_recovery_time_p95_s(self) -> float:
        samples = self.restart_recovery_time_s_samples
        return 0.0 if not samples else round(
            float(np.percentile(np.array(samples), 95)), 2)

    @property
    def order_roundtrip_p50_ms(self) -> float:
        s = self.order_roundtrip_ms_samples
        return 0.0 if not s else round(float(np.percentile(np.array(s), 50)), 1)

    @property
    def order_roundtrip_p95_ms(self) -> float:
        s = self.order_roundtrip_ms_samples
        return 0.0 if not s else round(float(np.percentile(np.array(s), 95)), 1)

    @property
    def intent_to_fill_p50_ms(self) -> float:
        s = self.intent_to_fill_ms_samples
        return 0.0 if not s else round(float(np.percentile(np.array(s), 50)), 1)

    @property
    def intent_to_fill_p95_ms(self) -> float:
        s = self.intent_to_fill_ms_samples
        return 0.0 if not s else round(float(np.percentile(np.array(s), 95)), 1)

    @property
    def exposure_utilization_avg_pct(self) -> float:
        s = self.exposure_utilization_pct_samples
        return 0.0 if not s else round(float(np.mean(np.array(s))), 2)


# ──────────────────────────────────────────────────────────────────────
#  Promotion Criteria & Stop Conditions
# ──────────────────────────────────────────────────────────────────────

PROMOTION_CRITERIA: Dict[str, Callable] = {
    "authority_match":          lambda m: m.authority_match >= 1.0,
    "execution_match":          lambda m: m.execution_match >= 1.0,
    "reconciliation_error":     lambda m: m.reconciliation_error == 0,
    "ghost_position_count":     lambda m: m.ghost_position_count == 0,
    "manual_intervention":      lambda m: m.manual_intervention == 0,
    "live_fill_rate":           lambda m: m.live_fill_rate >= 0.95,
    "state_recovery_time":      lambda m: m.state_recovery_time_max_s <= 60.0,
    "unmatched_execution":      lambda m: m.unmatched_execution == 0,
    "broker_cash_diff":         lambda m: m.broker_cash_diff_incidents == 0,
    "broker_position_diff":     lambda m: m.broker_position_diff_incidents == 0,
    "broker_order_diff":        lambda m: m.broker_order_diff_incidents == 0,
    "restart_recovery_success": lambda m: m.restart_recovery_rate >= 1.0,
}

STOP_CONDITIONS_MAP: Dict[str, Callable] = {
    "unexpected_position":     lambda m: m.unexpected_position >= 1,
    "cash_negative":           lambda m: m.cash_negative >= 1,
    "duplicate_submit":        lambda m: m.duplicate_submit >= 1,
    "orphan_order":            lambda m: m.orphan_order >= 1,
    "manual_override":         lambda m: m.manual_override >= 1,
    "broker_state_unknown":    lambda m: m.broker_state_unknown >= 1,
    "intent_queue_depth":      lambda m: m.intent_queue_depth_max > 1,
    "clock_skew":              lambda m: m.clock_skew_sec_max > 2.0,
    "unmatched_execution":     lambda m: m.unmatched_execution >= 1,
    "broker_cash_diff":        lambda m: m.broker_cash_diff_incidents > 0,
    "broker_position_diff":    lambda m: m.broker_position_diff_incidents > 0,
    "broker_order_diff":       lambda m: m.broker_order_diff_incidents > 0,
    "rollback":                lambda m: m.risk_guard_fired,
}


def check_stop_conditions(metrics: SandboxMetrics) -> Optional[str]:
    for name, fn in STOP_CONDITIONS_MAP.items():
        try:
            if fn(metrics):
                return name
        except Exception:
            pass
    return None


def check_promotion_criteria(metrics: SandboxMetrics) -> Dict[str, bool]:
    return {name: fn(metrics) for name, fn in PROMOTION_CRITERIA.items()}


# ──────────────────────────────────────────────────────────────────────
#  Score Computation
# ──────────────────────────────────────────────────────────────────────

_CRITICAL_COUNTERS = [
    "chain_break_equiv", "duplicate_submit", "orphan_order",
    "manual_override", "unmatched_execution",
]


def compute_live_reliability_score(m: SandboxMetrics) -> float:
    """
    0-100 composite:
      api_uptime     20pt
      fill_quality   20pt
      signal_match   15pt
      authority_match 15pt
      exec_match     15pt
      critical_zero  15pt  (-3pt per violated critical counter)
    """
    uptime_score    = (m.api_uptime_pct / 100.0) * 20.0
    fill_score      = m.live_fill_rate * 20.0
    signal_score    = 15.0 if m.signal_match >= 1.0 else m.signal_match * 15.0
    authority_score = 15.0 if m.authority_match >= 1.0 else m.authority_match * 15.0
    exec_score      = 15.0 if m.execution_match >= 1.0 else m.execution_match * 15.0

    # Critical integrity violations
    critical_violations = sum([
        m.reconciliation_error > 0,
        m.duplicate_submit > 0,
        m.orphan_order > 0,
        m.unmatched_execution > 0,
        m.ghost_position_count > 0,
    ])
    critical_score = max(0.0, 15.0 - critical_violations * 3.0)

    return round(
        uptime_score + fill_score + signal_score + authority_score
        + exec_score + critical_score,
        2,
    )


def compute_broker_consistency_score(m: SandboxMetrics) -> float:
    """
    0-100: 3 equal components — cash, position, order diff.
    Any incident deducts the full component.
    """
    score = 0.0
    score += 33.0 if m.broker_cash_diff_incidents == 0 else 0.0
    score += 33.0 if m.broker_position_diff_incidents == 0 else 0.0
    score += 34.0 if m.broker_order_diff_incidents == 0 else 0.0
    return round(score, 2)


def compute_restart_recovery_score(m: SandboxMetrics) -> float:
    """
    0-100:
      restart_recovery_rate × 50
      time component:  30 × max(0, 1 - p95_time / 60)
      silent_failure=0: 20
    """
    rate_score   = m.restart_recovery_rate * 50.0
    p95          = m.restart_recovery_time_p95_s
    time_score   = max(0.0, 1.0 - p95 / 60.0) * 30.0 if p95 > 0 else 30.0
    silent_score = 20.0 if m.silent_failure_count == 0 else 0.0
    return round(rate_score + time_score + silent_score, 2)


def compute_capital_activation_ratio(m: SandboxMetrics) -> float:
    """
    total_notional_filled / (n_trading_days × MAX_NOTIONAL).
    Represents sandbox budget utilization rate.
    """
    budget = m.n_trading_days * MAX_NOTIONAL
    if budget <= 0:
        return 0.0
    return round(m.total_notional_filled / budget, 4)


# ──────────────────────────────────────────────────────────────────────
#  Production Decision
# ──────────────────────────────────────────────────────────────────────

def determine_production_decision(
    live_score: float,
    broker_score: float,
    recovery_score: float,
    promo: Dict[str, bool],
    stop_fired: str,
    risk_guard_fired: bool,
    filled_orders: int,
) -> str:
    """Returns GO_SCALE | GO_HOLD | ROLLBACK."""
    if stop_fired or risk_guard_fired:
        return "ROLLBACK"
    if broker_score < 100.0:
        return "ROLLBACK"

    all_pass = all(promo.values())
    has_fills = filled_orders > 0

    if all_pass and has_fills and live_score >= 95.0 and recovery_score >= 95.0:
        return "GO_SCALE"
    if live_score >= 85.0 and recovery_score >= 90.0:
        return "GO_HOLD"
    return "ROLLBACK"


# ──────────────────────────────────────────────────────────────────────
#  Telemetry
# ──────────────────────────────────────────────────────────────────────

def append_sandbox_telemetry(record: dict, path: Path = SANDBOX_TELEMETRY) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[SANDBOX] telemetry write failed: %s", e)
