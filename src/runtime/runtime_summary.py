"""
src/runtime/runtime_summary.py
Immutable data structures for the governance compression pipeline.

All fields match the required output schema exactly.
No logic here — pure data definitions consumed by evaluators and renderers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Severity
# ─────────────────────────────────────────────────────────────────────────────

INFO     = "INFO"
WARN     = "WARN"
CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Section dataclasses — match required output schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunSection:
    run_status:   str            # "ok" | "failed" | "partial" | "unknown"
    run_mode:     str            # "live" | "dry" | "unknown"
    run_id:       str
    epoch_id:     str
    start_time:   Optional[float]
    end_time:     Optional[float]
    duration_sec: Optional[float]
    replay_stable:      bool = True   # True = no replay divergence detected
    deployment_blocked: bool = True   # True = deployment gate blocking live orders


@dataclass
class LockSection:
    acquisition:        str             # "ok" | "blocked" | "recovered" | "none" | "unknown"
    stale_detected:     bool
    reclaimed:          bool
    concurrent_blocked: bool            # True = concurrent attempt was blocked by lock
    lock_owner_pid:     Optional[int]
    lock_age_sec:       Optional[float]
    heartbeat_gap_sec:  Optional[float]
    reclaim_reason:     str
    reclaim_success:    bool
    orphan_lock_detected: bool
    # extensions
    lease_expiry:       Optional[float]  # absolute epoch of heartbeat expiry
    fencing_token:      Optional[str]    # deterministic hash for deduplication


@dataclass
class UniverseSection:
    live_count:       int
    shadow_count:     int
    promoted_symbols: List[str]
    filtered_price:   int
    filtered_risk:    int


@dataclass
class AllocationSection:
    capital_efficiency:    float
    missed_entries:        int
    rejected_allocations:  int
    sector_utilization:    Dict[str, float]
    concentration_score:   float         # max single sector weight


@dataclass
class ExecutionSection:
    planned_orders:          int
    submitted_orders:        int
    rejected_orders:         int
    retry_count:             int
    idempotency_pass:        bool
    duplicate_order_blocked: bool
    shadow_orders:           int = 0   # planned but not submitted (deployment blocked)
    rejected_candidates:     int = 0   # candidates rejected before order planning


@dataclass
class DataHealthSection:
    stale_snapshots:       int
    cache_reuse_rate:      float
    freshness_score:       float         # 0..1
    missing_data_symbols:  List[str]
    snapshot_latency_p95:  Optional[float]
    snapshot_latency_max:  Optional[float]


@dataclass
class GovernanceSection:
    fail_closed:             bool
    replay_match:            bool
    broker_authoritative:    bool
    quarantine_events:       int
    divergence_score:        float        # 0..1; 0 = perfect replay match
    reconciliation_required: bool
    reconciliation_success:  bool


@dataclass
class RiskSection:
    portfolio_exposure:   float           # total yen
    max_sector_exposure:  float           # fraction 0..1
    drawdown:             float           # fraction 0..1
    emergency_flags:      List[str]
    sector_concentration: float = 0.0    # max single sector weight (0..1)


@dataclass
class SummarySection:
    concise_operator_explanation: str
    top_anomalies:                List[str]
    recommended_actions:          List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly event — produced by SeverityClassifier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    severity:    str       # INFO | WARN | CRITICAL
    code:        str       # machine-readable tag, e.g. "STALE_LOCK"
    description: str
    section:     str       # which section it belongs to


# ─────────────────────────────────────────────────────────────────────────────
# Top-level container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GovernanceSummary:
    run:         RunSection
    lock:        LockSection
    universe:    UniverseSection
    allocation:  AllocationSection
    execution:   ExecutionSection
    data_health: DataHealthSection
    governance:  GovernanceSection
    risk:        RiskSection
    summary:     SummarySection
    # injected after classification
    anomalies:   List[AnomalyEvent] = field(default_factory=list)
    runtime_score: int              = 0


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel defaults — used when telemetry is missing
# ─────────────────────────────────────────────────────────────────────────────

def _unknown_run() -> RunSection:
    return RunSection(
        run_status="unknown", run_mode="unknown",
        run_id="", epoch_id="",
        start_time=None, end_time=None, duration_sec=None,
        replay_stable=True, deployment_blocked=True,
    )

def _unknown_lock() -> LockSection:
    return LockSection(
        acquisition="unknown", stale_detected=False, reclaimed=False,
        concurrent_blocked=False, lock_owner_pid=None,
        lock_age_sec=None, heartbeat_gap_sec=None,
        reclaim_reason="", reclaim_success=False,
        orphan_lock_detected=False,
        lease_expiry=None, fencing_token=None,
    )

def _unknown_universe() -> UniverseSection:
    return UniverseSection(
        live_count=0, shadow_count=0, promoted_symbols=[],
        filtered_price=0, filtered_risk=0,
    )

def _unknown_allocation() -> AllocationSection:
    return AllocationSection(
        capital_efficiency=0.0, missed_entries=0,
        rejected_allocations=0, sector_utilization={},
        concentration_score=0.0,
    )

def _unknown_execution() -> ExecutionSection:
    return ExecutionSection(
        planned_orders=0, submitted_orders=0, rejected_orders=0,
        retry_count=0, idempotency_pass=True, duplicate_order_blocked=False,
        shadow_orders=0, rejected_candidates=0,
    )

def _unknown_data_health() -> DataHealthSection:
    return DataHealthSection(
        stale_snapshots=0, cache_reuse_rate=0.0, freshness_score=1.0,
        missing_data_symbols=[], snapshot_latency_p95=None, snapshot_latency_max=None,
    )

def _unknown_governance() -> GovernanceSection:
    return GovernanceSection(
        fail_closed=True, replay_match=True, broker_authoritative=True,
        quarantine_events=0, divergence_score=0.0,
        reconciliation_required=False, reconciliation_success=True,
    )

def _unknown_risk() -> RiskSection:
    return RiskSection(
        portfolio_exposure=0.0, max_sector_exposure=0.0,
        drawdown=0.0, emergency_flags=[],
        sector_concentration=0.0,
    )

def _unknown_summary() -> SummarySection:
    return SummarySection(
        concise_operator_explanation="No telemetry available.",
        top_anomalies=[], recommended_actions=[],
    )


def empty_summary() -> GovernanceSummary:
    """Returns a fully-populated GovernanceSummary with safe 'unknown' defaults."""
    return GovernanceSummary(
        run         = _unknown_run(),
        lock        = _unknown_lock(),
        universe    = _unknown_universe(),
        allocation  = _unknown_allocation(),
        execution   = _unknown_execution(),
        data_health = _unknown_data_health(),
        governance  = _unknown_governance(),
        risk        = _unknown_risk(),
        summary     = _unknown_summary(),
    )
