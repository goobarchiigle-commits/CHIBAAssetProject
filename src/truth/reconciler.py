"""
reconciler.py — Multi-source truth reconciliation engine

Orchestrates the full truth pipeline each cycle:
  1. Replay intent snapshots → execution records (authoritative order journal)
  2. Mark stale PENDING orders (age > threshold) → UNKNOWN
  3. Build position truth: intent journal + broker API overlay
  4. Detect orphan broker positions (no matching intent record)
  5. Score truth confidence from broker availability, unknowns, staleness, orphans
  6. Compute capital truth: three deployable views (conservative/optimistic/effective)

Output: TruthState — the single authoritative runtime state for all allocation decisions.

Invariants upheld:
  - UNKNOWN orders are never auto-resolved (only explicit broker confirmation)
  - Orphans are never silently merged into confirmed holdings
  - Deployment is blocked when confidence < MIN_CONFIDENCE_TO_DEPLOY (0.70)
  - Capital truth is always recomputed from live records — never cached across cycles
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.truth.execution_truth import (
    ExecutionRecord, mark_unknown_stale,
    TERMINAL_STATES, UNKNOWN_STATES, PENDING_STATES,
    compute_exposure_breakdown,
)
from src.truth.intent_snapshot import (
    IntentSnapshot, load_chain, replay_chain, get_tail_hash, verify_chain,
)
from src.truth.position_truth import (
    PositionRecord, OrphanRecord, PositionTruth, build_position_truth,
)
from src.truth.capital_truth import (
    CapitalTruth, compute_capital_truth, check_capital_invariants,
    CASH_BUFFER_DEFAULT, MIN_CONFIDENCE_TO_DEPLOY,
)
from src.truth.confidence import ConfidenceScore, confidence_from_truth_state

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


@dataclass
class TruthState:
    """
    Complete authoritative runtime state for one reconciliation cycle.

    All allocation decisions must read from this struct — never from
    individual component files directly.
    """
    execution_records:    list          # list[ExecutionRecord]
    position_truth:       PositionTruth
    capital_truth:        CapitalTruth
    confidence:           ConfidenceScore

    reconciled_at:        str
    trading_day:          str
    broker_available:     bool
    stale_orders_promoted: int          # PENDING → UNKNOWN promotions this cycle
    chain_valid:          bool          # False if snapshot chain fails verification
    chain_violations:     list          # list[str]

    schema_version: int = 1

    @property
    def deployment_allowed(self) -> bool:
        return self.capital_truth.deployment_allowed

    @property
    def effective_deployable(self) -> float:
        return self.capital_truth.effective_deployable

    @property
    def n_unknown(self) -> int:
        return sum(1 for r in self.execution_records if r.state in UNKNOWN_STATES)

    @property
    def n_pending(self) -> int:
        return sum(1 for r in self.execution_records if r.state in PENDING_STATES)

    @property
    def n_orphan(self) -> int:
        return len(self.position_truth.orphans)


def reconcile(
    snapshot_path: Path,
    broker_positions: dict,          # {symbol: {lots, avg_cost, market_price, sector}}
    actual_equity: float,
    broker_available: bool,
    sector_map: dict,                # {symbol: sector_str}
    trading_day: str = "",
    cash_buffer_ratio: float = CASH_BUFFER_DEFAULT,
    stale_after_minutes: int = 30,
    reconcile_age_sec: float = 0.0,
) -> TruthState:
    """
    Full truth reconciliation — authoritative pipeline for one cycle.

    Args:
        snapshot_path:      path to intent_snapshots.jsonl
        broker_positions:   live broker position data ({symbol: {lots, avg_cost, ...}})
        actual_equity:      broker-reported total account equity (JPY)
        broker_available:   True if broker API responded successfully this cycle
        sector_map:         sector string per symbol (for position grouping)
        trading_day:        YYYY-MM-DD; defaults to today JST
        cash_buffer_ratio:  fraction of equity held as cash reserve
        stale_after_minutes: PENDING orders older than this → promoted to UNKNOWN
        reconcile_age_sec:  seconds since last reconciliation (for staleness scoring)
    """
    now_str = datetime.now(JST).isoformat()
    if not trading_day:
        trading_day = datetime.now(JST).date().isoformat()

    # ── Step 1: Load and replay intent snapshots ──────────────────────────────
    snapshots = load_chain(snapshot_path)
    chain_valid, chain_violations = verify_chain(snapshots)
    if not chain_valid:
        for v in chain_violations:
            logger.warning("truth reconcile: chain violation: %s", v)

    execution_records: list[ExecutionRecord] = list(replay_chain(snapshots).values())

    # ── Step 2: Mark stale PENDING → UNKNOWN ──────────────────────────────────
    before_unknown = sum(1 for r in execution_records if r.state in UNKNOWN_STATES)
    execution_records = mark_unknown_stale(execution_records, stale_after_minutes)
    after_unknown = sum(1 for r in execution_records if r.state in UNKNOWN_STATES)
    stale_promoted = max(0, after_unknown - before_unknown)

    # ── Step 3: Build position truth (inferred + broker overlay + orphans) ────
    position_truth = build_position_truth(
        execution_records = execution_records,
        broker_positions  = broker_positions if broker_available else {},
        sector_map        = sector_map,
        trading_day       = trading_day,
    )

    # ── Step 4: Score confidence ───────────────────────────────────────────────
    confidence = confidence_from_truth_state(
        broker_available  = broker_available,
        execution_records = execution_records,
        orphan_records    = position_truth.orphans,
        reconcile_age_sec = reconcile_age_sec,
    )

    # ── Step 5: Compute capital truth ─────────────────────────────────────────
    breakdown = compute_exposure_breakdown(execution_records)
    orphan_val = sum(getattr(o, "estimated_value", 0.0) for o in position_truth.orphans)

    capital_truth = compute_capital_truth(
        actual_equity      = actual_equity,
        confirmed_exposure = breakdown.confirmed_value,
        pending_exposure   = breakdown.pending_value,
        uncertain_exposure = breakdown.uncertain_value,
        orphan_exposure    = orphan_val,
        confidence         = confidence.score,
        n_unknown_orders   = breakdown.total_uncertain_orders,
        n_pending_orders   = breakdown.total_pending_orders,
        n_orphan_positions = len(position_truth.orphans),
        cash_buffer_ratio  = cash_buffer_ratio,
    )

    # Log any capital invariant violations
    violations = check_capital_invariants(capital_truth)
    for v in violations:
        logger.error("truth reconcile: capital invariant violation: %s", v)

    return TruthState(
        execution_records    = execution_records,
        position_truth       = position_truth,
        capital_truth        = capital_truth,
        confidence           = confidence,
        reconciled_at        = now_str,
        trading_day          = trading_day,
        broker_available     = broker_available,
        stale_orders_promoted = stale_promoted,
        chain_valid          = chain_valid,
        chain_violations     = chain_violations,
    )


def reconcile_from_records(
    execution_records: list[ExecutionRecord],
    broker_positions: dict,
    actual_equity: float,
    broker_available: bool,
    sector_map: dict,
    trading_day: str = "",
    cash_buffer_ratio: float = CASH_BUFFER_DEFAULT,
    stale_after_minutes: int = 30,
    reconcile_age_sec: float = 0.0,
) -> TruthState:
    """
    Reconcile from pre-loaded execution records (skips snapshot load/replay).
    Useful when caller already has records from load_execution_records().
    """
    now_str = datetime.now(JST).isoformat()
    if not trading_day:
        trading_day = datetime.now(JST).date().isoformat()

    before_unknown = sum(1 for r in execution_records if r.state in UNKNOWN_STATES)
    records = mark_unknown_stale(execution_records, stale_after_minutes)
    after_unknown = sum(1 for r in records if r.state in UNKNOWN_STATES)
    stale_promoted = max(0, after_unknown - before_unknown)

    position_truth = build_position_truth(
        execution_records = records,
        broker_positions  = broker_positions if broker_available else {},
        sector_map        = sector_map,
        trading_day       = trading_day,
    )

    confidence = confidence_from_truth_state(
        broker_available  = broker_available,
        execution_records = records,
        orphan_records    = position_truth.orphans,
        reconcile_age_sec = reconcile_age_sec,
    )

    breakdown = compute_exposure_breakdown(records)
    orphan_val = sum(getattr(o, "estimated_value", 0.0) for o in position_truth.orphans)

    capital_truth = compute_capital_truth(
        actual_equity      = actual_equity,
        confirmed_exposure = breakdown.confirmed_value,
        pending_exposure   = breakdown.pending_value,
        uncertain_exposure = breakdown.uncertain_value,
        orphan_exposure    = orphan_val,
        confidence         = confidence.score,
        n_unknown_orders   = breakdown.total_uncertain_orders,
        n_pending_orders   = breakdown.total_pending_orders,
        n_orphan_positions = len(position_truth.orphans),
        cash_buffer_ratio  = cash_buffer_ratio,
    )

    violations = check_capital_invariants(capital_truth)
    for v in violations:
        logger.error("truth reconcile: capital invariant violation: %s", v)

    return TruthState(
        execution_records    = records,
        position_truth       = position_truth,
        capital_truth        = capital_truth,
        confidence           = confidence,
        reconciled_at        = now_str,
        trading_day          = trading_day,
        broker_available     = broker_available,
        stale_orders_promoted = stale_promoted,
        chain_valid          = True,
        chain_violations     = [],
    )
