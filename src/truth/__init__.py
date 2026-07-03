"""
src/truth/ — Deterministic portfolio truth core

Provides authoritative ground truth for the runtime's portfolio state under
execution uncertainty. All capital allocation decisions should read from here.

Layers:
  execution_truth  — monotonic order lifecycle state machine
  position_truth   — authoritative holdings from broker + intent journal
  capital_truth    — deployable capital under uncertainty (3 views)
  intent_snapshot  — SHA-256 hash-chained append-only audit log
  confidence       — composite truth confidence score [0, 1]
  reconciler       — full pipeline orchestrator → TruthState
  invariants       — cross-layer runtime assertions
  telemetry        — append-only JSONL observation log

Quick start:
  from src.truth import reconcile, TruthState
  state = reconcile(snapshot_path, broker_positions, equity, broker_available, ...)
  if state.deployment_allowed:
      capital = state.effective_deployable
"""

# ── Execution truth ───────────────────────────────────────────────────────────
from src.truth.execution_truth import (
    ExecutionRecord,
    ExposureBreakdown,
    InvalidTransitionError,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
    UNCERTAIN_STATES,
    UNKNOWN_STATES,
    PENDING_STATES,
    ALL_VALID_STATES,
    can_transition,
    transition_state,
    compute_exposure_breakdown,
    record_to_dict,
    record_from_dict,
    append_execution_record,
    load_execution_records,
    mark_unknown_stale,
)

# ── Position truth ────────────────────────────────────────────────────────────
from src.truth.position_truth import (
    PositionRecord,
    OrphanRecord,
    PositionTruth,
    build_position_from_records,
    merge_broker_position,
    build_position_truth,
    save_position_truth,
    load_position_truth,
)

# ── Capital truth ─────────────────────────────────────────────────────────────
from src.truth.capital_truth import (
    CapitalTruth,
    compute_capital_truth,
    build_capital_truth_from_components,
    check_capital_invariants,
    save_capital_truth,
    load_capital_truth,
    MIN_CONFIDENCE_TO_DEPLOY,
    CASH_BUFFER_DEFAULT,
    EXPOSURE_TOLERANCE,
)

# ── Intent snapshot ───────────────────────────────────────────────────────────
from src.truth.intent_snapshot import (
    IntentSnapshot,
    GENESIS_HASH,
    make_snapshot,
    finalize_snapshot,
    append_snapshot,
    load_chain,
    verify_chain,
    replay_chain,
    get_tail_hash,
)

# ── Confidence scoring ────────────────────────────────────────────────────────
from src.truth.confidence import (
    ConfidenceScore,
    compute_confidence,
    confidence_from_truth_state,
    score_broker_factor,
    score_unknown_factor,
    score_staleness_factor,
    score_orphan_factor,
    BROKER_WEIGHT,
    UNKNOWN_WEIGHT,
    STALE_WEIGHT,
    ORPHAN_WEIGHT,
    STALE_HALF_LIFE_SEC,
)

# ── Reconciler ────────────────────────────────────────────────────────────────
from src.truth.reconciler import (
    TruthState,
    reconcile,
    reconcile_from_records,
)

# ── Invariants ────────────────────────────────────────────────────────────────
from src.truth.invariants import (
    check_execution_invariants,
    check_position_invariants,
    check_capital_layer_invariants,
    check_cross_layer_invariants,
    check_all_invariants,
    assert_no_violations,
)

# ── Telemetry ─────────────────────────────────────────────────────────────────
from src.truth.telemetry import (
    TruthTelemetryEntry,
    build_telemetry_entry,
    append_telemetry,
    load_telemetry,
    load_recent_telemetry,
)

__all__ = [
    # execution_truth
    "ExecutionRecord", "ExposureBreakdown", "InvalidTransitionError",
    "VALID_TRANSITIONS", "TERMINAL_STATES", "UNCERTAIN_STATES",
    "UNKNOWN_STATES", "PENDING_STATES", "ALL_VALID_STATES",
    "can_transition", "transition_state", "compute_exposure_breakdown",
    "record_to_dict", "record_from_dict",
    "append_execution_record", "load_execution_records", "mark_unknown_stale",
    # position_truth
    "PositionRecord", "OrphanRecord", "PositionTruth",
    "build_position_from_records", "merge_broker_position", "build_position_truth",
    "save_position_truth", "load_position_truth",
    # capital_truth
    "CapitalTruth", "compute_capital_truth", "build_capital_truth_from_components",
    "check_capital_invariants", "save_capital_truth", "load_capital_truth",
    "MIN_CONFIDENCE_TO_DEPLOY", "CASH_BUFFER_DEFAULT", "EXPOSURE_TOLERANCE",
    # intent_snapshot
    "IntentSnapshot", "GENESIS_HASH",
    "make_snapshot", "finalize_snapshot", "append_snapshot",
    "load_chain", "verify_chain", "replay_chain", "get_tail_hash",
    # confidence
    "ConfidenceScore", "compute_confidence", "confidence_from_truth_state",
    "score_broker_factor", "score_unknown_factor",
    "score_staleness_factor", "score_orphan_factor",
    "BROKER_WEIGHT", "UNKNOWN_WEIGHT", "STALE_WEIGHT", "ORPHAN_WEIGHT",
    "STALE_HALF_LIFE_SEC",
    # reconciler
    "TruthState", "reconcile", "reconcile_from_records",
    # invariants
    "check_execution_invariants", "check_position_invariants",
    "check_capital_layer_invariants", "check_cross_layer_invariants",
    "check_all_invariants", "assert_no_violations",
    # telemetry
    "TruthTelemetryEntry", "build_telemetry_entry",
    "append_telemetry", "load_telemetry", "load_recent_telemetry",
]
