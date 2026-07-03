"""
src/execution
Unified deterministic execution kernel.
"""
from src.execution.intent import (
    ExecutionIntent, IntentStatus,
    ACTIVE_STATUSES, ACTIVE_STATUS_VALUES, TERMINAL_STATUSES, VALID_TRANSITIONS,
    apply_transition, make_intent, make_intent_id,
)
from src.execution.journal import IntentJournal, DEFAULT_JOURNAL_PATH
from src.execution.dd_engine import compute_drawdown, assess_risk
from src.execution.reconcile import ReconcileResult, reconcile_open_orders
from src.execution.replay import ReplayFrame, replay_intent_journal, replay_execution_state
from src.execution.alloc import FillAwareAlloc, compute_fill_aware_alloc
from src.execution.adaptive_alloc import (
    AdaptiveAllocator,
    SectorDegradeResult,
    ForecastFrame,
    REJECT_THRESHOLD,
)
from src.execution.deferred_queue import (
    DeferredEntry, DeferredQueue,
    make_deferred_entry, make_entry_id,
    ALLOWED_REJECT_REASONS, DEFAULT_TTL_DAYS, MAX_RETRY_COUNT,
)
from src.execution.fractional_demand import FractionalAccumulator, AccumulatorState
from src.execution.broker_truth import assert_positions_match, assert_deferred_isolated
from src.execution.deferred_metrics import (
    DeferredMetrics, build_metrics, emit_metrics,
    track_missed_entry, update_forward_returns,
)
from src.execution.deferred_retry import (
    RetryEvaluation,
    evaluate_retry_candidates,
    apply_retry_result,
)
from src.execution.replay_guard import (
    ReplayGuardConfig, DEFAULT_CONFIG as REPLAY_GUARD_DEFAULT_CONFIG,
    ReplayGuard,
    StormProtectionResult,
)
from src.execution.regime_drift import (
    MarketRegime,
    RegimeDriftResult,
    check_regime_drift,
    RSR_RANK_DRIFT_LIMIT,
    SECTOR_RANK_DRIFT_LIMIT,
)

__all__ = [
    # intent / journal
    "ExecutionIntent", "IntentStatus",
    "ACTIVE_STATUSES", "ACTIVE_STATUS_VALUES", "TERMINAL_STATUSES", "VALID_TRANSITIONS",
    "apply_transition", "make_intent", "make_intent_id",
    "IntentJournal", "DEFAULT_JOURNAL_PATH",
    # risk / reconcile
    "compute_drawdown", "assess_risk",
    "ReconcileResult", "reconcile_open_orders",
    # replay / alloc
    "ReplayFrame", "replay_intent_journal", "replay_execution_state",
    "FillAwareAlloc", "compute_fill_aware_alloc",
    "AdaptiveAllocator", "SectorDegradeResult", "ForecastFrame", "REJECT_THRESHOLD",
    # deferred entry layer
    "DeferredEntry", "DeferredQueue",
    "make_deferred_entry", "make_entry_id",
    "ALLOWED_REJECT_REASONS", "DEFAULT_TTL_DAYS", "MAX_RETRY_COUNT",
    # fractional demand accumulator
    "FractionalAccumulator", "AccumulatorState",
    # broker truth enforcement
    "assert_positions_match", "assert_deferred_isolated",
    # metrics / observability
    "DeferredMetrics", "build_metrics", "emit_metrics",
    "track_missed_entry", "update_forward_returns",
    # retry engine
    "RetryEvaluation", "evaluate_retry_candidates", "apply_retry_result",
    # storm protection
    "ReplayGuardConfig", "REPLAY_GUARD_DEFAULT_CONFIG",
    "ReplayGuard", "StormProtectionResult",
    # regime drift invalidation
    "MarketRegime", "RegimeDriftResult", "check_regime_drift",
    "RSR_RANK_DRIFT_LIMIT", "SECTOR_RANK_DRIFT_LIMIT",
]
