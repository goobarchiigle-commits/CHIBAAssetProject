"""
src/allocation/ — Economic exposure & capital efficiency layer (Phase 5A)

Transforms "execution-safe adaptive runtime" into
"capital-efficient compound-growth infrastructure."

Modules:
  exposure_core        — rolling correlation, effective independent N, theme clustering
  efficiency_ranker    — deterministic capital efficiency scoring and ranking
  explainability       — append-only allocation decision rationale
  alpha_lifecycle      — per-symbol alpha persistence and decay tracking
  stability_metrics    — allocator observability and instability detection
  complexity_budget    — factor admission governance with hard cap

Integration:
  All allocation decisions consume verified truth-layer state only.
  Never allocate from assumed or inferred exposure.

Quick start:
  from src.allocation import (
      ExposureState, update_exposure_state, compute_exposure_report,
      CandidateSignal, rank_candidates,
      AllocationDecision, build_allocation_decision, append_allocation_decision,
  )
"""

# ── Exposure core ──────────────────────────────────────────────────────────────
from src.allocation.exposure_core import (
    PairCorrelation,
    ThemeCluster,
    ExposureReport,
    ExposureState,
    compute_pair_correlations,
    compute_theme_clusters,
    compute_exposure_report,
    update_exposure_state,
    load_exposure_state,
    save_exposure_state,
    CORR_WINDOW,
    COLLAPSE_THRESHOLD,
    COLLAPSE_STREAK_REQUIRED,
)

# ── Efficiency ranker ──────────────────────────────────────────────────────────
from src.allocation.efficiency_ranker import (
    CandidateSignal,
    EfficiencyScore,
    compute_efficiency_score,
    rank_candidates,
    VOLUME_THRESHOLD_YEN,
    MIN_SCORE,
)

# ── Explainability ─────────────────────────────────────────────────────────────
from src.allocation.explainability import (
    AllocationDecision,
    record_to_dict,
    append_allocation_decision,
    load_allocation_decisions,
    load_recent_decisions,
    build_allocation_decision,
)

# ── Alpha lifecycle ────────────────────────────────────────────────────────────
from src.allocation.alpha_lifecycle import (
    AlphaRecord,
    AlphaLifecycleState,
    build_lifecycle_state,
    build_lifecycle_map,
    append_alpha_record,
    load_alpha_records,
    save_lifecycle_state,
    load_lifecycle_state,
    LIFECYCLE_WINDOW,
    DECAY_THRESHOLD,
)

# ── Stability metrics ──────────────────────────────────────────────────────────
from src.allocation.stability_metrics import (
    RankSnapshot,
    AllocatorStabilityObservation,
    StabilityWindow,
    update_stability_window,
    compute_stability_observation,
    append_stability_observation,
    load_stability_window,
    save_stability_window,
    load_recent_stability,
)

# ── Complexity budget ──────────────────────────────────────────────────────────
from src.allocation.complexity_budget import (
    AllocatorFactor,
    ComplexityBudget,
    ComplexityAuditRecord,
    can_admit_factor,
    admit_factor,
    deactivate_factor,
    append_audit_record,
    load_complexity_budget,
    save_complexity_budget,
    load_audit_records,
    MAX_FACTORS,
    MIN_MARGINAL_CAGR_BPS,
)

# ── Intent audit ──────────────────────────────────────────────────────────────
from src.allocation.intent_audit import (
    AllocationIntentRecord,
    append_intent_record,
    load_intent_records,
)

# ── Skipped trade outcome attribution ─────────────────────────────────────────
from src.allocation.skipped_trade_outcomes import (
    SkippedTradeRecord,
    SkippedTradeOutcome,
    append_skipped_trade,
    append_skipped_outcome,
    load_skipped_trades,
    load_skipped_outcomes,
    enrich_skipped_trade,
    SKIP_ALLOC_CAP,
    SKIP_CONCENTRATION_THROTTLE,
    SKIP_TRUTH_CONFIDENCE,
    VALID_SKIP_REASONS,
)

# ── Allocation outcome tracking ────────────────────────────────────────────────
from src.allocation.allocation_outcomes import (
    AllocationOutcomeRecord,
    build_outcome_record,
    enrich_outcome_record,
    append_outcome_record,
    load_outcome_records,
)

# ── JSONL monthly rotation ─────────────────────────────────────────────────────
from src.allocation.jsonl_rotation import (
    needs_rotation,
    rotate_monthly,
    rotate_if_needed,
    list_archives,
)

# ── Stale state visibility warnings ───────────────────────────────────────────
from src.allocation.stale_visibility import (
    StaleVisibilityWarning,
    append_stale_warning,
    load_stale_warnings,
    check_exposure_staleness,
    check_lifecycle_staleness,
    check_stability_staleness,
    check_fallback_activation,
    emit_stale_warnings,
    WARN_STALE_EXPOSURE,
    WARN_STALE_LIFECYCLE,
    WARN_STALE_STABILITY,
    WARN_FALLBACK_ACTIVATED,
    DEFAULT_EXPOSURE_STALE_SEC,
    DEFAULT_LIFECYCLE_STALE_SEC,
    DEFAULT_STABILITY_STALE_SEC,
)

# ── Exposure-aware sizing ──────────────────────────────────────────────────────
from src.allocation.sizing import (
    SizingResult,
    compute_exposure_aware_size,
    gate_on_truth_confidence,
    detect_stale_state,
    compute_allocation_cap,
    apply_concentration_throttle,
    compute_concentration_penalty,
    TRUTH_CONFIDENCE_GATE,
    STALE_STATE_AGE_SEC,
    MAX_CONCENTRATION_PCT,
)

__all__ = [
    # exposure_core
    "PairCorrelation", "ThemeCluster", "ExposureReport", "ExposureState",
    "compute_pair_correlations", "compute_theme_clusters", "compute_exposure_report",
    "update_exposure_state", "load_exposure_state", "save_exposure_state",
    "CORR_WINDOW", "COLLAPSE_THRESHOLD", "COLLAPSE_STREAK_REQUIRED",
    # efficiency_ranker
    "CandidateSignal", "EfficiencyScore",
    "compute_efficiency_score", "rank_candidates",
    "VOLUME_THRESHOLD_YEN", "MIN_SCORE",
    # explainability
    "AllocationDecision", "record_to_dict",
    "append_allocation_decision", "load_allocation_decisions",
    "load_recent_decisions", "build_allocation_decision",
    # alpha_lifecycle
    "AlphaRecord", "AlphaLifecycleState",
    "build_lifecycle_state", "build_lifecycle_map",
    "append_alpha_record", "load_alpha_records",
    "save_lifecycle_state", "load_lifecycle_state",
    "LIFECYCLE_WINDOW", "DECAY_THRESHOLD",
    # stability_metrics
    "RankSnapshot", "AllocatorStabilityObservation", "StabilityWindow",
    "update_stability_window", "compute_stability_observation",
    "append_stability_observation", "load_stability_window",
    "save_stability_window", "load_recent_stability",
    # complexity_budget
    "AllocatorFactor", "ComplexityBudget", "ComplexityAuditRecord",
    "can_admit_factor", "admit_factor", "deactivate_factor",
    "append_audit_record", "load_complexity_budget",
    "save_complexity_budget", "load_audit_records",
    "MAX_FACTORS", "MIN_MARGINAL_CAGR_BPS",
    # intent_audit
    "AllocationIntentRecord", "append_intent_record", "load_intent_records",
    # sizing
    "SizingResult", "compute_exposure_aware_size",
    "gate_on_truth_confidence", "detect_stale_state",
    "compute_allocation_cap", "apply_concentration_throttle",
    "compute_concentration_penalty",
    "TRUTH_CONFIDENCE_GATE", "STALE_STATE_AGE_SEC", "MAX_CONCENTRATION_PCT",
    # skipped_trade_outcomes
    "SkippedTradeRecord", "SkippedTradeOutcome",
    "append_skipped_trade", "append_skipped_outcome",
    "load_skipped_trades", "load_skipped_outcomes",
    "enrich_skipped_trade",
    "SKIP_ALLOC_CAP", "SKIP_CONCENTRATION_THROTTLE", "SKIP_TRUTH_CONFIDENCE",
    "VALID_SKIP_REASONS",
    # allocation_outcomes
    "AllocationOutcomeRecord",
    "build_outcome_record", "enrich_outcome_record",
    "append_outcome_record", "load_outcome_records",
    # jsonl_rotation
    "needs_rotation", "rotate_monthly", "rotate_if_needed", "list_archives",
    # stale_visibility
    "StaleVisibilityWarning",
    "append_stale_warning", "load_stale_warnings",
    "check_exposure_staleness", "check_lifecycle_staleness",
    "check_stability_staleness", "check_fallback_activation",
    "emit_stale_warnings",
    "WARN_STALE_EXPOSURE", "WARN_STALE_LIFECYCLE",
    "WARN_STALE_STABILITY", "WARN_FALLBACK_ACTIVATED",
    "DEFAULT_EXPOSURE_STALE_SEC", "DEFAULT_LIFECYCLE_STALE_SEC",
    "DEFAULT_STABILITY_STALE_SEC",
]
