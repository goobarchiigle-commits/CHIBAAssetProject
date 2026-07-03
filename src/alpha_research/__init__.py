"""
src/alpha_research/ — Research operating system (Phase 5B)

Deterministic, governance-driven research runtime for continuous
alpha discovery, evaluation, promotion, containment, and retirement.

Never touches production capital allocation state.
Never introduces future leakage.
All records append-only and replay-safe.

Modules:
  exploration_registry   — append-only candidate lifecycle registry
  shadow_runtime         — virtual shadow allocations without real capital
  promotion_governance   — deterministic alpha promotion criteria
  retirement_engine      — automatic retirement on decay signals
  research_budget        — hard caps on exploration resource consumption
  research_half_life     — adaptive capacity metrics for the research system
  exploration_telemetry  — consolidated research pipeline analytics

Quick start:
  from src.alpha_research import (
      ExplorationCandidate, register_candidate, transition_candidate,
      evaluate_promotion, evaluate_retirement,
      ResearchBudget, evaluate_admission,
  )
"""

# ── Exploration registry ───────────────────────────────────────────────────────
from src.alpha_research.exploration_registry import (
    ExplorationCandidate,
    StatusChange,
    InvalidTransitionError,
    VALID_STATUSES,
    VALID_TRANSITIONS,
    can_transition,
    validate_transition,
    register_candidate,
    record_status_change,
    transition_candidate,
    load_registry,
    load_status_changes,
    save_registry_snapshot,
    get_active_candidates,
    get_by_status,
)

# ── Shadow runtime ─────────────────────────────────────────────────────────────
from src.alpha_research.shadow_runtime import (
    ShadowFill,
    ShadowPortfolioState,
    ShadowObservation,
    simulate_shadow_fill,
    apply_shadow_fill,
    build_shadow_observation,
    save_shadow_portfolio,
    load_shadow_portfolio,
    append_shadow_fill,
    append_shadow_observation,
    load_shadow_observations,
)

# ── Promotion governance ───────────────────────────────────────────────────────
from src.alpha_research.promotion_governance import (
    PromotionCriteria,
    PromotionDecision,
    ShadowPerformanceSummary,
    evaluate_promotion,
    compute_shadow_sharpe,
    compute_stability_from_observations,
    append_promotion_decision,
    load_promotion_decisions,
)

# ── Retirement engine ──────────────────────────────────────────────────────────
from src.alpha_research.retirement_engine import (
    RetirementSignal,
    RetirementConfig,
    RetirementRecord,
    evaluate_edge_decay,
    evaluate_slippage_deterioration,
    evaluate_underperformance,
    evaluate_capital_inefficiency,
    evaluate_retirement,
    append_retirement_record,
    load_retirement_records,
)

# ── Research budget ────────────────────────────────────────────────────────────
from src.alpha_research.research_budget import (
    ResearchBudget,
    ResearchBudgetState,
    BudgetDecision,
    check_shadow_capacity,
    check_capital_budget,
    check_theme_budget,
    is_congested,
    evaluate_admission,
    release_budget,
    append_budget_decision,
    load_research_budget,
    save_research_budget,
    load_budget_state,
    save_budget_state,
    load_budget_decisions,
)

# ── Research half-life ─────────────────────────────────────────────────────────
from src.alpha_research.research_half_life import (
    CandidateLifecycleRecord,
    ResearchHalfLifeMetrics,
    build_lifecycle_record,
    compute_half_life_metrics,
    append_half_life_metrics,
    load_half_life_metrics,
)

# ── Exploration telemetry ──────────────────────────────────────────────────────
from src.alpha_research.exploration_telemetry import (
    ExplorationTelemetrySnapshot,
    build_telemetry_snapshot,
    append_telemetry_snapshot,
    load_telemetry_snapshots,
    load_recent_snapshots,
)

__all__ = [
    # exploration_registry
    "ExplorationCandidate", "StatusChange", "InvalidTransitionError",
    "VALID_STATUSES", "VALID_TRANSITIONS",
    "can_transition", "validate_transition",
    "register_candidate", "record_status_change", "transition_candidate",
    "load_registry", "load_status_changes", "save_registry_snapshot",
    "get_active_candidates", "get_by_status",
    # shadow_runtime
    "ShadowFill", "ShadowPortfolioState", "ShadowObservation",
    "simulate_shadow_fill", "apply_shadow_fill", "build_shadow_observation",
    "save_shadow_portfolio", "load_shadow_portfolio",
    "append_shadow_fill", "append_shadow_observation", "load_shadow_observations",
    # promotion_governance
    "PromotionCriteria", "PromotionDecision", "ShadowPerformanceSummary",
    "evaluate_promotion", "compute_shadow_sharpe", "compute_stability_from_observations",
    "append_promotion_decision", "load_promotion_decisions",
    # retirement_engine
    "RetirementSignal", "RetirementConfig", "RetirementRecord",
    "evaluate_edge_decay", "evaluate_slippage_deterioration",
    "evaluate_underperformance", "evaluate_capital_inefficiency",
    "evaluate_retirement", "append_retirement_record", "load_retirement_records",
    # research_budget
    "ResearchBudget", "ResearchBudgetState", "BudgetDecision",
    "check_shadow_capacity", "check_capital_budget", "check_theme_budget",
    "is_congested", "evaluate_admission", "release_budget",
    "append_budget_decision", "load_research_budget", "save_research_budget",
    "load_budget_state", "save_budget_state", "load_budget_decisions",
    # research_half_life
    "CandidateLifecycleRecord", "ResearchHalfLifeMetrics",
    "build_lifecycle_record", "compute_half_life_metrics",
    "append_half_life_metrics", "load_half_life_metrics",
    # exploration_telemetry
    "ExplorationTelemetrySnapshot", "build_telemetry_snapshot",
    "append_telemetry_snapshot", "load_telemetry_snapshots", "load_recent_snapshots",
]
