"""
src/analytics/future_leader_paper_execution_engine.py
Phase 3C — Future Leader Paper Execution Engine
(Stateful Execution Survivability Replay Layer)

Purpose:
  Measure whether research alpha survives the full execution lifecycle after
  portfolio admission. Extends the expectancy chain one step further:

    research_expectancy
    ↓ (shadow deployment)
    shadow_expectancy
    ↓ (admission governance)
    admitted_expectancy
    ↓ (THIS LAYER)
    executed_expectancy / exited_position_expectancy

  NOT a paper trading simulator. NOT an execution simulator.
  Responsibility: stateful execution survivability telemetry only.

Architecture:
  PaperExecutionInput (promotion + shadow + routing reports)
  ↓ FAIL_CLOSED gate (BLOCKED/RESEARCH_ONLY → rollback all)
  ↓ synthetic position building (ALLOW/REDUCED_SIZE → position; DENY → skip)
  ↓ lifecycle transition logic (deterministic FAIL_CLOSED ordering)
  ↓ NAV drawdown computation
  ↓ execution + lifecycle decay metrics
  ↓ capital occupancy telemetry
  ↓ congestion telemetry
  ↓ portfolio state determination
  ↓ paper_execution_allowed gate
  → PaperExecutionReport (append-only JSONL)

INVARIANTS (変更禁止):
  - observation_only=True: reads routing/shadow/promotion reports only
  - FAIL_CLOSED: blockers/collapse/ambiguity → ROLLED_BACK or EXITED early
  - deterministic: identical inputs → identical hash and lifecycle replay
  - append-only JSONL: synthetic ledger and reports are immutable
  - no execution path connection: no live orders, no allocator, no broker

禁止:
  - live broker execution / broker API
  - slippage optimizer / smart order routing / latency optimizer
  - adaptive execution RL / execution optimizer
  - predictive_entry_enabled mutation / allocator mutation
  - realistic fills / synthetic fill simulation
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.analytics.future_leader_promotion_readiness import (
    PromotionReadinessReport,
    PromotionState,
    PromotionReadinessInputs,
    evaluate_promotion_readiness,
    _load_jsonl_dir,
    _load_regime_tags_from_dir,
    load_promotion_reports,
)
from src.analytics.future_leader_shadow_deployment_validator import (
    ShadowDeploymentInput,
    ShadowDeploymentReport,
    validate_shadow_deployment,
    load_shadow_deployment_reports,
    append_shadow_deployment_report,
)
from src.analytics.future_leader_limited_capital_router import (
    AdmissionDecision,
    LimitedCapitalRoutingInput,
    LimitedCapitalRoutingReport,
    RoutingSuppressionReason,
    VirtualAdmissionDecision,
    route_limited_capital,
    load_routing_reports,
    append_routing_report,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# Constants (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

# Rollback / exit triggers
_ROLLBACK_PERSISTENCE_THRESHOLD:  float = 0.40   # persistence < this → ROLLED_BACK
_ROLLBACK_DEGRADATION_THRESHOLD:  float = 0.40   # degradation < this → EXIT_BY_DRIFT_COLLAPSE
_DRAWDOWN_THRESHOLD:              float = -0.15  # CLAUDE.md max_dd_limit

# Paper execution readiness gate thresholds
_EXEC_DRIFT_THRESHOLD:       float = 0.80
_LIFECYCLE_DECAY_THRESHOLD:  float = 0.70

# Capital occupancy constants
_MAX_POSITIONS: int   = 3     # CLAUDE.md max_positions=3
_MAX_WEIGHT:    float = 0.10  # shadow deployment per-position cap
_HOLD_WINDOW:   int   = 10    # FORWARD_WINDOW (business days)

# Congestion threshold (overlapping positions ≥ this → CONGESTED)
_CONGESTION_THRESHOLD: int = _MAX_POSITIONS  # = 3

# Max lock duration for occupancy ratio (worst case: 3 positions × 10 days × 0.10 weight)
_MAX_LOCK_DURATION: float = float(_MAX_POSITIONS * _HOLD_WINDOW * _MAX_WEIGHT)  # = 3.0

# Capital reuse efficiency denominator guard
_EFFICIENCY_DENOMINATOR_GUARD: float = 1.0

# Drawdown guard for CAPITAL_SUPPRESSED trigger (slightly softer than FAIL_CLOSED threshold)
_CAPITAL_SUPPRESSED_DRAWDOWN: float = -0.10


# ─────────────────────────────────────────────────────────────────────────────
# Enums (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticPositionState(Enum):
    PENDING_ADMISSION = "pending_admission"
    ADMITTED          = "admitted"
    ACTIVE            = "active"
    REDUCED           = "reduced"
    EXIT_CANDIDATE    = "exit_candidate"
    EXITED            = "exited"
    ROLLED_BACK       = "rolled_back"


class PositionTransitionReason(Enum):
    ADMISSION_ALLOWED       = "admission_allowed"
    REDUCED_BY_CONGESTION   = "reduced_by_congestion"
    REDUCED_BY_DRAWDOWN     = "reduced_by_drawdown"
    ROLLBACK_BY_PERSISTENCE = "rollback_by_persistence"
    EXIT_BY_HOLDING_EXPIRY  = "exit_by_holding_expiry"
    EXIT_BY_DRIFT_COLLAPSE  = "exit_by_drift_collapse"
    EXIT_BY_REGIME_COLLAPSE = "exit_by_regime_collapse"


class SyntheticPortfolioState(Enum):
    STABLE             = "stable"
    DEGRADED           = "degraded"
    CONGESTED          = "congested"
    ROLLBACK_MODE      = "rollback_mode"
    CAPITAL_SUPPRESSED = "capital_suppressed"


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle PnL multipliers (変更禁止)
# Each transition reason → fraction of shadow_expectancy realized
# ─────────────────────────────────────────────────────────────────────────────

_LIFECYCLE_PNL_FACTOR: Dict[PositionTransitionReason, float] = {
    PositionTransitionReason.ADMISSION_ALLOWED:       1.00,  # full hold baseline (unrealized)
    PositionTransitionReason.EXIT_BY_HOLDING_EXPIRY:  1.00,  # full hold completed
    PositionTransitionReason.REDUCED_BY_CONGESTION:   0.75,  # reduced but survived
    PositionTransitionReason.REDUCED_BY_DRAWDOWN:     0.60,  # reduced due to drawdown pressure
    PositionTransitionReason.EXIT_BY_REGIME_COLLAPSE: 0.40,  # regime forced exit
    PositionTransitionReason.EXIT_BY_DRIFT_COLLAPSE:  0.50,  # early exit on drift collapse
    PositionTransitionReason.ROLLBACK_BY_PERSISTENCE: 0.30,  # persistence collapse — partial salvage
}

# Final states (unrealized_pnl → 0, realized_pnl set)
_FINAL_STATES = frozenset({
    SyntheticPositionState.EXITED,
    SyntheticPositionState.ROLLED_BACK,
})

# Routing eligible states (mirrors router)
_EXECUTION_ELIGIBLE_STATES = frozenset({
    PromotionState.SHADOW_READY,
    PromotionState.PAPER_READY,
    PromotionState.LIMITED_EXECUTION_READY,
    PromotionState.PRODUCTION_READY,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperExecutionInput:
    """
    Input bundle for paper execution lifecycle replay.
    All three upstream reports must be provided.
    previous_positions: {symbol: SyntheticPositionState.value} for transition tracking.
    """
    promotion_report: PromotionReadinessReport
    shadow_report:    ShadowDeploymentReport
    routing_report:   LimitedCapitalRoutingReport
    regime_tag:       str
    evaluation_date:  str
    previous_positions: Dict[str, str]  # {symbol: state_value}


@dataclass
class SyntheticPosition:
    """
    One synthetic position in the paper execution ledger.
    observation_only=True: no execution path connection.
    """
    symbol: str

    state:              SyntheticPositionState
    transition_reason:  PositionTransitionReason

    synthetic_entry_date: str
    synthetic_exit_date:  Optional[str]

    proposed_weight: float
    active_weight:   float

    unrealized_pnl:  float  # shadow_exp × active_weight (while position active)
    realized_pnl:    float  # shadow_exp × active_weight × lifecycle_factor (on exit)

    holding_duration_days: int
    rollback_triggered:    bool

    def to_dict(self) -> dict:
        return {
            "symbol":                 self.symbol,
            "state":                  self.state.value,
            "transition_reason":      self.transition_reason.value,
            "synthetic_entry_date":   self.synthetic_entry_date,
            "synthetic_exit_date":    self.synthetic_exit_date,
            "proposed_weight":        round(self.proposed_weight, 6),
            "active_weight":          round(self.active_weight, 6),
            "unrealized_pnl":         round(self.unrealized_pnl, 6),
            "realized_pnl":           round(self.realized_pnl, 6),
            "holding_duration_days":  self.holding_duration_days,
            "rollback_triggered":     self.rollback_triggered,
        }


@dataclass
class SyntheticExecutionLedger:
    """
    Append-only synthetic position ledger for one evaluation period.
    observation_only=True: no execution path connection.
    """
    evaluation_date: str
    positions:       List[SyntheticPosition]

    def to_dict(self) -> dict:
        return {
            "evaluation_date": self.evaluation_date,
            "positions":       [p.to_dict() for p in self.positions],
        }


@dataclass
class PaperExecutionReport:
    """
    Stateful execution survivability telemetry output.
    observation_only=True: no execution path connection.

    Extends the expectancy chain:
      research_expectancy → shadow → admitted → executed → exited

    JSONL schema: see to_dict().
    Future integration: when paper_execution_allowed=True,
    connect to AutoRollbackGovernor for live throttling governance.
    """
    timestamp:       str
    evaluation_date: str
    promotion_state: str
    regime_tag:      str
    portfolio_state: SyntheticPortfolioState

    ledger: SyntheticExecutionLedger

    # Expectancy chain
    research_expectancy:          float
    shadow_expectancy:            float
    admitted_expectancy:          float
    executed_expectancy:          float
    exited_position_expectancy:   float

    # Core drift metrics
    execution_drift_ratio:   Optional[float]  # executed / admitted
    lifecycle_decay_ratio:   Optional[float]  # exited / admitted
    execution_persistence_score: float        # fraction completing full hold

    # Drawdown + rollback
    synthetic_max_drawdown:  float
    rollback_trigger_rate:   float

    # Capital occupancy
    average_holding_duration:  float
    overlapping_cohort_count:  int
    portfolio_congestion_score: float
    capital_lock_duration:     float
    capital_occupancy_ratio:   float
    capital_reuse_efficiency:  float

    # Execution gate
    paper_execution_allowed: bool

    deterministic_hash: str

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "evaluation_date": self.evaluation_date,
            "promotion_state": self.promotion_state,
            "regime_tag":      self.regime_tag,
            "portfolio_state": self.portfolio_state.value,

            "ledger": self.ledger.to_dict(),

            "research_expectancy":        round(self.research_expectancy, 6),
            "shadow_expectancy":          round(self.shadow_expectancy, 6),
            "admitted_expectancy":        round(self.admitted_expectancy, 6),
            "executed_expectancy":        round(self.executed_expectancy, 6),
            "exited_position_expectancy": round(self.exited_position_expectancy, 6),

            "execution_drift_ratio": (
                round(self.execution_drift_ratio, 4)
                if self.execution_drift_ratio is not None else None
            ),
            "lifecycle_decay_ratio": (
                round(self.lifecycle_decay_ratio, 4)
                if self.lifecycle_decay_ratio is not None else None
            ),
            "execution_persistence_score": round(self.execution_persistence_score, 4),

            "synthetic_max_drawdown": round(self.synthetic_max_drawdown, 6),
            "rollback_trigger_rate":  round(self.rollback_trigger_rate, 4),

            "average_holding_duration":   round(self.average_holding_duration, 2),
            "overlapping_cohort_count":   self.overlapping_cohort_count,
            "portfolio_congestion_score": round(self.portfolio_congestion_score, 4),
            "capital_lock_duration":      round(self.capital_lock_duration, 4),
            "capital_occupancy_ratio":    round(self.capital_occupancy_ratio, 4),
            "capital_reuse_efficiency":   round(self.capital_reuse_efficiency, 4),

            "paper_execution_allowed": self.paper_execution_allowed,
            "deterministic_hash":      self.deterministic_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Suppression reason → position transition reason mapping (pure)
# ─────────────────────────────────────────────────────────────────────────────

_SUPPRESSION_TO_TRANSITION: Dict[RoutingSuppressionReason, PositionTransitionReason] = {
    RoutingSuppressionReason.NONE:                 PositionTransitionReason.ADMISSION_ALLOWED,
    RoutingSuppressionReason.RISK_OFF:             PositionTransitionReason.REDUCED_BY_CONGESTION,
    RoutingSuppressionReason.CONCENTRATION_EXCESS: PositionTransitionReason.REDUCED_BY_CONGESTION,
    RoutingSuppressionReason.CONFIDENCE_WEAKNESS:  PositionTransitionReason.REDUCED_BY_CONGESTION,
    RoutingSuppressionReason.DEPLOYMENT_DRIFT:     PositionTransitionReason.REDUCED_BY_DRAWDOWN,
    RoutingSuppressionReason.PERSISTENCE_COLLAPSE: PositionTransitionReason.REDUCED_BY_DRAWDOWN,
    RoutingSuppressionReason.BLOCKED_PROMOTION:    PositionTransitionReason.ROLLBACK_BY_PERSISTENCE,
}


def _map_suppression_to_transition(
    reason: RoutingSuppressionReason,
) -> PositionTransitionReason:
    return _SUPPRESSION_TO_TRANSITION.get(reason, PositionTransitionReason.REDUCED_BY_CONGESTION)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle exit determination (pure — FAIL_CLOSED priority ordering)
# ─────────────────────────────────────────────────────────────────────────────

def _determine_lifecycle_exit(
    initial_reason:      PositionTransitionReason,
    persistence_score:   float,
    degradation_ratio:   float,
    deployment_blockers: List[str],
    regime_tag:          str,
) -> Tuple[SyntheticPositionState, PositionTransitionReason]:
    """
    Determine final lifecycle state and transition reason for one position.

    FAIL_CLOSED priority order (checked first → highest severity):
      1. deployment_blockers present → ROLLED_BACK
      2. persistence collapse (< 0.40) → ROLLED_BACK
      3. deployment drift collapse (< 0.40) → EXITED via DRIFT_COLLAPSE
      4. risk_off regime → EXITED via REGIME_COLLAPSE
      5. normal completion → EXITED via HOLDING_EXPIRY
         (REDUCED positions retain their reduction reason on normal exit)

    Ambiguity → ROLLBACK_BY_PERSISTENCE (FAIL_CLOSED).
    """
    # 1. Deployment blockers — highest severity
    if deployment_blockers:
        return SyntheticPositionState.ROLLED_BACK, PositionTransitionReason.ROLLBACK_BY_PERSISTENCE

    # 2. Persistence collapse
    if persistence_score < _ROLLBACK_PERSISTENCE_THRESHOLD:
        return SyntheticPositionState.ROLLED_BACK, PositionTransitionReason.ROLLBACK_BY_PERSISTENCE

    # 3. Deployment drift collapse
    if degradation_ratio < _ROLLBACK_DEGRADATION_THRESHOLD:
        return SyntheticPositionState.EXITED, PositionTransitionReason.EXIT_BY_DRIFT_COLLAPSE

    # 4. Risk-off regime exit
    if regime_tag == "risk_off":
        return SyntheticPositionState.EXITED, PositionTransitionReason.EXIT_BY_REGIME_COLLAPSE

    # 5. Normal completion — REDUCED positions retain their reduction reason
    if initial_reason in (
        PositionTransitionReason.REDUCED_BY_CONGESTION,
        PositionTransitionReason.REDUCED_BY_DRAWDOWN,
    ):
        return SyntheticPositionState.EXITED, initial_reason

    return SyntheticPositionState.EXITED, PositionTransitionReason.EXIT_BY_HOLDING_EXPIRY


# ─────────────────────────────────────────────────────────────────────────────
# Position building + lifecycle evolution (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _build_synthetic_positions(
    admission_decisions: List[VirtualAdmissionDecision],
    shadow_report:       ShadowDeploymentReport,
    regime_tag:          str,
    evaluation_date:     str,
) -> List[SyntheticPosition]:
    """
    Build synthetic positions from routing decisions and evolve full lifecycle.

    DENY decisions → no position (skipped).
    ALLOW → ACTIVE → exit/rollback.
    REDUCED_SIZE → REDUCED → exit/rollback.

    Deterministic: symbols sorted; FAIL_CLOSED rules applied uniformly.
    Pure function: no I/O, no side effects.
    """
    degradation_ratio   = shadow_report.deployment_degradation_ratio or 0.0
    persistence_score   = shadow_report.deployment_persistence_score
    deployment_blockers = shadow_report.deployment_blockers
    shadow_exp          = shadow_report.shadow_expectancy
    avg_hold            = max(1, int(round(shadow_report.average_holding_duration)))

    positions: List[SyntheticPosition] = []

    for d in sorted(admission_decisions, key=lambda x: x.symbol):
        if d.decision == AdmissionDecision.DENY:
            continue

        # Initial admission reason
        if d.decision == AdmissionDecision.ALLOW:
            initial_reason = PositionTransitionReason.ADMISSION_ALLOWED
        else:  # REDUCED_SIZE
            initial_reason = _map_suppression_to_transition(d.suppression_reason)

        # Determine final lifecycle state and exit reason
        final_state, final_reason = _determine_lifecycle_exit(
            initial_reason=initial_reason,
            persistence_score=persistence_score,
            degradation_ratio=degradation_ratio,
            deployment_blockers=deployment_blockers,
            regime_tag=regime_tag,
        )

        # PnL computation
        lifecycle_factor = _LIFECYCLE_PNL_FACTOR.get(final_reason, 0.0)
        realized_pnl     = round(shadow_exp * d.adjusted_weight * lifecycle_factor, 6)
        unrealized_pnl   = (
            round(shadow_exp * d.adjusted_weight, 6)
            if final_state not in _FINAL_STATES
            else 0.0
        )
        rollback_triggered = final_state == SyntheticPositionState.ROLLED_BACK

        positions.append(SyntheticPosition(
            symbol=d.symbol,
            state=final_state,
            transition_reason=final_reason,
            synthetic_entry_date=evaluation_date,
            synthetic_exit_date=evaluation_date if final_state in _FINAL_STATES else None,
            proposed_weight=d.proposed_weight,
            active_weight=d.adjusted_weight,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            holding_duration_days=avg_hold,
            rollback_triggered=rollback_triggered,
        ))

    return positions


# ─────────────────────────────────────────────────────────────────────────────
# NAV drawdown (pure — mirrors shadow deployment validator pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_synthetic_nav_drawdown(positions: List[SyntheticPosition]) -> float:
    """
    Max drawdown from sequential position-weighted NAV curve.
    Each position contributes: NAV *= (1 + position_return × active_weight).
    Returns max drawdown as a negative float (0.0 if no drawdown).
    Sorted by symbol for deterministic ordering.
    """
    if not positions:
        return 0.0
    nav          = 1.0
    peak         = 1.0
    max_drawdown = 0.0
    for p in sorted(positions, key=lambda x: x.symbol):
        if p.active_weight > 1e-9:
            position_return = p.realized_pnl / p.active_weight
        else:
            position_return = 0.0
        nav  *= (1.0 + position_return * p.active_weight)
        peak  = max(peak, nav)
        dd    = (nav - peak) / max(peak, 1e-9)
        max_drawdown = min(max_drawdown, dd)
    return round(max_drawdown, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Execution metrics computation (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_execution_metrics(
    positions:          List[SyntheticPosition],
    routing_report:     LimitedCapitalRoutingReport,
    shadow_report:      ShadowDeploymentReport,
) -> dict:
    """
    Compute all execution survivability metrics from synthetic positions.
    Pure function: no I/O, no side effects.
    """
    # Denominator: total proposed weight across ALL admission decisions (including DENY)
    # — consistent with router's admitted_expectancy denominator
    all_decisions       = routing_report.admission_decisions
    total_proposed_w    = sum(d.proposed_weight for d in all_decisions)
    total_admitted_w    = sum(
        d.adjusted_weight for d in all_decisions
        if d.decision != AdmissionDecision.DENY
    )
    admitted_expectancy = routing_report.admitted_expectancy
    n_admitted          = len(positions)

    # ── Expectancy metrics ─────────────────────────────────────────────────────

    # executed_expectancy: all positions that went through lifecycle (including rollbacks)
    executed_expectancy = (
        sum(p.realized_pnl for p in positions) / max(total_proposed_w, 1e-9)
    )

    # exited_position_expectancy: only positions that completed full hold
    full_hold_positions = [
        p for p in positions
        if p.transition_reason == PositionTransitionReason.EXIT_BY_HOLDING_EXPIRY
    ]
    exited_position_expectancy = (
        sum(p.realized_pnl for p in full_hold_positions) / max(total_proposed_w, 1e-9)
    )

    # Drift and decay ratios
    if admitted_expectancy is not None and abs(admitted_expectancy) > 1e-9:
        execution_drift_ratio: Optional[float] = round(
            executed_expectancy / admitted_expectancy, 4
        )
        lifecycle_decay_ratio: Optional[float] = round(
            exited_position_expectancy / admitted_expectancy, 4
        )
    else:
        execution_drift_ratio = None
        lifecycle_decay_ratio = None

    # execution_persistence_score: fraction of admitted positions that completed full hold
    execution_persistence_score = (
        len(full_hold_positions) / max(n_admitted, 1)
    )

    # ── Rollback metrics ───────────────────────────────────────────────────────
    n_rolled_back    = sum(1 for p in positions if p.rollback_triggered)
    rollback_trigger_rate = n_rolled_back / max(n_admitted, 1)

    # ── Capital occupancy ─────────────────────────────────────────────────────
    avg_holding = (
        float(np.mean([p.holding_duration_days for p in positions]))
        if positions else float(shadow_report.average_holding_duration)
    )
    # capital_lock_duration: sum(weight × hold_days) — proxy for capital tied up
    capital_lock_duration = sum(
        p.active_weight * p.holding_duration_days for p in positions
    )
    # Overlapping cohort count: all admitted positions are simultaneously active
    overlapping_cohort_count = n_admitted
    # Portfolio congestion score: fraction of max capacity used
    portfolio_congestion_score = overlapping_cohort_count / max(_MAX_POSITIONS, 1)

    # capital_occupancy_ratio: lock duration vs theoretical maximum
    capital_occupancy_ratio = (
        capital_lock_duration / _MAX_LOCK_DURATION
        if _MAX_LOCK_DURATION > 1e-9 else 0.0
    )

    # capital_reuse_efficiency: complete exits / lock duration
    n_complete_exits = len(full_hold_positions)
    capital_reuse_efficiency = (
        n_complete_exits / max(capital_lock_duration, _EFFICIENCY_DENOMINATOR_GUARD)
    )

    return {
        "executed_expectancy":          round(executed_expectancy, 6),
        "exited_position_expectancy":   round(exited_position_expectancy, 6),
        "execution_drift_ratio":        execution_drift_ratio,
        "lifecycle_decay_ratio":        lifecycle_decay_ratio,
        "execution_persistence_score":  round(execution_persistence_score, 4),
        "rollback_trigger_rate":        round(rollback_trigger_rate, 4),
        "average_holding_duration":     round(avg_holding, 2),
        "overlapping_cohort_count":     overlapping_cohort_count,
        "portfolio_congestion_score":   round(portfolio_congestion_score, 4),
        "capital_lock_duration":        round(capital_lock_duration, 4),
        "capital_occupancy_ratio":      round(min(1.0, capital_occupancy_ratio), 4),
        "capital_reuse_efficiency":     round(capital_reuse_efficiency, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio state determination (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _determine_portfolio_state(
    execution_drift_ratio:   Optional[float],
    lifecycle_decay_ratio:   Optional[float],
    rollback_trigger_rate:   float,
    persistence_score:       float,
    synthetic_max_drawdown:  float,
    overlapping_cohort_count: int,
    deployment_blockers:     List[str],
) -> SyntheticPortfolioState:
    """
    Determine portfolio-level synthetic state.
    Priority ordering: ROLLBACK_MODE > CAPITAL_SUPPRESSED > CONGESTED > DEGRADED > STABLE.

    FAIL_CLOSED: blockers or persistence collapse → ROLLBACK_MODE.
    """
    # Highest severity: rollback conditions
    if (rollback_trigger_rate > 0.0
            or persistence_score < _ROLLBACK_PERSISTENCE_THRESHOLD
            or deployment_blockers):
        return SyntheticPortfolioState.ROLLBACK_MODE

    # Drawdown exceeded capital suppression threshold
    if synthetic_max_drawdown < _CAPITAL_SUPPRESSED_DRAWDOWN:
        return SyntheticPortfolioState.CAPITAL_SUPPRESSED

    # Congestion: too many overlapping positions
    if overlapping_cohort_count >= _CONGESTION_THRESHOLD:
        return SyntheticPortfolioState.CONGESTED

    # Degraded: execution drift or lifecycle decay below threshold
    drift_degraded  = execution_drift_ratio is not None and execution_drift_ratio  < _EXEC_DRIFT_THRESHOLD
    decay_degraded  = lifecycle_decay_ratio is not None and lifecycle_decay_ratio   < _LIFECYCLE_DECAY_THRESHOLD
    if drift_degraded or decay_degraded:
        return SyntheticPortfolioState.DEGRADED

    return SyntheticPortfolioState.STABLE


# ─────────────────────────────────────────────────────────────────────────────
# Paper execution readiness gate (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_paper_execution_allowed(
    promotion_report:        PromotionReadinessReport,
    shadow_report:           ShadowDeploymentReport,
    execution_drift_ratio:   Optional[float],
    lifecycle_decay_ratio:   Optional[float],
    synthetic_max_drawdown:  float,
    portfolio_state:         SyntheticPortfolioState,
) -> bool:
    """
    paper_execution_allowed: True only when ALL conditions satisfied.
    FAIL_CLOSED: any None ratio is treated as failing its threshold.

    Future integration point: when True + SHADOW_READY → connect AutoRollbackGovernor.
    """
    if not promotion_report.promotion_ready:
        return False
    if not shadow_report.is_deployment_viable:
        return False
    if execution_drift_ratio is None or execution_drift_ratio < _EXEC_DRIFT_THRESHOLD:
        return False
    if lifecycle_decay_ratio is None or lifecycle_decay_ratio < _LIFECYCLE_DECAY_THRESHOLD:
        return False
    if synthetic_max_drawdown < _DRAWDOWN_THRESHOLD:
        return False
    if portfolio_state in (
        SyntheticPortfolioState.ROLLBACK_MODE,
        SyntheticPortfolioState.CAPITAL_SUPPRESSED,
    ):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic hash (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deterministic_hash(inputs: PaperExecutionInput) -> str:
    """
    SHA256 of canonical input.
    Anchored to promotion + shadow + routing hashes for full chain auditability.
    """
    canonical = {
        "promotion_hash":    inputs.promotion_report.deterministic_hash,
        "shadow_hash":       inputs.shadow_report.deterministic_hash,
        "routing_hash":      inputs.routing_report.deterministic_hash,
        "regime_tag":        inputs.regime_tag,
        "previous_positions": dict(sorted(inputs.previous_positions.items())),
        "evaluation_date":   inputs.evaluation_date,
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# FAIL_CLOSED rollback-all helper (pure — ineligible state fast-path)
# ─────────────────────────────────────────────────────────────────────────────

def _rollback_all(
    inputs:    PaperExecutionInput,
    det_hash:  str,
    timestamp: str,
) -> PaperExecutionReport:
    """
    Fast-path for ineligible promotion states: all positions ROLLED_BACK.
    """
    shadow  = inputs.shadow_report
    promo   = inputs.promotion_report
    routing = inputs.routing_report
    avg_hold = max(1, int(round(shadow.average_holding_duration)))

    positions: List[SyntheticPosition] = [
        SyntheticPosition(
            symbol=d.symbol,
            state=SyntheticPositionState.ROLLED_BACK,
            transition_reason=PositionTransitionReason.ROLLBACK_BY_PERSISTENCE,
            synthetic_entry_date=inputs.evaluation_date,
            synthetic_exit_date=inputs.evaluation_date,
            proposed_weight=d.proposed_weight,
            active_weight=d.adjusted_weight,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            holding_duration_days=avg_hold,
            rollback_triggered=True,
        )
        for d in sorted(routing.admission_decisions, key=lambda x: x.symbol)
        if d.decision != AdmissionDecision.DENY
    ]

    ledger = SyntheticExecutionLedger(
        evaluation_date=inputs.evaluation_date,
        positions=positions,
    )
    n = len(positions)
    return PaperExecutionReport(
        timestamp=timestamp,
        evaluation_date=inputs.evaluation_date,
        promotion_state=promo.promotion_state.value,
        regime_tag=inputs.regime_tag,
        portfolio_state=SyntheticPortfolioState.ROLLBACK_MODE,
        ledger=ledger,
        research_expectancy=shadow.research_expectancy,
        shadow_expectancy=shadow.shadow_expectancy,
        admitted_expectancy=routing.admitted_expectancy,
        executed_expectancy=0.0,
        exited_position_expectancy=0.0,
        execution_drift_ratio=None,
        lifecycle_decay_ratio=None,
        execution_persistence_score=0.0,
        synthetic_max_drawdown=0.0,
        rollback_trigger_rate=1.0 if n else 0.0,
        average_holding_duration=float(avg_hold),
        overlapping_cohort_count=n,
        portfolio_congestion_score=n / max(_MAX_POSITIONS, 1),
        capital_lock_duration=0.0,
        capital_occupancy_ratio=0.0,
        capital_reuse_efficiency=0.0,
        paper_execution_allowed=False,
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main execution entry point (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def run_paper_execution(
    inputs: PaperExecutionInput,
) -> PaperExecutionReport:
    """
    Run stateful execution survivability replay.

    FAIL_CLOSED:
      - BLOCKED or RESEARCH_ONLY → all positions ROLLED_BACK
      - deployment_blockers → all positions ROLLED_BACK
      - persistence_score < 0.40 → all positions ROLLED_BACK
      - ambiguity → ROLLBACK_BY_PERSISTENCE

    Deterministic: same inputs → same lifecycle replay + same hash.
    observation_only=True: no execution mutation.

    Future integration points:
      paper_execution_allowed=True + SHADOW_READY →
        1. Connect AutoRollbackGovernor.evaluate(report)
        2. Connect MicroCapitalLiveDeployment.route() guard
      PAPER_READY → full paper allocation hook
    """
    ts       = datetime.now(JST).isoformat()
    det_hash = _compute_deterministic_hash(inputs)
    promo    = inputs.promotion_report
    shadow   = inputs.shadow_report
    routing  = inputs.routing_report

    # ── FAIL_CLOSED lifecycle gate ───────────────────────────────────────────
    if promo.promotion_state not in _EXECUTION_ELIGIBLE_STATES:
        return _rollback_all(inputs, det_hash, ts)

    # ── Build + evolve synthetic positions ──────────────────────────────────
    positions = _build_synthetic_positions(
        admission_decisions=routing.admission_decisions,
        shadow_report=shadow,
        regime_tag=inputs.regime_tag,
        evaluation_date=inputs.evaluation_date,
    )

    ledger = SyntheticExecutionLedger(
        evaluation_date=inputs.evaluation_date,
        positions=positions,
    )

    # ── NAV drawdown ─────────────────────────────────────────────────────────
    synthetic_max_drawdown = _compute_synthetic_nav_drawdown(positions)

    # ── Execution metrics ────────────────────────────────────────────────────
    metrics = _compute_execution_metrics(positions, routing, shadow)

    # ── Portfolio state ───────────────────────────────────────────────────────
    portfolio_state = _determine_portfolio_state(
        execution_drift_ratio=metrics["execution_drift_ratio"],
        lifecycle_decay_ratio=metrics["lifecycle_decay_ratio"],
        rollback_trigger_rate=metrics["rollback_trigger_rate"],
        persistence_score=shadow.deployment_persistence_score,
        synthetic_max_drawdown=synthetic_max_drawdown,
        overlapping_cohort_count=metrics["overlapping_cohort_count"],
        deployment_blockers=shadow.deployment_blockers,
    )

    # ── Paper execution readiness gate ───────────────────────────────────────
    paper_allowed = _compute_paper_execution_allowed(
        promotion_report=promo,
        shadow_report=shadow,
        execution_drift_ratio=metrics["execution_drift_ratio"],
        lifecycle_decay_ratio=metrics["lifecycle_decay_ratio"],
        synthetic_max_drawdown=synthetic_max_drawdown,
        portfolio_state=portfolio_state,
    )

    if positions:
        logger.info(
            "[PAPER_EXEC] positions=%d rollback=%.2f drift=%s decay=%s"
            " state=%s allowed=%s",
            len(positions),
            metrics["rollback_trigger_rate"],
            metrics["execution_drift_ratio"],
            metrics["lifecycle_decay_ratio"],
            portfolio_state.value,
            paper_allowed,
        )

    return PaperExecutionReport(
        timestamp=ts,
        evaluation_date=inputs.evaluation_date,
        promotion_state=promo.promotion_state.value,
        regime_tag=inputs.regime_tag,
        portfolio_state=portfolio_state,
        ledger=ledger,
        research_expectancy=shadow.research_expectancy,
        shadow_expectancy=shadow.shadow_expectancy,
        admitted_expectancy=routing.admitted_expectancy,
        executed_expectancy=metrics["executed_expectancy"],
        exited_position_expectancy=metrics["exited_position_expectancy"],
        execution_drift_ratio=metrics["execution_drift_ratio"],
        lifecycle_decay_ratio=metrics["lifecycle_decay_ratio"],
        execution_persistence_score=metrics["execution_persistence_score"],
        synthetic_max_drawdown=synthetic_max_drawdown,
        rollback_trigger_rate=metrics["rollback_trigger_rate"],
        average_holding_duration=metrics["average_holding_duration"],
        overlapping_cohort_count=metrics["overlapping_cohort_count"],
        portfolio_congestion_score=metrics["portfolio_congestion_score"],
        capital_lock_duration=metrics["capital_lock_duration"],
        capital_occupancy_ratio=metrics["capital_occupancy_ratio"],
        capital_reuse_efficiency=metrics["capital_reuse_efficiency"],
        paper_execution_allowed=paper_allowed,
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence (append-only)
# ─────────────────────────────────────────────────────────────────────────────

def append_paper_execution_report(
    report:  PaperExecutionReport,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append PaperExecutionReport to today's JSONL.
    append-only: never overwrites existing records.
    Fail-open: I/O errors are logged but not re-raised.

    Path: {log_dir}/paper_execution_reports_YYYYMMDD.jsonl
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"paper_execution_reports_{today.strftime('%Y%m%d')}.jsonl"
        line     = json.dumps(report.to_dict(), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[PAPER_EXEC] report appended → %s state=%s allowed=%s"
            " drift=%s decay=%s hash=%.8s…",
            log_path.name,
            report.portfolio_state.value,
            report.paper_execution_allowed,
            report.execution_drift_ratio,
            report.lifecycle_decay_ratio,
            report.deterministic_hash,
        )
    except Exception as e:
        logger.warning("[PAPER_EXEC] report append failed (fail-open): %s", e)


def append_execution_ledger(
    ledger:  SyntheticExecutionLedger,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append SyntheticExecutionLedger to today's ledger JSONL.
    append-only: never overwrites existing records.
    Fail-open: I/O errors are logged but not re-raised.

    Path: {log_dir}/synthetic_execution_ledger_YYYYMMDD.jsonl
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"synthetic_execution_ledger_{today.strftime('%Y%m%d')}.jsonl"
        line     = json.dumps(ledger.to_dict(), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[PAPER_EXEC] ledger appended → %s n_positions=%d",
            log_path.name, len(ledger.positions),
        )
    except Exception as e:
        logger.warning("[PAPER_EXEC] ledger append failed (fail-open): %s", e)


def load_paper_execution_reports(log_dir: Path) -> List[dict]:
    """Load all paper execution report JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("paper_execution_reports_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


def load_execution_ledgers(log_dir: Path) -> List[dict]:
    """Load all synthetic execution ledger JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("synthetic_execution_ledger_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def format_paper_execution_section(report: PaperExecutionReport) -> str:
    """Render PAPER EXECUTION section for daily report. Pure function."""
    lines = [
        "=== PAPER EXECUTION ENGINE ===",
        f"state:    {report.promotion_state}",
        f"regime:   {report.regime_tag}",
        f"portfolio: {report.portfolio_state.value}",
        f"allowed:  {report.paper_execution_allowed}",
        "",
        "expectancy chain:",
        f"  research:   {report.research_expectancy*100:+.2f}%",
        f"  shadow:     {report.shadow_expectancy*100:+.2f}%",
        f"  admitted:   {report.admitted_expectancy*100:+.2f}%",
        f"  executed:   {report.executed_expectancy*100:+.2f}%",
        f"  exited:     {report.exited_position_expectancy*100:+.2f}%",
        "",
        "execution metrics:",
        f"  drift_ratio:       {report.execution_drift_ratio}",
        f"  lifecycle_decay:   {report.lifecycle_decay_ratio}",
        f"  persistence_score: {report.execution_persistence_score:.3f}",
        f"  rollback_rate:     {report.rollback_trigger_rate:.3f}",
        f"  max_drawdown:      {report.synthetic_max_drawdown*100:+.2f}%",
        "",
        "capital occupancy:",
        f"  avg_hold: {report.average_holding_duration:.1f}d",
        f"  cohorts:  {report.overlapping_cohort_count}",
        f"  congestion: {report.portfolio_congestion_score:.3f}",
        f"  lock_dur:   {report.capital_lock_duration:.4f}",
        f"  occupancy:  {report.capital_occupancy_ratio:.3f}",
        f"  reuse_eff:  {report.capital_reuse_efficiency:.3f}",
    ]

    if report.ledger.positions:
        lines += ["", "positions:"]
        for p in report.ledger.positions:
            marker = "✗" if p.rollback_triggered else "✓"
            lines.append(
                f"  {marker} {p.symbol:10s}  {p.state.value:12s}"
                f"  w={p.active_weight:.4f}  pnl={p.realized_pnl*100:+.3f}%"
                f"  [{p.transition_reason.value}]"
            )

    lines += ["", f"hash: {report.deterministic_hash[:16]}…"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_paper_execution_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    promotion_log_dir:     Path,
    shadow_log_dir:        Path,
    router_log_dir:        Path,
    paper_exec_log_dir:    Path,
    reports_dir:           Path,
    candidate_symbols:     List[str],
    symbol_sectors:        Dict[str, str],
    symbol_themes:         Dict[str, str],
    regime_tag:            str,
    today:                 Optional[date] = None,
    persist:               bool = True,
) -> Optional[PaperExecutionReport]:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Pipeline:
      1. Load/recompute promotion readiness report.
      2. Load/recompute shadow deployment report.
      3. Load/recompute routing report (uses latest previous decisions for transitions).
      4. Load previous position states for lifecycle transition telemetry.
      5. Run run_paper_execution().
      6. Append to paper_execution_reports_YYYYMMDD.jsonl.
      7. Append synthetic_execution_ledger_YYYYMMDD.jsonl.
      8. Append PAPER EXECUTION section to daily future_leader_report.

    observation_only=True: no execution mutation.
    FAIL_CLOSED enforced inside run_paper_execution.

    Positioned after run_future_leader_limited_capital_router_hook in run_live_signal.py.

    Future integration points (activate ONLY when each layer is ready):
      paper_execution_allowed=True + SHADOW_READY →
        1. Call AutoRollbackGovernor.evaluate(report)
        2. Gate MicroCapitalLiveDeployment.route()
      PAPER_READY →
        1. Full paper allocation hook
        2. Position sizing with FreezeBreaker guard
    """
    if today is None:
        today = datetime.now(JST).date()

    try:
        surv_recs = _load_jsonl_dir(survivability_log_dir, "future_leader_survivability_*.jsonl")
        fail_recs = _load_jsonl_dir(failure_log_dir, "future_leader_failure_*.jsonl")
        reg_tags  = _load_regime_tags_from_dir(promotion_log_dir.parent / "regime")

        if not surv_recs:
            logger.info("[PAPER_EXEC] no survivability records — hook skipped")
            return None

        # Re-evaluate upstream reports from current data
        promo_inputs = PromotionReadinessInputs(
            survivability_records=surv_recs,
            failure_records=fail_recs,
            regime_tags=reg_tags,
            evaluation_date=today.isoformat(),
        )
        promo_report = evaluate_promotion_readiness(promo_inputs)

        shadow_inputs = ShadowDeploymentInput(
            promotion_report=promo_report,
            survivability_records=surv_recs,
            failure_records=fail_recs,
            evaluation_date=today.isoformat(),
        )
        shadow_report = validate_shadow_deployment(shadow_inputs)

        # Load previous routing decisions for transition telemetry
        prev_routing = load_routing_reports(router_log_dir)
        prev_routing_decisions: Dict[str, str] = {}
        if prev_routing:
            for d in prev_routing[-1].get("admission_decisions", []):
                sym = d.get("symbol", "")
                dec = d.get("decision", "")
                if sym and dec:
                    prev_routing_decisions[sym] = dec

        routing_inputs = LimitedCapitalRoutingInput(
            promotion_report=promo_report,
            shadow_report=shadow_report,
            candidate_symbols=candidate_symbols,
            symbol_sectors=symbol_sectors,
            symbol_themes=symbol_themes,
            regime_tag=regime_tag,
            previous_decisions=prev_routing_decisions,
            evaluation_date=today.isoformat(),
        )
        routing_report = route_limited_capital(routing_inputs)

        # Load previous position states for lifecycle transition telemetry
        prev_exec = load_paper_execution_reports(paper_exec_log_dir)
        prev_positions: Dict[str, str] = {}
        if prev_exec:
            for pos in prev_exec[-1].get("ledger", {}).get("positions", []):
                sym   = pos.get("symbol", "")
                state = pos.get("state", "")
                if sym and state:
                    prev_positions[sym] = state

        exec_inputs = PaperExecutionInput(
            promotion_report=promo_report,
            shadow_report=shadow_report,
            routing_report=routing_report,
            regime_tag=regime_tag,
            evaluation_date=today.isoformat(),
            previous_positions=prev_positions,
        )
        exec_report = run_paper_execution(exec_inputs)

        if persist:
            append_paper_execution_report(exec_report, paper_exec_log_dir, today=today)
            append_execution_ledger(exec_report.ledger, paper_exec_log_dir, today=today)

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"future_leader_report_{today.strftime('%Y%m%d')}.md"
        section     = format_paper_execution_section(exec_report)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info(
            "[PAPER_EXEC] hook done: portfolio=%s allowed=%s"
            " drift=%s decay=%s rollback=%.2f",
            exec_report.portfolio_state.value,
            exec_report.paper_execution_allowed,
            exec_report.execution_drift_ratio,
            exec_report.lifecycle_decay_ratio,
            exec_report.rollback_trigger_rate,
        )
        return exec_report

    except Exception as e:
        logger.warning("[PAPER_EXEC] hook failed (fail-open): %s", e)
        return None
