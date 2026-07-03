"""
src/deployment/future_leader_micro_capital_live_deployment.py
Phase 4B — Future Leader Micro Capital Live Deployment
(Self-Correcting Live Deployment Validation Layer)

Purpose:
  Connect synthetic deployment governance to minimal live capital and
  measure synthetic vs live survivability drift.

  NOT profit maximization. NOT execution automation.
  Responsibility: live deployment survivability verification only.

  Core question this layer answers:
    "Does synthetic alpha survive live friction?"

Architecture:
  Upstream reports (Promotion + Shadow + Router + Paper + Rollback)
  ↓ Live Deployment Eligibility Gate (FAIL_CLOSED — all 8 conditions required)
  ↓ Live metrics computation (live_execution_drift_ratio / live_survivability_ratio)
  ↓ Deployment state machine (FAIL_CLOSED priority ordering)
  ↓ Execution decision (governed — single symbol, 2% cap)
  ↓ Position reconciliation (mismatch → LIVE_ROLLBACK)
  ↓ Recovery observation gate (5-day mandatory post-rollback)
  → MicroCapitalDeploymentReport (append-only JSONL)
  → LiveExecutionLedger (append-only JSONL)

HARD CONSTRAINTS (変更禁止):
  MAX_LIVE_POSITIONS = 1          — single-symbol routing only
  MAX_POSITION_WEIGHT = 0.02      — 2% of total capital
  MAX_TOTAL_EXPOSURE = 0.02       — same as position cap (1 position)
  MAX_DAILY_ORDERS = 2            — EOD execution limit
  LIVE_RECOVERY_OBSERVATION_DAYS = 5  — post-rollback mandatory rest

INVARIANTS (変更禁止):
  - FAIL_CLOSED: ambiguity/missing data → most restrictive outcome
  - deterministic: same inputs → same outputs and hash
  - append-only JSONL: immutable telemetry records
  - no averaging down, no leverage, no intraday, no dynamic scaling
  - no broker mutation outside governed entry/exit/rollback actions

禁止:
  - RL sizing / adaptive ML execution / HFT / smart routing optimizer
  - multi-position routing
  - predictive_entry_enabled mutation / allocator mutation
  - autonomous universe mutation
  - API key hardcoding (use os.environ only)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.analytics.future_leader_promotion_readiness import (
    PromotionReadinessReport,
    PromotionState,
    PromotionReadinessInputs,
    evaluate_promotion_readiness,
    _load_jsonl_dir,
    _load_regime_tags_from_dir,
)
from src.analytics.future_leader_shadow_deployment_validator import (
    ShadowDeploymentInput,
    ShadowDeploymentReport,
    validate_shadow_deployment,
)
from src.analytics.future_leader_limited_capital_router import (
    AdmissionDecision,
    LimitedCapitalRoutingInput,
    LimitedCapitalRoutingReport,
    VirtualAdmissionDecision,
    route_limited_capital,
    load_routing_reports,
)
from src.analytics.future_leader_paper_execution_engine import (
    PaperExecutionInput,
    PaperExecutionReport,
    SyntheticExecutionLedger,
    run_paper_execution,
    load_paper_execution_reports,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# Hard constraints (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LIVE_POSITIONS: int   = 1
MAX_POSITION_WEIGHT: float = 0.02
MAX_TOTAL_EXPOSURE:  float = 0.02
MAX_DAILY_ORDERS:    int   = 2
LIVE_RECOVERY_OBSERVATION_DAYS: int = 5

# Eligibility gate thresholds
_ELIGIBILITY_MIN_DRIFT:     float = 0.80
_ELIGIBILITY_MIN_DECAY:     float = 0.70
_ELIGIBILITY_MAX_DRAWDOWN:  float = -0.10   # synthetic_max_drawdown must be strictly > this
_ELIGIBILITY_MAX_OCCUPANCY: float = 0.70

# FAIL_CLOSED rollback triggers
_LIVE_DRIFT_ROLLBACK_THRESHOLD:      float = 0.60
_OVERNIGHT_GAP_ROLLBACK_THRESHOLD:   float = -0.08
_ROLLBACK_PRESSURE_REDUCE_THRESHOLD: float = 0.80

# Recovery readiness thresholds (post-rollback — stricter than eligibility)
_RECOVERY_MIN_DRIFT:       float = 0.80
_RECOVERY_MIN_PERSISTENCE: float = 0.50   # higher bar than general 0.40
_RECOVERY_MAX_DRAWDOWN:    float = -0.05  # tighter than eligibility -0.10

# Reconciliation tolerance
_RECONCILIATION_TOLERANCE: float = 0.005  # 0.5% weight difference


# ─────────────────────────────────────────────────────────────────────────────
# Enums (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

class LiveDeploymentState(Enum):
    IDLE                 = "idle"
    PENDING_ENTRY        = "pending_entry"
    LIVE_ACTIVE          = "live_active"
    LIVE_REDUCED         = "live_reduced"
    LIVE_EXIT_PENDING    = "live_exit_pending"
    LIVE_ROLLBACK        = "live_rollback"
    LIVE_FROZEN          = "live_frozen"
    RECOVERY_OBSERVATION = "recovery_observation"


class LiveDeploymentFailureReason(Enum):
    ORDER_REJECTED        = "order_rejected"
    API_TIMEOUT           = "api_timeout"
    POSITION_MISMATCH     = "position_mismatch"
    LATENCY_EXCESS        = "latency_excess"
    PRICE_GAP_EXCESS      = "price_gap_excess"
    ROLLBACK_TRIGGERED    = "rollback_triggered"
    GOVERNANCE_BLOCKED    = "governance_blocked"
    API_INTEGRITY_UNKNOWN = "api_integrity_unknown"


# Promotion states eligible for live deployment (stricter than paper execution)
_LIVE_ELIGIBLE_PROMOTION_STATES = frozenset({
    PromotionState.PAPER_READY,
    PromotionState.LIMITED_EXECUTION_READY,
    PromotionState.PRODUCTION_READY,
})

# Rollback governor states that permit live entry
_LIVE_ELIGIBLE_ROLLBACK_STATES = frozenset({"stable", "watchlist"})

# States from which recovery observation logic applies
_POST_ROLLBACK_STATES = frozenset({
    LiveDeploymentState.LIVE_ROLLBACK,
    LiveDeploymentState.RECOVERY_OBSERVATION,
})


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4A stub — AutoRollbackGovernanceReport
# Contract: defines what Phase 4B expects from the Auto Rollback Governor.
# Replace with import from src.analytics.future_leader_auto_rollback_governor
# when Phase 4A is implemented.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoRollbackGovernanceReport:
    """
    Phase 4A Auto Rollback Governor interface stub.

    rollback_state values:
      stable        — healthy, full deployment permitted
      watchlist     — elevated risk, deployment permitted with monitoring
      soft_rollback — partial rollback, new entries suspended
      full_rollback — immediate exit of all live positions
      suspended     — governance frozen

    rollback_pressure_score: [0.0, 1.0]
      > 0.80 → reduce live exposure
      == 1.0 → full rollback territory

    recommended_exposure_multiplier: [0.0, 1.0]
      Applied to MAX_POSITION_WEIGHT when reducing.
    """
    timestamp:                      str
    evaluation_date:                str
    rollback_state:                 str
    rollback_pressure_score:        float
    recommended_exposure_multiplier: float
    deployment_blockers:            List[str]
    deterministic_hash:             str


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LivePositionSnapshot:
    """
    Current live position state as reported by broker + internal ledger.
    Used as input to the next evaluation cycle for lifecycle tracking.
    """
    symbol:           str
    entry_date:       str
    exit_date:        Optional[str]
    target_weight:    float
    actual_weight:    float
    entry_price:      float
    current_price:    Optional[float]
    realized_pnl:     float
    unrealized_pnl:   float
    deployment_state: LiveDeploymentState
    rollback_triggered: bool
    broker_order_id:  Optional[str]

    def to_dict(self) -> dict:
        return {
            "symbol":             self.symbol,
            "entry_date":         self.entry_date,
            "exit_date":          self.exit_date,
            "target_weight":      round(self.target_weight, 6),
            "actual_weight":      round(self.actual_weight, 6),
            "entry_price":        self.entry_price,
            "current_price":      self.current_price,
            "realized_pnl":       round(self.realized_pnl, 6),
            "unrealized_pnl":     round(self.unrealized_pnl, 6),
            "deployment_state":   self.deployment_state.value,
            "rollback_triggered": self.rollback_triggered,
            "broker_order_id":    self.broker_order_id,
        }


@dataclass
class LiveExecutionDecision:
    """
    Governed execution action for this evaluation cycle.
    order_eligible=True only when action=="enter" and all gates pass.
    """
    symbol:               Optional[str]
    action:               str   # idle/enter/hold/reduce/exit/rollback/freeze
    target_weight:        float
    failure_reason:       Optional[LiveDeploymentFailureReason]
    governance_rationale: str
    order_eligible:       bool

    def to_dict(self) -> dict:
        return {
            "symbol":               self.symbol,
            "action":               self.action,
            "target_weight":        round(self.target_weight, 6),
            "failure_reason":       self.failure_reason.value if self.failure_reason else None,
            "governance_rationale": self.governance_rationale,
            "order_eligible":       self.order_eligible,
        }


@dataclass
class LiveExecutionLedger:
    """Append-only live execution ledger for one evaluation period."""
    evaluation_date: str
    decisions:       List[LiveExecutionDecision]
    positions:       List[LivePositionSnapshot]

    def to_dict(self) -> dict:
        return {
            "evaluation_date": self.evaluation_date,
            "decisions":       [d.to_dict() for d in self.decisions],
            "positions":       [p.to_dict() for p in self.positions],
        }


@dataclass
class MicroCapitalDeploymentInput:
    """
    Full input bundle for micro capital deployment evaluation.

    Upstream governance reports + broker telemetry + live state.
    Security: API credentials must be loaded from os.environ; never passed here.
    """
    promotion_report:  PromotionReadinessReport
    shadow_report:     ShadowDeploymentReport
    routing_report:    LimitedCapitalRoutingReport
    paper_report:      PaperExecutionReport
    rollback_report:   AutoRollbackGovernanceReport

    evaluation_date: str
    regime_tag:      str

    # Current live state (broker-sourced; None = no open position)
    current_live_position:    Optional[LivePositionSnapshot]
    current_deployment_state: LiveDeploymentState

    # Reconciliation inputs
    broker_positions:        Dict[str, float]  # {symbol: actual_weight} from broker API
    reconciliation_mismatch: bool              # True if broker ≠ internal ledger

    # Broker telemetry
    broker_failure_detected:      bool
    api_integrity_unknown:        bool
    broker_timeout:               bool
    deterministic_replay_mismatch: bool
    order_latency_ms:             Optional[float]
    overnight_gap_impact:         float   # overnight price change as fraction

    # Order governance
    daily_orders_placed: int   # orders already placed today

    # Recovery tracking
    recovery_days_elapsed: int

    # Live performance telemetry (realized from actual trades; None = no trades yet)
    live_expectancy:   Optional[float]
    live_max_drawdown: float

    # Historical broker telemetry (computed from JSONL in integration hook)
    broker_rejection_rate: float

    # Transition tracking
    previous_deployments: Dict[str, str]  # {symbol: state_value}


@dataclass
class MicroCapitalDeploymentReport:
    """
    Self-correcting live deployment validation telemetry output.

    Core metric: live_survivability_ratio = live_expectancy / research_expectancy
    Measures: does synthetic alpha survive live friction?

    JSONL schema: see to_dict().
    Future integration: when live_survivability_ratio is stable over N days,
    connect Adaptive Capital Scaling governor.
    """
    timestamp:       str
    evaluation_date: str
    promotion_state: str
    regime_tag:      str

    live_deployment_state:       LiveDeploymentState
    allow_live_micro_deployment: bool

    ledger: LiveExecutionLedger

    # Expectancy chain (research → synthetic → live)
    research_expectancy:  float
    synthetic_expectancy: float
    live_expectancy:      Optional[float]

    # Drift and survivability
    execution_drift_ratio:      Optional[float]   # paper layer: executed / admitted
    live_execution_drift_ratio: Optional[float]   # live / synthetic
    live_survivability_ratio:   Optional[float]   # live / research  ← primary metric

    # Drawdown
    synthetic_max_drawdown: float
    live_max_drawdown:      float

    # Rollback
    rollback_triggered:    bool
    rollback_state:        str
    rollback_trigger_rate: float

    # Broker telemetry
    broker_rejection_rate: float
    order_latency_ms:      Optional[float]
    overnight_gap_impact:  float

    # Governance counters
    governance_freeze_count:       int
    reconciliation_mismatch_count: int

    # Capital
    capital_occupancy_ratio: float

    deterministic_hash: str

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "evaluation_date": self.evaluation_date,
            "promotion_state": self.promotion_state,
            "regime_tag":      self.regime_tag,

            "live_deployment_state":       self.live_deployment_state.value,
            "allow_live_micro_deployment": self.allow_live_micro_deployment,

            "ledger": self.ledger.to_dict(),

            "research_expectancy":  round(self.research_expectancy, 6),
            "synthetic_expectancy": round(self.synthetic_expectancy, 6),
            "live_expectancy": (
                round(self.live_expectancy, 6) if self.live_expectancy is not None else None
            ),

            "execution_drift_ratio": (
                round(self.execution_drift_ratio, 4)
                if self.execution_drift_ratio is not None else None
            ),
            "live_execution_drift_ratio": (
                round(self.live_execution_drift_ratio, 4)
                if self.live_execution_drift_ratio is not None else None
            ),
            "live_survivability_ratio": (
                round(self.live_survivability_ratio, 4)
                if self.live_survivability_ratio is not None else None
            ),

            "synthetic_max_drawdown": round(self.synthetic_max_drawdown, 6),
            "live_max_drawdown":      round(self.live_max_drawdown, 6),

            "rollback_triggered":   self.rollback_triggered,
            "rollback_state":       self.rollback_state,
            "rollback_trigger_rate": round(self.rollback_trigger_rate, 4),

            "broker_rejection_rate": round(self.broker_rejection_rate, 4),
            "order_latency_ms":      self.order_latency_ms,
            "overnight_gap_impact":  round(self.overnight_gap_impact, 6),

            "governance_freeze_count":       self.governance_freeze_count,
            "reconciliation_mismatch_count": self.reconciliation_mismatch_count,

            "capital_occupancy_ratio": round(self.capital_occupancy_ratio, 4),

            "deterministic_hash": self.deterministic_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions — eligibility and reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_live_deployment_eligibility(
    promotion_state:       PromotionState,
    deployment_viable:     bool,
    paper_execution_allowed: bool,
    rollback_state:        str,
    execution_drift_ratio: Optional[float],
    lifecycle_decay_ratio: Optional[float],
    synthetic_max_drawdown: float,
    capital_occupancy_ratio: float,
) -> bool:
    """
    FAIL_CLOSED eligibility gate.
    ALL eight conditions must be satisfied; any None is treated as failing.

    Stricter than paper_execution_allowed:
      - requires PAPER_READY+ (not SHADOW_READY)
      - requires rollback governor approval
      - tighter drawdown threshold (> -0.10)
      - capital occupancy guard (< 0.70)
    """
    if promotion_state not in _LIVE_ELIGIBLE_PROMOTION_STATES:
        return False
    if not deployment_viable:
        return False
    if not paper_execution_allowed:
        return False
    if rollback_state not in _LIVE_ELIGIBLE_ROLLBACK_STATES:
        return False
    if execution_drift_ratio is None or execution_drift_ratio < _ELIGIBILITY_MIN_DRIFT:
        return False
    if lifecycle_decay_ratio is None or lifecycle_decay_ratio < _ELIGIBILITY_MIN_DECAY:
        return False
    if synthetic_max_drawdown <= _ELIGIBILITY_MAX_DRAWDOWN:   # must be strictly > -0.10
        return False
    if capital_occupancy_ratio >= _ELIGIBILITY_MAX_OCCUPANCY:
        return False
    return True


def _compute_recovery_readiness(
    recovery_days_elapsed:       int,
    execution_drift_ratio:       Optional[float],
    deployment_persistence_score: float,
    synthetic_max_drawdown:      float,
    broker_failure_detected:     bool,
) -> bool:
    """
    FAIL_CLOSED post-rollback recovery gate.
    Uses stricter thresholds than general eligibility.
    Only after passing here can the state machine exit RECOVERY_OBSERVATION.
    """
    if recovery_days_elapsed < LIVE_RECOVERY_OBSERVATION_DAYS:
        return False
    if execution_drift_ratio is None or execution_drift_ratio < _RECOVERY_MIN_DRIFT:
        return False
    if deployment_persistence_score < _RECOVERY_MIN_PERSISTENCE:
        return False
    if synthetic_max_drawdown < _RECOVERY_MAX_DRAWDOWN:
        return False
    if broker_failure_detected:
        return False
    return True


def _check_position_reconciliation(
    internal_positions: Dict[str, float],
    broker_positions:   Dict[str, float],
    tolerance:          float = _RECONCILIATION_TOLERANCE,
) -> Tuple[bool, List[str]]:
    """
    Reconcile internal ledger weights vs broker-reported weights.
    Any symbol discrepancy beyond tolerance → not reconciled.
    Returns (is_reconciled, mismatch_details).
    FAIL_CLOSED: extra broker positions and missing broker positions both fail.
    """
    mismatches: List[str] = []
    for sym in sorted(set(internal_positions) | set(broker_positions)):
        internal = internal_positions.get(sym, 0.0)
        broker   = broker_positions.get(sym, 0.0)
        if abs(internal - broker) > tolerance:
            mismatches.append(
                f"{sym}: internal={internal:.4f} broker={broker:.4f}"
                f" diff={abs(internal - broker):.4f}"
            )
    return len(mismatches) == 0, mismatches


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions — symbol selection and metrics
# ─────────────────────────────────────────────────────────────────────────────

def _select_deployment_symbol(
    routing_report: LimitedCapitalRoutingReport,
) -> Optional[str]:
    """
    Select single symbol for live deployment from ALLOW decisions.
    Deterministic: highest readiness_score, tie-break by symbol ascending.
    MAX_LIVE_POSITIONS=1 enforced by design.
    """
    allowed = [
        d for d in routing_report.admission_decisions
        if d.decision == AdmissionDecision.ALLOW
    ]
    if not allowed:
        return None
    best = sorted(allowed, key=lambda d: (-d.readiness_score, d.symbol))[0]
    return best.symbol


def _compute_live_metrics(inputs: MicroCapitalDeploymentInput) -> Dict[str, Any]:
    """
    Compute live vs synthetic drift and survivability ratios. Pure function.

    live_execution_drift_ratio = live_expectancy / synthetic_expectancy
      → measures alpha loss due to live friction (slippage, timing, rejects)

    live_survivability_ratio = live_expectancy / research_expectancy
      → PRIMARY METRIC: does research alpha survive live deployment?
    """
    synthetic_exp = inputs.paper_report.executed_expectancy
    live_exp      = inputs.live_expectancy
    research_exp  = inputs.shadow_report.research_expectancy

    live_drift: Optional[float] = None
    if live_exp is not None and abs(synthetic_exp) > 1e-9:
        live_drift = round(live_exp / synthetic_exp, 4)

    live_surv: Optional[float] = None
    if live_exp is not None and abs(research_exp) > 1e-9:
        live_surv = round(live_exp / research_exp, 4)

    return {
        "synthetic_expectancy":       round(synthetic_exp, 6),
        "live_expectancy":            round(live_exp, 6) if live_exp is not None else None,
        "live_execution_drift_ratio": live_drift,
        "live_survivability_ratio":   live_surv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions — failure reason helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rollback_failure_reason(
    inputs: MicroCapitalDeploymentInput,
) -> LiveDeploymentFailureReason:
    if inputs.reconciliation_mismatch:
        return LiveDeploymentFailureReason.POSITION_MISMATCH
    if inputs.overnight_gap_impact < _OVERNIGHT_GAP_ROLLBACK_THRESHOLD:
        return LiveDeploymentFailureReason.PRICE_GAP_EXCESS
    if inputs.deterministic_replay_mismatch:
        return LiveDeploymentFailureReason.GOVERNANCE_BLOCKED
    return LiveDeploymentFailureReason.ROLLBACK_TRIGGERED


def _freeze_failure_reason(
    inputs: MicroCapitalDeploymentInput,
) -> LiveDeploymentFailureReason:
    if inputs.broker_timeout:
        return LiveDeploymentFailureReason.API_TIMEOUT
    if inputs.api_integrity_unknown:
        return LiveDeploymentFailureReason.API_INTEGRITY_UNKNOWN
    return LiveDeploymentFailureReason.GOVERNANCE_BLOCKED


# ─────────────────────────────────────────────────────────────────────────────
# Pure function — state machine (FAIL_CLOSED priority ordering)
# ─────────────────────────────────────────────────────────────────────────────

def _determine_deployment_state(
    inputs:                     MicroCapitalDeploymentInput,
    eligibility:                bool,
    live_execution_drift_ratio: Optional[float],
) -> LiveDeploymentState:
    """
    Portfolio-level deployment governance state.

    Priority ordering (highest severity first):
      1. LIVE_ROLLBACK  — position mismatch / live drift collapse /
                          gap collapse / replay mismatch / governor full rollback
      2. LIVE_FROZEN    — broker timeout / API integrity unknown / broker failure
      3. RECOVERY_OBSERVATION — post-rollback mandatory 5-day period
      4. LIVE_REDUCED   — rollback pressure > 0.80
      5. LIVE_ACTIVE    — position exists and healthy
      6. PENDING_ENTRY  — eligible, no position
      7. IDLE           — not eligible

    This is a deterministic pure function: same inputs → same state.
    """
    rollback = inputs.rollback_report

    # ── Priority 1: LIVE_ROLLBACK ─────────────────────────────────────────
    if inputs.reconciliation_mismatch:
        return LiveDeploymentState.LIVE_ROLLBACK
    if (live_execution_drift_ratio is not None
            and live_execution_drift_ratio < _LIVE_DRIFT_ROLLBACK_THRESHOLD):
        return LiveDeploymentState.LIVE_ROLLBACK
    if inputs.overnight_gap_impact < _OVERNIGHT_GAP_ROLLBACK_THRESHOLD:
        return LiveDeploymentState.LIVE_ROLLBACK
    if rollback.rollback_state == "full_rollback":
        return LiveDeploymentState.LIVE_ROLLBACK
    if inputs.deterministic_replay_mismatch:
        return LiveDeploymentState.LIVE_ROLLBACK

    # ── Priority 2: LIVE_FROZEN ───────────────────────────────────────────
    if inputs.broker_timeout:
        return LiveDeploymentState.LIVE_FROZEN
    if inputs.api_integrity_unknown:
        return LiveDeploymentState.LIVE_FROZEN
    if inputs.broker_failure_detected:
        return LiveDeploymentState.LIVE_FROZEN

    # ── Priority 3: Recovery observation ─────────────────────────────────
    if inputs.current_deployment_state in _POST_ROLLBACK_STATES:
        recovery_complete = _compute_recovery_readiness(
            recovery_days_elapsed=inputs.recovery_days_elapsed,
            execution_drift_ratio=inputs.paper_report.execution_drift_ratio,
            deployment_persistence_score=inputs.shadow_report.deployment_persistence_score,
            synthetic_max_drawdown=inputs.paper_report.synthetic_max_drawdown,
            broker_failure_detected=inputs.broker_failure_detected,
        )
        if not recovery_complete:
            return LiveDeploymentState.RECOVERY_OBSERVATION
        # Recovery complete — fall through to normal evaluation

    # ── Not eligible → IDLE ───────────────────────────────────────────────
    if not eligibility:
        return LiveDeploymentState.IDLE

    # ── Has live position: govern its state ──────────────────────────────
    if inputs.current_live_position is not None:
        if rollback.rollback_pressure_score > _ROLLBACK_PRESSURE_REDUCE_THRESHOLD:
            return LiveDeploymentState.LIVE_REDUCED
        return LiveDeploymentState.LIVE_ACTIVE

    # ── No position + eligible: await entry ──────────────────────────────
    return LiveDeploymentState.PENDING_ENTRY


# ─────────────────────────────────────────────────────────────────────────────
# Pure function — execution decision
# ─────────────────────────────────────────────────────────────────────────────

def _make_execution_decision(
    inputs:           MicroCapitalDeploymentInput,
    deployment_state: LiveDeploymentState,
    eligibility:      bool,
) -> LiveExecutionDecision:
    """
    Derive governed execution action from deployment state. Pure function.
    Enforces MAX_POSITION_WEIGHT=0.02 and MAX_DAILY_ORDERS=2.
    order_eligible=True only when action=="enter" and all gates pass.
    """
    symbol = _select_deployment_symbol(inputs.routing_report) if eligibility else None
    pos    = inputs.current_live_position

    if deployment_state == LiveDeploymentState.LIVE_ROLLBACK:
        return LiveExecutionDecision(
            symbol=pos.symbol if pos else symbol,
            action="rollback",
            target_weight=0.0,
            failure_reason=_rollback_failure_reason(inputs),
            governance_rationale="FAIL_CLOSED: rollback conditions met — exit all positions",
            order_eligible=False,
        )

    if deployment_state == LiveDeploymentState.LIVE_FROZEN:
        return LiveExecutionDecision(
            symbol=pos.symbol if pos else None,
            action="freeze",
            target_weight=pos.target_weight if pos else 0.0,
            failure_reason=_freeze_failure_reason(inputs),
            governance_rationale="LIVE_FROZEN: new entries and scaling suspended",
            order_eligible=False,
        )

    if deployment_state == LiveDeploymentState.RECOVERY_OBSERVATION:
        days = inputs.recovery_days_elapsed
        return LiveExecutionDecision(
            symbol=None,
            action="idle",
            target_weight=0.0,
            failure_reason=None,
            governance_rationale=(
                f"RECOVERY_OBSERVATION: {days}/{LIVE_RECOVERY_OBSERVATION_DAYS}d — "
                "await full recovery before re-entry"
            ),
            order_eligible=False,
        )

    if deployment_state == LiveDeploymentState.LIVE_REDUCED:
        multiplier = min(1.0, inputs.rollback_report.recommended_exposure_multiplier * 0.5)
        target_w   = round(MAX_POSITION_WEIGHT * multiplier, 6)
        return LiveExecutionDecision(
            symbol=pos.symbol if pos else symbol,
            action="reduce",
            target_weight=target_w,
            failure_reason=None,
            governance_rationale=(
                f"rollback_pressure={inputs.rollback_report.rollback_pressure_score:.2f}"
                f" > {_ROLLBACK_PRESSURE_REDUCE_THRESHOLD} — reduce exposure"
            ),
            order_eligible=False,
        )

    if deployment_state == LiveDeploymentState.LIVE_ACTIVE:
        return LiveExecutionDecision(
            symbol=pos.symbol if pos else symbol,
            action="hold",
            target_weight=pos.target_weight if pos else MAX_POSITION_WEIGHT,
            failure_reason=None,
            governance_rationale="position healthy — hold",
            order_eligible=False,
        )

    if deployment_state == LiveDeploymentState.PENDING_ENTRY:
        if inputs.daily_orders_placed >= MAX_DAILY_ORDERS:
            return LiveExecutionDecision(
                symbol=symbol,
                action="idle",
                target_weight=0.0,
                failure_reason=LiveDeploymentFailureReason.GOVERNANCE_BLOCKED,
                governance_rationale=(
                    f"MAX_DAILY_ORDERS={MAX_DAILY_ORDERS} reached — defer to next cycle"
                ),
                order_eligible=False,
            )
        if symbol is None:
            return LiveExecutionDecision(
                symbol=None,
                action="idle",
                target_weight=0.0,
                failure_reason=None,
                governance_rationale="eligible but no ALLOW symbol in routing decisions",
                order_eligible=False,
            )
        return LiveExecutionDecision(
            symbol=symbol,
            action="enter",
            target_weight=MAX_POSITION_WEIGHT,   # hard cap — no dynamic scaling
            failure_reason=None,
            governance_rationale=(
                f"enter {symbol} weight={MAX_POSITION_WEIGHT} (MAX_POSITION_WEIGHT)"
            ),
            order_eligible=True,
        )

    # IDLE
    return LiveExecutionDecision(
        symbol=None,
        action="idle",
        target_weight=0.0,
        failure_reason=None,
        governance_rationale="deployment not eligible — awaiting governance clearance",
        order_eligible=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure function — deterministic hash
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deterministic_hash(inputs: MicroCapitalDeploymentInput) -> str:
    """
    SHA256 anchored to all upstream layer hashes.
    Ensures full chain auditability: research → shadow → router → paper → rollback → live.
    """
    canonical = {
        "promotion_hash":  inputs.promotion_report.deterministic_hash,
        "shadow_hash":     inputs.shadow_report.deterministic_hash,
        "routing_hash":    inputs.routing_report.deterministic_hash,
        "paper_hash":      inputs.paper_report.deterministic_hash,
        "rollback_hash":   inputs.rollback_report.deterministic_hash,
        "regime_tag":      inputs.regime_tag,
        "evaluation_date": inputs.evaluation_date,
        "broker_positions": dict(sorted(inputs.broker_positions.items())),
        "reconciliation_mismatch":      inputs.reconciliation_mismatch,
        "overnight_gap_impact":         inputs.overnight_gap_impact,
        "live_expectancy":              inputs.live_expectancy,
        "live_max_drawdown":            inputs.live_max_drawdown,
        "recovery_days_elapsed":        inputs.recovery_days_elapsed,
        "current_deployment_state":     inputs.current_deployment_state.value,
        "daily_orders_placed":          inputs.daily_orders_placed,
        "deterministic_replay_mismatch": inputs.deterministic_replay_mismatch,
        "previous_deployments": dict(sorted(inputs.previous_deployments.items())),
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_micro_capital_deployment(
    inputs: MicroCapitalDeploymentInput,
) -> MicroCapitalDeploymentReport:
    """
    Evaluate governed micro capital live deployment for one cycle.

    FAIL_CLOSED priority chain:
      1. reconciliation_mismatch or live drift collapse → LIVE_ROLLBACK
      2. broker failure / timeout / API unknown → LIVE_FROZEN
      3. post-rollback recovery not complete → RECOVERY_OBSERVATION
      4. eligibility gate fails → IDLE
      5. pressure > 0.80 → LIVE_REDUCED
      6. position exists and healthy → LIVE_ACTIVE
      7. eligible + no position → PENDING_ENTRY

    Hard constraints enforced:
      MAX_POSITION_WEIGHT = 0.02
      MAX_LIVE_POSITIONS  = 1
      MAX_DAILY_ORDERS    = 2

    Deterministic: same inputs → same state + same hash.
    observation_governance=True: no unsupervised broker mutation.

    Future integration points (activate ONLY when each layer is ready):
      live_survivability_ratio stable over N days →
        1. Connect Adaptive Capital Scaling governor
        2. Ramp towards Production Deployment OS
      live_execution_drift_ratio consistently > 0.90 →
        1. Promote to PRODUCTION_READY via PromotionReadiness
    """
    ts       = datetime.now(JST).isoformat()
    det_hash = _compute_deterministic_hash(inputs)

    promo    = inputs.promotion_report
    shadow   = inputs.shadow_report
    paper    = inputs.paper_report
    rollback = inputs.rollback_report

    # ── Eligibility gate ─────────────────────────────────────────────────
    eligibility = _compute_live_deployment_eligibility(
        promotion_state=promo.promotion_state,
        deployment_viable=shadow.is_deployment_viable,
        paper_execution_allowed=paper.paper_execution_allowed,
        rollback_state=rollback.rollback_state,
        execution_drift_ratio=paper.execution_drift_ratio,
        lifecycle_decay_ratio=paper.lifecycle_decay_ratio,
        synthetic_max_drawdown=paper.synthetic_max_drawdown,
        capital_occupancy_ratio=paper.capital_occupancy_ratio,
    )

    # ── Live metrics ──────────────────────────────────────────────────────
    metrics    = _compute_live_metrics(inputs)
    live_drift = metrics["live_execution_drift_ratio"]

    # ── Deployment state ──────────────────────────────────────────────────
    deployment_state = _determine_deployment_state(inputs, eligibility, live_drift)

    # ── Execution decision ────────────────────────────────────────────────
    decision = _make_execution_decision(inputs, deployment_state, eligibility)

    # ── Position snapshot update ──────────────────────────────────────────
    rollback_triggered = deployment_state == LiveDeploymentState.LIVE_ROLLBACK
    positions: List[LivePositionSnapshot] = []

    if inputs.current_live_position is not None:
        pos = inputs.current_live_position
        positions.append(LivePositionSnapshot(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=inputs.evaluation_date if rollback_triggered else pos.exit_date,
            target_weight=pos.target_weight,
            actual_weight=pos.actual_weight,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            realized_pnl=pos.realized_pnl,
            unrealized_pnl=pos.unrealized_pnl,
            deployment_state=deployment_state,
            rollback_triggered=rollback_triggered,
            broker_order_id=pos.broker_order_id,
        ))

    ledger = LiveExecutionLedger(
        evaluation_date=inputs.evaluation_date,
        decisions=[decision],
        positions=positions,
    )

    # ── Governance counters ───────────────────────────────────────────────
    freeze_count = (
        1 if deployment_state in (
            LiveDeploymentState.LIVE_FROZEN,
            LiveDeploymentState.LIVE_ROLLBACK,
        ) else 0
    )
    recon_count = 1 if inputs.reconciliation_mismatch else 0

    if deployment_state not in (LiveDeploymentState.IDLE, LiveDeploymentState.RECOVERY_OBSERVATION):
        logger.info(
            "[MICRO_LIVE] state=%s eligible=%s live_drift=%s surv=%s"
            " rollback=%s freeze=%s",
            deployment_state.value, eligibility,
            metrics["live_execution_drift_ratio"],
            metrics["live_survivability_ratio"],
            rollback_triggered, freeze_count,
        )

    return MicroCapitalDeploymentReport(
        timestamp=ts,
        evaluation_date=inputs.evaluation_date,
        promotion_state=promo.promotion_state.value,
        regime_tag=inputs.regime_tag,
        live_deployment_state=deployment_state,
        allow_live_micro_deployment=eligibility,
        ledger=ledger,
        research_expectancy=shadow.research_expectancy,
        synthetic_expectancy=metrics["synthetic_expectancy"],
        live_expectancy=metrics["live_expectancy"],
        execution_drift_ratio=paper.execution_drift_ratio,
        live_execution_drift_ratio=live_drift,
        live_survivability_ratio=metrics["live_survivability_ratio"],
        synthetic_max_drawdown=paper.synthetic_max_drawdown,
        live_max_drawdown=inputs.live_max_drawdown,
        rollback_triggered=rollback_triggered,
        rollback_state=rollback.rollback_state,
        rollback_trigger_rate=paper.rollback_trigger_rate,
        broker_rejection_rate=inputs.broker_rejection_rate,
        order_latency_ms=inputs.order_latency_ms,
        overnight_gap_impact=inputs.overnight_gap_impact,
        governance_freeze_count=freeze_count,
        reconciliation_mismatch_count=recon_count,
        capital_occupancy_ratio=paper.capital_occupancy_ratio,
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence (append-only)
# ─────────────────────────────────────────────────────────────────────────────

def append_deployment_report(
    report:  MicroCapitalDeploymentReport,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append MicroCapitalDeploymentReport to today's JSONL.
    Fail-open: I/O errors are logged but not re-raised.
    Path: {log_dir}/micro_capital_deployment_report_YYYYMMDD.jsonl
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"micro_capital_deployment_report_{today.strftime('%Y%m%d')}.jsonl"
        line = json.dumps(report.to_dict(), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[MICRO_LIVE] report appended → %s state=%s allowed=%s"
            " live_surv=%s hash=%.8s…",
            path.name,
            report.live_deployment_state.value,
            report.allow_live_micro_deployment,
            report.live_survivability_ratio,
            report.deterministic_hash,
        )
    except Exception as e:
        logger.warning("[MICRO_LIVE] report append failed (fail-open): %s", e)


def append_execution_ledger(
    ledger:  LiveExecutionLedger,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append LiveExecutionLedger to today's ledger JSONL.
    Fail-open: I/O errors are logged but not re-raised.
    Path: {log_dir}/live_execution_ledger_YYYYMMDD.jsonl
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"live_execution_ledger_{today.strftime('%Y%m%d')}.jsonl"
        line = json.dumps(ledger.to_dict(), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[MICRO_LIVE] ledger appended → %s n_decisions=%d",
            path.name, len(ledger.decisions),
        )
    except Exception as e:
        logger.warning("[MICRO_LIVE] ledger append failed (fail-open): %s", e)


def load_deployment_reports(log_dir: Path) -> List[dict]:
    """Load all deployment report JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("micro_capital_deployment_report_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


def load_execution_ledgers(log_dir: Path) -> List[dict]:
    """Load all live execution ledger JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("live_execution_ledger_*.jsonl")):
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

def format_deployment_section(report: MicroCapitalDeploymentReport) -> str:
    """Render MICRO CAPITAL DEPLOYMENT section for daily report. Pure function."""
    state  = report.live_deployment_state.value
    flag   = "✓" if report.allow_live_micro_deployment else "✗"
    rb_flag = "✗ ROLLBACK" if report.rollback_triggered else "✓ OK"

    lines = [
        "=== MICRO CAPITAL LIVE DEPLOYMENT ===",
        f"state:    {state}",
        f"eligible: {flag}",
        f"regime:   {report.regime_tag}",
        f"rollback: {rb_flag}  [{report.rollback_state}]",
        "",
        "expectancy chain (research → synthetic → live):",
        f"  research:  {report.research_expectancy*100:+.2f}%",
        f"  synthetic: {report.synthetic_expectancy*100:+.2f}%",
    ]
    if report.live_expectancy is not None:
        lines.append(f"  live:      {report.live_expectancy*100:+.2f}%")
    else:
        lines.append("  live:      N/A (no trades yet)")

    lines += [
        "",
        "survivability metrics:",
        f"  execution_drift:       {report.execution_drift_ratio}",
        f"  live_execution_drift:  {report.live_execution_drift_ratio}",
        f"  live_survivability:    {report.live_survivability_ratio}  ← primary",
        "",
        "drawdown:",
        f"  synthetic: {report.synthetic_max_drawdown*100:+.2f}%",
        f"  live:      {report.live_max_drawdown*100:+.2f}%",
        "",
        "broker telemetry:",
        f"  latency:   {report.order_latency_ms}ms",
        f"  reject_rt: {report.broker_rejection_rate:.3f}",
        f"  gap:       {report.overnight_gap_impact*100:+.2f}%",
        f"  freezes:   {report.governance_freeze_count}",
        f"  recon_err: {report.reconciliation_mismatch_count}",
        "",
        "capital:",
        f"  occupancy: {report.capital_occupancy_ratio:.3f}",
    ]

    if report.ledger.decisions:
        d = report.ledger.decisions[0]
        lines += [
            "",
            f"decision: {d.action}  symbol={d.symbol}  w={d.target_weight:.4f}"
            f"  eligible={d.order_eligible}",
        ]
        if d.failure_reason:
            lines.append(f"  failure: {d.failure_reason.value}")
        lines.append(f"  rationale: {d.governance_rationale}")

    lines += ["", f"hash: {report.deterministic_hash[:16]}…"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def run_micro_capital_deployment_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    promotion_log_dir:     Path,
    shadow_log_dir:        Path,
    router_log_dir:        Path,
    paper_exec_log_dir:    Path,
    deployment_log_dir:    Path,
    reports_dir:           Path,
    candidate_symbols:     List[str],
    symbol_sectors:        Dict[str, str],
    symbol_themes:         Dict[str, str],
    regime_tag:            str,
    # Broker telemetry (supplied by run_live_signal.py from broker API)
    broker_positions:      Dict[str, float],
    reconciliation_mismatch: bool,
    broker_failure_detected: bool,
    api_integrity_unknown:   bool,
    broker_timeout:          bool,
    deterministic_replay_mismatch: bool,
    order_latency_ms:        Optional[float],
    overnight_gap_impact:    float,
    daily_orders_placed:     int,
    recovery_days_elapsed:   int,
    live_expectancy:         Optional[float],
    live_max_drawdown:       float,
    broker_rejection_rate:   float,
    current_live_position:   Optional[LivePositionSnapshot],
    current_deployment_state: LiveDeploymentState,
    previous_deployments:    Dict[str, str],
    rollback_report:         Optional[AutoRollbackGovernanceReport],
    today:                   Optional[date] = None,
    persist:                 bool = True,
) -> Optional[MicroCapitalDeploymentReport]:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Pipeline:
      1. Re-evaluate upstream reports (promotion → shadow → routing → paper).
      2. Accept pre-computed broker telemetry from run_live_signal.py.
      3. Use AutoRollbackGovernanceReport stub (replace with Phase 4A when built).
      4. Run evaluate_micro_capital_deployment().
      5. Append deployment report + ledger.
      6. Append MICRO CAPITAL DEPLOYMENT section to daily future_leader_report.

    Security: API credentials loaded from os.environ — never hardcoded.
    FAIL_CLOSED enforced inside evaluate_micro_capital_deployment.
    observation_governance=True: no unsupervised broker mutation.

    Positioned after run_future_leader_paper_execution_hook in run_live_signal.py.

    Future integration points (activate ONLY when ready):
      live_survivability_ratio stable → connect Adaptive Capital Scaling
      order_eligible=True → connect kabu station REST API (localhost:18080)
    """
    if today is None:
        today = datetime.now(JST).date()

    try:
        surv_recs = _load_jsonl_dir(survivability_log_dir, "future_leader_survivability_*.jsonl")
        fail_recs = _load_jsonl_dir(failure_log_dir, "future_leader_failure_*.jsonl")
        reg_tags  = _load_regime_tags_from_dir(promotion_log_dir.parent / "regime")

        if not surv_recs:
            logger.info("[MICRO_LIVE] no survivability records — hook skipped")
            return None

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

        prev_routing = load_routing_reports(router_log_dir)
        prev_routing_decisions: Dict[str, str] = {}
        if prev_routing:
            for d in prev_routing[-1].get("admission_decisions", []):
                sym, dec = d.get("symbol", ""), d.get("decision", "")
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

        prev_exec = load_paper_execution_reports(paper_exec_log_dir)
        prev_positions: Dict[str, str] = {}
        if prev_exec:
            for pos in prev_exec[-1].get("ledger", {}).get("positions", []):
                sym, st = pos.get("symbol", ""), pos.get("state", "")
                if sym and st:
                    prev_positions[sym] = st

        paper_inputs = PaperExecutionInput(
            promotion_report=promo_report,
            shadow_report=shadow_report,
            routing_report=routing_report,
            regime_tag=regime_tag,
            evaluation_date=today.isoformat(),
            previous_positions=prev_positions,
        )
        paper_report = run_paper_execution(paper_inputs)

        # Use provided rollback report or build a stable stub
        if rollback_report is None:
            rollback_report = AutoRollbackGovernanceReport(
                timestamp=datetime.now(JST).isoformat(),
                evaluation_date=today.isoformat(),
                rollback_state="stable",
                rollback_pressure_score=0.0,
                recommended_exposure_multiplier=1.0,
                deployment_blockers=[],
                deterministic_hash="0" * 64,
            )

        exec_inputs = MicroCapitalDeploymentInput(
            promotion_report=promo_report,
            shadow_report=shadow_report,
            routing_report=routing_report,
            paper_report=paper_report,
            rollback_report=rollback_report,
            evaluation_date=today.isoformat(),
            regime_tag=regime_tag,
            current_live_position=current_live_position,
            current_deployment_state=current_deployment_state,
            broker_positions=broker_positions,
            reconciliation_mismatch=reconciliation_mismatch,
            broker_failure_detected=broker_failure_detected,
            api_integrity_unknown=api_integrity_unknown,
            broker_timeout=broker_timeout,
            deterministic_replay_mismatch=deterministic_replay_mismatch,
            order_latency_ms=order_latency_ms,
            overnight_gap_impact=overnight_gap_impact,
            daily_orders_placed=daily_orders_placed,
            recovery_days_elapsed=recovery_days_elapsed,
            live_expectancy=live_expectancy,
            live_max_drawdown=live_max_drawdown,
            broker_rejection_rate=broker_rejection_rate,
            previous_deployments=previous_deployments,
        )
        exec_report = evaluate_micro_capital_deployment(exec_inputs)

        if persist:
            append_deployment_report(exec_report, deployment_log_dir, today=today)
            append_execution_ledger(exec_report.ledger, deployment_log_dir, today=today)

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"future_leader_report_{today.strftime('%Y%m%d')}.md"
        section     = format_deployment_section(exec_report)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info(
            "[MICRO_LIVE] hook done: state=%s eligible=%s surv=%s rollback=%s",
            exec_report.live_deployment_state.value,
            exec_report.allow_live_micro_deployment,
            exec_report.live_survivability_ratio,
            exec_report.rollback_triggered,
        )
        return exec_report

    except Exception as e:
        logger.warning("[MICRO_LIVE] hook failed (fail-open): %s", e)
        return None
