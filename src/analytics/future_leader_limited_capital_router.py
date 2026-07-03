"""
src/analytics/future_leader_limited_capital_router.py
Phase 3B — Future Leader Limited Capital Router
(Shadow Admission Governance Layer)

Purpose:
  Portfolio governance telemetry accumulation.
  Generates virtual admission / suppression / rollback decisions daily
  to measure whether research alpha survives portfolio admission.

  NOT an allocator. NOT an execution simulator.
  Responsibility: shadow admission governance telemetry only.

Architecture:
  PromotionReadinessReport + ShadowDeploymentReport
  ↓ FAIL_CLOSED lifecycle gate (BLOCKED/RESEARCH_ONLY → DENY all)
  ↓ virtual exposure snapshot (sector/theme/concentration tracking)
  ↓ admission decision logic (ALLOW/REDUCED_SIZE/DENY per symbol)
  ↓ capital suppression (risk_off multiplier)
  ↓ rollback simulation (persistence collapse → DENY)
  ↓ portfolio metrics: admitted_expectancy / unsuppressed_shadow_expectancy
  ↓ drift detection: deployment_drift_ratio / suppression_alpha_preservation_ratio
  ↓ state transition telemetry (previous → current decision)
  → LimitedCapitalRoutingReport (append-only JSONL)

INVARIANTS (変更禁止):
  - observation_only=True: reads promotion/shadow reports, writes governance telemetry only
  - FAIL_CLOSED: any blocker or ambiguity → DENY
  - deterministic: identical inputs → identical hash and outputs
  - append-only JSONL: routing decisions are immutable
  - no execution path connection: no live orders, no allocator, no signal mutation

禁止:
  - live order generation / predictive_entry_enabled mutation
  - allocator mutation / universe write
  - Kelly sizing / RL / dynamic optimizer / adaptive exposure model
  - broker API / synthetic fills / slippage engine
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

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# Constants (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

# Admission thresholds — FAIL_CLOSED boundaries
_DEGRADATION_DENY_THRESHOLD:     float = 0.50   # ratio < this → DENY
_DEGRADATION_REDUCED_THRESHOLD:  float = 0.75   # ratio < this (but >= DENY) → REDUCED_SIZE
_PERSISTENCE_DENY_THRESHOLD:     float = 0.40   # persistence < this → DENY (rollback sim)
_PERSISTENCE_REDUCED_THRESHOLD:  float = 0.60   # persistence < this → REDUCED_SIZE

_CONFIDENCE_REDUCED_THRESHOLD:   float = 0.55   # confidence < this → REDUCED_SIZE

# Concentration suppression thresholds
_SECTOR_EXPOSURE_DENY_THRESHOLD:    float = 0.50   # sector exposure > this → DENY
_SECTOR_EXPOSURE_REDUCED_THRESHOLD: float = 0.35   # sector exposure > this → REDUCED_SIZE
_POSITION_WEIGHT_DENY_THRESHOLD:    float = 0.10   # position weight > this → DENY

# Capital suppression multiplier (risk_off regime)
_RISK_OFF_SIZE_MULTIPLIER: float = 0.50

# Fallback proposed weight when shadow_report.allocation_weight is unavailable
# (shadow report caps allocation at MAX_WEIGHT=0.10, so this is only for degenerate cases)
_BASE_PROPOSED_WEIGHT: float = 1.0 / 3  # ≈ 0.3333 — never used when shadow report has weight

# Correlated cluster: sector with exposure above this threshold
_CLUSTER_EXPOSURE_THRESHOLD: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Enums (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

class AdmissionDecision(Enum):
    ALLOW        = "allow"
    REDUCED_SIZE = "reduced_size"
    DENY         = "deny"


class RoutingSuppressionReason(Enum):
    NONE                 = "none"
    RISK_OFF             = "risk_off"
    PERSISTENCE_COLLAPSE = "persistence_collapse"
    CONCENTRATION_EXCESS = "concentration_excess"
    DEPLOYMENT_DRIFT     = "deployment_drift"
    BLOCKED_PROMOTION    = "blocked_promotion"
    CONFIDENCE_WEAKNESS  = "confidence_weakness"


# ─────────────────────────────────────────────────────────────────────────────
# Routing-eligible lifecycle states (BLOCKED / RESEARCH_ONLY → DENY all)
# ─────────────────────────────────────────────────────────────────────────────

_ROUTING_ELIGIBLE_STATES = frozenset({
    PromotionState.SHADOW_READY,
    PromotionState.PAPER_READY,
    PromotionState.LIMITED_EXECUTION_READY,
    PromotionState.PRODUCTION_READY,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LimitedCapitalRoutingInput:
    """
    Input bundle for limited capital routing governance.

    candidate_symbols:   symbols to route (sorted for determinism)
    symbol_sectors:      {symbol: sector_str}  ("unknown" if missing)
    symbol_themes:       {symbol: theme_str}   ("unknown" if missing)
    regime_tag:          current market regime (risk_on/risk_off/range/volatile)
    previous_decisions:  {symbol: AdmissionDecision.value} from prior run
    evaluation_date:     ISO date YYYY-MM-DD
    """
    promotion_report:   PromotionReadinessReport
    shadow_report:      ShadowDeploymentReport
    candidate_symbols:  List[str]
    symbol_sectors:     Dict[str, str]
    symbol_themes:      Dict[str, str]
    regime_tag:         str
    previous_decisions: Dict[str, str]
    evaluation_date:    str


@dataclass
class VirtualAdmissionDecision:
    """
    Per-symbol virtual admission decision.
    observation_only=True: no execution path connection.
    """
    symbol:             str
    decision:           AdmissionDecision
    proposed_weight:    float
    adjusted_weight:    float
    suppression_reason: RoutingSuppressionReason

    readiness_score:              float
    promotion_confidence:         float
    deployment_degradation_ratio: float
    deployment_persistence_score: float

    previous_decision:   Optional[str]
    current_decision:    str
    decision_transition: str

    def to_dict(self) -> dict:
        return {
            "symbol":                       self.symbol,
            "decision":                     self.decision.value,
            "proposed_weight":              round(self.proposed_weight, 6),
            "adjusted_weight":              round(self.adjusted_weight, 6),
            "suppression_reason":           self.suppression_reason.value,
            "readiness_score":              round(self.readiness_score, 4),
            "promotion_confidence":         round(self.promotion_confidence, 4),
            "deployment_degradation_ratio": round(self.deployment_degradation_ratio, 4),
            "deployment_persistence_score": round(self.deployment_persistence_score, 4),
            "previous_decision":            self.previous_decision,
            "current_decision":             self.current_decision,
            "decision_transition":          self.decision_transition,
        }


@dataclass
class VirtualExposureSnapshot:
    """
    Portfolio-level virtual exposure at time of routing.
    Simple accumulation — no advanced correlation engine.
    """
    sector_exposure:          Dict[str, float]
    theme_exposure:           Dict[str, float]
    position_concentration:   float
    correlated_cluster_count: int

    def to_dict(self) -> dict:
        return {
            "sector_exposure":         {k: round(v, 4) for k, v in self.sector_exposure.items()},
            "theme_exposure":          {k: round(v, 4) for k, v in self.theme_exposure.items()},
            "position_concentration":  round(self.position_concentration, 4),
            "correlated_cluster_count": self.correlated_cluster_count,
        }


@dataclass
class LimitedCapitalRoutingReport:
    """
    Portfolio governance telemetry output.
    observation_only=True: no execution path connection.

    Key metrics:
      admitted_expectancy              — post-suppression portfolio expectancy proxy
      unsuppressed_shadow_expectancy   — shadow_expectancy without admission filtering
      deployment_drift_ratio           — admitted_expectancy / research_expectancy
      suppression_alpha_preservation_ratio — admitted_expectancy / unsuppressed

    JSONL schema: see to_dict().
    Future integration: connect to PaperExecutionEngine after Paper validation.
    """
    timestamp:       str
    evaluation_date: str
    promotion_state: str
    regime_tag:      str

    admission_decisions: List[VirtualAdmissionDecision]
    exposure_snapshot:   VirtualExposureSnapshot

    research_expectancy:            float
    shadow_expectancy:              float
    unsuppressed_shadow_expectancy: float
    admitted_expectancy:            float

    deployment_drift_ratio:               Optional[float]
    suppression_alpha_preservation_ratio: Optional[float]

    admission_rate:              float
    reduced_size_rate:           float
    denial_rate:                 float
    persistence_suppression_rate: float
    exposure_concentration:      float

    deterministic_hash: str

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "evaluation_date": self.evaluation_date,
            "promotion_state": self.promotion_state,
            "regime_tag":      self.regime_tag,

            "admission_decisions": [d.to_dict() for d in self.admission_decisions],
            "exposure_snapshot":   self.exposure_snapshot.to_dict(),

            "research_expectancy":            round(self.research_expectancy, 6),
            "shadow_expectancy":              round(self.shadow_expectancy, 6),
            "unsuppressed_shadow_expectancy": round(self.unsuppressed_shadow_expectancy, 6),
            "admitted_expectancy":            round(self.admitted_expectancy, 6),

            "deployment_drift_ratio": (
                round(self.deployment_drift_ratio, 4)
                if self.deployment_drift_ratio is not None else None
            ),
            "suppression_alpha_preservation_ratio": (
                round(self.suppression_alpha_preservation_ratio, 4)
                if self.suppression_alpha_preservation_ratio is not None else None
            ),

            "admission_rate":              round(self.admission_rate, 4),
            "reduced_size_rate":           round(self.reduced_size_rate, 4),
            "denial_rate":                 round(self.denial_rate, 4),
            "persistence_suppression_rate": round(self.persistence_suppression_rate, 4),
            "exposure_concentration":      round(self.exposure_concentration, 4),

            "deterministic_hash": self.deterministic_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Admission decision logic (pure — deterministic rule-based)
# ─────────────────────────────────────────────────────────────────────────────

def _decide_single_admission(
    proposed_weight:     float,
    degradation_ratio:   float,
    persistence_score:   float,
    confidence:          float,
    regime_tag:          str,
    sector:              str,
    sector_weights:      Dict[str, float],
    deployment_blockers: List[str],
) -> Tuple[AdmissionDecision, RoutingSuppressionReason, float]:
    """
    Evaluate admission for a single symbol.
    Returns (decision, suppression_reason, adjusted_weight).

    Rule priority: DENY checked first (FAIL_CLOSED), then REDUCED_SIZE, then ALLOW.
    Ambiguity → DENY.
    """
    # ── DENY conditions (FAIL_CLOSED — checked first) ──────────────────────

    if deployment_blockers:
        return AdmissionDecision.DENY, RoutingSuppressionReason.BLOCKED_PROMOTION, 0.0

    if persistence_score < _PERSISTENCE_DENY_THRESHOLD:
        return AdmissionDecision.DENY, RoutingSuppressionReason.PERSISTENCE_COLLAPSE, 0.0

    if degradation_ratio < _DEGRADATION_DENY_THRESHOLD:
        return AdmissionDecision.DENY, RoutingSuppressionReason.DEPLOYMENT_DRIFT, 0.0

    if proposed_weight > _POSITION_WEIGHT_DENY_THRESHOLD:
        return AdmissionDecision.DENY, RoutingSuppressionReason.CONCENTRATION_EXCESS, 0.0

    # Cumulative sector exposure after adding this symbol
    projected_sector_exp = sector_weights.get(sector, 0.0) + proposed_weight
    if projected_sector_exp > _SECTOR_EXPOSURE_DENY_THRESHOLD:
        return AdmissionDecision.DENY, RoutingSuppressionReason.CONCENTRATION_EXCESS, 0.0

    # ── REDUCED_SIZE conditions ──────────────────────────────────────────────

    # risk_off regime: capital suppression with 0.50x multiplier
    if regime_tag == "risk_off":
        adj = round(proposed_weight * _RISK_OFF_SIZE_MULTIPLIER, 6)
        return AdmissionDecision.REDUCED_SIZE, RoutingSuppressionReason.RISK_OFF, adj

    # Moderate deployment degradation
    if degradation_ratio < _DEGRADATION_REDUCED_THRESHOLD:
        adj = round(proposed_weight * 0.75, 6)
        return AdmissionDecision.REDUCED_SIZE, RoutingSuppressionReason.DEPLOYMENT_DRIFT, adj

    # Weak persistence (below REDUCED threshold, above DENY)
    if persistence_score < _PERSISTENCE_REDUCED_THRESHOLD:
        adj = round(proposed_weight * 0.75, 6)
        return AdmissionDecision.REDUCED_SIZE, RoutingSuppressionReason.PERSISTENCE_COLLAPSE, adj

    # Weak confidence
    if confidence < _CONFIDENCE_REDUCED_THRESHOLD:
        adj = round(proposed_weight * 0.75, 6)
        return AdmissionDecision.REDUCED_SIZE, RoutingSuppressionReason.CONFIDENCE_WEAKNESS, adj

    # Elevated (but not excessive) sector concentration
    if projected_sector_exp > _SECTOR_EXPOSURE_REDUCED_THRESHOLD:
        adj = round(proposed_weight * 0.60, 6)
        return AdmissionDecision.REDUCED_SIZE, RoutingSuppressionReason.CONCENTRATION_EXCESS, adj

    # ── ALLOW ────────────────────────────────────────────────────────────────
    return AdmissionDecision.ALLOW, RoutingSuppressionReason.NONE, proposed_weight


# ─────────────────────────────────────────────────────────────────────────────
# Exposure snapshot (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _build_exposure_snapshot(
    decisions:      List[VirtualAdmissionDecision],
    symbol_sectors: Dict[str, str],
    symbol_themes:  Dict[str, str],
) -> VirtualExposureSnapshot:
    """
    Accumulate virtual exposure from admitted (non-DENY) decisions.
    Pure accumulation — no correlation engine.
    """
    sector_exp: Dict[str, float] = {}
    theme_exp:  Dict[str, float] = {}
    max_weight: float = 0.0

    for d in decisions:
        if d.decision == AdmissionDecision.DENY:
            continue
        w = d.adjusted_weight
        if w > max_weight:
            max_weight = w

        sector = symbol_sectors.get(d.symbol, "unknown")
        sector_exp[sector] = round(sector_exp.get(sector, 0.0) + w, 6)

        theme = symbol_themes.get(d.symbol, "unknown")
        theme_exp[theme] = round(theme_exp.get(theme, 0.0) + w, 6)

    cluster_count = sum(1 for v in sector_exp.values() if v > _CLUSTER_EXPOSURE_THRESHOLD)

    return VirtualExposureSnapshot(
        sector_exposure=sector_exp,
        theme_exposure=theme_exp,
        position_concentration=round(max_weight, 6),
        correlated_cluster_count=cluster_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio expectancy proxy (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_admitted_expectancy(
    decisions:        List[VirtualAdmissionDecision],
    shadow_expectancy: float,
) -> float:
    """
    admitted_expectancy proxy.

    Uses weight-ratio scaling: admitted / proposed.
    Rationale: shadow_expectancy reflects the full-portfolio baseline.
    Reducing admitted weight proportionally reduces expected return.

    When per-symbol forward returns are available (future layer),
    this will be replaced with a weighted mean of admitted-symbol returns.
    """
    if not decisions:
        return 0.0
    total_proposed = sum(d.proposed_weight for d in decisions)
    if total_proposed <= 1e-9:
        return 0.0
    total_admitted = sum(d.adjusted_weight for d in decisions)
    return round(shadow_expectancy * (total_admitted / total_proposed), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Decision transition telemetry (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_decision_transition(previous: Optional[str], current: str) -> str:
    if previous is None:
        return f"new->{current}"
    if previous == current:
        return f"{current}->same"
    return f"{previous}->{current}"


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic hash (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deterministic_hash(inputs: LimitedCapitalRoutingInput) -> str:
    """
    SHA256 of canonical input.
    Anchored to promotion + shadow hashes for full chain auditability.
    """
    canonical = {
        "promotion_hash":    inputs.promotion_report.deterministic_hash,
        "shadow_hash":       inputs.shadow_report.deterministic_hash,
        "candidate_symbols": sorted(inputs.candidate_symbols),
        "symbol_sectors":    dict(sorted(inputs.symbol_sectors.items())),
        "symbol_themes":     dict(sorted(inputs.symbol_themes.items())),
        "regime_tag":        inputs.regime_tag,
        "previous_decisions": dict(sorted(inputs.previous_decisions.items())),
        "evaluation_date":   inputs.evaluation_date,
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Deny-all helper (pure — used for ineligible state fast-path)
# ─────────────────────────────────────────────────────────────────────────────

def _deny_all(
    inputs:    LimitedCapitalRoutingInput,
    det_hash:  str,
    timestamp: str,
    reason:    RoutingSuppressionReason,
) -> LimitedCapitalRoutingReport:
    shadow = inputs.shadow_report
    report = inputs.promotion_report
    proposed = shadow.allocation_weight if shadow.allocation_weight > 1e-9 else _BASE_PROPOSED_WEIGHT
    decisions: List[VirtualAdmissionDecision] = [
        VirtualAdmissionDecision(
            symbol=sym,
            decision=AdmissionDecision.DENY,
            proposed_weight=proposed,
            adjusted_weight=0.0,
            suppression_reason=reason,
            readiness_score=report.readiness_score,
            promotion_confidence=report.promotion_confidence,
            deployment_degradation_ratio=shadow.deployment_degradation_ratio or 0.0,
            deployment_persistence_score=shadow.deployment_persistence_score,
            previous_decision=inputs.previous_decisions.get(sym),
            current_decision=AdmissionDecision.DENY.value,
            decision_transition=_compute_decision_transition(
                inputs.previous_decisions.get(sym),
                AdmissionDecision.DENY.value,
            ),
        )
        for sym in sorted(inputs.candidate_symbols)
    ]
    exposure = VirtualExposureSnapshot(
        sector_exposure={}, theme_exposure={},
        position_concentration=0.0, correlated_cluster_count=0,
    )
    n = len(decisions)
    return LimitedCapitalRoutingReport(
        timestamp=timestamp,
        evaluation_date=inputs.evaluation_date,
        promotion_state=report.promotion_state.value,
        regime_tag=inputs.regime_tag,
        admission_decisions=decisions,
        exposure_snapshot=exposure,
        research_expectancy=shadow.research_expectancy,
        shadow_expectancy=shadow.shadow_expectancy,
        unsuppressed_shadow_expectancy=shadow.shadow_expectancy,
        admitted_expectancy=0.0,
        deployment_drift_ratio=None,
        suppression_alpha_preservation_ratio=None,
        admission_rate=0.0,
        reduced_size_rate=0.0,
        denial_rate=1.0 if n else 0.0,
        persistence_suppression_rate=0.0,
        exposure_concentration=0.0,
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main routing entry point (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def route_limited_capital(
    inputs: LimitedCapitalRoutingInput,
) -> LimitedCapitalRoutingReport:
    """
    Generate virtual admission governance decisions.

    FAIL_CLOSED:
      - BLOCKED or RESEARCH_ONLY → all candidates denied
      - deployment blockers present → all candidates denied
      - any ambiguity → DENY

    Deterministic: same inputs → same outputs and deterministic_hash.
    observation_only=True: no execution mutation.

    Future integration points:
      SHADOW_READY   → connect to PaperExecutionEngine (future)
      PAPER_READY+   → connect to AutoRollbackGovernor (future)
      LIMITED_EXECUTION_READY → connect to MicroCapitalLiveDeployment (future)
    """
    ts       = datetime.now(JST).isoformat()
    det_hash = _compute_deterministic_hash(inputs)
    report   = inputs.promotion_report
    shadow   = inputs.shadow_report

    # ── FAIL_CLOSED lifecycle gate ──────────────────────────────────────────
    if report.promotion_state not in _ROUTING_ELIGIBLE_STATES:
        return _deny_all(inputs, det_hash, ts, RoutingSuppressionReason.BLOCKED_PROMOTION)

    # ── Extract portfolio-level metrics ─────────────────────────────────────
    degradation_ratio   = shadow.deployment_degradation_ratio or 0.0
    persistence_score   = shadow.deployment_persistence_score
    deployment_blockers = shadow.deployment_blockers

    # Proposed weight: use shadow report's computed allocation (already capped at 0.10).
    # Falls back to _BASE_PROPOSED_WEIGHT only if shadow weight is zero (degenerate).
    proposed_weight = shadow.allocation_weight if shadow.allocation_weight > 1e-9 else _BASE_PROPOSED_WEIGHT

    # ── Per-symbol admission (sorted for determinism) ───────────────────────
    decisions: List[VirtualAdmissionDecision] = []
    running_sector_weights: Dict[str, float] = {}

    for sym in sorted(inputs.candidate_symbols):
        sector = inputs.symbol_sectors.get(sym, "unknown")

        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=proposed_weight,
            degradation_ratio=degradation_ratio,
            persistence_score=persistence_score,
            confidence=report.promotion_confidence,
            regime_tag=inputs.regime_tag,
            sector=sector,
            sector_weights=running_sector_weights,
            deployment_blockers=deployment_blockers,
        )

        # Accumulate sector exposure only for non-denied decisions
        if dec != AdmissionDecision.DENY:
            running_sector_weights[sector] = round(
                running_sector_weights.get(sector, 0.0) + adj_w, 6
            )

        prev = inputs.previous_decisions.get(sym)
        decisions.append(VirtualAdmissionDecision(
            symbol=sym,
            decision=dec,
            proposed_weight=proposed_weight,
            adjusted_weight=adj_w,
            suppression_reason=reason,
            readiness_score=report.readiness_score,
            promotion_confidence=report.promotion_confidence,
            deployment_degradation_ratio=degradation_ratio,
            deployment_persistence_score=persistence_score,
            previous_decision=prev,
            current_decision=dec.value,
            decision_transition=_compute_decision_transition(prev, dec.value),
        ))

    # ── Exposure snapshot ────────────────────────────────────────────────────
    exposure = _build_exposure_snapshot(decisions, inputs.symbol_sectors, inputs.symbol_themes)

    # ── Portfolio-level rates ────────────────────────────────────────────────
    n           = len(decisions)
    n_allow     = sum(1 for d in decisions if d.decision == AdmissionDecision.ALLOW)
    n_reduced   = sum(1 for d in decisions if d.decision == AdmissionDecision.REDUCED_SIZE)
    n_denied    = sum(1 for d in decisions if d.decision == AdmissionDecision.DENY)
    n_pers_deny = sum(
        1 for d in decisions
        if d.decision == AdmissionDecision.DENY
        and d.suppression_reason == RoutingSuppressionReason.PERSISTENCE_COLLAPSE
    )

    admission_rate             = n_allow   / n if n else 0.0
    reduced_size_rate          = n_reduced / n if n else 0.0
    denial_rate                = n_denied  / n if n else 1.0
    persistence_suppression_rate = n_pers_deny / n if n else 0.0

    # ── Counterfactual expectancy metrics ────────────────────────────────────
    # unsuppressed_shadow_expectancy = shadow_expectancy from shadow deployment
    # (portfolio without admission constraints — shadow simulation baseline)
    unsuppressed = shadow.shadow_expectancy

    # admitted_expectancy = shadow_expectancy scaled by admitted-weight fraction
    admitted_exp = _compute_admitted_expectancy(decisions, shadow.shadow_expectancy)

    # deployment_drift_ratio = admitted_expectancy / research_expectancy
    # measures: "does portfolio admission further erode research alpha?"
    if shadow.research_expectancy > 1e-9:
        drift_ratio: Optional[float] = round(admitted_exp / shadow.research_expectancy, 4)
    else:
        drift_ratio = None

    # suppression_alpha_preservation_ratio = admitted_expectancy / unsuppressed
    # > 1.0 → suppression preserved alpha (denied low-quality symbols)
    # < 1.0 → suppression cost alpha (over-suppression)
    if abs(unsuppressed) > 1e-9:
        preservation_ratio: Optional[float] = round(admitted_exp / unsuppressed, 4)
    else:
        preservation_ratio = None

    # Max sector exposure among admitted positions
    exposure_concentration = (
        max(exposure.sector_exposure.values())
        if exposure.sector_exposure else 0.0
    )

    return LimitedCapitalRoutingReport(
        timestamp=ts,
        evaluation_date=inputs.evaluation_date,
        promotion_state=report.promotion_state.value,
        regime_tag=inputs.regime_tag,
        admission_decisions=decisions,
        exposure_snapshot=exposure,
        research_expectancy=shadow.research_expectancy,
        shadow_expectancy=shadow.shadow_expectancy,
        unsuppressed_shadow_expectancy=unsuppressed,
        admitted_expectancy=admitted_exp,
        deployment_drift_ratio=drift_ratio,
        suppression_alpha_preservation_ratio=preservation_ratio,
        admission_rate=round(admission_rate, 4),
        reduced_size_rate=round(reduced_size_rate, 4),
        denial_rate=round(denial_rate, 4),
        persistence_suppression_rate=round(persistence_suppression_rate, 4),
        exposure_concentration=round(exposure_concentration, 4),
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence (append-only)
# ─────────────────────────────────────────────────────────────────────────────

def append_routing_report(
    report:  LimitedCapitalRoutingReport,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append one LimitedCapitalRoutingReport to today's JSONL.
    append-only: never overwrites existing records.
    Fail-open: I/O errors are logged but not re-raised.

    Path: {log_dir}/routing_decisions_YYYYMMDD.jsonl
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"routing_decisions_{today.strftime('%Y%m%d')}.jsonl"
        line     = json.dumps(report.to_dict(), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[LC_ROUTER] appended → %s state=%s regime=%s admit=%.2f deny=%.2f drift=%s hash=%.8s…",
            log_path.name,
            report.promotion_state,
            report.regime_tag,
            report.admission_rate,
            report.denial_rate,
            report.deployment_drift_ratio,
            report.deterministic_hash,
        )
    except Exception as e:
        logger.warning("[LC_ROUTER] append failed (fail-open): %s", e)


def load_routing_reports(log_dir: Path) -> List[dict]:
    """Load all routing decisions JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("routing_decisions_*.jsonl")):
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

def format_routing_section(report: LimitedCapitalRoutingReport) -> str:
    """Render LIMITED CAPITAL ROUTER section for daily report. Pure function."""
    lines = [
        "=== LIMITED CAPITAL ROUTER ===",
        f"state:   {report.promotion_state}",
        f"regime:  {report.regime_tag}",
        "",
        "expectancy chain:",
        f"  research:       {report.research_expectancy*100:+.2f}%",
        f"  shadow:         {report.shadow_expectancy*100:+.2f}%",
        f"  unsuppressed:   {report.unsuppressed_shadow_expectancy*100:+.2f}%",
        f"  admitted:       {report.admitted_expectancy*100:+.2f}%",
        "",
        "degradation metrics:",
        f"  drift_ratio:         {report.deployment_drift_ratio}",
        f"  preservation_ratio:  {report.suppression_alpha_preservation_ratio}",
        "",
        "admission rates:",
        f"  allow={report.admission_rate:.2f}  reduced={report.reduced_size_rate:.2f}"
        f"  deny={report.denial_rate:.2f}  pers_deny={report.persistence_suppression_rate:.2f}",
        "",
        f"exposure_concentration: {report.exposure_concentration:.3f}",
    ]

    if report.admission_decisions:
        lines += ["", "decisions:"]
        for d in report.admission_decisions:
            marker = "✓" if d.decision == AdmissionDecision.ALLOW else (
                "~" if d.decision == AdmissionDecision.REDUCED_SIZE else "✗"
            )
            lines.append(
                f"  {marker} {d.symbol:10s}  {d.current_decision:12s}"
                f"  w={d.adjusted_weight:.4f}  {d.suppression_reason.value}"
                f"  [{d.decision_transition}]"
            )

    lines += ["", f"hash: {report.deterministic_hash[:16]}…"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_limited_capital_router_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    promotion_log_dir:     Path,
    shadow_log_dir:        Path,
    router_log_dir:        Path,
    reports_dir:           Path,
    candidate_symbols:     List[str],
    symbol_sectors:        Dict[str, str],
    symbol_themes:         Dict[str, str],
    regime_tag:            str,
    today:                 Optional[date] = None,
    persist:               bool = True,
) -> Optional[LimitedCapitalRoutingReport]:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Pipeline:
      1. Loads survivability / failure / regime records.
      2. Evaluates PromotionReadinessReport (current).
      3. Validates ShadowDeploymentReport (current).
      4. Loads previous routing decisions for transition telemetry.
      5. Runs route_limited_capital().
      6. Appends to routing_decisions_YYYYMMDD.jsonl.
      7. Appends LIMITED CAPITAL ROUTER section to daily future_leader_report.

    observation_only=True: no execution mutation.
    FAIL_CLOSED enforced inside route_limited_capital.

    Positioned after run_future_leader_shadow_deployment_hook in run_live_signal.py.

    Future integration points (activate ONLY when each layer is ready):
      is_deployment_viable=True + SHADOW_READY →
        call PaperExecutionEngine.simulate(routing_report)
      PAPER_READY → call AutoRollbackGovernor.evaluate()
      LIMITED_EXECUTION_READY → call MicroCapitalLiveDeployment.route()
    """
    if today is None:
        today = datetime.now(JST).date()

    try:
        surv_recs = _load_jsonl_dir(survivability_log_dir, "future_leader_survivability_*.jsonl")
        fail_recs = _load_jsonl_dir(failure_log_dir, "future_leader_failure_*.jsonl")
        reg_tags  = _load_regime_tags_from_dir(promotion_log_dir.parent / "regime")

        if not surv_recs:
            logger.info("[LC_ROUTER] no survivability records — hook skipped")
            return None

        # Re-evaluate promotion readiness from current data
        promo_inputs = PromotionReadinessInputs(
            survivability_records=surv_recs,
            failure_records=fail_recs,
            regime_tags=reg_tags,
            evaluation_date=today.isoformat(),
        )
        promo_report = evaluate_promotion_readiness(promo_inputs)

        # Re-evaluate shadow deployment from current data
        shadow_inputs = ShadowDeploymentInput(
            promotion_report=promo_report,
            survivability_records=surv_recs,
            failure_records=fail_recs,
            evaluation_date=today.isoformat(),
        )
        shadow_report = validate_shadow_deployment(shadow_inputs)

        # Load previous routing decisions for transition telemetry
        prev_records = load_routing_reports(router_log_dir)
        prev_decisions: Dict[str, str] = {}
        if prev_records:
            last_decisions = prev_records[-1].get("admission_decisions", [])
            for d in last_decisions:
                sym = d.get("symbol", "")
                dec = d.get("decision", "")
                if sym and dec:
                    prev_decisions[sym] = dec

        routing_inputs = LimitedCapitalRoutingInput(
            promotion_report=promo_report,
            shadow_report=shadow_report,
            candidate_symbols=candidate_symbols,
            symbol_sectors=symbol_sectors,
            symbol_themes=symbol_themes,
            regime_tag=regime_tag,
            previous_decisions=prev_decisions,
            evaluation_date=today.isoformat(),
        )
        routing_report = route_limited_capital(routing_inputs)

        if persist:
            append_routing_report(routing_report, router_log_dir, today=today)

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"future_leader_report_{today.strftime('%Y%m%d')}.md"
        section     = format_routing_section(routing_report)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info(
            "[LC_ROUTER] hook done: state=%s regime=%s admit=%.2f deny=%.2f"
            " drift=%s preservation=%s blockers=%d",
            routing_report.promotion_state,
            routing_report.regime_tag,
            routing_report.admission_rate,
            routing_report.denial_rate,
            routing_report.deployment_drift_ratio,
            routing_report.suppression_alpha_preservation_ratio,
            len([d for d in routing_report.admission_decisions
                 if d.decision == AdmissionDecision.DENY]),
        )
        return routing_report

    except Exception as e:
        logger.warning("[LC_ROUTER] hook failed (fail-open): %s", e)
        return None
