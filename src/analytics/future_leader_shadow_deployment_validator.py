"""
src/analytics/future_leader_shadow_deployment_validator.py
Phase 3A: Minimal Shadow Deployment Validator

Purpose:
  Measure how much research alpha degrades when deployed as a portfolio.
  Bridges: research OS → deployment survivability validation.

  deployment_degradation_ratio = shadow_portfolio_expectancy / research_expectancy

  FAIL_CLOSED:
    - BLOCKED or RESEARCH_ONLY promotion state → simulation prohibited
    - degradation_ratio < 0.50 → BLOCKED (alpha collapsed in deployment)
    - shadow_max_drawdown < -0.15 → BLOCKED (CLAUDE.md max_dd_limit)
    - promotion_confidence < 0.40 → BLOCKED (confidence_collapse)
    - recent cohorts severely underperform → BLOCKED (survivability_breakdown)

Architecture:
  ShadowDeploymentInput (PromotionReadinessReport + survivability records)
  ↓ FAIL_CLOSED gate: eligible state check
  ↓ deterministic sizing: base_weight * readiness_score * confidence (cap 0.10)
  ↓ cohort simulation: group by obs_date, top-3 by score, weighted portfolio return
  ↓ metrics: research_expectancy vs shadow_expectancy → degradation_ratio
  ↓ blocker check: FAIL_CLOSED
  → ShadowDeploymentReport (append-only JSONL)

INVARIANTS (変更禁止):
  - observation_only=True: no live execution, no allocator mutation
  - deterministic: same inputs → same outputs and hash
  - append-only JSONL: shadow deployment audit trail
  - FAIL_CLOSED: any blocker or ineligible state → is_deployment_viable=False
  - sizing: deterministic only, no optimization or Kelly

禁止:
  - live order generation / predictive_entry_enabled mutation
  - advanced optimizer / Kelly sizing / RL / dynamic capital routing
  - execution simulation / slippage engine / portfolio optimization
  - auto promotion / allocator mutation / universe write
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.analytics.future_leader_promotion_readiness import (
    PromotionReadinessReport,
    PromotionState,
    _load_jsonl_dir,
    _load_regime_tags_from_dir,
    load_promotion_reports,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# Constants (変更禁止)
# ─────────────────────────────────────────────────────────────────────────────

MAX_SHADOW_POSITIONS:  int   = 3     # CLAUDE.md max_positions=3
MAX_WEIGHT:            float = 0.10  # conservative per-position cap (< 0.25 production)
_BASE_WEIGHT:          float = 1.0 / MAX_SHADOW_POSITIONS  # = 0.333 before scaling
FORWARD_WINDOW:        int   = 10    # forward materialization window (business days)

DEGRADATION_THRESHOLD: float = 0.50   # ratio below this → BLOCKED
DRAWDOWN_THRESHOLD:    float = -0.15  # CLAUDE.md max_dd_limit
CONFIDENCE_THRESHOLD:  float = 0.40   # confidence below this → BLOCKED
RECENT_SPLIT_RATIO:    float = 0.50   # recent = last 50% of cohorts
SURVIVABILITY_BREAKDOWN_RATIO: float = 0.30  # recent_mean < ratio * shadow_mean → BLOCKED

MIN_SHADOW_COHORTS:    int   = 3     # minimum cohorts for meaningful simulation
_CLEAN_QUALITY_MIN:    float = 1.0   # data_quality_weight threshold

SHADOW_DEPLOYMENT_BLOCKERS = {
    "deployment_degradation_excessive",  # degradation_ratio < 0.50
    "shadow_drawdown_excessive",          # max_drawdown < -0.15
    "survivability_breakdown",            # recent cohorts severely underperform
    "confidence_collapse",                # promotion_confidence < 0.40
}

# States that allow deployment simulation (BLOCKED and RESEARCH_ONLY excluded)
_DEPLOYMENT_ELIGIBLE_STATES = frozenset({
    PromotionState.SHADOW_READY,
    PromotionState.PAPER_READY,
    PromotionState.LIMITED_EXECUTION_READY,
    PromotionState.PRODUCTION_READY,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowDeploymentInput:
    """
    Input bundle for shadow deployment simulation.
    promotion_report must be SHADOW_READY or above for simulation to proceed.
    """
    promotion_report:      PromotionReadinessReport
    survivability_records: List[dict]  # raw future_leader_survivability_*.jsonl records
    failure_records:       List[dict]  # raw future_leader_failure_*.jsonl records ([] = unavailable)
    evaluation_date:       str         # ISO date YYYY-MM-DD


@dataclass
class ShadowPosition:
    """
    One virtual position in the shadow portfolio.
    observation_only=True: never connected to execution path.
    """
    symbol:               str
    entry_date:           str
    entry_price:          float
    allocation_weight:    float   # deterministic sizing output
    current_return:       float   # = fwd10 (already materialized)
    holding_days:         int     # from failure record, else FORWARD_WINDOW
    readiness_score:      float   # portfolio-level (from promotion_report)
    promotion_confidence: float   # portfolio-level (from promotion_report)


@dataclass
class ShadowPortfolioSnapshot:
    """
    One cohort's portfolio state (grouped by observation_date).
    """
    observation_date: str
    n_positions:      int
    position_symbols: List[str]
    portfolio_return: float   # equal-weighted mean of position returns
    portfolio_mae:    float   # worst position return (drawdown proxy)
    allocation_weight: float  # per-position weight from deterministic sizing


@dataclass
class ShadowDeploymentReport:
    """
    Shadow deployment validation output.
    observation_only=True: governance telemetry, no execution mutation.

    JSONL schema: see to_dict().
    Future integration: when is_deployment_viable=True and state=SHADOW_READY,
    connect to LimitedCapitalRouter.route() after Paper validation.
    """
    timestamp:         str
    evaluation_date:   str
    promotion_state:   str   # PromotionState.value string

    # Core deployment metrics
    research_expectancy:          float          # unweighted mean of all CLEAN fwd10
    shadow_expectancy:            float          # mean of cohort portfolio returns
    deployment_degradation_ratio: Optional[float]  # None if research_expectancy <= 0

    shadow_max_drawdown:           float   # from NAV curve of cohort returns
    survivability_adjusted_return: float   # shadow_expectancy * (0.5 + 0.5 * survival_frac)
    deployment_persistence_score:  float   # degradation_ratio * survival_adjustment (0–1)
    average_holding_duration:      float   # mean holding days across positions

    # FAIL_CLOSED
    deployment_blockers:    List[str]
    is_deployment_viable:   bool   # False if state ineligible or any blocker

    # Simulation metadata
    total_research_records:     int
    shadow_positions_simulated: int
    cohorts_simulated:          int
    allocation_weight:          float  # per-position deterministic weight

    # Determinism
    deterministic_hash: str

    def to_dict(self) -> dict:
        return {
            "timestamp":                   self.timestamp,
            "evaluation_date":             self.evaluation_date,
            "promotion_state":             self.promotion_state,
            "research_expectancy":         round(self.research_expectancy, 6),
            "shadow_expectancy":           round(self.shadow_expectancy, 6),
            "deployment_degradation_ratio": (
                round(self.deployment_degradation_ratio, 4)
                if self.deployment_degradation_ratio is not None else None
            ),
            "shadow_max_drawdown":           round(self.shadow_max_drawdown, 6),
            "survivability_adjusted_return": round(self.survivability_adjusted_return, 6),
            "deployment_persistence_score":  round(self.deployment_persistence_score, 4),
            "average_holding_duration":      round(self.average_holding_duration, 2),
            "deployment_blockers":           self.deployment_blockers,
            "is_deployment_viable":          self.is_deployment_viable,
            "total_research_records":        self.total_research_records,
            "shadow_positions_simulated":    self.shadow_positions_simulated,
            "cohorts_simulated":             self.cohorts_simulated,
            "allocation_weight":             round(self.allocation_weight, 6),
            "deterministic_hash":            self.deterministic_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN filter (pure — mirrors promotion_readiness pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_clean_survivability(records: List[dict]) -> List[dict]:
    """
    CLEAN complete: dq >= 1.0, insufficient_forward_data=False.
    Deduplicates (symbol, observation_date) — first occurrence wins.
    """
    seen = set()
    out: List[dict] = []
    for r in records:
        key = (r.get("symbol", ""), r.get("observation_date", ""))
        if not (key[0] and key[1]) or key in seen:
            continue
        seen.add(key)
        if (float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN
                and not r.get("insufficient_forward_data", True)):
            out.append(r)
    return out


def _build_failure_lookup(
    failure_records: List[dict],
) -> Dict[Tuple[str, str], dict]:
    """
    Build {(symbol, observation_date): failure_record} from CLEAN failure records.
    Last occurrence wins.
    """
    lookup: Dict[Tuple[str, str], dict] = {}
    for r in failure_records:
        sym = r.get("symbol", "")
        dt  = r.get("observation_date", "")
        if (sym and dt
                and float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN):
            lookup[(sym, dt)] = r
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic sizing (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_allocation_weight(report: PromotionReadinessReport) -> float:
    """
    Deterministic sizing: base_weight * readiness_score * confidence, capped at MAX_WEIGHT.

    base_weight = 1 / MAX_SHADOW_POSITIONS (= 0.333)
    No optimization, no Kelly, no dynamic routing.
    """
    raw = _BASE_WEIGHT * report.readiness_score * report.promotion_confidence
    return round(min(raw, MAX_WEIGHT), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Cohort simulation (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _holding_days_for(
    symbol:         str,
    obs_date:       str,
    failure_lookup: Dict[Tuple[str, str], dict],
) -> int:
    """Return actual holding days from failure record, else FORWARD_WINDOW."""
    fail = failure_lookup.get((symbol, obs_date))
    if (fail is not None
            and fail.get("failure_detected", False)
            and fail.get("failure_day_offset") is not None):
        return max(1, int(fail["failure_day_offset"]))
    return FORWARD_WINDOW


def _simulate_cohort(
    obs_date:        str,
    candidates:      List[dict],
    alloc_weight:    float,
    readiness_score: float,
    confidence:      float,
    failure_lookup:  Dict[Tuple[str, str], dict],
    max_positions:   int,
) -> Tuple[ShadowPortfolioSnapshot, List[ShadowPosition]]:
    """
    Simulate one cohort (all records for obs_date).

    Selection: sort by future_leader_score desc, take top max_positions.
    Portfolio return: equal-weighted mean of selected position returns.
    """
    sorted_cands = sorted(
        candidates,
        key=lambda r: float(r.get("future_leader_score", 0.0)),
        reverse=True,
    )
    selected = sorted_cands[:max_positions]

    positions: List[ShadowPosition] = []
    returns:   List[float]          = []

    for rec in selected:
        sym   = rec.get("symbol", "")
        fwd10 = rec.get("forward_return_10d")
        if not sym or fwd10 is None:
            continue
        fwd10 = float(fwd10)
        pos   = ShadowPosition(
            symbol=sym,
            entry_date=obs_date,
            entry_price=float(rec.get("close_price_at_observation", 0.0)),
            allocation_weight=alloc_weight,
            current_return=fwd10,
            holding_days=_holding_days_for(sym, obs_date, failure_lookup),
            readiness_score=readiness_score,
            promotion_confidence=confidence,
        )
        positions.append(pos)
        returns.append(fwd10)

    if not returns:
        return ShadowPortfolioSnapshot(
            observation_date=obs_date,
            n_positions=0,
            position_symbols=[],
            portfolio_return=0.0,
            portfolio_mae=0.0,
            allocation_weight=alloc_weight,
        ), []

    portfolio_return = float(np.mean(returns))
    portfolio_mae    = float(min(returns))

    return ShadowPortfolioSnapshot(
        observation_date=obs_date,
        n_positions=len(positions),
        position_symbols=[p.symbol for p in positions],
        portfolio_return=round(portfolio_return, 6),
        portfolio_mae=round(portfolio_mae, 6),
        allocation_weight=alloc_weight,
    ), positions


def _simulate_full_portfolio(
    clean_surv:      List[dict],
    alloc_weight:    float,
    readiness_score: float,
    confidence:      float,
    failure_lookup:  Dict[Tuple[str, str], dict],
    max_positions:   int = MAX_SHADOW_POSITIONS,
) -> Tuple[List[ShadowPortfolioSnapshot], List[ShadowPosition]]:
    """
    Simulate shadow portfolio across all unique observation_dates.
    Returns (snapshots sorted by date, all positions).
    Deterministic: groups sorted by date, cohorts sorted by score.
    """
    groups: Dict[str, List[dict]] = {}
    for rec in clean_surv:
        d = rec.get("observation_date", "")
        if d:
            groups.setdefault(d, []).append(rec)

    all_snapshots: List[ShadowPortfolioSnapshot] = []
    all_positions: List[ShadowPosition]          = []

    for obs_date in sorted(groups.keys()):
        snap, positions = _simulate_cohort(
            obs_date, groups[obs_date],
            alloc_weight, readiness_score, confidence,
            failure_lookup, max_positions,
        )
        if snap.n_positions > 0:
            all_snapshots.append(snap)
            all_positions.extend(positions)

    return all_snapshots, all_positions


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_nav_drawdown(snapshots: List[ShadowPortfolioSnapshot]) -> float:
    """
    Compute max drawdown from NAV curve of sequential cohort returns.
    Each cohort contributes: NAV *= (1 + portfolio_return * allocation_weight).
    Returns max drawdown as a negative float (0.0 if no drawdown).
    """
    if not snapshots:
        return 0.0
    nav          = 1.0
    peak         = 1.0
    max_drawdown = 0.0
    for snap in snapshots:
        nav  *= (1.0 + snap.portfolio_return * snap.allocation_weight)
        peak  = max(peak, nav)
        dd    = (nav - peak) / max(peak, 1e-9)
        max_drawdown = min(max_drawdown, dd)
    return round(max_drawdown, 6)


def _compute_shadow_metrics(
    snapshots:  List[ShadowPortfolioSnapshot],
    positions:  List[ShadowPosition],
    clean_surv: List[dict],
) -> Dict:
    """
    Compute all shadow deployment metrics from simulation outputs.
    Pure function: no I/O, no side effects.
    """
    # Research expectancy — unweighted mean of ALL CLEAN fwd10 (baseline)
    all_fwd10 = [
        float(r["forward_return_10d"])
        for r in clean_surv
        if r.get("forward_return_10d") is not None
    ]
    research_expectancy = float(np.mean(all_fwd10)) if all_fwd10 else 0.0

    # Shadow expectancy — mean of cohort portfolio returns
    cohort_returns      = [snap.portfolio_return for snap in snapshots]
    shadow_expectancy   = float(np.mean(cohort_returns)) if cohort_returns else 0.0

    # Degradation ratio
    if research_expectancy > 1e-9:
        degradation_ratio: Optional[float] = round(shadow_expectancy / research_expectancy, 4)
    else:
        degradation_ratio = None  # undefined when research baseline is zero/negative

    # Max drawdown from NAV curve
    shadow_max_drawdown = _compute_nav_drawdown(snapshots)

    # Survival fraction (fraction of positions with positive return)
    if positions:
        survival_fraction = float(np.mean([p.current_return > 0 for p in positions]))
    else:
        survival_fraction = 0.5

    # Survivability adjusted return
    survivability_adjusted_return = round(
        shadow_expectancy * (0.5 + 0.5 * survival_fraction), 6
    )

    # Deployment persistence score = degradation_ratio * survival_adjustment
    if degradation_ratio is not None:
        surv_adj  = 0.5 + 0.5 * survival_fraction
        pers_score = round(max(0.0, degradation_ratio * surv_adj), 4)
    else:
        pers_score = 0.0

    # Average holding duration
    avg_holding = (
        float(np.mean([p.holding_days for p in positions]))
        if positions else float(FORWARD_WINDOW)
    )

    # Recent cohort mean (last 50%) for survivability_breakdown check
    recent_n      = max(1, len(snapshots) // 2)
    recent_snaps  = snapshots[-recent_n:]
    recent_returns = [s.portfolio_return for s in recent_snaps]
    recent_mean   = float(np.mean(recent_returns)) if recent_returns else shadow_expectancy

    return {
        "research_expectancy":          round(research_expectancy, 6),
        "shadow_expectancy":            round(shadow_expectancy, 6),
        "deployment_degradation_ratio": degradation_ratio,
        "shadow_max_drawdown":          shadow_max_drawdown,
        "survivability_adjusted_return": survivability_adjusted_return,
        "deployment_persistence_score": pers_score,
        "average_holding_duration":     round(avg_holding, 2),
        "survival_fraction":            round(survival_fraction, 4),
        "recent_cohort_mean":           round(recent_mean, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FAIL_CLOSED blocker check (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _check_shadow_blockers(
    metrics:          Dict,
    promotion_report: PromotionReadinessReport,
) -> Tuple[List[str], List[str]]:
    """
    Evaluate SHADOW_DEPLOYMENT_BLOCKERS.
    Returns (blockers: List[str], warnings: List[str]).

    All checks are independent; multiple blockers may trigger simultaneously.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    ratio = metrics.get("deployment_degradation_ratio")
    dd    = metrics.get("shadow_max_drawdown", 0.0)
    shadow_mean  = metrics.get("shadow_expectancy", 0.0)
    recent_mean  = metrics.get("recent_cohort_mean", shadow_mean)

    # 1. Deployment degradation excessive
    if ratio is None:
        blockers.append("deployment_degradation_excessive")
        warnings.append("degradation_ratio undefined (research_expectancy <= 0)")
    elif ratio < DEGRADATION_THRESHOLD:
        blockers.append("deployment_degradation_excessive")
        warnings.append(
            f"degradation_ratio={ratio:.3f} < {DEGRADATION_THRESHOLD} "
            f"(shadow={shadow_mean*100:+.1f}% vs research={metrics.get('research_expectancy',0)*100:+.1f}%)"
        )

    # 2. Shadow drawdown excessive
    if dd < DRAWDOWN_THRESHOLD:
        blockers.append("shadow_drawdown_excessive")
        warnings.append(
            f"shadow_max_drawdown={dd*100:.1f}% < {DRAWDOWN_THRESHOLD*100:.0f}%"
        )

    # 3. Survivability breakdown — recent cohorts severely underperform
    if (shadow_mean > 1e-6
            and recent_mean < SURVIVABILITY_BREAKDOWN_RATIO * shadow_mean):
        blockers.append("survivability_breakdown")
        warnings.append(
            f"recent_cohort_mean={recent_mean*100:+.2f}% < "
            f"{SURVIVABILITY_BREAKDOWN_RATIO:.0%} of shadow_mean={shadow_mean*100:+.2f}%"
        )

    # 4. Confidence collapse
    if promotion_report.promotion_confidence < CONFIDENCE_THRESHOLD:
        blockers.append("confidence_collapse")
        warnings.append(
            f"promotion_confidence={promotion_report.promotion_confidence:.3f} "
            f"< {CONFIDENCE_THRESHOLD}"
        )

    return blockers, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic hash (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deterministic_hash(inputs: ShadowDeploymentInput) -> str:
    """
    SHA256 of canonical input representation.
    Anchored to promotion_report.deterministic_hash for chain auditability.
    """
    def _sort_recs(recs: List[dict]) -> List[dict]:
        return sorted(recs, key=lambda r: (r.get("symbol", ""), r.get("observation_date", "")))

    canonical = {
        "promotion_hash":  inputs.promotion_report.deterministic_hash,
        "survivability":   _sort_recs(inputs.survivability_records),
        "failure":         _sort_recs(inputs.failure_records),
        "evaluation_date": inputs.evaluation_date,
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Main validation entry point (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def validate_shadow_deployment(
    inputs: ShadowDeploymentInput,
) -> ShadowDeploymentReport:
    """
    Validate research alpha survivability under shadow portfolio deployment.

    FAIL_CLOSED:
      - BLOCKED or RESEARCH_ONLY promotion state → not viable, no simulation
      - any SHADOW_DEPLOYMENT_BLOCKER → is_deployment_viable=False

    Deterministic: same inputs → same outputs and same deterministic_hash.
    observation_only=True: no execution mutation.

    Returns ShadowDeploymentReport with full deployment metrics.
    """
    ts         = datetime.now(JST).isoformat()
    det_hash   = _compute_deterministic_hash(inputs)
    report     = inputs.promotion_report
    state      = report.promotion_state

    # FAIL_CLOSED gate: ineligible states produce a minimal BLOCKED report
    if state not in _DEPLOYMENT_ELIGIBLE_STATES:
        return ShadowDeploymentReport(
            timestamp=ts,
            evaluation_date=inputs.evaluation_date,
            promotion_state=state.value,
            research_expectancy=0.0,
            shadow_expectancy=0.0,
            deployment_degradation_ratio=None,
            shadow_max_drawdown=0.0,
            survivability_adjusted_return=0.0,
            deployment_persistence_score=0.0,
            average_holding_duration=float(FORWARD_WINDOW),
            deployment_blockers=["ineligible_promotion_state"],
            is_deployment_viable=False,
            total_research_records=0,
            shadow_positions_simulated=0,
            cohorts_simulated=0,
            allocation_weight=0.0,
            deterministic_hash=det_hash,
        )

    # Compute allocation weight (deterministic)
    alloc_weight = _compute_allocation_weight(report)

    # Filter CLEAN survivability records
    clean_surv = _filter_clean_survivability(inputs.survivability_records)

    # Insufficient data → BLOCKED
    if len(clean_surv) < MIN_SHADOW_COHORTS:
        return ShadowDeploymentReport(
            timestamp=ts,
            evaluation_date=inputs.evaluation_date,
            promotion_state=state.value,
            research_expectancy=0.0,
            shadow_expectancy=0.0,
            deployment_degradation_ratio=None,
            shadow_max_drawdown=0.0,
            survivability_adjusted_return=0.0,
            deployment_persistence_score=0.0,
            average_holding_duration=float(FORWARD_WINDOW),
            deployment_blockers=["insufficient_survivability_records"],
            is_deployment_viable=False,
            total_research_records=len(clean_surv),
            shadow_positions_simulated=0,
            cohorts_simulated=0,
            allocation_weight=alloc_weight,
            deterministic_hash=det_hash,
        )

    # Build failure lookup for holding duration computation
    failure_lookup = _build_failure_lookup(inputs.failure_records)

    # Run shadow portfolio simulation
    snapshots, positions = _simulate_full_portfolio(
        clean_surv, alloc_weight,
        report.readiness_score, report.promotion_confidence,
        failure_lookup,
    )

    if len(snapshots) < MIN_SHADOW_COHORTS:
        return ShadowDeploymentReport(
            timestamp=ts,
            evaluation_date=inputs.evaluation_date,
            promotion_state=state.value,
            research_expectancy=0.0,
            shadow_expectancy=0.0,
            deployment_degradation_ratio=None,
            shadow_max_drawdown=0.0,
            survivability_adjusted_return=0.0,
            deployment_persistence_score=0.0,
            average_holding_duration=float(FORWARD_WINDOW),
            deployment_blockers=["insufficient_shadow_cohorts"],
            is_deployment_viable=False,
            total_research_records=len(clean_surv),
            shadow_positions_simulated=len(positions),
            cohorts_simulated=len(snapshots),
            allocation_weight=alloc_weight,
            deterministic_hash=det_hash,
        )

    # Compute metrics
    metrics = _compute_shadow_metrics(snapshots, positions, clean_surv)

    # FAIL_CLOSED blocker check
    blockers, warnings = _check_shadow_blockers(metrics, report)

    is_viable = len(blockers) == 0

    if blockers:
        logger.info(
            "[SHADOW_DEPLOY] blockers: %s (state=%s ratio=%s)",
            blockers, state.value, metrics.get("deployment_degradation_ratio"),
        )

    return ShadowDeploymentReport(
        timestamp=ts,
        evaluation_date=inputs.evaluation_date,
        promotion_state=state.value,
        research_expectancy=metrics["research_expectancy"],
        shadow_expectancy=metrics["shadow_expectancy"],
        deployment_degradation_ratio=metrics["deployment_degradation_ratio"],
        shadow_max_drawdown=metrics["shadow_max_drawdown"],
        survivability_adjusted_return=metrics["survivability_adjusted_return"],
        deployment_persistence_score=metrics["deployment_persistence_score"],
        average_holding_duration=metrics["average_holding_duration"],
        deployment_blockers=sorted(blockers),
        is_deployment_viable=is_viable,
        total_research_records=len(clean_surv),
        shadow_positions_simulated=len(positions),
        cohorts_simulated=len(snapshots),
        allocation_weight=alloc_weight,
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence (append-only)
# ─────────────────────────────────────────────────────────────────────────────

def append_shadow_deployment_report(
    report:  ShadowDeploymentReport,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append ShadowDeploymentReport to today's JSONL.
    append-only: never overwrites existing records.
    Fail-open: I/O errors are logged but not re-raised.

    JSONL path: {log_dir}/shadow_deployment_YYYYMMDD.jsonl
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"shadow_deployment_{today.strftime('%Y%m%d')}.jsonl"
        line     = json.dumps(report.to_dict(), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[SHADOW_DEPLOY] appended → %s viable=%s ratio=%s hash=%.8s…",
            log_path.name,
            report.is_deployment_viable,
            report.deployment_degradation_ratio,
            report.deterministic_hash,
        )
    except Exception as e:
        logger.warning("[SHADOW_DEPLOY] append failed (fail-open): %s", e)


def load_shadow_deployment_reports(log_dir: Path) -> List[dict]:
    """Load all shadow deployment JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("shadow_deployment_*.jsonl")):
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

def format_shadow_deployment_section(report: ShadowDeploymentReport) -> str:
    """Render SHADOW DEPLOYMENT section for daily report. Pure function."""
    lines = [
        "=== SHADOW DEPLOYMENT VALIDATOR ===",
        f"state:         {report.promotion_state}",
        f"viable:        {report.is_deployment_viable}",
        "",
        f"research_exp:  {report.research_expectancy*100:+.2f}%",
        f"shadow_exp:    {report.shadow_expectancy*100:+.2f}%",
        f"degradation:   {(report.deployment_degradation_ratio or 0.0):.3f}",
        f"max_drawdown:  {report.shadow_max_drawdown*100:+.2f}%",
        f"persistence:   {report.deployment_persistence_score:.3f}",
        f"avg_holding:   {report.average_holding_duration:.1f}d",
        "",
        f"records:  {report.total_research_records} (positions={report.shadow_positions_simulated},"
        f" cohorts={report.cohorts_simulated})",
        f"weight:   {report.allocation_weight:.4f}",
    ]

    if report.deployment_blockers:
        lines += ["", "BLOCKERS (FAIL_CLOSED):"]
        for b in report.deployment_blockers:
            lines.append(f"  [BLOCKED] {b}")

    lines += ["", f"hash: {report.deterministic_hash[:16]}…"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_shadow_deployment_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    promotion_log_dir:     Path,
    shadow_log_dir:        Path,
    reports_dir:           Path,
    today:                 Optional[date] = None,
    persist:               bool = True,
) -> Optional[ShadowDeploymentReport]:
    """
    Integration hook for run_live_signal.py. Fail-open.

    1. Loads the most recent promotion readiness report.
    2. Loads raw survivability and failure records.
    3. Runs validate_shadow_deployment().
    4. Appends to shadow_deployment_YYYYMMDD.jsonl.
    5. Appends SHADOW DEPLOYMENT section to daily future_leader_report.

    observation_only=True: no execution mutation.
    FAIL_CLOSED enforced inside validate_shadow_deployment.

    Positioned after run_future_leader_promotion_readiness_hook in run_live_signal.py.

    Future integration points (activate ONLY when each layer is ready):
      is_deployment_viable=True + SHADOW_READY →
        1. Call LimitedCapitalRouter.route() after Paper validation
        2. Connect AutoRollback guard
      is_deployment_viable=True + PRODUCTION_READY →
        1. Connect DeployableUniverseGovernor
        2. Full Allocator integration
    """
    if today is None:
        today = datetime.now(JST).date()

    try:
        # Load most recent promotion report
        promo_recs = load_promotion_reports(promotion_log_dir)
        if not promo_recs:
            logger.info("[SHADOW_DEPLOY] no promotion reports — hook skipped")
            return None

        latest_promo_dict = promo_recs[-1]

        # Reconstruct PromotionReadinessReport from dict
        # (minimal reconstruction for validation — not full round-trip)
        from src.analytics.future_leader_promotion_readiness import (
            evaluate_promotion_readiness,
            PromotionReadinessInputs,
        )
        # Re-evaluate using live data so report is current
        surv_recs = _load_jsonl_dir(survivability_log_dir, "future_leader_survivability_*.jsonl")
        fail_recs = _load_jsonl_dir(failure_log_dir, "future_leader_failure_*.jsonl")
        from src.analytics.future_leader_promotion_readiness import _load_regime_tags_from_dir
        regime_log_dir = promotion_log_dir.parent / "regime"
        reg_tags   = _load_regime_tags_from_dir(regime_log_dir)

        if not surv_recs:
            logger.info("[SHADOW_DEPLOY] no survivability records — hook skipped")
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
        report = validate_shadow_deployment(shadow_inputs)

        if persist:
            append_shadow_deployment_report(report, shadow_log_dir, today=today)

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"future_leader_report_{today.strftime('%Y%m%d')}.md"
        section     = format_shadow_deployment_section(report)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info(
            "[SHADOW_DEPLOY] hook done: viable=%s ratio=%s drawdown=%.3f blockers=%s",
            report.is_deployment_viable,
            report.deployment_degradation_ratio,
            report.shadow_max_drawdown,
            report.deployment_blockers,
        )
        return report

    except Exception as e:
        logger.warning("[SHADOW_DEPLOY] hook failed (fail-open): %s", e)
        return None
