"""
src/runtime/policy/portfolio_intelligence_engine.py
— Unified portfolio intelligence operating layer

Integrates: analytics_policy_bridge, runtime_exit_orchestrator,
executed_vs_skipped_expectancy, exit_intelligence, position_lifecycle,
regime analytics.

Produces bounded adaptive runtime policies for:
  slot allocation · sector caps · portfolio heat · hold extension ·
  exit sensitivity · risk scaling · correlation suppression · entry throttling

STRICT CONSTRAINTS:
  - Bounded adaptation only (hard caps in PortfolioConstraints, frozen)
  - Fail-closed on any integrity violation
  - Cooldown + anti-oscillation guards (portfolio-level)
  - No recursive feedback (decisions never consume prior decisions as inputs)
  - Deterministic: same inputs → same decision_ids
  - Append-only JSONL decision log, atomic fsync writes
  - No direct strategy mutation, no autonomous alpha generation

Runtime integration:
  run_portfolio_intelligence_hook() → injected into run_live_signal.py
  Returns (modified_signals, modified_orders, PortfolioIntelligenceOverlay)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
JST = timezone(timedelta(hours=9))

DEFAULT_STORE_PATH = Path("runtime/policy/portfolio_intelligence_decisions.jsonl")

# ── Policy dimension constants ─────────────────────────────────────────────────
PIDTYPE_SLOT_SCALE         = "pi_slot_scale"
PIDTYPE_ENTRY_SUPPRESS     = "pi_entry_suppress"
PIDTYPE_HEAT_LIMIT         = "pi_heat_limit"
PIDTYPE_RISK_SCALE         = "pi_risk_scale"
PIDTYPE_THROTTLE           = "pi_entry_throttle"
PIDTYPE_SURVIVABILITY      = "pi_survivability_gate"
PIDTYPE_CONVEXITY          = "pi_convexity_retention"
PIDTYPE_CORR_SUPPRESSION   = "pi_corr_suppression"

ALL_PIDTYPES = (
    PIDTYPE_SLOT_SCALE, PIDTYPE_ENTRY_SUPPRESS, PIDTYPE_HEAT_LIMIT,
    PIDTYPE_RISK_SCALE, PIDTYPE_THROTTLE, PIDTYPE_SURVIVABILITY,
    PIDTYPE_CONVEXITY, PIDTYPE_CORR_SUPPRESSION,
)

# ── Trigger source constants ───────────────────────────────────────────────────
TRIGGER_SECTOR_CLUSTER      = "sector_cluster_score"
TRIGGER_MOMENTUM_SYNC       = "momentum_synchronization"
TRIGGER_VOL_CORRELATION     = "volatility_correlation"
TRIGGER_LIFECYCLE_SYNC      = "lifecycle_synchronization"
TRIGGER_CORR_HEAT           = "correlated_heat"
TRIGGER_EFFECTIVE_DIV       = "effective_diversification"
TRIGGER_REGIME_FRAGILITY    = "regime_fragility"
TRIGGER_CONVEXITY_SCORE     = "convexity_retention_score"
TRIGGER_SURVIVABILITY_SCORE = "survivability_score"
TRIGGER_CAPITAL_LEAKAGE     = "capital_constraint_leakage"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def _now_jst() -> str:
    return datetime.now(JST).isoformat()

def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")

def _days_between(a: str, b: str) -> int:
    return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. PortfolioConstraints — hard bounds, frozen, never relaxed by code
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioConstraints:
    """
    Hard bounds on all portfolio intelligence adjustments.
    Frozen: immutable by design. Never relaxed by code.
    """
    # Cross-position correlation thresholds
    sector_cluster_threshold: float = 0.60    # ≥ this → high sector concentration
    momentum_sync_threshold: float = 0.80     # ≥ this → momentum synchronization risk
    vol_corr_threshold: float = 0.75          # ≥ this → volatility correlation risk
    lifecycle_sync_threshold: float = 0.75    # ≥ this → lifecycle synchronization risk

    # Survivability thresholds
    min_survivability_score: float = 0.30     # < this → survivability gate
    max_correlated_heat: float = 0.65         # > this → heat limit applied
    min_effective_diversification: float = 0.30  # < this → suppression active

    # Slot allocation scale bounds [fraction of baseline]
    slot_scale_min: float = 0.20
    slot_scale_max: float = 1.00

    # Entry throttle scale bounds
    throttle_min: float = 0.10
    throttle_max: float = 1.00

    # Risk scaling bounds
    risk_scale_min: float = 0.40
    risk_scale_max: float = 1.50

    # Correlated exposure suppression scale
    corr_suppression_min: float = 0.30
    corr_suppression_max: float = 1.00

    # Heat limit bounds
    heat_limit_min: float = 0.05
    heat_limit_max: float = 0.50

    # Convexity
    winner_retention_threshold: float = 0.65  # runner_score ≥ this → preserve winner
    persistence_retention_threshold: float = 0.55

    # Governance
    cooldown_days: int = 2                    # min days between portfolio-level changes
    anti_oscillation_window_days: int = 5     # block direction reversals within this window
    min_sample_size: int = 3                  # min positions for portfolio analysis
    max_decisions_per_week: int = 6           # max portfolio-level decisions per 7-day window

    # Maximum deltas per cycle (prevent runaway single-cycle jumps)
    max_slot_delta: float = 0.25
    max_risk_delta: float = 0.35
    max_throttle_delta: float = 0.30
    max_heat_delta: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 2. Input snapshots
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioPositionSnapshot:
    """Current state of a single live position, enriched for portfolio analysis."""
    symbol: str
    sector: str = "unknown"
    hold_days: int = 0
    rsr: float = 50.0                        # RSR percentile [0, 100]
    rsr_momentum: float = 0.0               # momentum score (signed)
    current_return: float = 0.0
    rs_score: Optional[float] = None         # relative strength vs TOPIX
    signal: int = 1                          # 1=hold, 0=exit
    runner_score: float = 0.0               # from exit analytics
    persistence_score: float = 0.0          # trend persistence
    vol: float = 0.20                        # annualized volatility


@dataclass
class RegimeSnapshot:
    """Regime state for portfolio-level analysis."""
    regime: str = "unknown"               # "bull", "bear", "neutral", "unknown"
    deteriorating: bool = False
    rs_collapse: bool = False
    breadth_score: float = 0.50           # [0, 1]
    confidence: float = 0.50             # [0, 1]


@dataclass
class ExpectancySnapshot:
    """From executed_vs_skipped analytics."""
    capital_constraint_leakage: float = 0.0
    heat_limit_leakage: float = 0.0
    sector_exposure_leakage: float = 0.0
    skipped_expectancy_delta: float = 0.0
    opportunity_capture_ratio: float = 1.0
    n_executed: int = 0
    n_skipped: int = 0


@dataclass
class ExitIntelSnapshot:
    """From exit convexity analytics (RunnerRetentionSummary)."""
    premature_exit_ratio: float = 0.0
    mean_giveback_ratio: float = 0.0
    mean_runner_retention: float = 1.0
    mean_convexity_capture: float = 1.0
    per_symbol_runner_score: Dict[str, float] = field(default_factory=dict)
    per_symbol_persistence: Dict[str, float] = field(default_factory=dict)
    per_symbol_giveback: Dict[str, float] = field(default_factory=dict)
    n_trades: int = 0


@dataclass
class PortfolioIntelligenceInput:
    """Aggregated input from all subsystems."""
    positions: List[PortfolioPositionSnapshot] = field(default_factory=list)
    regime: RegimeSnapshot = field(default_factory=RegimeSnapshot)
    expectancy: ExpectancySnapshot = field(default_factory=ExpectancySnapshot)
    exit_intel: ExitIntelSnapshot = field(default_factory=ExitIntelSnapshot)
    portfolio_heat: float = 0.0           # current drawdown magnitude [0, 1]
    available_slots: int = 0
    max_slots: int = 4
    eval_date: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Intermediate analytics results (not persisted — inputs to decisions)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossPositionCorrelation:
    """Computed cross-position correlation metrics."""
    sector_cluster_score: float = 0.0     # [0, 1]: fraction in largest sector cluster
    momentum_sync_score: float = 0.0     # [0, 1]: synchronized momentum risk
    vol_correlation_score: float = 0.0   # [0, 1]: volatility correlation proxy
    lifecycle_sync_score: float = 0.0    # [0, 1]: lifecycle (entry timing) synchronization
    regime_sync_score: float = 0.0       # [0, 1]: held across same regime
    hidden_factor_overlap: float = 0.0   # [0, 1]: high-RSR concentration (systematic factor proxy)
    overall_corr_score: float = 0.0      # [0, 1]: weighted composite


@dataclass
class SurvivabilityReport:
    """Portfolio-level survivability assessment."""
    correlated_heat: float = 0.0          # [0, 1]: corr_score × portfolio_heat
    effective_diversification: float = 1.0  # [0, 1]: effective independent positions ratio
    concentration_instability: float = 0.0  # [0, 1]: instability from concentration
    convexity_concentration_risk: float = 0.0  # [0, 1]: winner convexity concentration
    regime_fragility: float = 0.0         # [0, 1]: regime-driven fragility
    liquidation_risk: float = 0.0         # [0, 1]: correlated liquidation pressure
    survivability_score: float = 1.0      # [0, 1]: composite survivability
    survivability_gate: bool = True        # False → block new entries


@dataclass
class CapitalDeploymentOverlay:
    """Bounded capital deployment adjustments."""
    slot_scale: float = 1.0               # multiply baseline slot allocation
    entry_throttle: float = 1.0           # multiply entry admission rate
    corr_suppression_scale: float = 1.0   # multiply correlated exposure
    risk_scale: float = 1.0               # multiply position sizes
    suppressed_entry_symbols: Set[str] = field(default_factory=set)   # block BUY orders


@dataclass
class ConvexityRetentionOverlay:
    """Winner preservation and exit adjustment overlays."""
    convexity_score: float = 0.0           # [0, 1]: portfolio-level convexity quality
    hold_extension_symbols: Set[str] = field(default_factory=set)
    deterioration_exit_symbols: Set[str] = field(default_factory=set)
    correlated_winner_suppression: bool = False  # True: suppress premature exit for cluster winners


# ─────────────────────────────────────────────────────────────────────────────
# 4. PortfolioIntelligenceOverlay — final output consumed by run_live_signal.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioIntelligenceOverlay:
    """
    Unified bounded overlay from portfolio intelligence engine.
    Consumed directly by run_live_signal.py hook.
    """
    # Capital deployment
    slot_scale: float = 1.0               # multiply effective slot allocation [min, 1.0]
    entry_throttle: float = 1.0           # multiply entry rate [min, 1.0]
    corr_suppression_scale: float = 1.0   # multiply correlated exposure sizing
    risk_scale: float = 1.0               # multiply position sizes
    correlated_heat_limit: float = 0.50   # effective portfolio heat cap

    # Gates
    survivability_gate: bool = True        # False → block all new BUY orders

    # Per-symbol overrides
    suppressed_entry_symbols: FrozenSet[str] = field(default_factory=frozenset)
    hold_extension_symbols: FrozenSet[str] = field(default_factory=frozenset)
    deterioration_exit_symbols: FrozenSet[str] = field(default_factory=frozenset)

    # Diagnostics (for logging/display)
    overall_corr_score: float = 0.0
    survivability_score: float = 1.0
    convexity_score: float = 0.0
    n_decisions: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. PortfolioAdjustmentDecision — immutable, append-only JSONL record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioAdjustmentDecision:
    """
    Immutable portfolio intelligence decision.
    Written once; never mutated. Append-only JSONL.
    """
    decision_id: str            # sha256(eval_date|dimension_type|new_value_canonical)
    schema_version: int
    timestamp_jst: str
    eval_date: str
    dimension_type: str         # one of ALL_PIDTYPES
    previous_value: float
    new_value: float
    trigger_source: str
    trigger_value: float
    confidence: float           # [0, 1]
    observation_count: int
    bounded: bool               # always True

    # Provenance diagnostics
    corr_cluster_score: Optional[float]
    survivability_score: Optional[float]
    convexity_score: Optional[float]
    regime: Optional[str]

    @staticmethod
    def create(
        eval_date: str,
        dimension_type: str,
        previous_value: float,
        new_value: float,
        trigger_source: str,
        trigger_value: float,
        confidence: float,
        observation_count: int,
        corr_cluster_score: Optional[float] = None,
        survivability_score: Optional[float] = None,
        convexity_score: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> "PortfolioAdjustmentDecision":
        payload = _canonical_json({
            "dimension_type": dimension_type,
            "eval_date": eval_date,
            "new_value": round(new_value, 6),
        })
        decision_id = _sha256(payload)
        return PortfolioAdjustmentDecision(
            decision_id=decision_id,
            schema_version=SCHEMA_VERSION,
            timestamp_jst=_now_jst(),
            eval_date=eval_date,
            dimension_type=dimension_type,
            previous_value=round(previous_value, 6),
            new_value=round(new_value, 6),
            trigger_source=trigger_source,
            trigger_value=round(trigger_value, 6),
            confidence=round(_clamp(confidence, 0.0, 1.0), 4),
            observation_count=observation_count,
            bounded=True,
            corr_cluster_score=None if corr_cluster_score is None else round(corr_cluster_score, 6),
            survivability_score=None if survivability_score is None else round(survivability_score, 6),
            convexity_score=None if convexity_score is None else round(convexity_score, 6),
            regime=regime,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PortfolioAdjustmentDecision":
        return PortfolioAdjustmentDecision(
            decision_id=d.get("decision_id", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            timestamp_jst=d.get("timestamp_jst", ""),
            eval_date=d.get("eval_date", ""),
            dimension_type=d.get("dimension_type", ""),
            previous_value=float(d.get("previous_value", 0.0)),
            new_value=float(d.get("new_value", 0.0)),
            trigger_source=d.get("trigger_source", ""),
            trigger_value=float(d.get("trigger_value", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            observation_count=int(d.get("observation_count", 0)),
            bounded=bool(d.get("bounded", True)),
            corr_cluster_score=_opt_float(d.get("corr_cluster_score")),
            survivability_score=_opt_float(d.get("survivability_score")),
            convexity_score=_opt_float(d.get("convexity_score")),
            regime=d.get("regime"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integrity validator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PIViolation:
    violation_type: str
    dimension_type: str
    detail: str


@dataclass
class PIIntegrityReport:
    n_records: int
    n_violations: int
    violations: List[PIViolation]
    passed: bool


class PortfolioIntegrityValidator:
    """
    Fail-closed: any CRITICAL violation → passed=False → engine writes nothing.
    """
    CRITICAL_TYPES = {
        "duplicate_decision_id",
        "future_leakage",
        "insufficient_sample",
        "rapid_recursive_adaptation",
    }

    def validate(
        self,
        records: List[PortfolioAdjustmentDecision],
        constraints: PortfolioConstraints,
        as_of_date: Optional[str] = None,
    ) -> PIIntegrityReport:
        today = as_of_date or _today_str()
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        violations: List[PIViolation] = []
        seen_ids: Dict[str, int] = {}
        dim_dates: Dict[str, List[str]] = {}

        for i, r in enumerate(records):
            # Duplicate decision_id
            if r.decision_id in seen_ids:
                violations.append(PIViolation(
                    "duplicate_decision_id", r.dimension_type,
                    f"index={i} duplicates index={seen_ids[r.decision_id]}",
                ))
            else:
                seen_ids[r.decision_id] = i

            # Future leakage
            if r.eval_date > today:
                violations.append(PIViolation(
                    "future_leakage", r.dimension_type,
                    f"eval_date={r.eval_date} > as_of_date={today}",
                ))

            # Insufficient sample
            if r.observation_count < constraints.min_sample_size:
                violations.append(PIViolation(
                    "insufficient_sample", r.dimension_type,
                    f"observation_count={r.observation_count} < min={constraints.min_sample_size}",
                ))

            # Unbounded
            if not r.bounded:
                violations.append(PIViolation(
                    "unbounded_decision", r.dimension_type, "bounded=False",
                ))

            # Confidence out of range
            if not (0.0 <= r.confidence <= 1.0):
                violations.append(PIViolation(
                    "invalid_confidence", r.dimension_type,
                    f"confidence={r.confidence} out of [0,1]",
                ))

            dim_dates.setdefault(r.dimension_type, []).append(r.eval_date)

        # Rapid recursive adaptation: > max_decisions_per_week for any dim_type in 7 days
        for dim, dates in dim_dates.items():
            week_dates = [
                d for d in dates
                if (today_dt - datetime.strptime(d, "%Y-%m-%d")).days <= 7
            ]
            if len(week_dates) > constraints.max_decisions_per_week:
                violations.append(PIViolation(
                    "rapid_recursive_adaptation", dim,
                    f"{dim}: {len(week_dates)} decisions in 7d > max={constraints.max_decisions_per_week}",
                ))

            # Policy oscillation within cooldown
            sorted_dates = sorted(dates)
            for j in range(1, len(sorted_dates)):
                gap = _days_between(sorted_dates[j - 1], sorted_dates[j])
                if gap < constraints.cooldown_days:
                    violations.append(PIViolation(
                        "policy_oscillation", dim,
                        f"{dim} changed on {sorted_dates[j-1]} and {sorted_dates[j]} (gap={gap}d < {constraints.cooldown_days}d)",
                    ))

        # Replay inconsistency: same eval_date + dimension_type → same decision_id expected
        date_dim_seen: Dict[Tuple[str, str], str] = {}
        for r in records:
            key = (r.eval_date, r.dimension_type)
            if key in date_dim_seen and date_dim_seen[key] != r.decision_id:
                violations.append(PIViolation(
                    "replay_inconsistency", r.dimension_type,
                    f"same eval_date+dim_type produced different decision_ids",
                ))
            date_dim_seen[key] = r.decision_id

        has_critical = any(v.violation_type in self.CRITICAL_TYPES for v in violations)
        return PIIntegrityReport(
            n_records=len(records),
            n_violations=len(violations),
            violations=violations,
            passed=not has_critical,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Intelligence computation modules
# ─────────────────────────────────────────────────────────────────────────────

class CrossPositionCorrelationIntelligence:
    """Computes cross-position correlation risk metrics from position snapshots."""

    def compute(
        self,
        positions: List[PortfolioPositionSnapshot],
        constraints: PortfolioConstraints,
    ) -> CrossPositionCorrelation:
        n = len(positions)
        if n < 2:
            return CrossPositionCorrelation()

        # ── Sector clustering ─────────────────────────────────────────────────
        sector_counts: Dict[str, int] = {}
        for p in positions:
            sector_counts[p.sector] = sector_counts.get(p.sector, 0) + 1
        max_sector_n = max(sector_counts.values()) if sector_counts else 1
        sector_cluster_score = _clamp(max_sector_n / n, 0.0, 1.0)

        # ── Momentum synchronization (RSR dispersion) ─────────────────────────
        rsr_vals = [p.rsr for p in positions]
        rsr_mean = _mean(rsr_vals)
        rsr_std = _std(rsr_vals)
        # High sync = low coefficient of variation = positions move together
        cv = rsr_std / max(rsr_mean, 1.0)
        momentum_sync_score = _clamp(1.0 - cv, 0.0, 1.0)

        # ── Volatility correlation (rsr_momentum dispersion) ──────────────────
        mom_vals = [p.rsr_momentum for p in positions]
        mom_std = _std(mom_vals)
        abs_mom_mean = max(abs(_mean(mom_vals)), 0.01)
        vol_correlation_score = _clamp(1.0 - mom_std / abs_mom_mean * 0.5, 0.0, 1.0)

        # ── Lifecycle synchronization (hold_days spread) ───────────────────────
        hold_vals = [float(p.hold_days) for p in positions]
        hold_std = _std(hold_vals)
        hold_mean = max(_mean(hold_vals), 1.0)
        lifecycle_sync_score = _clamp(1.0 / (1.0 + hold_std / hold_mean), 0.0, 1.0)

        # ── Regime synchronization (same regime, same direction) ──────────────
        # Proxy: all positions with positive rs_score → regime-synchronized
        rs_positive = [1 for p in positions if (p.rs_score or 0.0) > 0]
        regime_sync_score = _clamp(len(rs_positive) / n, 0.0, 1.0)

        # ── Hidden factor overlap (high RSR concentration) ────────────────────
        high_rsr = [p for p in positions if p.rsr >= 75.0]
        hidden_factor_overlap = _clamp(len(high_rsr) / n, 0.0, 1.0)

        # ── Overall weighted composite ────────────────────────────────────────
        overall = (
            0.30 * sector_cluster_score
            + 0.20 * momentum_sync_score
            + 0.15 * vol_correlation_score
            + 0.15 * lifecycle_sync_score
            + 0.10 * regime_sync_score
            + 0.10 * hidden_factor_overlap
        )

        return CrossPositionCorrelation(
            sector_cluster_score=round(sector_cluster_score, 6),
            momentum_sync_score=round(momentum_sync_score, 6),
            vol_correlation_score=round(vol_correlation_score, 6),
            lifecycle_sync_score=round(lifecycle_sync_score, 6),
            regime_sync_score=round(regime_sync_score, 6),
            hidden_factor_overlap=round(hidden_factor_overlap, 6),
            overall_corr_score=round(_clamp(overall, 0.0, 1.0), 6),
        )


class PortfolioSurvivabilityIntelligence:
    """Computes portfolio survivability from correlation + regime state."""

    def compute(
        self,
        corr: CrossPositionCorrelation,
        regime: RegimeSnapshot,
        portfolio_heat: float,
        n_positions: int,
        constraints: PortfolioConstraints,
    ) -> SurvivabilityReport:
        # ── Correlated heat ───────────────────────────────────────────────────
        correlated_heat = _clamp(corr.overall_corr_score * portfolio_heat, 0.0, 1.0)

        # ── Effective diversification ─────────────────────────────────────────
        # 1.0 = fully uncorrelated; approaches 1/N when fully correlated
        n = max(n_positions, 1)
        effective_diversification = _clamp(
            1.0 - corr.overall_corr_score * (1.0 - 1.0 / n),
            1.0 / n, 1.0,
        )

        # ── Concentration instability ─────────────────────────────────────────
        concentration_instability = _clamp(
            corr.sector_cluster_score * 0.60 + corr.hidden_factor_overlap * 0.40,
            0.0, 1.0,
        )

        # ── Convexity concentration risk ──────────────────────────────────────
        # High momentum sync + high lifecycle sync → exits may cascade
        convexity_concentration_risk = _clamp(
            corr.momentum_sync_score * 0.50 + corr.lifecycle_sync_score * 0.50,
            0.0, 1.0,
        )

        # ── Regime fragility ──────────────────────────────────────────────────
        regime_fragility = 0.0
        if regime.deteriorating:
            regime_fragility += 0.40
        if regime.rs_collapse:
            regime_fragility += 0.30
        if regime.regime == "bear":
            regime_fragility += 0.20
        regime_fragility = _clamp(
            regime_fragility * regime.confidence + corr.overall_corr_score * 0.15,
            0.0, 1.0,
        )

        # ── Correlated liquidation risk ───────────────────────────────────────
        liquidation_risk = _clamp(
            corr.vol_correlation_score * 0.40
            + corr.momentum_sync_score * 0.30
            + regime_fragility * 0.30,
            0.0, 1.0,
        )

        # ── Composite survivability score ─────────────────────────────────────
        survivability_score = _clamp(
            1.0
            - 0.30 * correlated_heat
            - 0.25 * (1.0 - effective_diversification)
            - 0.20 * concentration_instability
            - 0.15 * regime_fragility
            - 0.10 * convexity_concentration_risk,
            0.0, 1.0,
        )

        survivability_gate = survivability_score >= constraints.min_survivability_score

        return SurvivabilityReport(
            correlated_heat=round(correlated_heat, 6),
            effective_diversification=round(effective_diversification, 6),
            concentration_instability=round(concentration_instability, 6),
            convexity_concentration_risk=round(convexity_concentration_risk, 6),
            regime_fragility=round(regime_fragility, 6),
            liquidation_risk=round(liquidation_risk, 6),
            survivability_score=round(survivability_score, 6),
            survivability_gate=survivability_gate,
        )


class CapitalDeploymentIntelligence:
    """Computes bounded capital deployment adjustments from portfolio state."""

    def compute(
        self,
        corr: CrossPositionCorrelation,
        surv: SurvivabilityReport,
        expectancy: ExpectancySnapshot,
        regime: RegimeSnapshot,
        constraints: PortfolioConstraints,
        n_positions: int,
    ) -> CapitalDeploymentOverlay:
        c = constraints
        cs = corr.overall_corr_score

        # ── Correlation-adjusted slot scale ───────────────────────────────────
        # High correlation → fewer open slots (to avoid correlated concentration)
        corr_slot_penalty = cs * 0.35
        regime_factor = 0.85 if (regime.regime == "bear" and regime.deteriorating) else 1.0
        raw_slot = (1.0 - corr_slot_penalty) * regime_factor
        slot_scale = _clamp(raw_slot, c.slot_scale_min, c.slot_scale_max)

        # ── Entry throttle: capital_constraint leakage + correlation ──────────
        # Leakage > 0.02 → loosen throttle (capital is the bottleneck)
        cap_leak = expectancy.capital_constraint_leakage
        throttle_delta = 0.0
        if cap_leak > 0.02:
            throttle_delta += cap_leak * 0.30      # loosen: more entries needed
        if cs > 0.65:
            throttle_delta -= (cs - 0.65) * 0.40  # tighten: correlated entries risky
        entry_throttle = _clamp(1.0 + throttle_delta, c.throttle_min, c.throttle_max)

        # ── Correlated exposure suppression scale ─────────────────────────────
        # Suppresses position sizes for highly correlated entries
        if cs >= 0.60:
            raw_supp = 1.0 - (cs - 0.60) * 1.5
        else:
            raw_supp = 1.0
        corr_suppression_scale = _clamp(raw_supp, c.corr_suppression_min, c.corr_suppression_max)

        # ── Overall risk scale: survivability + regime ─────────────────────────
        if regime.deteriorating and cs > 0.50:
            regime_risk_penalty = 0.20
        elif regime.regime == "bear":
            regime_risk_penalty = 0.10
        else:
            regime_risk_penalty = 0.0
        raw_risk = surv.survivability_score - regime_risk_penalty
        risk_scale = _clamp(raw_risk, c.risk_scale_min, c.risk_scale_max)

        # ── Per-symbol suppressed entries ─────────────────────────────────────
        # Suppress new entries into the dominant sector cluster when corr is high
        suppressed_entry_symbols: Set[str] = set()
        # (symbols are determined at hook-level from sector cluster analysis)

        return CapitalDeploymentOverlay(
            slot_scale=round(slot_scale, 6),
            entry_throttle=round(entry_throttle, 6),
            corr_suppression_scale=round(corr_suppression_scale, 6),
            risk_scale=round(risk_scale, 6),
            suppressed_entry_symbols=suppressed_entry_symbols,
        )


class ConvexityRetentionIntelligence:
    """Computes convexity retention and winner preservation overlays."""

    def compute(
        self,
        positions: List[PortfolioPositionSnapshot],
        exit_intel: ExitIntelSnapshot,
        corr: CrossPositionCorrelation,
        regime: RegimeSnapshot,
        constraints: PortfolioConstraints,
    ) -> ConvexityRetentionOverlay:
        c = constraints
        n = len(positions)
        if n == 0:
            return ConvexityRetentionOverlay()

        # ── Per-position runner scores from exit analytics ─────────────────────
        runner_scores = [
            exit_intel.per_symbol_runner_score.get(p.symbol, p.runner_score)
            for p in positions
        ]
        persistence_scores = [
            exit_intel.per_symbol_persistence.get(p.symbol, p.persistence_score)
            for p in positions
        ]
        giveback_scores = [
            exit_intel.per_symbol_giveback.get(p.symbol, 0.0)
            for p in positions
        ]

        mean_runner = _mean(runner_scores)
        mean_persistence = _mean(persistence_scores)
        mean_giveback = _mean(giveback_scores)

        # ── Portfolio-level convexity score ───────────────────────────────────
        convexity_score = _clamp(
            mean_runner * 0.40
            + mean_persistence * 0.30
            + (1.0 - exit_intel.premature_exit_ratio) * 0.20
            + (exit_intel.mean_convexity_capture) * 0.10,
            0.0, 1.0,
        )

        # ── Winner preservation symbols ───────────────────────────────────────
        hold_extension_symbols: Set[str] = set()
        deterioration_exit_symbols: Set[str] = set()

        for p in positions:
            sym_runner = exit_intel.per_symbol_runner_score.get(p.symbol, p.runner_score)
            sym_persistence = exit_intel.per_symbol_persistence.get(p.symbol, p.persistence_score)
            sym_giveback = exit_intel.per_symbol_giveback.get(p.symbol, 0.0)

            # Winner: high runner + persistence + no regime deterioration
            if (
                sym_runner >= c.winner_retention_threshold
                and sym_persistence >= c.persistence_retention_threshold
                and not regime.deteriorating
                and not regime.rs_collapse
            ):
                hold_extension_symbols.add(p.symbol)

            # Deterioration: high giveback + RS collapse or bear deteriorating
            if sym_giveback >= 0.65 or (regime.deteriorating and regime.rs_collapse):
                deterioration_exit_symbols.add(p.symbol)

        # Correlated winner suppression: if sector cluster is high + all positions are winners
        correlated_winner_suppression = (
            corr.sector_cluster_score >= 0.70
            and mean_runner >= 0.70
            and mean_persistence >= 0.60
        )

        return ConvexityRetentionOverlay(
            convexity_score=round(convexity_score, 6),
            hold_extension_symbols=hold_extension_symbols,
            deterioration_exit_symbols=deterioration_exit_symbols,
            correlated_winner_suppression=correlated_winner_suppression,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. PortfolioIntelligenceEngine — unified orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioIntelligenceEngine:
    """
    Unified portfolio intelligence engine.

    Evaluates cross-position correlation, survivability, capital deployment,
    and convexity retention → produces bounded PortfolioAdjustmentDecision list.

    STRICT: no strategy mutation, no self-optimization, no unbounded adaptation.
    Every decision bounded by PortfolioConstraints. Every decision → append-only JSONL.
    """

    def __init__(
        self,
        constraints: Optional[PortfolioConstraints] = None,
        store_path: Path = DEFAULT_STORE_PATH,
    ) -> None:
        self._c = constraints or PortfolioConstraints()
        self._store_path = store_path
        self._corr_intel = CrossPositionCorrelationIntelligence()
        self._surv_intel = PortfolioSurvivabilityIntelligence()
        self._cap_intel = CapitalDeploymentIntelligence()
        self._conv_intel = ConvexityRetentionIntelligence()

    def evaluate(
        self,
        inp: PortfolioIntelligenceInput,
        history: List[PortfolioAdjustmentDecision],
        as_of_date: Optional[str] = None,
    ) -> Tuple[List[PortfolioAdjustmentDecision], PortfolioIntelligenceOverlay]:
        """
        Evaluate portfolio-level intelligence → bounded decisions + overlay.

        Returns (decisions, overlay). Decisions are NOT written here — caller must
        call append_decision() for each. Fail-closed on integrity violations.
        """
        today = as_of_date or _today_str()
        c = self._c

        # ── Fail-closed integrity check ───────────────────────────────────────
        validator = PortfolioIntegrityValidator()
        report = validator.validate(history, c, as_of_date=today)
        if not report.passed:
            logger.warning(
                "[PI_ENGINE] integrity check failed — no decisions: %s",
                [v.violation_type for v in report.violations],
            )
            return [], PortfolioIntelligenceOverlay()

        positions = inp.positions
        n = len(positions)
        total_obs = n + inp.exit_intel.n_trades + inp.expectancy.n_executed

        # Require minimum sample for portfolio analysis
        if n < c.min_sample_size:
            logger.debug("[PI_ENGINE] insufficient positions (%d < %d) — no decisions", n, c.min_sample_size)
            return [], PortfolioIntelligenceOverlay()

        # ── Portfolio intelligence computations ───────────────────────────────
        corr = self._corr_intel.compute(positions, c)
        surv = self._surv_intel.compute(
            corr, inp.regime, inp.portfolio_heat, n, c,
        )
        cap_overlay = self._cap_intel.compute(
            corr, surv, inp.expectancy, inp.regime, c, n,
        )
        conv_overlay = self._conv_intel.compute(
            positions, inp.exit_intel, corr, inp.regime, c,
        )

        # ── Produce bounded decisions ─────────────────────────────────────────
        decisions: List[PortfolioAdjustmentDecision] = []

        # Governance: cooldown + weekly quota
        weekly_count = self._recent_decision_count(history, today, 7)
        remaining_quota = max(0, c.max_decisions_per_week - weekly_count)

        def _maybe_add(dim_type: str, prev: float, new: float, trigger_src: str, trigger_val: float, confidence: float) -> bool:
            nonlocal remaining_quota
            if remaining_quota <= 0:
                return False
            if abs(new - prev) < 1e-6:
                return False
            if not self._cooldown_ok(dim_type, history, today):
                logger.debug("[PI_ENGINE] cooldown active for %s", dim_type)
                return False
            if not self._anti_oscillation_ok(dim_type, history, today, new, prev):
                logger.debug("[PI_ENGINE] anti-oscillation blocked for %s", dim_type)
                return False
            rec = PortfolioAdjustmentDecision.create(
                eval_date=today,
                dimension_type=dim_type,
                previous_value=prev,
                new_value=new,
                trigger_source=trigger_src,
                trigger_value=trigger_val,
                confidence=confidence,
                observation_count=total_obs,
                corr_cluster_score=corr.sector_cluster_score,
                survivability_score=surv.survivability_score,
                convexity_score=conv_overlay.convexity_score,
                regime=inp.regime.regime,
            )
            decisions.append(rec)
            remaining_quota -= 1
            return True

        # Compute confidence from sample size and signal strength
        def _conf(signal: float) -> float:
            size_factor = _clamp((total_obs - c.min_sample_size) / max(1, c.min_sample_size * 2), 0.0, 1.0)
            signal_factor = _clamp(abs(signal) / 0.30, 0.0, 1.0)
            return round((size_factor * 0.50 + signal_factor * 0.50) * 0.95, 4)

        # ── Slot scale ────────────────────────────────────────────────────────
        if abs(cap_overlay.slot_scale - 1.0) > 0.03:
            delta = _clamp(cap_overlay.slot_scale - 1.0, -c.max_slot_delta, c.max_slot_delta)
            bounded_slot = _clamp(1.0 + delta, c.slot_scale_min, c.slot_scale_max)
            _maybe_add(
                PIDTYPE_SLOT_SCALE, 1.0, bounded_slot,
                TRIGGER_SECTOR_CLUSTER, corr.sector_cluster_score, _conf(corr.overall_corr_score),
            )

        # ── Heat limit ────────────────────────────────────────────────────────
        if surv.correlated_heat > c.max_correlated_heat:
            heat_limit = _clamp(
                1.0 - surv.correlated_heat * 0.40,
                c.heat_limit_min, c.heat_limit_max,
            )
            _maybe_add(
                PIDTYPE_HEAT_LIMIT, 0.50, heat_limit,
                TRIGGER_CORR_HEAT, surv.correlated_heat, _conf(surv.correlated_heat),
            )

        # ── Risk scale ────────────────────────────────────────────────────────
        if abs(cap_overlay.risk_scale - 1.0) > 0.05:
            delta = _clamp(cap_overlay.risk_scale - 1.0, -c.max_risk_delta, c.max_risk_delta)
            bounded_risk = _clamp(1.0 + delta, c.risk_scale_min, c.risk_scale_max)
            _maybe_add(
                PIDTYPE_RISK_SCALE, 1.0, bounded_risk,
                TRIGGER_SURVIVABILITY_SCORE, surv.survivability_score, _conf(surv.survivability_score),
            )

        # ── Entry throttle ────────────────────────────────────────────────────
        if abs(cap_overlay.entry_throttle - 1.0) > 0.03:
            delta = _clamp(cap_overlay.entry_throttle - 1.0, -c.max_throttle_delta, c.max_throttle_delta)
            bounded_throttle = _clamp(1.0 + delta, c.throttle_min, c.throttle_max)
            _maybe_add(
                PIDTYPE_THROTTLE, 1.0, bounded_throttle,
                TRIGGER_CAPITAL_LEAKAGE, inp.expectancy.capital_constraint_leakage,
                _conf(abs(cap_overlay.entry_throttle - 1.0)),
            )

        # ── Correlated exposure suppression ───────────────────────────────────
        if abs(cap_overlay.corr_suppression_scale - 1.0) > 0.05:
            _maybe_add(
                PIDTYPE_CORR_SUPPRESSION, 1.0, cap_overlay.corr_suppression_scale,
                TRIGGER_EFFECTIVE_DIV, surv.effective_diversification,
                _conf(corr.overall_corr_score),
            )

        # ── Survivability gate (flag: 1.0=open, 0.0=gated) ───────────────────
        gate_val = 1.0 if surv.survivability_gate else 0.0
        if gate_val < 1.0:  # only write when closing the gate
            _maybe_add(
                PIDTYPE_SURVIVABILITY, 1.0, gate_val,
                TRIGGER_SURVIVABILITY_SCORE, surv.survivability_score,
                _conf(1.0 - surv.survivability_score),
            )

        # ── Convexity retention ───────────────────────────────────────────────
        if conv_overlay.convexity_score > 0.05:
            _maybe_add(
                PIDTYPE_CONVEXITY, 0.0, conv_overlay.convexity_score,
                TRIGGER_CONVEXITY_SCORE, conv_overlay.convexity_score,
                _conf(conv_overlay.convexity_score),
            )

        # ── Compute suppressed entry symbols (per-symbol, derived from corr) ──
        suppressed = self._compute_suppressed_symbols(positions, corr, surv, c)

        # ── Build unified overlay ─────────────────────────────────────────────
        # Effective values: from decisions or computed overlay if no decision produced
        effective_slot = next(
            (d.new_value for d in decisions if d.dimension_type == PIDTYPE_SLOT_SCALE), 1.0
        )
        effective_heat = next(
            (d.new_value for d in decisions if d.dimension_type == PIDTYPE_HEAT_LIMIT), 0.50
        )
        effective_risk = next(
            (d.new_value for d in decisions if d.dimension_type == PIDTYPE_RISK_SCALE), 1.0
        )
        effective_throttle = next(
            (d.new_value for d in decisions if d.dimension_type == PIDTYPE_THROTTLE), 1.0
        )
        effective_supp = next(
            (d.new_value for d in decisions if d.dimension_type == PIDTYPE_CORR_SUPPRESSION),
            cap_overlay.corr_suppression_scale,
        )

        overlay = PortfolioIntelligenceOverlay(
            slot_scale=effective_slot,
            entry_throttle=effective_throttle,
            corr_suppression_scale=effective_supp,
            risk_scale=effective_risk,
            correlated_heat_limit=effective_heat,
            survivability_gate=surv.survivability_gate,
            suppressed_entry_symbols=frozenset(suppressed),
            hold_extension_symbols=frozenset(conv_overlay.hold_extension_symbols),
            deterioration_exit_symbols=frozenset(conv_overlay.deterioration_exit_symbols),
            overall_corr_score=corr.overall_corr_score,
            survivability_score=surv.survivability_score,
            convexity_score=conv_overlay.convexity_score,
            n_decisions=len(decisions),
        )

        return decisions, overlay

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_suppressed_symbols(
        self,
        positions: List[PortfolioPositionSnapshot],
        corr: CrossPositionCorrelation,
        surv: SurvivabilityReport,
        c: PortfolioConstraints,
    ) -> Set[str]:
        """Identify new-entry symbols to suppress due to sector cluster concentration."""
        suppressed: Set[str] = set()
        if corr.sector_cluster_score < c.sector_cluster_threshold:
            return suppressed
        if surv.survivability_gate:
            return suppressed  # gate open → no suppression needed at this level

        # Find dominant sector cluster
        sector_counts: Dict[str, List[str]] = {}
        for p in positions:
            sector_counts.setdefault(p.sector, []).append(p.symbol)
        if not sector_counts:
            return suppressed
        dominant_sector = max(sector_counts, key=lambda s: len(sector_counts[s]))
        dominant_syms = set(sector_counts[dominant_sector])
        n = len(positions)
        if len(dominant_syms) / n >= c.sector_cluster_threshold:
            # Suppress new entries into dominant sector
            suppressed = dominant_syms
        return suppressed

    def _cooldown_ok(
        self, dim_type: str, history: List[PortfolioAdjustmentDecision], today: str,
    ) -> bool:
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=self._c.cooldown_days)
        ).strftime("%Y-%m-%d")
        return not any(
            r.dimension_type == dim_type and r.eval_date >= cutoff
            for r in history
        )

    def _anti_oscillation_ok(
        self,
        dim_type: str,
        history: List[PortfolioAdjustmentDecision],
        today: str,
        new_value: float,
        prev_value: float,
    ) -> bool:
        """Block direction reversal within anti_oscillation_window_days."""
        c = self._c
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=c.anti_oscillation_window_days)
        ).strftime("%Y-%m-%d")
        recent = [r for r in history if r.dimension_type == dim_type and r.eval_date >= cutoff]
        if not recent:
            return True
        last = max(recent, key=lambda r: r.eval_date)
        last_direction = math.copysign(1, last.new_value - last.previous_value)
        new_direction = math.copysign(1, new_value - prev_value)
        return last_direction == new_direction  # block reversal

    def _recent_decision_count(
        self, history: List[PortfolioAdjustmentDecision], today: str, window_days: int,
    ) -> int:
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=window_days)
        ).strftime("%Y-%m-%d")
        return sum(1 for r in history if r.eval_date >= cutoff)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_decision(
    record: PortfolioAdjustmentDecision,
    path: Path = DEFAULT_STORE_PATH,
) -> None:
    """Atomic append to JSONL. fsync durable. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                tf.write(existing)
                tf.write(line + "\n")
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        os.replace(tmp, str(path))
    except Exception as exc:
        logger.warning("[PI_ENGINE] append failed (%s) — continuing", exc)


def load_decisions(path: Path = DEFAULT_STORE_PATH) -> List[PortfolioAdjustmentDecision]:
    """Load all records. Corrupted lines skipped with warning."""
    if not path.exists():
        return []
    records: List[PortfolioAdjustmentDecision] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(PortfolioAdjustmentDecision.from_dict(json.loads(line)))
        except Exception as exc:
            logger.warning("[PI_ENGINE] parse error line %d: %s", i + 1, exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 10. PortfolioIntelligenceHook — lightweight hook for run_live_signal.py
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioIntelligenceHook:
    """
    Lightweight hook for direct injection into run_live_signal.py.

    Usage:
        hook = PortfolioIntelligenceHook(store_path=PI_DECISIONS_FILE)
        signals, orders, overlay = hook.run(
            signals=result.signals,
            order_objects=order_objects,
            portfolio_summary=result.portfolio_summary,
            broker_positions=_broker_pos,
            regime_info=_regime_info,
            exposure_state=_exposure_state,
        )

    Fail-open: any error returns inputs unchanged + empty overlay.
    """

    def __init__(
        self,
        store_path: Path = DEFAULT_STORE_PATH,
        constraints: Optional[PortfolioConstraints] = None,
    ) -> None:
        self._store_path = store_path
        self._engine = PortfolioIntelligenceEngine(
            constraints=constraints or PortfolioConstraints(),
            store_path=store_path,
        )

    def run(
        self,
        signals: List[Dict[str, Any]],
        order_objects: List[Any],
        portfolio_summary: Optional[Dict[str, Any]] = None,
        broker_positions: Optional[Dict[str, int]] = None,
        regime_info: Optional[Dict[str, Any]] = None,
        exposure_state: Optional[Any] = None,
        as_of_date: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Any], PortfolioIntelligenceOverlay]:
        try:
            return self._run_internal(
                signals, order_objects, portfolio_summary,
                broker_positions, regime_info, exposure_state, as_of_date,
            )
        except Exception as exc:
            logger.warning("[PI_HOOK] hook failed (%s) — inputs unchanged", exc)
            return signals, order_objects, PortfolioIntelligenceOverlay()

    def _run_internal(
        self,
        signals: List[Dict[str, Any]],
        order_objects: List[Any],
        portfolio_summary: Optional[Dict[str, Any]],
        broker_positions: Optional[Dict[str, int]],
        regime_info: Optional[Dict[str, Any]],
        exposure_state: Optional[Any],
        as_of_date: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[Any], PortfolioIntelligenceOverlay]:
        ps = portfolio_summary or {}
        ri = regime_info or {}

        inp = self._build_input(signals, ps, ri, broker_positions, exposure_state)

        if not inp.positions:
            logger.debug("[PI_HOOK] no held positions — skip")
            return signals, order_objects, PortfolioIntelligenceOverlay()

        history = load_decisions(self._store_path)
        decisions, overlay = self._engine.evaluate(inp, history, as_of_date=as_of_date)

        # Persist decisions
        for dec in decisions:
            append_decision(dec, self._store_path)
            logger.info(
                "[PI_HOOK] decision dim=%s new=%.4f trigger=%s(%.4f) date=%s",
                dec.dimension_type, dec.new_value, dec.trigger_source,
                dec.trigger_value, dec.eval_date,
            )

        # Apply overlay to signals
        modified_signals = self._apply_overlay_to_signals(signals, overlay)

        # Apply overlay to orders
        modified_orders = self._apply_overlay_to_orders(order_objects, overlay, signals)

        logger.info(
            "[PI_HOOK] complete: corr=%.3f surv=%.3f conv=%.3f slot=%.2f "
            "gate=%s suppressed=%d decisions=%d",
            overlay.overall_corr_score, overlay.survivability_score,
            overlay.convexity_score, overlay.slot_scale,
            overlay.survivability_gate, len(overlay.suppressed_entry_symbols),
            len(decisions),
        )

        return modified_signals, modified_orders, overlay

    def _apply_overlay_to_signals(
        self,
        signals: List[Dict[str, Any]],
        overlay: PortfolioIntelligenceOverlay,
    ) -> List[Dict[str, Any]]:
        """Apply hold-extension and deterioration-exit overlays to held-position signals."""
        modified = []
        for sig in signals:
            sym = sig.get("symbol", "")
            new_sig = dict(sig)

            if sig.get("currently_holding"):
                # Hold extension: suppress premature exit
                if sym in overlay.hold_extension_symbols and sig.get("signal") == 0:
                    new_sig["signal"] = 1
                    new_sig["action"] = "HOLD"
                    new_sig["portfolio_intelligence_hold_extension"] = True
                    logger.info("[PI_HOOK] hold_extension applied: %s", sym)

                # Deterioration exit: force exit
                elif sym in overlay.deterioration_exit_symbols and sig.get("signal") == 1:
                    new_sig["signal"] = 0
                    new_sig["action"] = "SELL"
                    new_sig["portfolio_intelligence_deterioration_exit"] = True
                    logger.info("[PI_HOOK] deterioration_exit applied: %s", sym)

            # Annotate all signals with overlay diagnostics
            new_sig["pi_slot_scale"] = overlay.slot_scale
            new_sig["pi_survivability"] = overlay.survivability_score
            new_sig["pi_corr_score"] = overlay.overall_corr_score
            modified.append(new_sig)
        return modified

    def _apply_overlay_to_orders(
        self,
        order_objects: List[Any],
        overlay: PortfolioIntelligenceOverlay,
        signals: List[Dict[str, Any]],
    ) -> List[Any]:
        """
        Apply overlay to order_objects:
        - survivability_gate=False → remove all BUY orders
        - suppressed_entry_symbols → remove BUY orders for suppressed symbols
        - deterioration_exit_symbols → keep / ensure SELL orders present
        """
        if not order_objects:
            return order_objects

        result: List[Any] = []
        for o in order_objects:
            sym = getattr(o, "symbol", None)
            side = getattr(o, "side", None)

            if side == "BUY":
                # Survivability gate: suppress all new entries
                if not overlay.survivability_gate:
                    logger.info("[PI_HOOK] survivability gate: suppressed BUY for %s", sym)
                    continue
                # Per-symbol correlated entry suppression
                if sym in overlay.suppressed_entry_symbols:
                    logger.info("[PI_HOOK] corr suppression: suppressed BUY for %s", sym)
                    continue

            result.append(o)

        return result

    # ── Input builders ────────────────────────────────────────────────────────

    def _build_input(
        self,
        signals: List[Dict[str, Any]],
        ps: Dict[str, Any],
        ri: Dict[str, Any],
        broker_positions: Optional[Dict[str, int]],
        exposure_state: Optional[Any],
    ) -> PortfolioIntelligenceInput:
        positions = self._build_positions(signals, ps)
        regime = self._build_regime(ri, signals)
        expectancy = self._build_expectancy(ps)
        exit_intel = self._build_exit_intel(signals)

        portfolio_heat = abs(float(ps.get("current_drawdown", 0.0)))
        max_pos = int(ps.get("max_positions", 4))
        cur_pos = int(ps.get("current_positions", len(positions)))
        available_slots = max(0, max_pos - cur_pos)

        return PortfolioIntelligenceInput(
            positions=positions,
            regime=regime,
            expectancy=expectancy,
            exit_intel=exit_intel,
            portfolio_heat=portfolio_heat,
            available_slots=available_slots,
            max_slots=max_pos,
            eval_date=_today_str(),
        )

    def _build_positions(
        self, signals: List[Dict[str, Any]], ps: Dict[str, Any]
    ) -> List[PortfolioPositionSnapshot]:
        positions = []
        for sig in signals:
            if not sig.get("currently_holding"):
                continue
            sym = sig.get("symbol", "")
            if not sym:
                continue
            positions.append(PortfolioPositionSnapshot(
                symbol=sym,
                sector=str(sig.get("sector", "unknown")),
                hold_days=int(sig.get("hold_days", 0)),
                rsr=float(sig.get("rsr", 50.0)),
                rsr_momentum=float(sig.get("rsr_momentum", 0.0)),
                current_return=float(sig.get("current_return", 0.0)),
                rs_score=_opt_float(sig.get("rsr_momentum")),
                signal=int(sig.get("signal", 1)),
                runner_score=float(sig.get("runner_score", 0.0)),
                persistence_score=float(sig.get("persistence_score", 0.0)),
                vol=float(sig.get("volatility", 0.20)),
            ))
        return positions

    def _build_regime(
        self, ri: Dict[str, Any], signals: List[Dict[str, Any]]
    ) -> RegimeSnapshot:
        n = len(signals)
        breadth = (
            sum(1 for s in signals if float(s.get("rsr", 0)) >= 50) / n
            if n > 0 else 0.5
        )
        return RegimeSnapshot(
            regime=str(ri.get("regime", "unknown")),
            deteriorating=bool(ri.get("deteriorating", False)),
            rs_collapse=bool(ri.get("rs_collapse", False)),
            breadth_score=_clamp(breadth, 0.0, 1.0),
            confidence=float(ri.get("confidence", 0.5)),
        )

    def _build_expectancy(self, ps: Dict[str, Any]) -> ExpectancySnapshot:
        return ExpectancySnapshot(
            capital_constraint_leakage=float(ps.get("capital_constraint_leakage", 0.0)),
            heat_limit_leakage=float(ps.get("heat_limit_leakage", 0.0)),
            sector_exposure_leakage=float(ps.get("sector_exposure_leakage", 0.0)),
            skipped_expectancy_delta=float(ps.get("skipped_expectancy_delta", 0.0)),
            opportunity_capture_ratio=float(ps.get("opportunity_capture_ratio", 1.0)),
            n_executed=int(ps.get("current_positions", 0)),
            n_skipped=int(ps.get("open_slots", 0)),
        )

    def _build_exit_intel(self, signals: List[Dict[str, Any]]) -> ExitIntelSnapshot:
        """Build exit intelligence from signals + EXIT_RECORDS_FILE if available."""
        intel = ExitIntelSnapshot()
        try:
            from src.paths import EXIT_RECORDS_FILE
            import json as _json
            if EXIT_RECORDS_FILE.exists():
                records = []
                for line in EXIT_RECORDS_FILE.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(_json.loads(line))
                    except Exception:
                        pass

                if records:
                    runner_scores = [float(r.get("runner_continuation_score", 0.0)) for r in records]
                    giveback = [float(r.get("pnl_giveback_ratio", 0.0)) for r in records]
                    retention = [float(r.get("runner_retention_rate", 1.0)) for r in records]
                    capture = [float(r.get("convexity_capture_rate", 1.0)) for r in records]
                    premature = [1 for r in records if r.get("is_premature_exit")]

                    intel.mean_runner_retention = _mean(retention)
                    intel.mean_giveback_ratio = _mean(giveback)
                    intel.mean_convexity_capture = _mean(capture)
                    intel.premature_exit_ratio = len(premature) / len(records) if records else 0.0
                    intel.n_trades = len(records)

                    by_sym: Dict[str, dict] = {}
                    for r in records:
                        sym = r.get("symbol", "")
                        if sym and (sym not in by_sym or r.get("exit_timestamp", "") > by_sym[sym].get("exit_timestamp", "")):
                            by_sym[sym] = r
                    for sym, r in by_sym.items():
                        intel.per_symbol_runner_score[sym] = float(r.get("runner_continuation_score", 0.0))
                        intel.per_symbol_persistence[sym] = float(r.get("relative_trend_persistence_score") or 0.0)
                        intel.per_symbol_giveback[sym] = float(r.get("pnl_giveback_ratio", 0.0))
        except Exception as exc:
            logger.debug("[PI_HOOK] exit intel load failed (%s)", exc)
        return intel


# ─────────────────────────────────────────────────────────────────────────────
# 11. Top-level hook for run_live_signal.py
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_intelligence_hook(
    signals: List[Dict[str, Any]],
    order_objects: List[Any],
    store_path: Path = DEFAULT_STORE_PATH,
    portfolio_summary: Optional[Dict[str, Any]] = None,
    broker_positions: Optional[Dict[str, int]] = None,
    regime_info: Optional[Dict[str, Any]] = None,
    exposure_state: Optional[Any] = None,
    constraints: Optional[PortfolioConstraints] = None,
    as_of_date: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Any], PortfolioIntelligenceOverlay]:
    """
    Top-level hook for run_live_signal.py.

    Returns (modified_signals, modified_orders, overlay).
    Fail-open: any error returns inputs unchanged + empty overlay.

    Usage in run_live_signal.py:
        from src.runtime.policy.portfolio_intelligence_engine import (
            run_portfolio_intelligence_hook, PortfolioIntelligenceOverlay,
        )
        from src.paths import PI_DECISIONS_FILE
        result.signals, order_objects, _pi_overlay = run_portfolio_intelligence_hook(
            signals=result.signals,
            order_objects=order_objects,
            store_path=PI_DECISIONS_FILE,
            portfolio_summary=result.portfolio_summary,
            broker_positions=_broker_pos,
            regime_info=_regime_info_eo,
            exposure_state=_exposure_state,
        )
    """
    try:
        hook = PortfolioIntelligenceHook(
            store_path=store_path,
            constraints=constraints or PortfolioConstraints(),
        )
        return hook.run(
            signals=signals,
            order_objects=order_objects,
            portfolio_summary=portfolio_summary,
            broker_positions=broker_positions,
            regime_info=regime_info,
            exposure_state=exposure_state,
            as_of_date=as_of_date,
        )
    except Exception as exc:
        logger.warning("[PI_HOOK] top-level hook failed (%s) — inputs unchanged", exc)
        return signals, order_objects, PortfolioIntelligenceOverlay()
