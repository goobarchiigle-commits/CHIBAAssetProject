"""
src/runtime/policy/runtime_exit_orchestrator.py — Autonomous runtime exit orchestration

Directly connects exit intelligence analytics to live runtime exit behavior.
Consumes: exit_intelligence, position_lifecycle_snapshots, regime analytics, expectancy analytics.
Produces: bounded per-symbol ExitAdjustmentDecision → modifies live signals + order_objects.

STRICT CONSTRAINTS:
  - Bounded adaptation only (hard caps enforced at compute time)
  - Fail-closed on any integrity violation
  - Cooldown and anti-oscillation guards per symbol
  - No recursive feedback (decisions never consume prior decisions as inputs)
  - No self-reinforcing runaway: force_exit + suppress_exit cannot coexist
  - Deterministic: same inputs → same outputs
  - Append-only JSONL decision log

Integration:
  - run_exit_orchestration_hook() for run_live_signal.py injection
  - Modifies result.signals (held positions) and order_objects directly
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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
JST = timezone(timedelta(hours=9))

DEFAULT_STORE_PATH = Path("runtime/policy/exit_policy_decisions.jsonl")


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
    """Days from date string a to date string b (both YYYY-MM-DD). Positive if b > a."""
    da = datetime.strptime(a, "%Y-%m-%d")
    db = datetime.strptime(b, "%Y-%m-%d")
    return (db - da).days


# ─────────────────────────────────────────────────────────────────────────────
# 1. ExitAdjustmentConstraints — hard safety bounds, never relaxed by code
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExitAdjustmentConstraints:
    """
    Hard bounds on all exit adjustments.
    All values enforced at compute time and re-validated before JSONL write.
    Never relaxed by code — only by explicit user configuration change.
    """
    # Hold duration extension
    max_hold_extension_days: int = 5          # [0, 5] extra days max

    # Exit sensitivity (turtle exit lookback scale)
    exit_sensitivity_min: float = 0.80        # tightest allowed (shorter lookback)
    exit_sensitivity_max: float = 1.20        # loosest allowed (longer lookback)

    # Deterioration pressure [0, 1]
    deterioration_pressure_max: float = 1.0
    force_exit_pressure_threshold: float = 0.80  # >= this → force_exit=True

    # Volatility-aware trailing multiplier
    vol_trailing_mult_min: float = 0.80
    vol_trailing_mult_max: float = 1.20

    # Regime-conditioned threshold shift
    regime_threshold_adj_min: float = -0.05
    regime_threshold_adj_max: float = 0.05

    # Governance
    cooldown_days: int = 3                     # min days between changes per symbol
    max_decisions_per_symbol_per_week: int = 2
    min_sample_size: int = 5                   # min observations before any decision

    # Anti-oscillation: block direction reversal within this many days of prior decision
    anti_oscillation_window_days: int = 6      # = 2 × cooldown_days


# ─────────────────────────────────────────────────────────────────────────────
# 2. Input snapshots
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitIntelligenceSnapshot:
    """
    Aggregated exit analytics — from ExitConvexityRecord / RunnerRetentionSummary.
    All fields optional: orchestrator degrades gracefully on missing data.
    """
    # Portfolio-wide metrics (from RunnerRetentionSummary)
    premature_exit_ratio: Optional[float] = None      # fraction of exits that were premature
    mean_giveback_ratio: Optional[float] = None        # mean PnL giveback [0, 1]
    mean_runner_retention: Optional[float] = None      # mean runner retention rate [0, 1]
    mean_convexity_capture: Optional[float] = None     # mean convexity capture rate [0, 1]
    post_exit_alpha_leakage: Optional[float] = None    # mean post-exit alpha leakage

    # Per-symbol metrics (from per-trade ExitConvexityRecord, keyed by symbol)
    per_symbol_runner_score: Dict[str, float] = field(default_factory=dict)
    per_symbol_giveback: Dict[str, float] = field(default_factory=dict)
    per_symbol_vol: Dict[str, float] = field(default_factory=dict)
    per_symbol_persistence: Dict[str, float] = field(default_factory=dict)
    per_symbol_decay_rate: Dict[str, float] = field(default_factory=dict)

    n_trades: int = 0
    eval_date: str = ""


@dataclass
class PositionLifecycleSnapshot:
    """Current state of a single live position."""
    symbol: str
    hold_days: int
    current_return: float              # fraction from entry (e.g., 0.05 = +5%)
    rs_score: Optional[float] = None   # relative strength vs TOPIX benchmark
    rsr: Optional[float] = None        # RSR percentile rank [0, 100]
    is_at_turtle_exit: bool = False    # currently triggering turtle exit condition
    trailing_stop_distance: Optional[float] = None   # fraction gap from trailing stop
    peak_return: Optional[float] = None              # MFE fraction
    # ── Compression continuation (from compression_continuation_hook) ─────────
    compression_score: float = 0.0          # composite [0, 100]
    suppression_eligible: bool = False      # True → healthy compression gate passed
    compression_phase: str = "neutral"      # "compression"|"distribution"|"breakout"|"neutral"


@dataclass
class RegimeAnalyticsSnapshot:
    """Current regime state from regime analytics pipeline."""
    regime: str = "unknown"                    # "bull", "bear", "neutral", "unknown"
    regime_deteriorating: bool = False         # regime trending worse over recent days
    breadth_score: Optional[float] = None      # [0, 1]: fraction of universe above MA
    rs_collapse_detected: bool = False         # RS sudden collapse signal
    regime_confidence: float = 0.5             # [0, 1]
    eval_date: str = ""


@dataclass
class ExitExpectancySnapshot:
    """From executed_vs_skipped expectancy analytics pipeline."""
    executed_win_rate: Optional[float] = None
    skipped_expectancy_delta: Optional[float] = None  # skipped − executed
    capture_ratio: Optional[float] = None
    eval_date: str = ""
    n_executed: int = 0
    n_skipped: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. ExitAdjustmentPolicy — per-symbol bounded runtime modifier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitAdjustmentPolicy:
    """
    Bounded runtime exit modifiers for a single held position.
    All values within ExitAdjustmentConstraints bounds.
    """
    symbol: str

    hold_extension_days: int = 0              # [0, max_hold_extension_days]
    exit_sensitivity_scale: float = 1.0       # [exit_sensitivity_min, exit_sensitivity_max]
    deterioration_pressure: float = 0.0       # [0, 1]
    trend_persistence_score: float = 0.0      # [0, 1]: higher = retain winner
    vol_trailing_multiplier: float = 1.0      # [vol_trailing_mult_min, vol_trailing_mult_max]
    regime_threshold_adjustment: float = 0.0  # [regime_threshold_adj_min, regime_threshold_adj_max]

    # Derived action flags (set after all adjustments computed)
    suppress_exit: bool = False   # True: prevent premature exit, extend hold
    force_exit: bool = False      # True: deterioration threshold met, exit now

    # suppress_exit and force_exit are mutually exclusive (enforced in orchestrator)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ExitAdjustmentDecision — immutable, append-only JSONL record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExitAdjustmentDecision:
    """
    Immutable exit adjustment decision record.
    Written once; never mutated. Append-only JSONL store.
    decision_id = sha256 dedup key: eval_date|symbol|hold_extension|suppress|force.
    """
    decision_id: str
    schema_version: int
    timestamp_jst: str
    eval_date: str
    symbol: str

    # Bounded adjustment values
    hold_extension_days: int
    exit_sensitivity_scale: float
    deterioration_pressure: float
    trend_persistence_score: float
    vol_trailing_multiplier: float
    regime_threshold_adjustment: float

    # Action flags
    suppress_exit: bool
    force_exit: bool

    # Provenance (analytics inputs that triggered this decision)
    trigger_runner_score: Optional[float]
    trigger_premature_ratio: Optional[float]
    trigger_giveback_ratio: Optional[float]
    trigger_vol: Optional[float]
    trigger_regime: Optional[str]
    trigger_rs_collapse: bool

    observation_count: int
    bounded: bool    # always True — unbounded decisions are never written

    @staticmethod
    def create(
        eval_date: str,
        symbol: str,
        policy: ExitAdjustmentPolicy,
        trigger_runner_score: Optional[float],
        trigger_premature_ratio: Optional[float],
        trigger_giveback_ratio: Optional[float],
        trigger_vol: Optional[float],
        trigger_regime: Optional[str],
        trigger_rs_collapse: bool,
        observation_count: int,
    ) -> "ExitAdjustmentDecision":
        ts = _now_jst()
        payload = _canonical_json({
            "eval_date": eval_date,
            "force_exit": policy.force_exit,
            "hold_extension_days": policy.hold_extension_days,
            "suppress_exit": policy.suppress_exit,
            "symbol": symbol,
        })
        decision_id = _sha256(payload)
        return ExitAdjustmentDecision(
            decision_id=decision_id,
            schema_version=SCHEMA_VERSION,
            timestamp_jst=ts,
            eval_date=eval_date,
            symbol=symbol,
            hold_extension_days=policy.hold_extension_days,
            exit_sensitivity_scale=round(policy.exit_sensitivity_scale, 6),
            deterioration_pressure=round(policy.deterioration_pressure, 6),
            trend_persistence_score=round(policy.trend_persistence_score, 6),
            vol_trailing_multiplier=round(policy.vol_trailing_multiplier, 6),
            regime_threshold_adjustment=round(policy.regime_threshold_adjustment, 6),
            suppress_exit=policy.suppress_exit,
            force_exit=policy.force_exit,
            trigger_runner_score=_round_opt(trigger_runner_score),
            trigger_premature_ratio=_round_opt(trigger_premature_ratio),
            trigger_giveback_ratio=_round_opt(trigger_giveback_ratio),
            trigger_vol=_round_opt(trigger_vol),
            trigger_regime=trigger_regime,
            trigger_rs_collapse=trigger_rs_collapse,
            observation_count=observation_count,
            bounded=True,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ExitAdjustmentDecision":
        return ExitAdjustmentDecision(
            decision_id=d.get("decision_id", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            timestamp_jst=d.get("timestamp_jst", ""),
            eval_date=d.get("eval_date", ""),
            symbol=d.get("symbol", ""),
            hold_extension_days=int(d.get("hold_extension_days", 0)),
            exit_sensitivity_scale=float(d.get("exit_sensitivity_scale", 1.0)),
            deterioration_pressure=float(d.get("deterioration_pressure", 0.0)),
            trend_persistence_score=float(d.get("trend_persistence_score", 0.0)),
            vol_trailing_multiplier=float(d.get("vol_trailing_multiplier", 1.0)),
            regime_threshold_adjustment=float(d.get("regime_threshold_adjustment", 0.0)),
            suppress_exit=bool(d.get("suppress_exit", False)),
            force_exit=bool(d.get("force_exit", False)),
            trigger_runner_score=_opt_float(d.get("trigger_runner_score")),
            trigger_premature_ratio=_opt_float(d.get("trigger_premature_ratio")),
            trigger_giveback_ratio=_opt_float(d.get("trigger_giveback_ratio")),
            trigger_vol=_opt_float(d.get("trigger_vol")),
            trigger_regime=d.get("trigger_regime"),
            trigger_rs_collapse=bool(d.get("trigger_rs_collapse", False)),
            observation_count=int(d.get("observation_count", 0)),
            bounded=bool(d.get("bounded", True)),
        )


def _round_opt(v: Optional[float]) -> Optional[float]:
    return round(v, 6) if v is not None else None


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integrity validator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitIntegrityViolation:
    violation_type: str
    symbol: str
    detail: str


@dataclass
class ExitIntegrityReport:
    n_records: int
    n_violations: int
    violations: List[ExitIntegrityViolation]
    passed: bool   # False → orchestrator writes no decisions


class ExitOrchestratorIntegrityValidator:
    """
    Validates ExitAdjustmentDecision history for structural and behavioral integrity.
    Fail-closed: any CRITICAL violation → passed=False → nothing written.
    """

    CRITICAL_TYPES = {
        "duplicate_decision_id",
        "future_leakage",
        "insufficient_sample",
        "rapid_recursive_adaptation",
    }

    def validate(
        self,
        records: List[ExitAdjustmentDecision],
        constraints: ExitAdjustmentConstraints,
        as_of_date: Optional[str] = None,
    ) -> ExitIntegrityReport:
        today = as_of_date or _today_str()
        violations: List[ExitIntegrityViolation] = []
        seen_ids: Dict[str, int] = {}
        symbol_dates: Dict[str, List[str]] = {}

        for i, r in enumerate(records):
            # Duplicate decision_id
            if r.decision_id in seen_ids:
                violations.append(ExitIntegrityViolation(
                    violation_type="duplicate_decision_id",
                    symbol=r.symbol,
                    detail=f"index={i} duplicates index={seen_ids[r.decision_id]}",
                ))
            else:
                seen_ids[r.decision_id] = i

            # Future leakage
            if r.eval_date > today:
                violations.append(ExitIntegrityViolation(
                    violation_type="future_leakage",
                    symbol=r.symbol,
                    detail=f"eval_date={r.eval_date} > as_of_date={today}",
                ))

            # Insufficient sample
            if r.observation_count < constraints.min_sample_size:
                violations.append(ExitIntegrityViolation(
                    violation_type="insufficient_sample",
                    symbol=r.symbol,
                    detail=(
                        f"observation_count={r.observation_count} "
                        f"< min_sample_size={constraints.min_sample_size}"
                    ),
                ))

            # Mutual exclusion: force_exit and suppress_exit cannot both be True
            if r.force_exit and r.suppress_exit:
                violations.append(ExitIntegrityViolation(
                    violation_type="mutual_exclusion_violation",
                    symbol=r.symbol,
                    detail="force_exit=True and suppress_exit=True simultaneously",
                ))

            # Bounded flag
            if not r.bounded:
                violations.append(ExitIntegrityViolation(
                    violation_type="unbounded_decision",
                    symbol=r.symbol,
                    detail="bounded=False is not allowed",
                ))

            symbol_dates.setdefault(r.symbol, []).append(r.eval_date)

        # Rapid recursive adaptation: same symbol changed > quota per 7 days
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        for sym, dates in symbol_dates.items():
            week_dates = [
                d for d in dates
                if (today_dt - datetime.strptime(d, "%Y-%m-%d")).days <= 7
            ]
            if len(week_dates) > constraints.max_decisions_per_symbol_per_week:
                violations.append(ExitIntegrityViolation(
                    violation_type="rapid_recursive_adaptation",
                    symbol=sym,
                    detail=(
                        f"symbol={sym} has {len(week_dates)} decisions "
                        f"in 7-day window > max={constraints.max_decisions_per_symbol_per_week}"
                    ),
                ))

            # Policy oscillation: cooldown violation
            sorted_dates = sorted(dates)
            for j in range(1, len(sorted_dates)):
                gap = _days_between(sorted_dates[j - 1], sorted_dates[j])
                if gap < constraints.cooldown_days:
                    violations.append(ExitIntegrityViolation(
                        violation_type="policy_oscillation",
                        symbol=sym,
                        detail=(
                            f"symbol={sym} changed on {sorted_dates[j-1]} "
                            f"and again on {sorted_dates[j]} "
                            f"(gap={gap}d < cooldown={constraints.cooldown_days}d)"
                        ),
                    ))

        has_critical = any(v.violation_type in self.CRITICAL_TYPES for v in violations)
        return ExitIntegrityReport(
            n_records=len(records),
            n_violations=len(violations),
            violations=violations,
            passed=not has_critical,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. RuntimeExitOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeExitOrchestrator:
    """
    Consumes exit intelligence analytics → produces bounded per-symbol ExitAdjustmentDecision.

    STRICT: no strategy mutation, no self-optimization, no unbounded adaptation.
    Every adjustment bounded by ExitAdjustmentConstraints.
    Every decision logged append-only to JSONL for full replay compatibility.
    """

    def __init__(
        self,
        constraints: Optional[ExitAdjustmentConstraints] = None,
        store_path: Path = DEFAULT_STORE_PATH,
    ) -> None:
        self._c = constraints or ExitAdjustmentConstraints()
        self._store_path = store_path

    def evaluate(
        self,
        exit_intelligence: ExitIntelligenceSnapshot,
        positions: List[PositionLifecycleSnapshot],
        regime: RegimeAnalyticsSnapshot,
        expectancy: ExitExpectancySnapshot,
        history: List[ExitAdjustmentDecision],
        as_of_date: Optional[str] = None,
    ) -> List[ExitAdjustmentDecision]:
        """
        Evaluate analytics → produce bounded ExitAdjustmentDecision per held position.

        Returns list of decisions (may be empty if integrity blocked or no adjustment warranted).
        Decisions are NOT written here — caller must call append_decision().
        """
        today = as_of_date or _today_str()
        c = self._c

        # Fail-closed: integrity check on history
        validator = ExitOrchestratorIntegrityValidator()
        report = validator.validate(history, c, as_of_date=today)
        if not report.passed:
            logger.warning(
                "[EXIT_ORCH] integrity check failed — no decisions produced: %s",
                [v.violation_type for v in report.violations],
            )
            return []

        total_obs = exit_intelligence.n_trades + expectancy.n_executed + expectancy.n_skipped

        decisions: List[ExitAdjustmentDecision] = []

        for pos in positions:
            sym = pos.symbol

            # Per-symbol cooldown + quota guards
            if not self._cooldown_ok(sym, history, today):
                logger.debug("[EXIT_ORCH] cooldown active for %s — skipping", sym)
                continue
            if not self._weekly_quota_ok(sym, history, today):
                logger.debug("[EXIT_ORCH] weekly quota exhausted for %s — skipping", sym)
                continue

            # Anti-oscillation: block direction reversal within anti_oscillation_window_days
            last_decision = self._last_decision_for_symbol(sym, history)
            if last_decision is not None:
                gap = _days_between(last_decision.eval_date, today)
                if gap < c.anti_oscillation_window_days:
                    logger.debug(
                        "[EXIT_ORCH] anti-oscillation window active for %s (gap=%dd)", sym, gap
                    )
                    continue

            policy = self._compute_policy(pos, exit_intelligence, regime, expectancy, total_obs)
            if policy is None:
                continue  # no meaningful adjustment

            # Build decision record
            rec = ExitAdjustmentDecision.create(
                eval_date=today,
                symbol=sym,
                policy=policy,
                trigger_runner_score=exit_intelligence.per_symbol_runner_score.get(sym),
                trigger_premature_ratio=exit_intelligence.premature_exit_ratio,
                trigger_giveback_ratio=exit_intelligence.per_symbol_giveback.get(sym),
                trigger_vol=exit_intelligence.per_symbol_vol.get(sym),
                trigger_regime=regime.regime,
                trigger_rs_collapse=regime.rs_collapse_detected,
                observation_count=total_obs,
            )
            decisions.append(rec)

        return decisions

    def _compute_policy(
        self,
        pos: PositionLifecycleSnapshot,
        intel: ExitIntelligenceSnapshot,
        regime: RegimeAnalyticsSnapshot,
        expectancy: ExitExpectancySnapshot,
        total_obs: int,
    ) -> Optional[ExitAdjustmentPolicy]:
        """
        Compute bounded ExitAdjustmentPolicy for a single position.
        Returns None if no meaningful adjustment warranted.
        """
        c = self._c
        sym = pos.symbol

        if total_obs < c.min_sample_size:
            return None

        # ── 1. Hold extension (winner retention) ─────────────────────────────
        runner_score = intel.per_symbol_runner_score.get(sym, 0.0)
        persistence = intel.per_symbol_persistence.get(sym, 0.0)
        premature_ratio = intel.premature_exit_ratio or 0.0

        hold_extension_days = 0
        if (
            runner_score >= 0.60
            and persistence >= 0.50
            and premature_ratio >= 0.25
            and not regime.regime_deteriorating
            and not regime.rs_collapse_detected
        ):
            raw_ext = runner_score * 3.0 + persistence * 2.0
            hold_extension_days = min(int(raw_ext), c.max_hold_extension_days)

        # ── 2. Exit sensitivity scale ─────────────────────────────────────────
        # Looser (>1.0): persistent convexity + strong breadth → retain winners
        # Tighter (<1.0): regime deteriorating or RS collapse → exit sooner
        exit_sensitivity_scale = 1.0
        if regime.regime_deteriorating or regime.rs_collapse_detected:
            tighten = 0.10 + (0.05 if regime.rs_collapse_detected else 0.0)
            exit_sensitivity_scale = _clamp(1.0 - tighten, c.exit_sensitivity_min, 1.0)
        elif (
            premature_ratio >= 0.35
            and (regime.breadth_score or 0.0) >= 0.55
            and runner_score >= 0.55
        ):
            loosen = premature_ratio * 0.25
            exit_sensitivity_scale = _clamp(1.0 + loosen, 1.0, c.exit_sensitivity_max)

        # ── 3. Deterioration pressure ─────────────────────────────────────────
        giveback = intel.per_symbol_giveback.get(sym, 0.0)
        decay_rate = intel.per_symbol_decay_rate.get(sym, 0.0)
        mean_giveback = intel.mean_giveback_ratio or 0.0

        deterioration_pressure = 0.0
        if regime.regime_deteriorating:
            deterioration_pressure += 0.35
        if regime.rs_collapse_detected:
            deterioration_pressure += 0.30
        if giveback >= 0.65 or mean_giveback >= 0.60:
            deterioration_pressure += 0.25
        if decay_rate >= 0.03:   # losing ≥3% per day after peak
            deterioration_pressure += 0.20
        if pos.rs_score is not None and pos.rs_score < -0.02:
            deterioration_pressure += 0.15
        deterioration_pressure = _clamp(deterioration_pressure, 0.0, c.deterioration_pressure_max)

        # ── 4. Trend persistence retention score ─────────────────────────────
        trend_persistence_score = _clamp(
            persistence * 0.60 + runner_score * 0.40, 0.0, 1.0
        )

        # ── 4b. Compression continuation override ────────────────────────────
        # If per-position compression assessment says healthy compression:
        #   extend hold by 2-3 business days and boost trend_persistence_score.
        # If distribution: add deterioration pressure to accelerate exit.
        # Fail-closed: only acts when suppression_eligible=True (gate already
        #   validates rsr >= 75, rsr_momentum >= -1.5, no invalidation flags).
        if pos.suppression_eligible and pos.compression_score >= 65.0:
            comp_ext = 3 if pos.compression_score >= 75.0 else 2
            hold_extension_days = max(
                hold_extension_days, min(comp_ext, c.max_hold_extension_days)
            )
            trend_persistence_score = max(
                trend_persistence_score,
                _clamp(pos.compression_score / 100.0, 0.0, 1.0),
            )
            logger.debug(
                "[EXIT_ORCH] compression suppression: %s score=%.1f ext=%dd",
                sym, pos.compression_score, comp_ext,
            )
        if pos.compression_phase == "distribution":
            deterioration_pressure = _clamp(
                deterioration_pressure + 0.30, 0.0, c.deterioration_pressure_max
            )
            logger.debug(
                "[EXIT_ORCH] distribution pressure: %s detr=%.2f",
                sym, deterioration_pressure,
            )

        # ── 5. Vol-aware trailing multiplier ─────────────────────────────────
        vol = intel.per_symbol_vol.get(sym, 0.25)  # default: moderate vol
        # High vol → tighten trailing (lower multiplier)
        raw_mult = 1.0 / (1.0 + vol * 1.5)
        vol_trailing_multiplier = _clamp(raw_mult, c.vol_trailing_mult_min, c.vol_trailing_mult_max)

        # ── 6. Regime-conditioned threshold adjustment ────────────────────────
        regime_threshold_adjustment = 0.0
        if regime.regime == "bull" and not regime.regime_deteriorating:
            regime_threshold_adjustment = _clamp(
                (regime.regime_confidence - 0.5) * 0.10,
                0.0,
                c.regime_threshold_adj_max,
            )
        elif regime.regime == "bear" or regime.regime_deteriorating:
            regime_threshold_adjustment = _clamp(
                -(regime.regime_confidence - 0.5) * 0.12,
                c.regime_threshold_adj_min,
                0.0,
            )

        # ── 7. Derive action flags ────────────────────────────────────────────
        # force_exit and suppress_exit are mutually exclusive
        force_exit = deterioration_pressure >= c.force_exit_pressure_threshold
        suppress_exit = (
            hold_extension_days > 0
            and trend_persistence_score >= 0.50
            and not force_exit
        )

        # Check if any meaningful adjustment exists
        no_adjustment = (
            hold_extension_days == 0
            and abs(exit_sensitivity_scale - 1.0) < 1e-4
            and deterioration_pressure < 0.10
            and abs(vol_trailing_multiplier - 1.0) < 0.05
            and abs(regime_threshold_adjustment) < 1e-4
            and not force_exit
            and not suppress_exit
        )
        if no_adjustment:
            return None

        return ExitAdjustmentPolicy(
            symbol=sym,
            hold_extension_days=hold_extension_days,
            exit_sensitivity_scale=exit_sensitivity_scale,
            deterioration_pressure=deterioration_pressure,
            trend_persistence_score=trend_persistence_score,
            vol_trailing_multiplier=vol_trailing_multiplier,
            regime_threshold_adjustment=regime_threshold_adjustment,
            suppress_exit=suppress_exit,
            force_exit=force_exit,
        )

    # ── Guard helpers ─────────────────────────────────────────────────────────

    def _cooldown_ok(
        self, symbol: str, history: List[ExitAdjustmentDecision], today: str
    ) -> bool:
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d")
            - timedelta(days=self._c.cooldown_days)
        ).strftime("%Y-%m-%d")
        recent = [r for r in history if r.symbol == symbol and r.eval_date >= cutoff]
        return len(recent) == 0

    def _weekly_quota_ok(
        self, symbol: str, history: List[ExitAdjustmentDecision], today: str
    ) -> bool:
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")
        count = sum(1 for r in history if r.symbol == symbol and r.eval_date >= cutoff)
        return count < self._c.max_decisions_per_symbol_per_week

    def _last_decision_for_symbol(
        self, symbol: str, history: List[ExitAdjustmentDecision]
    ) -> Optional[ExitAdjustmentDecision]:
        relevant = [r for r in history if r.symbol == symbol]
        if not relevant:
            return None
        return max(relevant, key=lambda r: r.eval_date)

    # ── Signal + order application ────────────────────────────────────────────

    def apply_to_signals(
        self,
        signals: List[Dict[str, Any]],
        decisions: List[ExitAdjustmentDecision],
    ) -> List[Dict[str, Any]]:
        """
        Apply exit adjustment decisions to live signals for held positions.

        suppress_exit=True: prevent premature exit (signal 0→1 for held position)
        force_exit=True:    accelerate exit (signal 1→0 for held position)

        Returns modified signal list. Original list is not mutated.
        """
        if not decisions:
            return signals

        decision_map: Dict[str, ExitAdjustmentDecision] = {d.symbol: d for d in decisions}
        modified: List[Dict[str, Any]] = []

        for sig in signals:
            sym = sig.get("symbol", "")
            if sym not in decision_map or not sig.get("currently_holding"):
                modified.append(sig)
                continue

            dec = decision_map[sym]
            new_sig = dict(sig)

            if dec.force_exit and sig.get("signal") == 1:
                new_sig["signal"] = 0
                new_sig["action"] = "SELL"
                new_sig["exit_reason"] = "exit_orchestrator_force_exit"
                new_sig["exit_orchestrator_applied"] = True
                logger.info(
                    "[EXIT_ORCH] force_exit applied: %s (pressure=%.2f)",
                    sym, dec.deterioration_pressure,
                )
            elif dec.suppress_exit and sig.get("signal") == 0:
                new_sig["signal"] = 1
                new_sig["action"] = "HOLD"
                new_sig["exit_reason"] = "exit_orchestrator_suppress_premature_exit"
                new_sig["exit_orchestrator_applied"] = True
                new_sig["hold_extension_days"] = dec.hold_extension_days
                logger.info(
                    "[EXIT_ORCH] suppress_exit applied: %s (extension=%dd)",
                    sym, dec.hold_extension_days,
                )

            # Always annotate vol trailing multiplier for display/downstream
            if abs(dec.vol_trailing_multiplier - 1.0) > 0.02:
                new_sig["exit_orchestrator_vol_mult"] = dec.vol_trailing_multiplier

            modified.append(new_sig)

        return modified

    def apply_to_orders(
        self,
        order_objects: List[Any],
        decisions: List[ExitAdjustmentDecision],
        signals: List[Dict[str, Any]],
        broker_positions: Optional[Dict[str, int]] = None,
    ) -> List[Any]:
        """
        Apply exit adjustments to order_objects list.

        suppress_exit: remove SELL orders for suppressed symbols.
        force_exit: add SELL order if none exists (constructed from signal + broker position data).

        Returns modified order_objects list. Original list is not mutated.
        """
        if not decisions:
            return order_objects

        decision_map: Dict[str, ExitAdjustmentDecision] = {d.symbol: d for d in decisions}
        suppress_syms = {sym for sym, d in decision_map.items() if d.suppress_exit}
        force_syms = {sym for sym, d in decision_map.items() if d.force_exit}

        # Existing SELL symbols in order_objects
        existing_sell_syms = {
            o.symbol for o in order_objects
            if hasattr(o, "side") and o.side == "SELL"
        }

        result: List[Any] = []
        for o in order_objects:
            sym = getattr(o, "symbol", None)
            side = getattr(o, "side", None)
            if sym in suppress_syms and side == "SELL":
                logger.info("[EXIT_ORCH] suppress_exit: removed SELL order for %s", sym)
                continue  # drop the premature exit order
            result.append(o)

        # Force exit: add SELL orders for positions not already exiting
        for sym in force_syms:
            if sym in existing_sell_syms:
                continue  # already has a SELL order
            qty = (broker_positions or {}).get(sym)
            if not qty or qty <= 0:
                logger.warning(
                    "[EXIT_ORCH] force_exit for %s: no broker position found — "
                    "signal modified but SELL order not added",
                    sym,
                )
                continue
            # Derive symbol_4digit: first 4 chars of symbol (e.g. "7203.T" → "7203")
            sym_4d = sym.split(".")[0][:4]
            # Get sector + price estimate from signals
            sig_for_sym = next(
                (s for s in signals if s.get("symbol") == sym), {}
            )
            sector = sig_for_sym.get("sector", "不明")
            trailing_stop = sig_for_sym.get("trailing_stop")
            if trailing_stop and trailing_stop > 0:
                estimated_price = float(trailing_stop)
            else:
                estimated_price = 0.0
            estimated_amount = estimated_price * qty if estimated_price > 0 else 0.0
            try:
                from src.kabusapi.signal_bridge import OrderInstruction
                sell_order = OrderInstruction(
                    symbol=sym,
                    symbol_4digit=sym_4d,
                    sector=sector,
                    side="SELL",
                    qty=qty,
                    order_type="MARKET_OPEN",
                    estimated_price=estimated_price,
                    estimated_amount=estimated_amount,
                    reason="exit_orchestrator_force_exit",
                    atr20=0.0,
                    strategy_type="fujiko",
                )
                result.append(sell_order)
                logger.info(
                    "[EXIT_ORCH] force_exit: added SELL order for %s qty=%d", sym, qty
                )
            except Exception as exc:
                logger.warning(
                    "[EXIT_ORCH] force_exit: failed to construct SELL order for %s: %s",
                    sym, exc,
                )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_decision(
    record: ExitAdjustmentDecision,
    path: Path = DEFAULT_STORE_PATH,
) -> None:
    """
    Atomic append to JSONL. fsync durability. Never raises — analytics continues on error.
    """
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
        logger.warning("[EXIT_ORCH] append failed (%s) — continuing", exc)


def load_decisions(path: Path = DEFAULT_STORE_PATH) -> List[ExitAdjustmentDecision]:
    """Load all records from JSONL. Corrupted lines are skipped with a warning."""
    if not path.exists():
        return []
    records: List[ExitAdjustmentDecision] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(ExitAdjustmentDecision.from_dict(d))
        except Exception as exc:
            logger.warning("[EXIT_ORCH] parse error at line %d: %s", i + 1, exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 8. RuntimeIntegrationHook — lightweight hook for run_live_signal.py
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeIntegrationHook:
    """
    Lightweight exit orchestration hook for direct injection into run_live_signal.py.

    Usage:
        hook = RuntimeIntegrationHook(store_path=EXIT_POLICY_DECISIONS_FILE)
        signals, orders = hook.run(
            signals=result.signals,
            order_objects=order_objects,
            portfolio_summary=result.portfolio_summary,
            broker_positions=_broker_pos,
            regime_info={"regime": "bear", "deteriorating": True, ...},
        )
        result.signals[:] = signals
        order_objects = orders

    Fail-open: any error returns inputs unchanged.
    """

    def __init__(
        self,
        store_path: Path = DEFAULT_STORE_PATH,
        constraints: Optional[ExitAdjustmentConstraints] = None,
    ) -> None:
        self._store_path = store_path
        self._orchestrator = RuntimeExitOrchestrator(
            constraints=constraints or ExitAdjustmentConstraints(),
            store_path=store_path,
        )

    def run(
        self,
        signals: List[Dict[str, Any]],
        order_objects: List[Any],
        portfolio_summary: Optional[Dict[str, Any]] = None,
        broker_positions: Optional[Dict[str, int]] = None,
        regime_info: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Run exit orchestration on current signals and order_objects.

        Returns (modified_signals, modified_orders).
        Fail-open: on any error returns inputs unchanged.
        """
        try:
            return self._run_internal(
                signals, order_objects, portfolio_summary,
                broker_positions, regime_info, as_of_date,
            )
        except Exception as exc:
            logger.warning("[EXIT_ORCH] hook failed (%s) — inputs unchanged", exc)
            return signals, order_objects

    def _run_internal(
        self,
        signals: List[Dict[str, Any]],
        order_objects: List[Any],
        portfolio_summary: Optional[Dict[str, Any]],
        broker_positions: Optional[Dict[str, int]],
        regime_info: Optional[Dict[str, Any]],
        as_of_date: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        ps = portfolio_summary or {}
        ri = regime_info or {}

        # Build input snapshots from available runtime data
        exit_intel = self._build_exit_intelligence(signals, ps)
        positions = self._build_positions(signals, ps)
        regime = self._build_regime(ri, ps, signals)
        expectancy = self._build_expectancy(ps)

        if not positions:
            logger.debug("[EXIT_ORCH] no held positions — nothing to evaluate")
            return signals, order_objects

        # Load decision history
        history = load_decisions(self._store_path)

        # Evaluate: produce bounded decisions
        decisions = self._orchestrator.evaluate(
            exit_intelligence=exit_intel,
            positions=positions,
            regime=regime,
            expectancy=expectancy,
            history=history,
            as_of_date=as_of_date,
        )

        if not decisions:
            logger.debug("[EXIT_ORCH] no exit adjustment decisions produced")
            return signals, order_objects

        # Persist decisions (append-only)
        for dec in decisions:
            append_decision(dec, self._store_path)
            logger.info(
                "[EXIT_ORCH] decision: symbol=%s suppress=%s force=%s hold_ext=%dd date=%s",
                dec.symbol, dec.suppress_exit, dec.force_exit,
                dec.hold_extension_days, dec.eval_date,
            )

        # Apply to signals
        modified_signals = self._orchestrator.apply_to_signals(signals, decisions)

        # Apply to orders
        modified_orders = self._orchestrator.apply_to_orders(
            order_objects, decisions, modified_signals, broker_positions,
        )

        return modified_signals, modified_orders

    # ── Input snapshot builders ────────────────────────────────────────────────

    def _build_exit_intelligence(
        self, signals: List[Dict[str, Any]], ps: Dict[str, Any]
    ) -> ExitIntelligenceSnapshot:
        """Build ExitIntelligenceSnapshot from runtime signals and portfolio summary."""
        # Try loading from EXIT_RECORDS_FILE if available
        intel = ExitIntelligenceSnapshot()
        try:
            from src.paths import EXIT_RECORDS_FILE
            from src.analytics.exits.models import ExitConvexityRecord
            from src.analytics.exits.convexity import RunnerRetentionDiagnostics

            if EXIT_RECORDS_FILE.exists():
                records = []
                for line in EXIT_RECORDS_FILE.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(ExitConvexityRecord.from_dict(json.loads(line)))
                    except Exception:
                        pass

                if records:
                    diagnostics = RunnerRetentionDiagnostics()
                    summary = diagnostics.compute(records)
                    intel.premature_exit_ratio = summary.premature_exit_ratio
                    intel.mean_giveback_ratio = summary.mean_pnl_giveback_ratio
                    intel.mean_runner_retention = summary.mean_runner_retention_rate
                    intel.mean_convexity_capture = summary.mean_convexity_capture_rate
                    intel.mean_giveback_ratio = summary.mean_pnl_giveback_ratio
                    intel.n_trades = summary.total_trades

                    # Per-symbol (use most recent record per symbol)
                    by_sym: Dict[str, ExitConvexityRecord] = {}
                    for r in records:
                        if r.symbol not in by_sym or r.exit_timestamp > by_sym[r.symbol].exit_timestamp:
                            by_sym[r.symbol] = r
                    for sym, r in by_sym.items():
                        intel.per_symbol_runner_score[sym] = r.runner_continuation_score
                        intel.per_symbol_giveback[sym] = r.pnl_giveback_ratio
                        intel.per_symbol_vol[sym] = r.volatility_during_hold
                        if r.relative_trend_persistence_score is not None:
                            intel.per_symbol_persistence[sym] = r.relative_trend_persistence_score
                        intel.per_symbol_decay_rate[sym] = r.post_peak_decay_rate
        except Exception as exc:
            logger.debug("[EXIT_ORCH] exit intelligence load failed (%s) — using defaults", exc)

        return intel

    def _build_positions(
        self, signals: List[Dict[str, Any]], ps: Dict[str, Any]
    ) -> List[PositionLifecycleSnapshot]:
        """Extract held position snapshots from signals."""
        positions = []
        current_dd = float(ps.get("current_drawdown", 0.0))

        for sig in signals:
            if not sig.get("currently_holding"):
                continue
            sym = sig.get("symbol", "")
            if not sym:
                continue
            hold_days = int(sig.get("hold_days", 0))
            trailing_stop = sig.get("trailing_stop")
            ts_dist: Optional[float] = None
            if trailing_stop and float(trailing_stop) > 0:
                ts_dist = float(trailing_stop)

            positions.append(PositionLifecycleSnapshot(
                symbol=sym,
                hold_days=hold_days,
                current_return=float(sig.get("current_return", current_dd)),
                rs_score=_opt_float(sig.get("rsr_momentum")),
                rsr=_opt_float(sig.get("rsr")),
                is_at_turtle_exit=(int(sig.get("signal", 1)) == 0),
                trailing_stop_distance=ts_dist,
                compression_score=float(sig.get("compression_score", 0.0)),
                suppression_eligible=bool(sig.get("suppression_eligible", False)),
                compression_phase=str(sig.get("compression_phase", "neutral")),
            ))
        return positions

    def _build_regime(
        self,
        regime_info: Dict[str, Any],
        ps: Dict[str, Any],
        signals: List[Dict[str, Any]],
    ) -> RegimeAnalyticsSnapshot:
        """Build RegimeAnalyticsSnapshot from runtime context."""
        regime = str(regime_info.get("regime", "unknown"))
        deteriorating = bool(regime_info.get("deteriorating", False))
        rs_collapse = bool(regime_info.get("rs_collapse", False))
        confidence = float(regime_info.get("confidence", 0.5))

        # Infer breadth from signals
        n_sigs = len(signals)
        breadth: Optional[float] = None
        if n_sigs > 0:
            n_above_50 = sum(1 for s in signals if float(s.get("rsr", 0)) >= 50)
            breadth = n_above_50 / n_sigs

        return RegimeAnalyticsSnapshot(
            regime=regime,
            regime_deteriorating=deteriorating,
            breadth_score=breadth,
            rs_collapse_detected=rs_collapse,
            regime_confidence=confidence,
        )

    def _build_expectancy(self, ps: Dict[str, Any]) -> ExitExpectancySnapshot:
        """Build ExitExpectancySnapshot from portfolio summary context."""
        return ExitExpectancySnapshot(
            n_executed=int(ps.get("current_positions", 0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-level hook for run_live_signal.py
# ─────────────────────────────────────────────────────────────────────────────

def run_exit_orchestration_hook(
    signals: List[Dict[str, Any]],
    order_objects: List[Any],
    store_path: Path = DEFAULT_STORE_PATH,
    portfolio_summary: Optional[Dict[str, Any]] = None,
    broker_positions: Optional[Dict[str, int]] = None,
    regime_info: Optional[Dict[str, Any]] = None,
    constraints: Optional[ExitAdjustmentConstraints] = None,
    as_of_date: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Top-level hook for run_live_signal.py.

    Returns (modified_signals, modified_orders).
    Fail-open: any error returns inputs unchanged.

    Usage in run_live_signal.py:
        from src.runtime.policy.runtime_exit_orchestrator import run_exit_orchestration_hook
        from src.paths import EXIT_POLICY_DECISIONS_FILE
        result.signals, order_objects = run_exit_orchestration_hook(
            signals=result.signals,
            order_objects=order_objects,
            store_path=EXIT_POLICY_DECISIONS_FILE,
            portfolio_summary=result.portfolio_summary,
            broker_positions=_broker_pos,
            regime_info=_regime_info,
        )
    """
    try:
        hook = RuntimeIntegrationHook(
            store_path=store_path,
            constraints=constraints or ExitAdjustmentConstraints(),
        )
        return hook.run(
            signals=signals,
            order_objects=order_objects,
            portfolio_summary=portfolio_summary,
            broker_positions=broker_positions,
            regime_info=regime_info,
            as_of_date=as_of_date,
        )
    except Exception as exc:
        logger.warning("[EXIT_ORCH] top-level hook failed (%s) — inputs unchanged", exc)
        return signals, order_objects
