"""
src/runtime/policy/analytics_policy_bridge.py — Analytics → runtime policy bridge

Safely connects observational analytics to bounded runtime decision policies.

STRICT CONSTRAINTS:
  - No autonomous strategy mutation
  - No self-optimizing loops
  - No parameter auto-tuning
  - Bounded adaptation only (hard safety bounds enforced at write time)
  - Fail-closed on integrity violations
  - Observational analytics are the only input — no direct trading decisions

Design:
  - AnalyticsPolicyBridge: reads analytics → produces bounded PolicyDecisionRecord
  - PolicyConstraints: hard safety bounds (never relaxed by code)
  - PolicyDecisionRecord: immutable, frozen, append-only JSONL
  - PolicyIntegrityValidator: validates no oscillation / rapid recursion / future leakage
  - ActivePolicyState: reads JSONL and surfaces current effective policy for live signal
  - All outputs: deterministic, UTF-8, fsync-durable, atomic append
  - Windows compatible
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1

JST = timezone(timedelta(hours=9))

# ── Policy type constants ──────────────────────────────────────────────────────
POLICY_SLOT_ALLOCATION      = "slot_allocation"
POLICY_SECTOR_EXPOSURE_CAP  = "sector_exposure_cap"
POLICY_PORTFOLIO_HEAT_LIMIT = "portfolio_heat_limit"
POLICY_HOLD_EXTENSION       = "hold_extension_eligibility"
POLICY_ENTRY_THROTTLING     = "entry_throttling"
POLICY_RISK_SCALING         = "risk_scaling"

ALL_POLICY_TYPES = (
    POLICY_SLOT_ALLOCATION,
    POLICY_SECTOR_EXPOSURE_CAP,
    POLICY_PORTFOLIO_HEAT_LIMIT,
    POLICY_HOLD_EXTENSION,
    POLICY_ENTRY_THROTTLING,
    POLICY_RISK_SCALING,
)

# ── Trigger metric constants ───────────────────────────────────────────────────
TRIGGER_SKIPPED_EXPECTANCY_DELTA   = "skipped_expectancy_delta"
TRIGGER_SKIPPED_WIN_RATE           = "skipped_win_rate"
TRIGGER_CAPTURE_RATIO              = "opportunity_capture_ratio"
TRIGGER_CONVEXITY_SCORE            = "skipped_convexity_score"
TRIGGER_SECTOR_MISS_RATE           = "sector_miss_rate"
TRIGGER_REGIME_MISS_RATE           = "regime_miss_rate"
TRIGGER_HEAT_LEAK_RATE             = "heat_leak_rate"
TRIGGER_LIFECYCLE_EXTENSION_SIGNAL = "lifecycle_extension_signal"

DEFAULT_STORE_PATH = Path("runtime/policy/policy_decisions.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# 1. PolicyConstraints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyConstraints:
    """
    Hard safety bounds. Never relaxed by code — only by explicit user configuration.
    All policy decisions are validated against these before being written.
    """
    max_allocation_change_per_week: float = 0.05   # max absolute Δ in allocation fraction/week
    max_exposure_delta: float = 0.10               # max absolute Δ in exposure cap
    cooldown_days_after_change: int = 3            # min days between changes to same policy_type
    min_sample_size: int = 10                      # min observations before any decision
    confidence_threshold: float = 0.70            # min confidence before applying change
    max_decisions_per_type_per_week: int = 1       # max decisions per policy_type per 7-day window
    max_total_decisions_per_week: int = 3          # max total decisions per 7-day window
    # Absolute hard bounds on policy values (regardless of delta)
    slot_allocation_min: float = 0.10
    slot_allocation_max: float = 1.00
    sector_exposure_cap_min: float = 0.10
    sector_exposure_cap_max: float = 0.50
    portfolio_heat_min: float = 0.05
    portfolio_heat_max: float = 0.50
    entry_throttle_min: float = 0.20
    entry_throttle_max: float = 1.00
    risk_scaling_min: float = 0.50
    risk_scaling_max: float = 1.50


# ─────────────────────────────────────────────────────────────────────────────
# 2. PolicyDecisionRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyDecisionRecord:
    """
    Immutable policy decision record. Written once; never mutated.
    Append-only JSONL store.
    """
    decision_id: str          # sha256 dedup key
    schema_version: int
    timestamp_jst: str        # ISO-8601 JST
    eval_date: str            # YYYY-MM-DD (JST)
    policy_type: str          # one of ALL_POLICY_TYPES
    previous_value: float
    new_value: float
    trigger_metric: str
    trigger_value: float
    confidence: float
    bounded: bool             # True if bounded by PolicyConstraints
    rollback_eligible: bool   # True if can be reverted on next cycle
    observation_count: int    # number of observations this decision is based on
    lookback_days: int        # analytics lookback window used

    @staticmethod
    def create(
        timestamp_jst: str,
        eval_date: str,
        policy_type: str,
        previous_value: float,
        new_value: float,
        trigger_metric: str,
        trigger_value: float,
        confidence: float,
        bounded: bool,
        rollback_eligible: bool,
        observation_count: int,
        lookback_days: int,
    ) -> "PolicyDecisionRecord":
        decision_id = _sha256(_canonical_json({
            "eval_date": eval_date,
            "new_value": new_value,
            "policy_type": policy_type,
            "timestamp_jst": timestamp_jst,
        }))
        return PolicyDecisionRecord(
            decision_id=decision_id,
            schema_version=SCHEMA_VERSION,
            timestamp_jst=timestamp_jst,
            eval_date=eval_date,
            policy_type=policy_type,
            previous_value=previous_value,
            new_value=new_value,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            confidence=confidence,
            bounded=bounded,
            rollback_eligible=rollback_eligible,
            observation_count=observation_count,
            lookback_days=lookback_days,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PolicyDecisionRecord":
        return PolicyDecisionRecord(
            decision_id=d.get("decision_id", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            timestamp_jst=d.get("timestamp_jst", ""),
            eval_date=d.get("eval_date", ""),
            policy_type=d.get("policy_type", ""),
            previous_value=float(d.get("previous_value", 0.0)),
            new_value=float(d.get("new_value", 0.0)),
            trigger_metric=d.get("trigger_metric", ""),
            trigger_value=float(d.get("trigger_value", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            bounded=bool(d.get("bounded", True)),
            rollback_eligible=bool(d.get("rollback_eligible", True)),
            observation_count=int(d.get("observation_count", 0)),
            lookback_days=int(d.get("lookback_days", 0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. AnalyticsPolicyBridge
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalyticsInput:
    """
    Aggregated analytics snapshot consumed by the bridge.
    All fields optional — bridge degrades gracefully on missing data.
    """
    # From executed_vs_skipped_expectancy
    skipped_expectancy_delta_5d: Optional[float] = None    # skipped − executed
    skipped_win_rate: Optional[float] = None
    opportunity_capture_ratio: Optional[float] = None
    skipped_convexity_score: Optional[float] = None
    n_executed: int = 0
    n_skipped: int = 0
    # Per-reason leakage (skip_reason → mean_5d_return)
    capital_constraint_leakage: Optional[float] = None
    heat_limit_leakage: Optional[float] = None
    sector_exposure_leakage: Optional[float] = None
    # From position_lifecycle / exit analytics
    lifecycle_hold_extension_signal: Optional[float] = None   # positive = should hold longer
    # From regime analytics
    regime_miss_rate: Optional[float] = None   # fraction of enriched skips in top regime
    eval_date: str = ""                        # YYYY-MM-DD of the snapshot


@dataclass
class BridgeDecision:
    """Proposed policy adjustment before integrity validation and write."""
    policy_type: str
    previous_value: float
    proposed_value: float
    trigger_metric: str
    trigger_value: float
    confidence: float
    observation_count: int
    eligible: bool
    block_reason: Optional[str]


class AnalyticsPolicyBridge:
    """
    Consumes analytics → produces bounded PolicyDecisionRecord objects.

    STRICT: no strategy mutation, no self-optimization, no autonomous parameter changes.
    Every decision is bounded by PolicyConstraints and validated by PolicyIntegrityValidator.
    """

    def __init__(
        self,
        constraints: Optional[PolicyConstraints] = None,
        store_path: Path = DEFAULT_STORE_PATH,
    ) -> None:
        self._constraints = constraints or PolicyConstraints()
        self._store_path = store_path

    def evaluate(
        self,
        analytics: AnalyticsInput,
        current_policy: Dict[str, float],
        history: List[PolicyDecisionRecord],
        as_of_date: Optional[str] = None,
    ) -> List[PolicyDecisionRecord]:
        """
        Evaluate analytics → produce bounded policy decisions.

        Args:
            analytics:      Aggregated analytics snapshot.
            current_policy: {policy_type: current_value} — baseline policy values.
            history:        Previously written PolicyDecisionRecord list.
            as_of_date:     Override date for testing (YYYY-MM-DD).

        Returns:
            List of new PolicyDecisionRecord (may be empty if no adjustment warranted).
            Records are NOT written here — caller must call append_decision().
        """
        today = as_of_date or _today_str()
        ts = datetime.now(JST).isoformat()

        # Fail-closed on integrity violations
        integrity = PolicyIntegrityValidator()
        report = integrity.validate(history, as_of_date=today)
        if not report.passed:
            logger.warning("[BRIDGE] integrity check failed — no decisions produced: %s",
                           [v.violation_type for v in report.violations])
            return []

        total_obs = analytics.n_executed + analytics.n_skipped

        decisions: List[PolicyDecisionRecord] = []

        # --- slot_allocation: capital_constraint leakage → loosen slot fraction ---
        decisions.extend(self._eval_slot_allocation(
            analytics, current_policy, history, today, ts, total_obs,
        ))

        # --- sector_exposure_cap: sector_exposure leakage → loosen cap ---
        decisions.extend(self._eval_sector_exposure(
            analytics, current_policy, history, today, ts, total_obs,
        ))

        # --- portfolio_heat_limit: heat leakage → loosen limit ---
        decisions.extend(self._eval_portfolio_heat(
            analytics, current_policy, history, today, ts, total_obs,
        ))

        # --- hold_extension_eligibility: lifecycle signal → enable/disable ---
        decisions.extend(self._eval_hold_extension(
            analytics, current_policy, history, today, ts, total_obs,
        ))

        # --- entry_throttling: low capture ratio → throttle less ---
        decisions.extend(self._eval_entry_throttling(
            analytics, current_policy, history, today, ts, total_obs,
        ))

        # --- risk_scaling: convexity / expectancy → adjust overall scale ---
        decisions.extend(self._eval_risk_scaling(
            analytics, current_policy, history, today, ts, total_obs,
        ))

        # Enforce max total decisions per week
        recent_total = self._recent_decision_count(history, today, window_days=7)
        allowed = self._constraints.max_total_decisions_per_week - recent_total
        decisions = decisions[:max(0, allowed)]

        return decisions

    # ── Per-policy evaluators ─────────────────────────────────────────────────

    def _eval_slot_allocation(
        self, a: AnalyticsInput, current: Dict[str, float],
        history: List[PolicyDecisionRecord], today: str, ts: str, total_obs: int,
    ) -> List[PolicyDecisionRecord]:
        c = self._constraints
        prev = current.get(POLICY_SLOT_ALLOCATION, 1.0)
        leak = a.capital_constraint_leakage
        if leak is None or total_obs < c.min_sample_size:
            return []
        if not (leak > 0.01):
            return []
        if not self._cooldown_ok(POLICY_SLOT_ALLOCATION, history, today):
            return []
        if not self._weekly_quota_ok(POLICY_SLOT_ALLOCATION, history, today):
            return []
        delta = _clamp(leak * 0.5, 0.0, c.max_allocation_change_per_week)
        proposed = _clamp(prev + delta, c.slot_allocation_min, c.slot_allocation_max)
        if abs(proposed - prev) < 1e-6:
            return []
        confidence = _compute_confidence(total_obs, leak, c.min_sample_size)
        if confidence < c.confidence_threshold:
            return []
        rec = PolicyDecisionRecord.create(
            timestamp_jst=ts, eval_date=today,
            policy_type=POLICY_SLOT_ALLOCATION,
            previous_value=round(prev, 6), new_value=round(proposed, 6),
            trigger_metric=TRIGGER_SKIPPED_EXPECTANCY_DELTA, trigger_value=round(leak, 6),
            confidence=round(confidence, 4), bounded=True, rollback_eligible=True,
            observation_count=total_obs, lookback_days=a.lookback_days if hasattr(a, "lookback_days") else 20,
        )
        return [rec]

    def _eval_sector_exposure(
        self, a: AnalyticsInput, current: Dict[str, float],
        history: List[PolicyDecisionRecord], today: str, ts: str, total_obs: int,
    ) -> List[PolicyDecisionRecord]:
        c = self._constraints
        prev = current.get(POLICY_SECTOR_EXPOSURE_CAP, 0.30)
        leak = a.sector_exposure_leakage
        if leak is None or total_obs < c.min_sample_size:
            return []
        if not (leak > 0.01):
            return []
        if not self._cooldown_ok(POLICY_SECTOR_EXPOSURE_CAP, history, today):
            return []
        if not self._weekly_quota_ok(POLICY_SECTOR_EXPOSURE_CAP, history, today):
            return []
        delta = _clamp(leak * 0.3, 0.0, c.max_exposure_delta)
        proposed = _clamp(prev + delta, c.sector_exposure_cap_min, c.sector_exposure_cap_max)
        if abs(proposed - prev) < 1e-6:
            return []
        confidence = _compute_confidence(total_obs, leak, c.min_sample_size)
        if confidence < c.confidence_threshold:
            return []
        rec = PolicyDecisionRecord.create(
            timestamp_jst=ts, eval_date=today,
            policy_type=POLICY_SECTOR_EXPOSURE_CAP,
            previous_value=round(prev, 6), new_value=round(proposed, 6),
            trigger_metric=TRIGGER_SECTOR_MISS_RATE, trigger_value=round(leak, 6),
            confidence=round(confidence, 4), bounded=True, rollback_eligible=True,
            observation_count=total_obs, lookback_days=20,
        )
        return [rec]

    def _eval_portfolio_heat(
        self, a: AnalyticsInput, current: Dict[str, float],
        history: List[PolicyDecisionRecord], today: str, ts: str, total_obs: int,
    ) -> List[PolicyDecisionRecord]:
        c = self._constraints
        prev = current.get(POLICY_PORTFOLIO_HEAT_LIMIT, 0.40)
        leak = a.heat_limit_leakage
        if leak is None or total_obs < c.min_sample_size:
            return []
        if not (leak > 0.01):
            return []
        if not self._cooldown_ok(POLICY_PORTFOLIO_HEAT_LIMIT, history, today):
            return []
        if not self._weekly_quota_ok(POLICY_PORTFOLIO_HEAT_LIMIT, history, today):
            return []
        delta = _clamp(leak * 0.2, 0.0, c.max_allocation_change_per_week)
        proposed = _clamp(prev + delta, c.portfolio_heat_min, c.portfolio_heat_max)
        if abs(proposed - prev) < 1e-6:
            return []
        confidence = _compute_confidence(total_obs, leak, c.min_sample_size)
        if confidence < c.confidence_threshold:
            return []
        rec = PolicyDecisionRecord.create(
            timestamp_jst=ts, eval_date=today,
            policy_type=POLICY_PORTFOLIO_HEAT_LIMIT,
            previous_value=round(prev, 6), new_value=round(proposed, 6),
            trigger_metric=TRIGGER_HEAT_LEAK_RATE, trigger_value=round(leak, 6),
            confidence=round(confidence, 4), bounded=True, rollback_eligible=True,
            observation_count=total_obs, lookback_days=20,
        )
        return [rec]

    def _eval_hold_extension(
        self, a: AnalyticsInput, current: Dict[str, float],
        history: List[PolicyDecisionRecord], today: str, ts: str, total_obs: int,
    ) -> List[PolicyDecisionRecord]:
        c = self._constraints
        prev = current.get(POLICY_HOLD_EXTENSION, 0.0)
        signal = a.lifecycle_hold_extension_signal
        if signal is None or total_obs < c.min_sample_size:
            return []
        if not self._cooldown_ok(POLICY_HOLD_EXTENSION, history, today):
            return []
        if not self._weekly_quota_ok(POLICY_HOLD_EXTENSION, history, today):
            return []
        proposed = 1.0 if signal > 0.02 else 0.0
        if abs(proposed - prev) < 1e-6:
            return []
        confidence = _compute_confidence(total_obs, abs(signal), c.min_sample_size)
        if confidence < c.confidence_threshold:
            return []
        rec = PolicyDecisionRecord.create(
            timestamp_jst=ts, eval_date=today,
            policy_type=POLICY_HOLD_EXTENSION,
            previous_value=round(prev, 6), new_value=round(proposed, 6),
            trigger_metric=TRIGGER_LIFECYCLE_EXTENSION_SIGNAL, trigger_value=round(signal, 6),
            confidence=round(confidence, 4), bounded=True, rollback_eligible=True,
            observation_count=total_obs, lookback_days=20,
        )
        return [rec]

    def _eval_entry_throttling(
        self, a: AnalyticsInput, current: Dict[str, float],
        history: List[PolicyDecisionRecord], today: str, ts: str, total_obs: int,
    ) -> List[PolicyDecisionRecord]:
        c = self._constraints
        prev = current.get(POLICY_ENTRY_THROTTLING, 1.0)
        cap_ratio = a.opportunity_capture_ratio
        delta_exp = a.skipped_expectancy_delta_5d
        if cap_ratio is None or delta_exp is None or total_obs < c.min_sample_size:
            return []
        # Only throttle less (increase rate) when skipped outperform and capture is low
        if not (delta_exp > 0.01 and cap_ratio < 0.80):
            return []
        if not self._cooldown_ok(POLICY_ENTRY_THROTTLING, history, today):
            return []
        if not self._weekly_quota_ok(POLICY_ENTRY_THROTTLING, history, today):
            return []
        delta = _clamp((1.0 - cap_ratio) * 0.2, 0.0, c.max_allocation_change_per_week)
        proposed = _clamp(prev + delta, c.entry_throttle_min, c.entry_throttle_max)
        if abs(proposed - prev) < 1e-6:
            return []
        confidence = _compute_confidence(total_obs, delta_exp, c.min_sample_size)
        if confidence < c.confidence_threshold:
            return []
        rec = PolicyDecisionRecord.create(
            timestamp_jst=ts, eval_date=today,
            policy_type=POLICY_ENTRY_THROTTLING,
            previous_value=round(prev, 6), new_value=round(proposed, 6),
            trigger_metric=TRIGGER_CAPTURE_RATIO, trigger_value=round(cap_ratio, 6),
            confidence=round(confidence, 4), bounded=True, rollback_eligible=True,
            observation_count=total_obs, lookback_days=20,
        )
        return [rec]

    def _eval_risk_scaling(
        self, a: AnalyticsInput, current: Dict[str, float],
        history: List[PolicyDecisionRecord], today: str, ts: str, total_obs: int,
    ) -> List[PolicyDecisionRecord]:
        c = self._constraints
        prev = current.get(POLICY_RISK_SCALING, 1.0)
        convexity = a.skipped_convexity_score
        delta_exp = a.skipped_expectancy_delta_5d
        if convexity is None or delta_exp is None or total_obs < c.min_sample_size:
            return []
        if not (convexity > 1.2 and delta_exp > 0.01):
            return []
        if not self._cooldown_ok(POLICY_RISK_SCALING, history, today):
            return []
        if not self._weekly_quota_ok(POLICY_RISK_SCALING, history, today):
            return []
        delta = _clamp(delta_exp * 0.3, 0.0, c.max_allocation_change_per_week)
        proposed = _clamp(prev + delta, c.risk_scaling_min, c.risk_scaling_max)
        if abs(proposed - prev) < 1e-6:
            return []
        confidence = _compute_confidence(total_obs, delta_exp, c.min_sample_size)
        if confidence < c.confidence_threshold:
            return []
        rec = PolicyDecisionRecord.create(
            timestamp_jst=ts, eval_date=today,
            policy_type=POLICY_RISK_SCALING,
            previous_value=round(prev, 6), new_value=round(proposed, 6),
            trigger_metric=TRIGGER_CONVEXITY_SCORE, trigger_value=round(convexity, 6),
            confidence=round(confidence, 4), bounded=True, rollback_eligible=True,
            observation_count=total_obs, lookback_days=20,
        )
        return [rec]

    # ── Guard helpers ─────────────────────────────────────────────────────────

    def _cooldown_ok(
        self,
        policy_type: str,
        history: List[PolicyDecisionRecord],
        today: str,
    ) -> bool:
        c = self._constraints
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=c.cooldown_days_after_change)
        ).strftime("%Y-%m-%d")
        recent = [r for r in history if r.policy_type == policy_type and r.eval_date >= cutoff]
        return len(recent) == 0

    def _weekly_quota_ok(
        self,
        policy_type: str,
        history: List[PolicyDecisionRecord],
        today: str,
    ) -> bool:
        c = self._constraints
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")
        per_type = sum(
            1 for r in history if r.policy_type == policy_type and r.eval_date >= cutoff
        )
        return per_type < c.max_decisions_per_type_per_week

    def _recent_decision_count(
        self,
        history: List[PolicyDecisionRecord],
        today: str,
        window_days: int,
    ) -> int:
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=window_days)
        ).strftime("%Y-%m-%d")
        return sum(1 for r in history if r.eval_date >= cutoff)


def _compute_confidence(n: int, signal_strength: float, min_n: int) -> float:
    """Heuristic confidence score in [0, 1]. Increases with sample size and signal strength."""
    if n < min_n:
        return 0.0
    size_factor = _clamp((n - min_n) / max(1, min_n * 3), 0.0, 1.0)
    signal_factor = _clamp(abs(signal_strength) / 0.10, 0.0, 1.0)
    return round((size_factor * 0.5 + signal_factor * 0.5) * 0.95, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PolicyIntegrityValidator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyViolation:
    violation_type: str
    decision_id: str
    detail: str


@dataclass
class PolicyIntegrityReport:
    n_records: int
    n_violations: int
    violations: List[PolicyViolation]
    passed: bool


class PolicyIntegrityValidator:
    """
    Validates a batch of PolicyDecisionRecord for structural and behavioral integrity.
    Fail-closed: any CRITICAL violation → passed=False → bridge writes nothing.
    """

    CRITICAL_TYPES = {
        "duplicate_decision_id",
        "future_leakage",
        "insufficient_sample",
        "rapid_recursive_adaptation",
    }

    def validate(
        self,
        records: List[PolicyDecisionRecord],
        as_of_date: Optional[str] = None,
        constraints: Optional[PolicyConstraints] = None,
    ) -> PolicyIntegrityReport:
        c = constraints or PolicyConstraints()
        today = as_of_date or _today_str()
        violations: List[PolicyViolation] = []

        seen_ids: Dict[str, int] = {}
        policy_dates: Dict[str, List[str]] = {}

        for i, r in enumerate(records):
            # Duplicate decision_id
            if r.decision_id in seen_ids:
                violations.append(PolicyViolation(
                    violation_type="duplicate_decision_id",
                    decision_id=r.decision_id,
                    detail=f"index={i} duplicates index={seen_ids[r.decision_id]}",
                ))
            else:
                seen_ids[r.decision_id] = i

            # Future leakage
            if r.eval_date > today:
                violations.append(PolicyViolation(
                    violation_type="future_leakage",
                    decision_id=r.decision_id,
                    detail=f"eval_date={r.eval_date} > as_of_date={today}",
                ))

            # Insufficient sample
            if r.observation_count < c.min_sample_size:
                violations.append(PolicyViolation(
                    violation_type="insufficient_sample",
                    decision_id=r.decision_id,
                    detail=(
                        f"observation_count={r.observation_count} "
                        f"< min_sample_size={c.min_sample_size}"
                    ),
                ))

            # Policy oscillation detection: same type changes value in opposite direction rapidly
            policy_dates.setdefault(r.policy_type, []).append(r.eval_date)

            # Invalid analytics linkage: confidence must be in [0, 1]
            if not (0.0 <= r.confidence <= 1.0):
                violations.append(PolicyViolation(
                    violation_type="invalid_confidence",
                    decision_id=r.decision_id,
                    detail=f"confidence={r.confidence} out of [0, 1]",
                ))

            # Bounded flag must be True (we never write unbounded decisions)
            if not r.bounded:
                violations.append(PolicyViolation(
                    violation_type="unbounded_decision",
                    decision_id=r.decision_id,
                    detail="bounded=False is not allowed",
                ))

        # Rapid recursive adaptation: same policy type changed > quota per week
        date_to_dt = lambda d: datetime.strptime(d, "%Y-%m-%d")
        today_dt = date_to_dt(today)
        for pt, dates in policy_dates.items():
            week_dates = [d for d in dates
                          if (today_dt - date_to_dt(d)).days <= 7]
            if len(week_dates) > c.max_decisions_per_type_per_week:
                violations.append(PolicyViolation(
                    violation_type="rapid_recursive_adaptation",
                    decision_id="batch",
                    detail=(
                        f"policy_type={pt} has {len(week_dates)} decisions "
                        f"in 7-day window > max={c.max_decisions_per_type_per_week}"
                    ),
                ))

        # Policy oscillation: same type changed back within cooldown window
        for pt, dates in policy_dates.items():
            sorted_dates = sorted(dates)
            for j in range(1, len(sorted_dates)):
                d0 = date_to_dt(sorted_dates[j - 1])
                d1 = date_to_dt(sorted_dates[j])
                gap = (d1 - d0).days
                if gap < c.cooldown_days_after_change:
                    violations.append(PolicyViolation(
                        violation_type="policy_oscillation",
                        decision_id="batch",
                        detail=(
                            f"policy_type={pt} changed on {sorted_dates[j-1]} "
                            f"and again on {sorted_dates[j]} "
                            f"(gap={gap}d < cooldown={c.cooldown_days_after_change}d)"
                        ),
                    ))

        has_critical = any(v.violation_type in self.CRITICAL_TYPES for v in violations)
        return PolicyIntegrityReport(
            n_records=len(records),
            n_violations=len(violations),
            violations=violations,
            passed=not has_critical,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ActivePolicyState — live signal integration layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActivePolicy:
    """Current effective value for a single policy type."""
    policy_type: str
    value: float
    decision_id: str
    eval_date: str
    confidence: float
    observation_count: int


class ActivePolicyState:
    """
    Reads the policy JSONL and surfaces the current effective policy for each type.
    Used by run_live_signal.py to apply bounded adjustments to runtime params.
    """

    def __init__(
        self,
        store_path: Path = DEFAULT_STORE_PATH,
        constraints: Optional[PolicyConstraints] = None,
        ttl_days: int = 30,
    ) -> None:
        self._store_path = store_path
        self._constraints = constraints or PolicyConstraints()
        self._ttl_days = ttl_days

    def load(self, as_of_date: Optional[str] = None) -> Dict[str, ActivePolicy]:
        """
        Load the latest valid policy for each type.
        Expired records (older than ttl_days) are ignored.
        Returns {} on any load error (fail-open for live signal: use defaults).
        """
        try:
            records = load_decisions(self._store_path)
        except Exception as exc:
            logger.warning("[POLICY] load failed (%s) — using defaults", exc)
            return {}

        today = as_of_date or _today_str()
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        cutoff = (today_dt - timedelta(days=self._ttl_days)).strftime("%Y-%m-%d")

        # Filter valid + non-expired
        valid = [
            r for r in records
            if r.eval_date >= cutoff and r.eval_date <= today
        ]

        # Latest per policy_type
        by_type: Dict[str, PolicyDecisionRecord] = {}
        for r in sorted(valid, key=lambda x: x.eval_date):
            by_type[r.policy_type] = r

        result: Dict[str, ActivePolicy] = {}
        for pt, rec in by_type.items():
            result[pt] = ActivePolicy(
                policy_type=pt,
                value=rec.new_value,
                decision_id=rec.decision_id,
                eval_date=rec.eval_date,
                confidence=rec.confidence,
                observation_count=rec.observation_count,
            )
        return result

    def apply_to_runtime_params(
        self,
        base_params: Dict[str, float],
        active: Dict[str, ActivePolicy],
        constraints: Optional[PolicyConstraints] = None,
    ) -> Dict[str, float]:
        """
        Apply active policy values to base runtime params with hard bounds.
        Returns a new dict — does not mutate base_params.
        All adjustments are bounded and logged.
        """
        c = constraints or self._constraints
        result = dict(base_params)

        for policy_type, ap in active.items():
            if policy_type not in base_params:
                logger.debug("[POLICY] no base param for %s — skipping", policy_type)
                continue
            base = base_params[policy_type]
            # Hard-bound the value regardless of stored value
            bounded_val = _bound_policy_value(policy_type, ap.value, c)
            # Cap total delta vs base
            delta = bounded_val - base
            max_delta = c.max_allocation_change_per_week
            if abs(delta) > max_delta:
                bounded_val = base + math.copysign(max_delta, delta)
                bounded_val = _bound_policy_value(policy_type, bounded_val, c)
            result[policy_type] = round(bounded_val, 6)
            logger.info(
                "[POLICY] %s: %.4f → %.4f (confidence=%.2f obs=%d date=%s)",
                policy_type, base, result[policy_type],
                ap.confidence, ap.observation_count, ap.eval_date,
            )
        return result


def _bound_policy_value(policy_type: str, value: float, c: PolicyConstraints) -> float:
    bounds = {
        POLICY_SLOT_ALLOCATION:      (c.slot_allocation_min,     c.slot_allocation_max),
        POLICY_SECTOR_EXPOSURE_CAP:  (c.sector_exposure_cap_min, c.sector_exposure_cap_max),
        POLICY_PORTFOLIO_HEAT_LIMIT: (c.portfolio_heat_min,      c.portfolio_heat_max),
        POLICY_HOLD_EXTENSION:       (0.0,                        1.0),
        POLICY_ENTRY_THROTTLING:     (c.entry_throttle_min,       c.entry_throttle_max),
        POLICY_RISK_SCALING:         (c.risk_scaling_min,          c.risk_scaling_max),
    }
    lo, hi = bounds.get(policy_type, (0.0, 1.0))
    return _clamp(value, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_decision(
    record: PolicyDecisionRecord,
    path: Path = DEFAULT_STORE_PATH,
) -> None:
    """
    Atomic append to JSONL. fsync durability. Never raises — analytics continues on error.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tf:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                tf.write(existing)
                tf.write(line + "\n")
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
        os.replace(tmp_path, str(path))
    except Exception as exc:
        logger.warning("[POLICY] append failed (%s) — continuing", exc)


def load_decisions(path: Path = DEFAULT_STORE_PATH) -> List[PolicyDecisionRecord]:
    """Load all records from JSONL. Corrupted lines are skipped with a warning."""
    if not path.exists():
        return []
    records: List[PolicyDecisionRecord] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(PolicyDecisionRecord.from_dict(d))
        except Exception as exc:
            logger.warning("[POLICY] parse error at line %d: %s", i + 1, exc)
    return records


def deduplicate_decisions(
    records: List[PolicyDecisionRecord],
) -> List[PolicyDecisionRecord]:
    """Deduplicate by decision_id. Stable sort by eval_date + policy_type."""
    seen: Dict[str, PolicyDecisionRecord] = {}
    for r in records:
        if r.decision_id not in seen:
            seen[r.decision_id] = r
    return sorted(seen.values(), key=lambda r: (r.eval_date, r.policy_type))


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration helper
# ─────────────────────────────────────────────────────────────────────────────

def run_policy_hook(
    store_path: Path = DEFAULT_STORE_PATH,
    base_params: Optional[Dict[str, float]] = None,
    as_of_date: Optional[str] = None,
    constraints: Optional[PolicyConstraints] = None,
) -> Dict[str, float]:
    """
    Lightweight hook for run_live_signal.py.
    Loads active policy, applies bounded adjustments to base_params, returns result.
    Fail-open: on any error returns base_params unchanged.

    Usage in run_live_signal.py:
        from src.runtime.policy.analytics_policy_bridge import run_policy_hook
        effective_params = run_policy_hook(
            store_path=POLICY_DECISIONS_FILE,
            base_params={
                "slot_allocation": 1.0,
                "sector_exposure_cap": MAX_SINGLE_WEIGHT,
                "portfolio_heat_limit": MAX_DD_LIMIT,
                "hold_extension_eligibility": 0.0,
                "entry_throttling": 1.0,
                "risk_scaling": 1.0,
            },
        )
    """
    bp = base_params or {}
    try:
        state = ActivePolicyState(
            store_path=store_path,
            constraints=constraints or PolicyConstraints(),
        )
        active = state.load(as_of_date=as_of_date)
        if not active:
            logger.debug("[POLICY] no active policy decisions — using base params")
            return dict(bp)
        result = state.apply_to_runtime_params(bp, active, constraints)
        logger.info("[POLICY] applied %d active policy adjustments", len(active))
        return result
    except Exception as exc:
        logger.warning("[POLICY] hook failed (%s) — using base params", exc)
        return dict(bp)
