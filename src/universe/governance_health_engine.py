"""
src/universe/governance_health_engine.py — Phase 3C: Governance Health Engine

Monitors universe governance pipeline health and manages operational modes.

Mode ladder:  NORMAL → WARNING → DEGRADED → SAFE_MODE → FREEZE

Each health check returns:
  HEALTHY          — check passed, sufficient data, no anomaly
  DEGRADED         — anomaly detected, sufficient data
  INSUFFICIENT_DATA — not enough observations (distinct from anomaly)

Mode hysteresis (MODE_CONFIRM_DAYS = 3):
  Any mode change confirms only after MODE_CONFIRM_DAYS consecutive observations
  at the target score level. Recovery requires RECOVERY_BUFFER above threshold.

Governance independence:
  Engine failure → FAIL_OPEN: execution continues with last known mode.
  Governance mode NEVER affects execution path.

Replay:
  governance_health.json contains all check results + mode state.
  Current-day mode judgment reproducible from the file alone.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Mode ladder (index 0 = best) ──────────────────────────────────────────────
MODE_NORMAL    = "NORMAL"
MODE_WARNING   = "WARNING"
MODE_DEGRADED  = "DEGRADED"
MODE_SAFE      = "SAFE_MODE"
MODE_FREEZE    = "FREEZE"

MODE_LADDER = [MODE_NORMAL, MODE_WARNING, MODE_DEGRADED, MODE_SAFE, MODE_FREEZE]

# ── Score thresholds (downgrade: score BELOW threshold → enter that mode) ─────
# Spec example: "score < 60 が3日連続 → SAFE_MODE" → SAFE_MODE threshold = 60
SCORE_THRESHOLDS: Dict[str, float] = {
    MODE_NORMAL:   80.0,
    MODE_WARNING:  70.0,
    MODE_DEGRADED: 60.0,
    MODE_SAFE:     30.0,
    MODE_FREEZE:    0.0,   # score < 30 → FREEZE
}

# ── Hysteresis ─────────────────────────────────────────────────────────────────
MODE_CONFIRM_DAYS: int  = 3     # consecutive observations before mode change confirms
RECOVERY_BUFFER: float  = 5.0  # upgrade requires score ≥ threshold + buffer

# ── Check status values ────────────────────────────────────────────────────────
STATUS_HEALTHY    = "HEALTHY"
STATUS_DEGRADED   = "DEGRADED"
STATUS_INSUF_DATA = "INSUFFICIENT_DATA"

_CHECK_SCORE_WEIGHTS = {
    STATUS_HEALTHY:    1.0,
    STATUS_INSUF_DATA: 0.5,   # uncertain → half credit
    STATUS_DEGRADED:   0.0,   # anomaly → no credit
}

_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GovernanceEnforcement:
    allow_promotions: bool
    allow_rotations: bool
    allow_new_demotions: bool
    allow_demotion_state_updates: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GovernanceEnforcement":
        return GovernanceEnforcement(
            allow_promotions=bool(d.get("allow_promotions", True)),
            allow_rotations=bool(d.get("allow_rotations", True)),
            allow_new_demotions=bool(d.get("allow_new_demotions", True)),
            allow_demotion_state_updates=bool(d.get("allow_demotion_state_updates", True)),
        )


_MODE_ENFORCEMENT: Dict[str, GovernanceEnforcement] = {
    MODE_NORMAL:   GovernanceEnforcement(True,  True,  True,  True),
    MODE_WARNING:  GovernanceEnforcement(True,  True,  True,  True),
    MODE_DEGRADED: GovernanceEnforcement(True,  True,  True,  True),
    MODE_SAFE:     GovernanceEnforcement(False, False, False, True),
    MODE_FREEZE:   GovernanceEnforcement(False, False, False, False),
}


# ─────────────────────────────────────────────────────────────────────────────
# Manual override
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManualOverride:
    override_mode: Optional[str]
    override_until: Optional[str]   # ISO datetime; None = active indefinitely
    override_reason: Optional[str]

    def is_active(self, now: str) -> bool:
        if not self.override_mode:
            return False
        if not self.override_until:
            return True
        try:
            exp = datetime.fromisoformat(self.override_until)
            cur = datetime.fromisoformat(now)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if cur.tzinfo is None:
                cur = cur.replace(tzinfo=timezone.utc)
            return cur < exp
        except Exception:
            return False

    @staticmethod
    def from_file(path: Path) -> Optional["ManualOverride"]:
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return ManualOverride(
                override_mode=d.get("override_mode"),
                override_until=d.get("override_until"),
                override_reason=d.get("override_reason"),
            )
        except Exception as exc:
            logger.warning("[GOVERNANCE_HEALTH] load_override failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_name: str
    status: str           # HEALTHY / DEGRADED / INSUFFICIENT_DATA
    value: float          # numeric summary for the check
    reason: str           # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CheckResult":
        return CheckResult(
            check_name=d["check_name"],
            status=d["status"],
            value=float(d.get("value", 0.0)),
            reason=d.get("reason", ""),
        )


@dataclass
class GovernanceHealthState:
    governance_mode: str
    mode_confirmed: bool
    pending_mode: Optional[str]
    pending_mode_days: int
    health_score: float
    checks: Dict[str, CheckResult]
    computed_at: str
    previous_live_size: int          # used for universe_churn check
    schema_version: int = _SCHEMA_VERSION
    enforcement: GovernanceEnforcement = None          # type: ignore[assignment]
    override_applied: bool = False
    override_reason: Optional[str] = None
    auto_freeze_mode: str = MODE_NORMAL                # from AutoFreezeEngine
    effective_mode: str = ""                           # max(governance_mode, auto_freeze_mode)
    effective_enforcement: GovernanceEnforcement = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.enforcement is None:
            self.enforcement = _MODE_ENFORCEMENT.get(
                self.governance_mode, _MODE_ENFORCEMENT[MODE_WARNING]
            )
        # effective_mode always recomputed for consistency
        self.effective_mode = _most_restrictive_mode(
            self.governance_mode, self.auto_freeze_mode
        )
        self.effective_enforcement = _MODE_ENFORCEMENT.get(
            self.effective_mode, _MODE_ENFORCEMENT[MODE_WARNING]
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = {k: v.to_dict() for k, v in self.checks.items()}
        # enforcement is serialized by asdict() as a nested dict
        return d

    @staticmethod
    def from_dict(d: dict) -> "GovernanceHealthState":
        mode = d.get("governance_mode", MODE_NORMAL)
        enf_data = d.get("enforcement")
        enforcement = (
            GovernanceEnforcement.from_dict(enf_data)
            if enf_data
            else _MODE_ENFORCEMENT.get(mode, _MODE_ENFORCEMENT[MODE_WARNING])
        )
        return GovernanceHealthState(
            governance_mode=mode,
            mode_confirmed=bool(d.get("mode_confirmed", True)),
            pending_mode=d.get("pending_mode"),
            pending_mode_days=int(d.get("pending_mode_days", 0)),
            health_score=float(d.get("health_score", 100.0)),
            checks={
                k: CheckResult.from_dict(v)
                for k, v in d.get("checks", {}).items()
            },
            computed_at=d.get("computed_at", ""),
            previous_live_size=int(d.get("previous_live_size", 0)),
            schema_version=int(d.get("schema_version", _SCHEMA_VERSION)),
            enforcement=enforcement,
            override_applied=bool(d.get("override_applied", False)),
            override_reason=d.get("override_reason"),
            auto_freeze_mode=d.get("auto_freeze_mode", MODE_NORMAL),
            # effective_mode / effective_enforcement recomputed by __post_init__
        )


@dataclass
class ModeTransitionRecord:
    timestamp: str
    old_mode: str
    new_mode: str
    health_score: float
    trigger_checks: List[str]   # check names that are DEGRADED and drove the transition
    confirm_day: int             # which confirm day this was (1, 2, or 3)
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class GovernanceHealthEngine:
    """
    Monitors governance pipeline health and manages the operational mode.

    Usage:
        engine = GovernanceHealthEngine(health_file, history_file, target_size=42)
        state  = engine.run(live_universe, shadow_universe, demotion_result, rotation_plan)
        engine.write_state(state)
    """

    def __init__(
        self,
        governance_health_file: Path,
        mode_history_file: Path,
        governance_override_file: Optional[Path] = None,
        target_universe_size: int = 42,
        mode_confirm_days: int = MODE_CONFIRM_DAYS,
        recovery_buffer: float = RECOVERY_BUFFER,
    ) -> None:
        self._health_file   = governance_health_file
        self._history_file  = mode_history_file
        self._override_file = governance_override_file
        self._target        = target_universe_size
        self._confirm_days  = mode_confirm_days
        self._recovery_buf  = recovery_buffer

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        live_universe: Dict[str, str],
        shadow_universe: Dict[str, str],
        demotion_result=None,    # Optional[DemotionActuationResult]
        rotation_plan=None,      # Optional[RotationPlan]
        auto_freeze_state=None,  # Optional[AutoFreezeState]
    ) -> GovernanceHealthState:
        """
        Run all health checks, compute score, apply mode hysteresis.
        effective_mode = max(governance_mode, auto_freeze_mode) — most restrictive wins.
        On any unhandled internal error → returns WARNING state (FAIL_OPEN, NORMAL禁止).
        """
        try:
            return self._run_internal(
                live_universe, shadow_universe,
                demotion_result, rotation_plan, auto_freeze_state,
            )
        except Exception as exc:
            logger.error("[GOVERNANCE_HEALTH] run() unhandled error → WARNING: %s", exc)
            return self._warning_fallback(str(exc))

    def _run_internal(
        self,
        live_universe: Dict[str, str],
        shadow_universe: Dict[str, str],
        demotion_result,
        rotation_plan,
        auto_freeze_state=None,
    ) -> GovernanceHealthState:
        now = _now_utc()
        prev = self.load_state()

        # ── Run checks ────────────────────────────────────────────────────────
        checks: Dict[str, CheckResult] = {}
        checks.update(self._check_universe_size(live_universe))
        checks.update(self._check_demotion_pipeline(demotion_result))
        checks.update(self._check_pending_exit_age(demotion_result))
        checks.update(self._check_shadow_coverage(live_universe, shadow_universe, rotation_plan))
        checks.update(self._check_rotation_plan_age(rotation_plan, now))
        checks.update(self._check_rotation_coverage(rotation_plan))
        checks.update(self._check_removal_activity(demotion_result))
        checks.update(self._check_recovery_rate(demotion_result))
        checks.update(self._check_candidate_freshness(shadow_universe, rotation_plan))
        checks.update(self._check_universe_churn(live_universe, prev))
        checks.update(self._check_governance_freshness(prev, now))

        # ── Score ─────────────────────────────────────────────────────────────
        health_score = _compute_score(checks)
        degraded_checks = [k for k, v in checks.items() if v.status == STATUS_DEGRADED]

        # ── Mode transition ───────────────────────────────────────────────────
        prev_mode     = prev.governance_mode if prev else MODE_NORMAL
        prev_pending  = prev.pending_mode    if prev else None
        prev_days     = prev.pending_mode_days if prev else 0

        new_mode, new_pending, new_days = _compute_mode_transition(
            current_mode    = prev_mode,
            current_pending = prev_pending,
            pending_days    = prev_days,
            health_score    = health_score,
            confirm_days    = self._confirm_days,
            recovery_buffer = self._recovery_buf,
        )

        mode_changed   = (new_mode != prev_mode)
        mode_confirmed = (new_pending is None)

        # ── Manual override ───────────────────────────────────────────────────
        override_applied = False
        override_reason: Optional[str] = None
        final_mode = new_mode

        if self._override_file is not None:
            override = ManualOverride.from_file(self._override_file)
            if override and override.is_active(now) and override.override_mode in MODE_LADDER:
                final_mode       = override.override_mode
                override_applied = True
                override_reason  = override.override_reason
                override_record  = ModeTransitionRecord(
                    timestamp=now,
                    old_mode=new_mode,
                    new_mode=final_mode,
                    health_score=health_score,
                    trigger_checks=["MANUAL_OVERRIDE"],
                    confirm_day=0,
                )
                self.write_transition(override_record)
                logger.warning(
                    "[GOVERNANCE_HEALTH] manual_override: computed=%s → override=%s reason=%s",
                    new_mode, final_mode, override_reason,
                )

        # ── Enforcement ───────────────────────────────────────────────────────
        enforcement = _MODE_ENFORCEMENT.get(final_mode, _MODE_ENFORCEMENT[MODE_WARNING])

        # ── Auto-freeze integration ────────────────────────────────────────────
        af_mode = (
            auto_freeze_state.auto_freeze_mode
            if auto_freeze_state is not None
            else MODE_NORMAL
        )

        state = GovernanceHealthState(
            governance_mode=final_mode,
            mode_confirmed=mode_confirmed,
            pending_mode=new_pending,
            pending_mode_days=new_days,
            health_score=health_score,
            checks=checks,
            computed_at=now,
            previous_live_size=len(live_universe),
            enforcement=enforcement,
            override_applied=override_applied,
            override_reason=override_reason,
            auto_freeze_mode=af_mode,
            # effective_mode and effective_enforcement computed by __post_init__
        )

        # ── Audit transition ──────────────────────────────────────────────────
        if mode_changed or (new_pending and new_days == 1):
            transition = ModeTransitionRecord(
                timestamp=now,
                old_mode=prev_mode,
                new_mode=new_mode if mode_changed else (new_pending or prev_mode),
                health_score=health_score,
                trigger_checks=degraded_checks,
                confirm_day=self._confirm_days if mode_changed else new_days,
            )
            self.write_transition(transition)
            if mode_changed:
                logger.warning(
                    "[GOVERNANCE_HEALTH] mode_change: %s → %s score=%.1f triggers=%s",
                    prev_mode, new_mode, health_score, degraded_checks,
                )
            else:
                logger.info(
                    "[GOVERNANCE_HEALTH] mode_pending: %s day=%d/%d score=%.1f",
                    new_pending, new_days, self._confirm_days, health_score,
                )

        logger.info(
            "[GOVERNANCE_HEALTH] mode=%s score=%.1f healthy=%d degraded=%d insuf=%d enforce=%s",
            final_mode, health_score,
            sum(1 for c in checks.values() if c.status == STATUS_HEALTHY),
            len(degraded_checks),
            sum(1 for c in checks.values() if c.status == STATUS_INSUF_DATA),
            enforcement,
        )

        return state

    def _warning_fallback(self, error_msg: str) -> GovernanceHealthState:
        """Return WARNING-mode state when run() itself fails (NORMAL禁止)."""
        now = _now_utc()
        check = CheckResult(
            check_name="engine_error",
            status=STATUS_DEGRADED,
            value=0.0,
            reason=f"engine_failure:{error_msg[:120]}",
        )
        return GovernanceHealthState(
            governance_mode=MODE_WARNING,
            mode_confirmed=True,
            pending_mode=None,
            pending_mode_days=0,
            health_score=0.0,
            checks={"engine_error": check},
            computed_at=now,
            previous_live_size=0,
            enforcement=_MODE_ENFORCEMENT[MODE_WARNING],
            override_applied=False,
            override_reason=None,
        )

    def load_enforcement(self) -> GovernanceEnforcement:
        """
        Load enforcement from saved governance_health.json.
        Returns NORMAL (permissive) on first run (no prior state).
        Returns WARNING enforcement on corrupt file.
        """
        state = self.load_state()
        if state is None:
            return _MODE_ENFORCEMENT[MODE_NORMAL]
        return _MODE_ENFORCEMENT.get(state.governance_mode, _MODE_ENFORCEMENT[MODE_WARNING])

    def write_state(self, state: GovernanceHealthState) -> None:
        """Atomic write of governance_health.json. FAIL_OPEN."""
        try:
            _write_atomic(
                self._health_file,
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            )
        except Exception as exc:
            logger.warning("[GOVERNANCE_HEALTH] write_state failed (FAIL_OPEN): %s", exc)

    def write_transition(self, record: ModeTransitionRecord) -> None:
        """Append mode transition record. FAIL_OPEN."""
        try:
            _append_jsonl(self._history_file, record.to_dict())
        except Exception as exc:
            logger.warning("[GOVERNANCE_HEALTH] write_transition failed (FAIL_OPEN): %s", exc)

    def load_state(self) -> Optional[GovernanceHealthState]:
        """Load governance_health.json. Returns None if absent or corrupt."""
        if not self._health_file.exists():
            return None
        try:
            return GovernanceHealthState.from_dict(
                json.loads(self._health_file.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            logger.warning("[GOVERNANCE_HEALTH] load_state failed: %s", exc)
            return None

    # ── Health checks ─────────────────────────────────────────────────────────

    def _check_universe_size(self, live: dict) -> Dict[str, CheckResult]:
        name = "universe_size"
        try:
            if not live:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "universe_empty")}
            size = len(live)
            off  = abs(size - self._target)
            if off <= 1:
                return {name: CheckResult(name, STATUS_HEALTHY, float(size), f"size={size}")}
            if off <= 2:
                return {name: CheckResult(name, STATUS_INSUF_DATA, float(size), f"size={size} off_by={off}")}
            return {name: CheckResult(name, STATUS_DEGRADED, float(size), f"size={size} off_by={off} target={self._target}")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_demotion_pipeline(self, dem_result) -> Dict[str, CheckResult]:
        name = "demotion_pipeline"
        try:
            if dem_result is None:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_demotion_result")}
            n = len(dem_result.demotion_candidates) + dem_result.pending_exit_count
            if n == 0:
                return {name: CheckResult(name, STATUS_HEALTHY, 0.0, "no_active_demotions")}
            if n <= 2:
                return {name: CheckResult(name, STATUS_INSUF_DATA, float(n), f"active_demotions={n}")}
            return {name: CheckResult(name, STATUS_DEGRADED, float(n), f"active_demotions={n} high_load")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_pending_exit_age(self, dem_result) -> Dict[str, CheckResult]:
        name = "pending_exit_age"
        try:
            if dem_result is None:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_demotion_result")}
            if dem_result.pending_exit_count == 0:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_pending_exits")}
            avg = dem_result.average_days_pending_exit
            if avg < 7:
                return {name: CheckResult(name, STATUS_HEALTHY, avg, f"avg_days={avg:.1f}")}
            if avg < 14:
                return {name: CheckResult(name, STATUS_INSUF_DATA, avg, f"avg_days={avg:.1f} moderate")}
            return {name: CheckResult(name, STATUS_DEGRADED, avg, f"avg_days={avg:.1f} overstayed")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_shadow_coverage(self, live: dict, shadow: dict, rotation_plan) -> Dict[str, CheckResult]:
        name = "shadow_coverage"
        try:
            shadow_only = len([s for s in shadow if s not in live])
            deficit = max(0, self._target - len(live))
            if deficit == 0:
                return {name: CheckResult(name, STATUS_HEALTHY, float(shadow_only), "no_deficit")}
            if shadow_only == 0:
                return {name: CheckResult(name, STATUS_DEGRADED, 0.0, f"no_candidates deficit={deficit}")}
            if shadow_only >= deficit:
                return {name: CheckResult(name, STATUS_HEALTHY, float(shadow_only), f"coverage={shadow_only}/{deficit}")}
            return {name: CheckResult(name, STATUS_INSUF_DATA, float(shadow_only), f"partial_coverage={shadow_only}/{deficit}")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_rotation_plan_age(self, plan, now: str) -> Dict[str, CheckResult]:
        name = "rotation_plan_age"
        try:
            if plan is None:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_plan")}
            age = _hours_since(plan.created_at, now) / 24.0
            if age < 1.0:
                return {name: CheckResult(name, STATUS_HEALTHY, round(age, 2), f"age={age:.1f}d fresh")}
            if age < 3.0:
                return {name: CheckResult(name, STATUS_INSUF_DATA, round(age, 2), f"age={age:.1f}d recent")}
            return {name: CheckResult(name, STATUS_DEGRADED, round(age, 2), f"age={age:.1f}d stale_plan")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_rotation_coverage(self, plan) -> Dict[str, CheckResult]:
        name = "rotation_coverage"
        try:
            if plan is None:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_plan")}
            if plan.deficit == 0:
                return {name: CheckResult(name, STATUS_HEALTHY, 0.0, "no_deficit")}
            coverage = len(plan.symbols_to_add) / plan.deficit
            if coverage >= 1.0:
                return {name: CheckResult(name, STATUS_HEALTHY, coverage, "fully_covered")}
            if coverage >= 0.5:
                return {name: CheckResult(name, STATUS_INSUF_DATA, coverage, f"partial={coverage:.0%}")}
            return {name: CheckResult(name, STATUS_DEGRADED, coverage, f"under_covered={coverage:.0%}")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_removal_activity(self, dem_result) -> Dict[str, CheckResult]:
        name = "removal_activity"
        try:
            if dem_result is None:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_demotion_result")}
            n = len(dem_result.symbols_to_remove)
            if n == 0:
                return {name: CheckResult(name, STATUS_HEALTHY, 0.0, "no_removals")}
            if n == 1:
                return {name: CheckResult(name, STATUS_INSUF_DATA, float(n), f"removed={n}")}
            return {name: CheckResult(name, STATUS_DEGRADED, float(n), f"removed={n} high_churn")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_recovery_rate(self, dem_result) -> Dict[str, CheckResult]:
        name = "recovery_rate"
        try:
            if dem_result is None:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_demotion_result")}
            total = dem_result.recovered_count + len(dem_result.demotion_candidates)
            if total == 0:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_transitions")}
            rate = dem_result.recovered_count / total
            if rate >= 0.3 or dem_result.recovered_count > 0:
                return {name: CheckResult(name, STATUS_HEALTHY, round(rate, 3), f"recovery_rate={rate:.0%}")}
            if len(dem_result.demotion_candidates) <= 1:
                return {name: CheckResult(name, STATUS_INSUF_DATA, round(rate, 3), f"low_volume candidates={len(dem_result.demotion_candidates)}")}
            return {name: CheckResult(name, STATUS_DEGRADED, round(rate, 3), f"no_recovery candidates={len(dem_result.demotion_candidates)}")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_candidate_freshness(self, shadow: dict, plan) -> Dict[str, CheckResult]:
        name = "candidate_freshness"
        try:
            if plan is None or not shadow:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_plan_or_shadow")}
            rejected = plan.rejected_candidates
            total    = len(plan.replacement_candidates) + len(rejected)
            if total == 0:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_candidates")}
            stale_ratio = len(rejected) / total
            if stale_ratio == 0.0:
                return {name: CheckResult(name, STATUS_HEALTHY, 0.0, "all_fresh")}
            if stale_ratio <= 0.3:
                return {name: CheckResult(name, STATUS_INSUF_DATA, round(stale_ratio, 3), f"stale={stale_ratio:.0%}")}
            return {name: CheckResult(name, STATUS_DEGRADED, round(stale_ratio, 3), f"high_stale={stale_ratio:.0%}")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_universe_churn(self, live: dict, prev: Optional[GovernanceHealthState]) -> Dict[str, CheckResult]:
        name = "universe_churn"
        try:
            if prev is None or prev.previous_live_size == 0:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "no_previous_state")}
            drop = prev.previous_live_size - len(live)
            if drop <= 0:
                return {name: CheckResult(name, STATUS_HEALTHY, float(abs(drop)), "stable_or_growing")}
            if drop <= 1:
                return {name: CheckResult(name, STATUS_INSUF_DATA, float(drop), f"small_drop={drop}")}
            return {name: CheckResult(name, STATUS_DEGRADED, float(drop), f"size_drop={drop} churn")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}

    def _check_governance_freshness(self, prev: Optional[GovernanceHealthState], now: str) -> Dict[str, CheckResult]:
        name = "governance_freshness"
        try:
            if prev is None or not prev.computed_at:
                return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, "first_run")}
            age_h = _hours_since(prev.computed_at, now)
            if age_h < 36:
                return {name: CheckResult(name, STATUS_HEALTHY, round(age_h, 1), f"age={age_h:.0f}h regular")}
            if age_h < 72:
                return {name: CheckResult(name, STATUS_INSUF_DATA, round(age_h, 1), f"age={age_h:.0f}h delayed")}
            return {name: CheckResult(name, STATUS_DEGRADED, round(age_h, 1), f"age={age_h:.0f}h stale")}
        except Exception as exc:
            return {name: CheckResult(name, STATUS_INSUF_DATA, 0.0, f"error:{exc}")}


# ─────────────────────────────────────────────────────────────────────────────
# Mode transition logic
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_raw_mode(score: float) -> str:
    """Map health score to mode via threshold ladder."""
    if score >= SCORE_THRESHOLDS[MODE_NORMAL]:    return MODE_NORMAL
    if score >= SCORE_THRESHOLDS[MODE_WARNING]:   return MODE_WARNING
    if score >= SCORE_THRESHOLDS[MODE_DEGRADED]:  return MODE_DEGRADED
    if score >= SCORE_THRESHOLDS[MODE_SAFE]:      return MODE_SAFE
    return MODE_FREEZE


def _compute_mode_transition(
    current_mode: str,
    current_pending: Optional[str],
    pending_days: int,
    health_score: float,
    confirm_days: int,
    recovery_buffer: float,
) -> Tuple[str, Optional[str], int]:
    """
    Returns (confirmed_mode, new_pending_mode, new_pending_days).
    Applies hysteresis: upgrades require RECOVERY_BUFFER above threshold.
    """
    raw_target = _score_to_raw_mode(health_score)

    current_idx = MODE_LADDER.index(current_mode)
    raw_idx     = MODE_LADDER.index(raw_target)
    is_upgrade  = raw_idx < current_idx   # moving toward NORMAL

    if is_upgrade:
        # Apply recovery buffer: harder to confirm improvement
        target = _score_to_raw_mode(health_score - recovery_buffer)
    else:
        target = raw_target

    if target == current_mode:
        # No change needed
        return current_mode, None, 0

    # Accumulate pending days
    if current_pending == target:
        new_days = pending_days + 1
    else:
        new_days = 1

    if new_days >= confirm_days:
        return target, None, 0
    else:
        return current_mode, target, new_days


def _most_restrictive_mode(mode_a: str, mode_b: str) -> str:
    """Return the more restrictive (higher ladder index) of two governance modes."""
    idx_a = MODE_LADDER.index(mode_a) if mode_a in MODE_LADDER else 0
    idx_b = MODE_LADDER.index(mode_b) if mode_b in MODE_LADDER else 0
    return MODE_LADDER[max(idx_a, idx_b)]


def _compute_score(checks: Dict[str, CheckResult]) -> float:
    """
    Health score [0, 100]:
      HEALTHY=1.0, INSUFFICIENT_DATA=0.5, DEGRADED=0.0
    """
    if not checks:
        return 100.0
    total  = len(checks)
    earned = sum(_CHECK_SCORE_WEIGHTS.get(c.status, 0.0) for c in checks.values())
    return round(earned / total * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_since(iso_str: str, now: str) -> float:
    try:
        t0 = datetime.fromisoformat(iso_str)
        t1 = datetime.fromisoformat(now)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
        return max(0.0, (t1 - t0).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
