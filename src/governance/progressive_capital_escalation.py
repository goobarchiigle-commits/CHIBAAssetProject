"""
progressive_capital_escalation.py — Progressive Capital Escalation Governance (Phase 6M)

Accelerates learning velocity by allowing successful ACTIVE experiments to receive
progressively larger capital allocations while preserving rollback protection.

Allocation ladder:
  Level 0:  0.5%  (initial)
  Level 1:  1.0%
  Level 2:  2.0%
  Level 3:  5.0%
  Level 4: 10.0%
  Level 5: 25.0%  (full-scale)

Decision states:
  HOLD             — stay at current level
  ESCALATE_LEVEL_1 — escalate to level 1
  ESCALATE_LEVEL_2 — escalate to level 2
  ESCALATE_LEVEL_3 — escalate to level 3
  FULL_SCALE       — escalate to level 4/5, or at level 5 performing well
  DEESCALATE       — drop one level

Persistence: 3 consecutive evaluations with conditions met before any level change.
Rollback governance is authoritative: ROLLED_BACK status overrides all escalation.

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open on errors
  never skip levels — one level per evaluation cycle
  O(n) in number of experiments
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"

# ── Decision states ────────────────────────────────────────────────────────────

DECISION_HOLD             = "HOLD"
DECISION_ESCALATE_LEVEL_1 = "ESCALATE_LEVEL_1"
DECISION_ESCALATE_LEVEL_2 = "ESCALATE_LEVEL_2"
DECISION_ESCALATE_LEVEL_3 = "ESCALATE_LEVEL_3"
DECISION_FULL_SCALE       = "FULL_SCALE"
DECISION_DEESCALATE       = "DEESCALATE"

# ── Allocation ladder (configurable constants) ─────────────────────────────────

_ALLOCATION_LEVELS: Dict[int, float] = {
    0: 0.005,   # 0.5%
    1: 0.010,   # 1.0%
    2: 0.020,   # 2.0%
    3: 0.050,   # 5.0%
    4: 0.100,   # 10.0%
    5: 0.250,   # 25.0%
}
_MAX_LEVEL = 5
_PERSISTENCE_DAYS = 3   # consecutive evaluations required before level change

# ── Escalation requirements per target level ───────────────────────────────────
# Keys: target_level → {obs_days, trades, expectancy_gt_ctrl, pnl_gt_ctrl, dd_lte_ctrl}

_ESCALATION_REQS: Dict[int, Dict[str, Any]] = {
    1: {"obs_days": 3,  "trades": 5,  "expectancy_gt": True,  "pnl_gt": False, "dd_lte": False},
    2: {"obs_days": 5,  "trades": 10, "expectancy_gt": True,  "pnl_gt": True,  "dd_lte": False},
    3: {"obs_days": 10, "trades": 20, "expectancy_gt": True,  "pnl_gt": True,  "dd_lte": False},
    4: {"obs_days": 15, "trades": 35, "expectancy_gt": True,  "pnl_gt": False, "dd_lte": True},
    5: {"obs_days": 20, "trades": 50, "expectancy_gt": True,  "pnl_gt": True,  "dd_lte": True},
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EscalationInput:
    experiment_id:     str
    evaluation_date:   str
    current_level:     int            # 0–5; current allocation_level in registry
    observation_days:  int
    trade_count:       int
    control_metrics:   Dict[str, float]
    experiment_metrics: Dict[str, float]


@dataclass
class EscalationDecision:
    experiment_id:              str
    evaluation_date:            str
    decision:                   str   # DECISION_* constant
    previous_level:             int
    new_level:                  int
    allocation_pct:             float
    reasons:                    List[str]
    consecutive_escalation_days:   int
    consecutive_deescalation_days: int
    schema_version:             str = SCHEMA_VERSION


@dataclass
class EscalationTelemetry:
    experiment_id:              str
    evaluation_date:            str
    decision:                   str
    previous_level:             int
    new_level:                  int
    allocation_pct:             float
    reasons:                    List[str]
    control_metrics:            Dict[str, float]
    experiment_metrics:         Dict[str, float]
    consecutive_escalation_days:   int
    consecutive_deescalation_days: int
    schema_version:             str = SCHEMA_VERSION


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, fallback: float = 0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _compute_days(from_date: str, to_date: str) -> int:
    from datetime import datetime
    try:
        d1 = datetime.fromisoformat(from_date)
        d2 = datetime.fromisoformat(to_date)
        return max(0, (d2 - d1).days)
    except Exception:
        return 0


def _allocation_pct(level: int) -> float:
    return _ALLOCATION_LEVELS.get(max(0, min(_MAX_LEVEL, level)), 0.005)


def _level_to_decision(target_level: int) -> str:
    _MAP = {
        1: DECISION_ESCALATE_LEVEL_1,
        2: DECISION_ESCALATE_LEVEL_2,
        3: DECISION_ESCALATE_LEVEL_3,
    }
    return _MAP.get(target_level, DECISION_FULL_SCALE)


def _load_all_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    try:
        if not path.exists():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[ESC] load_all_jsonl %s failed: %s", path, exc)
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
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
        Path(tmp).replace(path)
    except Exception as exc:
        logger.warning("[ESC] append_jsonl failed: %s", exc)


# ── Condition evaluation ───────────────────────────────────────────────────────

def _check_escalation_conditions(
    inp:          EscalationInput,
    target_level: int,
    ctrl:         Dict[str, float],
    exp:          Dict[str, float],
) -> Tuple[bool, List[str]]:
    """
    Return (conditions_met, reasons_list).
    On success: reasons_list = met condition names.
    On failure: reasons_list = unmet condition names.
    """
    reqs = _ESCALATION_REQS.get(target_level)
    if reqs is None:
        return False, [f"INVALID_TARGET_LEVEL_{target_level}"]

    met: List[str] = []
    failed: List[str] = []

    if inp.observation_days >= reqs["obs_days"]:
        met.append("OBS_DAYS_OK")
    else:
        failed.append(f"NEED_OBS_DAYS_{reqs['obs_days']}_HAVE_{inp.observation_days}")

    if inp.trade_count >= reqs["trades"]:
        met.append("TRADES_OK")
    else:
        failed.append(f"NEED_TRADES_{reqs['trades']}_HAVE_{inp.trade_count}")

    exp_exp  = _safe_float(exp.get("expectancy"), 0.0)
    ctrl_exp = _safe_float(ctrl.get("expectancy"), 0.0)
    if reqs["expectancy_gt"]:
        if exp_exp > ctrl_exp:
            met.append("EXPECTANCY_GT_CTRL")
        else:
            failed.append("EXPECTANCY_NOT_GT_CTRL")

    if reqs["pnl_gt"]:
        exp_pnl  = _safe_float(exp.get("pnl"), 0.0)
        ctrl_pnl = _safe_float(ctrl.get("pnl"), 0.0)
        if exp_pnl > ctrl_pnl:
            met.append("PNL_GT_CTRL")
        else:
            failed.append("PNL_NOT_GT_CTRL")

    if reqs["dd_lte"]:
        exp_dd  = abs(_safe_float(exp.get("max_drawdown"), 0.0))
        ctrl_dd = abs(_safe_float(ctrl.get("max_drawdown"), 0.0))
        if exp_dd <= ctrl_dd:
            met.append("DD_LTE_CTRL")
        else:
            failed.append("DD_GT_CTRL")

    if failed:
        return False, failed
    return True, met


def _is_weakened(ctrl: Dict[str, float], exp: Dict[str, float]) -> bool:
    """De-escalation condition: expectancy < control OR pnl < control."""
    exp_exp  = _safe_float(exp.get("expectancy"), 0.0)
    ctrl_exp = _safe_float(ctrl.get("expectancy"), 0.0)
    exp_pnl  = _safe_float(exp.get("pnl"), 0.0)
    ctrl_pnl = _safe_float(ctrl.get("pnl"), 0.0)
    return (exp_exp < ctrl_exp) or (exp_pnl < ctrl_pnl)


# ── Pure evaluation ────────────────────────────────────────────────────────────

def evaluate_escalation(
    inp:  EscalationInput,
    prev: Optional[EscalationDecision],
) -> EscalationDecision:
    """
    Pure function: evaluate one experiment's capital escalation.

    Priority:
      1. De-escalation (performance weakened) — takes precedence
      2. Full-scale steady-state (at max level, healthy)
      3. Escalation (conditions met for 3 consecutive days)
      4. HOLD

    Never raises. Never skips levels.
    """
    ctrl = inp.control_metrics
    exp  = inp.experiment_metrics

    prev_esc_consec   = prev.consecutive_escalation_days   if prev else 0
    prev_deesc_consec = prev.consecutive_deescalation_days if prev else 0

    # ── 1. De-escalation check ─────────────────────────────────────────────────
    if _is_weakened(ctrl, exp):
        new_deesc_consec = prev_deesc_consec + 1
        if new_deesc_consec >= _PERSISTENCE_DAYS and inp.current_level > 0:
            new_level = inp.current_level - 1
            return EscalationDecision(
                experiment_id=inp.experiment_id,
                evaluation_date=inp.evaluation_date,
                decision=DECISION_DEESCALATE,
                previous_level=inp.current_level,
                new_level=new_level,
                allocation_pct=_allocation_pct(new_level),
                reasons=[f"WEAKENED_{new_deesc_consec}_CONSECUTIVE_DAYS"],
                consecutive_escalation_days=0,
                consecutive_deescalation_days=0,
            )
        # Accumulating de-escalation pressure
        return EscalationDecision(
            experiment_id=inp.experiment_id,
            evaluation_date=inp.evaluation_date,
            decision=DECISION_HOLD,
            previous_level=inp.current_level,
            new_level=inp.current_level,
            allocation_pct=_allocation_pct(inp.current_level),
            reasons=[f"DEESC_ACCUMULATING_{new_deesc_consec}_OF_{_PERSISTENCE_DAYS}"],
            consecutive_escalation_days=0,
            consecutive_deescalation_days=new_deesc_consec,
        )

    # Not weakened → reset de-escalation counter
    new_deesc_consec = 0

    # ── 2. At maximum level ────────────────────────────────────────────────────
    if inp.current_level >= _MAX_LEVEL:
        return EscalationDecision(
            experiment_id=inp.experiment_id,
            evaluation_date=inp.evaluation_date,
            decision=DECISION_FULL_SCALE,
            previous_level=inp.current_level,
            new_level=_MAX_LEVEL,
            allocation_pct=_allocation_pct(_MAX_LEVEL),
            reasons=["AT_MAX_LEVEL"],
            consecutive_escalation_days=prev_esc_consec,
            consecutive_deescalation_days=0,
        )

    # ── 3. Escalation check ────────────────────────────────────────────────────
    next_level = inp.current_level + 1
    conditions_met, reasons = _check_escalation_conditions(inp, next_level, ctrl, exp)

    if conditions_met:
        new_esc_consec = prev_esc_consec + 1
        if new_esc_consec >= _PERSISTENCE_DAYS:
            new_level = next_level
            return EscalationDecision(
                experiment_id=inp.experiment_id,
                evaluation_date=inp.evaluation_date,
                decision=_level_to_decision(new_level),
                previous_level=inp.current_level,
                new_level=new_level,
                allocation_pct=_allocation_pct(new_level),
                reasons=reasons + [f"PERSISTED_{new_esc_consec}_DAYS"],
                consecutive_escalation_days=0,
                consecutive_deescalation_days=0,
            )
        # Accumulating escalation pressure
        return EscalationDecision(
            experiment_id=inp.experiment_id,
            evaluation_date=inp.evaluation_date,
            decision=DECISION_HOLD,
            previous_level=inp.current_level,
            new_level=inp.current_level,
            allocation_pct=_allocation_pct(inp.current_level),
            reasons=reasons + [f"ESC_ACCUMULATING_{new_esc_consec}_OF_{_PERSISTENCE_DAYS}"],
            consecutive_escalation_days=new_esc_consec,
            consecutive_deescalation_days=0,
        )

    # ── 4. HOLD — conditions not met ──────────────────────────────────────────
    return EscalationDecision(
        experiment_id=inp.experiment_id,
        evaluation_date=inp.evaluation_date,
        decision=DECISION_HOLD,
        previous_level=inp.current_level,
        new_level=inp.current_level,
        allocation_pct=_allocation_pct(inp.current_level),
        reasons=reasons,
        consecutive_escalation_days=0,
        consecutive_deescalation_days=0,
    )


# ── Telemetry ──────────────────────────────────────────────────────────────────

def build_escalation_telemetry(
    inp:      EscalationInput,
    decision: EscalationDecision,
) -> EscalationTelemetry:
    return EscalationTelemetry(
        experiment_id=inp.experiment_id,
        evaluation_date=inp.evaluation_date,
        decision=decision.decision,
        previous_level=decision.previous_level,
        new_level=decision.new_level,
        allocation_pct=decision.allocation_pct,
        reasons=list(decision.reasons),
        control_metrics=dict(inp.control_metrics),
        experiment_metrics=dict(inp.experiment_metrics),
        consecutive_escalation_days=decision.consecutive_escalation_days,
        consecutive_deescalation_days=decision.consecutive_deescalation_days,
    )


def append_escalation_telemetry(tel: EscalationTelemetry, path: Path) -> None:
    try:
        data = {
            "experiment_id":               tel.experiment_id,
            "evaluation_date":             tel.evaluation_date,
            "decision":                    tel.decision,
            "previous_level":              tel.previous_level,
            "new_level":                   tel.new_level,
            "allocation_pct":              tel.allocation_pct,
            "reasons":                     tel.reasons,
            "control_metrics":             tel.control_metrics,
            "experiment_metrics":          tel.experiment_metrics,
            "consecutive_escalation_days": tel.consecutive_escalation_days,
            "consecutive_deescalation_days": tel.consecutive_deescalation_days,
            "schema_version":              tel.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[ESC] append_escalation_telemetry failed: %s", exc)


def load_latest_escalation_decision(
    path:          Path,
    experiment_id: str,
) -> Optional[EscalationDecision]:
    """Load the most recent EscalationDecision for the given experiment_id."""
    latest_date = ""
    latest_rec: Optional[dict] = None
    for r in _load_all_jsonl(path):
        if str(r.get("experiment_id", "")) != experiment_id:
            continue
        d = str(r.get("evaluation_date", ""))
        if not d:
            continue
        if d > latest_date:
            latest_date = d
            latest_rec  = r
    if latest_rec is None:
        return None
    return EscalationDecision(
        experiment_id=experiment_id,
        evaluation_date=latest_date,
        decision=str(latest_rec.get("decision", DECISION_HOLD)),
        previous_level=_safe_int(latest_rec.get("previous_level"), 0),
        new_level=_safe_int(latest_rec.get("new_level"), 0),
        allocation_pct=_safe_float(latest_rec.get("allocation_pct"), 0.005),
        reasons=list(latest_rec.get("reasons") or []),
        consecutive_escalation_days=_safe_int(
            latest_rec.get("consecutive_escalation_days"), 0
        ),
        consecutive_deescalation_days=_safe_int(
            latest_rec.get("consecutive_deescalation_days"), 0
        ),
    )


# ── Registry integration ───────────────────────────────────────────────────────

def apply_escalation_decision(
    decision:      EscalationDecision,
    registry_path: Path,
) -> bool:
    """
    Apply escalation decision: update registry allocation_level + allocation_pct.
    HOLD → no registry change.
    FAIL_OPEN: returns False on error; registry unchanged.
    Logs [ESC_UP] on escalation, [ESC_DOWN] on de-escalation.
    """
    try:
        if decision.new_level == decision.previous_level:
            return True  # HOLD or FULL_SCALE steady-state — no registry change
        from src.experiments.active_experiment_registry import update_experiment_escalation
        event = (
            f"{decision.evaluation_date}:{decision.decision}"
            f":L{decision.previous_level}→L{decision.new_level}"
            f":{decision.allocation_pct:.3f}"
        )
        ok = update_experiment_escalation(
            experiment_id=decision.experiment_id,
            new_level=decision.new_level,
            new_allocation_pct=decision.allocation_pct,
            escalation_event=event,
            registry_path=registry_path,
        )
        if ok:
            if decision.new_level > decision.previous_level:
                logger.info("[ESC_UP] %s L%d→L%d alloc=%.1f%%",
                            decision.experiment_id,
                            decision.previous_level, decision.new_level,
                            decision.allocation_pct * 100)
            else:
                logger.info("[ESC_DOWN] %s L%d→L%d alloc=%.1f%%",
                            decision.experiment_id,
                            decision.previous_level, decision.new_level,
                            decision.allocation_pct * 100)
        return ok
    except Exception as exc:
        logger.warning("[ESC] apply_escalation_decision failed (fail-open): %s", exc)
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

def evaluate_all_experiment_escalations(
    registry_path:    Path,
    performance_path: Path,
    escalation_path:  Path,
    today:            str,
) -> List[EscalationDecision]:
    """
    Load all ACTIVE experiments, evaluate capital escalation, emit telemetry,
    and apply decisions to registry.

    FAIL_OPEN: returns [] on top-level error; individual failures logged and skipped.
    Rollback governance override: ROLLED_BACK registry status → experiment skipped.
    """
    try:
        from src.experiments.active_experiment_registry import (
            load_registry, STATUS_ACTIVE,
        )
        from src.analytics.experiment_performance import (
            load_performance_records, compute_performance_summary,
        )

        configs = load_registry(registry_path)
        active  = [c for c in configs if c.status == STATUS_ACTIVE]

        if not active:
            return []

        decisions: List[EscalationDecision] = []

        for exp in active:
            try:
                obs_days    = _compute_days(exp.start_date, today)
                all_records = load_performance_records(performance_path, exp.experiment_id)
                ctrl_metrics = compute_performance_summary(all_records, group="CONTROL")
                exp_metrics  = compute_performance_summary(all_records, group="EXPERIMENT")

                inp = EscalationInput(
                    experiment_id=exp.experiment_id,
                    evaluation_date=today,
                    current_level=exp.allocation_level,
                    observation_days=obs_days,
                    trade_count=_safe_int(exp_metrics.get("trades"), 0),
                    control_metrics=ctrl_metrics,
                    experiment_metrics=exp_metrics,
                )

                prev     = load_latest_escalation_decision(escalation_path, exp.experiment_id)
                decision = evaluate_escalation(inp, prev)

                tel = build_escalation_telemetry(inp, decision)
                append_escalation_telemetry(tel, escalation_path)
                apply_escalation_decision(decision, registry_path)

                decisions.append(decision)

                if decision.decision != DECISION_HOLD:
                    logger.info("[ESC] %s %s L%d→L%d",
                                exp.experiment_id, decision.decision,
                                decision.previous_level, decision.new_level)
                else:
                    logger.debug("[ESC] %s HOLD L%d", exp.experiment_id, decision.new_level)

            except Exception as exc:
                logger.warning("[ESC] experiment %s escalation failed: %s", exp.experiment_id, exc)

        return decisions

    except Exception as exc:
        logger.warning("[ESC] evaluate_all_experiment_escalations failed: %s", exc)
        return []
