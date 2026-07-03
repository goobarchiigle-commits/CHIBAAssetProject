"""
automatic_rollback.py — Automatic Rollback Governance (Phase 6L-Lite)

Fail-safe governance layer that automatically identifies underperforming
ACTIVE experiments and transitions them to rollback states.

Decision states:
  HEALTHY             — no triggers or insufficient sample
  REVIEW_REQUIRED     — 1 trigger condition met
  ROLLBACK_CANDIDATE  — 2+ trigger conditions met
  ROLLED_BACK         — ROLLBACK_CANDIDATE for >= 3 consecutive days

Trigger conditions (require minimum sample first):
  A: experiment_expectancy < control_expectancy - 0.25
  B: experiment_avg_return  < control_avg_return  - 0.01 (1%)
  C: abs(experiment_max_drawdown) > abs(control_max_drawdown) * 1.5

Persistence:
  ROLLBACK_CANDIDATE for _ROLLBACK_PERSISTENCE_DAYS consecutive days → ROLLED_BACK

Fail-safe:
  Any error → no state change, experiment remains ACTIVE.

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open on errors
  O(n) in number of experiments × JSONL lines
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"

DECISION_HEALTHY            = "HEALTHY"
DECISION_REVIEW_REQUIRED    = "REVIEW_REQUIRED"
DECISION_ROLLBACK_CANDIDATE = "ROLLBACK_CANDIDATE"
DECISION_ROLLED_BACK        = "ROLLED_BACK"

_MIN_TRADES          = 30
_MIN_OBS_DAYS        = 21
_ROLLBACK_PERSISTENCE_DAYS = 3
_REVIEW_TRIGGERS_MIN     = 1
_CANDIDATE_TRIGGERS_MIN  = 2

# Trigger thresholds
_EXPECTANCY_THRESHOLD = 0.25   # R-units below control
_AVG_RETURN_THRESHOLD = 0.01   # 1.0% (decimal) below control
# Drawdown: experiment magnitude > control magnitude × 1.5

_REASON_INSUFFICIENT   = "INSUFFICIENT_SAMPLE"
_REASON_LOW_EXPECTANCY = "LOW_EXPECTANCY"
_REASON_LOW_AVG_RETURN = "LOW_AVG_RETURN"
_REASON_EXCESS_DD      = "EXCESS_DRAWDOWN"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RollbackEvaluationInput:
    experiment_id:      str
    evaluation_date:    str
    trade_count:        int           # EXPERIMENT trade count
    observation_days:   int           # days since experiment start
    control_metrics:    Dict[str, float]
    experiment_metrics: Dict[str, float]


@dataclass
class RollbackDecision:
    experiment_id:             str
    evaluation_date:           str
    decision:                  str    # DECISION_* constant
    trigger_reasons:           List[str]
    consecutive_candidate_days: int
    schema_version:            str = SCHEMA_VERSION


@dataclass
class RollbackTelemetry:
    experiment_id:             str
    evaluation_date:           str
    decision:                  str
    trigger_reasons:           List[str]
    control_metrics:           Dict[str, float]
    experiment_metrics:        Dict[str, float]
    consecutive_candidate_days: int = 0
    schema_version:            str = SCHEMA_VERSION


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
    """Return (to_date - from_date).days. Returns 0 on error."""
    try:
        d1 = datetime.fromisoformat(from_date)
        d2 = datetime.fromisoformat(to_date)
        return max(0, (d2 - d1).days)
    except Exception:
        return 0


def _load_all_jsonl(path: Path) -> List[dict]:
    """Load all valid JSON lines. FAIL_OPEN → empty list."""
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
        logger.debug("[RB] load_all_jsonl %s failed: %s", path, exc)
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic append-only JSONL write. FAIL_OPEN."""
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
        logger.warning("[RB] append_jsonl failed: %s", exc)


# ── Decision loading ───────────────────────────────────────────────────────────

def load_latest_decision(
    rollback_path:  Path,
    experiment_id:  str,
) -> Optional[RollbackDecision]:
    """Load the most recent RollbackDecision for the given experiment_id."""
    latest_date: str = ""
    latest_rec:  Optional[dict] = None
    for r in _load_all_jsonl(rollback_path):
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
    return RollbackDecision(
        experiment_id=experiment_id,
        evaluation_date=latest_date,
        decision=str(latest_rec.get("decision", DECISION_HEALTHY)),
        trigger_reasons=list(latest_rec.get("trigger_reasons", [])),
        consecutive_candidate_days=_safe_int(latest_rec.get("consecutive_candidate_days"), 0),
    )


# ── Pure evaluation ────────────────────────────────────────────────────────────

def evaluate_experiment(
    inp:           RollbackEvaluationInput,
    prev_decision: Optional[RollbackDecision],
) -> RollbackDecision:
    """
    Pure function: evaluate one experiment against rollback criteria.

    Returns a RollbackDecision. Never raises.
    """
    # ── Minimum sample protection ──────────────────────────────────────────────
    if inp.trade_count < _MIN_TRADES or inp.observation_days < _MIN_OBS_DAYS:
        return RollbackDecision(
            experiment_id=inp.experiment_id,
            evaluation_date=inp.evaluation_date,
            decision=DECISION_HEALTHY,
            trigger_reasons=[_REASON_INSUFFICIENT],
            consecutive_candidate_days=0,
        )

    ctrl = inp.control_metrics
    exp  = inp.experiment_metrics

    # ── Evaluate trigger conditions ────────────────────────────────────────────
    triggers: List[str] = []

    # A: expectancy
    ctrl_exp = _safe_float(ctrl.get("expectancy"), 0.0)
    exp_exp  = _safe_float(exp.get("expectancy"),  0.0)
    if exp_exp < ctrl_exp - _EXPECTANCY_THRESHOLD:
        triggers.append(_REASON_LOW_EXPECTANCY)

    # B: avg_return (1% = 0.01 in decimal)
    ctrl_ret = _safe_float(ctrl.get("avg_return"), 0.0)
    exp_ret  = _safe_float(exp.get("avg_return"),  0.0)
    if exp_ret < ctrl_ret - _AVG_RETURN_THRESHOLD:
        triggers.append(_REASON_LOW_AVG_RETURN)

    # C: max_drawdown magnitude
    ctrl_dd_mag = abs(_safe_float(ctrl.get("max_drawdown"), 0.0))
    exp_dd_mag  = abs(_safe_float(exp.get("max_drawdown"),  0.0))
    if exp_dd_mag > ctrl_dd_mag * 1.5:
        triggers.append(_REASON_EXCESS_DD)

    n_triggers = len(triggers)

    # ── Classify ───────────────────────────────────────────────────────────────
    if n_triggers >= _CANDIDATE_TRIGGERS_MIN:
        base = DECISION_ROLLBACK_CANDIDATE
    elif n_triggers >= _REVIEW_TRIGGERS_MIN:
        base = DECISION_REVIEW_REQUIRED
    else:
        base = DECISION_HEALTHY

    # ── Persistence check for ROLLBACK_CANDIDATE → ROLLED_BACK ────────────────
    if base == DECISION_ROLLBACK_CANDIDATE:
        prev_consec = (
            prev_decision.consecutive_candidate_days
            if (prev_decision and prev_decision.decision == DECISION_ROLLBACK_CANDIDATE)
            else 0
        )
        new_consec = prev_consec + 1
        final_decision = (
            DECISION_ROLLED_BACK
            if new_consec >= _ROLLBACK_PERSISTENCE_DAYS
            else DECISION_ROLLBACK_CANDIDATE
        )
        return RollbackDecision(
            experiment_id=inp.experiment_id,
            evaluation_date=inp.evaluation_date,
            decision=final_decision,
            trigger_reasons=triggers,
            consecutive_candidate_days=new_consec,
        )

    return RollbackDecision(
        experiment_id=inp.experiment_id,
        evaluation_date=inp.evaluation_date,
        decision=base,
        trigger_reasons=triggers,
        consecutive_candidate_days=0,
    )


# ── Telemetry builder ──────────────────────────────────────────────────────────

def build_rollback_telemetry(
    inp:      RollbackEvaluationInput,
    decision: RollbackDecision,
) -> RollbackTelemetry:
    """Build a RollbackTelemetry record from evaluation inputs and decision."""
    return RollbackTelemetry(
        experiment_id=inp.experiment_id,
        evaluation_date=inp.evaluation_date,
        decision=decision.decision,
        trigger_reasons=list(decision.trigger_reasons),
        control_metrics=dict(inp.control_metrics),
        experiment_metrics=dict(inp.experiment_metrics),
        consecutive_candidate_days=decision.consecutive_candidate_days,
    )


def append_rollback_telemetry(tel: RollbackTelemetry, path: Path) -> None:
    """Serialize one RollbackTelemetry to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "experiment_id":             tel.experiment_id,
            "evaluation_date":           tel.evaluation_date,
            "decision":                  tel.decision,
            "trigger_reasons":           tel.trigger_reasons,
            "control_metrics":           tel.control_metrics,
            "experiment_metrics":        tel.experiment_metrics,
            "consecutive_candidate_days": tel.consecutive_candidate_days,
            "schema_version":            tel.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[RB] append_rollback_telemetry failed: %s", exc)


# ── Registry integration ───────────────────────────────────────────────────────

def apply_rollback_decision(
    decision:      RollbackDecision,
    registry_path: Path,
) -> bool:
    """
    Apply a rollback decision by updating the experiment registry status.

    HEALTHY     → no change
    REVIEW_REQUIRED    → status = STATUS_REVIEW_REQUIRED
    ROLLBACK_CANDIDATE → status = STATUS_ROLLBACK_CANDIDATE
    ROLLED_BACK        → status = STATUS_ROLLED_BACK

    Fail-safe: returns False on any error; experiment remains ACTIVE.
    Logs [RB_REVIEW], [RB_CANDIDATE], or [RB_APPLIED] on state change.
    """
    try:
        from src.experiments.active_experiment_registry import (
            STATUS_REVIEW_REQUIRED, STATUS_ROLLBACK_CANDIDATE, STATUS_ROLLED_BACK,
            update_experiment_status,
        )
        d = decision.decision
        if d == DECISION_HEALTHY:
            return True
        if d == DECISION_REVIEW_REQUIRED:
            logger.info("[RB_REVIEW] %s → REVIEW_REQUIRED", decision.experiment_id)
            return update_experiment_status(decision.experiment_id, STATUS_REVIEW_REQUIRED, registry_path)
        if d == DECISION_ROLLBACK_CANDIDATE:
            logger.info("[RB_CANDIDATE] %s → ROLLBACK_CANDIDATE (day %d/%d)",
                        decision.experiment_id,
                        decision.consecutive_candidate_days,
                        _ROLLBACK_PERSISTENCE_DAYS)
            return update_experiment_status(decision.experiment_id, STATUS_ROLLBACK_CANDIDATE, registry_path)
        if d == DECISION_ROLLED_BACK:
            logger.info("[RB_APPLIED] %s → ROLLED_BACK (persisted %d days)",
                        decision.experiment_id,
                        decision.consecutive_candidate_days)
            return update_experiment_status(decision.experiment_id, STATUS_ROLLED_BACK, registry_path)
        return True
    except Exception as exc:
        logger.warning("[RB] apply_rollback_decision failed (fail-safe, no change): %s", exc)
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

def evaluate_all_active_experiments(
    registry_path:    Path,
    performance_path: Path,
    rollback_path:    Path,
    today:            str,
) -> List[RollbackDecision]:
    """
    Load all ACTIVE experiments, evaluate each against rollback criteria,
    append telemetry, and apply decisions to registry.

    FAIL_OPEN: returns [] on top-level error; individual failures logged and skipped.
    """
    try:
        from src.experiments.active_experiment_registry import (
            load_registry,
            STATUS_ACTIVE, STATUS_REVIEW_REQUIRED, STATUS_ROLLBACK_CANDIDATE,
        )
        from src.analytics.experiment_performance import (
            load_performance_records, compute_performance_summary,
        )

        configs = load_registry(registry_path)
        # Evaluate ACTIVE + in-progress rollback states so persistence counter advances
        _evaluable = {STATUS_ACTIVE, STATUS_REVIEW_REQUIRED, STATUS_ROLLBACK_CANDIDATE}
        active  = [c for c in configs if c.status in _evaluable]

        if not active:
            return []

        decisions: List[RollbackDecision] = []

        for exp in active:
            try:
                obs_days    = _compute_days(exp.start_date, today)
                all_records = load_performance_records(performance_path, exp.experiment_id)
                ctrl_metrics = compute_performance_summary(all_records, group="CONTROL")
                exp_metrics  = compute_performance_summary(all_records, group="EXPERIMENT")

                inp = RollbackEvaluationInput(
                    experiment_id=exp.experiment_id,
                    evaluation_date=today,
                    trade_count=_safe_int(exp_metrics.get("trades"), 0),
                    observation_days=obs_days,
                    control_metrics=ctrl_metrics,
                    experiment_metrics=exp_metrics,
                )

                prev     = load_latest_decision(rollback_path, exp.experiment_id)
                decision = evaluate_experiment(inp, prev)

                tel = build_rollback_telemetry(inp, decision)
                append_rollback_telemetry(tel, rollback_path)
                apply_rollback_decision(decision, registry_path)

                decisions.append(decision)

                if decision.decision == DECISION_ROLLED_BACK:
                    logger.info("[RB] %s rolled back", exp.experiment_id)
                elif decision.decision == DECISION_ROLLBACK_CANDIDATE:
                    logger.info("[RB] %s candidate day=%d", exp.experiment_id,
                                decision.consecutive_candidate_days)
                elif decision.decision == DECISION_REVIEW_REQUIRED:
                    logger.info("[RB] %s under review triggers=%s",
                                exp.experiment_id, decision.trigger_reasons)

            except Exception as exc:
                logger.warning("[RB] experiment %s evaluation failed: %s", exp.experiment_id, exc)

        return decisions

    except Exception as exc:
        logger.warning("[RB] evaluate_all_active_experiments failed: %s", exc)
        return []
