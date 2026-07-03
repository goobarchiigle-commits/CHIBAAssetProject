"""
active_experiment_registry.py — Active Experiment Registry (Phase 6J-Plus)

Closed-loop production experimentation layer.
Manages experiment lifecycle and deterministic symbol routing.

Design:
  fail-closed routing — registry unavailable → all symbols default CONTROL
  observation_only    — no signals, no orders, no parameter changes
  deterministic       — sha256(symbol) routing is stable across restarts
  O(n)                — linear in registry size and symbol count

Registry storage: JSON (mutable state, not JSONL)
Assignment telemetry: JSONL (append-only)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.analytics.production_promotion_gate import GateRunResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"

STATUS_PENDING           = "PENDING"
STATUS_ACTIVE            = "ACTIVE"
STATUS_PAUSED            = "PAUSED"
STATUS_ROLLED_BACK       = "ROLLED_BACK"
STATUS_COMPLETED         = "COMPLETED"
STATUS_REVIEW_REQUIRED   = "REVIEW_REQUIRED"
STATUS_ROLLBACK_CANDIDATE = "ROLLBACK_CANDIDATE"

GROUP_CONTROL    = "CONTROL"
GROUP_EXPERIMENT = "EXPERIMENT"

EVAL_CONTINUE = "CONTINUE"
EVAL_ROLLBACK = "ROLLBACK"
EVAL_PROMOTE  = "PROMOTE"

_APPROVAL_SOURCE_GATE  = "PRODUCTION_PROMOTION_GATE"
_DEFAULT_ALLOCATION    = 0.5    # 50/50 split
_DEFAULT_ROLLBACK_THR  = -0.10  # -10% drawdown
_DEFAULT_MIN_SAMPLE    = 10
_DEFAULT_MAX_POSITIONS = 1
_PROMOTE_WIN_RATE_EDGE = 0.05   # experiment must beat control by 5pp


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    experiment_id:       str
    feature_name:        str
    status:              str    # STATUS_* constant
    allocation_pct:      float  # fraction [0.0, 1.0] — portion routed to EXPERIMENT
    max_positions:       int
    start_date:          str    # ISO date
    approval_source:     str
    rollback_threshold:  float  # max_drawdown below which → ROLLBACK (e.g. -0.10)
    minimum_sample_size: int
    schema_version:      str       = SCHEMA_VERSION
    allocation_level:    int       = 0   # Phase 6M: current escalation level (0-5)
    escalation_history:  List[str] = field(default_factory=list)  # Phase 6M: audit trail


@dataclass
class ExperimentAssignment:
    experiment_id:  str
    symbol:         str
    group:          str   # GROUP_CONTROL or GROUP_EXPERIMENT
    assigned_at:    str   # ISO date
    schema_version: str = SCHEMA_VERSION


@dataclass
class ExperimentEvaluation:
    experiment_id:       str
    date:                str
    n_control:           int
    n_experiment:        int
    control_win_rate:    float
    experiment_win_rate: float
    status:              str   # EVAL_* constant
    schema_version:      str = SCHEMA_VERSION


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
        logger.warning("[EXP] append_jsonl failed: %s", exc)


# ── Registry I/O ───────────────────────────────────────────────────────────────

def load_registry(path: Path) -> List[ExperimentConfig]:
    """
    Load experiment registry from JSON file.
    FAIL_CLOSED: returns [] on any error so callers default to CONTROL.
    """
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("[EXP] load_registry: unexpected format in %s", path)
            return []
        configs: List[ExperimentConfig] = []
        for item in data.get("experiments", []):
            if not isinstance(item, dict):
                continue
            configs.append(ExperimentConfig(
                experiment_id=str(item.get("experiment_id", "")),
                feature_name=str(item.get("feature_name", "")),
                status=str(item.get("status", STATUS_PENDING)),
                allocation_pct=_safe_float(item.get("allocation_pct"), 0.0),
                max_positions=_safe_int(item.get("max_positions"), _DEFAULT_MAX_POSITIONS),
                start_date=str(item.get("start_date", "")),
                approval_source=str(item.get("approval_source", "")),
                rollback_threshold=_safe_float(item.get("rollback_threshold"), _DEFAULT_ROLLBACK_THR),
                minimum_sample_size=_safe_int(item.get("minimum_sample_size"), _DEFAULT_MIN_SAMPLE),
                schema_version=str(item.get("schema_version", SCHEMA_VERSION)),
                allocation_level=_safe_int(item.get("allocation_level"), 0),
                escalation_history=list(item.get("escalation_history") or []),
            ))
        return configs
    except Exception as exc:
        logger.warning("[EXP] load_registry failed (FAIL_CLOSED → []): %s", exc)
        return []


def save_registry(configs: List[ExperimentConfig], path: Path) -> bool:
    """Atomic save of registry to JSON. Returns True on success. FAIL_OPEN."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": SCHEMA_VERSION,
            "experiments": [
                {
                    "experiment_id":       c.experiment_id,
                    "feature_name":        c.feature_name,
                    "status":              c.status,
                    "allocation_pct":      c.allocation_pct,
                    "max_positions":       c.max_positions,
                    "start_date":          c.start_date,
                    "approval_source":     c.approval_source,
                    "rollback_threshold":  c.rollback_threshold,
                    "minimum_sample_size": c.minimum_sample_size,
                    "schema_version":      c.schema_version,
                    "allocation_level":    c.allocation_level,
                    "escalation_history":  list(c.escalation_history),
                }
                for c in configs
            ],
        }
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(path)
        return True
    except Exception as exc:
        logger.warning("[EXP] save_registry failed: %s", exc)
        return False


# ── Registry queries ───────────────────────────────────────────────────────────

def get_active_experiments(configs: List[ExperimentConfig]) -> List[ExperimentConfig]:
    """Return only ACTIVE experiments."""
    return [c for c in configs if c.status == STATUS_ACTIVE]


# ── Deterministic routing ──────────────────────────────────────────────────────

def assign_symbol_to_experiment(
    symbol:     str,
    experiment: ExperimentConfig,
    today:      str,
) -> ExperimentAssignment:
    """
    Deterministic routing: sha256(symbol) → bucket [0.0, 1.0) → CONTROL/EXPERIMENT.

    Same symbol always maps to same group for a given experiment's allocation_pct.
    Allocation_pct is clamped to [0.0, 1.0].
    """
    alloc = max(0.0, min(1.0, experiment.allocation_pct))
    hash_hex = hashlib.sha256(symbol.encode()).hexdigest()
    hash_int  = int(hash_hex, 16)
    # Use mod 10000 for fine-grained bucket (0.0001 resolution)
    bucket = (hash_int % 10_000) / 10_000.0
    group  = GROUP_EXPERIMENT if bucket < alloc else GROUP_CONTROL
    return ExperimentAssignment(
        experiment_id=experiment.experiment_id,
        symbol=symbol,
        group=group,
        assigned_at=today,
    )


def route_symbols_to_experiments(
    symbols:     List[str],
    experiments: List[ExperimentConfig],
    today:       str,
) -> Dict[str, List[ExperimentAssignment]]:
    """
    Route each symbol through all active experiments.
    Returns {symbol: [ExperimentAssignment, ...]} for all active experiments.
    FAIL_CLOSED: if experiments is empty, returns {} (all symbols default CONTROL).
    """
    if not experiments or not symbols:
        return {}
    result: Dict[str, List[ExperimentAssignment]] = {}
    for sym in symbols:
        assignments: List[ExperimentAssignment] = []
        for exp in experiments:
            assignments.append(assign_symbol_to_experiment(sym, exp, today))
        result[sym] = assignments
    return result


def append_assignment(assignment: ExperimentAssignment, path: Path) -> None:
    """Serialize one ExperimentAssignment to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "experiment_id":  assignment.experiment_id,
            "symbol":         assignment.symbol,
            "group":          assignment.group,
            "assigned_at":    assignment.assigned_at,
            "schema_version": assignment.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[EXP] append_assignment failed: %s", exc)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_assignment(
    experiment:          ExperimentConfig,
    performance_records: List[dict],
    today:               str,
) -> ExperimentEvaluation:
    """
    Compare CONTROL vs EXPERIMENT performance records.
    Returns CONTINUE / ROLLBACK / PROMOTE verdict.
    FAIL_OPEN: returns CONTINUE on any error.
    """
    try:
        ctrl_recs = [r for r in performance_records if r.get("group") == GROUP_CONTROL]
        exp_recs  = [r for r in performance_records if r.get("group") == GROUP_EXPERIMENT]

        def _win_rate(recs: List[dict]) -> float:
            if not recs:
                return 0.0
            wins = sum(1 for r in recs if _safe_float(r.get("pnl"), 0.0) > 0)
            return round(wins / len(recs), 4)

        ctrl_wr = _win_rate(ctrl_recs)
        exp_wr  = _win_rate(exp_recs)
        n_exp   = len(exp_recs)

        if n_exp < experiment.minimum_sample_size:
            verdict = EVAL_CONTINUE
        else:
            # Check drawdown against rollback_threshold
            exp_ddlist = [_safe_float(r.get("max_drawdown"), 0.0) for r in exp_recs]
            worst_dd   = min(exp_ddlist) if exp_ddlist else 0.0
            if worst_dd < experiment.rollback_threshold:
                verdict = EVAL_ROLLBACK
            elif exp_wr >= ctrl_wr + _PROMOTE_WIN_RATE_EDGE:
                verdict = EVAL_PROMOTE
            else:
                verdict = EVAL_CONTINUE

        return ExperimentEvaluation(
            experiment_id=experiment.experiment_id,
            date=today,
            n_control=len(ctrl_recs),
            n_experiment=n_exp,
            control_win_rate=ctrl_wr,
            experiment_win_rate=exp_wr,
            status=verdict,
        )
    except Exception as exc:
        logger.warning("[EXP] evaluate_assignment failed: %s", exc)
        return ExperimentEvaluation(
            experiment_id=experiment.experiment_id,
            date=today,
            n_control=0,
            n_experiment=0,
            control_win_rate=0.0,
            experiment_win_rate=0.0,
            status=EVAL_CONTINUE,
        )


# ── Promotion integration ──────────────────────────────────────────────────────

def create_experiments_from_gate_approval(
    gate_records:   List[dict],
    registry_path:  Path,
    today:          str,
) -> List[ExperimentConfig]:
    """
    For each APPROVED gate record, create an ACTIVE experiment entry
    if no ACTIVE experiment already exists for that feature.

    gate_records: list of dicts with 'decision', 'candidate' fields
    Returns list of newly created ExperimentConfig entries.
    """
    try:
        existing = load_registry(registry_path)
        existing_ids: Set[str] = {c.experiment_id for c in existing}
        active_features: Set[str] = {
            c.feature_name for c in existing if c.status == STATUS_ACTIVE
        }

        new_configs: List[ExperimentConfig] = []
        for rec in gate_records:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("decision", "")) != "APPROVED":
                continue
            candidate = str(rec.get("candidate", ""))
            if not candidate:
                continue
            if candidate in active_features:
                continue
            exp_id = f"EXP_{candidate}_{today}"
            if exp_id in existing_ids:
                continue
            new_exp = ExperimentConfig(
                experiment_id=exp_id,
                feature_name=candidate,
                status=STATUS_ACTIVE,
                allocation_pct=_DEFAULT_ALLOCATION,
                max_positions=_DEFAULT_MAX_POSITIONS,
                start_date=today,
                approval_source=_APPROVAL_SOURCE_GATE,
                rollback_threshold=_DEFAULT_ROLLBACK_THR,
                minimum_sample_size=_DEFAULT_MIN_SAMPLE,
            )
            new_configs.append(new_exp)
            active_features.add(candidate)
            existing_ids.add(exp_id)

        if new_configs:
            save_registry(existing + new_configs, registry_path)

        return new_configs
    except Exception as exc:
        logger.warning("[EXP] create_experiments_from_gate_approval failed: %s", exc)
        return []


# ── Registry mutation ─────────────────────────────────────────────────────────

def update_experiment_status(
    experiment_id: str,
    new_status:    str,
    registry_path: Path,
) -> bool:
    """
    Update the status of a single experiment in the registry.
    Returns True on success. FAIL_OPEN (returns False on error).
    If experiment_id not found, no change is made (returns True).
    """
    try:
        configs = load_registry(registry_path)
        updated = False
        for c in configs:
            if c.experiment_id == experiment_id and c.status != new_status:
                c.status = new_status
                updated = True
                break
        if updated:
            return save_registry(configs, registry_path)
        return True
    except Exception as exc:
        logger.warning("[EXP] update_experiment_status failed for %s: %s", experiment_id, exc)
        return False


def update_experiment_escalation(
    experiment_id:    str,
    new_level:        int,
    new_allocation_pct: float,
    escalation_event: str,
    registry_path:    Path,
) -> bool:
    """
    Update allocation_level, allocation_pct, and append to escalation_history.
    Returns True on success. FAIL_OPEN (returns False on error).
    If experiment_id not found, no change is made (returns True).
    """
    try:
        configs = load_registry(registry_path)
        updated = False
        for c in configs:
            if c.experiment_id == experiment_id:
                c.allocation_level   = new_level
                c.allocation_pct     = new_allocation_pct
                c.escalation_history = list(c.escalation_history) + [escalation_event]
                updated = True
                break
        if updated:
            return save_registry(configs, registry_path)
        return True
    except Exception as exc:
        logger.warning("[EXP] update_experiment_escalation failed for %s: %s", experiment_id, exc)
        return False


# ── Report formatter ───────────────────────────────────────────────────────────

def format_experiment_report(
    active_experiments: List[ExperimentConfig],
    routing:            Dict[str, List[ExperimentAssignment]],
) -> str:
    """Return experiment routing summary string."""
    if not active_experiments:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("ACTIVE EXPERIMENT REGISTRY")
    lines.append("─" * 60)
    for exp in active_experiments:
        lines.append(
            f"  {exp.experiment_id:<36}  alloc={exp.allocation_pct:.0%}"
            f"  src={exp.approval_source}"
        )
    if routing:
        lines.append("─" * 60)
        lines.append(f"{'Symbol':<14} {'Experiment':<36} Group")
        lines.append("─" * 60)
        for sym, assignments in routing.items():
            for a in assignments:
                lines.append(f"  {sym:<12} {a.experiment_id:<36} {a.group}")
    lines.append(sep)
    return "\n".join(lines)
