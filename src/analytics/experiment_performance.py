"""
experiment_performance.py — Experiment Performance Tracker (Phase 6J-Plus)

Tracks per-experiment, per-group performance metrics.
Append-only JSONL. Observation-only — no execution changes.

Metrics tracked per record:
  trades, win_rate, expectancy, avg_return, max_drawdown, pnl
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"

GROUP_CONTROL    = "CONTROL"
GROUP_EXPERIMENT = "EXPERIMENT"


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class ExperimentOutcome:
    experiment_id:  str
    date:           str
    symbol:         str
    group:          str    # GROUP_CONTROL or GROUP_EXPERIMENT
    trades:         int
    win_rate:       float
    expectancy:     float
    avg_return:     float
    max_drawdown:   float
    pnl:            float
    schema_version: str = SCHEMA_VERSION


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
        logger.warning("[EXPPERF] append_jsonl failed: %s", exc)


# ── I/O ────────────────────────────────────────────────────────────────────────

def record_experiment_outcome(outcome: ExperimentOutcome, path: Path) -> None:
    """Append one ExperimentOutcome to the JSONL performance file. FAIL_OPEN."""
    try:
        data = {
            "experiment_id":  outcome.experiment_id,
            "date":           outcome.date,
            "symbol":         outcome.symbol,
            "group":          outcome.group,
            "trades":         outcome.trades,
            "win_rate":       outcome.win_rate,
            "expectancy":     outcome.expectancy,
            "avg_return":     outcome.avg_return,
            "max_drawdown":   outcome.max_drawdown,
            "pnl":            outcome.pnl,
            "schema_version": outcome.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[EXPPERF] record_experiment_outcome failed: %s", exc)


def load_performance_records(
    path:          Path,
    experiment_id: Optional[str] = None,
) -> List[dict]:
    """
    Load performance records from JSONL.
    If experiment_id is given, filter to that experiment.
    FAIL_OPEN → empty list on error.
    """
    records: List[dict] = []
    try:
        if not path.exists():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if experiment_id is None or str(r.get("experiment_id", "")) == experiment_id:
                    records.append(r)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[EXPPERF] load_performance_records %s failed: %s", path, exc)
    return records


# ── Analytics ──────────────────────────────────────────────────────────────────

def compute_performance_summary(
    records: List[dict],
    group:   Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute aggregate performance metrics from a list of outcome records.
    If group is given, filter to that group first.
    Returns dict with: trades, win_rate, expectancy, avg_return, max_drawdown, pnl.
    """
    filtered = records if group is None else [
        r for r in records if r.get("group") == group
    ]
    if not filtered:
        return {
            "trades": 0, "win_rate": 0.0, "expectancy": 0.0,
            "avg_return": 0.0, "max_drawdown": 0.0, "pnl": 0.0,
        }

    total_trades   = sum(_safe_int(r.get("trades"), 0) for r in filtered)
    pnl_values     = [_safe_float(r.get("pnl"), 0.0) for r in filtered]
    return_values  = [_safe_float(r.get("avg_return"), 0.0) for r in filtered]
    dd_values      = [_safe_float(r.get("max_drawdown"), 0.0) for r in filtered]
    wr_values      = [_safe_float(r.get("win_rate"), 0.0) for r in filtered]

    total_pnl      = sum(pnl_values)
    avg_return     = sum(return_values) / len(return_values) if return_values else 0.0
    max_drawdown   = min(dd_values) if dd_values else 0.0
    avg_win_rate   = sum(wr_values) / len(wr_values) if wr_values else 0.0

    # Expectancy: avg_return × win_rate (simplified, sign-aware)
    exp_values = [
        _safe_float(r.get("expectancy"), _safe_float(r.get("avg_return"), 0.0))
        for r in filtered
    ]
    expectancy = sum(exp_values) / len(exp_values) if exp_values else 0.0

    return {
        "trades":       total_trades,
        "win_rate":     round(avg_win_rate, 4),
        "expectancy":   round(expectancy, 4),
        "avg_return":   round(avg_return, 4),
        "max_drawdown": round(max_drawdown, 4),
        "pnl":          round(total_pnl, 4),
    }


def format_performance_report(
    experiment_id: str,
    records:       List[dict],
) -> str:
    """Return side-by-side CONTROL vs EXPERIMENT performance comparison."""
    ctrl = compute_performance_summary(records, group=GROUP_CONTROL)
    exp  = compute_performance_summary(records, group=GROUP_EXPERIMENT)

    if ctrl["trades"] == 0 and exp["trades"] == 0:
        return ""

    lines: List[str] = []
    sep = "═" * 70
    lines.append(sep)
    lines.append(f"EXPERIMENT PERFORMANCE: {experiment_id}")
    lines.append("─" * 70)
    lines.append(f"{'Metric':<18} {'CONTROL':>12} {'EXPERIMENT':>12}")
    lines.append("─" * 70)
    for metric in ("trades", "win_rate", "expectancy", "avg_return", "max_drawdown", "pnl"):
        cv = ctrl[metric]
        ev = exp[metric]
        if isinstance(cv, float):
            lines.append(f"  {metric:<16} {cv:>12.4f} {ev:>12.4f}")
        else:
            lines.append(f"  {metric:<16} {cv:>12} {ev:>12}")
    lines.append(sep)
    return "\n".join(lines)
