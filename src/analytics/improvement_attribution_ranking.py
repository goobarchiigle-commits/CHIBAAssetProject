"""
improvement_attribution_ranking.py — Improvement Attribution Ranking (Phase 6N)

Cross-experiment ranking that identifies which improvements produce the highest
real-world value and should receive future capital allocation priority.

Inputs (all read-only):
  EXPERIMENT_PERFORMANCE_FILE     — per-record pnl/return/expectancy/win_rate/drawdown
  ACTIVE_EXPERIMENT_REGISTRY_FILE — allocation_level / escalation_history per experiment
  ROLLBACK_GOVERNANCE_FILE        — rollback_count per experiment
  CAPITAL_ESCALATION_FILE         — escalation_count per experiment

Composite ranking score (0-100):
  50% profitability + consistency component
  30% drawdown penalty (inverted)
  10% survivability (rollback_count penalized)
  10% escalation success (escalation_count rewarded)

Output:
  IMPROVEMENT_RANKING_FILE — append-only JSONL
    one record per (date, experiment_id) per run

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open        — never blocks live execution
  analytics only   — ranking never affects execution
  O(n)             — linear in experiment count × JSONL lines
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

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Score weights ──────────────────────────────────────────────────────────────

_W_PROFITABILITY = 0.30   # total_pnl (normalized)
_W_CONSISTENCY   = 0.20   # win_rate × expectancy
_W_DD_PENALTY    = 0.30   # max_drawdown (inverted, lower is better)
_W_SURVIVABILITY = 0.10   # rollback_count penalized
_W_ESCALATION    = 0.10   # escalation_count rewarded

_MAX_ESCALATION_REWARD = 5   # level-5 max
_MAX_ROLLBACK_PENALTY  = 5   # cap penalty denominator


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ImprovementRankingInput:
    experiment_id:     str
    total_pnl:         float
    avg_return:        float
    expectancy:        float
    win_rate:          float
    max_drawdown:      float
    allocation_level:  int
    escalation_count:  int
    rollback_count:    int
    observation_days:  int


@dataclass
class ImprovementRankingResult:
    experiment_id:    str
    date:             str
    rank:             int
    ranking_score:    float      # 0-100
    metrics:          Dict[str, Any]
    schema_version:   str = SCHEMA_VERSION


@dataclass
class ImprovementRankingTelemetry:
    date:             str
    total_experiments: int
    ranked:           List[ImprovementRankingResult]
    top_experiment:   str
    top_score:        float
    schema_version:   str = SCHEMA_VERSION


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        logger.debug("[IAR] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[IAR] append_jsonl failed: %s", exc)


def _load_registry_json(registry_path: Path) -> List[dict]:
    """Load active_experiment_registry.json (JSON, not JSONL). FAIL_OPEN."""
    try:
        if not registry_path.exists():
            return []
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            # may be {"experiments": [...]}
            for k in ("experiments", "registry"):
                if isinstance(raw.get(k), list):
                    return raw[k]
            return list(raw.values()) if all(isinstance(v, dict) for v in raw.values()) else []
    except Exception as exc:
        logger.debug("[IAR] load_registry_json failed: %s", exc)
    return []


# ── Per-experiment aggregation ────────────────────────────────────────────────

def _aggregate_performance(
    perf_records: List[dict],
    experiment_id: str,
) -> Dict[str, float]:
    """Aggregate performance records for one experiment (EXPERIMENT group only)."""
    rows = [
        r for r in perf_records
        if str(r.get("experiment_id", "")) == experiment_id
        and str(r.get("group", "")) == "EXPERIMENT"
    ]
    if not rows:
        return {
            "total_pnl": 0.0, "avg_return": 0.0, "expectancy": 0.0,
            "win_rate": 0.0, "max_drawdown": 0.0, "observation_days": 0,
        }
    total_pnl   = sum(_safe_float(r.get("pnl"), 0.0) for r in rows)
    avg_return  = sum(_safe_float(r.get("avg_return"), 0.0) for r in rows) / len(rows)
    expectancy  = sum(_safe_float(r.get("expectancy"), 0.0) for r in rows) / len(rows)
    win_rate    = sum(_safe_float(r.get("win_rate"), 0.0) for r in rows) / len(rows)
    dd_values   = [_safe_float(r.get("max_drawdown"), 0.0) for r in rows]
    max_drawdown = min(dd_values) if dd_values else 0.0

    # observation_days: unique dates in records
    dates = set(str(r.get("date", "")) for r in rows if r.get("date"))
    return {
        "total_pnl":       round(total_pnl, 4),
        "avg_return":      round(avg_return, 4),
        "expectancy":      round(expectancy, 4),
        "win_rate":        round(win_rate, 4),
        "max_drawdown":    round(max_drawdown, 4),
        "observation_days": len(dates),
    }


def _count_escalations(escalation_records: List[dict], experiment_id: str) -> int:
    """Count escalation decisions (non-HOLD, non-DEESCALATE) for an experiment."""
    _ESCALATE_DECISIONS = {
        "ESCALATE_LEVEL_1", "ESCALATE_LEVEL_2",
        "ESCALATE_LEVEL_3", "FULL_SCALE",
    }
    count = 0
    for r in escalation_records:
        if str(r.get("experiment_id", "")) != experiment_id:
            continue
        if str(r.get("decision", "")) in _ESCALATE_DECISIONS:
            count += 1
    return count


def _count_rollbacks(rollback_records: List[dict], experiment_id: str) -> int:
    """Count rollback events (ROLLED_BACK decision) for an experiment."""
    count = 0
    for r in rollback_records:
        if str(r.get("experiment_id", "")) != experiment_id:
            continue
        if str(r.get("decision", "")) == "ROLLED_BACK":
            count += 1
    return count


def _get_allocation_level(registry_records: List[dict], experiment_id: str) -> int:
    """Return latest allocation_level for the experiment from registry."""
    for r in registry_records:
        if str(r.get("experiment_id", "")) == experiment_id:
            return _safe_int(r.get("allocation_level"), 0)
    return 0


def _get_experiment_ids(registry_records: List[dict]) -> List[str]:
    """Return all unique experiment_id values from the registry."""
    seen: List[str] = []
    for r in registry_records:
        eid = str(r.get("experiment_id", ""))
        if eid and eid not in seen:
            seen.append(eid)
    return seen


# ── Composite score ───────────────────────────────────────────────────────────

def _normalize_minmax(values: List[float]) -> List[float]:
    """Min-max normalize to [0, 1]. Returns zeros if all values equal."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _dd_score(max_drawdown: float) -> float:
    """
    Convert max_drawdown (negative fraction) to [0, 1] survivability score.
    max_drawdown=0.0 → 1.0 (no loss), max_drawdown=-0.50 → 0.0 (50% loss).
    """
    return max(0.0, min(1.0, 1.0 + max_drawdown * 2.0))


def _survivability_score(rollback_count: int) -> float:
    """0 rollbacks → 1.0; each rollback halves score, capped at _MAX_ROLLBACK_PENALTY."""
    if rollback_count <= 0:
        return 1.0
    capped = min(rollback_count, _MAX_ROLLBACK_PENALTY)
    return 1.0 / (1.0 + capped)


def _escalation_score(escalation_count: int) -> float:
    """More escalations → higher score (capped at _MAX_ESCALATION_REWARD)."""
    capped = min(escalation_count, _MAX_ESCALATION_REWARD)
    return capped / _MAX_ESCALATION_REWARD


def compute_ranking_scores(
    inputs: List[ImprovementRankingInput],
) -> List[float]:
    """
    Compute normalized ranking scores (0-100) for all experiments.

    Profitability+consistency component uses min-max across all experiments
    so scores are relative to the peer group.
    """
    if not inputs:
        return []

    # Component 1: profitability (normalized total_pnl)
    pnl_raw     = [inp.total_pnl for inp in inputs]
    pnl_normed  = _normalize_minmax(pnl_raw)

    # Component 2: consistency (win_rate × max(expectancy, 0) — already in [0, 1] territory)
    consist_raw  = [
        inp.win_rate * max(0.0, inp.expectancy)
        for inp in inputs
    ]
    consist_norm = _normalize_minmax(consist_raw)

    scores: List[float] = []
    for i, inp in enumerate(inputs):
        p_score    = pnl_normed[i]                            # [0, 1]
        c_score    = consist_norm[i]                           # [0, 1]
        dd_s       = _dd_score(inp.max_drawdown)               # [0, 1]
        surv_s     = _survivability_score(inp.rollback_count)  # [0, 1]
        esc_s      = _escalation_score(inp.escalation_count)   # [0, 1]

        raw = (
            _W_PROFITABILITY * p_score
            + _W_CONSISTENCY * c_score
            + _W_DD_PENALTY  * dd_s
            + _W_SURVIVABILITY * surv_s
            + _W_ESCALATION  * esc_s
        )
        scores.append(round(min(100.0, max(0.0, raw * 100.0)), 2))

    return scores


# ── Ranking builder ───────────────────────────────────────────────────────────

def build_ranking_results(
    inputs: List[ImprovementRankingInput],
    today:  str,
) -> List[ImprovementRankingResult]:
    """Assign ranks (1 = best) to all experiments. Stable sort by score desc."""
    if not inputs:
        return []

    scores = compute_ranking_scores(inputs)
    combined: List[Tuple[ImprovementRankingInput, float]] = list(zip(inputs, scores))
    combined.sort(key=lambda x: x[1], reverse=True)

    results: List[ImprovementRankingResult] = []
    for rank_i, (inp, score) in enumerate(combined, start=1):
        metrics = {
            "total_pnl":        inp.total_pnl,
            "avg_return":       inp.avg_return,
            "expectancy":       inp.expectancy,
            "win_rate":         inp.win_rate,
            "max_drawdown":     inp.max_drawdown,
            "allocation_level": inp.allocation_level,
            "escalation_count": inp.escalation_count,
            "rollback_count":   inp.rollback_count,
            "observation_days": inp.observation_days,
        }
        results.append(ImprovementRankingResult(
            experiment_id=inp.experiment_id,
            date=today,
            rank=rank_i,
            ranking_score=score,
            metrics=metrics,
        ))
    return results


# ── Telemetry I/O ─────────────────────────────────────────────────────────────

def append_ranking_record(result: ImprovementRankingResult, path: Path) -> None:
    """Serialize one ImprovementRankingResult to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "experiment_id":  result.experiment_id,
            "date":           result.date,
            "rank":           result.rank,
            "ranking_score":  result.ranking_score,
            "metrics":        result.metrics,
            "schema_version": result.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[IAR] append_ranking_record failed: %s", exc)


def load_ranking_records(
    path:          Path,
    experiment_id: Optional[str] = None,
) -> List[dict]:
    """Load ranking telemetry. FAIL_OPEN → empty list."""
    rows = _load_all_jsonl(path)
    if experiment_id is not None:
        rows = [r for r in rows if str(r.get("experiment_id", "")) == experiment_id]
    return rows


def _load_evaluated_pairs(output_file: Path) -> set:
    """Return set of (date, experiment_id) already written."""
    pairs: set = set()
    for r in _load_all_jsonl(output_file):
        d  = str(r.get("date", ""))
        eid = str(r.get("experiment_id", ""))
        if d and eid:
            pairs.add((d, eid))
    return pairs


# ── Main entry point ──────────────────────────────────────────────────────────

def run_improvement_attribution_ranking(
    performance_path:  Path,
    registry_path:     Path,
    rollback_path:     Path,
    escalation_path:   Path,
    output_path:       Path,
    today:             str,
) -> ImprovementRankingTelemetry:
    """
    Build cross-experiment ranking for today.

    Idempotent: (date, experiment_id) pair never re-written.
    Never raises. FAIL_OPEN on all errors.
    """
    try:
        evaluated = _load_evaluated_pairs(output_path)

        # Load source data
        perf_records       = _load_all_jsonl(performance_path)
        registry_records   = _load_registry_json(registry_path)
        rollback_records   = _load_all_jsonl(rollback_path)
        escalation_records = _load_all_jsonl(escalation_path)

        experiment_ids = _get_experiment_ids(registry_records)

        # Also pick up any experiment_id mentioned in performance but not registry
        for r in perf_records:
            eid = str(r.get("experiment_id", ""))
            if eid and eid not in experiment_ids:
                experiment_ids.append(eid)

        inputs: List[ImprovementRankingInput] = []
        for eid in experiment_ids:
            if (today, eid) in evaluated:
                continue
            perf = _aggregate_performance(perf_records, eid)
            alloc_level    = _get_allocation_level(registry_records, eid)
            esc_count      = _count_escalations(escalation_records, eid)
            rb_count       = _count_rollbacks(rollback_records, eid)
            inputs.append(ImprovementRankingInput(
                experiment_id=eid,
                total_pnl=perf["total_pnl"],
                avg_return=perf["avg_return"],
                expectancy=perf["expectancy"],
                win_rate=perf["win_rate"],
                max_drawdown=perf["max_drawdown"],
                allocation_level=alloc_level,
                escalation_count=esc_count,
                rollback_count=rb_count,
                observation_days=_safe_int(perf["observation_days"], 0),
            ))

        ranked = build_ranking_results(inputs, today)
        for result in ranked:
            append_ranking_record(result, output_path)

        top_exp   = ranked[0].experiment_id if ranked else ""
        top_score = ranked[0].ranking_score if ranked else 0.0

        return ImprovementRankingTelemetry(
            date=today,
            total_experiments=len(ranked),
            ranked=ranked,
            top_experiment=top_exp,
            top_score=top_score,
        )

    except Exception as exc:
        logger.warning("[IAR] run_improvement_attribution_ranking failed: %s", exc)
        return ImprovementRankingTelemetry(
            date=today,
            total_experiments=0,
            ranked=[],
            top_experiment="",
            top_score=0.0,
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_ranking_report(telemetry: ImprovementRankingTelemetry) -> str:
    """Return IMPROVEMENT ATTRIBUTION RANKING report section string."""
    if not telemetry.ranked:
        return ""
    lines: List[str] = []
    sep = "═" * 72
    lines.append(sep)
    lines.append("IMPROVEMENT ATTRIBUTION RANKING")
    lines.append(f"Date: {telemetry.date}  |  Experiments: {telemetry.total_experiments}")
    lines.append("─" * 72)
    lines.append(
        f"{'Rank':<5} {'Experiment':<22} {'Score':>6}  "
        f"{'PnL':>8}  {'WinR':>6}  {'DD':>7}  {'Esc':>4}  {'RB':>3}"
    )
    lines.append("─" * 72)
    for r in telemetry.ranked:
        m = r.metrics
        lines.append(
            f"  #{r.rank:<4} {r.experiment_id:<22} {r.ranking_score:>6.1f}  "
            f"{m.get('total_pnl', 0.0):>8.2f}  "
            f"{m.get('win_rate', 0.0):>6.3f}  "
            f"{m.get('max_drawdown', 0.0):>7.3f}  "
            f"{m.get('escalation_count', 0):>4}  "
            f"{m.get('rollback_count', 0):>3}"
        )
    lines.append("─" * 72)
    lines.append(
        f"Top: {telemetry.top_experiment}  (score={telemetry.top_score:.1f})"
    )
    lines.append(sep)
    return "\n".join(lines)
