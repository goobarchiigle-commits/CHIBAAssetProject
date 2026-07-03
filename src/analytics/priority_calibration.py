"""
priority_calibration.py — Priority Calibration Intelligence

Verifies whether priority_score has a monotonic relationship with future returns,
providing empirical evidence before any priority-aware sizing or allocation decisions.

Tasks:
  1. Decompose priority_score into 10-point buckets (0-10, 10-20, ..., 90-100)
  2. Per-bucket stats: avg_5d_return, avg_10d_return, win_rate, sample_size
  3. Monotonicity Score — Spearman rank correlation between bucket rank and return rank
  4. Calibration Error — fraction of adjacent bucket pairs violating monotonic ordering
  5. Telemetry — append-only priority_calibration.jsonl (1 record/run)
  6. Report — PRIORITY CALIBRATION section (top/worst bucket, monotonicity, verdict)

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — single-pass per record; no re-sorting of full dataset
  append-only JSONL — 1 record per run, never rewritten

Key metrics:
  priority_monotonicity      — Spearman ρ in [−1, 1]; 1.0 = perfect monotonic
  priority_calibration_error — fraction of adjacent violations in [0, 1]; 0.0 = perfect
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
TARGET_5D  = "subsequent_5d_return"
TARGET_10D = "subsequent_10d_return"   # optional field; None if absent

# ── Bucket configuration ──────────────────────────────────────────────────────
BUCKET_EDGES: List[int] = list(range(0, 101, 10))   # [0,10,20,...,100]
BUCKET_COUNT: int = len(BUCKET_EDGES) - 1            # 10 buckets

def _bucket_label(i: int) -> str:
    return f"{BUCKET_EDGES[i]}-{BUCKET_EDGES[i+1]}"

BUCKET_LABELS: List[str] = [_bucket_label(i) for i in range(BUCKET_COUNT)]

LOOKBACK_DAYS_DEFAULT: int = 90
MIN_BUCKET_SAMPLES:    int = 3     # min records for a bucket to count in monotonicity/calibration


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PriorityBucket:
    label:           str            # "0-10", "10-20", ...
    low:             float
    high:            float
    midpoint:        float
    n:               int
    avg_5d_return:   float
    avg_10d_return:  Optional[float]    # None if no 10d data available
    win_rate_5d:     float
    win_rate_10d:    Optional[float]


@dataclass
class CalibrationOutput:
    date:                      str
    n_records_total:           int    # CP records in lookback window
    n_materialized:            int    # records with finite 5d return
    lookback_days:             int
    min_bucket_samples:        int
    buckets:                   List[PriorityBucket]
    n_buckets_populated:       int    # buckets with n >= min_bucket_samples
    priority_monotonicity:     float  # Spearman ρ [−1,1]  ← 最重要KPI
    priority_calibration_error: float # adjacent violation rate [0,1]  ← 最重要KPI
    top_bucket:                str    # label of bucket with highest avg_5d_return
    worst_bucket:              str    # label of bucket with lowest avg_5d_return
    schema_version:            str = SCHEMA_VERSION


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe(v: Any, fallback: Optional[float] = 0.0) -> Optional[float]:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _load_jsonl(path: Path) -> List[dict]:
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
        logger.warning("[PCI] load_jsonl %s failed: %s", path, exc)
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic JSONL append. FAIL_OPEN."""
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
        logger.warning("[PCI] append_jsonl failed: %s", exc)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_filter_cp(
    cp_file: Path,
    lookback_days: int,
    today: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Load CP telemetry. Return (within_window, materialized).
    materialized: subset with finite subsequent_5d_return AND finite priority_score.
    FAIL_OPEN → ([], []).
    """
    try:
        today_dt = datetime.strptime(today, "%Y-%m-%d")
    except (ValueError, TypeError):
        today_dt = datetime.now(_JST).replace(tzinfo=None)
    cutoff_dt = today_dt - timedelta(days=lookback_days)

    within: List[dict] = []
    materialized: List[dict] = []
    for rec in _load_jsonl(cp_file):
        ds = rec.get("date", "")
        try:
            dt = datetime.strptime(str(ds), "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        if dt < cutoff_dt:
            continue
        within.append(rec)
        rv = _safe(rec.get(TARGET_5D), None)
        ps = _safe(rec.get("priority_score"), None)
        if rv is not None and ps is not None:
            materialized.append(rec)
    return within, materialized


# ── Task 1+2: Bucket decomposition ───────────────────────────────────────────

def _bucket_index(priority_score: float) -> int:
    """Map priority_score [0,100] to bucket index [0,9]. Clamps endpoints."""
    if priority_score >= BUCKET_EDGES[-1]:
        return BUCKET_COUNT - 1
    if priority_score < BUCKET_EDGES[0]:
        return 0
    for i in range(BUCKET_COUNT):
        if BUCKET_EDGES[i] <= priority_score < BUCKET_EDGES[i + 1]:
            return i
    return BUCKET_COUNT - 1


def compute_buckets(records: List[dict]) -> List[PriorityBucket]:
    """
    Task 1+2: Single-pass O(n) bucket accumulation.
    For each bucket: n, sum_5d, sum_10d, n_with_10d, wins_5d, wins_10d.
    """
    # Accumulators per bucket
    ns          = [0]   * BUCKET_COUNT
    sum5        = [0.0] * BUCKET_COUNT
    sum10       = [0.0] * BUCKET_COUNT
    n10         = [0]   * BUCKET_COUNT
    wins5       = [0]   * BUCKET_COUNT
    wins10      = [0]   * BUCKET_COUNT

    for rec in records:
        ps = _safe(rec.get("priority_score"), None)
        r5 = _safe(rec.get(TARGET_5D), None)
        if ps is None or r5 is None:
            continue
        idx = _bucket_index(ps)
        ns[idx]    += 1
        sum5[idx]  += r5
        wins5[idx] += 1 if r5 > 0 else 0
        r10 = _safe(rec.get(TARGET_10D), None)
        if r10 is not None:
            sum10[idx]  += r10
            n10[idx]    += 1
            wins10[idx] += 1 if r10 > 0 else 0

    buckets: List[PriorityBucket] = []
    for i in range(BUCKET_COUNT):
        n = ns[i]
        avg5  = round(sum5[i] / n, 6)     if n > 0 else 0.0
        wr5   = round(wins5[i] / n, 4)    if n > 0 else 0.0
        avg10: Optional[float] = None
        wr10:  Optional[float] = None
        if n10[i] > 0:
            avg10 = round(sum10[i] / n10[i], 6)
            wr10  = round(wins10[i] / n10[i], 4)
        buckets.append(PriorityBucket(
            label=BUCKET_LABELS[i],
            low=float(BUCKET_EDGES[i]),
            high=float(BUCKET_EDGES[i + 1]),
            midpoint=float(BUCKET_EDGES[i] + 5),
            n=n,
            avg_5d_return=avg5,
            avg_10d_return=avg10,
            win_rate_5d=wr5,
            win_rate_10d=wr10,
        ))
    return buckets


# ── Task 3: Monotonicity Score (Spearman ρ) ──────────────────────────────────

def _rank(values: List[float]) -> List[int]:
    """
    Return rank (1-based, ascending) for each element.
    Ties get the same rank (average-rank not needed for strict inequality buckets).
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0] * n
    for rank, i in enumerate(order, 1):
        ranks[i] = rank
    return ranks


def compute_monotonicity(buckets: List[PriorityBucket], min_samples: int) -> float:
    """
    Task 3: Spearman rank correlation between bucket priority order and avg_5d_return.

    Uses only populated buckets (n >= min_samples).
    Returns value in [−1, 1]; 0.0 if fewer than 2 populated buckets.
    """
    populated = [b for b in buckets if b.n >= min_samples]
    k = len(populated)
    if k < 2:
        return 0.0

    # Bucket order (low → high priority) is already sorted by index.
    # priority_rank[i] = i+1 (natural order of populated buckets by midpoint)
    priority_ranks = list(range(1, k + 1))
    returns = [b.avg_5d_return for b in populated]
    return_ranks = _rank(returns)

    d_sq_sum = sum((pr - rr) ** 2 for pr, rr in zip(priority_ranks, return_ranks))
    rho = 1.0 - (6.0 * d_sq_sum) / (k * (k * k - 1))
    return round(rho, 4)


# ── Task 4: Calibration Error ─────────────────────────────────────────────────

def compute_calibration_error(buckets: List[PriorityBucket], min_samples: int) -> float:
    """
    Task 4: Fraction of adjacent bucket pairs that violate monotonic ordering.

    For each pair (lower_priority_bucket, higher_priority_bucket):
      violation if avg_5d_return(higher) < avg_5d_return(lower).

    Returns value in [0, 1]; 0.0 = perfect monotonicity; 1.0 = fully inverted.
    Returns 0.0 if fewer than 2 populated buckets.
    """
    populated = [b for b in buckets if b.n >= min_samples]
    k = len(populated)
    if k < 2:
        return 0.0

    n_pairs = k - 1
    violations = sum(
        1 for i in range(n_pairs)
        if populated[i + 1].avg_5d_return < populated[i].avg_5d_return
    )
    return round(violations / n_pairs, 4)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_priority_calibration(
    cp_file:       Path,
    output_file:   Path,
    today:         Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    min_samples:   int = MIN_BUCKET_SAMPLES,
) -> CalibrationOutput:
    """
    Load CP telemetry, compute priority calibration metrics, append JSONL.

    Returns CalibrationOutput always. Never raises.
    """
    if today is None:
        today = datetime.now(_JST).strftime("%Y-%m-%d")

    def _empty() -> CalibrationOutput:
        return CalibrationOutput(
            date=today,
            n_records_total=0,
            n_materialized=0,
            lookback_days=lookback_days,
            min_bucket_samples=min_samples,
            buckets=[
                PriorityBucket(
                    label=BUCKET_LABELS[i],
                    low=float(BUCKET_EDGES[i]),
                    high=float(BUCKET_EDGES[i + 1]),
                    midpoint=float(BUCKET_EDGES[i] + 5),
                    n=0,
                    avg_5d_return=0.0,
                    avg_10d_return=None,
                    win_rate_5d=0.0,
                    win_rate_10d=None,
                )
                for i in range(BUCKET_COUNT)
            ],
            n_buckets_populated=0,
            priority_monotonicity=0.0,
            priority_calibration_error=0.0,
            top_bucket="",
            worst_bucket="",
        )

    try:
        within, materialized = load_and_filter_cp(cp_file, lookback_days, today)
        buckets = compute_buckets(materialized)

        populated = [b for b in buckets if b.n >= min_samples]
        n_pop = len(populated)

        mono  = compute_monotonicity(buckets, min_samples)
        calib = compute_calibration_error(buckets, min_samples)

        # Top / worst by avg_5d_return among populated buckets
        top_bucket    = ""
        worst_bucket  = ""
        if populated:
            top_bucket   = max(populated, key=lambda b: b.avg_5d_return).label
            worst_bucket = min(populated, key=lambda b: b.avg_5d_return).label

        result = CalibrationOutput(
            date=today,
            n_records_total=len(within),
            n_materialized=len(materialized),
            lookback_days=lookback_days,
            min_bucket_samples=min_samples,
            buckets=buckets,
            n_buckets_populated=n_pop,
            priority_monotonicity=mono,
            priority_calibration_error=calib,
            top_bucket=top_bucket,
            worst_bucket=worst_bucket,
        )

        append_calibration_record(result, output_file)
        logger.info(
            "[PCI] n=%d mat=%d buckets=%d monotonicity=%.4f calib_err=%.4f",
            result.n_records_total,
            result.n_materialized,
            result.n_buckets_populated,
            result.priority_monotonicity,
            result.priority_calibration_error,
        )
        return result

    except Exception as exc:
        logger.warning("[PCI] run_priority_calibration failed: %s", exc)
        return _empty()


# ── Task 5: JSONL persistence ─────────────────────────────────────────────────

def append_calibration_record(output: CalibrationOutput, path: Path) -> None:
    """Append one aggregated record to priority_calibration.jsonl. FAIL_OPEN."""
    try:
        record = {
            "date":                        output.date,
            "n_records_total":             output.n_records_total,
            "n_materialized":              output.n_materialized,
            "lookback_days":               output.lookback_days,
            "min_bucket_samples":          output.min_bucket_samples,
            "n_buckets_populated":         output.n_buckets_populated,
            "priority_monotonicity":       output.priority_monotonicity,
            "priority_calibration_error":  output.priority_calibration_error,
            "top_bucket":                  output.top_bucket,
            "worst_bucket":                output.worst_bucket,
            "buckets": [
                {
                    "label":          b.label,
                    "low":            b.low,
                    "high":           b.high,
                    "midpoint":       b.midpoint,
                    "n":              b.n,
                    "avg_5d_return":  b.avg_5d_return,
                    "avg_10d_return": b.avg_10d_return,
                    "win_rate_5d":    b.win_rate_5d,
                    "win_rate_10d":   b.win_rate_10d,
                }
                for b in output.buckets
            ],
            "schema_version": output.schema_version,
        }
        _append_jsonl(record, path)
    except Exception as exc:
        logger.warning("[PCI] append_calibration_record failed: %s", exc)


# ── Task 6: Report formatter ──────────────────────────────────────────────────

def format_calibration_report(output: CalibrationOutput) -> str:
    """DRY/LIVE PRIORITY CALIBRATION section. Returns '' if no data. FAIL_OPEN."""
    try:
        if output.n_buckets_populated == 0:
            return (
                f"\n── PRIORITY CALIBRATION INTELLIGENCE ────────────────────────────────\n"
                f"  データ不足: {output.n_materialized}/{output.n_records_total} 件 materialized"
                f" / populated buckets: {output.n_buckets_populated}\n"
                f"  (最低 {output.min_bucket_samples} 件/bucket、ルックバック {output.lookback_days}日)\n"
                f"──────────────────────────────────────────────────────────────────────"
            )

        lines = [
            "",
            "── PRIORITY CALIBRATION INTELLIGENCE ────────────────────────────────",
            f"  分析日: {output.date}  期間: {output.lookback_days}日"
            f"  サンプル: {output.n_materialized}/{output.n_records_total}件"
            f"  populated buckets: {output.n_buckets_populated}/{BUCKET_COUNT}",
            "",
        ]

        # Bucket table
        lines.append(
            f"  {'Bucket':<9} {'N':>5}  {'avg_5d':>8}  {'avg_10d':>8}  {'win_5d':>7}  {'win_10d':>7}"
        )
        lines.append("  " + "-" * 56)
        for b in output.buckets:
            if b.n == 0:
                continue
            avg10_str = f"{b.avg_10d_return:>+8.2%}" if b.avg_10d_return is not None else f"{'n/a':>8}"
            win10_str = f"{b.win_rate_10d:>7.0%}"    if b.win_rate_10d  is not None else f"{'n/a':>7}"
            marker = " ★" if b.label == output.top_bucket else ("  ✗" if b.label == output.worst_bucket else "   ")
            lines.append(
                f"  {b.label:<9} {b.n:>5}"
                f"  {b.avg_5d_return:>+8.2%}"
                f"  {avg10_str}"
                f"  {b.win_rate_5d:>7.0%}"
                f"  {win10_str}"
                f"{marker}"
            )
        lines.append("")

        # KPI summary
        mono_str = f"{output.priority_monotonicity:>+.3f}"
        calib_str = f"{output.priority_calibration_error:.1%}"
        lines.append(
            f"  最重要KPI:"
        )
        lines.append(
            f"    priority_monotonicity      = {mono_str}"
            f"  (Spearman ρ: 1.0=完全単調)"
        )
        lines.append(
            f"    priority_calibration_error = {calib_str}"
            f"  (隣接バケット違反率: 0%=完全単調)"
        )
        lines.append("")

        # Top / worst
        if output.top_bucket:
            lines.append(f"  Best bucket:  {output.top_bucket}")
        if output.worst_bucket:
            lines.append(f"  Worst bucket: {output.worst_bucket}")
        lines.append("")

        # Calibration verdict
        mono = output.priority_monotonicity
        calib_err = output.priority_calibration_error
        if mono >= 0.7 and calib_err <= 0.2:
            verdict = "強い単調性: priority_score は将来リターンを良好に序列化している"
        elif mono >= 0.4 and calib_err <= 0.4:
            verdict = "中程度の単調性: 傾向あり・継続モニタリング推奨"
        elif mono >= 0.0:
            verdict = "弱い単調性: priority-aware sizing には追加データが必要"
        else:
            verdict = "単調性なし/逆相関: priority_score の再校正を検討"

        lines.append(f"  判定: {verdict}")
        lines.append("──────────────────────────────────────────────────────────────────────")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[PCI] format_calibration_report failed: %s", exc)
        return ""
