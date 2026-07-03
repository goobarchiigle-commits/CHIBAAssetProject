"""
hold_duration_calibration.py — Hold Duration Calibration Intelligence

Quantifies the relationship between hold_days and forward return, identifying
the optimal holding duration from empirical CP telemetry data.

Tasks:
  1. Decompose hold_days into duration buckets: 1-3d, 4-6d, 7-10d, 11-15d, 16-20d, 21d+
  2. Per-bucket: avg_5d_return, avg_10d_return, win_rate, sample_size
  3. Suppression split: suppressed vs non_suppressed per-bucket and overall comparison
  4. Phase split: compression / breakout / neutral distribution and expected value
  5. Telemetry — append-only hold_duration_calibration.jsonl (1 record/run)
  6. Report — HOLD DURATION CALIBRATION section

Design:
  observation_only — no exit, entry, addon, sizing, or allocator changes
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — single-pass per file; n = records in JSONL
  append-only JSONL — 1 record per run, never rewritten

Key metrics:
  best_duration_bucket      — bucket label with highest avg_5d_return
  duration_ev_spread        — max(avg_5d) − min(avg_5d) across populated buckets
  suppression_return_delta  — avg_5d(suppressed) − avg_5d(non_suppressed)
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"
TARGET_5D  = "subsequent_5d_return"
TARGET_10D = "subsequent_10d_return"   # optional field

# ── Bucket configuration ──────────────────────────────────────────────────────
# (low_inclusive, high_inclusive_or_None)
DURATION_BUCKET_DEFS: List[Tuple[int, Optional[int]]] = [
    (1,  3),
    (4,  6),
    (7,  10),
    (11, 15),
    (16, 20),
    (21, None),
]
DURATION_BUCKET_LABELS: List[str] = [
    "1-3d", "4-6d", "7-10d", "11-15d", "16-20d", "21d+"
]
BUCKET_COUNT: int = len(DURATION_BUCKET_DEFS)

LOOKBACK_DAYS_DEFAULT: int = 90
MIN_BUCKET_SAMPLES:    int = 3

# Known phase values; others collected under "other"
KNOWN_PHASES = ("compression", "breakout", "neutral")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DurationBucket:
    label:          str
    low:            int
    high:           Optional[int]       # None = open-ended (21d+)
    n:              int
    avg_5d_return:  float
    avg_10d_return: Optional[float]
    win_rate_5d:    float
    win_rate_10d:   Optional[float]
    # Suppression sub-groups within this bucket
    suppressed_n:          int = 0
    suppressed_avg_5d:     float = 0.0
    non_suppressed_n:      int = 0
    non_suppressed_avg_5d: float = 0.0


@dataclass
class SuppressionComparison:
    suppressed_n:            int
    suppressed_avg_5d:       float
    suppressed_win_rate:     float
    non_suppressed_n:        int
    non_suppressed_avg_5d:   float
    non_suppressed_win_rate: float
    suppression_return_delta: float   # suppressed − non_suppressed


@dataclass
class PhaseStats:
    phase:        str
    n:            int
    avg_5d_return: float
    win_rate:     float
    modal_bucket: str           # most common duration bucket for this phase


@dataclass
class HoldDurationOutput:
    date:                      str
    n_records_total:           int
    n_materialized:            int
    lookback_days:             int
    min_bucket_samples:        int
    buckets:                   List[DurationBucket]
    n_buckets_populated:       int
    best_duration_bucket:      str    # 最重要KPI
    worst_duration_bucket:     str
    duration_ev_spread:        float  # 最重要KPI
    suppression:               SuppressionComparison
    suppression_return_delta:  float  # 最重要KPI (mirror of suppression.delta)
    phase_stats:               List[PhaseStats]
    phase_duration_leader:     str    # phase with highest avg_5d_return
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


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _safe_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return None


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
        logger.warning("[HDC] load_jsonl %s failed: %s", path, exc)
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
        logger.warning("[HDC] append_jsonl failed: %s", exc)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_filter_cp(
    cp_file: Path,
    lookback_days: int,
    today: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Load CP telemetry. Return (within_window, materialized).
    materialized: finite subsequent_5d_return AND finite hold_days ≥ 1.
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
        rv  = _safe(rec.get(TARGET_5D), None)
        hd  = _safe_int(rec.get("hold_days"))
        if rv is not None and hd is not None and hd >= 1:
            materialized.append(rec)
    return within, materialized


# ── Task 1+2: Duration bucket decomposition ───────────────────────────────────

def _bucket_index(hold_days: int) -> int:
    """Map hold_days to bucket index [0, BUCKET_COUNT-1]. Clamps below 1 to 0."""
    hd = max(1, hold_days)
    for i, (lo, hi) in enumerate(DURATION_BUCKET_DEFS):
        if hi is None or hd <= hi:
            if hd >= lo:
                return i
    return BUCKET_COUNT - 1


def compute_duration_buckets(records: List[dict]) -> List[DurationBucket]:
    """
    Task 1+2 + Task 3 sub-groups: single-pass O(n) accumulation.
    """
    # Per-bucket accumulators
    ns      = [0]   * BUCKET_COUNT
    sum5    = [0.0] * BUCKET_COUNT
    sum10   = [0.0] * BUCKET_COUNT
    n10     = [0]   * BUCKET_COUNT
    wins5   = [0]   * BUCKET_COUNT
    wins10  = [0]   * BUCKET_COUNT
    # Suppression sub-groups
    sup_n    = [0]   * BUCKET_COUNT
    sup_s5   = [0.0] * BUCKET_COUNT
    nsup_n   = [0]   * BUCKET_COUNT
    nsup_s5  = [0.0] * BUCKET_COUNT

    for rec in records:
        hd = _safe_int(rec.get("hold_days"))
        r5 = _safe(rec.get(TARGET_5D), None)
        if hd is None or r5 is None:
            continue
        idx = _bucket_index(hd)
        ns[idx]   += 1
        sum5[idx] += r5
        wins5[idx] += 1 if r5 > 0 else 0
        r10 = _safe(rec.get(TARGET_10D), None)
        if r10 is not None:
            sum10[idx]  += r10
            n10[idx]    += 1
            wins10[idx] += 1 if r10 > 0 else 0
        supp = _safe_bool(rec.get("suppression_eligible"))
        if supp is True:
            sup_n[idx]  += 1
            sup_s5[idx] += r5
        elif supp is False:
            nsup_n[idx]  += 1
            nsup_s5[idx] += r5

    buckets: List[DurationBucket] = []
    for i, (lo, hi) in enumerate(DURATION_BUCKET_DEFS):
        n = ns[i]
        avg5  = round(sum5[i] / n, 6)  if n > 0 else 0.0
        wr5   = round(wins5[i] / n, 4) if n > 0 else 0.0
        avg10: Optional[float] = None
        wr10:  Optional[float] = None
        if n10[i] > 0:
            avg10 = round(sum10[i] / n10[i], 6)
            wr10  = round(wins10[i] / n10[i], 4)
        sup_avg  = round(sup_s5[i] / sup_n[i], 6)  if sup_n[i]  > 0 else 0.0
        nsup_avg = round(nsup_s5[i] / nsup_n[i], 6) if nsup_n[i] > 0 else 0.0
        buckets.append(DurationBucket(
            label=DURATION_BUCKET_LABELS[i],
            low=lo,
            high=hi,
            n=n,
            avg_5d_return=avg5,
            avg_10d_return=avg10,
            win_rate_5d=wr5,
            win_rate_10d=wr10,
            suppressed_n=sup_n[i],
            suppressed_avg_5d=sup_avg,
            non_suppressed_n=nsup_n[i],
            non_suppressed_avg_5d=nsup_avg,
        ))
    return buckets


# ── Task 3: Suppression comparison ────────────────────────────────────────────

def compute_suppression_comparison(records: List[dict]) -> SuppressionComparison:
    """
    Task 3: Overall suppressed vs non_suppressed comparison across all durations.
    FAIL_OPEN → zero-filled result.
    """
    sup_n   = 0;  sup_s5  = 0.0;  sup_wins  = 0
    nsup_n  = 0;  nsup_s5 = 0.0;  nsup_wins = 0

    for rec in records:
        r5   = _safe(rec.get(TARGET_5D), None)
        supp = _safe_bool(rec.get("suppression_eligible"))
        if r5 is None or supp is None:
            continue
        if supp:
            sup_n   += 1
            sup_s5  += r5
            sup_wins += 1 if r5 > 0 else 0
        else:
            nsup_n   += 1
            nsup_s5  += r5
            nsup_wins += 1 if r5 > 0 else 0

    sup_avg  = round(sup_s5  / sup_n,  6) if sup_n  > 0 else 0.0
    nsup_avg = round(nsup_s5 / nsup_n, 6) if nsup_n > 0 else 0.0
    sup_wr   = round(sup_wins  / sup_n,  4) if sup_n  > 0 else 0.0
    nsup_wr  = round(nsup_wins / nsup_n, 4) if nsup_n > 0 else 0.0
    delta    = round(sup_avg - nsup_avg, 6)

    return SuppressionComparison(
        suppressed_n=sup_n,
        suppressed_avg_5d=sup_avg,
        suppressed_win_rate=sup_wr,
        non_suppressed_n=nsup_n,
        non_suppressed_avg_5d=nsup_avg,
        non_suppressed_win_rate=nsup_wr,
        suppression_return_delta=delta,
    )


# ── Task 4: Phase split ───────────────────────────────────────────────────────

def compute_phase_stats(records: List[dict]) -> List[PhaseStats]:
    """
    Task 4: Per-phase avg_5d_return, win_rate, n, and modal duration bucket.

    Phases: compression, breakout, neutral, other (catch-all).
    Phases with n=0 are omitted from the result.
    FAIL_OPEN → [].
    """
    # Accumulators: phase → {sum5, wins, n, bucket_count[]}
    ph_s5:    Dict[str, float]     = defaultdict(float)
    ph_wins:  Dict[str, int]       = defaultdict(int)
    ph_n:     Dict[str, int]       = defaultdict(int)
    ph_buck:  Dict[str, List[int]] = defaultdict(lambda: [0] * BUCKET_COUNT)

    for rec in records:
        r5   = _safe(rec.get(TARGET_5D), None)
        hd   = _safe_int(rec.get("hold_days"))
        phase = str(rec.get("current_phase", "")).strip().lower()
        if r5 is None or hd is None:
            continue
        if phase not in KNOWN_PHASES:
            phase = "other"
        idx = _bucket_index(max(1, hd))
        ph_s5[phase]    += r5
        ph_wins[phase]  += 1 if r5 > 0 else 0
        ph_n[phase]     += 1
        ph_buck[phase][idx] += 1

    result: List[PhaseStats] = []
    for phase in list(KNOWN_PHASES) + ["other"]:
        n = ph_n[phase]
        if n == 0:
            continue
        avg5 = round(ph_s5[phase] / n, 6)
        wr   = round(ph_wins[phase] / n, 4)
        buck_counts = ph_buck[phase]
        modal_idx = buck_counts.index(max(buck_counts))
        result.append(PhaseStats(
            phase=phase,
            n=n,
            avg_5d_return=avg5,
            win_rate=wr,
            modal_bucket=DURATION_BUCKET_LABELS[modal_idx],
        ))
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def run_hold_duration_calibration(
    cp_file:       Path,
    output_file:   Path,
    today:         Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    min_samples:   int = MIN_BUCKET_SAMPLES,
) -> HoldDurationOutput:
    """
    Load CP telemetry, compute hold duration calibration, append JSONL.
    Returns HoldDurationOutput always. Never raises.
    """
    if today is None:
        today = datetime.now(_JST).strftime("%Y-%m-%d")

    def _empty() -> HoldDurationOutput:
        return HoldDurationOutput(
            date=today,
            n_records_total=0,
            n_materialized=0,
            lookback_days=lookback_days,
            min_bucket_samples=min_samples,
            buckets=[
                DurationBucket(
                    label=DURATION_BUCKET_LABELS[i],
                    low=DURATION_BUCKET_DEFS[i][0],
                    high=DURATION_BUCKET_DEFS[i][1],
                    n=0, avg_5d_return=0.0, avg_10d_return=None,
                    win_rate_5d=0.0, win_rate_10d=None,
                )
                for i in range(BUCKET_COUNT)
            ],
            n_buckets_populated=0,
            best_duration_bucket="",
            worst_duration_bucket="",
            duration_ev_spread=0.0,
            suppression=SuppressionComparison(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0),
            suppression_return_delta=0.0,
            phase_stats=[],
            phase_duration_leader="",
        )

    try:
        within, materialized = load_and_filter_cp(cp_file, lookback_days, today)
        buckets    = compute_duration_buckets(materialized)
        suppression = compute_suppression_comparison(materialized)
        phase_stats = compute_phase_stats(materialized)

        populated = [b for b in buckets if b.n >= min_samples]
        n_pop = len(populated)

        best_bkt  = ""
        worst_bkt = ""
        ev_spread = 0.0
        if populated:
            best_bkt  = max(populated, key=lambda b: b.avg_5d_return).label
            worst_bkt = min(populated, key=lambda b: b.avg_5d_return).label
            ev_spread = round(
                max(b.avg_5d_return for b in populated) -
                min(b.avg_5d_return for b in populated),
                6,
            )

        phase_leader = ""
        if phase_stats:
            phase_leader = max(phase_stats, key=lambda p: p.avg_5d_return).phase

        result = HoldDurationOutput(
            date=today,
            n_records_total=len(within),
            n_materialized=len(materialized),
            lookback_days=lookback_days,
            min_bucket_samples=min_samples,
            buckets=buckets,
            n_buckets_populated=n_pop,
            best_duration_bucket=best_bkt,
            worst_duration_bucket=worst_bkt,
            duration_ev_spread=ev_spread,
            suppression=suppression,
            suppression_return_delta=suppression.suppression_return_delta,
            phase_stats=phase_stats,
            phase_duration_leader=phase_leader,
        )

        append_duration_record(result, output_file)
        logger.info(
            "[HDC] n=%d mat=%d buckets=%d best=%s spread=%.4f sup_delta=%.4f",
            result.n_records_total,
            result.n_materialized,
            result.n_buckets_populated,
            result.best_duration_bucket,
            result.duration_ev_spread,
            result.suppression_return_delta,
        )
        return result

    except Exception as exc:
        logger.warning("[HDC] run_hold_duration_calibration failed: %s", exc)
        return _empty()


# ── Task 5: JSONL persistence ─────────────────────────────────────────────────

def append_duration_record(output: HoldDurationOutput, path: Path) -> None:
    """Append one aggregated record to hold_duration_calibration.jsonl. FAIL_OPEN."""
    try:
        sup = output.suppression
        record = {
            "date":                   output.date,
            "n_records_total":        output.n_records_total,
            "n_materialized":         output.n_materialized,
            "lookback_days":          output.lookback_days,
            "min_bucket_samples":     output.min_bucket_samples,
            "n_buckets_populated":    output.n_buckets_populated,
            "best_duration_bucket":   output.best_duration_bucket,
            "worst_duration_bucket":  output.worst_duration_bucket,
            "duration_ev_spread":     output.duration_ev_spread,
            "suppression_return_delta": output.suppression_return_delta,
            "phase_duration_leader":  output.phase_duration_leader,
            "suppression": {
                "suppressed_n":            sup.suppressed_n,
                "suppressed_avg_5d":       sup.suppressed_avg_5d,
                "suppressed_win_rate":     sup.suppressed_win_rate,
                "non_suppressed_n":        sup.non_suppressed_n,
                "non_suppressed_avg_5d":   sup.non_suppressed_avg_5d,
                "non_suppressed_win_rate": sup.non_suppressed_win_rate,
                "suppression_return_delta": sup.suppression_return_delta,
            },
            "phase_stats": [
                {
                    "phase":         p.phase,
                    "n":             p.n,
                    "avg_5d_return": p.avg_5d_return,
                    "win_rate":      p.win_rate,
                    "modal_bucket":  p.modal_bucket,
                }
                for p in output.phase_stats
            ],
            "buckets": [
                {
                    "label":               b.label,
                    "low":                 b.low,
                    "high":                b.high,
                    "n":                   b.n,
                    "avg_5d_return":       b.avg_5d_return,
                    "avg_10d_return":      b.avg_10d_return,
                    "win_rate_5d":         b.win_rate_5d,
                    "win_rate_10d":        b.win_rate_10d,
                    "suppressed_n":        b.suppressed_n,
                    "suppressed_avg_5d":   b.suppressed_avg_5d,
                    "non_suppressed_n":    b.non_suppressed_n,
                    "non_suppressed_avg_5d": b.non_suppressed_avg_5d,
                }
                for b in output.buckets
            ],
            "schema_version": output.schema_version,
        }
        _append_jsonl(record, path)
    except Exception as exc:
        logger.warning("[HDC] append_duration_record failed: %s", exc)


# ── Task 6: Report formatter ──────────────────────────────────────────────────

def format_duration_report(output: HoldDurationOutput) -> str:
    """DRY/LIVE HOLD DURATION CALIBRATION section. Returns '' if no data. FAIL_OPEN."""
    try:
        if output.n_buckets_populated == 0:
            return (
                f"\n── HOLD DURATION CALIBRATION ─────────────────────────────────────────\n"
                f"  データ不足: {output.n_materialized}/{output.n_records_total} 件 materialized"
                f" / populated buckets: {output.n_buckets_populated}\n"
                f"  (最低 {output.min_bucket_samples} 件/bucket、ルックバック {output.lookback_days}日)\n"
                f"──────────────────────────────────────────────────────────────────────"
            )

        lines = [
            "",
            "── HOLD DURATION CALIBRATION ─────────────────────────────────────────",
            f"  分析日: {output.date}  期間: {output.lookback_days}日"
            f"  サンプル: {output.n_materialized}/{output.n_records_total}件"
            f"  populated: {output.n_buckets_populated}/{BUCKET_COUNT}",
            "",
        ]

        # Duration bucket table
        lines.append(
            f"  {'Bucket':<9} {'N':>5}  {'avg_5d':>8}  {'avg_10d':>8}"
            f"  {'win_5d':>7}  {'sup_n':>5}  {'sup_avg':>8}  {'nsup_avg':>9}"
        )
        lines.append("  " + "-" * 72)
        for b in output.buckets:
            if b.n == 0:
                continue
            avg10_s = f"{b.avg_10d_return:>+8.2%}" if b.avg_10d_return is not None else f"{'n/a':>8}"
            marker  = " ★" if b.label == output.best_duration_bucket else (
                      "  ✗" if b.label == output.worst_duration_bucket else "   ")
            sup_s  = f"{b.suppressed_avg_5d:>+8.2%}" if b.suppressed_n > 0 else f"{'n/a':>8}"
            nsup_s = f"{b.non_suppressed_avg_5d:>+9.2%}" if b.non_suppressed_n > 0 else f"{'n/a':>9}"
            lines.append(
                f"  {b.label:<9} {b.n:>5}"
                f"  {b.avg_5d_return:>+8.2%}"
                f"  {avg10_s}"
                f"  {b.win_rate_5d:>7.0%}"
                f"  {b.suppressed_n:>5}"
                f"  {sup_s}"
                f"  {nsup_s}"
                f"{marker}"
            )
        lines.append("")

        # Suppression comparison
        sup = output.suppression
        if sup.suppressed_n > 0 or sup.non_suppressed_n > 0:
            delta_sign = "+" if sup.suppression_return_delta >= 0 else ""
            lines.append("  Suppression Comparison (全期間):")
            lines.append(
                f"    suppressed    : n={sup.suppressed_n:>4}"
                f"  avg_5d={sup.suppressed_avg_5d:>+8.2%}"
                f"  win={sup.suppressed_win_rate:.0%}"
            )
            lines.append(
                f"    non-suppressed: n={sup.non_suppressed_n:>4}"
                f"  avg_5d={sup.non_suppressed_avg_5d:>+8.2%}"
                f"  win={sup.non_suppressed_win_rate:.0%}"
            )
            lines.append(
                f"    suppression_return_delta = {delta_sign}{sup.suppression_return_delta:>+.2%}"
                f"  ← 最重要KPI"
            )
            lines.append("")

        # Phase breakdown
        if output.phase_stats:
            lines.append("  Phase Distribution:")
            for ps in sorted(output.phase_stats, key=lambda p: p.avg_5d_return, reverse=True):
                leader_mark = " ★" if ps.phase == output.phase_duration_leader else ""
                lines.append(
                    f"    {ps.phase:<14} n={ps.n:>4}"
                    f"  avg_5d={ps.avg_5d_return:>+8.2%}"
                    f"  win={ps.win_rate:.0%}"
                    f"  modal={ps.modal_bucket}{leader_mark}"
                )
            lines.append("")

        # KPI summary
        lines.append(f"  最重要KPI:")
        lines.append(f"    best_duration_bucket     = {output.best_duration_bucket}")
        lines.append(f"    worst_duration_bucket    = {output.worst_duration_bucket}")
        lines.append(f"    duration_ev_spread       = {output.duration_ev_spread:>+.2%}")
        lines.append(
            f"    suppression_return_delta = {output.suppression_return_delta:>+.2%}"
        )
        if output.phase_duration_leader:
            lines.append(
                f"    phase_duration_leader    = {output.phase_duration_leader}"
            )
        lines.append("──────────────────────────────────────────────────────────────────────")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[HDC] format_duration_report failed: %s", exc)
        return ""
