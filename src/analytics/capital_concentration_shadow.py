"""
capital_concentration_shadow.py — Capital Concentration Shadow Intelligence

Observes whether priority_score-proportional capital weighting would outperform
equal-weight allocation using materialized 5d forward returns from CP telemetry.

Tasks:
  1. Compute actual_weight = 1/n and shadow_weight = priority_score-proportional
     per date (per portfolio snapshot)
  2. Compare actual_portfolio_return vs shadow_portfolio_return using 5d returns
  3. concentration_alpha = shadow_return - actual_return  [最重要KPI]
  4. Bucket analysis by priority_score gap (max - min) across positions
  5. Append-only capital_concentration_shadow.jsonl (1 record/run)
  6. DRY/LIVE CAPITAL CONCENTRATION SHADOW report

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — single-pass per file; n = records in JSONL
  append-only JSONL — 1 record per run, never rewritten

Key metrics:
  mean_concentration_alpha  — mean(shadow_return - actual_return) per date
  positive_alpha_rate       — fraction of dates where shadow > actual
  gap_high_mean_alpha       — mean alpha when priority_score gap >= 30
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
TARGET_5D = "subsequent_5d_return"

# ── Configuration ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS_DEFAULT: int = 90
MIN_POSITIONS_PER_DATE: int = 2   # dates with <2 positions are skipped (trivial alpha=0)

# Gap buckets: priority_score spread across positions on the same date
GAP_HIGH_THRESHOLD: float = 30.0   # gap >= 30 → high differentiation
GAP_MID_LOW: float = 10.0          # 10 <= gap < 30 → mid; < 10 → low


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DateSnapshot:
    date: str
    n_positions: int
    actual_return: float          # equal-weight portfolio return
    shadow_return: float          # priority-proportional portfolio return
    concentration_alpha: float    # shadow_return - actual_return
    priority_score_gap: float     # max_priority - min_priority across positions
    mean_priority: float          # mean priority_score across positions


@dataclass
class GapBucketStats:
    bucket: str        # "high", "mid", "low"
    n_dates: int
    mean_alpha: float
    mean_actual: float
    mean_shadow: float
    positive_alpha_rate: float


@dataclass
class ConcentrationShadowOutput:
    date: str
    n_records_total: int        # CP records in lookback window
    n_materialized: int         # records with finite 5d return
    n_dates_analyzed: int       # dates with >= MIN_POSITIONS_PER_DATE positions
    lookback_days: int
    min_positions: int
    mean_actual_return: float
    mean_shadow_return: float
    mean_concentration_alpha: float   # 最重要KPI
    positive_alpha_rate: float
    gap_high: GapBucketStats
    gap_mid: GapBucketStats
    gap_low: GapBucketStats
    date_snapshots: List[DateSnapshot] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


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
    """Load all valid JSON records. FAIL_OPEN → empty list."""
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
        logger.warning("[CCS] load_jsonl %s failed: %s", path, exc)
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
        logger.warning("[CCS] append_jsonl failed: %s", exc)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_filter_cp(
    cp_file: Path,
    lookback_days: int,
    today: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Load CP telemetry. Return (within_window, materialized).
    within_window: all records in lookback window.
    materialized:  subset with finite subsequent_5d_return AND finite priority_score.
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
        if rv is not None and ps is not None and ps > 0.0:
            materialized.append(rec)
    return within, materialized


# ── Core analytics ────────────────────────────────────────────────────────────

def group_by_date(records: List[dict]) -> Dict[str, List[dict]]:
    """Group materialized CP records by date. O(n)."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        date = str(rec.get("date", ""))
        if date:
            groups[date].append(rec)
    return dict(groups)


def compute_date_snapshot(date: str, positions: List[dict]) -> Optional[DateSnapshot]:
    """
    Compute one DateSnapshot for a single portfolio date.

    actual_weight_i  = 1/n  (equal weight)
    shadow_weight_i  = priority_score_i / sum(priority_scores)

    Returns None if n < MIN_POSITIONS_PER_DATE or sum_priority == 0.
    FAIL_OPEN.
    """
    try:
        n = len(positions)
        if n < MIN_POSITIONS_PER_DATE:
            return None

        returns = [float(_safe(p.get(TARGET_5D), 0.0)) for p in positions]
        scores  = [float(_safe(p.get("priority_score"), 0.0)) for p in positions]

        sum_scores = sum(scores)
        if sum_scores <= 0.0:
            return None

        actual_return = sum(r / n for r in returns)
        shadow_return = sum(s / sum_scores * r for s, r in zip(scores, returns))

        return DateSnapshot(
            date=date,
            n_positions=n,
            actual_return=round(actual_return, 6),
            shadow_return=round(shadow_return, 6),
            concentration_alpha=round(shadow_return - actual_return, 6),
            priority_score_gap=round(max(scores) - min(scores), 4),
            mean_priority=round(sum_scores / n, 4),
        )
    except Exception as exc:
        logger.debug("[CCS] compute_date_snapshot(%s) failed: %s", date, exc)
        return None


def _gap_bucket(gap: float) -> str:
    if gap >= GAP_HIGH_THRESHOLD:
        return "high"
    if gap >= GAP_MID_LOW:
        return "mid"
    return "low"


def _bucket_stats(snapshots: List[DateSnapshot], bucket: str) -> GapBucketStats:
    """Aggregate stats for one gap bucket."""
    subset = [s for s in snapshots if _gap_bucket(s.priority_score_gap) == bucket]
    n = len(subset)
    if n == 0:
        return GapBucketStats(
            bucket=bucket,
            n_dates=0,
            mean_alpha=0.0,
            mean_actual=0.0,
            mean_shadow=0.0,
            positive_alpha_rate=0.0,
        )
    mean_alpha  = sum(s.concentration_alpha for s in subset) / n
    mean_actual = sum(s.actual_return for s in subset) / n
    mean_shadow = sum(s.shadow_return for s in subset) / n
    pos_rate    = sum(1 for s in subset if s.concentration_alpha > 0) / n
    return GapBucketStats(
        bucket=bucket,
        n_dates=n,
        mean_alpha=round(mean_alpha, 6),
        mean_actual=round(mean_actual, 6),
        mean_shadow=round(mean_shadow, 6),
        positive_alpha_rate=round(pos_rate, 4),
    )


def compute_concentration_shadow(
    materialized: List[dict],
    lookback_days: int,
    today: str,
    n_records_total: int,
) -> ConcentrationShadowOutput:
    """
    Task 1-4: Build per-date snapshots and aggregate.
    FAIL_OPEN — returns empty output on any error.
    """
    empty = ConcentrationShadowOutput(
        date=today,
        n_records_total=n_records_total,
        n_materialized=len(materialized),
        n_dates_analyzed=0,
        lookback_days=lookback_days,
        min_positions=MIN_POSITIONS_PER_DATE,
        mean_actual_return=0.0,
        mean_shadow_return=0.0,
        mean_concentration_alpha=0.0,
        positive_alpha_rate=0.0,
        gap_high=GapBucketStats("high", 0, 0.0, 0.0, 0.0, 0.0),
        gap_mid=GapBucketStats("mid",  0, 0.0, 0.0, 0.0, 0.0),
        gap_low=GapBucketStats("low",  0, 0.0, 0.0, 0.0, 0.0),
        date_snapshots=[],
    )

    try:
        by_date = group_by_date(materialized)
        snapshots: List[DateSnapshot] = []
        for date, positions in sorted(by_date.items()):
            snap = compute_date_snapshot(date, positions)
            if snap is not None:
                snapshots.append(snap)

        nd = len(snapshots)
        if nd == 0:
            return empty

        mean_actual = sum(s.actual_return for s in snapshots) / nd
        mean_shadow = sum(s.shadow_return for s in snapshots) / nd
        mean_alpha  = sum(s.concentration_alpha for s in snapshots) / nd
        pos_rate    = sum(1 for s in snapshots if s.concentration_alpha > 0) / nd

        return ConcentrationShadowOutput(
            date=today,
            n_records_total=n_records_total,
            n_materialized=len(materialized),
            n_dates_analyzed=nd,
            lookback_days=lookback_days,
            min_positions=MIN_POSITIONS_PER_DATE,
            mean_actual_return=round(mean_actual, 6),
            mean_shadow_return=round(mean_shadow, 6),
            mean_concentration_alpha=round(mean_alpha, 6),
            positive_alpha_rate=round(pos_rate, 4),
            gap_high=_bucket_stats(snapshots, "high"),
            gap_mid=_bucket_stats(snapshots, "mid"),
            gap_low=_bucket_stats(snapshots, "low"),
            date_snapshots=snapshots,
        )
    except Exception as exc:
        logger.warning("[CCS] compute_concentration_shadow failed: %s", exc)
        return empty


# ── Main entry point ──────────────────────────────────────────────────────────

def run_concentration_shadow(
    cp_file:       Path,
    output_file:   Path,
    today:         Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> ConcentrationShadowOutput:
    """
    Load CP telemetry, compute concentration shadow, append JSONL.

    Returns ConcentrationShadowOutput always (empty values if insufficient data).
    Never raises.
    """
    if today is None:
        today = datetime.now(_JST).strftime("%Y-%m-%d")

    try:
        within, materialized = load_and_filter_cp(cp_file, lookback_days, today)
        result = compute_concentration_shadow(
            materialized=materialized,
            lookback_days=lookback_days,
            today=today,
            n_records_total=len(within),
        )
        append_concentration_record(result, output_file)
        logger.info(
            "[CCS] n=%d materialized=%d dates=%d alpha=%.4f pos_rate=%.1f%%",
            result.n_records_total,
            result.n_materialized,
            result.n_dates_analyzed,
            result.mean_concentration_alpha,
            result.positive_alpha_rate * 100,
        )
        return result
    except Exception as exc:
        logger.warning("[CCS] run_concentration_shadow failed: %s", exc)
        return ConcentrationShadowOutput(
            date=today,
            n_records_total=0,
            n_materialized=0,
            n_dates_analyzed=0,
            lookback_days=lookback_days,
            min_positions=MIN_POSITIONS_PER_DATE,
            mean_actual_return=0.0,
            mean_shadow_return=0.0,
            mean_concentration_alpha=0.0,
            positive_alpha_rate=0.0,
            gap_high=GapBucketStats("high", 0, 0.0, 0.0, 0.0, 0.0),
            gap_mid=GapBucketStats("mid",  0, 0.0, 0.0, 0.0, 0.0),
            gap_low=GapBucketStats("low",  0, 0.0, 0.0, 0.0, 0.0),
        )


# ── Task 5: JSONL persistence ─────────────────────────────────────────────────

def append_concentration_record(
    output: ConcentrationShadowOutput,
    path: Path,
) -> None:
    """Append one aggregated record to capital_concentration_shadow.jsonl. FAIL_OPEN."""
    try:
        record = {
            "date":                    output.date,
            "n_records_total":         output.n_records_total,
            "n_materialized":          output.n_materialized,
            "n_dates_analyzed":        output.n_dates_analyzed,
            "lookback_days":           output.lookback_days,
            "min_positions":           output.min_positions,
            "mean_actual_return":      output.mean_actual_return,
            "mean_shadow_return":      output.mean_shadow_return,
            "mean_concentration_alpha": output.mean_concentration_alpha,
            "positive_alpha_rate":     output.positive_alpha_rate,
            "gap_high": {
                "n_dates":             output.gap_high.n_dates,
                "mean_alpha":          output.gap_high.mean_alpha,
                "mean_actual":         output.gap_high.mean_actual,
                "mean_shadow":         output.gap_high.mean_shadow,
                "positive_alpha_rate": output.gap_high.positive_alpha_rate,
            },
            "gap_mid": {
                "n_dates":             output.gap_mid.n_dates,
                "mean_alpha":          output.gap_mid.mean_alpha,
                "mean_actual":         output.gap_mid.mean_actual,
                "mean_shadow":         output.gap_mid.mean_shadow,
                "positive_alpha_rate": output.gap_mid.positive_alpha_rate,
            },
            "gap_low": {
                "n_dates":             output.gap_low.n_dates,
                "mean_alpha":          output.gap_low.mean_alpha,
                "mean_actual":         output.gap_low.mean_actual,
                "mean_shadow":         output.gap_low.mean_shadow,
                "positive_alpha_rate": output.gap_low.positive_alpha_rate,
            },
            "schema_version": output.schema_version,
        }
        _append_jsonl(record, path)
    except Exception as exc:
        logger.warning("[CCS] append_concentration_record failed: %s", exc)


# ── Task 6: Report formatter ──────────────────────────────────────────────────

def format_concentration_report(output: ConcentrationShadowOutput) -> str:
    """DRY/LIVE CAPITAL CONCENTRATION SHADOW section. Returns '' if no data. FAIL_OPEN."""
    try:
        if output.n_dates_analyzed == 0:
            return (
                f"\n── CAPITAL CONCENTRATION SHADOW ──────────────────────────────────────\n"
                f"  データ不足: {output.n_materialized}/{output.n_records_total} 件 materialized"
                f" / {output.n_dates_analyzed} 日分析済み\n"
                f"  (最低 {output.min_positions} ポジション/日、ルックバック {output.lookback_days}日)\n"
                f"──────────────────────────────────────────────────────────────────────"
            )

        alpha_sign = "+" if output.mean_concentration_alpha >= 0 else ""
        lines = [
            "",
            "── CAPITAL CONCENTRATION SHADOW ──────────────────────────────────────",
            f"  分析日: {output.date}  期間: {output.lookback_days}日"
            f"  サンプル: {output.n_materialized}件  分析日数: {output.n_dates_analyzed}日",
            "",
            f"  {'実際 (等ウェイト)':>22}: {output.mean_actual_return:>+9.2%}  avg daily 5d return",
            f"  {'影 (priority比例)':>22}: {output.mean_shadow_return:>+9.2%}  avg daily 5d return",
            f"  {'concentration_alpha':>22}: "
            f"{alpha_sign}{output.mean_concentration_alpha:>+9.2%}"
            f"  ← 最重要KPI",
            f"  {'正alpha率':>22}: {output.positive_alpha_rate:>9.1%}"
            f"  ({sum(1 for s in output.date_snapshots if s.concentration_alpha > 0)}"
            f"/{output.n_dates_analyzed}日)",
            "",
        ]

        # Gap bucket analysis
        lines.append("  Priority Score Gap Bucket Analysis:")
        lines.append(
            f"  {'Bucket':<12} {'N日':>5} {'mean_alpha':>11}"
            f"  {'actual':>8}  {'shadow':>8}  {'pos_rate':>8}"
        )
        lines.append("  " + "-" * 60)
        for bkt, label in [
            (output.gap_high, f"gap≥{GAP_HIGH_THRESHOLD:.0f}"),
            (output.gap_mid,  f"gap{GAP_MID_LOW:.0f}-{GAP_HIGH_THRESHOLD:.0f}"),
            (output.gap_low,  f"gap<{GAP_MID_LOW:.0f}"),
        ]:
            if bkt.n_dates == 0:
                lines.append(f"  {label:<12} {'0':>5}  {'n/a':>11}  {'n/a':>8}  {'n/a':>8}  {'n/a':>8}")
            else:
                lines.append(
                    f"  {label:<12} {bkt.n_dates:>5}"
                    f"  {bkt.mean_alpha:>+11.2%}"
                    f"  {bkt.mean_actual:>+8.2%}"
                    f"  {bkt.mean_shadow:>+8.2%}"
                    f"  {bkt.positive_alpha_rate:>8.1%}"
                )
        lines.append("")

        # Verdict
        if output.mean_concentration_alpha > 0.001:
            verdict = "priority比例が等ウェイトを上回る → priority-aware sizing 検討候補"
        elif output.mean_concentration_alpha < -0.001:
            verdict = "等ウェイトが優勢 → priority比例の優位性なし"
        else:
            verdict = "差異軽微 → 継続モニタリング"
        lines.append(f"  判定: {verdict}")
        lines.append("──────────────────────────────────────────────────────────────────────")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[CCS] format_concentration_report failed: %s", exc)
        return ""
