"""
forward_continuation_outcome.py — Forward Continuation Outcome Intelligence

Cross-telemetry aggregation: identifies which intelligence signals best predict
5d/10d forward returns for held positions.

Primary data source:
  CONTINUATION_PRIORITY_TELEMETRY_FILE — priority_score + 7 signals + 5d return
Secondary (joined by (symbol, date)):
  BREAKOUT_QUALITY_TELEMETRY_FILE      — BQ component scores for breakout events

Key metric (最重要KPI):
  information_ratio = mean_5d(top_signal_half) - mean_5d(bottom_signal_half)
  Answers: which signal explains future 5d return most?

Outputs:
  1. signal_ranks       — ranked signals by information_ratio
  2. priority_bucket_ev — E[5d_return] by priority_score bucket [0-30/30-45/45-65/65-80/80-100]
  3. tier_ev            — E[5d_return] by priority_tier
  4. phase_ev           — E[5d_return] by current_phase
  5. combination_ev     — tier × phase E[5d_return]
  6. FCO report section — DRY/LIVE print summary

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — single-pass per file; n = records in JSONL
  append-only JSONL — aggregated KPI record per run (output_file)
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Configuration ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS_DEFAULT: int = 90        # analyse most-recent N calendar days
MIN_RECORDS: int = 5                   # minimum materialized records required
TARGET_5D = "subsequent_5d_return"     # column name in CP telemetry

NUMERIC_SIGNALS: List[str] = [
    "priority_score",
    "compression_score",
    "breakout_quality_score",
    "rsr",
    "rsr_momentum",
    "mfe_pct",
    "hold_days",
]

PRIORITY_BINS: List[float] = [0.0, 30.0, 45.0, 65.0, 80.0, 100.01]  # 100.01 to include 100.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BucketEV:
    bucket_label: str
    lo: float
    hi: float
    n: int
    mean_5d_return: float
    win_rate: float


@dataclass
class CategoryEV:
    category: str
    n: int
    mean_5d_return: float
    win_rate: float


@dataclass
class SignalRank:
    signal_name: str
    information_ratio: float    # mean_5d_top_half - mean_5d_bottom_half
    mean_top_half: float
    mean_bottom_half: float
    win_rate_top: float
    win_rate_bottom: float
    n_total: int


@dataclass
class CombinationEV:
    tier: str
    phase: str
    n: int
    mean_5d_return: float
    win_rate: float


@dataclass
class FcoResult:
    date: str
    n_records_total: int            # CP records in lookback window
    n_materialized: int             # records with non-null subsequent_5d_return
    lookback_days: int
    signal_ranks: List[SignalRank]
    top_signal: str
    top_signal_ir: float            # information_ratio of top_signal
    priority_bucket_ev: List[BucketEV]
    tier_ev: List[CategoryEV]
    phase_ev: List[CategoryEV]
    combination_ev: List[CombinationEV]
    overall_mean_5d: float
    overall_win_rate: float
    schema_version: str = SCHEMA_VERSION


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe_float(v: Any, fallback: Optional[float] = 0.0) -> Optional[float]:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _load_jsonl(path: Path) -> List[dict]:
    """Load all valid JSON records from a JSONL file. FAIL_OPEN → empty list."""
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
        logger.warning("[FCO] load_jsonl %s failed: %s", path, exc)
    return records


def _build_bq_index(bq_path: Path) -> Dict[Tuple[str, str], dict]:
    """Index BQ telemetry by (symbol, date). FAIL_OPEN → empty dict."""
    index: Dict[Tuple[str, str], dict] = {}
    try:
        for rec in _load_jsonl(bq_path):
            sym  = str(rec.get("symbol", ""))
            date = str(rec.get("date", ""))
            if sym and date:
                index[(sym, date)] = rec
    except Exception as exc:
        logger.warning("[FCO] bq_index failed: %s", exc)
    return index


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
        logger.warning("[FCO] append_jsonl failed: %s", exc)


# ── Record preparation ────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def filter_materialized(
    cp_records: List[dict],
    lookback_days: int,
    today: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Split CP records into:
      - within_window: records within lookback_days (all, including un-materialized)
      - materialized:  subset with non-null 5d return → used for analysis

    Returns (within_window, materialized).
    """
    today_dt = _parse_date(today)
    if today_dt is None:
        today_dt = datetime.now(_JST).replace(tzinfo=None)

    cutoff = today_dt - timedelta(days=lookback_days)

    within: List[dict] = []
    materialized: List[dict] = []
    for rec in cp_records:
        date_str = str(rec.get("date", ""))
        dt = _parse_date(date_str)
        if dt is None:
            continue
        if dt < cutoff:
            continue
        within.append(rec)
        rv = rec.get(TARGET_5D)
        if rv is None:
            continue
        try:
            rv_f = float(rv)
            if math.isfinite(rv_f):
                materialized.append(rec)
        except (TypeError, ValueError):
            pass
    return within, materialized


def enrich_with_bq(
    records: List[dict],
    bq_index: Dict[Tuple[str, str], dict],
) -> List[dict]:
    """Merge BQ telemetry fields into CP records (in-memory, no writes). O(n)."""
    enriched = []
    for rec in records:
        merged = dict(rec)
        key = (str(rec.get("symbol", "")), str(rec.get("date", "")))
        bq = bq_index.get(key)
        if bq:
            for bq_field in (
                "breakout_phase_type", "cpr_score", "usr_score",
                "ger_score", "re_score", "close_position_in_range",
                "upper_shadow_ratio",
            ):
                if bq_field in bq and bq_field not in merged:
                    merged[bq_field] = bq[bq_field]
            if "breakout_quality_score" not in merged or merged.get("breakout_quality_score") is None:
                if "breakout_quality_score" in bq:
                    merged["breakout_quality_score"] = bq["breakout_quality_score"]
        enriched.append(merged)
    return enriched


# ── Analytics ─────────────────────────────────────────────────────────────────

def compute_bucket_ev(
    records: List[dict],
    signal_key: str = "priority_score",
    bins: Optional[List[float]] = None,
) -> List[BucketEV]:
    """E[5d_return] per signal bucket. Returns empty list on error."""
    try:
        if bins is None:
            bins = PRIORITY_BINS
        n_buckets = len(bins) - 1
        sums   = [0.0] * n_buckets
        wins   = [0] * n_buckets
        counts = [0] * n_buckets

        for rec in records:
            sv = _safe_float(rec.get(signal_key), None)
            rv = rec.get(TARGET_5D)
            if sv is None or rv is None:
                continue
            try:
                rv_f = float(rv)
                if not math.isfinite(rv_f):
                    continue
            except (TypeError, ValueError):
                continue

            bucket_idx = n_buckets - 1  # default: last bucket
            for i in range(n_buckets - 1):
                if sv < bins[i + 1]:
                    bucket_idx = i
                    break

            sums[bucket_idx] += rv_f
            wins[bucket_idx] += 1 if rv_f > 0 else 0
            counts[bucket_idx] += 1

        result = []
        for i in range(n_buckets):
            n = counts[i]
            label = f"{bins[i]:.0f}-{bins[i+1]:.0f}".rstrip("1").rstrip(".")
            result.append(BucketEV(
                bucket_label=label,
                lo=bins[i],
                hi=bins[i + 1],
                n=n,
                mean_5d_return=round(sums[i] / n, 6) if n > 0 else 0.0,
                win_rate=round(wins[i] / n, 4) if n > 0 else 0.0,
            ))
        return result
    except Exception as exc:
        logger.warning("[FCO] bucket_ev failed: %s", exc)
        return []


def compute_category_ev(
    records: List[dict],
    signal_key: str,
) -> List[CategoryEV]:
    """E[5d_return] per categorical value. Sorted by mean_5d_return descending."""
    try:
        acc: Dict[str, List[float]] = defaultdict(list)
        for rec in records:
            cat = str(rec.get(signal_key, "unknown") or "unknown")
            rv = rec.get(TARGET_5D)
            if rv is None:
                continue
            try:
                rv_f = float(rv)
                if math.isfinite(rv_f):
                    acc[cat].append(rv_f)
            except (TypeError, ValueError):
                pass

        result = []
        for cat, returns in sorted(acc.items(), key=lambda x: -(sum(x[1]) / len(x[1]) if x[1] else 0)):
            n = len(returns)
            result.append(CategoryEV(
                category=cat,
                n=n,
                mean_5d_return=round(sum(returns) / n, 6) if n > 0 else 0.0,
                win_rate=round(sum(1 for r in returns if r > 0) / n, 4) if n > 0 else 0.0,
            ))
        return result
    except Exception as exc:
        logger.warning("[FCO] category_ev(%s) failed: %s", signal_key, exc)
        return []


def rank_signals(records: List[dict]) -> List[SignalRank]:
    """
    Rank numeric signals by information_ratio = mean_5d(top_half) - mean_5d(bottom_half).
    Median split: top half = signals >= median, bottom half = signals < median.
    Returns list sorted by |information_ratio| descending.
    FAIL_OPEN → empty list.
    """
    try:
        results: List[SignalRank] = []
        for sig in NUMERIC_SIGNALS:
            try:
                pairs: List[Tuple[float, float]] = []
                for rec in records:
                    sv = _safe_float(rec.get(sig), None)
                    rv = rec.get(TARGET_5D)
                    if sv is None or rv is None:
                        continue
                    try:
                        rv_f = float(rv)
                        if math.isfinite(sv) and math.isfinite(rv_f):
                            pairs.append((sv, rv_f))
                    except (TypeError, ValueError):
                        pass

                if len(pairs) < MIN_RECORDS:
                    continue

                pairs.sort(key=lambda x: x[0])
                mid = len(pairs) // 2
                bottom = pairs[:mid]
                top    = pairs[mid:]

                if not bottom or not top:
                    continue

                mean_bot = sum(r for _, r in bottom) / len(bottom)
                mean_top = sum(r for _, r in top)    / len(top)
                win_bot  = sum(1 for _, r in bottom if r > 0) / len(bottom)
                win_top  = sum(1 for _, r in top    if r > 0) / len(top)

                results.append(SignalRank(
                    signal_name=sig,
                    information_ratio=round(mean_top - mean_bot, 6),
                    mean_top_half=round(mean_top, 6),
                    mean_bottom_half=round(mean_bot, 6),
                    win_rate_top=round(win_top, 4),
                    win_rate_bottom=round(win_bot, 4),
                    n_total=len(pairs),
                ))
            except Exception as sig_exc:
                logger.debug("[FCO] rank signal %s failed: %s", sig, sig_exc)

        results.sort(key=lambda r: abs(r.information_ratio), reverse=True)
        return results
    except Exception as exc:
        logger.warning("[FCO] rank_signals failed: %s", exc)
        return []


def compute_combination_ev(records: List[dict]) -> List[CombinationEV]:
    """E[5d_return] for tier × phase combinations. Sorted by mean_5d_return desc."""
    try:
        acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for rec in records:
            tier  = str(rec.get("priority_tier", "unknown") or "unknown")
            phase = str(rec.get("current_phase",  "unknown") or "unknown")
            rv    = rec.get(TARGET_5D)
            if rv is None:
                continue
            try:
                rv_f = float(rv)
                if math.isfinite(rv_f):
                    acc[(tier, phase)].append(rv_f)
            except (TypeError, ValueError):
                pass

        result = []
        for (tier, phase), returns in sorted(
            acc.items(),
            key=lambda x: -(sum(x[1]) / len(x[1]) if x[1] else 0),
        ):
            n = len(returns)
            result.append(CombinationEV(
                tier=tier,
                phase=phase,
                n=n,
                mean_5d_return=round(sum(returns) / n, 6) if n > 0 else 0.0,
                win_rate=round(sum(1 for r in returns if r > 0) / n, 4) if n > 0 else 0.0,
            ))
        return result
    except Exception as exc:
        logger.warning("[FCO] combination_ev failed: %s", exc)
        return []


def _overall_stats(records: List[dict]) -> Tuple[float, float]:
    """(overall_mean_5d, overall_win_rate) for materialized records."""
    try:
        returns = []
        for rec in records:
            rv = rec.get(TARGET_5D)
            if rv is None:
                continue
            try:
                rv_f = float(rv)
                if math.isfinite(rv_f):
                    returns.append(rv_f)
            except (TypeError, ValueError):
                pass
        if not returns:
            return 0.0, 0.0
        mean = sum(returns) / len(returns)
        win  = sum(1 for r in returns if r > 0) / len(returns)
        return round(mean, 6), round(win, 4)
    except Exception:
        return 0.0, 0.0


# ── Main entry point ──────────────────────────────────────────────────────────

def run_fco_analysis(
    cp_file:       Path,
    bq_file:       Path,
    output_file:   Path,
    today:         Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> FcoResult:
    """
    Load CP + BQ telemetry, compute cross-signal analytics, append result JSONL.

    Returns FcoResult always (empty/zero values if insufficient data).
    Never raises.
    """
    if today is None:
        today = datetime.now(_JST).strftime("%Y-%m-%d")

    empty = FcoResult(
        date=today,
        n_records_total=0,
        n_materialized=0,
        lookback_days=lookback_days,
        signal_ranks=[],
        top_signal="",
        top_signal_ir=0.0,
        priority_bucket_ev=[],
        tier_ev=[],
        phase_ev=[],
        combination_ev=[],
        overall_mean_5d=0.0,
        overall_win_rate=0.0,
    )

    try:
        # Load sources
        cp_records = _load_jsonl(cp_file)
        bq_index   = _build_bq_index(bq_file)

        # Filter + split
        within, materialized = filter_materialized(cp_records, lookback_days, today)
        if len(materialized) < MIN_RECORDS:
            logger.info(
                "[FCO] insufficient materialized records: %d (need %d)",
                len(materialized), MIN_RECORDS,
            )
            result = FcoResult(
                date=today,
                n_records_total=len(within),
                n_materialized=len(materialized),
                lookback_days=lookback_days,
                signal_ranks=[],
                top_signal="",
                top_signal_ir=0.0,
                priority_bucket_ev=compute_bucket_ev(materialized),
                tier_ev=[],
                phase_ev=[],
                combination_ev=[],
                overall_mean_5d=0.0,
                overall_win_rate=0.0,
            )
            append_fco_record(result, output_file)
            return result

        # Enrich with BQ
        enriched = enrich_with_bq(materialized, bq_index)

        # Compute analytics
        signal_ranks     = rank_signals(enriched)
        bucket_ev        = compute_bucket_ev(enriched)
        tier_ev          = compute_category_ev(enriched, "priority_tier")
        phase_ev         = compute_category_ev(enriched, "current_phase")
        combination_ev   = compute_combination_ev(enriched)
        mean_5d, win_rate = _overall_stats(enriched)

        top_signal    = signal_ranks[0].signal_name if signal_ranks else ""
        top_signal_ir = signal_ranks[0].information_ratio if signal_ranks else 0.0

        result = FcoResult(
            date=today,
            n_records_total=len(within),
            n_materialized=len(enriched),
            lookback_days=lookback_days,
            signal_ranks=signal_ranks,
            top_signal=top_signal,
            top_signal_ir=top_signal_ir,
            priority_bucket_ev=bucket_ev,
            tier_ev=tier_ev,
            phase_ev=phase_ev,
            combination_ev=combination_ev,
            overall_mean_5d=mean_5d,
            overall_win_rate=win_rate,
        )

        append_fco_record(result, output_file)
        logger.info(
            "[FCO] analysis complete: n=%d materialized=%d top_signal=%s IR=%.4f",
            len(within), len(enriched), top_signal, top_signal_ir,
        )
        return result

    except Exception as exc:
        logger.warning("[FCO] run_fco_analysis failed: %s", exc)
        return empty


def append_fco_record(result: FcoResult, output_file: Path) -> None:
    """Append aggregated FCO result to JSONL. FAIL_OPEN."""
    try:
        record = {
            "date":             result.date,
            "n_records_total":  result.n_records_total,
            "n_materialized":   result.n_materialized,
            "lookback_days":    result.lookback_days,
            "top_signal":       result.top_signal,
            "top_signal_ir":    result.top_signal_ir,
            "overall_mean_5d":  result.overall_mean_5d,
            "overall_win_rate": result.overall_win_rate,
            "signal_ranks": [
                {
                    "signal": r.signal_name,
                    "ir":     r.information_ratio,
                    "top":    r.mean_top_half,
                    "bot":    r.mean_bottom_half,
                    "wt":     r.win_rate_top,
                    "wb":     r.win_rate_bottom,
                    "n":      r.n_total,
                }
                for r in result.signal_ranks
            ],
            "priority_bucket_ev": [
                {
                    "bucket": b.bucket_label,
                    "n":      b.n,
                    "mean":   b.mean_5d_return,
                    "win":    b.win_rate,
                }
                for b in result.priority_bucket_ev
            ],
            "tier_ev": [
                {"tier": c.category, "n": c.n, "mean": c.mean_5d_return, "win": c.win_rate}
                for c in result.tier_ev
            ],
            "phase_ev": [
                {"phase": c.category, "n": c.n, "mean": c.mean_5d_return, "win": c.win_rate}
                for c in result.phase_ev
            ],
            "combination_ev": [
                {
                    "tier":  c.tier,
                    "phase": c.phase,
                    "n":     c.n,
                    "mean":  c.mean_5d_return,
                    "win":   c.win_rate,
                }
                for c in result.combination_ev
            ],
            "schema_version": result.schema_version,
        }
        _append_jsonl(record, output_file)
    except Exception as exc:
        logger.warning("[FCO] append_fco_record failed: %s", exc)


# ── Report formatter ──────────────────────────────────────────────────────────

def format_fco_report(result: FcoResult) -> str:
    """DRY/LIVE print report. Returns empty string if no data. FAIL_OPEN."""
    try:
        if result.n_materialized < MIN_RECORDS:
            return (
                f"\n── FORWARD CONTINUATION OUTCOME ──────────────────────────────────\n"
                f"  データ不足: {result.n_materialized}/{result.n_records_total} 件 materialized\n"
                f"  (最低 {MIN_RECORDS} 件必要 / ルックバック {result.lookback_days}日)\n"
                f"──────────────────────────────────────────────────────────────────"
            )

        lines = [
            "",
            "── FORWARD CONTINUATION OUTCOME INTELLIGENCE ─────────────────────",
            f"  分析日: {result.date}  期間: {result.lookback_days}日  "
            f"サンプル: {result.n_materialized}/{result.n_records_total}件",
            f"  全体5d平均: {result.overall_mean_5d:+.2%}  勝率: {result.overall_win_rate:.0%}",
            "",
        ]

        # Section 1: Signal ranking
        if result.signal_ranks:
            lines.append("  [1] 予測シグナルランキング (5d forward return)")
            lines.append(
                f"  {'Rank':<5} {'シグナル':<26} {'IR':>7}  "
                f"{'Top半分':>8}  {'Bot半分':>8}  {'勝率↑':>6}  {'勝率↓':>6}  N"
            )
            lines.append("  " + "-" * 80)
            for rank, sr in enumerate(result.signal_ranks, 1):
                lines.append(
                    f"  {rank:<5} {sr.signal_name:<26}"
                    f" {sr.information_ratio:>+7.2%}"
                    f"  {sr.mean_top_half:>+8.2%}"
                    f"  {sr.mean_bottom_half:>+8.2%}"
                    f"  {sr.win_rate_top:>6.0%}"
                    f"  {sr.win_rate_bottom:>6.0%}"
                    f"  {sr.n_total}"
                )
            lines.append("")

        # Section 2: Priority score buckets
        if result.priority_bucket_ev:
            lines.append("  [2] priority_score バケット別期待値")
            lines.append(f"  {'バケット':<12} {'E[5d]':>8}  {'勝率':>6}  N")
            lines.append("  " + "-" * 40)
            for b in result.priority_bucket_ev:
                if b.n > 0:
                    lines.append(
                        f"  {b.bucket_label:<12}"
                        f" {b.mean_5d_return:>+8.2%}"
                        f"  {b.win_rate:>6.0%}"
                        f"  {b.n}"
                    )
            lines.append("")

        # Section 3: Tier EV
        if result.tier_ev:
            lines.append("  [3] priority_tier 別期待値")
            lines.append(f"  {'Tier':<24} {'E[5d]':>8}  {'勝率':>6}  N")
            lines.append("  " + "-" * 50)
            for c in result.tier_ev:
                if c.n > 0:
                    lines.append(
                        f"  {c.category:<24}"
                        f" {c.mean_5d_return:>+8.2%}"
                        f"  {c.win_rate:>6.0%}"
                        f"  {c.n}"
                    )
            lines.append("")

        # Section 4: Phase EV
        if result.phase_ev:
            lines.append("  [4] current_phase 別期待値")
            lines.append(f"  {'Phase':<26} {'E[5d]':>8}  {'勝率':>6}  N")
            lines.append("  " + "-" * 50)
            for c in result.phase_ev:
                if c.n > 0:
                    lines.append(
                        f"  {c.category:<26}"
                        f" {c.mean_5d_return:>+8.2%}"
                        f"  {c.win_rate:>6.0%}"
                        f"  {c.n}"
                    )
            lines.append("")

        # Section 5: Combination EV (top 6 only to keep report concise)
        top_combos = [c for c in result.combination_ev if c.n >= 2][:6]
        if top_combos:
            lines.append("  [5] tier × phase 組み合わせ期待値 (n≥2, 上位6件)")
            lines.append(f"  {'Tier':<24} {'Phase':<22} {'E[5d]':>8}  {'勝率':>6}  N")
            lines.append("  " + "-" * 70)
            for c in top_combos:
                lines.append(
                    f"  {c.tier:<24}"
                    f" {c.phase:<22}"
                    f" {c.mean_5d_return:>+8.2%}"
                    f"  {c.win_rate:>6.0%}"
                    f"  {c.n}"
                )
            lines.append("")

        # Summary footer
        if result.signal_ranks:
            top = result.signal_ranks[0]
            lines.append(
                f"  最重要KPI: 「{top.signal_name}」が5d return を最も説明"
                f" (IR={top.information_ratio:+.2%})"
            )
        lines.append("──────────────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[FCO] format_fco_report failed: %s", exc)
        return ""
