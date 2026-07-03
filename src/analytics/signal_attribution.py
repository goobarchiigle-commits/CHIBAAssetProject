"""
signal_attribution.py — Signal Attribution Intelligence

Quantifies which intelligence signals explain future 5d return (continuation alpha)
using data already accumulated in CP telemetry + slot_pressure telemetry.

Tasks:
  1. Signal IR ranking    — IR = mean_5d(top_half) - mean_5d(bottom_half)
  2. Predictive ranking   — ordered list by |IR|, top signal identified
  3. Signal Interaction   — 2×2 table for top-2 signal pair (HH/HL/LH/LL)
  4. Stability check      — min_sample_size gate; skip unstable signals
  5. Telemetry            — append-only signal_attribution.jsonl (1 record/run)
  6. Report               — DRY/LIVE SIGNAL ATTRIBUTION section

Signals evaluated:
  From CP telemetry (per symbol):
    priority_score, compression_score, breakout_quality_score,
    rsr, rsr_momentum, mfe_pct, giveback_ratio, hold_days
  From slot_pressure (date-level, joined by date):
    slot_pressure_score

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — single-pass per file; n = records in JSONL
  append-only JSONL — 1 record per run, never rewritten

Key metrics:
  top_signal        — signal with highest |IR|
  top_signal_ir     — its information ratio
  top_interaction   — "signal_a × signal_b" label for top pair
  interaction_ev    — E[5d_return] for HH bucket of top pair
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
TARGET_5D = "subsequent_5d_return"

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_SAMPLE_SIZE: int = 5
LOOKBACK_DAYS_DEFAULT: int = 90
TOP_N_SIGNALS: int = 5          # signals to show in report
TOP_INTERACTION_N: int = 2      # top-N signals for 2×2 interaction

# Signals sourced from CP telemetry
CP_NUMERIC_SIGNALS: List[str] = [
    "priority_score",
    "compression_score",
    "breakout_quality_score",
    "rsr",
    "rsr_momentum",
    "mfe_pct",
    "giveback_ratio",
    "hold_days",
]

# Signals added by enrichment (joined by date from slot_pressure)
ENRICHED_SIGNALS: List[str] = ["slot_pressure_score"]

ALL_SIGNALS: List[str] = CP_NUMERIC_SIGNALS + ENRICHED_SIGNALS


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SignalAttributionResult:
    signal_name: str
    signal_ir: float           # top_half_mean - bottom_half_mean (最重要KPI)
    sample_size: int           # n pairs used (both signal + 5d return non-null)
    top_half_mean: float       # mean 5d return for records with signal >= median
    bottom_half_mean: float    # mean 5d return for records with signal < median
    top_half_win_rate: float   # win_rate (>0) for top half
    bottom_half_win_rate: float
    median_split: float        # the median value used for splitting


@dataclass
class InteractionBucket:
    signal_a: str
    signal_b: str
    category: str              # "HH", "HL", "LH", "LL"
    n: int
    mean_5d_return: float
    win_rate: float


@dataclass
class SignalAttributionOutput:
    date: str
    n_records_total: int       # CP records in lookback window
    n_materialized: int        # records with non-null 5d return
    lookback_days: int
    min_sample_size: int
    attribution_results: List[SignalAttributionResult]  # sorted desc by |IR|
    top_signal: str
    top_signal_ir: float
    top_interaction: str       # e.g. "priority_score × compression_score"
    interaction_ev: float      # E[5d] for HH bucket of top pair
    interaction_table: List[InteractionBucket]
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
        logger.warning("[SA] load_jsonl %s failed: %s", path, exc)
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
        logger.warning("[SA] append_jsonl failed: %s", exc)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_filter_cp(
    cp_file: Path,
    lookback_days: int,
    today: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Load CP telemetry. Return (within_window, materialized).
    within_window: all records in lookback window.
    materialized:  subset with finite subsequent_5d_return.
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


def build_sp_index(sp_file: Path) -> Dict[str, float]:
    """
    Build date → slot_pressure_score index from SLOT_PRESSURE_FILE.
    One record per day; uses most recent record for a given date.
    FAIL_OPEN → empty dict.
    """
    index: Dict[str, float] = {}
    try:
        for rec in _load_jsonl(sp_file):
            date = str(rec.get("date", ""))
            score_raw = rec.get("pressure_score")
            if not date or score_raw is None:
                continue
            sv = _safe(score_raw, None)
            if sv is not None:
                index[date] = sv
    except Exception as exc:
        logger.warning("[SA] build_sp_index failed: %s", exc)
    return index


def enrich_with_sp(
    records: List[dict],
    sp_index: Dict[str, float],
) -> List[dict]:
    """
    Add slot_pressure_score to each CP record by joining on date.
    Records without a matching date entry get slot_pressure_score=None.
    In-memory enrichment — no writes. O(n).
    """
    result = []
    for rec in records:
        date = str(rec.get("date", ""))
        score = sp_index.get(date)
        if score is not None:
            merged = dict(rec)
            merged["slot_pressure_score"] = score
            result.append(merged)
        else:
            result.append(rec)
    return result


# ── Core analytics ────────────────────────────────────────────────────────────

def _extract_pairs(
    records: List[dict],
    signal_name: str,
) -> List[Tuple[float, float]]:
    """Extract finite (signal_value, 5d_return) pairs. O(n)."""
    pairs: List[Tuple[float, float]] = []
    for rec in records:
        sv = _safe(rec.get(signal_name), None)
        rv = _safe(rec.get(TARGET_5D), None)
        if sv is not None and rv is not None:
            pairs.append((sv, rv))
    return pairs


def compute_signal_ir(
    records: List[dict],
    signal_name: str,
    min_samples: int = MIN_SAMPLE_SIZE,
) -> Optional[SignalAttributionResult]:
    """
    Task 1 + 4: Compute IR for one signal. Returns None if below min_samples.

    Median split:
      top_half  = records with signal_value >= median → mean_5d
      bottom_half = records with signal_value < median → mean_5d
      IR = top_half_mean - bottom_half_mean
    """
    try:
        pairs = _extract_pairs(records, signal_name)
        if len(pairs) < min_samples:
            return None

        pairs.sort(key=lambda x: x[0])
        n = len(pairs)
        mid = n // 2
        # median: midpoint of sorted list
        median_val = pairs[mid][0] if n % 2 == 1 else (pairs[mid - 1][0] + pairs[mid][0]) / 2.0

        bottom = pairs[:mid]
        top    = pairs[mid:]
        if not bottom or not top:
            return None

        def _stats(half: List[Tuple[float, float]]) -> Tuple[float, float]:
            rets = [r for _, r in half]
            m = sum(rets) / len(rets)
            w = sum(1 for r in rets if r > 0) / len(rets)
            return round(m, 6), round(w, 4)

        mean_bot, win_bot = _stats(bottom)
        mean_top, win_top = _stats(top)

        return SignalAttributionResult(
            signal_name=signal_name,
            signal_ir=round(mean_top - mean_bot, 6),
            sample_size=n,
            top_half_mean=mean_top,
            bottom_half_mean=mean_bot,
            top_half_win_rate=win_top,
            bottom_half_win_rate=win_bot,
            median_split=round(median_val, 4),
        )
    except Exception as exc:
        logger.debug("[SA] compute_signal_ir(%s) failed: %s", signal_name, exc)
        return None


def rank_all_signals(
    records: List[dict],
    min_samples: int = MIN_SAMPLE_SIZE,
) -> List[SignalAttributionResult]:
    """
    Task 2: Compute IR for all signals; return sorted desc by |IR|.
    Signals below min_samples are silently excluded.
    FAIL_OPEN → empty list.
    """
    results: List[SignalAttributionResult] = []
    try:
        for sig in ALL_SIGNALS:
            r = compute_signal_ir(records, sig, min_samples)
            if r is not None:
                results.append(r)
        results.sort(key=lambda r: abs(r.signal_ir), reverse=True)
    except Exception as exc:
        logger.warning("[SA] rank_all_signals failed: %s", exc)
    return results


def compute_interaction(
    records: List[dict],
    signal_a: str,
    signal_b: str,
    min_samples: int = MIN_SAMPLE_SIZE,
) -> List[InteractionBucket]:
    """
    Task 3: 2×2 interaction table for signal_a × signal_b.

    Each record labelled H (>= median) or L (< median) for each signal.
    Groups: HH, HL, LH, LL.
    Buckets with n < 1 are included with n=0 (for display completeness).
    FAIL_OPEN → empty list.
    """
    try:
        # Build valid triples: (signal_a_val, signal_b_val, 5d_return)
        triples: List[Tuple[float, float, float]] = []
        for rec in records:
            va = _safe(rec.get(signal_a), None)
            vb = _safe(rec.get(signal_b), None)
            rv = _safe(rec.get(TARGET_5D), None)
            if va is not None and vb is not None and rv is not None:
                triples.append((va, vb, rv))

        if len(triples) < min_samples:
            return []

        # Compute medians via sort
        sorted_a = sorted(t[0] for t in triples)
        sorted_b = sorted(t[1] for t in triples)
        n = len(triples)
        mid = n // 2
        med_a = sorted_a[mid] if n % 2 == 1 else (sorted_a[mid - 1] + sorted_a[mid]) / 2.0
        med_b = sorted_b[mid] if n % 2 == 1 else (sorted_b[mid - 1] + sorted_b[mid]) / 2.0

        # Accumulate
        cats = {"HH": [], "HL": [], "LH": [], "LL": []}
        for va, vb, rv in triples:
            ha = "H" if va >= med_a else "L"
            hb = "H" if vb >= med_b else "L"
            cats[ha + hb].append(rv)

        result: List[InteractionBucket] = []
        for cat in ("HH", "HL", "LH", "LL"):
            returns = cats[cat]
            nc = len(returns)
            result.append(InteractionBucket(
                signal_a=signal_a,
                signal_b=signal_b,
                category=cat,
                n=nc,
                mean_5d_return=round(sum(returns) / nc, 6) if nc > 0 else 0.0,
                win_rate=round(sum(1 for r in returns if r > 0) / nc, 4) if nc > 0 else 0.0,
            ))
        return result
    except Exception as exc:
        logger.warning("[SA] compute_interaction(%s×%s) failed: %s", signal_a, signal_b, exc)
        return []


# ── Main entry point ──────────────────────────────────────────────────────────

def run_signal_attribution(
    cp_file:       Path,
    sp_file:       Path,
    output_file:   Path,
    today:         Optional[str] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    min_samples:   int = MIN_SAMPLE_SIZE,
) -> SignalAttributionOutput:
    """
    Load telemetry, compute signal IR rankings and interactions, append JSONL.

    Returns SignalAttributionOutput always (empty values if insufficient data).
    Never raises.
    """
    if today is None:
        today = datetime.now(_JST).strftime("%Y-%m-%d")

    empty = SignalAttributionOutput(
        date=today,
        n_records_total=0,
        n_materialized=0,
        lookback_days=lookback_days,
        min_sample_size=min_samples,
        attribution_results=[],
        top_signal="",
        top_signal_ir=0.0,
        top_interaction="",
        interaction_ev=0.0,
        interaction_table=[],
    )

    try:
        # Load + filter CP telemetry
        within, materialized = load_and_filter_cp(cp_file, lookback_days, today)

        # Enrich with slot_pressure
        sp_index   = build_sp_index(sp_file)
        enriched   = enrich_with_sp(materialized, sp_index)

        if len(enriched) < min_samples:
            logger.info(
                "[SA] insufficient records: %d (need %d)", len(enriched), min_samples,
            )
            result = SignalAttributionOutput(
                date=today,
                n_records_total=len(within),
                n_materialized=len(enriched),
                lookback_days=lookback_days,
                min_sample_size=min_samples,
                attribution_results=[],
                top_signal="",
                top_signal_ir=0.0,
                top_interaction="",
                interaction_ev=0.0,
                interaction_table=[],
            )
            append_attribution_record(result, output_file)
            return result

        # Task 1+2: rank signals
        ranked = rank_all_signals(enriched, min_samples)

        top_signal    = ranked[0].signal_name if ranked else ""
        top_signal_ir = ranked[0].signal_ir   if ranked else 0.0

        # Task 3: interaction for top-2 signals
        interaction_table: List[InteractionBucket] = []
        top_interaction = ""
        interaction_ev  = 0.0

        if len(ranked) >= TOP_INTERACTION_N:
            sig_a = ranked[0].signal_name
            sig_b = ranked[1].signal_name
            interaction_table = compute_interaction(enriched, sig_a, sig_b, min_samples)
            if interaction_table:
                top_interaction = f"{sig_a} × {sig_b}"
                hh = next((b for b in interaction_table if b.category == "HH"), None)
                if hh:
                    interaction_ev = hh.mean_5d_return

        result = SignalAttributionOutput(
            date=today,
            n_records_total=len(within),
            n_materialized=len(enriched),
            lookback_days=lookback_days,
            min_sample_size=min_samples,
            attribution_results=ranked,
            top_signal=top_signal,
            top_signal_ir=top_signal_ir,
            top_interaction=top_interaction,
            interaction_ev=interaction_ev,
            interaction_table=interaction_table,
        )

        append_attribution_record(result, output_file)
        logger.info(
            "[SA] attribution: n=%d materialized=%d top=%s IR=%.4f interaction=%s ev=%.4f",
            len(within), len(enriched), top_signal, top_signal_ir,
            top_interaction, interaction_ev,
        )
        return result

    except Exception as exc:
        logger.warning("[SA] run_signal_attribution failed: %s", exc)
        return empty


def append_attribution_record(output: SignalAttributionOutput, path: Path) -> None:
    """Append one aggregated record to signal_attribution.jsonl. FAIL_OPEN."""
    try:
        record = {
            "date":             output.date,
            "n_records_total":  output.n_records_total,
            "n_materialized":   output.n_materialized,
            "lookback_days":    output.lookback_days,
            "min_sample_size":  output.min_sample_size,
            "top_signal":       output.top_signal,
            "top_signal_ir":    output.top_signal_ir,
            "top_interaction":  output.top_interaction,
            "interaction_ev":   output.interaction_ev,
            "attribution_results": [
                {
                    "signal":   r.signal_name,
                    "ir":       r.signal_ir,
                    "n":        r.sample_size,
                    "top_mean": r.top_half_mean,
                    "bot_mean": r.bottom_half_mean,
                    "top_win":  r.top_half_win_rate,
                    "bot_win":  r.bottom_half_win_rate,
                    "median":   r.median_split,
                }
                for r in output.attribution_results
            ],
            "interaction_table": [
                {
                    "signal_a": b.signal_a,
                    "signal_b": b.signal_b,
                    "cat":      b.category,
                    "n":        b.n,
                    "mean_5d":  b.mean_5d_return,
                    "win":      b.win_rate,
                }
                for b in output.interaction_table
            ],
            "schema_version": output.schema_version,
        }
        _append_jsonl(record, path)
    except Exception as exc:
        logger.warning("[SA] append_attribution_record failed: %s", exc)


# ── Report formatter ──────────────────────────────────────────────────────────

def format_attribution_report(output: SignalAttributionOutput) -> str:
    """DRY/LIVE SIGNAL ATTRIBUTION section. Returns '' if no data. FAIL_OPEN."""
    try:
        if output.n_materialized < output.min_sample_size:
            return (
                f"\n── SIGNAL ATTRIBUTION INTELLIGENCE ──────────────────────────────────\n"
                f"  データ不足: {output.n_materialized}/{output.n_records_total} 件 materialized\n"
                f"  (最低 {output.min_sample_size} 件必要 / ルックバック {output.lookback_days}日)\n"
                f"──────────────────────────────────────────────────────────────────────"
            )

        lines = [
            "",
            "── SIGNAL ATTRIBUTION INTELLIGENCE ──────────────────────────────────",
            f"  分析日: {output.date}  期間: {output.lookback_days}日"
            f"  サンプル: {output.n_materialized}/{output.n_records_total}件"
            f"  最低N: {output.min_sample_size}",
            "",
        ]

        # Top Signals
        top_n = output.attribution_results[:TOP_N_SIGNALS]
        if top_n:
            lines.append("  Top Signals (5d return Information Ratio):")
            lines.append(
                f"  {'Rank':<5} {'Signal':<28} {'IR':>8}   "
                f"{'Top_mean':>9}  {'Bot_mean':>9}  {'Top_win':>7}  {'Bot_win':>7}  N"
            )
            lines.append("  " + "-" * 82)
            for i, r in enumerate(top_n, 1):
                lines.append(
                    f"  {i:<5} {r.signal_name:<28}"
                    f" {r.signal_ir:>+8.2%}"
                    f"   {r.top_half_mean:>+9.2%}"
                    f"  {r.bottom_half_mean:>+9.2%}"
                    f"  {r.top_half_win_rate:>7.0%}"
                    f"  {r.bottom_half_win_rate:>7.0%}"
                    f"  {r.sample_size}"
                )
            lines.append("")

        # Interaction
        if output.interaction_table and output.top_interaction:
            lines.append(f"  Top Interaction: {output.top_interaction}")
            lines.append("")

            # Determine signal names from first entry
            sig_a = output.interaction_table[0].signal_a if output.interaction_table else "sig_a"
            sig_b = output.interaction_table[0].signal_b if output.interaction_table else "sig_b"
            b_map = {b.category: b for b in output.interaction_table}

            def _cell(cat: str) -> str:
                b = b_map.get(cat)
                if b is None or b.n == 0:
                    return f"{'n/a':>14}"
                return f"{b.mean_5d_return:>+7.2%} (n={b.n:>2})"

            lines.append(
                f"  {'':28} {sig_b + '_HIGH':<20}  {sig_b + '_LOW':<20}"
            )
            lines.append(
                f"  {sig_a + '_HIGH':<28} {_cell('HH'):<20}  {_cell('HL'):<20}"
            )
            lines.append(
                f"  {sig_a + '_LOW':<28} {_cell('LH'):<20}  {_cell('LL'):<20}"
            )
            lines.append("")

            # Best bucket
            best = max(output.interaction_table, key=lambda b: b.mean_5d_return if b.n > 0 else -999)
            if best.n > 0:
                lines.append(
                    f"  Best: {best.signal_a}_{'HIGH' if best.category[0]=='H' else 'LOW'}"
                    f" × {best.signal_b}_{'HIGH' if best.category[1]=='H' else 'LOW'}"
                    f"  avg_5d={best.mean_5d_return:+.2%}  win={best.win_rate:.0%}  n={best.n}"
                )
                lines.append("")

        # KPI footer
        if output.top_signal:
            lines.append(
                f"  最重要KPI: 「{output.top_signal}」 IR={output.top_signal_ir:+.2%}"
            )
        if output.top_interaction:
            lines.append(
                f"  最強組み合わせ: {output.top_interaction}"
                f"  HH期待値={output.interaction_ev:+.2%}"
            )
        lines.append("──────────────────────────────────────────────────────────────────────")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[SA] format_attribution_report failed: %s", exc)
        return ""
