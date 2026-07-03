"""
src/analytics/exits/reports.py — Exit analytics persistence + report generation

Persistence:
  append_exit_record     — JSONL append (dedup by record_id)
  append_policy_replay   — JSONL append (dedup by replay_id)
  persist_holding_path   — per-trade JSON (atomic, keyed by record_id)
  write_parquet          — pandas parquet for analytics tables
  write_report_atomic    — atomic temp→replace markdown write

Report generation:
  generate_markdown_report — deterministic markdown from records + replays

All outputs: deterministic for fixed inputs. Append-only JSONL. UTC timestamps.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.analytics.exits.models import (
    ExitConvexityRecord,
    HoldingPath,
    PolicyReplayResult,
    RunnerRetentionSummary,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

_SENTINEL = object()


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via temp file + os.replace (Windows-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _load_jsonl_ids(path: Path, id_field: str) -> set:
    if not path.exists():
        return set()
    ids: set = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                v = d.get(id_field)
                if v:
                    ids.add(v)
            except Exception:
                pass
    except Exception:
        pass
    return ids


# ── ExitConvexityRecord persistence ──────────────────────────────────────────

def append_exit_record(record: ExitConvexityRecord, path: Path) -> bool:
    """
    Append one ExitConvexityRecord to JSONL. Dedup by record_id.

    Returns True if appended, False if duplicate or error.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = _load_jsonl_ids(path, "record_id")
        if record.record_id in existing_ids:
            logger.debug("[EXIT_RECORDS] duplicate record_id=%s skipped", record.record_id)
            return False
        line = _canonical_json(record.to_dict())
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception as exc:
        logger.warning("[EXIT_RECORDS] append failed record_id=%s: %s", record.record_id, exc)
        return False


def load_exit_records(path: Path) -> List[ExitConvexityRecord]:
    """Load all ExitConvexityRecord from JSONL. Skips corrupt lines."""
    if not path.exists():
        return []
    records: List[ExitConvexityRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(ExitConvexityRecord.from_dict(d))
        except Exception as exc:
            logger.warning("[EXIT_RECORDS] parse error: %s", exc)
    return records


# ── PolicyReplayResult persistence ───────────────────────────────────────────

def append_policy_replay(result: PolicyReplayResult, path: Path) -> bool:
    """Append one PolicyReplayResult. Dedup by replay_id."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = _load_jsonl_ids(path, "replay_id")
        if result.replay_id in existing_ids:
            logger.debug("[POLICY_REPLAY] duplicate replay_id=%s skipped", result.replay_id)
            return False
        line = _canonical_json(result.to_dict())
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception as exc:
        logger.warning("[POLICY_REPLAY] append failed replay_id=%s: %s", result.replay_id, exc)
        return False


def load_policy_replays(path: Path) -> List[PolicyReplayResult]:
    if not path.exists():
        return []
    results: List[PolicyReplayResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            results.append(PolicyReplayResult.from_dict(d))
        except Exception as exc:
            logger.warning("[POLICY_REPLAY] parse error: %s", exc)
    return results


# ── HoldingPath persistence ───────────────────────────────────────────────────

def persist_holding_path(hp: HoldingPath, directory: Path) -> None:
    """
    Persist one HoldingPath as a per-trade JSON file.
    Filename: {record_id[:16]}.json (deterministic, Windows-safe).
    Atomic write: old file replaced atomically.
    """
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{hp.record_id[:16]}.json"
    target = directory / filename
    content = _canonical_json(hp.to_dict())
    _atomic_write_text(target, content)


def load_holding_path(record_id: str, directory: Path) -> Optional[HoldingPath]:
    """Load HoldingPath by record_id. Returns None if not found."""
    filename = f"{record_id[:16]}.json"
    path = directory / filename
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return HoldingPath.from_dict(d)
    except Exception as exc:
        logger.warning("[HOLDING_PATH] load failed record_id=%s: %s", record_id, exc)
        return None


# ── Parquet analytics tables ──────────────────────────────────────────────────

def write_exit_records_parquet(records: List[ExitConvexityRecord], path: Path) -> None:
    """Write ExitConvexityRecord list to parquet via pandas. Atomic temp→replace."""
    try:
        import pandas as pd
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in records]
        df = pd.DataFrame(rows)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_parquet_", suffix=".parquet")
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False, engine="pyarrow")
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception as exc:
        logger.warning("[EXIT_PARQUET] write failed: %s", exc)


def write_policy_replays_parquet(results: List[PolicyReplayResult], path: Path) -> None:
    """Write PolicyReplayResult list to parquet via pandas."""
    try:
        import pandas as pd
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_parquet_", suffix=".parquet")
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False, engine="pyarrow")
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception as exc:
        logger.warning("[POLICY_REPLAY_PARQUET] write failed: %s", exc)


# ── Markdown report generation ────────────────────────────────────────────────

def _pct(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _fmt(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _percentile(values: List[float], p: float) -> float:
    """Compute p-th percentile (0-100) of sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def generate_markdown_report(
    records: List[ExitConvexityRecord],
    policy_replays: Optional[List[PolicyReplayResult]] = None,
    summary: Optional[RunnerRetentionSummary] = None,
    report_date: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """
    Generate deterministic markdown report from exit analytics.

    Same inputs → byte-identical output (uses sorted ordering throughout).
    """
    policy_replays = policy_replays or []
    date_str = report_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    generated_at = datetime.now(timezone.utc).isoformat()

    lines: List[str] = []
    a = lines.append

    a(f"# Exit Convexity Analytics Report — {date_str}")
    a(f"")
    a(f"Generated: {generated_at}  ")
    a(f"Total completed trades: {len(records)}  ")
    a(f"")

    # ── Summary Statistics ─────────────────────────────────────────────────────
    a("## Summary Statistics")
    a("")
    if records:
        mfes = [r.max_favorable_excursion for r in records]
        maes = [r.max_adverse_excursion for r in records]
        returns = [r.realized_return for r in records]
        captures = [r.convexity_capture_rate for r in records]
        efficiencies = [r.exit_efficiency_score for r in records]
        givebacks = [r.pnl_giveback_ratio for r in records]

        n = len(records)
        a(f"| Metric | Mean | P25 | P50 | P75 |")
        a(f"|--------|------|-----|-----|-----|")
        a(f"| Realized Return | {_pct(sum(returns)/n)} | {_pct(_percentile(returns,25))} | {_pct(_percentile(returns,50))} | {_pct(_percentile(returns,75))} |")
        a(f"| MFE | {_pct(sum(mfes)/n)} | {_pct(_percentile(mfes,25))} | {_pct(_percentile(mfes,50))} | {_pct(_percentile(mfes,75))} |")
        a(f"| MAE | {_pct(sum(maes)/n)} | {_pct(_percentile(maes,25))} | {_pct(_percentile(maes,50))} | {_pct(_percentile(maes,75))} |")
        a(f"| Convexity Capture Rate | {_pct(sum(captures)/n)} | {_pct(_percentile(captures,25))} | {_pct(_percentile(captures,50))} | {_pct(_percentile(captures,75))} |")
        a(f"| Exit Efficiency Score | {_fmt(sum(efficiencies)/n,4)} | {_fmt(_percentile(efficiencies,25),4)} | {_fmt(_percentile(efficiencies,50),4)} | {_fmt(_percentile(efficiencies,75),4)} |")
        a(f"| PnL Giveback Ratio | {_pct(sum(givebacks)/n)} | {_pct(_percentile(givebacks,25))} | {_pct(_percentile(givebacks,50))} | {_pct(_percentile(givebacks,75))} |")
    else:
        a("_No records._")
    a("")

    # ── Convexity Leakage Diagnostics ─────────────────────────────────────────
    a("## Convexity Leakage Diagnostics")
    a("")
    if records:
        premature = [r for r in records if r.is_premature_exit]
        truncated = [r for r in records if r.is_convexity_truncated]
        vol_insens = [r for r in records if r.is_volatility_insensitive]
        runner_kept = [r for r in records if r.is_runner_retained]
        n = len(records)
        a(f"| Condition | Count | Rate |")
        a(f"|-----------|-------|------|")
        a(f"| Premature Exit | {len(premature)} | {_pct(len(premature)/n)} |")
        a(f"| Convexity Truncated | {len(truncated)} | {_pct(len(truncated)/n)} |")
        a(f"| Volatility Insensitive | {len(vol_insens)} | {_pct(len(vol_insens)/n)} |")
        a(f"| Runner Retained | {len(runner_kept)} | {_pct(len(runner_kept)/n)} |")
    else:
        a("_No records._")
    a("")

    # ── Exit Inefficiency Rankings ─────────────────────────────────────────────
    a("## Exit Inefficiency Rankings")
    a("")
    a("### Lowest Convexity Capture (Worst Exits)")
    a("")
    worst = sorted(records, key=lambda r: r.convexity_capture_rate)[:top_n]
    if worst:
        a(f"| Symbol | Capture Rate | Realized Return | MFE | Exit Reason |")
        a(f"|--------|-------------|----------------|-----|-------------|")
        for r in worst:
            a(f"| {r.symbol} | {_pct(r.convexity_capture_rate)} | {_pct(r.realized_return)} | {_pct(r.max_favorable_excursion)} | {r.exit_reason} |")
    else:
        a("_No records._")
    a("")

    a("### Largest PnL Giveback")
    a("")
    biggest_giveback = sorted(records, key=lambda r: -r.pnl_giveback_ratio)[:top_n]
    if biggest_giveback:
        a(f"| Symbol | Giveback Ratio | MFE | Realized Return | Hold Days |")
        a(f"|--------|---------------|-----|----------------|-----------|")
        for r in biggest_giveback:
            a(f"| {r.symbol} | {_pct(r.pnl_giveback_ratio)} | {_pct(r.max_favorable_excursion)} | {_pct(r.realized_return)} | {r.holding_duration_days:.1f} |")
    else:
        a("_No records._")
    a("")

    # ── Top Alpha Leakage Trades ───────────────────────────────────────────────
    a("## Top Alpha Leakage Trades")
    a("")
    with_leakage = [r for r in records if r.post_exit_alpha_leakage is not None]
    top_leakage = sorted(with_leakage, key=lambda r: -(r.post_exit_alpha_leakage or 0.0))[:top_n]
    if top_leakage:
        a(f"| Symbol | Post-Exit 5d Return | Realized Return | Runner Continuation Score |")
        a(f"|--------|--------------------|-----------------|-----------------------------|")
        for r in top_leakage:
            a(f"| {r.symbol} | {_pct(r.post_exit_return_5d)} | {_pct(r.realized_return)} | {_fmt(r.runner_continuation_score)} |")
    else:
        a("_No post-exit continuation data available._")
    a("")

    # ── Largest Post-Exit Continuations ───────────────────────────────────────
    a("## Largest Post-Exit Continuations")
    a("")
    with_5d = [r for r in records if r.post_exit_return_5d is not None]
    top_cont = sorted(with_5d, key=lambda r: -(r.post_exit_return_5d or 0.0))[:top_n]
    if top_cont:
        a(f"| Symbol | Post-Exit 5d | Benchmark-Relative 5d | Sector-Relative | Is Premature |")
        a(f"|--------|-------------|----------------------|-----------------|--------------|")
        for r in top_cont:
            a(f"| {r.symbol} | {_pct(r.post_exit_return_5d)} | {_pct(r.benchmark_relative_post_exit_return_5d)} | {_pct(r.sector_relative_post_exit_return)} | {'Yes' if r.is_premature_exit else 'No'} |")
    else:
        a("_No post-exit continuation data available._")
    a("")

    # ── Trailing Policy Comparisons ───────────────────────────────────────────
    a("## Trailing Policy Comparisons")
    a("")
    if policy_replays:
        policy_names = sorted(set(r.policy_name for r in policy_replays))
        a(f"| Policy | Trades | Mean Hyp Return | Mean Convexity Retention | Mean Runner Retention |")
        a(f"|--------|--------|----------------|--------------------------|----------------------|")
        for pname in policy_names:
            prec = [r for r in policy_replays if r.policy_name == pname]
            if not prec:
                continue
            n = len(prec)
            mean_ret = sum(r.hypothetical_realized_return for r in prec) / n
            mean_conv = sum(r.convexity_retention for r in prec) / n
            mean_rr = sum(r.runner_retention for r in prec) / n
            a(f"| {pname} | {n} | {_pct(mean_ret)} | {_pct(mean_conv)} | {_pct(mean_rr)} |")
    else:
        a("_No policy replay data available._")
    a("")

    # ── Trend Persistence Observations ────────────────────────────────────────
    a("## Trend Persistence Observations")
    a("")
    trend_vals = [r.relative_trend_persistence_score for r in records if r.relative_trend_persistence_score is not None]
    persist_caps = [r.trend_persistence_capture for r in records if r.trend_persistence_capture is not None]
    if trend_vals:
        mean_trend = sum(trend_vals) / len(trend_vals)
        a(f"- Mean relative trend persistence score: {_fmt(mean_trend)}")
        a(f"- Trades with strong post-exit trend (score > 0.67): {sum(1 for v in trend_vals if v > 0.67)}")
    if persist_caps:
        mean_cap = sum(persist_caps) / len(persist_caps)
        a(f"- Mean trend persistence capture: {_pct(mean_cap)}")
    if not trend_vals and not persist_caps:
        a("_No trend persistence data available._")
    a("")

    # ── Runner Retention Summary ───────────────────────────────────────────────
    if summary is not None:
        a("## Runner Retention Summary")
        a("")
        a(f"| Metric | Value |")
        a(f"|--------|-------|")
        a(f"| Total Trades | {summary.total_trades} |")
        a(f"| Premature Exit Rate | {_pct(summary.premature_exit_rate)} |")
        a(f"| Convexity Truncation Rate | {_pct(summary.convexity_truncation_rate)} |")
        a(f"| Mean Convexity Capture Rate | {_pct(summary.mean_convexity_capture_rate)} |")
        a(f"| Mean Exit Efficiency Score | {_fmt(summary.mean_exit_efficiency_score)} |")
        a(f"| Mean PnL Giveback Ratio | {_pct(summary.mean_pnl_giveback_ratio)} |")
        a(f"| Mean Runner Retention Rate | {_pct(summary.mean_runner_retention_rate)} |")
        if summary.mean_post_exit_alpha_leakage is not None:
            a(f"| Mean Post-Exit Alpha Leakage | {_pct(summary.mean_post_exit_alpha_leakage)} |")
        a("")

    a("---")
    a(f"_Generated by src/analytics/exits/reports.py — schema_version={SCHEMA_VERSION}_")
    a("")

    return "\n".join(lines)


def write_report_atomic(content: str, path: Path) -> None:
    """Atomically write markdown report. Overwrites existing."""
    try:
        _atomic_write_text(path, content)
    except Exception as exc:
        logger.warning("[EXIT_REPORT] write failed path=%s: %s", path, exc)
        raise
