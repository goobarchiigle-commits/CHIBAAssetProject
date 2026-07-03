"""
src/analytics/missed_alpha_enrichment.py
Missed Alpha Enrichment — join missed alpha records with forward return data.

Purpose:
  Materialize forward returns (5d/10d/MFE) for missed alpha records
  using future_leader_survivability JSONL data.
  analytics only — no execution mutation.

INVARIANTS:
  - pure function pipeline: no side effects on enrichment logic
  - append-only JSONL for enriched records
  - lookahead contamination forbidden: only uses data from survivability logs
    (which contain realized forward returns for past observations)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .missed_alpha_registry import (
    MissedAlphaRecord,
    MissedAlphaReason,
    append_missed_alpha,
    load_missed_alpha,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichedMissedAlphaSummary:
    """
    Aggregated summary of missed alpha enrichment for a date range.
    """
    total_records:           int
    enriched_count:          int
    unenriched_count:        int
    avg_fwd_ret_5d:          Optional[float]
    avg_fwd_ret_10d:         Optional[float]
    avg_fwd_mfe:             Optional[float]
    top_missed_by_fwd_ret:   List[Dict[str, Any]]   # top 5 by fwd_ret_10d
    reason_breakdown:        Dict[str, int]          # reason → count
    high_cost_threshold_pct: float = 0.05           # records with fwd_ret_10d >= 5% = "high cost"
    high_cost_count:         int = 0


def _load_survivability_index(survivability_log_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Build (symbol, entry_date) → survivability record index.
    Used to join missed alpha with forward returns.
    """
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not survivability_log_dir.exists():
        return index
    for p in sorted(survivability_log_dir.glob("future_leader_survivability_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sym = str(rec.get("symbol", ""))
                obs_date = str(rec.get("observation_date", rec.get("entry_date", "")))
                if sym and obs_date:
                    index[(sym, obs_date)] = rec
            except Exception:
                pass
    return index


def enrich_missed_alpha_records(
    records:                 List[MissedAlphaRecord],
    survivability_log_dir:   Path,
) -> List[MissedAlphaRecord]:
    """
    Materialize forward returns for each MissedAlphaRecord.
    Returns new list of enriched records (original list unchanged).
    Unenrichable records returned as-is (enriched=False).
    """
    if not records:
        return []

    surv_index = _load_survivability_index(survivability_log_dir)
    enriched: List[MissedAlphaRecord] = []

    for rec in records:
        if rec.enriched:
            enriched.append(rec)
            continue

        key = (rec.symbol, rec.record_date)
        surv = surv_index.get(key)

        if surv is None:
            enriched.append(rec)
            continue

        fwd_ret_5d  = _opt_float(surv.get("fwd_ret_5d"))
        fwd_ret_10d = _opt_float(surv.get("fwd_ret_10d"))
        fwd_mfe     = _opt_float(surv.get("fwd_mfe"))

        from dataclasses import replace
        enriched.append(replace(
            rec,
            fwd_ret_5d=fwd_ret_5d,
            fwd_ret_10d=fwd_ret_10d,
            fwd_mfe=fwd_mfe,
            enriched=(fwd_ret_5d is not None or fwd_ret_10d is not None),
        ))

    return enriched


def summarize_missed_alpha(records: List[MissedAlphaRecord]) -> EnrichedMissedAlphaSummary:
    """
    Aggregate enriched missed alpha records into summary statistics.
    """
    enriched_recs = [r for r in records if r.enriched]
    reason_breakdown: Dict[str, int] = {}
    for r in records:
        k = r.reason.value if isinstance(r.reason, MissedAlphaReason) else str(r.reason)
        reason_breakdown[k] = reason_breakdown.get(k, 0) + 1

    def _avg(vals: List[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    fwd_5d_vals  = [r.fwd_ret_5d  for r in enriched_recs if r.fwd_ret_5d  is not None]
    fwd_10d_vals = [r.fwd_ret_10d for r in enriched_recs if r.fwd_ret_10d is not None]
    mfe_vals     = [r.fwd_mfe     for r in enriched_recs if r.fwd_mfe     is not None]

    threshold = 0.05
    high_cost = [r for r in enriched_recs if r.fwd_ret_10d is not None and r.fwd_ret_10d >= threshold]

    # Top 5 by fwd_ret_10d descending
    top5 = sorted(
        [r for r in enriched_recs if r.fwd_ret_10d is not None],
        key=lambda r: r.fwd_ret_10d,  # type: ignore[arg-type]
        reverse=True,
    )[:5]

    return EnrichedMissedAlphaSummary(
        total_records=len(records),
        enriched_count=len(enriched_recs),
        unenriched_count=len(records) - len(enriched_recs),
        avg_fwd_ret_5d=_avg(fwd_5d_vals),
        avg_fwd_ret_10d=_avg(fwd_10d_vals),
        avg_fwd_mfe=_avg(mfe_vals),
        top_missed_by_fwd_ret=[r.to_dict() for r in top5],
        reason_breakdown=reason_breakdown,
        high_cost_threshold_pct=threshold,
        high_cost_count=len(high_cost),
    )


def run_missed_alpha_enrichment_hook(
    missed_alpha_log_dir:    Path,
    survivability_log_dir:   Path,
    enriched_log_dir:        Path,
    evaluation_date:         str,
) -> Optional[EnrichedMissedAlphaSummary]:
    """
    Convenience hook for run_live_signal.py integration.
    Loads today's missed alpha records, enriches them, writes enriched JSONL.
    Returns summary or None if no records.
    """
    date_str = evaluation_date.replace("-", "")
    records = load_missed_alpha(missed_alpha_log_dir, date_str=date_str)
    if not records:
        logger.info("[MISSED_ALPHA_ENRICH] no records for %s", evaluation_date)
        return None

    enriched = enrich_missed_alpha_records(records, survivability_log_dir)
    summary  = summarize_missed_alpha(enriched)

    # Write enriched records
    enriched_log_dir.mkdir(parents=True, exist_ok=True)
    out_path = enriched_log_dir / f"enriched_missed_alpha_{date_str}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    logger.info(
        "[MISSED_ALPHA_ENRICH] date=%s total=%d enriched=%d avg_fwd10d=%s high_cost=%d",
        evaluation_date,
        summary.total_records,
        summary.enriched_count,
        f"{summary.avg_fwd_ret_10d:.4f}" if summary.avg_fwd_ret_10d is not None else "N/A",
        summary.high_cost_count,
    )
    return summary


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
