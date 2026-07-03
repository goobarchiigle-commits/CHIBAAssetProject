"""
src/analytics/future_leader_expectancy.py — Future Leader Expectancy Analytics

Purpose:
  Validate predictive quality of future_leader_score, absolute penalties,
  and RSR zone classification against materialized forward returns.

INVARIANTS (変更禁止):
  - observation_only: no execution path connection
  - CLEAN observations only: PARTIAL_FAILURE / MAJOR_FAILURE excluded
  - pure functions: I/O and aggregation separated
  - no weighted expectancy / Bayesian adjustment / ML ranking / regime segmentation

禁止:
  - signal mutation
  - allocator linkage
  - governance linkage
  - auto activation
  - deployable universe mutation
  - ranking replacement
  - predictive_entry_enabled mutation
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum sample count for statistically meaningful display
_MIN_SAMPLE: int = 5

# CLEAN filter: data_quality_weight must be exactly 1.0 (CLEAN only)
_CLEAN_QUALITY_MIN: float = 1.0

# Score bucket definitions: (label, lo, hi) where hi=None means ∞
_SCORE_BUCKETS: List[Tuple[str, float, Optional[float]]] = [
    ("70+",   70.0, None),
    ("60-70", 60.0, 70.0),
    ("50-60", 50.0, 60.0),
    ("<50",    0.0, 50.0),
]

# Absolute penalty names (matches future_leader_screener.py Phase 1 penalties)
_PENALTY_NAMES: List[str] = [
    "negative_accumulation",
    "low_drift",
    "weak_close_strength",
    "negative_rs_delta",
]

# RSR zone display order
_RSR_ZONES: List[str] = ["emerging", "mature", "weak"]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpectancySummary:
    """Aggregate forward-return stats for CLEAN complete observations."""
    record_count:      int
    mean_return_5d:    Optional[float]
    median_return_5d:  Optional[float]
    mean_return_10d:   Optional[float]
    median_return_10d: Optional[float]
    median_mfe_10d:    Optional[float]
    median_mae_10d:    Optional[float]


@dataclass
class BucketStats:
    """Forward-return stats for one score bucket or RSR zone."""
    label:             str
    count:             int
    mean_return_5d:    Optional[float]
    median_return_5d:  Optional[float]
    mean_return_10d:   Optional[float]
    median_return_10d: Optional[float]
    median_mfe_10d:    Optional[float]
    median_mae_10d:    Optional[float]


@dataclass
class PenaltyStats:
    """Forward-return stats for observations where a specific penalty triggered."""
    penalty_name:      str
    triggered_count:   int
    mean_return_10d:   Optional[float]
    median_return_10d: Optional[float]
    median_mfe_10d:    Optional[float]
    median_mae_10d:    Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers (separated from computation)
# ─────────────────────────────────────────────────────────────────────────────

def load_survivability_records(log_dir: Path) -> List[dict]:
    """Load all survivability JSONL records from log_dir."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("future_leader_survivability_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


def load_candidate_records(candidates_log_path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Load candidates JSONL → {(symbol, observation_date): record}.
    First occurrence per key wins (stable dedup across re-runs).
    Used for penalty attribution: provides absolute_penalties field.
    """
    lookup: Dict[Tuple[str, str], dict] = {}
    if not candidates_log_path.exists():
        return lookup
    try:
        for line in candidates_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                dt  = rec.get("observation_date", "")
                if sym and dt:
                    key = (sym, dt)
                    if key not in lookup:
                        lookup[key] = rec
            except Exception:
                pass
    except Exception:
        pass
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_clean_complete(records: List[dict]) -> List[dict]:
    """
    CLEAN-only filter for expectancy analysis.
    Excludes: PARTIAL_FAILURE, MAJOR_FAILURE, insufficient_forward_data=True.
    """
    return [
        r for r in records
        if float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN
        and not r.get("insufficient_forward_data", True)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation utilities (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _opt_mean(vals: List[float]) -> Optional[float]:
    return round(float(np.mean(vals)), 6) if vals else None


def _opt_median(vals: List[float]) -> Optional[float]:
    return round(float(np.median(vals)), 6) if vals else None


def _bucket_stats(label: str, recs: List[dict]) -> BucketStats:
    fwd5  = [float(r["forward_return_5d"])  for r in recs if r.get("forward_return_5d")  is not None]
    fwd10 = [float(r["forward_return_10d"]) for r in recs if r.get("forward_return_10d") is not None]
    mfe   = [float(r["mfe_10d"])  for r in recs if r.get("mfe_10d")  is not None]
    mae   = [float(r["mae_10d"])  for r in recs if r.get("mae_10d")  is not None]
    return BucketStats(
        label=label,
        count=len(recs),
        mean_return_5d=_opt_mean(fwd5),
        median_return_5d=_opt_median(fwd5),
        mean_return_10d=_opt_mean(fwd10),
        median_return_10d=_opt_median(fwd10),
        median_mfe_10d=_opt_median(mfe),
        median_mae_10d=_opt_median(mae),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure computation functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_expectancy_summary(records: List[dict]) -> ExpectancySummary:
    """
    Aggregate basic metrics from CLEAN complete records.
    records must already be filtered by filter_clean_complete().
    """
    fwd5  = [float(r["forward_return_5d"])  for r in records if r.get("forward_return_5d")  is not None]
    fwd10 = [float(r["forward_return_10d"]) for r in records if r.get("forward_return_10d") is not None]
    mfe   = [float(r["mfe_10d"])  for r in records if r.get("mfe_10d")  is not None]
    mae   = [float(r["mae_10d"])  for r in records if r.get("mae_10d")  is not None]
    return ExpectancySummary(
        record_count=len(records),
        mean_return_5d=_opt_mean(fwd5),
        median_return_5d=_opt_median(fwd5),
        mean_return_10d=_opt_mean(fwd10),
        median_return_10d=_opt_median(fwd10),
        median_mfe_10d=_opt_median(mfe),
        median_mae_10d=_opt_median(mae),
    )


def compute_score_bucket_analysis(records: List[dict]) -> List[BucketStats]:
    """
    Bucket CLEAN complete records by future_leader_score.
    Buckets: 70+, 60-70, 50-60, <50 (ordered highest first).
    records must already be filtered by filter_clean_complete().
    """
    result: List[BucketStats] = []
    for label, lo, hi in _SCORE_BUCKETS:
        if hi is None:
            bucket = [r for r in records if float(r.get("future_leader_score", 0.0)) >= lo]
        else:
            bucket = [r for r in records if lo <= float(r.get("future_leader_score", 0.0)) < hi]
        result.append(_bucket_stats(label, bucket))
    return result


def compute_penalty_attribution(
    survivability_records: List[dict],
    candidate_lookup: Dict[Tuple[str, str], dict],
) -> List[PenaltyStats]:
    """
    Attribute forward returns to absolute penalty triggers.

    Multi-penalty overlap permitted: each record may appear in multiple groups.
    Interaction analysis is intentionally excluded (single attribution only).
    survivability_records must already be filtered by filter_clean_complete().
    """
    penalty_groups: Dict[str, List[dict]] = {name: [] for name in _PENALTY_NAMES}

    for surv_rec in survivability_records:
        sym  = surv_rec.get("symbol", "")
        dt   = surv_rec.get("observation_date", "")
        cand = candidate_lookup.get((sym, dt), {})
        abs_pens: dict = cand.get("absolute_penalties", {})
        for pname in _PENALTY_NAMES:
            if pname in abs_pens:
                penalty_groups[pname].append(surv_rec)

    result: List[PenaltyStats] = []
    for pname in _PENALTY_NAMES:
        recs  = penalty_groups[pname]
        fwd10 = [float(r["forward_return_10d"]) for r in recs if r.get("forward_return_10d") is not None]
        mfe   = [float(r["mfe_10d"]) for r in recs if r.get("mfe_10d") is not None]
        mae   = [float(r["mae_10d"]) for r in recs if r.get("mae_10d") is not None]
        result.append(PenaltyStats(
            penalty_name=pname,
            triggered_count=len(recs),
            mean_return_10d=_opt_mean(fwd10),
            median_return_10d=_opt_median(fwd10),
            median_mfe_10d=_opt_median(mfe),
            median_mae_10d=_opt_median(mae),
        ))
    return result


def compute_rsr_zone_analysis(records: List[dict]) -> List[BucketStats]:
    """
    Group CLEAN complete records by rsr_zone.
    Order: emerging, mature, weak.
    records must already be filtered by filter_clean_complete().
    """
    result: List[BucketStats] = []
    for zone in _RSR_ZONES:
        bucket = [r for r in records if r.get("rsr_zone") == zone]
        result.append(_bucket_stats(zone, bucket))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Render functions (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "N/A"


def _render_bucket(b: BucketStats) -> List[str]:
    if b.count < _MIN_SAMPLE:
        return [f"{b.label}:", f"  count: {b.count}", "  (insufficient sample)"]
    lines = [f"{b.label}:", f"  count: {b.count}"]
    if b.mean_return_5d is not None:
        lines.append(f"  5d mean:    {_pct(b.mean_return_5d)}")
    if b.mean_return_10d is not None:
        lines.append(f"  10d mean:   {_pct(b.mean_return_10d)}")
    if b.median_mfe_10d is not None:
        lines.append(f"  median MFE: {_pct(b.median_mfe_10d)}")
    if b.median_mae_10d is not None:
        lines.append(f"  median MAE: {_pct(b.median_mae_10d)}")
    return lines


def render_expectancy_report(
    summary: ExpectancySummary,
    score_buckets: List[BucketStats],
    penalty_stats: List[PenaltyStats],
    zone_stats: List[BucketStats],
) -> str:
    """
    Render full expectancy analysis report as text.
    Pure function: no I/O.
    """
    if summary.record_count < _MIN_SAMPLE:
        return (
            "=== FUTURE LEADER EXPECTANCY ===\n"
            "No statistically meaningful sample yet\n"
            f"(CLEAN complete records: {summary.record_count},"
            f" min required: {_MIN_SAMPLE})"
        )

    lines: List[str] = [
        "=== FUTURE LEADER EXPECTANCY ===",
        f"CLEAN complete records: {summary.record_count}",
        f"  10d mean:   {_pct(summary.mean_return_10d)}",
        f"  10d median: {_pct(summary.median_return_10d)}",
        f"  5d mean:    {_pct(summary.mean_return_5d)}",
        f"  median MFE: {_pct(summary.median_mfe_10d)}",
        f"  median MAE: {_pct(summary.median_mae_10d)}",
    ]

    lines += ["", "=== SCORE BUCKET EXPECTANCY ==="]
    for b in score_buckets:
        lines += _render_bucket(b)

    lines += ["", "=== PENALTY EXPECTANCY ==="]
    has_any = any(p.triggered_count >= _MIN_SAMPLE for p in penalty_stats)
    if not has_any:
        lines.append("(insufficient penalty-triggered samples)")
    else:
        for p in penalty_stats:
            if p.triggered_count == 0:
                continue
            if p.triggered_count < _MIN_SAMPLE:
                lines += [
                    f"{p.penalty_name}:",
                    f"  count: {p.triggered_count}",
                    "  (insufficient sample)",
                ]
            else:
                lines += [
                    f"{p.penalty_name}:",
                    f"  count: {p.triggered_count}",
                    f"  10d mean:   {_pct(p.mean_return_10d)}",
                    f"  median MFE: {_pct(p.median_mfe_10d)}",
                    f"  median MAE: {_pct(p.median_mae_10d)}",
                ]

    lines += ["", "=== RSR ZONE EXPECTANCY ==="]
    for b in zone_stats:
        lines += _render_bucket(b)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for report integration
# ─────────────────────────────────────────────────────────────────────────────

def format_future_leader_expectancy_section(
    survivability_log_dir: Path,
    candidates_log_path: Path,
) -> str:
    """
    Load, compute, and render full expectancy section.
    Fail-open: returns empty string on any error.
    observation_only=True: no execution path connection.
    """
    try:
        all_recs   = load_survivability_records(survivability_log_dir)
        clean_recs = filter_clean_complete(all_recs)
        cand_map   = load_candidate_records(candidates_log_path)

        summary       = compute_expectancy_summary(clean_recs)
        score_buckets = compute_score_bucket_analysis(clean_recs)
        penalty_stats = compute_penalty_attribution(clean_recs, cand_map)
        zone_stats    = compute_rsr_zone_analysis(clean_recs)

        return render_expectancy_report(summary, score_buckets, penalty_stats, zone_stats)
    except Exception as e:
        logger.warning("[FL_EXPECTANCY] render failed (fail-open): %s", e)
        return ""
