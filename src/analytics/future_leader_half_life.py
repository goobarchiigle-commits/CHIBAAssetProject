"""
src/analytics/future_leader_half_life.py — Future Leader Alpha Persistence Half-Life

Purpose:
  Quantify how long alpha persists after Future Leader detection.
  Institutional drift survivability telemetry layer.

INVARIANTS (変更禁止):
  - observation_only=True: no execution path connection
  - CLEAN-only aggregation: data_quality_weight >= 1.0, insufficient_forward_data=False
  - PARTIAL_FAILURE: persisted by prior modules; excluded from aggregation here
  - pure functions: I/O and computation separated
  - lookahead-safe: all inputs are materialized survivability/failure records

禁止:
  - order generation / signal mutation / allocator linkage / broker linkage
  - governance linkage / deployable universe mutation
  - predictive_entry_enabled mutation
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

HALF_LIFE_WINDOW: int = 20    # theoretical survivability horizon (business days)
_CLEAN_QUALITY_MIN: float = 1.0
_MIN_SAMPLE: int = 5
_EXPECTANCY_DAYS: List[int] = [3, 5, 10, 20]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HoldingHalfLifeSummary:
    """
    Alpha persistence half-life summary for Future Leader observations.
    observation_only=True: not connected to execution path.
    """
    sample_size:            int
    median_survival_days:   float
    half_life_day:          Optional[int]    # first day expectancy ≤ 50% of peak; None = beyond window
    expectancy_day_3:       Optional[float]  # None = not in survivability materialization window
    expectancy_day_5:       Optional[float]
    expectancy_day_10:      Optional[float]
    expectancy_day_20:      Optional[float]  # None = not in survivability materialization window
    expectancy_decay_slope: Optional[float]  # negative → stronger persistence collapse
    clean_sample_ratio:     float            # clean / total failure records


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers (separated from computation)
# ─────────────────────────────────────────────────────────────────────────────

def _load_failure_records(failure_log_dir: Path) -> List[dict]:
    records: List[dict] = []
    if not failure_log_dir.exists():
        return records
    for p in sorted(failure_log_dir.glob("future_leader_failure_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    return records


def _load_survivability_records(survivability_log_dir: Path) -> List[dict]:
    records: List[dict] = []
    if not survivability_log_dir.exists():
        return records
    for p in sorted(survivability_log_dir.glob("future_leader_survivability_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_by_key(records: List[dict]) -> List[dict]:
    """
    Deduplicate records by (symbol, observation_date). First occurrence wins.
    Prevents duplicate counting when JSONL accumulates repeated materializations.
    """
    seen: Dict[Tuple[str, str], bool] = {}
    out: List[dict] = []
    for r in records:
        key = (r.get("symbol", ""), r.get("observation_date", ""))
        if key[0] and key[1] and key not in seen:
            seen[key] = True
            out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN filters
# ─────────────────────────────────────────────────────────────────────────────

def filter_clean_failure(records: List[dict]) -> List[dict]:
    """
    CLEAN-only failure records: data_quality_weight >= 1.0.
    Deduplication applied first.
    """
    deduped = _dedup_by_key(records)
    return [r for r in deduped if float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN]


def filter_clean_survivability(records: List[dict]) -> List[dict]:
    """
    CLEAN-only survivability: data_quality_weight >= 1.0 AND insufficient_forward_data=False.
    Deduplication applied first.
    """
    deduped = _dedup_by_key(records)
    return [
        r for r in deduped
        if float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN
        and not r.get("insufficient_forward_data", True)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Survival days per record
# ─────────────────────────────────────────────────────────────────────────────

def compute_survival_days(rec: dict) -> Optional[int]:
    """
    Survival days for one failure record.
    failure_detected=True  → failure_day_offset (1-based business day within window).
    failure_detected=False → HALF_LIFE_WINDOW (20d cap: no failure detected).
    Returns None when failure_detected=True but failure_day_offset is missing.
    """
    if rec.get("failure_detected", False):
        offset = rec.get("failure_day_offset")
        if offset is None:
            return None
        return int(offset)
    return HALF_LIFE_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
# Survival curve  (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def compute_survival_curve(clean_failure_records: List[dict]) -> Dict[str, float]:
    """
    Compute survival curve from CLEAN failure records.

    At day d: fraction of records still "alive"
      = failure_detected=False  OR  failure_day_offset > d.

    Returns {"day_1": ratio, ..., "day_20": ratio}.
    Returns all-zero dict when no records.
    """
    if not clean_failure_records:
        return {f"day_{d}": 0.0 for d in range(1, HALF_LIFE_WINDOW + 1)}

    n = len(clean_failure_records)
    result: Dict[str, float] = {}

    for d in range(1, HALF_LIFE_WINDOW + 1):
        alive = 0
        for rec in clean_failure_records:
            if not rec.get("failure_detected", False):
                alive += 1
            else:
                offset = rec.get("failure_day_offset")
                if offset is not None and int(offset) > d:
                    alive += 1
        result[f"day_{d}"] = round(alive / n, 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Expectancy computation  (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_expectancy_at_days(clean_surv: List[dict]) -> Dict[int, Optional[float]]:
    """
    Compute mean forward return at each available day from CLEAN survivability records.

    Available from existing materialization:
      day_5  → forward_return_5d
      day_10 → forward_return_10d

    day_3 and day_20 are not in the current survivability materialization window → None.
    Minimum _MIN_SAMPLE records required per window to report expectancy.
    """
    result: Dict[int, Optional[float]] = {d: None for d in _EXPECTANCY_DAYS}

    vals_5  = [float(r["forward_return_5d"])  for r in clean_surv if r.get("forward_return_5d")  is not None]
    vals_10 = [float(r["forward_return_10d"]) for r in clean_surv if r.get("forward_return_10d") is not None]

    if len(vals_5)  >= _MIN_SAMPLE:
        result[5]  = round(float(np.mean(vals_5)),  6)
    if len(vals_10) >= _MIN_SAMPLE:
        result[10] = round(float(np.mean(vals_10)), 6)

    return result


def _compute_half_life_day(expectancy_map: Dict[int, Optional[float]]) -> Optional[int]:
    """
    Find first available day where expectancy ≤ 50% of peak expectancy.
    Uses only the days present in expectancy_map with non-None values.
    Returns None when alpha persists through all observed days or no data.
    """
    available = [(d, v) for d, v in sorted(expectancy_map.items()) if v is not None]
    if not available:
        return None

    peak = max(v for _, v in available)
    if peak <= 0:
        return None

    threshold = peak * 0.5
    for day, val in available:
        if val <= threshold:
            return day

    return None


def _compute_decay_slope(expectancy_map: Dict[int, Optional[float]]) -> Optional[float]:
    """
    Linear regression slope on available (day, expectancy) points.
    Requires at least 2 non-None data points.
    Negative slope = stronger persistence collapse.
    """
    points = [(d, v) for d, v in sorted(expectancy_map.items()) if v is not None]
    if len(points) < 2:
        return None

    days = np.array([p[0] for p in points], dtype=float)
    vals = np.array([p[1] for p in points], dtype=float)

    coeffs = np.polyfit(days, vals, 1)   # [slope, intercept]
    return round(float(coeffs[0]), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Main computation  (pure — accepts pre-filtered records)
# ─────────────────────────────────────────────────────────────────────────────

def compute_half_life_summary(
    clean_failure_records:        List[dict],
    clean_survivability_records:  List[dict],
    total_failure_count:          int,
) -> HoldingHalfLifeSummary:
    """
    Compute HoldingHalfLifeSummary from CLEAN records.
    observation_only=True: pure analytics, no execution path connection.

    clean_failure_records: pre-filtered by filter_clean_failure()
    clean_survivability_records: pre-filtered by filter_clean_survivability()
    total_failure_count: len(all_failure_records) before CLEAN filter
    """
    n = len(clean_failure_records)
    clean_sample_ratio = round(n / max(total_failure_count, 1), 4)

    if n == 0:
        return HoldingHalfLifeSummary(
            sample_size=0,
            median_survival_days=0.0,
            half_life_day=None,
            expectancy_day_3=None,
            expectancy_day_5=None,
            expectancy_day_10=None,
            expectancy_day_20=None,
            expectancy_decay_slope=None,
            clean_sample_ratio=clean_sample_ratio,
        )

    # Survival days
    survival_days_list: List[int] = []
    for r in clean_failure_records:
        s = compute_survival_days(r)
        if s is not None:
            survival_days_list.append(s)

    median_survival = float(np.median(survival_days_list)) if survival_days_list else 0.0

    # Expectancy
    exp_map    = _compute_expectancy_at_days(clean_survivability_records)
    half_life  = _compute_half_life_day(exp_map)
    decay_slope = _compute_decay_slope(exp_map)

    return HoldingHalfLifeSummary(
        sample_size=n,
        median_survival_days=round(median_survival, 1),
        half_life_day=half_life,
        expectancy_day_3=exp_map.get(3),
        expectancy_day_5=exp_map.get(5),
        expectancy_day_10=exp_map.get(10),
        expectancy_day_20=exp_map.get(20),
        expectancy_decay_slope=decay_slope,
        clean_sample_ratio=clean_sample_ratio,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report formatters  (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _pct_or_na(v: Optional[float]) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "N/A"


def _day_or_na(v: Optional[int]) -> str:
    return f"{v}d" if v is not None else "N/A"


def format_half_life_section(
    summary: HoldingHalfLifeSummary,
    survival_curve: Dict[str, float],
) -> str:
    """
    Render HOLDING HALF-LIFE + SURVIVAL CURVE report section.
    Pure function: no I/O.
    """
    if summary.sample_size < _MIN_SAMPLE:
        return (
            "=== HOLDING HALF-LIFE ===\n"
            f"(insufficient CLEAN sample: {summary.sample_size} / min {_MIN_SAMPLE})"
        )

    lines = [
        "=== HOLDING HALF-LIFE ===",
        f"sample: {summary.sample_size}",
        f"median survival: {summary.median_survival_days:.0f}d",
        f"half-life day: {_day_or_na(summary.half_life_day)}",
    ]

    if summary.expectancy_decay_slope is not None:
        lines.append(f"decay slope: {summary.expectancy_decay_slope:+.2f}")

    exp_pairs = [
        ("day3",  summary.expectancy_day_3),
        ("day5",  summary.expectancy_day_5),
        ("day10", summary.expectancy_day_10),
        ("day20", summary.expectancy_day_20),
    ]
    exp_lines = [f"  {label}: {_pct_or_na(val)}" for label, val in exp_pairs if val is not None]
    if exp_lines:
        lines += ["", "expectancy:"] + exp_lines

    if survival_curve:
        lines += ["", "=== SURVIVAL CURVE ==="]
        for d in range(1, HALF_LIFE_WINDOW + 1):
            key = f"day_{d}"
            if key in survival_curve:
                lines.append(f"  {key}: {survival_curve[key] * 100:.0f}%")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_half_life_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    reports_dir:           Path,
    today:                 Optional[date] = None,
) -> None:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Loads materialized survivability and failure records, computes alpha persistence
    half-life summary, and appends the HOLDING HALF-LIFE section to today's
    future_leader_report_YYYYMMDD.md.

    observation_only=True: no execution path connection.
    Positioned after future leader failure clustering in run_live_signal.py.
    """
    if today is None:
        today = datetime.now(JST).date()

    today_str   = today.strftime("%Y%m%d")
    report_path = reports_dir / f"future_leader_report_{today_str}.md"

    try:
        all_fail = _load_failure_records(failure_log_dir)
        all_surv = _load_survivability_records(survivability_log_dir)

        clean_fail = filter_clean_failure(all_fail)
        clean_surv = filter_clean_survivability(all_surv)

        if not clean_fail:
            logger.info("[FL_HALF_LIFE] no CLEAN failure records — section skipped")
            return

        summary        = compute_half_life_summary(clean_fail, clean_surv, len(all_fail))
        survival_curve = compute_survival_curve(clean_fail)
        section        = format_half_life_section(summary, survival_curve)

        reports_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info(
            "[FL_HALF_LIFE] section written → %s (sample=%d, half_life_day=%s, decay_slope=%s)",
            report_path, summary.sample_size, summary.half_life_day, summary.expectancy_decay_slope,
        )

    except Exception as e:
        logger.warning("[FL_HALF_LIFE] hook failed (fail-open): %s", e)
