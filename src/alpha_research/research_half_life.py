"""
research_half_life.py — Adaptive capacity metrics for the research system

Measures the research system's own adaptability, not just alpha profitability.

Tracks:
  - candidate survival duration (days from DISCOVERED to RETIRED/PRODUCTION)
  - promotion half-life (median days to first promotion attempt)
  - retirement acceleration (rate at which retirement pace is increasing)
  - regime adaptation latency
  - exploration-to-production latency
  - promotion durability (days from PRODUCTION to RETIRED)

No future leakage. Deterministic calculations.
Append-only telemetry.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CandidateLifecycleRecord:
    """Duration tracking for one candidate. Built from registry events."""
    candidate_id: str
    discovered_at: str
    shadow_at: str = ""
    candidate_at: str = ""
    production_at: str = ""
    retired_at: str = ""
    final_status: str = "DISCOVERED"
    days_to_shadow: Optional[int] = None
    days_to_production: Optional[int] = None
    days_in_production: Optional[int] = None
    total_lifespan_days: Optional[int] = None


@dataclass
class ResearchHalfLifeMetrics:
    """Snapshot of research system adaptive capacity."""
    timestamp: str
    trading_day: str
    n_candidates_total: int
    n_promoted: int
    n_retired: int
    n_active: int
    avg_survival_days: float           # mean days from discovery to exit
    median_promotion_days: float       # median days from DISCOVERED to PRODUCTION
    promotion_success_rate: float      # promoted / (promoted + retired)
    retirement_acceleration: float     # positive = retirement pace increasing
    exploration_to_production_latency: float   # avg days DISCOVERED → PRODUCTION
    avg_production_durability_days: float      # avg days in PRODUCTION before retire
    schema_version: int = SCHEMA_VERSION


# ── Pure computation ───────────────────────────────────────────────────────────

def _days_between(date_a: str, date_b: str) -> Optional[int]:
    """Parse YYYY-MM-DD or ISO timestamps. Returns day difference or None."""
    if not date_a or not date_b:
        return None
    try:
        # Accept both date-only and full ISO
        a = date_a[:10]
        b = date_b[:10]
        from datetime import date as _date
        da = _date.fromisoformat(a)
        db = _date.fromisoformat(b)
        return abs((db - da).days)
    except Exception:
        return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_retirement_acceleration(retirement_counts: list[int]) -> float:
    """
    Acceleration = slope of retirement count over rolling windows.
    Positive = retirement pace increasing (bad sign for research quality).
    Uses simple first-difference on rolling window counts.
    """
    if len(retirement_counts) < 2:
        return 0.0
    diffs = [retirement_counts[i] - retirement_counts[i - 1]
             for i in range(1, len(retirement_counts))]
    return round(_mean(diffs), 4)


def build_lifecycle_record(
    candidate_id: str,
    status_changes: list,              # list[StatusChange] for this candidate
    discovery_time: str,
) -> CandidateLifecycleRecord:
    """
    Build lifecycle duration record from status change history.
    No future leakage: uses only completed transitions.
    """
    record = CandidateLifecycleRecord(
        candidate_id=candidate_id,
        discovered_at=discovery_time,
    )
    for change in status_changes:
        if change.candidate_id != candidate_id:
            continue
        if change.to_status == "SHADOW":
            record.shadow_at = change.timestamp
        elif change.to_status == "CANDIDATE":
            record.candidate_at = change.timestamp
        elif change.to_status == "PRODUCTION":
            record.production_at = change.timestamp
        elif change.to_status == "RETIRED":
            record.retired_at = change.timestamp
            record.final_status = "RETIRED"

    if record.production_at:
        record.final_status = "PRODUCTION"

    record.days_to_shadow = _days_between(record.discovered_at, record.shadow_at)
    record.days_to_production = _days_between(record.discovered_at, record.production_at)
    if record.production_at and record.retired_at:
        record.days_in_production = _days_between(record.production_at, record.retired_at)

    if record.retired_at:
        record.total_lifespan_days = _days_between(record.discovered_at, record.retired_at)
    elif record.production_at:
        record.total_lifespan_days = record.days_to_production

    return record


def compute_half_life_metrics(
    timestamp: str,
    trading_day: str,
    lifecycle_records: list[CandidateLifecycleRecord],
    retirement_counts_rolling: list[int],   # per-period retirement counts
) -> ResearchHalfLifeMetrics:
    """
    Compute research system adaptive capacity metrics.
    Deterministic: same inputs → same output.
    """
    total = len(lifecycle_records)
    promoted = sum(1 for r in lifecycle_records if r.production_at)
    retired = sum(1 for r in lifecycle_records if r.final_status == "RETIRED")
    active = total - retired

    survival_days = [
        r.total_lifespan_days for r in lifecycle_records
        if r.total_lifespan_days is not None
    ]
    avg_survival = round(_mean([float(d) for d in survival_days]), 2)

    promotion_latencies = [
        r.days_to_production for r in lifecycle_records
        if r.days_to_production is not None
    ]
    median_promo = round(_median([float(d) for d in promotion_latencies]), 2)

    promo_rate = round(promoted / (promoted + retired), 4) if (promoted + retired) > 0 else 0.0

    retirement_accel = _compute_retirement_acceleration(retirement_counts_rolling)

    prod_durations = [
        r.days_in_production for r in lifecycle_records
        if r.days_in_production is not None
    ]
    avg_durability = round(_mean([float(d) for d in prod_durations]), 2)

    return ResearchHalfLifeMetrics(
        timestamp=timestamp,
        trading_day=trading_day,
        n_candidates_total=total,
        n_promoted=promoted,
        n_retired=retired,
        n_active=active,
        avg_survival_days=avg_survival,
        median_promotion_days=median_promo,
        promotion_success_rate=promo_rate,
        retirement_acceleration=retirement_accel,
        exploration_to_production_latency=median_promo,
        avg_production_durability_days=avg_durability,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def append_half_life_metrics(metrics: ResearchHalfLifeMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(metrics), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_half_life_metrics(path: Path) -> list[ResearchHalfLifeMetrics]:
    if not path.exists():
        return []
    result: list[ResearchHalfLifeMetrics] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            result.append(ResearchHalfLifeMetrics(
                timestamp=data.get("timestamp", ""),
                trading_day=data.get("trading_day", ""),
                n_candidates_total=int(data.get("n_candidates_total", 0)),
                n_promoted=int(data.get("n_promoted", 0)),
                n_retired=int(data.get("n_retired", 0)),
                n_active=int(data.get("n_active", 0)),
                avg_survival_days=float(data.get("avg_survival_days", 0.0)),
                median_promotion_days=float(data.get("median_promotion_days", 0.0)),
                promotion_success_rate=float(data.get("promotion_success_rate", 0.0)),
                retirement_acceleration=float(data.get("retirement_acceleration", 0.0)),
                exploration_to_production_latency=float(data.get("exploration_to_production_latency", 0.0)),
                avg_production_durability_days=float(data.get("avg_production_durability_days", 0.0)),
            ))
        except Exception as exc:
            logger.warning("half_life_metrics parse error (%s) — skipped", exc)
    return result
