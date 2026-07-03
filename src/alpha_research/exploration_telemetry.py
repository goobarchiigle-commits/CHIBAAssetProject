"""
exploration_telemetry.py — Research quality analytics telemetry

Consolidated, schema-versioned telemetry snapshot of the entire
exploration pipeline. Append-only JSONL.

Tracks:
  - exploration hit-rate (candidates → production)
  - promotion success ratio
  - average alpha lifespan
  - post-promotion decay time
  - factor proliferation rate
  - capital efficiency trend
  - exploration crowding
  - shadow runtime saturation
  - exploration capital utilization
  - candidate congestion
  - allocator contamination risk

Deterministic serialization. Replay-safe reconstruction.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


@dataclass
class ExplorationTelemetrySnapshot:
    """One telemetry snapshot. Append-only."""
    timestamp: str
    trading_day: str

    # Research pipeline health
    total_candidates: int
    active_candidates: int
    shadow_candidates: int
    production_candidates: int
    retired_candidates: int

    # Efficiency metrics
    exploration_hit_rate: float          # produced / (produced + retired) [0,1]
    promotion_success_ratio: float       # promoted / attempted
    avg_alpha_lifespan_days: float
    post_promotion_decay_days: float     # avg days in production before retire

    # Capacity metrics
    shadow_saturation: float             # active_shadow / max_shadow [0,1]
    capital_utilization: float           # exploration_capital / production_capital
    candidate_congestion: float          # active / hard_cap [0,1]
    factor_proliferation_rate: float     # new factors per period

    # Risk indicators
    allocator_contamination_risk: float  # [0,1] proxy: shadow_saturation * crowding
    research_turnover: float             # retirements / period

    schema_version: int = SCHEMA_VERSION


def build_telemetry_snapshot(
    timestamp: str,
    trading_day: str,
    total_candidates: int,
    active_candidates: int,
    shadow_candidates: int,
    production_candidates: int,
    retired_candidates: int,
    n_promoted: int,
    n_promotion_attempts: int,
    avg_alpha_lifespan_days: float,
    post_promotion_decay_days: float,
    max_shadow_candidates: int,
    exploration_capital: float,
    production_capital: float,
    max_candidates_hard_cap: int,
    n_new_factors_this_period: int,
    n_retired_this_period: int,
    shadow_crowding_score: float = 0.0,
) -> ExplorationTelemetrySnapshot:
    """Compute all derived metrics from inputs."""
    # hit-rate: how often does exploration reach production?
    denominator = n_promoted + retired_candidates
    hit_rate = round(n_promoted / denominator, 4) if denominator > 0 else 0.0

    # promotion success ratio
    promo_ratio = round(n_promoted / n_promotion_attempts, 4) if n_promotion_attempts > 0 else 0.0

    # saturation metrics
    shadow_sat = round(shadow_candidates / max_shadow_candidates, 4) if max_shadow_candidates > 0 else 0.0
    cap_util = round(exploration_capital / production_capital, 4) if production_capital > 0 else 0.0
    congestion = round(active_candidates / max_candidates_hard_cap, 4) if max_candidates_hard_cap > 0 else 0.0

    # contamination risk = saturation × crowding proxy
    contamination = round(min(1.0, shadow_sat * (1.0 + shadow_crowding_score)), 4)

    return ExplorationTelemetrySnapshot(
        timestamp=timestamp,
        trading_day=trading_day,
        total_candidates=total_candidates,
        active_candidates=active_candidates,
        shadow_candidates=shadow_candidates,
        production_candidates=production_candidates,
        retired_candidates=retired_candidates,
        exploration_hit_rate=hit_rate,
        promotion_success_ratio=promo_ratio,
        avg_alpha_lifespan_days=round(avg_alpha_lifespan_days, 2),
        post_promotion_decay_days=round(post_promotion_decay_days, 2),
        shadow_saturation=shadow_sat,
        capital_utilization=cap_util,
        candidate_congestion=congestion,
        factor_proliferation_rate=float(n_new_factors_this_period),
        allocator_contamination_risk=contamination,
        research_turnover=float(n_retired_this_period),
    )


def append_telemetry_snapshot(snapshot: ExplorationTelemetrySnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_telemetry_snapshots(path: Path) -> list[ExplorationTelemetrySnapshot]:
    if not path.exists():
        return []
    snapshots: list[ExplorationTelemetrySnapshot] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            snapshots.append(ExplorationTelemetrySnapshot(
                timestamp=data.get("timestamp", ""),
                trading_day=data.get("trading_day", ""),
                total_candidates=int(data.get("total_candidates", 0)),
                active_candidates=int(data.get("active_candidates", 0)),
                shadow_candidates=int(data.get("shadow_candidates", 0)),
                production_candidates=int(data.get("production_candidates", 0)),
                retired_candidates=int(data.get("retired_candidates", 0)),
                exploration_hit_rate=float(data.get("exploration_hit_rate", 0.0)),
                promotion_success_ratio=float(data.get("promotion_success_ratio", 0.0)),
                avg_alpha_lifespan_days=float(data.get("avg_alpha_lifespan_days", 0.0)),
                post_promotion_decay_days=float(data.get("post_promotion_decay_days", 0.0)),
                shadow_saturation=float(data.get("shadow_saturation", 0.0)),
                capital_utilization=float(data.get("capital_utilization", 0.0)),
                candidate_congestion=float(data.get("candidate_congestion", 0.0)),
                factor_proliferation_rate=float(data.get("factor_proliferation_rate", 0.0)),
                allocator_contamination_risk=float(data.get("allocator_contamination_risk", 0.0)),
                research_turnover=float(data.get("research_turnover", 0.0)),
            ))
        except Exception as exc:
            logger.warning("telemetry_snapshot parse error (%s) — skipped", exc)
    return snapshots


def load_recent_snapshots(path: Path, n: int = 20) -> list[ExplorationTelemetrySnapshot]:
    """Load last N snapshots without reading entire file."""
    all_snaps = load_telemetry_snapshots(path)
    return all_snaps[-n:] if len(all_snaps) > n else all_snaps
