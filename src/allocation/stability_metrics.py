"""
stability_metrics.py — Allocator stability telemetry

Detects allocator instability before it damages compound growth quality.
The allocator itself must be observable.

Tracks:
- Rank turnover: how many candidates changed ranking position
- Concentration oscillation: volatility of sector concentration readings
- Aggressiveness volatility: volatility of aggression state values
- Factor instability: proxy for scoring inconsistency

All calculations are deterministic and replay-safe.
Append-only JSONL telemetry.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
STABILITY_WINDOW: int = 10          # rolling observations for stability metrics
TURNOVER_ALERT_THRESHOLD: float = 0.67   # > 2/3 of ranks changed = unstable
CONCENTRATION_ALERT_THRESHOLD: float = 0.15  # std > 0.15 = oscillating
AGGRESSION_ALERT_THRESHOLD: float = 0.20     # std > 0.20 = volatile
SCHEMA_VERSION: int = 1


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RankSnapshot:
    """Ordered list of symbols by rank at one point in time."""
    trading_day: str
    ranked_symbols: list    # [best, ..., worst] by efficiency score


@dataclass
class AllocatorStabilityObservation:
    """One stability telemetry observation. Append-only."""
    timestamp: str
    trading_day: str
    rank_turnover: float               # fraction of ranks changed [0, 1]
    concentration_oscillation: float   # rolling std of sector_concentration
    aggressiveness_volatility: float   # rolling std of aggression values
    stability_state: str               # "STABLE" | "UNSTABLE" | "CRITICAL"
    n_observations_used: int           # rolling window actually used
    schema_version: int = SCHEMA_VERSION


@dataclass
class StabilityWindow:
    """Rolling window of observations for stability computation."""
    rank_snapshots: list = field(default_factory=list)       # list[RankSnapshot dicts]
    concentration_history: list = field(default_factory=list)  # list[float]
    aggression_history: list = field(default_factory=list)     # list[float]
    schema_version: int = SCHEMA_VERSION


# ── Pure helpers ───────────────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


def _compute_rank_turnover(
    previous: list[str],
    current: list[str],
) -> float:
    """
    Fraction of symbols that changed rank position.
    Handles different-length rankings by padding with empty positions.
    Returns [0, 1]. 0 = identical ranking. 1 = completely different.
    """
    if not previous and not current:
        return 0.0
    max_len = max(len(previous), len(current))
    if max_len == 0:
        return 0.0

    changed = 0
    for i in range(max_len):
        sym_prev = previous[i] if i < len(previous) else ""
        sym_curr = current[i] if i < len(current) else ""
        if sym_prev != sym_curr:
            changed += 1

    return round(changed / max_len, 4)


def _classify_stability(
    rank_turnover: float,
    concentration_oscillation: float,
    aggressiveness_volatility: float,
) -> str:
    """STABLE / UNSTABLE / CRITICAL based on threshold crossings."""
    critical_count = sum([
        rank_turnover >= TURNOVER_ALERT_THRESHOLD,
        concentration_oscillation >= CONCENTRATION_ALERT_THRESHOLD * 2,
        aggressiveness_volatility >= AGGRESSION_ALERT_THRESHOLD * 2,
    ])
    unstable_count = sum([
        rank_turnover >= TURNOVER_ALERT_THRESHOLD,
        concentration_oscillation >= CONCENTRATION_ALERT_THRESHOLD,
        aggressiveness_volatility >= AGGRESSION_ALERT_THRESHOLD,
    ])

    if critical_count >= 2:
        return "CRITICAL"
    if unstable_count >= 2:
        return "UNSTABLE"
    return "STABLE"


# ── Core computation ───────────────────────────────────────────────────────────

def update_stability_window(
    window: StabilityWindow,
    rank_snapshot: RankSnapshot,
    sector_concentration: float,
    aggression_value: float,
    max_history: int = STABILITY_WINDOW,
) -> StabilityWindow:
    """
    Append new observations and return updated window.
    Immutable update — returns new StabilityWindow.
    """
    new_snapshots = list(window.rank_snapshots[-(max_history - 1):])
    new_snapshots.append(asdict(rank_snapshot))

    new_conc = list(window.concentration_history[-(max_history - 1):])
    new_conc.append(round(max(0.0, min(1.0, float(sector_concentration))), 4))

    new_agg = list(window.aggression_history[-(max_history - 1):])
    new_agg.append(round(max(0.0, min(1.0, float(aggression_value))), 4))

    return StabilityWindow(
        rank_snapshots=new_snapshots,
        concentration_history=new_conc,
        aggression_history=new_agg,
    )


def compute_stability_observation(
    timestamp: str,
    trading_day: str,
    window: StabilityWindow,
) -> AllocatorStabilityObservation:
    """
    Compute current stability metrics from rolling window.
    Deterministic — same window → same result.
    """
    snapshots = window.rank_snapshots
    conc_hist = window.concentration_history
    agg_hist = window.aggression_history

    # Rank turnover: compare most recent two snapshots
    if len(snapshots) >= 2:
        prev_ranked = snapshots[-2].get("ranked_symbols", [])
        curr_ranked = snapshots[-1].get("ranked_symbols", [])
        rank_turnover = _compute_rank_turnover(prev_ranked, curr_ranked)
    else:
        rank_turnover = 0.0

    conc_osc = round(_std(conc_hist), 4) if len(conc_hist) >= 2 else 0.0
    agg_vol = round(_std(agg_hist), 4) if len(agg_hist) >= 2 else 0.0
    state = _classify_stability(rank_turnover, conc_osc, agg_vol)
    n_obs = len(snapshots)

    return AllocatorStabilityObservation(
        timestamp=timestamp,
        trading_day=trading_day,
        rank_turnover=rank_turnover,
        concentration_oscillation=conc_osc,
        aggressiveness_volatility=agg_vol,
        stability_state=state,
        n_observations_used=n_obs,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def append_stability_observation(
    observation: AllocatorStabilityObservation,
    path: Path,
) -> None:
    """Append observation to append-only JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(observation), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_stability_window(path: Path) -> StabilityWindow:
    """Load rolling stability window from JSON."""
    if not path.exists():
        return StabilityWindow()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StabilityWindow(
            rank_snapshots=data.get("rank_snapshots", []),
            concentration_history=data.get("concentration_history", []),
            aggression_history=data.get("aggression_history", []),
        )
    except Exception as exc:
        logger.warning("stability_window load failed (%s) — default", exc)
        return StabilityWindow()


def save_stability_window(window: StabilityWindow, path: Path) -> None:
    """Atomic save of rolling window state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(window), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_recent_stability(path: Path, n: int = 20) -> list[AllocatorStabilityObservation]:
    """Load last N stability observations from JSONL."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        recent = [l for l in lines[-n:] if l.strip()]
        result = []
        for line in recent:
            data = json.loads(line)
            result.append(AllocatorStabilityObservation(
                timestamp=data.get("timestamp", ""),
                trading_day=data.get("trading_day", ""),
                rank_turnover=float(data.get("rank_turnover", 0.0)),
                concentration_oscillation=float(data.get("concentration_oscillation", 0.0)),
                aggressiveness_volatility=float(data.get("aggressiveness_volatility", 0.0)),
                stability_state=data.get("stability_state", "STABLE"),
                n_observations_used=int(data.get("n_observations_used", 0)),
            ))
        return result
    except Exception as exc:
        logger.warning("stability load failed (%s)", exc)
        return []
