"""
opportunity_cost.py — Opportunity Cost Tracking

Records capital left idle when deployment was blocked despite positive edge.
Key insight: idle capital has a cost = missed_alpha × missed_days.

opportunity_cost_bps = expected_net_alpha_bps × idle_fraction × holding_period_days

Use case:
  - If FROZEN state persists 5 days with 10 bps/day edge → 50 bps lost
  - Feeds into CAGR-oriented telemetry for operator awareness
  - Does NOT change behavior (informational only — no feedback to aggression)

Append-only JSONL for audit trail.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OpportunityCostRecord:
    trading_day: str
    deployment_state: str           # FROZEN, DEFENSIVE, etc.
    idle_capital: float             # JPY sitting unused
    effective_capital: float        # total effective capital
    idle_fraction: float            # idle / effective [0,1]
    expected_net_alpha_bps: float   # post-cost alpha estimate
    opportunity_cost_bps: float     # = net_alpha × idle_fraction
    cumulative_cost_bps: float      # running total
    reason: str                     # why capital was idle


@dataclass
class OpportunityCostState:
    cumulative_cost_bps: float = 0.0
    total_idle_days: int = 0
    last_record_day: str = ""
    schema_version: int = 1


def compute_opportunity_cost_bps(
    expected_net_alpha_bps: float,
    idle_fraction: float,
) -> float:
    """
    Per-day opportunity cost in bps.

    idle_fraction = 1.0 → full capital idle (FROZEN)
    idle_fraction = 0.5 → half capital idle (DEFENSIVE)
    """
    if expected_net_alpha_bps <= 0.0:
        return 0.0
    fraction = max(0.0, min(1.0, idle_fraction))
    return round(expected_net_alpha_bps * fraction, 4)


def build_opportunity_cost_record(
    trading_day: str,
    deployment_state: str,
    deployable_capital: float,
    effective_capital: float,
    actual_deployed: float,
    expected_net_alpha_bps: float,
    cumulative_cost_bps: float,
) -> OpportunityCostRecord:
    """
    Build a daily opportunity cost record.

    Args:
        deployable_capital: what was available to deploy
        actual_deployed:    what was actually deployed (order value)
        expected_net_alpha_bps: from AlphaImpactEstimate.expected_net_alpha_bps
    """
    if effective_capital <= 0.0:
        idle_fraction = 0.0
    else:
        idle = max(0.0, deployable_capital - actual_deployed)
        idle_fraction = min(1.0, idle / effective_capital)

    idle_capital = idle_fraction * effective_capital
    cost_bps = compute_opportunity_cost_bps(expected_net_alpha_bps, idle_fraction)
    new_cumulative = cumulative_cost_bps + cost_bps

    state_reasons = {
        "FROZEN":    "execution quality freeze",
        "DEFENSIVE": "weak edge / high drawdown",
        "RECOVERING":"post-freeze recovery window",
    }
    reason = state_reasons.get(deployment_state, f"state={deployment_state}")
    if actual_deployed >= deployable_capital * 0.95:
        reason = "fully deployed"

    return OpportunityCostRecord(
        trading_day=trading_day,
        deployment_state=deployment_state,
        idle_capital=round(idle_capital, 2),
        effective_capital=round(effective_capital, 2),
        idle_fraction=round(idle_fraction, 4),
        expected_net_alpha_bps=round(expected_net_alpha_bps, 4),
        opportunity_cost_bps=round(cost_bps, 4),
        cumulative_cost_bps=round(new_cumulative, 4),
        reason=reason,
    )


def update_opportunity_cost_state(
    state: OpportunityCostState,
    record: OpportunityCostRecord,
) -> OpportunityCostState:
    """Update running state from a new daily record."""
    idle_day = 1 if record.idle_fraction > 0.05 else 0
    return OpportunityCostState(
        cumulative_cost_bps=record.cumulative_cost_bps,
        total_idle_days=state.total_idle_days + idle_day,
        last_record_day=record.trading_day,
    )


def append_opportunity_cost_record(record: OpportunityCostRecord, path: Path) -> None:
    """Append-only JSONL write. Never overwrites existing records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_opportunity_cost_state(path: Path) -> OpportunityCostState:
    """Reconstruct state by scanning the tail of the JSONL log."""
    if not path.exists():
        return OpportunityCostState()
    try:
        last_cumulative = 0.0
        total_idle = 0
        last_day = ""
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                last_cumulative = float(rec.get("cumulative_cost_bps", 0.0))
                last_day = rec.get("trading_day", "")
                if float(rec.get("idle_fraction", 0.0)) > 0.05:
                    total_idle += 1
        return OpportunityCostState(
            cumulative_cost_bps=last_cumulative,
            total_idle_days=total_idle,
            last_record_day=last_day,
        )
    except Exception as exc:
        logger.warning("opportunity_cost_state load failed (%s) — default", exc)
        return OpportunityCostState()
