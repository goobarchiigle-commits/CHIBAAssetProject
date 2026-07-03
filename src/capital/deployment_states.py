"""
deployment_states.py — Extended deployment state machine

States:
  RAMPING      — normal capital deployment
  FROZEN       — no growth (execution quality gate)
  RECOVERING   — half growth (post-freeze recovery)
  AGGRESSIVE   — 2× growth (high edge + quality)
  DEFENSIVE    — 0.5× growth (weak edge, not frozen)
  OPPORTUNISTIC— 1.5× growth (tactical opportunity detected)
  EXPLORATION  — concurrent with other states; exploration budget active

Transition principles:
  - Expansion: requires multiple consecutive confirming periods
  - Contraction: immediate on single deterioration signal
  - FROZEN/RECOVERING: driven by execution quality (from deployment_ramp)
  - AGGRESSIVE/DEFENSIVE/OPPORTUNISTIC: driven by aggression score + edge
  - EXPLORATION: orthogonal flag, can be active alongside any other state
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_STATES = frozenset({
    "RAMPING", "FROZEN", "RECOVERING",
    "AGGRESSIVE", "DEFENSIVE", "OPPORTUNISTIC", "EXPLORATION",
})

# Growth limit multipliers for each state
STATE_GROWTH_MULTIPLIERS: dict[str, float] = {
    "RAMPING":      1.00,
    "AGGRESSIVE":   2.00,
    "OPPORTUNISTIC":1.50,
    "DEFENSIVE":    0.50,
    "RECOVERING":   0.50,
    "FROZEN":       0.00,
    "EXPLORATION":  1.00,  # exploration doesn't change core growth
}

# Minimum consecutive periods required to enter each elevated state
_ENTER_AGGRESSIVE_MIN: int = 3    # require 3 days of high aggression
_ENTER_OPPORTUNISTIC_MIN: int = 2
_EXIT_AGGRESSIVE_THRESHOLD: float = 0.75  # exit if ema drops below this
_ENTER_DEFENSIVE_THRESHOLD: float = 0.45  # enter DEFENSIVE if ema below this


@dataclass
class DeploymentStateRecord:
    state: str = "RAMPING"
    exploration_active: bool = False
    consecutive_high_aggression: int = 0
    consecutive_defensive: int = 0
    periods_since_state_change: int = 0
    schema_version: int = 1


def transition_deployment_state(
    record: DeploymentStateRecord,
    aggression_ema: float,
    edge_score: float,
    execution_mode: str,      # from DeploymentRampState.mode
    drawdown: float = 0.0,    # abs value (0.15 = 15% DD)
) -> DeploymentStateRecord:
    """
    Compute next deployment state.

    Contraction is immediate. Expansion requires multiple confirming periods.
    FROZEN/RECOVERING driven by execution quality.
    """
    prev_state = record.state
    consecutive_high = record.consecutive_high_aggression
    consecutive_defensive = record.consecutive_defensive
    periods = record.periods_since_state_change

    # Execution quality overrides (highest priority)
    if execution_mode == "FROZEN":
        new_state = "FROZEN"
        consecutive_high = 0
        consecutive_defensive = 0
    elif execution_mode == "RECOVERING":
        new_state = "RECOVERING"
        consecutive_high = 0
    else:
        # Aggression/edge-driven transitions
        if aggression_ema >= 0.88 and edge_score >= 0.75 and drawdown <= 0.05:
            consecutive_high += 1
            if consecutive_high >= _ENTER_AGGRESSIVE_MIN:
                new_state = "AGGRESSIVE"
            else:
                new_state = prev_state if prev_state not in ("FROZEN", "RECOVERING") else "RAMPING"
        elif aggression_ema >= 0.72 and edge_score >= 0.60:
            consecutive_high = max(0, consecutive_high - 1)
            if prev_state == "AGGRESSIVE" and aggression_ema < _EXIT_AGGRESSIVE_THRESHOLD:
                new_state = "OPPORTUNISTIC"
                consecutive_high = 0
            elif prev_state in ("AGGRESSIVE", "OPPORTUNISTIC"):
                new_state = "OPPORTUNISTIC" if aggression_ema >= 0.75 else "RAMPING"
            else:
                consecutive_periods = record.consecutive_high_aggression + 1
                new_state = "OPPORTUNISTIC" if consecutive_periods >= _ENTER_OPPORTUNISTIC_MIN else "RAMPING"
                consecutive_high = consecutive_periods
        elif aggression_ema <= _ENTER_DEFENSIVE_THRESHOLD or drawdown >= 0.12:
            consecutive_high = 0
            consecutive_defensive += 1
            new_state = "DEFENSIVE"
        else:
            consecutive_high = max(0, consecutive_high - 1)
            consecutive_defensive = 0
            new_state = "RAMPING"

    # Track periods in state
    if new_state != prev_state:
        periods = 0
        if new_state not in ("FROZEN", "RECOVERING"):
            logger.info(
                "DeploymentState %s → %s (ema=%.3f edge=%.3f exec=%s)",
                prev_state, new_state, aggression_ema, edge_score, execution_mode,
            )
    else:
        periods += 1

    return DeploymentStateRecord(
        state=new_state,
        exploration_active=record.exploration_active,
        consecutive_high_aggression=max(0, consecutive_high),
        consecutive_defensive=max(0, consecutive_defensive),
        periods_since_state_change=periods,
    )


def get_growth_multiplier(record: DeploymentStateRecord) -> float:
    """Returns growth limit multiplier for the current state."""
    return STATE_GROWTH_MULTIPLIERS.get(record.state, 1.0)


def load_deployment_state_record(path: Path) -> DeploymentStateRecord:
    if not path.exists():
        return DeploymentStateRecord()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = set(DeploymentStateRecord.__dataclass_fields__)
        return DeploymentStateRecord(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("deployment_state load failed (%s) — default RAMPING", exc)
        return DeploymentStateRecord()


def save_deployment_state_record(record: DeploymentStateRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
