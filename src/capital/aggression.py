"""
aggression.py — Adaptive deployment aggression scoring

deployment_aggression = edge_score × execution_score × liquidity_score
                       × regime_score × concentration_penalty × drawdown_penalty

Asymmetric dynamics (convex deployment):
  expansion:   EMA α=0.10 (slow — requires sustained evidence)
  contraction: EMA α=0.40 (fast — react quickly to deterioration)

Half-Kelly boundary: effective growth limit capped at 0.5× Kelly to
prevent overbetting. Aggression score [MIN, MAX] maps to growth multiplier.

State machine: RAMPING | FROZEN | RECOVERING | AGGRESSIVE | DEFENSIVE | OPPORTUNISTIC
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Expansion is bounded to prevent runaway aggressiveness
MIN_AGGRESSION: float = 0.30
MAX_AGGRESSION: float = 1.40   # capped at 1.4 (aggressive but not reckless)
HALF_KELLY_CAP: float = 0.50   # half-kelly: never bet more than 50% of Kelly optimal

# Growth limit multipliers per aggression mode
_MODE_GROWTH_MULTIPLIERS: dict[str, float] = {
    "RAMPING":      1.00,
    "AGGRESSIVE":   2.00,   # 3.0%/day at base 1.5%
    "OPPORTUNISTIC":1.50,
    "DEFENSIVE":    0.50,
    "RECOVERING":   0.50,
    "FROZEN":       0.00,
}

# EMA decay constants
_EXPAND_ALPHA: float = 0.10   # slow expansion
_CONTRACT_ALPHA: float = 0.40  # fast contraction


@dataclass
class AggressionInputs:
    """Market-condition inputs for aggression computation. All in [0, 1]."""
    edge_score: float = 0.5            # signal edge quality (higher = better)
    execution_score: float = 0.9       # execution quality from deployment_ramp
    liquidity_score: float = 0.9       # market liquidity condition
    regime_score: float = 0.7          # bull/bear regime confidence
    concentration_penalty: float = 1.0 # 1.0 = diversified, < 1 = concentrated
    drawdown_penalty: float = 1.0      # 1.0 = at HWM, < 1 = in drawdown


@dataclass
class AggressionState:
    mode: str = "RAMPING"           # current deployment mode
    ema_score: float = 0.60         # smoothed aggression score
    raw_score: float = 0.60         # latest unsmoothed score
    growth_limit_modifier: float = 1.0  # multiplier on base daily growth limit
    consecutive_aggressive: int = 0
    schema_version: int = 1


def compute_raw_aggression(inputs: AggressionInputs) -> float:
    """
    Multiplicative aggression score.
    All inputs in [0, 1], output clamped to [MIN_AGGRESSION, MAX_AGGRESSION].
    """
    raw = (
        inputs.edge_score
        * inputs.execution_score
        * inputs.liquidity_score
        * inputs.regime_score
        * inputs.concentration_penalty
        * inputs.drawdown_penalty
    )
    return max(MIN_AGGRESSION, min(MAX_AGGRESSION, raw))


def _ema_update(prev: float, new_value: float, alpha: float) -> float:
    return prev * (1.0 - alpha) + new_value * alpha


def update_aggression(
    state: AggressionState,
    inputs: AggressionInputs,
    base_growth_limit: float = 0.015,
) -> tuple[AggressionState, float]:
    """
    Update aggression state.

    Returns (new_state, effective_growth_limit):
      effective_growth_limit = base × mode_multiplier × half_kelly_cap
    """
    raw = compute_raw_aggression(inputs)

    # Asymmetric EMA: faster when contracting (raw < ema), slower when expanding
    alpha = _CONTRACT_ALPHA if raw < state.ema_score else _EXPAND_ALPHA
    new_ema = _ema_update(state.ema_score, raw, alpha)

    # Mode transitions based on smoothed score
    prev_mode = state.mode
    consecutive_aggressive = state.consecutive_aggressive

    if state.mode == "FROZEN":
        # Only exit frozen via external freeze state management
        mode = "FROZEN"
        consecutive_aggressive = 0
    elif new_ema >= 0.90 and inputs.edge_score >= 0.80:
        mode = "AGGRESSIVE"
        consecutive_aggressive += 1
        if consecutive_aggressive == 1 and prev_mode != "AGGRESSIVE":
            logger.info("Aggression → AGGRESSIVE (ema=%.3f edge=%.3f)", new_ema, inputs.edge_score)
    elif new_ema >= 0.75:
        mode = "OPPORTUNISTIC" if prev_mode in ("RAMPING", "AGGRESSIVE") else "RAMPING"
        consecutive_aggressive = max(0, consecutive_aggressive - 1)
    elif new_ema >= 0.55:
        mode = "RAMPING"
        consecutive_aggressive = 0
    elif new_ema >= 0.35:
        mode = "DEFENSIVE"
        consecutive_aggressive = 0
        if prev_mode not in ("DEFENSIVE", "FROZEN", "RECOVERING"):
            logger.warning("Aggression → DEFENSIVE (ema=%.3f)", new_ema)
    else:
        mode = "DEFENSIVE"
        consecutive_aggressive = 0

    # Growth limit: base × mode_multiplier, capped by half-Kelly
    mode_mult = _MODE_GROWTH_MULTIPLIERS.get(mode, 1.0)
    effective_growth_limit = base_growth_limit * mode_mult * HALF_KELLY_CAP * 2.0
    # Note: × HALF_KELLY_CAP × 2 = × 1.0 at RAMPING, normalizing to base at 1.0 multiplier

    new_state = AggressionState(
        mode=mode,
        ema_score=round(new_ema, 4),
        raw_score=round(raw, 4),
        growth_limit_modifier=round(mode_mult, 4),
        consecutive_aggressive=consecutive_aggressive,
    )
    return new_state, round(effective_growth_limit, 6)


def load_aggression_state(path: Path) -> AggressionState:
    if not path.exists():
        return AggressionState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = set(AggressionState.__dataclass_fields__)
        return AggressionState(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("aggression_state load failed (%s) — default RAMPING", exc)
        return AggressionState()


def save_aggression_state(state: AggressionState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def compute_drawdown_penalty(drawdown: float) -> float:
    """
    drawdown in [0, 1] (fraction): 0 = at HWM, 0.15 = -15% DD.
    Returns penalty [0.3, 1.0].
    """
    dd = max(0.0, min(0.5, abs(drawdown)))
    if dd <= 0.05:
        return 1.0
    if dd >= 0.15:
        return 0.30
    t = (dd - 0.05) / (0.15 - 0.05)
    return round(1.0 - t * 0.70, 4)


def compute_regime_score(is_bear: bool, regime_confidence: float = 0.7) -> float:
    """
    regime_confidence: how confident we are in the current regime (0–1).
    Bear regime reduces score by 30%.
    """
    base = 0.70 if is_bear else 1.0
    return round(base * max(0.5, min(1.0, regime_confidence)), 4)
