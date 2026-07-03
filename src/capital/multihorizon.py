"""
multihorizon.py — Time-scale-separated confidence system

Prevents short-term execution noise from dominating long-term allocation.

Four independent horizons with separate EMA decay rates:
  execution:  1–5d  window, fast decay (α=0.40)  — intraday quality
  signal:     5–21d window, medium decay (α=0.15) — signal quality
  regime:     21–63d window, slow decay (α=0.06)  — market regime
  strategic:  63–252d window, very slow (α=0.02)  — long-term confidence

Aggregate confidence = geometric mean with optional horizon weights.
Short-term deterioration cannot collapse long-term allocation.

Design: A single bad execution day shifts execution_confidence but has
~5% impact on strategic_confidence (since α_exec doesn't feed into strategic).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# EMA decay rates per horizon (higher = faster)
_EMA_EXECUTION: float = 0.40
_EMA_SIGNAL: float = 0.15
_EMA_REGIME: float = 0.06
_EMA_STRATEGIC: float = 0.02

# Weights for aggregate confidence
_WEIGHTS: dict[str, float] = {
    "execution":  0.20,
    "signal":     0.30,
    "regime":     0.30,
    "strategic":  0.20,
}


@dataclass
class MultiHorizonConfidence:
    execution_confidence: float = 0.70   # 1–5d
    signal_confidence: float = 0.65      # 5–21d
    regime_confidence: float = 0.60      # 21–63d
    strategic_confidence: float = 0.60   # 63–252d
    aggregate_confidence: float = 0.63   # weighted geometric mean
    schema_version: int = 1


def _ema(prev: float, new_val: float, alpha: float) -> float:
    return prev * (1.0 - alpha) + new_val * alpha


def _weighted_geometric_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted geometric mean of confidence values."""
    log_sum = 0.0
    weight_sum = 0.0
    for key, val in values.items():
        w = weights.get(key, 0.0)
        if val <= 0.0:
            return 0.0
        log_sum += w * math.log(max(1e-9, val))
        weight_sum += w
    if weight_sum <= 0.0:
        return 0.0
    return math.exp(log_sum / weight_sum)


def update_multihorizon(
    state: MultiHorizonConfidence,
    execution_observation: float,  # [0,1] today's execution quality
    signal_observation: float,     # [0,1] today's signal quality
    regime_observation: float,     # [0,1] today's regime confidence
    strategic_observation: float,  # [0,1] today's strategic assessment
) -> MultiHorizonConfidence:
    """
    Update all horizons with independent EMA decay rates.
    Inputs clamped to [0, 1].
    """
    exec_obs = max(0.0, min(1.0, execution_observation))
    sig_obs  = max(0.0, min(1.0, signal_observation))
    reg_obs  = max(0.0, min(1.0, regime_observation))
    str_obs  = max(0.0, min(1.0, strategic_observation))

    new_exec = _ema(state.execution_confidence,  exec_obs, _EMA_EXECUTION)
    new_sig  = _ema(state.signal_confidence,     sig_obs,  _EMA_SIGNAL)
    new_reg  = _ema(state.regime_confidence,     reg_obs,  _EMA_REGIME)
    new_str  = _ema(state.strategic_confidence,  str_obs,  _EMA_STRATEGIC)

    vals = {
        "execution": new_exec, "signal": new_sig,
        "regime": new_reg,     "strategic": new_str,
    }
    aggregate = _weighted_geometric_mean(vals, _WEIGHTS)

    return MultiHorizonConfidence(
        execution_confidence=round(new_exec, 4),
        signal_confidence=round(new_sig, 4),
        regime_confidence=round(new_reg, 4),
        strategic_confidence=round(new_str, 4),
        aggregate_confidence=round(aggregate, 4),
    )


def load_multihorizon(path: Path) -> MultiHorizonConfidence:
    if not path.exists():
        return MultiHorizonConfidence()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = set(MultiHorizonConfidence.__dataclass_fields__)
        return MultiHorizonConfidence(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("multihorizon load failed (%s) — default", exc)
        return MultiHorizonConfidence()


def save_multihorizon(state: MultiHorizonConfidence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
