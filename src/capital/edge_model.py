"""
edge_model.py — Edge persistence and quality modeling

Edge confidence MUST NOT rely primarily on recent PnL.
Uses multi-window consistency (5d/21d/63d) for robustness against overfitting.

At runtime, proxy metrics from signal output are used:
  - rsr_breadth_75: fraction of universe with RSR >= 75 (cross-sectional edge)
  - signal_strength: RSR avg of top signals / 100
  - momentum_quality: fraction of signals with positive RSR momentum
  - regime_alignment: how well signal direction aligns with regime

persistence_score reflects:
  - variance (stability) of the above metrics over time
  - NOT just mean level (mean can be high but unstable = low persistence)

Anti-overfitting:
  - All windows are > 5 days (no 1-2 day reactions)
  - Persistence requires multi-window agreement
  - Alpha decay detection: if short-window >> long-window → potential overfitting
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PERSISTENCE_WINDOW_SHORT: int = 5
PERSISTENCE_WINDOW_MED: int = 21
PERSISTENCE_WINDOW_LONG: int = 63


@dataclass
class EdgeObservation:
    """Single-day edge observation from signal output."""
    trading_day: str
    rsr_breadth_75: float          # fraction of universe with RSR >= 75
    signal_strength: float         # normalized 0-1 RSR quality of top signals
    momentum_quality: float        # fraction of signals with positive RSR momentum
    regime_alignment: float        # 0-1, how well signal aligns with regime
    net_alpha_proxy_bps: float     # estimated net alpha proxy (post-cost)


@dataclass
class EdgeModelState:
    observations: list = field(default_factory=list)  # rolling EdgeObservation dicts
    persistence_score_5d: float = 0.5
    persistence_score_21d: float = 0.5
    persistence_score_63d: float = 0.5
    alpha_decay_rate: float = 0.0   # positive = decaying, negative = improving
    composite_edge_score: float = 0.5
    schema_version: int = 1


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_persistence_score(values: list[float], window: int) -> float:
    """
    Persistence = stability × level of recent observations.
    Uses coefficient of variation (CV): low CV = high persistence.
    """
    if len(values) < 2:
        return 0.5

    recent = values[-window:]
    if len(recent) < 2:
        return 0.5

    mean = _mean(recent)
    std = _std(recent)

    if mean <= 0.0:
        return 0.0

    cv = std / mean  # coefficient of variation
    stability = max(0.0, 1.0 - cv * 2.0)  # CV=0.5 → stability=0
    score = mean * stability
    return round(max(0.0, min(1.0, score)), 4)


def _compute_alpha_decay(values: list[float], short_w: int = 5, long_w: int = 21) -> float:
    """
    Positive value = recent performance exceeds long-term (potential mean reversion).
    Negative value = recent deterioration.
    Used to detect overfitting to recent regime.
    """
    if len(values) < long_w:
        return 0.0
    short_mean = _mean(values[-short_w:])
    long_mean = _mean(values[-long_w:])
    if long_mean <= 0:
        return 0.0
    return round((short_mean - long_mean) / long_mean, 4)


def update_edge_model(
    state: EdgeModelState,
    observation: EdgeObservation,
    max_history: int = PERSISTENCE_WINDOW_LONG * 3,
) -> EdgeModelState:
    """
    Update edge model with new daily observation.
    Returns updated state. Deterministic — no randomness.
    """
    obs_dict = asdict(observation)
    history = list(state.observations[-(max_history - 1):])
    history.append(obs_dict)

    # Extract composite quality series
    quality_series = [
        (o["rsr_breadth_75"] * 0.35 + o["signal_strength"] * 0.30
         + o["momentum_quality"] * 0.20 + o["regime_alignment"] * 0.15)
        for o in history
    ]

    p5  = _compute_persistence_score(quality_series, PERSISTENCE_WINDOW_SHORT)
    p21 = _compute_persistence_score(quality_series, PERSISTENCE_WINDOW_MED)
    p63 = _compute_persistence_score(quality_series, PERSISTENCE_WINDOW_LONG)

    # Alpha decay: high = short-window exceeds long-window (potential overfit)
    decay = _compute_alpha_decay(quality_series)

    # Multi-window agreement: require consistency across windows
    agreement = 1.0 - abs(p5 - p63) * 2.0   # penalize inconsistency
    agreement = max(0.0, min(1.0, agreement))

    # Composite edge score: weighted average with anti-overfitting penalty
    composite = p5 * 0.20 + p21 * 0.40 + p63 * 0.40
    composite *= agreement
    # Penalize if alpha_decay is very high (recent PnL > historical → suspect)
    if decay > 0.30:
        composite *= max(0.7, 1.0 - decay * 0.5)

    return EdgeModelState(
        observations=history,
        persistence_score_5d=p5,
        persistence_score_21d=p21,
        persistence_score_63d=p63,
        alpha_decay_rate=decay,
        composite_edge_score=round(max(0.0, min(1.0, composite)), 4),
    )


def build_edge_observation(
    trading_day: str,
    rsr_breadth_75: float,
    top_signal_rsr_avg: float,
    rsr_momentum_positive_fraction: float,
    regime_is_favorable: bool,
    net_alpha_proxy_bps: float = 0.0,
) -> EdgeObservation:
    """Build EdgeObservation from signal bridge output."""
    signal_strength = min(1.0, top_signal_rsr_avg / 100.0) if top_signal_rsr_avg > 0 else 0.0
    regime_alignment = 1.0 if regime_is_favorable else 0.6
    return EdgeObservation(
        trading_day=trading_day,
        rsr_breadth_75=round(max(0.0, min(1.0, rsr_breadth_75)), 4),
        signal_strength=round(max(0.0, min(1.0, signal_strength)), 4),
        momentum_quality=round(max(0.0, min(1.0, rsr_momentum_positive_fraction)), 4),
        regime_alignment=round(regime_alignment, 4),
        net_alpha_proxy_bps=round(net_alpha_proxy_bps, 2),
    )


def load_edge_model(path: Path) -> EdgeModelState:
    if not path.exists():
        return EdgeModelState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return EdgeModelState(
            observations=data.get("observations", []),
            persistence_score_5d=float(data.get("persistence_score_5d", 0.5)),
            persistence_score_21d=float(data.get("persistence_score_21d", 0.5)),
            persistence_score_63d=float(data.get("persistence_score_63d", 0.5)),
            alpha_decay_rate=float(data.get("alpha_decay_rate", 0.0)),
            composite_edge_score=float(data.get("composite_edge_score", 0.5)),
        )
    except Exception as exc:
        logger.warning("edge_model load failed (%s) — default", exc)
        return EdgeModelState()


def save_edge_model(state: EdgeModelState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
