"""
reflexivity.py — Liquidity Reflexivity Protection

Detect when our own orders are moving the market (self-impact).

Signal: fill_ratio degrades as participation_rate increases.
If fill_ratio correlates negatively with participation_rate over recent
observations → we are reflexively impacting our own execution quality.

Method: simple linear regression over a rolling window.
  slope < REFLEXIVITY_SLOPE_THRESHOLD → reflexivity detected
  (negative slope = higher participation → worse fills)

Response: reduce max_participation_rate (applied in position_sizer).
  Normal:   max_participation = BASE_PARTICIPATION (5%)
  Mild:     max_participation = BASE × 0.70
  Moderate: max_participation = BASE × 0.50
  Severe:   max_participation = BASE × 0.30

Append-only JSONL for regression history.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REFLEXIVITY_WINDOW: int = 10            # observations for regression
REFLEXIVITY_SLOPE_MILD: float = -0.30   # slope < this → mild reflexivity
REFLEXIVITY_SLOPE_MODERATE: float = -0.60
REFLEXIVITY_SLOPE_SEVERE: float = -1.00

BASE_PARTICIPATION_RATE: float = 0.05   # 5% (matches market_impact.py default)

# Participation multipliers by severity
_PARTICIPATION_MULTIPLIERS: dict[str, float] = {
    "NORMAL":   1.00,
    "MILD":     0.70,
    "MODERATE": 0.50,
    "SEVERE":   0.30,
}


@dataclass
class ReflexivityObservation:
    trading_day: str
    participation_rate: float   # fraction of ADV traded
    fill_ratio: float           # actual fills / requested
    order_value: float          # JPY
    adv_yen: float              # average daily volume


@dataclass
class ReflexivityState:
    observations: list = field(default_factory=list)  # rolling ReflexivityObservation dicts
    slope: float = 0.0                  # regression slope (fill_ratio ~ participation_rate)
    r_squared: float = 0.0             # regression fit quality
    severity: str = "NORMAL"           # NORMAL / MILD / MODERATE / SEVERE
    adjusted_participation_rate: float = BASE_PARTICIPATION_RATE
    schema_version: int = 1


def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """
    OLS: y = slope × x + intercept.
    Returns (slope, intercept, r_squared).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R²
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return round(slope, 6), round(intercept, 6), round(max(0.0, r_sq), 6)


def _classify_severity(slope: float, r_squared: float) -> str:
    """Classify reflexivity severity. R² < 0.20 → insufficient evidence → NORMAL."""
    if r_squared < 0.20:
        return "NORMAL"
    if slope <= REFLEXIVITY_SLOPE_SEVERE:
        return "SEVERE"
    if slope <= REFLEXIVITY_SLOPE_MODERATE:
        return "MODERATE"
    if slope <= REFLEXIVITY_SLOPE_MILD:
        return "MILD"
    return "NORMAL"


def update_reflexivity(
    state: ReflexivityState,
    observation: ReflexivityObservation,
    max_history: int = REFLEXIVITY_WINDOW * 3,
) -> ReflexivityState:
    """
    Update reflexivity state with a new execution observation.

    Regression is computed over the last REFLEXIVITY_WINDOW observations.
    """
    obs_dict = asdict(observation)
    history = list(state.observations[-(max_history - 1):])
    history.append(obs_dict)

    # Extract regression inputs from recent window
    recent = history[-REFLEXIVITY_WINDOW:]
    participation = [o["participation_rate"] for o in recent]
    fill_ratios   = [o["fill_ratio"] for o in recent]

    slope, _, r_sq = _linear_regression(participation, fill_ratios)
    severity = _classify_severity(slope, r_sq)
    multiplier = _PARTICIPATION_MULTIPLIERS[severity]
    adjusted = round(BASE_PARTICIPATION_RATE * multiplier, 4)

    if severity != state.severity:
        logger.info(
            "Reflexivity severity %s → %s (slope=%.3f r²=%.3f) → participation=%.3f",
            state.severity, severity, slope, r_sq, adjusted,
        )

    return ReflexivityState(
        observations=history,
        slope=slope,
        r_squared=r_sq,
        severity=severity,
        adjusted_participation_rate=adjusted,
    )


def get_participation_rate(state: ReflexivityState) -> float:
    """Returns the adjusted max participation rate to use in position_sizer."""
    return state.adjusted_participation_rate


def append_reflexivity_observation(observation: ReflexivityObservation, path: Path) -> None:
    """Append-only JSONL for audit trail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(observation), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_reflexivity_state(path: Path) -> ReflexivityState:
    if not path.exists():
        return ReflexivityState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = set(ReflexivityState.__dataclass_fields__)
        return ReflexivityState(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("reflexivity_state load failed (%s) — default", exc)
        return ReflexivityState()


def save_reflexivity_state(state: ReflexivityState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
