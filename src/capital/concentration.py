"""
concentration.py — Correlation-aware capital protection

Herfindahl-based effective position count (effective_N).
4 correlated positions must not be treated as 4 independent risks.

effective_N = (Σ w_i)² / Σ w_i²  — Herfindahl reciprocal

concentration_penalty = effective_N / target_N (capped at [MIN, 1.0])

Factor/theme concentration detection:
- Multiple semiconductor stocks → same factor exposure
- Cluster concentration: same RSR cluster → similar momentum factor
- Sector concentration: already handled by existing sector cap

penalty affects:
- deployment_aggression (via inputs.concentration_penalty)
- position sizing (via sizing scalar)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MIN_PENALTY: float = 0.40  # even fully concentrated positions get some capital
DEFAULT_TARGET_N: int = 3   # matches max_positions config


@dataclass
class ConcentrationMetrics:
    effective_n: float               # Herfindahl reciprocal
    concentration_penalty: float     # [MIN_PENALTY, 1.0]
    weights: list                    # position weights used for calculation
    sector_concentration: float      # max sector weight
    schema_version: int = 1


def compute_effective_n(weights: list[float]) -> float:
    """
    Herfindahl reciprocal: effective number of independent positions.
    weights: list of position weights (need not sum to 1).

    Examples:
      [0.25, 0.25, 0.25, 0.25] → effective_N = 4.0 (fully diversified)
      [1.0, 0.0, 0.0, 0.0]    → effective_N = 1.0 (fully concentrated)
      [0.50, 0.25, 0.25]       → effective_N ≈ 2.67
    """
    if not weights:
        return 0.0
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    w_norm = [w / total for w in weights]
    sum_sq = sum(w ** 2 for w in w_norm)
    return round(1.0 / sum_sq, 4) if sum_sq > 0 else float(len(weights))


def compute_concentration_penalty(
    weights: list[float],
    target_n: int = DEFAULT_TARGET_N,
    min_penalty: float = MIN_PENALTY,
) -> float:
    """
    Returns concentration_penalty ∈ [min_penalty, 1.0].
    effective_N = target_N → penalty = 1.0 (fully diversified within target)
    effective_N < target_N → penalty < 1.0
    """
    if not weights:
        return min_penalty

    eff_n = compute_effective_n(weights)
    if target_n <= 0:
        return 1.0

    ratio = eff_n / target_n
    penalty = max(min_penalty, min(1.0, ratio))
    return round(penalty, 4)


def compute_concentration_metrics(
    position_weights: dict,   # {symbol: weight}
    sector_map: dict,         # {symbol: sector}
    target_n: int = DEFAULT_TARGET_N,
) -> ConcentrationMetrics:
    """
    Compute full concentration metrics from current positions.

    Args:
        position_weights: {symbol: weight} (weight = value / total_portfolio)
        sector_map:       {symbol: sector}
    """
    if not position_weights:
        return ConcentrationMetrics(
            effective_n=0.0,
            concentration_penalty=1.0,
            weights=[],
            sector_concentration=0.0,
        )

    weights = list(position_weights.values())
    eff_n = compute_effective_n(weights)
    penalty = compute_concentration_penalty(weights, target_n)

    # Sector concentration: max single sector weight
    sector_weights: dict[str, float] = {}
    total_w = sum(weights) or 1.0
    for sym, w in position_weights.items():
        sector = sector_map.get(sym, "不明")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w / total_w
    max_sector_w = max(sector_weights.values()) if sector_weights else 0.0

    return ConcentrationMetrics(
        effective_n=eff_n,
        concentration_penalty=penalty,
        weights=weights,
        sector_concentration=round(max_sector_w, 4),
    )


def load_concentration_metrics(path: Path) -> Optional[ConcentrationMetrics]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConcentrationMetrics(
            effective_n=float(data.get("effective_n", 0.0)),
            concentration_penalty=float(data.get("concentration_penalty", 1.0)),
            weights=data.get("weights", []),
            sector_concentration=float(data.get("sector_concentration", 0.0)),
        )
    except Exception as exc:
        logger.warning("concentration_metrics load failed (%s)", exc)
        return None


def save_concentration_metrics(metrics: ConcentrationMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
