"""
market_impact.py — Nonlinear market impact estimation

Conservative quadratic participation model.
Impact grows faster than linearly — large orders in thin markets are penalized.

For Japanese equities, at 5% daily volume participation:
  impact ≈ 100×0.05 + 2000×0.0025 = 5 + 5 = 10 bps (reasonable for mid-cap)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketImpactConfig:
    linear_coeff_bps: float = 100.0       # bps per 1.0 participation rate
    quadratic_coeff_bps: float = 2000.0   # bps per (participation rate)²
    min_impact_bps: float = 2.0           # floor: minimum cost always present
    max_impact_bps: float = 200.0         # ceiling: extreme protection


def estimate_market_impact_bps(
    order_value: float,
    adv_value: float,
    cfg: MarketImpactConfig | None = None,
) -> float:
    """
    Estimate market impact in bps using quadratic participation model.

    Args:
        order_value: total order value in yen
        adv_value:   average daily volume value in yen (0 → max impact)

    Returns: market impact estimate in bps
    """
    if cfg is None:
        cfg = MarketImpactConfig()

    if adv_value <= 0.0:
        return cfg.max_impact_bps

    participation = min(1.0, order_value / adv_value)
    impact = (
        cfg.linear_coeff_bps * participation
        + cfg.quadratic_coeff_bps * participation ** 2
    )
    impact = max(cfg.min_impact_bps, min(cfg.max_impact_bps, impact))
    return round(impact, 2)


def impact_adjusted_price(
    price: float,
    impact_bps: float,
    side: str,
) -> float:
    """
    Apply impact to price.
    BUY: price increases (we pay more). SELL: price decreases (we receive less).
    """
    adjustment = price * impact_bps / 10_000.0
    if side.upper() == "BUY":
        return price + adjustment
    return max(0.0, price - adjustment)
