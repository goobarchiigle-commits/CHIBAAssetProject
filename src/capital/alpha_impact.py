"""
alpha_impact.py — Alpha-after-impact deployment gate

Deploy capital ONLY when expected post-cost edge remains positive.

expected_net_alpha_bps = gross_alpha_bps
                       - market_impact_bps
                       - spread_cost_bps
                       - slippage_cost_bps
                       - queue_decay_bps

Deployment gate: proceed only if expected_net_alpha_bps > MIN_THRESHOLD.

For Japanese equities, typical costs:
  - spread:        2–5 bps (mid-cap)
  - slippage:      5–15 bps (0.5–1.5% participation)
  - market impact: 5–20 bps (per estimate from market_impact.py)
  - queue decay:   1–3 bps (morning order priority)

Gross alpha proxy from RSR/momentum signal:
  RSR=80 in bull regime → estimate ~20–30 bps/day expected alpha
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


MIN_NET_ALPHA_THRESHOLD_BPS: float = 5.0   # deploy if net alpha > 5 bps
SPREAD_COST_BPS_DEFAULT: float = 3.0       # typical bid-ask spread / 2
QUEUE_DECAY_BPS_DEFAULT: float = 2.0       # priority decay in morning auction


@dataclass
class AlphaImpactEstimate:
    gross_alpha_bps: float
    market_impact_bps: float
    spread_cost_bps: float
    slippage_cost_bps: float
    queue_decay_bps: float
    expected_net_alpha_bps: float
    deploy: bool                   # True if net alpha > threshold
    reason: str


def estimate_net_alpha(
    gross_alpha_bps: float,
    market_impact_bps: float,
    slippage_bps: float = 8.0,
    spread_cost_bps: float = SPREAD_COST_BPS_DEFAULT,
    queue_decay_bps: float = QUEUE_DECAY_BPS_DEFAULT,
    min_threshold_bps: float = MIN_NET_ALPHA_THRESHOLD_BPS,
) -> AlphaImpactEstimate:
    """
    Compute expected net alpha and deployment decision.

    Args:
        gross_alpha_bps:   estimated gross edge per trade (from RSR/signal quality)
        market_impact_bps: from market_impact.estimate_market_impact_bps()
        slippage_bps:      realized or expected slippage
        spread_cost_bps:   bid-ask half-spread
        queue_decay_bps:   position-in-queue priority loss

    Returns: AlphaImpactEstimate with deployment decision.
    """
    total_cost = market_impact_bps + spread_cost_bps + slippage_bps + queue_decay_bps
    net_alpha = gross_alpha_bps - total_cost

    deploy = net_alpha > min_threshold_bps
    reason = (
        f"net_alpha={net_alpha:.1f}bps > {min_threshold_bps}bps"
        if deploy
        else f"net_alpha={net_alpha:.1f}bps ≤ {min_threshold_bps}bps (gross={gross_alpha_bps:.1f}, cost={total_cost:.1f})"
    )

    return AlphaImpactEstimate(
        gross_alpha_bps=round(gross_alpha_bps, 2),
        market_impact_bps=round(market_impact_bps, 2),
        spread_cost_bps=round(spread_cost_bps, 2),
        slippage_cost_bps=round(slippage_bps, 2),
        queue_decay_bps=round(queue_decay_bps, 2),
        expected_net_alpha_bps=round(net_alpha, 2),
        deploy=deploy,
        reason=reason,
    )


def estimate_gross_alpha_from_signal(
    rsr: float,
    rsr_momentum: float,
    regime_is_favorable: bool,
) -> float:
    """
    Proxy gross alpha estimate from signal quality.
    Higher RSR and positive momentum → higher expected alpha.

    Returns estimated gross alpha in bps.
    Not a prediction; an order-of-magnitude proxy for deployment gating.
    """
    if rsr <= 75.0:
        return 0.0

    # RSR excess above threshold (75 = threshold)
    rsr_excess = (rsr - 75.0) / 25.0  # 0.0 at RSR=75, 1.0 at RSR=100
    mom_bonus = max(0.0, min(10.0, rsr_momentum * 5.0))
    regime_mult = 1.0 if regime_is_favorable else 0.6

    # Calibrated to produce ~15-30 bps at RSR=80-90 in bull regime
    gross_alpha = (10.0 + rsr_excess * 20.0 + mom_bonus) * regime_mult
    return round(gross_alpha, 2)


def compute_post_cost_alpha_retention(
    gross_alpha_bps: float,
    net_alpha_bps: float,
) -> float:
    """Fraction of gross alpha retained after all costs. [0, 1]."""
    if gross_alpha_bps <= 0.0:
        return 0.0
    return round(max(0.0, min(1.0, net_alpha_bps / gross_alpha_bps)), 4)
