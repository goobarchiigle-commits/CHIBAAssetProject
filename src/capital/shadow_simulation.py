"""
shadow_simulation.py — Parallel shadow capital simulation

Computes execution quality estimates across capital tiers in parallel.
No impact on live execution path. Append-only telemetry output.
Replay-safe: deterministic given inputs, no external I/O in simulation core.

Tiers: [1x=current, 2x, 5x] configurable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.capital.market_impact import estimate_market_impact_bps, MarketImpactConfig
from src.capital.position_sizer import SizerConfig, size_position

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


@dataclass
class ShadowTierResult:
    tier: str                     # "1x", "2x", "5x"
    capital: float
    fill_ratio_estimate: float    # estimated fill quality [0, 1]
    partial_fill_ratio: float     # estimated partial fill rate
    slippage_bps: float           # weighted average slippage estimate
    participation_rate: float     # max single-stock participation rate
    deployability_score: float    # deployed / theoretical [0, 1]
    market_impact_bps: float      # weighted average market impact
    liquidity_saturation_score: float  # weighted avg stress [0, 1]
    total_order_value: float      # total deployed yen
    adv_coverage_ratio: float     # total_order_value / sum(adv)


@dataclass
class ShadowSimulationRecord:
    ts: str
    trading_day: str
    base_capital: float
    tiers: list            # list[ShadowTierResult as dict]
    schema_version: int = 1


def simulate_tier(
    tier_label: str,
    capital_multiplier: float,
    base_capital: float,
    signals: list[dict],
    impact_cfg: Optional[MarketImpactConfig] = None,
) -> ShadowTierResult:
    """
    Simulate execution quality for one capital tier.

    Args:
        signals: list of dicts with keys {symbol, weight, price, adv_yen}
                 weight = fractional [0, 1] target allocation

    Pure computation — no I/O, no side effects.
    """
    tier_capital = base_capital * capital_multiplier
    sizer_cfg = SizerConfig()

    total_theoretical = 0.0
    total_deployed = 0.0
    total_adv = 0.0
    weighted_impact = 0.0
    weighted_stress = 0.0
    max_participation = 0.0

    for sig in signals:
        weight = float(sig.get("weight", 0.0))
        price  = float(sig.get("price", 0.0))
        adv    = float(sig.get("adv_yen", 0.0))
        sym    = str(sig.get("symbol", ""))

        if price <= 0.0 or weight <= 0.0:
            continue

        sized = size_position(
            symbol=sym,
            signal_weight=weight,
            price=price,
            risk_adjusted_capital=tier_capital,
            adv_yen=adv,
            cfg=sizer_cfg,
            impact_cfg=impact_cfg,
        )

        total_theoretical += sized.theoretical_target
        total_deployed    += sized.estimated_value
        total_adv         += adv

        if sized.estimated_value > 0.0:
            weighted_impact += sized.estimated_impact_bps * sized.estimated_value
            weighted_stress += sized.liquidity_stress_score * sized.estimated_value
            max_participation = max(max_participation, sized.participation_rate)

    avg_impact = weighted_impact / total_deployed if total_deployed > 0.0 else 0.0
    avg_stress = weighted_stress / total_deployed if total_deployed > 0.0 else 0.0

    # Heuristic fill ratio: degrades as participation exceeds 5% (normal max)
    # At 5% participation: fill = 1.0; at 20% participation: fill ~= 0.5
    participation_excess = max(0.0, max_participation - 0.05) / 0.15
    fill_ratio = max(0.50, 1.0 - 0.50 * min(1.0, participation_excess))
    partial_fill = min(0.50, max_participation * 2.0)

    deployability = total_deployed / total_theoretical if total_theoretical > 0.0 else 0.0

    return ShadowTierResult(
        tier=tier_label,
        capital=round(tier_capital, 0),
        fill_ratio_estimate=round(fill_ratio, 3),
        partial_fill_ratio=round(partial_fill, 3),
        slippage_bps=round(avg_impact, 2),
        participation_rate=round(max_participation, 4),
        deployability_score=round(min(1.0, deployability), 3),
        market_impact_bps=round(avg_impact, 2),
        liquidity_saturation_score=round(avg_stress, 3),
        total_order_value=round(total_deployed, 0),
        adv_coverage_ratio=round(total_deployed / total_adv if total_adv > 0.0 else 0.0, 4),
    )


def run_shadow_simulation(
    base_capital: float,
    signals: list[dict],
    tiers: Optional[list[int]] = None,
    trading_day: Optional[str] = None,
    impact_cfg: Optional[MarketImpactConfig] = None,
) -> ShadowSimulationRecord:
    """
    Run shadow simulation across all capital tiers.

    Returns ShadowSimulationRecord — pure data, no I/O.
    tiers: list of multipliers, default [1, 2, 5]
    """
    if tiers is None:
        tiers = [1, 2, 5]
    if trading_day is None:
        trading_day = datetime.now(JST).date().isoformat()

    tier_results = []
    for multiplier in tiers:
        label = f"{multiplier}x"
        result = simulate_tier(label, float(multiplier), base_capital, signals, impact_cfg)
        tier_results.append(asdict(result))

    return ShadowSimulationRecord(
        ts=datetime.now(JST).isoformat(),
        trading_day=trading_day,
        base_capital=base_capital,
        tiers=tier_results,
    )


def append_shadow_record(record: ShadowSimulationRecord, path: Path) -> None:
    """Append-only shadow simulation telemetry write. No live path impact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.debug("shadow simulation record appended: %s", path)
