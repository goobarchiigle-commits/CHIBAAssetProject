"""
position_sizer.py — Liquidity-constrained position sizer

Pipeline:
  signal_weight × risk_adjusted_capital
    → theoretical_target
    → liquidity_constrained_target  (ADV participation limit)
    → deployable_target             (max_single_position cap)
    → execution_safe_target         (execution scalar applied)
    → lots                          (rounded to lot size)

Replaces direct signal-to-capital multiplication.
All intermediate values preserved for telemetry and audit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.capital.intraday_liquidity import (
    compute_interval_volume_yen,
    compute_liquidity_stress_score,
)
from src.capital.market_impact import estimate_market_impact_bps, MarketImpactConfig

logger = logging.getLogger(__name__)

LOT_SIZE = 100  # Tokyo Stock Exchange standard lot size


@dataclass
class SizerConfig:
    max_participation_rate: float = 0.05       # max 5% of ADV
    max_single_position_ratio: float = 0.25    # max 25% of risk_adjusted_capital
    min_expected_fill_ratio: float = 0.80      # expected fill threshold
    execution_scalar_bounds: tuple = (0.7, 1.0)
    lot_size: int = LOT_SIZE


@dataclass
class SizedPosition:
    symbol: str
    theoretical_target: float       # signal_weight × risk_adjusted_capital
    liquidity_constrained: float    # after ADV participation limit
    deployable_target: float        # after max_single_position cap
    execution_safe_target: float    # after execution scalar
    lots: int                       # final lot count (integer)
    estimated_value: float          # lots × price × lot_size
    participation_rate: float       # estimated_value / adv_yen
    estimated_impact_bps: float     # market impact estimate
    liquidity_stress_score: float   # [0, 1]: 1.0 = saturated
    sizing_notes: list              # audit trail of constraints applied


def size_position(
    symbol: str,
    signal_weight: float,
    price: float,
    risk_adjusted_capital: float,
    adv_yen: float,
    execution_scalar: float = 1.0,
    now: Optional[datetime] = None,
    cfg: Optional[SizerConfig] = None,
    impact_cfg: Optional[MarketImpactConfig] = None,
) -> SizedPosition:
    """
    Run full sizing pipeline for a single symbol.

    Args:
        signal_weight:          fractional weight [0, 1] from strategy signal
        price:                  entry price per share (yen)
        risk_adjusted_capital:  from CapitalState
        adv_yen:                average daily volume in yen (0 = unknown)
        execution_scalar:       from DeploymentRamp / FreezeState [0, 1]
    """
    if cfg is None:
        cfg = SizerConfig()

    notes: list[str] = []

    if price <= 0.0 or risk_adjusted_capital <= 0.0:
        return _zero_position(symbol, "invalid_price_or_capital")

    # Step 1: theoretical
    theoretical = signal_weight * risk_adjusted_capital

    # Step 2: ADV liquidity constraint
    if adv_yen > 0.0:
        max_adv_order = adv_yen * cfg.max_participation_rate
        if theoretical > max_adv_order:
            notes.append(
                f"adv_capped {theoretical:.0f}→{max_adv_order:.0f} "
                f"(participation>{cfg.max_participation_rate:.0%})"
            )
        liq_constrained = min(theoretical, max_adv_order)
    else:
        liq_constrained = theoretical
        notes.append("adv_unknown: skipped participation limit")

    # Step 3: max single-position cap
    max_single = cfg.max_single_position_ratio * risk_adjusted_capital
    if liq_constrained > max_single:
        notes.append(
            f"pos_capped {liq_constrained:.0f}→{max_single:.0f} "
            f"(>{cfg.max_single_position_ratio:.0%})"
        )
    deployable = min(liq_constrained, max_single)

    # Step 4: execution safety scalar
    exec_s = max(cfg.execution_scalar_bounds[0], min(cfg.execution_scalar_bounds[1], execution_scalar))
    exec_safe = deployable * exec_s
    if exec_s < 1.0:
        notes.append(f"exec_scalar={exec_s:.3f}")

    # Step 5: round to lots
    lots = max(0, int(exec_safe // (price * cfg.lot_size)))
    estimated_value = lots * price * cfg.lot_size

    # Participation rate
    participation = estimated_value / adv_yen if adv_yen > 0.0 else 0.0

    # Market impact
    impact_bps = estimate_market_impact_bps(estimated_value, adv_yen, impact_cfg)

    # Intraday liquidity stress
    stress = compute_liquidity_stress_score(estimated_value, adv_yen, now)

    return SizedPosition(
        symbol=symbol,
        theoretical_target=round(theoretical, 2),
        liquidity_constrained=round(liq_constrained, 2),
        deployable_target=round(deployable, 2),
        execution_safe_target=round(exec_safe, 2),
        lots=lots,
        estimated_value=round(estimated_value, 0),
        participation_rate=round(participation, 4),
        estimated_impact_bps=impact_bps,
        liquidity_stress_score=stress,
        sizing_notes=notes,
    )


def _zero_position(symbol: str, reason: str) -> SizedPosition:
    return SizedPosition(
        symbol=symbol,
        theoretical_target=0.0,
        liquidity_constrained=0.0,
        deployable_target=0.0,
        execution_safe_target=0.0,
        lots=0,
        estimated_value=0.0,
        participation_rate=0.0,
        estimated_impact_bps=0.0,
        liquidity_stress_score=1.0,
        sizing_notes=[f"zero:{reason}"],
    )
