"""
risk_scalars.py — Market condition scalars for capital scaling

Computes volatility_scalar, liquidity_scalar, execution_scalar.
Each scalar ∈ (0, 1] — can only reduce, never increase, deployable_capital.
All computations are deterministic given inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScalarConfig:
    # Volatility scalar
    vol_normal_threshold: float = 0.15    # annualized vol: normal regime ceiling
    vol_stress_threshold: float = 0.30    # annualized vol: full stress
    vol_scalar_min: float = 0.50          # minimum scalar during high volatility

    # Liquidity scalar (based on portfolio ADV ratio)
    adv_ratio_normal: float = 0.03        # portfolio_value / total_adv: normal
    adv_ratio_stress: float = 0.10        # portfolio_value / total_adv: stressed
    liq_scalar_min: float = 0.50

    # Execution quality scalar
    exec_quality_min: float = 0.60        # quality score at which scalar = exec_scalar_min
    exec_scalar_min: float = 0.70
    exec_scalar_max: float = 1.00


def compute_volatility_scalar(
    realized_vol_annualized: float,
    cfg: Optional[ScalarConfig] = None,
) -> float:
    """
    Returns volatility scalar ∈ [vol_scalar_min, 1.0].
    High volatility → smaller scalar → reduced deployment.
    Linear interpolation between normal and stress thresholds.
    """
    if cfg is None:
        cfg = ScalarConfig()

    if realized_vol_annualized <= cfg.vol_normal_threshold:
        return 1.0

    if realized_vol_annualized >= cfg.vol_stress_threshold:
        return cfg.vol_scalar_min

    t = (realized_vol_annualized - cfg.vol_normal_threshold) / (
        cfg.vol_stress_threshold - cfg.vol_normal_threshold
    )
    scalar = 1.0 - t * (1.0 - cfg.vol_scalar_min)
    return round(max(cfg.vol_scalar_min, min(1.0, scalar)), 4)


def compute_liquidity_scalar(
    portfolio_adv_ratio: float,
    cfg: Optional[ScalarConfig] = None,
) -> float:
    """
    Returns liquidity scalar ∈ [liq_scalar_min, 1.0].
    portfolio_adv_ratio = total_portfolio_value / sum(adv_of_holdings).
    Higher ratio → portfolio is large relative to its holdings' liquidity → lower scalar.
    """
    if cfg is None:
        cfg = ScalarConfig()

    if portfolio_adv_ratio <= cfg.adv_ratio_normal:
        return 1.0

    if portfolio_adv_ratio >= cfg.adv_ratio_stress:
        return cfg.liq_scalar_min

    t = (portfolio_adv_ratio - cfg.adv_ratio_normal) / (
        cfg.adv_ratio_stress - cfg.adv_ratio_normal
    )
    scalar = 1.0 - t * (1.0 - cfg.liq_scalar_min)
    return round(max(cfg.liq_scalar_min, min(1.0, scalar)), 4)


def compute_execution_scalar(
    quality_score: float,
    cfg: Optional[ScalarConfig] = None,
) -> float:
    """
    Returns execution scalar ∈ [exec_scalar_min, exec_scalar_max].
    quality_score from deployment_ramp.compute_quality_score().
    Below exec_quality_min → scalar = exec_scalar_min.
    """
    if cfg is None:
        cfg = ScalarConfig()

    clamped = max(0.0, min(1.0, quality_score))

    if clamped >= 1.0:
        return cfg.exec_scalar_max

    if clamped <= cfg.exec_quality_min:
        return cfg.exec_scalar_min

    t = (clamped - cfg.exec_quality_min) / (1.0 - cfg.exec_quality_min)
    scalar = cfg.exec_scalar_min + t * (cfg.exec_scalar_max - cfg.exec_scalar_min)
    return round(max(cfg.exec_scalar_min, min(cfg.exec_scalar_max, scalar)), 4)
