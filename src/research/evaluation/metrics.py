"""Regime-aware performance metrics. All functions are pure and deterministic."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma == 0.0 or np.isnan(sigma):
        return 0.0
    return mu / sigma * float(np.sqrt(periods_per_year))


def compute_sortino(
    returns: pd.Series,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < target]
    if len(downside) == 0:
        return float("inf") if float(returns.mean()) > target else 0.0
    downside_std = float(np.sqrt((downside ** 2).mean()))
    if downside_std == 0.0:
        return 0.0
    return float(returns.mean()) / downside_std * float(np.sqrt(periods_per_year))


def compute_max_drawdown(equity: pd.Series) -> float:
    """Returns negative value, e.g. -0.12 for -12% drawdown."""
    if len(equity) == 0:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / (roll_max.abs() + 1e-9)
    return float(dd.min())


def compute_turnover(positions: pd.DataFrame) -> float:
    """Mean absolute daily aggregate position change."""
    if len(positions) < 2:
        return 0.0
    return float(positions.diff().abs().sum(axis=1).mean())


def compute_hit_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0.0).mean())


def compute_tail_risk(returns: pd.Series, percentile: float = 0.05) -> float:
    """Expected Shortfall (CVaR) at given percentile. Returns negative value."""
    if len(returns) == 0:
        return 0.0
    threshold = float(returns.quantile(percentile))
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return threshold
    return float(tail.mean())


def compute_regime_metrics(
    returns: pd.Series,
    labels: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Compute Sharpe/drawdown/hit_rate per regime label.

    Aligns returns and labels on shared index. Each regime with >= 2 observations
    gets its own metrics block.
    """
    r_aligned, l_aligned = returns.align(labels, join="inner")
    result: Dict[str, Dict[str, float]] = {}
    for regime in sorted(l_aligned.unique()):
        mask = l_aligned == regime
        r_reg = r_aligned[mask]
        if len(r_reg) < 2:
            result[regime] = {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": float((r_reg > 0).mean()) if len(r_reg) > 0 else 0.0,
                "trade_count": int(mask.sum()),
            }
            continue
        equity = (1.0 + r_reg).cumprod()
        result[regime] = {
            "sharpe": compute_sharpe(r_reg),
            "max_drawdown": compute_max_drawdown(equity),
            "hit_rate": compute_hit_rate(r_reg),
            "trade_count": int(mask.sum()),
        }
    return result


def compute_oos_robustness(
    is_sharpes: list[float],
    oos_sharpes: list[float],
    min_oos_sharpe: float = 0.0,
) -> float:
    """Fraction of folds where OOS Sharpe >= min_oos_sharpe."""
    if not oos_sharpes:
        return 0.0
    passing = sum(1 for s in oos_sharpes if s >= min_oos_sharpe)
    return passing / len(oos_sharpes)


def compute_sharpe_decay(is_sharpe: float, oos_sharpe: float) -> float:
    """IS Sharpe minus OOS Sharpe. Positive = decay."""
    return is_sharpe - oos_sharpe
