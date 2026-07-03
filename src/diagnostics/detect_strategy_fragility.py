"""
detect_strategy_fragility — strict, deterministic fragility decomposition.

Inputs
------
daily_decomp : DataFrame[date]
    columns: long_contribution, short_contribution, gross, cost, net
symbol_df : DataFrame[symbol]
    columns: cumulative_pnl, avg_position, turnover_contribution
daily_ret : Series[date]       — gross daily return (pre-cost), NOT recomputed
daily_ret_net : Series[date]   — net daily return (post-cost), NOT recomputed

Note: top3_persistence and rolling_herfindahl require per-symbol per-date series.
With the current schema (symbol_df is static), rolling_herfindahl is broadcast
from the static HHI and top3_persistence is NaN. Pass a per-symbol per-date
DataFrame as `symbol_daily` kwarg to enable full computation (see below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

EPS: float = 1e-12
ROLL: int = 20


def detect_strategy_fragility(
    daily_decomp: pd.DataFrame,
    symbol_df: pd.DataFrame,
    daily_ret: pd.Series,
    daily_ret_net: pd.Series,
    *,
    symbol_daily: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    symbol_daily : optional DataFrame[date x symbol]
        Per-symbol per-date realized contribution. When provided, enables
        rolling_herfindahl (per-window) and top3_persistence computation.
        If None, rolling_herfindahl is the static HHI broadcast to all dates
        and top3_persistence is NaN.

    Returns
    -------
    dict with keys:
        temporal_stability, dependency, cost_sensitivity, asymmetry_risk
    """
    # ── column aliases (preserve exact names, no copies) ─────────────────────
    long_c  = daily_decomp["long_contribution"]
    short_c = daily_decomp["short_contribution"]
    gross_c = daily_decomp["gross"]
    cost_c  = daily_decomp["cost"]
    net_c   = daily_decomp["net"]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Temporal stability
    # ─────────────────────────────────────────────────────────────────────────
    rolling_var_long  = long_c.rolling(ROLL).var()
    rolling_var_short = short_c.rolling(ROLL).var()
    rolling_var_net   = net_c.rolling(ROLL).var()

    var_net      = float(net_c.var())
    mean_net_val = float(net_c.mean())
    stability_score = var_net / max(abs(mean_net_val), EPS)

    temporal_stability: dict[str, Any] = {
        "rolling_var_long":  rolling_var_long,
        "rolling_var_short": rolling_var_short,
        "rolling_var_net":   rolling_var_net,
        "stability_score":   stability_score,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Dependency
    # ─────────────────────────────────────────────────────────────────────────
    cum_pnl     = symbol_df["cumulative_pnl"]
    top3_symbols = cum_pnl.nlargest(3).index.tolist()

    if symbol_daily is not None:
        # symbol_daily: DataFrame indexed by date, columns = symbols
        # rolling top3 by realized contribution per window
        n_dates = len(symbol_daily)

        top3_sets: list[frozenset] = []
        for i in range(n_dates):
            start = max(0, i - ROLL + 1)
            window = symbol_daily.iloc[start : i + 1]
            contrib = window.sum()
            top3_sets.append(frozenset(contrib.nlargest(3).index))

        # persistence: fraction of consecutive (i, i+1) pairs sharing ≥1 symbol
        if len(top3_sets) > 1:
            hits = sum(
                1
                for a, b in zip(top3_sets[:-1], top3_sets[1:])
                if len(a & b) > 0
            )
            top3_persistence = float(hits) / (len(top3_sets) - 1)
        else:
            top3_persistence = float("nan")

        # rolling Herfindahl per date
        hhi_values: list[float] = []
        for i in range(n_dates):
            start = max(0, i - ROLL + 1)
            window = symbol_daily.iloc[start : i + 1]
            contrib_abs = window.sum().abs()
            total       = float(contrib_abs.sum())
            if total < EPS:
                hhi_values.append(float("nan"))
            else:
                shares = contrib_abs / total
                hhi_values.append(float((shares ** 2).sum()))

        rolling_herfindahl = pd.Series(
            hhi_values, index=symbol_daily.index, dtype=float
        )

    else:
        # static fallback — symbol_daily not provided
        top3_persistence = float("nan")

        abs_pnl   = cum_pnl.abs()
        total_abs = float(abs_pnl.sum())
        if total_abs < EPS:
            hhi_static = float("nan")
        else:
            shares     = abs_pnl / total_abs
            hhi_static = float((shares ** 2).sum())

        rolling_herfindahl = pd.Series(
            hhi_static, index=daily_decomp.index, dtype=float
        )

    dependency: dict[str, Any] = {
        "top3_symbols":       top3_symbols,
        "top3_persistence":   top3_persistence,
        "rolling_herfindahl": rolling_herfindahl,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Cost sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    mean_gross = float(daily_ret.mean())
    mean_net   = float(daily_ret_net.mean())
    decay      = mean_gross - mean_net

    sum_gross = float(gross_c.sum())
    sum_cost  = float(cost_c.sum())
    cost_impact_ratio = sum_cost / max(abs(sum_gross), EPS)

    cost_sensitivity: dict[str, Any] = {
        "mean_gross":        mean_gross,
        "mean_net":          mean_net,
        "decay":             decay,
        "cost_impact_ratio": cost_impact_ratio,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Asymmetry risk
    # ─────────────────────────────────────────────────────────────────────────
    vol_long  = float(long_c.std())
    vol_short = float(short_c.std())
    vol_net   = float(net_c.std())

    # Drawdown mask: running-max based, strictly backward-looking
    cum_curve = (1.0 + daily_ret_net).cumprod()
    hwm       = cum_curve.cummax()
    in_dd     = cum_curve < hwm  # boolean Series, aligned to daily_ret_net.index

    # Align decomp to dd mask (inner join to handle any index mismatch)
    long_dd  = long_c.reindex(in_dd.index)
    short_dd = short_c.reindex(in_dd.index)

    dd_long_contribution  = float(long_dd[in_dd].sum())
    dd_short_contribution = float(short_dd[in_dd].sum())

    asymmetry_risk: dict[str, Any] = {
        "vol_long":               vol_long,
        "vol_short":              vol_short,
        "vol_net":                vol_net,
        "drawdown_mask":          in_dd,
        "dd_long_contribution":   dd_long_contribution,
        "dd_short_contribution":  dd_short_contribution,
    }

    return {
        "temporal_stability": temporal_stability,
        "dependency":         dependency,
        "cost_sensitivity":   cost_sensitivity,
        "asymmetry_risk":     asymmetry_risk,
    }
