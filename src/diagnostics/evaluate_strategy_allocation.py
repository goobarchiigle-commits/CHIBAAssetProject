"""
evaluate_strategy_allocation — deterministic 3-state capital allocation.

Inputs
------
daily_decomp : DataFrame[date]
    columns: long_contribution, short_contribution, gross, cost, net
symbol_df    : DataFrame[symbol]
    columns: cumulative_pnl, avg_position, turnover_contribution
daily_ret    : Series[date]  — gross daily return (available; not consumed here)

Sources
-------
mean_net / var_net  ← daily_decomp["net"]          (no recomputation)
mean_turnover       ← symbol_df["turnover_contribution"]  (cross-sectional mean)

Constants (module-level, fixed — no tuning)
-------------------------------------------
EPS                 = 1e-12
STABILITY_THRESHOLD = 1.0   — stability_score > this triggers fragility flag
                              guarded by abs(mean_net) > EPS to suppress
                              false positives when mean ≈ 0
TOP3_SHARE_THRESHOLD = 0.5  — concentration flag threshold

Decision states
---------------
DISCARD          — HIGH_FRAGILITY and (not HIGH_RETURN or not HIGH_EFFICIENCY)
ALLOCATE_LIMITED — HIGH_FRAGILITY and HIGH_RETURN and HIGH_EFFICIENCY
ALLOCATE         — not HIGH_FRAGILITY
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

EPS: float                  = 1e-12
STABILITY_THRESHOLD: float  = 1.0
TOP3_SHARE_THRESHOLD: float = 0.5


class _Metrics(TypedDict):
    mean_net:        float
    stability_score: float
    top3_share:      float
    efficiency:      float
    mean_turnover:   float


class _Flags(TypedDict):
    HIGH_FRAGILITY:  bool
    HIGH_RETURN:     bool
    HIGH_EFFICIENCY: bool


class AllocationResult(TypedDict):
    decision: str  # "DISCARD" | "ALLOCATE_LIMITED" | "ALLOCATE"
    metrics:  _Metrics
    flags:    _Flags


def evaluate_strategy_allocation(
    daily_decomp: pd.DataFrame,
    symbol_df:    pd.DataFrame,
    daily_ret:    pd.Series,          # available for callers; not consumed
) -> AllocationResult:
    # ── 1. Base metrics ───────────────────────────────────────────────────────
    net_col  = daily_decomp["net"]
    mean_net = float(net_col.mean())
    var_net  = float(net_col.var())

    stability_score = var_net / max(abs(mean_net), EPS)

    cum_pnl   = symbol_df["cumulative_pnl"]
    total_pnl = float(cum_pnl.sum())
    top3_share = float(cum_pnl.nlargest(3).sum()) / max(abs(total_pnl), EPS)

    mean_turnover = float(symbol_df["turnover_contribution"].sum()) / len(symbol_df)
    efficiency    = mean_net / max(mean_turnover, EPS)

    # ── 2. Return classification ──────────────────────────────────────────────
    high_return: bool = mean_net > 0

    # ── 3. Fragility classification ───────────────────────────────────────────
    # stability arm guarded by abs(mean_net) > EPS: prevents var / near-zero
    # mean from producing a spuriously large score when the strategy is flat
    stability_triggered = (
        stability_score > STABILITY_THRESHOLD and abs(mean_net) > EPS
    )
    high_fragility: bool = stability_triggered or (top3_share > TOP3_SHARE_THRESHOLD)

    # ── 4. Efficiency classification ──────────────────────────────────────────
    high_efficiency: bool = efficiency > 0

    # ── 5. Allocation decision (strict 3-state) ───────────────────────────────
    if high_fragility and not high_return:
        decision = "DISCARD"
    elif high_fragility and high_return:
        decision = "ALLOCATE_LIMITED" if high_efficiency else "DISCARD"
    else:
        decision = "ALLOCATE"

    return AllocationResult(
        decision=decision,
        metrics=_Metrics(
            mean_net        = mean_net,
            stability_score = stability_score,
            top3_share      = top3_share,
            efficiency      = efficiency,
            mean_turnover   = mean_turnover,
        ),
        flags=_Flags(
            HIGH_FRAGILITY  = high_fragility,
            HIGH_RETURN     = high_return,
            HIGH_EFFICIENCY = high_efficiency,
        ),
    )
