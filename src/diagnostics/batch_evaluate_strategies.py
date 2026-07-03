"""
batch_evaluate_strategies — deterministic ranking and selection across strategies.

Each strategy dict must contain:
    strategy_id       : any hashable
    metrics           : dict  (raw; not consumed here)
    daily_decomp      : DataFrame[date]  — must have "net" column
    symbol_df         : DataFrame[symbol]
    allocation_result : AllocationResult from evaluate_strategy_allocation
                        shape: {decision, metrics{mean_net, stability_score,
                                top3_share, efficiency, mean_turnover},
                                flags{HIGH_FRAGILITY, HIGH_RETURN, HIGH_EFFICIENCY}}

Ranking (strict lexicographic priority)
----------------------------------------
1. decision   — ALLOCATE < ALLOCATE_LIMITED < DISCARD  (ascending rank int)
2. mean_net   — descending
3. efficiency — descending

Sort algorithm: mergesort (stable) → identical keys preserve insertion order.

Summary lists
-------------
high_return_but_discarded : HIGH_RETURN=True  AND decision="DISCARD"
low_fragility_low_return  : HIGH_FRAGILITY=False AND HIGH_RETURN=False
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

_DECISION_RANK: dict[str, int] = {
    "ALLOCATE":         0,
    "ALLOCATE_LIMITED": 1,
    "DISCARD":          2,
}

_ALL_DECISIONS: tuple[str, ...] = ("ALLOCATE", "ALLOCATE_LIMITED", "DISCARD")


class Strategy(TypedDict):
    strategy_id:       Any
    metrics:           dict          # raw; not consumed directly
    daily_decomp:      pd.DataFrame
    symbol_df:         pd.DataFrame
    allocation_result: dict          # AllocationResult


class _Summary(TypedDict):
    count_by_decision:         dict[str, int]
    high_return_but_discarded: list[Any]
    low_fragility_low_return:  list[Any]


class BatchResult(TypedDict):
    ranked_table: pd.DataFrame   # strategy_id, decision, mean_net,
                                 # stability_score, top3_share, efficiency
    summary:      _Summary


def batch_evaluate_strategies(strategies: list[Strategy]) -> BatchResult:
    rows: list[dict] = []

    for s in strategies:
        ar    = s["allocation_result"]
        m     = ar["metrics"]
        flags = ar["flags"]

        # mean_net from daily_decomp["net"] per spec §1 (consistent with
        # evaluate_strategy_allocation which also derives it from the same column)
        mean_net = float(s["daily_decomp"]["net"].mean())

        rows.append({
            "strategy_id":     s["strategy_id"],
            "decision":        ar["decision"],
            "_rank":           _DECISION_RANK[ar["decision"]],
            "mean_net":        mean_net,
            "stability_score": float(m["stability_score"]),
            "top3_share":      float(m["top3_share"]),
            "efficiency":      float(m["efficiency"]),
            "_high_return":    bool(flags["HIGH_RETURN"]),
            "_high_fragility": bool(flags["HIGH_FRAGILITY"]),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        ranked_table = pd.DataFrame(
            columns=["strategy_id", "decision", "mean_net",
                     "stability_score", "top3_share", "efficiency"]
        )
        return BatchResult(
            ranked_table=ranked_table,
            summary=_Summary(
                count_by_decision         = {d: 0 for d in _ALL_DECISIONS},
                high_return_but_discarded = [],
                low_fragility_low_return  = [],
            ),
        )

    # ── Ranking ───────────────────────────────────────────────────────────────
    df.sort_values(
        by        = ["_rank", "mean_net", "efficiency"],
        ascending = [True,    False,       False],
        kind      = "mergesort",  # stable → insertion order breaks remaining ties
        inplace   = True,
    )
    df.reset_index(drop=True, inplace=True)

    # ── Output table ──────────────────────────────────────────────────────────
    _OUTPUT_COLS = [
        "strategy_id", "decision", "mean_net",
        "stability_score", "top3_share", "efficiency",
    ]
    ranked_table = df[_OUTPUT_COLS].copy()

    # ── Summary ───────────────────────────────────────────────────────────────
    raw_counts = df.groupby("decision", sort=False)["strategy_id"].count().to_dict()
    count_by_decision: dict[str, int] = {d: raw_counts.get(d, 0) for d in _ALL_DECISIONS}

    high_return_but_discarded: list[Any] = (
        df.loc[df["_high_return"] & (df["decision"] == "DISCARD"), "strategy_id"]
        .tolist()
    )

    low_fragility_low_return: list[Any] = (
        df.loc[~df["_high_fragility"] & ~df["_high_return"], "strategy_id"]
        .tolist()
    )

    return BatchResult(
        ranked_table=ranked_table,
        summary=_Summary(
            count_by_decision         = count_by_decision,
            high_return_but_discarded = high_return_but_discarded,
            low_fragility_low_return  = low_fragility_low_return,
        ),
    )
