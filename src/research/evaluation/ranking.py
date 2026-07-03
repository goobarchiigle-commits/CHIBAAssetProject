"""Deterministic experiment ranking.

Same inputs → same ranking, always.
Rejected experiments sort to the bottom.
Tie-breaking uses (experiment_id, fold_id) for stability.
"""
from __future__ import annotations

from typing import List, NamedTuple

from src.research.contracts.result import ResultContract
from src.research.evaluation.rejection import check_rejection


class RankedResult(NamedTuple):
    result: ResultContract
    score: float
    rejected: bool
    rejection_reasons: List[str]


def _score(r: ResultContract) -> float:
    """Composite ranking score. Higher = better.

    Weights:
      0.35 OOS robustness
      0.25 Sharpe (capped at MAX_SHARPE to prevent outlier dominance)
      0.20 drawdown containment
      0.10 Sharpe decay penalty
      0.10 hit rate
    """
    sharpe_capped = min(r.sharpe, 3.0)
    dd_score = 1.0 - min(abs(r.max_drawdown), 0.5) / 0.5
    decay_score = 1.0 / (1.0 + max(r.stability_metrics.sharpe_decay, 0.0))
    return (
        0.35 * r.stability_metrics.oos_robustness
        + 0.25 * sharpe_capped / 3.0          # normalise to [0,1]
        + 0.20 * dd_score
        + 0.10 * decay_score
        + 0.10 * r.hit_rate
    )


def rank_results(results: List[ResultContract]) -> List[RankedResult]:
    """Rank results descending by score. Rejected → score = -inf → bottom.

    Deterministic: identical ResultContract lists → identical ordering.
    """
    ranked: List[RankedResult] = []
    for r in results:
        rejected, reasons = check_rejection(r)
        score = _score(r) if not rejected else float("-inf")
        ranked.append(RankedResult(result=r, score=score, rejected=rejected, rejection_reasons=reasons))

    ranked.sort(key=lambda x: (-x.score, x.result.experiment_id, x.result.fold_id))
    return ranked
