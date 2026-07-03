"""Regime-aware evaluation — metrics, rejection, deterministic ranking."""
from src.research.evaluation.metrics import (
    compute_sharpe, compute_sortino, compute_max_drawdown,
    compute_turnover, compute_hit_rate, compute_tail_risk,
    compute_regime_metrics,
)
from src.research.evaluation.rejection import check_rejection
from src.research.evaluation.ranking import rank_results

__all__ = [
    "compute_sharpe",
    "compute_sortino",
    "compute_max_drawdown",
    "compute_turnover",
    "compute_hit_rate",
    "compute_tail_risk",
    "compute_regime_metrics",
    "check_rejection",
    "rank_results",
]
