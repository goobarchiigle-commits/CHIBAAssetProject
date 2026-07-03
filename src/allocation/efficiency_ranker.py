"""
efficiency_ranker.py — Capital efficiency scoring and candidate ranking

Objective: maximize long-term compound efficiency per unit of
  - deployable capital
  - liquidity consumed
  - correlation consumed
  - drawdown risk consumed

Formula:
    capital_efficiency_score =
        expected_geometric_return
        / (liquidity_penalty * correlation_penalty * drawdown_penalty)

Intentionally simple and explainable. Avoid pseudo-precision.
All calculations are deterministic and replay-safe.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from src.allocation.exposure_core import ExposureState, _pearson, CORR_WINDOW, MIN_OBSERVATIONS_FOR_CORR

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VOLUME_THRESHOLD_YEN: float = 500_000_000.0   # 5億円/日 = "liquid enough"
MIN_VOLUME_YEN: float = 50_000_000.0           # 5000万未満 = illiquid
DRAWDOWN_PENALTY_SCALE: float = 2.0            # how hard to penalize high symbol DD
MAX_CORR_PENALTY: float = 2.0                  # cap on correlation penalty
MAX_LIQUIDITY_PENALTY: float = 3.0             # cap on liquidity penalty
MAX_DRAWDOWN_PENALTY: float = 2.0              # cap on drawdown penalty
MIN_SCORE: float = 0.001                       # floor on efficiency score


@dataclass
class CandidateSignal:
    """Single candidate signal to be ranked."""
    symbol: str
    rsr_rank: float               # RSR rank value [0, 100]
    post_cost_alpha_bps: float    # expected net alpha after costs
    avg_daily_volume_yen: float   # daily turnover estimate
    sector: str
    estimated_symbol_dd: float    # proxy max-drawdown for this symbol [0, 1]


@dataclass
class EfficiencyScore:
    """Deterministic capital efficiency score for one candidate."""
    symbol: str
    capital_efficiency_score: float    # composite score (higher = better)
    expected_geometric_return: float   # post-cost alpha [0, ∞)
    liquidity_penalty: float           # ≥ 1.0
    correlation_penalty: float         # ≥ 1.0
    drawdown_penalty: float            # ≥ 1.0
    correlation_with_portfolio: float  # max pairwise correlation [-1, 1]
    rank: int = 0                      # set by rank_candidates()


def _compute_liquidity_penalty(avg_daily_volume_yen: float) -> float:
    """
    Penalty for low liquidity.
    Very liquid → 1.0; illiquid → up to MAX_LIQUIDITY_PENALTY.
    """
    if avg_daily_volume_yen >= VOLUME_THRESHOLD_YEN:
        return 1.0
    if avg_daily_volume_yen <= 0.0:
        return MAX_LIQUIDITY_PENALTY
    # Linear interpolation between MIN_VOLUME and THRESHOLD
    # MIN_VOLUME → MAX_PENALTY; THRESHOLD → 1.0
    if avg_daily_volume_yen <= MIN_VOLUME_YEN:
        return MAX_LIQUIDITY_PENALTY
    ratio = (avg_daily_volume_yen - MIN_VOLUME_YEN) / (VOLUME_THRESHOLD_YEN - MIN_VOLUME_YEN)
    penalty = MAX_LIQUIDITY_PENALTY - (MAX_LIQUIDITY_PENALTY - 1.0) * ratio
    return round(max(1.0, min(MAX_LIQUIDITY_PENALTY, penalty)), 4)


def _compute_correlation_penalty(
    candidate_symbol: str,
    existing_symbols: list[str],
    exposure_state: ExposureState,
) -> tuple[float, float]:
    """
    Penalty for adding a symbol that is correlated with existing positions.
    Returns (penalty, max_correlation).
    penalty = 1 + max_abs_correlation_with_any_existing
    """
    if not existing_symbols:
        return 1.0, 0.0

    max_abs_corr = 0.0
    cand_buf = exposure_state.return_buffers.get(candidate_symbol, [])

    for sym in existing_symbols:
        existing_buf = exposure_state.return_buffers.get(sym, [])
        common_n = min(len(cand_buf), len(existing_buf), CORR_WINDOW)
        if common_n < MIN_OBSERVATIONS_FOR_CORR:
            continue
        r = _pearson(cand_buf[-common_n:], existing_buf[-common_n:])
        if r is not None:
            max_abs_corr = max(max_abs_corr, abs(r))

    penalty = 1.0 + max_abs_corr
    return round(min(MAX_CORR_PENALTY, penalty), 4), round(max_abs_corr, 4)


def _compute_drawdown_penalty(estimated_symbol_dd: float) -> float:
    """
    Penalty for high single-symbol drawdown risk.
    estimated_symbol_dd: [0, 1] fraction. 0.20 = 20% expected max DD.
    penalty = 1 + dd * SCALE (capped at MAX)
    """
    dd = max(0.0, min(1.0, estimated_symbol_dd))
    penalty = 1.0 + dd * DRAWDOWN_PENALTY_SCALE
    return round(min(MAX_DRAWDOWN_PENALTY, penalty), 4)


def _expected_geometric_return(post_cost_alpha_bps: float) -> float:
    """
    Convert post-cost alpha bps to a geometric return proxy [0, ∞).
    Clamped at 0 for negative alpha.
    """
    return round(max(0.0, post_cost_alpha_bps / 100.0), 6)


def compute_efficiency_score(
    candidate: CandidateSignal,
    existing_positions: list[str],
    exposure_state: ExposureState,
) -> EfficiencyScore:
    """
    Compute capital efficiency score for a single candidate.

    Args:
        candidate:          signal to score
        existing_positions: symbols already in portfolio
        exposure_state:     rolling return history for correlation lookup
    """
    geo_ret = _expected_geometric_return(candidate.post_cost_alpha_bps)
    lp = _compute_liquidity_penalty(candidate.avg_daily_volume_yen)
    cp, max_corr = _compute_correlation_penalty(
        candidate.symbol, existing_positions, exposure_state
    )
    dp = _compute_drawdown_penalty(candidate.estimated_symbol_dd)

    denom = lp * cp * dp
    if geo_ret <= 0.0 or denom <= 0.0:
        score = MIN_SCORE
    else:
        score = geo_ret / denom

    score = round(max(MIN_SCORE, score), 6)

    return EfficiencyScore(
        symbol=candidate.symbol,
        capital_efficiency_score=score,
        expected_geometric_return=geo_ret,
        liquidity_penalty=lp,
        correlation_penalty=cp,
        drawdown_penalty=dp,
        correlation_with_portfolio=max_corr,
    )


def rank_candidates(
    candidates: list[CandidateSignal],
    existing_positions: list[str],
    exposure_state: ExposureState,
    max_to_rank: int = 10,
) -> list[EfficiencyScore]:
    """
    Score and rank all candidates by capital efficiency.
    Returns sorted list (best first) with rank field set.
    Deterministic — identical inputs → identical output.
    """
    scored: list[EfficiencyScore] = []
    for cand in candidates:
        s = compute_efficiency_score(cand, existing_positions, exposure_state)
        scored.append(s)

    # Sort descending by score, then ascending by symbol for tie-breaking
    scored.sort(key=lambda x: (-x.capital_efficiency_score, x.symbol))
    scored = scored[:max_to_rank]

    for i, s in enumerate(scored):
        s.rank = i + 1

    return scored
