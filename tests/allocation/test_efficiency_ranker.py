"""Tests for src/allocation/efficiency_ranker.py"""
import pytest
from src.allocation.exposure_core import ExposureState
from src.allocation.efficiency_ranker import (
    CandidateSignal,
    EfficiencyScore,
    _compute_liquidity_penalty,
    _compute_correlation_penalty,
    _compute_drawdown_penalty,
    _expected_geometric_return,
    compute_efficiency_score,
    rank_candidates,
    VOLUME_THRESHOLD_YEN,
    MIN_VOLUME_YEN,
    MAX_LIQUIDITY_PENALTY,
    MAX_CORR_PENALTY,
    MAX_DRAWDOWN_PENALTY,
    MIN_SCORE,
)


def _make_candidate(symbol="8035", alpha_bps=20.0, volume=1e9, dd=0.10):
    return CandidateSignal(
        symbol=symbol,
        rsr_rank=80.0,
        post_cost_alpha_bps=alpha_bps,
        avg_daily_volume_yen=volume,
        sector="電気機器",
        estimated_symbol_dd=dd,
    )


# ── liquidity penalty ──────────────────────────────────────────────────────────

def test_liquidity_penalty_very_liquid():
    assert _compute_liquidity_penalty(VOLUME_THRESHOLD_YEN) == pytest.approx(1.0)
    assert _compute_liquidity_penalty(VOLUME_THRESHOLD_YEN * 2) == pytest.approx(1.0)


def test_liquidity_penalty_zero_volume():
    assert _compute_liquidity_penalty(0.0) == pytest.approx(MAX_LIQUIDITY_PENALTY)


def test_liquidity_penalty_min_volume():
    assert _compute_liquidity_penalty(MIN_VOLUME_YEN) == pytest.approx(MAX_LIQUIDITY_PENALTY)


def test_liquidity_penalty_intermediate():
    mid = (MIN_VOLUME_YEN + VOLUME_THRESHOLD_YEN) / 2
    penalty = _compute_liquidity_penalty(mid)
    assert 1.0 < penalty < MAX_LIQUIDITY_PENALTY


def test_liquidity_penalty_bounded():
    for v in [0, 1e6, 1e8, 1e9, 1e12]:
        p = _compute_liquidity_penalty(v)
        assert 1.0 <= p <= MAX_LIQUIDITY_PENALTY


# ── correlation penalty ────────────────────────────────────────────────────────

def test_correlation_penalty_no_existing():
    state = ExposureState()
    penalty, corr = _compute_correlation_penalty("A", [], state)
    assert penalty == pytest.approx(1.0)
    assert corr == pytest.approx(0.0)


def test_correlation_penalty_no_history():
    state = ExposureState()  # empty buffers
    penalty, corr = _compute_correlation_penalty("A", ["B"], state)
    assert penalty == pytest.approx(1.0)  # insufficient data → no penalty
    assert corr == pytest.approx(0.0)


def test_correlation_penalty_high_correlation():
    returns = [float(i) for i in range(1, 21)]
    state = ExposureState(return_buffers={"A": returns, "B": returns})
    penalty, corr = _compute_correlation_penalty("A", ["B"], state)
    assert corr == pytest.approx(1.0, abs=1e-3)
    assert penalty == pytest.approx(2.0, abs=1e-3)


def test_correlation_penalty_capped():
    # Even with corr=1.0, penalty never exceeds MAX_CORR_PENALTY
    returns = [float(i) for i in range(1, 21)]
    state = ExposureState(return_buffers={
        "A": returns, "B": returns, "C": returns
    })
    penalty, _ = _compute_correlation_penalty("A", ["B", "C"], state)
    assert penalty <= MAX_CORR_PENALTY


# ── drawdown penalty ───────────────────────────────────────────────────────────

def test_drawdown_penalty_zero():
    assert _compute_drawdown_penalty(0.0) == pytest.approx(1.0)


def test_drawdown_penalty_moderate():
    p = _compute_drawdown_penalty(0.20)
    assert 1.0 < p < MAX_DRAWDOWN_PENALTY


def test_drawdown_penalty_max():
    p = _compute_drawdown_penalty(1.0)
    assert p == pytest.approx(MAX_DRAWDOWN_PENALTY)


def test_drawdown_penalty_bounded():
    for dd in [0.0, 0.1, 0.5, 1.0, 2.0]:
        p = _compute_drawdown_penalty(dd)
        assert 1.0 <= p <= MAX_DRAWDOWN_PENALTY


# ── geometric return ───────────────────────────────────────────────────────────

def test_geometric_return_positive():
    assert _expected_geometric_return(20.0) == pytest.approx(0.2, abs=1e-6)


def test_geometric_return_negative_clamped():
    assert _expected_geometric_return(-10.0) == pytest.approx(0.0)


def test_geometric_return_zero():
    assert _expected_geometric_return(0.0) == pytest.approx(0.0)


# ── compute_efficiency_score ───────────────────────────────────────────────────

def test_efficiency_score_positive_alpha():
    state = ExposureState()
    cand = _make_candidate(alpha_bps=20.0, volume=1e9, dd=0.10)
    score = compute_efficiency_score(cand, [], state)
    assert score.capital_efficiency_score > MIN_SCORE
    assert score.expected_geometric_return == pytest.approx(0.2, abs=1e-6)
    assert score.liquidity_penalty == pytest.approx(1.0)


def test_efficiency_score_negative_alpha():
    state = ExposureState()
    cand = _make_candidate(alpha_bps=-5.0)
    score = compute_efficiency_score(cand, [], state)
    assert score.capital_efficiency_score == pytest.approx(MIN_SCORE)


def test_efficiency_score_lower_for_illiquid():
    state = ExposureState()
    liquid = _make_candidate(volume=1e9)
    illiquid = _make_candidate(volume=1e7)
    s_liquid = compute_efficiency_score(liquid, [], state)
    s_illiquid = compute_efficiency_score(illiquid, [], state)
    assert s_liquid.capital_efficiency_score > s_illiquid.capital_efficiency_score


def test_efficiency_score_lower_for_high_dd():
    state = ExposureState()
    low_dd = _make_candidate(dd=0.05)
    high_dd = _make_candidate(dd=0.50)
    s_low = compute_efficiency_score(low_dd, [], state)
    s_high = compute_efficiency_score(high_dd, [], state)
    assert s_low.capital_efficiency_score > s_high.capital_efficiency_score


def test_efficiency_score_deterministic():
    state = ExposureState()
    cand = _make_candidate()
    s1 = compute_efficiency_score(cand, [], state)
    s2 = compute_efficiency_score(cand, [], state)
    assert s1.capital_efficiency_score == s2.capital_efficiency_score


# ── rank_candidates ────────────────────────────────────────────────────────────

def test_rank_candidates_sorted():
    state = ExposureState()
    cands = [
        _make_candidate("A", alpha_bps=10.0, volume=1e8),
        _make_candidate("B", alpha_bps=30.0, volume=1e9),
        _make_candidate("C", alpha_bps=20.0, volume=1e9),
    ]
    ranked = rank_candidates(cands, [], state)
    assert ranked[0].symbol == "B"  # highest alpha, liquid
    assert ranked[0].rank == 1
    assert ranked[-1].rank == len(ranked)


def test_rank_candidates_rank_field():
    state = ExposureState()
    cands = [_make_candidate(str(i), alpha_bps=float(i * 5)) for i in range(5)]
    ranked = rank_candidates(cands, [], state)
    for i, s in enumerate(ranked):
        assert s.rank == i + 1


def test_rank_candidates_empty():
    state = ExposureState()
    ranked = rank_candidates([], [], state)
    assert ranked == []


def test_rank_candidates_max_to_rank():
    state = ExposureState()
    cands = [_make_candidate(str(i)) for i in range(10)]
    ranked = rank_candidates(cands, [], state, max_to_rank=3)
    assert len(ranked) <= 3


def test_rank_candidates_deterministic():
    state = ExposureState()
    cands = [_make_candidate(str(i), alpha_bps=float(i * 5)) for i in range(5)]
    r1 = rank_candidates(cands, [], state)
    r2 = rank_candidates(cands, [], state)
    assert [s.symbol for s in r1] == [s.symbol for s in r2]
