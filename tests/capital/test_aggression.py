"""Tests for src/capital/aggression.py"""
import pytest
from src.capital.aggression import (
    AggressionInputs,
    AggressionState,
    compute_raw_aggression,
    update_aggression,
    compute_drawdown_penalty,
    compute_regime_score,
)


def _inputs(**kwargs):
    defaults = dict(
        edge_score=0.70, execution_score=0.85,
        liquidity_score=0.90, regime_score=0.85,
        concentration_penalty=1.0, drawdown_penalty=1.0,
    )
    defaults.update(kwargs)
    return AggressionInputs(**defaults)


def _state(**kwargs):
    defaults = dict(
        mode="RAMPING", ema_score=0.65, raw_score=0.65,
        growth_limit_modifier=1.0, consecutive_aggressive=0,
    )
    defaults.update(kwargs)
    return AggressionState(**defaults)


class TestComputeRawAggression:
    def test_all_perfect(self):
        inp = _inputs(edge_score=1.0, execution_score=1.0, liquidity_score=1.0,
                      regime_score=1.0, concentration_penalty=1.0, drawdown_penalty=1.0)
        assert compute_raw_aggression(inp) == pytest.approx(1.0)

    def test_zero_edge_returns_min(self):
        from src.capital.aggression import MIN_AGGRESSION
        inp = _inputs(edge_score=0.0)
        assert compute_raw_aggression(inp) == pytest.approx(MIN_AGGRESSION)

    def test_half_edge(self):
        inp = _inputs(edge_score=0.5, execution_score=1.0, liquidity_score=1.0,
                      regime_score=1.0, concentration_penalty=1.0, drawdown_penalty=1.0)
        assert compute_raw_aggression(inp) == pytest.approx(0.5)

    def test_multiplicative_nature(self):
        inp = _inputs(edge_score=0.8, execution_score=0.9, liquidity_score=1.0,
                      regime_score=1.0, concentration_penalty=1.0, drawdown_penalty=1.0)
        raw = compute_raw_aggression(inp)
        assert raw == pytest.approx(0.8 * 0.9, rel=1e-3)

    def test_clamped_to_max(self):
        inp = _inputs(edge_score=2.0, execution_score=2.0)  # unrealistic high values
        from src.capital.aggression import MAX_AGGRESSION
        assert compute_raw_aggression(inp) <= MAX_AGGRESSION

    def test_clamped_to_min(self):
        inp = _inputs(edge_score=0.01, execution_score=0.01)
        from src.capital.aggression import MIN_AGGRESSION
        assert compute_raw_aggression(inp) >= MIN_AGGRESSION


class TestAsymmetricEMA:
    def test_expansion_uses_slow_alpha(self):
        state = _state(ema_score=0.60)
        inp = _inputs(edge_score=1.0, execution_score=1.0, liquidity_score=1.0,
                      regime_score=1.0, concentration_penalty=1.0, drawdown_penalty=1.0)
        new, _ = update_aggression(state, inp)
        # raw > ema → expansion path (alpha=0.10)
        expected_ema = 0.60 * 0.90 + new.raw_score * 0.10
        assert new.ema_score == pytest.approx(expected_ema, rel=1e-3)

    def test_contraction_uses_fast_alpha(self):
        state = _state(ema_score=0.90)
        inp = _inputs(edge_score=0.1, execution_score=0.5, liquidity_score=0.5,
                      regime_score=0.5, concentration_penalty=0.5, drawdown_penalty=0.5)
        new, _ = update_aggression(state, inp)
        # raw < ema → contraction path (alpha=0.40)
        expected_ema = 0.90 * 0.60 + new.raw_score * 0.40
        assert new.ema_score == pytest.approx(expected_ema, rel=1e-3)


class TestModeTransitions:
    def test_aggressive_requires_consecutive_periods(self):
        state = _state(ema_score=0.91, consecutive_aggressive=2)
        inp = _inputs(edge_score=1.0, execution_score=1.0, liquidity_score=1.0,
                      regime_score=1.0, concentration_penalty=1.0, drawdown_penalty=1.0)
        new, _ = update_aggression(state, inp)
        # 3rd consecutive period → AGGRESSIVE
        assert new.mode == "AGGRESSIVE"

    def test_defensive_below_threshold(self):
        # EMA already below 0.45 → immediately DEFENSIVE
        state = _state(ema_score=0.40, mode="RAMPING")
        inp = _inputs(edge_score=0.3, execution_score=0.5, liquidity_score=0.5,
                      regime_score=0.5, concentration_penalty=0.5, drawdown_penalty=0.5)
        new, _ = update_aggression(state, inp)
        assert new.mode == "DEFENSIVE"

    def test_ramping_normal_conditions(self):
        state = _state(ema_score=0.65)
        inp = _inputs()
        new, _ = update_aggression(state, inp)
        assert new.mode in ("RAMPING", "OPPORTUNISTIC")

    def test_growth_multiplier_frozen(self):
        state = _state(mode="FROZEN", ema_score=0.30)
        inp = _inputs(edge_score=0.1, execution_score=0.1, liquidity_score=0.1,
                      regime_score=0.1, concentration_penalty=0.5, drawdown_penalty=0.4)
        new, growth_limit = update_aggression(state, inp)
        assert growth_limit == pytest.approx(0.0)


class TestDrawdownPenalty:
    def test_no_dd(self):
        assert compute_drawdown_penalty(0.0) == pytest.approx(1.0)

    def test_small_dd(self):
        p = compute_drawdown_penalty(0.04)
        assert p == pytest.approx(1.0)

    def test_at_15pct_dd(self):
        p = compute_drawdown_penalty(0.15)
        assert p == pytest.approx(0.30, rel=0.05)

    def test_beyond_max_dd_clamped(self):
        p = compute_drawdown_penalty(0.25)
        assert p >= 0.30  # floor

    def test_monotonically_decreasing(self):
        penalties = [compute_drawdown_penalty(dd) for dd in [0.0, 0.05, 0.10, 0.15]]
        for i in range(len(penalties) - 1):
            assert penalties[i] >= penalties[i + 1]


class TestRegimeScore:
    def test_bull_regime(self):
        score = compute_regime_score(is_bear=False, regime_confidence=1.0)
        assert score == pytest.approx(1.0)

    def test_bear_regime(self):
        score = compute_regime_score(is_bear=True, regime_confidence=1.0)
        assert score < 1.0
        assert score > 0.0

    def test_low_confidence_dampens(self):
        bull_high = compute_regime_score(is_bear=False, regime_confidence=1.0)
        bull_low = compute_regime_score(is_bear=False, regime_confidence=0.5)
        assert bull_low < bull_high

    def test_bear_low_confidence(self):
        score = compute_regime_score(is_bear=True, regime_confidence=0.4)
        assert 0.0 < score < 1.0
