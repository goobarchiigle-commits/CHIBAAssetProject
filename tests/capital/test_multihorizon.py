"""Tests for src/capital/multihorizon.py"""
import pytest
from src.capital.multihorizon import (
    MultiHorizonConfidence,
    update_multihorizon,
    _weighted_geometric_mean,
    _ema,
    _EMA_EXECUTION,
    _EMA_SIGNAL,
    _EMA_REGIME,
    _EMA_STRATEGIC,
    _WEIGHTS,
)


def _state(**kwargs):
    defaults = dict(
        execution_confidence=0.70, signal_confidence=0.65,
        regime_confidence=0.60, strategic_confidence=0.60,
        aggregate_confidence=0.63,
    )
    defaults.update(kwargs)
    return MultiHorizonConfidence(**defaults)


class TestEMA:
    def test_ema_moves_toward_new_value(self):
        result = _ema(0.50, 1.0, 0.40)
        assert result > 0.50

    def test_ema_slow_alpha(self):
        result = _ema(0.50, 1.0, 0.10)
        assert result == pytest.approx(0.50 * 0.90 + 1.0 * 0.10)

    def test_ema_zero_alpha_unchanged(self):
        result = _ema(0.50, 1.0, 0.0)
        assert result == pytest.approx(0.50)


class TestWeightedGeometricMean:
    def test_uniform_weights_equal_value(self):
        vals = {"a": 0.8, "b": 0.8, "c": 0.8, "d": 0.8}
        weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        result = _weighted_geometric_mean(vals, weights)
        assert result == pytest.approx(0.8, rel=1e-4)

    def test_zero_value_returns_zero(self):
        vals = {"a": 0.0, "b": 0.8}
        weights = {"a": 0.5, "b": 0.5}
        result = _weighted_geometric_mean(vals, weights)
        assert result == pytest.approx(0.0)

    def test_empty_weights_returns_zero(self):
        result = _weighted_geometric_mean({}, {})
        assert result == pytest.approx(0.0)


class TestUpdateMultihorizon:
    def test_execution_moves_fast(self):
        state = _state(execution_confidence=0.50)
        new = update_multihorizon(state, 1.0, 0.65, 0.60, 0.60)
        expected = 0.50 * (1 - _EMA_EXECUTION) + 1.0 * _EMA_EXECUTION
        assert new.execution_confidence == pytest.approx(expected, rel=1e-4)

    def test_strategic_moves_very_slow(self):
        state = _state(strategic_confidence=0.50)
        new = update_multihorizon(state, 0.70, 0.65, 0.60, 1.0)
        expected = 0.50 * (1 - _EMA_STRATEGIC) + 1.0 * _EMA_STRATEGIC
        assert new.strategic_confidence == pytest.approx(expected, rel=1e-4)

    def test_bad_execution_barely_moves_strategic(self):
        state = _state(execution_confidence=0.80, strategic_confidence=0.80)
        # Single bad execution day
        new = update_multihorizon(state, 0.0, 0.65, 0.60, 0.60)
        exec_drop = state.execution_confidence - new.execution_confidence
        strat_drop = state.strategic_confidence - new.strategic_confidence
        # Strategic drops much less than execution
        assert exec_drop > strat_drop * 5

    def test_clamps_observations_to_01(self):
        state = _state()
        new = update_multihorizon(state, 1.5, -0.3, 2.0, -1.0)
        assert 0.0 <= new.execution_confidence <= 1.0
        assert 0.0 <= new.signal_confidence <= 1.0
        assert 0.0 <= new.regime_confidence <= 1.0
        assert 0.0 <= new.strategic_confidence <= 1.0

    def test_aggregate_in_range(self):
        state = _state()
        new = update_multihorizon(state, 0.8, 0.7, 0.6, 0.6)
        assert 0.0 < new.aggregate_confidence < 1.0

    def test_four_horizons_independent(self):
        state = _state(
            execution_confidence=0.50,
            signal_confidence=0.50,
            regime_confidence=0.50,
            strategic_confidence=0.50,
        )
        # All same observation
        new = update_multihorizon(state, 1.0, 1.0, 1.0, 1.0)
        # Execution moves fastest
        assert new.execution_confidence > new.signal_confidence
        assert new.signal_confidence > new.regime_confidence
        assert new.regime_confidence > new.strategic_confidence

    def test_deterministic(self):
        state = _state()
        r1 = update_multihorizon(state, 0.8, 0.7, 0.65, 0.60)
        r2 = update_multihorizon(state, 0.8, 0.7, 0.65, 0.60)
        assert r1.aggregate_confidence == r2.aggregate_confidence
