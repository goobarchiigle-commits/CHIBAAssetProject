"""Tests for src/capital/concentration.py"""
import pytest
from src.capital.concentration import (
    compute_effective_n,
    compute_concentration_penalty,
    compute_concentration_metrics,
    MIN_PENALTY,
    DEFAULT_TARGET_N,
)


class TestComputeEffectiveN:
    def test_equal_weights_4(self):
        weights = [0.25, 0.25, 0.25, 0.25]
        assert compute_effective_n(weights) == pytest.approx(4.0, rel=1e-3)

    def test_fully_concentrated(self):
        weights = [1.0, 0.0, 0.0, 0.0]
        assert compute_effective_n(weights) == pytest.approx(1.0, rel=1e-3)

    def test_half_half(self):
        weights = [0.50, 0.50]
        assert compute_effective_n(weights) == pytest.approx(2.0, rel=1e-3)

    def test_unequal_3_positions(self):
        weights = [0.50, 0.25, 0.25]
        result = compute_effective_n(weights)
        assert 2.0 < result < 3.0

    def test_empty_returns_zero(self):
        assert compute_effective_n([]) == pytest.approx(0.0)

    def test_unnormalized_weights(self):
        # 1000, 1000, 1000, 1000 → same as 0.25 each
        weights = [1000.0, 1000.0, 1000.0, 1000.0]
        assert compute_effective_n(weights) == pytest.approx(4.0, rel=1e-3)

    def test_single_position(self):
        assert compute_effective_n([1.0]) == pytest.approx(1.0, rel=1e-3)


class TestConcentrationPenalty:
    def test_fully_diversified(self):
        weights = [1/3, 1/3, 1/3]
        penalty = compute_concentration_penalty(weights, target_n=3)
        assert penalty == pytest.approx(1.0, rel=1e-3)

    def test_fully_concentrated_returns_min_penalty(self):
        weights = [1.0, 0.0, 0.0]
        penalty = compute_concentration_penalty(weights, target_n=3)
        assert penalty == pytest.approx(MIN_PENALTY)

    def test_empty_returns_min_penalty(self):
        assert compute_concentration_penalty([]) == pytest.approx(MIN_PENALTY)

    def test_penalty_in_range(self):
        for _ in range(10):
            import random
            weights = [random.random() for _ in range(3)]
            p = compute_concentration_penalty(weights, target_n=3)
            assert MIN_PENALTY <= p <= 1.0

    def test_higher_concentration_lower_penalty(self):
        equal = [1/3, 1/3, 1/3]
        concentrated = [0.8, 0.15, 0.05]
        p_equal = compute_concentration_penalty(equal)
        p_concentrated = compute_concentration_penalty(concentrated)
        assert p_equal >= p_concentrated


class TestConcentrationMetrics:
    def test_empty_positions(self):
        metrics = compute_concentration_metrics({}, {})
        assert metrics.effective_n == pytest.approx(0.0)
        assert metrics.concentration_penalty == pytest.approx(1.0)
        assert metrics.sector_concentration == pytest.approx(0.0)

    def test_single_sector(self):
        weights = {"7203.T": 0.5, "9984.T": 0.5}
        sectors = {"7203.T": "輸送用機器", "9984.T": "輸送用機器"}
        m = compute_concentration_metrics(weights, sectors)
        assert m.sector_concentration == pytest.approx(1.0, rel=1e-3)

    def test_two_sectors(self):
        weights = {"7203.T": 0.5, "9984.T": 0.5}
        sectors = {"7203.T": "輸送用機器", "9984.T": "情報通信"}
        m = compute_concentration_metrics(weights, sectors)
        assert m.sector_concentration == pytest.approx(0.5, rel=1e-3)

    def test_missing_sector_defaults(self):
        weights = {"XXXX.T": 1.0}
        m = compute_concentration_metrics(weights, {})
        assert m.effective_n == pytest.approx(1.0)

    def test_three_equal_positions(self):
        weights = {"A": 1/3, "B": 1/3, "C": 1/3}
        sectors = {"A": "s1", "B": "s2", "C": "s3"}
        m = compute_concentration_metrics(weights, sectors, target_n=3)
        assert m.effective_n == pytest.approx(3.0, rel=1e-3)
        assert m.concentration_penalty == pytest.approx(1.0, rel=1e-3)
