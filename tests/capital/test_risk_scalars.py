"""Tests for risk_scalars.py"""
import pytest

from src.capital.risk_scalars import (
    ScalarConfig,
    compute_volatility_scalar,
    compute_liquidity_scalar,
    compute_execution_scalar,
)


class TestVolatilityScalar:
    def test_low_vol_returns_one(self):
        # Below normal threshold → full deployment
        assert compute_volatility_scalar(0.10) == 1.0

    def test_high_vol_returns_min(self):
        cfg = ScalarConfig(vol_scalar_min=0.50, vol_stress_threshold=0.30)
        assert compute_volatility_scalar(0.35, cfg) == 0.50

    def test_interpolation(self):
        # At midpoint between normal (0.15) and stress (0.30)
        cfg = ScalarConfig(vol_normal_threshold=0.15, vol_stress_threshold=0.30, vol_scalar_min=0.50)
        scalar = compute_volatility_scalar(0.225, cfg)
        assert scalar == pytest.approx(0.75, abs=0.01)

    def test_exactly_at_normal_threshold(self):
        cfg = ScalarConfig(vol_normal_threshold=0.15)
        assert compute_volatility_scalar(0.15, cfg) == 1.0

    def test_exactly_at_stress_threshold(self):
        cfg = ScalarConfig(vol_stress_threshold=0.30, vol_scalar_min=0.50)
        assert compute_volatility_scalar(0.30, cfg) == pytest.approx(0.50)

    def test_result_within_bounds(self):
        cfg = ScalarConfig(vol_scalar_min=0.50)
        for vol in [0.0, 0.10, 0.20, 0.30, 0.50]:
            s = compute_volatility_scalar(vol, cfg)
            assert 0.50 <= s <= 1.0


class TestLiquidityScalar:
    def test_low_ratio_returns_one(self):
        assert compute_liquidity_scalar(0.01) == 1.0

    def test_high_ratio_returns_min(self):
        cfg = ScalarConfig(adv_ratio_stress=0.10, liq_scalar_min=0.50)
        assert compute_liquidity_scalar(0.15, cfg) == 0.50

    def test_interpolation(self):
        cfg = ScalarConfig(adv_ratio_normal=0.03, adv_ratio_stress=0.10, liq_scalar_min=0.50)
        scalar = compute_liquidity_scalar(0.065, cfg)
        assert scalar == pytest.approx(0.75, abs=0.01)

    def test_result_within_bounds(self):
        cfg = ScalarConfig(liq_scalar_min=0.50)
        for ratio in [0.0, 0.05, 0.10, 0.20]:
            s = compute_liquidity_scalar(ratio, cfg)
            assert 0.50 <= s <= 1.0


class TestExecutionScalar:
    def test_perfect_quality_returns_max(self):
        cfg = ScalarConfig(exec_scalar_max=1.0)
        assert compute_execution_scalar(1.0, cfg) == 1.0

    def test_below_min_returns_min(self):
        cfg = ScalarConfig(exec_quality_min=0.60, exec_scalar_min=0.70)
        assert compute_execution_scalar(0.50, cfg) == 0.70

    def test_at_min_quality_returns_min_scalar(self):
        cfg = ScalarConfig(exec_quality_min=0.60, exec_scalar_min=0.70)
        assert compute_execution_scalar(0.60, cfg) == 0.70

    def test_interpolation(self):
        cfg = ScalarConfig(exec_quality_min=0.60, exec_scalar_min=0.70, exec_scalar_max=1.0)
        # At quality=0.80 (midpoint of 0.60→1.0): scalar = 0.70 + 0.5×0.30 = 0.85
        scalar = compute_execution_scalar(0.80, cfg)
        assert scalar == pytest.approx(0.85, abs=0.01)

    def test_result_within_bounds(self):
        cfg = ScalarConfig(exec_scalar_min=0.70, exec_scalar_max=1.0)
        for q in [0.0, 0.5, 0.7, 0.9, 1.0]:
            s = compute_execution_scalar(q, cfg)
            assert 0.70 <= s <= 1.0
