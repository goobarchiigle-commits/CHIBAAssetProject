"""Tests for market_impact.py"""
import pytest

from src.capital.market_impact import (
    MarketImpactConfig,
    estimate_market_impact_bps,
    impact_adjusted_price,
)


class TestEstimateMarketImpact:
    def test_zero_adv_returns_max(self):
        cfg = MarketImpactConfig(max_impact_bps=200.0)
        result = estimate_market_impact_bps(1_000_000, 0.0, cfg)
        assert result == 200.0

    def test_tiny_order_near_floor(self):
        # 0.01% participation → very small impact
        adv = 1_000_000_000  # 1B yen
        order = 100_000       # 100K yen
        result = estimate_market_impact_bps(order, adv)
        assert result <= 10.0

    def test_5pct_participation(self):
        # At 5% participation: 100×0.05 + 2000×0.0025 = 5 + 5 = 10 bps
        adv = 1_000_000_000
        order = 50_000_000   # 50M yen = 5% of ADV
        result = estimate_market_impact_bps(order, adv)
        assert result == pytest.approx(10.0, abs=0.5)

    def test_impact_increases_with_participation(self):
        adv = 100_000_000
        small = estimate_market_impact_bps(1_000_000, adv)
        large = estimate_market_impact_bps(10_000_000, adv)
        assert large > small

    def test_max_cap_respected(self):
        cfg = MarketImpactConfig(max_impact_bps=50.0)
        # Force high participation
        result = estimate_market_impact_bps(999_999_999, 1_000_000, cfg)
        assert result == 50.0

    def test_min_floor_respected(self):
        cfg = MarketImpactConfig(min_impact_bps=5.0)
        # Tiny order → floor
        result = estimate_market_impact_bps(1_000, 1_000_000_000, cfg)
        assert result == 5.0

    def test_nonlinear_quadratic(self):
        # At 5%: 100×0.05 + 2000×0.0025 = 10 bps
        # At 10%: 100×0.10 + 2000×0.01  = 30 bps  → ratio=3.0 > 2.0
        adv = 1_000_000_000
        impact_5pct  = estimate_market_impact_bps(50_000_000, adv)   # 5%
        impact_10pct = estimate_market_impact_bps(100_000_000, adv)  # 10%
        ratio = impact_10pct / impact_5pct
        assert ratio > 2.0  # quadratic term dominates at higher participation


class TestImpactAdjustedPrice:
    def test_buy_increases_price(self):
        adjusted = impact_adjusted_price(1000.0, 10.0, "BUY")
        assert adjusted > 1000.0
        assert adjusted == pytest.approx(1000.0 + 1000.0 * 10 / 10000)

    def test_sell_decreases_price(self):
        adjusted = impact_adjusted_price(1000.0, 10.0, "SELL")
        assert adjusted < 1000.0

    def test_zero_impact_no_change(self):
        assert impact_adjusted_price(1000.0, 0.0, "BUY") == 1000.0

    def test_sell_price_not_negative(self):
        result = impact_adjusted_price(1.0, 50000.0, "SELL")
        assert result >= 0.0
