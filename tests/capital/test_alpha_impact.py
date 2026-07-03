"""Tests for src/capital/alpha_impact.py"""
import pytest
from src.capital.alpha_impact import (
    estimate_net_alpha,
    estimate_gross_alpha_from_signal,
    compute_post_cost_alpha_retention,
    MIN_NET_ALPHA_THRESHOLD_BPS,
)


class TestEstimateNetAlpha:
    def test_deploy_when_net_positive(self):
        est = estimate_net_alpha(gross_alpha_bps=30.0, market_impact_bps=5.0,
                                 slippage_bps=8.0, spread_cost_bps=3.0, queue_decay_bps=2.0)
        # net = 30 - (5+8+3+2) = 12 > 5 threshold
        assert est.deploy is True
        assert est.expected_net_alpha_bps == pytest.approx(12.0)

    def test_no_deploy_when_net_below_threshold(self):
        est = estimate_net_alpha(gross_alpha_bps=15.0, market_impact_bps=5.0,
                                 slippage_bps=8.0, spread_cost_bps=3.0, queue_decay_bps=2.0)
        # net = 15 - 18 = -3 ≤ 5
        assert est.deploy is False

    def test_no_deploy_at_exactly_threshold(self):
        # net == threshold → no deploy (requires strictly greater)
        total_cost = 15.0
        gross = total_cost + MIN_NET_ALPHA_THRESHOLD_BPS  # net = exactly threshold
        est = estimate_net_alpha(gross_alpha_bps=gross, market_impact_bps=5.0,
                                 slippage_bps=5.0, spread_cost_bps=3.0, queue_decay_bps=2.0)
        assert est.deploy is False  # net == threshold, not strictly greater

    def test_total_cost_sum(self):
        est = estimate_net_alpha(gross_alpha_bps=50.0, market_impact_bps=10.0,
                                 slippage_bps=8.0, spread_cost_bps=3.0, queue_decay_bps=2.0)
        assert est.market_impact_bps == pytest.approx(10.0)
        assert est.spread_cost_bps == pytest.approx(3.0)
        assert est.slippage_cost_bps == pytest.approx(8.0)
        assert est.queue_decay_bps == pytest.approx(2.0)
        assert est.expected_net_alpha_bps == pytest.approx(27.0)

    def test_reason_contains_net_alpha(self):
        est = estimate_net_alpha(30.0, 5.0)
        assert "net_alpha" in est.reason

    def test_default_params(self):
        est = estimate_net_alpha(gross_alpha_bps=25.0, market_impact_bps=5.0)
        # defaults: slippage=8, spread=3, queue=2 → total=18 → net=7 > 5
        assert est.deploy is True
        assert est.expected_net_alpha_bps == pytest.approx(7.0)


class TestEstimateGrossAlpha:
    def test_below_threshold_rsr(self):
        gross = estimate_gross_alpha_from_signal(rsr=70.0, rsr_momentum=0.0, regime_is_favorable=True)
        assert gross == pytest.approx(0.0)

    def test_at_threshold(self):
        gross = estimate_gross_alpha_from_signal(rsr=75.0, rsr_momentum=0.0, regime_is_favorable=True)
        assert gross >= 0.0

    def test_high_rsr_bull(self):
        gross = estimate_gross_alpha_from_signal(rsr=85.0, rsr_momentum=0.5, regime_is_favorable=True)
        assert 15.0 <= gross <= 40.0  # calibrated to 15-30 bps range

    def test_bear_regime_dampens(self):
        bull = estimate_gross_alpha_from_signal(80.0, 0.5, True)
        bear = estimate_gross_alpha_from_signal(80.0, 0.5, False)
        assert bear < bull

    def test_momentum_bonus_positive(self):
        base = estimate_gross_alpha_from_signal(80.0, 0.0, True)
        with_mom = estimate_gross_alpha_from_signal(80.0, 1.0, True)
        assert with_mom > base

    def test_monotonic_in_rsr(self):
        values = [estimate_gross_alpha_from_signal(rsr, 0.0, True) for rsr in [75, 80, 85, 90, 100]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestPostCostRetention:
    def test_full_retention(self):
        r = compute_post_cost_alpha_retention(10.0, 10.0)
        assert r == pytest.approx(1.0)

    def test_zero_retention_negative_net(self):
        r = compute_post_cost_alpha_retention(10.0, -5.0)
        assert r == pytest.approx(0.0)

    def test_partial_retention(self):
        r = compute_post_cost_alpha_retention(20.0, 10.0)
        assert r == pytest.approx(0.5)

    def test_zero_gross_returns_zero(self):
        r = compute_post_cost_alpha_retention(0.0, 5.0)
        assert r == pytest.approx(0.0)

    def test_capped_at_one(self):
        r = compute_post_cost_alpha_retention(5.0, 10.0)
        assert r <= 1.0
