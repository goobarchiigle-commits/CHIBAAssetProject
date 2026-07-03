"""Tests for position_sizer.py"""
import pytest

from src.capital.position_sizer import (
    SizerConfig,
    SizedPosition,
    size_position,
    LOT_SIZE,
)


class TestSizePosition:
    def _baseline(self):
        return dict(
            symbol="9984.T",
            signal_weight=0.25,
            price=5000.0,          # ¥5,000/share → ¥500K/lot
            risk_adjusted_capital=4_000_000,
            adv_yen=2_000_000_000, # 2B yen daily volume
        )

    def test_basic_sizing(self):
        pos = size_position(**self._baseline())
        assert pos.lots > 0
        assert pos.estimated_value > 0

    def test_zero_price_returns_zero(self):
        params = {**self._baseline(), "price": 0.0}
        pos = size_position(**params)
        assert pos.lots == 0

    def test_zero_capital_returns_zero(self):
        params = {**self._baseline(), "risk_adjusted_capital": 0.0}
        pos = size_position(**params)
        assert pos.lots == 0

    def test_adv_constraint_applied(self):
        # Use very small ADV → adv constraint should limit position
        params = {**self._baseline(), "adv_yen": 1_000_000}  # 1M yen ADV
        pos = size_position(**params)
        cfg = SizerConfig()
        max_adv_order = 1_000_000 * cfg.max_participation_rate
        assert pos.estimated_value <= max_adv_order + 1  # +1 for rounding

    def test_max_single_position_constraint(self):
        params = {**self._baseline(), "signal_weight": 0.50}  # 50% signal
        pos = size_position(**params)
        cfg = SizerConfig()
        max_single = cfg.max_single_position_ratio * 4_000_000
        assert pos.estimated_value <= max_single + 1

    def test_execution_scalar_reduces_size(self):
        params = {**self._baseline()}
        pos_full = size_position(**params, execution_scalar=1.0)
        pos_half = size_position(**params, execution_scalar=0.7)
        assert pos_half.lots <= pos_full.lots

    def test_result_rounded_to_lots(self):
        pos = size_position(**self._baseline())
        # estimated_value should be a multiple of (price × lot_size)
        remainder = pos.estimated_value % (self._baseline()["price"] * LOT_SIZE)
        assert remainder == pytest.approx(0.0, abs=1.0)

    def test_participation_rate_computed(self):
        pos = size_position(**self._baseline())
        expected_participation = pos.estimated_value / self._baseline()["adv_yen"]
        assert pos.participation_rate == pytest.approx(expected_participation, abs=0.001)

    def test_notes_populated_when_constrained(self):
        # Small ADV triggers constraint note
        params = {**self._baseline(), "adv_yen": 500_000}
        pos = size_position(**params)
        # theoretical > max_adv_order → note about adv_capped
        assert any("adv_capped" in n for n in pos.sizing_notes)

    def test_no_adv_skips_constraint(self):
        params = {**self._baseline(), "adv_yen": 0.0}
        pos = size_position(**params)
        assert any("adv_unknown" in n for n in pos.sizing_notes)

    def test_high_price_stock_sizing(self):
        # High-priced stock: ¥50,000/share → ¥5M/lot > capital×25%
        params = dict(
            symbol="6857.T",
            signal_weight=0.25,
            price=50_000.0,
            risk_adjusted_capital=4_000_000,
            adv_yen=5_000_000_000,
        )
        pos = size_position(**params)
        # max_single = 4M × 0.25 = 1M → lots = 1M // (50K × 100) = 0
        # At ¥50K/share with 4M capital, cannot afford a full lot
        assert pos.lots == 0

    def test_achievable_stock_sizing(self):
        # ¥3,000/share → ¥300K/lot (within 4M × 25% = 1M)
        params = dict(
            symbol="7203.T",
            signal_weight=0.25,
            price=3_000.0,
            risk_adjusted_capital=4_000_000,
            adv_yen=10_000_000_000,
        )
        pos = size_position(**params)
        assert pos.lots >= 1
