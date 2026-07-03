"""Tests for admission_control.py"""
import pytest

from src.capital.admission_control import (
    AdmissionConfig,
    AdmissionDecision,
    assess_admission,
    filter_universe_by_capital,
    LOT_SIZE,
)


class TestAssessAdmission:
    def test_held_always_admitted(self):
        decision = assess_admission("6857.T", 50_000.0, 3_000_000, held=True)
        assert decision.admitted
        assert decision.reason == "held_position"

    def test_unknown_price_conservative_admit(self):
        decision = assess_admission("XXXX.T", 0.0, 3_000_000, held=False)
        assert decision.admitted
        assert "unknown" in decision.reason

    def test_affordable_stock_admitted(self):
        # ¥3,000/share × 100 = ¥300K; 3M × 0.30 = ¥810K (after 10% buffer) → admitted
        decision = assess_admission("7203.T", 3_000.0, 3_000_000)
        assert decision.admitted

    def test_expensive_stock_rejected_at_low_capital(self):
        # ¥50,000/share × 100 = ¥5M > 3M × 0.30 × 0.90 → rejected
        decision = assess_admission("6857.T", 50_000.0, 3_000_000)
        assert not decision.admitted

    def test_expensive_stock_admitted_at_high_capital(self):
        # ¥50,000 × 100 = ¥5M; 20M × 0.30 × 0.90 = ¥5.4M → admitted
        decision = assess_admission("6857.T", 50_000.0, 20_000_000)
        assert decision.admitted

    def test_capital_progression_3m(self):
        # At 3M: effective_max = min(3M*0.90*0.30, 3M*0.25) = min(810K, 750K) = 750K
        # → max price = 750K / 100 = ¥7,500/share
        cfg = AdmissionConfig()
        max_by_ratio = 3_000_000 * (1 - cfg.cash_safety_buffer) * cfg.max_lot_cost_ratio
        max_by_pos = 3_000_000 * cfg.max_single_position_ratio
        effective_max = min(max_by_ratio, max_by_pos)
        max_price = effective_max / LOT_SIZE
        decision = assess_admission("X.T", max_price - 1, 3_000_000, cfg=cfg)
        assert decision.admitted

    def test_capital_progression_4m(self):
        # At 4M: ¥10,800/share
        cfg = AdmissionConfig()
        max_price = 4_000_000 * (1 - cfg.cash_safety_buffer) * cfg.max_lot_cost_ratio / LOT_SIZE
        decision = assess_admission("X.T", max_price + 1, 4_000_000, cfg=cfg)
        assert not decision.admitted

    def test_zero_effective_capital_rejects(self):
        decision = assess_admission("7203.T", 3_000.0, 0.0)
        assert not decision.admitted

    def test_lot_cost_ratio_in_result(self):
        decision = assess_admission("7203.T", 3_000.0, 3_000_000)
        expected_ratio = (3_000.0 * LOT_SIZE) / 3_000_000
        assert decision.lot_cost_ratio == pytest.approx(expected_ratio, abs=0.001)

    def test_max_single_position_also_limits(self):
        # max_single_position_ratio=0.25 < max_lot_cost_ratio=0.30
        cfg = AdmissionConfig(max_lot_cost_ratio=0.30, max_single_position_ratio=0.15)
        # lot_cost = 4000 × 100 = 400K; 4M × 0.25 = 1M; 4M×0.90×0.30 = 1.08M
        # effective_max = min(1.08M, 4M×0.15=600K) = 600K
        # 400K < 600K → admitted
        decision = assess_admission("X.T", 4_000.0, 4_000_000, cfg=cfg)
        assert decision.admitted


class TestFilterUniverseByCapital:
    def test_filters_expensive_stocks(self):
        tickers = {
            "7203.T": "輸送用機器",     # ¥3,000 → affordable at 3M
            "6857.T": "電機精密",        # ¥50,000 → not affordable at 3M
            "8306.T": "銀行業",          # ¥1,500 → affordable
        }
        prices = {
            "7203.T": 3_000.0,
            "6857.T": 50_000.0,
            "8306.T": 1_500.0,
        }
        filtered, rejected = filter_universe_by_capital(
            tickers, prices, 3_000_000, set()
        )
        assert "7203.T" in filtered
        assert "8306.T" in filtered
        assert "6857.T" not in filtered
        assert len(rejected) == 1
        assert rejected[0].symbol == "6857.T"

    def test_held_expensive_not_filtered(self):
        tickers = {"6857.T": "電機精密"}
        prices = {"6857.T": 50_000.0}
        filtered, rejected = filter_universe_by_capital(
            tickers, prices, 3_000_000, {"6857.T"}
        )
        assert "6857.T" in filtered
        assert len(rejected) == 0

    def test_unknown_price_stays_in(self):
        tickers = {"9999.T": "不明"}
        prices = {}
        filtered, rejected = filter_universe_by_capital(
            tickers, prices, 3_000_000, set()
        )
        assert "9999.T" in filtered

    def test_all_affordable_none_rejected(self):
        tickers = {"7203.T": "自動車", "8306.T": "銀行業"}
        prices = {"7203.T": 3_000.0, "8306.T": 1_500.0}
        filtered, rejected = filter_universe_by_capital(
            tickers, prices, 10_000_000, set()
        )
        assert len(filtered) == 2
        assert len(rejected) == 0
