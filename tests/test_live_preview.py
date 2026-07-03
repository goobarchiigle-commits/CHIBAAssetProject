"""
tests/test_live_preview.py
print_live_preview() の表示内容を検証する。
"""
import io
import sys
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock


# ── テスト用スタブ ────────────────────────────────────────────────────────────

@dataclass
class _OrderStub:
    symbol: str
    side: str
    qty: int
    estimated_price: float
    estimated_amount: float
    reason: str
    strategy_type: str = "fujiko"
    atr20: float = 0.0


def _make_signal(
    symbol: str,
    signal: int,
    currently_holding: bool,
    rsr: float = 80.0,
    rsr_rank: int = 1,
    hold_days: int = 0,
    sector: str = "テスト",
    unrealized_pnl_pct: float = 0.0,
    stop_price: float = 0,
) -> dict:
    return {
        "symbol":             symbol,
        "sector":             sector,
        "signal":             signal,
        "strategy_type":      "fujiko",
        "rsr":                rsr,
        "rsr_rank":           rsr_rank,
        "sepa_score":         3,
        "rsr_momentum":       0.5,
        "hold_days":          hold_days,
        "currently_holding":  currently_holding,
        "reason":             "TEST_REASON",
        "stop_price":         stop_price,
        "trailing_stop":      stop_price,
        "entry_price":        1000.0,
        "unrealized_pnl_pct": unrealized_pnl_pct,
    }


def _make_result(signals, portfolio_summary=None):
    r = MagicMock()
    r.signals = signals
    r.orders  = []
    r.warnings = []
    r.portfolio_summary = portfolio_summary or {
        "available_cash":    1_000_000,
        "current_equity":    3_000_000,
        "current_positions": 1,
        "max_positions":     3,
    }
    return r


def _capture(result, order_objects) -> str:
    from src.live.preview import print_live_preview
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        print_live_preview(result, order_objects)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ── テストケース ──────────────────────────────────────────────────────────────

class TestLivePreviewHeader(unittest.TestCase):
    """ヘッダーと集計行が常に出力される。"""

    def test_header_always_shown(self):
        result = _make_result([])
        out = _capture(result, [])
        self.assertIn("LIVE PREVIEW", out)
        self.assertIn("planned_buy_count", out)
        self.assertIn("planned_sell_count", out)
        self.assertIn("planned_capital_usage", out)
        self.assertIn("available_cash_now", out)
        self.assertIn("cash_after_orders", out)

    def test_no_orders_shows_no_order_message(self):
        result = _make_result([])
        out = _capture(result, [])
        self.assertIn("発注予定なし", out)

    def test_summary_counts_zero(self):
        result = _make_result([])
        out = _capture(result, [])
        self.assertIn("planned_buy_count    : 0", out)
        self.assertIn("planned_sell_count   : 0", out)


class TestLivePreviewBuyOrders(unittest.TestCase):
    """BUY候補セクション。"""

    def _setup(self, n_buy: int = 1):
        sigs = [_make_signal(f"100{i}.T", signal=1, currently_holding=False,
                              rsr=85.0, rsr_rank=i + 1)
                for i in range(n_buy)]
        orders = [_OrderStub(
            symbol=f"100{i}.T",
            side="BUY",
            qty=100,
            estimated_price=1_500.0,
            estimated_amount=150_000.0,
            reason="RSR_ENTRY",
        ) for i in range(n_buy)]
        result = _make_result(sigs, {"available_cash": 500_000, "current_equity": 3_000_000,
                                      "current_positions": 0, "max_positions": 3})
        return result, orders

    def test_buy_section_header(self):
        result, orders = self._setup(1)
        out = _capture(result, orders)
        self.assertIn("[BUY候補]", out)
        self.assertIn("1件", out)

    def test_buy_symbol_shown(self):
        result, orders = self._setup(1)
        out = _capture(result, orders)
        self.assertIn("1000.T", out)

    def test_buy_qty_shown(self):
        result, orders = self._setup(1)
        out = _capture(result, orders)
        self.assertIn("100株", out)

    def test_buy_amount_shown(self):
        result, orders = self._setup(1)
        out = _capture(result, orders)
        self.assertIn("150,000", out)

    def test_buy_rsr_shown(self):
        result, orders = self._setup(1)
        out = _capture(result, orders)
        self.assertIn("85.0", out)

    def test_planned_buy_count_matches(self):
        result, orders = self._setup(2)
        out = _capture(result, orders)
        self.assertIn("planned_buy_count    : 2", out)

    def test_capital_usage_summed(self):
        result, orders = self._setup(2)
        out = _capture(result, orders)
        self.assertIn("¥300,000", out)  # 150,000 × 2

    def test_cash_after_reduced_by_buy(self):
        result, orders = self._setup(1)
        out = _capture(result, orders)
        # available_cash=500,000 - buy=150,000 = 350,000
        self.assertIn("350,000", out)

    def test_reason_truncated_to_35chars(self):
        sigs = [_make_signal("1000.T", signal=1, currently_holding=False)]
        orders = [_OrderStub("1000.T", "BUY", 100, 1500.0, 150_000.0,
                              reason="A" * 50)]
        result = _make_result(sigs)
        out = _capture(result, orders)
        # reason should appear but truncated
        self.assertIn("A" * 35, out)
        self.assertNotIn("A" * 36, out)


class TestLivePreviewSellOrders(unittest.TestCase):
    """SELL候補セクション。"""

    def _setup(self):
        sigs = [_make_signal("2000.T", signal=-1, currently_holding=True,
                              rsr=45.0, rsr_rank=30, hold_days=10,
                              unrealized_pnl_pct=-0.05, stop_price=950.0)]
        orders = [_OrderStub("2000.T", "SELL", 100, 950.0, 95_000.0, "TRAIL_STOP")]
        result = _make_result(sigs, {"available_cash": 200_000, "current_equity": 3_000_000,
                                      "current_positions": 1, "max_positions": 3})
        return result, orders

    def test_sell_section_header(self):
        result, orders = self._setup()
        out = _capture(result, orders)
        self.assertIn("[SELL候補]", out)

    def test_sell_symbol_shown(self):
        result, orders = self._setup()
        out = _capture(result, orders)
        self.assertIn("2000.T", out)

    def test_sell_hold_days_shown(self):
        result, orders = self._setup()
        out = _capture(result, orders)
        self.assertIn("10d", out)

    def test_planned_sell_count_matches(self):
        result, orders = self._setup()
        out = _capture(result, orders)
        self.assertIn("planned_sell_count   : 1", out)

    def test_cash_after_increased_by_sell(self):
        result, orders = self._setup()
        out = _capture(result, orders)
        # available_cash=200,000 + sell=95,000 = 295,000
        self.assertIn("295,000", out)


class TestLivePreviewHoldContinue(unittest.TestCase):
    """HOLD継続セクション。"""

    def _setup(self):
        sigs = [_make_signal("3000.T", signal=1, currently_holding=True,
                              rsr=90.0, rsr_rank=2, hold_days=15,
                              unrealized_pnl_pct=0.12, stop_price=1_800.0)]
        result = _make_result(sigs)
        return result

    def test_hold_section_header(self):
        result = self._setup()
        out = _capture(result, [])
        self.assertIn("[HOLD継続]", out)

    def test_hold_symbol_shown(self):
        result = self._setup()
        out = _capture(result, [])
        self.assertIn("3000.T", out)

    def test_hold_rsr_shown(self):
        result = self._setup()
        out = _capture(result, [])
        self.assertIn("90.0", out)

    def test_hold_days_shown(self):
        result = self._setup()
        out = _capture(result, [])
        self.assertIn("15d", out)

    def test_hold_pnl_shown(self):
        result = self._setup()
        out = _capture(result, [])
        self.assertIn("+12.0%", out)

    def test_hold_stop_price_shown(self):
        result = self._setup()
        out = _capture(result, [])
        self.assertIn("1,800", out)

    def test_hold_no_order_shows_no_order_message_absent(self):
        # HOLDのみ → "発注予定なし" は表示しない（[HOLD継続]セクションが代替）
        result = self._setup()
        out = _capture(result, [])
        self.assertNotIn("発注予定なし", out)

    def test_no_stop_price_shows_dash(self):
        sigs = [_make_signal("3001.T", signal=1, currently_holding=True, stop_price=0)]
        result = _make_result(sigs)
        out = _capture(result, [])
        self.assertIn("—", out)


class TestLivePreviewShadowBuy(unittest.TestCase):
    """SHADOW_BUY は別セクションで表示される。"""

    def test_shadow_buy_section(self):
        sigs = [_make_signal("5000.T", signal=1, currently_holding=False)]
        orders = [_OrderStub("5000.T", "SHADOW_BUY", 100, 2000.0, 200_000.0, "SHADOW")]
        result = _make_result(sigs)
        out = _capture(result, orders)
        self.assertIn("[SHADOW BUY候補]", out)
        self.assertIn("5000.T", out)

    def test_shadow_not_counted_in_buy_count(self):
        sigs = [_make_signal("5000.T", signal=1, currently_holding=False)]
        orders = [_OrderStub("5000.T", "SHADOW_BUY", 100, 2000.0, 200_000.0, "SHADOW")]
        result = _make_result(sigs)
        out = _capture(result, orders)
        self.assertIn("planned_buy_count    : 0", out)

    def test_shadow_not_counted_in_capital_usage(self):
        sigs = [_make_signal("5000.T", signal=1, currently_holding=False)]
        orders = [_OrderStub("5000.T", "SHADOW_BUY", 100, 2000.0, 200_000.0, "SHADOW")]
        result = _make_result(sigs, {"available_cash": 500_000, "current_equity": 3_000_000,
                                      "current_positions": 0, "max_positions": 3})
        out = _capture(result, orders)
        self.assertIn("planned_capital_usage: ¥0", out)


class TestLivePreviewMixedOrders(unittest.TestCase):
    """BUY + SELL + HOLD 混在ケース。"""

    def test_mixed_all_sections_shown(self):
        sigs = [
            _make_signal("1000.T", signal=1, currently_holding=False, rsr=85.0, rsr_rank=1),
            _make_signal("2000.T", signal=-1, currently_holding=True, rsr=40.0, rsr_rank=35, hold_days=20),
            _make_signal("3000.T", signal=1, currently_holding=True, rsr=90.0, rsr_rank=2, hold_days=5),
        ]
        orders = [
            _OrderStub("1000.T", "BUY",  100, 1500.0, 150_000.0, "RSR_ENTRY"),
            _OrderStub("2000.T", "SELL", 100,  800.0,  80_000.0, "EXIT"),
        ]
        result = _make_result(sigs, {"available_cash": 600_000, "current_equity": 3_000_000,
                                      "current_positions": 2, "max_positions": 3})
        out = _capture(result, orders)
        self.assertIn("[BUY候補]", out)
        self.assertIn("[SELL候補]", out)
        self.assertIn("[HOLD継続]", out)
        self.assertIn("1000.T", out)
        self.assertIn("2000.T", out)
        self.assertIn("3000.T", out)

    def test_mixed_cash_calculation(self):
        sigs = [
            _make_signal("1000.T", signal=1, currently_holding=False),
            _make_signal("2000.T", signal=-1, currently_holding=True, hold_days=10),
        ]
        orders = [
            _OrderStub("1000.T", "BUY",  100, 1500.0, 150_000.0, "BUY"),
            _OrderStub("2000.T", "SELL", 100,  900.0,  90_000.0, "SELL"),
        ]
        result = _make_result(sigs, {"available_cash": 400_000, "current_equity": 3_000_000,
                                      "current_positions": 1, "max_positions": 3})
        out = _capture(result, orders)
        # 400,000 - 150,000 + 90,000 = 340,000
        self.assertIn("340,000", out)


class TestLivePreviewFailOpen(unittest.TestCase):
    """例外発生時は FAIL_OPEN（呼び出し元が握りつぶす）。"""

    def test_corrupted_portfolio_summary(self):
        """portfolio_summary が None でも print_live_preview は例外を伝播させない（ps が or {} で保護済み）。"""
        from src.live.preview import print_live_preview
        result = _make_result([])
        result.portfolio_summary = None
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            print_live_preview(result, [])  # should not raise
        finally:
            sys.stdout = old
        out = buf.getvalue()
        self.assertIn("LIVE PREVIEW", out)


if __name__ == "__main__":
    unittest.main()
