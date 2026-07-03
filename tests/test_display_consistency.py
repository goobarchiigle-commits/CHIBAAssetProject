"""
tests/test_display_consistency.py
表示層一貫性テスト: 保有継続と新規BUY候補の分離を保証する。
"""
import io
import sys
import unittest
from unittest.mock import MagicMock


def _make_signal(
    symbol: str,
    signal: int,
    currently_holding: bool,
    entry_price: float = 0,
    trailing_stop: float = 0,
    unrealized_pnl_pct: float = 0.0,
    strategy_type: str = "fujiko",
    sector: str = "テスト",
    rsr: float = 80.0,
    sepa_score: int = 3,
    rsr_momentum: float = 0.5,
    stop_price: float = 0,
) -> dict:
    return {
        "symbol":             symbol,
        "sector":             sector,
        "signal":             signal,
        "strategy_type":      strategy_type,
        "rsr":                rsr,
        "sepa_score":         sepa_score,
        "rsr_momentum":       rsr_momentum,
        "hold_days":          5 if currently_holding else 0,
        "currently_holding":  currently_holding,
        "reason":             "TEST",
        "stop_price":         stop_price,
        "trailing_stop":      trailing_stop,
        "entry_price":        entry_price,
        "unrealized_pnl_pct": unrealized_pnl_pct,
    }


def _make_result(signals, orders=None, warnings=None):
    r = MagicMock()
    r.signals  = signals
    r.orders   = orders or []
    r.warnings = warnings or []
    return r


def _capture_print_signals(result) -> str:
    """print_signals() の出力を文字列として返す。"""
    from src.run_morning_signal import print_signals
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        print_signals(result)
    finally:
        sys.stdout = old
    return buf.getvalue()


class TestHeldPositionNotRenderedAsBUY(unittest.TestCase):
    """保有継続銘柄が ✅ BUY と表示されないことを検証する。"""

    def test_held_signal1_shows_hold_not_buy(self):
        signals = [
            _make_signal("9104.T", signal=1, currently_holding=True,
                         entry_price=6351, trailing_stop=6167,
                         unrealized_pnl_pct=0.025),
            _make_signal("7203.T", signal=1, currently_holding=False),
        ]
        result = _make_result(signals)
        output = _capture_print_signals(result)

        # 保有銘柄は HOLD 表示
        self.assertIn("📌 HOLD", output, "保有継続銘柄が HOLD 表示されていない")
        # 保有銘柄に BUY ラベルが付かない（✅ BUY の行に 9104.T が現れない）
        for line in output.splitlines():
            if "9104.T" in line and "ランキング" not in line and "保有継続" not in line:
                self.assertNotIn("✅ BUY", line, f"保有継続銘柄に ✅ BUY が表示: {line}")

    def test_new_candidate_still_shows_buy(self):
        signals = [
            _make_signal("7203.T", signal=1, currently_holding=False),
        ]
        result = _make_result(signals)
        output = _capture_print_signals(result)
        self.assertIn("✅ BUY", output, "新規BUY候補が ✅ BUY と表示されていない")

    def test_non_buy_signal_no_buy_label(self):
        signals = [
            _make_signal("6501.T", signal=0, currently_holding=False),
        ]
        result = _make_result(signals)
        output = _capture_print_signals(result)
        self.assertNotIn("✅ BUY", output)
        self.assertNotIn("📌 HOLD", output)


class TestNoOrderMessageIncludesHolding(unittest.TestCase):
    """注文なし時メッセージに保有継続銘柄が含まれることを検証する。"""

    def test_no_order_with_holding_shows_symbol(self):
        signals = [
            _make_signal("9104.T", signal=1, currently_holding=True,
                         entry_price=6351, trailing_stop=6167,
                         unrealized_pnl_pct=0.025),
        ]
        result = _make_result(signals, orders=[])
        output = _capture_print_signals(result)

        self.assertIn("新規注文なし", output, "保有継続時に「新規注文なし」と表示されていない")
        self.assertIn("9104.T", output, "保有銘柄コードがメッセージに含まれていない")
        self.assertIn("保有継続", output, "「保有継続」文言がメッセージに含まれていない")

    def test_no_order_no_holding_shows_generic_message(self):
        signals = [
            _make_signal("7203.T", signal=0, currently_holding=False),
        ]
        result = _make_result(signals, orders=[])
        output = _capture_print_signals(result)

        self.assertIn("本日の注文なし", output)
        self.assertNotIn("新規注文なし", output)

    def test_multiple_holding_symbols_all_shown(self):
        signals = [
            _make_signal("9104.T", signal=1, currently_holding=True,
                         entry_price=6000, trailing_stop=5800),
            _make_signal("7203.T", signal=0, currently_holding=True,
                         entry_price=2500, trailing_stop=2400),
        ]
        result = _make_result(signals, orders=[])
        output = _capture_print_signals(result)
        self.assertIn("9104.T", output)
        self.assertIn("7203.T", output)


class TestHoldingSectionDisplayed(unittest.TestCase):
    """保有継続セクションが表示されることを検証する。"""

    def test_hold_section_shows_entry_and_stop(self):
        signals = [
            _make_signal("9104.T", signal=1, currently_holding=True,
                         entry_price=6351, trailing_stop=6167,
                         unrealized_pnl_pct=0.025),
        ]
        result = _make_result(signals)
        output = _capture_print_signals(result)

        self.assertIn("📌 保有継続", output)
        self.assertIn("avg=", output)
        self.assertIn("last_stop=", output)
        self.assertIn("pnl=", output)

    def test_no_hold_section_when_no_holdings(self):
        signals = [
            _make_signal("7203.T", signal=1, currently_holding=False),
        ]
        result = _make_result(signals)
        output = _capture_print_signals(result)
        # 保有なし → 保有継続セクションは表示されない
        self.assertNotIn("📌 保有継続", output)


if __name__ == "__main__":
    unittest.main()
