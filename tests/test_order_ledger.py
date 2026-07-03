import unittest
import shutil
from datetime import date
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

from src.kabusapi.signal_bridge import AbortError
import src.live.order_ledger as ol
from src.live.order_ledger import OrderLedger


@contextmanager
def _workspace_tempdir():
    base_dir = Path(__file__).resolve().parents[1] / ".codex_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / f"test_order_ledger_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield str(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestOrderLedger(unittest.TestCase):

    def test_first_order_allowed(self):
        """初回発注は許可される"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger()
                result = ledger.check_and_record("5411.T", "BUY", qty=100)
        self.assertTrue(result["allowed"])

    def test_duplicate_blocked(self):
        """同一 execution_key の重複は拒否"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger()
                ledger.check_and_record("5411.T", "BUY", qty=100)
                with self.assertRaises(AbortError):
                    ledger.check_and_record("5411.T", "BUY", qty=100)

    def test_daily_limit_exceeded(self):
        """1日 MAX_ORDERS_PER_DAY 超過で拒否"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                with patch.object(ol, "MAX_ORDERS_PER_DAY", 2):
                    ledger = OrderLedger()
                    ledger.check_and_record("A.T", "BUY")
                    ledger.check_and_record("B.T", "BUY")
                    with self.assertRaises(RuntimeError):
                        ledger.check_and_record("C.T", "BUY")

    def test_daily_count_increments(self):
        """発注記録のたびに daily_count が増える"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger()
                self.assertEqual(ledger.daily_count(), 0)
                ledger.check_and_record("A.T", "BUY")
                self.assertEqual(ledger.daily_count(), 1)
                ledger.check_and_record("B.T", "BUY")
                self.assertEqual(ledger.daily_count(), 2)

    def test_different_side_same_symbol_allowed(self):
        """同一銘柄でも BUY/SELL は別 execution_key なので許可"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger()
                r1 = ledger.check_and_record("5411.T", "BUY")
                r2 = ledger.check_and_record("5411.T", "SELL")
        self.assertTrue(r1["allowed"])
        self.assertTrue(r2["allowed"])

    def test_execution_key_format(self):
        """execution_key は date_symbol_side 形式"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                d = date(2026, 4, 7)
                ledger = OrderLedger(trade_date=d)
                key = ledger.execution_key("5411.T", "BUY")
        self.assertEqual(key, "2026-04-07_5411.T_BUY")

    def test_old_date_cleared(self):
        """前日のレジャーは当日初期化時にクリアされる"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            # 昨日の記録を作成
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger_yesterday = OrderLedger(trade_date=date(2026, 4, 6))
                ledger_yesterday.check_and_record("5411.T", "BUY")
                self.assertEqual(ledger_yesterday.daily_count(), 1)

            # 今日のレジャーを開くとクリアされている
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger_today = OrderLedger(trade_date=date(2026, 4, 7))
                self.assertEqual(ledger_today.daily_count(), 0)

    def test_shadow_sell_records_do_not_block_live_sell(self):
        """shadow SELL 記録は live SELL の重複判定対象に含めない"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 4, 14))
                shadow_5401 = ledger.record_shadow("5401.T", "SELL", qty=100)
                shadow_5411 = ledger.record_shadow("5411.T", "SELL", qty=100)

                self.assertTrue(shadow_5401["allowed"])
                self.assertTrue(shadow_5411["allowed"])
                self.assertEqual(ledger.daily_count(), 0)
                self.assertFalse(ledger.is_duplicate("5401.T", "SELL"))
                self.assertFalse(ledger.is_duplicate("5411.T", "SELL"))

                live_5401 = ledger.check_and_record("5401.T", "SELL", qty=100)
                live_5411 = ledger.check_and_record("5411.T", "SELL", qty=100)

        self.assertTrue(live_5401["allowed"])
        self.assertTrue(live_5411["allowed"])


class TestMarkFailed(unittest.TestCase):
    """mark_failed: pending → failed 更新のテスト"""

    def test_mark_failed_updates_status(self):
        """check_and_record 後に mark_failed すると execution_status が failed になること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 4, 14))
                ledger.check_and_record("5401.T", "SELL", qty=100)
                ledger.mark_failed("5401.T", "SELL", error="500 Internal Server Error")
                key = ledger.execution_key("5401.T", "SELL")
                order = ledger._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "failed")
        self.assertIn("500", order["error"])

    def test_mark_failed_allows_reorder(self):
        """failed 状態は daily_count に含まれないので再発注を妨げないこと"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 4, 14))
                ledger.check_and_record("5401.T", "SELL", qty=100)
                self.assertEqual(ledger.daily_count(), 1)
                ledger.mark_failed("5401.T", "SELL", error="500")
                self.assertEqual(ledger.daily_count(), 0)

    def test_mark_failed_unknown_key_does_not_raise(self):
        """存在しない key に mark_failed しても例外が出ないこと"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 4, 14))
                ledger.mark_failed("9999.T", "SELL", error="no entry")  # raises しない

    def test_mark_failed_persists_to_file(self):
        """mark_failed 後にファイルを再ロードしても failed が読み込まれること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 4, 14))
                ledger.check_and_record("5401.T", "SELL", qty=100)
                ledger.mark_failed("5401.T", "SELL", error="500")

                ledger2 = OrderLedger(trade_date=date(2026, 4, 14))
                key = ledger2.execution_key("5401.T", "SELL")
                order = ledger2._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "failed")


class TestMarkSubmitted(unittest.TestCase):
    """
    mark_submitted: pending → submitted 状態遷移のテスト。

    回帰: run_morning_signal.py が発注成功後に mark_submitted() を呼ぶようになった
    ことを保証する。2026-06-04 に同日BUY未着でCBが誤発動した根本原因の修正確認。
    """

    def _make_ledger(self, tmp_path: Path, trade_date=None):
        import src.live.order_ledger as ol
        with patch.object(ol, "LEDGER_PATH", tmp_path / "ledger.json"):
            return OrderLedger(trade_date=trade_date or date(2026, 6, 5)), ol

    def test_mark_submitted_updates_status(self):
        """check_and_record 後に mark_submitted すると execution_status が submitted になること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)
                ledger.mark_submitted("5301.T", "BUY", order_id="20260605A01N00000001")
                key = ledger.execution_key("5301.T", "BUY")
                order = ledger._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "submitted")
        self.assertEqual(order["order_id"], "20260605A01N00000001")

    def test_mark_submitted_persists_to_file(self):
        """mark_submitted 後にファイルを再ロードしても submitted が読み込まれること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)
                ledger.mark_submitted("5301.T", "BUY", order_id="ORD001")

                ledger2 = OrderLedger(trade_date=date(2026, 6, 5))
                key = ledger2.execution_key("5301.T", "BUY")
                order = ledger2._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "submitted")
        self.assertEqual(order["order_id"], "ORD001")

    def test_mark_submitted_unknown_key_does_not_raise(self):
        """存在しない key に mark_submitted しても例外が出ないこと"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.mark_submitted("9999.T", "BUY", order_id="ORD_MISSING")

    def test_mark_submitted_empty_order_id_fallback(self):
        """order_id が空文字（ブローカー未返却）でも例外なく記録されること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)
                ledger.mark_submitted("5301.T", "BUY", order_id="")
                key = ledger.execution_key("5301.T", "BUY")
                order = ledger._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "submitted")
        self.assertEqual(order["order_id"], "")

    def test_mark_submitted_is_duplicate(self):
        """submitted は is_duplicate で True（二重発注防止対象）"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)
                ledger.mark_submitted("5301.T", "BUY", order_id="ORD001")
                result = ledger.is_duplicate("5301.T", "BUY")
        self.assertTrue(result, "submitted order must be counted as duplicate")

    def test_mark_submitted_counted_in_daily_count(self):
        """submitted は daily_count に含まれること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)
                count_before = ledger.daily_count()
                ledger.mark_submitted("5301.T", "BUY", order_id="ORD001")
                count_after = ledger.daily_count()
        self.assertEqual(count_before, 1)
        self.assertEqual(count_after, 1, "daily_count must not change after pending→submitted")

    def test_mark_submitted_sell_side(self):
        """SELL 側でも mark_submitted が正しく動作すること"""
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("6981.T", "SELL", qty=100, price=5000.0)
                ledger.mark_submitted("6981.T", "SELL", order_id="ORD002")
                key = ledger.execution_key("6981.T", "SELL")
                order = ledger._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "submitted")

    def test_morning_signal_success_path_calls_mark_submitted(self):
        """
        run_morning_signal.py の成功経路が mark_submitted を呼ぶことを確認。
        send_results の success=True エントリーに対して submitted 遷移が起きること。
        """
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                # simulate check_and_record (called before send)
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)

                # simulate send_results loop from run_morning_signal.py
                send_results = [
                    {"symbol": "5301.T", "side": "BUY", "qty": 400,
                     "success": True, "order_id": "20260605A01N00000001"},
                ]
                for r in send_results:
                    if r.get("success"):
                        ledger.mark_submitted(r["symbol"], r["side"], r.get("order_id", ""))
                    else:
                        ledger.mark_failed(r["symbol"], r["side"],
                                           error=r.get("error", "unknown"))

                key = ledger.execution_key("5301.T", "BUY")
                order = ledger._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "submitted")
        self.assertEqual(order["order_id"], "20260605A01N00000001")

    def test_morning_signal_failure_path_calls_mark_failed(self):
        """
        run_morning_signal.py の失敗経路が mark_failed を呼ぶことを確認。
        send_results の success=False エントリーに対して failed 遷移が起きること。
        """
        with _workspace_tempdir() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            import src.live.order_ledger as ol
            with patch.object(ol, "LEDGER_PATH", ledger_path):
                ledger = OrderLedger(trade_date=date(2026, 6, 5))
                ledger.check_and_record("5301.T", "BUY", qty=400, price=1852.5)

                send_results = [
                    {"symbol": "5301.T", "side": "BUY", "qty": 400,
                     "success": False, "error": "503 Service Unavailable"},
                ]
                for r in send_results:
                    if r.get("success"):
                        ledger.mark_submitted(r["symbol"], r["side"], r.get("order_id", ""))
                    else:
                        ledger.mark_failed(r["symbol"], r["side"],
                                           error=r.get("error", "unknown"))

                key = ledger.execution_key("5301.T", "BUY")
                order = ledger._ledger["orders"][key]
        self.assertEqual(order["execution_status"], "failed")
        self.assertIn("503", order["error"])


class TestMarketOrderPriceExclusion(unittest.TestCase):
    """send_order: 成行注文時に Price が payload から除外されること"""

    def setUp(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import src.kabusapi.client as client_module
        from src.kabusapi.client import _CACHE, KabuClient, Side, Exchange, OrderType
        self._CACHE = _CACHE
        self._CACHE._token = ""
        self._CACHE._fetched_at = 0.0
        self._CACHE._fetch_count = 0
        from unittest.mock import patch as _patch, MagicMock
        with _patch.dict("os.environ", {
            "KABU_API_PASSWORD": "testpass",
            "KABU_TRADE_PASSWORD": "tradepass",
            "KABU_ACCOUNT_TYPE": "4",
        }):
            self.client = KabuClient()
        self.client._token = "tok_test"
        self.Side = Side
        self.Exchange = Exchange
        self.OrderType = OrderType

    def _payload(self, order_type, side=None) -> dict:
        from unittest.mock import MagicMock, patch
        side = side or self.Side.SELL
        exchange = self.Exchange.TSE if side == self.Side.SELL else self.Exchange.SOR
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"OrderId": "20260414A02N00000001"}
        mock_resp.raise_for_status.return_value = None
        with patch.object(self.client._session, "post", return_value=mock_resp) as mp:
            self.client.send_order(
                symbol=    "5401",
                exchange=  exchange,
                side=      side,
                qty=       100,
                order_type=order_type,
                price=     0.0,
            )
        _, kwargs = mp.call_args
        return kwargs.get("json", {})

    def test_market_sell_has_no_price(self):
        payload = self._payload(self.OrderType.MARKET, self.Side.SELL)
        self.assertNotIn("Price", payload, "MARKET SELL に Price が含まれてはならない")

    def test_market_open_sell_has_no_price(self):
        payload = self._payload(self.OrderType.MARKET_OPEN, self.Side.SELL)
        self.assertNotIn("Price", payload, "MARKET_OPEN SELL に Price が含まれてはならない")

    def test_market_buy_has_no_price(self):
        payload = self._payload(self.OrderType.MARKET, self.Side.BUY)
        self.assertNotIn("Price", payload, "MARKET BUY に Price が含まれてはならない（回帰）")

    def test_market_open_buy_has_no_price(self):
        payload = self._payload(self.OrderType.MARKET_OPEN, self.Side.BUY)
        self.assertNotIn("Price", payload, "MARKET_OPEN BUY に Price が含まれてはならない（回帰）")

    def test_limit_sell_keeps_price(self):
        """指値注文は Price を残すこと"""
        payload = self._payload(self.OrderType.LIMIT, self.Side.SELL)
        self.assertIn("Price", payload, "指値 SELL は Price が必要")


if __name__ == "__main__":
    unittest.main()
