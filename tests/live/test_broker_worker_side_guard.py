"""
tests/live/test_broker_worker_side_guard.py

2026-07-07 4銘柄同時保有インシデントの回帰テスト。

src/live/broker_worker.py は _submit_orders_process_isolated() から
BrokerProcessSupervisor 経由で子プロセスとして起動される、実際に
kabu API へ発注する経路（run_live_signal.py の「プロセス分離モード」）。
以前は side="SHADOW_BUY" を Side.BUY にマッピングしており、
observation_only のはずの Shadow候補が実際にブローカーへ送信されていた
（5301.T 300株 ¥527,550 の不正発注の直接原因）。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.live.broker_worker as bw


class TestBrokerWorkerSideGuard(unittest.TestCase):

    def _order(self, side: str, **extra) -> dict:
        return {
            "symbol_4digit":    "5301",
            "symbol":           "5301.T",
            "side":             side,
            "qty":              300,
            "client_order_id":  "COI-TEST",
            **extra,
        }

    def test_unknown_side_never_calls_broker(self):
        """side が BUY/SELL 以外なら client.send_order を一切呼ばずfail-closedで返す。"""
        client = MagicMock()
        result = bw._submit_single_order(client, self._order("SHADOW_BUY"), front_order_type=10)

        client.send_order.assert_not_called()
        self.assertFalse(result["success"])
        self.assertIn("rejected_unknown_side", result["error"])
        self.assertIsNone(result["order_id"])

    def test_buy_side_maps_to_side_buy_and_calls_broker(self):
        from src.kabusapi.client import Side
        client = MagicMock()
        client.send_order.return_value = MagicMock(success=True, order_id="OID1", result_code=0)

        result = bw._submit_single_order(client, self._order("BUY"), front_order_type=10)

        self.assertTrue(client.send_order.called)
        _, kwargs = client.send_order.call_args
        self.assertEqual(kwargs["side"], Side.BUY)
        self.assertTrue(result["success"])

    def test_sell_side_maps_to_side_sell_and_calls_broker(self):
        from src.kabusapi.client import Side
        client = MagicMock()
        client.send_order.return_value = MagicMock(success=True, order_id="OID2", result_code=0)

        result = bw._submit_single_order(client, self._order("SELL"), front_order_type=10)

        self.assertTrue(client.send_order.called)
        _, kwargs = client.send_order.call_args
        self.assertEqual(kwargs["side"], Side.SELL)
        self.assertTrue(result["success"])

    def test_result_forwards_estimated_price_and_strategy_type(self):
        """entry-metadata-loss follow-up: 入力にある estimated_price/strategy_type を
        base_result へ転記すること（欠落バグの防御的多重化）。"""
        client = MagicMock()
        client.send_order.return_value = MagicMock(success=True, order_id="OID3", result_code=0)

        order = self._order("BUY", estimated_price=7449.0, strategy_type="trend_follow")
        result = bw._submit_single_order(client, order, front_order_type=10)

        self.assertEqual(result["estimated_price"], 7449.0)
        self.assertEqual(result["strategy_type"], "trend_follow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
