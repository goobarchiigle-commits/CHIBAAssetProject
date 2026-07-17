"""
src/kabusapi/test_send_order_entry_freeze_guard.py
Entry Freeze Mode（資産保全・2026-07-17）の最終防波堤検証。

KabuClient.send_order() は全BUY発注経路（signal_bridge._send_orders /
broker_worker.py子プロセス / run_morning_signal.py）が最終的に収束する関数であり、
上流ゲート（_build_orders等）が万一バイパスされてもここでBUYを遮断する
"最後の砦"として entry_freeze の最終チェックを持つ。本テストはそれを検証する。

実行:
    cd C:/ai-trading
    python -m pytest src/kabusapi/test_send_order_entry_freeze_guard.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import load_strategy_config
from src.kabusapi.client import KabuClient, Side, OrderType


def _make_authenticated_client() -> KabuClient:
    """__init__（トークン取得・API接続）を経由せず、最小限の状態だけを持つ
    KabuClientインスタンスを生成する（ネットワーク・認証情報不要）。"""
    client = object.__new__(KabuClient)
    client._token           = "dummy-token"
    client._trade_password  = "dummy-pw"
    client._account_type    = 4
    return client


class TestSendOrderEntryFreezeGuard(unittest.TestCase):

    def setUp(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        load_strategy_config.cache_clear()

    def tearDown(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        load_strategy_config.cache_clear()

    def test_buy_rejected_without_http_call_when_frozen(self):
        os.environ["ENTRY_FREEZE_ENABLED"] = "1"
        client = _make_authenticated_client()

        with patch.object(
            client, "_request_with_token_retry",
            side_effect=AssertionError("HTTP call must NOT happen when frozen"),
        ):
            result = client.send_order(
                symbol="1234", side=Side.BUY, qty=100, order_type=OrderType.MARKET,
            )

        self.assertFalse(result.success, "frozen中のBUYはsuccess=Falseで返ること")
        self.assertEqual(result.result_code, -1)
        self.assertEqual(result.raw.get("rejected"), "entry_freeze")

    def test_sell_still_reaches_http_call_when_frozen(self):
        """freeze中でもSELLはガードを素通りしHTTP送信ロジックへ到達すること
        （実際のHTTP呼び出しはモックし、ガードで止まっていないことのみ検証）。"""
        os.environ["ENTRY_FREEZE_ENABLED"] = "1"
        client = _make_authenticated_client()

        called = {"hit": False}

        def _fake_request(method, url, **kwargs):
            called["hit"] = True
            raise RuntimeError("stop here — HTTP call was reached as expected")

        with patch.object(client, "_request_with_token_retry", side_effect=_fake_request):
            with self.assertRaises(RuntimeError):
                client.send_order(
                    symbol="5678", side=Side.SELL, qty=100, order_type=OrderType.MARKET,
                )
        self.assertTrue(called["hit"], "SELLはfreeze中でもHTTP送信ロジックへ到達すること")

    def test_buy_reaches_http_call_when_not_frozen(self):
        """回帰guard: freeze無効時はBUYも通常通りHTTP送信ロジックへ到達すること。"""
        os.environ["ENTRY_FREEZE_ENABLED"] = "0"
        client = _make_authenticated_client()

        called = {"hit": False}

        def _fake_request(method, url, **kwargs):
            called["hit"] = True
            raise RuntimeError("stop here — HTTP call was reached as expected")

        with patch.object(client, "_request_with_token_retry", side_effect=_fake_request):
            with self.assertRaises(RuntimeError):
                client.send_order(
                    symbol="1234", side=Side.BUY, qty=100, order_type=OrderType.MARKET,
                )
        self.assertTrue(called["hit"], "freeze無効時はBUYもHTTP送信ロジックへ到達すること")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
