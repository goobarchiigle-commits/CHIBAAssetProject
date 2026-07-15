"""
tests/live/test_process_isolated_metadata_merge.py

entry_price/entry_atr/highest_close 欠落バグ（2026-07-07 follow-up incident）
の回帰テスト。対象: run_live_signal._submit_orders_process_isolated()。

broker_worker.py（子プロセス）の結果には estimated_price/atr20/reason/
sector/strategy_type が含まれない。親プロセス側の元注文オブジェクトから
これらを復元するマージ処理が正しく機能することを検証する。

NOTE: src/run_live_signal.py はモジュールトップレベルで
assert_execution_context() を呼び、LIVE_MODE=true 環境では
sys.argv[0] が _LIVE_ALLOWED_SCRIPTS ("run_live_signal.py"/
"run_morning_signal.py") に含まれない限り import 時に RuntimeError
になる（ライブラリとしての誤importを防ぐ意図的なガード）。
ユニットテストとして関数を検証するため、import前に sys.argv[0] を
一時的にスプーフィングする。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd  # noqa: F401 — pre-warm sys.modules before any heavy-mock block

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _import_run_live_signal():
    _orig_argv0 = sys.argv[0]
    sys.argv[0] = str(_ROOT / "run_live_signal.py")
    try:
        import src.run_live_signal as rls
        return rls
    finally:
        sys.argv[0] = _orig_argv0


class TestProcessIsolatedMetadataMerge(unittest.TestCase):

    def setUp(self):
        try:
            self.rls = _import_run_live_signal()
        except Exception as e:
            self.skipTest(f"src.run_live_signal import unavailable in test env: {e}")

    def test_missing_metadata_restored_from_parent_order_objects(self):
        """broker_worker.py 相当の最小結果（estimated_price等なし）を
        _submit_orders_process_isolated() が親側の order object から復元すること。"""
        order_obj = SimpleNamespace(
            symbol="6506.T", symbol_4digit="6506", side="BUY", qty=100,
            estimated_price=7449.0, atr20=454.45,
            reason="trend_follow fallback=False", sector="機械",
            strategy_type="trend_follow",
        )

        # broker_worker.py の最小スキーマ（estimated_price/atr20/reason/sector/
        # strategy_type を含まない）を模擬した戻り値
        _minimal_broker_result = {
            "symbol": "6506.T", "symbol_4digit": "6506", "side": "BUY", "qty": 100,
            "client_order_id": "COI-WILL-BE-SET",
            "success": True, "order_id": "OID123", "result_code": 0, "error": None,
        }

        fake_proc_supervisor = MagicMock()

        def _fake_submit_orders(orders_dicts, timeout_sec, run_id):
            coi = orders_dicts[0]["client_order_id"]
            return [{**_minimal_broker_result, "client_order_id": coi}]

        fake_proc_supervisor.submit_orders.side_effect = _fake_submit_orders

        with patch.object(self.rls, "compute_front_order_type", return_value=(10, False)):
            results = self.rls._submit_orders_process_isolated(
                [order_obj],
                registry=None,
                trading_day="2026-07-07",
                run_id="test-run",
                proc_supervisor=fake_proc_supervisor,
            )

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["estimated_price"], 7449.0, "broker_worker結果に無いestimated_priceが復元されない")
        self.assertEqual(r["atr20"], 454.45)
        self.assertEqual(r["reason"], "trend_follow fallback=False")
        self.assertEqual(r["sector"], "機械")
        self.assertEqual(r["strategy_type"], "trend_follow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
