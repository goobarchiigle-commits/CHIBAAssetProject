"""
src/test_run_ce_shadow_tracking.py
src.live.ce_shadow_tracking.run_ce_shadow_tracking() の回帰テスト（2026-07-15 SSOT統合）。

背景: run_morning_signal.py（2026-07-15廃止）のCapital Efficiency機能を
run_live_signal.pyへShadow Modeとして移植した（実体は独立モジュール
src/live/ce_shadow_tracking.py。run_live_signal.py自体はassert_execution_context()
によりpytest等の未承認スクリプトからのimportをLIVE_MODE=true環境下でブロック
するため、テスト容易性のため切り出した）。実発注数量(order.qty)への影響が
ゼロであることを検証する。

実行:
    python -m pytest src/test_run_ce_shadow_tracking.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.live.ce_shadow_tracking import run_ce_shadow_tracking


@dataclass
class _FakeOrder:
    symbol:           str
    side:             str
    qty:              int
    estimated_price:  float = 1000.0


class TestRunCeShadowTracking(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._runtime_dir = Path(self._tmpdir.name) / "runtime"
        self._logs_dir    = Path(self._tmpdir.name) / "logs"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run(self, signals, orders):
        return run_ce_shadow_tracking(
            signals, orders, runtime_dir=self._runtime_dir, logs_dir=self._logs_dir,
        )

    def test_does_not_mutate_order_qty(self):
        """Tier0固定: order.qtyは呼び出し前後で不変であること（実発注数量への影響ゼロ）。"""
        orders = [_FakeOrder(symbol="5301.T", side="BUY", qty=300)]
        signals = [{"symbol": "5301.T", "rsr": 91.9}]
        original_qty = orders[0].qty

        self._run(signals, orders)

        self.assertEqual(orders[0].qty, original_qty)

    def test_sell_orders_ignored(self):
        orders = [_FakeOrder(symbol="8053.T", side="SELL", qty=100)]
        signals = [{"symbol": "8053.T", "rsr": 63.6}]
        result = self._run(signals, orders)
        self.assertEqual(result, {})

    def test_no_orders_returns_empty_dict(self):
        result = self._run([], [])
        self.assertEqual(result, {})

    def test_buy_order_produces_shadow_metadata(self):
        orders = [_FakeOrder(symbol="6981.T", side="BUY", qty=100, estimated_price=9066.0)]
        signals = [{"symbol": "6981.T", "rsr": 100.0}]

        result = self._run(signals, orders)

        self.assertIn("6981.T", result)
        meta = result["6981.T"]
        self.assertEqual(meta["actual_qty"], 100)
        # 新規state(サンプル不足)ではea=0.0固定 → scale=1.0 → shadow_qty==base_qty
        self.assertEqual(meta["shadow_qty"], 100)
        self.assertEqual(meta["ea"], 0.0)
        self.assertIn("confidence", meta)

    def test_shadow_qty_always_multiple_of_100_and_at_least_100(self):
        orders = [_FakeOrder(symbol="6981.T", side="BUY", qty=100)]
        signals = [{"symbol": "6981.T", "rsr": 100.0}]
        result = self._run(signals, orders)
        shadow_qty = result["6981.T"]["shadow_qty"]
        self.assertGreaterEqual(shadow_qty, 100)
        self.assertEqual(shadow_qty % 100, 0)

    def test_creates_ce_state_pkl(self):
        orders = [_FakeOrder(symbol="6981.T", side="BUY", qty=100)]
        signals = [{"symbol": "6981.T", "rsr": 100.0}]
        self._run(signals, orders)
        ce_state_file = self._runtime_dir / "ce_state.pkl"
        self.assertTrue(ce_state_file.exists())

    def test_fail_open_on_broken_ce_state_file(self):
        """壊れたce_state.pklがあっても例外を外へ漏らさずFAIL_OPENすること。"""
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        (self._runtime_dir / "ce_state.pkl").write_bytes(b"not a valid pickle")

        orders = [_FakeOrder(symbol="6981.T", side="BUY", qty=100)]
        signals = [{"symbol": "6981.T", "rsr": 100.0}]
        # 例外を送出しないこと（クラッシュしないことがテスト対象）
        try:
            self._run(signals, orders)
        except Exception as exc:  # pragma: no cover - このテストは例外なしを期待
            self.fail(f"run_ce_shadow_tracking raised unexpectedly: {exc}")

    def test_multiple_buy_orders_each_tracked_independently(self):
        orders = [
            _FakeOrder(symbol="8035.T", side="BUY", qty=100),
            _FakeOrder(symbol="6857.T", side="BUY", qty=200),
        ]
        signals = [
            {"symbol": "8035.T", "rsr": 96.8},
            {"symbol": "6857.T", "rsr": 93.5},
        ]
        result = self._run(signals, orders)
        self.assertEqual(set(result.keys()), {"8035.T", "6857.T"})
        self.assertEqual(result["8035.T"]["actual_qty"], 100)
        self.assertEqual(result["6857.T"]["actual_qty"], 200)


if __name__ == "__main__":
    unittest.main()
