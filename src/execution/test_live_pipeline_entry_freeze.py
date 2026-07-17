"""
src/execution/test_live_pipeline_entry_freeze.py
Entry Freeze Mode（資産保全・2026-07-17）の live_pipeline.generate_orders() 側検証。

背景: pipeline.py（引数なし実行時のデフォルト分岐）が signal_bridge.py と全く独立に
run_live_pipeline → generate_orders → execute_orders という発注経路を持つことが
Study "Entry Freeze" タスクの全探索で判明した（KABU_API_KEY未設定のため現状は
到達時にKeyErrorで停止するが、将来設定されれば実発注し得る残存経路）。
generate_orders() 側にも同一のEntry Freezeゲートを追加したため、その検証を行う。

実行:
    cd C:/ai-trading
    python -m pytest src/execution/test_live_pipeline_entry_freeze.py -v
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

import pandas as pd
from unittest.mock import patch

from src.config_loader import load_strategy_config
from src.execution.live_pipeline import generate_orders, execute_orders


class TestLivePipelineEntryFreeze(unittest.TestCase):

    def setUp(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        load_strategy_config.cache_clear()

    def tearDown(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        load_strategy_config.cache_clear()

    def _rebalance_inputs(self):
        # decision="ALLOCATE"(mult=1.0) x kill_state="KEEP"(mult=1.0) → mult=1.0
        allocation_result = {"decision": "ALLOCATE"}
        kill_state = "KEEP"
        full_target_sizes = {"1234.T": 100, "5678.T": 0}   # 1234=新規BUY方向 / 5678=手仕舞いSELL方向
        current_positions = {"5678.T": 50}
        return allocation_result, kill_state, full_target_sizes, current_positions

    def test_buy_blocked_when_frozen(self):
        os.environ["ENTRY_FREEZE_ENABLED"] = "1"
        alloc, kill_state, targets, positions = self._rebalance_inputs()
        orders = generate_orders(alloc, kill_state, targets, positions)
        sides = sorted(o["side"] for o in orders)
        self.assertNotIn("BUY", sides, "Entry Freeze中はBUY方向のrebalanceを生成してはならない")

    def test_sell_still_generated_when_frozen(self):
        os.environ["ENTRY_FREEZE_ENABLED"] = "1"
        alloc, kill_state, targets, positions = self._rebalance_inputs()
        orders = generate_orders(alloc, kill_state, targets, positions)
        sell_orders = [o for o in orders if o["side"] == "SELL"]
        self.assertEqual(len(sell_orders), 1, "SELL方向のrebalanceはfreeze中でも生成されること")
        self.assertEqual(sell_orders[0]["symbol"], "5678.T")

    def test_buy_generated_when_not_frozen(self):
        """回帰guard: freeze無効時は従来通りBUY方向のrebalanceが生成されること。
        strategy.yamlの既定は2026-07-17時点でentry_freeze.enabled=trueのため、
        ここでは環境変数で明示的にfreezeを無効化してテストする。"""
        os.environ["ENTRY_FREEZE_ENABLED"] = "0"
        load_strategy_config.cache_clear()
        alloc, kill_state, targets, positions = self._rebalance_inputs()
        orders = generate_orders(alloc, kill_state, targets, positions)
        sides = sorted(o["side"] for o in orders)
        self.assertIn("BUY", sides, "freeze無効時は従来通りBUYが生成されること（回帰guard）")


class TestExecuteOrdersFinalGuard(unittest.TestCase):
    """execute_orders() 自体の最終防波堤（generate_orders()を経由せず直接
    呼ばれた場合でもPOST直前でBUYを遮断できること）を検証する。"""

    def setUp(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        os.environ["KABU_API_KEY"] = "dummy-test-key"
        load_strategy_config.cache_clear()

    def tearDown(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        os.environ.pop("KABU_API_KEY", None)
        load_strategy_config.cache_clear()

    def test_buy_blocked_before_post_when_frozen(self):
        os.environ["ENTRY_FREEZE_ENABLED"] = "1"
        orders = [{"symbol": "1234.T", "side": "BUY", "qty": 100, "target": 100}]
        market_data = pd.DataFrame({"symbol": ["1234.T"], "open": [1000.0]})

        with patch("src.execution.live_pipeline.requests.post",
                   side_effect=AssertionError("requests.post must NOT be called when frozen")):
            executed = execute_orders(
                orders, "2026-07-17", market_data, set(),
                decision="ALLOCATE", kill_state="KEEP", strategy_id="test",
            )
        self.assertEqual(executed, [], "frozen中はexecuted結果も空であること")

    def test_sell_still_reaches_post_when_frozen(self):
        os.environ["ENTRY_FREEZE_ENABLED"] = "1"
        orders = [{"symbol": "5678.T", "side": "SELL", "qty": 50, "target": 0}]
        market_data = pd.DataFrame({"symbol": ["5678.T"], "open": [1000.0]})

        with patch("src.execution.live_pipeline.requests.post",
                   side_effect=RuntimeError("stop here — POST was reached as expected")):
            with self.assertRaises(RuntimeError):
                execute_orders(
                    orders, "2026-07-17", market_data, set(),
                    decision="ALLOCATE", kill_state="KEEP", strategy_id="test",
                )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
