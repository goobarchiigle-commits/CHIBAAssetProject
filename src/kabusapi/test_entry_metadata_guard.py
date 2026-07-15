"""
src/kabusapi/test_entry_metadata_guard.py

entry_price/entry_atr/highest_close 欠落バグの回帰テスト
（process-isolated実発注後にportfolio_stateへ0.0が書き込まれた
2026-07-07 follow-up incident, 5301.Tで発覚）。

対象: SignalBridge.update_state_after_execution() の fail-closed 挙動。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd  # noqa: F401 — pre-warm sys.modules so the later
                      # patch.dict(sys.modules,...) heavy-mock blocks don't
                      # transitively remove numpy/pandas on __exit__ (numpy
                      # C-extensions cannot be re-imported in-process).

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _import_bridge_symbols():
    heavy = ["kabusapi", "kabu_station_api", "pandas_datareader"]
    mocks = {m: MagicMock() for m in heavy}
    with patch.dict("sys.modules", mocks):
        from src.kabusapi.signal_bridge import SignalBridge
    return SignalBridge


def _make_bridge(SignalBridge, *, client=None):
    bridge = MagicMock(spec=SignalBridge)
    bridge._cfg = MagicMock()
    bridge._cfg.risk.reentry_cooldown_days = 5
    bridge._client = client
    bridge._load_portfolio_state = MagicMock(return_value={
        "position_entry_dates":     {},
        "position_entry_prices":    {},
        "position_entry_atrs":      {},
        "position_entry_rsrs":      {},
        "position_highest_closes":  {},
        "position_qtys":            {},
        "reentry_blocked":          {},
        "available_cash":           1_000_000.0,
        "shadow_positions":         {},
        "position_strategy_types":  {},
    })
    saved = {}
    def _save(state):
        saved.clear()
        saved.update(state)
    bridge._save_portfolio_state = MagicMock(side_effect=_save)
    bridge._recover_entry_price_from_broker = SignalBridge._recover_entry_price_from_broker.__get__(
        bridge, type(bridge)
    )
    bridge.update_state_after_execution = SignalBridge.update_state_after_execution.__get__(
        bridge, type(bridge)
    )
    return bridge, saved


class TestNormalBuyStillWorks(unittest.TestCase):
    """通常BUY / TrendFollow: estimated_price が正しく渡ればそのまま保存される（回帰基準）。"""

    def setUp(self):
        self.SignalBridge = _import_bridge_symbols()

    def test_normal_buy_records_entry_price_and_atr(self):
        bridge, saved = _make_bridge(self.SignalBridge)
        send_results = [{
            "symbol": "7203.T", "side": "BUY", "success": True,
            "qty": 100, "estimated_price": 3000.0, "atr20": 80.0,
            "reason": "BUY[フジコ法]: RSR=80.0", "sector": "輸送用機器",
            "strategy_type": "fujiko",
        }]
        bridge.update_state_after_execution(send_results, today_str="2026-07-08")
        self.assertEqual(saved["position_entry_prices"]["7203.T"], 3000.0)
        self.assertEqual(saved["position_highest_closes"]["7203.T"], 3000.0)
        self.assertEqual(saved["position_entry_atrs"]["7203.T"], 80.0)
        self.assertNotIn("7203.T", saved.get("entry_metadata_missing", {}))

    def test_trend_follow_buy_records_entry_price_and_atr(self):
        bridge, saved = _make_bridge(self.SignalBridge)
        send_results = [{
            "symbol": "6506.T", "side": "BUY", "success": True,
            "qty": 100, "estimated_price": 7449.0, "atr20": 454.45,
            "reason": "trend_follow fallback=False", "sector": "機械",
            "strategy_type": "trend_follow",
        }]
        bridge.update_state_after_execution(send_results, today_str="2026-07-07")
        self.assertEqual(saved["position_entry_prices"]["6506.T"], 7449.0)
        self.assertEqual(saved["position_entry_atrs"]["6506.T"], 454.45)
        self.assertEqual(saved["position_strategy_types"]["6506.T"], "trend_follow")


class TestFailClosedOnMissingPrice(unittest.TestCase):
    """process-isolated経路のようにestimated_priceが無いsend_resultを模擬。"""

    def setUp(self):
        self.SignalBridge = _import_bridge_symbols()

    def test_missing_price_does_not_write_zero(self):
        """broker recovery も失敗する場合、0.0を書き込まずentry_metadata_missingに記録する。"""
        client = MagicMock()
        client.get_positions.return_value = []  # ブローカー側にも情報なし
        bridge, saved = _make_bridge(self.SignalBridge, client=client)
        bridge.universe_tickers = {"5301.T": "化学"}
        # 「ブローカーにも情報が無い」を明示的に模擬する（spec MagicMockの
        # 未設定属性はMagicMockの魔法メソッド既定値で偽の非ゼロ値を返しうるため）。
        bridge._get_current_positions = MagicMock(return_value={})

        # broker_worker.py の最小スキーマを模擬（estimated_price/atr20/reason 無し）
        send_results = [{
            "symbol": "5301.T", "side": "BUY", "success": True,
            "qty": 300, "client_order_id": "COI-X",
        }]
        bridge.update_state_after_execution(send_results, today_str="2026-07-07")

        self.assertNotIn("5301.T", saved["position_entry_prices"],
                          "0.0をentry_priceとして書き込んではいけない")
        self.assertNotIn("5301.T", saved["position_highest_closes"])
        self.assertIn("5301.T", saved.get("entry_metadata_missing", {}),
                       "復元不可の場合は監査用レジストリに記録する")
        # 自動売買は継続する想定 — 例外を投げずに保存まで完了していること
        self.assertTrue(bridge._save_portfolio_state.called)

    def test_missing_price_recovered_from_broker_avg_price(self):
        """send_resultにestimated_priceが無くても、broker実avg_priceがあれば復元する。"""
        client = MagicMock()
        client.get_positions.return_value = [
            {"Symbol": "5301", "LeavesQty": 300, "Price": 1758.5},
        ]
        bridge, saved = _make_bridge(self.SignalBridge, client=client)
        bridge.universe_tickers = {"5301.T": "化学"}
        bridge.live = False
        bridge._positions_api_status = {}
        bridge._get_current_positions = self.SignalBridge._get_current_positions.__get__(
            bridge, type(bridge)
        )

        send_results = [{
            "symbol": "5301.T", "side": "BUY", "success": True,
            "qty": 300, "client_order_id": "COI-X",
        }]
        with patch("src.common.position_normalizer.filter_live_positions",
                   return_value=[{"Symbol": "5301", "LeavesQty": 300, "Price": 1758.5}]):
            bridge.update_state_after_execution(send_results, today_str="2026-07-07")

        self.assertEqual(saved["position_entry_prices"]["5301.T"], 1758.5)
        self.assertNotIn("5301.T", saved.get("entry_metadata_missing", {}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
