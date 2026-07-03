"""
tests/test_state_update.py
P0-2: 発注結果に応じた portfolio_state 更新のテスト
"""
import sys
import json
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _make_bridge(tmp_state_file: Path):
    """テスト用の SignalBridge を最小設定で生成する（API 接続なし）。"""
    from src.kabusapi.signal_bridge import SignalBridge
    bridge = SignalBridge.__new__(SignalBridge)
    bridge._state_file = tmp_state_file
    bridge.capital = 3_000_000
    bridge._client = None
    return bridge


_DEFAULT_STATE = {
    "cb_state": "NORMAL",
    "equity_peak": 3_000_000,
    "cb_cooldown_end_date": None,
    "recovery_threshold": None,
    "position_entry_dates": {},
    "position_entry_prices": {},
    "position_entry_atrs": {},
    "position_highest_closes": {},
    "reentry_blocked": {},
    "last_updated": None,
}


class TestStateUpdateAfterExecution(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.state_file = Path(self._tmpdir.name) / "portfolio_state.json"
        self.state_file.write_text(
            json.dumps(_DEFAULT_STATE), encoding="utf-8"
        )
        self.bridge = _make_bridge(self.state_file)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_state_updated_when_order_id_present(self):
        """order_id が取得できた場合、portfolio_state のエントリーが更新される"""
        send_results = [{
            "symbol": "7203.T",
            "side": "BUY",
            "qty": 100,
            "estimated_price": 3000.0,
            "atr20": 80.0,
            "sector": "輸送用機器",
            "reason": "turtle_entry",
            "strategy_type": "fujiko",
            "order_id": "20260407A01N12345678",
            "success": True,
            "result_code": 0,
        }]
        self.bridge.update_state_after_execution(send_results, "2026-04-07")

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertIn("7203.T", state["position_entry_dates"])
        self.assertEqual(state["position_entry_dates"]["7203.T"], "2026-04-07")
        self.assertEqual(state["position_entry_prices"]["7203.T"], 3000.0)

    def test_state_not_updated_when_genuinely_failed(self):
        """success=False の場合は portfolio_state を更新しない"""
        send_results = [{
            "symbol": "7203.T",
            "side": "BUY",
            "qty": 100,
            "estimated_price": 3000.0,
            "atr20": 80.0,
            "sector": "輸送用機器",
            "reason": "turtle_entry",
            "strategy_type": "fujiko",
            "order_id": "",
            "success": False,
            "result_code": 4001013,
        }]
        self.bridge.update_state_after_execution(send_results, "2026-04-07")

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertNotIn("7203.T", state["position_entry_dates"])


if __name__ == "__main__":
    unittest.main()
