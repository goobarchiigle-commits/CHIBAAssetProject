"""
tests/test_cb_guard_compensation.py

P0 CB_GUARD_COMP バグ修正の検証テスト（position_qtys書込み部分のみ現存）。

根本原因（2026-06-04インシデント）:
  update_state_after_execution がBUY約定後に position_qtys を書かなかった。

修正: BUYブロックに pos_qtys[sym] = int(qty) を追加 (SELL は pop)。
この部分（TestPositionQtysWrittenOnBuy）は独立した価値があるため現存する。

2026-07-19 追記: 上記バグが誘発したCB発動抑制機構（SOURCE1/_cb_guard_compensation
経由のPendingOrderState）自体は、stale portfolio_stateとの区別が原理的に不可能な
構造的欠陥（実機検証で5日前に確定売却済みの3銘柄を「決済待ち」と誤認し
¥1,748,850を誤算出）と、SOURCE2(ledger-based)がrun_live_signal.pyから
呼ばれず実質死コードだった事実が判明したため、Broker-as-Sole-SSOTの方針に
従い完全撤去した。旧`TestCbGuardCompensationSource1`/`TestCbNotTriggeredAfterCompensation`
（SOURCE1の算出値そのものを検証していたテスト）は、検証対象の関数
`_cb_guard_compensation()`が削除されたため本ファイルから削除した。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_bridge(state_file: Path):
    from src.kabusapi.signal_bridge import SignalBridge
    bridge = SignalBridge.__new__(SignalBridge)
    bridge._state_file = state_file
    bridge.capital = 3_000_000
    bridge._client = None
    bridge._positions_api_status = {}  # _save_portfolio_state が参照する
    return bridge


def _initial_state(**overrides) -> dict:
    base = {
        "cb_state":               "NORMAL",
        "equity_peak":            3_000_000,
        "cb_cooldown_end_date":   None,
        "recovery_threshold":     None,
        "position_entry_dates":   {},
        "position_entry_prices":  {},
        "position_entry_atrs":    {},
        "position_highest_closes": {},
        "position_qtys":          {},
        "reentry_blocked":        {},
        "last_updated":           None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# テスト 1: update_state_after_execution が position_qtys を書くか
# ---------------------------------------------------------------------------

class TestPositionQtysWrittenOnBuy(unittest.TestCase):
    """Fix 検証: BUY後に position_qtys[sym] が即時保存されるか。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_file = Path(self._tmp.name) / "portfolio_state.json"
        self.state_file.write_text(json.dumps(_initial_state()), encoding="utf-8")
        self.bridge = _make_bridge(self.state_file)

    def tearDown(self):
        self._tmp.cleanup()

    def test_buy_writes_position_qtys(self):
        """BUY成功 → position_qtys に qty が書かれる。"""
        send_results = [{
            "symbol":          "5301.T",
            "side":            "BUY",
            "qty":             400,
            "estimated_price": 1852.5,
            "atr20":           45.0,
            "sector":          "化学",
            "reason":          "turtle_entry",
            "strategy_type":   "fujiko",
            "order_id":        "20260604A01N00000001",
            "success":         True,
            "result_code":     0,
        }]
        self.bridge.update_state_after_execution(send_results, "2026-06-04")

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertIn("5301.T", state["position_qtys"],
                      "position_qtys に 5301.T が存在しない (BUG: 書き込み欠落)")
        self.assertEqual(state["position_qtys"]["5301.T"], 400)

    def test_buy_also_writes_entry_price(self):
        """従来の entry_prices / entry_dates も正常に書かれるか（リグレッション）。"""
        send_results = [{
            "symbol":          "5301.T",
            "side":            "BUY",
            "qty":             400,
            "estimated_price": 1852.5,
            "atr20":           45.0,
            "sector":          "化学",
            "reason":          "turtle_entry",
            "strategy_type":   "fujiko",
            "order_id":        "20260604A01N00000001",
            "success":         True,
            "result_code":     0,
        }]
        self.bridge.update_state_after_execution(send_results, "2026-06-04")

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["position_entry_prices"]["5301.T"], 1852.5)
        self.assertEqual(state["position_entry_dates"]["5301.T"], "2026-06-04")

    def test_sell_removes_position_qtys(self):
        """SELL後は position_qtys からシンボルが除去される。"""
        init = _initial_state(
            position_qtys         = {"5301.T": 400},
            position_entry_prices = {"5301.T": 1852.5},
            position_entry_dates  = {"5301.T": "2026-06-04"},
            position_highest_closes = {"5301.T": 1852.5},
        )
        self.state_file.write_text(json.dumps(init), encoding="utf-8")
        bridge = _make_bridge(self.state_file)

        send_results = [{
            "symbol":          "5301.T",
            "side":            "SELL",
            "qty":             400,
            "estimated_price": 1900.0,
            "atr20":           0.0,
            "sector":          "化学",
            "reason":          "turtle_exit",
            "strategy_type":   "",
            "order_id":        "20260620A01N00000001",
            "success":         True,
            "result_code":     0,
        }]
        bridge.update_state_after_execution(send_results, "2026-06-20")

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertNotIn("5301.T", state.get("position_qtys", {}),
                         "SELL後も position_qtys に残存している")

    def test_failed_order_does_not_write_position_qtys(self):
        """success=False の場合は position_qtys を変更しない。"""
        send_results = [{
            "symbol":          "5301.T",
            "side":            "BUY",
            "qty":             400,
            "estimated_price": 1852.5,
            "atr20":           45.0,
            "sector":          "化学",
            "reason":          "turtle_entry",
            "strategy_type":   "fujiko",
            "order_id":        "",
            "success":         False,
            "result_code":     4001013,
        }]
        self.bridge.update_state_after_execution(send_results, "2026-06-04")

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertNotIn("5301.T", state.get("position_qtys", {}))


if __name__ == "__main__":
    unittest.main()
