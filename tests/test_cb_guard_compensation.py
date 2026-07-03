"""
tests/test_cb_guard_compensation.py

P0 CB_GUARD_COMP バグ修正の検証テスト。

根本原因:
  update_state_after_execution がBUY約定後に position_qtys を書かなかった。
  → _cb_pre_qtys に新規BUYシンボルが含まれず SOURCE1 補償がゼロになった。
  → T+2決済ラグで equity が急落し、誤った DD=-22% でCB_ACTIVEが発動した。

修正: BUYブロックに pos_qtys[sym] = int(qty) を追加 (SELL は pop)。

再現ケース (2026-06-04):
  5301.T BUY 400株 @ ¥1852.5 → 運用資金¥741,000が決済ラグで消える
  equity=3,184,309 / peak=4,089,109 → 補償後 cb_equity=3,925,309 → DD=-4.01%
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


# ---------------------------------------------------------------------------
# テスト 2: _cb_guard_compensation SOURCE1 の直接検証
# ---------------------------------------------------------------------------

class TestCbGuardCompensationSource1(unittest.TestCase):
    """_cb_guard_compensation の SOURCE1 (state-based) 補償ロジックを直接検証。"""

    def _run(self, pre_commit_qtys, pre_commit_costs, current_positions,
             current_equity=3_184_309.0, ledger_path=None, today="2026-06-04"):
        from src.kabusapi.signal_bridge import _cb_guard_compensation
        return _cb_guard_compensation(
            current_equity    = current_equity,
            current_positions = current_positions,
            pre_commit_qtys   = pre_commit_qtys,
            pre_commit_costs  = pre_commit_costs,
            ledger_path       = ledger_path,
            today_str         = today,
        )

    def test_source1_compensates_settlement_lag(self):
        """
        2026-06-04 再現ケース:
        5301.T が position_qtys にあり (fix後) ブローカーには未着 → SOURCE1 = ¥741,000。
        """
        state_comp, state_syms, ledger_comp, ledger_syms = self._run(
            pre_commit_qtys  = {"6506.T": 100, "6981.T": 100, "5301.T": 400},
            pre_commit_costs = {"6506.T": 6987.0, "6981.T": 4864.0, "5301.T": 1852.5},
            current_positions = {
                "6506.T": {"qty": 100, "avg_price": 6987.0},
                "6981.T": {"qty": 100, "avg_price": 4864.0},
                # 5301.T: T+2決済ラグで未着
            },
        )
        self.assertEqual(state_comp, 400 * 1852.5,
                         f"SOURCE1 expected ¥741,000 got ¥{state_comp}")
        self.assertIn("5301.T", state_syms)
        self.assertEqual(ledger_comp, 0.0, "SOURCE2 should be zero (no ledger)")

    def test_no_compensation_when_position_already_in_broker(self):
        """ブローカーにポジションが既に着荷している場合は補償なし。"""
        state_comp, state_syms, ledger_comp, ledger_syms = self._run(
            pre_commit_qtys   = {"5301.T": 400},
            pre_commit_costs  = {"5301.T": 1852.5},
            current_positions = {"5301.T": {"qty": 400, "avg_price": 1852.5}},
        )
        self.assertEqual(state_comp, 0.0, "着荷済みシンボルを誤補償してはならない")
        self.assertNotIn("5301.T", state_syms)

    def test_no_compensation_when_pre_qtys_empty(self):
        """
        BUG再現: fix前の状態（position_qtys に 5301.T が無い）では補償ゼロ。
        """
        state_comp, state_syms, ledger_comp, ledger_syms = self._run(
            pre_commit_qtys   = {"6506.T": 100, "6981.T": 100},  # 5301.T 欠落
            pre_commit_costs  = {"6506.T": 6987.0, "6981.T": 4864.0, "5301.T": 1852.5},
            current_positions = {
                "6506.T": {"qty": 100, "avg_price": 6987.0},
                "6981.T": {"qty": 100, "avg_price": 4864.0},
            },
        )
        self.assertEqual(state_comp, 0.0, "BUG再現: SOURCE1補償ゼロを確認")


# ---------------------------------------------------------------------------
# テスト 3: 6/4 シナリオ 完全統合 — CB_ACTIVE にならないことを証明
# ---------------------------------------------------------------------------

CB_DD_TRIGGER = 0.15  # signal_bridge.CB_DD_TRIGGER と同じ定数


class TestCbNotTriggeredAfterCompensation(unittest.TestCase):
    """
    修正後シナリオ:
      equity=3,184,309 / peak=4,089,109
      5301.T BUY 400 @ ¥1852.5 → position_qtys に記録 (fix)
      → SOURCE1 補償=¥741,000
      → cb_equity=3,925,309
      → DD=(3,925,309/4,089,109)-1 = -4.01%  → -15%以内 → CB_ACTIVE 発動しない
    """

    def test_cb_equity_calculation(self):
        """補償後 cb_equity の数値が正しいことを確認。"""
        from src.kabusapi.signal_bridge import _cb_guard_compensation

        current_equity = 3_184_309.0
        equity_peak    = 4_089_109.0

        state_comp, _, ledger_comp, _ = _cb_guard_compensation(
            current_equity    = current_equity,
            current_positions = {
                "6506.T": {"qty": 100, "avg_price": 6987.0},
                "6981.T": {"qty": 100, "avg_price": 4864.0},
            },
            pre_commit_qtys  = {"6506.T": 100, "6981.T": 100, "5301.T": 400},
            pre_commit_costs = {"6506.T": 6987.0, "6981.T": 4864.0, "5301.T": 1852.5},
            ledger_path      = None,
            today_str        = "2026-06-04",
        )

        comp_total = state_comp + ledger_comp
        cb_equity  = current_equity + comp_total
        dd_adjusted = cb_equity / equity_peak - 1.0

        self.assertAlmostEqual(comp_total, 741_000.0, places=0,
                               msg="comp_total が ¥741,000 でない")
        self.assertAlmostEqual(cb_equity, 3_925_309.0, places=0,
                               msg="cb_equity が ¥3,925,309 でない")
        self.assertGreater(dd_adjusted, -CB_DD_TRIGGER,
                           msg=f"DD={dd_adjusted:.4f} が閾値 -{CB_DD_TRIGGER} 以下 → 誤CB発動")

    def test_dd_raw_would_have_triggered_cb(self):
        """補償なしの生 DD は -15% を超えること（バグ再現の確認）。"""
        current_equity = 3_184_309.0
        equity_peak    = 4_089_109.0
        dd_raw = current_equity / equity_peak - 1.0
        self.assertLessEqual(dd_raw, -CB_DD_TRIGGER,
                             msg="補償なしDDは -15% 以下のはず (バグ再現失敗)")

    def test_comp_source1_label(self):
        """SOURCE1 補償が ¥741,000 であることを明示的に検証。"""
        from src.kabusapi.signal_bridge import _cb_guard_compensation

        state_comp, state_syms, ledger_comp, ledger_syms = _cb_guard_compensation(
            current_equity    = 3_184_309.0,
            current_positions = {
                "6506.T": {"qty": 100, "avg_price": 6987.0},
                "6981.T": {"qty": 100, "avg_price": 4864.0},
            },
            pre_commit_qtys  = {"6506.T": 100, "6981.T": 100, "5301.T": 400},
            pre_commit_costs = {"6506.T": 6987.0, "6981.T": 4864.0, "5301.T": 1852.5},
            ledger_path      = None,
            today_str        = "2026-06-04",
        )

        self.assertEqual(state_comp,   400 * 1852.5, "comp_source1 ≠ ¥741,000")
        self.assertEqual(ledger_comp,  0.0,          "comp_source2 ≠ ¥0")
        self.assertEqual(state_comp + ledger_comp, 741_000.0, "comp_total ≠ ¥741,000")
        self.assertIn("5301.T", state_syms)
        self.assertEqual(ledger_syms, [])


if __name__ == "__main__":
    unittest.main()
