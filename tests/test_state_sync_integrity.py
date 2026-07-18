"""
tests/test_state_sync_integrity.py
STATE_SYNC_INTEGRITY (2026-07-19) の回帰テスト。

背景: broker が get_positions() に成功し「真の保有0件」を返した場合でも、
旧実装は current_positions が空であることだけを見て無条件に
portfolio_state["position_qtys"]（前回runの永続化値）へフォールバックして
いた。この経路には _apply_position_missing_streak_gate() のような
decay保護（MAX_POSITION_MISSING_STREAK）が一切なく、brokerが繰り返し
「保有0」を正しく返し続ける限り無期限にstateの古いpositionsで
上書きされ続ける構造的欠陥だった（2026-07-17 equity_peak異常値インシデントの
調査中に発見）。

対象:
  - broker成功・真の保有0 → フォールバックしない（brokerをSSOTとして信頼）
  - broker失敗/未接続 → 従来通りstateへフォールバック
  - current_positionsが非空 → 何もしない（素通し）
  - stateにposition_qtysが無い → フォールバックしようがない
"""
import unittest

from src.kabusapi.signal_bridge import _apply_full_empty_fallback_gate


class TestFullEmptyFallbackGate(unittest.TestCase):

    def test_broker_success_with_true_zero_positions_does_not_fallback(self):
        """brokerがget_positions()に成功し保有0件を返した場合、
        stateに古いpositionsが残っていてもフォールバックしないこと（中核テスト）。"""
        result = _apply_full_empty_fallback_gate(
            {},  # current_positions（broker成功・空）
            broker_positions_ok=True,
            saved_qtys={"5301.T": 300, "6506.T": 100, "6981.T": 100},
            saved_prices={"5301.T": 1758.5, "6506.T": 7349.0, "6981.T": 4864.0},
        )
        self.assertEqual(result, {}, "brokerの「真の保有0」はSSOTとして信頼され、state復元が起きてはならない")

    def test_broker_failure_falls_back_to_state(self):
        """broker API呼び出し自体が失敗/未接続の場合は、従来通りstateへフォールバックすること。"""
        result = _apply_full_empty_fallback_gate(
            {},
            broker_positions_ok=False,
            saved_qtys={"5301.T": 300, "6981.T": 100},
            saved_prices={"5301.T": 1758.5, "6981.T": 4864.0},
        )
        self.assertEqual(result, {
            "5301.T": {"qty": 300, "avg_price": 1758.5},
            "6981.T": {"qty": 100, "avg_price": 4864.0},
        })

    def test_non_empty_current_positions_passthrough(self):
        """current_positionsが既に非空なら、broker成否に関わらず何もしないこと。"""
        given = {"7203.T": {"qty": 100, "avg_price": 2500.0}}
        result = _apply_full_empty_fallback_gate(
            given, broker_positions_ok=True,
            saved_qtys={"5301.T": 300}, saved_prices={"5301.T": 1758.5},
        )
        self.assertEqual(result, given)

    def test_no_saved_state_nothing_to_restore(self):
        """stateにposition_qtysが無ければ、broker失敗時でも復元しようがなく空のまま。"""
        result = _apply_full_empty_fallback_gate(
            {}, broker_positions_ok=False, saved_qtys={}, saved_prices={},
        )
        self.assertEqual(result, {})

    def test_zero_qty_saved_entries_excluded_from_fallback(self):
        """フォールバック時、qty<=0のstateエントリは復元対象から除外されること。"""
        result = _apply_full_empty_fallback_gate(
            {}, broker_positions_ok=False,
            saved_qtys={"5301.T": 300, "6506.T": 0}, saved_prices={"5301.T": 1758.5, "6506.T": 7349.0},
        )
        self.assertEqual(result, {"5301.T": {"qty": 300, "avg_price": 1758.5}})

    def test_broker_success_zero_positions_no_saved_state_noop(self):
        """broker成功・保有0・stateも空の通常ケースでは何も起きないこと。"""
        result = _apply_full_empty_fallback_gate(
            {}, broker_positions_ok=True, saved_qtys={}, saved_prices={},
        )
        self.assertEqual(result, {})

    def test_pure_function_does_not_mutate_input(self):
        """current_positions/saved_qtysを破壊的に変更しないこと。"""
        current = {}
        saved = {"5301.T": 300}
        _apply_full_empty_fallback_gate(
            current, broker_positions_ok=False, saved_qtys=saved, saved_prices={"5301.T": 1758.5},
        )
        self.assertEqual(current, {})
        self.assertEqual(saved, {"5301.T": 300})


if __name__ == "__main__":
    unittest.main()
