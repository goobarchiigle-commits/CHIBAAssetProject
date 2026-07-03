"""
tests/test_ghost_position_fix.py
GHOST_POSITION_FIX (2026-07-03) の回帰テスト。

実インシデント: 5301.T が実ブローカーで売却済みにも関わらず、無条件の
「部分補完」ロジックが毎run portfolio_state から復活させ続け、
compute_live_equity() の market_value を ¥595,550 過大評価 →
誤 equity_peak (¥4,706,291、実際は一度も到達していない) が永続化された。

対象: _apply_position_missing_streak_gate()
  (src/kabusapi/signal_bridge.py, run() の「部分補完」ブロックから切り出した純粋関数)
"""
import unittest

from src.kabusapi.signal_bridge import (
    MAX_POSITION_MISSING_STREAK,
    _apply_position_missing_streak_gate,
)


class TestMissingStreakGate(unittest.TestCase):
    def test_broker_api_failure_never_supplements(self):
        """broker_positions_ok=False のときは一切補完しない（broker失敗を売却と誤判定しない）。"""
        current_positions, streak = _apply_position_missing_streak_gate(
            {}, {"5301.T": 300}, {"5301.T": 1801.0}, {},
            broker_positions_ok=False,
        )
        self.assertEqual(current_positions, {})
        self.assertEqual(streak, {})

    def test_first_miss_supplements_and_increments_streak(self):
        """初回欠落は補完され、streakが1になる。"""
        current_positions, streak = _apply_position_missing_streak_gate(
            {"6981.T": {"qty": 100, "avg_price": 4864.0}},
            {"6981.T": 100, "5301.T": 300},
            {"5301.T": 1801.0},
            {},
            broker_positions_ok=True,
        )
        self.assertIn("5301.T", current_positions)
        self.assertEqual(current_positions["5301.T"]["qty"], 300)
        self.assertEqual(streak["5301.T"], 1)

    def test_streak_accumulates_across_calls_up_to_threshold(self):
        """MAX_POSITION_MISSING_STREAK 未満の間は補完され続ける。"""
        streak = {}
        current_positions = {}
        for _ in range(MAX_POSITION_MISSING_STREAK):
            current_positions, streak = _apply_position_missing_streak_gate(
                {}, {"5301.T": 300}, {"5301.T": 1801.0}, streak,
                broker_positions_ok=True,
            )
            self.assertIn("5301.T", current_positions)
        self.assertEqual(streak["5301.T"], MAX_POSITION_MISSING_STREAK)

    def test_streak_exceeds_threshold_stops_supplementing_and_clears_entry(self):
        """
        しきい値到達後は補完を停止し（=幽霊ポジション永久復活を防止）、
        streak エントリ自体もクリアされる（次回 commit_broker_snapshot で
        position_qtys から完全除去される前提の設計）。
        """
        streak = {"5301.T": MAX_POSITION_MISSING_STREAK}
        current_positions, streak_after = _apply_position_missing_streak_gate(
            {}, {"5301.T": 300}, {"5301.T": 1801.0}, streak,
            broker_positions_ok=True,
        )
        self.assertNotIn("5301.T", current_positions)  # 補完されない
        self.assertNotIn("5301.T", streak_after)        # streak自体もクリア

    def test_symbol_found_by_broker_resets_streak(self):
        """broker が銘柄を再度返却したら streak はリセットされる。"""
        current_positions, streak = _apply_position_missing_streak_gate(
            {"5301.T": {"qty": 300, "avg_price": 1801.0}},  # broker が今回返却
            {"5301.T": 300},
            {"5301.T": 1801.0},
            {"5301.T": 1},  # 前回1回欠落していた
            broker_positions_ok=True,
        )
        self.assertNotIn("5301.T", streak)  # リセットされ、追跡対象から外れる

    def test_zero_qty_saved_position_never_supplemented(self):
        current_positions, streak = _apply_position_missing_streak_gate(
            {}, {"5301.T": 0}, {"5301.T": 1801.0}, {},
            broker_positions_ok=True,
        )
        self.assertEqual(current_positions, {})
        self.assertEqual(streak, {})


if __name__ == "__main__":
    unittest.main()
