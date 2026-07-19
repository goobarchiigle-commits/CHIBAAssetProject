"""
src/live/test_entry_metadata_recovery.py
entry_metadata_recovery.py の回帰テスト（2026-07-15全面改訂）。

背景: 2026-07-14/15 RCAで、旧実装がentry_dateを一度も書き込まない設計バグを
持っていたことが判明（signal_bridge.pyの時間ストップ判定を無効化していた）。
本テストは4ソース横断検索・entry_date伝播・detected_at保持・execution_quality
のBUY/SELL誤判定除外を検証する。

実行:
    python -m pytest src/live/test_entry_metadata_recovery.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.live.entry_metadata_recovery import (
    _find_from_execution_quality,
    _find_from_executed_signals,
    _find_from_logs_live_orders,
    _find_from_order_journal,
    recover_missing_entry_metadata,
)


class TestFindFromOrderJournal(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orders_dir = Path(self._tmpdir.name) / "orders"
        (self._orders_dir / "ack").mkdir(parents=True)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_finds_buy_with_created_at_date(self):
        rec = {
            "symbol": "5301.T", "side": "SHADOW_BUY", "qty": 300, "price": 1758.5,
            "created_at": "2026-07-07T09:00:06+0900",
        }
        (self._orders_dir / "ack" / "20260707_084405_5301.T_SHADOW_BUY.json").write_text(
            json.dumps(rec), encoding="utf-8"
        )
        result = _find_from_order_journal(self._orders_dir, "5301.T")
        self.assertIsNotNone(result)
        self.assertEqual(result["entry_date"], "2026-07-07")
        self.assertEqual(result["estimated_price"], 1758.5)
        self.assertEqual(result["confidence"], "high")

    def test_ignores_sell_side(self):
        rec = {
            "symbol": "5301.T", "side": "SELL", "qty": 300, "price": 1758.5,
            "created_at": "2026-07-07T09:00:06+0900",
        }
        (self._orders_dir / "ack" / "20260707_084405_5301.T_SELL.json").write_text(
            json.dumps(rec), encoding="utf-8"
        )
        result = _find_from_order_journal(self._orders_dir, "5301.T")
        self.assertIsNone(result)

    def test_returns_none_when_no_match(self):
        result = _find_from_order_journal(self._orders_dir, "9999.T")
        self.assertIsNone(result)


class TestFindFromLogsLiveOrders(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_extracts_date_from_filename_and_atr_from_order(self):
        data = [{"orders": [
            {"symbol": "6506.T", "side": "BUY", "estimated_price": 7449.0, "atr20": 454.45},
        ]}]
        (self._dir / "20260707_084405_orders.json").write_text(json.dumps(data), encoding="utf-8")
        result = _find_from_logs_live_orders(self._dir, "6506.T")
        self.assertIsNotNone(result)
        self.assertEqual(result["entry_date"], "2026-07-07")
        self.assertEqual(result["atr20"], 454.45)

    def test_picks_most_recent_when_multiple_files_match(self):
        old = [{"orders": [{"symbol": "6506.T", "side": "BUY", "estimated_price": 6000.0, "atr20": 300.0}]}]
        new = [{"orders": [{"symbol": "6506.T", "side": "BUY", "estimated_price": 7449.0, "atr20": 454.45}]}]
        (self._dir / "20260513_084405_orders.json").write_text(json.dumps(old), encoding="utf-8")
        (self._dir / "20260707_084405_orders.json").write_text(json.dumps(new), encoding="utf-8")
        result = _find_from_logs_live_orders(self._dir, "6506.T")
        self.assertEqual(result["entry_date"], "2026-07-07")
        self.assertEqual(result["estimated_price"], 7449.0)


class TestFindFromExecutedSignals(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_finds_from_send_results(self):
        data = {"send_results": [{
            "symbol": "6981.T", "side": "BUY", "success": True,
            "estimated_price": 4864.0, "atr20": 201.85,
            "order_submit_time": "2026-04-28T13:00:15+0900",
        }]}
        (self._dir / "signal_20260428_130016_executed.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = _find_from_executed_signals(self._dir, "6981.T")
        self.assertIsNotNone(result)
        self.assertEqual(result["entry_date"], "2026-04-28")
        self.assertEqual(result["estimated_price"], 4864.0)
        self.assertEqual(result["atr20"], 201.85)

    def test_ignores_failed_send_result(self):
        data = {"send_results": [{
            "symbol": "6981.T", "side": "BUY", "success": False,
            "estimated_price": 4864.0, "order_submit_time": "2026-04-28T13:00:15+0900",
        }]}
        (self._dir / "signal_20260428_130016_executed.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = _find_from_executed_signals(self._dir, "6981.T")
        self.assertIsNone(result)


class TestFindFromExecutionQualityAmbiguousSide(unittest.TestCase):
    """execution_quality.jsonlはside欠如のためBUY/SELL両方が混在する低信頼度ソース。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_excludes_known_sell_date(self):
        """既知のSELL日付（data/signals側で確認済み）は採用しないこと。"""
        rec = {
            "symbol": "6981.T", "entry_signal_time": "2026-07-14T08:41:05+0900",
            "planned_entry_price": 9066.0, "fill_status": "submitted",
        }
        (self._dir / "20260714.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        result = _find_from_execution_quality(self._dir, "6981.T", known_sell_dates={"2026-07-14"})
        self.assertIsNone(result)

    def test_accepts_non_sell_date_as_low_confidence(self):
        rec = {
            "symbol": "6981.T", "entry_signal_time": "2026-04-28T13:00:06+0900",
            "planned_entry_price": 4936.0, "fill_status": "submitted",
        }
        (self._dir / "20260428.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        result = _find_from_execution_quality(self._dir, "6981.T", known_sell_dates=set())
        self.assertIsNotNone(result)
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["entry_date"], "2026-04-28")


class TestRecoverMissingEntryMetadataIntegration(unittest.TestCase):
    """recover_missing_entry_metadata()の統合テスト（実際のportfolio_state dict操作）。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._logs_live_dir     = self._root / "logs_live"
        self._orders_dir        = self._root / "orders"
        self._signals_dir       = self._root / "signals"
        self._exec_quality_dir  = self._root / "exec_quality"
        self._audit_log         = self._root / "audit.jsonl"
        for d in (self._logs_live_dir, self._orders_dir, self._signals_dir, self._exec_quality_dir):
            d.mkdir(parents=True)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_state(self, **overrides):
        state = {
            "position_qtys":           {"6981.T": 100},
            "position_entry_prices":   {},
            "position_entry_atrs":     {},
            "position_highest_closes": {},
            "position_entry_dates":    {},
            "position_strategy_types": {},
            "entry_metadata_missing":  {},
        }
        state.update(overrides)
        return state

    def test_sets_entry_date_on_successful_recovery(self):
        """コア回帰: 復元成功時にposition_entry_datesへ必ず書き込まれること
        （旧実装は価格/ATRを復元してもentry_dateを書かない設計バグがあった）。"""
        data = {"send_results": [{
            "symbol": "6981.T", "side": "BUY", "success": True,
            "estimated_price": 4864.0, "atr20": 201.85,
            "order_submit_time": "2026-04-28T13:00:15+0900",
        }]}
        (self._signals_dir / "signal_20260428_130016_executed.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        state = self._make_state()
        result = recover_missing_entry_metadata(
            state, logs_live_dir=self._logs_live_dir, audit_log_path=self._audit_log,
            held_positions=state["position_qtys"],
            orders_dir=self._orders_dir, signals_dir=self._signals_dir,
            exec_quality_dir=self._exec_quality_dir,
        )
        self.assertEqual(len(result["recovered"]), 1)
        self.assertEqual(state["position_entry_dates"]["6981.T"], "2026-04-28")
        self.assertEqual(state["position_entry_prices"]["6981.T"], 4864.0)
        self.assertEqual(state["position_entry_atrs"]["6981.T"], 201.85)
        self.assertNotIn("6981.T", state["entry_metadata_missing"])

    def test_detected_at_preserved_across_repeated_failed_attempts(self):
        """Phase4: 復元不能な銘柄について、detected_atが毎日上書きされないこと。"""
        state = self._make_state(entry_metadata_missing={
            "6981.T": {"detected_at": "2026-07-01", "entry_date": "", "qty": 100,
                       "reason": "no_matching_buy_record_in_any_source", "recovery_attempts": 3},
        })
        result = recover_missing_entry_metadata(
            state, logs_live_dir=self._logs_live_dir, audit_log_path=self._audit_log,
            held_positions=state["position_qtys"],
            orders_dir=self._orders_dir, signals_dir=self._signals_dir,
            exec_quality_dir=self._exec_quality_dir,
        )
        self.assertEqual(len(result["unrecoverable"]), 1)
        self.assertEqual(state["entry_metadata_missing"]["6981.T"]["detected_at"], "2026-07-01")
        self.assertEqual(state["entry_metadata_missing"]["6981.T"]["recovery_attempts"], 4)

    def test_unrecoverable_when_no_source_matches(self):
        state = self._make_state()
        result = recover_missing_entry_metadata(
            state, logs_live_dir=self._logs_live_dir, audit_log_path=self._audit_log,
            held_positions=state["position_qtys"],
            orders_dir=self._orders_dir, signals_dir=self._signals_dir,
            exec_quality_dir=self._exec_quality_dir,
        )
        self.assertEqual(len(result["unrecoverable"]), 1)
        self.assertIn("6981.T", state["entry_metadata_missing"])
        self.assertEqual(state["entry_metadata_missing"]["6981.T"]["entry_date"], "")

    def test_already_complete_position_skipped(self):
        """entry_date/price/atr全て揃っている銘柄は何も変更しないこと。"""
        state = self._make_state(
            position_entry_dates={"6981.T": "2026-04-28"},
            position_entry_prices={"6981.T": 4864.0},
            position_entry_atrs={"6981.T": 201.85},
        )
        result = recover_missing_entry_metadata(
            state, logs_live_dir=self._logs_live_dir, audit_log_path=self._audit_log,
            held_positions=state["position_qtys"],
            orders_dir=self._orders_dir, signals_dir=self._signals_dir,
            exec_quality_dir=self._exec_quality_dir,
        )
        self.assertEqual(result["recovered"], [])
        self.assertEqual(result["unrecoverable"], [])
        self.assertEqual(state["position_entry_dates"]["6981.T"], "2026-04-28")


if __name__ == "__main__":
    unittest.main()
