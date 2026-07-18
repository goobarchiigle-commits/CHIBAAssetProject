"""
src/test_startup_check.py
startup_check._compute_startup_equity() の回帰テスト（2026-07-15 SSOT修正）。

背景: 2026-07-14 08:41 incidentのRCAで、旧実装が state ファイルの
available_cash（stale）を無条件に使い、DD=-21.9%という誤警告を出したことが
判明した（実際のDDは-10.2%）。本テストは broker snapshot 優先ロジックと
FAIL_OPENフォールバックの両方を検証する。

実行:
    python -m pytest src/test_startup_check.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.startup_check import _compute_startup_equity
import src.portfolio.equity as _equity_mod


class _IsolatedSnapshotMixin:
    """_compute_startup_equity() は内部で compute_live_equity(persist_snapshot=True)
    を呼び出し、src.portfolio.equity._SNAPSHOT_FILE（logs/equity_snapshots.jsonl の
    実ファイル）へ書き込む。src.startup_check._BASE_DIR のパッチだけでは
    equity.py 側の独立した _SNAPSHOT_FILE 定数までは差し替わらず、本番ログを
    汚染していた（2026-07-17〜18に実際に混入が確認された）。全テストクラス共通で
    _SNAPSHOT_FILE を一時ファイルへリダイレクトし、本番ログへの書き込みを防ぐ。"""

    def setUp(self):
        super().setUp()
        import tempfile
        self._snap_tmpdir = tempfile.TemporaryDirectory()
        self._snap_patcher = patch.object(
            _equity_mod, "_SNAPSHOT_FILE", Path(self._snap_tmpdir.name) / "equity_snapshots.jsonl",
        )
        self._snap_patcher.start()

    def tearDown(self):
        self._snap_patcher.stop()
        self._snap_tmpdir.cleanup()
        super().tearDown()


def _base_state(**overrides) -> dict:
    state = {
        "equity_peak":     4_110_741.0,
        "available_cash":  3_210_391.0,   # 2026-07-14 incidentで実際に観測された stale 値
        "position_qtys":   {"5301.T": 300, "6506.T": 100, "6981.T": 100},
        "position_entry_prices": {"5301.T": 1741.0, "6506.T": 7349.0, "6981.T": 4864.0},
        "snapshot_avg_costs":    {"5301.T": 1741.0, "6506.T": 7349.0, "6981.T": 4864.0},
        "cb_state": "NORMAL",
    }
    state.update(overrides)
    return state


class TestComputeStartupEquityBrokerAvailable(_IsolatedSnapshotMixin, unittest.TestCase):
    """
    broker snapshot取得成功時: state file の stale available_cash を無視すること。

    実環境の cache/ohlcv/ に本物のOHLCVデータが存在すると compute_live_equity() が
    avg_price ではなく実勢終値を使う（本番同一の意図通りの挙動）ため、
    金額アサーションが環境依存にならないよう _BASE_DIR を空の一時dirへ差し替え、
    OHLCVキャッシュ不在（=avg_price使用）を強制して決定的にテストする。
    """

    def setUp(self):
        super().setUp()
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch("src.startup_check._BASE_DIR", Path(self._tmpdir.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()
        super().tearDown()

    @patch("src.kabusapi.client.KabuClient")
    def test_uses_broker_cash_not_stale_state_cash(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 1_706_591.0}
        mock_client.get_positions.return_value = [
            {"Symbol": "5301", "LeavesQty": 300, "Price": 1741.0},
            {"Symbol": "6506", "LeavesQty": 100, "Price": 7349.0},
            {"Symbol": "6981", "LeavesQty": 100, "Price": 4864.0},
        ]
        mock_client_cls.return_value = mock_client

        result = _compute_startup_equity(_base_state())

        self.assertTrue(result["broker_available"])
        self.assertEqual(result["equity_src"], "broker_snapshot")
        self.assertEqual(result["cash_used"], 1_706_591.0)
        # stale state cash (3,210,391) は使われていないこと
        self.assertNotEqual(result["cash_used"], 3_210_391.0)
        # OHLCVキャッシュ不在(一時空dir) → avg_price評価: cash + sum(qty*avg_price)
        # = 1,706,591 + (300*1741 + 100*7349 + 100*4864) = 1,706,591 + 1,743,600 = 3,450,191
        self.assertAlmostEqual(result["current_equity"], 3_450_191.0, delta=1.0)

    @patch("src.kabusapi.client.KabuClient")
    def test_dd_computed_from_broker_equity_not_stale_estimate(self, mock_client_cls):
        """2026-07-14 incident の回帰確認: 旧実装はDD=-21.9%を誤って出していた。
        broker snapshot 経由なら実態に近いDD（浅い側）になること。"""
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 1_706_591.0}
        mock_client.get_positions.return_value = [
            {"Symbol": "5301", "LeavesQty": 300, "Price": 1741.0},
            {"Symbol": "6506", "LeavesQty": 100, "Price": 7349.0},
            {"Symbol": "6981", "LeavesQty": 100, "Price": 4864.0},
        ]
        mock_client_cls.return_value = mock_client

        result = _compute_startup_equity(_base_state())
        # 旧実装のDD(-21.9%)より明確に浅い（stale cashを使っていない証拠）
        self.assertGreater(result["dd"], -21.9)

    @patch("src.kabusapi.client.KabuClient")
    def test_zero_qty_positions_excluded(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 1_000_000.0}
        mock_client.get_positions.return_value = [
            {"Symbol": "5301", "LeavesQty": 0, "Price": 1741.0},   # 売却済み・qty=0
            {"Symbol": "6506", "LeavesQty": 100, "Price": 7349.0},
        ]
        mock_client_cls.return_value = mock_client

        result = _compute_startup_equity(_base_state())
        self.assertTrue(result["broker_available"])
        # 1,000,000 + 100*7349 = 1,734,900
        self.assertAlmostEqual(result["current_equity"], 1_734_900.0, delta=1.0)


class TestComputeStartupEquityFailOpen(_IsolatedSnapshotMixin, unittest.TestCase):
    """broker snapshot取得失敗時: FAIL-CLOSED（Broker-as-Sole-SSOT, 2026-07-18）。

    旧実装は state ファイルの stale な available_cash/position_qtys へ FAIL_OPEN
    していたが、これが2026-07-15〜17 equity_peak異常値インシデントの一因だった
    （state実測より新しいbroker実態が食い違うケースを検知できなかった）。
    broker取得失敗時は current_equity/cash_used を計算せず（0.0のまま）、
    呼び出し側（run_startup_check）が ok=False として扱う。
    """

    @patch("src.kabusapi.client.KabuClient")
    def test_broker_exception_leaves_equity_unavailable(self, mock_client_cls):
        mock_client_cls.side_effect = ConnectionError("API unreachable")

        result = _compute_startup_equity(_base_state())

        self.assertFalse(result["broker_available"])
        self.assertEqual(result["cash_used"], 0.0)   # state file へのフォールバックは行わない
        self.assertEqual(result["current_equity"], 0.0)
        self.assertEqual(result["equity_src"], "unavailable")

    @patch("src.kabusapi.client.KabuClient")
    def test_missing_wallet_field_leaves_equity_unavailable(self, mock_client_cls):
        """StockAccountWallet キー欠損時もFAIL-CLOSEDすること（フォールバックしない）。"""
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"UnexpectedKey": 123}
        mock_client.get_positions.return_value = []
        mock_client_cls.return_value = mock_client

        result = _compute_startup_equity(_base_state())
        self.assertFalse(result["broker_available"])
        self.assertEqual(result["cash_used"], 0.0)

    @patch("src.kabusapi.client.KabuClient")
    def test_no_crash_with_no_positions_and_broker_available(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 3_000_000.0}
        mock_client.get_positions.return_value = []
        mock_client_cls.return_value = mock_client

        result = _compute_startup_equity(_base_state(position_qtys={}))
        self.assertTrue(result["broker_available"])
        self.assertEqual(result["current_equity"], 3_000_000.0)


class TestDDBreachAndPeakAnomalyWarnings(_IsolatedSnapshotMixin, unittest.TestCase):

    @patch("src.kabusapi.client.KabuClient")
    def test_dd_breach_warning_present_when_below_threshold(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 1_000_000.0}
        mock_client.get_positions.return_value = []
        mock_client_cls.return_value = mock_client

        # equity=1,000,000 vs peak=4,110,741 → DD ≈ -75.7% (BUY_STOP閾値-15%を大きく下回る)
        result = _compute_startup_equity(_base_state(position_qtys={}))
        self.assertTrue(result["dd_breach"])
        self.assertTrue(any("DD警告" in w for w in result["warnings"]))

    @patch("src.kabusapi.client.KabuClient")
    def test_no_dd_breach_when_equity_near_peak(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 4_000_000.0}
        mock_client.get_positions.return_value = []
        mock_client_cls.return_value = mock_client

        result = _compute_startup_equity(_base_state(position_qtys={}))
        self.assertFalse(result["dd_breach"])
        self.assertFalse(any("DD警告" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
