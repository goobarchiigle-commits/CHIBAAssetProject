"""
tests/test_sync_positions.py
src/scripts/sync_positions.py の回帰テスト（Broker-as-Sole-SSOT, 2026-07-18）。

背景: 旧実装は独自にqty/avg_price構築・state["last_equity"]直接書き込み・
手動pruningループを行っており、SignalBridge本体とは別の資産計算経路だった。
fetch_broker_snapshot() + compute_live_equity() + commit_broker_snapshot() への
一本化を検証する。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.scripts.sync_positions as sp


def _write_state(path: Path, **overrides) -> None:
    state = {
        "schema_version": 3, "equity_peak": 3_000_000.0, "cb_state": "NORMAL",
        "available_cash": 1_000_000.0, "position_qtys": {}, "positions_count": 0,
        "last_equity": 3_000_000.0, "generation_id": 1,
        "position_entry_dates": {}, "position_entry_prices": {},
    }
    state.update(overrides)
    path.write_text(json.dumps(state), encoding="utf-8")


class SyncPositionsTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.state_file = Path(self._tmpdir.name) / "portfolio_state.json"
        _write_state(self.state_file)
        self._patcher = mock.patch.object(sp, "PORTFOLIO_STATE_FILE", self.state_file)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _run(self, argv_extra: list[str]) -> int:
        argv = ["sync_positions.py"] + argv_extra
        with mock.patch.object(sys, "argv", argv):
            return sp.main()

    def _load_state(self) -> dict:
        return json.loads(self.state_file.read_text(encoding="utf-8"))


class TestApiConnectFailure(SyncPositionsTestBase):
    @mock.patch("src.scripts.sync_positions.KabuClient")
    def test_token_fetch_failure_returns_1(self, mock_client_cls):
        mock_client_cls.side_effect = ConnectionError("unreachable")
        rc = self._run(["--force"])
        self.assertEqual(rc, 1)


class TestBrokerSnapshotFailure(SyncPositionsTestBase):
    @mock.patch("src.scripts.sync_positions.KabuClient")
    def test_positions_fetch_failure_returns_1(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 1_000_000.0}
        mock_client.get_positions.side_effect = RuntimeError("API error")
        mock_client_cls.return_value = mock_client
        rc = self._run(["--force"])
        self.assertEqual(rc, 1)
        # state は変更されない
        state = self._load_state()
        self.assertEqual(state["available_cash"], 1_000_000.0)


class TestEmptyPositions(SyncPositionsTestBase):
    @mock.patch("src.scripts.sync_positions.KabuClient")
    def test_force_clears_entry_metadata(self, mock_client_cls):
        _write_state(
            self.state_file,
            position_entry_dates={"5301.T": "2026-07-01"},
            position_entry_prices={"5301.T": 1700.0},
        )
        mock_client = mock.MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 3_000_000.0}
        mock_client.get_positions.return_value = []
        mock_client_cls.return_value = mock_client

        rc = self._run(["--force"])
        self.assertEqual(rc, 0)
        state = self._load_state()
        self.assertEqual(state["position_entry_dates"], {})
        self.assertEqual(state["position_entry_prices"], {})


class TestNormalSync(SyncPositionsTestBase):
    @mock.patch("src.scripts.sync_positions.KabuClient")
    def test_commits_broker_snapshot(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 1_500_000.0}
        mock_client.get_positions.return_value = [
            {"Symbol": "7203", "LeavesQty": 100, "Price": 2500.0, "CurrentPrice": 2600.0},
        ]
        mock_client_cls.return_value = mock_client

        rc = self._run(["--force", "--entry-date", "2026-07-18"])
        self.assertEqual(rc, 0)

        state = self._load_state()
        self.assertEqual(state["available_cash"], 1_500_000.0)
        self.assertEqual(state["position_qtys"], {"7203.T": 100})
        # last_equity = cash + qty * CurrentPrice = 1,500,000 + 100*2600 = 1,760,000
        self.assertAlmostEqual(state["last_equity"], 1_760_000.0, delta=1.0)
        self.assertEqual(state["position_entry_dates"]["7203.T"], "2026-07-18")
        self.assertEqual(state["position_entry_prices"]["7203.T"], 2500.0)
        # equity_peak は sync_positions からは絶対に書き換えない
        self.assertEqual(state["equity_peak"], 3_000_000.0)

    @mock.patch("src.scripts.sync_positions.KabuClient")
    def test_removes_stale_symbol_not_in_broker(self, mock_client_cls):
        _write_state(
            self.state_file,
            position_qtys={"6981.T": 100},
            position_entry_dates={"6981.T": "2026-07-01"},
            position_entry_prices={"6981.T": 4800.0},
        )
        mock_client = mock.MagicMock()
        mock_client.get_wallet_cash.return_value = {"StockAccountWallet": 3_000_000.0}
        mock_client.get_positions.return_value = [
            {"Symbol": "7203", "LeavesQty": 100, "Price": 2500.0, "CurrentPrice": 2600.0},
        ]
        mock_client_cls.return_value = mock_client

        rc = self._run(["--force"])
        self.assertEqual(rc, 0)
        state = self._load_state()
        self.assertNotIn("6981.T", state["position_entry_dates"])
        self.assertNotIn("6981.T", state["position_qtys"])


if __name__ == "__main__":
    unittest.main()
