"""
tests/test_duplicate_prevention.py
P1-4: 重複発注防止 + P0-1: drift 検知テスト
"""
import sys
import unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import MagicMock
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


@contextmanager
def _workspace_tempdir():
    base_dir = Path(__file__).resolve().parents[1] / ".codex_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / f"test_duplicate_prevention_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_dir
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestIdempotencyLock(unittest.TestCase):
    """already_ordered_today / mark_ordered のロックファイル操作テスト"""

    def setUp(self):
        self._tmpdir = _workspace_tempdir()
        self._tmp_path = self._tmpdir.__enter__()
        self._lock_file = self._tmp_path / "order_lock.json"
        import src.run_morning_signal as rms
        self._rms = rms
        self._orig_lock = rms.ORDER_LOCK_FILE
        rms.ORDER_LOCK_FILE = self._lock_file

    def tearDown(self):
        self._rms.ORDER_LOCK_FILE = self._orig_lock
        self._tmpdir.__exit__(None, None, None)

    def test_lock_recorded_before_order(self):
        """mark_ordered() でロックが記録され、already_ordered_today() が True を返す"""
        self.assertFalse(self._rms.already_ordered_today("7203.T"))
        self._rms.mark_ordered("7203.T", "BUY")
        self.assertTrue(self._rms.already_ordered_today("7203.T"))

    def test_same_symbol_not_ordered_twice(self):
        """ロック済み銘柄は already_ordered_today() が True、別銘柄は False"""
        self._rms.mark_ordered("7203.T", "BUY")
        self.assertTrue(self._rms.already_ordered_today("7203.T"))
        self.assertFalse(self._rms.already_ordered_today("9984.T"))


class TestDriftDetection(unittest.TestCase):
    """position_sync.sync_and_validate_state() のテスト"""

    def _call(self, broker: dict, local: dict) -> dict:
        from src.live.position_sync import sync_and_validate_state
        mock_bridge = MagicMock()
        mock_bridge.get_broker_positions.return_value = broker
        mock_bridge.get_local_positions.return_value = local
        return sync_and_validate_state(mock_bridge), mock_bridge

    def test_drift_detection_raises_abort(self):
        """broker にあってローカルにないポジション → AbortError"""
        from src.kabusapi.signal_bridge import AbortError
        from src.live.position_sync import sync_and_validate_state

        mock_bridge = MagicMock()
        mock_bridge.get_broker_positions.return_value = {"7203.T": 100}
        mock_bridge.get_local_positions.return_value = {}

        with self.assertRaises(AbortError):
            sync_and_validate_state(mock_bridge)

        mock_bridge.overwrite_local_positions.assert_called_once_with({"7203.T": 100})

    def test_no_drift_when_positions_match(self):
        """broker と local が一致 → drift なし"""
        result, bridge = self._call({"7203.T": 100}, {"7203.T": 1})
        self.assertFalse(result["drift_detected"])
        bridge.overwrite_local_positions.assert_not_called()

    def test_sync_failure_raises_abort(self):
        """API エラー時は AbortError"""
        from src.kabusapi.signal_bridge import AbortError
        from src.live.position_sync import sync_and_validate_state
        mock_bridge = MagicMock()
        mock_bridge.get_broker_positions.side_effect = ConnectionError("API unreachable")
        with self.assertRaises(AbortError):
            sync_and_validate_state(mock_bridge)

    def test_local_extra_symbol_drift_no_halt(self):
        """ローカルにあってbrokerにない → drift=True（既決済扱い）"""
        result, bridge = self._call({}, {"7203.T": 1})
        self.assertTrue(result["drift_detected"])
        self.assertTrue(result["synced"])
        self.assertEqual(result["broker_syms"], set())
        self.assertEqual(result["local_syms"], {"7203.T"})
        bridge.overwrite_local_positions.assert_called_once_with({})


if __name__ == "__main__":
    unittest.main()
