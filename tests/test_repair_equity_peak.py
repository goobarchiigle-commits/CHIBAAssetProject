"""
tests/test_repair_equity_peak.py
src/scripts/repair_equity_peak.py の回帰テスト（EQUITY_PEAK_HARDENING, 2026-07-03）。

対象:
  - dry-run 既定で書き込みが発生しないこと
  - --apply で state ファイルへ確定書き込みされること
  - --force なしで確認プロンプトを拒否すると書き込まれないこと
  - median / max 方式の切替
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.scripts.repair_equity_peak as repair_mod
import src.portfolio.equity as eq_mod


def _write_snapshot(path: Path, equities: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, eq in enumerate(equities):
            f.write(json.dumps({
                "timestamp": f"2026-06-{i + 1:02d}T08:44:00+0900",
                "equity": eq,
            }) + "\n")


def _write_state(path: Path, equity_peak: float) -> None:
    path.write_text(json.dumps({
        "schema_version": 3, "equity_peak": equity_peak, "cb_state": "NORMAL",
        "available_cash": 1_000_000.0, "position_qtys": {}, "positions_count": 0,
        "last_equity": equity_peak, "generation_id": 1,
        "candidate_peak": {"value": 9_999_999.0, "staged_date": "2026-06-01",
                            "reason": "test", "current_equity_at_stage": 9_999_999.0},
    }), encoding="utf-8")


class RepairEquityPeakTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self.state_file    = tmp / "state.json"
        self.snapshot_file = tmp / "equity_snapshots.jsonl"
        self.audit_file    = tmp / "equity_peak_audit.jsonl"

        _write_snapshot(self.snapshot_file, [3_000_000.0, 3_100_000.0, 3_200_000.0, 4_000_000.0])
        _write_state(self.state_file, equity_peak=3_000_000.0)

        self._orig_audit_file = eq_mod._PEAK_AUDIT_FILE
        eq_mod._PEAK_AUDIT_FILE = self.audit_file

    def tearDown(self):
        eq_mod._PEAK_AUDIT_FILE = self._orig_audit_file
        self._tmpdir.cleanup()

    def _run(self, extra_args: list[str], confirm_input: str | None = None) -> int:
        argv = [
            "repair_equity_peak.py",
            "--state-file", str(self.state_file),
            "--snapshot-file", str(self.snapshot_file),
        ] + extra_args
        with mock.patch.object(sys, "argv", argv):
            if confirm_input is not None:
                with mock.patch("builtins.input", return_value=confirm_input):
                    return repair_mod.main()
            return repair_mod.main()

    def _load_state(self) -> dict:
        return json.loads(self.state_file.read_text(encoding="utf-8"))


class TestDryRunDefault(RepairEquityPeakTestBase):
    def test_dry_run_does_not_write(self):
        rc = self._run([])
        self.assertEqual(rc, 0)
        state = self._load_state()
        self.assertEqual(state["equity_peak"], 3_000_000.0)  # 未変更
        self.assertFalse(self.audit_file.exists())


class TestApply(RepairEquityPeakTestBase):
    def test_apply_force_writes_new_peak(self):
        rc = self._run(["--apply", "--force", "--method", "max"])
        self.assertEqual(rc, 0)
        state = self._load_state()
        self.assertEqual(state["equity_peak"], 4_000_000.0)  # max(全equity)
        self.assertIsNone(state["candidate_peak"])  # 手動修復でクリアされる
        self.assertTrue(self.audit_file.exists())
        record = json.loads(self.audit_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(record["action"], "APPLIED")
        self.assertEqual(record["mode"], "manual")
        self.assertEqual(record["new_peak"], 4_000_000.0)

    def test_apply_without_force_declines_on_no(self):
        rc = self._run(["--apply"], confirm_input="n")
        self.assertEqual(rc, 2)
        state = self._load_state()
        self.assertEqual(state["equity_peak"], 3_000_000.0)  # 変更されない

    def test_apply_without_force_confirms_on_yes(self):
        rc = self._run(["--apply", "--method", "max"], confirm_input="y")
        self.assertEqual(rc, 0)
        state = self._load_state()
        self.assertEqual(state["equity_peak"], 4_000_000.0)


class TestMissingSnapshot(unittest.TestCase):
    def test_missing_snapshot_file_returns_1(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            state_file    = tmp / "state.json"
            snapshot_file = tmp / "does_not_exist.jsonl"
            _write_state(state_file, equity_peak=3_000_000.0)
            argv = [
                "repair_equity_peak.py",
                "--state-file", str(state_file),
                "--snapshot-file", str(snapshot_file),
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = repair_mod.main()
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
