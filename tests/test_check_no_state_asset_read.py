"""
tests/test_check_no_state_asset_read.py
scripts/check_no_state_asset_read.py の回帰テスト（Broker-as-Sole-SSOT, 2026-07-19）。
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_no_state_asset_read.py"
_spec = importlib.util.spec_from_file_location("check_no_state_asset_read", _SCRIPT_PATH)
check_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_mod)


class TestCleanRepoPasses(unittest.TestCase):
    def test_current_repo_has_no_violations(self):
        """現状のリポジトリは違反ゼロであること（このテスト自体がCIゲート）。"""
        violations = check_mod.find_violations()
        self.assertEqual(
            violations, [],
            f"未許可ファイルでの資産系フィールド読み取りを検出: {violations}",
        )


class TestPatternDetection(unittest.TestCase):
    def test_detects_dict_get_pattern(self):
        self.assertIsNotNone(check_mod._PATTERN.search('state.get("available_cash", 0)'))

    def test_detects_bracket_pattern(self):
        self.assertIsNotNone(check_mod._PATTERN.search('state["position_qtys"]'))

    def test_does_not_match_unrelated_key(self):
        self.assertIsNone(check_mod._PATTERN.search('state.get("equity_peak", 0)'))

    def test_snapshot_hash_not_guarded(self):
        """snapshot_hash/snapshot_tsはbacktest側の同名キーと衝突するため対象外。"""
        self.assertIsNone(check_mod._PATTERN.search('meta.get("snapshot_hash")'))


class TestFileExclusion(unittest.TestCase):
    def test_test_file_name_excluded(self):
        self.assertTrue(check_mod._is_test_file("src/foo/test_bar.py"))
        self.assertTrue(check_mod._is_test_file("src/foo/bar_test.py"))
        self.assertFalse(check_mod._is_test_file("src/foo/bar.py"))

    def test_allowed_files_frozenset_contains_core_modules(self):
        for expected in (
            "src/portfolio/state_store.py",
            "src/portfolio/equity.py",
            "src/portfolio/broker_source.py",
        ):
            self.assertIn(expected, check_mod.ALLOWED_FILES)


class TestViolationDetectionEndToEnd(unittest.TestCase):
    """find_violations() を一時ファイルで再現し、実際に検知することを確認する。"""

    def test_finds_violation_in_disallowed_file(self):
        with tempfile.TemporaryDirectory() as td:
            probe_dir = Path(td)
            probe_file = probe_dir / "some_module.py"
            probe_file.write_text(
                'def f():\n    state = {}\n    return state.get("available_cash", 0)\n',
                encoding="utf-8",
            )
            violations = check_mod.find_violations(src_dir=probe_dir, root_dir=probe_dir)
            self.assertEqual(len(violations), 1)
            self.assertIn("available_cash", violations[0][2])

    def test_allowed_file_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            probe_dir = Path(td) / "src" / "portfolio"
            probe_dir.mkdir(parents=True)
            probe_file = probe_dir / "state_store.py"
            probe_file.write_text(
                'def f():\n    state = {}\n    return state.get("available_cash", 0)\n',
                encoding="utf-8",
            )
            violations = check_mod.find_violations(src_dir=Path(td), root_dir=Path(td))
            self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
