"""
src/test_config_loader_entry_freeze.py
Entry Freeze Mode（資産保全・2026-07-17）の config_loader 側検証。

実行:
    cd C:/ai-trading
    python -m pytest src/test_config_loader_entry_freeze.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import resolve_entry_freeze, load_strategy_config


class TestResolveEntryFreeze(unittest.TestCase):
    """純粋関数のみを対象。ファイルI/O・lru_cacheに依存しない。"""

    def test_no_env_var_falls_back_to_yaml_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENTRY_FREEZE_ENABLED", None)
            enabled, reason = resolve_entry_freeze(False, "Research Freeze")
            self.assertFalse(enabled)
            self.assertEqual(reason, "Research Freeze")

    def test_no_env_var_falls_back_to_yaml_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENTRY_FREEZE_ENABLED", None)
            enabled, reason = resolve_entry_freeze(True, "Manual Freeze")
            self.assertTrue(enabled)
            self.assertEqual(reason, "Manual Freeze")

    def test_env_var_forces_enable_regardless_of_yaml(self):
        for truthy in ("1", "true", "True", "yes", "on"):
            with patch.dict(os.environ, {"ENTRY_FREEZE_ENABLED": truthy}):
                enabled, reason = resolve_entry_freeze(False, "Research Freeze")
                self.assertTrue(enabled, f"env={truthy!r} must force freeze ON")
                self.assertIn("env override", reason)

    def test_env_var_forces_disable_regardless_of_yaml(self):
        for falsy in ("0", "false", "False", "no", "off"):
            with patch.dict(os.environ, {"ENTRY_FREEZE_ENABLED": falsy}):
                enabled, reason = resolve_entry_freeze(True, "Manual Freeze")
                self.assertFalse(enabled, f"env={falsy!r} must force freeze OFF")

    def test_env_var_enable_matches_yaml_enable(self):
        """env=trueかつyaml側も既にenabled=Trueの場合、reasonはyaml側を維持する。"""
        with patch.dict(os.environ, {"ENTRY_FREEZE_ENABLED": "1"}):
            enabled, reason = resolve_entry_freeze(True, "Manual Freeze")
            self.assertTrue(enabled)
            self.assertEqual(reason, "Manual Freeze")


class TestLoadStrategyConfigEntryFreeze(unittest.TestCase):
    """実ファイル strategy.yaml を読み込む統合テスト。"""

    def setUp(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        load_strategy_config.cache_clear()

    def tearDown(self):
        os.environ.pop("ENTRY_FREEZE_ENABLED", None)
        load_strategy_config.cache_clear()

    def test_default_committed_state_is_frozen(self):
        """commit時点のstrategy.yamlはentry_freeze.enabled=trueであること
        （2026-07-17 Study100/101帰結によりEntry Freeze Modeを有効化してcommit。
        解除にはユーザーの明示操作が必要＝意図せずfreezeが解除された状態で
        commitされていないことの回帰guard）。"""
        cfg = load_strategy_config()
        self.assertTrue(cfg.entry_freeze.enabled)
        self.assertEqual(cfg.entry_freeze.reason, "Research Freeze")

    def test_env_override_flips_loaded_config_off(self):
        os.environ["ENTRY_FREEZE_ENABLED"] = "0"
        load_strategy_config.cache_clear()
        cfg = load_strategy_config()
        self.assertFalse(cfg.entry_freeze.enabled)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
