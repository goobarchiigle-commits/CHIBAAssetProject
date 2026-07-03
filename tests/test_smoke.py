import importlib
import unittest

from src.paths import BACKTEST_DATASET_DIR, DEFAULT_DATA_VERSION


class SmokeTests(unittest.TestCase):
    def test_default_snapshot_detected(self) -> None:
        self.assertTrue(DEFAULT_DATA_VERSION)
        self.assertTrue((BACKTEST_DATASET_DIR / DEFAULT_DATA_VERSION).exists())

    def test_target_modules_import(self) -> None:
        importlib.import_module("src.run_live_signal")
        importlib.import_module("src.backtest.composite_alpha_bt")


if __name__ == "__main__":
    unittest.main()
