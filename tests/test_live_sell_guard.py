import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import src.run_morning_signal as rms


class TestLiveSellGuard(unittest.TestCase):

    def test_sell_skipped_when_broker_position_is_zero(self):
        orders = [SimpleNamespace(symbol="5401.T", side="SELL", qty=100)]
        with patch.object(rms, "already_ordered_today", return_value=False):
            guarded = rms._apply_live_order_guards(orders, {"5401.T": 0})
        self.assertEqual(guarded, [])

    def test_sell_qty_capped_by_broker_position(self):
        order = SimpleNamespace(symbol="5411.T", side="SELL", qty=300)
        with patch.object(rms, "already_ordered_today", return_value=False):
            guarded = rms._apply_live_order_guards([order], {"5411.T": 100})
        self.assertEqual(len(guarded), 1)
        self.assertEqual(guarded[0].qty, 100)

    def test_one_shot_lock_skips_live_order(self):
        order = SimpleNamespace(symbol="5411.T", side="SELL", qty=100)
        with patch.object(rms, "already_ordered_today", return_value=True):
            guarded = rms._apply_live_order_guards([order], {"5411.T": 100})
        self.assertEqual(guarded, [])


if __name__ == "__main__":
    unittest.main()
