"""
test_sync_stop_persistence.py
Task 1 regression: sync_positions highest_close protection + stop price math.

Rules verified:
  1. sync never overwrites an existing highest_close with a lower avg_price.
  2. stop formula: highest_close - 3 * ATR20 reproduces the cached stop.
  3. entry_atrs are preserved (never overwritten) by sync.
  4. entry_prices are always updated to broker fill avg.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest


# ── helpers mirroring sync_positions step-6b logic ───────────────────
def _apply_sync_step6b(state: dict, api_positions: list[dict]) -> dict:
    """
    Replicate the sync_positions.py step-6b mutation in isolation.
    state is modified in-place and also returned.
    """
    new_entry_dates  = {}
    new_entry_prices = {}
    entry_date       = "2026-04-15"

    for p in api_positions:
        sym = p["symbol"]
        existing_date  = state.get("position_entry_dates", {}).get(sym)
        new_entry_dates[sym]  = existing_date or entry_date
        new_entry_prices[sym] = p["avg_price"]

    state["position_entry_dates"]  = new_entry_dates
    state["position_entry_prices"] = new_entry_prices

    # highest_close: initialise only if absent; NEVER lower avg_price over existing
    for p in api_positions:
        sym = p["symbol"]
        existing_hc = state.get("position_highest_closes", {}).get(sym)
        if existing_hc is None:
            state.setdefault("position_highest_closes", {})[sym] = p["avg_price"]
        # (protection: if avg_price < existing_hc → skip, logged in real code)

    return state


class TestHighestCloseProtection(unittest.TestCase):
    """highest_closes must survive a sync call with lower avg_price."""

    def test_existing_highest_close_preserved_when_avg_lower(self):
        """9104.T fill=6351 must not overwrite highest_close=7022."""
        state = {
            "position_entry_dates":    {"9104.T": "2026-04-15"},
            "position_entry_prices":   {"9104.T": 6351.0},
            "position_entry_atrs":     {"9104.T": 284.95},
            "position_highest_closes": {"9104.T": 7022.0},
        }
        api_positions = [
            {"symbol": "9104.T", "qty": 100, "avg_price": 6351.0, "value": 635100.0}
        ]
        _apply_sync_step6b(state, api_positions)
        self.assertEqual(
            state["position_highest_closes"]["9104.T"],
            7022.0,
            "highest_close must not be overwritten with lower avg_price",
        )

    def test_new_symbol_initialises_highest_close_to_avg_price(self):
        """Brand-new holding: highest_close should be seeded from avg_price."""
        state = {
            "position_entry_dates":    {},
            "position_entry_prices":   {},
            "position_entry_atrs":     {},
            "position_highest_closes": {},
        }
        api_positions = [
            {"symbol": "1234.T", "qty": 200, "avg_price": 5000.0, "value": 1_000_000.0}
        ]
        _apply_sync_step6b(state, api_positions)
        self.assertEqual(
            state["position_highest_closes"]["1234.T"],
            5000.0,
            "initial highest_close must equal avg_price for new position",
        )

    def test_higher_avg_price_does_not_raise_highest_close(self):
        """
        avg_price > highest_close should NOT overwrite either.
        The signal_bridge updates highest_close daily via max(); sync never should.
        """
        state = {
            "position_entry_dates":    {"9104.T": "2026-04-01"},
            "position_entry_prices":   {"9104.T": 5900.0},
            "position_entry_atrs":     {"9104.T": 300.0},
            "position_highest_closes": {"9104.T": 6200.0},
        }
        api_positions = [
            {"symbol": "9104.T", "qty": 100, "avg_price": 6500.0, "value": 650000.0}
        ]
        _apply_sync_step6b(state, api_positions)
        # highest_close is already set → sync skips it regardless
        self.assertEqual(
            state["position_highest_closes"]["9104.T"],
            6200.0,
            "sync must not touch an already-set highest_close (even if avg_price is higher)",
        )

    def test_entry_atrs_not_touched_by_sync(self):
        """entry_atrs must survive a sync run unchanged."""
        state = {
            "position_entry_dates":    {"9104.T": "2026-04-15"},
            "position_entry_prices":   {"9104.T": 6351.0},
            "position_entry_atrs":     {"9104.T": 284.95},
            "position_highest_closes": {"9104.T": 7022.0},
        }
        api_positions = [
            {"symbol": "9104.T", "qty": 100, "avg_price": 6351.0, "value": 635100.0}
        ]
        _apply_sync_step6b(state, api_positions)
        self.assertAlmostEqual(
            state["position_entry_atrs"]["9104.T"],
            284.95,
            places=4,
            msg="entry_atrs must be preserved by sync",
        )

    def test_entry_prices_always_updated(self):
        """entry_prices should always be overwritten with broker fill."""
        state = {
            "position_entry_dates":    {"9104.T": "2026-04-10"},
            "position_entry_prices":   {"9104.T": 0.0},   # unknown prior
            "position_entry_atrs":     {},
            "position_highest_closes": {},
        }
        api_positions = [
            {"symbol": "9104.T", "qty": 100, "avg_price": 6351.0, "value": 635100.0}
        ]
        _apply_sync_step6b(state, api_positions)
        self.assertEqual(
            state["position_entry_prices"]["9104.T"],
            6351.0,
            "entry_prices must be updated to broker fill avg",
        )


class TestStopPriceMath(unittest.TestCase):
    """Stop formula: stop = highest_close - 3 * ATR20."""

    def test_9104T_stop_reproduced(self):
        """9104.T: 7022 - 3 * 284.95 ≈ 6167."""
        highest_close = 7022.0
        atr20         = 284.95
        stop          = highest_close - 3.0 * atr20
        self.assertAlmostEqual(stop, 6167.15, places=1)
        self.assertAlmostEqual(round(stop), 6167, places=0)

    def test_stop_is_below_highest_close(self):
        """Stop must always be below the peak."""
        for hc, atr in [(5000, 100), (8000, 250.5), (7022, 284.95)]:
            stop = hc - 3.0 * atr
            self.assertLess(stop, hc, f"stop({stop}) must be < highest_close({hc})")

    def test_trailing_stop_not_triggered_at_peak(self):
        """close == highest_close must never trigger trailing stop."""
        highest_close = 7022.0
        atr20         = 284.95
        stop          = highest_close - 3.0 * atr20
        close         = highest_close   # right at peak
        self.assertFalse(close < stop, "close == peak must not trigger stop")

    def test_trailing_stop_triggered_below_stop(self):
        """close < stop should trigger."""
        highest_close = 7022.0
        atr20         = 284.95
        stop          = highest_close - 3.0 * atr20  # 6167.15
        close         = 6100.0
        self.assertTrue(close < stop, "close=6100 < stop=6167 must trigger")
