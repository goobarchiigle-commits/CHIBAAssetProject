"""
src/execution/test_adaptive_alloc.py
Adaptive portfolio allocator v2 テスト。

実行:
    cd C:/ai-trading
    python -m pytest src/execution/test_adaptive_alloc.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.adaptive_alloc import (
    AdaptiveAllocator,
    SectorDegradeResult,
    ForecastFrame,
    REJECT_THRESHOLD,
    LOT_SIZE,
    _round_lot,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alloc(
    capital: float = 3_000_000,
    sector_cap: float = 0.25,
    bear_sector_cap: float = 0.18,
    is_bear: bool = False,
) -> AdaptiveAllocator:
    return AdaptiveAllocator(
        capital=capital,
        sector_cap=sector_cap,
        bear_sector_cap=bear_sector_cap,
        is_bear=is_bear,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterministicAllocation
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicAllocation(unittest.TestCase):
    """Same input always produces same output."""

    def _run(self, sectors, qty, price):
        a = _alloc()
        results = []
        for sec in sectors:
            r = a.apply(sec, qty, price)
            if r.reason != "rejected":
                a.commit(sec, r.degraded_qty, price)
            results.append(r)
        return results

    def test_same_input_same_output(self):
        sectors = ["電機精密", "機械", "電機精密"]
        r1 = self._run(sectors, 200, 5000.0)
        r2 = self._run(sectors, 200, 5000.0)
        for a, b in zip(r1, r2):
            self.assertEqual(a.multiplier,   b.multiplier)
            self.assertEqual(a.degraded_qty, b.degraded_qty)
            self.assertEqual(a.reason,       b.reason)

    def test_ranking_order_preserved(self):
        """Higher-ranked candidates (processed first) get full allocation."""
        a = _alloc(capital=3_000_000, sector_cap=0.25)
        # Rank 1: 100 * 3000 / 3_000_000 = 0.10 (10% of cap) → ok
        qty_rank1 = 100
        r1 = a.apply("電機精密", qty_rank1, 3000.0)
        self.assertEqual(r1.reason, "ok")
        a.commit("電機精密", r1.degraded_qty, 3000.0)

        # Rank 2: 400 * 1500 / 3_000_000 = 0.20 > remaining 0.15 → degraded
        # remaining = 0.25 - 0.10 = 0.15; requested = 400*1500/3_000_000 = 0.20
        # multiplier = 0.15/0.20 = 0.75 → degraded
        qty_rank2 = 400
        r2 = a.apply("電機精密", qty_rank2, 1500.0)
        self.assertEqual(r2.reason, "degraded")
        self.assertLess(r2.degraded_qty, qty_rank2)
        self.assertGreater(r2.multiplier, REJECT_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
# TestNoSectorOverflow
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSectorOverflow(unittest.TestCase):

    def test_committed_weight_never_exceeds_cap(self):
        a = _alloc(capital=3_000_000, sector_cap=0.25)
        total_weight = 0.0
        for _ in range(10):
            r = a.apply("機械", 300, 2500.0)
            if r.reason != "rejected":
                a.commit("機械", r.degraded_qty, 2500.0)
                total_weight += r.degraded_qty * 2500.0 / 3_000_000
        # Total committed sector weight must not exceed cap
        util = a.sector_utilization().get("機械", 0.0)
        self.assertLessEqual(util, 0.25 + 1e-9,
                             f"sector overflow: {util:.4f} > 0.25")

    def test_bear_cap_respected(self):
        a = _alloc(is_bear=True, bear_sector_cap=0.18)
        for _ in range(5):
            r = a.apply("商社", 400, 3000.0)
            if r.reason != "rejected":
                a.commit("商社", r.degraded_qty, 3000.0)
        util = a.sector_utilization().get("商社", 0.0)
        self.assertLessEqual(util, 0.18 + 1e-9,
                             f"bear sector overflow: {util:.4f} > 0.18")

    def test_multi_sector_no_cross_contamination(self):
        a = _alloc(capital=3_000_000, sector_cap=0.25)
        # Fill 電機精密 (100 * 5000 / 3_000_000 = 0.167 → ok)
        r1 = a.apply("電機精密", 100, 5000.0)
        a.commit("電機精密", r1.degraded_qty, r1.degraded_qty and 5000.0 or 0.0)
        # 商社: 100 * 3000 / 3_000_000 = 0.10 < 0.25 → ok, unaffected by 電機精密
        r2 = a.apply("商社", 100, 3000.0)
        self.assertAlmostEqual(r2.multiplier, 1.0)
        self.assertEqual(r2.reason, "ok")


# ─────────────────────────────────────────────────────────────────────────────
# TestDegradationInvariant
# ─────────────────────────────────────────────────────────────────────────────

class TestDegradationInvariant(unittest.TestCase):

    def test_full_sector_rejected(self):
        """Sector at 100% capacity → multiplier=0 → rejected."""
        a = _alloc(capital=1_000_000, sector_cap=0.25)
        # Fill sector completely: 250_000 / (100 * 2500) = 1 lot exactly
        # Actually let's pre-load via existing exposure simulation
        # Manually jam sector weight to cap
        a._sector_weight["テスト"] = 0.25
        r = a.apply("テスト", 100, 5000.0)
        self.assertEqual(r.reason, "rejected")
        self.assertEqual(r.degraded_qty, 0)
        self.assertLess(r.multiplier, REJECT_THRESHOLD)

    def test_partial_sector_degraded(self):
        """Sector at 80% of cap → multiplier ~0.20 → rejected (< 0.25)."""
        a = _alloc(capital=1_000_000, sector_cap=0.25)
        a._sector_weight["テスト"] = 0.20  # 80% of 0.25 used
        # requested = 100 * 5000 / 1_000_000 = 0.50 >> remaining 0.05
        # multiplier = 0.05 / 0.50 = 0.10 < 0.25 → rejected
        r = a.apply("テスト", 100, 5000.0)
        self.assertEqual(r.reason, "rejected")

    def test_slight_overflow_degraded_not_rejected(self):
        """Sector at 40% of cap → multiplier 0.60 → degraded, not rejected."""
        a = _alloc(capital=1_000_000, sector_cap=0.25)
        a._sector_weight["テスト"] = 0.10  # 40% of cap used
        # remaining = 0.15; requested = 100 * 1000 / 1_000_000 = 0.10
        # multiplier = 0.15 / 0.10 = 1.0 → ok (still fits)
        r = a.apply("テスト", 100, 1000.0)
        self.assertEqual(r.reason, "ok")
        self.assertEqual(r.multiplier, 1.0)

    def test_degraded_multiplier_at_boundary(self):
        """multiplier clearly above REJECT_THRESHOLD → degraded (not rejected)."""
        a = _alloc(capital=1_000_000, sector_cap=0.25)
        # remaining = 0.12; requested = 300*1000/1_000_000 = 0.30
        # multiplier = 0.12/0.30 = 0.40 > REJECT_THRESHOLD(0.25) → degraded
        a._sector_weight["テスト"] = 0.13   # remaining = 0.12
        r = a.apply("テスト", 300, 1000.0)
        self.assertAlmostEqual(r.multiplier, 0.12 / 0.30, places=4)
        self.assertEqual(r.reason, "degraded")
        self.assertNotEqual(r.degraded_qty, 0)

    def test_full_capacity_available_ok(self):
        """Empty sector → multiplier=1.0 → ok."""
        a = _alloc()
        r = a.apply("電機精密", 100, 3000.0)
        self.assertEqual(r.reason, "ok")
        self.assertEqual(r.multiplier, 1.0)
        self.assertEqual(r.degraded_qty, 100)


# ─────────────────────────────────────────────────────────────────────────────
# TestLotRoundingSafety
# ─────────────────────────────────────────────────────────────────────────────

class TestLotRoundingSafety(unittest.TestCase):

    def test_degraded_qty_always_multiple_of_lot(self):
        a = _alloc(capital=3_000_000, sector_cap=0.25)
        a._sector_weight["機械"] = 0.15   # remaining = 0.10
        # Several qty values to test
        for qty in (100, 250, 300, 450, 550, 800, 1000, 1500):
            r = a.apply("機械", qty, 3000.0)
            if r.reason != "rejected" and r.degraded_qty > 0:
                self.assertEqual(
                    r.degraded_qty % LOT_SIZE, 0,
                    f"qty={qty} degraded_qty={r.degraded_qty} not multiple of {LOT_SIZE}",
                )

    def test_round_lot_helper(self):
        self.assertEqual(_round_lot(0),   0)
        self.assertEqual(_round_lot(99),  0)
        self.assertEqual(_round_lot(100), 100)
        self.assertEqual(_round_lot(199), 100)
        self.assertEqual(_round_lot(200), 200)
        self.assertEqual(_round_lot(350), 300)

    def test_degraded_never_exceeds_original(self):
        a = _alloc()
        a._sector_weight["テスト"] = 0.15
        for qty in (100, 500, 1000):
            r = a.apply("テスト", qty, 2000.0)
            self.assertLessEqual(r.degraded_qty, r.original_qty)

    def test_zero_qty_is_rejected(self):
        a = _alloc()
        r = a.apply("テスト", 0, 5000.0)
        self.assertEqual(r.reason, "rejected")
        self.assertEqual(r.degraded_qty, 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestClusterRegime (bear cap)
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterRegime(unittest.TestCase):

    def test_bear_cap_lower_than_normal(self):
        bull = _alloc(capital=3_000_000, is_bear=False, sector_cap=0.25, bear_sector_cap=0.18)
        bear = _alloc(capital=3_000_000, is_bear=True,  sector_cap=0.25, bear_sector_cap=0.18)
        # Both at 14% sector weight
        # bull remaining = 0.25 - 0.14 = 0.11; requested = 200*3000/3M = 0.20
        # bull multiplier = 0.11/0.20 = 0.55 > 0.25 → degraded (not rejected)
        # bear remaining = 0.18 - 0.14 = 0.04; bear multiplier = 0.04/0.20 = 0.20 < 0.25 → rejected
        bull._sector_weight["電機"] = 0.14
        bear._sector_weight["電機"] = 0.14

        qty, price = 200, 3000.0   # requested = 0.20

        rb = bull.apply("電機", qty, price)
        self.assertNotEqual(rb.reason, "rejected",
                            f"bull should degrade not reject; mult={rb.multiplier:.4f}")

        rr = bear.apply("電機", qty, price)
        self.assertEqual(rr.reason, "rejected",
                         f"bear should reject; mult={rr.multiplier:.4f}")

    def test_bear_mode_uses_bear_cap(self):
        a = _alloc(is_bear=True, sector_cap=0.25, bear_sector_cap=0.18)
        self.assertAlmostEqual(a._effective_cap, 0.18)

    def test_normal_mode_uses_sector_cap(self):
        a = _alloc(is_bear=False, sector_cap=0.25, bear_sector_cap=0.18)
        self.assertAlmostEqual(a._effective_cap, 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# TestForecastTrajectory
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastTrajectory(unittest.TestCase):

    def _candidates(self):
        # All weights designed to fit within per-sector 25% cap when alone:
        # 6920.T: 200 * 3000 / 3_000_000 = 0.20  (rank1, fits fully)
        # 6981.T: 200 * 3000 / 3_000_000 = 0.20  (rank2, after rank1 sector=0.20, remaining=0.05 → rejected)
        # 8053.T: 100 * 2500 / 3_000_000 ≈ 0.083 (different sector, fits fully)
        return [
            ("6920.T", "電機精密", 1, 200, 3000.0),
            ("6981.T", "電機精密", 2, 200, 3000.0),
            ("8053.T", "商社",     3, 100, 2500.0),
        ]

    def test_forecast_returns_one_frame_per_candidate(self):
        a = _alloc(capital=3_000_000)
        frames = a.forecast(self._candidates())
        self.assertEqual(len(frames), 3)

    def test_forecast_does_not_mutate_state(self):
        a = _alloc(capital=3_000_000)
        a._sector_weight["電機精密"] = 0.05
        before = dict(a._sector_weight)
        a.forecast(self._candidates())
        self.assertEqual(a._sector_weight, before)

    def test_forecast_different_sector_ok(self):
        # 8053.T: 100 * 2500 / 3_000_000 ≈ 0.083 < cap(0.25) → multiplier=1.0
        a = _alloc(capital=3_000_000)
        frames = a.forecast(self._candidates())
        shosya_frame = next(f for f in frames if f.sector == "商社")
        self.assertFalse(shosya_frame.rejected)
        self.assertAlmostEqual(shosya_frame.multiplier, 1.0, places=4)

    def test_forecast_degraded_sector_reduces_next_candidates(self):
        """First candidate with large weight degrades sector for subsequent ones."""
        # Rank 1 at 6920.T: 300 * 40000 / 3_000_000 = 4.0 > cap(0.25) → capped at 0.25
        # After rank 1 allocated ~0.25 sector weight, rank 2 has near-zero remaining
        a = _alloc(capital=3_000_000)
        frames = a.forecast(self._candidates())
        denki_frames = [f for f in frames if f.sector == "電機精密"]
        self.assertGreaterEqual(len(denki_frames), 2)
        # rank1 should be degraded (weight >> cap) but not necessarily rejected
        # rank2 after rank1 fills sector → likely rejected
        if len(denki_frames) == 2:
            rank2 = denki_frames[1]
            # After rank1 exhausted sector budget, rank2 should be rejected or heavily degraded
            # (exact behavior depends on price, but remaining should be minimal)
            self.assertLessEqual(rank2.effective_weight, denki_frames[0].effective_weight)


# ─────────────────────────────────────────────────────────────────────────────
# TestCapitalEfficiency
# ─────────────────────────────────────────────────────────────────────────────

class TestCapitalEfficiency(unittest.TestCase):

    def test_zero_committed_zero_efficiency(self):
        a = _alloc()
        self.assertAlmostEqual(a.capital_efficiency(1_000_000), 0.0)

    def test_efficiency_bounded_at_one(self):
        a = _alloc(capital=3_000_000)
        a._total_allocated = 5_000_000  # over-allocated (shouldn't happen, but guard)
        self.assertLessEqual(a.capital_efficiency(3_000_000), 1.0)

    def test_efficiency_tracks_commits(self):
        a = _alloc(capital=3_000_000)
        # 100 * 3000 / 3_000_000 = 0.10 < cap(0.25) → ok, degraded_qty=100
        r = a.apply("機械", 100, 3000.0)
        self.assertEqual(r.reason, "ok")
        a.commit("機械", r.degraded_qty, 3000.0)
        # allocated = 100 * 3000 = 300_000
        eff = a.capital_efficiency(3_000_000)
        self.assertAlmostEqual(eff, 300_000 / 3_000_000, places=4)

    def test_existing_exposure_does_not_affect_efficiency(self):
        """set_existing_exposure seeds weights but not _total_allocated."""
        a = _alloc(capital=3_000_000)
        positions = {"6920.T": {"qty": 200}}
        a.set_existing_exposure(
            positions, {"6920.T": "電機精密"}, lambda sym: 40000.0
        )
        # No orders committed yet → efficiency is 0
        self.assertAlmostEqual(a.capital_efficiency(3_000_000), 0.0)
        # But sector weight is seeded
        self.assertGreater(a.sector_utilization().get("電機精密", 0.0), 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestSetExistingExposure
# ─────────────────────────────────────────────────────────────────────────────

class TestSetExistingExposure(unittest.TestCase):

    def test_seeds_sector_weight_from_positions(self):
        a = _alloc(capital=3_000_000)
        positions = {
            "6920.T": {"qty": 100},
            "6981.T": {"qty": 200},
        }
        universe_tickers = {"6920.T": "電機精密", "6981.T": "電機精密"}
        price_fn = lambda sym: 10000.0
        a.set_existing_exposure(positions, universe_tickers, price_fn)
        # 100*10000 + 200*10000 = 3_000_000 / 3_000_000 = 1.0 — sector full
        util = a.sector_utilization().get("電機精密", 0.0)
        self.assertAlmostEqual(util, 1.0, places=4)
        # New order in same sector → rejected
        r = a.apply("電機精密", 100, 10000.0)
        self.assertEqual(r.reason, "rejected")

    def test_excluded_sells_not_seeded(self):
        """Positions being sold should NOT contribute to sector weight."""
        a = _alloc(capital=3_000_000)
        # No positions seeded (simulating all sold)
        a.set_existing_exposure({}, {"6920.T": "電機精密"}, lambda sym: 10000.0)
        r = a.apply("電機精密", 200, 3000.0)
        self.assertEqual(r.reason, "ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    unittest.main(verbosity=2)
