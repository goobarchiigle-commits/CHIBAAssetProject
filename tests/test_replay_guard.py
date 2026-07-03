"""
tests/test_replay_guard.py
Deterministic tests for deferred replay storm protection.

Covers:
  - cycle throttle (max 2 per cycle, highest score first)
  - sector replay cap (max 1 per sector per cycle)
  - open auction window block (08:59:00–09:00:30 JST)
  - cooldown enforcement (30 min after failed retry)
  - cooldown expiry (old failure → admitted)
  - non-eligible pass-through (should_retry=False → non_eligible)
  - metric counters (skipped / throttled / sector_blocked)
  - all guards independent (correct partitioning)
  - admitted entries are highest-score first
  - last_failed_at serialisation roundtrip
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.execution.deferred_queue import make_deferred_entry, DeferredQueue
from src.execution.deferred_retry import RetryEvaluation, apply_retry_result
from src.execution.replay_guard import (
    ReplayGuard, ReplayGuardConfig, StormProtectionResult,
)

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(**kwargs) -> ReplayGuardConfig:
    defaults = dict(
        max_retries_per_cycle           = 2,
        max_entries_per_sector_per_cycle = 1,
        cooldown_minutes                = 30,
        auction_block_start             = (8, 59, 0),
        auction_block_end               = (9,  0, 30),
    )
    defaults.update(kwargs)
    return ReplayGuardConfig(**defaults)


def _jst(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 5, 9, hour, minute, second, tzinfo=JST)


def _eval(
    symbol:        str,
    sector:        str   = "電子部品",
    score:         float = 0.80,
    should_retry:  bool  = True,
    drop_reason:   str | None = None,
    last_failed_at: str  = "",
) -> RetryEvaluation:
    entry = make_deferred_entry(
        symbol          = symbol,
        sector          = sector,
        strategy        = "turtle",
        signal_score    = score,
        desired_weight  = 0.33,
        degraded_weight = 0.10,
        reject_reason   = "sector_cap",
    )
    entry.last_failed_at = last_failed_at
    return RetryEvaluation(
        entry        = entry,
        should_retry = should_retry,
        drop_reason  = drop_reason,
        new_score    = score if should_retry else None,
    )


def _guard(**kwargs) -> ReplayGuard:
    return ReplayGuard(_cfg(**kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cycle throttle
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleThrottle(unittest.TestCase):
    NOW = _jst(10, 0)  # well outside auction window

    def test_only_two_admitted_from_five_eligible(self) -> None:
        # Use distinct sectors so sector cap (=1) does not interfere
        g = _guard(max_retries_per_cycle=2)
        sectors = ["S1", "S2", "S3", "S4", "S5"]
        evals = [_eval(f"{i}.T", sector=sectors[i], score=float(5 - i)) for i in range(5)]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),  2)
        self.assertEqual(len(r.throttled), 3)

    def test_highest_score_admitted_first(self) -> None:
        # Input already in score-descending order; admitted follows input order
        g = _guard(max_retries_per_cycle=2)
        evals = [
            _eval("A.T", sector="S1", score=0.50),
            _eval("B.T", sector="S2", score=0.95),
            _eval("C.T", sector="S3", score=0.75),
        ]
        r = g.filter(evals, now=self.NOW)
        admitted_symbols = [ev.entry.symbol for ev in r.admitted]
        self.assertEqual(admitted_symbols, ["A.T", "B.T"])  # input order preserved

    def test_single_eligible_under_quota_is_admitted(self) -> None:
        g = _guard(max_retries_per_cycle=2)
        evals = [_eval("A.T")]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),  1)
        self.assertEqual(len(r.throttled), 0)

    def test_zero_eligible_admitted_none(self) -> None:
        g = _guard(max_retries_per_cycle=2)
        r = g.filter([], now=self.NOW)
        self.assertEqual(len(r.admitted), 0)

    def test_throttled_count_is_correct(self) -> None:
        # Distinct sectors to isolate the cycle throttle (sector cap not the limiter)
        g = _guard(max_retries_per_cycle=1)
        evals = [_eval(f"{i}.T", sector=f"S{i}") for i in range(4)]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(r.deferred_replay_throttled, 3)

    def test_quota_zero_throttles_all(self) -> None:
        g = _guard(max_retries_per_cycle=0)
        evals = [_eval("A.T", sector="S1"), _eval("B.T", sector="S2")]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),  0)
        self.assertEqual(len(r.throttled), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sector replay cap
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorReplayCap(unittest.TestCase):
    NOW = _jst(10, 0)

    def test_second_entry_same_sector_blocked(self) -> None:
        g = _guard(max_retries_per_cycle=5, max_entries_per_sector_per_cycle=1)
        evals = [
            _eval("A.T", sector="電子部品", score=0.9),
            _eval("B.T", sector="電子部品", score=0.8),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),       1)
        self.assertEqual(len(r.blocked_sector), 1)
        self.assertEqual(r.admitted[0].entry.symbol, "A.T")

    def test_different_sectors_both_admitted(self) -> None:
        g = _guard(max_retries_per_cycle=5, max_entries_per_sector_per_cycle=1)
        evals = [
            _eval("A.T", sector="電子部品"),
            _eval("B.T", sector="機械"),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),       2)
        self.assertEqual(len(r.blocked_sector), 0)

    def test_sector_cap_two_allows_two_per_sector(self) -> None:
        g = _guard(max_retries_per_cycle=5, max_entries_per_sector_per_cycle=2)
        evals = [
            _eval("A.T", sector="電子部品"),
            _eval("B.T", sector="電子部品"),
            _eval("C.T", sector="電子部品"),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),       2)
        self.assertEqual(len(r.blocked_sector), 1)

    def test_sector_blocked_count_metric(self) -> None:
        g = _guard(max_retries_per_cycle=5, max_entries_per_sector_per_cycle=1)
        evals = [
            _eval("A.T", sector="電子部品"),
            _eval("B.T", sector="電子部品"),
            _eval("C.T", sector="電子部品"),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(r.deferred_sector_blocked, 2)

    def test_throttle_and_sector_block_interact_correctly(self) -> None:
        # max_retries=2, sector_cap=1, 3 eligible in same sector.
        # Sector guard fires before cycle throttle:
        #   Entry A.T → admitted (sector=0<1, cycle=0<2) → sector=1, cycle=1
        #   Entry B.T → blocked_sector (sector=1>=1)
        #   Entry C.T → blocked_sector (sector=1>=1)
        # Result: admitted=1, blocked_sector=2, throttled=0
        g = _guard(max_retries_per_cycle=2, max_entries_per_sector_per_cycle=1)
        evals = [
            _eval("A.T", sector="電子部品", score=0.9),
            _eval("B.T", sector="電子部品", score=0.8),
            _eval("C.T", sector="電子部品", score=0.7),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),       1)
        self.assertEqual(len(r.blocked_sector), 2)
        self.assertEqual(len(r.throttled),      0)
        self.assertEqual(r.admitted[0].entry.symbol, "A.T")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Open auction window
# ─────────────────────────────────────────────────────────────────────────────

class TestAuctionWindow(unittest.TestCase):
    def test_at_auction_start_all_blocked(self) -> None:
        g = _guard()
        r = g.filter([_eval("A.T"), _eval("B.T")], now=_jst(8, 59, 0))
        self.assertEqual(len(r.blocked_auction), 2)
        self.assertEqual(len(r.admitted),        0)

    def test_during_auction_window_all_blocked(self) -> None:
        g = _guard()
        r = g.filter([_eval("A.T")], now=_jst(8, 59, 45))
        self.assertEqual(len(r.blocked_auction), 1)

    def test_at_auction_end_boundary_all_blocked(self) -> None:
        g = _guard()
        r = g.filter([_eval("A.T")], now=_jst(9, 0, 30))
        self.assertEqual(len(r.blocked_auction), 1)

    def test_one_second_after_end_admitted(self) -> None:
        g = _guard()
        now = datetime(2026, 5, 9, 9, 0, 31, tzinfo=JST)
        r   = g.filter([_eval("A.T")], now=now)
        self.assertEqual(len(r.admitted),        1)
        self.assertEqual(len(r.blocked_auction), 0)

    def test_before_auction_window_admitted(self) -> None:
        g = _guard()
        r = g.filter([_eval("A.T")], now=_jst(8, 58, 59))
        self.assertEqual(len(r.admitted), 1)

    def test_auction_block_skipped_metric(self) -> None:
        g = _guard()
        r = g.filter([_eval("A.T"), _eval("B.T")], now=_jst(8, 59, 15))
        self.assertEqual(r.deferred_replay_skipped, 2)

    def test_non_eligible_not_counted_as_auction_blocked(self) -> None:
        g = _guard()
        evals = [
            _eval("A.T", should_retry=True),
            _eval("B.T", should_retry=False, drop_reason="signal_invalidated"),
        ]
        r = g.filter(evals, now=_jst(8, 59, 15))
        self.assertEqual(len(r.blocked_auction), 1)  # only the eligible one
        self.assertEqual(len(r.non_eligible),    1)

    def test_non_jst_now_converted_correctly(self) -> None:
        # UTC 23:59:00 = JST 08:59:00
        now_utc = datetime(2026, 5, 8, 23, 59, 0, tzinfo=timezone.utc)
        g = _guard()
        r = g.filter([_eval("A.T")], now=now_utc)
        self.assertEqual(len(r.blocked_auction), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cooldown
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldown(unittest.TestCase):
    NOW = _jst(10, 0)

    def _failed_ts(self, minutes_ago: float) -> str:
        ts = self.NOW - timedelta(minutes=minutes_ago)
        return ts.strftime("%Y-%m-%dT%H:%M:%S%z")

    def test_recent_failure_blocks_retry(self) -> None:
        g = _guard(cooldown_minutes=30)
        ev = _eval("A.T", last_failed_at=self._failed_ts(10))  # 10 min ago
        r  = g.filter([ev], now=self.NOW)
        self.assertEqual(len(r.blocked_cooldown), 1)
        self.assertEqual(len(r.admitted),         0)

    def test_expired_cooldown_admits_entry(self) -> None:
        g = _guard(cooldown_minutes=30)
        ev = _eval("A.T", last_failed_at=self._failed_ts(31))  # 31 min ago
        r  = g.filter([ev], now=self.NOW)
        self.assertEqual(len(r.admitted),         1)
        self.assertEqual(len(r.blocked_cooldown), 0)

    def test_no_last_failed_at_not_blocked(self) -> None:
        g = _guard(cooldown_minutes=30)
        ev = _eval("A.T", last_failed_at="")
        r  = g.filter([ev], now=self.NOW)
        self.assertEqual(len(r.admitted),         1)
        self.assertEqual(len(r.blocked_cooldown), 0)

    def test_exactly_at_cooldown_boundary_admitted(self) -> None:
        # Exactly 30 min ago → cooldown_remaining=0.0 → admitted
        g = _guard(cooldown_minutes=30)
        ev = _eval("A.T", last_failed_at=self._failed_ts(30))
        r  = g.filter([ev], now=self.NOW)
        self.assertEqual(len(r.admitted), 1)

    def test_malformed_last_failed_at_not_blocked(self) -> None:
        g = _guard(cooldown_minutes=30)
        ev = _eval("A.T", last_failed_at="not-a-timestamp")
        r  = g.filter([ev], now=self.NOW)
        self.assertEqual(len(r.admitted), 1)

    def test_apply_retry_result_stamps_last_failed_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q  = DeferredQueue(Path(tmp) / "deferred_entries.json")
            ev = _eval("A.T")
            q.add(ev.entry)
            apply_retry_result(q, ev, executed=False)
            updated = q.load()[0]
            self.assertTrue(updated.last_failed_at != "")

    def test_last_failed_at_persists_in_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q  = DeferredQueue(Path(tmp) / "deferred_entries.json")
            ev = _eval("A.T")
            q.add(ev.entry)
            apply_retry_result(q, ev, executed=False)
            loaded = q.load()[0]
            # Should be blocked on reload
            g    = _guard(cooldown_minutes=30)
            now  = datetime.now(JST)
            ev2  = RetryEvaluation(
                entry        = loaded,
                should_retry = True,
                drop_reason  = None,
                new_score    = 0.8,
            )
            r = g.filter([ev2], now=now)
            self.assertEqual(len(r.blocked_cooldown), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Non-eligible pass-through
# ─────────────────────────────────────────────────────────────────────────────

class TestNonEligiblePassThrough(unittest.TestCase):
    NOW = _jst(10, 0)

    def test_non_eligible_goes_to_non_eligible_not_throttled(self) -> None:
        g  = _guard()
        ev = _eval("A.T", should_retry=False, drop_reason="signal_invalidated")
        r  = g.filter([ev], now=self.NOW)
        self.assertEqual(len(r.non_eligible), 1)
        self.assertEqual(len(r.throttled),    0)
        self.assertEqual(len(r.admitted),     0)

    def test_mixed_eligible_and_non_eligible(self) -> None:
        # Use distinct sectors so sector cap (=1 per sector) doesn't block
        g = _guard(max_retries_per_cycle=2)
        evals = [
            _eval("A.T", sector="S1", should_retry=True),
            _eval("B.T", sector="S2", should_retry=False, drop_reason="ttl_expired"),
            _eval("C.T", sector="S3", should_retry=True),
            _eval("D.T", sector="S4", should_retry=False, drop_reason="signal_invalidated"),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),     2)
        self.assertEqual(len(r.non_eligible), 2)
        self.assertEqual(len(r.throttled),    0)

    def test_non_eligible_do_not_count_against_cycle_quota(self) -> None:
        # Non-eligible entries bypass all guards → cycle quota unaffected
        g = _guard(max_retries_per_cycle=2)
        evals = [
            _eval("A.T", sector="S1", should_retry=False, drop_reason="ttl_expired"),
            _eval("B.T", sector="S2", should_retry=False, drop_reason="ttl_expired"),
            _eval("C.T", sector="S3", should_retry=True),
            _eval("D.T", sector="S4", should_retry=True),
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),     2)
        self.assertEqual(len(r.non_eligible), 2)
        self.assertEqual(len(r.throttled),    0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. StormProtectionResult metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestStormProtectionResult(unittest.TestCase):
    NOW = _jst(10, 0)

    def test_total_suppressed_sums_all_non_admitted(self) -> None:
        g = _guard(max_retries_per_cycle=1, max_entries_per_sector_per_cycle=1,
                   cooldown_minutes=30)
        failed_ts = (self.NOW - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S%z")
        evals = [
            _eval("A.T", sector="電子部品", score=0.9),               # admitted
            _eval("B.T", sector="電子部品", score=0.8),               # sector blocked
            _eval("C.T", sector="機械",    score=0.7),                # throttled (quota=1)
            _eval("D.T", sector="化学",    score=0.6,
                  last_failed_at=failed_ts),                           # cooldown
        ]
        r = g.filter(evals, now=self.NOW)
        self.assertEqual(len(r.admitted),         1)
        self.assertEqual(len(r.blocked_sector),   1)
        self.assertEqual(len(r.throttled),        1)
        self.assertEqual(len(r.blocked_cooldown), 1)
        self.assertEqual(r.total_suppressed,      3)

    def test_all_zero_on_empty_input(self) -> None:
        r = ReplayGuard().filter([], now=self.NOW)
        self.assertEqual(r.total_suppressed,          0)
        self.assertEqual(r.deferred_replay_skipped,   0)
        self.assertEqual(r.deferred_replay_throttled, 0)
        self.assertEqual(r.deferred_sector_blocked,   0)

    def test_auction_window_all_skipped(self) -> None:
        g = _guard()
        evals = [_eval("A.T"), _eval("B.T"), _eval("C.T")]
        r = g.filter(evals, now=_jst(8, 59, 15))
        self.assertEqual(r.deferred_replay_skipped,   3)
        self.assertEqual(r.deferred_replay_throttled, 0)
        self.assertEqual(r.deferred_sector_blocked,   0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Guard ordering — auction checked before cooldown/sector/throttle
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardOrdering(unittest.TestCase):
    def test_auction_takes_precedence_over_cooldown(self) -> None:
        now       = _jst(8, 59, 15)
        failed_ts = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S%z")
        g  = _guard()
        ev = _eval("A.T", last_failed_at=failed_ts)
        r  = g.filter([ev], now=now)
        # Even though it's in cooldown, the auction block fires first
        self.assertEqual(len(r.blocked_auction),  1)
        self.assertEqual(len(r.blocked_cooldown), 0)

    def test_cooldown_takes_precedence_over_sector_block(self) -> None:
        now       = _jst(10, 0)
        failed_ts = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S%z")
        g  = _guard(max_entries_per_sector_per_cycle=0)
        ev = _eval("A.T", last_failed_at=failed_ts)
        r  = g.filter([ev], now=now)
        # Cooldown fires before sector cap
        self.assertEqual(len(r.blocked_cooldown), 1)
        self.assertEqual(len(r.blocked_sector),   0)

    def test_sector_block_takes_precedence_over_throttle(self) -> None:
        now = _jst(10, 0)
        g   = _guard(max_retries_per_cycle=5, max_entries_per_sector_per_cycle=1)
        evals = [
            _eval("A.T", sector="電子部品"),
            _eval("B.T", sector="電子部品"),  # sector blocked, not throttled
        ]
        r = g.filter(evals, now=now)
        self.assertEqual(len(r.blocked_sector), 1)
        self.assertEqual(len(r.throttled),      0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. last_failed_at serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestLastFailedAtSerialisation(unittest.TestCase):
    def test_roundtrip_via_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q  = DeferredQueue(Path(tmp) / "deferred_entries.json")
            ev = _eval("6762.T")
            q.add(ev.entry)
            apply_retry_result(q, ev, executed=False)
            loaded = q.load()[0]
            d = loaded.to_dict()
            from src.execution.deferred_queue import DeferredEntry
            restored = DeferredEntry.from_dict(d)
            self.assertEqual(restored.last_failed_at, loaded.last_failed_at)
            self.assertTrue(len(restored.last_failed_at) > 0)

    def test_empty_last_failed_at_roundtrips(self) -> None:
        e = make_deferred_entry(
            symbol="A.T", sector="S", strategy="t",
            signal_score=0.5, desired_weight=0.3,
            degraded_weight=0.1, reject_reason="sector_cap",
        )
        self.assertEqual(e.last_failed_at, "")
        from src.execution.deferred_queue import DeferredEntry
        restored = DeferredEntry.from_dict(e.to_dict())
        self.assertEqual(restored.last_failed_at, "")


if __name__ == "__main__":
    unittest.main()
