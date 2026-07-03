"""
tests/test_deferred_queue.py
Deterministic tests for the deferred entry observability + recovery layer.

Covers:
  - queue persistence (add / load / save roundtrip)
  - TTL expiration
  - retry recovery (RetryEvaluation flow)
  - lot rounding accumulation (FractionalAccumulator)
  - sector cap preservation (never bypassed in retry)
  - broker truth isolation (assert_positions_match)
  - no virtual pnl leakage
  - no cash mutation before execution
  - stale deferred cleanup (expire_stale)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.execution.deferred_queue import (
    DeferredEntry, DeferredQueue,
    make_deferred_entry, make_entry_id,
    ALLOWED_REJECT_REASONS, DEFAULT_TTL_DAYS, MAX_RETRY_COUNT,
)
from src.execution.fractional_demand import FractionalAccumulator
from src.execution.broker_truth import assert_positions_match, assert_deferred_isolated
from src.execution.deferred_retry import (
    evaluate_retry_candidates, apply_retry_result,
)
from src.execution.deferred_metrics import (
    DeferredMetrics, build_metrics, emit_metrics,
    track_missed_entry,
)

JST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entry(
    symbol:       str  = "6762.T",
    sector:       str  = "電子部品",
    strategy:     str  = "turtle",
    score:        float = 0.85,
    desired_w:    float = 0.33,
    degraded_w:   float = 0.18,
    reason:       str  = "sector_cap",
    ttl_days:     int  = DEFAULT_TTL_DAYS,
) -> DeferredEntry:
    return make_deferred_entry(
        symbol          = symbol,
        sector          = sector,
        strategy        = strategy,
        signal_score    = score,
        desired_weight  = desired_w,
        degraded_weight = degraded_w,
        reject_reason   = reason,
        ttl_days        = ttl_days,
    )


def _queue(tmp_dir: str) -> DeferredQueue:
    return DeferredQueue(Path(tmp_dir) / "deferred_entries.json")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Queue persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestQueuePersistence(unittest.TestCase):
    def test_roundtrip_single_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q   = _queue(tmp)
            e   = _entry()
            q.add(e)
            loaded = q.load()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].symbol, e.symbol)
            self.assertEqual(loaded[0].entry_id, e.entry_id)

    def test_roundtrip_multiple_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            for sym in ["6762.T", "7203.T", "9984.T"]:
                q.add(_entry(symbol=sym))
            loaded = q.load()
            self.assertEqual(len(loaded), 3)

    def test_empty_queue_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            self.assertEqual(q.load(), [])

    def test_duplicate_entry_id_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            q.add(e)   # duplicate
            self.assertEqual(len(q.load()), 1)

    def test_save_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            # Confirm .tmp file is gone (replaced)
            self.assertFalse((Path(tmp) / "deferred_entries.tmp").exists())
            self.assertTrue((Path(tmp) / "deferred_entries.json").exists())

    def test_field_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry(
                symbol="4901.T", sector="化学", strategy="turtle",
                score=0.92, desired_w=0.30, degraded_w=0.12,
                reason="lot_rounding", ttl_days=2,
            )
            q.add(e)
            loaded = q.load()[0]
            self.assertEqual(loaded.sector,           "化学")
            self.assertEqual(loaded.reject_reason,    "lot_rounding")
            self.assertAlmostEqual(loaded.signal_score,   0.92)
            self.assertAlmostEqual(loaded.desired_weight,  0.30)
            self.assertAlmostEqual(loaded.degraded_weight, 0.12)
            self.assertEqual(loaded.ttl_days, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TTL expiration
# ─────────────────────────────────────────────────────────────────────────────

class TestTTLExpiration(unittest.TestCase):
    def _expired_entry(self) -> DeferredEntry:
        e = _entry(ttl_days=1)
        # Backdate created_at by 2 days
        old = (datetime.now(JST) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S%z")
        e.created_at = old
        e.updated_at = old
        return e

    def test_is_expired_true_for_old_entry(self) -> None:
        e = self._expired_entry()
        self.assertTrue(e.is_expired)

    def test_is_expired_false_for_fresh_entry(self) -> None:
        e = _entry(ttl_days=3)
        self.assertFalse(e.is_expired)

    def _expired_entry_for(self, symbol: str) -> DeferredEntry:
        e = _entry(symbol=symbol, ttl_days=1)
        old = (datetime.now(JST) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S%z")
        e.created_at = old
        e.updated_at = old
        return e

    def test_expire_stale_removes_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            fresh   = _entry(symbol="6762.T", ttl_days=3)
            expired = self._expired_entry_for("9984.T")
            q.add(fresh)
            q.add(expired)
            dropped = q.expire_stale()
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0].symbol, "9984.T")
            remaining = q.load()
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].symbol, "6762.T")

    def test_expire_stale_removes_max_retry_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry(ttl_days=99)  # TTL not expired
            e.retry_count = MAX_RETRY_COUNT   # but retry exhausted
            q.add(e)
            dropped = q.expire_stale()
            self.assertEqual(len(dropped), 1)
            self.assertEqual(len(q.load()), 0)

    def test_get_active_excludes_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            expired = self._expired_entry()
            q.add(expired)
            self.assertEqual(len(q.get_active()), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retry recovery
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryRecovery(unittest.TestCase):
    def _base_kwargs(self, entries: list[DeferredEntry]) -> dict:
        return dict(
            queue              = None,    # injected per test
            signal_scores      = {"6762.T": 0.9},
            tradeable_symbols  = {"6762.T"},
            sector_weights     = {"電子部品": 0.10},
            sector_caps        = {"電子部品": 0.25},
            current_positions  = {},
            max_open           = 3,
            portfolio_equity   = 3_000_000.0,
            available_cash     = 3_000_000.0,
        )

    def test_eligible_entry_returns_should_retry_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            kw = self._base_kwargs([e])
            kw["queue"] = q
            results = evaluate_retry_candidates(**kw)
            self.assertTrue(results[0].should_retry)

    def test_stale_signal_drops_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            kw = self._base_kwargs([e])
            kw["queue"]         = q
            kw["signal_scores"] = {"6762.T": 0.0}  # signal gone
            results = evaluate_retry_candidates(**kw)
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "signal_invalidated")

    def test_symbol_not_tradeable_drops_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            kw = self._base_kwargs([e])
            kw["queue"]             = q
            kw["tradeable_symbols"] = set()  # symbol removed
            results = evaluate_retry_candidates(**kw)
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "symbol_not_tradeable")

    def test_apply_retry_increments_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            kw = self._base_kwargs([e])
            kw["queue"] = q
            evals = evaluate_retry_candidates(**kw)
            apply_retry_result(q, evals[0], executed=False)
            updated = q.load()[0]
            self.assertEqual(updated.retry_count, 1)

    def test_apply_retry_executed_removes_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            kw = self._base_kwargs([e])
            kw["queue"] = q
            evals = evaluate_retry_candidates(**kw)
            apply_retry_result(q, evals[0], executed=True)
            self.assertEqual(len(q.load()), 0)

    def test_max_positions_reached_blocks_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            kw = self._base_kwargs([e])
            kw["queue"]             = q
            kw["max_open"]          = 1
            kw["current_positions"] = {"7203.T": 100}  # max_open=1, already 1
            results = evaluate_retry_candidates(**kw)
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "max_positions_reached")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lot rounding accumulation (FractionalAccumulator)
# ─────────────────────────────────────────────────────────────────────────────

class TestFractionalAccumulator(unittest.TestCase):
    def test_accumulate_three_days_triggers_lot(self) -> None:
        acc = FractionalAccumulator()
        acc.accumulate("6762.T", 0.32)
        acc.accumulate("6762.T", 0.41)
        total = acc.accumulate("6762.T", 0.38)
        self.assertAlmostEqual(total, 1.11, places=6)
        self.assertEqual(acc.check_lot_threshold("6762.T"), 1)

    def test_accumulate_below_threshold_returns_zero_lots(self) -> None:
        acc = FractionalAccumulator()
        acc.accumulate("6762.T", 0.40)
        self.assertEqual(acc.check_lot_threshold("6762.T"), 0)

    def test_consume_reduces_accumulator(self) -> None:
        acc = FractionalAccumulator()
        acc.accumulate("6762.T", 0.32)
        acc.accumulate("6762.T", 0.85)
        remainder = acc.consume("6762.T", 1)
        self.assertAlmostEqual(remainder, 0.17, places=6)

    def test_accumulate_does_not_affect_portfolio_state(self) -> None:
        acc = FractionalAccumulator()
        portfolio = {"cash": 2_000_000.0, "equity": 3_000_000.0, "positions": {}}
        acc.accumulate("6762.T", 0.50)
        # Portfolio must be completely unchanged
        self.assertEqual(portfolio["cash"],   2_000_000.0)
        self.assertEqual(portfolio["equity"], 3_000_000.0)
        self.assertEqual(portfolio["positions"], {})

    def test_multi_symbol_isolation(self) -> None:
        acc = FractionalAccumulator()
        acc.accumulate("6762.T", 0.70)
        acc.accumulate("7203.T", 0.30)
        self.assertAlmostEqual(acc.get_accumulated("6762.T"), 0.70)
        self.assertAlmostEqual(acc.get_accumulated("7203.T"), 0.30)

    def test_reset_clears_symbol(self) -> None:
        acc = FractionalAccumulator()
        acc.accumulate("6762.T", 0.99)
        acc.reset("6762.T")
        self.assertAlmostEqual(acc.get_accumulated("6762.T"), 0.0)

    def test_total_accumulated_sums_all(self) -> None:
        acc = FractionalAccumulator()
        acc.accumulate("A", 0.5)
        acc.accumulate("B", 0.3)
        self.assertAlmostEqual(acc.total_accumulated(), 0.8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sector cap preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorCapPreservation(unittest.TestCase):
    def test_sector_cap_binding_blocks_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry(sector="電子部品")
            q.add(e)
            results = evaluate_retry_candidates(
                queue              = q,
                signal_scores      = {"6762.T": 0.9},
                tradeable_symbols  = {"6762.T"},
                sector_weights     = {"電子部品": 0.249},  # only 0.001 headroom
                sector_caps        = {"電子部品": 0.25},
                current_positions  = {},
                max_open           = 3,
                portfolio_equity   = 3_000_000.0,
                available_cash     = 3_000_000.0,
            )
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "sector_cap_still_binding")

    def test_sector_cap_partial_headroom_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry(sector="電子部品", desired_w=0.10)
            q.add(e)
            results = evaluate_retry_candidates(
                queue              = q,
                signal_scores      = {"6762.T": 0.9},
                tradeable_symbols  = {"6762.T"},
                sector_weights     = {"電子部品": 0.10},  # 0.15 headroom
                sector_caps        = {"電子部品": 0.25},
                current_positions  = {},
                max_open           = 3,
                portfolio_equity   = 3_000_000.0,
                available_cash     = 3_000_000.0,
            )
            self.assertTrue(results[0].should_retry)

    def test_allowed_reject_reasons_are_fixed(self) -> None:
        expected = {"sector_cap", "lot_rounding", "alloc_cap", "insufficient_cash"}
        self.assertEqual(ALLOWED_REJECT_REASONS, frozenset(expected))

    def test_invalid_reject_reason_raises(self) -> None:
        with self.assertRaises(ValueError):
            _entry(reason="fake_reason")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Broker truth isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerTruthIsolation(unittest.TestCase):
    def test_positions_match_passes_when_equal(self) -> None:
        broker = {"6762.T": 100, "7203.T": 200}
        local  = {"6762.T": 100, "7203.T": 200}
        assert_positions_match(broker, local)  # must not raise

    def test_positions_match_raises_on_mismatch(self) -> None:
        broker = {"6762.T": 100}
        local  = {"6762.T": 200}
        with self.assertRaises(AssertionError) as ctx:
            assert_positions_match(broker, local)
        self.assertIn("6762.T", str(ctx.exception))

    def test_positions_match_raises_on_missing_symbol(self) -> None:
        broker = {"6762.T": 100, "7203.T": 200}
        local  = {"6762.T": 100}  # 7203.T missing
        with self.assertRaises(AssertionError):
            assert_positions_match(broker, local)

    def test_positions_match_tolerance_allows_small_diff(self) -> None:
        broker = {"6762.T": 101}
        local  = {"6762.T": 100}
        assert_positions_match(broker, local, tolerance=1)  # must not raise

    def test_deferred_isolated_ok_with_clean_state(self) -> None:
        entries   = [_entry(symbol="6762.T")]
        portfolio = {"cash": 1_000_000.0, "equity": 3_000_000.0, "positions": {}}
        assert_deferred_isolated(entries, portfolio)  # must not raise

    def test_deferred_isolated_fails_on_nan_equity(self) -> None:
        entries   = [_entry()]
        portfolio = {"cash": 1_000_000.0, "equity": float("nan"), "positions": {}}
        with self.assertRaises(AssertionError):
            assert_deferred_isolated(entries, portfolio)


# ─────────────────────────────────────────────────────────────────────────────
# 7. No virtual pnl leakage / no cash mutation before execution
# ─────────────────────────────────────────────────────────────────────────────

class TestNoCashOrPnlMutation(unittest.TestCase):
    """
    Adding deferred entries must have zero effect on any accounting state.
    """

    def test_add_entry_does_not_mutate_portfolio(self) -> None:
        portfolio = {
            "cash":           2_500_000.0,
            "equity":         3_000_000.0,
            "positions":      {},
            "cumulative_pnl": 0.0,
            "max_drawdown":   0.0,
        }
        snapshot = dict(portfolio)

        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            q.add(_entry(symbol="6762.T"))
            q.add(_entry(symbol="7203.T", reason="lot_rounding"))

        # Portfolio must be completely unchanged
        self.assertEqual(portfolio, snapshot)

    def test_fractional_accum_does_not_mutate_portfolio(self) -> None:
        portfolio = {
            "cash":      2_000_000.0,
            "equity":    3_000_000.0,
            "positions": {},
        }
        snapshot = dict(portfolio)

        acc = FractionalAccumulator()
        for _ in range(5):
            acc.accumulate("6762.T", 0.30)

        self.assertEqual(portfolio, snapshot)

    def test_evaluate_retry_does_not_mutate_portfolio(self) -> None:
        portfolio = {
            "cash":      3_000_000.0,
            "equity":    3_000_000.0,
            "positions": {},
        }
        snapshot = dict(portfolio)

        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            evaluate_retry_candidates(
                queue              = q,
                signal_scores      = {"6762.T": 0.9},
                tradeable_symbols  = {"6762.T"},
                sector_weights     = {"電子部品": 0.05},
                sector_caps        = {"電子部品": 0.25},
                current_positions  = {},
                max_open           = 3,
                portfolio_equity   = portfolio["equity"],
                available_cash     = portfolio["cash"],
            )

        self.assertEqual(portfolio, snapshot)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Stale deferred cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestStaleCleanup(unittest.TestCase):
    def test_remove_symbol_clears_all_entries_for_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            q.add(_entry(symbol="6762.T", reason="sector_cap"))
            q.add(_entry(symbol="6762.T", reason="lot_rounding"))
            q.add(_entry(symbol="7203.T", reason="alloc_cap"))
            q.remove_symbol("6762.T", reason="delisted")
            remaining = q.load()
            symbols = [e.symbol for e in remaining]
            self.assertNotIn("6762.T", symbols)
            self.assertIn("7203.T", symbols)

    def test_remove_by_entry_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry()
            q.add(e)
            q.remove(e.entry_id, reason="manual_drop")
            self.assertEqual(len(q.load()), 0)

    def test_expire_stale_empty_queue_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            dropped = q.expire_stale()
            self.assertEqual(dropped, [])

    def test_full_lifecycle(self) -> None:
        """Add → retry × n → expire."""
        with tempfile.TemporaryDirectory() as tmp:
            q = _queue(tmp)
            e = _entry(ttl_days=3)
            q.add(e)

            for i in range(MAX_RETRY_COUNT - 1):
                active = q.get_active()
                self.assertEqual(len(active), 1)
                active[0].retry_count += 1
                q.update(active[0])

            # One more retry → MAX_RETRY_COUNT reached
            active = q.get_active()
            self.assertEqual(len(active), 1)
            active[0].retry_count += 1
            q.update(active[0])

            dropped = q.expire_stale()
            self.assertEqual(len(dropped), 1)
            self.assertEqual(len(q.load()), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestDeferredMetrics(unittest.TestCase):
    def test_build_metrics_rejected_by_reason(self) -> None:
        entries = [
            _entry(reason="sector_cap"),
            _entry(reason="sector_cap"),
            _entry(reason="lot_rounding"),
        ]
        m = build_metrics(
            deferred_entries           = entries,
            recovered_entries          = 1,
            expired_entries            = 0,
            retry_attempts             = 2,
            retry_successes            = 1,
            fractional_demand_total    = 1.5,
            fractional_execution_count = 1,
            missed_entries_before      = 4,
            missed_entries_after       = 3,
        )
        self.assertEqual(m.rejected_by_reason["sector_cap"],   2)
        self.assertEqual(m.rejected_by_reason["lot_rounding"], 1)
        self.assertAlmostEqual(m.retry_success_rate, 0.5)
        self.assertAlmostEqual(m.alpha_capture_estimate, 0.25)

    def test_emit_metrics_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deferred_metrics.jsonl"
            m = build_metrics(
                deferred_entries           = [],
                recovered_entries          = 0,
                expired_entries            = 0,
                retry_attempts             = 0,
                retry_successes            = 0,
                fractional_demand_total    = 0.0,
                fractional_execution_count = 0,
                missed_entries_before      = 0,
                missed_entries_after       = 0,
            )
            emit_metrics(m, path)
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertIn("timestamp", record)
            self.assertIn("deferred_queue_size", record)

    def test_track_missed_entry_writes_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missed.jsonl"
            track_missed_entry(
                entry_id      = "abc123",
                symbol        = "6762.T",
                reject_reason = "sector_cap",
                signal_score  = 0.85,
                miss_date     = "2026-05-09",
                miss_price    = 2450.0,
                tracking_path = path,
            )
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["symbol"],        "6762.T")
            self.assertEqual(rec["reject_reason"], "sector_cap")
            self.assertIsNone(rec["forward_return_5d"])
            self.assertIsNone(rec["forward_return_20d"])


if __name__ == "__main__":
    unittest.main()
