"""
tests/test_regime_drift.py
Deterministic tests for regime drift invalidation.

Covers:
  - MarketRegime enum (all values, validation, coerce)
  - check_regime_drift: regime change drop
  - check_regime_drift: RSR rank decay drop (> 20 limit)
  - check_regime_drift: sector rank decay drop (> 10 limit)
  - check_regime_drift: boundary values (exactly at limit → pass)
  - check_regime_drift: None fields skip checks (backward compatibility)
  - check_regime_drift: no drop when context unchanged
  - evaluate_retry_candidates: drift check wired in correctly
  - context field serialisation roundtrip via from_dict / to_dict
  - make_deferred_entry accepts optional context kwargs
  - DeferredMetrics carries regime drift counters
  - build_metrics accepts drift counters
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile

from src.execution.deferred_queue import (
    DeferredEntry, make_deferred_entry, DeferredQueue,
)
from src.execution.regime_drift import (
    MarketRegime, RegimeDriftResult, check_regime_drift,
    RSR_RANK_DRIFT_LIMIT, SECTOR_RANK_DRIFT_LIMIT,
)
from src.execution.deferred_retry import RetryEvaluation, evaluate_retry_candidates
from src.execution.deferred_metrics import build_metrics

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entry(
    symbol:                    str       = "6762.T",
    rsr_rank:                  int | None = None,
    sector_rank:               int | None = None,
    regime:                    str | None = None,
) -> DeferredEntry:
    return make_deferred_entry(
        symbol                    = symbol,
        sector                    = "電子部品",
        strategy                  = "turtle",
        signal_score              = 0.85,
        desired_weight            = 0.33,
        degraded_weight           = 0.10,
        reject_reason             = "sector_cap",
        rsr_rank_at_creation      = rsr_rank,
        sector_rank_at_creation   = sector_rank,
        market_regime_at_creation = regime,
    )


def _no_drift(result: RegimeDriftResult) -> None:
    assert not result.should_drop, f"Expected no drop, got reason={result.reason!r}"


def _drift(result: RegimeDriftResult, expected_reason: str) -> None:
    assert result.should_drop, "Expected drop but should_drop=False"
    assert result.reason == expected_reason, (
        f"Expected reason={expected_reason!r}, got {result.reason!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. MarketRegime enum
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketRegimeEnum(unittest.TestCase):
    def test_all_required_values_exist(self) -> None:
        required = {"risk_on", "risk_off", "commodity", "defensives",
                    "rates_up", "tech_leadership"}
        actual   = {m.value for m in MarketRegime}
        self.assertEqual(actual, required)

    def test_is_valid_accepts_all_values(self) -> None:
        for m in MarketRegime:
            self.assertTrue(MarketRegime.is_valid(m.value))

    def test_is_valid_rejects_unknown(self) -> None:
        self.assertFalse(MarketRegime.is_valid("unknown_regime"))
        self.assertFalse(MarketRegime.is_valid(""))
        self.assertFalse(MarketRegime.is_valid(None))

    def test_coerce_returns_enum_for_valid(self) -> None:
        result = MarketRegime.coerce("risk_on")
        self.assertEqual(result, MarketRegime.RISK_ON)

    def test_coerce_returns_none_for_invalid(self) -> None:
        self.assertIsNone(MarketRegime.coerce("bad"))
        self.assertIsNone(MarketRegime.coerce(None))

    def test_str_equality(self) -> None:
        self.assertEqual(MarketRegime.RISK_ON, "risk_on")
        self.assertNotEqual(MarketRegime.RISK_ON, "risk_off")

    def test_serialises_as_string(self) -> None:
        # str(Enum) in Python gives "MarketRegime.RISK_ON"; .value gives "risk_on"
        self.assertEqual(MarketRegime.RISK_ON.value, "risk_on")


# ─────────────────────────────────────────────────────────────────────────────
# 2. check_regime_drift — regime change
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeChangeDrop(unittest.TestCase):
    def test_regime_changed_triggers_drop(self) -> None:
        e = _entry(regime="risk_on")
        r = check_regime_drift(e, None, None, current_regime="risk_off")
        _drift(r, "regime_changed")

    def test_regime_unchanged_no_drop(self) -> None:
        e = _entry(regime="risk_on")
        r = check_regime_drift(e, None, None, current_regime="risk_on")
        _no_drift(r)

    def test_no_creation_regime_skips_check(self) -> None:
        e = _entry(regime=None)
        r = check_regime_drift(e, None, None, current_regime="risk_off")
        _no_drift(r)

    def test_no_current_regime_skips_check(self) -> None:
        e = _entry(regime="risk_on")
        r = check_regime_drift(e, None, None, current_regime=None)
        _no_drift(r)

    def test_all_regime_pairs_that_differ_trigger_drop(self) -> None:
        for a in MarketRegime:
            for b in MarketRegime:
                if a != b:
                    e = _entry(regime=a.value)
                    r = check_regime_drift(e, None, None, current_regime=b.value)
                    _drift(r, "regime_changed")

    def test_detail_contains_both_regimes(self) -> None:
        e = _entry(regime="risk_on")
        r = check_regime_drift(e, None, None, current_regime="risk_off")
        self.assertIn("risk_on",  r.detail)
        self.assertIn("risk_off", r.detail)
        self.assertIn(e.entry_id, r.detail)


# ─────────────────────────────────────────────────────────────────────────────
# 3. check_regime_drift — RSR rank decay
# ─────────────────────────────────────────────────────────────────────────────

class TestRSRRankDecay(unittest.TestCase):
    def test_rsr_drift_over_limit_triggers_drop(self) -> None:
        e = _entry(rsr_rank=5)
        r = check_regime_drift(e, current_rsr_rank=5 + RSR_RANK_DRIFT_LIMIT + 1,
                               current_sector_rank=None, current_regime=None)
        _drift(r, "rsr_rank_decay")

    def test_rsr_drift_exactly_at_limit_no_drop(self) -> None:
        e = _entry(rsr_rank=5)
        r = check_regime_drift(e, current_rsr_rank=5 + RSR_RANK_DRIFT_LIMIT,
                               current_sector_rank=None, current_regime=None)
        _no_drift(r)

    def test_rsr_drift_improvement_no_drop(self) -> None:
        e = _entry(rsr_rank=30)
        r = check_regime_drift(e, current_rsr_rank=5,  # much better
                               current_sector_rank=None, current_regime=None)
        _no_drift(r)

    def test_no_creation_rsr_rank_skips_check(self) -> None:
        e = _entry(rsr_rank=None)
        r = check_regime_drift(e, current_rsr_rank=99,
                               current_sector_rank=None, current_regime=None)
        _no_drift(r)

    def test_no_current_rsr_rank_skips_check(self) -> None:
        e = _entry(rsr_rank=5)
        r = check_regime_drift(e, current_rsr_rank=None,
                               current_sector_rank=None, current_regime=None)
        _no_drift(r)

    def test_detail_contains_ranks_and_drift(self) -> None:
        e = _entry(rsr_rank=1)
        r = check_regime_drift(e, current_rsr_rank=1 + RSR_RANK_DRIFT_LIMIT + 5,
                               current_sector_rank=None, current_regime=None)
        self.assertIn("rsr_rank_at_creation=1", r.detail)
        self.assertIn(e.symbol, r.detail)

    def test_rsr_limit_constant_is_20(self) -> None:
        self.assertEqual(RSR_RANK_DRIFT_LIMIT, 20)


# ─────────────────────────────────────────────────────────────────────────────
# 4. check_regime_drift — sector rank decay
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorRankDecay(unittest.TestCase):
    def test_sector_drift_over_limit_triggers_drop(self) -> None:
        e = _entry(sector_rank=3)
        r = check_regime_drift(e, current_rsr_rank=None,
                               current_sector_rank=3 + SECTOR_RANK_DRIFT_LIMIT + 1,
                               current_regime=None)
        _drift(r, "sector_rank_decay")

    def test_sector_drift_exactly_at_limit_no_drop(self) -> None:
        e = _entry(sector_rank=3)
        r = check_regime_drift(e, current_rsr_rank=None,
                               current_sector_rank=3 + SECTOR_RANK_DRIFT_LIMIT,
                               current_regime=None)
        _no_drift(r)

    def test_sector_drift_improvement_no_drop(self) -> None:
        e = _entry(sector_rank=15)
        r = check_regime_drift(e, current_rsr_rank=None,
                               current_sector_rank=2,
                               current_regime=None)
        _no_drift(r)

    def test_no_creation_sector_rank_skips_check(self) -> None:
        e = _entry(sector_rank=None)
        r = check_regime_drift(e, current_rsr_rank=None,
                               current_sector_rank=99,
                               current_regime=None)
        _no_drift(r)

    def test_sector_limit_constant_is_10(self) -> None:
        self.assertEqual(SECTOR_RANK_DRIFT_LIMIT, 10)

    def test_detail_contains_sector_ranks(self) -> None:
        e = _entry(sector_rank=1)
        r = check_regime_drift(e, current_rsr_rank=None,
                               current_sector_rank=1 + SECTOR_RANK_DRIFT_LIMIT + 3,
                               current_regime=None)
        self.assertIn("sector_rank_at_creation=1", r.detail)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Guard ordering — regime checked before rank
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardOrdering(unittest.TestCase):
    def test_regime_change_fires_before_rsr_decay(self) -> None:
        # Both conditions true — regime should fire first
        e = _entry(rsr_rank=1, sector_rank=1, regime="risk_on")
        r = check_regime_drift(
            e,
            current_rsr_rank    = 1 + RSR_RANK_DRIFT_LIMIT + 10,
            current_sector_rank = 1 + SECTOR_RANK_DRIFT_LIMIT + 10,
            current_regime      = "risk_off",
        )
        _drift(r, "regime_changed")

    def test_rsr_decay_fires_before_sector_decay(self) -> None:
        # No regime change, both rank conditions true
        e = _entry(rsr_rank=1, sector_rank=1, regime=None)
        r = check_regime_drift(
            e,
            current_rsr_rank    = 1 + RSR_RANK_DRIFT_LIMIT + 5,
            current_sector_rank = 1 + SECTOR_RANK_DRIFT_LIMIT + 5,
            current_regime      = None,
        )
        _drift(r, "rsr_rank_decay")

    def test_all_none_context_never_drops(self) -> None:
        e = _entry(rsr_rank=None, sector_rank=None, regime=None)
        r = check_regime_drift(e, current_rsr_rank=999,
                               current_sector_rank=999, current_regime="risk_off")
        _no_drift(r)


# ─────────────────────────────────────────────────────────────────────────────
# 6. evaluate_retry_candidates — drift check integrated
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftInEvaluateRetry(unittest.TestCase):
    def _base_kw(self, q: DeferredQueue) -> dict:
        return dict(
            queue              = q,
            signal_scores      = {"6762.T": 0.9},
            tradeable_symbols  = {"6762.T"},
            sector_weights     = {"電子部品": 0.05},
            sector_caps        = {"電子部品": 0.25},
            current_positions  = {},
            max_open           = 3,
            portfolio_equity   = 3_000_000.0,
            available_cash     = 3_000_000.0,
        )

    def test_regime_change_drops_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry(regime="risk_on")
            q.add(e)
            kw = self._base_kw(q)
            kw["current_regime"] = "risk_off"
            results = evaluate_retry_candidates(**kw)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "regime_changed")

    def test_rsr_decay_drops_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry(rsr_rank=5)
            q.add(e)
            kw = self._base_kw(q)
            kw["current_rsr_ranks"] = {"6762.T": 5 + RSR_RANK_DRIFT_LIMIT + 1}
            results = evaluate_retry_candidates(**kw)
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "rsr_rank_decay")

    def test_sector_decay_drops_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry(sector_rank=2)
            q.add(e)
            kw = self._base_kw(q)
            kw["current_sector_ranks"] = {"6762.T": 2 + SECTOR_RANK_DRIFT_LIMIT + 1}
            results = evaluate_retry_candidates(**kw)
            self.assertFalse(results[0].should_retry)
            self.assertEqual(results[0].drop_reason, "sector_rank_decay")

    def test_no_context_fields_no_drift_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry()  # all None context
            q.add(e)
            kw = self._base_kw(q)
            kw["current_rsr_ranks"]    = {"6762.T": 999}
            kw["current_sector_ranks"] = {"6762.T": 999}
            kw["current_regime"]       = "risk_off"
            results = evaluate_retry_candidates(**kw)
            self.assertTrue(results[0].should_retry)

    def test_without_drift_kwargs_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry(regime="risk_on")
            q.add(e)
            # Call WITHOUT any drift kwargs — should still work
            results = evaluate_retry_candidates(**self._base_kw(q))
            self.assertTrue(results[0].should_retry)

    def test_rank_stable_entry_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry(rsr_rank=5, sector_rank=3, regime="risk_on")
            q.add(e)
            kw = self._base_kw(q)
            kw["current_rsr_ranks"]    = {"6762.T": 5 + RSR_RANK_DRIFT_LIMIT}   # exactly at limit
            kw["current_sector_ranks"] = {"6762.T": 3 + SECTOR_RANK_DRIFT_LIMIT}  # exactly at limit
            kw["current_regime"]       = "risk_on"
            results = evaluate_retry_candidates(**kw)
            self.assertTrue(results[0].should_retry)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Serialisation roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestContextFieldSerialisation(unittest.TestCase):
    def test_all_context_fields_preserved(self) -> None:
        e = _entry(rsr_rank=7, sector_rank=3, regime="risk_on")
        d = e.to_dict()
        restored = DeferredEntry.from_dict(d)
        self.assertEqual(restored.rsr_rank_at_creation,     7)
        self.assertEqual(restored.sector_rank_at_creation,  3)
        self.assertEqual(restored.market_regime_at_creation, "risk_on")

    def test_none_context_fields_preserved_as_none(self) -> None:
        e = _entry()
        d = e.to_dict()
        restored = DeferredEntry.from_dict(d)
        self.assertIsNone(restored.rsr_rank_at_creation)
        self.assertIsNone(restored.sector_rank_at_creation)
        self.assertIsNone(restored.market_regime_at_creation)

    def test_old_entry_without_context_fields_loads_safely(self) -> None:
        raw = {
            "entry_id": "abc123", "symbol": "6762.T", "sector": "電子部品",
            "strategy": "turtle", "signal_score": 0.8,
            "desired_weight": 0.33, "degraded_weight": 0.10,
            "reject_reason": "sector_cap", "created_at": "2026-05-01T09:00:00+0900",
            "ttl_days": 3,
        }
        e = DeferredEntry.from_dict(raw)
        self.assertIsNone(e.rsr_rank_at_creation)
        self.assertIsNone(e.sector_rank_at_creation)
        self.assertIsNone(e.market_regime_at_creation)

    def test_context_fields_persist_via_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = DeferredQueue(Path(tmp) / "q.json")
            e = _entry(rsr_rank=12, sector_rank=4, regime="tech_leadership")
            q.add(e)
            loaded = q.load()[0]
            self.assertEqual(loaded.rsr_rank_at_creation,     12)
            self.assertEqual(loaded.sector_rank_at_creation,   4)
            self.assertEqual(loaded.market_regime_at_creation, "tech_leadership")


# ─────────────────────────────────────────────────────────────────────────────
# 8. DeferredMetrics — regime drift counters
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeDriftMetrics(unittest.TestCase):
    def _base_metrics_kw(self) -> dict:
        return dict(
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

    def test_default_drift_counters_are_zero(self) -> None:
        m = build_metrics(**self._base_metrics_kw())
        self.assertEqual(m.deferred_regime_invalidations, 0)
        self.assertEqual(m.deferred_rank_decay_drops, 0)

    def test_regime_invalidation_counter_set(self) -> None:
        kw = self._base_metrics_kw()
        kw["deferred_regime_invalidations"] = 3
        m  = build_metrics(**kw)
        self.assertEqual(m.deferred_regime_invalidations, 3)

    def test_rank_decay_counter_set(self) -> None:
        kw = self._base_metrics_kw()
        kw["deferred_rank_decay_drops"] = 5
        m  = build_metrics(**kw)
        self.assertEqual(m.deferred_rank_decay_drops, 5)

    def test_to_dict_includes_drift_counters(self) -> None:
        kw = self._base_metrics_kw()
        kw["deferred_regime_invalidations"] = 2
        kw["deferred_rank_decay_drops"]     = 4
        d = build_metrics(**kw).to_dict()
        self.assertEqual(d["deferred_regime_invalidations"], 2)
        self.assertEqual(d["deferred_rank_decay_drops"],     4)

    def test_backward_compat_no_drift_kwargs(self) -> None:
        # old callers that don't pass drift counters must still work
        m = build_metrics(**self._base_metrics_kw())
        self.assertEqual(m.deferred_regime_invalidations, 0)
        self.assertEqual(m.deferred_rank_decay_drops,     0)


if __name__ == "__main__":
    unittest.main()
