"""
tests/analytics/test_alpha_leakage.py

Tests for the deployable-alpha leakage analytics layer:
  - src/analytics/skipped_opportunity_analytics.py
  - src/analytics/deployable_alpha_metrics.py
  - src/analytics/daily_leakage_summary.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.analytics.skipped_opportunity_analytics import (
    ENRICHMENT_ENRICHED,
    ENRICHMENT_EXPIRED,
    ENRICHMENT_LOOKBACK_DAYS,
    ENRICHMENT_PENDING,
    REJECTION_ALLOC_CAP,
    REJECTION_CONCENTRATION_THROTTLE,
    REJECTION_DUPLICATE,
    REJECTION_EPOCH_BLOCKED,
    REJECTION_HIGH_PRICE,
    REJECTION_INSUFFICIENT_CASH,
    REJECTION_RATE_LIMIT,
    REJECTION_SECTOR_SUPPRESSION,
    REJECTION_STALE_SIGNAL,
    REJECTION_TRUTH_CONFIDENCE,
    SkippedOpportunityRecord,
    append_skipped_opportunity,
    deduplicate_records,
    enrich_forward_returns,
    load_skipped_opportunities,
    load_skipped_opportunities_for_date,
)
from src.analytics.deployable_alpha_metrics import (
    DeployableAlphaMetrics,
    append_alpha_metrics,
    compute_alpha_metrics,
    load_alpha_metrics,
)
from src.analytics.daily_leakage_summary import (
    DailyLeakageSummary,
    append_daily_summary,
    generate_daily_summary,
    get_summary_for_date,
    load_daily_summaries,
    log_daily_summary,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

FIXED_TS = "2026-05-17T09:00:00+00:00"
FIXED_DATE = "2026-05-17"
RUN_ID = "run-analytics-001"
CAPITAL = 3_000_000.0


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def _make_record(
    symbol: str = "1234.T",
    rejection_reason: str = REJECTION_ALLOC_CAP,
    signal_strength: float = 80.0,
    predicted_rank: int = 1,
    available_cash: float = 500_000.0,
    alloc_cap: int = 0,
    intended_size: int = 100,
    price: float = 2500.0,
    run_id: str = RUN_ID,
    ts: str = FIXED_TS,
    sector: str = "輸送用機器",
    concentration: float = 0.1,
) -> SkippedOpportunityRecord:
    return SkippedOpportunityRecord.create(
        run_id=run_id,
        timestamp=ts,
        symbol=symbol,
        strategy_id="fujiko_v2",
        signal_strength=signal_strength,
        predicted_rank=predicted_rank,
        rejection_reason=rejection_reason,
        available_cash=available_cash,
        alloc_cap=alloc_cap,
        intended_position_size=intended_size,
        sector_state=sector,
        concentration_state=concentration,
        price_at_signal=price,
    )


def _make_price_history(
    price_at_signal: float = 2500.0,
    returns: List[float] = None,
) -> List[float]:
    """prices[0]=entry, prices[1..5]=forward prices"""
    if returns is None:
        returns = [0.01, 0.02, 0.015, 0.025, 0.03]
    return [price_at_signal] + [round(price_at_signal * (1 + r), 2) for r in returns]


# ══════════════════════════════════════════════════════════════════════════════
# 1. SkippedOpportunityRecord — creation and determinism
# ══════════════════════════════════════════════════════════════════════════════

class TestSkippedOpportunityRecord:
    def test_record_id_deterministic(self):
        r1 = _make_record()
        r2 = _make_record()
        assert r1.record_id == r2.record_id

    def test_different_symbol_different_id(self):
        r1 = _make_record(symbol="1234.T")
        r2 = _make_record(symbol="5678.T")
        assert r1.record_id != r2.record_id

    def test_different_reason_different_id(self):
        r1 = _make_record(rejection_reason=REJECTION_ALLOC_CAP)
        r2 = _make_record(rejection_reason=REJECTION_TRUTH_CONFIDENCE)
        assert r1.record_id != r2.record_id

    def test_different_run_id_different_record_id(self):
        r1 = _make_record(run_id="run-A")
        r2 = _make_record(run_id="run-B")
        assert r1.record_id != r2.record_id

    def test_initial_enrichment_status_is_pending(self):
        r = _make_record()
        assert r.enrichment_status == ENRICHMENT_PENDING

    def test_initial_forward_returns_are_none(self):
        r = _make_record()
        assert r.forward_return_1d is None
        assert r.forward_return_3d is None
        assert r.forward_return_5d is None
        assert r.estimated_missed_pnl is None

    def test_all_rejection_reasons_are_distinct_strings(self):
        reasons = [
            REJECTION_ALLOC_CAP, REJECTION_HIGH_PRICE, REJECTION_SECTOR_SUPPRESSION,
            REJECTION_CONCENTRATION_THROTTLE, REJECTION_INSUFFICIENT_CASH,
            REJECTION_STALE_SIGNAL, REJECTION_TRUTH_CONFIDENCE,
            REJECTION_RATE_LIMIT, REJECTION_DUPLICATE, REJECTION_EPOCH_BLOCKED,
        ]
        assert len(set(reasons)) == 10

    def test_record_captures_all_required_fields(self):
        r = _make_record(
            symbol="9201.T",
            signal_strength=85.5,
            predicted_rank=3,
            available_cash=750_000.0,
            alloc_cap=50,
            intended_size=200,
            price=3800.0,
            sector="電気機器",
            concentration=0.22,
        )
        assert r.symbol == "9201.T"
        assert r.signal_strength == 85.5
        assert r.predicted_rank == 3
        assert r.available_cash == 750_000.0
        assert r.alloc_cap == 50
        assert r.intended_position_size == 200
        assert r.sector_state == "電気機器"
        assert r.concentration_state == 0.22
        assert r.price_at_signal == 3800.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Forward return enrichment — correctness and determinism
# ══════════════════════════════════════════════════════════════════════════════

class TestForwardReturnEnrichment:
    def test_enrichment_fills_forward_returns(self):
        rec = _make_record(price=2500.0, intended_size=100)
        prices = _make_price_history(2500.0, [0.01, 0.00, 0.02, 0.00, 0.03])
        enriched = enrich_forward_returns([rec], {"1234.T": prices})
        assert len(enriched) == 1
        e = enriched[0]
        assert e.enrichment_status == ENRICHMENT_ENRICHED
        assert e.forward_return_1d == pytest.approx(0.01, rel=1e-4)
        assert e.forward_return_3d == pytest.approx(0.02, rel=1e-4)
        assert e.forward_return_5d == pytest.approx(0.03, rel=1e-4)

    def test_enrichment_deterministic_same_inputs(self):
        rec = _make_record(price=2500.0)
        prices = _make_price_history(2500.0, [0.01, 0.02, 0.015, 0.025, 0.03])
        e1 = enrich_forward_returns([rec], {"1234.T": prices})
        e2 = enrich_forward_returns([rec], {"1234.T": prices})
        assert e1[0].forward_return_5d == e2[0].forward_return_5d
        assert e1[0].estimated_missed_pnl == e2[0].estimated_missed_pnl

    def test_mfe_is_max_of_window(self):
        rec = _make_record(price=1000.0)
        returns = [0.01, -0.02, 0.05, -0.01, 0.03]
        prices = _make_price_history(1000.0, returns)
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert e.max_favorable_excursion == pytest.approx(0.05, rel=1e-4)

    def test_mae_is_min_of_window(self):
        rec = _make_record(price=1000.0)
        returns = [0.01, -0.02, 0.05, -0.03, 0.03]
        prices = _make_price_history(1000.0, returns)
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert e.max_adverse_excursion == pytest.approx(-0.03, rel=1e-4)

    def test_estimated_missed_pnl_correct(self):
        rec = _make_record(price=2000.0, intended_size=100)
        # forward_return_5d = 0.05 → hypothetical_return = 0.05
        prices = [2000.0, 2020.0, 2040.0, 2060.0, 2080.0, 2100.0]
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        # hyp = (2100-2000)/2000 = 0.05; pnl = 100 * 2000 * 0.05 = 10000
        assert e.estimated_missed_pnl == pytest.approx(10000.0, rel=1e-4)

    def test_hypothetical_return_uses_5d_when_available(self):
        rec = _make_record(price=1000.0)
        prices = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert e.hypothetical_return == pytest.approx(0.05, rel=1e-4)

    def test_hypothetical_return_falls_back_to_3d(self):
        rec = _make_record(price=1000.0)
        prices = [1000.0, 1010.0, 1020.0, 1030.0]  # only 4 prices (t0,t1,t2,t3)
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert e.forward_return_5d is None
        assert e.hypothetical_return == pytest.approx(0.03, rel=1e-4)

    def test_hypothetical_return_falls_back_to_1d(self):
        rec = _make_record(price=1000.0)
        prices = [1000.0, 1010.0]  # only t0, t+1
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert e.forward_return_3d is None
        assert e.forward_return_5d is None
        assert e.hypothetical_return == pytest.approx(0.01, rel=1e-4)

    def test_missing_symbol_stays_pending(self):
        rec = _make_record(symbol="9999.T")
        # pin now_ts to 1 day after record to prevent expiry check from triggering
        now_ts = "2026-05-18T09:00:00+00:00"
        e = enrich_forward_returns([rec], {}, now_ts=now_ts)[0]
        assert e.enrichment_status == ENRICHMENT_PENDING

    def test_already_enriched_record_unchanged(self):
        rec = _make_record()
        prices = _make_price_history()
        enriched = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert enriched.enrichment_status == ENRICHMENT_ENRICHED
        # Re-enrich should not change values
        re_enriched = enrich_forward_returns([enriched], {"1234.T": prices})[0]
        assert re_enriched.forward_return_5d == enriched.forward_return_5d

    def test_expired_record_marked_expired(self):
        old_ts = "2020-01-01T09:00:00+00:00"
        rec = _make_record(ts=old_ts)
        now_ts = datetime.now(timezone.utc).isoformat()
        e = enrich_forward_returns([rec], {}, now_ts=now_ts)[0]
        assert e.enrichment_status == ENRICHMENT_EXPIRED

    def test_forward_return_negative_captured(self):
        rec = _make_record(price=2000.0, intended_size=100)
        prices = [2000.0, 1980.0, 1960.0, 1940.0, 1920.0, 1900.0]
        e = enrich_forward_returns([rec], {"1234.T": prices})[0]
        assert e.forward_return_5d < 0
        assert e.estimated_missed_pnl is not None
        assert e.estimated_missed_pnl < 0

    def test_zero_entry_price_no_crash(self):
        rec = _make_record(price=0.0)
        prices = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        result = enrich_forward_returns([rec], {"1234.T": prices})
        assert len(result) == 1  # no crash

    def test_empty_records_empty_result(self):
        result = enrich_forward_returns([], {"1234.T": [1000.0, 1010.0]})
        assert result == []

    def test_enrichment_does_not_mutate_original(self):
        rec = _make_record(price=2500.0)
        prices = _make_price_history()
        enrich_forward_returns([rec], {"1234.T": prices})
        # Original still pending
        assert rec.enrichment_status == ENRICHMENT_PENDING
        assert rec.forward_return_1d is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Deduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    def test_duplicate_records_deduplicated(self):
        r1 = _make_record()
        r2 = _make_record()  # same record_id
        assert r1.record_id == r2.record_id
        result = deduplicate_records([r1, r2])
        assert len(result) == 1

    def test_different_records_both_kept(self):
        r1 = _make_record(symbol="1234.T")
        r2 = _make_record(symbol="5678.T")
        result = deduplicate_records([r1, r2])
        assert len(result) == 2

    def test_enriched_preferred_over_pending(self):
        rec = _make_record(price=2500.0)
        prices = _make_price_history()
        enriched = enrich_forward_returns([rec], {"1234.T": prices})[0]
        result = deduplicate_records([rec, enriched])
        assert len(result) == 1
        assert result[0].enrichment_status == ENRICHMENT_ENRICHED

    def test_output_sorted_by_timestamp(self):
        r1 = _make_record(symbol="1234.T", ts="2026-05-17T09:00:00+00:00")
        r2 = _make_record(symbol="5678.T", ts="2026-05-17T08:00:00+00:00")
        result = deduplicate_records([r1, r2])
        assert result[0].timestamp < result[1].timestamp


# ══════════════════════════════════════════════════════════════════════════════
# 4. Persistence — append-only, load, date filter, missing data
# ══════════════════════════════════════════════════════════════════════════════

class TestSkippedOpportunityPersistence:
    def test_roundtrip_single_record(self, tmp: Path):
        path = tmp / "skipped.jsonl"
        rec = _make_record()
        append_skipped_opportunity(rec, path)
        loaded = load_skipped_opportunities(path)
        assert len(loaded) == 1
        assert loaded[0].record_id == rec.record_id
        assert loaded[0].symbol == rec.symbol
        assert loaded[0].enrichment_status == ENRICHMENT_PENDING

    def test_multiple_records_append_order_preserved(self, tmp: Path):
        path = tmp / "skipped.jsonl"
        symbols = ["1234.T", "5678.T", "9012.T"]
        for i, sym in enumerate(symbols):
            ts = f"2026-05-17T09:0{i}:00+00:00"
            append_skipped_opportunity(_make_record(symbol=sym, ts=ts), path)
        loaded = load_skipped_opportunities(path)
        assert len(loaded) == 3
        assert [r.symbol for r in loaded] == symbols

    def test_enriched_record_roundtrip(self, tmp: Path):
        path = tmp / "skipped.jsonl"
        rec = _make_record(price=2500.0, intended_size=100)
        prices = _make_price_history()
        enriched = enrich_forward_returns([rec], {"1234.T": prices})[0]
        append_skipped_opportunity(enriched, path)
        loaded = load_skipped_opportunities(path)
        assert loaded[0].enrichment_status == ENRICHMENT_ENRICHED
        assert loaded[0].forward_return_5d is not None
        assert loaded[0].estimated_missed_pnl is not None

    def test_missing_file_returns_empty(self, tmp: Path):
        assert load_skipped_opportunities(tmp / "missing.jsonl") == []

    def test_corrupted_line_skipped(self, tmp: Path):
        path = tmp / "skipped.jsonl"
        append_skipped_opportunity(_make_record(), path)
        with path.open("a") as f:
            f.write("not json\n")
        loaded = load_skipped_opportunities(path)
        assert len(loaded) == 1

    def test_load_for_date_filters_correctly(self, tmp: Path):
        path = tmp / "skipped.jsonl"
        r1 = _make_record(symbol="1234.T", ts="2026-05-17T09:00:00+00:00")
        r2 = _make_record(symbol="5678.T", ts="2026-05-18T09:00:00+00:00")
        append_skipped_opportunity(r1, path)
        append_skipped_opportunity(r2, path)
        today = load_skipped_opportunities_for_date(path, "2026-05-17")
        assert len(today) == 1
        assert today[0].symbol == "1234.T"

    def test_write_failure_does_not_raise(self, tmp: Path):
        # Write to a non-existent deep path — should fail-safe
        path = tmp / "a" / "b" / "c" / "d" / "skipped.jsonl"
        rec = _make_record()
        # Should not raise (creates dirs)
        append_skipped_opportunity(rec, path)
        assert path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# 5. DeployableAlphaMetrics — computation correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestDeployableAlphaMetrics:
    def test_zero_skipped_zero_executed(self):
        metrics = compute_alpha_metrics([], 0)
        assert metrics.opportunity_utilization_ratio == 0.0
        assert metrics.skipped_count == 0
        assert metrics.executed_count == 0

    def test_utilization_ratio_all_executed(self):
        metrics = compute_alpha_metrics([], executed_count=5)
        assert metrics.opportunity_utilization_ratio == 1.0

    def test_utilization_ratio_half_executed(self):
        skipped = [_make_record() for _ in range(3)]
        metrics = compute_alpha_metrics(skipped, executed_count=3)
        assert metrics.opportunity_utilization_ratio == pytest.approx(0.5, rel=1e-6)

    def test_utilization_ratio_all_skipped(self):
        skipped = [_make_record(symbol=str(i)) for i in range(4)]
        metrics = compute_alpha_metrics(skipped, executed_count=0)
        assert metrics.opportunity_utilization_ratio == 0.0

    def test_skipped_pnl_estimate_sum_of_enriched(self):
        rec1 = _make_record(symbol="1234.T", price=1000.0, intended_size=100)
        rec2 = _make_record(symbol="5678.T", price=2000.0, intended_size=50)
        prices1 = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]  # 5d=+5%
        prices2 = [2000.0, 2020.0, 2040.0, 2060.0, 2080.0, 2100.0]  # 5d=+5%
        enriched1 = enrich_forward_returns([rec1], {"1234.T": prices1})[0]
        enriched2 = enrich_forward_returns([rec2], {"5678.T": prices2})[0]
        # pnl1 = 100 * 1000 * 0.05 = 5000; pnl2 = 50 * 2000 * 0.05 = 5000; total=10000
        metrics = compute_alpha_metrics([enriched1, enriched2], executed_count=0)
        assert metrics.skipped_pnl_estimate == pytest.approx(10000.0, rel=1e-4)

    def test_skipped_pnl_ignores_pending_records(self):
        pending = _make_record()
        assert pending.estimated_missed_pnl is None
        metrics = compute_alpha_metrics([pending], executed_count=0)
        assert metrics.skipped_pnl_estimate == 0.0

    def test_conversion_rate_zero_when_no_realized_pnl(self):
        rec = _make_record(price=1000.0, intended_size=100)
        prices = [1000.0] * 6
        enriched = enrich_forward_returns([rec], {"1234.T": prices})[0]
        metrics = compute_alpha_metrics([enriched], executed_count=1, realized_pnl=0.0)
        assert metrics.deployable_alpha_conversion_rate == 0.0

    def test_conversion_rate_one_when_no_missed_pnl(self):
        metrics = compute_alpha_metrics([], executed_count=1, realized_pnl=5000.0)
        assert metrics.deployable_alpha_conversion_rate == 1.0

    def test_idle_cash_ratio_computed_from_skipped_cash(self):
        r1 = _make_record(available_cash=1_000_000.0)
        r2 = _make_record(symbol="5678.T", available_cash=2_000_000.0)
        metrics = compute_alpha_metrics([r1, r2], executed_count=0, capital=3_000_000.0)
        # mean cash = 1.5M; ratio = 1.5M / 3M = 0.5
        assert metrics.idle_cash_ratio == pytest.approx(0.5, rel=1e-6)
        assert metrics.mean_available_cash_at_skip == pytest.approx(1_500_000.0, rel=1e-6)

    def test_exclusion_reason_distribution_normalized(self):
        reasons = [REJECTION_ALLOC_CAP, REJECTION_ALLOC_CAP, REJECTION_TRUTH_CONFIDENCE]
        skipped = [_make_record(rejection_reason=r, symbol=str(i)) for i, r in enumerate(reasons)]
        metrics = compute_alpha_metrics(skipped, executed_count=0)
        dist = metrics.exclusion_reason_distribution
        assert dist[REJECTION_ALLOC_CAP] == pytest.approx(2/3, rel=1e-6)
        assert dist[REJECTION_TRUTH_CONFIDENCE] == pytest.approx(1/3, rel=1e-6)
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_enrichment_coverage_zero_when_all_pending(self):
        skipped = [_make_record(symbol=str(i)) for i in range(3)]
        metrics = compute_alpha_metrics(skipped, executed_count=0)
        assert metrics.enrichment_coverage == 0.0

    def test_enrichment_coverage_one_when_all_enriched(self):
        records = []
        for i in range(3):
            rec = _make_record(symbol=str(i), price=1000.0)
            prices = _make_price_history(1000.0)
            enriched = enrich_forward_returns([rec], {str(i): prices})[0]
            records.append(enriched)
        metrics = compute_alpha_metrics(records, executed_count=0)
        assert metrics.enrichment_coverage == 1.0

    def test_metrics_deterministic(self):
        skipped = [_make_record(symbol=str(i)) for i in range(3)]
        m1 = compute_alpha_metrics(skipped, executed_count=2, capital=CAPITAL)
        m2 = compute_alpha_metrics(skipped, executed_count=2, capital=CAPITAL)
        assert m1.opportunity_utilization_ratio == m2.opportunity_utilization_ratio
        assert m1.exclusion_reason_distribution == m2.exclusion_reason_distribution


class TestAlphaMetricsPersistence:
    def test_roundtrip(self, tmp: Path):
        path = tmp / "metrics.jsonl"
        metrics = compute_alpha_metrics([], executed_count=5, capital=CAPITAL)
        append_alpha_metrics(metrics, path)
        loaded = load_alpha_metrics(path)
        assert len(loaded) == 1
        assert loaded[0].executed_count == 5
        assert loaded[0].opportunity_utilization_ratio == 1.0

    def test_missing_file_returns_empty(self, tmp: Path):
        assert load_alpha_metrics(tmp / "missing.jsonl") == []

    def test_corrupted_line_skipped(self, tmp: Path):
        path = tmp / "metrics.jsonl"
        metrics = compute_alpha_metrics([], executed_count=1)
        append_alpha_metrics(metrics, path)
        with path.open("a") as f:
            f.write("bad json\n")
        loaded = load_alpha_metrics(path)
        assert len(loaded) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 6. DailyLeakageSummary — generation and determinism
# ══════════════════════════════════════════════════════════════════════════════

class TestDailyLeakageSummary:
    def test_summary_id_deterministic(self):
        s1 = generate_daily_summary(FIXED_DATE, RUN_ID, [], 0)
        s2 = generate_daily_summary(FIXED_DATE, RUN_ID, [], 0)
        assert s1.summary_id == s2.summary_id

    def test_different_date_different_summary_id(self):
        s1 = generate_daily_summary("2026-05-17", RUN_ID, [], 0)
        s2 = generate_daily_summary("2026-05-18", RUN_ID, [], 0)
        assert s1.summary_id != s2.summary_id

    def test_counts_correct(self):
        skipped = [_make_record(symbol=str(i)) for i in range(3)]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, skipped, executed_count=2)
        assert summary.skipped_count == 3
        assert summary.executed_count == 2
        assert summary.total_signals_seen == 5

    def test_top_missed_by_signal_sorted_descending(self):
        r1 = _make_record(symbol="A", signal_strength=90.0)
        r2 = _make_record(symbol="B", signal_strength=70.0)
        r3 = _make_record(symbol="C", signal_strength=80.0)
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [r1, r2, r3], 0)
        top = summary.top_missed_by_signal
        assert top[0]["signal_strength"] >= top[1]["signal_strength"]
        assert top[1]["signal_strength"] >= top[2]["signal_strength"]

    def test_top_missed_by_signal_capped_at_top_n(self):
        skipped = [_make_record(symbol=str(i), signal_strength=float(i)) for i in range(10)]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, skipped, 0, top_n=3)
        assert len(summary.top_missed_by_signal) == 3

    def test_top_missed_by_pnl_only_enriched(self):
        rec_pending = _make_record(symbol="pending.T")
        rec_enriched = _make_record(symbol="enriched.T", price=1000.0, intended_size=100)
        prices = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1060.0]  # 5d=+6%
        enriched = enrich_forward_returns([rec_enriched], {"enriched.T": prices})[0]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [rec_pending, enriched], 0)
        # Only enriched record in top_by_pnl
        assert len(summary.top_missed_by_pnl) == 1
        assert summary.top_missed_by_pnl[0]["symbol"] == "enriched.T"

    def test_exclusion_breakdown_all_reasons_present(self):
        reasons = [REJECTION_ALLOC_CAP, REJECTION_TRUTH_CONFIDENCE, REJECTION_HIGH_PRICE]
        skipped = [_make_record(rejection_reason=r, symbol=str(i)) for i, r in enumerate(reasons)]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, skipped, 0)
        bd_reasons = {entry["reason"] for entry in summary.exclusion_breakdown}
        for r in reasons:
            assert r in bd_reasons

    def test_idle_capital_peak_is_max_cash(self):
        r1 = _make_record(available_cash=500_000.0)
        r2 = _make_record(symbol="5678.T", available_cash=1_500_000.0)
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [r1, r2], 0)
        assert summary.idle_capital_peak_jpy == pytest.approx(1_500_000.0, rel=1e-6)

    def test_idle_capital_mean_is_average(self):
        r1 = _make_record(available_cash=500_000.0)
        r2 = _make_record(symbol="5678.T", available_cash=1_500_000.0)
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [r1, r2], 0)
        assert summary.idle_capital_mean_jpy == pytest.approx(1_000_000.0, rel=1e-6)

    def test_empty_skipped_no_crash(self):
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [], executed_count=5)
        assert summary.skipped_count == 0
        assert summary.executed_count == 5
        assert summary.opportunity_utilization_ratio == 1.0

    def test_opportunity_utilization_matches_metrics(self):
        skipped = [_make_record(symbol=str(i)) for i in range(3)]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, skipped, executed_count=3)
        assert summary.opportunity_utilization_ratio == pytest.approx(0.5, rel=1e-6)

    def test_skipped_pnl_matches_enriched_sum(self):
        rec = _make_record(symbol="1234.T", price=1000.0, intended_size=100)
        prices = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1100.0]  # 5d=+10%
        enriched = enrich_forward_returns([rec], {"1234.T": prices})[0]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [enriched], executed_count=0)
        # pnl = 100 * 1000 * 0.10 = 10000
        assert summary.skipped_pnl_estimate == pytest.approx(10000.0, rel=1e-4)

    def test_log_daily_summary_does_not_raise(self):
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [], 0)
        log_daily_summary(summary)  # should not raise


class TestDailyLeakageSummaryPersistence:
    def test_roundtrip(self, tmp: Path):
        path = tmp / "leakage.jsonl"
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, [], 5)
        append_daily_summary(summary, path)
        loaded = load_daily_summaries(path)
        assert len(loaded) == 1
        assert loaded[0].summary_id == summary.summary_id
        assert loaded[0].date == FIXED_DATE

    def test_multiple_summaries_preserved(self, tmp: Path):
        path = tmp / "leakage.jsonl"
        for i in range(3):
            date = f"2026-05-1{i + 5}"
            s = generate_daily_summary(date, RUN_ID, [], i)
            append_daily_summary(s, path)
        loaded = load_daily_summaries(path)
        assert len(loaded) == 3

    def test_get_summary_for_date_returns_latest(self, tmp: Path):
        path = tmp / "leakage.jsonl"
        s1 = generate_daily_summary(FIXED_DATE, "run-A", [], 1)
        s2 = generate_daily_summary(FIXED_DATE, "run-B", [], 2)
        append_daily_summary(s1, path)
        append_daily_summary(s2, path)
        result = get_summary_for_date(path, FIXED_DATE)
        assert result is not None
        assert result.run_id == "run-B"

    def test_get_summary_for_date_returns_none_when_missing(self, tmp: Path):
        path = tmp / "leakage.jsonl"
        append_daily_summary(generate_daily_summary(FIXED_DATE, RUN_ID, [], 0), path)
        assert get_summary_for_date(path, "2099-01-01") is None

    def test_missing_file_returns_empty(self, tmp: Path):
        assert load_daily_summaries(tmp / "missing.jsonl") == []

    def test_corrupted_line_skipped(self, tmp: Path):
        path = tmp / "leakage.jsonl"
        append_daily_summary(generate_daily_summary(FIXED_DATE, RUN_ID, [], 0), path)
        with path.open("a") as f:
            f.write("bad json\n")
        loaded = load_daily_summaries(path)
        assert len(loaded) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. Stale data and missing data handling
# ══════════════════════════════════════════════════════════════════════════════

class TestStaleAndMissingData:
    def test_stale_signal_reason_accepted(self):
        r = _make_record(rejection_reason=REJECTION_STALE_SIGNAL)
        assert r.rejection_reason == REJECTION_STALE_SIGNAL

    def test_record_with_zero_cash_does_not_affect_mean(self):
        r1 = _make_record(available_cash=0.0)
        r2 = _make_record(symbol="5678.T", available_cash=1_000_000.0)
        metrics = compute_alpha_metrics([r1, r2], executed_count=0, capital=CAPITAL)
        # Only r2's cash counts (r1.available_cash == 0 excluded from mean)
        assert metrics.mean_available_cash_at_skip == pytest.approx(1_000_000.0, rel=1e-6)

    def test_none_forward_returns_handled_in_metrics(self):
        recs = [_make_record(symbol=str(i)) for i in range(5)]
        # All pending (forward_returns=None)
        metrics = compute_alpha_metrics(recs, executed_count=0)
        assert metrics.skipped_pnl_estimate == 0.0
        assert metrics.enrichment_coverage == 0.0

    def test_partial_enrichment_handled(self):
        rec_pending = _make_record(symbol="A")
        rec_enriched = _make_record(symbol="B", price=1000.0, intended_size=100)
        prices = _make_price_history(1000.0, [0.05, 0.05, 0.05, 0.05, 0.05])
        enriched = enrich_forward_returns([rec_enriched], {"B": prices})[0]
        metrics = compute_alpha_metrics([rec_pending, enriched], executed_count=0)
        assert metrics.enrichment_coverage == pytest.approx(0.5, rel=1e-6)
        assert metrics.skipped_pnl_estimate > 0

    def test_summary_with_no_enriched_records_has_zero_pnl(self):
        skipped = [_make_record(symbol=str(i)) for i in range(5)]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, skipped, 0)
        assert summary.skipped_pnl_estimate == 0.0
        assert len(summary.top_missed_by_pnl) == 0

    def test_exclusion_breakdown_pnl_zero_when_no_enrichment(self):
        skipped = [_make_record(symbol=str(i)) for i in range(3)]
        summary = generate_daily_summary(FIXED_DATE, RUN_ID, skipped, 0)
        for entry in summary.exclusion_breakdown:
            assert entry["estimated_missed_pnl"] == 0.0

    def test_expired_records_excluded_from_pnl_computation(self):
        old_ts = "2020-01-01T09:00:00+00:00"
        rec = _make_record(ts=old_ts)
        now = datetime.now(timezone.utc).isoformat()
        expired = enrich_forward_returns([rec], {}, now_ts=now)[0]
        assert expired.enrichment_status == ENRICHMENT_EXPIRED
        # Expired records have no forward_return data → estimated_missed_pnl=None
        assert expired.estimated_missed_pnl is None
        metrics = compute_alpha_metrics([expired], executed_count=0)
        assert metrics.skipped_pnl_estimate == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 8. Replay determinism — same inputs → same outputs
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayDeterminism:
    def test_record_id_stable_across_runs(self):
        """record_id must be identical on replay (same run_id+symbol+reason+ts)."""
        ids = set()
        for _ in range(5):
            r = _make_record()
            ids.add(r.record_id)
        assert len(ids) == 1

    def test_metrics_stable_across_calls(self):
        skipped = [_make_record(symbol=str(i)) for i in range(4)]
        results = [
            compute_alpha_metrics(skipped, executed_count=2, capital=CAPITAL)
            for _ in range(3)
        ]
        ratios = [r.opportunity_utilization_ratio for r in results]
        assert len(set(ratios)) == 1

    def test_summary_id_stable_across_calls(self):
        """summary_id depends only on date + run_id."""
        ids = {
            generate_daily_summary(FIXED_DATE, RUN_ID, [], 0).summary_id
            for _ in range(3)
        }
        assert len(ids) == 1

    def test_enrichment_deterministic_with_same_prices(self):
        rec = _make_record(price=2000.0)
        prices = [2000.0, 2020.0, 2040.0, 2060.0, 2080.0, 2100.0]
        results = [
            enrich_forward_returns([rec], {"1234.T": prices})[0]
            for _ in range(3)
        ]
        pnls = [r.estimated_missed_pnl for r in results]
        assert len(set(pnls)) == 1

    def test_full_pipeline_replay(self, tmp: Path):
        """Write records, reload, re-enrich, re-compute — all values stable."""
        path = tmp / "skipped.jsonl"
        metrics_path = tmp / "metrics.jsonl"
        summary_path = tmp / "leakage.jsonl"

        # Original run
        rec = _make_record(symbol="1234.T", price=2000.0, intended_size=100)
        append_skipped_opportunity(rec, path)

        # Enrichment pass
        prices = [2000.0, 2020.0, 2040.0, 2060.0, 2080.0, 2100.0]
        loaded = load_skipped_opportunities(path)
        enriched_list = enrich_forward_returns(loaded, {"1234.T": prices})
        for e in enriched_list:
            append_skipped_opportunity(e, path)

        # Metrics computation (pass 1)
        all_records = deduplicate_records(load_skipped_opportunities(path))
        metrics1 = compute_alpha_metrics(all_records, executed_count=1, capital=CAPITAL)
        append_alpha_metrics(metrics1, metrics_path)

        # Replay: reload and recompute
        all_records2 = deduplicate_records(load_skipped_opportunities(path))
        metrics2 = compute_alpha_metrics(all_records2, executed_count=1, capital=CAPITAL)

        assert metrics1.skipped_pnl_estimate == metrics2.skipped_pnl_estimate
        assert metrics1.opportunity_utilization_ratio == metrics2.opportunity_utilization_ratio

        # Daily summary
        summary1 = generate_daily_summary(FIXED_DATE, RUN_ID, all_records, 1, capital=CAPITAL)
        summary2 = generate_daily_summary(FIXED_DATE, RUN_ID, all_records2, 1, capital=CAPITAL)
        assert summary1.summary_id == summary2.summary_id
        assert summary1.skipped_pnl_estimate == summary2.skipped_pnl_estimate

    def test_timestamp_consistency_in_loaded_records(self, tmp: Path):
        """Timestamps survive JSONL roundtrip unchanged."""
        path = tmp / "skipped.jsonl"
        rec = _make_record(ts=FIXED_TS)
        append_skipped_opportunity(rec, path)
        loaded = load_skipped_opportunities(path)
        assert loaded[0].timestamp == FIXED_TS
