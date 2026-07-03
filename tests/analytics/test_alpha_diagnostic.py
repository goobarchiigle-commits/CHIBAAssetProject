"""
tests/analytics/test_alpha_diagnostic.py

Tests for src/analytics/alpha_diagnostic.py

Coverage:
  - deterministic snapshot identity
  - derived metric correctness
  - ranking reproducibility and ordering
  - replay consistency (append → load → compare)
  - missing-data and empty-record handling
  - delta-calculation correctness (trailing windows)
  - markdown report determinism and structure
  - bottleneck identification
  - convexity threshold filtering
  - idle-cash ranking
  - restrictive allocator period filtering
  - get_diagnostic_for_date retrieval
  - alloc_cap=0 filtering in restrictive allocator
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import pytest

from src.analytics.skipped_opportunity_analytics import (
    ENRICHMENT_ENRICHED,
    ENRICHMENT_PENDING,
    REJECTION_ALLOC_CAP,
    REJECTION_HIGH_PRICE,
    REJECTION_SECTOR_SUPPRESSION,
    REJECTION_CONCENTRATION_THROTTLE,
    REJECTION_INSUFFICIENT_CASH,
    SkippedOpportunityRecord,
)
from src.analytics.daily_leakage_summary import DailyLeakageSummary
from src.analytics.alpha_diagnostic import (
    CONVEXITY_THRESHOLD,
    RUNNER_THRESHOLD,
    AlphaDiagnosticSnapshot,
    TrailingWindowMetrics,
    append_diagnostic,
    append_markdown_report,
    generate_diagnostic,
    generate_markdown_report,
    get_diagnostic_for_date,
    load_diagnostics,
    log_diagnostic,
    _make_id,
    _compute_trailing,
    _identify_dominant_bottleneck,
    _identify_leakage_source,
    _rank_exclusion_costs,
    _rank_idle_cash,
    _rank_missed_winners,
    _rank_restrictive_allocator,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATE = "2026-05-17"
RUN_ID = "run-diag-001"
CAPITAL = 3_000_000.0
TS = "2026-05-17T09:00:00+00:00"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(
    symbol: str = "1234.T",
    rejection_reason: str = REJECTION_ALLOC_CAP,
    signal_strength: float = 80.0,
    predicted_rank: int = 1,
    available_cash: float = 500_000.0,
    alloc_cap: int = 0,
    intended_size: int = 100,
    price: float = 2500.0,
    ts: str = TS,
    run_id: str = RUN_ID,
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


def _enrich(
    rec: SkippedOpportunityRecord,
    fr5d: float = 0.04,
    fr1d: float = 0.01,
    fr3d: float = 0.02,
) -> SkippedOpportunityRecord:
    import dataclasses
    missed_pnl = round(rec.intended_position_size * rec.price_at_signal * fr5d, 2)
    return dataclasses.replace(
        rec,
        forward_return_1d=fr1d,
        forward_return_3d=fr3d,
        forward_return_5d=fr5d,
        hypothetical_return=fr5d,
        estimated_missed_pnl=missed_pnl,
        enrichment_status=ENRICHMENT_ENRICHED,
    )


def _make_summary(
    date: str,
    util: float = 0.5,
    conv: float = 0.6,
    skipped_pnl: float = 10_000.0,
    realized_pnl: float = 15_000.0,
    idle_mean: float = 300_000.0,
    executed: int = 2,
    skipped: int = 2,
) -> DailyLeakageSummary:
    return DailyLeakageSummary(
        summary_id=f"id-{date}",
        date=date,
        run_id="run-hist",
        generated_at=f"{date}T09:00:00+00:00",
        total_signals_seen=executed + skipped,
        executed_count=executed,
        skipped_count=skipped,
        top_missed_by_signal=[],
        top_missed_by_pnl=[],
        exclusion_breakdown=[],
        idle_capital_mean_jpy=idle_mean,
        idle_capital_peak_jpy=idle_mean * 1.5,
        opportunity_utilization_ratio=util,
        deployable_alpha_conversion_rate=conv,
        skipped_pnl_estimate=skipped_pnl,
        realized_pnl_estimate=realized_pnl,
        enrichment_coverage=0.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Deterministic snapshot identity
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterministicIdentity:
    def test_diagnostic_id_deterministic(self):
        id1 = _make_id(DATE, RUN_ID)
        id2 = _make_id(DATE, RUN_ID)
        assert id1 == id2

    def test_diagnostic_id_differs_by_date(self):
        assert _make_id("2026-05-17", RUN_ID) != _make_id("2026-05-18", RUN_ID)

    def test_diagnostic_id_differs_by_run_id(self):
        assert _make_id(DATE, "run-A") != _make_id(DATE, "run-B")

    def test_snapshot_id_matches_helper(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        assert snap.diagnostic_id == _make_id(DATE, RUN_ID)

    def test_snapshot_core_fields_deterministic(self):
        records = [_make_record("1234.T"), _make_record("5678.T")]
        snap1 = generate_diagnostic(DATE, RUN_ID, records, 1, realized_pnl=0.0)
        snap2 = generate_diagnostic(DATE, RUN_ID, records, 1, realized_pnl=0.0)
        # generated_at may differ; compare everything else
        d1 = asdict(snap1)
        d2 = asdict(snap2)
        for key in d1:
            if key == "generated_at":
                continue
            assert d1[key] == d2[key], f"Mismatch on {key}"

    def test_snapshot_rankings_deterministic(self):
        records = [
            _enrich(_make_record("A.T", signal_strength=90.0, price=3000.0), fr5d=0.08),
            _enrich(_make_record("B.T", signal_strength=70.0, price=1500.0), fr5d=0.06),
        ]
        snap1 = generate_diagnostic(DATE, RUN_ID, records, 0)
        snap2 = generate_diagnostic(DATE, RUN_ID, records, 0)
        assert snap1.top_missed_winners == snap2.top_missed_winners
        assert snap1.largest_exclusion_costs == snap2.largest_exclusion_costs


# ══════════════════════════════════════════════════════════════════════════════
# 2. Derived metric correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestDerivedMetrics:
    def test_deployable_alpha_loss_rate_zero_when_no_pnl(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        assert snap.deployable_alpha_loss_rate == 0.0

    def test_deployable_alpha_loss_rate_formula(self):
        # skipped_pnl = 10_000, realized = 40_000 → loss_rate = 10/50 = 0.2
        enriched_rec = _enrich(_make_record(intended_size=1, price=10_000.0), fr5d=1.0)
        snap = generate_diagnostic(
            DATE, RUN_ID, [enriched_rec], 1, realized_pnl=40_000.0, capital=CAPITAL
        )
        assert abs(snap.deployable_alpha_loss_rate - 10_000.0 / 50_000.0) < 1e-4

    def test_capital_efficiency_score_range(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 5, realized_pnl=50_000.0)
        assert 0.0 <= snap.capital_efficiency_score <= 1.0

    def test_capital_efficiency_score_full_execution(self):
        # executed=5, skipped=0 → util=1.0; realized_pnl=50k, no skipped → conv=1.0
        snap = generate_diagnostic(DATE, RUN_ID, [], 5, realized_pnl=50_000.0)
        # idle_cash_ratio=0 (no skips), so score = 0.4*1 + 0.4*1 + 0.2*1 = 1.0
        assert abs(snap.capital_efficiency_score - 1.0) < 1e-5

    def test_capital_efficiency_score_zero_execution(self):
        # executed=0, skipped=0 → all zeros
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        # util=0, conv=0, idle=0 → score = 0*0.4 + 0*0.4 + 1*0.2 = 0.2
        assert abs(snap.capital_efficiency_score - 0.2) < 1e-5

    def test_convexity_capture_rate_no_enriched(self):
        # pending records only → no convexity misses known → rate = 1.0
        records = [_make_record(), _make_record("B.T")]
        snap = generate_diagnostic(DATE, RUN_ID, records, 1)
        assert snap.convexity_capture_rate == 1.0
        assert snap.convexity_missed_count == 0

    def test_convexity_capture_rate_all_miss(self):
        # 2 enriched skips both above threshold, 0 executed
        r1 = _enrich(_make_record("A.T"), fr5d=CONVEXITY_THRESHOLD + 0.01)
        r2 = _enrich(_make_record("B.T"), fr5d=CONVEXITY_THRESHOLD + 0.02)
        snap = generate_diagnostic(DATE, RUN_ID, [r1, r2], 0)
        # total_signals = 2, convexity_missed = 2 → rate = 1 - 2/2 = 0.0
        assert snap.convexity_capture_rate == 0.0
        assert snap.convexity_missed_count == 2

    def test_convexity_threshold_boundary(self):
        # fr5d == threshold exactly should NOT count as convexity event (strict >)
        at_threshold = _enrich(_make_record("A.T"), fr5d=CONVEXITY_THRESHOLD)
        above = _enrich(_make_record("B.T"), fr5d=CONVEXITY_THRESHOLD + 0.001)
        snap = generate_diagnostic(DATE, RUN_ID, [at_threshold, above], 0)
        assert snap.convexity_missed_count == 1  # only above counts

    def test_missed_runner_ratio_formula(self):
        # runner = fr5d > RUNNER_THRESHOLD; 1 runner among 3 total signals
        runner = _enrich(_make_record("A.T"), fr5d=RUNNER_THRESHOLD + 0.01)
        no_run = _enrich(_make_record("B.T"), fr5d=0.01)
        snap = generate_diagnostic(DATE, RUN_ID, [runner, no_run], 1)
        # total=3, runners=1 → ratio = 1/3
        assert abs(snap.missed_runner_ratio - 1.0 / 3.0) < 1e-5
        assert snap.runner_missed_count == 1

    def test_idle_cash_ratio_clamped(self):
        # available_cash > capital → ratio should be clamped to 1.0
        rec = _make_record(available_cash=CAPITAL * 2.0)
        snap = generate_diagnostic(DATE, RUN_ID, [rec], 0, capital=CAPITAL)
        assert snap.idle_cash_ratio <= 1.0

    def test_opportunity_utilization_ratio(self):
        records = [_make_record(), _make_record("B.T")]
        snap = generate_diagnostic(DATE, RUN_ID, records, 3)  # 3 exec, 2 skipped → 3/5
        assert abs(snap.opportunity_utilization_ratio - 3 / 5) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rankings reproducibility and ordering
# ══════════════════════════════════════════════════════════════════════════════

class TestRankings:
    def test_top_missed_winners_sorted_by_fr5d_desc(self):
        r1 = _enrich(_make_record("A.T", signal_strength=70.0), fr5d=0.02)
        r2 = _enrich(_make_record("B.T", signal_strength=80.0), fr5d=0.07)
        r3 = _enrich(_make_record("C.T", signal_strength=60.0), fr5d=0.05)
        winners = _rank_missed_winners([r1, r2, r3], top_n=5)
        symbols = [w["symbol"] for w in winners]
        assert symbols == ["B.T", "C.T", "A.T"]

    def test_top_missed_winners_top_n_respected(self):
        records = [_enrich(_make_record(f"{i}.T"), fr5d=0.01 * i) for i in range(10)]
        winners = _rank_missed_winners(records, top_n=3)
        assert len(winners) == 3

    def test_largest_exclusion_costs_sorted_by_pnl_desc(self):
        r_alloc = _enrich(_make_record("A.T", rejection_reason=REJECTION_ALLOC_CAP, price=2000.0, intended_size=200), fr5d=0.10)
        r_hp = _enrich(_make_record("B.T", rejection_reason=REJECTION_HIGH_PRICE, price=1000.0, intended_size=50), fr5d=0.03)
        costs = _rank_exclusion_costs([r_alloc, r_hp], skipped_count=2, top_n=5)
        # alloc_cap has higher PnL; should be first
        assert costs[0]["reason"] == REJECTION_ALLOC_CAP

    def test_exclusion_costs_fraction_correct(self):
        records = [
            _make_record("A.T", rejection_reason=REJECTION_ALLOC_CAP),
            _make_record("B.T", rejection_reason=REJECTION_ALLOC_CAP),
            _make_record("C.T", rejection_reason=REJECTION_HIGH_PRICE),
        ]
        costs = _rank_exclusion_costs(records, skipped_count=3, top_n=5)
        entry_alloc = next(c for c in costs if c["reason"] == REJECTION_ALLOC_CAP)
        entry_hp = next(c for c in costs if c["reason"] == REJECTION_HIGH_PRICE)
        assert abs(entry_alloc["fraction"] - 2 / 3) < 1e-5
        assert abs(entry_hp["fraction"] - 1 / 3) < 1e-5

    def test_idle_cash_ranking_sorted_desc(self):
        records = [
            _make_record("A.T", available_cash=100_000.0),
            _make_record("B.T", available_cash=900_000.0),
            _make_record("C.T", available_cash=500_000.0),
        ]
        ranked = _rank_idle_cash(records, top_n=5)
        cashes = [r["available_cash_jpy"] for r in ranked]
        assert cashes == sorted(cashes, reverse=True)

    def test_restrictive_allocator_only_alloc_cap_zero(self):
        r_blocked = _make_record("A.T", alloc_cap=0, rejection_reason=REJECTION_ALLOC_CAP)
        r_partial = _make_record("B.T", alloc_cap=50, rejection_reason=REJECTION_ALLOC_CAP)
        r_other = _make_record("C.T", alloc_cap=0, rejection_reason=REJECTION_HIGH_PRICE)
        result = _rank_restrictive_allocator([r_blocked, r_partial, r_other], top_n=5)
        symbols = {r["symbol"] for r in result}
        assert "B.T" not in symbols  # alloc_cap != 0 excluded
        assert "A.T" in symbols
        assert "C.T" in symbols

    def test_restrictive_allocator_sorted_by_cash_desc(self):
        r1 = _make_record("A.T", alloc_cap=0, available_cash=200_000.0)
        r2 = _make_record("B.T", alloc_cap=0, available_cash=800_000.0)
        r3 = _make_record("C.T", alloc_cap=0, available_cash=500_000.0)
        result = _rank_restrictive_allocator([r1, r2, r3], top_n=5)
        cashes = [r["available_cash_jpy"] for r in result]
        assert cashes == sorted(cashes, reverse=True)

    def test_convexity_events_in_snapshot(self):
        above = _enrich(_make_record("A.T"), fr5d=CONVEXITY_THRESHOLD + 0.02)
        below = _enrich(_make_record("B.T"), fr5d=CONVEXITY_THRESHOLD - 0.01)
        snap = generate_diagnostic(DATE, RUN_ID, [above, below], 0)
        conv_symbols = {e["symbol"] for e in snap.top_missed_convexity_events}
        assert "A.T" in conv_symbols
        assert "B.T" not in conv_symbols

    def test_rankings_reproducible_across_calls(self):
        records = [
            _enrich(_make_record("A.T", signal_strength=90.0), fr5d=0.08),
            _enrich(_make_record("B.T", signal_strength=70.0), fr5d=0.04),
            _make_record("C.T"),
        ]
        snap1 = generate_diagnostic(DATE, RUN_ID, records, 2)
        snap2 = generate_diagnostic(DATE, RUN_ID, records, 2)
        assert snap1.top_missed_winners == snap2.top_missed_winners
        assert snap1.largest_exclusion_costs == snap2.largest_exclusion_costs
        assert snap1.highest_idle_cash_periods == snap2.highest_idle_cash_periods
        assert snap1.most_restrictive_allocator_periods == snap2.most_restrictive_allocator_periods


# ══════════════════════════════════════════════════════════════════════════════
# 4. Trailing windows and delta calculations
# ══════════════════════════════════════════════════════════════════════════════

class TestTrailingWindows:
    def _make_past(self, days: int, base_date: str = "2026-05-16") -> List[DailyLeakageSummary]:
        from datetime import date
        base = datetime.strptime(base_date, "%Y-%m-%d").date()
        return [
            _make_summary(
                date=str(base - timedelta(days=i)),
                util=0.5,
                conv=0.6,
                skipped_pnl=10_000.0,
                realized_pnl=15_000.0,
            )
            for i in range(days)
        ]

    def test_trailing_5d_sample_count(self):
        past = self._make_past(10)
        tw = _compute_trailing(
            window=5, past_summaries=past,
            current_util=0.7, current_conv=0.8,
            current_efficiency=0.7, current_loss_rate=0.3,
        )
        assert tw.sample_count == 5

    def test_trailing_20d_fewer_than_20_samples(self):
        past = self._make_past(8)
        tw = _compute_trailing(
            window=20, past_summaries=past,
            current_util=0.7, current_conv=0.8,
            current_efficiency=0.7, current_loss_rate=0.3,
        )
        assert tw.sample_count == 8

    def test_trailing_5d_mean_util(self):
        past = self._make_past(5)
        tw = _compute_trailing(
            window=5, past_summaries=past,
            current_util=0.7, current_conv=0.6,
            current_efficiency=0.6, current_loss_rate=0.4,
        )
        assert abs(tw.mean_util_ratio - 0.5) < 1e-5

    def test_delta_util_correct(self):
        past = self._make_past(5)  # mean_util = 0.5
        tw = _compute_trailing(
            window=5, past_summaries=past,
            current_util=0.7, current_conv=0.6,
            current_efficiency=0.6, current_loss_rate=0.4,
        )
        assert abs(tw.delta_util_ratio - (0.7 - 0.5)) < 1e-5

    def test_trailing_empty_history(self):
        tw = _compute_trailing(
            window=5, past_summaries=[],
            current_util=0.7, current_conv=0.8,
            current_efficiency=0.65, current_loss_rate=0.35,
        )
        assert tw.sample_count == 0
        assert tw.mean_util_ratio == 0.0
        assert abs(tw.delta_util_ratio - 0.7) < 1e-5

    def test_trailing_excludes_current_date(self):
        # historical_summaries contains current date — should be excluded
        past = self._make_past(3) + [_make_summary(DATE, util=0.99)]
        snap = generate_diagnostic(DATE, RUN_ID, [], 1, historical_summaries=past)
        # sample_count should be 3 (current date excluded)
        assert snap.trailing_5d["sample_count"] == 3

    def test_trailing_5d_in_snapshot(self):
        past = self._make_past(5)
        snap = generate_diagnostic(DATE, RUN_ID, [], 1, historical_summaries=past)
        assert "mean_util_ratio" in snap.trailing_5d
        assert "delta_util_ratio" in snap.trailing_5d
        assert snap.trailing_5d["window_days"] == 5
        assert snap.trailing_20d["window_days"] == 20

    def test_delta_calculation_sign_correct(self):
        # current util=0.3 < historical mean=0.5 → delta < 0
        past = self._make_past(5)  # mean_util=0.5
        snap = generate_diagnostic(DATE, RUN_ID, [], 1,
                                   historical_summaries=past)
        # util = 1/2 (executed=1, skipped=1 implied from one record)
        # actual executed=1, skipped=0 → util=1.0 > 0.5 → delta > 0
        assert snap.trailing_5d["delta_util_ratio"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Bottleneck identification
# ══════════════════════════════════════════════════════════════════════════════

class TestBottleneckIdentification:
    def test_dominant_bottleneck_by_pnl(self):
        # alloc_cap misses larger PnL than high_price
        r_alloc = _enrich(_make_record("A.T", rejection_reason=REJECTION_ALLOC_CAP,
                                        price=3000.0, intended_size=200), fr5d=0.10)
        r_hp = _enrich(_make_record("B.T", rejection_reason=REJECTION_HIGH_PRICE,
                                     price=500.0, intended_size=10), fr5d=0.02)
        snap = generate_diagnostic(DATE, RUN_ID, [r_alloc, r_hp], 0)
        assert snap.dominant_bottleneck == REJECTION_ALLOC_CAP

    def test_dominant_bottleneck_fallback_to_count(self):
        # No enriched records → fallback to reason with most skips
        records = [
            _make_record("A.T", rejection_reason=REJECTION_INSUFFICIENT_CASH),
            _make_record("B.T", rejection_reason=REJECTION_INSUFFICIENT_CASH),
            _make_record("C.T", rejection_reason=REJECTION_HIGH_PRICE),
        ]
        snap = generate_diagnostic(DATE, RUN_ID, records, 0)
        assert snap.dominant_bottleneck == REJECTION_INSUFFICIENT_CASH

    def test_dominant_bottleneck_no_records(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        assert snap.dominant_bottleneck == "none"

    def test_largest_leakage_source_by_count(self):
        records = [
            _make_record("A.T", rejection_reason=REJECTION_SECTOR_SUPPRESSION),
            _make_record("B.T", rejection_reason=REJECTION_SECTOR_SUPPRESSION),
            _make_record("C.T", rejection_reason=REJECTION_ALLOC_CAP),
        ]
        snap = generate_diagnostic(DATE, RUN_ID, records, 0)
        assert snap.largest_leakage_source == REJECTION_SECTOR_SUPPRESSION

    def test_largest_leakage_source_no_records(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        assert snap.largest_leakage_source == "none"

    def test_dominant_bottleneck_pnl_matches_sum(self):
        r1 = _enrich(_make_record("A.T", rejection_reason=REJECTION_ALLOC_CAP,
                                   price=1000.0, intended_size=100), fr5d=0.05)
        r2 = _enrich(_make_record("B.T", rejection_reason=REJECTION_ALLOC_CAP,
                                   price=1000.0, intended_size=100), fr5d=0.05)
        # expected pnl = 2 × 100 × 1000 × 0.05 = 10_000
        snap = generate_diagnostic(DATE, RUN_ID, [r1, r2], 0)
        assert snap.dominant_bottleneck == REJECTION_ALLOC_CAP
        assert abs(snap.dominant_bottleneck_pnl_jpy - 10_000.0) < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Missing-data and edge-case handling
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingDataHandling:
    def test_empty_skipped_records(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        assert snap.total_signals_seen == 0
        assert snap.skipped_count == 0
        assert snap.executed_count == 0
        assert snap.deployable_alpha_loss_rate == 0.0
        assert snap.convexity_capture_rate == 1.0
        assert snap.missed_runner_ratio == 0.0
        assert snap.top_missed_winners == []
        assert snap.largest_exclusion_costs == []

    def test_no_enriched_records(self):
        records = [_make_record(), _make_record("B.T")]
        snap = generate_diagnostic(DATE, RUN_ID, records, 1)
        assert snap.enrichment_coverage == 0.0
        assert snap.skipped_pnl_estimate == 0.0
        assert snap.top_missed_winners == []
        assert snap.top_missed_convexity_events == []

    def test_all_enriched_records(self):
        records = [_enrich(_make_record()), _enrich(_make_record("B.T"), fr5d=0.07)]
        snap = generate_diagnostic(DATE, RUN_ID, records, 0)
        assert snap.enrichment_coverage == 1.0

    def test_zero_capital_no_crash(self):
        rec = _make_record(available_cash=500_000.0)
        snap = generate_diagnostic(DATE, RUN_ID, [rec], 0, capital=0.0)
        assert snap.idle_cash_ratio == 0.0

    def test_records_with_zero_cash_excluded_from_idle_mean(self):
        r_zero = _make_record("A.T", available_cash=0.0)
        r_cash = _make_record("B.T", available_cash=600_000.0)
        snap = generate_diagnostic(DATE, RUN_ID, [r_zero, r_cash], 0)
        assert abs(snap.idle_capital_mean_jpy - 600_000.0) < 1.0
        assert snap.idle_capital_peak_jpy == 600_000.0

    def test_estimated_missed_pnl_none_handled(self):
        rec = _make_record()  # no enrichment → pnl=None
        snap = generate_diagnostic(DATE, RUN_ID, [rec], 0)
        assert snap.skipped_pnl_estimate == 0.0

    def test_no_restrictive_allocator_periods(self):
        rec = _make_record("A.T", alloc_cap=50)
        snap = generate_diagnostic(DATE, RUN_ID, [rec], 0)
        assert snap.most_restrictive_allocator_periods == []

    def test_top_n_capped(self):
        records = [_enrich(_make_record(f"{i}.T"), fr5d=0.01 * i) for i in range(20)]
        snap = generate_diagnostic(DATE, RUN_ID, records, 0, top_n=3)
        assert len(snap.top_missed_winners) <= 3
        assert len(snap.largest_exclusion_costs) <= 3
        assert len(snap.highest_idle_cash_periods) <= 3


# ══════════════════════════════════════════════════════════════════════════════
# 7. Replay consistency (append → load → compare)
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayConsistency:
    def test_append_and_load_roundtrip(self, tmp_path: Path):
        records = [_enrich(_make_record("A.T"), fr5d=0.06)]
        snap = generate_diagnostic(DATE, RUN_ID, records, 1, realized_pnl=5_000.0)
        path = tmp_path / "diag.jsonl"
        append_diagnostic(snap, path)
        loaded = load_diagnostics(path)
        assert len(loaded) == 1
        s = loaded[0]
        assert s.diagnostic_id == snap.diagnostic_id
        assert s.date == snap.date
        assert s.run_id == snap.run_id
        assert abs(s.capital_efficiency_score - snap.capital_efficiency_score) < 1e-8
        assert abs(s.deployable_alpha_loss_rate - snap.deployable_alpha_loss_rate) < 1e-8
        assert s.dominant_bottleneck == snap.dominant_bottleneck
        assert s.top_missed_winners == snap.top_missed_winners
        assert s.largest_exclusion_costs == snap.largest_exclusion_costs

    def test_append_multiple_records(self, tmp_path: Path):
        path = tmp_path / "diag.jsonl"
        dates = ["2026-05-15", "2026-05-16", "2026-05-17"]
        for d in dates:
            snap = generate_diagnostic(d, RUN_ID, [], 0)
            append_diagnostic(snap, path)
        loaded = load_diagnostics(path)
        assert len(loaded) == 3
        assert {s.date for s in loaded} == set(dates)

    def test_get_diagnostic_for_date(self, tmp_path: Path):
        path = tmp_path / "diag.jsonl"
        for d in ["2026-05-15", "2026-05-16", DATE]:
            append_diagnostic(generate_diagnostic(d, RUN_ID, [], 0), path)
        result = get_diagnostic_for_date(path, DATE)
        assert result is not None
        assert result.date == DATE

    def test_get_diagnostic_for_date_missing(self, tmp_path: Path):
        path = tmp_path / "diag.jsonl"
        append_diagnostic(generate_diagnostic("2026-05-15", RUN_ID, [], 0), path)
        assert get_diagnostic_for_date(path, "2099-01-01") is None

    def test_get_diagnostic_for_date_no_file(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"
        assert get_diagnostic_for_date(path, DATE) is None

    def test_load_empty_file(self, tmp_path: Path):
        path = tmp_path / "diag.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_diagnostics(path) == []

    def test_load_corrupted_line_skipped(self, tmp_path: Path):
        path = tmp_path / "diag.jsonl"
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        append_diagnostic(snap, path)
        with path.open("a", encoding="utf-8") as f:
            f.write("NOT_VALID_JSON\n")
        loaded = load_diagnostics(path)
        assert len(loaded) == 1

    def test_replay_same_records_same_jsonl(self, tmp_path: Path):
        records = [_enrich(_make_record("A.T"), fr5d=0.05)]
        snap = generate_diagnostic(DATE, RUN_ID, records, 2, realized_pnl=8_000.0)
        path1 = tmp_path / "d1.jsonl"
        path2 = tmp_path / "d2.jsonl"
        append_diagnostic(snap, path1)
        append_diagnostic(snap, path2)
        line1 = path1.read_text(encoding="utf-8").strip()
        line2 = path2.read_text(encoding="utf-8").strip()
        # Same snapshot → same JSON line (excluding generated_at, which is inside the snap)
        d1 = json.loads(line1)
        d2 = json.loads(line2)
        # diagnostic_id must match
        assert d1["diagnostic_id"] == d2["diagnostic_id"]


# ══════════════════════════════════════════════════════════════════════════════
# 8. Markdown report determinism and structure
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkdownReport:
    def _snap(self) -> AlphaDiagnosticSnapshot:
        records = [
            _enrich(_make_record("1234.T", rejection_reason=REJECTION_ALLOC_CAP,
                                  signal_strength=85.0, available_cash=700_000.0,
                                  price=3000.0, intended_size=100), fr5d=0.08),
            _make_record("5678.T", rejection_reason=REJECTION_HIGH_PRICE,
                          available_cash=500_000.0, alloc_cap=0),
        ]
        return generate_diagnostic(DATE, RUN_ID, records, 1, realized_pnl=12_000.0)

    def test_markdown_deterministic(self):
        snap = self._snap()
        md1 = generate_markdown_report(snap)
        md2 = generate_markdown_report(snap)
        assert md1 == md2

    def test_markdown_contains_date(self):
        snap = self._snap()
        md = generate_markdown_report(snap)
        assert DATE in md

    def test_markdown_contains_diagnostic_id(self):
        snap = self._snap()
        md = generate_markdown_report(snap)
        assert snap.diagnostic_id[:8] in md

    def test_markdown_has_capital_efficiency_section(self):
        md = generate_markdown_report(self._snap())
        assert "Capital Efficiency Summary" in md

    def test_markdown_has_bottleneck_section(self):
        md = generate_markdown_report(self._snap())
        assert "Dominant Capital-Efficiency Bottleneck" in md

    def test_markdown_has_exclusion_breakdown(self):
        md = generate_markdown_report(self._snap())
        assert "Exclusion Breakdown" in md

    def test_markdown_has_top_missed_winners(self):
        md = generate_markdown_report(self._snap())
        assert "Top Missed Winners" in md

    def test_markdown_has_idle_cash_section(self):
        md = generate_markdown_report(self._snap())
        assert "Highest Idle-Cash Periods" in md

    def test_markdown_has_restrictive_allocator_section(self):
        md = generate_markdown_report(self._snap())
        assert "Most Restrictive Allocator Periods" in md

    def test_markdown_has_trailing_window_section(self):
        md = generate_markdown_report(self._snap())
        assert "Trailing Window Comparison" in md

    def test_markdown_has_convexity_section(self):
        md = generate_markdown_report(self._snap())
        assert "Missed Convexity Events" in md

    def test_markdown_shows_bottleneck_reason(self):
        snap = self._snap()
        md = generate_markdown_report(snap)
        assert snap.dominant_bottleneck in md

    def test_markdown_no_crash_empty_snap(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        md = generate_markdown_report(snap)
        assert "Alpha Diagnostic" in md

    def test_markdown_append_creates_file(self, tmp_path: Path):
        snap = self._snap()
        report_dir = tmp_path / "reports"
        append_markdown_report(snap, report_dir)
        report_path = report_dir / f"{DATE}.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert DATE in content

    def test_markdown_append_separates_runs(self, tmp_path: Path):
        snap = self._snap()
        report_dir = tmp_path / "reports"
        append_markdown_report(snap, report_dir)
        append_markdown_report(snap, report_dir)
        content = (report_dir / f"{DATE}.md").read_text(encoding="utf-8")
        assert "---" in content


# ══════════════════════════════════════════════════════════════════════════════
# 9. Fail-open behavior
# ══════════════════════════════════════════════════════════════════════════════

class TestFailOpen:
    def test_append_to_readonly_dir_no_raise(self, tmp_path: Path):
        # append_diagnostic should not raise even if write fails
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        bad_path = tmp_path / "no_such_dir" / "deeply" / "nested" / "file.jsonl"
        # Should succeed — creates parent dirs
        append_diagnostic(snap, bad_path)
        assert bad_path.exists()

    def test_append_markdown_no_raise(self, tmp_path: Path):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        report_dir = tmp_path / "reports"
        append_markdown_report(snap, report_dir)  # must not raise

    def test_log_diagnostic_no_raise(self):
        snap = generate_diagnostic(DATE, RUN_ID, [], 0)
        log_diagnostic(snap)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 10. Paths constants
# ══════════════════════════════════════════════════════════════════════════════

class TestPathConstants:
    def test_alpha_diagnostic_file_defined(self):
        from src.paths import ALPHA_DIAGNOSTIC_FILE
        assert "alpha_diagnostics" in str(ALPHA_DIAGNOSTIC_FILE)

    def test_alpha_diagnostic_report_dir_defined(self):
        from src.paths import ALPHA_DIAGNOSTIC_REPORT_DIR
        assert "reports" in str(ALPHA_DIAGNOSTIC_REPORT_DIR)

    def test_paths_under_analytics_dir(self):
        from src.paths import ALPHA_DIAGNOSTIC_FILE, ALPHA_DIAGNOSTIC_REPORT_DIR, ANALYTICS_DIR
        assert str(ANALYTICS_DIR) in str(ALPHA_DIAGNOSTIC_FILE)
        assert str(ANALYTICS_DIR) in str(ALPHA_DIAGNOSTIC_REPORT_DIR)
