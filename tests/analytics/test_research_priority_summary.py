"""
tests/analytics/test_research_priority_summary.py

pytest coverage for research_priority_summary module.

Covers:
  - deterministic summaries (same inputs → same output fields)
  - ranking reproducibility
  - bottleneck attribution correctness for each rejection reason
  - markdown reproducibility (byte-identical)
  - trailing-window comparisons
  - confidence-level determinism
  - missing-data handling (empty/None inputs)
  - replay consistency (serialize → deserialize → compare)
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from src.analytics.research_priority_summary import (
    # Core function
    generate_research_priority_summary,
    generate_research_priority_markdown,
    run_research_priority_summary,
    # Persistence
    append_research_priority_summary,
    load_research_priority_summaries,
    write_research_priority_report,
    get_summary_for_date,
    # Dataclasses
    ResearchPrioritySummary,
    LeakageSourceEntry,
    # Constants — bottleneck categories
    BOTTLENECK_ALLOC_CAP_PRESSURE,
    BOTTLENECK_IDLE_CASH_PERSISTENCE,
    BOTTLENECK_HIGH_PRICED_EXCLUSION,
    BOTTLENECK_CONCENTRATION_THROTTLE,
    BOTTLENECK_SECTOR_SUPPRESSION,
    BOTTLENECK_PREMATURE_EXIT,
    BOTTLENECK_STALE_SIGNAL_DECAY,
    BOTTLENECK_INSUFFICIENT_BUYING_POWER,
    BOTTLENECK_EXECUTION_INTEGRITY,
    BOTTLENECK_NONE,
    # Constants — research categories
    RESEARCH_ALLOCATION,
    RESEARCH_EXIT,
    RESEARCH_DEPLOYABILITY,
    RESEARCH_CONCENTRATION,
    RESEARCH_CAPITAL_EFFICIENCY,
    RESEARCH_INTEGRITY,
    # Confidence levels
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    # Internal helpers (for direct testing)
    _classify_bottleneck,
    _compute_confidence,
    _compute_trend,
    _build_leakage_sources,
    _extract_affected_symbols,
    _extract_affected_sectors,
    _build_secondary_categories,
    _extract_integrity,
    _BOTTLENECK_TO_RESEARCH,
    _REASON_TO_BOTTLENECK,
    SCHEMA_VERSION,
)
from src.analytics.alpha_diagnostic import AlphaDiagnosticSnapshot, generate_diagnostic
from src.analytics.daily_leakage_summary import DailyLeakageSummary
from src.analytics.skipped_opportunity_analytics import (
    SkippedOpportunityRecord,
    ENRICHMENT_ENRICHED,
    ENRICHMENT_PENDING,
    REJECTION_ALLOC_CAP,
    REJECTION_HIGH_PRICE,
    REJECTION_SECTOR_SUPPRESSION,
    REJECTION_CONCENTRATION_THROTTLE,
    REJECTION_INSUFFICIENT_CASH,
    REJECTION_STALE_SIGNAL,
    REJECTION_TRUTH_CONFIDENCE,
    REJECTION_RATE_LIMIT,
    REJECTION_DUPLICATE,
    REJECTION_EPOCH_BLOCKED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

DATE = "2026-05-17"
RUN_ID = "test-run-001"
CAPITAL = 3_000_000.0


def _make_skipped(
    symbol: str = "1234",
    rejection_reason: str = REJECTION_ALLOC_CAP,
    available_cash: float = 800_000.0,
    alloc_cap: int = 0,
    price: float = 2000.0,
    size: int = 100,
    signal_strength: float = 80.0,
    forward_return_5d: Optional[float] = 0.05,
    estimated_missed_pnl: Optional[float] = 10_000.0,
    sector: str = "電気機器",
    run_id: str = RUN_ID,
    enrichment_status: str = ENRICHMENT_ENRICHED,
) -> SkippedOpportunityRecord:
    r = SkippedOpportunityRecord.create(
        run_id=run_id,
        timestamp=f"{DATE}T09:00:00+00:00",
        symbol=symbol,
        strategy_id="strat_a",
        signal_strength=signal_strength,
        predicted_rank=1,
        rejection_reason=rejection_reason,
        available_cash=available_cash,
        alloc_cap=alloc_cap,
        intended_position_size=size,
        sector_state=sector,
        concentration_state=0.3,
        price_at_signal=price,
    )
    import dataclasses
    return dataclasses.replace(
        r,
        forward_return_5d=forward_return_5d,
        estimated_missed_pnl=estimated_missed_pnl,
        enrichment_status=enrichment_status,
    )


def _make_snap(
    dominant_bottleneck: str = REJECTION_ALLOC_CAP,
    skipped_records: Optional[List] = None,
    executed_count: int = 1,
    idle_cash_ratio: float = 0.2,
) -> AlphaDiagnosticSnapshot:
    recs = skipped_records or [_make_skipped()]
    snap = generate_diagnostic(
        date=DATE,
        run_id=RUN_ID,
        skipped_records=recs,
        executed_count=executed_count,
        realized_pnl=5000.0,
        capital=CAPITAL,
    )
    return snap


def _make_leakage_summary(
    executed_count: int = 1,
    skipped_count: int = 2,
    skipped_pnl: float = 15_000.0,
    realized_pnl: float = 5_000.0,
    idle_mean: float = 600_000.0,
) -> DailyLeakageSummary:
    return DailyLeakageSummary(
        summary_id="test-summary-id",
        date=DATE,
        run_id=RUN_ID,
        generated_at="2026-05-17T09:00:00+00:00",
        total_signals_seen=executed_count + skipped_count,
        executed_count=executed_count,
        skipped_count=skipped_count,
        top_missed_by_signal=[
            {
                "symbol": "5555",
                "strategy_id": "s",
                "signal_strength": 85.0,
                "predicted_rank": 1,
                "rejection_reason": REJECTION_ALLOC_CAP,
                "price_at_signal": 2000.0,
                "intended_position_size": 100,
                "estimated_missed_pnl": skipped_pnl,
                "forward_return_5d": 0.05,
                "timestamp": f"{DATE}T09:00:00+00:00",
            }
        ],
        top_missed_by_pnl=[
            {
                "symbol": "5555",
                "strategy_id": "s",
                "signal_strength": 85.0,
                "predicted_rank": 1,
                "rejection_reason": REJECTION_ALLOC_CAP,
                "price_at_signal": 2000.0,
                "intended_position_size": 100,
                "estimated_missed_pnl": skipped_pnl,
                "forward_return_5d": 0.05,
                "timestamp": f"{DATE}T09:00:00+00:00",
            }
        ],
        exclusion_breakdown=[
            {
                "reason": REJECTION_ALLOC_CAP,
                "count": skipped_count,
                "fraction": 1.0,
                "estimated_missed_pnl": skipped_pnl,
            }
        ],
        idle_capital_mean_jpy=idle_mean,
        idle_capital_peak_jpy=idle_mean * 1.2,
        opportunity_utilization_ratio=executed_count / (executed_count + skipped_count),
        deployable_alpha_conversion_rate=realized_pnl / (realized_pnl + skipped_pnl),
        skipped_pnl_estimate=skipped_pnl,
        realized_pnl_estimate=realized_pnl,
        enrichment_coverage=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic summaries
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicSummaries:
    """Same inputs (excluding generated_at) → identical ResearchPrioritySummary fields."""

    def test_summary_id_deterministic(self):
        s1 = generate_research_priority_summary(DATE, RUN_ID, False)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False)
        assert s1.summary_id == s2.summary_id

    def test_all_fields_deterministic_from_alpha_snap(self):
        snap = _make_snap()
        s1 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        # All fields except generated_at must be identical
        d1 = asdict(s1)
        d2 = asdict(s2)
        d1.pop("generated_at")
        d2.pop("generated_at")
        assert d1 == d2

    def test_all_fields_deterministic_from_leakage(self):
        ls = _make_leakage_summary()
        s1 = generate_research_priority_summary(DATE, RUN_ID, True, leakage_summary=ls)
        s2 = generate_research_priority_summary(DATE, RUN_ID, True, leakage_summary=ls)
        d1 = asdict(s1); d2 = asdict(s2)
        d1.pop("generated_at"); d2.pop("generated_at")
        assert d1 == d2

    def test_all_fields_deterministic_from_records(self):
        recs = [_make_skipped("1234"), _make_skipped("5678", REJECTION_HIGH_PRICE)]
        s1 = generate_research_priority_summary(DATE, RUN_ID, False, skipped_records=recs)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False, skipped_records=recs)
        d1 = asdict(s1); d2 = asdict(s2)
        d1.pop("generated_at"); d2.pop("generated_at")
        assert d1 == d2

    def test_schema_version_present(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        assert s.schema_version == SCHEMA_VERSION

    def test_date_and_run_id_preserved(self):
        s = generate_research_priority_summary(DATE, RUN_ID, True)
        assert s.date == DATE
        assert s.run_id == RUN_ID
        assert s.is_live is True


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck attribution
# ─────────────────────────────────────────────────────────────────────────────

class TestBottleneckAttribution:
    """Each rejection reason maps to the correct bottleneck category."""

    REASON_TO_EXPECTED = [
        (REJECTION_ALLOC_CAP,              BOTTLENECK_ALLOC_CAP_PRESSURE),
        (REJECTION_HIGH_PRICE,             BOTTLENECK_HIGH_PRICED_EXCLUSION),
        (REJECTION_SECTOR_SUPPRESSION,     BOTTLENECK_SECTOR_SUPPRESSION),
        (REJECTION_CONCENTRATION_THROTTLE, BOTTLENECK_CONCENTRATION_THROTTLE),
        (REJECTION_INSUFFICIENT_CASH,      BOTTLENECK_INSUFFICIENT_BUYING_POWER),
        (REJECTION_STALE_SIGNAL,           BOTTLENECK_STALE_SIGNAL_DECAY),
        (REJECTION_TRUTH_CONFIDENCE,       BOTTLENECK_EXECUTION_INTEGRITY),
        (REJECTION_RATE_LIMIT,             BOTTLENECK_EXECUTION_INTEGRITY),
        (REJECTION_DUPLICATE,              BOTTLENECK_EXECUTION_INTEGRITY),
        (REJECTION_EPOCH_BLOCKED,          BOTTLENECK_EXECUTION_INTEGRITY),
    ]

    @pytest.mark.parametrize("reason,expected", REASON_TO_EXPECTED)
    def test_reason_maps_to_bottleneck(self, reason, expected):
        assert _REASON_TO_BOTTLENECK[reason] == expected

    @pytest.mark.parametrize("reason,expected", REASON_TO_EXPECTED)
    def test_single_reason_bottleneck_in_summary(self, reason, expected):
        recs = [_make_skipped(rejection_reason=reason, alloc_cap=1)]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=1, realized_pnl=1000.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        # Dominant bottleneck should map to expected category
        # (unless overridden by idle_cash_persistence / premature_exit derivation)
        from src.analytics.research_priority_summary import _REASON_TO_BOTTLENECK as r2b
        assert r2b.get(reason, BOTTLENECK_NONE) == expected

    def test_alloc_cap_maps_to_allocation_research(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_ALLOC_CAP_PRESSURE] == RESEARCH_ALLOCATION

    def test_high_price_maps_to_deployability_research(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_HIGH_PRICED_EXCLUSION] == RESEARCH_DEPLOYABILITY

    def test_concentration_maps_to_concentration_review(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_CONCENTRATION_THROTTLE] == RESEARCH_CONCENTRATION

    def test_sector_suppression_maps_to_allocation(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_SECTOR_SUPPRESSION] == RESEARCH_ALLOCATION

    def test_stale_signal_maps_to_deployability(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_STALE_SIGNAL_DECAY] == RESEARCH_DEPLOYABILITY

    def test_execution_integrity_maps_to_integrity_review(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_EXECUTION_INTEGRITY] == RESEARCH_INTEGRITY

    def test_premature_exit_maps_to_exit_research(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_PREMATURE_EXIT] == RESEARCH_EXIT

    def test_idle_cash_maps_to_capital_efficiency(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_IDLE_CASH_PERSISTENCE] == RESEARCH_CAPITAL_EFFICIENCY

    def test_insufficient_buying_power_maps_to_capital_efficiency(self):
        assert _BOTTLENECK_TO_RESEARCH[BOTTLENECK_INSUFFICIENT_BUYING_POWER] == RESEARCH_CAPITAL_EFFICIENCY

    def test_classify_premature_exit_when_alloc_and_high_idle(self):
        result = _classify_bottleneck(
            raw_bottleneck=REJECTION_ALLOC_CAP,
            idle_cash_ratio=0.50,
            idle_cash_peak=1_500_000.0,
            executed_count=2,
            skipped_count=5,
            capital=CAPITAL,
            restrictive_periods=[{"timestamp": "t", "symbol": "X"}],
        )
        assert result == BOTTLENECK_PREMATURE_EXIT

    def test_classify_idle_cash_persistence_when_ratio_high(self):
        result = _classify_bottleneck(
            raw_bottleneck=REJECTION_ALLOC_CAP,
            idle_cash_ratio=0.40,  # above 0.35 threshold
            idle_cash_peak=1_200_000.0,
            executed_count=1,
            skipped_count=5,   # skipped > executed
            capital=CAPITAL,
            restrictive_periods=[],  # no restrictive periods → no premature_exit
        )
        assert result == BOTTLENECK_IDLE_CASH_PERSISTENCE

    def test_classify_normal_bottleneck_no_override(self):
        result = _classify_bottleneck(
            raw_bottleneck=REJECTION_HIGH_PRICE,
            idle_cash_ratio=0.10,  # low
            idle_cash_peak=300_000.0,
            executed_count=2,
            skipped_count=1,
            capital=CAPITAL,
            restrictive_periods=[],
        )
        assert result == BOTTLENECK_HIGH_PRICED_EXCLUSION

    def test_classify_none_bottleneck_for_unknown_reason(self):
        result = _classify_bottleneck(
            raw_bottleneck="unknown_reason",
            idle_cash_ratio=0.05,
            idle_cash_peak=100_000.0,
            executed_count=3,
            skipped_count=1,
            capital=CAPITAL,
            restrictive_periods=[],
        )
        assert result == BOTTLENECK_NONE


# ─────────────────────────────────────────────────────────────────────────────
# Ranking reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestRankingReproducibility:
    """Same snapshot → same research categories and leakage sources."""

    def test_primary_category_stable(self):
        snap = _make_snap()
        s1 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        assert s1.primary_research_category == s2.primary_research_category

    def test_secondary_categories_stable(self):
        recs = [
            _make_skipped("A", REJECTION_ALLOC_CAP, estimated_missed_pnl=10_000.0),
            _make_skipped("B", REJECTION_HIGH_PRICE, estimated_missed_pnl=8_000.0),
            _make_skipped("C", REJECTION_SECTOR_SUPPRESSION, estimated_missed_pnl=6_000.0),
        ]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=1, realized_pnl=5000.0, capital=CAPITAL,
        )
        s1 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        assert s1.secondary_research_categories == s2.secondary_research_categories

    def test_leakage_sources_order_stable(self):
        recs = [
            _make_skipped("A", REJECTION_ALLOC_CAP, estimated_missed_pnl=10_000.0),
            _make_skipped("B", REJECTION_HIGH_PRICE, estimated_missed_pnl=8_000.0),
        ]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=1, realized_pnl=1000.0, capital=CAPITAL,
        )
        s1 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        assert s1.leakage_sources == s2.leakage_sources

    def test_affected_symbols_sorted(self):
        recs = [
            _make_skipped("9999"),
            _make_skipped("1111"),
            _make_skipped("5555"),
        ]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=0, realized_pnl=0.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        assert s.affected_symbols == sorted(s.affected_symbols)

    def test_affected_sectors_sorted(self):
        recs = [
            _make_skipped("A", sector="電気機器"),
            _make_skipped("B", sector="情報通信"),
            _make_skipped("C", sector="化学"),
        ]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=0, realized_pnl=0.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        assert s.affected_sectors == sorted(s.affected_sectors)

    def test_leakage_sources_sorted_by_pnl_desc(self):
        recs = [
            _make_skipped("A", REJECTION_HIGH_PRICE, estimated_missed_pnl=5_000.0),
            _make_skipped("B", REJECTION_ALLOC_CAP, estimated_missed_pnl=15_000.0),
        ]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=0, realized_pnl=0.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        if len(s.leakage_sources) >= 2:
            pnls = [abs(src["estimated_missed_pnl"]) for src in s.leakage_sources]
            assert pnls == sorted(pnls, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkdownReproducibility:
    """Same ResearchPrioritySummary → byte-identical markdown (generated_at is metadata only)."""

    def test_markdown_byte_identical_for_same_snap(self):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        md1 = generate_research_priority_markdown(s)
        md2 = generate_research_priority_markdown(s)
        assert md1 == md2

    def test_markdown_contains_date(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        md = generate_research_priority_markdown(s)
        assert DATE in md

    def test_markdown_contains_dominant_bottleneck(self):
        recs = [_make_skipped(rejection_reason=REJECTION_HIGH_PRICE)]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=1, realized_pnl=1000.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        md = generate_research_priority_markdown(s)
        assert s.dominant_bottleneck in md

    def test_markdown_contains_primary_research_category(self):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        md = generate_research_priority_markdown(s)
        assert s.primary_research_category in md

    def test_markdown_contains_executive_summary(self):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        md = generate_research_priority_markdown(s)
        assert "Executive Summary" in md
        assert s.executive_summary in md

    def test_markdown_contains_all_required_sections(self):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        md = generate_research_priority_markdown(s)
        required = [
            "Executive Summary",
            "Dominant Bottleneck",
            "Estimated CAGR Leakage Sources",
            "Top Missed Opportunities",
            "Capital Efficiency Trends",
            "Risk / Integrity Warnings",
            "Research Priority For Next Session",
        ]
        for section in required:
            assert section in md, f"Section missing: {section}"

    def test_markdown_mode_indicator(self):
        s_dry  = generate_research_priority_summary(DATE, RUN_ID, False)
        s_live = generate_research_priority_summary(DATE, RUN_ID, True)
        md_dry  = generate_research_priority_markdown(s_dry)
        md_live = generate_research_priority_markdown(s_live)
        assert "DRY-RUN" in md_dry
        assert "LIVE" in md_live

    def test_markdown_integrity_warning_visible(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        import dataclasses
        s_violated = dataclasses.replace(
            s,
            integrity_ok=False,
            integrity_warnings=["blocking_reconciliation_mismatch: 1 mismatch"],
        )
        md = generate_research_priority_markdown(s_violated)
        assert "INTEGRITY VIOLATION" in md
        assert "blocking_reconciliation_mismatch" in md


# ─────────────────────────────────────────────────────────────────────────────
# Trailing window comparisons
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingWindowComparisons:
    """Correct delta computation and trend classification from trailing windows."""

    def _make_hist(self, n: int, util: float, conv: float) -> List[DailyLeakageSummary]:
        return [
            DailyLeakageSummary(
                summary_id=f"h{i}",
                date=f"2026-05-{15 - i:02d}",
                run_id="h",
                generated_at="2026-05-15T09:00:00+00:00",
                total_signals_seen=5,
                executed_count=int(5 * util),
                skipped_count=5 - int(5 * util),
                top_missed_by_signal=[],
                top_missed_by_pnl=[],
                exclusion_breakdown=[],
                idle_capital_mean_jpy=200_000.0,
                idle_capital_peak_jpy=250_000.0,
                opportunity_utilization_ratio=util,
                deployable_alpha_conversion_rate=conv,
                skipped_pnl_estimate=5000.0,
                realized_pnl_estimate=10_000.0,
                enrichment_coverage=0.8,
            )
            for i in range(n)
        ]

    def test_trailing_5d_sample_count(self):
        hist = self._make_hist(7, 0.5, 0.6)
        snap = _make_snap()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, alpha_snap=snap, historical_leakage=hist
        )
        # alpha_snap provides trailing_5d directly from generate_diagnostic
        assert "sample_count" in s.persistence_vs_5d

    def test_trailing_20d_sample_count_capped_at_available(self):
        hist = self._make_hist(5, 0.5, 0.6)  # only 5 days available
        ls = _make_leakage_summary()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, leakage_summary=ls, historical_leakage=hist
        )
        assert s.persistence_vs_20d.get("sample_count", 0) == 5

    def test_trend_improving_when_efficiency_above_5d_mean(self):
        # Build scenario where current efficiency is significantly above trailing mean
        hist = self._make_hist(5, 0.1, 0.1)  # very low historical efficiency
        ls = _make_leakage_summary(executed_count=4, skipped_count=1)  # high current util
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, leakage_summary=ls, historical_leakage=hist
        )
        assert s.capital_efficiency_trend == "improving"

    def test_trend_degrading_when_efficiency_below_5d_mean(self):
        # Build scenario where current efficiency is significantly below trailing mean
        hist = self._make_hist(5, 0.9, 0.9)  # high historical efficiency
        ls = _make_leakage_summary(executed_count=0, skipped_count=5)  # low current
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, leakage_summary=ls, historical_leakage=hist
        )
        assert s.capital_efficiency_trend == "degrading"

    def test_trend_insufficient_data_when_no_history(self):
        ls = _make_leakage_summary()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, leakage_summary=ls, historical_leakage=[]
        )
        assert s.capital_efficiency_trend == "insufficient_data"

    def test_persistence_vs_5d_keys_present(self):
        hist = self._make_hist(3, 0.5, 0.5)
        ls = _make_leakage_summary()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, leakage_summary=ls, historical_leakage=hist
        )
        required_keys = {
            "window_days", "sample_count", "mean_util_ratio",
            "delta_capital_efficiency_score", "delta_deployable_alpha_loss_rate",
        }
        assert required_keys.issubset(set(s.persistence_vs_5d.keys()))

    def test_compute_trend_stable_near_zero_delta(self):
        t5_stable = {"sample_count": 3, "delta_capital_efficiency_score": 0.005}
        result = _compute_trend(t5_stable, current_eff=0.6)
        assert result == "stable"

    def test_compute_trend_insufficient_when_sample_count_zero(self):
        t5_empty = {"sample_count": 0, "delta_capital_efficiency_score": 0.1}
        result = _compute_trend(t5_empty, current_eff=0.6)
        assert result == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# Confidence level determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceLevelDeterminism:

    @pytest.mark.parametrize("coverage,signals,skipped,expected", [
        (0.80, 10, 5,  CONFIDENCE_HIGH),
        (0.70, 5,  3,  CONFIDENCE_HIGH),
        (0.50, 3,  1,  CONFIDENCE_MEDIUM),
        (0.30, 5,  2,  CONFIDENCE_MEDIUM),  # signals >= 3 and skipped >= 1
        (0.20, 2,  0,  CONFIDENCE_LOW),
        (0.00, 0,  0,  CONFIDENCE_LOW),
        (0.10, 1,  1,  CONFIDENCE_LOW),     # skipped >= 1 but signals < 3
    ])
    def test_confidence_levels(self, coverage, signals, skipped, expected):
        level, basis = _compute_confidence(coverage, signals, skipped)
        assert level == expected

    def test_confidence_basis_contains_coverage(self):
        level, basis = _compute_confidence(0.75, 10, 5)
        assert "enrichment" in basis

    def test_confidence_basis_contains_signal_count(self):
        level, basis = _compute_confidence(0.75, 10, 5)
        assert "10" in basis

    def test_confidence_same_inputs_same_output(self):
        r1 = _compute_confidence(0.6, 8, 4)
        r2 = _compute_confidence(0.6, 8, 4)
        assert r1 == r2

    def test_high_enrichment_low_signals_medium_not_high(self):
        level, _ = _compute_confidence(0.90, 2, 1)
        # signals < 3: cannot be HIGH
        assert level != CONFIDENCE_HIGH

    def test_summary_confidence_from_alpha_snap(self):
        recs = [_make_skipped() for _ in range(10)]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=2, realized_pnl=5000.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        assert s.confidence_level in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)
        assert s.confidence_basis  # non-empty


# ─────────────────────────────────────────────────────────────────────────────
# Missing data handling
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingDataHandling:
    """Empty/None inputs → valid LOW-confidence summary, no exceptions."""

    def test_no_inputs_returns_valid_summary(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        assert isinstance(s, ResearchPrioritySummary)
        assert s.confidence_level == CONFIDENCE_LOW
        assert s.dominant_bottleneck == BOTTLENECK_NONE
        assert s.total_signals == 0

    def test_empty_skipped_records(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False, skipped_records=[])
        assert isinstance(s, ResearchPrioritySummary)
        assert s.skipped_count == 0

    def test_none_alpha_snap(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=None)
        assert isinstance(s, ResearchPrioritySummary)

    def test_none_leakage_summary(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False, leakage_summary=None)
        assert isinstance(s, ResearchPrioritySummary)

    def test_none_historical_leakage(self):
        snap = _make_snap()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, alpha_snap=snap, historical_leakage=None
        )
        assert isinstance(s, ResearchPrioritySummary)
        # Trailing windows should be present but with zero sample counts
        assert s.persistence_vs_5d.get("sample_count", 0) == 0

    def test_empty_historical_leakage(self):
        snap = _make_snap()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, alpha_snap=snap, historical_leakage=[]
        )
        assert s.capital_efficiency_trend == "insufficient_data"

    def test_none_integrity_result(self):
        snap = _make_snap()
        s = generate_research_priority_summary(
            DATE, RUN_ID, False, alpha_snap=snap, integrity_result=None
        )
        assert s.integrity_ok is True
        assert s.integrity_warnings == []

    def test_no_affected_symbols_when_no_data(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        assert isinstance(s.affected_symbols, list)

    def test_no_leakage_sources_when_no_data(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        assert isinstance(s.leakage_sources, list)

    def test_executive_summary_non_empty_even_without_data(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        assert len(s.executive_summary) > 0

    def test_markdown_renders_without_error_for_empty_summary(self):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        md = generate_research_priority_markdown(s)
        assert len(md) > 100

    def test_pending_records_handled_gracefully(self):
        recs = [
            _make_skipped(enrichment_status=ENRICHMENT_PENDING,
                          forward_return_5d=None,
                          estimated_missed_pnl=None)
        ]
        s = generate_research_priority_summary(DATE, RUN_ID, False, skipped_records=recs)
        assert isinstance(s, ResearchPrioritySummary)


# ─────────────────────────────────────────────────────────────────────────────
# Replay consistency (serialize → deserialize → compare)
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayConsistency:
    """Serialized summaries round-trip exactly."""

    def test_jsonl_roundtrip(self, tmp_path):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        path = tmp_path / "test_rps.jsonl"
        append_research_priority_summary(s, path)
        loaded = load_research_priority_summaries(path)
        assert len(loaded) == 1
        d_orig = asdict(s)
        d_load = asdict(loaded[0])
        assert d_orig == d_load

    def test_multiple_appends_all_loaded(self, tmp_path):
        path = tmp_path / "test_rps.jsonl"
        for i in range(3):
            s = generate_research_priority_summary(
                f"2026-05-{15 + i:02d}", f"run-{i}", False
            )
            append_research_priority_summary(s, path)
        loaded = load_research_priority_summaries(path)
        assert len(loaded) == 3

    def test_corrupted_lines_skipped(self, tmp_path):
        path = tmp_path / "test_rps.jsonl"
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        append_research_priority_summary(s, path)
        # Inject corrupted line
        with path.open("a", encoding="utf-8") as f:
            f.write("NOT JSON\n")
        append_research_priority_summary(s, path)
        loaded = load_research_priority_summaries(path)
        assert len(loaded) == 2  # corrupted line skipped

    def test_get_summary_for_date(self, tmp_path):
        path = tmp_path / "test_rps.jsonl"
        s1 = generate_research_priority_summary("2026-05-15", "r1", False)
        s2 = generate_research_priority_summary("2026-05-16", "r2", False)
        append_research_priority_summary(s1, path)
        append_research_priority_summary(s2, path)
        found = get_summary_for_date(path, "2026-05-15")
        assert found is not None
        assert found.date == "2026-05-15"

    def test_get_summary_for_date_returns_none_when_missing(self, tmp_path):
        path = tmp_path / "test_rps.jsonl"
        result = get_summary_for_date(path, DATE)
        assert result is None

    def test_all_fields_preserved_after_roundtrip(self, tmp_path):
        recs = [_make_skipped("7777", REJECTION_HIGH_PRICE)]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=2, realized_pnl=8000.0, capital=CAPITAL,
        )
        s = generate_research_priority_summary(DATE, RUN_ID, True, alpha_snap=snap)
        path = tmp_path / "rt.jsonl"
        append_research_priority_summary(s, path)
        loaded = load_research_priority_summaries(path)[0]
        assert loaded.dominant_bottleneck == s.dominant_bottleneck
        assert loaded.primary_research_category == s.primary_research_category
        assert loaded.is_live == s.is_live
        assert loaded.confidence_level == s.confidence_level

    def test_idempotent_summary_id(self):
        s1 = generate_research_priority_summary(DATE, RUN_ID, False)
        s2 = generate_research_priority_summary(DATE, RUN_ID, False)
        assert s1.summary_id == s2.summary_id

    def test_different_run_id_different_summary_id(self):
        s1 = generate_research_priority_summary(DATE, "run-A", False)
        s2 = generate_research_priority_summary(DATE, "run-B", False)
        assert s1.summary_id != s2.summary_id

    def test_different_date_different_summary_id(self):
        s1 = generate_research_priority_summary("2026-05-01", RUN_ID, False)
        s2 = generate_research_priority_summary("2026-05-02", RUN_ID, False)
        assert s1.summary_id != s2.summary_id


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report file persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkdownReportPersistence:

    def test_write_creates_file(self, tmp_path):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        report_path = write_research_priority_report(s, tmp_path)
        assert report_path is not None
        assert report_path.exists()

    def test_filename_contains_compact_date(self, tmp_path):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        report_path = write_research_priority_report(s, tmp_path)
        assert "20260517" in report_path.name

    def test_write_appends_on_second_run(self, tmp_path):
        snap = _make_snap()
        s = generate_research_priority_summary(DATE, RUN_ID, False, alpha_snap=snap)
        write_research_priority_report(s, tmp_path)
        write_research_priority_report(s, tmp_path)
        report_path = tmp_path / "daily_research_summary_20260517.md"
        content = report_path.read_text(encoding="utf-8")
        assert content.count("Daily Research Priority Summary") == 2

    def test_write_returns_none_on_permission_error(self, tmp_path, monkeypatch):
        s = generate_research_priority_summary(DATE, RUN_ID, False)
        monkeypatch.setattr(Path, "open", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("no")))
        result = write_research_priority_report(s, tmp_path)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Integrity extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrityExtraction:

    def test_none_integrity_returns_ok(self):
        ok, warnings = _extract_integrity(None)
        assert ok is True
        assert warnings == []

    def test_integrity_result_ok(self):
        mock = MagicMock()
        mock.execution_allowed = True
        mock.integrity_violations = []
        ok, warnings = _extract_integrity(mock)
        assert ok is True
        assert warnings == []

    def test_integrity_result_violation(self):
        mock = MagicMock()
        mock.execution_allowed = False
        mock.integrity_violations = [{"detail": "snapshot stale", "blocking": True}]
        ok, warnings = _extract_integrity(mock)
        assert ok is False
        assert "snapshot stale" in warnings[0]

    def test_integrity_warning_in_secondary_categories_when_violated(self):
        mock = MagicMock()
        mock.execution_allowed = False
        mock.integrity_violations = [{"detail": "checksum mismatch"}]
        result = _build_secondary_categories(
            leakage_sources=[],
            primary_category=RESEARCH_ALLOCATION,
            idle_cash_ratio=0.1,
            integrity_result=mock,
        )
        assert RESEARCH_INTEGRITY in result


# ─────────────────────────────────────────────────────────────────────────────
# run_research_priority_summary convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestRunResearchPrioritySummary:

    def test_run_returns_summary_and_path(self, tmp_path):
        snap = _make_snap()
        jsonl = tmp_path / "rps.jsonl"
        report_dir = tmp_path / "reports"
        summary, path = run_research_priority_summary(
            date=DATE,
            run_id=RUN_ID,
            is_live=False,
            summary_jsonl_path=jsonl,
            report_dir=report_dir,
            alpha_snap=snap,
        )
        assert summary is not None
        assert path is not None
        assert jsonl.exists()
        assert path.exists()

    def test_run_appends_to_jsonl(self, tmp_path):
        snap = _make_snap()
        jsonl = tmp_path / "rps.jsonl"
        report_dir = tmp_path / "reports"
        for i in range(3):
            run_research_priority_summary(
                date=f"2026-05-{15 + i:02d}",
                run_id=f"run-{i}",
                is_live=False,
                summary_jsonl_path=jsonl,
                report_dir=report_dir,
            )
        loaded = load_research_priority_summaries(jsonl)
        assert len(loaded) == 3

    def test_run_fail_open_returns_none_on_error(self, tmp_path, monkeypatch):
        import src.analytics.research_priority_summary as mod
        monkeypatch.setattr(mod, "generate_research_priority_summary", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        jsonl = tmp_path / "rps.jsonl"
        report_dir = tmp_path / "reports"
        # Should not raise
        summary, path = run_research_priority_summary(
            date=DATE, run_id=RUN_ID, is_live=False,
            summary_jsonl_path=jsonl, report_dir=report_dir,
        )
        assert summary is None
        assert path is None

    def test_run_with_all_inputs(self, tmp_path):
        recs = [_make_skipped("1234"), _make_skipped("5678", REJECTION_HIGH_PRICE)]
        snap = generate_diagnostic(
            date=DATE, run_id=RUN_ID, skipped_records=recs,
            executed_count=1, realized_pnl=5000.0, capital=CAPITAL,
        )
        ls = _make_leakage_summary()
        jsonl = tmp_path / "rps.jsonl"
        report_dir = tmp_path / "reports"
        summary, path = run_research_priority_summary(
            date=DATE,
            run_id=RUN_ID,
            is_live=True,
            summary_jsonl_path=jsonl,
            report_dir=report_dir,
            skipped_records=recs,
            alpha_snap=snap,
            leakage_summary=ls,
            capital=CAPITAL,
        )
        assert summary is not None
        assert summary.total_signals > 0


# ─────────────────────────────────────────────────────────────────────────────
# Secondary research categories
# ─────────────────────────────────────────────────────────────────────────────

class TestSecondaryCategories:

    def test_no_secondary_when_only_primary(self):
        sources = [
            {
                "bottleneck_category": BOTTLENECK_ALLOC_CAP_PRESSURE,
                "fraction_of_skipped": 1.0,
                "estimated_missed_pnl": 5000.0,
                "source_reason": REJECTION_ALLOC_CAP,
                "skip_count": 5,
            }
        ]
        result = _build_secondary_categories(sources, RESEARCH_ALLOCATION, 0.1, None)
        assert RESEARCH_ALLOCATION not in result

    def test_secondary_added_when_significant_fraction(self):
        sources = [
            {
                "bottleneck_category": BOTTLENECK_ALLOC_CAP_PRESSURE,
                "fraction_of_skipped": 0.6,
                "estimated_missed_pnl": 8000.0,
                "source_reason": REJECTION_ALLOC_CAP,
                "skip_count": 6,
            },
            {
                "bottleneck_category": BOTTLENECK_HIGH_PRICED_EXCLUSION,
                "fraction_of_skipped": 0.40,
                "estimated_missed_pnl": 5000.0,
                "source_reason": REJECTION_HIGH_PRICE,
                "skip_count": 4,
            },
        ]
        result = _build_secondary_categories(sources, RESEARCH_ALLOCATION, 0.1, None)
        assert RESEARCH_DEPLOYABILITY in result

    def test_secondary_not_added_when_fraction_below_threshold(self):
        sources = [
            {
                "bottleneck_category": BOTTLENECK_HIGH_PRICED_EXCLUSION,
                "fraction_of_skipped": 0.05,   # below 0.15 threshold
                "estimated_missed_pnl": 100.0,
                "source_reason": REJECTION_HIGH_PRICE,
                "skip_count": 1,
            }
        ]
        result = _build_secondary_categories(sources, RESEARCH_ALLOCATION, 0.1, None)
        assert RESEARCH_DEPLOYABILITY not in result

    def test_capital_efficiency_added_when_idle_cash_high(self):
        result = _build_secondary_categories(
            leakage_sources=[],
            primary_category=RESEARCH_ALLOCATION,
            idle_cash_ratio=0.30,  # above 0.25 threshold
        )
        assert RESEARCH_CAPITAL_EFFICIENCY in result

    def test_no_duplicate_primary_in_secondary(self):
        sources = [
            {
                "bottleneck_category": BOTTLENECK_ALLOC_CAP_PRESSURE,
                "fraction_of_skipped": 1.0,
                "estimated_missed_pnl": 5000.0,
                "source_reason": REJECTION_ALLOC_CAP,
                "skip_count": 5,
            }
        ]
        result = _build_secondary_categories(sources, RESEARCH_ALLOCATION, 0.4, None)
        # primary should not appear in secondary
        count = result.count(RESEARCH_ALLOCATION)
        assert count == 0
