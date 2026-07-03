"""
tests/analytics/test_executed_vs_skipped_expectancy.py

Tests for src/analytics/executed_vs_skipped_expectancy.py
"""
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from src.analytics.executed_vs_skipped_expectancy import (
    ALL_SKIP_REASONS,
    ATR_BUCKETS,
    DEFAULT_STORE_PATH,
    SKIP_CAPITAL_CONSTRAINT,
    SKIP_COOLDOWN,
    SKIP_DUPLICATE_SIGNAL,
    SKIP_EXECUTION_FAILURE,
    SKIP_LIQUIDITY_FILTER,
    SKIP_MANUAL_OVERRIDE,
    SKIP_PORTFOLIO_HEAT,
    SKIP_SECTOR_EXPOSURE,
    SKIP_UNIVERSE_FILTER,
    SKIP_UNKNOWN,
    SKIP_VOLATILITY_FILTER,
    CounterfactualPortfolioAnalyzer,
    CounterfactualSummary,
    ExpectancyAnalyticsEngine,
    ExpectancyReport,
    ExpectancySlice,
    IntegrityReport,
    IntegrityValidator,
    IntegrityViolation,
    SCHEMA_VERSION,
    SkipReasonClassifier,
    TradeOpportunityRecord,
    _atr_bucket_label,
    _opt_float,
    _safe_mean,
    _sha256,
    _slot_bucket_label,
    _win_rate,
    append_opportunity,
    deduplicate_opportunities,
    enrich_forward_returns,
    load_opportunities,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(
    symbol="6506.T",
    eval_date="2026-05-19",
    executed=True,
    skip_reason=None,
    capital_available_pct=0.31,
    portfolio_heat=0.42,
    slot_utilization=0.80,
    sector="transportation_equipment",
    market_regime="trend_persistent",
    atr_pct=0.028,
    rs_rank=4,
    entry_score=0.82,
    liquidity_score=0.77,
    position_lifecycle_available=True,
    forward_5d_return=None,
    forward_20d_return=None,
    enrichment_status="pending",
) -> TradeOpportunityRecord:
    return TradeOpportunityRecord.create(
        eval_date=eval_date,
        symbol=symbol,
        executed=executed,
        skip_reason=skip_reason,
        capital_available_pct=capital_available_pct,
        portfolio_heat=portfolio_heat,
        slot_utilization=slot_utilization,
        sector=sector,
        market_regime=market_regime,
        atr_pct=atr_pct,
        rs_rank=rs_rank,
        entry_score=entry_score,
        liquidity_score=liquidity_score,
        position_lifecycle_available=position_lifecycle_available,
        forward_5d_return=forward_5d_return,
        forward_20d_return=forward_20d_return,
        enrichment_status=enrichment_status,
    )


def _enriched(r: TradeOpportunityRecord, f5: float = 0.05, f20: float = 0.12) -> TradeOpportunityRecord:
    import dataclasses
    return dataclasses.replace(r, forward_5d_return=f5, forward_20d_return=f20, enrichment_status="enriched")


# ─────────────────────────────────────────────────────────────────────────────
# TradeOpportunityRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeOpportunityRecord:
    def test_create_executed(self):
        r = _make_record(executed=True, skip_reason=None)
        assert r.executed is True
        assert r.skip_reason is None
        assert r.schema_version == SCHEMA_VERSION
        assert len(r.opportunity_id) == 64  # sha256 hex

    def test_create_skipped(self):
        r = _make_record(executed=False, skip_reason=SKIP_CAPITAL_CONSTRAINT)
        assert r.executed is False
        assert r.skip_reason == SKIP_CAPITAL_CONSTRAINT

    def test_opportunity_id_determinism(self):
        r1 = _make_record(symbol="7203.T", eval_date="2026-05-01", executed=True)
        r2 = _make_record(symbol="7203.T", eval_date="2026-05-01", executed=True)
        assert r1.opportunity_id == r2.opportunity_id

    def test_opportunity_id_differs_by_executed(self):
        r1 = _make_record(symbol="7203.T", eval_date="2026-05-01", executed=True, skip_reason=None)
        r2 = _make_record(symbol="7203.T", eval_date="2026-05-01", executed=False, skip_reason=SKIP_CAPITAL_CONSTRAINT)
        assert r1.opportunity_id != r2.opportunity_id

    def test_opportunity_id_differs_by_skip_reason(self):
        r1 = _make_record(executed=False, skip_reason=SKIP_CAPITAL_CONSTRAINT)
        r2 = _make_record(executed=False, skip_reason=SKIP_LIQUIDITY_FILTER)
        assert r1.opportunity_id != r2.opportunity_id

    def test_round_trip_serialization(self):
        r = _make_record(forward_5d_return=0.041, forward_20d_return=0.118, enrichment_status="enriched")
        d = r.to_dict()
        r2 = TradeOpportunityRecord.from_dict(d)
        assert r2.opportunity_id == r.opportunity_id
        assert r2.forward_5d_return == pytest.approx(0.041)
        assert r2.forward_20d_return == pytest.approx(0.118)
        assert r2.enrichment_status == "enriched"

    def test_from_dict_missing_optional_fields(self):
        d = {
            "opportunity_id": "abc",
            "schema_version": 1,
            "eval_date": "2026-05-01",
            "symbol": "6506.T",
            "executed": True,
            "skip_reason": None,
            "capital_available_pct": 0.5,
            "portfolio_heat": 0.3,
            "slot_utilization": 0.5,
            "sector": "auto",
            "market_regime": "bull",
            "atr_pct": 0.02,
            "rs_rank": 1,
            "entry_score": 0.8,
            "liquidity_score": 0.9,
            "position_lifecycle_available": True,
            "enrichment_status": "pending",
        }
        r = TradeOpportunityRecord.from_dict(d)
        assert r.forward_5d_return is None
        assert r.forward_20d_return is None

    def test_immutability_via_dataclass(self):
        r = _make_record()
        with pytest.raises((AttributeError, TypeError)):
            r.symbol = "CHANGED"  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# SkipReasonClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestSkipReasonClassifier:
    def setup_method(self):
        self.clf = SkipReasonClassifier()

    def test_classify_none_returns_unknown(self):
        assert self.clf.classify(None) == SKIP_UNKNOWN

    def test_classify_capital_constraint(self):
        assert self.clf.classify("insufficient_cash") == SKIP_CAPITAL_CONSTRAINT
        assert self.clf.classify("alloc_cap_exceeded") == SKIP_CAPITAL_CONSTRAINT
        assert self.clf.classify("capital_limit") == SKIP_CAPITAL_CONSTRAINT

    def test_classify_portfolio_heat(self):
        assert self.clf.classify("portfolio_heat_limit") == SKIP_PORTFOLIO_HEAT
        assert self.clf.classify("dd_limit_reached") == SKIP_PORTFOLIO_HEAT

    def test_classify_sector_exposure(self):
        assert self.clf.classify("sector_exposure_limit") == SKIP_SECTOR_EXPOSURE
        assert self.clf.classify("concentration_throttle") == SKIP_SECTOR_EXPOSURE

    def test_classify_liquidity_filter(self):
        assert self.clf.classify("liquidity_gate") == SKIP_LIQUIDITY_FILTER
        assert self.clf.classify("low_volume") == SKIP_LIQUIDITY_FILTER

    def test_classify_volatility_filter(self):
        assert self.clf.classify("atr_too_high") == SKIP_VOLATILITY_FILTER
        assert self.clf.classify("volatility_filter") == SKIP_VOLATILITY_FILTER

    def test_classify_universe_filter(self):
        assert self.clf.classify("universe_filter") == SKIP_UNIVERSE_FILTER
        assert self.clf.classify("epoch_blocked") == SKIP_UNIVERSE_FILTER
        assert self.clf.classify("manifest_rejected") == SKIP_UNIVERSE_FILTER

    def test_classify_duplicate_signal(self):
        assert self.clf.classify("duplicate_signal") == SKIP_DUPLICATE_SIGNAL
        assert self.clf.classify("already_ordered") == SKIP_DUPLICATE_SIGNAL

    def test_classify_cooldown(self):
        assert self.clf.classify("cooldown_active") == SKIP_COOLDOWN
        assert self.clf.classify("rate_limit") == SKIP_COOLDOWN
        assert self.clf.classify("throttle_applied") == SKIP_COOLDOWN

    def test_classify_execution_failure(self):
        assert self.clf.classify("execution_failed") == SKIP_EXECUTION_FAILURE
        assert self.clf.classify("broker_error") == SKIP_EXECUTION_FAILURE

    def test_classify_manual_override(self):
        assert self.clf.classify("manual_override") == SKIP_MANUAL_OVERRIDE
        assert self.clf.classify("override_active") == SKIP_MANUAL_OVERRIDE

    def test_classify_unknown(self):
        assert self.clf.classify("some_weird_reason_xyz") == SKIP_UNKNOWN
        assert self.clf.classify("") == SKIP_UNKNOWN

    def test_classify_all_canonical_reasons_pass_through(self):
        for reason in ALL_SKIP_REASONS:
            r = _make_record(executed=False, skip_reason=reason)
            result = self.clf.classify_record(r)
            assert result == reason, f"{reason} → {result}"

    def test_classify_record_executed_returns_empty(self):
        r = _make_record(executed=True, skip_reason=None)
        assert self.clf.classify_record(r) == ""

    def test_classify_record_raw_reason_gets_classified(self):
        r = _make_record(executed=False, skip_reason="not_enough_cash_available")
        result = self.clf.classify_record(r)
        assert result == SKIP_CAPITAL_CONSTRAINT


# ─────────────────────────────────────────────────────────────────────────────
# ExpectancyAnalyticsEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectancyAnalyticsEngine:
    def setup_method(self):
        self.engine = ExpectancyAnalyticsEngine()

    def _mixed_dataset(self):
        executed = [
            _enriched(_make_record(symbol="A", market_regime="bull", sector="tech", atr_pct=0.02), f5=0.03, f20=0.08),
            _enriched(_make_record(symbol="B", market_regime="bull", sector="tech", atr_pct=0.03), f5=0.01, f20=0.04),
            _enriched(_make_record(symbol="C", market_regime="bear", sector="auto", atr_pct=0.05), f5=-0.02, f20=-0.06),
        ]
        skipped = [
            _enriched(_make_record(symbol="D", executed=False, skip_reason=SKIP_CAPITAL_CONSTRAINT,
                                   market_regime="bull", sector="tech", atr_pct=0.02), f5=0.08, f20=0.15),
            _enriched(_make_record(symbol="E", executed=False, skip_reason=SKIP_LIQUIDITY_FILTER,
                                   market_regime="bear", sector="auto", atr_pct=0.05), f5=0.04, f20=0.10),
        ]
        return executed + skipped

    def test_overall_expectancy(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        assert isinstance(report, ExpectancyReport)
        assert report.n_total == 5
        assert report.overall.n_executed == 3
        assert report.overall.n_skipped == 2

    def test_expectancy_delta_positive_when_skipped_outperforms(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        # Skipped 5d avg = (0.08+0.04)/2=0.06; Executed avg = (0.03+0.01-0.02)/3=0.0067
        assert report.overall.expectancy_delta_5d is not None
        assert report.overall.expectancy_delta_5d > 0

    def test_by_skip_reason_slices_populated(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        assert SKIP_CAPITAL_CONSTRAINT in report.by_skip_reason
        assert SKIP_LIQUIDITY_FILTER in report.by_skip_reason

    def test_by_regime_slices(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        assert "bull" in report.by_regime
        assert "bear" in report.by_regime

    def test_by_sector_slices(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        assert "tech" in report.by_sector
        assert "auto" in report.by_sector

    def test_by_atr_bucket(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        assert len(report.by_atr_bucket) > 0

    def test_by_slot_utilization(self):
        records = [
            _enriched(_make_record(slot_utilization=0.2), f5=0.02),
            _enriched(_make_record(symbol="X", executed=False, skip_reason=SKIP_COOLDOWN,
                                   slot_utilization=0.8), f5=0.05),
        ]
        report = self.engine.analyze(records)
        assert len(report.by_slot_utilization) > 0

    def test_opportunity_capture_ratio(self):
        recs = self._mixed_dataset()
        report = self.engine.analyze(recs)
        assert report.overall.opportunity_capture_ratio == pytest.approx(3 / 5, rel=1e-3)

    def test_empty_records_returns_report(self):
        report = self.engine.analyze([])
        assert report.overall.n_executed == 0
        assert report.overall.n_skipped == 0
        assert report.overall.executed_expectancy_5d is None

    def test_pending_records_excluded(self):
        recs = [_make_record(enrichment_status="pending")]
        report = self.engine.analyze(recs)
        assert report.overall.n_executed == 0

    def test_skipped_convexity_score(self):
        skipped = [
            _enriched(_make_record(symbol="X", executed=False, skip_reason=SKIP_COOLDOWN), f5=0.1),
            _enriched(_make_record(symbol="Y", executed=False, skip_reason=SKIP_COOLDOWN,
                                   eval_date="2026-05-18"), f5=0.2),
        ]
        executed = [_enriched(_make_record(), f5=0.01)]
        report = self.engine.analyze(executed + skipped)
        assert report.overall.skipped_convexity_score is not None
        assert report.overall.skipped_convexity_score >= 0

    def test_win_rate_computed(self):
        recs = [
            _enriched(_make_record(symbol="A"), f5=0.05),
            _enriched(_make_record(symbol="B", eval_date="2026-05-18"), f5=-0.02),
            _enriched(_make_record(symbol="C", eval_date="2026-05-17"), f5=0.01),
        ]
        report = self.engine.analyze(recs)
        assert report.overall.executed_win_rate == pytest.approx(2 / 3, rel=1e-3)

    def test_regime_segmentation_correctness(self):
        bull_exec = _enriched(_make_record(market_regime="bull"), f5=0.03)
        bear_exec = _enriched(_make_record(symbol="Z", market_regime="bear"), f5=-0.01)
        bull_skip = _enriched(_make_record(symbol="Y", executed=False, skip_reason=SKIP_CAPITAL_CONSTRAINT,
                                           market_regime="bull"), f5=0.07)
        report = self.engine.analyze([bull_exec, bear_exec, bull_skip])
        bull_slice = report.by_regime["bull"]
        assert bull_slice.n_executed == 1
        assert bull_slice.n_skipped == 1

    def test_sector_segmentation_correctness(self):
        r1 = _enriched(_make_record(sector="steel"), f5=0.02)
        r2 = _enriched(_make_record(symbol="X", executed=False, skip_reason=SKIP_SECTOR_EXPOSURE,
                                    sector="steel"), f5=0.06)
        report = self.engine.analyze([r1, r2])
        steel = report.by_sector["steel"]
        assert steel.n_executed == 1
        assert steel.n_skipped == 1
        assert steel.expectancy_delta_5d == pytest.approx(0.04, rel=1e-3)

    def test_atr_bucket_segmentation(self):
        r_low = _enriched(_make_record(atr_pct=0.01), f5=0.02)
        r_high = _enriched(_make_record(symbol="X", atr_pct=0.05), f5=0.04)
        report = self.engine.analyze([r_low, r_high])
        assert "<1.5%" in report.by_atr_bucket
        assert "4%+" in report.by_atr_bucket


# ─────────────────────────────────────────────────────────────────────────────
# CounterfactualPortfolioAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestCounterfactualPortfolioAnalyzer:
    def setup_method(self):
        self.analyzer = CounterfactualPortfolioAnalyzer()

    def _dataset(self):
        executed = [
            _enriched(_make_record(symbol="A", sector="auto", market_regime="bull"), f5=0.02, f20=0.05),
            _enriched(_make_record(symbol="B", sector="tech", market_regime="bull"), f5=0.01, f20=0.03),
        ]
        skipped = [
            _enriched(_make_record(symbol="C", executed=False, skip_reason=SKIP_CAPITAL_CONSTRAINT,
                                   sector="auto", market_regime="bull"), f5=0.09, f20=0.18),
            _enriched(_make_record(symbol="D", executed=False, skip_reason=SKIP_LIQUIDITY_FILTER,
                                   sector="tech", market_regime="bear"), f5=0.05, f20=0.12),
        ]
        return executed + skipped

    def test_missed_alpha_positive_when_skipped_outperforms(self):
        summary = self.analyzer.analyze(self._dataset())
        assert isinstance(summary, CounterfactualSummary)
        assert summary.estimated_missed_alpha_5d is not None
        assert summary.estimated_missed_alpha_5d > 0

    def test_n_skipped_with_attribution(self):
        summary = self.analyzer.analyze(self._dataset())
        assert summary.n_skipped_with_attribution == 2

    def test_persistent_skip_bias_keys(self):
        summary = self.analyzer.analyze(self._dataset())
        assert SKIP_CAPITAL_CONSTRAINT in summary.persistent_skip_bias
        assert SKIP_LIQUIDITY_FILTER in summary.persistent_skip_bias

    def test_systematic_under_allocation_populated(self):
        summary = self.analyzer.analyze(self._dataset())
        assert isinstance(summary.systematic_under_allocation, list)
        # Both skip reasons have positive skipped returns → under-allocation
        for reason in [SKIP_CAPITAL_CONSTRAINT, SKIP_LIQUIDITY_FILTER]:
            assert reason in summary.systematic_under_allocation

    def test_top_missed_sectors(self):
        summary = self.analyzer.analyze(self._dataset())
        assert isinstance(summary.top_missed_sectors, list)
        assert len(summary.top_missed_sectors) <= 5

    def test_top_missed_regimes(self):
        summary = self.analyzer.analyze(self._dataset())
        assert isinstance(summary.top_missed_regimes, list)

    def test_empty_records(self):
        summary = self.analyzer.analyze([])
        assert summary.n_skipped_with_attribution == 0
        assert summary.estimated_missed_alpha_5d is None

    def test_no_skipped_records(self):
        executed = [_enriched(_make_record(), f5=0.03)]
        summary = self.analyzer.analyze(executed)
        assert summary.n_skipped_with_attribution == 0
        assert summary.mean_skipped_5d_return is None

    def test_no_executed_records(self):
        skipped = [
            _enriched(_make_record(executed=False, skip_reason=SKIP_COOLDOWN), f5=0.04)
        ]
        summary = self.analyzer.analyze(skipped)
        assert summary.mean_executed_5d_return is None
        assert summary.estimated_missed_alpha_5d is None

    def test_missing_attribution_handling(self):
        records = [
            _make_record(enrichment_status="pending"),   # no attribution
            _enriched(_make_record(symbol="B", executed=False,
                                   skip_reason=SKIP_CAPITAL_CONSTRAINT), f5=0.06),
        ]
        summary = self.analyzer.analyze(records)
        assert summary.n_skipped_with_attribution == 1

    def test_20d_missed_alpha(self):
        executed = [_enriched(_make_record(), f5=0.02, f20=0.05)]
        skipped  = [_enriched(_make_record(symbol="X", executed=False,
                                           skip_reason=SKIP_PORTFOLIO_HEAT), f5=0.06, f20=0.20)]
        summary = self.analyzer.analyze(executed + skipped)
        assert summary.estimated_missed_alpha_20d == pytest.approx(0.15, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# IntegrityValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrityValidator:
    def setup_method(self):
        self.v = IntegrityValidator()

    def test_clean_records_pass(self):
        records = [
            _make_record(symbol="A"),
            _make_record(symbol="B", eval_date="2026-05-18"),
        ]
        report = self.v.validate(records, as_of_date="2026-05-20")
        assert report.passed is True
        assert report.n_violations == 0

    def test_duplicate_opportunity_id_detected(self):
        r = _make_record()
        report = self.v.validate([r, r], as_of_date="2026-05-20")
        assert not report.passed
        types = [v.violation_type for v in report.violations]
        assert "duplicate_opportunity_id" in types

    def test_future_leakage_detected(self):
        r = _make_record(eval_date="2099-01-01")
        report = self.v.validate([r], as_of_date="2026-05-20")
        assert not report.passed
        types = [v.violation_type for v in report.violations]
        assert "future_leakage" in types

    def test_enriched_without_attribution_flagged(self):
        import dataclasses
        r = dataclasses.replace(_make_record(), enrichment_status="enriched",
                                forward_5d_return=None, forward_20d_return=None)
        report = self.v.validate([r], as_of_date="2026-05-20")
        types = [v.violation_type for v in report.violations]
        assert "missing_forward_attribution" in types

    def test_enriched_with_partial_attribution_passes(self):
        import dataclasses
        r = dataclasses.replace(_make_record(), enrichment_status="enriched",
                                forward_5d_return=0.04, forward_20d_return=None)
        report = self.v.validate([r], as_of_date="2026-05-20")
        types = [v.violation_type for v in report.violations]
        assert "missing_forward_attribution" not in types

    def test_lifecycle_linkage_missing_flagged(self):
        import dataclasses
        r = dataclasses.replace(_make_record(executed=True), position_lifecycle_available=False)
        report = self.v.validate([r], as_of_date="2026-05-20")
        types = [v.violation_type for v in report.violations]
        assert "lifecycle_linkage_missing" in types

    def test_skipped_without_lifecycle_ok(self):
        r = _make_record(executed=False, skip_reason=SKIP_COOLDOWN, position_lifecycle_available=False)
        report = self.v.validate([r], as_of_date="2026-05-20")
        types = [v.violation_type for v in report.violations]
        assert "lifecycle_linkage_missing" not in types

    def test_n_records_correct(self):
        records = [_make_record(symbol=s) for s in ["A", "B", "C"]]
        report = self.v.validate(records, as_of_date="2026-05-20")
        assert report.n_records == 3

    def test_multiple_violations_reported(self):
        r = _make_record(eval_date="2099-01-01")
        # Same record twice = duplicate + future_leakage
        report = self.v.validate([r, r], as_of_date="2026-05-20")
        assert report.n_violations >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "evs.jsonl"
        r1 = _make_record(symbol="A")
        r2 = _make_record(symbol="B", eval_date="2026-05-18")
        append_opportunity(r1, path)
        append_opportunity(r2, path)
        loaded = load_opportunities(path)
        assert len(loaded) == 2
        symbols = {r.symbol for r in loaded}
        assert symbols == {"A", "B"}

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert load_opportunities(path) == []

    def test_load_skips_corrupted_lines(self, tmp_path):
        path = tmp_path / "evs.jsonl"
        r = _make_record()
        append_opportunity(r, path)
        with path.open("a", encoding="utf-8") as f:
            f.write("NOT_JSON\n")
        loaded = load_opportunities(path)
        assert len(loaded) == 1

    def test_append_durability_utf8(self, tmp_path):
        path = tmp_path / "evs.jsonl"
        r = _make_record(sector="輸送用機器")
        append_opportunity(r, path)
        raw = path.read_text(encoding="utf-8")
        assert "輸送用機器" in raw

    def test_serialization_sort_keys_deterministic(self, tmp_path):
        path1 = tmp_path / "a.jsonl"
        path2 = tmp_path / "b.jsonl"
        r = _make_record()
        append_opportunity(r, path1)
        append_opportunity(r, path2)
        assert path1.read_text(encoding="utf-8") == path2.read_text(encoding="utf-8")

    def test_append_fail_open(self, tmp_path):
        path = Path("/nonexistent_directory_xyz/evs.jsonl")
        r = _make_record()
        # Should not raise
        append_opportunity(r, path)

    def test_load_preserves_forward_returns(self, tmp_path):
        path = tmp_path / "evs.jsonl"
        r = _enriched(_make_record(), f5=0.041, f20=0.118)
        append_opportunity(r, path)
        loaded = load_opportunities(path)
        assert loaded[0].forward_5d_return == pytest.approx(0.041)
        assert loaded[0].forward_20d_return == pytest.approx(0.118)


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_dedup_identical_records(self):
        r = _make_record()
        result = deduplicate_opportunities([r, r])
        assert len(result) == 1

    def test_dedup_prefers_enriched_over_pending(self):
        r_pending = _make_record()
        r_enriched = _enriched(r_pending)
        result = deduplicate_opportunities([r_pending, r_enriched])
        assert len(result) == 1
        assert result[0].enrichment_status == "enriched"

    def test_dedup_stable_sort_by_eval_date(self):
        r1 = _make_record(symbol="A", eval_date="2026-05-01")
        r2 = _make_record(symbol="B", eval_date="2026-05-02")
        r3 = _make_record(symbol="C", eval_date="2026-04-30")
        result = deduplicate_opportunities([r2, r1, r3])
        dates = [r.eval_date for r in result]
        assert dates == sorted(dates)

    def test_dedup_distinct_records_kept(self):
        r1 = _make_record(symbol="A")
        r2 = _make_record(symbol="B")
        result = deduplicate_opportunities([r1, r2])
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichForwardReturns:
    def test_enriches_pending_record(self):
        r = _make_record(symbol="6506.T")
        result = enrich_forward_returns([r], {"6506.T": {"5d": 0.041, "20d": 0.118}})
        assert len(result) == 1
        assert result[0].enrichment_status == "enriched"
        assert result[0].forward_5d_return == pytest.approx(0.041)
        assert result[0].forward_20d_return == pytest.approx(0.118)

    def test_skips_already_enriched(self):
        r = _enriched(_make_record(), f5=0.05, f20=0.10)
        result = enrich_forward_returns([r], {"6506.T": {"5d": 0.99, "20d": 0.99}})
        assert result[0].forward_5d_return == pytest.approx(0.05)

    def test_no_data_leaves_pending(self):
        r = _make_record(symbol="UNKNOWN")
        result = enrich_forward_returns([r], {})
        assert result[0].enrichment_status == "pending"

    def test_partial_data_enriched(self):
        r = _make_record(symbol="6506.T")
        result = enrich_forward_returns([r], {"6506.T": {"5d": 0.03, "20d": None}})
        assert result[0].enrichment_status == "enriched"
        assert result[0].forward_5d_return == pytest.approx(0.03)
        assert result[0].forward_20d_return is None

    def test_deterministic_output(self):
        r = _make_record(symbol="7203.T")
        fwd = {"7203.T": {"5d": 0.02, "20d": 0.08}}
        r1 = enrich_forward_returns([r], fwd)
        r2 = enrich_forward_returns([r], fwd)
        assert r1[0].forward_5d_return == r2[0].forward_5d_return
        assert r1[0].forward_20d_return == r2[0].forward_20d_return


# ─────────────────────────────────────────────────────────────────────────────
# All skip reasons covered
# ─────────────────────────────────────────────────────────────────────────────

class TestAllSkipReasons:
    """Ensure every skip reason can be created, stored, and analyzed."""

    def test_all_reasons_roundtrip(self, tmp_path):
        path = tmp_path / "evs.jsonl"
        clf = SkipReasonClassifier()
        engine = ExpectancyAnalyticsEngine()
        records = []
        for i, reason in enumerate(ALL_SKIP_REASONS):
            r = _enriched(
                _make_record(
                    symbol=f"SYM{i:02d}.T",
                    eval_date=f"2026-05-{i+1:02d}",
                    executed=False,
                    skip_reason=reason,
                ),
                f5=float(i) * 0.01,
                f20=float(i) * 0.02,
            )
            records.append(r)
            append_opportunity(r, path)

        loaded = load_opportunities(path)
        assert len(loaded) == len(ALL_SKIP_REASONS)

        for r in loaded:
            classified = clf.classify_record(r)
            assert classified in ALL_SKIP_REASONS, f"Unrecognized: {r.skip_reason} → {classified}"

        report = engine.analyze(records)
        assert report.overall.n_skipped == len(ALL_SKIP_REASONS)

    def test_mixed_executed_skipped(self):
        engine = ExpectancyAnalyticsEngine()
        executed = [
            _enriched(_make_record(symbol=f"E{i}", eval_date=f"2026-05-{i+1:02d}"), f5=0.02)
            for i in range(5)
        ]
        skipped = [
            _enriched(_make_record(symbol=f"S{i}", eval_date=f"2026-05-{i+1:02d}",
                                   executed=False, skip_reason=r), f5=0.05)
            for i, r in enumerate(ALL_SKIP_REASONS)
        ]
        report = engine.analyze(executed + skipped)
        assert report.overall.n_executed == 5
        assert report.overall.n_skipped == len(ALL_SKIP_REASONS)
        assert report.overall.expectancy_delta_5d > 0


# ─────────────────────────────────────────────────────────────────────────────
# Replay determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayDeterminism:
    def test_opportunity_id_stable_across_runs(self):
        kwargs = dict(
            eval_date="2026-05-19", symbol="6506.T", executed=False,
            skip_reason=SKIP_CAPITAL_CONSTRAINT, capital_available_pct=0.31,
            portfolio_heat=0.42, slot_utilization=0.80,
            sector="transportation_equipment", market_regime="trend_persistent",
            atr_pct=0.028, rs_rank=4, entry_score=0.82, liquidity_score=0.77,
            position_lifecycle_available=True,
        )
        r1 = TradeOpportunityRecord.create(**kwargs)
        r2 = TradeOpportunityRecord.create(**kwargs)
        assert r1.opportunity_id == r2.opportunity_id

    def test_expectancy_report_deterministic(self):
        records = [
            _enriched(_make_record(symbol="A", market_regime="bull"), f5=0.03),
            _enriched(_make_record(symbol="B", executed=False, skip_reason=SKIP_COOLDOWN,
                                   market_regime="bull"), f5=0.07),
        ]
        engine = ExpectancyAnalyticsEngine()
        r1 = engine.analyze(records)
        r2 = engine.analyze(records)
        assert r1.overall.executed_expectancy_5d == r2.overall.executed_expectancy_5d
        assert r1.overall.skipped_expectancy_5d == r2.overall.skipped_expectancy_5d
        assert r1.overall.expectancy_delta_5d == r2.overall.expectancy_delta_5d

    def test_serialization_deterministic(self):
        r = _make_record()
        d1 = json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False)
        d2 = json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False)
        assert d1 == d2


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_safe_mean_empty(self):
        assert _safe_mean([]) is None

    def test_safe_mean_ignores_none(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_win_rate_empty(self):
        assert _win_rate([]) is None

    def test_win_rate_all_positive(self):
        assert _win_rate([0.1, 0.2, 0.3]) == pytest.approx(1.0)

    def test_win_rate_mixed(self):
        assert _win_rate([0.1, -0.1, 0.1]) == pytest.approx(2 / 3, rel=1e-3)

    def test_atr_bucket_low(self):
        assert _atr_bucket_label(0.01) == "<1.5%"

    def test_atr_bucket_mid(self):
        assert _atr_bucket_label(0.02) == "1.5-2.5%"

    def test_atr_bucket_high(self):
        assert _atr_bucket_label(0.06) == "4%+"

    def test_slot_bucket_low(self):
        assert _slot_bucket_label(0.1) == "0-33%"

    def test_slot_bucket_mid(self):
        assert _slot_bucket_label(0.5) == "33-67%"

    def test_slot_bucket_high(self):
        assert _slot_bucket_label(0.9) == "67-100%"

    def test_opt_float_none(self):
        assert _opt_float(None) is None

    def test_opt_float_valid(self):
        assert _opt_float("0.041") == pytest.approx(0.041)

    def test_opt_float_invalid(self):
        assert _opt_float("not_a_number") is None

    def test_sha256_deterministic(self):
        assert _sha256("hello") == _sha256("hello")
        assert _sha256("hello") != _sha256("world")
