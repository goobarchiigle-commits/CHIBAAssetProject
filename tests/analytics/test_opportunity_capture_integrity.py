"""
tests/analytics/test_opportunity_capture_integrity.py

Opportunity Capture完全稼働化（2026-07-08）の回帰テスト。
対象: build_skipped_opportunity_records() / enrich_and_rewrite_store() /
      run_id が EVS(build_opportunity_records)と一致すること /
      analyze_opportunity_capture()のmissed_alpha_score/forward_return計算。
"""
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.analytics.executed_vs_skipped_expectancy import build_opportunity_records
from src.analytics.skipped_opportunity_analytics import (
    build_skipped_opportunity_records,
    enrich_and_rewrite_store,
    load_skipped_opportunities,
    append_skipped_opportunity,
    REJECTION_POSITION_FULL,
    REJECTION_HIGH_PRICE,
    ENRICHMENT_ENRICHED,
    ENRICHMENT_PENDING,
)
from src.analytics.weekly_market_intelligence import analyze_opportunity_capture


def _sig(symbol, rsr=80.0, rank=1, holding=False, entry_price=0.0):
    return {
        "symbol": symbol, "signal": 1, "currently_holding": holding,
        "rsr": rsr, "rsr_rank": rank, "sector": "テスト", "entry_price": entry_price,
    }


class TestBuildSkippedOpportunityRecords:
    def test_capacity_skip_maps_to_position_full(self):
        signals = [_sig("8035.T", rank=2)]
        stage_audit = [{"symbol": "8035.T", "stage": "CAPACITY", "passed": False, "reason": "position_full"}]
        recs = build_skipped_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="run1", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", available_cash=1_000_000.0,
        )
        assert len(recs) == 1
        assert recs[0].rejection_reason == REJECTION_POSITION_FULL
        assert recs[0].run_id == "run1"

    def test_executed_symbol_produces_no_skipped_record(self):
        signals = [_sig("7203.T")]
        send_results = [{"symbol": "7203.T", "side": "BUY", "success": True}]
        recs = build_skipped_opportunity_records(
            signals=signals, stage_audit=[], send_results=send_results,
            run_id="run1", run_timestamp="2026-06-23T08:41:18+0900", mode="LIVE",
            source_script="run_live_signal.py", available_cash=1_000_000.0,
        )
        assert recs == []

    def test_capital_skip_maps_to_high_price(self):
        signals = [_sig("8035.T")]
        stage_audit = [{"symbol": "8035.T", "stage": "CAPITAL", "passed": False}]
        recs = build_skipped_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="run1", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", available_cash=1_000_000.0,
        )
        assert recs[0].rejection_reason == REJECTION_HIGH_PRICE


class TestRunIdConsistencyAcrossStores:
    """Phase3: Stage Audit・EVS・Opportunity Captureが同一run_idで紐付く。"""

    def test_evs_and_skipped_share_run_id_from_same_inputs(self):
        signals = [_sig("8035.T", rank=2), _sig("6920.T", rank=3)]
        stage_audit = [
            {"symbol": "8035.T", "stage": "CAPACITY", "passed": False, "reason": "position_full"},
            {"symbol": "6920.T", "stage": "CAPACITY", "passed": False, "reason": "position_full"},
        ]
        common_kwargs = dict(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="shared_run_20260629_084416", run_timestamp="2026-06-29T08:44:16+0900",
            mode="LIVE", source_script="run_live_signal.py",
        )
        evs_recs = build_opportunity_records(
            **common_kwargs, capital_available_pct=0.5, portfolio_heat=0.0,
            market_regime="unknown", max_positions=3,
        )
        sor_recs = build_skipped_opportunity_records(**common_kwargs, available_cash=1_000_000.0)

        assert len(evs_recs) == 2
        assert len(sor_recs) == 2
        assert {r.run_id for r in evs_recs} == {"shared_run_20260629_084416"}
        assert {r.run_id for r in sor_recs} == {"shared_run_20260629_084416"}
        assert {r.symbol for r in evs_recs} == {r.symbol for r in sor_recs}


class TestMultipleRunsSameDay:
    def test_different_runs_produce_distinguishable_records(self):
        signals = [_sig("2802.T", rsr=78.6, rank=14)]
        stage_audit_skip = [{"symbol": "2802.T", "stage": "CAPACITY", "passed": False}]
        run_a = build_skipped_opportunity_records(
            signals=signals, stage_audit=stage_audit_skip, send_results=[],
            run_id="20260623_054137", run_timestamp="2026-06-23T05:41:37+0900",
            mode="LIVE", source_script="run_live_signal.py", available_cash=1_000_000.0,
        )
        signals_bought = [_sig("2802.T", rsr=82.5, rank=9)]
        run_b = build_skipped_opportunity_records(
            signals=signals_bought, stage_audit=[], send_results=[
                {"symbol": "2802.T", "side": "BUY", "success": True},
            ],
            run_id="20260623_084118", run_timestamp="2026-06-23T08:41:18+0900",
            mode="LIVE", source_script="run_live_signal.py", available_cash=1_000_000.0,
        )
        assert len(run_a) == 1  # skipped in the early run
        assert len(run_b) == 0  # actually bought in the later run
        assert run_a[0].record_id != ""


class TestEnrichmentPipeline:
    def test_enrich_and_rewrite_store_fills_forward_returns(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "skipped_opportunities.jsonl"
            signals = [_sig("8035.T", rank=2)]
            stage_audit = [{"symbol": "8035.T", "stage": "CAPACITY", "passed": False}]
            recs = build_skipped_opportunity_records(
                signals=signals, stage_audit=stage_audit, send_results=[],
                run_id="run1", run_timestamp="2026-06-01T08:44:00+0900", mode="LIVE",
                source_script="run_live_signal.py", available_cash=1_000_000.0,
            )
            for r in recs:
                append_skipped_opportunity(r, path)

            loaded_before = load_skipped_opportunities(path)
            assert loaded_before[0].enrichment_status == ENRICHMENT_PENDING

            def _price_fetcher(sym, iso_date):
                return [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

            n = enrich_and_rewrite_store(path, _price_fetcher, now_ts="2026-06-10T00:00:00+00:00")
            assert n == 1

            loaded_after = load_skipped_opportunities(path)
            assert loaded_after[0].enrichment_status == ENRICHMENT_ENRICHED
            assert loaded_after[0].forward_return_5d == pytest.approx(0.05, rel=1e-3)

    def test_enrich_and_rewrite_store_empty_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nonexistent.jsonl"
            n = enrich_and_rewrite_store(path, lambda s, d: None)
            assert n == 0


class TestAnalyzeOpportunityCaptureUsesEnrichedData:
    def test_missed_alpha_computed_from_enriched_skipped(self):
        skipped = [{
            "rejection_reason": "position_full",
            "enrichment_status": "enriched",
            "forward_return_5d": 0.03,
        }]
        result = analyze_opportunity_capture(
            metrics=[{"date": "2026-06-29", "blocked_by_rsr": 0, "blocked_by_breakout": 0,
                      "candidate_count": 1, "raw_buy_count": 3}],
            skipped=skipped,
            week_start=date(2026, 6, 23), week_end=date(2026, 6, 29),
        )
        assert result.n_enriched_skipped == 1
        assert result.mean_forward_return_5d == pytest.approx(0.03)
        assert result.missed_alpha_score is not None
        assert not any("missing" in m for m in result.insufficiency_markers)

    def test_capture_ratio_uses_correct_metric_field_names(self):
        result = analyze_opportunity_capture(
            metrics=[{"date": "2026-06-29", "blocked_by_rsr": 5, "blocked_by_breakout": 2,
                      "candidate_count": 1, "raw_buy_count": 8}],
            skipped=[],
            week_start=date(2026, 6, 23), week_end=date(2026, 6, 29),
        )
        assert result.opportunity_capture_ratio == pytest.approx(1 / 8)
        assert result.false_negative_rate == pytest.approx(7 / 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
