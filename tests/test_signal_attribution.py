"""
Tests for src/analytics/signal_attribution.py — Signal Attribution Intelligence
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import pytest

from src.analytics.signal_attribution import (
    ALL_SIGNALS,
    CP_NUMERIC_SIGNALS,
    ENRICHED_SIGNALS,
    MIN_SAMPLE_SIZE,
    SCHEMA_VERSION,
    TARGET_5D,
    TOP_N_SIGNALS,
    InteractionBucket,
    SignalAttributionOutput,
    SignalAttributionResult,
    _extract_pairs,
    _safe,
    append_attribution_record,
    build_sp_index,
    compute_interaction,
    compute_signal_ir,
    enrich_with_sp,
    format_attribution_report,
    load_and_filter_cp,
    rank_all_signals,
    run_signal_attribution,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _cp_rec(
    symbol: str = "1234.T",
    date: str = "2026-05-01",
    priority_score: float = 70.0,
    compression_score: float = 60.0,
    breakout_quality_score: float = 75.0,
    rsr: float = 80.0,
    rsr_momentum: float = 1.5,
    mfe_pct: float = 0.08,
    giveback_ratio: float = 0.10,
    hold_days: int = 12,
    subsequent_5d_return: Optional[float] = 0.05,
    materialized_5d: bool = True,
) -> dict:
    return {
        "date": date,
        "symbol": symbol,
        "priority_score": priority_score,
        "compression_score": compression_score,
        "breakout_quality_score": breakout_quality_score,
        "rsr": rsr,
        "rsr_momentum": rsr_momentum,
        "mfe_pct": mfe_pct,
        "giveback_ratio": giveback_ratio,
        "hold_days": hold_days,
        TARGET_5D: subsequent_5d_return,
        "materialized_5d": materialized_5d,
        "schema_version": "v1",
    }


def _make_cp_file(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "cp.jsonl"
    _write_jsonl(p, records)
    return p


def _make_sp_file(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "sp.jsonl"
    _write_jsonl(p, records)
    return p


def _make_out(tmp_path: Path) -> Path:
    return tmp_path / "sa" / "signal_attribution.jsonl"


def _diverse_records(n: int = 12) -> List[dict]:
    """n materialized CP records with monotone-correlated signals and 5d returns."""
    records = []
    for i in range(n):
        score = 15.0 + i * 7.0
        ret   = -0.06 + i * 0.012
        records.append(_cp_rec(
            symbol=f"{1000+i}.T",
            date=f"2026-04-{(i % 28) + 1:02d}",
            priority_score=score,
            compression_score=30.0 + i * 5.0,
            breakout_quality_score=40.0 + i * 4.0,
            rsr=65.0 + i * 2.5,
            rsr_momentum=-1.0 + i * 0.2,
            mfe_pct=0.01 + i * 0.007,
            giveback_ratio=0.30 - i * 0.02,
            hold_days=3 + i * 2,
            subsequent_5d_return=round(ret, 4),
        ))
    return records


# ── _safe ─────────────────────────────────────────────────────────────────────

class TestSafe:
    def test_normal(self):
        assert _safe(1.5) == 1.5

    def test_none_default(self):
        assert _safe(None) == 0.0

    def test_none_custom_fallback(self):
        assert _safe(None, fallback=None) is None

    def test_inf(self):
        assert _safe(float("inf")) == 0.0

    def test_nan(self):
        assert _safe(float("nan")) == 0.0

    def test_bad_type(self):
        assert _safe("abc", fallback=-1.0) == -1.0


# ── load_and_filter_cp ────────────────────────────────────────────────────────

class TestLoadAndFilterCp:
    def test_within_and_materialized(self, tmp_path):
        recs = [
            _cp_rec(date="2026-05-01", subsequent_5d_return=0.05),
            _cp_rec(date="2026-05-02", subsequent_5d_return=None),
            _cp_rec(date="2020-01-01", subsequent_5d_return=0.03),  # old
        ]
        cp = _make_cp_file(tmp_path, recs)
        within, mat = load_and_filter_cp(cp, lookback_days=180, today="2026-05-30")
        assert len(within) == 2
        assert len(mat) == 1

    def test_lookback_cutoff(self, tmp_path):
        recs = [
            _cp_rec(date="2026-05-20", subsequent_5d_return=0.02),
            _cp_rec(date="2026-01-01", subsequent_5d_return=0.05),
        ]
        cp = _make_cp_file(tmp_path, recs)
        _, mat = load_and_filter_cp(cp, lookback_days=30, today="2026-05-30")
        assert len(mat) == 1

    def test_missing_file(self, tmp_path):
        w, m = load_and_filter_cp(tmp_path / "missing.jsonl", 90, "2026-05-30")
        assert w == [] and m == []

    def test_invalid_date_skipped(self, tmp_path):
        recs = [{"date": "bad", "symbol": "X", TARGET_5D: 0.01}]
        cp = _make_cp_file(tmp_path, recs)
        w, m = load_and_filter_cp(cp, 90, "2026-05-30")
        assert w == [] and m == []


# ── build_sp_index ────────────────────────────────────────────────────────────

class TestBuildSpIndex:
    def test_builds_index(self, tmp_path):
        recs = [
            {"date": "2026-05-01", "pressure_score": 45.0},
            {"date": "2026-05-02", "pressure_score": 72.0},
        ]
        sp = _make_sp_file(tmp_path, recs)
        idx = build_sp_index(sp)
        assert idx["2026-05-01"] == 45.0
        assert idx["2026-05-02"] == 72.0

    def test_missing_file(self, tmp_path):
        assert build_sp_index(tmp_path / "no.jsonl") == {}

    def test_skips_missing_date_or_score(self, tmp_path):
        recs = [
            {"date": "", "pressure_score": 50.0},
            {"date": "2026-05-01", "pressure_score": None},
        ]
        sp = _make_sp_file(tmp_path, recs)
        idx = build_sp_index(sp)
        assert idx == {}

    def test_later_record_overwrites(self, tmp_path):
        recs = [
            {"date": "2026-05-01", "pressure_score": 30.0},
            {"date": "2026-05-01", "pressure_score": 80.0},
        ]
        sp = _make_sp_file(tmp_path, recs)
        idx = build_sp_index(sp)
        assert idx["2026-05-01"] == 80.0


# ── enrich_with_sp ────────────────────────────────────────────────────────────

class TestEnrichWithSp:
    def test_adds_slot_pressure_score(self):
        records = [_cp_rec(date="2026-05-01")]
        idx = {"2026-05-01": 55.0}
        enriched = enrich_with_sp(records, idx)
        assert enriched[0]["slot_pressure_score"] == 55.0

    def test_no_match_leaves_record_without_field(self):
        records = [_cp_rec(date="2026-05-01")]
        enriched = enrich_with_sp(records, {})
        assert "slot_pressure_score" not in enriched[0]

    def test_does_not_mutate_original(self):
        rec = _cp_rec(date="2026-05-01")
        original_keys = set(rec.keys())
        enrich_with_sp([rec], {"2026-05-01": 60.0})
        assert set(rec.keys()) == original_keys

    def test_mixed_match(self):
        records = [
            _cp_rec(date="2026-05-01"),
            _cp_rec(date="2026-05-02"),
        ]
        idx = {"2026-05-01": 40.0}
        enriched = enrich_with_sp(records, idx)
        assert enriched[0]["slot_pressure_score"] == 40.0
        assert "slot_pressure_score" not in enriched[1]


# ── _extract_pairs ────────────────────────────────────────────────────────────

class TestExtractPairs:
    def test_basic_extraction(self):
        records = [
            _cp_rec(priority_score=70.0, subsequent_5d_return=0.05),
            _cp_rec(priority_score=40.0, subsequent_5d_return=-0.02),
        ]
        pairs = _extract_pairs(records, "priority_score")
        assert len(pairs) == 2
        assert (70.0, 0.05) in pairs

    def test_skips_none_signal(self):
        records = [
            {TARGET_5D: 0.05},  # missing priority_score
            _cp_rec(priority_score=70.0, subsequent_5d_return=0.05),
        ]
        pairs = _extract_pairs(records, "priority_score")
        assert len(pairs) == 1

    def test_skips_none_return(self):
        records = [_cp_rec(priority_score=70.0, subsequent_5d_return=None)]
        assert _extract_pairs(records, "priority_score") == []

    def test_skips_inf_values(self):
        records = [{"priority_score": float("inf"), TARGET_5D: 0.05}]
        assert _extract_pairs(records, "priority_score") == []


# ── compute_signal_ir ─────────────────────────────────────────────────────────

class TestComputeSignalIR:
    def test_positive_ir_for_correlated_signal(self):
        records = _diverse_records(12)
        r = compute_signal_ir(records, "priority_score")
        assert r is not None
        assert r.signal_ir > 0

    def test_negative_ir_for_anti_correlated(self):
        # giveback_ratio decreases as priority_score increases in _diverse_records
        records = _diverse_records(12)
        r = compute_signal_ir(records, "giveback_ratio")
        assert r is not None
        assert r.signal_ir < 0

    def test_none_when_below_min_samples(self):
        records = _diverse_records(2)
        assert compute_signal_ir(records, "priority_score", min_samples=5) is None

    def test_returns_result_fields(self):
        records = _diverse_records(10)
        r = compute_signal_ir(records, "rsr")
        assert r is not None
        assert r.signal_name == "rsr"
        assert 0 < r.sample_size <= 10
        assert 0.0 <= r.top_half_win_rate <= 1.0
        assert 0.0 <= r.bottom_half_win_rate <= 1.0

    def test_median_split_is_finite(self):
        records = _diverse_records(10)
        r = compute_signal_ir(records, "priority_score")
        assert r is not None
        assert math.isfinite(r.median_split)

    def test_ir_sign_matches_direction(self):
        records = _diverse_records(12)
        r = compute_signal_ir(records, "priority_score")
        assert r is not None
        # top_half (high score) should have higher mean return
        assert r.top_half_mean > r.bottom_half_mean
        assert r.signal_ir == pytest.approx(r.top_half_mean - r.bottom_half_mean, abs=1e-9)

    def test_missing_signal_field_returns_none_for_insufficient_data(self, tmp_path):
        # Records missing the signal → pairs = [] → None
        records = [{TARGET_5D: 0.05}] * 3  # no "slot_pressure_score" field
        assert compute_signal_ir(records, "slot_pressure_score", min_samples=2) is None

    def test_all_returns_same_gives_zero_ir(self):
        records = [_cp_rec(priority_score=float(i), subsequent_5d_return=0.05)
                   for i in range(10)]
        r = compute_signal_ir(records, "priority_score")
        assert r is not None
        assert r.signal_ir == pytest.approx(0.0, abs=1e-9)


# ── rank_all_signals ──────────────────────────────────────────────────────────

class TestRankAllSignals:
    def test_returns_list_of_results(self):
        records = _diverse_records(12)
        ranked = rank_all_signals(records)
        assert len(ranked) > 0
        assert all(isinstance(r, SignalAttributionResult) for r in ranked)

    def test_sorted_by_abs_ir_descending(self):
        records = _diverse_records(12)
        ranked = rank_all_signals(records)
        irs = [abs(r.signal_ir) for r in ranked]
        assert irs == sorted(irs, reverse=True)

    def test_signal_names_are_known(self):
        records = _diverse_records(12)
        ranked = rank_all_signals(records)
        for r in ranked:
            assert r.signal_name in ALL_SIGNALS

    def test_below_min_samples_excluded(self):
        records = _diverse_records(2)
        ranked = rank_all_signals(records, min_samples=5)
        assert ranked == []

    def test_slot_pressure_included_when_enriched(self):
        records = [
            dict(_cp_rec(date=f"2026-04-{i+1:02d}",
                         priority_score=float(i*8),
                         subsequent_5d_return=i*0.01),
                 slot_pressure_score=float(i*5))
            for i in range(10)
        ]
        ranked = rank_all_signals(records, min_samples=5)
        names = [r.signal_name for r in ranked]
        assert "slot_pressure_score" in names


# ── compute_interaction ───────────────────────────────────────────────────────

class TestComputeInteraction:
    def test_returns_4_buckets(self):
        records = _diverse_records(12)
        buckets = compute_interaction(records, "priority_score", "compression_score")
        assert len(buckets) == 4
        cats = {b.category for b in buckets}
        assert cats == {"HH", "HL", "LH", "LL"}

    def test_hh_has_highest_return_for_correlated_signals(self):
        """Both signals correlated with return → HH should have highest E[5d]."""
        records = _diverse_records(12)
        buckets = compute_interaction(records, "priority_score", "compression_score")
        hh = next(b for b in buckets if b.category == "HH")
        ll = next(b for b in buckets if b.category == "LL")
        # With positive IR for both signals, HH > LL
        if hh.n > 0 and ll.n > 0:
            assert hh.mean_5d_return > ll.mean_5d_return

    def test_bucket_n_sums_to_total_pairs(self):
        records = _diverse_records(12)
        buckets = compute_interaction(records, "priority_score", "compression_score")
        total = sum(b.n for b in buckets)
        # All records have both signals and 5d return → total = 12
        assert total == 12

    def test_win_rate_in_range(self):
        records = _diverse_records(12)
        for b in compute_interaction(records, "priority_score", "rsr"):
            assert 0.0 <= b.win_rate <= 1.0

    def test_empty_when_below_min_samples(self):
        records = _diverse_records(2)
        assert compute_interaction(records, "priority_score", "compression_score",
                                   min_samples=5) == []

    def test_signal_names_preserved_in_buckets(self):
        records = _diverse_records(10)
        buckets = compute_interaction(records, "priority_score", "rsr_momentum")
        for b in buckets:
            assert b.signal_a == "priority_score"
            assert b.signal_b == "rsr_momentum"

    def test_missing_signal_returns_empty(self):
        records = _diverse_records(10)
        # slot_pressure_score not in records → pairs = [] → empty
        result = compute_interaction(records, "priority_score", "slot_pressure_score",
                                     min_samples=5)
        assert result == []


# ── run_signal_attribution ────────────────────────────────────────────────────

class TestRunSignalAttribution:
    def test_full_run_produces_output(self, tmp_path):
        records = _diverse_records(12)
        cp  = _make_cp_file(tmp_path, records)
        sp  = _make_sp_file(tmp_path, [])
        out = _make_out(tmp_path)

        result = run_signal_attribution(
            cp_file=cp, sp_file=sp, output_file=out, today="2026-05-30",
        )
        assert isinstance(result, SignalAttributionOutput)
        assert result.n_materialized == 12
        assert result.top_signal in ALL_SIGNALS
        assert out.exists()

    def test_appends_to_jsonl(self, tmp_path):
        records = _diverse_records(10)
        cp  = _make_cp_file(tmp_path, records)
        sp  = _make_sp_file(tmp_path, [])
        out = _make_out(tmp_path)

        run_signal_attribution(cp_file=cp, sp_file=sp, output_file=out, today="2026-05-30")
        run_signal_attribution(cp_file=cp, sp_file=sp, output_file=out, today="2026-05-30")
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_top_signal_has_highest_abs_ir(self, tmp_path):
        records = _diverse_records(12)
        cp  = _make_cp_file(tmp_path, records)
        sp  = _make_sp_file(tmp_path, [])
        out = _make_out(tmp_path)

        result = run_signal_attribution(cp_file=cp, sp_file=sp, output_file=out,
                                        today="2026-05-30")
        top_ir = abs(result.top_signal_ir)
        for r in result.attribution_results:
            assert abs(r.signal_ir) <= top_ir + 1e-9

    def test_interaction_uses_top2_signals(self, tmp_path):
        records = _diverse_records(12)
        cp  = _make_cp_file(tmp_path, records)
        sp  = _make_sp_file(tmp_path, [])
        out = _make_out(tmp_path)

        result = run_signal_attribution(cp_file=cp, sp_file=sp, output_file=out,
                                        today="2026-05-30")
        if result.top_interaction:
            assert "×" in result.top_interaction
            parts = [p.strip() for p in result.top_interaction.split("×")]
            assert len(parts) == 2
            assert parts[0] == result.attribution_results[0].signal_name
            assert parts[1] == result.attribution_results[1].signal_name

    def test_insufficient_data_returns_empty(self, tmp_path):
        cp  = _make_cp_file(tmp_path, [_cp_rec()])
        out = _make_out(tmp_path)
        result = run_signal_attribution(
            cp_file=cp, sp_file=tmp_path/"no.jsonl",
            output_file=out, today="2026-05-30",
        )
        assert result.n_materialized < MIN_SAMPLE_SIZE
        assert result.attribution_results == []

    def test_missing_cp_returns_empty(self, tmp_path):
        out = _make_out(tmp_path)
        result = run_signal_attribution(
            cp_file=tmp_path/"no.jsonl", sp_file=tmp_path/"no.jsonl",
            output_file=out, today="2026-05-30",
        )
        assert result.n_records_total == 0

    def test_sp_enrichment_adds_slot_signal(self, tmp_path):
        records = _diverse_records(12)
        sp_recs = [
            {"date": r["date"], "pressure_score": float(i * 7)}
            for i, r in enumerate(records)
        ]
        cp  = _make_cp_file(tmp_path, records)
        sp  = _make_sp_file(tmp_path, sp_recs)
        out = _make_out(tmp_path)

        result = run_signal_attribution(cp_file=cp, sp_file=sp, output_file=out,
                                        today="2026-05-30")
        names = [r.signal_name for r in result.attribution_results]
        assert "slot_pressure_score" in names

    def test_never_raises(self, tmp_path):
        result = run_signal_attribution(
            cp_file=tmp_path/"x", sp_file=tmp_path/"y",
            output_file=tmp_path/"a"/"b"/"c.jsonl",
            today="2026-05-30",
        )
        assert isinstance(result, SignalAttributionOutput)

    def test_schema_version_in_jsonl(self, tmp_path):
        records = _diverse_records(10)
        cp  = _make_cp_file(tmp_path, records)
        out = _make_out(tmp_path)
        run_signal_attribution(cp_file=cp, sp_file=tmp_path/"no.jsonl",
                               output_file=out, today="2026-05-30")
        rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert rec["schema_version"] == SCHEMA_VERSION

    def test_lookback_filters_old_records(self, tmp_path):
        records = [
            _cp_rec(date="2026-05-01", subsequent_5d_return=0.05),
            _cp_rec(date="2020-01-01", subsequent_5d_return=0.08),
        ]
        cp  = _make_cp_file(tmp_path, records)
        out = _make_out(tmp_path)
        result = run_signal_attribution(
            cp_file=cp, sp_file=tmp_path/"no.jsonl",
            output_file=out, today="2026-05-30",
            lookback_days=90,
        )
        assert result.n_records_total == 1


# ── append_attribution_record ─────────────────────────────────────────────────

class TestAppendAttributionRecord:
    def _empty_output(self) -> SignalAttributionOutput:
        return SignalAttributionOutput(
            date="2026-05-30",
            n_records_total=0,
            n_materialized=0,
            lookback_days=90,
            min_sample_size=5,
            attribution_results=[],
            top_signal="",
            top_signal_ir=0.0,
            top_interaction="",
            interaction_ev=0.0,
            interaction_table=[],
        )

    def test_creates_file(self, tmp_path):
        out = tmp_path / "sa.jsonl"
        append_attribution_record(self._empty_output(), out)
        assert out.exists()
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_appends_multiple(self, tmp_path):
        out = tmp_path / "sa.jsonl"
        append_attribution_record(self._empty_output(), out)
        append_attribution_record(self._empty_output(), out)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_fail_open_on_bad_path(self):
        append_attribution_record(
            self._empty_output(),
            Path("/invalid/path/that/cannot/exist/sa.jsonl"),
        )


# ── format_attribution_report ─────────────────────────────────────────────────

class TestFormatAttributionReport:
    def _full_output(self) -> SignalAttributionOutput:
        return SignalAttributionOutput(
            date="2026-05-30",
            n_records_total=18,
            n_materialized=15,
            lookback_days=90,
            min_sample_size=5,
            attribution_results=[
                SignalAttributionResult("priority_score", 0.052, 15, 0.081, 0.029, 0.75, 0.40, 65.0),
                SignalAttributionResult("compression_score", 0.048, 15, 0.073, 0.025, 0.70, 0.42, 58.0),
                SignalAttributionResult("rsr_momentum", 0.027, 15, 0.063, 0.036, 0.67, 0.50, 0.5),
            ],
            top_signal="priority_score",
            top_signal_ir=0.052,
            top_interaction="priority_score × compression_score",
            interaction_ev=0.081,
            interaction_table=[
                InteractionBucket("priority_score", "compression_score", "HH", 4, 0.092, 0.80),
                InteractionBucket("priority_score", "compression_score", "HL", 4, 0.065, 0.65),
                InteractionBucket("priority_score", "compression_score", "LH", 4, 0.043, 0.55),
                InteractionBucket("priority_score", "compression_score", "LL", 3, 0.010, 0.33),
            ],
        )

    def test_returns_string(self):
        rpt = format_attribution_report(self._full_output())
        assert isinstance(rpt, str) and len(rpt) > 0

    def test_contains_section_header(self):
        rpt = format_attribution_report(self._full_output())
        assert "SIGNAL ATTRIBUTION" in rpt

    def test_contains_top_signals(self):
        rpt = format_attribution_report(self._full_output())
        assert "Top Signals" in rpt
        assert "priority_score" in rpt

    def test_contains_interaction_section(self):
        rpt = format_attribution_report(self._full_output())
        assert "Top Interaction" in rpt
        assert "priority_score × compression_score" in rpt

    def test_contains_hh_ll_labels(self):
        rpt = format_attribution_report(self._full_output())
        assert "HH" in rpt or "HIGH" in rpt

    def test_contains_kpi_footer(self):
        rpt = format_attribution_report(self._full_output())
        assert "最重要KPI" in rpt

    def test_insufficient_data_shows_warning(self):
        output = SignalAttributionOutput(
            date="2026-05-30", n_records_total=2, n_materialized=2,
            lookback_days=90, min_sample_size=5,
            attribution_results=[], top_signal="", top_signal_ir=0.0,
            top_interaction="", interaction_ev=0.0, interaction_table=[],
        )
        rpt = format_attribution_report(output)
        assert "データ不足" in rpt

    def test_never_raises(self):
        output = SignalAttributionOutput(
            date="X", n_records_total=0, n_materialized=0, lookback_days=90,
            min_sample_size=5, attribution_results=[], top_signal="",
            top_signal_ir=0.0, top_interaction="", interaction_ev=0.0,
            interaction_table=[],
        )
        try:
            format_attribution_report(output)
        except Exception as exc:
            pytest.fail(f"format_attribution_report raised: {exc}")

    def test_top_n_signals_shown(self):
        results = [
            SignalAttributionResult(f"sig_{i}", float(i), 10, 0.05, 0.01, 0.6, 0.4, 50.0)
            for i in range(8)
        ]
        results.sort(key=lambda r: abs(r.signal_ir), reverse=True)
        output = SignalAttributionOutput(
            date="2026-05-30", n_records_total=10, n_materialized=10,
            lookback_days=90, min_sample_size=5,
            attribution_results=results,
            top_signal="sig_7", top_signal_ir=7.0,
            top_interaction="", interaction_ev=0.0, interaction_table=[],
        )
        rpt = format_attribution_report(output)
        # Should show at most TOP_N_SIGNALS entries
        shown_count = sum(1 for r in results[:TOP_N_SIGNALS] if r.signal_name in rpt)
        assert shown_count >= 1


# ── Integration: stability check ──────────────────────────────────────────────

class TestStabilityCheck:
    def test_min_sample_size_excludes_signals_with_few_records(self, tmp_path):
        # Only 4 records — below default MIN_SAMPLE_SIZE=5
        records = _diverse_records(4)
        cp  = _make_cp_file(tmp_path, records)
        out = _make_out(tmp_path)
        result = run_signal_attribution(
            cp_file=cp, sp_file=tmp_path/"no.jsonl",
            output_file=out, today="2026-05-30",
            min_samples=5,
        )
        assert result.attribution_results == []

    def test_exactly_min_sample_size_is_included(self):
        records = _diverse_records(5)
        r = compute_signal_ir(records, "priority_score", min_samples=5)
        assert r is not None
        assert r.sample_size == 5

    def test_one_below_min_excluded(self):
        records = _diverse_records(4)
        r = compute_signal_ir(records, "priority_score", min_samples=5)
        assert r is None
