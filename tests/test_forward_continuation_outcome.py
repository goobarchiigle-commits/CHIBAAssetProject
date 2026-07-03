"""
Tests for src/analytics/forward_continuation_outcome.py
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest

from src.analytics.forward_continuation_outcome import (
    MIN_RECORDS,
    NUMERIC_SIGNALS,
    PRIORITY_BINS,
    SCHEMA_VERSION,
    TARGET_5D,
    BucketEV,
    CategoryEV,
    CombinationEV,
    FcoResult,
    SignalRank,
    _build_bq_index,
    _load_jsonl,
    _overall_stats,
    _safe_float,
    append_fco_record,
    compute_bucket_ev,
    compute_category_ev,
    compute_combination_ev,
    enrich_with_bq,
    filter_materialized,
    format_fco_report,
    rank_signals,
    run_fco_analysis,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _cp_record(
    symbol: str = "1234.T",
    date: str = "2026-05-01",
    priority_score: float = 70.0,
    priority_tier: str = "high_quality",
    compression_score: float = 60.0,
    breakout_quality_score: float = 75.0,
    rsr: float = 80.0,
    rsr_momentum: float = 1.5,
    mfe_pct: float = 0.08,
    hold_days: int = 12,
    current_phase: str = "healthy_breakout",
    suppression_eligible: bool = False,
    subsequent_5d_return: Optional[float] = 0.05,
    materialized_5d: bool = True,
) -> dict:
    return {
        "date": date,
        "symbol": symbol,
        "priority_score": priority_score,
        "priority_tier": priority_tier,
        "compression_score": compression_score,
        "breakout_quality_score": breakout_quality_score,
        "rsr": rsr,
        "rsr_momentum": rsr_momentum,
        "mfe_pct": mfe_pct,
        "hold_days": hold_days,
        "current_phase": current_phase,
        "suppression_eligible": suppression_eligible,
        TARGET_5D: subsequent_5d_return,
        "materialized_5d": materialized_5d,
        "schema_version": "v1",
    }


def _make_cp_file(tmp_path: Path, records: List[dict]) -> Path:
    p = tmp_path / "cp_telemetry.jsonl"
    _write_jsonl(p, records)
    return p


def _make_bq_file(tmp_path: Path, records: List[dict]) -> Path:
    p = tmp_path / "bq_telemetry.jsonl"
    _write_jsonl(p, records)
    return p


def _make_fco_out(tmp_path: Path) -> Path:
    return tmp_path / "fco_aggregated.jsonl"


def _make_diverse_records(n: int = 12) -> List[dict]:
    """Generate n diverse materialized CP records with varying signals and 5d returns."""
    import random
    random.seed(42)
    records = []
    phases = ["healthy_breakout", "compression", "breakout", "distribution"]
    tiers = ["core_continuation", "high_quality", "neutral", "exit_candidate"]
    for i in range(n):
        score = 20.0 + i * 6.0  # 20 to 86
        ret = -0.05 + i * 0.012  # -0.05 to +0.082
        records.append(_cp_record(
            symbol=f"{1000 + i}.T",
            date=f"2026-04-{(i % 28) + 1:02d}",
            priority_score=score,
            priority_tier=tiers[i % 4],
            compression_score=30.0 + i * 5.0,
            breakout_quality_score=40.0 + i * 4.0,
            rsr=65.0 + i * 2.5,
            rsr_momentum=-1.0 + i * 0.2,
            mfe_pct=0.01 + i * 0.008,
            hold_days=3 + i * 2,
            current_phase=phases[i % 4],
            subsequent_5d_return=round(ret, 4),
            materialized_5d=True,
        ))
    return records


# ── _safe_float ───────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(1.5) == 1.5

    def test_int(self):
        assert _safe_float(10) == 10.0

    def test_string_float(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_none_default(self):
        assert _safe_float(None) == 0.0

    def test_none_custom_fallback(self):
        assert _safe_float(None, fallback=None) is None

    def test_inf_returns_fallback(self):
        assert _safe_float(float("inf")) == 0.0

    def test_nan_returns_fallback(self):
        assert _safe_float(float("nan")) == 0.0

    def test_bad_string_returns_fallback(self):
        assert _safe_float("abc", fallback=99.0) == 99.0


# ── _load_jsonl ───────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_loads_valid_records(self, tmp_path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [{"a": 1}, {"b": 2}])
        recs = _load_jsonl(p)
        assert len(recs) == 2
        assert recs[0]["a"] == 1

    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_jsonl(tmp_path / "missing.jsonl") == []

    def test_skips_bad_lines(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")
        recs = _load_jsonl(p)
        assert len(recs) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert _load_jsonl(p) == []


# ── _build_bq_index ───────────────────────────────────────────────────────────

class TestBqIndex:
    def test_indexes_by_symbol_date(self, tmp_path):
        recs = [
            {"symbol": "1234.T", "date": "2026-04-01", "quality_score": 80.0},
            {"symbol": "5678.T", "date": "2026-04-02", "quality_score": 60.0},
        ]
        p = _make_bq_file(tmp_path, recs)
        idx = _build_bq_index(p)
        assert ("1234.T", "2026-04-01") in idx
        assert idx[("1234.T", "2026-04-01")]["quality_score"] == 80.0

    def test_missing_file(self, tmp_path):
        assert _build_bq_index(tmp_path / "no.jsonl") == {}

    def test_skips_records_without_symbol_or_date(self, tmp_path):
        recs = [{"symbol": "", "date": "2026-04-01"}, {"symbol": "X", "date": ""}]
        p = _make_bq_file(tmp_path, recs)
        assert _build_bq_index(p) == {}


# ── filter_materialized ───────────────────────────────────────────────────────

class TestFilterMaterialized:
    def test_within_and_materialized_split(self):
        recs = [
            _cp_record(date="2026-05-01", subsequent_5d_return=0.05, materialized_5d=True),
            _cp_record(date="2026-05-02", subsequent_5d_return=None, materialized_5d=False),
            _cp_record(date="2020-01-01", subsequent_5d_return=0.03, materialized_5d=True),
        ]
        within, mat = filter_materialized(recs, lookback_days=180, today="2026-05-30")
        assert len(within) == 2           # old record excluded
        assert len(mat) == 1             # only materialized in window

    def test_lookback_cutoff(self):
        recs = [
            _cp_record(date="2026-05-20", subsequent_5d_return=0.02),
            _cp_record(date="2026-01-01", subsequent_5d_return=0.05),
        ]
        _, mat = filter_materialized(recs, lookback_days=30, today="2026-05-30")
        assert len(mat) == 1             # only recent record

    def test_none_return_excluded_from_materialized(self):
        recs = [_cp_record(date="2026-05-01", subsequent_5d_return=None)]
        _, mat = filter_materialized(recs, lookback_days=90, today="2026-05-30")
        assert mat == []

    def test_invalid_date_skipped(self):
        recs = [{"date": "not-a-date", "symbol": "X", TARGET_5D: 0.01, "materialized_5d": True}]
        within, mat = filter_materialized(recs, lookback_days=90, today="2026-05-30")
        assert within == []
        assert mat == []


# ── enrich_with_bq ────────────────────────────────────────────────────────────

class TestEnrichWithBq:
    def test_merges_bq_fields(self):
        records = [_cp_record(symbol="1234.T", date="2026-05-01")]
        bq_index = {("1234.T", "2026-05-01"): {"cpr_score": 85.0, "breakout_phase_type": "healthy_breakout"}}
        enriched = enrich_with_bq(records, bq_index)
        assert enriched[0]["cpr_score"] == 85.0
        assert enriched[0]["breakout_phase_type"] == "healthy_breakout"

    def test_no_bq_match_leaves_record_unchanged(self):
        records = [_cp_record(symbol="1234.T", date="2026-05-01")]
        enriched = enrich_with_bq(records, {})
        assert "cpr_score" not in enriched[0]

    def test_does_not_overwrite_existing_fields(self):
        records = [_cp_record(symbol="X", date="2026-05-01", breakout_quality_score=70.0)]
        bq_index = {("X", "2026-05-01"): {"breakout_quality_score": 99.0}}
        enriched = enrich_with_bq(records, bq_index)
        assert enriched[0]["breakout_quality_score"] == 70.0  # not overwritten

    def test_fills_missing_bq_score(self):
        records = [_cp_record(symbol="X", date="2026-05-01")]
        del records[0]["breakout_quality_score"]
        bq_index = {("X", "2026-05-01"): {"breakout_quality_score": 88.0}}
        enriched = enrich_with_bq(records, bq_index)
        assert enriched[0]["breakout_quality_score"] == 88.0


# ── compute_bucket_ev ─────────────────────────────────────────────────────────

class TestComputeBucketEV:
    def test_correct_bucket_assignment(self):
        records = [
            _cp_record(priority_score=25.0, subsequent_5d_return=0.10),  # bucket 0-30
            _cp_record(priority_score=60.0, subsequent_5d_return=0.04),  # bucket 45-65
            _cp_record(priority_score=85.0, subsequent_5d_return=0.08),  # bucket 80-100
        ]
        buckets = compute_bucket_ev(records)
        # Find bucket 0-30
        b0 = next(b for b in buckets if b.lo == 0.0)
        assert b0.n == 1
        assert b0.mean_5d_return == pytest.approx(0.10)
        assert b0.win_rate == 1.0

    def test_empty_bucket_has_zero_mean(self):
        records = [_cp_record(priority_score=85.0, subsequent_5d_return=0.05)]
        buckets = compute_bucket_ev(records)
        b0 = next(b for b in buckets if b.lo == 0.0)
        assert b0.n == 0
        assert b0.mean_5d_return == 0.0

    def test_win_rate_calculation(self):
        records = [
            _cp_record(priority_score=70.0, subsequent_5d_return=0.05),
            _cp_record(priority_score=72.0, subsequent_5d_return=-0.02),
        ]
        buckets = compute_bucket_ev(records)
        b = next(b for b in buckets if b.lo == 65.0)
        assert b.n == 2
        assert b.win_rate == pytest.approx(0.5)

    def test_none_returns_skipped(self):
        records = [
            _cp_record(priority_score=70.0, subsequent_5d_return=None),
            _cp_record(priority_score=70.0, subsequent_5d_return=0.05),
        ]
        buckets = compute_bucket_ev(records)
        b = next(b for b in buckets if b.lo == 65.0)
        assert b.n == 1

    def test_empty_records_returns_zero_buckets(self):
        buckets = compute_bucket_ev([])
        assert all(b.n == 0 for b in buckets)

    def test_custom_bins(self):
        records = [_cp_record(priority_score=50.0, subsequent_5d_return=0.02)]
        buckets = compute_bucket_ev(records, bins=[0.0, 40.0, 60.0, 101.0])
        assert len(buckets) == 3
        b_mid = next(b for b in buckets if b.lo == 40.0)
        assert b_mid.n == 1


# ── compute_category_ev ───────────────────────────────────────────────────────

class TestComputeCategoryEV:
    def test_groups_by_category(self):
        records = [
            _cp_record(priority_tier="core_continuation", subsequent_5d_return=0.10),
            _cp_record(priority_tier="core_continuation", subsequent_5d_return=0.06),
            _cp_record(priority_tier="exit_candidate",    subsequent_5d_return=-0.03),
        ]
        cats = compute_category_ev(records, "priority_tier")
        core = next(c for c in cats if c.category == "core_continuation")
        assert core.n == 2
        assert core.mean_5d_return == pytest.approx(0.08)
        assert core.win_rate == 1.0

    def test_sorted_by_mean_return(self):
        records = [
            _cp_record(current_phase="phase_a", subsequent_5d_return=0.01),
            _cp_record(current_phase="phase_b", subsequent_5d_return=0.09),
        ]
        cats = compute_category_ev(records, "current_phase")
        assert cats[0].category == "phase_b"

    def test_empty_records(self):
        assert compute_category_ev([], "priority_tier") == []

    def test_skips_none_return(self):
        records = [
            _cp_record(priority_tier="high_quality", subsequent_5d_return=None),
            _cp_record(priority_tier="high_quality", subsequent_5d_return=0.04),
        ]
        cats = compute_category_ev(records, "priority_tier")
        hq = next(c for c in cats if c.category == "high_quality")
        assert hq.n == 1


# ── rank_signals ──────────────────────────────────────────────────────────────

class TestRankSignals:
    def test_returns_signal_ranks(self):
        records = _make_diverse_records(12)
        ranks = rank_signals(records)
        assert len(ranks) > 0
        assert all(isinstance(r, SignalRank) for r in ranks)

    def test_sorted_by_abs_ir_descending(self):
        records = _make_diverse_records(12)
        ranks = rank_signals(records)
        irs = [abs(r.information_ratio) for r in ranks]
        assert irs == sorted(irs, reverse=True)

    def test_all_numeric_signals_attempted(self):
        records = _make_diverse_records(12)
        ranks = rank_signals(records)
        ranked_names = {r.signal_name for r in ranks}
        assert ranked_names.issubset(set(NUMERIC_SIGNALS))

    def test_high_correlation_signal_ranks_first(self):
        """priority_score perfectly correlates with 5d_return → should rank high."""
        records = []
        for i in range(10):
            score = 20.0 + i * 8.0
            records.append(_cp_record(
                symbol=f"{i}.T",
                date=f"2026-04-{i+1:02d}",
                priority_score=score,
                compression_score=50.0,   # flat → no correlation
                breakout_quality_score=50.0,
                rsr=75.0,
                rsr_momentum=0.0,
                mfe_pct=0.05,
                hold_days=10,
                subsequent_5d_return=round(score / 1000.0, 4),  # perfect linear
            ))
        ranks = rank_signals(records)
        assert ranks[0].signal_name == "priority_score"

    def test_insufficient_records_returns_empty(self):
        records = _make_diverse_records(2)
        ranks = rank_signals(records)
        assert ranks == []

    def test_top_half_higher_than_bottom_when_positive_ir(self):
        records = _make_diverse_records(12)
        ranks = rank_signals(records)
        for r in ranks:
            if r.information_ratio > 0:
                assert r.mean_top_half > r.mean_bottom_half

    def test_win_rates_in_range(self):
        records = _make_diverse_records(12)
        for r in rank_signals(records):
            assert 0.0 <= r.win_rate_top <= 1.0
            assert 0.0 <= r.win_rate_bottom <= 1.0


# ── compute_combination_ev ────────────────────────────────────────────────────

class TestComputeCombinationEV:
    def test_cross_tabulation(self):
        records = [
            _cp_record(priority_tier="core_continuation", current_phase="healthy_breakout",
                       subsequent_5d_return=0.10),
            _cp_record(priority_tier="core_continuation", current_phase="healthy_breakout",
                       subsequent_5d_return=0.06),
            _cp_record(priority_tier="exit_candidate", current_phase="distribution",
                       subsequent_5d_return=-0.04),
        ]
        combos = compute_combination_ev(records)
        core_hb = next(c for c in combos
                       if c.tier == "core_continuation" and c.phase == "healthy_breakout")
        assert core_hb.n == 2
        assert core_hb.mean_5d_return == pytest.approx(0.08)

    def test_sorted_by_mean_return(self):
        records = [
            _cp_record(priority_tier="tier_a", current_phase="phase_x", subsequent_5d_return=0.01),
            _cp_record(priority_tier="tier_b", current_phase="phase_y", subsequent_5d_return=0.09),
        ]
        combos = compute_combination_ev(records)
        assert combos[0].mean_5d_return > combos[1].mean_5d_return

    def test_empty_records(self):
        assert compute_combination_ev([]) == []


# ── _overall_stats ────────────────────────────────────────────────────────────

class TestOverallStats:
    def test_correct_mean_and_win_rate(self):
        records = [
            _cp_record(subsequent_5d_return=0.10),
            _cp_record(subsequent_5d_return=-0.02),
            _cp_record(subsequent_5d_return=0.04),
        ]
        mean, win = _overall_stats(records)
        assert mean == pytest.approx((0.10 - 0.02 + 0.04) / 3, abs=1e-5)
        assert win == pytest.approx(2 / 3, abs=1e-4)

    def test_empty_returns_zeros(self):
        mean, win = _overall_stats([])
        assert mean == 0.0
        assert win == 0.0

    def test_skips_none_returns(self):
        records = [
            _cp_record(subsequent_5d_return=None),
            _cp_record(subsequent_5d_return=0.06),
        ]
        mean, win = _overall_stats(records)
        assert mean == pytest.approx(0.06)
        assert win == 1.0


# ── run_fco_analysis ──────────────────────────────────────────────────────────

class TestRunFcoAnalysis:
    def test_full_run_sufficient_data(self, tmp_path):
        records = _make_diverse_records(12)
        cp_file  = _make_cp_file(tmp_path, records)
        bq_file  = _make_bq_file(tmp_path, [])
        out_file = _make_fco_out(tmp_path)

        result = run_fco_analysis(
            cp_file=cp_file, bq_file=bq_file,
            output_file=out_file, today="2026-05-30",
        )
        assert isinstance(result, FcoResult)
        assert result.n_materialized == 12
        assert result.n_records_total == 12
        assert len(result.signal_ranks) > 0
        assert result.top_signal in NUMERIC_SIGNALS
        assert out_file.exists()

    def test_result_appended_to_jsonl(self, tmp_path):
        records = _make_diverse_records(10)
        cp_file  = _make_cp_file(tmp_path, records)
        bq_file  = _make_bq_file(tmp_path, [])
        out_file = _make_fco_out(tmp_path)

        run_fco_analysis(cp_file=cp_file, bq_file=bq_file,
                         output_file=out_file, today="2026-05-30")
        run_fco_analysis(cp_file=cp_file, bq_file=bq_file,
                         output_file=out_file, today="2026-05-30")

        lines = [l for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2  # two runs appended

    def test_insufficient_data_returns_empty_result(self, tmp_path):
        records = [_cp_record(date="2026-05-01", subsequent_5d_return=0.05)]  # only 1
        cp_file  = _make_cp_file(tmp_path, records)
        bq_file  = _make_bq_file(tmp_path, [])
        out_file = _make_fco_out(tmp_path)

        result = run_fco_analysis(
            cp_file=cp_file, bq_file=bq_file,
            output_file=out_file, today="2026-05-30",
        )
        assert result.n_materialized < MIN_RECORDS
        assert result.signal_ranks == []

    def test_missing_cp_file_returns_empty(self, tmp_path):
        bq_file  = _make_bq_file(tmp_path, [])
        out_file = _make_fco_out(tmp_path)
        result = run_fco_analysis(
            cp_file=tmp_path / "missing.jsonl",
            bq_file=bq_file,
            output_file=out_file,
            today="2026-05-30",
        )
        assert result.n_records_total == 0
        assert result.signal_ranks == []

    def test_lookback_filters_old_records(self, tmp_path):
        records = [
            _cp_record(date="2026-05-01", subsequent_5d_return=0.05),  # recent
            _cp_record(date="2024-01-01", subsequent_5d_return=0.08),  # old
        ]
        cp_file  = _make_cp_file(tmp_path, records)
        bq_file  = _make_bq_file(tmp_path, [])
        out_file = _make_fco_out(tmp_path)

        result = run_fco_analysis(
            cp_file=cp_file, bq_file=bq_file,
            output_file=out_file, today="2026-05-30",
            lookback_days=90,
        )
        assert result.n_records_total == 1  # only recent

    def test_bq_enrichment_used_in_analysis(self, tmp_path):
        records = _make_diverse_records(10)
        bq_records = [
            {"symbol": r["symbol"], "date": r["date"],
             "breakout_phase_type": "healthy_breakout", "cpr_score": 90.0}
            for r in records[:5]
        ]
        cp_file  = _make_cp_file(tmp_path, records)
        bq_file  = _make_bq_file(tmp_path, bq_records)
        out_file = _make_fco_out(tmp_path)

        result = run_fco_analysis(
            cp_file=cp_file, bq_file=bq_file,
            output_file=out_file, today="2026-05-30",
        )
        assert result.n_materialized == 10  # all records processed

    def test_schema_version_in_output(self, tmp_path):
        records = _make_diverse_records(10)
        cp_file  = _make_cp_file(tmp_path, records)
        out_file = _make_fco_out(tmp_path)

        run_fco_analysis(cp_file=cp_file, bq_file=tmp_path / "no.jsonl",
                         output_file=out_file, today="2026-05-30")
        lines = [l for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = json.loads(lines[0])
        assert rec["schema_version"] == SCHEMA_VERSION

    def test_never_raises(self, tmp_path):
        # Pass completely invalid paths — should not raise
        result = run_fco_analysis(
            cp_file=tmp_path / "x",
            bq_file=tmp_path / "y",
            output_file=tmp_path / "out" / "fco.jsonl",
            today="2026-05-30",
        )
        assert isinstance(result, FcoResult)


# ── append_fco_record ─────────────────────────────────────────────────────────

class TestAppendFcoRecord:
    def _empty_result(self) -> FcoResult:
        return FcoResult(
            date="2026-05-30",
            n_records_total=0,
            n_materialized=0,
            lookback_days=90,
            signal_ranks=[],
            top_signal="",
            top_signal_ir=0.0,
            priority_bucket_ev=[],
            tier_ev=[],
            phase_ev=[],
            combination_ev=[],
            overall_mean_5d=0.0,
            overall_win_rate=0.0,
        )

    def test_creates_file_and_appends(self, tmp_path):
        out = tmp_path / "fco.jsonl"
        result = self._empty_result()
        append_fco_record(result, out)
        assert out.exists()
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["date"] == "2026-05-30"

    def test_appends_multiple_times(self, tmp_path):
        out = tmp_path / "fco.jsonl"
        append_fco_record(self._empty_result(), out)
        append_fco_record(self._empty_result(), out)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_fail_open_on_invalid_path(self):
        # Should not raise even on completely invalid path
        append_fco_record(
            self._empty_result(),
            Path("/invalid/path/that/cannot/exist/fco.jsonl"),
        )


# ── format_fco_report ─────────────────────────────────────────────────────────

class TestFormatFcoReport:
    def _result_with_data(self) -> FcoResult:
        return FcoResult(
            date="2026-05-30",
            n_records_total=15,
            n_materialized=12,
            lookback_days=90,
            signal_ranks=[
                SignalRank(
                    signal_name="priority_score",
                    information_ratio=0.062,
                    mean_top_half=0.081,
                    mean_bottom_half=0.019,
                    win_rate_top=0.75,
                    win_rate_bottom=0.42,
                    n_total=12,
                ),
                SignalRank(
                    signal_name="rsr",
                    information_ratio=0.031,
                    mean_top_half=0.065,
                    mean_bottom_half=0.034,
                    win_rate_top=0.67,
                    win_rate_bottom=0.50,
                    n_total=12,
                ),
            ],
            top_signal="priority_score",
            top_signal_ir=0.062,
            priority_bucket_ev=[
                BucketEV("0-30", 0, 30, 2, -0.02, 0.3),
                BucketEV("30-45", 30, 45, 3, 0.01, 0.5),
                BucketEV("45-65", 45, 65, 4, 0.04, 0.6),
                BucketEV("65-80", 65, 80, 2, 0.07, 0.7),
                BucketEV("80-100", 80, 100, 1, 0.09, 0.8),
            ],
            tier_ev=[
                CategoryEV("core_continuation", 4, 0.08, 0.75),
                CategoryEV("high_quality",      5, 0.04, 0.60),
                CategoryEV("exit_candidate",     3, -0.01, 0.33),
            ],
            phase_ev=[
                CategoryEV("healthy_breakout", 6, 0.09, 0.83),
                CategoryEV("compression",      4, 0.03, 0.50),
                CategoryEV("distribution",     2, -0.02, 0.30),
            ],
            combination_ev=[
                CombinationEV("core_continuation", "healthy_breakout", 3, 0.10, 0.9),
                CombinationEV("exit_candidate",     "distribution",     2, -0.03, 0.2),
            ],
            overall_mean_5d=0.045,
            overall_win_rate=0.62,
        )

    def test_returns_non_empty_string(self):
        result = self._result_with_data()
        rpt = format_fco_report(result)
        assert isinstance(rpt, str)
        assert len(rpt) > 0

    def test_contains_section_headers(self):
        rpt = format_fco_report(self._result_with_data())
        assert "[1]" in rpt  # signal ranking
        assert "[2]" in rpt  # bucket ev
        assert "[3]" in rpt  # tier ev
        assert "[4]" in rpt  # phase ev

    def test_contains_top_signal_name(self):
        rpt = format_fco_report(self._result_with_data())
        assert "priority_score" in rpt

    def test_contains_kpi_summary(self):
        rpt = format_fco_report(self._result_with_data())
        assert "最重要KPI" in rpt

    def test_insufficient_data_shows_warning(self):
        result = FcoResult(
            date="2026-05-30",
            n_records_total=3,
            n_materialized=2,
            lookback_days=90,
            signal_ranks=[],
            top_signal="",
            top_signal_ir=0.0,
            priority_bucket_ev=[],
            tier_ev=[],
            phase_ev=[],
            combination_ev=[],
            overall_mean_5d=0.0,
            overall_win_rate=0.0,
        )
        rpt = format_fco_report(result)
        assert "データ不足" in rpt

    def test_never_raises(self):
        # Completely empty result
        result = FcoResult(
            date="X", n_records_total=0, n_materialized=0,
            lookback_days=90, signal_ranks=[], top_signal="",
            top_signal_ir=0.0, priority_bucket_ev=[], tier_ev=[],
            phase_ev=[], combination_ev=[], overall_mean_5d=0.0,
            overall_win_rate=0.0,
        )
        try:
            format_fco_report(result)
        except Exception as exc:
            pytest.fail(f"format_fco_report raised: {exc}")

    def test_contains_overall_stats(self):
        rpt = format_fco_report(self._result_with_data())
        assert "4.50%" in rpt or "4.5%" in rpt  # overall_mean_5d

    def test_combination_section_present_when_n_ge_2(self):
        rpt = format_fco_report(self._result_with_data())
        assert "[5]" in rpt


# ── Integration: signal ranking correctness ───────────────────────────────────

class TestSignalRankingIntegration:
    def test_anti_correlated_signal_has_negative_ir(self):
        """When higher signal → lower 5d return, IR should be negative."""
        records = []
        for i in range(10):
            records.append(_cp_record(
                symbol=f"{i}.T",
                date=f"2026-04-{i+1:02d}",
                priority_score=10.0 + i * 9.0,  # 10 to 91
                rsr=75.0,
                compression_score=50.0,
                breakout_quality_score=50.0,
                rsr_momentum=0.0,
                mfe_pct=0.05,
                hold_days=10,
                subsequent_5d_return=round(0.10 - i * 0.02, 4),  # 0.10 down to -0.08
            ))
        ranks = rank_signals(records)
        ps_rank = next((r for r in ranks if r.signal_name == "priority_score"), None)
        assert ps_rank is not None
        assert ps_rank.information_ratio < 0

    def test_uncorrelated_signal_ranks_below_correlated(self):
        """A signal uncorrelated with 5d return should rank lower than a correlated one."""
        records = []
        for i in range(10):
            score = 10.0 + i * 9.0          # 10→91 — perfectly correlated with return
            uncorrelated_rsr = 70.0 + (i % 3)  # cycles 70/71/72 — no correlation
            records.append(_cp_record(
                symbol=f"{i}.T",
                date=f"2026-04-{i+1:02d}",
                priority_score=score,
                rsr=uncorrelated_rsr,
                compression_score=50.0 + i * 0.1,  # near-flat
                breakout_quality_score=50.0,
                rsr_momentum=0.0,
                mfe_pct=0.05,
                hold_days=10,
                subsequent_5d_return=round(score / 1000.0, 4),  # linear of priority_score only
            ))
        ranks = rank_signals(records)
        ps_rank  = next((r for r in ranks if r.signal_name == "priority_score"), None)
        rsr_rank = next((r for r in ranks if r.signal_name == "rsr"), None)
        assert ps_rank is not None and rsr_rank is not None
        assert abs(ps_rank.information_ratio) > abs(rsr_rank.information_ratio)

    def test_ir_ordering_stable_across_runs(self):
        records = _make_diverse_records(14)
        r1 = rank_signals(records)
        r2 = rank_signals(records)
        assert [r.signal_name for r in r1] == [r.signal_name for r in r2]
