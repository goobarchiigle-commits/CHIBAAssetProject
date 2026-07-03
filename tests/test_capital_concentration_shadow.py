"""
tests/test_capital_concentration_shadow.py

Tests for src/analytics/capital_concentration_shadow.py
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from src.analytics.capital_concentration_shadow import (
    GAP_HIGH_THRESHOLD,
    GAP_MID_LOW,
    MIN_POSITIONS_PER_DATE,
    GapBucketStats,
    ConcentrationShadowOutput,
    DateSnapshot,
    _safe,
    _load_jsonl,
    _append_jsonl,
    load_and_filter_cp,
    group_by_date,
    compute_date_snapshot,
    _gap_bucket,
    _bucket_stats,
    compute_concentration_shadow,
    run_concentration_shadow,
    append_concentration_record,
    format_concentration_report,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _cp_record(
    date="2026-05-01",
    symbol="1234",
    priority_score=60.0,
    subsequent_5d_return=0.05,
    **kwargs,
):
    rec = {
        "date": date,
        "symbol": symbol,
        "priority_score": priority_score,
        "subsequent_5d_return": subsequent_5d_return,
        "priority_tier": "tier_2",
        "compression_score": 50.0,
        "rsr": 80.0,
        "current_phase": "phase_2",
        "schema_version": "v1",
    }
    rec.update(kwargs)
    return rec


def _write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_output(
    date="2026-05-01",
    n_records_total=10,
    n_materialized=8,
    n_dates_analyzed=5,
    mean_actual_return=0.03,
    mean_shadow_return=0.04,
    mean_concentration_alpha=0.01,
    positive_alpha_rate=0.6,
):
    return ConcentrationShadowOutput(
        date=date,
        n_records_total=n_records_total,
        n_materialized=n_materialized,
        n_dates_analyzed=n_dates_analyzed,
        lookback_days=90,
        min_positions=2,
        mean_actual_return=mean_actual_return,
        mean_shadow_return=mean_shadow_return,
        mean_concentration_alpha=mean_concentration_alpha,
        positive_alpha_rate=positive_alpha_rate,
        gap_high=GapBucketStats("high", 3, 0.02, 0.02, 0.04, 0.67),
        gap_mid=GapBucketStats("mid",  1, 0.005, 0.03, 0.035, 0.5),
        gap_low=GapBucketStats("low",  1, -0.005, 0.04, 0.035, 0.4),
        date_snapshots=[],
    )


# ── TestSafe ─────────────────────────────────────────────────────────────────

class TestSafe:
    def test_none_returns_fallback(self):
        assert _safe(None, 0.0) == 0.0

    def test_none_returns_none_fallback(self):
        assert _safe(None, None) is None

    def test_finite_float(self):
        assert _safe(3.14, 0.0) == pytest.approx(3.14)

    def test_int_input(self):
        assert _safe(5, 0.0) == pytest.approx(5.0)

    def test_string_number(self):
        assert _safe("2.5", 0.0) == pytest.approx(2.5)

    def test_nan_returns_fallback(self):
        assert _safe(float("nan"), 0.0) == pytest.approx(0.0)

    def test_inf_returns_fallback(self):
        assert _safe(float("inf"), 0.0) == pytest.approx(0.0)

    def test_invalid_string(self):
        assert _safe("abc", 0.0) == pytest.approx(0.0)


# ── TestLoadJsonl ─────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_jsonl(tmp_path / "missing.jsonl") == []

    def test_loads_valid_records(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _write_jsonl([{"a": 1}, {"b": 2}], p)
        records = _load_jsonl(p)
        assert len(records) == 2
        assert records[0]["a"] == 1

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        assert len(_load_jsonl(p)) == 2

    def test_skips_invalid_json(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a":1}\nnot json\n{"b":2}\n', encoding="utf-8")
        assert len(_load_jsonl(p)) == 2


# ── TestAppendJsonl ───────────────────────────────────────────────────────────

class TestAppendJsonl:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _append_jsonl({"x": 1}, p)
        assert p.exists()
        records = _load_jsonl(p)
        assert records[0]["x"] == 1

    def test_appends_to_existing(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _append_jsonl({"x": 1}, p)
        _append_jsonl({"x": 2}, p)
        records = _load_jsonl(p)
        assert len(records) == 2
        assert records[1]["x"] == 2

    def test_atomic_no_partial(self, tmp_path):
        p = tmp_path / "out.jsonl"
        for i in range(10):
            _append_jsonl({"i": i}, p)
        records = _load_jsonl(p)
        assert len(records) == 10


# ── TestLoadAndFilterCp ───────────────────────────────────────────────────────

class TestLoadAndFilterCp:
    def test_returns_empty_for_missing_file(self, tmp_path):
        within, mat = load_and_filter_cp(tmp_path / "missing.jsonl", 90, "2026-05-01")
        assert within == [] and mat == []

    def test_filters_by_lookback(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        old = _cp_record(date="2025-01-01", symbol="A")
        recent = _cp_record(date="2026-05-01", symbol="B")
        _write_jsonl([old, recent], p)
        within, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert all(r["symbol"] == "B" for r in within)

    def test_materialized_requires_positive_priority(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        zero_ps = _cp_record(date="2026-05-01", symbol="Z", priority_score=0.0)
        good = _cp_record(date="2026-05-01", symbol="G", priority_score=50.0)
        _write_jsonl([zero_ps, good], p)
        within, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(within) == 2
        assert len(mat) == 1
        assert mat[0]["symbol"] == "G"

    def test_materialized_requires_finite_5d_return(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        no_ret = _cp_record(date="2026-05-01", symbol="A", subsequent_5d_return=None)
        good = _cp_record(date="2026-05-01", symbol="B", subsequent_5d_return=0.03)
        _write_jsonl([no_ret, good], p)
        _, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(mat) == 1
        assert mat[0]["symbol"] == "B"

    def test_skips_invalid_date(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        bad = {"date": "not-a-date", "symbol": "X"}
        good = _cp_record(date="2026-05-01")
        _write_jsonl([bad, good], p)
        within, _ = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(within) == 1

    def test_returns_both_all_and_materialized(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        recs = [
            _cp_record(date="2026-05-01", symbol="A", subsequent_5d_return=0.05),
            _cp_record(date="2026-05-01", symbol="B", subsequent_5d_return=None),
        ]
        _write_jsonl(recs, p)
        within, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(within) == 2
        assert len(mat) == 1


# ── TestGroupByDate ───────────────────────────────────────────────────────────

class TestGroupByDate:
    def test_groups_correctly(self):
        recs = [
            _cp_record(date="2026-05-01", symbol="A"),
            _cp_record(date="2026-05-01", symbol="B"),
            _cp_record(date="2026-05-02", symbol="C"),
        ]
        groups = group_by_date(recs)
        assert len(groups["2026-05-01"]) == 2
        assert len(groups["2026-05-02"]) == 1

    def test_empty_input(self):
        assert group_by_date([]) == {}

    def test_skips_missing_date(self):
        recs = [{"symbol": "A"}, _cp_record(date="2026-05-01", symbol="B")]
        groups = group_by_date(recs)
        assert "2026-05-01" in groups
        assert "" not in groups or groups.get("") == []


# ── TestComputeDateSnapshot ───────────────────────────────────────────────────

class TestComputeDateSnapshot:
    def test_basic_two_positions(self):
        positions = [
            _cp_record(priority_score=80.0, subsequent_5d_return=0.10),
            _cp_record(priority_score=40.0, subsequent_5d_return=0.02),
        ]
        snap = compute_date_snapshot("2026-05-01", positions)
        assert snap is not None
        assert snap.n_positions == 2
        # actual = (0.10 + 0.02) / 2 = 0.06
        assert snap.actual_return == pytest.approx(0.06)
        # shadow = 80/120 * 0.10 + 40/120 * 0.02 = 0.0667 + 0.0067 ≈ 0.0733
        expected_shadow = (80 / 120) * 0.10 + (40 / 120) * 0.02
        assert snap.shadow_return == pytest.approx(expected_shadow, abs=1e-5)
        assert snap.concentration_alpha == pytest.approx(expected_shadow - 0.06, abs=1e-5)

    def test_returns_none_for_single_position(self):
        positions = [_cp_record(priority_score=60.0, subsequent_5d_return=0.05)]
        snap = compute_date_snapshot("2026-05-01", positions)
        assert snap is None

    def test_returns_none_for_zero_total_priority(self):
        positions = [
            _cp_record(priority_score=0.0, subsequent_5d_return=0.05),
            _cp_record(priority_score=0.0, subsequent_5d_return=0.03),
        ]
        snap = compute_date_snapshot("2026-05-01", positions)
        assert snap is None

    def test_equal_priority_gives_zero_alpha(self):
        positions = [
            _cp_record(priority_score=50.0, subsequent_5d_return=0.05),
            _cp_record(priority_score=50.0, subsequent_5d_return=0.03),
        ]
        snap = compute_date_snapshot("2026-05-01", positions)
        assert snap is not None
        assert snap.concentration_alpha == pytest.approx(0.0, abs=1e-6)

    def test_priority_gap_computed_correctly(self):
        positions = [
            _cp_record(priority_score=90.0, subsequent_5d_return=0.05),
            _cp_record(priority_score=30.0, subsequent_5d_return=0.03),
        ]
        snap = compute_date_snapshot("2026-05-01", positions)
        assert snap.priority_score_gap == pytest.approx(60.0)

    def test_three_positions(self):
        positions = [
            _cp_record(priority_score=80.0, subsequent_5d_return=0.10),
            _cp_record(priority_score=60.0, subsequent_5d_return=0.05),
            _cp_record(priority_score=40.0, subsequent_5d_return=0.00),
        ]
        snap = compute_date_snapshot("2026-05-01", positions)
        assert snap is not None
        total = 80 + 60 + 40  # 180
        expected_shadow = (80/180)*0.10 + (60/180)*0.05 + (40/180)*0.00
        assert snap.shadow_return == pytest.approx(expected_shadow, abs=1e-5)


# ── TestGapBucket ─────────────────────────────────────────────────────────────

class TestGapBucket:
    def test_high(self):
        assert _gap_bucket(GAP_HIGH_THRESHOLD) == "high"
        assert _gap_bucket(50.0) == "high"

    def test_mid(self):
        assert _gap_bucket(GAP_MID_LOW) == "mid"
        assert _gap_bucket(20.0) == "mid"
        assert _gap_bucket(GAP_HIGH_THRESHOLD - 0.1) == "mid"

    def test_low(self):
        assert _gap_bucket(0.0) == "low"
        assert _gap_bucket(GAP_MID_LOW - 0.1) == "low"


# ── TestBucketStats ───────────────────────────────────────────────────────────

class TestBucketStats:
    def _make_snap(self, gap, alpha):
        return DateSnapshot(
            date="2026-05-01",
            n_positions=2,
            actual_return=0.03,
            shadow_return=0.03 + alpha,
            concentration_alpha=alpha,
            priority_score_gap=gap,
            mean_priority=50.0,
        )

    def test_empty_bucket(self):
        stats = _bucket_stats([], "high")
        assert stats.n_dates == 0
        assert stats.mean_alpha == 0.0

    def test_single_snap_high_gap(self):
        snap = self._make_snap(gap=35.0, alpha=0.02)
        stats = _bucket_stats([snap], "high")
        assert stats.n_dates == 1
        assert stats.mean_alpha == pytest.approx(0.02)

    def test_positive_alpha_rate(self):
        snaps = [
            self._make_snap(gap=35.0, alpha=0.02),
            self._make_snap(gap=40.0, alpha=-0.01),
            self._make_snap(gap=50.0, alpha=0.03),
        ]
        stats = _bucket_stats(snaps, "high")
        assert stats.n_dates == 3
        assert stats.positive_alpha_rate == pytest.approx(2/3, abs=1e-4)

    def test_mid_bucket_filters_correctly(self):
        snaps = [
            self._make_snap(gap=5.0, alpha=0.01),   # low
            self._make_snap(gap=15.0, alpha=0.02),  # mid
            self._make_snap(gap=35.0, alpha=0.03),  # high
        ]
        mid_stats = _bucket_stats(snaps, "mid")
        assert mid_stats.n_dates == 1
        assert mid_stats.mean_alpha == pytest.approx(0.02)


# ── TestComputeConcentrationShadow ────────────────────────────────────────────

class TestComputeConcentrationShadow:
    def _two_pos_date(self, date, ps_a, ps_b, ret_a, ret_b):
        return [
            _cp_record(date=date, symbol="A", priority_score=ps_a, subsequent_5d_return=ret_a),
            _cp_record(date=date, symbol="B", priority_score=ps_b, subsequent_5d_return=ret_b),
        ]

    def test_basic_positive_alpha(self):
        recs = self._two_pos_date("2026-05-01", 80.0, 20.0, 0.10, 0.00)
        result = compute_concentration_shadow(recs, 90, "2026-05-30", 2)
        assert result.n_dates_analyzed == 1
        # actual = 0.05, shadow = 0.8*0.10 + 0.2*0.00 = 0.08
        assert result.mean_actual_return == pytest.approx(0.05)
        assert result.mean_shadow_return == pytest.approx(0.08)
        assert result.mean_concentration_alpha == pytest.approx(0.03, abs=1e-5)

    def test_zero_alpha_equal_priority(self):
        recs = self._two_pos_date("2026-05-01", 50.0, 50.0, 0.05, 0.03)
        result = compute_concentration_shadow(recs, 90, "2026-05-30", 2)
        assert result.mean_concentration_alpha == pytest.approx(0.0, abs=1e-6)

    def test_empty_materialized(self):
        result = compute_concentration_shadow([], 90, "2026-05-30", 0)
        assert result.n_dates_analyzed == 0
        assert result.mean_concentration_alpha == 0.0

    def test_single_position_dates_excluded(self):
        recs = [
            _cp_record(date="2026-05-01", symbol="A", priority_score=70.0, subsequent_5d_return=0.05),
        ]
        result = compute_concentration_shadow(recs, 90, "2026-05-30", 1)
        assert result.n_dates_analyzed == 0

    def test_multi_date_aggregation(self):
        recs = (
            self._two_pos_date("2026-05-01", 80.0, 20.0, 0.10, 0.00)  # alpha>0
            + self._two_pos_date("2026-05-02", 20.0, 80.0, 0.10, 0.00)  # alpha<0
        )
        result = compute_concentration_shadow(recs, 90, "2026-05-30", 4)
        assert result.n_dates_analyzed == 2
        assert abs(result.mean_concentration_alpha) < 0.01  # nearly cancels

    def test_positive_alpha_rate(self):
        recs = (
            self._two_pos_date("2026-05-01", 80.0, 20.0, 0.10, 0.00)  # alpha>0
            + self._two_pos_date("2026-05-02", 50.0, 50.0, 0.05, 0.03)  # alpha=0
        )
        result = compute_concentration_shadow(recs, 90, "2026-05-30", 4)
        # 1 out of 2 dates has alpha > 0
        assert result.positive_alpha_rate == pytest.approx(0.5)

    def test_gap_bucket_distribution(self):
        recs = (
            self._two_pos_date("2026-05-01", 90.0, 10.0, 0.10, 0.02)   # gap=80 → high
            + self._two_pos_date("2026-05-02", 65.0, 50.0, 0.05, 0.03) # gap=15 → mid
            + self._two_pos_date("2026-05-03", 52.0, 48.0, 0.04, 0.04) # gap=4  → low
        )
        result = compute_concentration_shadow(recs, 90, "2026-05-30", 6)
        assert result.gap_high.n_dates == 1
        assert result.gap_mid.n_dates == 1
        assert result.gap_low.n_dates == 1


# ── TestRunConcentrationShadow ────────────────────────────────────────────────

class TestRunConcentrationShadow:
    def test_missing_cp_file_returns_empty(self, tmp_path):
        cp = tmp_path / "missing.jsonl"
        out = tmp_path / "out.jsonl"
        result = run_concentration_shadow(cp, out, today="2026-05-30")
        assert result.n_dates_analyzed == 0

    def test_writes_jsonl(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "ccs.jsonl"
        recs = [
            _cp_record(date="2026-05-01", symbol="A", priority_score=80.0, subsequent_5d_return=0.10),
            _cp_record(date="2026-05-01", symbol="B", priority_score=20.0, subsequent_5d_return=0.02),
        ]
        _write_jsonl(recs, cp)
        run_concentration_shadow(cp, out, today="2026-05-30")
        assert out.exists()
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_appends_on_second_call(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "ccs.jsonl"
        recs = [
            _cp_record(date="2026-05-01", symbol="A", priority_score=80.0, subsequent_5d_return=0.10),
            _cp_record(date="2026-05-01", symbol="B", priority_score=20.0, subsequent_5d_return=0.02),
        ]
        _write_jsonl(recs, cp)
        run_concentration_shadow(cp, out, today="2026-05-30")
        run_concentration_shadow(cp, out, today="2026-05-31")
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_never_raises_on_corrupt_file(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        cp.write_text("not json\n{bad}\n", encoding="utf-8")
        result = run_concentration_shadow(cp, out, today="2026-05-30")
        assert result is not None

    def test_concentration_alpha_positive_for_high_priority_winner(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = [
            _cp_record(date="2026-05-01", symbol="A", priority_score=90.0, subsequent_5d_return=0.15),
            _cp_record(date="2026-05-01", symbol="B", priority_score=10.0, subsequent_5d_return=0.01),
        ]
        _write_jsonl(recs, cp)
        result = run_concentration_shadow(cp, out, today="2026-05-30")
        # Shadow overweights the winner → positive alpha
        assert result.mean_concentration_alpha > 0


# ── TestAppendConcentrationRecord ─────────────────────────────────────────────

class TestAppendConcentrationRecord:
    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "ccs.jsonl"
        output = _make_output()
        append_concentration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["mean_concentration_alpha"] == output.mean_concentration_alpha
        assert "gap_high" in rec
        assert "gap_mid" in rec
        assert "gap_low" in rec
        assert rec["schema_version"] == "v1"

    def test_gap_bucket_fields_present(self, tmp_path):
        p = tmp_path / "ccs.jsonl"
        output = _make_output()
        append_concentration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        for bkt in ("gap_high", "gap_mid", "gap_low"):
            assert "n_dates" in rec[bkt]
            assert "mean_alpha" in rec[bkt]
            assert "positive_alpha_rate" in rec[bkt]

    def test_appends_without_truncating(self, tmp_path):
        p = tmp_path / "ccs.jsonl"
        append_concentration_record(_make_output(date="2026-05-01"), p)
        append_concentration_record(_make_output(date="2026-05-02"), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2


# ── TestFormatConcentrationReport ─────────────────────────────────────────────

class TestFormatConcentrationReport:
    def test_empty_data_returns_no_data_msg(self):
        output = _make_output(n_dates_analyzed=0)
        rpt = format_concentration_report(output)
        assert "データ不足" in rpt

    def test_positive_alpha_shows_verdict(self):
        output = _make_output(mean_concentration_alpha=0.02)
        rpt = format_concentration_report(output)
        assert "priority比例" in rpt
        assert "concentration_alpha" in rpt

    def test_negative_alpha_shows_equal_weight_verdict(self):
        output = _make_output(
            mean_concentration_alpha=-0.01,
            mean_shadow_return=0.02,
            mean_actual_return=0.03,
        )
        rpt = format_concentration_report(output)
        assert "等ウェイト" in rpt

    def test_report_contains_kpi_values(self):
        output = _make_output(
            mean_actual_return=0.03,
            mean_shadow_return=0.04,
            mean_concentration_alpha=0.01,
        )
        rpt = format_concentration_report(output)
        assert "CAPITAL CONCENTRATION SHADOW" in rpt
        assert "3.00%" in rpt or "+3.00%" in rpt or "+3%" in rpt
        assert "4.00%" in rpt or "+4.00%" in rpt or "+4%" in rpt

    def test_report_contains_gap_buckets(self):
        output = _make_output()
        rpt = format_concentration_report(output)
        assert "gap" in rpt.lower() or "Gap" in rpt

    def test_fail_open_on_corrupt_output(self):
        class BadOutput:
            n_dates_analyzed = None  # will cause attr error

        rpt = format_concentration_report(BadOutput())
        assert isinstance(rpt, str)

    def test_returns_string(self):
        output = _make_output()
        rpt = format_concentration_report(output)
        assert isinstance(rpt, str)

    def test_minor_alpha_shows_monitoring_verdict(self):
        output = _make_output(mean_concentration_alpha=0.0001)
        rpt = format_concentration_report(output)
        assert "モニタリング" in rpt or "軽微" in rpt


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_positive_alpha(self, tmp_path):
        """End-to-end: high-priority symbol wins → positive concentration_alpha."""
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "ccs.jsonl"
        # Three dates; high-priority winner on each date
        recs = []
        for i, d in enumerate(["2026-05-01", "2026-05-02", "2026-05-03"]):
            recs.append(_cp_record(date=d, symbol="A", priority_score=85.0, subsequent_5d_return=0.12))
            recs.append(_cp_record(date=d, symbol="B", priority_score=15.0, subsequent_5d_return=0.01))
        _write_jsonl(recs, cp)
        result = run_concentration_shadow(cp, out, today="2026-05-30")
        assert result.n_dates_analyzed == 3
        assert result.mean_concentration_alpha > 0
        assert result.positive_alpha_rate == 1.0
        # JSONL written
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["mean_concentration_alpha"] > 0

    def test_full_pipeline_equal_allocation_baseline(self, tmp_path):
        """Equal priority → zero alpha (baseline check)."""
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for d in ["2026-05-01", "2026-05-02"]:
            recs.append(_cp_record(date=d, symbol="A", priority_score=50.0, subsequent_5d_return=0.08))
            recs.append(_cp_record(date=d, symbol="B", priority_score=50.0, subsequent_5d_return=0.04))
        _write_jsonl(recs, cp)
        result = run_concentration_shadow(cp, out, today="2026-05-30")
        assert result.mean_concentration_alpha == pytest.approx(0.0, abs=1e-6)

    def test_report_format_for_positive_alpha(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = [
            _cp_record(date="2026-05-01", symbol="A", priority_score=90.0, subsequent_5d_return=0.15),
            _cp_record(date="2026-05-01", symbol="B", priority_score=10.0, subsequent_5d_return=0.01),
        ]
        _write_jsonl(recs, cp)
        result = run_concentration_shadow(cp, out, today="2026-05-30")
        rpt = format_concentration_report(result)
        assert "CAPITAL CONCENTRATION SHADOW" in rpt
        assert len(rpt) > 100
