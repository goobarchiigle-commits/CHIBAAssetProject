"""
tests/test_hold_duration_calibration.py

Tests for src/analytics/hold_duration_calibration.py
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from src.analytics.hold_duration_calibration import (
    BUCKET_COUNT,
    DURATION_BUCKET_DEFS,
    DURATION_BUCKET_LABELS,
    MIN_BUCKET_SAMPLES,
    DurationBucket,
    SuppressionComparison,
    PhaseStats,
    HoldDurationOutput,
    _safe,
    _safe_int,
    _safe_bool,
    _load_jsonl,
    _append_jsonl,
    _bucket_index,
    load_and_filter_cp,
    compute_duration_buckets,
    compute_suppression_comparison,
    compute_phase_stats,
    run_hold_duration_calibration,
    append_duration_record,
    format_duration_report,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _rec(
    date="2026-05-01",
    symbol="1234",
    hold_days=5,
    subsequent_5d_return=0.05,
    subsequent_10d_return=None,
    suppression_eligible=False,
    current_phase="neutral",
    priority_score=60.0,
    **kwargs,
):
    rec = {
        "date": date,
        "symbol": symbol,
        "hold_days": hold_days,
        "subsequent_5d_return": subsequent_5d_return,
        "suppression_eligible": suppression_eligible,
        "current_phase": current_phase,
        "priority_score": priority_score,
        "schema_version": "v1",
    }
    if subsequent_10d_return is not None:
        rec["subsequent_10d_return"] = subsequent_10d_return
    rec.update(kwargs)
    return rec


def _write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_sup():
    return SuppressionComparison(
        suppressed_n=5,
        suppressed_avg_5d=0.04,
        suppressed_win_rate=0.6,
        non_suppressed_n=10,
        non_suppressed_avg_5d=0.03,
        non_suppressed_win_rate=0.5,
        suppression_return_delta=0.01,
    )


def _make_output(n_buckets_populated=3, best_bucket="7-10d", worst_bucket="1-3d"):
    buckets = [
        DurationBucket(
            label=DURATION_BUCKET_LABELS[i],
            low=DURATION_BUCKET_DEFS[i][0],
            high=DURATION_BUCKET_DEFS[i][1],
            n=5 if i < 3 else 0,
            avg_5d_return=0.01 * (i + 1) if i < 3 else 0.0,
            avg_10d_return=None,
            win_rate_5d=0.5 if i < 3 else 0.0,
            win_rate_10d=None,
        )
        for i in range(BUCKET_COUNT)
    ]
    return HoldDurationOutput(
        date="2026-05-30",
        n_records_total=20,
        n_materialized=15,
        lookback_days=90,
        min_bucket_samples=3,
        buckets=buckets,
        n_buckets_populated=n_buckets_populated,
        best_duration_bucket=best_bucket,
        worst_duration_bucket=worst_bucket,
        duration_ev_spread=0.02,
        suppression=_make_sup(),
        suppression_return_delta=0.01,
        phase_stats=[
            PhaseStats("compression", 5, 0.04, 0.6, "7-10d"),
            PhaseStats("breakout",    7, 0.06, 0.7, "4-6d"),
            PhaseStats("neutral",     3, 0.02, 0.4, "1-3d"),
        ],
        phase_duration_leader="breakout",
    )


# ── TestSafeHelpers ───────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_none(self):
        assert _safe(None, 0.0) == 0.0

    def test_safe_float(self):
        assert _safe(3.14, 0.0) == pytest.approx(3.14)

    def test_safe_nan(self):
        assert _safe(float("nan"), 0.0) == pytest.approx(0.0)

    def test_safe_int_valid(self):
        assert _safe_int(5) == 5

    def test_safe_int_float(self):
        assert _safe_int(4.9) == 4

    def test_safe_int_string(self):
        assert _safe_int("7") == 7

    def test_safe_int_none(self):
        assert _safe_int(None) is None

    def test_safe_int_invalid(self):
        assert _safe_int("abc") is None

    def test_safe_bool_true(self):
        assert _safe_bool(True)  is True
        assert _safe_bool("true") is True
        assert _safe_bool("1")    is True

    def test_safe_bool_false(self):
        assert _safe_bool(False)   is False
        assert _safe_bool("false") is False
        assert _safe_bool("0")     is False

    def test_safe_bool_none(self):
        assert _safe_bool(None) is None

    def test_safe_bool_int_one(self):
        assert _safe_bool(1) is True

    def test_safe_bool_int_zero(self):
        assert _safe_bool(0) is False


# ── TestLoadJsonl ─────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_missing_file(self, tmp_path):
        assert _load_jsonl(tmp_path / "x.jsonl") == []

    def test_loads_records(self, tmp_path):
        p = tmp_path / "t.jsonl"
        _write_jsonl([{"a": 1}, {"b": 2}], p)
        assert len(_load_jsonl(p)) == 2

    def test_skips_invalid(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"a":1}\nnot json\n{"b":2}\n', encoding="utf-8")
        assert len(_load_jsonl(p)) == 2


# ── TestAppendJsonl ───────────────────────────────────────────────────────────

class TestAppendJsonl:
    def test_creates_and_appends(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _append_jsonl({"x": 1}, p)
        _append_jsonl({"x": 2}, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2


# ── TestBucketIndex ───────────────────────────────────────────────────────────

class TestBucketIndex:
    def test_hold_1(self):
        assert _bucket_index(1) == 0    # 1-3d

    def test_hold_3(self):
        assert _bucket_index(3) == 0    # 1-3d

    def test_hold_4(self):
        assert _bucket_index(4) == 1    # 4-6d

    def test_hold_6(self):
        assert _bucket_index(6) == 1    # 4-6d

    def test_hold_7(self):
        assert _bucket_index(7) == 2    # 7-10d

    def test_hold_10(self):
        assert _bucket_index(10) == 2   # 7-10d

    def test_hold_11(self):
        assert _bucket_index(11) == 3   # 11-15d

    def test_hold_15(self):
        assert _bucket_index(15) == 3   # 11-15d

    def test_hold_16(self):
        assert _bucket_index(16) == 4   # 16-20d

    def test_hold_20(self):
        assert _bucket_index(20) == 4   # 16-20d

    def test_hold_21(self):
        assert _bucket_index(21) == 5   # 21d+

    def test_hold_100(self):
        assert _bucket_index(100) == 5  # 21d+

    def test_hold_zero_clamps(self):
        assert _bucket_index(0) == 0    # clamped to 1 → 1-3d

    def test_hold_negative_clamps(self):
        assert _bucket_index(-5) == 0


# ── TestLoadAndFilterCp ───────────────────────────────────────────────────────

class TestLoadAndFilterCp:
    def test_missing_file(self, tmp_path):
        w, m = load_and_filter_cp(tmp_path / "x.jsonl", 90, "2026-05-30")
        assert w == [] and m == []

    def test_filters_old_records(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        _write_jsonl([
            _rec(date="2025-01-01", symbol="A"),
            _rec(date="2026-05-01", symbol="B"),
        ], p)
        within, _ = load_and_filter_cp(p, 90, "2026-05-30")
        assert all(r["symbol"] == "B" for r in within)

    def test_materialized_needs_5d_return(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        _write_jsonl([
            _rec(date="2026-05-01", symbol="A", subsequent_5d_return=None),
            _rec(date="2026-05-01", symbol="B", subsequent_5d_return=0.05),
        ], p)
        _, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(mat) == 1 and mat[0]["symbol"] == "B"

    def test_materialized_needs_hold_days(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        _write_jsonl([
            {"date": "2026-05-01", "subsequent_5d_return": 0.05},  # no hold_days
            _rec(date="2026-05-01", symbol="B", hold_days=5),
        ], p)
        _, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(mat) == 1 and mat[0]["symbol"] == "B"

    def test_materialized_requires_hold_days_ge_1(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        _write_jsonl([
            _rec(date="2026-05-01", symbol="A", hold_days=0),
            _rec(date="2026-05-01", symbol="B", hold_days=1),
        ], p)
        _, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(mat) == 1 and mat[0]["symbol"] == "B"


# ── TestComputeDurationBuckets ────────────────────────────────────────────────

class TestComputeDurationBuckets:
    def test_returns_six_buckets(self):
        assert len(compute_duration_buckets([])) == BUCKET_COUNT

    def test_empty_input_all_zero(self):
        buckets = compute_duration_buckets([])
        assert all(b.n == 0 for b in buckets)

    def test_single_record_correct_bucket(self):
        recs = [_rec(hold_days=5, subsequent_5d_return=0.08)]
        buckets = compute_duration_buckets(recs)
        assert buckets[1].n == 1     # 4-6d
        assert buckets[1].avg_5d_return == pytest.approx(0.08)

    def test_avg_computed_correctly(self):
        recs = [
            _rec(hold_days=2, subsequent_5d_return=0.04),
            _rec(hold_days=3, subsequent_5d_return=0.06),
        ]
        buckets = compute_duration_buckets(recs)
        assert buckets[0].n == 2
        assert buckets[0].avg_5d_return == pytest.approx(0.05)

    def test_win_rate_5d(self):
        recs = [
            _rec(hold_days=8, subsequent_5d_return=0.05),
            _rec(hold_days=9, subsequent_5d_return=-0.02),
            _rec(hold_days=10, subsequent_5d_return=0.01),
        ]
        buckets = compute_duration_buckets(recs)
        assert buckets[2].n == 3      # 7-10d
        assert buckets[2].win_rate_5d == pytest.approx(2/3, abs=1e-4)

    def test_10d_return_optional(self):
        recs = [_rec(hold_days=12, subsequent_5d_return=0.05,
                     subsequent_10d_return=0.09)]
        buckets = compute_duration_buckets(recs)
        assert buckets[3].avg_10d_return == pytest.approx(0.09)

    def test_10d_none_when_absent(self):
        recs = [_rec(hold_days=5, subsequent_5d_return=0.05)]
        buckets = compute_duration_buckets(recs)
        assert buckets[1].avg_10d_return is None

    def test_suppression_sub_group(self):
        recs = [
            _rec(hold_days=5, subsequent_5d_return=0.08, suppression_eligible=True),
            _rec(hold_days=6, subsequent_5d_return=0.02, suppression_eligible=False),
        ]
        buckets = compute_duration_buckets(recs)
        b = buckets[1]  # 4-6d
        assert b.suppressed_n == 1
        assert b.suppressed_avg_5d == pytest.approx(0.08)
        assert b.non_suppressed_n == 1
        assert b.non_suppressed_avg_5d == pytest.approx(0.02)

    def test_21d_plus_bucket(self):
        recs = [
            _rec(hold_days=21, subsequent_5d_return=0.10),
            _rec(hold_days=45, subsequent_5d_return=0.12),
        ]
        buckets = compute_duration_buckets(recs)
        assert buckets[5].n == 2     # 21d+

    def test_bucket_labels(self):
        buckets = compute_duration_buckets([])
        assert buckets[0].label == "1-3d"
        assert buckets[5].label == "21d+"


# ── TestComputeSuppressionComparison ─────────────────────────────────────────

class TestComputeSuppressionComparison:
    def test_empty_records(self):
        sup = compute_suppression_comparison([])
        assert sup.suppressed_n == 0
        assert sup.non_suppressed_n == 0
        assert sup.suppression_return_delta == 0.0

    def test_suppressed_higher(self):
        recs = [
            _rec(subsequent_5d_return=0.10, suppression_eligible=True),
            _rec(subsequent_5d_return=0.10, suppression_eligible=True),
            _rec(subsequent_5d_return=0.02, suppression_eligible=False),
            _rec(subsequent_5d_return=0.02, suppression_eligible=False),
        ]
        sup = compute_suppression_comparison(recs)
        assert sup.suppressed_n == 2
        assert sup.non_suppressed_n == 2
        assert sup.suppressed_avg_5d == pytest.approx(0.10)
        assert sup.non_suppressed_avg_5d == pytest.approx(0.02)
        assert sup.suppression_return_delta == pytest.approx(0.08)

    def test_suppressed_lower(self):
        recs = [
            _rec(subsequent_5d_return=0.02, suppression_eligible=True),
            _rec(subsequent_5d_return=0.08, suppression_eligible=False),
        ]
        sup = compute_suppression_comparison(recs)
        assert sup.suppression_return_delta < 0

    def test_win_rate_computed(self):
        recs = [
            _rec(subsequent_5d_return=0.05, suppression_eligible=True),
            _rec(subsequent_5d_return=-0.02, suppression_eligible=True),
            _rec(subsequent_5d_return=0.03, suppression_eligible=False),
        ]
        sup = compute_suppression_comparison(recs)
        assert sup.suppressed_win_rate == pytest.approx(0.5)
        assert sup.non_suppressed_win_rate == pytest.approx(1.0)

    def test_skips_records_without_suppression_field(self):
        recs = [
            {"subsequent_5d_return": 0.05},   # no suppression_eligible
            _rec(subsequent_5d_return=0.03, suppression_eligible=True),
        ]
        sup = compute_suppression_comparison(recs)
        assert sup.suppressed_n == 1


# ── TestComputePhaseStats ─────────────────────────────────────────────────────

class TestComputePhaseStats:
    def test_empty_records(self):
        assert compute_phase_stats([]) == []

    def test_known_phases_present(self):
        recs = [
            _rec(hold_days=5, subsequent_5d_return=0.04, current_phase="compression"),
            _rec(hold_days=8, subsequent_5d_return=0.08, current_phase="breakout"),
            _rec(hold_days=3, subsequent_5d_return=0.02, current_phase="neutral"),
        ]
        stats = compute_phase_stats(recs)
        phases = {s.phase for s in stats}
        assert "compression" in phases
        assert "breakout" in phases
        assert "neutral" in phases

    def test_unknown_phase_bucketed_as_other(self):
        recs = [_rec(hold_days=5, subsequent_5d_return=0.03, current_phase="special")]
        stats = compute_phase_stats(recs)
        phases = {s.phase for s in stats}
        assert "other" in phases

    def test_avg_5d_correct(self):
        recs = [
            _rec(hold_days=5, subsequent_5d_return=0.04, current_phase="compression"),
            _rec(hold_days=6, subsequent_5d_return=0.06, current_phase="compression"),
        ]
        stats = compute_phase_stats(recs)
        comp = next(s for s in stats if s.phase == "compression")
        assert comp.avg_5d_return == pytest.approx(0.05)

    def test_modal_bucket_identified(self):
        recs = [
            _rec(hold_days=5, subsequent_5d_return=0.05, current_phase="breakout"),
            _rec(hold_days=6, subsequent_5d_return=0.04, current_phase="breakout"),
            _rec(hold_days=8, subsequent_5d_return=0.06, current_phase="breakout"),
        ]
        stats = compute_phase_stats(recs)
        brk = next(s for s in stats if s.phase == "breakout")
        # 2 records in 4-6d, 1 in 7-10d → modal = 4-6d
        assert brk.modal_bucket == "4-6d"

    def test_omits_zero_n_phases(self):
        recs = [_rec(hold_days=5, subsequent_5d_return=0.05, current_phase="compression")]
        stats = compute_phase_stats(recs)
        # breakout and neutral have n=0 and should not appear
        phases_with_data = {s.phase for s in stats}
        assert "breakout" not in phases_with_data
        assert "neutral" not in phases_with_data

    def test_win_rate_correct(self):
        recs = [
            _rec(hold_days=5, subsequent_5d_return=0.05, current_phase="neutral"),
            _rec(hold_days=5, subsequent_5d_return=-0.02, current_phase="neutral"),
            _rec(hold_days=5, subsequent_5d_return=0.03, current_phase="neutral"),
        ]
        stats = compute_phase_stats(recs)
        neu = next(s for s in stats if s.phase == "neutral")
        assert neu.win_rate == pytest.approx(2/3, abs=1e-4)


# ── TestRunHoldDurationCalibration ────────────────────────────────────────────

class TestRunHoldDurationCalibration:
    def test_missing_file_returns_empty(self, tmp_path):
        result = run_hold_duration_calibration(
            cp_file=tmp_path / "x.jsonl",
            output_file=tmp_path / "out.jsonl",
            today="2026-05-30",
        )
        assert result.n_records_total == 0

    def test_writes_jsonl(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "hdc.jsonl"
        recs = [_rec(hold_days=i+1, subsequent_5d_return=0.01*i) for i in range(10)]
        _write_jsonl(recs, cp)
        run_hold_duration_calibration(cp, out, today="2026-05-30")
        assert out.exists()
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_appends_on_second_call(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "hdc.jsonl"
        recs = [_rec(hold_days=5, subsequent_5d_return=0.05)]
        _write_jsonl(recs, cp)
        run_hold_duration_calibration(cp, out, today="2026-05-30")
        run_hold_duration_calibration(cp, out, today="2026-05-31")
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_never_raises_on_corrupt_file(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        cp.write_text("not json\n", encoding="utf-8")
        result = run_hold_duration_calibration(cp, out, today="2026-05-30")
        assert result is not None

    def test_best_bucket_identified(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        # 21d+ bucket has highest return
        for _ in range(5):
            recs.append(_rec(hold_days=2,  subsequent_5d_return=0.01))
            recs.append(_rec(hold_days=5,  subsequent_5d_return=0.03))
            recs.append(_rec(hold_days=25, subsequent_5d_return=0.10))
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30", min_samples=3)
        assert result.best_duration_bucket == "21d+"

    def test_duration_ev_spread(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for _ in range(5):
            recs.append(_rec(hold_days=2,  subsequent_5d_return=0.01))
            recs.append(_rec(hold_days=25, subsequent_5d_return=0.06))
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30", min_samples=3)
        assert result.duration_ev_spread == pytest.approx(0.05, abs=1e-4)

    def test_suppression_delta_computed(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = [
            _rec(hold_days=5, subsequent_5d_return=0.08, suppression_eligible=True),
            _rec(hold_days=5, subsequent_5d_return=0.08, suppression_eligible=True),
            _rec(hold_days=5, subsequent_5d_return=0.02, suppression_eligible=False),
            _rec(hold_days=5, subsequent_5d_return=0.02, suppression_eligible=False),
        ]
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30", min_samples=1)
        assert result.suppression_return_delta == pytest.approx(0.06, abs=1e-4)


# ── TestAppendDurationRecord ──────────────────────────────────────────────────

class TestAppendDurationRecord:
    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "hdc.jsonl"
        output = _make_output()
        append_duration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["best_duration_bucket"] == output.best_duration_bucket
        assert rec["duration_ev_spread"] == output.duration_ev_spread
        assert rec["schema_version"] == "v1"

    def test_buckets_present(self, tmp_path):
        p = tmp_path / "hdc.jsonl"
        output = _make_output()
        append_duration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert isinstance(rec["buckets"], list)
        assert len(rec["buckets"]) == BUCKET_COUNT

    def test_suppression_block_present(self, tmp_path):
        p = tmp_path / "hdc.jsonl"
        output = _make_output()
        append_duration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert "suppression" in rec
        assert "suppression_return_delta" in rec["suppression"]

    def test_phase_stats_present(self, tmp_path):
        p = tmp_path / "hdc.jsonl"
        output = _make_output()
        append_duration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert isinstance(rec["phase_stats"], list)

    def test_appends_without_truncating(self, tmp_path):
        p = tmp_path / "hdc.jsonl"
        append_duration_record(_make_output(), p)
        append_duration_record(_make_output(), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2


# ── TestFormatDurationReport ──────────────────────────────────────────────────

class TestFormatDurationReport:
    def test_no_populated_buckets_returns_data_shortage(self):
        output = _make_output(n_buckets_populated=0)
        rpt = format_duration_report(output)
        assert "データ不足" in rpt

    def test_contains_header(self):
        output = _make_output()
        rpt = format_duration_report(output)
        assert "HOLD DURATION CALIBRATION" in rpt

    def test_contains_kpi_section(self):
        output = _make_output()
        rpt = format_duration_report(output)
        assert "best_duration_bucket" in rpt
        assert "duration_ev_spread" in rpt
        assert "suppression_return_delta" in rpt

    def test_shows_best_bucket(self):
        output = _make_output(best_bucket="7-10d")
        rpt = format_duration_report(output)
        assert "7-10d" in rpt

    def test_shows_suppression_comparison(self):
        output = _make_output()
        rpt = format_duration_report(output)
        assert "suppressed" in rpt.lower()

    def test_shows_phase_section(self):
        output = _make_output()
        rpt = format_duration_report(output)
        assert "breakout" in rpt

    def test_returns_string(self):
        output = _make_output()
        assert isinstance(format_duration_report(output), str)

    def test_fail_open_on_bad_input(self):
        class Bad:
            n_buckets_populated = None
        rpt = format_duration_report(Bad())
        assert isinstance(rpt, str)


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_longer_holds_win(self, tmp_path):
        """End-to-end: 21d+ bucket has highest avg return."""
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "hdc.jsonl"
        recs = []
        for _ in range(5):
            recs.append(_rec(date="2026-05-01", hold_days=2,  subsequent_5d_return=0.01))
            recs.append(_rec(date="2026-05-02", hold_days=5,  subsequent_5d_return=0.03))
            recs.append(_rec(date="2026-05-03", hold_days=8,  subsequent_5d_return=0.05))
            recs.append(_rec(date="2026-05-04", hold_days=13, subsequent_5d_return=0.07))
            recs.append(_rec(date="2026-05-05", hold_days=18, subsequent_5d_return=0.09))
            recs.append(_rec(date="2026-05-06", hold_days=25, subsequent_5d_return=0.12))
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30", min_samples=3)
        assert result.best_duration_bucket == "21d+"
        assert result.worst_duration_bucket == "1-3d"
        assert result.duration_ev_spread > 0
        # JSONL written
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["best_duration_bucket"] == "21d+"

    def test_suppression_positive_delta(self, tmp_path):
        """Suppressed holds outperform → positive delta."""
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for _ in range(5):
            recs.append(_rec(hold_days=10, subsequent_5d_return=0.10, suppression_eligible=True))
            recs.append(_rec(hold_days=8,  subsequent_5d_return=0.03, suppression_eligible=False))
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30")
        assert result.suppression_return_delta > 0

    def test_phase_leader_identified(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for _ in range(5):
            recs.append(_rec(hold_days=8, subsequent_5d_return=0.08, current_phase="breakout"))
            recs.append(_rec(hold_days=5, subsequent_5d_return=0.03, current_phase="neutral"))
            recs.append(_rec(hold_days=4, subsequent_5d_return=0.04, current_phase="compression"))
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30")
        assert result.phase_duration_leader == "breakout"

    def test_report_format_complete(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for i, hd in enumerate([2, 5, 8, 13, 18, 25]):
            for _ in range(4):
                recs.append(_rec(
                    hold_days=hd,
                    subsequent_5d_return=0.01 * (i + 1),
                    suppression_eligible=(i % 2 == 0),
                    current_phase=["compression", "breakout", "neutral"][i % 3],
                ))
        _write_jsonl(recs, cp)
        result = run_hold_duration_calibration(cp, out, today="2026-05-30", min_samples=3)
        rpt = format_duration_report(result)
        assert "HOLD DURATION CALIBRATION" in rpt
        assert "best_duration_bucket" in rpt
        assert len(rpt) > 200
