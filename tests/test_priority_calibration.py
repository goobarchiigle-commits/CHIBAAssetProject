"""
tests/test_priority_calibration.py

Tests for src/analytics/priority_calibration.py
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from src.analytics.priority_calibration import (
    BUCKET_COUNT,
    BUCKET_EDGES,
    BUCKET_LABELS,
    MIN_BUCKET_SAMPLES,
    PriorityBucket,
    CalibrationOutput,
    _safe,
    _load_jsonl,
    _append_jsonl,
    _bucket_index,
    _rank,
    load_and_filter_cp,
    compute_buckets,
    compute_monotonicity,
    compute_calibration_error,
    run_priority_calibration,
    append_calibration_record,
    format_calibration_report,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _rec(
    date="2026-05-01",
    symbol="1234",
    priority_score=60.0,
    subsequent_5d_return=0.05,
    subsequent_10d_return=None,
    **kwargs,
):
    rec = {
        "date": date,
        "symbol": symbol,
        "priority_score": priority_score,
        "subsequent_5d_return": subsequent_5d_return,
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


def _make_output(
    n_records_total=20,
    n_materialized=15,
    n_buckets_populated=5,
    priority_monotonicity=0.8,
    priority_calibration_error=0.2,
    top_bucket="70-80",
    worst_bucket="10-20",
):
    buckets = [
        PriorityBucket(
            label=BUCKET_LABELS[i],
            low=float(BUCKET_EDGES[i]),
            high=float(BUCKET_EDGES[i + 1]),
            midpoint=float(BUCKET_EDGES[i] + 5),
            n=3 if i in (1, 3, 5, 7, 9) else 0,
            avg_5d_return=0.01 * (i + 1) if i in (1, 3, 5, 7, 9) else 0.0,
            avg_10d_return=None,
            win_rate_5d=0.5 + 0.05 * i if i in (1, 3, 5, 7, 9) else 0.0,
            win_rate_10d=None,
        )
        for i in range(BUCKET_COUNT)
    ]
    return CalibrationOutput(
        date="2026-05-30",
        n_records_total=n_records_total,
        n_materialized=n_materialized,
        lookback_days=90,
        min_bucket_samples=MIN_BUCKET_SAMPLES,
        buckets=buckets,
        n_buckets_populated=n_buckets_populated,
        priority_monotonicity=priority_monotonicity,
        priority_calibration_error=priority_calibration_error,
        top_bucket=top_bucket,
        worst_bucket=worst_bucket,
    )


# ── TestSafe ─────────────────────────────────────────────────────────────────

class TestSafe:
    def test_none_fallback(self):
        assert _safe(None, 0.0) == 0.0

    def test_none_none_fallback(self):
        assert _safe(None, None) is None

    def test_finite_float(self):
        assert _safe(3.14, 0.0) == pytest.approx(3.14)

    def test_nan_returns_fallback(self):
        assert _safe(float("nan"), 0.0) == pytest.approx(0.0)

    def test_inf_returns_fallback(self):
        assert _safe(float("inf"), 0.0) == pytest.approx(0.0)

    def test_string_number(self):
        assert _safe("7.5", 0.0) == pytest.approx(7.5)

    def test_invalid_string(self):
        assert _safe("xyz", 0.0) == pytest.approx(0.0)


# ── TestLoadJsonl ─────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_missing_file_empty(self, tmp_path):
        assert _load_jsonl(tmp_path / "x.jsonl") == []

    def test_loads_records(self, tmp_path):
        p = tmp_path / "t.jsonl"
        _write_jsonl([{"a": 1}, {"b": 2}], p)
        assert len(_load_jsonl(p)) == 2

    def test_skips_blank(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        assert len(_load_jsonl(p)) == 2

    def test_skips_invalid_json(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"a":1}\nnot json\n{"b":2}\n', encoding="utf-8")
        assert len(_load_jsonl(p)) == 2


# ── TestAppendJsonl ───────────────────────────────────────────────────────────

class TestAppendJsonl:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _append_jsonl({"x": 1}, p)
        assert p.exists()

    def test_appends_two_records(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _append_jsonl({"x": 1}, p)
        _append_jsonl({"x": 2}, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_atomic_ten_appends(self, tmp_path):
        p = tmp_path / "out.jsonl"
        for i in range(10):
            _append_jsonl({"i": i}, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 10


# ── TestBucketIndex ───────────────────────────────────────────────────────────

class TestBucketIndex:
    def test_zero(self):
        assert _bucket_index(0.0) == 0

    def test_five(self):
        assert _bucket_index(5.0) == 0

    def test_ten(self):
        assert _bucket_index(10.0) == 1

    def test_fifty(self):
        assert _bucket_index(50.0) == 5

    def test_ninety(self):
        assert _bucket_index(90.0) == 9

    def test_hundred_clamps_to_last(self):
        assert _bucket_index(100.0) == 9

    def test_over_hundred_clamps(self):
        assert _bucket_index(105.0) == 9

    def test_negative_clamps_to_zero(self):
        assert _bucket_index(-5.0) == 0

    def test_all_edges_correct(self):
        # Each edge value goes to expected bucket
        for i, edge in enumerate(BUCKET_EDGES[:-1]):
            assert _bucket_index(float(edge)) == i


# ── TestRank ──────────────────────────────────────────────────────────────────

class TestRank:
    def test_ascending_gives_natural_ranks(self):
        assert _rank([1.0, 2.0, 3.0]) == [1, 2, 3]

    def test_descending_gives_reversed_ranks(self):
        assert _rank([3.0, 2.0, 1.0]) == [3, 2, 1]

    def test_single_element(self):
        assert _rank([5.0]) == [1]

    def test_ties_stable(self):
        ranks = _rank([1.0, 1.0])
        assert sorted(ranks) == [1, 2]


# ── TestLoadAndFilterCp ───────────────────────────────────────────────────────

class TestLoadAndFilterCp:
    def test_empty_for_missing_file(self, tmp_path):
        within, mat = load_and_filter_cp(tmp_path / "x.jsonl", 90, "2026-05-30")
        assert within == [] and mat == []

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
            _rec(date="2026-05-01", symbol="B", subsequent_5d_return=0.03),
        ], p)
        _, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(mat) == 1 and mat[0]["symbol"] == "B"

    def test_materialized_needs_priority_score(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        _write_jsonl([
            {"date": "2026-05-01", "subsequent_5d_return": 0.05},
            _rec(date="2026-05-01", symbol="B", subsequent_5d_return=0.05),
        ], p)
        _, mat = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(mat) == 1 and mat[0]["symbol"] == "B"

    def test_skips_bad_date(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        _write_jsonl([
            {"date": "bad", "priority_score": 50, "subsequent_5d_return": 0.05},
            _rec(date="2026-05-01"),
        ], p)
        within, _ = load_and_filter_cp(p, 90, "2026-05-30")
        assert len(within) == 1


# ── TestComputeBuckets ────────────────────────────────────────────────────────

class TestComputeBuckets:
    def test_returns_ten_buckets(self):
        buckets = compute_buckets([])
        assert len(buckets) == 10

    def test_empty_input_all_zero(self):
        buckets = compute_buckets([])
        assert all(b.n == 0 for b in buckets)

    def test_single_record_correct_bucket(self):
        recs = [_rec(priority_score=55.0, subsequent_5d_return=0.10)]
        buckets = compute_buckets(recs)
        assert buckets[5].n == 1         # 50-60
        assert buckets[5].avg_5d_return == pytest.approx(0.10)

    def test_hundred_goes_to_last_bucket(self):
        recs = [_rec(priority_score=100.0, subsequent_5d_return=0.05)]
        buckets = compute_buckets(recs)
        assert buckets[9].n == 1

    def test_avg_computed_correctly(self):
        recs = [
            _rec(priority_score=10.0, subsequent_5d_return=0.04),
            _rec(priority_score=15.0, subsequent_5d_return=0.06),
        ]
        buckets = compute_buckets(recs)
        assert buckets[1].n == 2
        assert buckets[1].avg_5d_return == pytest.approx(0.05)

    def test_win_rate_5d(self):
        recs = [
            _rec(priority_score=20.0, subsequent_5d_return=0.05),
            _rec(priority_score=25.0, subsequent_5d_return=-0.02),
            _rec(priority_score=28.0, subsequent_5d_return=0.01),
        ]
        buckets = compute_buckets(recs)
        assert buckets[2].n == 3
        assert buckets[2].win_rate_5d == pytest.approx(2 / 3, abs=1e-4)

    def test_10d_return_optional(self):
        recs = [
            _rec(priority_score=30.0, subsequent_5d_return=0.05,
                 subsequent_10d_return=0.08),
        ]
        buckets = compute_buckets(recs)
        assert buckets[3].avg_10d_return == pytest.approx(0.08)
        assert buckets[3].win_rate_10d == pytest.approx(1.0)

    def test_10d_none_when_absent(self):
        recs = [_rec(priority_score=40.0, subsequent_5d_return=0.05)]
        buckets = compute_buckets(recs)
        assert buckets[4].avg_10d_return is None
        assert buckets[4].win_rate_10d is None

    def test_bucket_labels_correct(self):
        buckets = compute_buckets([])
        assert buckets[0].label == "0-10"
        assert buckets[9].label == "90-100"

    def test_bucket_midpoints(self):
        buckets = compute_buckets([])
        assert buckets[0].midpoint == pytest.approx(5.0)
        assert buckets[9].midpoint == pytest.approx(95.0)

    def test_multiple_buckets_populated(self):
        recs = [
            _rec(priority_score=5.0,  subsequent_5d_return=0.01),
            _rec(priority_score=55.0, subsequent_5d_return=0.05),
            _rec(priority_score=95.0, subsequent_5d_return=0.09),
        ]
        buckets = compute_buckets(recs)
        assert buckets[0].n == 1
        assert buckets[5].n == 1
        assert buckets[9].n == 1


# ── TestComputeMonotonicity ───────────────────────────────────────────────────

class TestComputeMonotonicity:
    def _make_buckets(self, n_and_avg5: list) -> list:
        """Helper: build synthetic bucket list for monotonicity tests."""
        result = []
        for i, (n, avg5) in enumerate(n_and_avg5):
            result.append(PriorityBucket(
                label=BUCKET_LABELS[i] if i < BUCKET_COUNT else f"{i*10}-{i*10+10}",
                low=float(i * 10),
                high=float(i * 10 + 10),
                midpoint=float(i * 10 + 5),
                n=n,
                avg_5d_return=avg5,
                avg_10d_return=None,
                win_rate_5d=0.5,
                win_rate_10d=None,
            ))
        return result

    def test_perfect_monotone_returns_one(self):
        # 5 buckets with perfectly increasing returns
        buckets = self._make_buckets([
            (5, 0.01), (5, 0.02), (5, 0.03), (5, 0.04), (5, 0.05),
            (0, 0.0),  (0, 0.0),  (0, 0.0),  (0, 0.0),  (0, 0.0),
        ])
        rho = compute_monotonicity(buckets, 3)
        assert rho == pytest.approx(1.0)

    def test_perfectly_reversed_returns_minus_one(self):
        buckets = self._make_buckets([
            (5, 0.05), (5, 0.04), (5, 0.03), (5, 0.02), (5, 0.01),
            (0, 0.0),  (0, 0.0),  (0, 0.0),  (0, 0.0),  (0, 0.0),
        ])
        rho = compute_monotonicity(buckets, 3)
        assert rho == pytest.approx(-1.0)

    def test_fewer_than_two_returns_zero(self):
        buckets = self._make_buckets([(5, 0.05)] + [(0, 0.0)] * 9)
        assert compute_monotonicity(buckets, 3) == 0.0

    def test_zero_populated_returns_zero(self):
        buckets = self._make_buckets([(0, 0.0)] * 10)
        assert compute_monotonicity(buckets, 3) == 0.0

    def test_min_samples_gate(self):
        # Two buckets but both below min_samples
        buckets = self._make_buckets([
            (2, 0.01), (2, 0.05),
        ] + [(0, 0.0)] * 8)
        # min_samples=3 → neither qualifies → return 0.0
        assert compute_monotonicity(buckets, 3) == 0.0

    def test_partial_monotone_between_neg_and_pos(self):
        # 3 populated buckets: ascending → ρ=1.0
        buckets = self._make_buckets([
            (5, 0.01), (0, 0.0), (5, 0.03), (0, 0.0), (5, 0.05),
            (0, 0.0),  (0, 0.0), (0, 0.0),  (0, 0.0), (0, 0.0),
        ])
        rho = compute_monotonicity(buckets, 3)
        assert rho == pytest.approx(1.0)

    def test_result_in_range(self):
        import random
        random.seed(42)
        buckets = self._make_buckets([
            (5, random.uniform(-0.1, 0.1)) for _ in range(10)
        ])
        rho = compute_monotonicity(buckets, 3)
        assert -1.0 <= rho <= 1.0


# ── TestComputeCalibrationError ───────────────────────────────────────────────

class TestComputeCalibrationError:
    def _make_buckets(self, n_and_avg5: list) -> list:
        result = []
        for i, (n, avg5) in enumerate(n_and_avg5):
            result.append(PriorityBucket(
                label=BUCKET_LABELS[i] if i < BUCKET_COUNT else f"{i*10}-{i*10+10}",
                low=float(i * 10),
                high=float(i * 10 + 10),
                midpoint=float(i * 10 + 5),
                n=n,
                avg_5d_return=avg5,
                avg_10d_return=None,
                win_rate_5d=0.5,
                win_rate_10d=None,
            ))
        return result

    def test_perfect_monotone_error_zero(self):
        buckets = self._make_buckets([
            (5, 0.01), (5, 0.02), (5, 0.03),
        ] + [(0, 0.0)] * 7)
        assert compute_calibration_error(buckets, 3) == pytest.approx(0.0)

    def test_fully_reversed_error_one(self):
        buckets = self._make_buckets([
            (5, 0.03), (5, 0.02), (5, 0.01),
        ] + [(0, 0.0)] * 7)
        assert compute_calibration_error(buckets, 3) == pytest.approx(1.0)

    def test_half_violations(self):
        # 3 buckets: pair (0,1) ok, pair (1,2) violated
        buckets = self._make_buckets([
            (5, 0.01), (5, 0.03), (5, 0.02),
        ] + [(0, 0.0)] * 7)
        assert compute_calibration_error(buckets, 3) == pytest.approx(0.5)

    def test_fewer_than_two_returns_zero(self):
        buckets = self._make_buckets([(5, 0.05)] + [(0, 0.0)] * 9)
        assert compute_calibration_error(buckets, 3) == 0.0

    def test_no_populated_returns_zero(self):
        buckets = self._make_buckets([(0, 0.0)] * 10)
        assert compute_calibration_error(buckets, 3) == 0.0

    def test_min_samples_gate(self):
        buckets = self._make_buckets([
            (2, 0.03), (2, 0.01),
        ] + [(0, 0.0)] * 8)
        assert compute_calibration_error(buckets, 3) == 0.0


# ── TestRunPriorityCalibration ────────────────────────────────────────────────

class TestRunPriorityCalibration:
    def test_missing_file_returns_empty(self, tmp_path):
        result = run_priority_calibration(
            cp_file=tmp_path / "x.jsonl",
            output_file=tmp_path / "out.jsonl",
            today="2026-05-30",
        )
        assert result.n_records_total == 0

    def test_writes_jsonl(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "pc.jsonl"
        recs = [_rec(priority_score=float(50 + i * 5), subsequent_5d_return=0.01 * i)
                for i in range(10)]
        _write_jsonl(recs, cp)
        run_priority_calibration(cp, out, today="2026-05-30")
        assert out.exists()
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_appends_on_second_call(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "pc.jsonl"
        recs = [_rec(priority_score=60.0, subsequent_5d_return=0.05) for _ in range(5)]
        _write_jsonl(recs, cp)
        run_priority_calibration(cp, out, today="2026-05-30")
        run_priority_calibration(cp, out, today="2026-05-31")
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_never_raises_on_corrupt_file(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        cp.write_text("not json\n{bad}\n", encoding="utf-8")
        result = run_priority_calibration(cp, out, today="2026-05-30")
        assert result is not None

    def test_monotonicity_positive_for_correlated_data(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        # Perfect correlation: higher priority → higher return
        recs = []
        for i in range(10):
            ps = float(5 + i * 10)        # 5, 15, 25, ..., 95
            for _ in range(5):
                recs.append(_rec(
                    date="2026-05-01",
                    priority_score=ps,
                    subsequent_5d_return=0.005 * (i + 1),
                ))
        _write_jsonl(recs, cp)
        result = run_priority_calibration(cp, out, today="2026-05-30")
        assert result.priority_monotonicity > 0.5

    def test_calibration_error_zero_for_perfect_data(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for i in range(10):
            ps = float(5 + i * 10)
            for _ in range(5):
                recs.append(_rec(priority_score=ps, subsequent_5d_return=0.01 * (i + 1)))
        _write_jsonl(recs, cp)
        result = run_priority_calibration(cp, out, today="2026-05-30")
        assert result.priority_calibration_error == pytest.approx(0.0)


# ── TestAppendCalibrationRecord ───────────────────────────────────────────────

class TestAppendCalibrationRecord:
    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "pc.jsonl"
        output = _make_output()
        append_calibration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["priority_monotonicity"] == output.priority_monotonicity
        assert rec["priority_calibration_error"] == output.priority_calibration_error
        assert "buckets" in rec
        assert rec["schema_version"] == "v1"

    def test_buckets_list_present(self, tmp_path):
        p = tmp_path / "pc.jsonl"
        output = _make_output()
        append_calibration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert isinstance(rec["buckets"], list)
        assert len(rec["buckets"]) == BUCKET_COUNT

    def test_bucket_fields_complete(self, tmp_path):
        p = tmp_path / "pc.jsonl"
        output = _make_output()
        append_calibration_record(output, p)
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        b = rec["buckets"][0]
        for field in ("label", "low", "high", "midpoint", "n", "avg_5d_return", "win_rate_5d"):
            assert field in b

    def test_appends_without_truncating(self, tmp_path):
        p = tmp_path / "pc.jsonl"
        append_calibration_record(_make_output(), p)
        append_calibration_record(_make_output(), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2


# ── TestFormatCalibrationReport ───────────────────────────────────────────────

class TestFormatCalibrationReport:
    def test_no_populated_buckets_returns_data_shortage(self):
        output = _make_output(n_buckets_populated=0)
        rpt = format_calibration_report(output)
        assert "データ不足" in rpt

    def test_contains_kpi_labels(self):
        output = _make_output()
        rpt = format_calibration_report(output)
        assert "priority_monotonicity" in rpt
        assert "priority_calibration_error" in rpt

    def test_contains_top_and_worst_bucket(self):
        output = _make_output(top_bucket="70-80", worst_bucket="10-20")
        rpt = format_calibration_report(output)
        assert "70-80" in rpt
        assert "10-20" in rpt

    def test_strong_monotonicity_verdict(self):
        output = _make_output(priority_monotonicity=0.9, priority_calibration_error=0.1)
        rpt = format_calibration_report(output)
        assert "強い単調性" in rpt

    def test_negative_monotonicity_verdict(self):
        output = _make_output(priority_monotonicity=-0.3, priority_calibration_error=0.8)
        rpt = format_calibration_report(output)
        assert "単調性なし" in rpt or "逆相関" in rpt

    def test_contains_header(self):
        output = _make_output()
        rpt = format_calibration_report(output)
        assert "PRIORITY CALIBRATION" in rpt

    def test_returns_string(self):
        output = _make_output()
        assert isinstance(format_calibration_report(output), str)

    def test_fail_open_on_bad_input(self):
        class Bad:
            n_buckets_populated = None
        rpt = format_calibration_report(Bad())
        assert isinstance(rpt, str)

    def test_medium_monotonicity_verdict(self):
        output = _make_output(priority_monotonicity=0.5, priority_calibration_error=0.3)
        rpt = format_calibration_report(output)
        assert "中程度" in rpt

    def test_weak_monotonicity_verdict(self):
        output = _make_output(priority_monotonicity=0.2, priority_calibration_error=0.5)
        rpt = format_calibration_report(output)
        assert "弱い" in rpt


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_perfect_correlation(self, tmp_path):
        """End-to-end: perfectly correlated data → monotonicity=1, error=0."""
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "pc.jsonl"
        recs = []
        for i in range(10):
            ps = float(5 + i * 10)
            for _ in range(5):
                recs.append(_rec(
                    date=f"2026-05-{i+1:02d}",
                    priority_score=ps,
                    subsequent_5d_return=0.005 * (i + 1),
                ))
        _write_jsonl(recs, cp)
        result = run_priority_calibration(cp, out, today="2026-05-30", min_samples=3)
        assert result.priority_monotonicity == pytest.approx(1.0)
        assert result.priority_calibration_error == pytest.approx(0.0)
        assert result.top_bucket == "90-100"
        # JSONL written
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["priority_monotonicity"] == pytest.approx(1.0)

    def test_full_pipeline_inverted_correlation(self, tmp_path):
        """Inverted data → negative monotonicity, high calibration error."""
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for i in range(10):
            ps = float(5 + i * 10)
            for _ in range(5):
                recs.append(_rec(
                    priority_score=ps,
                    subsequent_5d_return=0.005 * (10 - i),  # high ps → low return
                ))
        _write_jsonl(recs, cp)
        result = run_priority_calibration(cp, out, today="2026-05-30", min_samples=3)
        assert result.priority_monotonicity < 0
        assert result.priority_calibration_error > 0.5

    def test_report_for_perfect_data(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = []
        for i in range(5):
            ps = float(10 + i * 15)
            for _ in range(5):
                recs.append(_rec(priority_score=ps, subsequent_5d_return=0.01 * (i + 1)))
        _write_jsonl(recs, cp)
        result = run_priority_calibration(cp, out, today="2026-05-30", min_samples=3)
        rpt = format_calibration_report(result)
        assert "PRIORITY CALIBRATION" in rpt
        assert "priority_monotonicity" in rpt
        assert len(rpt) > 100

    def test_10d_return_included_when_present(self, tmp_path):
        cp = tmp_path / "cp.jsonl"
        out = tmp_path / "out.jsonl"
        recs = [
            _rec(priority_score=55.0, subsequent_5d_return=0.05,
                 subsequent_10d_return=0.09),
            _rec(priority_score=56.0, subsequent_5d_return=0.04,
                 subsequent_10d_return=0.07),
            _rec(priority_score=57.0, subsequent_5d_return=0.06,
                 subsequent_10d_return=0.10),
        ]
        _write_jsonl(recs, cp)
        result = run_priority_calibration(cp, out, today="2026-05-30", min_samples=3)
        bucket_50_60 = result.buckets[5]
        assert bucket_50_60.avg_10d_return == pytest.approx(
            (0.09 + 0.07 + 0.10) / 3, abs=1e-5
        )
