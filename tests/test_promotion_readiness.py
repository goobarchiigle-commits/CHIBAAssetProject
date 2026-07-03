"""
test_promotion_readiness.py — Promotion Readiness Engine (Phase 6B-Lite)

Tests (100% coverage):
  TestSampleSizeScore / TestComputeReadinessScore / TestComputeStatus
  TestOutcomeStats / TestLatestEvidenceScore / TestLatestShadowStatus
  TestLoadEvaluatedKeys / TestLatestPerCandidate / TestComputeSummary
  TestFormatReport / TestAppendReadinessRecord / TestLoadAllJsonl
  TestRunPromotionReadiness / TestIntegration
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.promotion_readiness import (
    PROMOTION_N_COMPLETED_MIN,
    PROMOTION_SUCCESS_RATE_MIN,
    SCHEMA_VERSION,
    SCORE_PROMOTION_THRESHOLD,
    SCORE_SHADOW_CONTINUE_THRESHOLD,
    STATUS_MORE_DATA,
    STATUS_PROMOTION_READY,
    STATUS_SHADOW_CONTINUE,
    W_EVIDENCE_SCORE,
    W_SAMPLE_SIZE,
    W_SUCCESS_RATE,
    OutcomeStats,
    PromotionReadinessRecord,
    PromotionReadinessRunResult,
    PromotionReadinessSummary,
    _CANDIDATE_ORDER,
    _SAMPLE_BREAKPOINTS,
    _compute_outcome_stats,
    _compute_readiness_score,
    _compute_status,
    _compute_summary,
    _latest_evidence_score,
    _latest_per_candidate,
    _latest_shadow_status,
    _load_all_jsonl,
    _load_evaluated_keys,
    _safe_float,
    _safe_int,
    _sample_size_score,
    append_readiness_record,
    format_readiness_report,
    run_promotion_readiness,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _ep_record(date: str, candidates: list[dict]) -> dict:
    return {
        "date": date,
        "candidates": candidates,
        "n_promotable": 0,
        "schema_version": "v1",
    }


def _ep_candidate(ct: str, score: int) -> dict:
    return {
        "candidate_type": ct,
        "evidence_score": score,
        "status": "PROMOTABLE",
        "source_module": "HDC",
        "supporting_kpis": {},
    }


def _sr_record(date: str, recommendations: list[dict]) -> dict:
    return {
        "date": date,
        "recommendations": recommendations,
        "schema_version": "v1",
    }


def _sr_rec(ct: str, status: str = "SHADOW_READY") -> dict:
    return {
        "candidate_type": ct,
        "shadow_status": status,
        "shadow_duration_days": 20,
        "evidence_score": 75,
        "shadow_priority_score": 75,
    }


def _outcome_rec(candidate: str, success: bool, delta: float = 0.01) -> dict:
    return {
        "date": "2026-06-01",
        "candidate": candidate,
        "shadow_start_date": "2026-05-12",
        "shadow_duration_days": 20,
        "before_kpi": 0.1,
        "after_kpi": 0.2,
        "delta": delta,
        "success": success,
        "schema_version": "v1",
    }


def _pr_record(date: str, candidate: str, score: float, status: str,
               n: int = 0, sr: float = 0.0) -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "readiness_score": score,
        "success_rate": sr,
        "n_completed": n,
        "avg_delta": 0.0,
        "evidence_score": 75,
        "sample_size_score": 60.0,
        "shadow_status": "SHADOW_READY",
        "status": status,
        "schema_version": SCHEMA_VERSION,
    }


def _make_record(ct: str, score: float, status: str, n: int = 3, sr: float = 66.7) -> PromotionReadinessRecord:
    return PromotionReadinessRecord(
        date="2026-06-30",
        candidate=ct,
        readiness_score=score,
        success_rate=sr,
        n_completed=n,
        avg_delta=0.01,
        evidence_score=75,
        sample_size_score=60.0,
        shadow_status="SHADOW_READY",
        status=status,
    )


# ─── TestSafeHelpers ─────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_int_valid(self):
        assert _safe_int(5) == 5

    def test_safe_int_string(self):
        assert _safe_int("10") == 10

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_invalid(self):
        assert _safe_int("bad") == 0

    def test_safe_int_fallback(self):
        assert _safe_int(None, 99) == 99

    def test_safe_float_valid(self):
        assert _safe_float(1.5) == pytest.approx(1.5)

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_fallback(self):
        assert _safe_float(None, 5.5) == pytest.approx(5.5)


# ─── TestSampleSizeScore ─────────────────────────────────────────────────────

class TestSampleSizeScore:
    def test_zero(self):
        assert _sample_size_score(0) == pytest.approx(0.0)

    def test_negative(self):
        assert _sample_size_score(-5) == pytest.approx(0.0)

    def test_one(self):
        assert _sample_size_score(1) == pytest.approx(20.0)

    def test_two(self):
        assert _sample_size_score(2) == pytest.approx(40.0)

    def test_three(self):
        assert _sample_size_score(3) == pytest.approx(60.0)

    def test_four_interpolated(self):
        # between (3,60) and (5,80): t=0.5 → 70
        assert _sample_size_score(4) == pytest.approx(70.0)

    def test_five(self):
        assert _sample_size_score(5) == pytest.approx(80.0)

    def test_six_interpolated(self):
        # between (5,80) and (10,100): t=0.2 → 84
        assert _sample_size_score(6) == pytest.approx(84.0)

    def test_nine_interpolated(self):
        # t=0.8 → 80 + 0.8*20 = 96
        assert _sample_size_score(9) == pytest.approx(96.0)

    def test_ten(self):
        assert _sample_size_score(10) == pytest.approx(100.0)

    def test_large(self):
        assert _sample_size_score(100) == pytest.approx(100.0)

    def test_breakpoints_monotone(self):
        scores = [_sample_size_score(n) for n in range(0, 12)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


# ─── TestComputeReadinessScore ────────────────────────────────────────────────

class TestComputeReadinessScore:
    def test_all_zero(self):
        assert _compute_readiness_score(0.0, 0, 0) == pytest.approx(0.0)

    def test_perfect_score(self):
        # 40*1.0 + 30*(100/100) + 30*(100/100) = 100
        assert _compute_readiness_score(1.0, 100, 10) == pytest.approx(100.0)

    def test_clamped_max(self):
        assert _compute_readiness_score(1.1, 110, 100) == pytest.approx(100.0)

    def test_no_outcomes_high_evidence(self):
        # 40*0 + 30*0.75 + 30*0 = 22.5
        assert _compute_readiness_score(0.0, 75, 0) == pytest.approx(22.5)

    def test_weight_success_dominant(self):
        # success_rate=1.0, evidence=0, n=0 → 40
        assert _compute_readiness_score(1.0, 0, 0) == pytest.approx(40.0)

    def test_n3_success_half_evidence_75(self):
        # 40*0.5 + 30*0.75 + 30*0.6 = 20 + 22.5 + 18 = 60.5
        assert _compute_readiness_score(0.5, 75, 3) == pytest.approx(60.5)

    def test_n3_success_full_evidence_100(self):
        # 40*1.0 + 30*1.0 + 30*0.6 = 40+30+18 = 88.0
        assert _compute_readiness_score(1.0, 100, 3) == pytest.approx(88.0)

    def test_rounded_to_1dp(self):
        score = _compute_readiness_score(0.667, 75, 3)
        assert score == round(score, 1)


# ─── TestComputeStatus ────────────────────────────────────────────────────────

class TestComputeStatus:
    def test_promotion_ready_all_conditions_met(self):
        # score=88, n=5, sr=0.8 → PROMOTION_READY
        assert _compute_status(88.0, 5, 0.8) == STATUS_PROMOTION_READY

    def test_promotion_ready_exact_thresholds(self):
        # score=80, n=3, sr=0.60 → PROMOTION_READY
        assert _compute_status(80.0, 3, 0.60) == STATUS_PROMOTION_READY

    def test_not_ready_score_too_low(self):
        # score=79, n=5, sr=0.8 → SHADOW_CONTINUE (score>=60)
        assert _compute_status(79.9, 5, 0.8) == STATUS_SHADOW_CONTINUE

    def test_not_ready_n_too_low(self):
        # score=85, n=2, sr=0.8 → SHADOW_CONTINUE (n_completed < 3)
        assert _compute_status(85.0, 2, 0.8) == STATUS_SHADOW_CONTINUE

    def test_not_ready_success_rate_too_low(self):
        # score=85, n=5, sr=0.59 → SHADOW_CONTINUE
        assert _compute_status(85.0, 5, 0.59) == STATUS_SHADOW_CONTINUE

    def test_shadow_continue_score_60(self):
        assert _compute_status(60.0, 0, 0.0) == STATUS_SHADOW_CONTINUE

    def test_shadow_continue_score_79(self):
        assert _compute_status(79.9, 1, 0.5) == STATUS_SHADOW_CONTINUE

    def test_more_data_score_59(self):
        assert _compute_status(59.9, 0, 0.0) == STATUS_MORE_DATA

    def test_more_data_zero_score(self):
        assert _compute_status(0.0, 0, 0.0) == STATUS_MORE_DATA


# ─── TestOutcomeStats ─────────────────────────────────────────────────────────

class TestOutcomeStats:
    def test_empty_records(self):
        s = _compute_outcome_stats([], "PRIORITY")
        assert s.n_completed == 0
        assert s.success_rate == 0.0
        assert s.avg_delta == 0.0

    def test_filters_by_candidate(self):
        recs = [
            _outcome_rec("PRIORITY", True, 0.1),
            _outcome_rec("EXIT_EXTENSION", False, -0.1),
        ]
        s = _compute_outcome_stats(recs, "PRIORITY")
        assert s.n_completed == 1
        assert s.success_rate == pytest.approx(1.0)

    def test_all_success(self):
        recs = [_outcome_rec("PRIORITY", True, 0.1)] * 3
        s = _compute_outcome_stats(recs, "PRIORITY")
        assert s.n_completed == 3
        assert s.success_rate == pytest.approx(1.0)

    def test_all_failure(self):
        recs = [_outcome_rec("PRIORITY", False, -0.1)] * 4
        s = _compute_outcome_stats(recs, "PRIORITY")
        assert s.n_completed == 4
        assert s.success_rate == pytest.approx(0.0)

    def test_mixed(self):
        recs = [
            _outcome_rec("PRIORITY", True, 0.2),
            _outcome_rec("PRIORITY", False, -0.1),
            _outcome_rec("PRIORITY", True, 0.3),
        ]
        s = _compute_outcome_stats(recs, "PRIORITY")
        assert s.n_completed == 3
        assert s.success_rate == pytest.approx(2 / 3)

    def test_avg_delta(self):
        recs = [
            _outcome_rec("PRIORITY", True, 0.2),
            _outcome_rec("PRIORITY", True, 0.4),
        ]
        s = _compute_outcome_stats(recs, "PRIORITY")
        assert s.avg_delta == pytest.approx(0.3)

    def test_candidate_not_present(self):
        recs = [_outcome_rec("EXIT_EXTENSION", True, 0.1)]
        s = _compute_outcome_stats(recs, "PRIORITY")
        assert s.n_completed == 0

    def test_all_candidate_types(self):
        for ct in _CANDIDATE_ORDER:
            recs = [_outcome_rec(ct, True, 0.01)]
            s = _compute_outcome_stats(recs, ct)
            assert s.n_completed == 1


# ─── TestLatestEvidenceScore ──────────────────────────────────────────────────

class TestLatestEvidenceScore:
    def test_found(self):
        ep = _ep_record("2026-06-01", [_ep_candidate("PRIORITY", 75)])
        assert _latest_evidence_score(ep, "PRIORITY") == 75

    def test_not_found(self):
        ep = _ep_record("2026-06-01", [_ep_candidate("EXIT_EXTENSION", 80)])
        assert _latest_evidence_score(ep, "PRIORITY") == 0

    def test_empty_ep(self):
        assert _latest_evidence_score({}, "PRIORITY") == 0

    def test_no_candidates_key(self):
        assert _latest_evidence_score({"date": "2026-06-01"}, "PRIORITY") == 0

    def test_invalid_score(self):
        ep = _ep_record("2026-06-01", [{"candidate_type": "PRIORITY", "evidence_score": "bad"}])
        assert _latest_evidence_score(ep, "PRIORITY") == 0

    def test_multiple_candidates(self):
        ep = _ep_record("2026-06-01", [
            _ep_candidate("EXIT_EXTENSION", 80),
            _ep_candidate("PRIORITY", 65),
            _ep_candidate("SIGNAL_LEADER", 55),
        ])
        assert _latest_evidence_score(ep, "EXIT_EXTENSION") == 80
        assert _latest_evidence_score(ep, "PRIORITY") == 65
        assert _latest_evidence_score(ep, "SIGNAL_LEADER") == 55


# ─── TestLatestShadowStatus ───────────────────────────────────────────────────

class TestLatestShadowStatus:
    def test_found(self):
        sr = _sr_record("2026-06-01", [_sr_rec("PRIORITY", "SHADOW_READY")])
        assert _latest_shadow_status(sr, "PRIORITY") == "SHADOW_READY"

    def test_not_found(self):
        sr = _sr_record("2026-06-01", [_sr_rec("EXIT_EXTENSION", "SHADOW_READY")])
        assert _latest_shadow_status(sr, "PRIORITY") == ""

    def test_empty_sr(self):
        assert _latest_shadow_status({}, "PRIORITY") == ""

    def test_continue_observe(self):
        sr = _sr_record("2026-06-01", [_sr_rec("PRIORITY", "CONTINUE_OBSERVE")])
        assert _latest_shadow_status(sr, "PRIORITY") == "CONTINUE_OBSERVE"

    def test_wait_data(self):
        sr = _sr_record("2026-06-01", [_sr_rec("PRIORITY", "WAIT_DATA")])
        assert _latest_shadow_status(sr, "PRIORITY") == "WAIT_DATA"


# ─── TestLoadEvaluatedKeys ────────────────────────────────────────────────────

class TestLoadEvaluatedKeys:
    def test_file_not_exist(self, tmp_path):
        assert _load_evaluated_keys(tmp_path / "missing.jsonl") == set()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_evaluated_keys(f) == set()

    def test_single_record(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [_pr_record("2026-06-30", "PRIORITY", 84.0, STATUS_PROMOTION_READY)])
        keys = _load_evaluated_keys(f)
        assert ("2026-06-30", "PRIORITY") in keys

    def test_multiple_records(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [
            _pr_record("2026-06-30", "PRIORITY", 84.0, STATUS_PROMOTION_READY),
            _pr_record("2026-06-30", "EXIT_EXTENSION", 61.0, STATUS_SHADOW_CONTINUE),
        ])
        keys = _load_evaluated_keys(f)
        assert len(keys) == 2
        assert ("2026-06-30", "EXIT_EXTENSION") in keys

    def test_missing_fields_skipped(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"no_date": True}])
        assert _load_evaluated_keys(f) == set()


# ─── TestLatestPerCandidate ───────────────────────────────────────────────────

class TestLatestPerCandidate:
    def test_empty(self):
        assert _latest_per_candidate([]) == []

    def test_single(self):
        records = [_pr_record("2026-06-30", "PRIORITY", 84.0, STATUS_PROMOTION_READY)]
        result = _latest_per_candidate(records)
        assert len(result) == 1
        assert result[0].candidate == "PRIORITY"

    def test_latest_date_wins(self):
        records = [
            _pr_record("2026-06-01", "PRIORITY", 50.0, STATUS_MORE_DATA),
            _pr_record("2026-06-30", "PRIORITY", 84.0, STATUS_PROMOTION_READY),
        ]
        result = _latest_per_candidate(records)
        assert len(result) == 1
        assert result[0].readiness_score == pytest.approx(84.0)
        assert result[0].date == "2026-06-30"

    def test_ordered_by_candidate_order(self):
        records = [
            _pr_record("2026-06-30", "CONTINUATION_SIGNAL", 40.0, STATUS_MORE_DATA),
            _pr_record("2026-06-30", "EXIT_EXTENSION", 84.0, STATUS_PROMOTION_READY),
        ]
        result = _latest_per_candidate(records)
        assert result[0].candidate == "EXIT_EXTENSION"
        assert result[1].candidate == "CONTINUATION_SIGNAL"

    def test_all_five_candidates(self):
        records = [_pr_record("2026-06-30", ct, 50.0, STATUS_SHADOW_CONTINUE)
                   for ct in _CANDIDATE_ORDER]
        result = _latest_per_candidate(records)
        assert len(result) == 5
        assert [r.candidate for r in result] == _CANDIDATE_ORDER

    def test_missing_candidate_excluded(self):
        records = [_pr_record("2026-06-30", "PRIORITY", 84.0, STATUS_PROMOTION_READY)]
        result = _latest_per_candidate(records)
        assert all(r.candidate == "PRIORITY" for r in result)
        assert len(result) == 1


# ─── TestComputeSummary ───────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty(self):
        s = _compute_summary([])
        assert s.promotion_ready_count == 0
        assert s.top_ready_candidate == ""
        assert s.avg_readiness_score == 0.0

    def test_no_ready(self):
        records = [
            _make_record("PRIORITY", 65.0, STATUS_SHADOW_CONTINUE),
            _make_record("EXIT_EXTENSION", 42.0, STATUS_MORE_DATA),
        ]
        s = _compute_summary(records)
        assert s.promotion_ready_count == 0
        assert s.top_ready_candidate == ""

    def test_one_ready(self):
        records = [
            _make_record("EXIT_EXTENSION", 85.0, STATUS_PROMOTION_READY),
            _make_record("PRIORITY", 55.0, STATUS_MORE_DATA),
        ]
        s = _compute_summary(records)
        assert s.promotion_ready_count == 1
        assert s.top_ready_candidate == "EXIT_EXTENSION"

    def test_multiple_ready_top_is_highest(self):
        records = [
            _make_record("EXIT_EXTENSION", 85.0, STATUS_PROMOTION_READY),
            _make_record("PRIORITY", 92.0, STATUS_PROMOTION_READY),
            _make_record("SIGNAL_LEADER", 40.0, STATUS_MORE_DATA),
        ]
        s = _compute_summary(records)
        assert s.promotion_ready_count == 2
        assert s.top_ready_candidate == "PRIORITY"

    def test_avg_readiness_score(self):
        records = [
            _make_record("EXIT_EXTENSION", 80.0, STATUS_PROMOTION_READY),
            _make_record("PRIORITY", 60.0, STATUS_SHADOW_CONTINUE),
            _make_record("SIGNAL_LEADER", 40.0, STATUS_MORE_DATA),
        ]
        s = _compute_summary(records)
        assert s.avg_readiness_score == pytest.approx((80.0 + 60.0 + 40.0) / 3, abs=0.1)

    def test_all_ready(self):
        records = [_make_record(ct, 85.0, STATUS_PROMOTION_READY) for ct in _CANDIDATE_ORDER]
        s = _compute_summary(records)
        assert s.promotion_ready_count == 5


# ─── TestFormatReport ─────────────────────────────────────────────────────────

class TestFormatReport:
    def test_empty_returns_empty(self):
        assert format_readiness_report([]) == ""

    def test_contains_header(self):
        records = [_make_record("PRIORITY", 84.0, STATUS_PROMOTION_READY)]
        report = format_readiness_report(records)
        assert "PROMOTION READINESS" in report

    def test_contains_status(self):
        records = [_make_record("PRIORITY", 84.0, STATUS_PROMOTION_READY)]
        report = format_readiness_report(records)
        assert "PROMOTION_READY" in report

    def test_contains_score(self):
        records = [_make_record("EXIT_EXTENSION", 84.0, STATUS_PROMOTION_READY)]
        report = format_readiness_report(records)
        assert "score=84" in report

    def test_contains_candidate_name(self):
        records = [_make_record("CAPITAL_CONCENTRATION", 61.0, STATUS_SHADOW_CONTINUE)]
        report = format_readiness_report(records)
        assert "CAPITAL_CONCENTRATION" in report

    def test_counts_in_header(self):
        records = [
            _make_record("EXIT_EXTENSION", 85.0, STATUS_PROMOTION_READY),
            _make_record("PRIORITY", 65.0, STATUS_SHADOW_CONTINUE),
            _make_record("SIGNAL_LEADER", 40.0, STATUS_MORE_DATA),
        ]
        report = format_readiness_report(records)
        assert "Ready: 1" in report
        assert "Continue: 1" in report
        assert "More Data: 1" in report

    def test_separator_present(self):
        records = [_make_record("PRIORITY", 50.0, STATUS_MORE_DATA)]
        report = format_readiness_report(records)
        assert "═" in report

    def test_all_candidates_present(self):
        records = [_make_record(ct, 50.0, STATUS_MORE_DATA) for ct in _CANDIDATE_ORDER]
        report = format_readiness_report(records)
        for ct in _CANDIDATE_ORDER:
            assert ct in report


# ─── TestAppendReadinessRecord ────────────────────────────────────────────────

class TestAppendReadinessRecord:
    def _rec(self, ct: str = "PRIORITY", status: str = STATUS_PROMOTION_READY) -> PromotionReadinessRecord:
        return PromotionReadinessRecord(
            date="2026-06-30", candidate=ct, readiness_score=84.0,
            success_rate=66.7, n_completed=3, avg_delta=0.015,
            evidence_score=75, sample_size_score=60.0,
            shadow_status="SHADOW_READY", status=status,
        )

    def test_creates_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        append_readiness_record(self._rec(), f)
        assert f.exists()
        records = _load_all_jsonl(f)
        assert len(records) == 1

    def test_all_fields_present(self, tmp_path):
        f = tmp_path / "out.jsonl"
        append_readiness_record(self._rec(), f)
        r = _load_all_jsonl(f)[0]
        assert r["date"] == "2026-06-30"
        assert r["candidate"] == "PRIORITY"
        assert r["readiness_score"] == pytest.approx(84.0)
        assert r["success_rate"] == pytest.approx(66.7)
        assert r["n_completed"] == 3
        assert r["shadow_status"] == "SHADOW_READY"
        assert r["status"] == STATUS_PROMOTION_READY
        assert r["schema_version"] == SCHEMA_VERSION

    def test_appends_multiple(self, tmp_path):
        f = tmp_path / "out.jsonl"
        for ct in ["PRIORITY", "EXIT_EXTENSION"]:
            append_readiness_record(self._rec(ct), f)
        records = _load_all_jsonl(f)
        assert len(records) == 2

    def test_creates_parent_dir(self, tmp_path):
        f = tmp_path / "new_dir" / "out.jsonl"
        append_readiness_record(self._rec(), f)
        assert f.exists()


# ─── TestLoadAllJsonl ─────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file(self, tmp_path):
        assert _load_all_jsonl(tmp_path / "missing.jsonl") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_all_jsonl(f) == []

    def test_valid(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"a": 1}, {"b": 2}])
        assert len(_load_all_jsonl(f)) == 2

    def test_skips_invalid_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\nBAD\n{"b":2}\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 2

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 2


# ─── TestRunPromotionReadiness ────────────────────────────────────────────────

TODAY = "2026-06-30"


class TestRunPromotionReadiness:
    def _setup(self, tmp_path):
        return (
            tmp_path / "ep.jsonl",
            tmp_path / "sr.jsonl",
            tmp_path / "outcome.jsonl",
            tmp_path / "out.jsonl",
        )

    def test_empty_all_files(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        # 5 candidates evaluated, all with n_completed=0 → MORE_DATA or low score
        assert len(result.new_records) == 5

    def test_all_candidates_evaluated(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        candidates = {r.candidate for r in result.new_records}
        assert candidates == set(_CANDIDATE_ORDER)

    def test_idempotent_second_run(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        r1 = run_promotion_readiness(ep, sr, oc, out, TODAY)
        r2 = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert len(r1.new_records) == 5
        assert len(r2.new_records) == 0
        assert len(_load_all_jsonl(out)) == 5

    def test_different_day_not_skipped(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        run_promotion_readiness(ep, sr, oc, out, "2026-06-29")
        r2 = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert len(r2.new_records) == 5

    def test_evidence_score_from_ep(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(ep, [_ep_record("2026-06-30", [_ep_candidate("PRIORITY", 80)])])
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        priority = next(r for r in result.new_records if r.candidate == "PRIORITY")
        assert priority.evidence_score == 80

    def test_shadow_status_from_sr(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(sr, [_sr_record("2026-06-30", [_sr_rec("EXIT_EXTENSION", "SHADOW_READY")])])
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        ext = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ext.shadow_status == "SHADOW_READY"

    def test_promotion_ready_when_conditions_met(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        # n=5, all success → success_rate=1.0, evidence=90 → high score
        _write_jsonl(ep, [_ep_record("2026-06-30", [_ep_candidate("PRIORITY", 90)])])
        _write_jsonl(oc, [_outcome_rec("PRIORITY", True, 0.1)] * 5)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        priority = next(r for r in result.new_records if r.candidate == "PRIORITY")
        assert priority.status == STATUS_PROMOTION_READY
        assert priority.n_completed == 5
        assert priority.success_rate == pytest.approx(100.0)

    def test_more_data_when_no_outcomes(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        # No evidence, no outcomes → very low score → MORE_DATA
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        for r in result.new_records:
            assert r.status == STATUS_MORE_DATA

    def test_shadow_continue_moderate_score(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        # evidence=75, n=2, sr=0.5 → 40*0.5 + 30*0.75 + 30*(40/100) = 20+22.5+12 = 54.5 → MORE_DATA
        # Let me use n=5, sr=0.5, evidence=60 → 40*0.5 + 30*0.6 + 30*(80/100) = 20+18+24 = 62 → SHADOW_CONTINUE
        _write_jsonl(ep, [_ep_record("2026-06-30", [_ep_candidate("SIGNAL_LEADER", 60)])])
        outcome_recs = (
            [_outcome_rec("SIGNAL_LEADER", True, 0.01)] * 3 +
            [_outcome_rec("SIGNAL_LEADER", False, -0.01)] * 3
        )
        _write_jsonl(oc, outcome_recs)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        sig = next(r for r in result.new_records if r.candidate == "SIGNAL_LEADER")
        assert sig.status in (STATUS_SHADOW_CONTINUE, STATUS_MORE_DATA)

    def test_success_rate_stored_as_percentage(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(oc, [
            _outcome_rec("PRIORITY", True, 0.1),
            _outcome_rec("PRIORITY", False, -0.1),
        ])
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        priority = next(r for r in result.new_records if r.candidate == "PRIORITY")
        # success_rate should be 50.0 (percentage)
        assert priority.success_rate == pytest.approx(50.0)

    def test_summary_promotion_ready_count(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(ep, [_ep_record("2026-06-30", [_ep_candidate("PRIORITY", 100)])])
        _write_jsonl(oc, [_outcome_rec("PRIORITY", True, 0.1)] * 10)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert result.summary.promotion_ready_count >= 1

    def test_summary_avg_readiness_score(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert 0.0 <= result.summary.avg_readiness_score <= 100.0

    def test_snapshot_in_result(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert len(result.snapshot) == 5

    def test_snapshot_ordered(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        candidates = [r.candidate for r in result.snapshot]
        assert candidates == _CANDIDATE_ORDER

    def test_fail_open_on_exception(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        # Provide invalid files that may cause issues
        ep.write_text("NOT JSON\n", encoding="utf-8")
        sr.write_text("NOT JSON\n", encoding="utf-8")
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert isinstance(result, PromotionReadinessRunResult)

    def test_schema_version_in_output(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        run_promotion_readiness(ep, sr, oc, out, TODAY)
        records = _load_all_jsonl(out)
        for r in records:
            assert r["schema_version"] == SCHEMA_VERSION

    def test_multiple_ep_records_uses_latest(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(ep, [
            _ep_record("2026-06-01", [_ep_candidate("PRIORITY", 40)]),
            _ep_record("2026-06-30", [_ep_candidate("PRIORITY", 90)]),
        ])
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        priority = next(r for r in result.new_records if r.candidate == "PRIORITY")
        assert priority.evidence_score == 90

    def test_multiple_outcome_records_accumulated(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        recs = [_outcome_rec("EXIT_EXTENSION", True, 0.02)] * 3
        recs[0]["date"] = "2026-06-01"
        recs[1]["date"] = "2026-06-10"
        recs[2]["date"] = "2026-06-20"
        _write_jsonl(oc, recs)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        ext = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ext.n_completed == 3

    def test_no_outcomes_gives_zero_sr(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        for r in result.new_records:
            assert r.n_completed == 0
            assert r.success_rate == pytest.approx(0.0)

    def test_sample_size_score_in_output(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(oc, [_outcome_rec("PRIORITY", True, 0.1)] * 3)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        priority = next(r for r in result.new_records if r.candidate == "PRIORITY")
        assert priority.sample_size_score == pytest.approx(60.0)

    def test_top_ready_candidate_in_summary(self, tmp_path):
        ep, sr, oc, out = self._setup(tmp_path)
        _write_jsonl(ep, [_ep_record("2026-06-30", [_ep_candidate("PRIORITY", 100)])])
        _write_jsonl(oc, [_outcome_rec("PRIORITY", True, 0.1)] * 10)
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        if result.summary.promotion_ready_count > 0:
            assert result.summary.top_ready_candidate != ""


# ─── TestIntegration ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_cycle_promotion_ready(self, tmp_path):
        ep  = tmp_path / "ep.jsonl"
        sr  = tmp_path / "sr.jsonl"
        oc  = tmp_path / "outcome.jsonl"
        out = tmp_path / "out.jsonl"

        # High evidence for all candidates
        _write_jsonl(ep, [_ep_record("2026-06-30", [_ep_candidate(ct, 90) for ct in _CANDIDATE_ORDER])])
        # 10 successful outcomes per candidate
        outcome_recs = [_outcome_rec(ct, True, 0.02) for ct in _CANDIDATE_ORDER for _ in range(10)]
        _write_jsonl(oc, outcome_recs)

        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert len(result.new_records) == 5
        # All should be PROMOTION_READY with 10 outcomes, 100% SR, 90 evidence
        for r in result.new_records:
            assert r.status == STATUS_PROMOTION_READY
        assert result.summary.promotion_ready_count == 5

    def test_report_format_integrated(self, tmp_path):
        ep  = tmp_path / "ep.jsonl"
        sr  = tmp_path / "sr.jsonl"
        oc  = tmp_path / "outcome.jsonl"
        out = tmp_path / "out.jsonl"
        result = run_promotion_readiness(ep, sr, oc, out, TODAY)
        report = format_readiness_report(result.snapshot)
        assert "PROMOTION READINESS" in report
        assert "More Data: 5" in report

    def test_idempotent_output_file_size(self, tmp_path):
        ep  = tmp_path / "ep.jsonl"
        sr  = tmp_path / "sr.jsonl"
        oc  = tmp_path / "outcome.jsonl"
        out = tmp_path / "out.jsonl"
        run_promotion_readiness(ep, sr, oc, out, TODAY)
        run_promotion_readiness(ep, sr, oc, out, TODAY)
        run_promotion_readiness(ep, sr, oc, out, TODAY)
        assert len(_load_all_jsonl(out)) == 5

    def test_candidate_order_constant(self):
        assert _CANDIDATE_ORDER == [
            "EXIT_EXTENSION",
            "CAPITAL_CONCENTRATION",
            "PRIORITY",
            "SIGNAL_LEADER",
            "CONTINUATION_SIGNAL",
        ]

    def test_thresholds_consistent(self):
        assert SCORE_PROMOTION_THRESHOLD == 80.0
        assert SCORE_SHADOW_CONTINUE_THRESHOLD == 60.0
        assert PROMOTION_N_COMPLETED_MIN == 3
        assert PROMOTION_SUCCESS_RATE_MIN == pytest.approx(0.60)

    def test_weights_sum_to_100(self):
        assert W_SUCCESS_RATE + W_EVIDENCE_SCORE + W_SAMPLE_SIZE == pytest.approx(100.0)

    def test_sample_breakpoints_monotone(self):
        ns  = [bp[0] for bp in _SAMPLE_BREAKPOINTS]
        scs = [bp[1] for bp in _SAMPLE_BREAKPOINTS]
        for i in range(len(ns) - 1):
            assert ns[i] < ns[i + 1]
            assert scs[i] <= scs[i + 1]
