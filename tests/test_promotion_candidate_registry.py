"""
test_promotion_candidate_registry.py — Promotion Candidate Registry (Phase 6C-Lite)

Tests (100% coverage):
  TestSafeHelpers / TestLoadAllJsonl / TestLoadActiveSet
  TestLoadActiveRegistry / TestCutoffDate / TestComputeSummary
  TestFormatReport / TestAppendRegistryRecord / TestLatestPrPerCandidate
  TestRunPromotionCandidateRegistry / TestIntegration
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.promotion_candidate_registry import (
    DEFAULT_LOOKBACK_DAYS,
    REGISTRY_STATUS_ACTIVE,
    SCHEMA_VERSION,
    RegistryRecord,
    RegistryRunResult,
    RegistrySummary,
    _CANDIDATE_ORDER,
    _STATUS_PROMOTION_READY,
    _compute_summary,
    _cutoff_date,
    _latest_pr_per_candidate,
    _load_active_registry,
    _load_active_set,
    _load_all_jsonl,
    _safe_float,
    _safe_int,
    append_registry_record,
    format_registry_report,
    run_promotion_candidate_registry,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _pr_rec(candidate: str, status: str, score: float = 84.0,
            sr: float = 66.7, n: int = 3, date: str = "2026-07-01") -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "readiness_score": score,
        "success_rate": sr,
        "n_completed": n,
        "status": status,
        "schema_version": "v1",
    }


def _reg_rec(candidate: str, date: str = "2026-07-05",
             score: float = 84.0, sr: float = 66.7, n: int = 3,
             status: str = REGISTRY_STATUS_ACTIVE) -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "readiness_score": score,
        "success_rate": sr,
        "n_completed": n,
        "registry_status": status,
        "schema_version": SCHEMA_VERSION,
    }


def _make_active_record(ct: str, date: str = "2026-07-05",
                        score: float = 84.0) -> RegistryRecord:
    return RegistryRecord(
        date=date, candidate=ct, readiness_score=score,
        success_rate=66.7, n_completed=3,
        registry_status=REGISTRY_STATUS_ACTIVE,
    )


TODAY = "2026-07-05"


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

    def test_safe_float_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)


# ─── TestLoadAllJsonl ─────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file(self, tmp_path):
        assert _load_all_jsonl(tmp_path / "missing.jsonl") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_all_jsonl(f) == []

    def test_valid_records(self, tmp_path):
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


# ─── TestLoadActiveSet ────────────────────────────────────────────────────────

class TestLoadActiveSet:
    def test_missing_file(self, tmp_path):
        assert _load_active_set(tmp_path / "missing.jsonl") == set()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_active_set(f) == set()

    def test_single_active(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("EXIT_EXTENSION")])
        assert "EXIT_EXTENSION" in _load_active_set(f)

    def test_non_active_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("EXIT_EXTENSION", status="INACTIVE")])
        assert _load_active_set(f) == set()

    def test_multiple_candidates(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [
            _reg_rec("EXIT_EXTENSION"),
            _reg_rec("PRIORITY"),
        ])
        s = _load_active_set(f)
        assert "EXIT_EXTENSION" in s
        assert "PRIORITY" in s
        assert len(s) == 2

    def test_duplicate_same_candidate(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [
            _reg_rec("EXIT_EXTENSION", date="2026-07-01"),
            _reg_rec("EXIT_EXTENSION", date="2026-07-05"),
        ])
        s = _load_active_set(f)
        assert len(s) == 1
        assert "EXIT_EXTENSION" in s

    def test_blank_candidate_skipped(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [{"registry_status": REGISTRY_STATUS_ACTIVE, "candidate": ""}])
        assert _load_active_set(f) == set()

    def test_missing_status_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [{"candidate": "PRIORITY"}])
        assert _load_active_set(f) == set()


# ─── TestLoadActiveRegistry ───────────────────────────────────────────────────

class TestLoadActiveRegistry:
    def test_empty_file(self, tmp_path):
        assert _load_active_registry(tmp_path / "missing.jsonl") == []

    def test_single_record(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("EXIT_EXTENSION")])
        result = _load_active_registry(f)
        assert len(result) == 1
        assert result[0].candidate == "EXIT_EXTENSION"
        assert result[0].registry_status == REGISTRY_STATUS_ACTIVE

    def test_non_active_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("EXIT_EXTENSION", status="INACTIVE")])
        assert _load_active_registry(f) == []

    def test_ordered_by_candidate_order(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [
            _reg_rec("CONTINUATION_SIGNAL"),
            _reg_rec("EXIT_EXTENSION"),
        ])
        result = _load_active_registry(f)
        assert result[0].candidate == "EXIT_EXTENSION"
        assert result[1].candidate == "CONTINUATION_SIGNAL"

    def test_latest_date_wins(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [
            _reg_rec("EXIT_EXTENSION", date="2026-07-01", score=80.0),
            _reg_rec("EXIT_EXTENSION", date="2026-07-10", score=90.0),
        ])
        result = _load_active_registry(f)
        assert len(result) == 1
        assert result[0].readiness_score == pytest.approx(90.0)
        assert result[0].date == "2026-07-10"

    def test_all_five_candidates(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec(ct) for ct in _CANDIDATE_ORDER])
        result = _load_active_registry(f)
        assert len(result) == 5
        assert [r.candidate for r in result] == _CANDIDATE_ORDER

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("UNKNOWN_CANDIDATE")])
        assert _load_active_registry(f) == []

    def test_missing_date_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [{"candidate": "EXIT_EXTENSION", "registry_status": REGISTRY_STATUS_ACTIVE}])
        assert _load_active_registry(f) == []

    def test_score_preserved(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("PRIORITY", score=91.5)])
        result = _load_active_registry(f)
        assert result[0].readiness_score == pytest.approx(91.5)

    def test_n_completed_preserved(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("PRIORITY", n=7)])
        result = _load_active_registry(f)
        assert result[0].n_completed == 7


# ─── TestCutoffDate ───────────────────────────────────────────────────────────

class TestCutoffDate:
    def test_seven_days_back(self):
        assert _cutoff_date("2026-07-05", 7) == "2026-06-28"

    def test_zero_lookback(self):
        assert _cutoff_date("2026-07-05", 0) == "2026-07-05"

    def test_one_day(self):
        assert _cutoff_date("2026-07-05", 1) == "2026-07-04"

    def test_cross_month(self):
        assert _cutoff_date("2026-07-05", 10) == "2026-06-25"

    def test_invalid_date_returns_empty(self):
        assert _cutoff_date("not-a-date", 7) == ""

    def test_thirty_days(self):
        assert _cutoff_date("2026-07-31", 30) == "2026-07-01"


# ─── TestComputeSummary ───────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty(self):
        s = _compute_summary([], TODAY)
        assert s.active_candidates == 0
        assert s.new_candidates_7d == 0
        assert s.avg_readiness_score == 0.0
        assert s.top_candidate == ""

    def test_single_recent(self):
        reg = [_make_active_record("EXIT_EXTENSION", date=TODAY, score=84.0)]
        s = _compute_summary(reg, TODAY)
        assert s.active_candidates == 1
        assert s.new_candidates_7d == 1
        assert s.avg_readiness_score == pytest.approx(84.0)
        assert s.top_candidate == "EXIT_EXTENSION"

    def test_new_7d_exact_boundary(self):
        # today=2026-07-05, cutoff=2026-06-28
        # record on 2026-06-28 → is "new" (>= cutoff)
        reg = [_make_active_record("EXIT_EXTENSION", date="2026-06-28")]
        s = _compute_summary(reg, TODAY, lookback_days=7)
        assert s.new_candidates_7d == 1

    def test_new_7d_outside_window(self):
        # record on 2026-06-27 → older than 7d
        reg = [_make_active_record("EXIT_EXTENSION", date="2026-06-27")]
        s = _compute_summary(reg, TODAY, lookback_days=7)
        assert s.new_candidates_7d == 0

    def test_mixed_old_and_new(self):
        reg = [
            _make_active_record("EXIT_EXTENSION", date="2026-06-20", score=80.0),  # old
            _make_active_record("PRIORITY", date=TODAY, score=90.0),               # new
        ]
        s = _compute_summary(reg, TODAY)
        assert s.active_candidates == 2
        assert s.new_candidates_7d == 1

    def test_avg_readiness_score(self):
        reg = [
            _make_active_record("EXIT_EXTENSION", score=80.0),
            _make_active_record("PRIORITY", score=90.0),
        ]
        s = _compute_summary(reg, TODAY)
        assert s.avg_readiness_score == pytest.approx(85.0)

    def test_top_candidate_highest_score(self):
        reg = [
            _make_active_record("EXIT_EXTENSION", score=80.0),
            _make_active_record("PRIORITY", score=92.0),
            _make_active_record("SIGNAL_LEADER", score=85.0),
        ]
        s = _compute_summary(reg, TODAY)
        assert s.top_candidate == "PRIORITY"

    def test_custom_lookback_days(self):
        reg = [_make_active_record("EXIT_EXTENSION", date="2026-07-03")]
        s = _compute_summary(reg, "2026-07-05", lookback_days=1)
        assert s.new_candidates_7d == 0  # only 2d ago, but lookback=1d

    def test_all_five_active(self):
        reg = [_make_active_record(ct, score=85.0) for ct in _CANDIDATE_ORDER]
        s = _compute_summary(reg, TODAY)
        assert s.active_candidates == 5

    def test_invalid_today_zero_new(self):
        reg = [_make_active_record("EXIT_EXTENSION", date=TODAY)]
        s = _compute_summary(reg, "bad-date")
        assert s.new_candidates_7d == 0


# ─── TestFormatReport ─────────────────────────────────────────────────────────

class TestFormatReport:
    def _summary(self, n_active=1, new_7d=1, avg=84.0, top="EXIT_EXTENSION"):
        return RegistrySummary(n_active, new_7d, avg, top)

    def test_empty_returns_empty_string(self):
        assert format_registry_report([], self._summary(0, 0, 0.0, ""), TODAY) == ""

    def test_header_present(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        report = format_registry_report(reg, self._summary(), TODAY)
        assert "PROMOTION CANDIDATE REGISTRY" in report

    def test_active_section(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        report = format_registry_report(reg, self._summary(), TODAY)
        assert "ACTIVE:" in report
        assert "EXIT_EXTENSION" in report

    def test_score_shown(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        report = format_registry_report(reg, self._summary(), TODAY)
        assert "score=84" in report

    def test_date_shown(self):
        reg = [_make_active_record("EXIT_EXTENSION", date=TODAY)]
        report = format_registry_report(reg, self._summary(), TODAY)
        assert TODAY in report

    def test_new_this_week_section(self):
        reg = [_make_active_record("EXIT_EXTENSION", date=TODAY, score=84.0)]
        report = format_registry_report(reg, self._summary(), TODAY)
        assert "NEW THIS WEEK:" in report
        assert "★ EXIT_EXTENSION" in report

    def test_no_new_this_week_section_when_all_old(self):
        reg = [_make_active_record("EXIT_EXTENSION", date="2026-06-01", score=84.0)]
        report = format_registry_report(reg, self._summary(new_7d=0), TODAY)
        assert "NEW THIS WEEK:" not in report

    def test_avg_readiness_shown(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        report = format_registry_report(reg, self._summary(avg=84.0), TODAY)
        assert "Average Readiness: 84.0" in report

    def test_top_candidate_shown(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        report = format_registry_report(reg, self._summary(top="EXIT_EXTENSION"), TODAY)
        assert "Top Candidate: EXIT_EXTENSION" in report

    def test_active_count_in_header(self):
        reg = [
            _make_active_record("EXIT_EXTENSION", score=84.0),
            _make_active_record("PRIORITY", score=88.0),
        ]
        summary = RegistrySummary(2, 2, 86.0, "PRIORITY")
        report = format_registry_report(reg, summary, TODAY)
        assert "Active: 2" in report

    def test_new_7d_count_in_header(self):
        reg = [_make_active_record("EXIT_EXTENSION", date=TODAY, score=84.0)]
        summary = RegistrySummary(1, 1, 84.0, "EXIT_EXTENSION")
        report = format_registry_report(reg, summary, TODAY)
        assert "New (7d): 1" in report

    def test_separator_present(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        report = format_registry_report(reg, self._summary(), TODAY)
        assert "═" in report

    def test_no_top_candidate_no_line(self):
        reg = [_make_active_record("EXIT_EXTENSION", score=84.0)]
        summary = RegistrySummary(1, 1, 84.0, "")
        report = format_registry_report(reg, summary, TODAY)
        assert "Top Candidate:" not in report


# ─── TestAppendRegistryRecord ─────────────────────────────────────────────────

class TestAppendRegistryRecord:
    def _rec(self, ct: str = "EXIT_EXTENSION") -> RegistryRecord:
        return RegistryRecord(
            date=TODAY, candidate=ct, readiness_score=84.0,
            success_rate=66.7, n_completed=3,
        )

    def test_creates_file(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        append_registry_record(self._rec(), f)
        assert f.exists()
        records = _load_all_jsonl(f)
        assert len(records) == 1

    def test_all_fields_present(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        append_registry_record(self._rec(), f)
        r = _load_all_jsonl(f)[0]
        assert r["date"] == TODAY
        assert r["candidate"] == "EXIT_EXTENSION"
        assert r["readiness_score"] == pytest.approx(84.0)
        assert r["success_rate"] == pytest.approx(66.7)
        assert r["n_completed"] == 3
        assert r["registry_status"] == REGISTRY_STATUS_ACTIVE
        assert r["schema_version"] == SCHEMA_VERSION

    def test_appends_multiple(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        for ct in ["EXIT_EXTENSION", "PRIORITY"]:
            append_registry_record(self._rec(ct), f)
        records = _load_all_jsonl(f)
        assert len(records) == 2

    def test_creates_parent_dir(self, tmp_path):
        f = tmp_path / "sub" / "reg.jsonl"
        append_registry_record(self._rec(), f)
        assert f.exists()

    def test_schema_version_in_output(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        append_registry_record(self._rec(), f)
        r = _load_all_jsonl(f)[0]
        assert r["schema_version"] == SCHEMA_VERSION


# ─── TestLatestPrPerCandidate ─────────────────────────────────────────────────

class TestLatestPrPerCandidate:
    def test_empty(self):
        assert _latest_pr_per_candidate([]) == {}

    def test_single(self):
        rec = _pr_rec("PRIORITY", "PROMOTION_READY")
        result = _latest_pr_per_candidate([rec])
        assert "PRIORITY" in result

    def test_latest_date_wins(self):
        r1 = _pr_rec("PRIORITY", "PROMOTION_READY", score=70.0, date="2026-07-01")
        r2 = _pr_rec("PRIORITY", "PROMOTION_READY", score=85.0, date="2026-07-05")
        result = _latest_pr_per_candidate([r1, r2])
        assert result["PRIORITY"]["readiness_score"] == pytest.approx(85.0)

    def test_multiple_candidates(self):
        records = [_pr_rec(ct, "PROMOTION_READY") for ct in ["PRIORITY", "EXIT_EXTENSION"]]
        result = _latest_pr_per_candidate(records)
        assert len(result) == 2

    def test_missing_date_skipped(self):
        result = _latest_pr_per_candidate([{"candidate": "PRIORITY"}])
        assert result == {}

    def test_missing_candidate_skipped(self):
        result = _latest_pr_per_candidate([{"date": "2026-07-01"}])
        assert result == {}


# ─── TestRunPromotionCandidateRegistry ───────────────────────────────────────

class TestRunPromotionCandidateRegistry:
    def _setup(self, tmp_path):
        return tmp_path / "pr.jsonl", tmp_path / "reg.jsonl"

    def test_empty_pr_file_no_new(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.new_records == []
        assert result.active_registry == []

    def test_not_promotion_ready_not_registered(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "SHADOW_CONTINUE")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.new_records == []

    def test_more_data_not_registered(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "MORE_DATA")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.new_records == []

    def test_promotion_ready_registered(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY", score=84.0)])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.new_records) == 1
        assert result.new_records[0].candidate == "EXIT_EXTENSION"
        assert result.new_records[0].readiness_score == pytest.approx(84.0)
        assert result.new_records[0].registry_status == REGISTRY_STATUS_ACTIVE

    def test_already_active_skipped(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", date="2026-07-01")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.new_records == []

    def test_idempotent_second_run(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        r1 = run_promotion_candidate_registry(pr, reg, TODAY)
        r2 = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(r1.new_records) == 1
        assert len(r2.new_records) == 0
        assert len(_load_all_jsonl(reg)) == 1

    def test_idempotent_different_day(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        run_promotion_candidate_registry(pr, reg, "2026-07-01")
        r2 = run_promotion_candidate_registry(pr, reg, "2026-07-10")
        assert len(r2.new_records) == 0

    def test_multiple_promotion_ready(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [
            _pr_rec("EXIT_EXTENSION", "PROMOTION_READY"),
            _pr_rec("PRIORITY", "PROMOTION_READY"),
        ])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        candidates = {r.candidate for r in result.new_records}
        assert "EXIT_EXTENSION" in candidates
        assert "PRIORITY" in candidates

    def test_mixed_statuses(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [
            _pr_rec("EXIT_EXTENSION", "PROMOTION_READY"),
            _pr_rec("PRIORITY", "SHADOW_CONTINUE"),
            _pr_rec("SIGNAL_LEADER", "MORE_DATA"),
        ])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.new_records) == 1
        assert result.new_records[0].candidate == "EXIT_EXTENSION"

    def test_all_five_candidates_registered(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec(ct, "PROMOTION_READY") for ct in _CANDIDATE_ORDER])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.new_records) == 5
        registered = {r.candidate for r in result.new_records}
        assert registered == set(_CANDIDATE_ORDER)

    def test_output_file_written(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY", score=84.3, sr=66.7, n=4)])
        run_promotion_candidate_registry(pr, reg, TODAY)
        records = _load_all_jsonl(reg)
        assert len(records) == 1
        assert records[0]["candidate"] == "EXIT_EXTENSION"
        assert records[0]["readiness_score"] == pytest.approx(84.3)
        assert records[0]["success_rate"] == pytest.approx(66.7)
        assert records[0]["n_completed"] == 4

    def test_active_registry_in_result(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.active_registry) == 1
        assert result.active_registry[0].candidate == "EXIT_EXTENSION"

    def test_summary_active_count(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.summary.active_candidates == 1

    def test_summary_new_7d(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.summary.new_candidates_7d == 1

    def test_summary_top_candidate(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [
            _pr_rec("EXIT_EXTENSION", "PROMOTION_READY", score=84.0),
            _pr_rec("PRIORITY", "PROMOTION_READY", score=92.0),
        ])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.summary.top_candidate == "PRIORITY"

    def test_summary_avg_readiness(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [
            _pr_rec("EXIT_EXTENSION", "PROMOTION_READY", score=80.0),
            _pr_rec("PRIORITY", "PROMOTION_READY", score=90.0),
        ])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.summary.avg_readiness_score == pytest.approx(85.0)

    def test_multiple_pr_records_uses_latest(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [
            _pr_rec("EXIT_EXTENSION", "SHADOW_CONTINUE", score=55.0, date="2026-06-01"),
            _pr_rec("EXIT_EXTENSION", "PROMOTION_READY", score=84.0, date="2026-07-05"),
        ])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.new_records) == 1
        assert result.new_records[0].readiness_score == pytest.approx(84.0)

    def test_latest_pr_shadow_continue_not_registered(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [
            _pr_rec("EXIT_EXTENSION", "PROMOTION_READY", date="2026-06-01"),
            _pr_rec("EXIT_EXTENSION", "SHADOW_CONTINUE", date="2026-07-05"),
        ])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.new_records == []

    def test_fail_open_on_exception(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        pr.write_text("NOT JSON\n", encoding="utf-8")
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert isinstance(result, RegistryRunResult)

    def test_schema_version_in_output(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        run_promotion_candidate_registry(pr, reg, TODAY)
        r = _load_all_jsonl(reg)[0]
        assert r["schema_version"] == SCHEMA_VERSION

    def test_date_in_output(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        run_promotion_candidate_registry(pr, reg, TODAY)
        r = _load_all_jsonl(reg)[0]
        assert r["date"] == TODAY

    def test_active_registry_ordered(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        _write_jsonl(pr, [_pr_rec(ct, "PROMOTION_READY") for ct in _CANDIDATE_ORDER])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        order = [r.candidate for r in result.active_registry]
        assert order == _CANDIDATE_ORDER

    def test_pre_existing_in_active_registry(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        # Pre-seed registry with one old candidate
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", date="2026-06-01")])
        # New PROMOTION_READY candidate
        _write_jsonl(pr, [_pr_rec("PRIORITY", "PROMOTION_READY")])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.new_records) == 1
        assert result.new_records[0].candidate == "PRIORITY"
        assert len(result.active_registry) == 2

    def test_summary_reflects_pre_existing(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        # Pre-seed with old candidate
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", date="2026-06-01", score=80.0)])
        _write_jsonl(pr, [_pr_rec("PRIORITY", "PROMOTION_READY", score=90.0)])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        # 2 active candidates total
        assert result.summary.active_candidates == 2
        # Only PRIORITY is "new" this week
        assert result.summary.new_candidates_7d == 1

    def test_empty_registry_no_active(self, tmp_path):
        pr, reg = self._setup(tmp_path)
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert result.summary.active_candidates == 0
        assert result.summary.top_candidate == ""


# ─── TestIntegration ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_cycle_all_candidates(self, tmp_path):
        pr  = tmp_path / "pr.jsonl"
        reg = tmp_path / "reg.jsonl"
        _write_jsonl(pr, [_pr_rec(ct, "PROMOTION_READY", score=85.0) for ct in _CANDIDATE_ORDER])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(result.new_records) == 5
        assert result.summary.active_candidates == 5
        assert result.summary.new_candidates_7d == 5

    def test_report_format_integrated(self, tmp_path):
        pr  = tmp_path / "pr.jsonl"
        reg = tmp_path / "reg.jsonl"
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY", score=84.0)])
        result = run_promotion_candidate_registry(pr, reg, TODAY)
        report = format_registry_report(result.active_registry, result.summary, TODAY)
        assert "PROMOTION CANDIDATE REGISTRY" in report
        assert "EXIT_EXTENSION" in report
        assert "ACTIVE:" in report

    def test_idempotent_after_three_runs(self, tmp_path):
        pr  = tmp_path / "pr.jsonl"
        reg = tmp_path / "reg.jsonl"
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        for _ in range(3):
            run_promotion_candidate_registry(pr, reg, TODAY)
        assert len(_load_all_jsonl(reg)) == 1

    def test_registry_preserves_registration_date(self, tmp_path):
        pr  = tmp_path / "pr.jsonl"
        reg = tmp_path / "reg.jsonl"
        _write_jsonl(pr, [_pr_rec("EXIT_EXTENSION", "PROMOTION_READY")])
        run_promotion_candidate_registry(pr, reg, "2026-07-01")
        result = run_promotion_candidate_registry(pr, reg, "2026-07-10")
        # Still only 1 record, registered on 2026-07-01
        records = _load_all_jsonl(reg)
        assert len(records) == 1
        assert records[0]["date"] == "2026-07-01"

    def test_candidate_order_invariant(self):
        assert _CANDIDATE_ORDER[0] == "EXIT_EXTENSION"
        assert _CANDIDATE_ORDER[-1] == "CONTINUATION_SIGNAL"
        assert len(_CANDIDATE_ORDER) == 5

    def test_default_lookback_days(self):
        assert DEFAULT_LOOKBACK_DAYS == 7

    def test_status_constant(self):
        assert _STATUS_PROMOTION_READY == "PROMOTION_READY"

    def test_registry_status_constant(self):
        assert REGISTRY_STATUS_ACTIVE == "ACTIVE"
