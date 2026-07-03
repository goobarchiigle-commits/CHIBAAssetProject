"""
test_promotion_deployment_planner.py — Promotion Deployment Planner (Phase 6D-Lite)

Tests (100% coverage):
  TestSafeHelpers / TestLoadAllJsonl / TestPlannedChangeMap
  TestLoadPlannedSet / TestLoadPlannedRecords / TestLoadActiveFromRegistry
  TestComputeSummary / TestFormatReport / TestAppendPlanRecord
  TestRunPromotionDeploymentPlanner / TestIntegration
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.promotion_deployment_planner import (
    DEPLOYMENT_STATUS_PLANNED,
    SCHEMA_VERSION,
    DeploymentPlanRecord,
    DeploymentPlanRunResult,
    DeploymentPlanSummary,
    _CANDIDATE_ORDER,
    _PLANNED_CHANGE_MAP,
    _REGISTRY_STATUS_ACTIVE,
    _compute_summary,
    _load_active_from_registry,
    _load_all_jsonl,
    _load_planned_records,
    _load_planned_set,
    _safe_float,
    _safe_int,
    append_plan_record,
    format_deployment_report,
    run_promotion_deployment_planner,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _reg_rec(candidate: str, date: str = "2026-07-05",
             score: float = 84.0, status: str = _REGISTRY_STATUS_ACTIVE) -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "readiness_score": score,
        "success_rate": 66.7,
        "n_completed": 3,
        "registry_status": status,
        "schema_version": "v1",
    }


def _plan_rec(candidate: str, date: str = "2026-07-15",
              score: float = 84.0,
              planned_change: str = "increase_exit_extension_shadow",
              status: str = DEPLOYMENT_STATUS_PLANNED) -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "planned_change": planned_change,
        "readiness_score": score,
        "deployment_status": status,
        "schema_version": SCHEMA_VERSION,
    }


def _make_plan_record(ct: str, score: float = 84.0,
                      date: str = "2026-07-15") -> DeploymentPlanRecord:
    return DeploymentPlanRecord(
        date=date,
        candidate=ct,
        planned_change=_PLANNED_CHANGE_MAP.get(ct, "unknown"),
        readiness_score=score,
    )


TODAY = "2026-07-15"


# ─── TestSafeHelpers ─────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float_valid(self):
        assert _safe_float(1.5) == pytest.approx(1.5)

    def test_safe_float_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_fallback(self):
        assert _safe_float(None, 5.5) == pytest.approx(5.5)

    def test_safe_int_valid(self):
        assert _safe_int(3) == 3

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_invalid(self):
        assert _safe_int("bad") == 0

    def test_safe_int_fallback(self):
        assert _safe_int(None, 7) == 7


# ─── TestLoadAllJsonl ─────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file(self, tmp_path):
        assert _load_all_jsonl(tmp_path / "missing.jsonl") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "f.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_all_jsonl(f) == []

    def test_valid_records(self, tmp_path):
        f = tmp_path / "f.jsonl"
        _write_jsonl(f, [{"a": 1}, {"b": 2}])
        assert len(_load_all_jsonl(f)) == 2

    def test_skips_invalid_lines(self, tmp_path):
        f = tmp_path / "f.jsonl"
        f.write_text('{"a":1}\nBAD\n{"b":2}\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 2

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "f.jsonl"
        f.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 2


# ─── TestPlannedChangeMap ─────────────────────────────────────────────────────

class TestPlannedChangeMap:
    def test_all_five_candidates_present(self):
        for ct in _CANDIDATE_ORDER:
            assert ct in _PLANNED_CHANGE_MAP

    def test_exit_extension_change(self):
        assert _PLANNED_CHANGE_MAP["EXIT_EXTENSION"] == "increase_exit_extension_shadow"

    def test_capital_concentration_change(self):
        assert _PLANNED_CHANGE_MAP["CAPITAL_CONCENTRATION"] == "priority_weighted_capital_shadow"

    def test_priority_change(self):
        assert _PLANNED_CHANGE_MAP["PRIORITY"] == "priority_threshold_shadow"

    def test_signal_leader_change(self):
        assert _PLANNED_CHANGE_MAP["SIGNAL_LEADER"] == "signal_weight_shadow"

    def test_continuation_signal_change(self):
        assert _PLANNED_CHANGE_MAP["CONTINUATION_SIGNAL"] == "continuation_boost_shadow"

    def test_no_unknown_candidates(self):
        # All keys are in _CANDIDATE_ORDER
        for ct in _PLANNED_CHANGE_MAP:
            assert ct in _CANDIDATE_ORDER

    def test_all_values_non_empty(self):
        for change in _PLANNED_CHANGE_MAP.values():
            assert len(change) > 0

    def test_all_values_unique(self):
        values = list(_PLANNED_CHANGE_MAP.values())
        assert len(values) == len(set(values))


# ─── TestLoadPlannedSet ───────────────────────────────────────────────────────

class TestLoadPlannedSet:
    def test_missing_file(self, tmp_path):
        assert _load_planned_set(tmp_path / "missing.jsonl") == set()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_planned_set(f) == set()

    def test_single_planned(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("EXIT_EXTENSION")])
        s = _load_planned_set(f)
        assert "EXIT_EXTENSION" in s

    def test_non_planned_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("EXIT_EXTENSION", status="DEPLOYED")])
        assert _load_planned_set(f) == set()

    def test_multiple_candidates(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec("EXIT_EXTENSION"),
            _plan_rec("PRIORITY"),
        ])
        s = _load_planned_set(f)
        assert len(s) == 2
        assert "PRIORITY" in s

    def test_duplicate_same_candidate_deduped(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec("EXIT_EXTENSION", date="2026-07-01"),
            _plan_rec("EXIT_EXTENSION", date="2026-07-15"),
        ])
        s = _load_planned_set(f)
        assert len(s) == 1

    def test_blank_candidate_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [{"deployment_status": DEPLOYMENT_STATUS_PLANNED, "candidate": ""}])
        assert _load_planned_set(f) == set()

    def test_missing_status_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [{"candidate": "EXIT_EXTENSION"}])
        assert _load_planned_set(f) == set()


# ─── TestLoadPlannedRecords ───────────────────────────────────────────────────

class TestLoadPlannedRecords:
    def test_empty(self, tmp_path):
        assert _load_planned_records(tmp_path / "missing.jsonl") == []

    def test_single_record(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("EXIT_EXTENSION")])
        result = _load_planned_records(f)
        assert len(result) == 1
        assert result[0].candidate == "EXIT_EXTENSION"
        assert result[0].planned_change == "increase_exit_extension_shadow"

    def test_non_planned_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("EXIT_EXTENSION", status="DEPLOYED")])
        assert _load_planned_records(f) == []

    def test_ordered_by_candidate_order(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec("CONTINUATION_SIGNAL"),
            _plan_rec("EXIT_EXTENSION"),
        ])
        result = _load_planned_records(f)
        assert result[0].candidate == "EXIT_EXTENSION"
        assert result[1].candidate == "CONTINUATION_SIGNAL"

    def test_latest_date_wins(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec("EXIT_EXTENSION", date="2026-07-01", score=80.0),
            _plan_rec("EXIT_EXTENSION", date="2026-07-15", score=90.0),
        ])
        result = _load_planned_records(f)
        assert len(result) == 1
        assert result[0].readiness_score == pytest.approx(90.0)

    def test_all_five_candidates(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec(ct, planned_change=_PLANNED_CHANGE_MAP[ct])
            for ct in _CANDIDATE_ORDER
        ])
        result = _load_planned_records(f)
        assert len(result) == 5
        assert [r.candidate for r in result] == _CANDIDATE_ORDER

    def test_missing_date_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [{"candidate": "EXIT_EXTENSION", "deployment_status": DEPLOYMENT_STATUS_PLANNED}])
        assert _load_planned_records(f) == []

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("UNKNOWN")])
        assert _load_planned_records(f) == []

    def test_planned_change_preserved(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("PRIORITY", planned_change="priority_threshold_shadow")])
        result = _load_planned_records(f)
        assert result[0].planned_change == "priority_threshold_shadow"


# ─── TestLoadActiveFromRegistry ───────────────────────────────────────────────

class TestLoadActiveFromRegistry:
    def test_empty(self, tmp_path):
        assert _load_active_from_registry(tmp_path / "missing.jsonl") == []

    def test_single_active(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("EXIT_EXTENSION")])
        result = _load_active_from_registry(f)
        assert len(result) == 1
        assert result[0]["candidate"] == "EXIT_EXTENSION"

    def test_inactive_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("EXIT_EXTENSION", status="INACTIVE")])
        assert _load_active_from_registry(f) == []

    def test_ordered_by_candidate_order(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [
            _reg_rec("CONTINUATION_SIGNAL"),
            _reg_rec("EXIT_EXTENSION"),
        ])
        result = _load_active_from_registry(f)
        assert result[0]["candidate"] == "EXIT_EXTENSION"
        assert result[1]["candidate"] == "CONTINUATION_SIGNAL"

    def test_latest_date_per_candidate(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [
            _reg_rec("EXIT_EXTENSION", date="2026-07-01", score=80.0),
            _reg_rec("EXIT_EXTENSION", date="2026-07-10", score=90.0),
        ])
        result = _load_active_from_registry(f)
        assert len(result) == 1
        assert result[0]["readiness_score"] == pytest.approx(90.0)

    def test_all_five_candidates(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec(ct) for ct in _CANDIDATE_ORDER])
        result = _load_active_from_registry(f)
        assert [r["candidate"] for r in result] == _CANDIDATE_ORDER

    def test_missing_date_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [{"candidate": "EXIT_EXTENSION", "registry_status": _REGISTRY_STATUS_ACTIVE}])
        assert _load_active_from_registry(f) == []

    def test_score_preserved(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_reg_rec("PRIORITY", score=91.5)])
        result = _load_active_from_registry(f)
        assert result[0]["readiness_score"] == pytest.approx(91.5)


# ─── TestComputeSummary ───────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty(self):
        s = _compute_summary([])
        assert s.planned_count == 0
        assert s.highest_priority_candidate == ""
        assert s.avg_readiness_score == 0.0

    def test_single_record(self):
        records = [_make_plan_record("EXIT_EXTENSION", score=84.0)]
        s = _compute_summary(records)
        assert s.planned_count == 1
        assert s.highest_priority_candidate == "EXIT_EXTENSION"
        assert s.avg_readiness_score == pytest.approx(84.0)

    def test_highest_priority_by_score(self):
        records = [
            _make_plan_record("EXIT_EXTENSION", score=84.0),
            _make_plan_record("PRIORITY", score=92.0),
        ]
        s = _compute_summary(records)
        assert s.highest_priority_candidate == "PRIORITY"

    def test_avg_readiness_score(self):
        records = [
            _make_plan_record("EXIT_EXTENSION", score=80.0),
            _make_plan_record("PRIORITY", score=90.0),
            _make_plan_record("SIGNAL_LEADER", score=85.0),
        ]
        s = _compute_summary(records)
        assert s.avg_readiness_score == pytest.approx(85.0)

    def test_planned_count(self):
        records = [_make_plan_record(ct) for ct in _CANDIDATE_ORDER]
        s = _compute_summary(records)
        assert s.planned_count == 5

    def test_rounded_avg(self):
        records = [
            _make_plan_record("EXIT_EXTENSION", score=84.1),
            _make_plan_record("PRIORITY", score=84.2),
        ]
        s = _compute_summary(records)
        assert s.avg_readiness_score == round((84.1 + 84.2) / 2, 1)


# ─── TestFormatReport ─────────────────────────────────────────────────────────

class TestFormatReport:
    def _summary(self, n=1, top="EXIT_EXTENSION", avg=84.0):
        return DeploymentPlanSummary(n, top, avg)

    def test_empty_returns_empty_string(self):
        assert format_deployment_report([], self._summary(0, "", 0.0)) == ""

    def test_header_present(self):
        records = [_make_plan_record("EXIT_EXTENSION", score=84.0)]
        report = format_deployment_report(records, self._summary())
        assert "PROMOTION DEPLOYMENT PLAN" in report

    def test_planned_count(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        report = format_deployment_report(records, self._summary(n=1))
        assert "Planned: 1" in report

    def test_highest_priority(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        report = format_deployment_report(records, self._summary(top="EXIT_EXTENSION"))
        assert "Highest Priority: EXIT_EXTENSION" in report

    def test_avg_readiness(self):
        records = [_make_plan_record("EXIT_EXTENSION", score=84.0)]
        report = format_deployment_report(records, self._summary(avg=84.0))
        assert "Average Readiness: 84.0" in report

    def test_planned_section(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        report = format_deployment_report(records, self._summary())
        assert "PLANNED:" in report

    def test_candidate_name_shown(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        report = format_deployment_report(records, self._summary())
        assert "EXIT_EXTENSION" in report

    def test_planned_change_shown(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        report = format_deployment_report(records, self._summary())
        assert "increase_exit_extension_shadow" in report

    def test_score_shown(self):
        records = [_make_plan_record("EXIT_EXTENSION", score=84.1)]
        report = format_deployment_report(records, self._summary(avg=84.1))
        assert "score=84.1" in report

    def test_separator_present(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        report = format_deployment_report(records, self._summary())
        assert "═" in report

    def test_no_highest_priority_when_empty(self):
        records = [_make_plan_record("EXIT_EXTENSION")]
        summary = DeploymentPlanSummary(1, "", 84.0)
        report = format_deployment_report(records, summary)
        assert "Highest Priority:" not in report

    def test_all_five_candidates_in_report(self):
        records = [_make_plan_record(ct) for ct in _CANDIDATE_ORDER]
        summary = DeploymentPlanSummary(5, "EXIT_EXTENSION", 84.0)
        report = format_deployment_report(records, summary)
        for ct in _CANDIDATE_ORDER:
            assert ct in report

    def test_all_planned_changes_in_report(self):
        records = [_make_plan_record(ct) for ct in _CANDIDATE_ORDER]
        summary = DeploymentPlanSummary(5, "EXIT_EXTENSION", 84.0)
        report = format_deployment_report(records, summary)
        for change in _PLANNED_CHANGE_MAP.values():
            assert change in report


# ─── TestAppendPlanRecord ─────────────────────────────────────────────────────

class TestAppendPlanRecord:
    def _rec(self, ct: str = "EXIT_EXTENSION") -> DeploymentPlanRecord:
        return DeploymentPlanRecord(
            date=TODAY, candidate=ct,
            planned_change=_PLANNED_CHANGE_MAP[ct],
            readiness_score=84.0,
        )

    def test_creates_file(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        append_plan_record(self._rec(), f)
        assert f.exists()

    def test_all_fields_present(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        append_plan_record(self._rec(), f)
        r = _load_all_jsonl(f)[0]
        assert r["date"] == TODAY
        assert r["candidate"] == "EXIT_EXTENSION"
        assert r["planned_change"] == "increase_exit_extension_shadow"
        assert r["readiness_score"] == pytest.approx(84.0)
        assert r["deployment_status"] == DEPLOYMENT_STATUS_PLANNED
        assert r["schema_version"] == SCHEMA_VERSION

    def test_appends_multiple(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        for ct in ["EXIT_EXTENSION", "PRIORITY"]:
            append_plan_record(self._rec(ct), f)
        assert len(_load_all_jsonl(f)) == 2

    def test_creates_parent_dir(self, tmp_path):
        f = tmp_path / "sub" / "plan.jsonl"
        append_plan_record(self._rec(), f)
        assert f.exists()

    def test_schema_version_correct(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        append_plan_record(self._rec(), f)
        assert _load_all_jsonl(f)[0]["schema_version"] == SCHEMA_VERSION


# ─── TestRunPromotionDeploymentPlanner ───────────────────────────────────────

class TestRunPromotionDeploymentPlanner:
    def _setup(self, tmp_path):
        return tmp_path / "reg.jsonl", tmp_path / "plan.jsonl"

    def test_empty_registry_no_new(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.new_records == []
        assert result.planned_records == []

    def test_inactive_not_planned(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", status="INACTIVE")])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.new_records == []

    def test_active_candidate_planned(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", score=84.0)])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(result.new_records) == 1
        rec = result.new_records[0]
        assert rec.candidate == "EXIT_EXTENSION"
        assert rec.planned_change == "increase_exit_extension_shadow"
        assert rec.readiness_score == pytest.approx(84.0)
        assert rec.deployment_status == DEPLOYMENT_STATUS_PLANNED

    def test_already_planned_skipped(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.new_records == []

    def test_idempotent_second_run(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        r1 = run_promotion_deployment_planner(reg, plan, TODAY)
        r2 = run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(r1.new_records) == 1
        assert len(r2.new_records) == 0
        assert len(_load_all_jsonl(plan)) == 1

    def test_idempotent_different_day(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        run_promotion_deployment_planner(reg, plan, "2026-07-01")
        r2 = run_promotion_deployment_planner(reg, plan, "2026-07-20")
        assert len(r2.new_records) == 0

    def test_multiple_active_candidates(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec(ct) for ct in ["EXIT_EXTENSION", "PRIORITY"]])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(result.new_records) == 2

    def test_all_five_candidates_planned(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec(ct) for ct in _CANDIDATE_ORDER])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(result.new_records) == 5
        candidates = {r.candidate for r in result.new_records}
        assert candidates == set(_CANDIDATE_ORDER)

    def test_planned_change_mapping_correct(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec(ct) for ct in _CANDIDATE_ORDER])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        by_ct = {r.candidate: r for r in result.new_records}
        for ct, expected_change in _PLANNED_CHANGE_MAP.items():
            assert by_ct[ct].planned_change == expected_change

    def test_planned_records_in_result(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(result.planned_records) == 1
        assert result.planned_records[0].candidate == "EXIT_EXTENSION"

    def test_summary_planned_count(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec(ct) for ct in ["EXIT_EXTENSION", "PRIORITY"]])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.summary.planned_count == 2

    def test_summary_highest_priority(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [
            _reg_rec("EXIT_EXTENSION", score=84.0),
            _reg_rec("PRIORITY", score=92.0),
        ])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.summary.highest_priority_candidate == "PRIORITY"

    def test_summary_avg_readiness(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [
            _reg_rec("EXIT_EXTENSION", score=80.0),
            _reg_rec("PRIORITY", score=90.0),
        ])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.summary.avg_readiness_score == pytest.approx(85.0)

    def test_schema_version_in_output(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        run_promotion_deployment_planner(reg, plan, TODAY)
        assert _load_all_jsonl(plan)[0]["schema_version"] == SCHEMA_VERSION

    def test_date_in_output(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        run_promotion_deployment_planner(reg, plan, TODAY)
        assert _load_all_jsonl(plan)[0]["date"] == TODAY

    def test_planned_records_ordered(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec(ct) for ct in _CANDIDATE_ORDER])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        order = [r.candidate for r in result.planned_records]
        assert order == _CANDIDATE_ORDER

    def test_fail_open_on_exception(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        reg.write_text("NOT JSON\n", encoding="utf-8")
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert isinstance(result, DeploymentPlanRunResult)

    def test_multiple_registry_records_uses_latest(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [
            _reg_rec("EXIT_EXTENSION", date="2026-07-01", score=80.0),
            _reg_rec("EXIT_EXTENSION", date="2026-07-10", score=90.0),
        ])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.new_records[0].readiness_score == pytest.approx(90.0)

    def test_pre_existing_planned_included_in_summary(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        # Pre-seed one planned
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION", score=84.0)])
        # New active candidate
        _write_jsonl(reg, [_reg_rec("PRIORITY", score=88.0)])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        # 1 new + 1 pre-existing = 2 planned
        assert result.summary.planned_count == 2

    def test_empty_registry_summary_zeros(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert result.summary.planned_count == 0
        assert result.summary.highest_priority_candidate == ""

    def test_readiness_score_preserved_in_output(self, tmp_path):
        reg, plan = self._setup(tmp_path)
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", score=91.7)])
        run_promotion_deployment_planner(reg, plan, TODAY)
        r = _load_all_jsonl(plan)[0]
        assert r["readiness_score"] == pytest.approx(91.7)


# ─── TestIntegration ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_cycle_all_candidates(self, tmp_path):
        reg  = tmp_path / "reg.jsonl"
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(reg, [_reg_rec(ct, score=85.0) for ct in _CANDIDATE_ORDER])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(result.new_records) == 5
        assert result.summary.planned_count == 5
        for r in result.new_records:
            assert r.planned_change == _PLANNED_CHANGE_MAP[r.candidate]

    def test_report_format_integrated(self, tmp_path):
        reg  = tmp_path / "reg.jsonl"
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION", score=84.0)])
        result = run_promotion_deployment_planner(reg, plan, TODAY)
        report = format_deployment_report(result.planned_records, result.summary)
        assert "PROMOTION DEPLOYMENT PLAN" in report
        assert "increase_exit_extension_shadow" in report

    def test_idempotent_three_runs(self, tmp_path):
        reg  = tmp_path / "reg.jsonl"
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(reg, [_reg_rec("EXIT_EXTENSION")])
        for _ in range(3):
            run_promotion_deployment_planner(reg, plan, TODAY)
        assert len(_load_all_jsonl(plan)) == 1

    def test_candidate_order_invariant(self):
        assert _CANDIDATE_ORDER == [
            "EXIT_EXTENSION",
            "CAPITAL_CONCENTRATION",
            "PRIORITY",
            "SIGNAL_LEADER",
            "CONTINUATION_SIGNAL",
        ]

    def test_status_constant(self):
        assert DEPLOYMENT_STATUS_PLANNED == "PLANNED"

    def test_registry_status_constant(self):
        assert _REGISTRY_STATUS_ACTIVE == "ACTIVE"

    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "v1"

    def test_report_empty_when_no_planned(self, tmp_path):
        result = run_promotion_deployment_planner(
            tmp_path / "reg.jsonl",
            tmp_path / "plan.jsonl",
            TODAY
        )
        report = format_deployment_report(result.planned_records, result.summary)
        assert report == ""
