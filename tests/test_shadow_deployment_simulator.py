"""Tests for src/analytics/shadow_deployment_simulator.py (Phase 6E-Lite)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.shadow_deployment_simulator import (
    SCHEMA_VERSION,
    SHADOW_DURATION_DAYS,
    SHADOW_STATUS_DEPLOYED,
    ShadowDeploymentRecord,
    ShadowDeploymentRunResult,
    ShadowDeploymentSummary,
    _CANDIDATE_ORDER,
    _PLAN_STATUS_PLANNED,
    _append_jsonl,
    _compute_summary,
    _load_all_jsonl,
    _load_deployed_records,
    _load_deployed_set,
    _load_planned_from_deployment_plan,
    _safe_float,
    _safe_int,
    append_shadow_record,
    format_shadow_deployment_report,
    run_shadow_deployment_simulator,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _plan_rec(candidate: str, planned_change: str = "change_x",
              date: str = "2026-07-01", status: str = "PLANNED") -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "planned_change": planned_change,
        "deployment_status": status,
        "schema_version": "v1",
    }


def _deployed_rec(candidate: str, planned_change: str = "change_x",
                  date: str = "2026-07-20", status: str = "DEPLOYED") -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "planned_change": planned_change,
        "shadow_status": status,
        "shadow_duration_days": 30,
        "schema_version": "v1",
    }


# ── TestSafeHelpers ───────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float_normal(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_fallback(self):
        assert _safe_float(None, 99.0) == 99.0

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_safe_float_string_number(self):
        assert _safe_float("7.5") == pytest.approx(7.5)

    def test_safe_float_bad_string(self):
        assert _safe_float("bad") == 0.0

    def test_safe_int_normal(self):
        assert _safe_int(5) == 5

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_string(self):
        assert _safe_int("30") == 30

    def test_safe_int_bad(self):
        assert _safe_int("bad") == 0

    def test_safe_int_fallback(self):
        assert _safe_int(None, 30) == 30


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_shadow_status_deployed(self):
        assert SHADOW_STATUS_DEPLOYED == "DEPLOYED"

    def test_plan_status_planned(self):
        assert _PLAN_STATUS_PLANNED == "PLANNED"

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_shadow_duration_days(self):
        assert SHADOW_DURATION_DAYS == 30

    def test_candidate_order_length(self):
        assert len(_CANDIDATE_ORDER) == 5

    def test_candidate_order_values(self):
        assert _CANDIDATE_ORDER == [
            "EXIT_EXTENSION",
            "CAPITAL_CONCENTRATION",
            "PRIORITY",
            "SIGNAL_LEADER",
            "CONTINUATION_SIGNAL",
        ]


# ── TestLoadAllJsonl ──────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        result = _load_all_jsonl(tmp_path / "missing.jsonl")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_all_jsonl(f) == []

    def test_valid_lines_loaded(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"a": 1}, {"b": 2}])
        result = _load_all_jsonl(f)
        assert len(result) == 2

    def test_bad_json_line_skipped(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text('{"ok": 1}\nnot_json\n{"ok": 2}\n', encoding="utf-8")
        result = _load_all_jsonl(f)
        assert len(result) == 2

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "blanks.jsonl"
        f.write_text('\n{"x": 1}\n\n', encoding="utf-8")
        result = _load_all_jsonl(f)
        assert len(result) == 1


# ── TestLoadDeployedSet ───────────────────────────────────────────────────────

class TestLoadDeployedSet:
    def test_empty_file(self, tmp_path):
        assert _load_deployed_set(tmp_path / "missing.jsonl") == set()

    def test_deployed_candidate_added(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_deployed_rec("EXIT_EXTENSION")])
        result = _load_deployed_set(f)
        assert "EXIT_EXTENSION" in result

    def test_non_deployed_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_deployed_rec("EXIT_EXTENSION", status="OTHER")])
        assert _load_deployed_set(f) == set()

    def test_multiple_candidates(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _deployed_rec("EXIT_EXTENSION"),
            _deployed_rec("PRIORITY"),
        ])
        result = _load_deployed_set(f)
        assert result == {"EXIT_EXTENSION", "PRIORITY"}

    def test_duplicate_candidate_included_once(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _deployed_rec("EXIT_EXTENSION"),
            _deployed_rec("EXIT_EXTENSION"),
        ])
        result = _load_deployed_set(f)
        assert len(result) == 1
        assert "EXIT_EXTENSION" in result

    def test_empty_candidate_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [{"shadow_status": "DEPLOYED", "candidate": ""}])
        assert _load_deployed_set(f) == set()


# ── TestLoadDeployedRecords ───────────────────────────────────────────────────

class TestLoadDeployedRecords:
    def test_empty_file(self, tmp_path):
        assert _load_deployed_records(tmp_path / "missing.jsonl") == []

    def test_candidate_order_preserved(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _deployed_rec("SIGNAL_LEADER"),
            _deployed_rec("EXIT_EXTENSION"),
        ])
        result = _load_deployed_records(f)
        assert [r.candidate for r in result] == ["EXIT_EXTENSION", "SIGNAL_LEADER"]

    def test_non_deployed_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_deployed_rec("EXIT_EXTENSION", status="OTHER")])
        assert _load_deployed_records(f) == []

    def test_latest_per_candidate_selected(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _deployed_rec("EXIT_EXTENSION", date="2026-07-01"),
            _deployed_rec("EXIT_EXTENSION", date="2026-07-20"),
        ])
        result = _load_deployed_records(f)
        assert len(result) == 1
        assert result[0].date == "2026-07-20"

    def test_fields_populated(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_deployed_rec("EXIT_EXTENSION", planned_change="my_change")])
        result = _load_deployed_records(f)
        r = result[0]
        assert r.candidate == "EXIT_EXTENSION"
        assert r.planned_change == "my_change"
        assert r.shadow_status == SHADOW_STATUS_DEPLOYED
        assert r.shadow_duration_days == 30

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_deployed_rec("UNKNOWN_CAND")])
        assert _load_deployed_records(f) == []


# ── TestLoadPlannedFromDeploymentPlan ─────────────────────────────────────────

class TestLoadPlannedFromDeploymentPlan:
    def test_empty_file(self, tmp_path):
        assert _load_planned_from_deployment_plan(tmp_path / "missing.jsonl") == []

    def test_planned_candidate_included(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("EXIT_EXTENSION")])
        result = _load_planned_from_deployment_plan(f)
        assert len(result) == 1
        assert result[0]["candidate"] == "EXIT_EXTENSION"

    def test_non_planned_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("EXIT_EXTENSION", status="OTHER")])
        assert _load_planned_from_deployment_plan(f) == []

    def test_candidate_order_preserved(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec("SIGNAL_LEADER"),
            _plan_rec("EXIT_EXTENSION"),
        ])
        result = _load_planned_from_deployment_plan(f)
        assert [r["candidate"] for r in result] == ["EXIT_EXTENSION", "SIGNAL_LEADER"]

    def test_latest_per_candidate_selected(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [
            _plan_rec("EXIT_EXTENSION", date="2026-07-01"),
            _plan_rec("EXIT_EXTENSION", date="2026-07-10"),
        ])
        result = _load_planned_from_deployment_plan(f)
        assert len(result) == 1
        assert result[0]["date"] == "2026-07-10"

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec("UNKNOWN_CAND")])
        assert _load_planned_from_deployment_plan(f) == []

    def test_all_five_planned(self, tmp_path):
        f = tmp_path / "plan.jsonl"
        _write_jsonl(f, [_plan_rec(c) for c in _CANDIDATE_ORDER])
        result = _load_planned_from_deployment_plan(f)
        assert len(result) == 5


# ── TestComputeSummary ────────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty_list(self):
        summary = _compute_summary([])
        assert summary.active_shadow_deployments == 0
        assert summary.deployment_count == 0
        assert summary.candidate_counts == {}

    def test_single_record(self):
        rec = ShadowDeploymentRecord(
            date="2026-07-20",
            candidate="EXIT_EXTENSION",
            planned_change="increase_exit_extension_shadow",
        )
        summary = _compute_summary([rec])
        assert summary.active_shadow_deployments == 1
        assert summary.deployment_count == 1
        assert summary.candidate_counts == {"EXIT_EXTENSION": 1}

    def test_multiple_records(self):
        records = [
            ShadowDeploymentRecord("2026-07-20", "EXIT_EXTENSION", "change_a"),
            ShadowDeploymentRecord("2026-07-20", "PRIORITY", "change_b"),
        ]
        summary = _compute_summary(records)
        assert summary.active_shadow_deployments == 2
        assert summary.deployment_count == 2
        assert "EXIT_EXTENSION" in summary.candidate_counts
        assert "PRIORITY" in summary.candidate_counts

    def test_candidate_counts_correct(self):
        records = [
            ShadowDeploymentRecord("2026-07-20", "EXIT_EXTENSION", "c"),
            ShadowDeploymentRecord("2026-07-21", "EXIT_EXTENSION", "c"),
        ]
        summary = _compute_summary(records)
        assert summary.candidate_counts["EXIT_EXTENSION"] == 2

    def test_five_candidates(self):
        records = [
            ShadowDeploymentRecord("2026-07-20", ct, "change")
            for ct in _CANDIDATE_ORDER
        ]
        summary = _compute_summary(records)
        assert summary.active_shadow_deployments == 5
        assert summary.deployment_count == 5
        assert len(summary.candidate_counts) == 5


# ── TestFormatReport ──────────────────────────────────────────────────────────

class TestFormatReport:
    def _make_record(self, candidate: str) -> ShadowDeploymentRecord:
        return ShadowDeploymentRecord(
            date="2026-07-20",
            candidate=candidate,
            planned_change=f"change_{candidate.lower()}",
        )

    def _make_summary(self, records: list) -> ShadowDeploymentSummary:
        return _compute_summary(records)

    def test_empty_returns_empty_string(self):
        summary = ShadowDeploymentSummary(0, 0, {})
        assert format_shadow_deployment_report([], summary) == ""

    def test_contains_header(self):
        r = self._make_record("EXIT_EXTENSION")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert "SHADOW DEPLOYMENTS" in rpt

    def test_contains_separator(self):
        r = self._make_record("EXIT_EXTENSION")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert "═" in rpt

    def test_contains_active_label(self):
        r = self._make_record("EXIT_EXTENSION")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert "ACTIVE:" in rpt

    def test_contains_candidate_name(self):
        r = self._make_record("EXIT_EXTENSION")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert "EXIT_EXTENSION" in rpt

    def test_contains_planned_change(self):
        r = self._make_record("PRIORITY")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert r.planned_change in rpt

    def test_contains_duration(self):
        r = self._make_record("EXIT_EXTENSION")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert "30d" in rpt

    def test_contains_date(self):
        r = self._make_record("EXIT_EXTENSION")
        s = self._make_summary([r])
        rpt = format_shadow_deployment_report([r], s)
        assert "2026-07-20" in rpt

    def test_active_count_in_report(self):
        records = [self._make_record(c) for c in ["EXIT_EXTENSION", "PRIORITY"]]
        s = self._make_summary(records)
        rpt = format_shadow_deployment_report(records, s)
        assert "Active: 2" in rpt

    def test_total_deployed_in_report(self):
        records = [self._make_record(c) for c in ["EXIT_EXTENSION", "PRIORITY"]]
        s = self._make_summary(records)
        rpt = format_shadow_deployment_report(records, s)
        assert "Total Deployed: 2" in rpt

    def test_all_five_candidates_in_report(self):
        records = [self._make_record(c) for c in _CANDIDATE_ORDER]
        s = self._make_summary(records)
        rpt = format_shadow_deployment_report(records, s)
        for ct in _CANDIDATE_ORDER:
            assert ct in rpt


# ── TestAppendShadowRecord ────────────────────────────────────────────────────

class TestAppendShadowRecord:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "out" / "shadow.jsonl"
        rec = ShadowDeploymentRecord("2026-07-20", "EXIT_EXTENSION", "change_x")
        append_shadow_record(rec, path)
        assert path.exists()

    def test_all_fields_present(self, tmp_path):
        path = tmp_path / "shadow.jsonl"
        rec = ShadowDeploymentRecord("2026-07-20", "EXIT_EXTENSION", "change_x")
        append_shadow_record(rec, path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-07-20"
        assert data["candidate"] == "EXIT_EXTENSION"
        assert data["planned_change"] == "change_x"
        assert data["shadow_status"] == SHADOW_STATUS_DEPLOYED
        assert data["shadow_duration_days"] == 30
        assert data["schema_version"] == SCHEMA_VERSION

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "shadow.jsonl"
        for ct in ["EXIT_EXTENSION", "PRIORITY"]:
            append_shadow_record(
                ShadowDeploymentRecord("2026-07-20", ct, "change"),
                path,
            )
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "shadow.jsonl"
        rec = ShadowDeploymentRecord("2026-07-20", "EXIT_EXTENSION", "change_x")
        append_shadow_record(rec, path)
        assert path.exists()

    def test_schema_version_correct(self, tmp_path):
        path = tmp_path / "shadow.jsonl"
        rec = ShadowDeploymentRecord("2026-07-20", "EXIT_EXTENSION", "change_x")
        append_shadow_record(rec, path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["schema_version"] == "v1"


# ── TestRunShadowDeploymentSimulator ──────────────────────────────────────────

class TestRunShadowDeploymentSimulator:
    def test_empty_plan_no_new(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.new_records == []
        assert result.deployed_records == []

    def test_non_planned_not_deployed(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION", status="OTHER")])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.new_records == []

    def test_planned_candidate_deployed(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result.new_records) == 1
        assert result.new_records[0].candidate == "EXIT_EXTENSION"
        assert result.new_records[0].shadow_status == SHADOW_STATUS_DEPLOYED

    def test_already_deployed_skipped(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        _write_jsonl(out,  [_deployed_rec("EXIT_EXTENSION")])
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.new_records == []

    def test_idempotent_second_run(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        run_shadow_deployment_simulator(plan, out, "2026-07-20")
        result2 = run_shadow_deployment_simulator(plan, out, "2026-07-21")
        assert result2.new_records == []

    def test_idempotent_different_day(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        run_shadow_deployment_simulator(plan, out, "2026-07-20")
        run_shadow_deployment_simulator(plan, out, "2026-07-25")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_multiple_planned_candidates(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [
            _plan_rec("EXIT_EXTENSION"),
            _plan_rec("PRIORITY"),
        ])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result.new_records) == 2

    def test_all_five_candidates_deployed(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec(c) for c in _CANDIDATE_ORDER])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result.new_records) == 5

    def test_planned_change_preserved(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [
            _plan_rec("EXIT_EXTENSION", planned_change="increase_exit_extension_shadow"),
        ])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.new_records[0].planned_change == "increase_exit_extension_shadow"

    def test_deployed_records_in_result(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result.deployed_records) == 1
        assert result.deployed_records[0].candidate == "EXIT_EXTENSION"

    def test_summary_deployment_count(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [
            _plan_rec("EXIT_EXTENSION"),
            _plan_rec("PRIORITY"),
        ])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.summary.deployment_count == 2

    def test_summary_active_shadow_deployments(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.summary.active_shadow_deployments == 1

    def test_summary_candidate_counts(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert "EXIT_EXTENSION" in result.summary.candidate_counts

    def test_schema_version_in_output(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        run_shadow_deployment_simulator(plan, out, "2026-07-20")
        data = json.loads(out.read_text(encoding="utf-8").strip())
        assert data["schema_version"] == "v1"

    def test_date_in_output(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        run_shadow_deployment_simulator(plan, out, "2026-07-20")
        data = json.loads(out.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-07-20"

    def test_shadow_duration_in_output(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        out = tmp_path / "shadow.jsonl"
        run_shadow_deployment_simulator(plan, out, "2026-07-20")
        data = json.loads(out.read_text(encoding="utf-8").strip())
        assert data["shadow_duration_days"] == 30

    def test_deployed_records_ordered(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [
            _plan_rec("SIGNAL_LEADER"),
            _plan_rec("EXIT_EXTENSION"),
        ])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        candidates = [r.candidate for r in result.deployed_records]
        assert candidates.index("EXIT_EXTENSION") < candidates.index("SIGNAL_LEADER")

    def test_fail_open_on_exception(self, tmp_path):
        # pass a non-existent directory path (not a file) as plan
        bad_plan = tmp_path / "nonexistent" / "plan.jsonl"
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(bad_plan, out, "2026-07-20")
        assert isinstance(result, ShadowDeploymentRunResult)
        assert result.new_records == []

    def test_multiple_plan_records_uses_latest(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        _write_jsonl(plan, [
            _plan_rec("EXIT_EXTENSION", date="2026-07-01"),
            _plan_rec("EXIT_EXTENSION", date="2026-07-10"),
        ])
        out = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result.new_records) == 1

    def test_pre_existing_deployed_included_in_summary(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("PRIORITY")])
        _write_jsonl(out,  [_deployed_rec("EXIT_EXTENSION")])
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        candidates = [r.candidate for r in result.deployed_records]
        assert "EXIT_EXTENSION" in candidates
        assert "PRIORITY" in candidates

    def test_empty_plan_summary_zeros(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert result.summary.active_shadow_deployments == 0
        assert result.summary.deployment_count == 0

    def test_result_date_matches_input(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        result = run_shadow_deployment_simulator(plan, out, "2026-08-01")
        assert result.date == "2026-08-01"


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_cycle_all_candidates(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [
            _plan_rec(ct, planned_change=f"change_{ct.lower()}")
            for ct in _CANDIDATE_ORDER
        ])
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result.new_records) == 5
        assert result.summary.deployment_count == 5
        assert set(result.summary.candidate_counts.keys()) == set(_CANDIDATE_ORDER)

    def test_report_format_integrated(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        rpt = format_shadow_deployment_report(result.deployed_records, result.summary)
        assert "SHADOW DEPLOYMENTS" in rpt
        assert "EXIT_EXTENSION" in rpt

    def test_idempotent_three_runs(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        for day in ["2026-07-20", "2026-07-21", "2026-07-22"]:
            run_shadow_deployment_simulator(plan, out, day)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_candidate_order_invariant(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec(c) for c in reversed(_CANDIDATE_ORDER)])
        result = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        for i, rec in enumerate(result.deployed_records):
            assert rec.candidate == _CANDIDATE_ORDER[i]

    def test_shadow_status_constant(self):
        assert SHADOW_STATUS_DEPLOYED == "DEPLOYED"

    def test_plan_status_constant(self):
        assert _PLAN_STATUS_PLANNED == "PLANNED"

    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "v1"

    def test_report_empty_when_no_deployed(self, tmp_path):
        summary = ShadowDeploymentSummary(0, 0, {})
        rpt = format_shadow_deployment_report([], summary)
        assert rpt == ""

    def test_partial_deploy_then_add_more(self, tmp_path):
        plan = tmp_path / "plan.jsonl"
        out  = tmp_path / "shadow.jsonl"
        _write_jsonl(plan, [_plan_rec("EXIT_EXTENSION")])
        result1 = run_shadow_deployment_simulator(plan, out, "2026-07-20")
        assert len(result1.new_records) == 1

        # add another candidate to plan
        with plan.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_plan_rec("PRIORITY")) + "\n")
        result2 = run_shadow_deployment_simulator(plan, out, "2026-07-21")
        assert len(result2.new_records) == 1
        assert result2.new_records[0].candidate == "PRIORITY"
        assert result2.summary.deployment_count == 2
