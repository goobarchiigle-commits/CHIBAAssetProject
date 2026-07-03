"""
tests/test_system_health_audit.py — Phase 7A tests

Full coverage of:
  Constants / dataclasses / helpers /
  telemetry file check / state transition check /
  inactive experiment check / orphaned registry check /
  daily summary builder / report I/O /
  main orchestrator / report formatter /
  edge cases / fail-open guarantees
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.analytics.system_health_audit import (
    # constants
    STATUS_HEALTHY, STATUS_WARNING, STATUS_CRITICAL,
    SCHEMA_VERSION,
    STALE_THRESHOLD_DAYS, INACTIVE_EXPERIMENT_DAYS, ORPHAN_MIN_DAYS,
    CHECK_EVIDENCE_GENERATION, CHECK_SHADOW_GENERATION,
    CHECK_PROMOTION_GENERATION, CHECK_GATE_GENERATION,
    CHECK_EXPERIMENT_ASSIGNMENT, CHECK_CAPITAL_ESCALATION,
    CHECK_ROLLBACK_EVALUATION, CHECK_IMPROVEMENT_RANKING,
    CHECK_STATE_TRANSITIONS, CHECK_INACTIVE_EXPERIMENTS,
    CHECK_ORPHANED_REGISTRY,
    # dataclasses
    CheckResult, DailySummary, SystemHealthReport,
    # helpers
    _safe_float, _safe_int, _load_jsonl, _load_registry_json,
    _parse_date, _latest_date_in_records, _days_since,
    _today_dt, _aggregate_status,
    # checks
    _check_telemetry_file, _check_state_transitions,
    _check_inactive_experiments, _check_orphaned_registry,
    # summary
    _build_daily_summary,
    # I/O
    _report_to_dict, write_health_report, load_health_report,
    # main
    run_system_health_audit,
    # formatter
    format_health_report,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

TODAY = "2026-05-30"
TODAY_DT = datetime(2026, 5, 30)


def _recent(days_ago: int = 0) -> str:
    return (TODAY_DT - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_registry(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"experiments": records}), encoding="utf-8")


def _reg(eid: str, status: str = "ACTIVE", level: int = 0,
         start: str = "2026-01-01") -> dict:
    return {"experiment_id": eid, "status": status,
            "allocation_level": level, "start_date": start}


def _perf(eid: str, date: str = TODAY) -> dict:
    return {"experiment_id": eid, "group": "EXPERIMENT", "date": date,
            "pnl": 100.0, "win_rate": 0.6}


def _esc(eid: str, decision: str = "ESCALATE_LEVEL_1", date: str = TODAY) -> dict:
    return {"experiment_id": eid, "decision": decision, "evaluation_date": date}


def _rb(eid: str, decision: str = "HEALTHY", date: str = TODAY) -> dict:
    return {"experiment_id": eid, "decision": decision, "evaluation_date": date}


def _rank(eid: str, rank: int = 1, date: str = TODAY) -> dict:
    return {"experiment_id": eid, "rank": rank, "ranking_score": 80.0, "date": date}


def _assign(eid: str, date: str = TODAY) -> dict:
    return {"experiment_id": eid, "assigned_at": date, "group": "EXPERIMENT"}


def _make_files(tmp_path: Path) -> dict:
    """Return a dict of all required file paths (all in tmp_path)."""
    names = [
        "evidence", "shadow", "promotion", "gate", "assignment",
        "escalation", "rollback", "ranking", "performance",
    ]
    files = {n: tmp_path / f"{n}.jsonl" for n in names}
    files["registry"] = tmp_path / "registry.json"
    files["output"]   = tmp_path / "system_health_report.json"
    return files


def _run(files: dict, today: str = TODAY) -> SystemHealthReport:
    return run_system_health_audit(
        evidence_file=files["evidence"],
        shadow_file=files["shadow"],
        promotion_file=files["promotion"],
        gate_file=files["gate"],
        assignment_file=files["assignment"],
        escalation_file=files["escalation"],
        rollback_file=files["rollback"],
        ranking_file=files["ranking"],
        registry_file=files["registry"],
        performance_file=files["performance"],
        output_file=files["output"],
        today=today,
    )


# ── 1. Constants ──────────────────────────────────────────────────────────────

def test_status_constants():
    assert STATUS_HEALTHY  == "HEALTHY"
    assert STATUS_WARNING  == "WARNING"
    assert STATUS_CRITICAL == "CRITICAL"


def test_schema_version():
    assert SCHEMA_VERSION == "v1"


def test_stale_threshold_positive():
    assert STALE_THRESHOLD_DAYS > 0


def test_inactive_threshold_positive():
    assert INACTIVE_EXPERIMENT_DAYS > 0


def test_check_name_constants():
    assert CHECK_EVIDENCE_GENERATION == "evidence_generation"
    assert CHECK_STATE_TRANSITIONS   == "state_transitions"
    assert CHECK_INACTIVE_EXPERIMENTS == "inactive_experiments"
    assert CHECK_ORPHANED_REGISTRY   == "orphaned_registry_entries"


# ── 2. Dataclasses ────────────────────────────────────────────────────────────

def test_check_result_fields():
    c = CheckResult(check_name="foo", status=STATUS_HEALTHY, message="ok")
    assert c.details == {}


def test_daily_summary_fields():
    s = DailySummary(
        active_experiments=3,
        escalated_experiments=["A"],
        rollback_candidates=[],
        rolled_back_experiments=[],
        top_ranked_improvements=["B"],
    )
    assert s.active_experiments == 3


def test_system_health_report_schema():
    r = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[], summary=DailySummary(0, [], [], [], []),
    )
    assert r.schema_version == SCHEMA_VERSION


# ── 3. Helpers ────────────────────────────────────────────────────────────────

def test_safe_float_normal():
    assert _safe_float(1.5) == 1.5


def test_safe_float_none():
    assert _safe_float(None) == 0.0


def test_safe_int_str():
    assert _safe_int("3") == 3


def test_safe_int_bad():
    assert _safe_int("x") == 0


def test_load_jsonl_missing(tmp_path):
    p = tmp_path / "x.jsonl"
    assert _load_jsonl(p) == []


def test_load_jsonl_reads(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, [{"a": 1}, {"b": 2}])
    rows = _load_jsonl(p)
    assert len(rows) == 2


def test_load_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\nBAD\n{"b":2}\n', encoding="utf-8")
    rows = _load_jsonl(p)
    assert len(rows) == 2


def test_load_registry_json_list(tmp_path):
    p = tmp_path / "reg.json"
    _write_registry(p, [_reg("EXP_A"), _reg("EXP_B")])
    rows = _load_registry_json(p)
    assert len(rows) == 2


def test_load_registry_json_missing(tmp_path):
    assert _load_registry_json(tmp_path / "x.json") == []


def test_load_registry_json_bad(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("NOT_JSON", encoding="utf-8")
    assert _load_registry_json(p) == []


def test_parse_date_iso():
    d = _parse_date("2026-05-30")
    assert d is not None
    assert d.year == 2026 and d.month == 5 and d.day == 30


def test_parse_date_empty():
    assert _parse_date("") is None


def test_parse_date_bad():
    assert _parse_date("not-a-date") is None


def test_latest_date_in_records_basic():
    rows = [{"date": "2026-05-28"}, {"date": "2026-05-30"}, {"date": "2026-05-29"}]
    d = _latest_date_in_records(rows)
    assert d is not None
    assert d.day == 30


def test_latest_date_in_records_empty():
    assert _latest_date_in_records([]) is None


def test_days_since_today():
    d = datetime(2026, 5, 30)
    assert _days_since(d, TODAY_DT) == 0


def test_days_since_yesterday():
    d = datetime(2026, 5, 29)
    assert _days_since(d, TODAY_DT) == 1


def test_days_since_none():
    assert _days_since(None, TODAY_DT) is None


def test_today_dt_parses():
    d = _today_dt("2026-05-30")
    assert d.year == 2026


def test_aggregate_status_healthy():
    assert _aggregate_status([STATUS_HEALTHY, STATUS_HEALTHY]) == STATUS_HEALTHY


def test_aggregate_status_warning():
    assert _aggregate_status([STATUS_HEALTHY, STATUS_WARNING]) == STATUS_WARNING


def test_aggregate_status_critical():
    assert _aggregate_status([STATUS_WARNING, STATUS_CRITICAL]) == STATUS_CRITICAL


def test_aggregate_status_empty():
    assert _aggregate_status([]) == STATUS_HEALTHY


# ── 4. _check_telemetry_file ──────────────────────────────────────────────────

def test_check_telemetry_missing(tmp_path):
    result = _check_telemetry_file("test_check", tmp_path / "x.jsonl", TODAY_DT)
    assert result.status == STATUS_WARNING
    assert "not found" in result.message.lower()


def test_check_telemetry_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    result = _check_telemetry_file("test_check", p, TODAY_DT)
    assert result.status == STATUS_WARNING


def test_check_telemetry_fresh_record(tmp_path):
    p = tmp_path / "fresh.jsonl"
    _write_jsonl(p, [{"date": TODAY}])
    result = _check_telemetry_file("test_check", p, TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_check_telemetry_stale_record(tmp_path):
    p = tmp_path / "stale.jsonl"
    stale_date = _recent(STALE_THRESHOLD_DAYS + 2)
    _write_jsonl(p, [{"date": stale_date}])
    result = _check_telemetry_file("test_check", p, TODAY_DT)
    assert result.status == STATUS_WARNING
    assert "stale" in result.message.lower()


def test_check_telemetry_no_date_field(tmp_path):
    p = tmp_path / "nodates.jsonl"
    _write_jsonl(p, [{"value": 1}])
    result = _check_telemetry_file("test_check", p, TODAY_DT)
    assert result.status == STATUS_WARNING


def test_check_telemetry_check_name_preserved(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, [{"date": TODAY}])
    result = _check_telemetry_file("my_check", p, TODAY_DT)
    assert result.check_name == "my_check"


def test_check_telemetry_details_populated(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, [{"date": TODAY}])
    result = _check_telemetry_file("my_check", p, TODAY_DT)
    assert "record_count" in result.details


# ── 5. _check_state_transitions ───────────────────────────────────────────────

def test_state_transitions_healthy():
    registry = [_reg("EXP_A", status="ACTIVE", level=1)]
    rollback = [_rb("EXP_A", "HEALTHY")]
    escalation = [_esc("EXP_A", "ESCALATE_LEVEL_1")]
    result = _check_state_transitions(registry, rollback, escalation)
    assert result.status == STATUS_HEALTHY


def test_state_transitions_rolled_back_but_active():
    registry = [_reg("EXP_A", status="ACTIVE")]
    rollback = [_rb("EXP_A", "ROLLED_BACK")]
    escalation = []
    result = _check_state_transitions(registry, rollback, escalation)
    assert result.status == STATUS_WARNING
    assert "EXP_A" in str(result.details)


def test_state_transitions_escalated_but_level_zero():
    registry = [_reg("EXP_A", status="ACTIVE", level=0)]
    rollback = []
    escalation = [_esc("EXP_A", "FULL_SCALE")]
    result = _check_state_transitions(registry, rollback, escalation)
    assert result.status == STATUS_WARNING


def test_state_transitions_empty_inputs():
    result = _check_state_transitions([], [], [])
    assert result.status == STATUS_HEALTHY


def test_state_transitions_deescalate_not_flagged():
    registry = [_reg("EXP_A", status="ACTIVE", level=0)]
    rollback = []
    escalation = [_esc("EXP_A", "DEESCALATE")]
    result = _check_state_transitions(registry, rollback, escalation)
    assert result.status == STATUS_HEALTHY


def test_state_transitions_rolled_back_status_ok():
    # If experiment is ROLLED_BACK in registry, no conflict even if in rollback file
    registry = [_reg("EXP_A", status="ROLLED_BACK")]
    rollback = [_rb("EXP_A", "ROLLED_BACK")]
    escalation = []
    result = _check_state_transitions(registry, rollback, escalation)
    assert result.status == STATUS_HEALTHY


def test_state_transitions_multiple_issues():
    registry = [
        _reg("EXP_A", status="ACTIVE"),
        _reg("EXP_B", status="ACTIVE", level=0),
    ]
    rollback   = [_rb("EXP_A", "ROLLED_BACK")]
    escalation = [_esc("EXP_B", "ESCALATE_LEVEL_2")]
    result = _check_state_transitions(registry, rollback, escalation)
    assert result.status == STATUS_WARNING
    assert len(result.details.get("issues", [])) == 2


# ── 6. _check_inactive_experiments ────────────────────────────────────────────

def test_inactive_experiments_all_recent(tmp_path):
    registry = [_reg("EXP_A", start="2026-01-01")]
    perf = [_perf("EXP_A", date=TODAY)]
    result = _check_inactive_experiments(registry, perf, TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_inactive_experiments_stale_perf():
    stale = _recent(INACTIVE_EXPERIMENT_DAYS + 2)
    registry = [_reg("EXP_A", start="2026-01-01")]
    perf = [_perf("EXP_A", date=stale)]
    result = _check_inactive_experiments(registry, perf, TODAY_DT)
    assert result.status == STATUS_WARNING
    assert "EXP_A" in str(result.details)


def test_inactive_experiments_no_perf():
    registry = [_reg("EXP_A", start="2026-01-01")]
    result = _check_inactive_experiments(registry, [], TODAY_DT)
    assert result.status == STATUS_WARNING


def test_inactive_experiments_too_new_skipped():
    # Experiment started today — no perf yet is fine
    registry = [_reg("EXP_A", start=TODAY)]
    result = _check_inactive_experiments(registry, [], TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_inactive_experiments_non_active_ignored():
    registry = [_reg("EXP_A", status="ROLLED_BACK", start="2026-01-01")]
    result = _check_inactive_experiments(registry, [], TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_inactive_experiments_empty_registry():
    result = _check_inactive_experiments([], [], TODAY_DT)
    assert result.status == STATUS_HEALTHY


# ── 7. _check_orphaned_registry ───────────────────────────────────────────────

def test_orphaned_registry_no_orphans():
    registry = [_reg("EXP_A", start="2026-01-01")]
    assign   = [_assign("EXP_A")]
    perf     = []
    result = _check_orphaned_registry(registry, assign, perf, TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_orphaned_registry_orphan_detected():
    registry = [_reg("EXP_A", start="2026-01-01")]
    assign   = []
    perf     = []
    result = _check_orphaned_registry(registry, assign, perf, TODAY_DT)
    assert result.status == STATUS_WARNING
    assert "EXP_A" in str(result.details)


def test_orphaned_registry_too_new_excluded():
    registry = [_reg("EXP_A", start=TODAY)]
    result = _check_orphaned_registry(registry, [], [], TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_orphaned_registry_perf_resolves_orphan():
    registry = [_reg("EXP_A", start="2026-01-01")]
    perf     = [_perf("EXP_A")]
    result = _check_orphaned_registry(registry, [], perf, TODAY_DT)
    assert result.status == STATUS_HEALTHY


def test_orphaned_registry_empty():
    result = _check_orphaned_registry([], [], [], TODAY_DT)
    assert result.status == STATUS_HEALTHY


# ── 8. _build_daily_summary ───────────────────────────────────────────────────

def test_build_daily_summary_active_count():
    registry = [_reg("EXP_A"), _reg("EXP_B"), _reg("EXP_C", status="ROLLED_BACK")]
    summary = _build_daily_summary(registry, [], [], [], TODAY)
    assert summary.active_experiments == 2


def test_build_daily_summary_escalated():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_2"), _esc("EXP_B", "HOLD")]
    summary = _build_daily_summary([], esc, [], [], TODAY)
    assert "EXP_A" in summary.escalated_experiments
    assert "EXP_B" not in summary.escalated_experiments


def test_build_daily_summary_rollback_candidates():
    rb = [_rb("EXP_A", "ROLLBACK_CANDIDATE"), _rb("EXP_B", "HEALTHY")]
    summary = _build_daily_summary([], [], rb, [], TODAY)
    assert "EXP_A" in summary.rollback_candidates
    assert "EXP_B" not in summary.rollback_candidates


def test_build_daily_summary_rolled_back():
    rb = [_rb("EXP_A", "ROLLED_BACK")]
    summary = _build_daily_summary([], [], rb, [], TODAY)
    assert "EXP_A" in summary.rolled_back_experiments


def test_build_daily_summary_top_ranked():
    ranking = [
        _rank("EXP_B", rank=1),
        _rank("EXP_A", rank=2),
        _rank("EXP_C", rank=3),
    ]
    summary = _build_daily_summary([], [], [], ranking, TODAY)
    assert summary.top_ranked_improvements[0] == "EXP_B"
    assert summary.top_ranked_improvements[1] == "EXP_A"


def test_build_daily_summary_empty_inputs():
    summary = _build_daily_summary([], [], [], [], TODAY)
    assert summary.active_experiments == 0
    assert summary.escalated_experiments == []


def test_build_daily_summary_dedup_escalated():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_1"), _esc("EXP_A", "ESCALATE_LEVEL_2")]
    summary = _build_daily_summary([], esc, [], [], TODAY)
    assert summary.escalated_experiments.count("EXP_A") == 1


# ── 9. Report I/O ─────────────────────────────────────────────────────────────

def test_report_to_dict_structure():
    report = SystemHealthReport(
        audit_date="2026-05-30",
        overall_status=STATUS_HEALTHY,
        checks=[CheckResult("c1", STATUS_HEALTHY, "ok")],
        summary=DailySummary(2, ["A"], [], [], ["B"]),
    )
    d = _report_to_dict(report)
    assert d["audit_date"] == "2026-05-30"
    assert d["overall_status"] == STATUS_HEALTHY
    assert len(d["checks"]) == 1
    assert d["summary"]["active_experiments"] == 2


def test_write_and_load_health_report(tmp_path):
    path = tmp_path / "report.json"
    report = SystemHealthReport(
        audit_date="2026-05-30",
        overall_status=STATUS_HEALTHY,
        checks=[CheckResult("c1", STATUS_HEALTHY, "ok", {"a": 1})],
        summary=DailySummary(1, [], [], [], []),
    )
    write_health_report(report, path)
    loaded = load_health_report(path)
    assert loaded is not None
    assert loaded["overall_status"] == STATUS_HEALTHY
    assert loaded["checks"][0]["check_name"] == "c1"


def test_load_health_report_missing(tmp_path):
    assert load_health_report(tmp_path / "missing.json") is None


def test_load_health_report_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("NOT_JSON", encoding="utf-8")
    assert load_health_report(p) is None


def test_write_health_report_creates_dirs(tmp_path):
    path = tmp_path / "sub" / "sub2" / "report.json"
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[], summary=DailySummary(0, [], [], [], []),
    )
    write_health_report(report, path)
    assert path.exists()


# ── 10. Main orchestrator ─────────────────────────────────────────────────────

def test_run_all_missing_files(tmp_path):
    files = _make_files(tmp_path)
    report = _run(files)
    assert isinstance(report, SystemHealthReport)
    assert report.overall_status in (STATUS_HEALTHY, STATUS_WARNING, STATUS_CRITICAL)
    assert len(report.checks) > 0


def test_run_writes_output_file(tmp_path):
    files = _make_files(tmp_path)
    _run(files)
    assert files["output"].exists()


def test_run_output_is_valid_json(tmp_path):
    files = _make_files(tmp_path)
    _run(files)
    loaded = load_health_report(files["output"])
    assert loaded is not None
    assert "overall_status" in loaded


def test_run_all_healthy_files(tmp_path):
    files = _make_files(tmp_path)
    for name in ["evidence", "shadow", "promotion", "gate", "assignment",
                 "escalation", "rollback", "ranking", "performance"]:
        _write_jsonl(files[name], [{"date": TODAY}])
    _write_registry(files["registry"], [_reg("EXP_A", start="2026-01-01")])
    _write_jsonl(files["assignment"], [_assign("EXP_A")])
    report = _run(files)
    assert report.overall_status in (STATUS_HEALTHY, STATUS_WARNING)


def test_run_detects_stale_evidence(tmp_path):
    files = _make_files(tmp_path)
    stale = _recent(STALE_THRESHOLD_DAYS + 3)
    _write_jsonl(files["evidence"], [{"date": stale}])
    report = _run(files)
    ev_check = next(c for c in report.checks if c.check_name == CHECK_EVIDENCE_GENERATION)
    assert ev_check.status == STATUS_WARNING


def test_run_detects_rolled_back_active(tmp_path):
    files = _make_files(tmp_path)
    _write_registry(files["registry"], [_reg("EXP_A", status="ACTIVE")])
    _write_jsonl(files["rollback"], [_rb("EXP_A", "ROLLED_BACK")])
    report = _run(files)
    st_check = next(c for c in report.checks if c.check_name == CHECK_STATE_TRANSITIONS)
    assert st_check.status == STATUS_WARNING


def test_run_detects_inactive_experiment(tmp_path):
    files = _make_files(tmp_path)
    _write_registry(files["registry"], [_reg("EXP_A", start="2026-01-01")])
    stale = _recent(INACTIVE_EXPERIMENT_DAYS + 3)
    _write_jsonl(files["performance"], [_perf("EXP_A", date=stale)])
    report = _run(files)
    ie_check = next(c for c in report.checks if c.check_name == CHECK_INACTIVE_EXPERIMENTS)
    assert ie_check.status == STATUS_WARNING


def test_run_detects_orphaned_registry(tmp_path):
    files = _make_files(tmp_path)
    _write_registry(files["registry"], [_reg("EXP_A", start="2026-01-01")])
    report = _run(files)
    or_check = next(c for c in report.checks if c.check_name == CHECK_ORPHANED_REGISTRY)
    assert or_check.status == STATUS_WARNING


def test_run_summary_active_count(tmp_path):
    files = _make_files(tmp_path)
    _write_registry(files["registry"], [
        _reg("EXP_A"), _reg("EXP_B"), _reg("EXP_C", status="ROLLED_BACK")
    ])
    report = _run(files)
    assert report.summary.active_experiments == 2


def test_run_summary_top_ranked(tmp_path):
    files = _make_files(tmp_path)
    _write_jsonl(files["ranking"], [
        _rank("EXP_A", rank=1),
        _rank("EXP_B", rank=2),
    ])
    report = _run(files)
    assert report.summary.top_ranked_improvements[0] == "EXP_A"


def test_run_has_expected_check_count(tmp_path):
    files = _make_files(tmp_path)
    report = _run(files)
    # 8 telemetry checks + 3 structural checks = 11
    assert len(report.checks) == 11


def test_run_schema_version(tmp_path):
    files = _make_files(tmp_path)
    report = _run(files)
    loaded = load_health_report(files["output"])
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_run_fail_open_bad_registry(tmp_path):
    files = _make_files(tmp_path)
    files["registry"].parent.mkdir(parents=True, exist_ok=True)
    files["registry"].write_text("BAD_JSON", encoding="utf-8")
    # Should not raise
    report = _run(files)
    assert isinstance(report, SystemHealthReport)


def test_run_overwrites_previous_report(tmp_path):
    files = _make_files(tmp_path)
    _run(files, today="2026-05-29")
    _run(files, today="2026-05-30")
    loaded = load_health_report(files["output"])
    assert loaded["audit_date"] == "2026-05-30"


# ── 11. Formatter ─────────────────────────────────────────────────────────────

def test_format_health_report_contains_header():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[], summary=DailySummary(0, [], [], [], []),
    )
    rpt = format_health_report(report)
    assert "SYSTEM HEALTH AUDIT" in rpt


def test_format_health_report_shows_status():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_WARNING,
        checks=[], summary=DailySummary(0, [], [], [], []),
    )
    rpt = format_health_report(report)
    assert STATUS_WARNING in rpt


def test_format_health_report_shows_check():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[CheckResult("my_check", STATUS_HEALTHY, "all good")],
        summary=DailySummary(0, [], [], [], []),
    )
    rpt = format_health_report(report)
    assert "my_check" in rpt
    assert "all good" in rpt


def test_format_health_report_shows_active_experiments():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[],
        summary=DailySummary(5, [], [], [], []),
    )
    rpt = format_health_report(report)
    assert "5" in rpt


def test_format_health_report_shows_top_improvements():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[],
        summary=DailySummary(0, [], [], [], ["EXP_TOP"]),
    )
    rpt = format_health_report(report)
    assert "EXP_TOP" in rpt


def test_format_health_report_shows_date():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[], summary=DailySummary(0, [], [], [], []),
    )
    rpt = format_health_report(report)
    assert "2026-05-30" in rpt


def test_format_health_report_returns_str():
    report = SystemHealthReport(
        audit_date="2026-05-30", overall_status=STATUS_HEALTHY,
        checks=[], summary=DailySummary(0, [], [], [], []),
    )
    assert isinstance(format_health_report(report), str)


# ── 12. Integration ───────────────────────────────────────────────────────────

def test_full_healthy_system(tmp_path):
    """End-to-end: fully healthy system with all telemetry fresh."""
    files = _make_files(tmp_path)

    for name in ["evidence", "shadow", "promotion", "gate"]:
        _write_jsonl(files[name], [{"date": TODAY}])

    _write_jsonl(files["assignment"], [_assign("EXP_A"), _assign("EXP_B")])
    _write_jsonl(files["escalation"], [_esc("EXP_A", "ESCALATE_LEVEL_1")])
    _write_jsonl(files["rollback"],   [_rb("EXP_A", "HEALTHY")])
    _write_jsonl(files["ranking"],    [_rank("EXP_A", 1), _rank("EXP_B", 2)])
    _write_jsonl(files["performance"], [_perf("EXP_A"), _perf("EXP_B")])
    _write_registry(files["registry"], [
        _reg("EXP_A", level=1, start="2026-01-01"),
        _reg("EXP_B", level=0, start="2026-01-01"),
    ])

    report = _run(files)

    # State transitions: EXP_A escalated to level=1 → consistent
    st_check = next(c for c in report.checks if c.check_name == CHECK_STATE_TRANSITIONS)
    assert st_check.status == STATUS_HEALTHY

    # Inactive: both have fresh perf
    ie_check = next(c for c in report.checks if c.check_name == CHECK_INACTIVE_EXPERIMENTS)
    assert ie_check.status == STATUS_HEALTHY

    # Orphan: both in assignment
    or_check = next(c for c in report.checks if c.check_name == CHECK_ORPHANED_REGISTRY)
    assert or_check.status == STATUS_HEALTHY

    # Summary
    assert report.summary.active_experiments == 2
    assert report.summary.top_ranked_improvements[0] == "EXP_A"

    # Report written
    loaded = load_health_report(files["output"])
    assert loaded is not None
    assert loaded["overall_status"] in (STATUS_HEALTHY, STATUS_WARNING)


def test_full_degraded_system(tmp_path):
    """End-to-end: multiple issues detected → WARNING or CRITICAL."""
    files = _make_files(tmp_path)

    stale = _recent(STALE_THRESHOLD_DAYS + 5)
    _write_jsonl(files["evidence"], [{"date": stale}])
    # Other telemetry files missing
    _write_registry(files["registry"], [_reg("EXP_A", status="ACTIVE", start="2026-01-01")])
    _write_jsonl(files["rollback"],    [_rb("EXP_A", "ROLLED_BACK")])

    report = _run(files)
    assert report.overall_status in (STATUS_WARNING, STATUS_CRITICAL)

    checks_by_name = {c.check_name: c for c in report.checks}
    assert checks_by_name[CHECK_EVIDENCE_GENERATION].status == STATUS_WARNING
    assert checks_by_name[CHECK_STATE_TRANSITIONS].status == STATUS_WARNING
