"""
tests/test_weekly_executive_review.py — Phase 7B tests

Full coverage of:
  Constants / dataclasses / helpers /
  _build_system_health / _build_experiment_summary /
  _build_improvement_ranking / _build_capital_deployment /
  _build_rollback_activity / _build_performance_summary /
  _build_key_alerts / serialization / I/O /
  main orchestrator / formatter / edge cases / fail-open
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.analytics.weekly_executive_review import (
    # constants
    SCHEMA_VERSION, LOOKBACK_DAYS,
    ALERT_CRITICAL, ALERT_WARNING, ALERT_INFO,
    _ALLOC_LEVEL_PCT, _ESC_DECISIONS,
    # dataclasses
    SystemHealthSection, ExperimentEntry, ExperimentSummarySection,
    RankedExperiment, ImprovementRankingSection,
    AllocationChange, CapitalDeploymentSection,
    RollbackEvent, RollbackActivitySection,
    PerformanceSummarySection, Alert, KeyAlertsSection,
    WeeklyExecutiveReview,
    # helpers
    _safe_float, _safe_int, _parse_date,
    _load_jsonl, _load_json, _load_registry,
    _within_window, _today_naive, _alloc_pct,
    # section builders
    _build_system_health, _build_experiment_summary,
    _build_improvement_ranking, _build_capital_deployment,
    _build_rollback_activity, _build_performance_summary,
    _build_key_alerts,
    # I/O
    _review_to_dict, write_review, load_review,
    # main
    run_weekly_executive_review,
    # formatter
    format_executive_review,
)


# ── Test fixtures ─────────────────────────────────────────────────────────────

TODAY      = "2026-05-31"
TODAY_DT   = datetime(2026, 5, 31)
WEEK_AGO   = (TODAY_DT - timedelta(days=6)).strftime("%Y-%m-%d")
OLD_DATE   = (TODAY_DT - timedelta(days=10)).strftime("%Y-%m-%d")


def _d(days_ago: int = 0) -> str:
    return (TODAY_DT - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"experiments": rows}), encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _reg(eid: str, status: str = "ACTIVE", level: int = 0,
         start: str = "2026-01-01") -> dict:
    return {"experiment_id": eid, "status": status,
            "allocation_level": level, "start_date": start}


def _perf(eid: str, date: str = TODAY, pnl: float = 100.0,
          win_rate: float = 0.6, expectancy: float = 0.5,
          max_drawdown: float = -0.05, trades: int = 5,
          group: str = "EXPERIMENT") -> dict:
    return {"experiment_id": eid, "group": group, "date": date,
            "pnl": pnl, "win_rate": win_rate, "expectancy": expectancy,
            "max_drawdown": max_drawdown, "trades": trades}


def _esc(eid: str, decision: str = "ESCALATE_LEVEL_1", date: str = TODAY,
         prev: int = 0, new: int = 1) -> dict:
    return {"experiment_id": eid, "decision": decision,
            "evaluation_date": date, "previous_level": prev, "new_level": new,
            "allocation_pct": _ALLOC_LEVEL_PCT.get(new, 0.005)}


def _rb(eid: str, decision: str = "HEALTHY", date: str = TODAY,
        reasons: list | None = None) -> dict:
    return {"experiment_id": eid, "decision": decision,
            "evaluation_date": date, "trigger_reasons": reasons or []}


def _rank(eid: str, rank: int = 1, score: float = 75.0,
          pnl: float = 1000.0, wr: float = 0.6, dd: float = -0.05,
          date: str = TODAY) -> dict:
    return {"experiment_id": eid, "rank": rank, "ranking_score": score,
            "date": date,
            "metrics": {"total_pnl": pnl, "win_rate": wr, "max_drawdown": dd}}


def _health_report(status: str = "HEALTHY", checks: list | None = None) -> dict:
    checks = checks or []
    return {
        "overall_status": status,
        "schema_version": "v1",
        "checks": checks,
        "summary": {"active_experiments": 2},
    }


def _check(name: str, status: str = "HEALTHY", msg: str = "ok") -> dict:
    return {"check_name": name, "status": status, "message": msg, "details": {}}


def _make_files(tmp_path: Path) -> dict:
    names = ["evidence", "shadow", "promotion", "gate",
             "escalation", "rollback", "ranking", "performance"]
    files = {n: tmp_path / f"{n}.jsonl" for n in names}
    files["registry"]     = tmp_path / "registry.json"
    files["health"]       = tmp_path / "health.json"
    files["output"]       = tmp_path / "weekly_executive_review.json"
    return files


def _run(files: dict, today: str = TODAY) -> WeeklyExecutiveReview:
    return run_weekly_executive_review(
        evidence_file=files["evidence"],
        shadow_file=files["shadow"],
        promotion_file=files["promotion"],
        gate_file=files["gate"],
        registry_file=files["registry"],
        escalation_file=files["escalation"],
        rollback_file=files["rollback"],
        ranking_file=files["ranking"],
        health_report_file=files["health"],
        performance_file=files["performance"],
        output_file=files["output"],
        today=today,
    )


def _make_default_review() -> WeeklyExecutiveReview:
    h = SystemHealthSection("HEALTHY", [], [], [], 11, 0, 0)
    e = ExperimentSummarySection([], [], [], [], 0)
    r = ImprovementRankingSection([], [], 0)
    c = CapitalDeploymentSection({}, [], 0.0)
    rb = RollbackActivitySection([], [], [])
    p = PerformanceSummarySection(0.0, 0.0, 0.0, 0.0, 0, 0)
    k = KeyAlertsSection([], 0, 0, 0)
    return WeeklyExecutiveReview(
        review_date=TODAY, lookback_days=LOOKBACK_DAYS,
        system_health=h, experiment_summary=e, improvement_ranking=r,
        capital_deployment=c, rollback_activity=rb,
        performance_summary=p, key_alerts=k,
    )


# ── 1. Constants ──────────────────────────────────────────────────────────────

def test_schema_version():
    assert SCHEMA_VERSION == "v1"


def test_lookback_days_positive():
    assert LOOKBACK_DAYS > 0


def test_alert_constants():
    assert ALERT_CRITICAL == "CRITICAL"
    assert ALERT_WARNING  == "WARNING"
    assert ALERT_INFO     == "INFO"


def test_esc_decisions_set():
    assert "ESCALATE_LEVEL_1" in _ESC_DECISIONS
    assert "FULL_SCALE"       in _ESC_DECISIONS
    assert "HOLD"         not in _ESC_DECISIONS


def test_alloc_level_pct_level0():
    assert _alloc_pct(0) == pytest.approx(0.005)


def test_alloc_level_pct_level5():
    assert _alloc_pct(5) == pytest.approx(0.250)


def test_alloc_level_pct_clamps():
    assert _alloc_pct(-1) == _alloc_pct(0)
    assert _alloc_pct(99) == _alloc_pct(5)


# ── 2. Dataclasses ────────────────────────────────────────────────────────────

def test_system_health_section_fields():
    s = SystemHealthSection("HEALTHY", [], [], [], 11, 0, 0)
    assert s.overall_status == "HEALTHY"
    assert s.check_count == 11


def test_experiment_entry_fields():
    e = ExperimentEntry("EXP_A", "ACTIVE", 2, 0.02, "2026-01-01")
    assert e.allocation_pct == 0.02


def test_ranked_experiment_fields():
    r = RankedExperiment(1, "EXP_A", 80.0, 1000.0, 0.6, -0.05)
    assert r.rank == 1


def test_alert_defaults():
    a = Alert(ALERT_INFO, "info msg")
    assert a.experiment_id == ""
    assert a.source == ""


def test_weekly_executive_review_schema():
    rev = _make_default_review()
    assert rev.schema_version == SCHEMA_VERSION


# ── 3. Helpers ────────────────────────────────────────────────────────────────

def test_safe_float_normal():
    assert _safe_float(2.5) == 2.5


def test_safe_float_none():
    assert _safe_float(None) == 0.0


def test_safe_float_nan():
    import math
    assert _safe_float(float("nan")) == 0.0


def test_safe_int_str():
    assert _safe_int("3") == 3


def test_safe_int_bad():
    assert _safe_int("x") == 0


def test_parse_date_iso():
    d = _parse_date("2026-05-31")
    assert d is not None and d.day == 31


def test_parse_date_empty():
    assert _parse_date("") is None


def test_load_jsonl_missing(tmp_path):
    assert _load_jsonl(tmp_path / "x.jsonl") == []


def test_load_jsonl_reads(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, [{"a": 1}, {"b": 2}])
    assert len(_load_jsonl(p)) == 2


def test_load_jsonl_bad_line_skipped(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\nBAD\n{"b":2}\n', encoding="utf-8")
    assert len(_load_jsonl(p)) == 2


def test_load_json_missing(tmp_path):
    assert _load_json(tmp_path / "x.json") is None


def test_load_json_reads(tmp_path):
    p = tmp_path / "x.json"
    _write_json(p, {"k": "v"})
    assert _load_json(p)["k"] == "v"


def test_load_registry_list(tmp_path):
    p = tmp_path / "r.json"
    _write_registry(p, [_reg("A"), _reg("B")])
    rows = _load_registry(p)
    assert len(rows) == 2


def test_load_registry_missing(tmp_path):
    assert _load_registry(tmp_path / "r.json") == []


def test_within_window_today():
    assert _within_window(TODAY, TODAY_DT, LOOKBACK_DAYS)


def test_within_window_old():
    assert not _within_window(OLD_DATE, TODAY_DT, LOOKBACK_DAYS)


def test_within_window_boundary():
    cutoff = (TODAY_DT - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    assert _within_window(cutoff, TODAY_DT, LOOKBACK_DAYS)


def test_today_naive_parses():
    d = _today_naive(TODAY)
    assert d.year == 2026


# ── 4. _build_system_health ───────────────────────────────────────────────────

def test_build_system_health_none():
    s = _build_system_health(None)
    assert s.overall_status == "UNKNOWN"
    assert s.check_count == 0


def test_build_system_health_healthy():
    report = _health_report("HEALTHY", [_check("c1"), _check("c2")])
    s = _build_system_health(report)
    assert s.overall_status == "HEALTHY"
    assert s.check_count == 2
    assert s.warning_count == 0


def test_build_system_health_counts_warnings():
    report = _health_report("WARNING", [
        _check("c1", "WARNING", "stale: c1"),
        _check("c2", "HEALTHY"),
    ])
    s = _build_system_health(report)
    assert s.warning_count == 1


def test_build_system_health_detects_stale():
    report = _health_report("WARNING", [
        _check("evidence_generation", "WARNING", "Stale: last record 5d ago"),
    ])
    s = _build_system_health(report)
    assert "evidence_generation" in s.stale_telemetry


def test_build_system_health_detects_missing():
    report = _health_report("WARNING", [
        _check("shadow_generation", "WARNING", "File not found: shadow.jsonl"),
    ])
    s = _build_system_health(report)
    assert "shadow_generation" in s.missing_telemetry


def test_build_system_health_detects_orphans():
    report = _health_report("WARNING", [
        {
            "check_name": "orphaned_registry_entries",
            "status": "WARNING",
            "message": "2 orphaned",
            "details": {"orphans": ["EXP_X", "EXP_Y"]},
        }
    ])
    s = _build_system_health(report)
    assert "EXP_X" in s.orphaned_experiments


def test_build_system_health_critical_count():
    report = _health_report("CRITICAL", [
        _check("c1", "CRITICAL", "broken"),
    ])
    s = _build_system_health(report)
    assert s.critical_count == 1


# ── 5. _build_experiment_summary ─────────────────────────────────────────────

def test_build_exp_summary_active_count():
    registry = [_reg("A"), _reg("B"), _reg("C", status="ROLLED_BACK")]
    s = _build_experiment_summary(registry, [], [], TODAY_DT)
    assert s.total_active == 2


def test_build_exp_summary_escalated_this_week():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_1", TODAY)]
    s = _build_experiment_summary([], esc, [], TODAY_DT)
    assert "EXP_A" in s.escalated_this_week


def test_build_exp_summary_escalated_old_excluded():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_1", OLD_DATE)]
    s = _build_experiment_summary([], esc, [], TODAY_DT)
    assert "EXP_A" not in s.escalated_this_week


def test_build_exp_summary_rollback_candidates():
    rb = [_rb("EXP_A", "ROLLBACK_CANDIDATE", TODAY)]
    s = _build_experiment_summary([], [], rb, TODAY_DT)
    assert "EXP_A" in s.rollback_candidates


def test_build_exp_summary_rolled_back_this_week():
    rb = [_rb("EXP_A", "ROLLED_BACK", TODAY)]
    s = _build_experiment_summary([], [], rb, TODAY_DT)
    assert "EXP_A" in s.rolled_back_this_week


def test_build_exp_summary_allocation_level():
    registry = [_reg("EXP_A", level=3)]
    s = _build_experiment_summary(registry, [], [], TODAY_DT)
    assert s.active_experiments[0].allocation_level == 3
    assert s.active_experiments[0].allocation_pct == pytest.approx(0.05)


def test_build_exp_summary_empty():
    s = _build_experiment_summary([], [], [], TODAY_DT)
    assert s.total_active == 0
    assert s.escalated_this_week == []


# ── 6. _build_improvement_ranking ────────────────────────────────────────────

def test_build_ranking_empty():
    r = _build_improvement_ranking([])
    assert r.total_ranked == 0
    assert r.top_10 == []
    assert r.bottom_10 == []


def test_build_ranking_single():
    rows = [_rank("EXP_A", 1, 80.0)]
    r = _build_improvement_ranking(rows)
    assert r.total_ranked == 1
    assert r.top_10[0].experiment_id == "EXP_A"


def test_build_ranking_top10_ordered():
    rows = [_rank(f"EXP_{i}", rank=i, score=100.0 - i * 5) for i in range(1, 16)]
    r = _build_improvement_ranking(rows)
    assert len(r.top_10) == 10
    assert r.top_10[0].rank == 1
    assert r.top_10[-1].rank == 10


def test_build_ranking_bottom10_worst_first():
    rows = [_rank(f"EXP_{i}", rank=i, score=100.0 - i * 5) for i in range(1, 16)]
    r = _build_improvement_ranking(rows)
    # bottom_10: worst (highest rank number) first
    assert r.bottom_10[0].rank >= r.bottom_10[-1].rank


def test_build_ranking_uses_latest_record():
    rows = [
        _rank("EXP_A", 3, 60.0, date="2026-05-29"),
        _rank("EXP_A", 1, 90.0, date=TODAY),  # newer → should win
    ]
    r = _build_improvement_ranking(rows)
    assert r.top_10[0].ranking_score == pytest.approx(90.0)


def test_build_ranking_metrics_populated():
    rows = [_rank("EXP_A", 1, 75.0, pnl=5000.0, wr=0.7, dd=-0.03)]
    r = _build_improvement_ranking(rows)
    assert r.top_10[0].total_pnl == pytest.approx(5000.0)
    assert r.top_10[0].win_rate  == pytest.approx(0.7)


# ── 7. _build_capital_deployment ─────────────────────────────────────────────

def test_build_capital_empty():
    c = _build_capital_deployment([], [], TODAY_DT)
    assert c.allocation_by_experiment == {}
    assert c.total_allocated_pct == 0.0


def test_build_capital_active_allocation():
    registry = [_reg("EXP_A", level=2), _reg("EXP_B", level=0)]
    c = _build_capital_deployment(registry, [], TODAY_DT)
    assert c.allocation_by_experiment["EXP_A"] == pytest.approx(0.02)
    assert c.allocation_by_experiment["EXP_B"] == pytest.approx(0.005)


def test_build_capital_total_pct():
    registry = [_reg("EXP_A", level=1), _reg("EXP_B", level=1)]
    c = _build_capital_deployment(registry, [], TODAY_DT)
    assert c.total_allocated_pct == pytest.approx(0.02)


def test_build_capital_non_active_excluded():
    registry = [_reg("EXP_A", level=3, status="ROLLED_BACK")]
    c = _build_capital_deployment(registry, [], TODAY_DT)
    assert "EXP_A" not in c.allocation_by_experiment


def test_build_capital_changes_this_week():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_2", TODAY, prev=1, new=2)]
    c = _build_capital_deployment([], esc, TODAY_DT)
    assert len(c.changes_this_week) == 1
    assert c.changes_this_week[0].experiment_id == "EXP_A"
    assert c.changes_this_week[0].new_level == 2


def test_build_capital_old_changes_excluded():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_1", OLD_DATE)]
    c = _build_capital_deployment([], esc, TODAY_DT)
    assert len(c.changes_this_week) == 0


def test_build_capital_deescalate_included():
    esc = [_esc("EXP_A", "DEESCALATE", TODAY, prev=2, new=1)]
    c = _build_capital_deployment([], esc, TODAY_DT)
    assert len(c.changes_this_week) == 1


def test_build_capital_dedup_changes():
    esc = [_esc("EXP_A", "ESCALATE_LEVEL_1", TODAY)] * 3
    c = _build_capital_deployment([], esc, TODAY_DT)
    assert len(c.changes_this_week) == 1


# ── 8. _build_rollback_activity ──────────────────────────────────────────────

def test_build_rollback_empty():
    rb = _build_rollback_activity([], TODAY_DT)
    assert rb.experiments_stopped == []
    assert rb.events_this_week == []


def test_build_rollback_stopped_this_week():
    rows = [_rb("EXP_A", "ROLLED_BACK", TODAY)]
    rb = _build_rollback_activity(rows, TODAY_DT)
    assert "EXP_A" in rb.experiments_stopped


def test_build_rollback_old_excluded():
    rows = [_rb("EXP_A", "ROLLED_BACK", OLD_DATE)]
    rb = _build_rollback_activity(rows, TODAY_DT)
    assert rb.experiments_stopped == []


def test_build_rollback_candidates():
    rows = [_rb("EXP_A", "ROLLBACK_CANDIDATE", TODAY)]
    rb = _build_rollback_activity(rows, TODAY_DT)
    assert "EXP_A" in rb.rollback_candidate_ids


def test_build_rollback_trigger_reasons():
    rows = [_rb("EXP_A", "ROLLED_BACK", TODAY, ["LOW_EXPECTANCY", "EXCESS_DRAWDOWN"])]
    rb = _build_rollback_activity(rows, TODAY_DT)
    assert rb.events_this_week[0].trigger_reasons == ["LOW_EXPECTANCY", "EXCESS_DRAWDOWN"]


def test_build_rollback_review_required_included():
    rows = [_rb("EXP_A", "REVIEW_REQUIRED", TODAY)]
    rb = _build_rollback_activity(rows, TODAY_DT)
    assert len(rb.events_this_week) == 1


def test_build_rollback_healthy_excluded():
    rows = [_rb("EXP_A", "HEALTHY", TODAY)]
    rb = _build_rollback_activity(rows, TODAY_DT)
    assert len(rb.events_this_week) == 0


# ── 9. _build_performance_summary ────────────────────────────────────────────

def test_build_perf_empty():
    p = _build_performance_summary([], TODAY_DT)
    assert p.aggregate_pnl == 0.0
    assert p.total_trades == 0


def test_build_perf_basic():
    rows = [_perf("EXP_A", pnl=500.0, trades=10, win_rate=0.7, expectancy=0.6)]
    p = _build_performance_summary(rows, TODAY_DT)
    assert p.aggregate_pnl == pytest.approx(500.0)
    assert p.total_trades == 10


def test_build_perf_control_excluded():
    rows = [
        _perf("EXP_A", pnl=9999.0, group="CONTROL"),
        _perf("EXP_A", pnl=100.0, group="EXPERIMENT"),
    ]
    p = _build_performance_summary(rows, TODAY_DT)
    assert p.aggregate_pnl == pytest.approx(100.0)


def test_build_perf_old_excluded():
    rows = [_perf("EXP_A", date=OLD_DATE, pnl=9999.0)]
    p = _build_performance_summary(rows, TODAY_DT)
    assert p.aggregate_pnl == 0.0


def test_build_perf_avg_metrics():
    rows = [
        _perf("EXP_A", expectancy=0.4, max_drawdown=-0.06, win_rate=0.5),
        _perf("EXP_B", expectancy=0.6, max_drawdown=-0.10, win_rate=0.7),
    ]
    p = _build_performance_summary(rows, TODAY_DT)
    assert p.avg_expectancy == pytest.approx(0.5, abs=0.01)
    assert p.avg_win_rate   == pytest.approx(0.6, abs=0.01)


def test_build_perf_experiment_count():
    rows = [_perf("EXP_A"), _perf("EXP_B"), _perf("EXP_A")]  # two unique eids
    p = _build_performance_summary(rows, TODAY_DT)
    assert p.experiment_count == 2


# ── 10. _build_key_alerts ────────────────────────────────────────────────────

def _make_health(status: str = "HEALTHY") -> SystemHealthSection:
    return SystemHealthSection(status, [], [], [], 11, 0 if status == "HEALTHY" else 1, 0)


def _make_exp_sum(total_active: int = 2) -> ExperimentSummarySection:
    entries = [ExperimentEntry(f"EXP_{i}", "ACTIVE", 0, 0.005, "") for i in range(total_active)]
    return ExperimentSummarySection(entries, [], [], [], total_active)


def _make_rb_section(stopped: list | None = None,
                     candidates: list | None = None) -> RollbackActivitySection:
    return RollbackActivitySection([], stopped or [], candidates or [])


def _make_perf(pnl: float = 500.0, dd: float = -0.05) -> PerformanceSummarySection:
    return PerformanceSummarySection(pnl, 0.3, dd, 0.6, 10, 2)


def _make_rank(n: int = 5) -> ImprovementRankingSection:
    top = [RankedExperiment(i, f"EXP_{i}", 90.0 - i * 3, 1000.0, 0.6, -0.05) for i in range(1, n+1)]
    bot = [RankedExperiment(n - i, f"EXP_BOT_{i}", 10.0 + i, -100.0, 0.3, -0.20) for i in range(min(3, n))]
    return ImprovementRankingSection(top, bot, n)


def _make_deploy(total_pct: float = 0.02) -> CapitalDeploymentSection:
    return CapitalDeploymentSection({"EXP_A": total_pct}, [], total_pct)


def test_build_key_alerts_health_critical():
    h = _make_health("CRITICAL")
    h.critical_count = 1
    k = _build_key_alerts(h, _make_exp_sum(), _make_rb_section(), _make_perf(), _make_rank(), _make_deploy())
    assert any(a.severity == ALERT_CRITICAL for a in k.alerts)
    assert k.critical_count >= 1


def test_build_key_alerts_health_warning():
    h = _make_health("WARNING")
    k = _build_key_alerts(h, _make_exp_sum(), _make_rb_section(), _make_perf(), _make_rank(), _make_deploy())
    assert any(a.severity == ALERT_WARNING for a in k.alerts)


def test_build_key_alerts_rolled_back_is_critical():
    h = _make_health()
    rb = _make_rb_section(stopped=["EXP_A"])
    k = _build_key_alerts(h, _make_exp_sum(), rb, _make_perf(), _make_rank(), _make_deploy())
    crit = [a for a in k.alerts if a.severity == ALERT_CRITICAL]
    assert any(a.experiment_id == "EXP_A" for a in crit)


def test_build_key_alerts_rollback_candidate_is_warning():
    h = _make_health()
    rb = _make_rb_section(candidates=["EXP_B"])
    k = _build_key_alerts(h, _make_exp_sum(), rb, _make_perf(), _make_rank(), _make_deploy())
    warns = [a for a in k.alerts if a.severity == ALERT_WARNING]
    assert any(a.experiment_id == "EXP_B" for a in warns)


def test_build_key_alerts_negative_pnl_is_warning():
    p = _make_perf(pnl=-500.0)
    k = _build_key_alerts(_make_health(), _make_exp_sum(), _make_rb_section(), p, _make_rank(), _make_deploy())
    assert any("negative" in a.message.lower() for a in k.alerts)


def test_build_key_alerts_excess_drawdown_warning():
    p = _make_perf(dd=-0.15)
    k = _build_key_alerts(_make_health(), _make_exp_sum(), _make_rb_section(), p, _make_rank(), _make_deploy())
    assert any("drawdown" in a.message.lower() for a in k.alerts)


def test_build_key_alerts_no_active_experiments():
    e = ExperimentSummarySection([], [], [], [], 0)
    k = _build_key_alerts(_make_health(), e, _make_rb_section(), _make_perf(), _make_rank(), _make_deploy())
    assert any("idle" in a.message.lower() or "no active" in a.message.lower() for a in k.alerts)


def test_build_key_alerts_high_capital_allocation():
    c = _make_deploy(total_pct=0.60)
    k = _build_key_alerts(_make_health(), _make_exp_sum(), _make_rb_section(), _make_perf(), _make_rank(), c)
    assert any("capital" in a.message.lower() or "allocated" in a.message.lower() for a in k.alerts)


def test_build_key_alerts_sorted_critical_first():
    rb = _make_rb_section(stopped=["EXP_A"])
    p = _make_perf(pnl=-100.0)
    k = _build_key_alerts(_make_health("WARNING"), _make_exp_sum(), rb, p, _make_rank(), _make_deploy())
    if len(k.alerts) > 1:
        _order = {ALERT_CRITICAL: 0, ALERT_WARNING: 1, ALERT_INFO: 2}
        for i in range(len(k.alerts) - 1):
            assert _order[k.alerts[i].severity] <= _order[k.alerts[i + 1].severity]


def test_build_key_alerts_healthy_no_issues():
    k = _build_key_alerts(
        _make_health("HEALTHY"), _make_exp_sum(3), _make_rb_section(),
        _make_perf(500.0, -0.05), _make_rank(), _make_deploy(0.02),
    )
    assert k.critical_count == 0


def test_build_key_alerts_missing_telemetry_alert():
    h = SystemHealthSection("WARNING", [], ["evidence_generation"], [], 11, 1, 0)
    k = _build_key_alerts(h, _make_exp_sum(), _make_rb_section(), _make_perf(), _make_rank(), _make_deploy())
    assert any("missing" in a.message.lower() for a in k.alerts)


def test_build_key_alerts_stale_telemetry_alert():
    h = SystemHealthSection("WARNING", ["shadow_generation"], [], [], 11, 1, 0)
    k = _build_key_alerts(h, _make_exp_sum(), _make_rb_section(), _make_perf(), _make_rank(), _make_deploy())
    assert any("stale" in a.message.lower() for a in k.alerts)


# ── 11. Serialization ─────────────────────────────────────────────────────────

def test_review_to_dict_keys():
    rev = _make_default_review()
    d = _review_to_dict(rev)
    for key in ["review_date", "lookback_days", "schema_version",
                "system_health", "experiment_summary", "improvement_ranking",
                "capital_deployment", "rollback_activity",
                "performance_summary", "key_alerts"]:
        assert key in d


def test_review_to_dict_system_health():
    rev = _make_default_review()
    d = _review_to_dict(rev)
    assert d["system_health"]["overall_status"] == "HEALTHY"


def test_review_to_dict_experiment_summary():
    rev = _make_default_review()
    d = _review_to_dict(rev)
    assert "total_active" in d["experiment_summary"]
    assert "active_experiments" in d["experiment_summary"]


def test_review_to_dict_alerts():
    rev = _make_default_review()
    rev.key_alerts.alerts.append(Alert(ALERT_INFO, "test", "EXP_X", "test_src"))
    rev.key_alerts.info_count = 1
    d = _review_to_dict(rev)
    alert = d["key_alerts"]["alerts"][0]
    assert alert["severity"] == ALERT_INFO
    assert alert["experiment_id"] == "EXP_X"


def test_write_and_load_review(tmp_path):
    path = tmp_path / "review.json"
    rev  = _make_default_review()
    write_review(rev, path)
    loaded = load_review(path)
    assert loaded is not None
    assert loaded["review_date"] == TODAY
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_load_review_missing(tmp_path):
    assert load_review(tmp_path / "missing.json") is None


def test_load_review_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("NOT_JSON", encoding="utf-8")
    assert load_review(p) is None


def test_write_review_creates_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "review.json"
    write_review(_make_default_review(), path)
    assert path.exists()


def test_write_review_overwrites(tmp_path):
    path = tmp_path / "rev.json"
    rev1 = _make_default_review()
    rev1.review_date = "2026-05-29"
    write_review(rev1, path)
    rev2 = _make_default_review()
    rev2.review_date = TODAY
    write_review(rev2, path)
    loaded = load_review(path)
    assert loaded["review_date"] == TODAY


# ── 12. Main orchestrator ─────────────────────────────────────────────────────

def test_run_all_missing_files(tmp_path):
    files = _make_files(tmp_path)
    rev = _run(files)
    assert isinstance(rev, WeeklyExecutiveReview)
    assert rev.review_date == TODAY


def test_run_writes_output(tmp_path):
    files = _make_files(tmp_path)
    _run(files)
    assert files["output"].exists()


def test_run_output_valid_json(tmp_path):
    files = _make_files(tmp_path)
    _run(files)
    loaded = load_review(files["output"])
    assert loaded is not None


def test_run_full_healthy(tmp_path):
    files = _make_files(tmp_path)
    _write_registry(files["registry"], [
        _reg("EXP_A", level=1, start="2026-01-01"),
        _reg("EXP_B", level=2, start="2026-01-01"),
    ])
    _write_jsonl(files["performance"], [
        _perf("EXP_A", pnl=500.0), _perf("EXP_B", pnl=300.0),
    ])
    _write_jsonl(files["escalation"], [
        _esc("EXP_A", "ESCALATE_LEVEL_1", TODAY, prev=0, new=1),
    ])
    _write_jsonl(files["rollback"], [_rb("EXP_A", "HEALTHY")])
    _write_jsonl(files["ranking"], [_rank("EXP_A", 1), _rank("EXP_B", 2)])
    _write_json(files["health"], _health_report("HEALTHY", [_check("c1")]))
    rev = _run(files)
    assert rev.experiment_summary.total_active == 2
    assert "EXP_A" in rev.experiment_summary.escalated_this_week
    assert rev.performance_summary.aggregate_pnl == pytest.approx(800.0)
    assert rev.improvement_ranking.top_10[0].experiment_id == "EXP_A"


def test_run_detects_rolled_back(tmp_path):
    files = _make_files(tmp_path)
    _write_jsonl(files["rollback"], [_rb("EXP_A", "ROLLED_BACK", TODAY)])
    rev = _run(files)
    assert "EXP_A" in rev.rollback_activity.experiments_stopped
    crit = [a for a in rev.key_alerts.alerts if a.severity == ALERT_CRITICAL]
    assert any(a.experiment_id == "EXP_A" for a in crit)


def test_run_detects_critical_health(tmp_path):
    files = _make_files(tmp_path)
    _write_json(files["health"], _health_report("CRITICAL", [_check("c1", "CRITICAL", "broken")]))
    rev = _run(files)
    assert rev.system_health.overall_status == "CRITICAL"
    assert any(a.severity == ALERT_CRITICAL for a in rev.key_alerts.alerts)


def test_run_capital_deployment(tmp_path):
    files = _make_files(tmp_path)
    _write_registry(files["registry"], [_reg("EXP_A", level=3)])
    rev = _run(files)
    assert "EXP_A" in rev.capital_deployment.allocation_by_experiment
    assert rev.capital_deployment.allocation_by_experiment["EXP_A"] == pytest.approx(0.05)


def test_run_lookback_window(tmp_path):
    files = _make_files(tmp_path)
    _write_jsonl(files["performance"], [
        _perf("EXP_A", date=TODAY,    pnl=100.0),
        _perf("EXP_B", date=OLD_DATE, pnl=9999.0),  # outside window
    ])
    rev = _run(files)
    assert rev.performance_summary.aggregate_pnl == pytest.approx(100.0)


def test_run_fail_open_bad_registry(tmp_path):
    files = _make_files(tmp_path)
    files["registry"].parent.mkdir(parents=True, exist_ok=True)
    files["registry"].write_text("BAD_JSON", encoding="utf-8")
    rev = _run(files)
    assert isinstance(rev, WeeklyExecutiveReview)


def test_run_overwrites_previous(tmp_path):
    files = _make_files(tmp_path)
    _run(files, today="2026-05-30")
    _run(files, today=TODAY)
    loaded = load_review(files["output"])
    assert loaded["review_date"] == TODAY


def test_run_schema_version(tmp_path):
    files = _make_files(tmp_path)
    _run(files)
    loaded = load_review(files["output"])
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_run_negative_pnl_alert(tmp_path):
    files = _make_files(tmp_path)
    _write_jsonl(files["performance"], [_perf("EXP_A", pnl=-5000.0)])
    rev = _run(files)
    assert any("negative" in a.message.lower() for a in rev.key_alerts.alerts)


# ── 13. Formatter ─────────────────────────────────────────────────────────────

def test_format_contains_header():
    rev = _make_default_review()
    rpt = format_executive_review(rev)
    assert "WEEKLY EXECUTIVE REVIEW" in rpt


def test_format_shows_date():
    rev = _make_default_review()
    rpt = format_executive_review(rev)
    assert TODAY in rpt


def test_format_shows_system_health():
    rev = _make_default_review()
    rpt = format_executive_review(rev)
    assert "SYSTEM HEALTH" in rpt
    assert "HEALTHY" in rpt


def test_format_shows_experiments():
    rev = _make_default_review()
    rev.experiment_summary.active_experiments = [
        ExperimentEntry("EXP_X", "ACTIVE", 1, 0.01, "2026-01-01")
    ]
    rev.experiment_summary.total_active = 1
    rpt = format_executive_review(rev)
    assert "EXP_X" in rpt


def test_format_shows_alerts():
    rev = _make_default_review()
    rev.key_alerts.alerts.append(Alert(ALERT_CRITICAL, "Urgent review needed", "EXP_Z"))
    rev.key_alerts.critical_count = 1
    rpt = format_executive_review(rev)
    assert "Urgent review needed" in rpt


def test_format_returns_str():
    rev = _make_default_review()
    assert isinstance(format_executive_review(rev), str)


def test_format_shows_performance():
    rev = _make_default_review()
    rev.performance_summary.aggregate_pnl = 12345.0
    rpt = format_executive_review(rev)
    assert "12345" in rpt


def test_format_shows_capital_deployment():
    rev = _make_default_review()
    rpt = format_executive_review(rev)
    assert "CAPITAL DEPLOYMENT" in rpt


# ── 14. Integration ───────────────────────────────────────────────────────────

def test_integration_full_scenario(tmp_path):
    """End-to-end: rich scenario → all 7 sections populated + report written."""
    files = _make_files(tmp_path)

    _write_registry(files["registry"], [
        _reg("EXP_WINNER", level=3, start="2026-01-01"),
        _reg("EXP_MID",    level=1, start="2026-02-01"),
        _reg("EXP_LOSER",  level=0, start="2026-03-01", status="ROLLED_BACK"),
    ])
    _write_jsonl(files["performance"], [
        _perf("EXP_WINNER", pnl=8000.0, win_rate=0.72, expectancy=0.8, max_drawdown=-0.03),
        _perf("EXP_MID",    pnl=1500.0, win_rate=0.55, expectancy=0.3, max_drawdown=-0.07),
        _perf("EXP_LOSER",  pnl=-2000.0, win_rate=0.30, expectancy=-0.2, max_drawdown=-0.25),
    ])
    _write_jsonl(files["escalation"], [
        _esc("EXP_WINNER", "ESCALATE_LEVEL_2", TODAY, prev=2, new=3),
        _esc("EXP_MID",    "ESCALATE_LEVEL_1", TODAY, prev=0, new=1),
    ])
    _write_jsonl(files["rollback"], [
        _rb("EXP_LOSER", "ROLLED_BACK", TODAY, ["LOW_EXPECTANCY", "EXCESS_DRAWDOWN"]),
    ])
    _write_jsonl(files["ranking"], [
        _rank("EXP_WINNER", 1, 90.0, pnl=8000.0),
        _rank("EXP_MID",    2, 65.0, pnl=1500.0),
        _rank("EXP_LOSER",  3, 10.0, pnl=-2000.0),
    ])
    _write_json(files["health"], _health_report(
        "WARNING", [
            _check("evidence_generation", "WARNING", "Stale: last 4d"),
            _check("shadow_generation",   "HEALTHY"),
        ]
    ))

    rev = _run(files)

    # System health
    assert rev.system_health.overall_status == "WARNING"
    assert "evidence_generation" in rev.system_health.stale_telemetry

    # Experiment summary
    assert rev.experiment_summary.total_active == 2
    assert "EXP_WINNER" in rev.experiment_summary.escalated_this_week
    assert "EXP_LOSER"  in rev.rollback_activity.experiments_stopped

    # Ranking
    assert rev.improvement_ranking.total_ranked == 3
    assert rev.improvement_ranking.top_10[0].experiment_id == "EXP_WINNER"

    # Capital
    assert "EXP_WINNER" in rev.capital_deployment.allocation_by_experiment
    assert len(rev.capital_deployment.changes_this_week) == 2

    # Performance
    assert rev.performance_summary.aggregate_pnl == pytest.approx(8000.0 + 1500.0 + (-2000.0))

    # Key alerts — rolled-back = CRITICAL
    crit = [a for a in rev.key_alerts.alerts if a.severity == ALERT_CRITICAL]
    assert any(a.experiment_id == "EXP_LOSER" for a in crit)

    # Report on disk
    loaded = load_review(files["output"])
    assert loaded is not None
    assert loaded["experiment_summary"]["total_active"] == 2

    # Formatter
    rpt = format_executive_review(rev)
    assert "EXP_WINNER" in rpt
    assert "ROLLBACK ACTIVITY" in rpt
