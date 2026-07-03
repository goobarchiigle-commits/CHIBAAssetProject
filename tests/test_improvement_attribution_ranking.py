"""
tests/test_improvement_attribution_ranking.py — Phase 6N tests

80+ tests covering:
  Constants / dataclasses / helpers /
  aggregation / score computation / ranking /
  telemetry I/O / idempotency / main orchestrator / report formatter /
  edge cases / fail-open guarantees
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.improvement_attribution_ranking import (
    # constants
    SCHEMA_VERSION,
    _W_PROFITABILITY, _W_CONSISTENCY, _W_DD_PENALTY,
    _W_SURVIVABILITY, _W_ESCALATION,
    _MAX_ESCALATION_REWARD, _MAX_ROLLBACK_PENALTY,
    # dataclasses
    ImprovementRankingInput, ImprovementRankingResult, ImprovementRankingTelemetry,
    # helpers
    _safe_float, _safe_int, _load_all_jsonl, _append_jsonl,
    _load_registry_json,
    # aggregation
    _aggregate_performance, _count_escalations, _count_rollbacks,
    _get_allocation_level, _get_experiment_ids,
    # scoring
    _normalize_minmax, _dd_score, _survivability_score, _escalation_score,
    compute_ranking_scores,
    # ranking
    build_ranking_results,
    # I/O
    append_ranking_record, load_ranking_records, _load_evaluated_pairs,
    # main
    run_improvement_attribution_ranking,
    # report
    format_ranking_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _perf_rec(
    eid: str,
    group: str = "EXPERIMENT",
    pnl: float = 1000.0,
    avg_return: float = 0.02,
    expectancy: float = 0.5,
    win_rate: float = 0.6,
    max_drawdown: float = -0.05,
    date: str = "2026-05-30",
) -> dict:
    return {
        "experiment_id": eid, "group": group, "date": date,
        "pnl": pnl, "avg_return": avg_return, "expectancy": expectancy,
        "win_rate": win_rate, "max_drawdown": max_drawdown,
        "trades": 10, "schema_version": "v1",
    }


def _esc_rec(eid: str, decision: str = "ESCALATE_LEVEL_1", date: str = "2026-05-30") -> dict:
    return {"experiment_id": eid, "decision": decision, "evaluation_date": date,
            "schema_version": "v1"}


def _rb_rec(eid: str, decision: str = "ROLLED_BACK", date: str = "2026-05-30") -> dict:
    return {"experiment_id": eid, "decision": decision, "evaluation_date": date,
            "schema_version": "v1"}


def _reg_record(eid: str, level: int = 0, status: str = "ACTIVE") -> dict:
    return {"experiment_id": eid, "allocation_level": level,
            "status": status, "schema_version": "v1"}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_registry_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def _make_input(
    eid: str = "EXP_A",
    pnl: float = 1000.0,
    avg_return: float = 0.02,
    expectancy: float = 0.5,
    win_rate: float = 0.6,
    max_drawdown: float = -0.05,
    allocation_level: int = 1,
    escalation_count: int = 2,
    rollback_count: int = 0,
    observation_days: int = 10,
) -> ImprovementRankingInput:
    return ImprovementRankingInput(
        experiment_id=eid, total_pnl=pnl, avg_return=avg_return,
        expectancy=expectancy, win_rate=win_rate, max_drawdown=max_drawdown,
        allocation_level=allocation_level, escalation_count=escalation_count,
        rollback_count=rollback_count, observation_days=observation_days,
    )


# ── 1. Constants ──────────────────────────────────────────────────────────────

def test_schema_version():
    assert SCHEMA_VERSION == "v1"


def test_weights_sum_to_one():
    total = _W_PROFITABILITY + _W_CONSISTENCY + _W_DD_PENALTY + _W_SURVIVABILITY + _W_ESCALATION
    assert abs(total - 1.0) < 1e-9


def test_max_escalation_reward_positive():
    assert _MAX_ESCALATION_REWARD > 0


def test_max_rollback_penalty_positive():
    assert _MAX_ROLLBACK_PENALTY > 0


# ── 2. Dataclasses ────────────────────────────────────────────────────────────

def test_improvement_ranking_input_fields():
    inp = _make_input()
    assert inp.experiment_id == "EXP_A"
    assert inp.total_pnl == 1000.0
    assert inp.win_rate == 0.6


def test_improvement_ranking_result_defaults():
    r = ImprovementRankingResult(
        experiment_id="EXP_A", date="2026-05-30", rank=1,
        ranking_score=75.0, metrics={},
    )
    assert r.schema_version == SCHEMA_VERSION


def test_improvement_ranking_telemetry_fields():
    t = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=3, ranked=[],
        top_experiment="EXP_A", top_score=80.0,
    )
    assert t.schema_version == SCHEMA_VERSION


# ── 3. Helpers ────────────────────────────────────────────────────────────────

def test_safe_float_normal():
    assert _safe_float(1.5) == 1.5


def test_safe_float_none():
    assert _safe_float(None) == 0.0


def test_safe_float_nan():
    assert _safe_float(float("nan")) == 0.0


def test_safe_float_inf():
    assert _safe_float(float("inf")) == 0.0


def test_safe_float_custom_fallback():
    assert _safe_float(None, -1.0) == -1.0


def test_safe_int_normal():
    assert _safe_int("5") == 5


def test_safe_int_bad():
    assert _safe_int("x") == 0


def test_load_all_jsonl_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    assert _load_all_jsonl(p) == []


def test_load_all_jsonl_missing(tmp_path):
    p = tmp_path / "missing.jsonl"
    assert _load_all_jsonl(p) == []


def test_load_all_jsonl_reads(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [{"a": 1}, {"a": 2}])
    rows = _load_all_jsonl(p)
    assert len(rows) == 2


def test_load_all_jsonl_bad_line_skipped(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"a":1}\nNOT_JSON\n{"b":2}\n', encoding="utf-8")
    rows = _load_all_jsonl(p)
    assert len(rows) == 2


def test_append_jsonl_creates_file(tmp_path):
    p = tmp_path / "sub" / "out.jsonl"
    _append_jsonl({"x": 1}, p)
    assert p.exists()
    rows = _load_all_jsonl(p)
    assert rows[0]["x"] == 1


def test_append_jsonl_accumulates(tmp_path):
    p = tmp_path / "t.jsonl"
    _append_jsonl({"n": 1}, p)
    _append_jsonl({"n": 2}, p)
    rows = _load_all_jsonl(p)
    assert len(rows) == 2


def test_load_registry_json_list(tmp_path):
    p = tmp_path / "reg.json"
    recs = [_reg_record("EXP_A"), _reg_record("EXP_B")]
    _write_registry_json(p, recs)
    loaded = _load_registry_json(p)
    assert len(loaded) == 2


def test_load_registry_json_missing(tmp_path):
    p = tmp_path / "missing.json"
    assert _load_registry_json(p) == []


def test_load_registry_json_bad(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("NOT_JSON", encoding="utf-8")
    assert _load_registry_json(p) == []


# ── 4. Aggregation ────────────────────────────────────────────────────────────

def test_aggregate_performance_basic():
    rows = [_perf_rec("EXP_A", pnl=500.0), _perf_rec("EXP_A", pnl=300.0)]
    result = _aggregate_performance(rows, "EXP_A")
    assert abs(result["total_pnl"] - 800.0) < 0.01


def test_aggregate_performance_filters_group():
    rows = [
        _perf_rec("EXP_A", group="CONTROL", pnl=9999.0),
        _perf_rec("EXP_A", group="EXPERIMENT", pnl=100.0),
    ]
    result = _aggregate_performance(rows, "EXP_A")
    assert abs(result["total_pnl"] - 100.0) < 0.01


def test_aggregate_performance_unknown_experiment():
    rows = [_perf_rec("EXP_A", pnl=500.0)]
    result = _aggregate_performance(rows, "EXP_Z")
    assert result["total_pnl"] == 0.0


def test_aggregate_performance_observation_days():
    rows = [
        _perf_rec("EXP_A", date="2026-05-28"),
        _perf_rec("EXP_A", date="2026-05-29"),
        _perf_rec("EXP_A", date="2026-05-29"),  # duplicate date
    ]
    result = _aggregate_performance(rows, "EXP_A")
    assert result["observation_days"] == 2  # unique dates


def test_aggregate_performance_max_drawdown_most_negative():
    rows = [
        _perf_rec("EXP_A", max_drawdown=-0.10),
        _perf_rec("EXP_A", max_drawdown=-0.05),
    ]
    result = _aggregate_performance(rows, "EXP_A")
    assert result["max_drawdown"] == pytest.approx(-0.10, abs=1e-4)


def test_count_escalations_basic():
    recs = [
        _esc_rec("EXP_A", "ESCALATE_LEVEL_1"),
        _esc_rec("EXP_A", "ESCALATE_LEVEL_2"),
        _esc_rec("EXP_A", "HOLD"),
        _esc_rec("EXP_B", "ESCALATE_LEVEL_1"),
    ]
    assert _count_escalations(recs, "EXP_A") == 2
    assert _count_escalations(recs, "EXP_B") == 1


def test_count_escalations_full_scale():
    recs = [_esc_rec("EXP_A", "FULL_SCALE")]
    assert _count_escalations(recs, "EXP_A") == 1


def test_count_escalations_deescalate_not_counted():
    recs = [_esc_rec("EXP_A", "DEESCALATE")]
    assert _count_escalations(recs, "EXP_A") == 0


def test_count_rollbacks_basic():
    recs = [
        _rb_rec("EXP_A", "ROLLED_BACK"),
        _rb_rec("EXP_A", "HEALTHY"),
        _rb_rec("EXP_B", "ROLLED_BACK"),
    ]
    assert _count_rollbacks(recs, "EXP_A") == 1
    assert _count_rollbacks(recs, "EXP_B") == 1


def test_count_rollbacks_zero():
    recs = [_rb_rec("EXP_A", "HEALTHY")]
    assert _count_rollbacks(recs, "EXP_A") == 0


def test_get_allocation_level_found():
    recs = [_reg_record("EXP_A", level=3)]
    assert _get_allocation_level(recs, "EXP_A") == 3


def test_get_allocation_level_missing():
    recs = [_reg_record("EXP_B", level=2)]
    assert _get_allocation_level(recs, "EXP_A") == 0


def test_get_experiment_ids_unique():
    recs = [_reg_record("EXP_A"), _reg_record("EXP_B"), _reg_record("EXP_A")]
    ids = _get_experiment_ids(recs)
    assert ids.count("EXP_A") == 1
    assert "EXP_B" in ids


# ── 5. Score computation ──────────────────────────────────────────────────────

def test_normalize_minmax_basic():
    vals = [0.0, 5.0, 10.0]
    normed = _normalize_minmax(vals)
    assert normed[0] == pytest.approx(0.0)
    assert normed[2] == pytest.approx(1.0)
    assert normed[1] == pytest.approx(0.5)


def test_normalize_minmax_all_equal():
    vals = [3.0, 3.0, 3.0]
    normed = _normalize_minmax(vals)
    assert all(v == 0.0 for v in normed)


def test_normalize_minmax_empty():
    assert _normalize_minmax([]) == []


def test_dd_score_zero_drawdown():
    assert _dd_score(0.0) == pytest.approx(1.0)


def test_dd_score_fifty_pct():
    # max_drawdown=-0.50 → score=0.0
    assert _dd_score(-0.50) == pytest.approx(0.0)


def test_dd_score_clamps_below_zero():
    # Beyond -0.50 → 0.0
    assert _dd_score(-1.0) == pytest.approx(0.0)


def test_dd_score_mild_drawdown():
    # -0.10 → 1.0 + (-0.10)*2 = 0.8
    assert _dd_score(-0.10) == pytest.approx(0.8)


def test_survivability_score_no_rollbacks():
    assert _survivability_score(0) == pytest.approx(1.0)


def test_survivability_score_one_rollback():
    s = _survivability_score(1)
    assert 0.0 < s < 1.0


def test_survivability_score_decreases_with_rollbacks():
    s1 = _survivability_score(1)
    s2 = _survivability_score(2)
    assert s1 > s2


def test_survivability_score_capped():
    s5 = _survivability_score(5)
    s100 = _survivability_score(100)
    assert s5 == s100  # both capped at _MAX_ROLLBACK_PENALTY


def test_escalation_score_zero():
    assert _escalation_score(0) == pytest.approx(0.0)


def test_escalation_score_max():
    assert _escalation_score(_MAX_ESCALATION_REWARD) == pytest.approx(1.0)


def test_escalation_score_capped():
    assert _escalation_score(100) == pytest.approx(1.0)


def test_escalation_score_intermediate():
    mid = _MAX_ESCALATION_REWARD // 2
    s = _escalation_score(mid)
    assert 0.0 < s < 1.0


def test_compute_ranking_scores_single():
    inp = [_make_input()]
    scores = compute_ranking_scores(inp)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 100.0


def test_compute_ranking_scores_empty():
    assert compute_ranking_scores([]) == []


def test_compute_ranking_scores_returns_list():
    inps = [_make_input("A"), _make_input("B")]
    scores = compute_ranking_scores(inps)
    assert len(scores) == 2


def test_compute_ranking_scores_all_in_range():
    inps = [
        _make_input("A", pnl=5000.0, win_rate=0.7, rollback_count=0, escalation_count=3),
        _make_input("B", pnl=-1000.0, win_rate=0.3, rollback_count=2, escalation_count=0),
    ]
    scores = compute_ranking_scores(inps)
    for s in scores:
        assert 0.0 <= s <= 100.0


def test_compute_ranking_scores_better_pnl_wins():
    inps = [
        _make_input("A", pnl=10000.0, rollback_count=0),
        _make_input("B", pnl=100.0, rollback_count=0),
    ]
    scores = compute_ranking_scores(inps)
    idx_a = 0  # A is first
    assert scores[idx_a] > scores[1]


def test_compute_ranking_scores_rollback_penalizes():
    inps = [
        _make_input("A", pnl=1000.0, rollback_count=0),
        _make_input("B", pnl=1000.0, rollback_count=3),
    ]
    scores = compute_ranking_scores(inps)
    assert scores[0] > scores[1]


def test_compute_ranking_scores_no_nan():
    inps = [_make_input("A", pnl=0.0, win_rate=0.0, expectancy=0.0)]
    scores = compute_ranking_scores(inps)
    assert not math.isnan(scores[0])


# ── 6. Ranking ────────────────────────────────────────────────────────────────

def test_build_ranking_results_empty():
    assert build_ranking_results([], "2026-05-30") == []


def test_build_ranking_results_single():
    inps = [_make_input("EXP_A")]
    results = build_ranking_results(inps, "2026-05-30")
    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].experiment_id == "EXP_A"


def test_build_ranking_results_order():
    inps = [
        _make_input("EXP_LOW",  pnl=10.0),
        _make_input("EXP_HIGH", pnl=9999.0),
    ]
    results = build_ranking_results(inps, "2026-05-30")
    assert results[0].experiment_id == "EXP_HIGH"
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_build_ranking_results_metrics_populated():
    inps = [_make_input("EXP_A", pnl=500.0, escalation_count=2)]
    results = build_ranking_results(inps, "2026-05-30")
    assert results[0].metrics["total_pnl"] == 500.0
    assert results[0].metrics["escalation_count"] == 2


def test_build_ranking_results_date():
    inps = [_make_input("EXP_A")]
    results = build_ranking_results(inps, "2026-05-30")
    assert results[0].date == "2026-05-30"


def test_build_ranking_results_schema_version():
    inps = [_make_input("EXP_A")]
    results = build_ranking_results(inps, "2026-05-30")
    assert results[0].schema_version == SCHEMA_VERSION


# ── 7. Telemetry I/O ──────────────────────────────────────────────────────────

def test_append_ranking_record_creates(tmp_path):
    p = tmp_path / "out.jsonl"
    r = ImprovementRankingResult(
        experiment_id="EXP_A", date="2026-05-30", rank=1,
        ranking_score=75.0, metrics={"total_pnl": 1000.0},
    )
    append_ranking_record(r, p)
    rows = _load_all_jsonl(p)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "EXP_A"
    assert rows[0]["rank"] == 1
    assert rows[0]["ranking_score"] == 75.0


def test_append_ranking_record_has_schema_version(tmp_path):
    p = tmp_path / "out.jsonl"
    r = ImprovementRankingResult(
        experiment_id="EXP_A", date="2026-05-30", rank=1,
        ranking_score=80.0, metrics={},
    )
    append_ranking_record(r, p)
    rows = _load_all_jsonl(p)
    assert rows[0]["schema_version"] == SCHEMA_VERSION


def test_load_ranking_records_filter(tmp_path):
    p = tmp_path / "out.jsonl"
    for eid in ["EXP_A", "EXP_B", "EXP_A"]:
        r = ImprovementRankingResult(
            experiment_id=eid, date="2026-05-30", rank=1,
            ranking_score=50.0, metrics={},
        )
        append_ranking_record(r, p)
    rows_a = load_ranking_records(p, experiment_id="EXP_A")
    assert len(rows_a) == 2
    rows_b = load_ranking_records(p, experiment_id="EXP_B")
    assert len(rows_b) == 1


def test_load_ranking_records_missing(tmp_path):
    p = tmp_path / "missing.jsonl"
    rows = load_ranking_records(p)
    assert rows == []


def test_load_evaluated_pairs_empty(tmp_path):
    p = tmp_path / "out.jsonl"
    pairs = _load_evaluated_pairs(p)
    assert pairs == set()


def test_load_evaluated_pairs_populated(tmp_path):
    p = tmp_path / "out.jsonl"
    r = ImprovementRankingResult(
        experiment_id="EXP_A", date="2026-05-30", rank=1,
        ranking_score=60.0, metrics={},
    )
    append_ranking_record(r, p)
    pairs = _load_evaluated_pairs(p)
    assert ("2026-05-30", "EXP_A") in pairs


# ── 8. Main orchestrator ──────────────────────────────────────────────────────

def _make_all_files(tmp_path: Path, date: str = "2026-05-30"):
    perf   = tmp_path / "perf.jsonl"
    reg    = tmp_path / "reg.json"
    rb     = tmp_path / "rollback.jsonl"
    esc    = tmp_path / "escalation.jsonl"
    output = tmp_path / "out.jsonl"
    return perf, reg, rb, esc, output


def test_run_ranking_empty_inputs(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert isinstance(tel, ImprovementRankingTelemetry)
    assert tel.total_experiments == 0
    assert tel.ranked == []


def test_run_ranking_single_experiment(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_A")])
    _write_registry_json(reg, [_reg_record("EXP_A", level=1)])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.total_experiments == 1
    assert tel.top_experiment == "EXP_A"
    assert 0.0 <= tel.top_score <= 100.0


def test_run_ranking_multiple_experiments(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [
        _perf_rec("EXP_A", pnl=5000.0),
        _perf_rec("EXP_B", pnl=100.0),
    ])
    _write_registry_json(reg, [
        _reg_record("EXP_A", level=3),
        _reg_record("EXP_B", level=0),
    ])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.total_experiments == 2
    assert tel.ranked[0].rank == 1
    assert tel.ranked[1].rank == 2


def test_run_ranking_idempotent(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_A")])
    _write_registry_json(reg, [_reg_record("EXP_A")])
    run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    rows = _load_all_jsonl(output)
    # Only one record per (date, experiment_id)
    assert len(rows) == 1


def test_run_ranking_different_dates(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_A")])
    _write_registry_json(reg, [_reg_record("EXP_A")])
    run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-31")
    rows = _load_all_jsonl(output)
    assert len(rows) == 2


def test_run_ranking_with_escalation_data(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_A")])
    _write_registry_json(reg, [_reg_record("EXP_A", level=2)])
    _write_jsonl(esc, [
        _esc_rec("EXP_A", "ESCALATE_LEVEL_1"),
        _esc_rec("EXP_A", "ESCALATE_LEVEL_2"),
    ])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.ranked[0].metrics["escalation_count"] == 2


def test_run_ranking_with_rollback_data(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_A")])
    _write_registry_json(reg, [_reg_record("EXP_A")])
    _write_jsonl(rb, [_rb_rec("EXP_A", "ROLLED_BACK")])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.ranked[0].metrics["rollback_count"] == 1


def test_run_ranking_writes_jsonl(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_A")])
    _write_registry_json(reg, [_reg_record("EXP_A")])
    run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert output.exists()
    rows = _load_all_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "EXP_A"


def test_run_ranking_fail_open_bad_perf(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    perf.parent.mkdir(parents=True, exist_ok=True)
    perf.write_text("GARBAGE_NOT_JSON\n", encoding="utf-8")
    _write_registry_json(reg, [_reg_record("EXP_A")])
    # Should not raise
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert isinstance(tel, ImprovementRankingTelemetry)


def test_run_ranking_experiment_from_perf_not_in_registry(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [_perf_rec("EXP_UNKNOWN")])
    _write_registry_json(reg, [])  # empty registry
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.total_experiments == 1
    assert tel.top_experiment == "EXP_UNKNOWN"


def test_run_ranking_control_group_excluded(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [
        _perf_rec("EXP_A", group="CONTROL", pnl=99999.0),  # should be ignored
        _perf_rec("EXP_A", group="EXPERIMENT", pnl=100.0),
    ])
    _write_registry_json(reg, [_reg_record("EXP_A")])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert abs(tel.ranked[0].metrics["total_pnl"] - 100.0) < 0.01


def test_run_ranking_top_experiment_is_highest_ranked(tmp_path):
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [
        _perf_rec("EXP_A", pnl=100.0),
        _perf_rec("EXP_B", pnl=9999.0),
    ])
    _write_registry_json(reg, [
        _reg_record("EXP_A"),
        _reg_record("EXP_B"),
    ])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.top_experiment == "EXP_B"


# ── 9. Report formatter ───────────────────────────────────────────────────────

def test_format_ranking_report_empty():
    tel = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=0, ranked=[],
        top_experiment="", top_score=0.0,
    )
    assert format_ranking_report(tel) == ""


def test_format_ranking_report_has_header():
    inps = [_make_input("EXP_A")]
    results = build_ranking_results(inps, "2026-05-30")
    tel = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=1, ranked=results,
        top_experiment="EXP_A", top_score=results[0].ranking_score,
    )
    rpt = format_ranking_report(tel)
    assert "IMPROVEMENT ATTRIBUTION RANKING" in rpt


def test_format_ranking_report_has_experiment_id():
    inps = [_make_input("MY_EXP")]
    results = build_ranking_results(inps, "2026-05-30")
    tel = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=1, ranked=results,
        top_experiment="MY_EXP", top_score=results[0].ranking_score,
    )
    rpt = format_ranking_report(tel)
    assert "MY_EXP" in rpt


def test_format_ranking_report_has_rank_indicator():
    inps = [_make_input("EXP_A"), _make_input("EXP_B", pnl=9999.0)]
    results = build_ranking_results(inps, "2026-05-30")
    tel = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=2, ranked=results,
        top_experiment=results[0].experiment_id, top_score=results[0].ranking_score,
    )
    rpt = format_ranking_report(tel)
    assert "#1" in rpt
    assert "#2" in rpt


def test_format_ranking_report_has_top_line():
    inps = [_make_input("EXP_A")]
    results = build_ranking_results(inps, "2026-05-30")
    tel = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=1, ranked=results,
        top_experiment="EXP_A", top_score=results[0].ranking_score,
    )
    rpt = format_ranking_report(tel)
    assert "Top:" in rpt


def test_format_ranking_report_returns_string():
    inps = [_make_input("EXP_A")]
    results = build_ranking_results(inps, "2026-05-30")
    tel = ImprovementRankingTelemetry(
        date="2026-05-30", total_experiments=1, ranked=results,
        top_experiment="EXP_A", top_score=50.0,
    )
    assert isinstance(format_ranking_report(tel), str)


# ── 10. Integration ───────────────────────────────────────────────────────────

def test_full_flow_three_experiments(tmp_path):
    """End-to-end: 3 experiments, ranking written to JSONL, report generated."""
    perf, reg, rb, esc, output = _make_all_files(tmp_path)
    _write_jsonl(perf, [
        _perf_rec("EXP_WINNER", pnl=50000.0, win_rate=0.75, max_drawdown=-0.03),
        _perf_rec("EXP_MID",    pnl=5000.0,  win_rate=0.55, max_drawdown=-0.10),
        _perf_rec("EXP_LOSER",  pnl=-1000.0, win_rate=0.30, max_drawdown=-0.25),
    ])
    _write_registry_json(reg, [
        _reg_record("EXP_WINNER", level=4),
        _reg_record("EXP_MID",    level=2),
        _reg_record("EXP_LOSER",  level=0),
    ])
    _write_jsonl(esc, [
        _esc_rec("EXP_WINNER", "ESCALATE_LEVEL_1"),
        _esc_rec("EXP_WINNER", "ESCALATE_LEVEL_2"),
        _esc_rec("EXP_WINNER", "ESCALATE_LEVEL_3"),
        _esc_rec("EXP_WINNER", "FULL_SCALE"),
        _esc_rec("EXP_MID",    "ESCALATE_LEVEL_1"),
    ])
    _write_jsonl(rb, [
        _rb_rec("EXP_LOSER", "ROLLED_BACK"),
    ])
    tel = run_improvement_attribution_ranking(perf, reg, rb, esc, output, "2026-05-30")
    assert tel.total_experiments == 3
    assert tel.ranked[0].rank == 1
    assert tel.ranked[0].experiment_id == "EXP_WINNER"
    assert tel.ranked[2].experiment_id == "EXP_LOSER"
    rows = _load_all_jsonl(output)
    assert len(rows) == 3
    rpt = format_ranking_report(tel)
    assert "EXP_WINNER" in rpt
    assert "#1" in rpt
