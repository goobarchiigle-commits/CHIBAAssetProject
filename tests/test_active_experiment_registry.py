"""
tests/test_active_experiment_registry.py — Active Experiment Registry (Phase 6J-Plus)
"""
import hashlib
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.active_experiment_registry import (
    SCHEMA_VERSION,
    STATUS_PENDING, STATUS_ACTIVE, STATUS_PAUSED, STATUS_ROLLED_BACK, STATUS_COMPLETED,
    GROUP_CONTROL, GROUP_EXPERIMENT,
    EVAL_CONTINUE, EVAL_ROLLBACK, EVAL_PROMOTE,
    ExperimentConfig, ExperimentAssignment, ExperimentEvaluation,
    load_registry, save_registry, get_active_experiments,
    assign_symbol_to_experiment, route_symbols_to_experiments,
    append_assignment, evaluate_assignment,
    create_experiments_from_gate_approval,
    format_experiment_report,
    _safe_float, _safe_int,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_config(
    exp_id="EXP_EXIT_EXTENSION_2026-05-30",
    feature="EXIT_EXTENSION",
    status=STATUS_ACTIVE,
    allocation_pct=0.5,
    max_positions=1,
    start_date="2026-05-30",
    approval_source="PRODUCTION_PROMOTION_GATE",
    rollback_threshold=-0.10,
    minimum_sample_size=10,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=exp_id,
        feature_name=feature,
        status=status,
        allocation_pct=allocation_pct,
        max_positions=max_positions,
        start_date=start_date,
        approval_source=approval_source,
        rollback_threshold=rollback_threshold,
        minimum_sample_size=minimum_sample_size,
    )


def _write_registry(path: Path, configs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "experiments": [
            {
                "experiment_id":       c.experiment_id,
                "feature_name":        c.feature_name,
                "status":              c.status,
                "allocation_pct":      c.allocation_pct,
                "max_positions":       c.max_positions,
                "start_date":          c.start_date,
                "approval_source":     c.approval_source,
                "rollback_threshold":  c.rollback_threshold,
                "minimum_sample_size": c.minimum_sample_size,
                "schema_version":      c.schema_version,
            }
            for c in configs
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── TestConstants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_status_pending(self):
        assert STATUS_PENDING == "PENDING"

    def test_status_active(self):
        assert STATUS_ACTIVE == "ACTIVE"

    def test_status_paused(self):
        assert STATUS_PAUSED == "PAUSED"

    def test_status_rolled_back(self):
        assert STATUS_ROLLED_BACK == "ROLLED_BACK"

    def test_status_completed(self):
        assert STATUS_COMPLETED == "COMPLETED"

    def test_group_control(self):
        assert GROUP_CONTROL == "CONTROL"

    def test_group_experiment(self):
        assert GROUP_EXPERIMENT == "EXPERIMENT"

    def test_eval_continue(self):
        assert EVAL_CONTINUE == "CONTINUE"

    def test_eval_rollback(self):
        assert EVAL_ROLLBACK == "ROLLBACK"

    def test_eval_promote(self):
        assert EVAL_PROMOTE == "PROMOTE"

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"


# ── TestDataclasses ────────────────────────────────────────────────────────────

class TestExperimentConfig:
    def test_creation(self):
        c = _make_config()
        assert c.experiment_id == "EXP_EXIT_EXTENSION_2026-05-30"
        assert c.feature_name == "EXIT_EXTENSION"
        assert c.status == STATUS_ACTIVE

    def test_defaults_schema_version(self):
        c = _make_config()
        assert c.schema_version == SCHEMA_VERSION

    def test_allocation_pct_stored(self):
        c = _make_config(allocation_pct=0.3)
        assert c.allocation_pct == pytest.approx(0.3)

    def test_rollback_threshold_stored(self):
        c = _make_config(rollback_threshold=-0.15)
        assert c.rollback_threshold == pytest.approx(-0.15)

    def test_minimum_sample_size_stored(self):
        c = _make_config(minimum_sample_size=20)
        assert c.minimum_sample_size == 20


class TestExperimentAssignment:
    def test_creation(self):
        a = ExperimentAssignment(
            experiment_id="EXP_X_2026-05-30",
            symbol="1234.T",
            group=GROUP_CONTROL,
            assigned_at="2026-05-30",
        )
        assert a.group == GROUP_CONTROL

    def test_defaults_schema_version(self):
        a = ExperimentAssignment("e", "sym", GROUP_EXPERIMENT, "2026-05-30")
        assert a.schema_version == SCHEMA_VERSION

    def test_group_field(self):
        a = ExperimentAssignment("e", "sym", GROUP_EXPERIMENT, "2026-05-30")
        assert a.group == GROUP_EXPERIMENT


class TestExperimentEvaluation:
    def test_creation(self):
        ev = ExperimentEvaluation(
            experiment_id="EXP_X",
            date="2026-05-30",
            n_control=10,
            n_experiment=8,
            control_win_rate=0.5,
            experiment_win_rate=0.6,
            status=EVAL_CONTINUE,
        )
        assert ev.status == EVAL_CONTINUE

    def test_defaults_schema_version(self):
        ev = ExperimentEvaluation("e", "2026-05-30", 0, 0, 0.0, 0.0, EVAL_CONTINUE)
        assert ev.schema_version == SCHEMA_VERSION


# ── TestSafeHelpers ────────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float_normal(self):
        assert _safe_float(1.5) == 1.5

    def test_safe_float_none(self):
        assert _safe_float(None, 0.0) == 0.0

    def test_safe_float_string(self):
        assert _safe_float("bad", 99.0) == 99.0

    def test_safe_int_normal(self):
        assert _safe_int(5) == 5

    def test_safe_int_none(self):
        assert _safe_int(None, 3) == 3

    def test_safe_int_string(self):
        assert _safe_int("bad", 0) == 0


# ── TestLoadRegistry ───────────────────────────────────────────────────────────

class TestLoadRegistry:
    def test_missing_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "reg.json"
        result = load_registry(p)
        assert result == []

    def test_empty_experiments_list(self, tmp_path):
        p = tmp_path / "reg.json"
        p.write_text(json.dumps({"experiments": []}), encoding="utf-8")
        result = load_registry(p)
        assert result == []

    def test_single_experiment_loaded(self, tmp_path):
        p = tmp_path / "reg.json"
        _write_registry(p, [_make_config()])
        result = load_registry(p)
        assert len(result) == 1
        assert result[0].experiment_id == "EXP_EXIT_EXTENSION_2026-05-30"

    def test_multiple_experiments_loaded(self, tmp_path):
        p = tmp_path / "reg.json"
        _write_registry(p, [_make_config(), _make_config(exp_id="EXP_B", feature="PRIORITY")])
        result = load_registry(p)
        assert len(result) == 2

    def test_malformed_json_returns_empty_fail_closed(self, tmp_path):
        p = tmp_path / "reg.json"
        p.write_text("not-json", encoding="utf-8")
        result = load_registry(p)
        assert result == []

    def test_non_dict_format_returns_empty(self, tmp_path):
        p = tmp_path / "reg.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = load_registry(p)
        assert result == []

    def test_status_preserved(self, tmp_path):
        p = tmp_path / "reg.json"
        _write_registry(p, [_make_config(status=STATUS_PAUSED)])
        result = load_registry(p)
        assert result[0].status == STATUS_PAUSED

    def test_allocation_pct_preserved(self, tmp_path):
        p = tmp_path / "reg.json"
        _write_registry(p, [_make_config(allocation_pct=0.3)])
        result = load_registry(p)
        assert result[0].allocation_pct == pytest.approx(0.3)

    def test_non_dict_entry_skipped(self, tmp_path):
        p = tmp_path / "reg.json"
        data = {"experiments": [None, {"experiment_id": "EXP_X", "feature_name": "F",
                                        "status": STATUS_ACTIVE, "allocation_pct": 0.5,
                                        "max_positions": 1, "start_date": "2026-05-30",
                                        "approval_source": "PPG",
                                        "rollback_threshold": -0.10,
                                        "minimum_sample_size": 10,
                                        "schema_version": "v1"}]}
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_registry(p)
        assert len(result) == 1

    def test_rollback_threshold_preserved(self, tmp_path):
        p = tmp_path / "reg.json"
        _write_registry(p, [_make_config(rollback_threshold=-0.20)])
        result = load_registry(p)
        assert result[0].rollback_threshold == pytest.approx(-0.20)

    def test_minimum_sample_size_preserved(self, tmp_path):
        p = tmp_path / "reg.json"
        _write_registry(p, [_make_config(minimum_sample_size=25)])
        result = load_registry(p)
        assert result[0].minimum_sample_size == 25


# ── TestSaveRegistry ───────────────────────────────────────────────────────────

class TestSaveRegistry:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "sub" / "reg.json"
        result = save_registry([_make_config()], p)
        assert result is True
        assert p.exists()

    def test_saves_experiments(self, tmp_path):
        p = tmp_path / "reg.json"
        save_registry([_make_config()], p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert len(data["experiments"]) == 1

    def test_round_trip(self, tmp_path):
        p = tmp_path / "reg.json"
        cfg = _make_config(allocation_pct=0.3, minimum_sample_size=20)
        save_registry([cfg], p)
        loaded = load_registry(p)
        assert loaded[0].allocation_pct == pytest.approx(0.3)
        assert loaded[0].minimum_sample_size == 20

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "reg.json"
        save_registry([_make_config()], p)
        save_registry([_make_config(), _make_config(exp_id="B", feature="PRIORITY")], p)
        loaded = load_registry(p)
        assert len(loaded) == 2

    def test_empty_list_saves_ok(self, tmp_path):
        p = tmp_path / "reg.json"
        result = save_registry([], p)
        assert result is True
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["experiments"] == []

    def test_schema_version_written(self, tmp_path):
        p = tmp_path / "reg.json"
        save_registry([_make_config()], p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["schema_version"] == SCHEMA_VERSION

    def test_returns_true_on_success(self, tmp_path):
        p = tmp_path / "reg.json"
        assert save_registry([], p) is True


# ── TestGetActiveExperiments ───────────────────────────────────────────────────

class TestGetActiveExperiments:
    def test_empty_input_returns_empty(self):
        assert get_active_experiments([]) == []

    def test_active_included(self):
        configs = [_make_config(status=STATUS_ACTIVE)]
        result = get_active_experiments(configs)
        assert len(result) == 1

    def test_pending_excluded(self):
        configs = [_make_config(status=STATUS_PENDING)]
        assert get_active_experiments(configs) == []

    def test_paused_excluded(self):
        configs = [_make_config(status=STATUS_PAUSED)]
        assert get_active_experiments(configs) == []

    def test_rolled_back_excluded(self):
        configs = [_make_config(status=STATUS_ROLLED_BACK)]
        assert get_active_experiments(configs) == []

    def test_completed_excluded(self):
        configs = [_make_config(status=STATUS_COMPLETED)]
        assert get_active_experiments(configs) == []

    def test_mixed_returns_only_active(self):
        configs = [
            _make_config(status=STATUS_ACTIVE),
            _make_config(exp_id="B", status=STATUS_PAUSED),
            _make_config(exp_id="C", status=STATUS_PENDING),
        ]
        result = get_active_experiments(configs)
        assert len(result) == 1
        assert result[0].status == STATUS_ACTIVE

    def test_multiple_active_all_returned(self):
        configs = [
            _make_config(exp_id="A", status=STATUS_ACTIVE),
            _make_config(exp_id="B", status=STATUS_ACTIVE),
        ]
        result = get_active_experiments(configs)
        assert len(result) == 2


# ── TestAssignSymbolToExperiment ───────────────────────────────────────────────

class TestAssignSymbolToExperiment:
    def test_returns_assignment(self):
        exp = _make_config(allocation_pct=0.5)
        result = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert isinstance(result, ExperimentAssignment)

    def test_group_is_control_or_experiment(self):
        exp = _make_config(allocation_pct=0.5)
        result = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert result.group in (GROUP_CONTROL, GROUP_EXPERIMENT)

    def test_deterministic_same_symbol(self):
        exp = _make_config(allocation_pct=0.5)
        r1 = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        r2 = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert r1.group == r2.group

    def test_deterministic_different_dates_same_result(self):
        exp = _make_config(allocation_pct=0.5)
        r1 = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        r2 = assign_symbol_to_experiment("1234.T", exp, "2026-06-01")
        assert r1.group == r2.group

    def test_allocation_zero_always_control(self):
        exp = _make_config(allocation_pct=0.0)
        for sym in ["1234.T", "5678.T", "9999.T", "AAPL", "TSLA"]:
            result = assign_symbol_to_experiment(sym, exp, "2026-05-30")
            assert result.group == GROUP_CONTROL, f"{sym} should be CONTROL"

    def test_allocation_one_always_experiment(self):
        exp = _make_config(allocation_pct=1.0)
        for sym in ["1234.T", "5678.T", "9999.T", "AAPL", "TSLA"]:
            result = assign_symbol_to_experiment(sym, exp, "2026-05-30")
            assert result.group == GROUP_EXPERIMENT, f"{sym} should be EXPERIMENT"

    def test_allocation_clamped_above_one(self):
        exp = _make_config(allocation_pct=1.5)  # clamped to 1.0
        result = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert result.group == GROUP_EXPERIMENT

    def test_allocation_clamped_below_zero(self):
        exp = _make_config(allocation_pct=-0.5)  # clamped to 0.0
        result = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert result.group == GROUP_CONTROL

    def test_experiment_id_set_correctly(self):
        exp = _make_config(exp_id="MY_EXP")
        result = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert result.experiment_id == "MY_EXP"

    def test_symbol_set_correctly(self):
        exp = _make_config()
        result = assign_symbol_to_experiment("5678.T", exp, "2026-05-30")
        assert result.symbol == "5678.T"

    def test_assigned_at_set_correctly(self):
        exp = _make_config()
        result = assign_symbol_to_experiment("1234.T", exp, "2026-05-30")
        assert result.assigned_at == "2026-05-30"

    def test_different_symbols_can_get_different_groups(self):
        """With 50% allocation, a large enough set should yield both groups."""
        exp = _make_config(allocation_pct=0.5)
        symbols = [f"{i:04d}.T" for i in range(100)]
        groups = {assign_symbol_to_experiment(s, exp, "2026-05-30").group for s in symbols}
        assert GROUP_CONTROL in groups
        assert GROUP_EXPERIMENT in groups

    def test_sha256_based_routing(self):
        """Verify routing uses sha256."""
        exp = _make_config(allocation_pct=0.5)
        sym = "1234.T"
        hash_hex = hashlib.sha256(sym.encode()).hexdigest()
        hash_int  = int(hash_hex, 16)
        bucket    = (hash_int % 10_000) / 10_000.0
        expected_group = GROUP_EXPERIMENT if bucket < 0.5 else GROUP_CONTROL
        result = assign_symbol_to_experiment(sym, exp, "2026-05-30")
        assert result.group == expected_group

    def test_routing_stable_across_experiment_id_change(self):
        """Group depends only on symbol hash and allocation_pct, not experiment_id."""
        exp1 = _make_config(exp_id="EXP_A", allocation_pct=0.5)
        exp2 = _make_config(exp_id="EXP_B", allocation_pct=0.5)
        r1 = assign_symbol_to_experiment("1234.T", exp1, "2026-05-30")
        r2 = assign_symbol_to_experiment("1234.T", exp2, "2026-05-30")
        assert r1.group == r2.group


# ── TestRouteSymbolsToExperiments ──────────────────────────────────────────────

class TestRouteSymbolsToExperiments:
    def test_empty_symbols_empty_result(self):
        exp = _make_config()
        result = route_symbols_to_experiments([], [exp], "2026-05-30")
        assert result == {}

    def test_empty_experiments_empty_result(self):
        result = route_symbols_to_experiments(["1234.T"], [], "2026-05-30")
        assert result == {}

    def test_single_symbol_single_experiment(self):
        exp = _make_config()
        result = route_symbols_to_experiments(["1234.T"], [exp], "2026-05-30")
        assert "1234.T" in result
        assert len(result["1234.T"]) == 1

    def test_multiple_symbols_routed(self):
        exp = _make_config()
        syms = ["1234.T", "5678.T", "9999.T"]
        result = route_symbols_to_experiments(syms, [exp], "2026-05-30")
        assert len(result) == 3

    def test_multiple_experiments_each_gets_assignment(self):
        exp1 = _make_config(exp_id="A")
        exp2 = _make_config(exp_id="B")
        result = route_symbols_to_experiments(["1234.T"], [exp1, exp2], "2026-05-30")
        assert len(result["1234.T"]) == 2

    def test_deterministic_output(self):
        exp = _make_config()
        r1 = route_symbols_to_experiments(["1234.T"], [exp], "2026-05-30")
        r2 = route_symbols_to_experiments(["1234.T"], [exp], "2026-05-30")
        assert r1["1234.T"][0].group == r2["1234.T"][0].group


# ── TestAppendAssignment ───────────────────────────────────────────────────────

class TestAppendAssignment:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "assignments.jsonl"
        a = ExperimentAssignment("EXP_A", "1234.T", GROUP_CONTROL, "2026-05-30")
        append_assignment(a, p)
        assert p.exists()

    def test_appends_record(self, tmp_path):
        p = tmp_path / "assignments.jsonl"
        a = ExperimentAssignment("EXP_A", "1234.T", GROUP_CONTROL, "2026-05-30")
        append_assignment(a, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["symbol"] == "1234.T"
        assert data["group"] == GROUP_CONTROL

    def test_two_appends_two_lines(self, tmp_path):
        p = tmp_path / "assignments.jsonl"
        a1 = ExperimentAssignment("EXP_A", "1234.T", GROUP_CONTROL, "2026-05-30")
        a2 = ExperimentAssignment("EXP_A", "5678.T", GROUP_EXPERIMENT, "2026-05-30")
        append_assignment(a1, p)
        append_assignment(a2, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2


# ── TestEvaluateAssignment ─────────────────────────────────────────────────────

class TestEvaluateAssignment:
    def _make_perf_record(self, group, pnl=100.0, max_drawdown=-0.05):
        return {"group": group, "pnl": pnl, "max_drawdown": max_drawdown,
                "win_rate": 1.0 if pnl > 0 else 0.0}

    def test_below_min_sample_returns_continue(self):
        exp = _make_config(minimum_sample_size=10)
        records = [self._make_perf_record(GROUP_EXPERIMENT) for _ in range(5)]
        result = evaluate_assignment(exp, records, "2026-05-30")
        assert result.status == EVAL_CONTINUE

    def test_rollback_on_bad_drawdown(self):
        exp = _make_config(rollback_threshold=-0.10, minimum_sample_size=3)
        records = [
            self._make_perf_record(GROUP_EXPERIMENT, pnl=100.0, max_drawdown=-0.15),
        ] * 3
        result = evaluate_assignment(exp, records, "2026-05-30")
        assert result.status == EVAL_ROLLBACK

    def test_promote_on_win_rate_edge(self):
        exp = _make_config(rollback_threshold=-0.30, minimum_sample_size=5)
        # CONTROL: 50% win rate (5 wins, 5 losses)
        ctrl = (
            [self._make_perf_record(GROUP_CONTROL, pnl=100.0)] * 5 +
            [self._make_perf_record(GROUP_CONTROL, pnl=-50.0)] * 5
        )
        # EXPERIMENT: 60% win rate (6 wins, 4 losses) → beats CONTROL by 10pp ≥ 5pp
        exp_records = (
            [self._make_perf_record(GROUP_EXPERIMENT, pnl=100.0)] * 6 +
            [self._make_perf_record(GROUP_EXPERIMENT, pnl=-50.0)] * 4
        )
        records = ctrl + exp_records
        result = evaluate_assignment(exp, records, "2026-05-30")
        assert result.status == EVAL_PROMOTE

    def test_continue_when_no_clear_winner(self):
        exp = _make_config(rollback_threshold=-0.30, minimum_sample_size=3)
        records = (
            [self._make_perf_record(GROUP_CONTROL, pnl=100.0)] * 5 +
            [self._make_perf_record(GROUP_EXPERIMENT, pnl=100.0)] * 5
        )
        result = evaluate_assignment(exp, records, "2026-05-30")
        assert result.status == EVAL_CONTINUE

    def test_n_control_n_experiment_counted(self):
        exp = _make_config(minimum_sample_size=1)
        records = (
            [self._make_perf_record(GROUP_CONTROL)] * 4 +
            [self._make_perf_record(GROUP_EXPERIMENT)] * 3
        )
        result = evaluate_assignment(exp, records, "2026-05-30")
        assert result.n_control == 4
        assert result.n_experiment == 3

    def test_empty_records_returns_continue(self):
        exp = _make_config(minimum_sample_size=10)
        result = evaluate_assignment(exp, [], "2026-05-30")
        assert result.status == EVAL_CONTINUE
        assert result.n_control == 0
        assert result.n_experiment == 0

    def test_experiment_id_set_in_evaluation(self):
        exp = _make_config(exp_id="MY_EXP")
        result = evaluate_assignment(exp, [], "2026-05-30")
        assert result.experiment_id == "MY_EXP"

    def test_date_set_in_evaluation(self):
        exp = _make_config()
        result = evaluate_assignment(exp, [], "2026-06-01")
        assert result.date == "2026-06-01"

    def test_win_rate_computed(self):
        exp = _make_config(minimum_sample_size=1)
        records = [
            {"group": GROUP_EXPERIMENT, "pnl": 100.0, "max_drawdown": -0.02, "win_rate": 1.0},
            {"group": GROUP_EXPERIMENT, "pnl": -50.0, "max_drawdown": -0.05, "win_rate": 0.0},
        ]
        result = evaluate_assignment(exp, records, "2026-05-30")
        assert result.experiment_win_rate == pytest.approx(0.5)


# ── TestCreateExperimentsFromGateApproval ──────────────────────────────────────

class TestCreateExperimentsFromGateApproval:
    def _gate_recs(self, decisions):
        return [{"decision": d, "candidate": c} for d, c in decisions]

    def test_approved_creates_experiment(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("APPROVED", "EXIT_EXTENSION")])
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert len(result) == 1
        assert result[0].feature_name == "EXIT_EXTENSION"
        assert result[0].status == STATUS_ACTIVE

    def test_review_required_skipped(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("REVIEW_REQUIRED", "EXIT_EXTENSION")])
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert result == []

    def test_rejected_skipped(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("REJECTED", "EXIT_EXTENSION")])
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert result == []

    def test_multiple_approved_creates_multiple(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("APPROVED", "EXIT_EXTENSION"), ("APPROVED", "PRIORITY")])
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert len(result) == 2

    def test_duplicate_feature_not_created_twice(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("APPROVED", "EXIT_EXTENSION")])
        create_experiments_from_gate_approval(recs, p, "2026-05-30")
        result2 = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        # No new experiments on second call
        assert result2 == []

    def test_existing_active_feature_not_duplicated(self, tmp_path):
        p = tmp_path / "reg.json"
        existing = _make_config(feature="EXIT_EXTENSION", status=STATUS_ACTIVE)
        _write_registry(p, [existing])
        recs = self._gate_recs([("APPROVED", "EXIT_EXTENSION")])
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert result == []

    def test_registry_updated_on_disk(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("APPROVED", "EXIT_EXTENSION")])
        create_experiments_from_gate_approval(recs, p, "2026-05-30")
        loaded = load_registry(p)
        assert len(loaded) == 1
        assert loaded[0].approval_source == "PRODUCTION_PROMOTION_GATE"

    def test_approval_source_set_correctly(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("APPROVED", "PRIORITY")])
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert result[0].approval_source == "PRODUCTION_PROMOTION_GATE"

    def test_start_date_set_to_today(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = self._gate_recs([("APPROVED", "SIGNAL_LEADER")])
        result = create_experiments_from_gate_approval(recs, p, "2026-06-15")
        assert result[0].start_date == "2026-06-15"

    def test_empty_gate_records_returns_empty(self, tmp_path):
        p = tmp_path / "reg.json"
        result = create_experiments_from_gate_approval([], p, "2026-05-30")
        assert result == []

    def test_non_dict_entries_in_gate_records_skipped(self, tmp_path):
        p = tmp_path / "reg.json"
        recs = [None, "bad", {"decision": "APPROVED", "candidate": "PRIORITY"}]
        result = create_experiments_from_gate_approval(recs, p, "2026-05-30")
        assert len(result) == 1


# ── TestFormatExperimentReport ─────────────────────────────────────────────────

class TestFormatExperimentReport:
    def test_empty_experiments_returns_empty(self):
        result = format_experiment_report([], {})
        assert result == ""

    def test_report_contains_header(self):
        exp = _make_config()
        rpt = format_experiment_report([exp], {})
        assert "ACTIVE EXPERIMENT REGISTRY" in rpt

    def test_report_contains_experiment_id(self):
        exp = _make_config(exp_id="EXP_UNIQUE_ID")
        rpt = format_experiment_report([exp], {})
        assert "EXP_UNIQUE_ID" in rpt

    def test_report_contains_routing_when_provided(self):
        exp = _make_config()
        routing = {"1234.T": [ExperimentAssignment(exp.experiment_id, "1234.T", GROUP_CONTROL, "2026-05-30")]}
        rpt = format_experiment_report([exp], routing)
        assert "1234.T" in rpt
        assert "CONTROL" in rpt

    def test_separator_lines_present(self):
        exp = _make_config()
        rpt = format_experiment_report([exp], {})
        assert "═" in rpt


# ── TestIntegration ────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_round_trip(self, tmp_path):
        p = tmp_path / "reg.json"
        cfg = _make_config(allocation_pct=0.5)
        save_registry([cfg], p)
        loaded = load_registry(p)
        active = get_active_experiments(loaded)
        assert len(active) == 1
        asgn = assign_symbol_to_experiment("1234.T", active[0], "2026-05-30")
        assert asgn.group in (GROUP_CONTROL, GROUP_EXPERIMENT)

    def test_gate_approval_to_registry_to_routing(self, tmp_path):
        p = tmp_path / "reg.json"
        gate_recs = [{"decision": "APPROVED", "candidate": "EXIT_EXTENSION"}]
        new = create_experiments_from_gate_approval(gate_recs, p, "2026-05-30")
        assert len(new) == 1
        loaded = load_registry(p)
        active = get_active_experiments(loaded)
        assert len(active) == 1
        routing = route_symbols_to_experiments(["1234.T", "5678.T"], active, "2026-05-30")
        assert len(routing) == 2

    def test_fail_closed_missing_registry(self, tmp_path):
        p = tmp_path / "missing_reg.json"
        loaded = load_registry(p)
        active = get_active_experiments(loaded)
        assert active == []
        # With no active experiments, routing returns empty → CONTROL default
        routing = route_symbols_to_experiments(["1234.T"], active, "2026-05-30")
        assert routing == {}

    def test_paths_constants_importable(self):
        from src.paths import (
            EXPERIMENTS_DIR,
            ACTIVE_EXPERIMENT_REGISTRY_FILE,
            EXPERIMENT_ASSIGNMENT_FILE,
            EXPERIMENT_PERFORMANCE_FILE,
        )
        assert EXPERIMENTS_DIR.name == "experiments"
        assert ACTIVE_EXPERIMENT_REGISTRY_FILE.suffix == ".json"
        assert EXPERIMENT_ASSIGNMENT_FILE.suffix == ".jsonl"
        assert EXPERIMENT_PERFORMANCE_FILE.suffix == ".jsonl"

    def test_no_execution_imports(self):
        import src.experiments.active_experiment_registry as m
        src_text = Path(m.__file__).read_text(encoding="utf-8")
        assert "signal_bridge" not in src_text
        assert "place_order" not in src_text
        assert "kabu" not in src_text.lower()

    def test_evaluate_then_rollback_status(self, tmp_path):
        p = tmp_path / "reg.json"
        cfg = _make_config(rollback_threshold=-0.10, minimum_sample_size=2)
        save_registry([cfg], p)
        loaded = load_registry(p)
        exp = get_active_experiments(loaded)[0]
        records = [
            {"group": GROUP_EXPERIMENT, "pnl": -200.0, "max_drawdown": -0.20, "win_rate": 0.0},
            {"group": GROUP_EXPERIMENT, "pnl": -100.0, "max_drawdown": -0.15, "win_rate": 0.0},
            {"group": GROUP_CONTROL, "pnl": 100.0, "max_drawdown": -0.02, "win_rate": 1.0},
        ]
        ev = evaluate_assignment(exp, records, "2026-05-30")
        assert ev.status == EVAL_ROLLBACK

    def test_determinism_across_multiple_calls(self):
        exp = _make_config(allocation_pct=0.5)
        sym = "7203.T"
        results = [assign_symbol_to_experiment(sym, exp, "2026-05-30").group for _ in range(10)]
        assert len(set(results)) == 1  # all same group
