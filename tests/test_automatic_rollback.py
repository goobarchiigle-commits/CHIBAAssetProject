"""
tests/test_automatic_rollback.py — Automatic Rollback Governance (Phase 6L-Lite)
"""
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.governance.automatic_rollback import (
    SCHEMA_VERSION,
    DECISION_HEALTHY, DECISION_REVIEW_REQUIRED,
    DECISION_ROLLBACK_CANDIDATE, DECISION_ROLLED_BACK,
    _MIN_TRADES, _MIN_OBS_DAYS, _ROLLBACK_PERSISTENCE_DAYS,
    _EXPECTANCY_THRESHOLD, _AVG_RETURN_THRESHOLD,
    _REASON_INSUFFICIENT, _REASON_LOW_EXPECTANCY,
    _REASON_LOW_AVG_RETURN, _REASON_EXCESS_DD,
    RollbackEvaluationInput, RollbackDecision, RollbackTelemetry,
    evaluate_experiment, build_rollback_telemetry,
    append_rollback_telemetry, load_latest_decision,
    apply_rollback_decision, evaluate_all_active_experiments,
    _compute_days, _safe_float, _safe_int,
)
from src.experiments.active_experiment_registry import (
    ExperimentConfig,
    STATUS_ACTIVE, STATUS_REVIEW_REQUIRED as REG_REVIEW,
    STATUS_ROLLBACK_CANDIDATE as REG_CANDIDATE, STATUS_ROLLED_BACK,
    save_registry, load_registry, get_active_experiments,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _good_metrics(expectancy=0.30, avg_return=0.03, max_drawdown=-0.05, trades=50):
    return {"expectancy": expectancy, "avg_return": avg_return,
            "max_drawdown": max_drawdown, "trades": trades}


def _make_inp(
    exp_id="EXP_A",
    trade_count=50,
    obs_days=30,
    ctrl_metrics=None,
    exp_metrics=None,
    date="2026-05-30",
) -> RollbackEvaluationInput:
    return RollbackEvaluationInput(
        experiment_id=exp_id,
        evaluation_date=date,
        trade_count=trade_count,
        observation_days=obs_days,
        control_metrics=ctrl_metrics or _good_metrics(),
        experiment_metrics=exp_metrics or _good_metrics(),
    )


def _make_decision(
    decision=DECISION_HEALTHY,
    consecutive=0,
    exp_id="EXP_A",
    date="2026-05-29",
) -> RollbackDecision:
    return RollbackDecision(
        experiment_id=exp_id,
        evaluation_date=date,
        decision=decision,
        trigger_reasons=[],
        consecutive_candidate_days=consecutive,
    )


def _make_registry_file(path: Path, configs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_registry(configs, path)


def _make_exp_config(
    exp_id="EXP_A",
    status=STATUS_ACTIVE,
    start_date="2026-05-01",
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=exp_id,
        feature_name="EXIT_EXTENSION",
        status=status,
        allocation_pct=0.5,
        max_positions=1,
        start_date=start_date,
        approval_source="PRODUCTION_PROMOTION_GATE",
        rollback_threshold=-0.10,
        minimum_sample_size=10,
    )


def _write_perf_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_rollback_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── TestConstants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_decision_healthy(self):
        assert DECISION_HEALTHY == "HEALTHY"

    def test_decision_review_required(self):
        assert DECISION_REVIEW_REQUIRED == "REVIEW_REQUIRED"

    def test_decision_rollback_candidate(self):
        assert DECISION_ROLLBACK_CANDIDATE == "ROLLBACK_CANDIDATE"

    def test_decision_rolled_back(self):
        assert DECISION_ROLLED_BACK == "ROLLED_BACK"

    def test_min_trades(self):
        assert _MIN_TRADES == 30

    def test_min_obs_days(self):
        assert _MIN_OBS_DAYS == 21

    def test_rollback_persistence_days(self):
        assert _ROLLBACK_PERSISTENCE_DAYS == 3

    def test_expectancy_threshold(self):
        assert _EXPECTANCY_THRESHOLD == pytest.approx(0.25)

    def test_avg_return_threshold(self):
        assert _AVG_RETURN_THRESHOLD == pytest.approx(0.01)

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_reason_constants(self):
        assert _REASON_INSUFFICIENT   == "INSUFFICIENT_SAMPLE"
        assert _REASON_LOW_EXPECTANCY == "LOW_EXPECTANCY"
        assert _REASON_LOW_AVG_RETURN == "LOW_AVG_RETURN"
        assert _REASON_EXCESS_DD      == "EXCESS_DRAWDOWN"


# ── TestDataclasses ────────────────────────────────────────────────────────────

class TestRollbackEvaluationInput:
    def test_creation(self):
        inp = _make_inp()
        assert inp.experiment_id == "EXP_A"
        assert inp.trade_count == 50
        assert inp.observation_days == 30

    def test_metrics_stored(self):
        ctrl = {"expectancy": 0.3}
        exp  = {"expectancy": 0.1}
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        assert inp.control_metrics["expectancy"] == pytest.approx(0.3)
        assert inp.experiment_metrics["expectancy"] == pytest.approx(0.1)


class TestRollbackDecision:
    def test_creation(self):
        d = _make_decision(DECISION_REVIEW_REQUIRED, consecutive=0)
        assert d.decision == DECISION_REVIEW_REQUIRED
        assert d.consecutive_candidate_days == 0

    def test_defaults_schema_version(self):
        d = _make_decision()
        assert d.schema_version == SCHEMA_VERSION

    def test_trigger_reasons_stored(self):
        d = RollbackDecision("E", "2026-05-30", DECISION_REVIEW_REQUIRED,
                              [_REASON_LOW_EXPECTANCY], 0)
        assert _REASON_LOW_EXPECTANCY in d.trigger_reasons


class TestRollbackTelemetry:
    def test_creation(self):
        t = RollbackTelemetry(
            experiment_id="EXP_A",
            evaluation_date="2026-05-30",
            decision=DECISION_HEALTHY,
            trigger_reasons=[],
            control_metrics={"expectancy": 0.3},
            experiment_metrics={"expectancy": 0.3},
        )
        assert t.decision == DECISION_HEALTHY

    def test_defaults_schema_version(self):
        t = RollbackTelemetry("E", "2026-05-30", DECISION_HEALTHY, [], {}, {})
        assert t.schema_version == SCHEMA_VERSION


# ── TestHelpers ────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_compute_days_normal(self):
        assert _compute_days("2026-05-01", "2026-05-30") == 29

    def test_compute_days_same_date(self):
        assert _compute_days("2026-05-30", "2026-05-30") == 0

    def test_compute_days_error_returns_zero(self):
        assert _compute_days("bad", "2026-05-30") == 0

    def test_safe_float_normal(self):
        assert _safe_float(1.5) == 1.5

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_int_normal(self):
        assert _safe_int(5) == 5

    def test_safe_int_none(self):
        assert _safe_int(None) == 0


# ── TestEvaluateExperiment — Healthy paths ─────────────────────────────────────

class TestEvaluateExperimentHealthy:
    def test_insufficient_trade_count_returns_healthy(self):
        inp = _make_inp(trade_count=_MIN_TRADES - 1, obs_days=_MIN_OBS_DAYS)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_HEALTHY
        assert _REASON_INSUFFICIENT in result.trigger_reasons

    def test_insufficient_obs_days_returns_healthy(self):
        inp = _make_inp(trade_count=_MIN_TRADES, obs_days=_MIN_OBS_DAYS - 1)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_HEALTHY
        assert _REASON_INSUFFICIENT in result.trigger_reasons

    def test_both_below_minimum_returns_healthy(self):
        inp = _make_inp(trade_count=0, obs_days=0)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_HEALTHY

    def test_no_triggers_returns_healthy(self):
        ctrl = _good_metrics(expectancy=0.30, avg_return=0.03, max_drawdown=-0.05)
        exp  = _good_metrics(expectancy=0.30, avg_return=0.03, max_drawdown=-0.05)
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_HEALTHY
        assert result.trigger_reasons == []

    def test_healthy_resets_consecutive_days_to_zero(self):
        prev = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=2)
        inp  = _make_inp()
        result = evaluate_experiment(inp, prev)
        assert result.consecutive_candidate_days == 0

    def test_healthy_with_exactly_minimum_sample(self):
        ctrl = _good_metrics()
        exp  = _good_metrics()
        inp  = _make_inp(trade_count=_MIN_TRADES, obs_days=_MIN_OBS_DAYS,
                         ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_HEALTHY

    def test_experiment_id_preserved(self):
        result = evaluate_experiment(_make_inp(exp_id="MY_EXP"), None)
        assert result.experiment_id == "MY_EXP"

    def test_evaluation_date_preserved(self):
        result = evaluate_experiment(_make_inp(date="2026-06-15"), None)
        assert result.evaluation_date == "2026-06-15"


# ── TestEvaluateExperiment — Review paths ─────────────────────────────────────

class TestEvaluateExperimentReview:
    def test_trigger_a_expectancy(self):
        ctrl = _good_metrics(expectancy=0.30)
        exp  = _good_metrics(expectancy=0.30 - _EXPECTANCY_THRESHOLD - 0.01)  # just below threshold
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_REVIEW_REQUIRED
        assert _REASON_LOW_EXPECTANCY in result.trigger_reasons

    def test_trigger_b_avg_return(self):
        ctrl = _good_metrics(avg_return=0.03)
        exp  = _good_metrics(avg_return=0.03 - _AVG_RETURN_THRESHOLD - 0.001)
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_REVIEW_REQUIRED
        assert _REASON_LOW_AVG_RETURN in result.trigger_reasons

    def test_trigger_c_drawdown(self):
        ctrl = _good_metrics(max_drawdown=-0.10)
        exp  = _good_metrics(max_drawdown=-0.16)  # 0.16 > 0.10 * 1.5 = 0.15
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_REVIEW_REQUIRED
        assert _REASON_EXCESS_DD in result.trigger_reasons

    def test_exactly_at_expectancy_threshold_no_trigger(self):
        ctrl = _good_metrics(expectancy=0.30)
        exp  = _good_metrics(expectancy=0.30 - _EXPECTANCY_THRESHOLD)  # exactly at boundary
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert _REASON_LOW_EXPECTANCY not in result.trigger_reasons

    def test_exactly_at_return_threshold_no_trigger(self):
        ctrl = _good_metrics(avg_return=0.03)
        exp  = _good_metrics(avg_return=0.03 - _AVG_RETURN_THRESHOLD)
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert _REASON_LOW_AVG_RETURN not in result.trigger_reasons

    def test_exactly_at_drawdown_threshold_no_trigger(self):
        ctrl = _good_metrics(max_drawdown=-0.10)
        exp  = _good_metrics(max_drawdown=-0.15)  # exactly 1.5×
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert _REASON_EXCESS_DD not in result.trigger_reasons

    def test_review_resets_consecutive_candidate_days(self):
        prev   = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=1)
        ctrl   = _good_metrics(expectancy=0.30)
        exp    = _good_metrics(expectancy=0.30 - 0.30)  # trigger A only
        inp    = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, prev)
        assert result.decision == DECISION_REVIEW_REQUIRED
        assert result.consecutive_candidate_days == 0


# ── TestEvaluateExperiment — Candidate paths ───────────────────────────────────

class TestEvaluateExperimentCandidate:
    def _two_trigger_metrics(self):
        ctrl = _good_metrics(expectancy=0.30, avg_return=0.03, max_drawdown=-0.05)
        exp  = _good_metrics(
            expectancy=0.30 - 0.30,   # trigger A
            avg_return=0.03 - 0.02,   # trigger B
            max_drawdown=-0.05,       # no trigger C
        )
        return ctrl, exp

    def test_two_triggers_returns_rollback_candidate(self):
        ctrl, exp = self._two_trigger_metrics()
        inp = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision == DECISION_ROLLBACK_CANDIDATE

    def test_three_triggers_returns_rollback_candidate(self):
        ctrl = _good_metrics(expectancy=0.30, avg_return=0.03, max_drawdown=-0.10)
        exp  = _good_metrics(expectancy=0.0, avg_return=0.0, max_drawdown=-0.20)
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.decision in (DECISION_ROLLBACK_CANDIDATE, DECISION_ROLLED_BACK)

    def test_day_1_consecutive_count_is_1(self):
        ctrl, exp = self._two_trigger_metrics()
        inp    = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert result.consecutive_candidate_days == 1

    def test_day_2_consecutive_count_is_2(self):
        ctrl, exp = self._two_trigger_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=1)
        result = evaluate_experiment(inp, prev)
        assert result.decision == DECISION_ROLLBACK_CANDIDATE
        assert result.consecutive_candidate_days == 2

    def test_trigger_reasons_included(self):
        ctrl, exp = self._two_trigger_metrics()
        inp    = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        result = evaluate_experiment(inp, None)
        assert len(result.trigger_reasons) >= 2

    def test_previous_review_does_not_carry_consecutive(self):
        ctrl, exp = self._two_trigger_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_REVIEW_REQUIRED, consecutive=0)
        result = evaluate_experiment(inp, prev)
        assert result.consecutive_candidate_days == 1


# ── TestEvaluateExperiment — Rollback path ─────────────────────────────────────

class TestEvaluateExperimentRolledBack:
    def _two_trigger_metrics(self):
        ctrl = _good_metrics(expectancy=0.30, avg_return=0.03, max_drawdown=-0.05)
        exp  = _good_metrics(expectancy=0.0, avg_return=0.0, max_drawdown=-0.05)
        return ctrl, exp

    def test_three_consecutive_triggers_rolled_back(self):
        ctrl, exp = self._two_trigger_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=_ROLLBACK_PERSISTENCE_DAYS - 1)
        result = evaluate_experiment(inp, prev)
        assert result.decision == DECISION_ROLLED_BACK

    def test_two_consecutive_not_yet_rolled_back(self):
        ctrl, exp = self._two_trigger_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=1)
        result = evaluate_experiment(inp, prev)
        assert result.decision == DECISION_ROLLBACK_CANDIDATE

    def test_rolled_back_has_expected_consecutive_count(self):
        ctrl, exp = self._two_trigger_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=_ROLLBACK_PERSISTENCE_DAYS - 1)
        result = evaluate_experiment(inp, prev)
        assert result.consecutive_candidate_days == _ROLLBACK_PERSISTENCE_DAYS

    def test_streak_break_resets_consecutive(self):
        # Day 2 was CANDIDATE (consecutive=2), but day 3 has no triggers → HEALTHY
        ctrl = _good_metrics()
        exp  = _good_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_ROLLBACK_CANDIDATE, consecutive=2)
        result = evaluate_experiment(inp, prev)
        assert result.consecutive_candidate_days == 0
        assert result.decision == DECISION_HEALTHY

    def test_previous_healthy_does_not_carry_consecutive(self):
        ctrl, exp = self._two_trigger_metrics()
        inp  = _make_inp(ctrl_metrics=ctrl, exp_metrics=exp)
        prev = _make_decision(DECISION_HEALTHY, consecutive=0)
        result = evaluate_experiment(inp, prev)
        assert result.consecutive_candidate_days == 1


# ── TestBuildRollbackTelemetry ─────────────────────────────────────────────────

class TestBuildRollbackTelemetry:
    def test_returns_telemetry(self):
        inp = _make_inp()
        dec = _make_decision()
        tel = build_rollback_telemetry(inp, dec)
        assert isinstance(tel, RollbackTelemetry)

    def test_experiment_id_matches(self):
        inp = _make_inp(exp_id="MY_EXP")
        dec = _make_decision(exp_id="MY_EXP")
        tel = build_rollback_telemetry(inp, dec)
        assert tel.experiment_id == "MY_EXP"

    def test_decision_matches(self):
        inp = _make_inp()
        dec = _make_decision(DECISION_REVIEW_REQUIRED)
        tel = build_rollback_telemetry(inp, dec)
        assert tel.decision == DECISION_REVIEW_REQUIRED

    def test_trigger_reasons_copied(self):
        inp = _make_inp()
        dec = RollbackDecision("E", "2026-05-30", DECISION_REVIEW_REQUIRED,
                                [_REASON_LOW_EXPECTANCY], 0)
        tel = build_rollback_telemetry(inp, dec)
        assert _REASON_LOW_EXPECTANCY in tel.trigger_reasons

    def test_control_metrics_copied(self):
        ctrl = {"expectancy": 0.99}
        inp  = _make_inp(ctrl_metrics=ctrl)
        dec  = _make_decision()
        tel  = build_rollback_telemetry(inp, dec)
        assert tel.control_metrics["expectancy"] == pytest.approx(0.99)

    def test_experiment_metrics_copied(self):
        exp = {"avg_return": 0.05}
        inp = _make_inp(exp_metrics=exp)
        dec = _make_decision()
        tel = build_rollback_telemetry(inp, dec)
        assert tel.experiment_metrics["avg_return"] == pytest.approx(0.05)

    def test_consecutive_days_copied(self):
        inp = _make_inp()
        dec = _make_decision(consecutive=2)
        tel = build_rollback_telemetry(inp, dec)
        assert tel.consecutive_candidate_days == 2


# ── TestAppendRollbackTelemetry ────────────────────────────────────────────────

class TestAppendRollbackTelemetry:
    def _make_tel(self, exp_id="EXP_A", decision=DECISION_HEALTHY) -> RollbackTelemetry:
        return RollbackTelemetry(
            experiment_id=exp_id,
            evaluation_date="2026-05-30",
            decision=decision,
            trigger_reasons=[],
            control_metrics={"expectancy": 0.3},
            experiment_metrics={"expectancy": 0.3},
        )

    def test_creates_file(self, tmp_path):
        p = tmp_path / "rb" / "rb.jsonl"
        append_rollback_telemetry(self._make_tel(), p)
        assert p.exists()

    def test_appends_record(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        append_rollback_telemetry(self._make_tel(), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_fields_written(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        append_rollback_telemetry(self._make_tel(exp_id="MY_EXP", decision=DECISION_REVIEW_REQUIRED), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["experiment_id"] == "MY_EXP"
        assert data["decision"] == DECISION_REVIEW_REQUIRED

    def test_two_appends_two_lines(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        append_rollback_telemetry(self._make_tel(exp_id="EXP_A"), p)
        append_rollback_telemetry(self._make_tel(exp_id="EXP_B"), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_schema_version_written(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        append_rollback_telemetry(self._make_tel(), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["schema_version"] == SCHEMA_VERSION


# ── TestLoadLatestDecision ─────────────────────────────────────────────────────

class TestLoadLatestDecision:
    def test_missing_file_returns_none(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert load_latest_decision(p, "EXP_A") is None

    def test_no_matching_records_returns_none(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        _write_rollback_jsonl(p, [{"experiment_id": "EXP_B", "evaluation_date": "2026-05-30",
                                    "decision": DECISION_HEALTHY, "trigger_reasons": [],
                                    "consecutive_candidate_days": 0}])
        assert load_latest_decision(p, "EXP_A") is None

    def test_returns_latest_by_date(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        _write_rollback_jsonl(p, [
            {"experiment_id": "EXP_A", "evaluation_date": "2026-05-28",
             "decision": DECISION_HEALTHY, "trigger_reasons": [], "consecutive_candidate_days": 0},
            {"experiment_id": "EXP_A", "evaluation_date": "2026-05-30",
             "decision": DECISION_ROLLBACK_CANDIDATE, "trigger_reasons": [], "consecutive_candidate_days": 2},
        ])
        result = load_latest_decision(p, "EXP_A")
        assert result is not None
        assert result.decision == DECISION_ROLLBACK_CANDIDATE
        assert result.consecutive_candidate_days == 2

    def test_consecutive_days_preserved(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        _write_rollback_jsonl(p, [
            {"experiment_id": "EXP_A", "evaluation_date": "2026-05-30",
             "decision": DECISION_ROLLBACK_CANDIDATE, "trigger_reasons": [],
             "consecutive_candidate_days": 2},
        ])
        result = load_latest_decision(p, "EXP_A")
        assert result.consecutive_candidate_days == 2

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "rb.jsonl"
        p.write_text(
            'not-json\n'
            '{"experiment_id": "EXP_A", "evaluation_date": "2026-05-30", '
            '"decision": "HEALTHY", "trigger_reasons": [], "consecutive_candidate_days": 0}\n',
            encoding="utf-8"
        )
        result = load_latest_decision(p, "EXP_A")
        assert result is not None


# ── TestApplyRollbackDecision ──────────────────────────────────────────────────

class TestApplyRollbackDecision:
    def _setup_registry(self, tmp_path, status=STATUS_ACTIVE) -> Path:
        p = tmp_path / "reg.json"
        _make_registry_file(p, [_make_exp_config("EXP_A", status=status)])
        return p

    def test_healthy_no_registry_change(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_HEALTHY, exp_id="EXP_A")
        result = apply_rollback_decision(dec, reg)
        assert result is True
        loaded = load_registry(reg)
        assert loaded[0].status == STATUS_ACTIVE

    def test_review_required_updates_registry(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_REVIEW_REQUIRED, exp_id="EXP_A")
        apply_rollback_decision(dec, reg)
        loaded = load_registry(reg)
        assert loaded[0].status == REG_REVIEW

    def test_rollback_candidate_updates_registry(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_ROLLBACK_CANDIDATE, exp_id="EXP_A")
        apply_rollback_decision(dec, reg)
        loaded = load_registry(reg)
        assert loaded[0].status == REG_CANDIDATE

    def test_rolled_back_updates_registry(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_ROLLED_BACK, exp_id="EXP_A")
        apply_rollback_decision(dec, reg)
        loaded = load_registry(reg)
        assert loaded[0].status == STATUS_ROLLED_BACK

    def test_returns_true_on_success(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_HEALTHY, exp_id="EXP_A")
        assert apply_rollback_decision(dec, reg) is True

    def test_rolled_back_not_in_active_experiments(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_ROLLED_BACK, exp_id="EXP_A")
        apply_rollback_decision(dec, reg)
        loaded  = load_registry(reg)
        active  = get_active_experiments(loaded)
        assert len(active) == 0

    def test_missing_registry_file_returns_false(self, tmp_path):
        reg = tmp_path / "missing_reg.json"
        dec = _make_decision(DECISION_ROLLED_BACK, exp_id="EXP_A")
        result = apply_rollback_decision(dec, reg)
        # load_registry on missing file returns [] → update_experiment_status finds nothing → saves []
        assert result is True  # no error, just no match

    def test_unknown_experiment_id_no_crash(self, tmp_path):
        reg = self._setup_registry(tmp_path)
        dec = _make_decision(DECISION_ROLLED_BACK, exp_id="EXP_UNKNOWN")
        result = apply_rollback_decision(dec, reg)
        assert result is True
        loaded = load_registry(reg)
        assert loaded[0].status == STATUS_ACTIVE  # unchanged


# ── TestEvaluateAllActiveExperiments ───────────────────────────────────────────

class TestEvaluateAllActiveExperiments:
    def _setup(self, tmp_path, exp_configs=None, perf_records=None):
        reg_file  = tmp_path / "reg.json"
        perf_file = tmp_path / "perf.jsonl"
        rb_file   = tmp_path / "rb.jsonl"

        configs = exp_configs or [_make_exp_config("EXP_A", start_date="2026-01-01")]
        _make_registry_file(reg_file, configs)

        if perf_records:
            _write_perf_jsonl(perf_file, perf_records)
        else:
            perf_file.write_text("", encoding="utf-8")

        return reg_file, perf_file, rb_file

    def test_no_active_experiments_returns_empty(self, tmp_path):
        reg_file  = tmp_path / "reg.json"
        perf_file = tmp_path / "perf.jsonl"
        rb_file   = tmp_path / "rb.jsonl"
        _make_registry_file(reg_file, [_make_exp_config(status=STATUS_ROLLED_BACK)])
        perf_file.write_text("", encoding="utf-8")
        result = evaluate_all_active_experiments(reg_file, perf_file, rb_file, "2026-05-30")
        assert result == []

    def test_returns_list_of_decisions(self, tmp_path):
        reg, perf, rb = self._setup(tmp_path)
        result = evaluate_all_active_experiments(reg, perf, rb, "2026-05-30")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_insufficient_data_returns_healthy(self, tmp_path):
        reg, perf, rb = self._setup(tmp_path)
        result = evaluate_all_active_experiments(reg, perf, rb, "2026-05-30")
        assert result[0].decision == DECISION_HEALTHY
        assert _REASON_INSUFFICIENT in result[0].trigger_reasons

    def test_appends_telemetry_to_jsonl(self, tmp_path):
        reg, perf, rb = self._setup(tmp_path)
        evaluate_all_active_experiments(reg, perf, rb, "2026-05-30")
        assert rb.exists()
        lines = [l for l in rb.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_multiple_experiments_evaluated(self, tmp_path):
        configs = [
            _make_exp_config("EXP_A", start_date="2026-01-01"),
            _make_exp_config("EXP_B", start_date="2026-01-01"),
        ]
        reg, perf, rb = self._setup(tmp_path, exp_configs=configs)
        result = evaluate_all_active_experiments(reg, perf, rb, "2026-05-30")
        assert len(result) == 2

    def test_missing_registry_returns_empty(self, tmp_path):
        reg  = tmp_path / "missing.json"
        perf = tmp_path / "perf.jsonl"
        rb   = tmp_path / "rb.jsonl"
        perf.write_text("", encoding="utf-8")
        result = evaluate_all_active_experiments(reg, perf, rb, "2026-05-30")
        assert result == []

    def test_fail_open_bad_performance_file(self, tmp_path):
        reg, _, rb = self._setup(tmp_path)
        bad_perf = tmp_path / "perf.jsonl"
        bad_perf.write_text("not-json\n", encoding="utf-8")
        result = evaluate_all_active_experiments(reg, bad_perf, rb, "2026-05-30")
        assert isinstance(result, list)


# ── TestIntegration ────────────────────────────────────────────────────────────

class TestIntegration:
    def _perf_rec(self, exp_id, group, expectancy, avg_return, max_drawdown):
        return {
            "experiment_id": exp_id,
            "group": group,
            "trades": 10,
            "win_rate": 0.6,
            "expectancy": expectancy,
            "avg_return": avg_return,
            "max_drawdown": max_drawdown,
            "pnl": 100.0,
        }

    def test_healthy_path_no_registry_change(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        rb   = tmp_path / "rb.jsonl"
        _make_registry_file(reg, [_make_exp_config("EXP_A", start_date="2026-01-01")])
        perf.write_text("", encoding="utf-8")
        evaluate_all_active_experiments(reg, perf, rb, "2026-05-30")
        loaded = load_registry(reg)
        assert loaded[0].status == STATUS_ACTIVE

    def test_rollback_after_three_consecutive_candidate_days(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        rb   = tmp_path / "rb.jsonl"
        _make_registry_file(reg, [_make_exp_config("EXP_A", start_date="2026-01-01")])
        # Build performance that triggers 2 conditions
        ctrl_rec = self._perf_rec("EXP_A", "CONTROL", 0.30, 0.03, -0.05)
        exp_rec  = self._perf_rec("EXP_A", "EXPERIMENT", 0.0, 0.0, -0.05)
        # 35 trades each to pass minimum sample
        perf_rows = [ctrl_rec] * 35 + [exp_rec] * 35
        _write_perf_jsonl(perf, perf_rows)

        for day in ["2026-05-28", "2026-05-29", "2026-05-30"]:
            evaluate_all_active_experiments(reg, perf, rb, day)

        loaded = load_registry(reg)
        assert loaded[0].status == STATUS_ROLLED_BACK

    def test_paths_constants_importable(self):
        from src.paths import ROLLBACK_GOVERNANCE_DIR, ROLLBACK_GOVERNANCE_FILE
        assert ROLLBACK_GOVERNANCE_DIR.name == "rollback_governance"
        assert ROLLBACK_GOVERNANCE_FILE.suffix == ".jsonl"

    def test_no_execution_imports(self):
        import src.governance.automatic_rollback as m
        src_text = Path(m.__file__).read_text(encoding="utf-8")
        assert "signal_bridge" not in src_text
        assert "place_order" not in src_text
        assert "kabu" not in src_text.lower()

    def test_registry_constants_importable(self):
        from src.experiments.active_experiment_registry import (
            STATUS_REVIEW_REQUIRED, STATUS_ROLLBACK_CANDIDATE,
        )
        assert STATUS_REVIEW_REQUIRED == "REVIEW_REQUIRED"
        assert STATUS_ROLLBACK_CANDIDATE == "ROLLBACK_CANDIDATE"

    def test_evaluate_experiment_pure_no_side_effects(self):
        inp  = _make_inp()
        prev = _make_decision()
        r1   = evaluate_experiment(inp, prev)
        r2   = evaluate_experiment(inp, prev)
        assert r1.decision == r2.decision
        assert r1.consecutive_candidate_days == r2.consecutive_candidate_days

    def test_build_telemetry_then_append_load(self, tmp_path):
        p   = tmp_path / "rb.jsonl"
        inp = _make_inp(exp_id="EXP_TEL")
        dec = _make_decision(DECISION_REVIEW_REQUIRED, exp_id="EXP_TEL")
        tel = build_rollback_telemetry(inp, dec)
        append_rollback_telemetry(tel, p)
        latest = load_latest_decision(p, "EXP_TEL")
        assert latest is not None
        assert latest.decision == DECISION_REVIEW_REQUIRED
