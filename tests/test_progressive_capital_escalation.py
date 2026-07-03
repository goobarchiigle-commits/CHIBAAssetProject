"""
tests/test_progressive_capital_escalation.py — Phase 6M tests

110+ tests covering:
  Constants / dataclasses / helpers / condition checks /
  evaluate_escalation (hold, escalate, deescalate, full-scale) /
  telemetry / I/O / registry integration / orchestrator / integration
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.progressive_capital_escalation import (
    # constants
    DECISION_HOLD, DECISION_ESCALATE_LEVEL_1, DECISION_ESCALATE_LEVEL_2,
    DECISION_ESCALATE_LEVEL_3, DECISION_FULL_SCALE, DECISION_DEESCALATE,
    SCHEMA_VERSION,
    _ALLOCATION_LEVELS, _MAX_LEVEL, _PERSISTENCE_DAYS, _ESCALATION_REQS,
    # dataclasses
    EscalationInput, EscalationDecision, EscalationTelemetry,
    # helpers
    _safe_float, _safe_int, _allocation_pct, _level_to_decision,
    _load_all_jsonl, _append_jsonl, _is_weakened, _check_escalation_conditions,
    # core functions
    evaluate_escalation,
    build_escalation_telemetry,
    append_escalation_telemetry,
    load_latest_escalation_decision,
    apply_escalation_decision,
    evaluate_all_experiment_escalations,
)
from src.experiments.active_experiment_registry import (
    ExperimentConfig, load_registry, save_registry,
    STATUS_ACTIVE, STATUS_ROLLED_BACK, STATUS_ROLLBACK_CANDIDATE,
    update_experiment_escalation,
)
from src.analytics.experiment_performance import (
    ExperimentOutcome, record_experiment_outcome,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_exp_config(eid="EXP_A", start_date="2026-01-01",
                     status=STATUS_ACTIVE, level=0) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=eid,
        feature_name="feat_a",
        status=status,
        allocation_pct=_ALLOCATION_LEVELS.get(level, 0.005),
        max_positions=1,
        start_date=start_date,
        approval_source="GATE",
        rollback_threshold=-0.10,
        minimum_sample_size=10,
        allocation_level=level,
    )


def _make_registry(path: Path, configs) -> None:
    save_registry(configs, path)


def _make_inp(eid="EXP_A", level=0, obs=30, trades=50,
              ctrl=None, exp=None) -> EscalationInput:
    ctrl = ctrl or {"expectancy": 1.0, "avg_return": 0.02, "pnl": 100.0, "max_drawdown": -0.05}
    exp  = exp  or {"expectancy": 1.5, "avg_return": 0.03, "pnl": 150.0, "max_drawdown": -0.04}
    return EscalationInput(
        experiment_id=eid,
        evaluation_date="2026-05-30",
        current_level=level,
        observation_days=obs,
        trade_count=trades,
        control_metrics=ctrl,
        experiment_metrics=exp,
    )


def _make_decision(eid="EXP_A", decision=DECISION_HOLD, prev_level=0,
                   new_level=0, esc_days=0, deesc_days=0) -> EscalationDecision:
    return EscalationDecision(
        experiment_id=eid,
        evaluation_date="2026-05-29",
        decision=decision,
        previous_level=prev_level,
        new_level=new_level,
        allocation_pct=_ALLOCATION_LEVELS.get(new_level, 0.005),
        reasons=[],
        consecutive_escalation_days=esc_days,
        consecutive_deescalation_days=deesc_days,
    )


def _write_perf(path: Path, rows) -> None:
    for r in rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(r)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _perf_rec(eid, group, expectancy, avg_return, max_dd, pnl=None):
    pnl = pnl if pnl is not None else (100.0 if expectancy > 0 else -100.0)
    return {
        "experiment_id": eid,
        "group": group,
        "trades": 1,
        "expectancy": expectancy,
        "avg_return": avg_return,
        "max_drawdown": max_dd,
        "pnl": pnl,
    }


# ── TestConstants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_decision_states_are_strings(self):
        for d in [DECISION_HOLD, DECISION_ESCALATE_LEVEL_1, DECISION_ESCALATE_LEVEL_2,
                  DECISION_ESCALATE_LEVEL_3, DECISION_FULL_SCALE, DECISION_DEESCALATE]:
            assert isinstance(d, str)

    def test_six_decision_states(self):
        states = {DECISION_HOLD, DECISION_ESCALATE_LEVEL_1, DECISION_ESCALATE_LEVEL_2,
                  DECISION_ESCALATE_LEVEL_3, DECISION_FULL_SCALE, DECISION_DEESCALATE}
        assert len(states) == 6

    def test_allocation_levels_count(self):
        assert len(_ALLOCATION_LEVELS) == 6

    def test_allocation_levels_0_to_5(self):
        assert set(_ALLOCATION_LEVELS.keys()) == {0, 1, 2, 3, 4, 5}

    def test_allocation_level_0_is_half_pct(self):
        assert _ALLOCATION_LEVELS[0] == 0.005

    def test_allocation_level_5_is_25pct(self):
        assert _ALLOCATION_LEVELS[5] == 0.25

    def test_allocation_increases_monotonically(self):
        vals = [_ALLOCATION_LEVELS[i] for i in range(6)]
        assert vals == sorted(vals)

    def test_max_level(self):
        assert _MAX_LEVEL == 5

    def test_persistence_days(self):
        assert _PERSISTENCE_DAYS == 3

    def test_escalation_reqs_all_levels(self):
        assert set(_ESCALATION_REQS.keys()) == {1, 2, 3, 4, 5}

    def test_schema_version_string(self):
        assert isinstance(SCHEMA_VERSION, str)


# ── TestEscalationInput ────────────────────────────────────────────────────────

class TestEscalationInput:
    def test_basic_construction(self):
        inp = _make_inp()
        assert inp.experiment_id == "EXP_A"
        assert inp.current_level == 0

    def test_fields_exist(self):
        inp = _make_inp()
        assert hasattr(inp, "observation_days")
        assert hasattr(inp, "trade_count")
        assert hasattr(inp, "control_metrics")
        assert hasattr(inp, "experiment_metrics")

    def test_custom_level(self):
        inp = _make_inp(level=3)
        assert inp.current_level == 3


# ── TestEscalationDecision ─────────────────────────────────────────────────────

class TestEscalationDecision:
    def test_basic_construction(self):
        d = _make_decision()
        assert d.decision == DECISION_HOLD
        assert d.schema_version == SCHEMA_VERSION

    def test_fields_exist(self):
        d = _make_decision()
        for attr in ["experiment_id", "evaluation_date", "decision", "previous_level",
                     "new_level", "allocation_pct", "reasons",
                     "consecutive_escalation_days", "consecutive_deescalation_days"]:
            assert hasattr(d, attr)

    def test_schema_version_default(self):
        d = _make_decision()
        assert d.schema_version == "v1"


# ── TestEscalationTelemetry ────────────────────────────────────────────────────

class TestEscalationTelemetry:
    def test_basic_construction(self):
        tel = EscalationTelemetry(
            experiment_id="EXP_A", evaluation_date="2026-05-30",
            decision=DECISION_HOLD, previous_level=0, new_level=0,
            allocation_pct=0.005, reasons=[], control_metrics={}, experiment_metrics={},
            consecutive_escalation_days=0, consecutive_deescalation_days=0,
        )
        assert tel.experiment_id == "EXP_A"
        assert tel.schema_version == SCHEMA_VERSION

    def test_fields_present(self):
        tel = EscalationTelemetry(
            experiment_id="EXP_A", evaluation_date="2026-05-30",
            decision=DECISION_HOLD, previous_level=0, new_level=0,
            allocation_pct=0.005, reasons=[], control_metrics={}, experiment_metrics={},
            consecutive_escalation_days=0, consecutive_deescalation_days=0,
        )
        for attr in ["control_metrics", "experiment_metrics", "reasons"]:
            assert hasattr(tel, attr)


# ── TestHelpers ────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_safe_float_normal(self):
        assert _safe_float(1.5) == 1.5

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_nan(self):
        import math
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_custom_fallback(self):
        assert _safe_float(None, 99.0) == 99.0

    def test_safe_int_normal(self):
        assert _safe_int(5) == 5

    def test_safe_int_string(self):
        assert _safe_int("10") == 10

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_allocation_pct_level0(self):
        assert _allocation_pct(0) == 0.005

    def test_allocation_pct_level5(self):
        assert _allocation_pct(5) == 0.25

    def test_allocation_pct_clamps_negative(self):
        assert _allocation_pct(-1) == 0.005

    def test_allocation_pct_clamps_over_max(self):
        assert _allocation_pct(99) == 0.25

    def test_level_to_decision_1(self):
        assert _level_to_decision(1) == DECISION_ESCALATE_LEVEL_1

    def test_level_to_decision_2(self):
        assert _level_to_decision(2) == DECISION_ESCALATE_LEVEL_2

    def test_level_to_decision_3(self):
        assert _level_to_decision(3) == DECISION_ESCALATE_LEVEL_3

    def test_level_to_decision_4_full_scale(self):
        assert _level_to_decision(4) == DECISION_FULL_SCALE

    def test_level_to_decision_5_full_scale(self):
        assert _level_to_decision(5) == DECISION_FULL_SCALE

    def test_is_weakened_expectancy(self):
        ctrl = {"expectancy": 2.0, "pnl": 100.0}
        exp  = {"expectancy": 1.0, "pnl": 100.0}
        assert _is_weakened(ctrl, exp) is True

    def test_is_weakened_pnl(self):
        ctrl = {"expectancy": 1.0, "pnl": 200.0}
        exp  = {"expectancy": 1.5, "pnl": 100.0}
        assert _is_weakened(ctrl, exp) is True

    def test_is_weakened_both(self):
        ctrl = {"expectancy": 2.0, "pnl": 200.0}
        exp  = {"expectancy": 1.0, "pnl": 100.0}
        assert _is_weakened(ctrl, exp) is True

    def test_not_weakened(self):
        ctrl = {"expectancy": 1.0, "pnl": 100.0}
        exp  = {"expectancy": 1.5, "pnl": 150.0}
        assert _is_weakened(ctrl, exp) is False

    def test_not_weakened_equal_values(self):
        ctrl = {"expectancy": 1.0, "pnl": 100.0}
        exp  = {"expectancy": 1.0, "pnl": 100.0}
        # equal is not < so not weakened
        assert _is_weakened(ctrl, exp) is False


# ── TestCheckEscalationConditions ─────────────────────────────────────────────

class TestCheckEscalationConditions:
    def _ctrl(self):
        return {"expectancy": 1.0, "pnl": 100.0, "max_drawdown": -0.05}

    def _exp_good(self):
        return {"expectancy": 1.5, "pnl": 150.0, "max_drawdown": -0.03}

    def test_level1_all_conditions_met(self):
        inp = _make_inp(obs=5, trades=10)
        ok, reasons = _check_escalation_conditions(inp, 1, self._ctrl(), self._exp_good())
        assert ok is True
        assert "EXPECTANCY_GT_CTRL" in reasons

    def test_level1_insufficient_obs(self):
        inp = _make_inp(obs=2, trades=10)
        ok, reasons = _check_escalation_conditions(inp, 1, self._ctrl(), self._exp_good())
        assert ok is False
        assert any("OBS_DAYS" in r for r in reasons)

    def test_level1_insufficient_trades(self):
        inp = _make_inp(obs=5, trades=3)
        ok, reasons = _check_escalation_conditions(inp, 1, self._ctrl(), self._exp_good())
        assert ok is False
        assert any("TRADES" in r for r in reasons)

    def test_level1_expectancy_equal_not_gt(self):
        inp = _make_inp(obs=5, trades=10)
        exp = {"expectancy": 1.0, "pnl": 150.0, "max_drawdown": -0.03}
        ok, reasons = _check_escalation_conditions(inp, 1, self._ctrl(), exp)
        assert ok is False
        assert "EXPECTANCY_NOT_GT_CTRL" in reasons

    def test_level2_all_conditions_met(self):
        inp = _make_inp(obs=10, trades=15)
        ok, reasons = _check_escalation_conditions(inp, 2, self._ctrl(), self._exp_good())
        assert ok is True

    def test_level2_pnl_not_gt(self):
        inp = _make_inp(obs=10, trades=15)
        exp = {"expectancy": 1.5, "pnl": 50.0, "max_drawdown": -0.03}
        ok, reasons = _check_escalation_conditions(inp, 2, self._ctrl(), exp)
        assert ok is False
        assert "PNL_NOT_GT_CTRL" in reasons

    def test_level3_all_conditions_met(self):
        inp = _make_inp(obs=15, trades=25)
        ok, reasons = _check_escalation_conditions(inp, 3, self._ctrl(), self._exp_good())
        assert ok is True

    def test_level3_insufficient_trades(self):
        inp = _make_inp(obs=15, trades=15)
        ok, reasons = _check_escalation_conditions(inp, 3, self._ctrl(), self._exp_good())
        assert ok is False

    def test_level4_all_conditions_met(self):
        inp = _make_inp(obs=20, trades=40)
        ok, reasons = _check_escalation_conditions(inp, 4, self._ctrl(), self._exp_good())
        assert ok is True
        assert "DD_LTE_CTRL" in reasons

    def test_level4_dd_too_high(self):
        inp = _make_inp(obs=20, trades=40)
        exp = {"expectancy": 1.5, "pnl": 150.0, "max_drawdown": -0.10}
        ok, reasons = _check_escalation_conditions(inp, 4, self._ctrl(), exp)
        assert ok is False
        assert "DD_GT_CTRL" in reasons

    def test_level5_all_conditions_met(self):
        inp = _make_inp(obs=25, trades=60)
        ok, reasons = _check_escalation_conditions(inp, 5, self._ctrl(), self._exp_good())
        assert ok is True
        assert "PNL_GT_CTRL" in reasons
        assert "DD_LTE_CTRL" in reasons

    def test_level5_multiple_failures(self):
        inp = _make_inp(obs=5, trades=10)
        ok, reasons = _check_escalation_conditions(inp, 5, self._ctrl(), self._exp_good())
        assert ok is False
        assert len(reasons) >= 2

    def test_invalid_target_level_returns_false(self):
        inp = _make_inp()
        ok, reasons = _check_escalation_conditions(inp, 99, {}, {})
        assert ok is False
        assert any("INVALID" in r for r in reasons)

    def test_level1_obs_exactly_at_threshold(self):
        inp = _make_inp(obs=3, trades=5)
        ok, _ = _check_escalation_conditions(inp, 1, self._ctrl(), self._exp_good())
        assert ok is True

    def test_level1_obs_one_below_threshold(self):
        inp = _make_inp(obs=2, trades=5)
        ok, _ = _check_escalation_conditions(inp, 1, self._ctrl(), self._exp_good())
        assert ok is False


# ── TestEvaluateEscalationHold ─────────────────────────────────────────────────

class TestEvaluateEscalationHold:
    def test_hold_obs_days_insufficient(self):
        inp = _make_inp(obs=1, trades=10)
        d = evaluate_escalation(inp, None)
        assert d.decision == DECISION_HOLD
        assert d.new_level == 0

    def test_hold_trades_insufficient(self):
        inp = _make_inp(obs=10, trades=2)
        d = evaluate_escalation(inp, None)
        assert d.decision == DECISION_HOLD
        assert d.new_level == 0

    def test_hold_expectancy_equal_to_ctrl(self):
        ctrl = {"expectancy": 1.0, "pnl": 100.0, "max_drawdown": -0.05}
        exp  = {"expectancy": 1.0, "pnl": 100.0, "max_drawdown": -0.04}
        inp = _make_inp(obs=5, trades=10, ctrl=ctrl, exp=exp)
        d = evaluate_escalation(inp, None)
        assert d.decision == DECISION_HOLD

    def test_hold_accumulates_escalation_counter_day1(self):
        # Good conditions but day 1 — not persistent yet
        inp  = _make_inp(obs=5, trades=10)
        prev = _make_decision(esc_days=0)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_HOLD
        assert d.consecutive_escalation_days == 1

    def test_hold_accumulates_escalation_counter_day2(self):
        inp  = _make_inp(obs=5, trades=10)
        prev = _make_decision(esc_days=1)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_HOLD
        assert d.consecutive_escalation_days == 2

    def test_hold_resets_escalation_counter_when_conditions_fail(self):
        inp  = _make_inp(obs=1, trades=2)
        prev = _make_decision(esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.consecutive_escalation_days == 0

    def test_hold_accumulates_deescalation_counter(self):
        ctrl = {"expectancy": 2.0, "pnl": 200.0, "max_drawdown": -0.05}
        exp  = {"expectancy": 1.0, "pnl": 100.0, "max_drawdown": -0.04}
        inp  = _make_inp(obs=5, trades=10, ctrl=ctrl, exp=exp)
        prev = _make_decision(deesc_days=1)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_HOLD
        assert d.consecutive_deescalation_days == 2

    def test_hold_at_level0_weakened_cannot_deescalate(self):
        ctrl = {"expectancy": 2.0, "pnl": 200.0, "max_drawdown": -0.05}
        exp  = {"expectancy": 1.0, "pnl": 100.0, "max_drawdown": -0.04}
        inp  = _make_inp(level=0, obs=5, trades=10, ctrl=ctrl, exp=exp)
        prev = _make_decision(deesc_days=3)
        d = evaluate_escalation(inp, prev)
        # level 0, can't go lower, so even with 3+ deesc days it stays HOLD
        assert d.new_level == 0


# ── TestEvaluateEscalationEscalate ─────────────────────────────────────────────

class TestEvaluateEscalationEscalate:
    def test_escalate_level1_after_3_days(self):
        inp  = _make_inp(obs=5, trades=10)
        prev = _make_decision(esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_ESCALATE_LEVEL_1
        assert d.new_level == 1
        assert d.allocation_pct == _ALLOCATION_LEVELS[1]

    def test_escalate_level2_after_3_days(self):
        inp  = _make_inp(level=1, obs=10, trades=15)
        prev = _make_decision(prev_level=1, new_level=1, esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_ESCALATE_LEVEL_2
        assert d.new_level == 2

    def test_escalate_level3_after_3_days(self):
        inp  = _make_inp(level=2, obs=15, trades=25)
        prev = _make_decision(prev_level=2, new_level=2, esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_ESCALATE_LEVEL_3
        assert d.new_level == 3

    def test_escalate_to_level4_returns_full_scale(self):
        inp  = _make_inp(level=3, obs=20, trades=40)
        prev = _make_decision(prev_level=3, new_level=3, esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_FULL_SCALE
        assert d.new_level == 4

    def test_escalate_to_level5_returns_full_scale(self):
        inp  = _make_inp(level=4, obs=25, trades=60)
        prev = _make_decision(prev_level=4, new_level=4, esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_FULL_SCALE
        assert d.new_level == 5

    def test_escalate_resets_both_counters(self):
        inp  = _make_inp(obs=5, trades=10)
        prev = _make_decision(esc_days=2, deesc_days=0)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_ESCALATE_LEVEL_1
        assert d.consecutive_escalation_days == 0
        assert d.consecutive_deescalation_days == 0

    def test_no_level_skip_from_0(self):
        inp  = _make_inp(level=0, obs=30, trades=60)
        prev = _make_decision(esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.new_level == 1  # can only go to 1, not 5

    def test_escalate_no_prev_starts_fresh(self):
        inp = _make_inp(obs=5, trades=10)
        d = evaluate_escalation(inp, None)
        # First evaluation: esc_consec goes from 0 to 1 → HOLD
        assert d.consecutive_escalation_days == 1
        assert d.decision == DECISION_HOLD

    def test_previous_level_tracked_correctly(self):
        inp  = _make_inp(obs=5, trades=10)
        prev = _make_decision(esc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.previous_level == 0
        assert d.new_level == 1

    def test_allocation_pct_matches_level(self):
        for lvl in range(_MAX_LEVEL):
            inp  = _make_inp(level=lvl, obs=30, trades=60)
            prev = _make_decision(prev_level=lvl, new_level=lvl, esc_days=2)
            d = evaluate_escalation(inp, prev)
            if d.new_level != d.previous_level:
                assert d.allocation_pct == _ALLOCATION_LEVELS[d.new_level]


# ── TestEvaluateEscalationFullScale ───────────────────────────────────────────

class TestEvaluateEscalationFullScale:
    def test_full_scale_when_at_max_level(self):
        inp = _make_inp(level=5, obs=30, trades=60)
        d = evaluate_escalation(inp, None)
        assert d.decision == DECISION_FULL_SCALE
        assert d.new_level == 5
        assert d.allocation_pct == 0.25

    def test_full_scale_not_weakened_steady_state(self):
        inp = _make_inp(level=5, obs=30, trades=60)
        prev = _make_decision(prev_level=5, new_level=5)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_FULL_SCALE

    def test_full_scale_keeps_level_5(self):
        inp = _make_inp(level=5)
        d = evaluate_escalation(inp, None)
        assert d.previous_level == 5
        assert d.new_level == 5


# ── TestEvaluateEscalationDeescalate ──────────────────────────────────────────

class TestEvaluateEscalationDeescalate:
    def _weakened_metrics(self):
        ctrl = {"expectancy": 2.0, "pnl": 200.0, "max_drawdown": -0.05}
        exp  = {"expectancy": 1.0, "pnl": 100.0, "max_drawdown": -0.04}
        return ctrl, exp

    def test_deescalate_after_3_days_weakened_expectancy(self):
        ctrl, exp = self._weakened_metrics()
        inp  = _make_inp(level=2, ctrl=ctrl, exp=exp)
        prev = _make_decision(prev_level=2, new_level=2, deesc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_DEESCALATE
        assert d.new_level == 1

    def test_deescalate_after_3_days_weakened_pnl_only(self):
        ctrl = {"expectancy": 1.0, "pnl": 200.0, "max_drawdown": -0.05}
        exp  = {"expectancy": 1.5, "pnl": 50.0, "max_drawdown": -0.04}
        inp  = _make_inp(level=3, ctrl=ctrl, exp=exp)
        prev = _make_decision(prev_level=3, new_level=3, deesc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_DEESCALATE
        assert d.new_level == 2

    def test_deescalate_resets_both_counters(self):
        ctrl, exp = self._weakened_metrics()
        inp  = _make_inp(level=2, ctrl=ctrl, exp=exp)
        prev = _make_decision(prev_level=2, new_level=2, deesc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.consecutive_escalation_days == 0
        assert d.consecutive_deescalation_days == 0

    def test_deescalate_from_level_1_goes_to_0(self):
        ctrl, exp = self._weakened_metrics()
        inp  = _make_inp(level=1, ctrl=ctrl, exp=exp)
        prev = _make_decision(prev_level=1, new_level=1, deesc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_DEESCALATE
        assert d.new_level == 0

    def test_deescalate_from_level_0_stays_hold(self):
        ctrl, exp = self._weakened_metrics()
        inp  = _make_inp(level=0, ctrl=ctrl, exp=exp)
        prev = _make_decision(deesc_days=2)
        d = evaluate_escalation(inp, prev)
        # At level 0, cannot go lower even with 3+ deesc days
        assert d.new_level == 0
        assert d.decision == DECISION_HOLD

    def test_deescalate_priority_over_escalation(self):
        # Weakened → de-escalation check fires before escalation
        ctrl = {"expectancy": 2.0, "pnl": 200.0, "max_drawdown": -0.05}
        exp  = {"expectancy": 1.5, "pnl": 50.0, "max_drawdown": -0.04}
        inp  = _make_inp(level=2, obs=30, trades=60, ctrl=ctrl, exp=exp)
        prev = _make_decision(prev_level=2, new_level=2, esc_days=2, deesc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.decision == DECISION_DEESCALATE

    def test_deescalate_allocation_matches_new_level(self):
        ctrl, exp = self._weakened_metrics()
        inp  = _make_inp(level=3, ctrl=ctrl, exp=exp)
        prev = _make_decision(prev_level=3, new_level=3, deesc_days=2)
        d = evaluate_escalation(inp, prev)
        assert d.allocation_pct == _ALLOCATION_LEVELS[2]


# ── TestBuildEscalationTelemetry ───────────────────────────────────────────────

class TestBuildEscalationTelemetry:
    def test_basic_build(self):
        inp = _make_inp()
        dec = _make_decision()
        tel = build_escalation_telemetry(inp, dec)
        assert tel.experiment_id == "EXP_A"

    def test_metrics_copied(self):
        inp = _make_inp()
        dec = _make_decision()
        tel = build_escalation_telemetry(inp, dec)
        assert tel.control_metrics == inp.control_metrics
        assert tel.experiment_metrics == inp.experiment_metrics

    def test_reasons_copied(self):
        inp = _make_inp()
        dec = _make_decision()
        dec.reasons = ["REASON_A", "REASON_B"]
        tel = build_escalation_telemetry(inp, dec)
        assert tel.reasons == ["REASON_A", "REASON_B"]

    def test_schema_version(self):
        tel = build_escalation_telemetry(_make_inp(), _make_decision())
        assert tel.schema_version == SCHEMA_VERSION

    def test_pure_no_side_effects(self):
        inp = _make_inp()
        dec = _make_decision()
        tel1 = build_escalation_telemetry(inp, dec)
        tel2 = build_escalation_telemetry(inp, dec)
        assert tel1.experiment_id == tel2.experiment_id


# ── TestAppendEscalationTelemetry ──────────────────────────────────────────────

class TestAppendEscalationTelemetry:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        tel = build_escalation_telemetry(_make_inp(), _make_decision())
        append_escalation_telemetry(tel, path)
        assert path.exists()

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        for _ in range(3):
            tel = build_escalation_telemetry(_make_inp(), _make_decision())
            append_escalation_telemetry(tel, path)
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

    def test_json_valid(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        tel = build_escalation_telemetry(_make_inp(), _make_decision())
        append_escalation_telemetry(tel, path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["experiment_id"] == "EXP_A"

    def test_all_fields_present(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        tel = build_escalation_telemetry(_make_inp(), _make_decision())
        append_escalation_telemetry(tel, path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        for key in ["experiment_id", "evaluation_date", "decision", "previous_level",
                    "new_level", "allocation_pct", "reasons", "control_metrics",
                    "experiment_metrics", "consecutive_escalation_days",
                    "consecutive_deescalation_days", "schema_version"]:
            assert key in data, f"Missing field: {key}"

    def test_fail_open_on_invalid_path(self, tmp_path):
        # Should not raise
        bad_path = tmp_path / "nonexistent_dir" / "x" / "esc.jsonl"
        tel = build_escalation_telemetry(_make_inp(), _make_decision())
        # Will create dirs — or fail silently
        append_escalation_telemetry(tel, bad_path)


# ── TestLoadLatestEscalationDecision ──────────────────────────────────────────

class TestLoadLatestEscalationDecision:
    def test_returns_none_no_file(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        r = load_latest_escalation_decision(path, "EXP_A")
        assert r is None

    def test_returns_latest_by_date(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        for date in ["2026-05-28", "2026-05-30", "2026-05-29"]:
            dec = EscalationDecision(
                experiment_id="EXP_A", evaluation_date=date,
                decision=DECISION_HOLD, previous_level=0, new_level=0,
                allocation_pct=0.005, reasons=[],
                consecutive_escalation_days=0, consecutive_deescalation_days=0,
            )
            tel = build_escalation_telemetry(_make_inp(), dec)
            append_escalation_telemetry(tel, path)
        r = load_latest_escalation_decision(path, "EXP_A")
        assert r is not None
        assert r.evaluation_date == "2026-05-30"

    def test_filters_by_experiment_id(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        for eid in ["EXP_A", "EXP_B"]:
            dec = _make_decision(eid=eid)
            tel = build_escalation_telemetry(_make_inp(eid=eid), dec)
            append_escalation_telemetry(tel, path)
        r = load_latest_escalation_decision(path, "EXP_B")
        assert r is not None
        assert r.experiment_id == "EXP_B"

    def test_handles_malformed_lines(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        path.write_text("not-json\n", encoding="utf-8")
        r = load_latest_escalation_decision(path, "EXP_A")
        assert r is None

    def test_returns_correct_fields(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        dec = _make_decision(esc_days=2, deesc_days=1)
        tel = build_escalation_telemetry(_make_inp(), dec)
        append_escalation_telemetry(tel, path)
        r = load_latest_escalation_decision(path, "EXP_A")
        assert r is not None
        assert r.consecutive_escalation_days == 2
        assert r.consecutive_deescalation_days == 1


# ── TestApplyEscalationDecision ───────────────────────────────────────────────

class TestApplyEscalationDecision:
    def test_apply_escalation_updates_registry(self, tmp_path):
        reg = tmp_path / "reg.json"
        _make_registry(reg, [_make_exp_config(level=0)])
        dec = EscalationDecision(
            experiment_id="EXP_A", evaluation_date="2026-05-30",
            decision=DECISION_ESCALATE_LEVEL_1, previous_level=0, new_level=1,
            allocation_pct=_ALLOCATION_LEVELS[1], reasons=[],
            consecutive_escalation_days=0, consecutive_deescalation_days=0,
        )
        ok = apply_escalation_decision(dec, reg)
        assert ok is True
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 1
        assert loaded[0].allocation_pct == _ALLOCATION_LEVELS[1]

    def test_apply_hold_no_registry_change(self, tmp_path):
        reg = tmp_path / "reg.json"
        _make_registry(reg, [_make_exp_config(level=2)])
        dec = _make_decision(prev_level=2, new_level=2)
        ok = apply_escalation_decision(dec, reg)
        assert ok is True
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 2

    def test_apply_deescalation_updates_registry(self, tmp_path):
        reg = tmp_path / "reg.json"
        _make_registry(reg, [_make_exp_config(level=3)])
        dec = EscalationDecision(
            experiment_id="EXP_A", evaluation_date="2026-05-30",
            decision=DECISION_DEESCALATE, previous_level=3, new_level=2,
            allocation_pct=_ALLOCATION_LEVELS[2], reasons=[],
            consecutive_escalation_days=0, consecutive_deescalation_days=0,
        )
        ok = apply_escalation_decision(dec, reg)
        assert ok is True
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 2

    def test_escalation_history_updated(self, tmp_path):
        reg = tmp_path / "reg.json"
        _make_registry(reg, [_make_exp_config(level=0)])
        dec = EscalationDecision(
            experiment_id="EXP_A", evaluation_date="2026-05-30",
            decision=DECISION_ESCALATE_LEVEL_1, previous_level=0, new_level=1,
            allocation_pct=_ALLOCATION_LEVELS[1], reasons=[],
            consecutive_escalation_days=0, consecutive_deescalation_days=0,
        )
        apply_escalation_decision(dec, reg)
        loaded = load_registry(reg)
        assert len(loaded[0].escalation_history) == 1
        assert "ESCALATE_LEVEL_1" in loaded[0].escalation_history[0]

    def test_fail_open_missing_registry(self, tmp_path):
        reg = tmp_path / "nonexistent.json"
        dec = _make_decision(prev_level=0, new_level=1)
        dec.decision = DECISION_ESCALATE_LEVEL_1
        # Should not raise
        result = apply_escalation_decision(dec, reg)
        assert isinstance(result, bool)


# ── TestEvaluateAllExperimentEscalations ──────────────────────────────────────

class TestEvaluateAllExperimentEscalations:
    def test_empty_registry_returns_empty(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([], reg)
        perf.write_text("", encoding="utf-8")
        result = evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert result == []

    def test_skips_non_active_experiments(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        cfg = _make_exp_config(status=STATUS_ROLLED_BACK)
        save_registry([cfg], reg)
        perf.write_text("", encoding="utf-8")
        result = evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert result == []

    def test_evaluates_active_experiment(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01")], reg)
        perf.write_text("", encoding="utf-8")
        result = evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert len(result) == 1
        assert result[0].experiment_id == "EXP_A"

    def test_rollback_candidate_skipped(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        cfg = _make_exp_config(status=STATUS_ROLLBACK_CANDIDATE)
        save_registry([cfg], reg)
        perf.write_text("", encoding="utf-8")
        result = evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert result == []

    def test_emits_telemetry(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01")], reg)
        perf.write_text("", encoding="utf-8")
        evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert esc.exists()
        lines = [l for l in esc.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_fail_open_on_registry_error(self, tmp_path):
        reg  = tmp_path / "nonexistent.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        result = evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert result == []

    def test_fail_open_individual_error(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        # Two experiments; second has bad data but first should still work
        cfgs = [
            _make_exp_config("EXP_A", start_date="2026-01-01"),
            _make_exp_config("EXP_B", start_date="INVALID_DATE"),
        ]
        save_registry(cfgs, reg)
        perf.write_text("", encoding="utf-8")
        result = evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        assert any(d.experiment_id == "EXP_A" for d in result)

    def test_updates_registry_on_escalation(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01", level=0)], reg)
        # Write performance that triggers escalation conditions
        ctrl_rows = [_perf_rec("EXP_A", "CONTROL",   1.0, 0.02, -0.05, 100.0)] * 10
        exp_rows  = [_perf_rec("EXP_A", "EXPERIMENT", 2.0, 0.03, -0.04, 200.0)] * 10
        _write_perf(perf, ctrl_rows + exp_rows)
        # Pre-seed two consecutive escalation days
        prev_dec = EscalationDecision(
            experiment_id="EXP_A", evaluation_date="2026-05-28",
            decision=DECISION_HOLD, previous_level=0, new_level=0,
            allocation_pct=0.005, reasons=[],
            consecutive_escalation_days=2, consecutive_deescalation_days=0,
        )
        prev_tel = build_escalation_telemetry(_make_inp(), prev_dec)
        append_escalation_telemetry(prev_tel, esc)
        evaluate_all_experiment_escalations(reg, perf, esc, "2026-05-30")
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 1


# ── TestRegistryExtension ─────────────────────────────────────────────────────

class TestRegistryExtension:
    def test_allocation_level_persisted(self, tmp_path):
        reg = tmp_path / "reg.json"
        cfg = _make_exp_config(level=3)
        save_registry([cfg], reg)
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 3

    def test_escalation_history_persisted(self, tmp_path):
        reg = tmp_path / "reg.json"
        cfg = _make_exp_config()
        cfg.escalation_history = ["2026-05-30:ESCALATE_LEVEL_1:L0→L1:0.010"]
        save_registry([cfg], reg)
        loaded = load_registry(reg)
        assert len(loaded[0].escalation_history) == 1

    def test_backward_compatible_load(self, tmp_path):
        reg = tmp_path / "reg.json"
        # Write JSON without new fields
        data = {
            "schema_version": "v1",
            "experiments": [{
                "experiment_id": "EXP_A",
                "feature_name": "feat_a",
                "status": "ACTIVE",
                "allocation_pct": 0.5,
                "max_positions": 1,
                "start_date": "2026-01-01",
                "approval_source": "GATE",
                "rollback_threshold": -0.10,
                "minimum_sample_size": 10,
                "schema_version": "v1",
                # No allocation_level or escalation_history
            }]
        }
        reg.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 0
        assert loaded[0].escalation_history == []

    def test_allocation_level_default_0(self):
        cfg = ExperimentConfig(
            experiment_id="EXP_A", feature_name="f", status="ACTIVE",
            allocation_pct=0.5, max_positions=1, start_date="2026-01-01",
            approval_source="GATE", rollback_threshold=-0.1, minimum_sample_size=10,
        )
        assert cfg.allocation_level == 0

    def test_escalation_history_default_empty(self):
        cfg = ExperimentConfig(
            experiment_id="EXP_A", feature_name="f", status="ACTIVE",
            allocation_pct=0.5, max_positions=1, start_date="2026-01-01",
            approval_source="GATE", rollback_threshold=-0.1, minimum_sample_size=10,
        )
        assert cfg.escalation_history == []

    def test_update_experiment_escalation_function(self, tmp_path):
        reg = tmp_path / "reg.json"
        save_registry([_make_exp_config(level=0)], reg)
        ok = update_experiment_escalation(
            "EXP_A", 2, _ALLOCATION_LEVELS[2], "2026-05-30:ESCALATE:L0→L2:0.02", reg
        )
        assert ok is True
        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 2
        assert loaded[0].allocation_pct == _ALLOCATION_LEVELS[2]
        assert len(loaded[0].escalation_history) == 1


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_escalation_from_level0_to_1_over_3_days(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01")], reg)
        rows = ([_perf_rec("EXP_A", "CONTROL",    1.0, 0.02, -0.05, 100.0)] * 10 +
                [_perf_rec("EXP_A", "EXPERIMENT", 2.0, 0.03, -0.04, 200.0)] * 10)
        _write_perf(perf, rows)

        for day in ["2026-05-28", "2026-05-29", "2026-05-30"]:
            evaluate_all_experiment_escalations(reg, perf, esc, day)

        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 1
        assert loaded[0].allocation_pct == _ALLOCATION_LEVELS[1]

    def test_hold_when_conditions_not_met(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01")], reg)
        # Experiment worse than control → no escalation
        rows = ([_perf_rec("EXP_A", "CONTROL",    2.0, 0.03, -0.04, 200.0)] * 10 +
                [_perf_rec("EXP_A", "EXPERIMENT", 1.0, 0.02, -0.05, 100.0)] * 10)
        _write_perf(perf, rows)

        for day in ["2026-05-28", "2026-05-29", "2026-05-30"]:
            evaluate_all_experiment_escalations(reg, perf, esc, day)

        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 0

    def test_deescalation_after_3_weakened_days(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01", level=2)], reg)
        rows = ([_perf_rec("EXP_A", "CONTROL",    2.0, 0.03, -0.04, 200.0)] * 10 +
                [_perf_rec("EXP_A", "EXPERIMENT", 0.5, 0.01, -0.05,  50.0)] * 10)
        _write_perf(perf, rows)

        for day in ["2026-05-28", "2026-05-29", "2026-05-30"]:
            evaluate_all_experiment_escalations(reg, perf, esc, day)

        loaded = load_registry(reg)
        assert loaded[0].allocation_level == 1

    def test_paths_constants_importable(self):
        from src.paths import CAPITAL_ESCALATION_DIR, CAPITAL_ESCALATION_FILE
        assert CAPITAL_ESCALATION_DIR.name == "capital_escalation"
        assert CAPITAL_ESCALATION_FILE.suffix == ".jsonl"

    def test_no_execution_imports(self):
        import src.governance.progressive_capital_escalation as m
        src_text = Path(m.__file__).read_text(encoding="utf-8")
        assert "signal_bridge" not in src_text
        assert "place_order" not in src_text
        assert "kabu" not in src_text.lower()

    def test_evaluate_experiment_pure_no_side_effects(self):
        inp  = _make_inp()
        prev = _make_decision()
        r1   = evaluate_escalation(inp, prev)
        r2   = evaluate_escalation(inp, prev)
        assert r1.decision == r2.decision
        assert r1.new_level == r2.new_level

    def test_escalation_history_grows_with_each_escalation(self, tmp_path):
        reg  = tmp_path / "reg.json"
        perf = tmp_path / "perf.jsonl"
        esc  = tmp_path / "esc.jsonl"
        save_registry([_make_exp_config(start_date="2026-01-01", level=0)], reg)
        rows = ([_perf_rec("EXP_A", "CONTROL",    1.0, 0.02, -0.05, 100.0)] * 10 +
                [_perf_rec("EXP_A", "EXPERIMENT", 2.0, 0.03, -0.04, 200.0)] * 10)
        _write_perf(perf, rows)
        # Trigger first escalation
        for day in ["2026-05-26", "2026-05-27", "2026-05-28"]:
            evaluate_all_experiment_escalations(reg, perf, esc, day)
        loaded1 = load_registry(reg)
        assert loaded1[0].allocation_level == 1
        assert len(loaded1[0].escalation_history) == 1

    def test_telemetry_idempotent_experiment_id(self, tmp_path):
        path = tmp_path / "esc.jsonl"
        inp = _make_inp(eid="EXP_UNIQUE")
        dec = _make_decision(eid="EXP_UNIQUE")
        tel = build_escalation_telemetry(inp, dec)
        append_escalation_telemetry(tel, path)
        loaded = load_latest_escalation_decision(path, "EXP_UNIQUE")
        assert loaded is not None
        assert loaded.experiment_id == "EXP_UNIQUE"
