"""
pilot/phase_gate.py — Study13 phase promotion + stop condition gates

Phase 1 → Phase 2 promotion gate
Phase 2 → Phase 3 promotion gate
Phase 2 safety: kill_switch / freeze / rollback / degrade
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.pilot.shadow_state import (
    PilotState,
    PHASE1_DAYS_MIN, TRACKING_ERR_MAX, REJECT_RATE_MAX, LATENCY_P99_MAX_MS,
    BACKTEST_CALMAR,
    KILL_SWITCH_DD_SLACK, FREEZE_REJECT_CONSEC, ROLLBACK_ALPHA_REAL, DEGRADE_FILL_RATE,
)


@dataclass
class GateCondition:
    name:   str
    passed: bool
    actual: str
    target: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "actual": self.actual, "target": self.target}


@dataclass
class GateResult:
    phase:      str
    promoted:   bool
    stopped:    bool
    stop_reason: Optional[str]
    conditions: list[GateCondition]
    summary:    str

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.conditions)


def _calmar(eq_history: list[float]) -> float:
    if len(eq_history) < 10: return 0.0
    eq = np.array(eq_history, dtype=float)
    if eq[0] <= 0: return 0.0
    n_years = len(eq) / 252
    cagr = (eq[-1] / eq[0]) ** (1 / max(n_years, 0.01)) - 1
    roll = pd.Series(eq).expanding().max()
    max_dd = float(((pd.Series(eq) - roll) / roll).min())
    if max_dd >= -1e-8: return float("inf")
    return cagr / abs(max_dd)


def _alpha_realization(eq_history: list[float], ideal_history: list[float],
                        window: int = 30) -> float:
    if len(eq_history) < 2 or len(ideal_history) < 2: return 1.0
    n = min(len(eq_history), len(ideal_history), window)
    eq_r   = eq_history[-n]
    eq_e   = eq_history[-1]
    id_r   = ideal_history[-n]
    id_e   = ideal_history[-1]
    if id_r <= 0 or id_e <= id_r: return 1.0
    live_ret  = (eq_e / max(eq_r, 1.0)) - 1
    ideal_ret = (id_e / max(id_r, 1.0)) - 1
    if abs(ideal_ret) < 1e-8: return 1.0
    return live_ret / ideal_ret


def check_phase1_gate(state: PilotState) -> GateResult:
    """
    Check all Phase 1 → Phase 2 promotion conditions.
    ALL must pass for promotion.
    """
    shadow_calmar  = _calmar(state.equity_history)
    calmar_target  = BACKTEST_CALMAR * 0.90
    reject_rate    = state.broker_reject_rate()
    tracking_err   = state.tracking_error_max()
    latency_p99    = state.latency_p99()

    conditions = [
        GateCondition(
            "trading_days >= 30",
            state.trading_days >= PHASE1_DAYS_MIN,
            str(state.trading_days),
            f">= {PHASE1_DAYS_MIN}",
        ),
        GateCondition(
            "shadow_calmar >= backtest×0.90",
            not math.isnan(shadow_calmar) and shadow_calmar >= calmar_target,
            f"{shadow_calmar:.3f}",
            f">= {calmar_target:.3f}",
        ),
        GateCondition(
            "position_diff_days == 0",
            state.n_position_diff_days == 0,
            str(state.n_position_diff_days),
            "== 0",
        ),
        GateCondition(
            "cash_drift == 0",
            state.n_cash_drift_events == 0,
            str(state.n_cash_drift_events),
            "== 0",
        ),
        GateCondition(
            "manual_intervention == 0",
            state.n_manual_intervention == 0,
            str(state.n_manual_intervention),
            "== 0",
        ),
        GateCondition(
            "rollback_counter == 0",
            state.n_rollback == 0,
            str(state.n_rollback),
            "== 0",
        ),
        GateCondition(
            "broker_reject < 0.5%",
            reject_rate < REJECT_RATE_MAX,
            f"{reject_rate*100:.3f}%",
            f"< {REJECT_RATE_MAX*100:.1f}%",
        ),
        GateCondition(
            "tracking_error < 1.0%",
            tracking_err < TRACKING_ERR_MAX,
            f"{tracking_err*100:.3f}%",
            f"< {TRACKING_ERR_MAX*100:.1f}%",
        ),
        GateCondition(
            "latency_p99 < 5000ms",
            latency_p99 < LATENCY_P99_MAX_MS,
            f"{latency_p99:.1f}ms",
            f"< {LATENCY_P99_MAX_MS:.0f}ms",
        ),
        GateCondition(
            "state_hash_break == 0",
            state.n_state_hash_break == 0,
            str(state.n_state_hash_break),
            "== 0",
        ),
        GateCondition(
            "duplicate_order == 0",
            state.n_duplicate_guard == 0,
            str(state.n_duplicate_guard),
            "== 0",
        ),
    ]

    all_pass = all(c.passed for c in conditions)
    summary = ("PROMOTE → PHASE2_LIVE" if all_pass
               else f"HOLD — {sum(1 for c in conditions if not c.passed)} 条件未達")
    return GateResult(
        phase="PHASE1", promoted=all_pass, stopped=False,
        stop_reason=None, conditions=conditions, summary=summary,
    )


def check_phase1_stop(state: PilotState, stop_reason: Optional[str]) -> GateResult:
    """
    Check Phase 1 immediate stop conditions.
    Any one triggers STOPPED.
    """
    stop_conditions = {
        "duplicate_order >= 1":    state.n_duplicate_guard >= 1,
        "cash_sync_error >= 1":    state.n_cash_drift_events >= 1,
        "state_hash_break >= 1":   state.n_state_hash_break >= 1,
        "latency_p99 >= 5000ms":   state.latency_p99() >= LATENCY_P99_MAX_MS,
        "explicit_stop":           stop_reason is not None and stop_reason in {
            "duplicate_order", "cash_sync_error", "state_hash_break", "latency_p99_exceeded"
        },
    }
    fired = [k for k, v in stop_conditions.items() if v]
    stopped = bool(fired)
    return GateResult(
        phase="PHASE1_STOP", promoted=False, stopped=stopped,
        stop_reason=fired[0] if fired else None,
        conditions=[GateCondition(k, not v, "FIRED" if v else "OK", "== 0")
                    for k, v in stop_conditions.items()],
        summary=f"STOP FIRED: {fired[0]}" if fired else "OK — no stop condition",
    )


def check_phase2_safety(
    state: PilotState,
    prod_max_dd: float,    # production portfolio max_dd (fraction, negative)
    fill_rate_30d: float,  # last 30d fill rate
) -> GateResult:
    """
    Phase 2 daily safety checks: kill_switch / freeze / rollback / degrade.
    """
    shadow_eq_series = state.equity_history
    shadow_dd = _calmar.__doc__ and 0.0  # placeholder
    # Rolling DD of shadow equity
    if len(shadow_eq_series) >= 5:
        eq  = pd.Series(shadow_eq_series)
        roll = eq.expanding().max()
        shadow_max_dd = float(((eq - roll) / roll).min())
    else:
        shadow_max_dd = 0.0

    kill_switch_thresh = prod_max_dd - (KILL_SWITCH_DD_SLACK / 100)
    alpha_real_30d = _alpha_realization(state.equity_history, state.expected_equity)
    consec_rej     = state.consec_reject

    conditions = [
        GateCondition(
            "kill_switch: shadow_DD < prod_DD - 2pp",
            shadow_max_dd >= kill_switch_thresh,
            f"{shadow_max_dd*100:.2f}%",
            f">= {kill_switch_thresh*100:.2f}%",
        ),
        GateCondition(
            "freeze: consec_reject < 3",
            consec_rej < FREEZE_REJECT_CONSEC,
            str(consec_rej),
            f"< {FREEZE_REJECT_CONSEC}",
        ),
        GateCondition(
            "rollback: alpha_real_30d >= 80%",
            alpha_real_30d >= ROLLBACK_ALPHA_REAL,
            f"{alpha_real_30d*100:.1f}%",
            f">= {ROLLBACK_ALPHA_REAL*100:.0f}%",
        ),
        GateCondition(
            "degrade: fill_rate >= 95%",
            fill_rate_30d >= DEGRADE_FILL_RATE,
            f"{fill_rate_30d*100:.1f}%",
            f">= {DEGRADE_FILL_RATE*100:.0f}%",
        ),
    ]

    kill  = not conditions[0].passed
    freeze = not conditions[1].passed
    rollback = not conditions[2].passed
    degrade  = not conditions[3].passed

    if kill:
        reason = "kill_switch"; action = "STOP: alloc=0%, position close"
    elif rollback:
        reason = "rollback"; action = "PHASE1_SHADOW再開"
    elif freeze:
        reason = "freeze"; action = "新規注文停止"
    elif degrade:
        reason = "degrade"; action = "注意モード: サイズ縮小検討"
    else:
        reason = None; action = "NORMAL"

    return GateResult(
        phase="PHASE2_SAFETY", promoted=False,
        stopped=kill, stop_reason=reason,
        conditions=conditions,
        summary=f"{action}",
    )


def check_phase3_gate(state: PilotState) -> GateResult:
    """
    Phase 2 → Phase 3 Production Promotion gate.
    30 trading days Phase 2, Calmar improvement, zero incidents.
    """
    calmar = _calmar(state.equity_history)
    conditions = [
        GateCondition(
            "phase2_days >= 30",
            state.phase2_days >= 30,
            str(state.phase2_days),
            ">= 30",
        ),
        GateCondition(
            "calmar >= backtest×0.90",
            calmar >= BACKTEST_CALMAR * 0.90,
            f"{calmar:.3f}",
            f">= {BACKTEST_CALMAR * 0.90:.3f}",
        ),
        GateCondition(
            "n_state_hash_break == 0",
            state.n_state_hash_break == 0,
            str(state.n_state_hash_break),
            "== 0",
        ),
        GateCondition(
            "n_rollback == 0",
            state.n_rollback == 0,
            str(state.n_rollback),
            "== 0",
        ),
        GateCondition(
            "n_manual_intervention == 0",
            state.n_manual_intervention == 0,
            str(state.n_manual_intervention),
            "== 0",
        ),
    ]
    all_pass = all(c.passed for c in conditions)
    return GateResult(
        phase="PHASE3",
        promoted=all_pass, stopped=False, stop_reason=None,
        conditions=conditions,
        summary="PROMOTE → PRODUCTION" if all_pass else "HOLD — Phase2継続",
    )
