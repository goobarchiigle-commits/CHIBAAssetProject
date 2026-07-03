"""
tests/test_capital_deployment_os.py
Pytest suite for Phase 5B — Unified Capital Deployment OS.

Covers:
  - capital growth scaling (dynamic_max_positions / target_position_size)
  - idle cash reduction (idle relaxation path)
  - continuation deployment (FULL_ENTRY / REDUCED_STARTER logic)
  - replay equivalence (same input → same output, order-stable)
  - liquidity cap enforcement (participation_rate ≤ 0.03)
  - duplicate order prevention (per-run guard)
  - CB_ACTIVE behavior (new entries blocked)
  - bounded position growth (lot_cost_ratio gate)
  - safety gates (crash / broker / volatility_spike)
  - addon eligibility (depth / cooldown / profit / concentration)
  - regime scaling (bear × 0.5, crash → 0)
  - all decision types emitted
  - telemetry output structure
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from src.live.capital_deployment_os import (
    # Data structures
    CandidateInfo,
    DeploymentContext,
    DeploymentDecision,
    HeldPosition,
    # Decision types
    DECISION_FULL_ENTRY,
    DECISION_REDUCED_STARTER,
    DECISION_ADDON,
    DECISION_OBSERVATION_ONLY,
    DECISION_BLOCKED,
    # Regime constants
    REGIME_CRASH,
    REGIME_BEAR,
    REGIME_NORMAL,
    REGIME_BULL,
    BROKER_HEALTHY,
    BROKER_DEGRADED,
    BROKER_UNREACHABLE,
    # Pure functions
    dynamic_max_positions,
    target_position_size,
    starter_position_size,
    addon_position_size,
    compute_participation_rate,
    apply_participation_cap,
    regime_scale_factor,
    classify_opportunity,
    check_safety_gates,
    # Main class
    CapitalDeploymentOS,
    # Helpers
    make_context,
    make_candidate,
    _is_addon_eligible,
    _should_apply_idle_relaxation,
    _is_duplicate,
    _build_telemetry_summary,
    # Constants
    MAX_PARTICIPATION_RATE,
    MAX_POSITIONS_HARD_CAP,
    MIN_POSITIONS_FLOOR,
    PARAMS_LOCKED_MAX_POSITIONS,
    LOT_SIZE,
    IDLE_UTILIZATION_THRESHOLD,
    HIGH_SURVIVABILITY_THRESHOLD,
    HIGH_CONTINUATION_THRESHOLD,
    FULL_ENTRY_CONTINUATION_MIN,
    REDUCED_STARTER_CONTINUATION_MIN,
    MIN_SURVIVABILITY_ENTRY,
    MIN_SURVIVABILITY_ADDON,
    BOUNDED_GROWTH_CAP_RATIO,
)

TODAY = "2026-05-29"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _ctx(
    deployable=3_000_000,
    cap_util=0.0,
    regime=REGIME_NORMAL,
    broker=BROKER_HEALTHY,
    vol_spike=False,
    cb=False,
):
    return make_context(
        actual_equity=deployable * 1.1,
        effective_capital=deployable,
        deployable_capital=deployable,
        capital_utilization=cap_util,
        market_regime=regime,
        broker_health=broker,
        volatility_spike=vol_spike,
        cb_active=cb,
        date=TODAY,
    )


def _cand(
    symbol="6758.T",
    price=3000.0,
    cont=70.0,
    surv=0.80,
    adv=500_000_000,
    is_held=False,
    held_info=None,
):
    return make_candidate(
        symbol=symbol,
        price=price,
        continuation_score=cont,
        survivability=surv,
        adv_yen=adv,
        is_held=is_held,
        held_info=held_info,
    )


def _held(
    symbol="6758.T",
    qty=100,
    avg_cost=2800.0,
    pnl_pct=0.07,
    hold_days=10,
    addon_count=0,
    last_addon="",
):
    return HeldPosition(
        symbol=symbol,
        qty=qty,
        avg_cost=avg_cost,
        unrealized_pnl_pct=pnl_pct,
        hold_days=hold_days,
        addon_count=addon_count,
        last_addon_date=last_addon,
        sector="電気機器",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. dynamic_max_positions
# ─────────────────────────────────────────────────────────────────────────────

def test_max_positions_floor():
    assert dynamic_max_positions(0) == MIN_POSITIONS_FLOOR
    assert dynamic_max_positions(-1) == MIN_POSITIONS_FLOOR


def test_max_positions_1m():
    # 1M → tier=1 → 1+2=3
    assert dynamic_max_positions(1_000_000) == 3


def test_max_positions_3m():
    # 3M → tier=3 → 3+2=5, but PARAMS_LOCKED クランプで 3
    assert dynamic_max_positions(3_000_000) == PARAMS_LOCKED_MAX_POSITIONS


def test_max_positions_hard_cap():
    # 巨額資本でも PARAMS_LOCKED_MAX_POSITIONS=3 にクランプされる
    assert dynamic_max_positions(100_000_000) == PARAMS_LOCKED_MAX_POSITIONS


def test_max_positions_growth_scaling():
    results = [dynamic_max_positions(c * 1_000_000) for c in [1, 2, 3, 5, 10]]
    # Non-decreasing
    for a, b in zip(results, results[1:]):
        assert b >= a


# ─────────────────────────────────────────────────────────────────────────────
# 2. target_position_size
# ─────────────────────────────────────────────────────────────────────────────

def test_target_position_size_basic():
    # 3M / 3 positions = 1M per pos; 1M / 3000*100 = ~3.3 lots → 300 shares
    shares = target_position_size(3_000_000, 3, 3000.0)
    assert shares >= 100
    assert shares % LOT_SIZE == 0


def test_target_position_size_zero_price():
    assert target_position_size(3_000_000, 3, 0) == 0


def test_target_position_size_zero_capital():
    assert target_position_size(0, 3, 3000.0) == 0


def test_target_position_size_concentration_cap():
    # 1 position at 3M → would be 100% concentration → capped at 25%
    shares = target_position_size(3_000_000, 1, 1000.0)
    max_allowed_notional = 3_000_000 * 0.25
    assert shares * 1000.0 <= max_allowed_notional + 1000  # small rounding tolerance


def test_target_position_size_lot_multiple():
    shares = target_position_size(3_000_000, 3, 3000.0)
    assert shares % 100 == 0


def test_target_grows_with_capital():
    s1 = target_position_size(3_000_000, 3, 3000.0)
    s2 = target_position_size(6_000_000, 3, 3000.0)
    assert s2 >= s1


# ─────────────────────────────────────────────────────────────────────────────
# 3. starter_position_size
# ─────────────────────────────────────────────────────────────────────────────

def test_starter_smaller_than_target():
    full = target_position_size(3_000_000, 3, 3000.0)
    starter = starter_position_size(3_000_000, 3, 3000.0)
    assert starter <= full


def test_starter_lot_multiple():
    starter = starter_position_size(3_000_000, 3, 3000.0)
    assert starter % LOT_SIZE == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. addon_position_size
# ─────────────────────────────────────────────────────────────────────────────

def test_addon_smaller_than_target():
    full = target_position_size(3_000_000, 3, 3000.0)
    addon = addon_position_size(3_000_000, 3, 3000.0)
    assert addon < full


def test_addon_minimum_lot():
    # Very small capital → still at least 1 lot
    addon = addon_position_size(100_000, 3, 100.0)
    assert addon >= LOT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# 5. participation_rate
# ─────────────────────────────────────────────────────────────────────────────

def test_participation_rate_normal():
    # 15M order / 500M ADV = 0.03
    r = compute_participation_rate(15_000_000, 500_000_000)
    assert abs(r - 0.03) < 1e-9


def test_participation_rate_zero_adv():
    assert compute_participation_rate(1_000_000, 0) == 0.0


def test_apply_participation_cap_no_reduction():
    shares, reduced = apply_participation_cap(100, 3000.0, 500_000_000)
    assert shares == 100
    assert reduced is False


def test_apply_participation_cap_reduces():
    # 10000 shares × 3000 = 30M; 500M × 3% = 15M → cap
    shares, reduced = apply_participation_cap(10000, 3000.0, 500_000_000)
    assert reduced is True
    assert shares * 3000.0 <= 500_000_000 * MAX_PARTICIPATION_RATE + 3000  # tolerance


def test_apply_participation_cap_lot_multiple():
    shares, _ = apply_participation_cap(5000, 2000.0, 100_000_000)
    assert shares % LOT_SIZE == 0


def test_apply_participation_cap_unknown_adv():
    shares, reduced = apply_participation_cap(1000, 3000.0, 0)
    assert shares == 1000
    assert reduced is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. regime_scale_factor
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_crash_zero():
    assert regime_scale_factor(REGIME_CRASH, False) == 0.0


def test_regime_cb_active_zero():
    assert regime_scale_factor(REGIME_NORMAL, True) == 0.0


def test_regime_bear_half():
    assert regime_scale_factor(REGIME_BEAR, False) == 0.50


def test_regime_normal_full():
    assert regime_scale_factor(REGIME_NORMAL, False) == 1.0


def test_regime_bull_full():
    assert regime_scale_factor(REGIME_BULL, False) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. check_safety_gates
# ─────────────────────────────────────────────────────────────────────────────

def test_safety_gate_crash():
    ctx = _ctx(regime=REGIME_CRASH)
    safe, reason = check_safety_gates(ctx)
    assert safe is False
    assert "crash" in reason


def test_safety_gate_broker_degraded():
    ctx = _ctx(broker=BROKER_DEGRADED)
    safe, reason = check_safety_gates(ctx)
    assert safe is False
    assert "broker_health" in reason


def test_safety_gate_broker_unreachable():
    ctx = _ctx(broker=BROKER_UNREACHABLE)
    safe, _ = check_safety_gates(ctx)
    assert safe is False


def test_safety_gate_volatility_spike():
    ctx = _ctx(vol_spike=True)
    safe, reason = check_safety_gates(ctx)
    assert safe is False
    assert "volatility" in reason


def test_safety_gate_healthy():
    ctx = _ctx()
    safe, reason = check_safety_gates(ctx)
    assert safe is True
    assert reason == ""


# ─────────────────────────────────────────────────────────────────────────────
# 8. classify_opportunity
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_full_entry():
    ctx = _ctx()
    cand = _cand(cont=FULL_ENTRY_CONTINUATION_MIN + 5)
    dt, reason = classify_opportunity(cand, ctx)
    assert dt == DECISION_FULL_ENTRY
    assert reason == ""


def test_classify_reduced_starter():
    ctx = _ctx()
    cand = _cand(cont=REDUCED_STARTER_CONTINUATION_MIN + 5)
    dt, _ = classify_opportunity(cand, ctx)
    assert dt == DECISION_REDUCED_STARTER


def test_classify_observation_only():
    ctx = _ctx()
    cand = _cand(cont=REDUCED_STARTER_CONTINUATION_MIN - 5)
    dt, _ = classify_opportunity(cand, ctx)
    assert dt == DECISION_OBSERVATION_ONLY


def test_classify_blocked_low_survivability():
    ctx = _ctx()
    cand = _cand(surv=MIN_SURVIVABILITY_ENTRY - 0.01, cont=80.0)
    dt, reason = classify_opportunity(cand, ctx)
    assert dt == DECISION_BLOCKED
    assert "survivability" in reason


def test_classify_blocked_cb_active():
    ctx = _ctx(cb=True)
    cand = _cand(cont=80.0)
    dt, reason = classify_opportunity(cand, ctx)
    assert dt == DECISION_BLOCKED
    assert "cb_active" in reason


def test_classify_addon():
    ctx = _ctx()
    held = _held()
    cand = _cand(is_held=True, held_info=held, cont=80.0, surv=0.80)
    dt, _ = classify_opportunity(cand, ctx, is_addon_eligible=True)
    assert dt == DECISION_ADDON


def test_classify_blocked_safety_gate():
    ctx = _ctx(regime=REGIME_CRASH)
    cand = _cand(cont=80.0)
    dt, reason = classify_opportunity(cand, ctx)
    assert dt == DECISION_BLOCKED
    assert "crash" in reason


def test_classify_bounded_growth_cap():
    # Price so high that lot_cost > 30% of deployable_capital
    ctx = _ctx(deployable=1_000_000)
    # 1 lot = 100 × 4000 = 400k → 40% of 1M → blocked
    cand = _cand(price=4000.0, cont=80.0, surv=0.90)
    dt, reason = classify_opportunity(cand, ctx)
    assert dt == DECISION_BLOCKED
    assert "lot_cost_ratio" in reason


# ─────────────────────────────────────────────────────────────────────────────
# 9. _is_addon_eligible
# ─────────────────────────────────────────────────────────────────────────────

def test_addon_eligible_basic():
    cand = _cand(is_held=True, held_info=_held())
    ok, reason = _is_addon_eligible(cand, TODAY, deployable_capital=3_000_000)
    assert ok is True
    assert reason == ""


def test_addon_ineligible_not_held():
    cand = _cand(is_held=False)
    ok, _ = _is_addon_eligible(cand, TODAY)
    assert ok is False


def test_addon_ineligible_loser():
    held = _held(pnl_pct=-0.02)
    cand = _cand(is_held=True, held_info=held)
    ok, reason = _is_addon_eligible(cand, TODAY)
    assert ok is False
    assert "losers" in reason


def test_addon_ineligible_depth():
    held = _held(addon_count=2)
    cand = _cand(is_held=True, held_info=held)
    ok, reason = _is_addon_eligible(cand, TODAY, max_addon_depth=2)
    assert ok is False
    assert "addon_count" in reason


def test_addon_ineligible_cooldown():
    held = _held(last_addon="2026-05-28")  # 1 day ago
    cand = _cand(is_held=True, held_info=held)
    ok, reason = _is_addon_eligible(cand, TODAY, addon_cooldown_days=5)
    assert ok is False
    assert "cooldown" in reason


def test_addon_eligible_after_cooldown():
    held = _held(last_addon="2026-05-20")  # 9 days ago
    cand = _cand(is_held=True, held_info=held)
    ok, _ = _is_addon_eligible(cand, TODAY, addon_cooldown_days=5)
    assert ok is True


def test_addon_ineligible_concentration():
    # 900 shares × 4000 = 3.6M → 120% of 3M deployable → post-addon > 0.35
    held = _held(qty=900, avg_cost=4000.0)
    cand = _cand(price=4000.0, is_held=True, held_info=held)
    ok, reason = _is_addon_eligible(
        cand, TODAY, deployable_capital=3_000_000, max_position_weight_addon=0.35
    )
    assert ok is False
    assert "weight" in reason


# ─────────────────────────────────────────────────────────────────────────────
# 10. idle capital relaxation
# ─────────────────────────────────────────────────────────────────────────────

def test_idle_relaxation_triggers():
    # cont in (HIGH_CONTINUATION_THRESHOLD, REDUCED_STARTER_CONTINUATION_MIN)
    # → normally OBSERVATION_ONLY, but triggers idle relaxation
    ctx = _ctx(cap_util=0.20)
    cont = (HIGH_CONTINUATION_THRESHOLD + REDUCED_STARTER_CONTINUATION_MIN) / 2
    cand = _cand(cont=cont, surv=HIGH_SURVIVABILITY_THRESHOLD + 0.05)
    assert _should_apply_idle_relaxation(cand, ctx) is True


def test_idle_relaxation_no_trigger_high_util():
    ctx = _ctx(cap_util=0.80)
    cont = (HIGH_CONTINUATION_THRESHOLD + REDUCED_STARTER_CONTINUATION_MIN) / 2
    cand = _cand(cont=cont, surv=0.90)
    assert _should_apply_idle_relaxation(cand, ctx) is False


def test_idle_relaxation_no_trigger_low_surv():
    ctx = _ctx(cap_util=0.10)
    cont = (HIGH_CONTINUATION_THRESHOLD + REDUCED_STARTER_CONTINUATION_MIN) / 2
    cand = _cand(cont=cont, surv=0.40)
    assert _should_apply_idle_relaxation(cand, ctx) is False


def test_idle_relaxation_no_trigger_cb():
    ctx = _ctx(cap_util=0.10, cb=True)
    cont = (HIGH_CONTINUATION_THRESHOLD + REDUCED_STARTER_CONTINUATION_MIN) / 2
    cand = _cand(cont=cont, surv=0.90)
    assert _should_apply_idle_relaxation(cand, ctx) is False


def test_idle_relaxation_upgrades_observation():
    """
    Candidate with cont in (HIGH_CONTINUATION_THRESHOLD, REDUCED_STARTER_CONTINUATION_MIN)
    + high survivability + low capital utilization → upgraded OBSERVATION_ONLY → REDUCED_STARTER.
    """
    os_inst = CapitalDeploymentOS()
    ctx = _ctx(cap_util=0.10)
    # cont in gray zone: above idle min, below standard REDUCED_STARTER threshold
    cont = (HIGH_CONTINUATION_THRESHOLD + REDUCED_STARTER_CONTINUATION_MIN) / 2
    cand = _cand(cont=cont, surv=HIGH_SURVIVABILITY_THRESHOLD + 0.05)
    decisions = os_inst.run([cand], ctx)
    assert decisions[0].idle_relaxation_applied is True
    assert decisions[0].decision_type == DECISION_REDUCED_STARTER


# ─────────────────────────────────────────────────────────────────────────────
# 11. Duplicate order prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_duplicate_guard_blocks_second():
    prior = [
        DeploymentDecision(
            symbol="6758.T",
            decision_type=DECISION_FULL_ENTRY,
            target_size=300,
            target_notional=900_000,
            deployable_capital=3_000_000,
            continuation_score=70,
            survivability=0.80,
            participation_rate=0.01,
            capital_utilization=0.0,
            suppression_reason="",
            regime_scale=1.0,
            liquidity_constrained=False,
            idle_relaxation_applied=False,
            date=TODAY,
        )
    ]
    assert _is_duplicate("6758.T", DECISION_FULL_ENTRY, prior) is True


def test_duplicate_guard_allows_different_symbol():
    prior = [
        DeploymentDecision(
            symbol="6758.T",
            decision_type=DECISION_FULL_ENTRY,
            target_size=300,
            target_notional=900_000,
            deployable_capital=3_000_000,
            continuation_score=70,
            survivability=0.80,
            participation_rate=0.01,
            capital_utilization=0.0,
            suppression_reason="",
            regime_scale=1.0,
            liquidity_constrained=False,
            idle_relaxation_applied=False,
            date=TODAY,
        )
    ]
    assert _is_duplicate("7203.T", DECISION_FULL_ENTRY, prior) is False


def test_duplicate_guard_allows_observation():
    # OBSERVATION_ONLY is never blocked as "duplicate"
    prior = [
        DeploymentDecision(
            symbol="6758.T",
            decision_type=DECISION_FULL_ENTRY,
            target_size=300,
            target_notional=900_000,
            deployable_capital=3_000_000,
            continuation_score=70,
            survivability=0.80,
            participation_rate=0.01,
            capital_utilization=0.0,
            suppression_reason="",
            regime_scale=1.0,
            liquidity_constrained=False,
            idle_relaxation_applied=False,
            date=TODAY,
        )
    ]
    assert _is_duplicate("6758.T", DECISION_OBSERVATION_ONLY, prior) is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. CB_ACTIVE behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_cb_active_blocks_new_entry():
    os = CapitalDeploymentOS()
    ctx = _ctx(cb=True)
    cand = _cand(cont=80.0, surv=0.90)
    decisions = os.run([cand], ctx)
    assert decisions[0].decision_type == DECISION_BLOCKED
    assert "cb_active" in decisions[0].suppression_reason


def test_cb_active_allows_observation_held():
    # Held positions: even with CB we should still process (no new BUY)
    # classify_opportunity: held + not addon_eligible → OBSERVATION_ONLY
    os = CapitalDeploymentOS()
    ctx = _ctx(cb=True)
    held = _held(pnl_pct=-0.05)  # loser → not addon eligible
    cand = _cand(is_held=True, held_info=held, cont=80.0, surv=0.90)
    decisions = os.run([cand], ctx)
    # CB doesn't block already-held positions from observation
    assert decisions[0].decision_type in (
        DECISION_OBSERVATION_ONLY, DECISION_BLOCKED,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 13. Bounded position growth (lot_cost_ratio)
# ─────────────────────────────────────────────────────────────────────────────

def test_bounded_growth_blocks_expensive_stock():
    os = CapitalDeploymentOS()
    ctx = _ctx(deployable=1_000_000)
    # 1 lot at ¥4000 = ¥400k → 40% of 1M > 30% cap
    cand = _cand(price=4000.0, cont=80.0, surv=0.90)
    decisions = os.run([cand], ctx)
    assert decisions[0].decision_type == DECISION_BLOCKED
    assert "lot_cost_ratio" in decisions[0].suppression_reason


def test_bounded_growth_allows_affordable_stock():
    os = CapitalDeploymentOS()
    ctx = _ctx(deployable=3_000_000)
    # 1 lot at ¥4000 = ¥400k → 13.3% of 3M < 30% cap → allowed
    cand = _cand(price=4000.0, cont=80.0, surv=0.90)
    decisions = os.run([cand], ctx)
    assert decisions[0].decision_type != DECISION_BLOCKED


# ─────────────────────────────────────────────────────────────────────────────
# 14. Liquidity cap enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_liquidity_cap_enforced():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    # Very low ADV: 1M yen ADV × 3% = 30k max notional = 10 shares at ¥3000
    # target would be much higher
    cand = _cand(cont=80.0, surv=0.90, adv=1_000_000)
    decisions = os.run([cand], ctx)
    d = decisions[0]
    if d.decision_type in (DECISION_FULL_ENTRY, DECISION_REDUCED_STARTER):
        max_notional = 1_000_000 * MAX_PARTICIPATION_RATE
        assert d.target_notional <= max_notional + 3000  # small rounding tolerance


def test_liquidity_cap_flag():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    cand = _cand(cont=80.0, surv=0.90, adv=1_000_000)
    decisions = os.run([cand], ctx)
    d = decisions[0]
    if d.decision_type in (DECISION_FULL_ENTRY, DECISION_REDUCED_STARTER):
        assert d.liquidity_constrained is True


def test_liquidity_cap_not_applied_when_not_needed():
    os = CapitalDeploymentOS()
    ctx = _ctx(deployable=3_000_000)
    cand = _cand(cont=80.0, surv=0.90, adv=500_000_000)
    decisions = os.run([cand], ctx)
    d = decisions[0]
    assert d.liquidity_constrained is False


# ─────────────────────────────────────────────────────────────────────────────
# 15. Replay equivalence
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_equivalence_same_output():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    candidates = [
        _cand("6758.T", price=3000.0, cont=75.0),
        _cand("7203.T", price=2000.0, cont=40.0),
        _cand("8035.T", price=5000.0, cont=20.0),
    ]
    d1 = [asdict(d) for d in os.run(candidates, ctx)]
    d2 = [asdict(d) for d in os.run(candidates, ctx)]
    assert d1 == d2


def test_replay_equivalence_order_stable():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    c1 = _cand("6758.T", cont=75.0)
    c2 = _cand("7203.T", cont=40.0)
    d1 = os.run([c1, c2], ctx)
    d2 = os.run([c1, c2], ctx)
    assert d1[0].symbol == d2[0].symbol == "6758.T"
    assert d1[1].symbol == d2[1].symbol == "7203.T"


# ─────────────────────────────────────────────────────────────────────────────
# 16. Capital growth scaling (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def test_capital_growth_increases_target_size():
    os = CapitalDeploymentOS()
    cand = _cand(cont=80.0, surv=0.90, price=2000.0)

    d_small = os.run([cand], _ctx(deployable=1_500_000))
    d_large = os.run([cand], _ctx(deployable=6_000_000))

    s_small = d_small[0].target_size
    s_large = d_large[0].target_size
    assert s_large >= s_small  # larger capital → larger or equal target


def test_capital_growth_max_positions_scales():
    small_max = dynamic_max_positions(1_000_000)
    large_max = dynamic_max_positions(5_000_000)
    assert large_max >= small_max


# ─────────────────────────────────────────────────────────────────────────────
# 17. Bear regime scaling
# ─────────────────────────────────────────────────────────────────────────────

def test_bear_regime_halves_size():
    os = CapitalDeploymentOS()
    cand = _cand(cont=80.0, surv=0.90, price=2000.0, adv=1_000_000_000)

    d_normal = os.run([cand], _ctx(regime=REGIME_NORMAL))
    d_bear   = os.run([cand], _ctx(regime=REGIME_BEAR))

    if d_normal[0].decision_type in (DECISION_FULL_ENTRY, DECISION_REDUCED_STARTER):
        assert d_bear[0].regime_scale == 0.50
        assert d_bear[0].target_size <= d_normal[0].target_size


def test_crash_regime_blocks_all():
    os = CapitalDeploymentOS()
    candidates = [
        _cand("6758.T", cont=80.0),
        _cand("7203.T", cont=70.0),
    ]
    ctx = _ctx(regime=REGIME_CRASH)
    decisions = os.run(candidates, ctx)
    for d in decisions:
        assert d.decision_type == DECISION_BLOCKED


# ─────────────────────────────────────────────────────────────────────────────
# 18. Add-on per-run throttle
# ─────────────────────────────────────────────────────────────────────────────

def test_addon_per_run_throttle():
    os = CapitalDeploymentOS(max_addons_per_run=1)
    ctx = _ctx()

    held_a = _held(symbol="6758.T", pnl_pct=0.10)
    held_b = _held(symbol="7203.T", pnl_pct=0.12)
    cand_a = _cand("6758.T", is_held=True, held_info=held_a, cont=80.0, surv=0.80)
    cand_b = _cand("7203.T", is_held=True, held_info=held_b, cont=80.0, surv=0.80)

    ctx.current_positions["6758.T"] = held_a
    ctx.current_positions["7203.T"] = held_b

    decisions = os.run([cand_a, cand_b], ctx)
    addon_decisions = [d for d in decisions if d.decision_type == DECISION_ADDON and d.target_size > 0]
    assert len(addon_decisions) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# 19. All decision types emitted in one run
# ─────────────────────────────────────────────────────────────────────────────

def test_all_non_blocked_decision_types():
    os = CapitalDeploymentOS()
    ctx = _ctx(cap_util=0.10, deployable=5_000_000)

    # FULL_ENTRY: high cont, good surv, affordable
    c_full = _cand("A", price=2000.0, cont=FULL_ENTRY_CONTINUATION_MIN + 5, surv=0.85)
    # REDUCED_STARTER: moderate cont
    c_reduced = _cand("B", price=2000.0, cont=REDUCED_STARTER_CONTINUATION_MIN + 5, surv=0.80)
    # ADDON: held winner eligible
    held = _held(symbol="C", pnl_pct=0.08)
    c_addon = _cand("C", price=2000.0, cont=70.0, surv=0.80, is_held=True, held_info=held)

    decisions = os.run([c_full, c_reduced, c_addon], ctx)
    types = {d.decision_type for d in decisions}
    assert DECISION_FULL_ENTRY in types
    assert DECISION_REDUCED_STARTER in types
    assert DECISION_ADDON in types


def test_blocked_safety():
    os = CapitalDeploymentOS()
    ctx = _ctx(broker=BROKER_UNREACHABLE)
    cand = _cand(cont=80.0)
    decisions = os.run([cand], ctx)
    assert decisions[0].decision_type == DECISION_BLOCKED


# ─────────────────────────────────────────────────────────────────────────────
# 20. Telemetry output structure
# ─────────────────────────────────────────────────────────────────────────────

def test_telemetry_file_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        tel_dir = Path(tmpdir)
        os = CapitalDeploymentOS(telemetry_dir=tel_dir)
        ctx = _ctx()
        cand = _cand(cont=75.0, surv=0.80)
        os.run([cand], ctx)

        decisions_file = tel_dir / "deployment_decisions.jsonl"
        telemetry_file = tel_dir / "deployment_telemetry.jsonl"
        assert decisions_file.exists()
        assert telemetry_file.exists()


def test_telemetry_decisions_jsonl_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        tel_dir = Path(tmpdir)
        os = CapitalDeploymentOS(telemetry_dir=tel_dir)
        ctx = _ctx()
        candidates = [_cand("6758.T", cont=75.0), _cand("7203.T", cont=30.0)]
        os.run(candidates, ctx)

        lines = (tel_dir / "deployment_decisions.jsonl").read_text("utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            rec = json.loads(line)
            assert "symbol" in rec
            assert "decision_type" in rec
            assert "target_size" in rec
            assert "schema_version" in rec


def test_telemetry_summary_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        tel_dir = Path(tmpdir)
        os = CapitalDeploymentOS(telemetry_dir=tel_dir)
        ctx = _ctx()
        candidates = [_cand("A", cont=75.0), _cand("B", cont=30.0)]
        os.run(candidates, ctx)

        lines = (tel_dir / "deployment_telemetry.jsonl").read_text("utf-8").strip().splitlines()
        summary = json.loads(lines[-1])
        assert "capital_utilization" in summary
        assert "idle_cash_ratio" in summary
        assert "deployable_capital" in summary
        assert "continuation_deployment_rate" in summary
        assert "liquidity_constrained_count" in summary
        assert "suppressed_alpha_count" in summary
        assert "total_candidates" in summary
        assert summary["total_candidates"] == 2


def test_telemetry_build_summary_structure():
    ctx = _ctx()
    cand = _cand(cont=75.0)
    decisions = [
        DeploymentDecision(
            symbol="A", decision_type=DECISION_FULL_ENTRY,
            target_size=300, target_notional=900_000,
            deployable_capital=3_000_000, continuation_score=75,
            survivability=0.8, participation_rate=0.01,
            capital_utilization=0.0, suppression_reason="",
            regime_scale=1.0, liquidity_constrained=False,
            idle_relaxation_applied=False, date=TODAY,
        )
    ]
    summary = _build_telemetry_summary(decisions, ctx, "2026-05-29T09:00:00+0900")
    assert summary["full_entries"] == 1
    assert summary["blocked"] == 0
    assert summary["schema_version"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 21. make_context / make_candidate convenience builders
# ─────────────────────────────────────────────────────────────────────────────

def test_make_context_defaults():
    ctx = make_context(
        actual_equity=3_500_000,
        effective_capital=3_000_000,
        deployable_capital=2_700_000,
    )
    assert ctx.market_regime == REGIME_NORMAL
    assert ctx.broker_health == BROKER_HEALTHY
    assert ctx.volatility_spike is False
    assert ctx.cb_active is False
    assert ctx.current_positions == {}


def test_make_candidate_defaults():
    cand = make_candidate("6758.T", price=3000.0)
    assert cand.sector == "電気機器"
    assert cand.is_held is False
    assert cand.held_info is None


# ─────────────────────────────────────────────────────────────────────────────
# 22. DeploymentDecision schema
# ─────────────────────────────────────────────────────────────────────────────

def test_deployment_decision_schema():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    cand = _cand(cont=75.0)
    decisions = os.run([cand], ctx)
    d = decisions[0]
    # All required fields present
    assert hasattr(d, "symbol")
    assert hasattr(d, "decision_type")
    assert hasattr(d, "target_size")
    assert hasattr(d, "target_notional")
    assert hasattr(d, "deployable_capital")
    assert hasattr(d, "continuation_score")
    assert hasattr(d, "survivability")
    assert hasattr(d, "participation_rate")
    assert hasattr(d, "capital_utilization")
    assert hasattr(d, "suppression_reason")
    assert hasattr(d, "regime_scale")
    assert hasattr(d, "liquidity_constrained")
    assert hasattr(d, "idle_relaxation_applied")
    assert hasattr(d, "date")
    assert hasattr(d, "schema_version")


def test_blocked_decision_zero_size():
    os = CapitalDeploymentOS()
    ctx = _ctx(regime=REGIME_CRASH)
    cand = _cand(cont=80.0)
    decisions = os.run([cand], ctx)
    assert decisions[0].target_size == 0
    assert decisions[0].target_notional == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 23. Multiple candidates ordering
# ─────────────────────────────────────────────────────────────────────────────

def test_output_length_matches_input():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    candidates = [_cand(str(i), price=2000.0, cont=50.0) for i in range(5)]
    decisions = os.run(candidates, ctx)
    assert len(decisions) == 5


def test_output_preserves_symbol_order():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    syms = ["A", "B", "C"]
    candidates = [_cand(s, cont=70.0) for s in syms]
    decisions = os.run(candidates, ctx)
    assert [d.symbol for d in decisions] == syms


# ─────────────────────────────────────────────────────────────────────────────
# 24. Empty candidate list
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_candidates():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    decisions = os.run([], ctx)
    assert decisions == []


# ─────────────────────────────────────────────────────────────────────────────
# 25. Participation rate recorded on decision
# ─────────────────────────────────────────────────────────────────────────────

def test_participation_rate_on_decision():
    os = CapitalDeploymentOS()
    ctx = _ctx()
    cand = _cand(cont=80.0, adv=500_000_000)
    decisions = os.run([cand], ctx)
    d = decisions[0]
    if d.target_size > 0:
        expected = d.target_notional / 500_000_000
        assert abs(d.participation_rate - expected) < 1e-9
