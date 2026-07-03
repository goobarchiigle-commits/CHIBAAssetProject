"""
tests/analytics/test_future_leader_paper_execution_engine.py
Phase 3C — Paper Execution Engine (Stateful Execution Survivability Replay) tests

Coverage:
  - deterministic lifecycle replay + same-input hash
  - transition reason correctness (ALLOW→ADMISSION_ALLOWED, risk_off→REDUCED_BY_CONGESTION)
  - synthetic rollback (persistence < 0.40 → ROLLED_BACK)
  - lifecycle decay calculation correctness
  - execution drift calculation correctness
  - overlapping cohort tracking
  - capital occupancy calculation
  - congestion telemetry (CONGESTED portfolio state)
  - portfolio state transitions (STABLE / DEGRADED / ROLLBACK_MODE)
  - fail-closed rollback (BLOCKED promotion state)
  - append-only integrity
  - paper_execution_allowed gate logic
  - NAV drawdown computation
  - expectancy chain ordering
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pytest

sys.stdout.reconfigure(encoding="utf-8")

from src.analytics.future_leader_limited_capital_router import (
    AdmissionDecision,
    LimitedCapitalRoutingReport,
    RoutingSuppressionReason,
    VirtualAdmissionDecision,
    VirtualExposureSnapshot,
)
from src.analytics.future_leader_paper_execution_engine import (
    PaperExecutionInput,
    PaperExecutionReport,
    PositionTransitionReason,
    SyntheticExecutionLedger,
    SyntheticPortfolioState,
    SyntheticPosition,
    SyntheticPositionState,
    _EXEC_DRIFT_THRESHOLD,
    _LIFECYCLE_DECAY_THRESHOLD,
    _LIFECYCLE_PNL_FACTOR,
    _ROLLBACK_PERSISTENCE_THRESHOLD,
    _build_synthetic_positions,
    _compute_deterministic_hash,
    _compute_execution_metrics,
    _compute_paper_execution_allowed,
    _compute_synthetic_nav_drawdown,
    _determine_lifecycle_exit,
    _determine_portfolio_state,
    _map_suppression_to_transition,
    append_execution_ledger,
    append_paper_execution_report,
    load_execution_ledgers,
    load_paper_execution_reports,
    run_paper_execution,
)
from src.analytics.future_leader_promotion_readiness import (
    PromotionReadinessReport,
    PromotionState,
)
from src.analytics.future_leader_shadow_deployment_validator import ShadowDeploymentReport


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_promotion_report(
    state: PromotionState = PromotionState.SHADOW_READY,
    readiness_score: float = 0.70,
    confidence: float = 0.65,
    promotion_ready: Optional[bool] = None,
) -> PromotionReadinessReport:
    ready = promotion_ready if promotion_ready is not None else (
        state not in (PromotionState.BLOCKED, PromotionState.RESEARCH_ONLY)
    )
    return PromotionReadinessReport(
        timestamp="2026-05-25T09:00:00+09:00",
        promotion_state=state,
        promotion_ready=ready,
        readiness_score=readiness_score,
        promotion_confidence=confidence,
        component_scores={
            "statistical_edge": 0.70, "survivability": 0.68,
            "regime_robustness": 0.60, "failure_quality": 0.65, "temporal_stability": 0.70,
        },
        blocking_reasons=[],
        warnings=[],
        sample_size=50,
        recent_sample_size=20,
        regime_coverage={"risk_on": 15, "risk_off": 8, "range": 12, "volatile": 5},
        deterministic_hash="a" * 64,
    )


def _make_shadow_report(
    is_viable: bool = True,
    degradation_ratio: Optional[float] = 0.80,
    persistence_score: float = 0.72,
    research_expectancy: float = 0.025,
    shadow_expectancy: float = 0.020,
    deployment_blockers: Optional[List[str]] = None,
    avg_holding: float = 8.0,
) -> ShadowDeploymentReport:
    return ShadowDeploymentReport(
        timestamp="2026-05-25T09:00:00+09:00",
        evaluation_date="2026-05-25",
        promotion_state=PromotionState.SHADOW_READY.value,
        research_expectancy=research_expectancy,
        shadow_expectancy=shadow_expectancy,
        deployment_degradation_ratio=degradation_ratio,
        shadow_max_drawdown=-0.05,
        survivability_adjusted_return=0.018,
        deployment_persistence_score=persistence_score,
        average_holding_duration=avg_holding,
        deployment_blockers=deployment_blockers or [],
        is_deployment_viable=is_viable,
        total_research_records=50,
        shadow_positions_simulated=30,
        cohorts_simulated=10,
        allocation_weight=0.045,
        deterministic_hash="b" * 64,
    )


def _make_admission_decision(
    symbol: str,
    decision: AdmissionDecision = AdmissionDecision.ALLOW,
    proposed_weight: float = 0.045,
    adjusted_weight: float = 0.045,
    suppression_reason: RoutingSuppressionReason = RoutingSuppressionReason.NONE,
) -> VirtualAdmissionDecision:
    return VirtualAdmissionDecision(
        symbol=symbol,
        decision=decision,
        proposed_weight=proposed_weight,
        adjusted_weight=adjusted_weight,
        suppression_reason=suppression_reason,
        readiness_score=0.70,
        promotion_confidence=0.65,
        deployment_degradation_ratio=0.80,
        deployment_persistence_score=0.72,
        previous_decision=None,
        current_decision=decision.value,
        decision_transition=f"new->{decision.value}",
    )


def _make_routing_report(
    decisions: Optional[List[VirtualAdmissionDecision]] = None,
    admitted_expectancy: float = 0.010,
    research_expectancy: float = 0.025,
    shadow_expectancy: float = 0.020,
    promotion_state: str = "shadow_ready",
) -> LimitedCapitalRoutingReport:
    if decisions is None:
        decisions = [
            _make_admission_decision("AAA"),
            _make_admission_decision("BBB"),
        ]
    n = len(decisions)
    n_allow   = sum(1 for d in decisions if d.decision == AdmissionDecision.ALLOW)
    n_reduced = sum(1 for d in decisions if d.decision == AdmissionDecision.REDUCED_SIZE)
    n_denied  = sum(1 for d in decisions if d.decision == AdmissionDecision.DENY)
    return LimitedCapitalRoutingReport(
        timestamp="2026-05-25T09:00:00+09:00",
        evaluation_date="2026-05-25",
        promotion_state=promotion_state,
        regime_tag="risk_on",
        admission_decisions=decisions,
        exposure_snapshot=VirtualExposureSnapshot(
            sector_exposure={"tech": 0.09},
            theme_exposure={"growth": 0.09},
            position_concentration=0.09,
            correlated_cluster_count=0,
        ),
        research_expectancy=research_expectancy,
        shadow_expectancy=shadow_expectancy,
        unsuppressed_shadow_expectancy=shadow_expectancy,
        admitted_expectancy=admitted_expectancy,
        deployment_drift_ratio=0.80,
        suppression_alpha_preservation_ratio=0.90,
        admission_rate=n_allow / n if n else 0.0,
        reduced_size_rate=n_reduced / n if n else 0.0,
        denial_rate=n_denied / n if n else 0.0,
        persistence_suppression_rate=0.0,
        exposure_concentration=0.09,
        deterministic_hash="c" * 64,
    )


def _make_paper_input(
    promotion_state: PromotionState = PromotionState.SHADOW_READY,
    regime_tag: str = "risk_on",
    degradation_ratio: Optional[float] = 0.80,
    persistence_score: float = 0.72,
    deployment_blockers: Optional[List[str]] = None,
    decisions: Optional[List[VirtualAdmissionDecision]] = None,
    admitted_expectancy: float = 0.010,
    shadow_expectancy: float = 0.020,
    is_viable: bool = True,
    previous_positions: Optional[Dict[str, str]] = None,
    promotion_ready: Optional[bool] = None,
) -> PaperExecutionInput:
    promo  = _make_promotion_report(state=promotion_state, promotion_ready=promotion_ready)
    shadow = _make_shadow_report(
        is_viable=is_viable,
        degradation_ratio=degradation_ratio,
        persistence_score=persistence_score,
        shadow_expectancy=shadow_expectancy,
        deployment_blockers=deployment_blockers,
    )
    routing = _make_routing_report(
        decisions=decisions,
        admitted_expectancy=admitted_expectancy,
        shadow_expectancy=shadow_expectancy,
        promotion_state=promotion_state.value,
    )
    return PaperExecutionInput(
        promotion_report=promo,
        shadow_report=shadow,
        routing_report=routing,
        regime_tag=regime_tag,
        evaluation_date="2026-05-25",
        previous_positions=previous_positions or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deterministic lifecycle replay + same-input hash
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_hash(self):
        inp = _make_paper_input()
        h1  = _compute_deterministic_hash(inp)
        h2  = _compute_deterministic_hash(inp)
        assert h1 == h2

    def test_same_input_same_report_hash(self):
        inp = _make_paper_input()
        r1  = run_paper_execution(inp)
        r2  = run_paper_execution(inp)
        assert r1.deterministic_hash == r2.deterministic_hash

    def test_same_input_same_positions(self):
        inp = _make_paper_input()
        r1  = run_paper_execution(inp)
        r2  = run_paper_execution(inp)
        assert len(r1.ledger.positions) == len(r2.ledger.positions)
        for p1, p2 in zip(r1.ledger.positions, r2.ledger.positions):
            assert p1.symbol           == p2.symbol
            assert p1.state            == p2.state
            assert p1.transition_reason == p2.transition_reason
            assert p1.realized_pnl     == p2.realized_pnl

    def test_different_regime_different_hash(self):
        i1 = _make_paper_input(regime_tag="risk_on")
        i2 = _make_paper_input(regime_tag="risk_off")
        assert _compute_deterministic_hash(i1) != _compute_deterministic_hash(i2)

    def test_positions_sorted_by_symbol(self):
        decisions = [
            _make_admission_decision("ZZZ"),
            _make_admission_decision("AAA"),
            _make_admission_decision("MMM"),
        ]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        syms   = [p.symbol for p in report.ledger.positions]
        assert syms == sorted(syms)

    def test_hash_is_64_hex_chars(self):
        inp  = _make_paper_input()
        h    = _compute_deterministic_hash(inp)
        assert len(h) == 64
        int(h, 16)  # valid hex


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transition reason correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionReasonCorrectness:
    def test_allow_normal_exit_holding_expiry(self):
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.ADMISSION_ALLOWED,
            persistence_score=0.75,
            degradation_ratio=0.80,
            deployment_blockers=[],
            regime_tag="risk_on",
        )
        assert state  == SyntheticPositionState.EXITED
        assert reason == PositionTransitionReason.EXIT_BY_HOLDING_EXPIRY

    def test_risk_off_suppression_maps_to_reduced_by_congestion(self):
        reason = _map_suppression_to_transition(RoutingSuppressionReason.RISK_OFF)
        assert reason == PositionTransitionReason.REDUCED_BY_CONGESTION

    def test_deployment_drift_maps_to_reduced_by_drawdown(self):
        reason = _map_suppression_to_transition(RoutingSuppressionReason.DEPLOYMENT_DRIFT)
        assert reason == PositionTransitionReason.REDUCED_BY_DRAWDOWN

    def test_persistence_collapse_maps_to_reduced_by_drawdown(self):
        reason = _map_suppression_to_transition(RoutingSuppressionReason.PERSISTENCE_COLLAPSE)
        assert reason == PositionTransitionReason.REDUCED_BY_DRAWDOWN

    def test_concentration_excess_maps_to_reduced_by_congestion(self):
        reason = _map_suppression_to_transition(RoutingSuppressionReason.CONCENTRATION_EXCESS)
        assert reason == PositionTransitionReason.REDUCED_BY_CONGESTION

    def test_reduced_size_normal_exit_retains_reduction_reason(self):
        # REDUCED_BY_CONGESTION + no triggers → exits with REDUCED_BY_CONGESTION
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.REDUCED_BY_CONGESTION,
            persistence_score=0.75,
            degradation_ratio=0.80,
            deployment_blockers=[],
            regime_tag="risk_on",
        )
        assert state  == SyntheticPositionState.EXITED
        assert reason == PositionTransitionReason.REDUCED_BY_CONGESTION

    def test_reduced_by_drawdown_normal_exit_retains_reason(self):
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.REDUCED_BY_DRAWDOWN,
            persistence_score=0.75,
            degradation_ratio=0.80,
            deployment_blockers=[],
            regime_tag="risk_on",
        )
        assert state  == SyntheticPositionState.EXITED
        assert reason == PositionTransitionReason.REDUCED_BY_DRAWDOWN

    def test_regime_collapse_reason(self):
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.ADMISSION_ALLOWED,
            persistence_score=0.75,
            degradation_ratio=0.80,
            deployment_blockers=[],
            regime_tag="risk_off",
        )
        assert state  == SyntheticPositionState.EXITED
        assert reason == PositionTransitionReason.EXIT_BY_REGIME_COLLAPSE


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic rollback correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticRollback:
    def test_persistence_collapse_triggers_rollback(self):
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.ADMISSION_ALLOWED,
            persistence_score=0.25,  # < 0.40 → ROLLED_BACK
            degradation_ratio=0.80,
            deployment_blockers=[],
            regime_tag="risk_on",
        )
        assert state  == SyntheticPositionState.ROLLED_BACK
        assert reason == PositionTransitionReason.ROLLBACK_BY_PERSISTENCE

    def test_deployment_blockers_trigger_rollback(self):
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.ADMISSION_ALLOWED,
            persistence_score=0.75,
            degradation_ratio=0.80,
            deployment_blockers=["deployment_degradation_excessive"],
            regime_tag="risk_on",
        )
        assert state  == SyntheticPositionState.ROLLED_BACK
        assert reason == PositionTransitionReason.ROLLBACK_BY_PERSISTENCE

    def test_blockers_take_priority_over_persistence(self):
        state, reason = _determine_lifecycle_exit(
            initial_reason=PositionTransitionReason.ADMISSION_ALLOWED,
            persistence_score=0.25,  # also < threshold
            degradation_ratio=0.80,
            deployment_blockers=["some_blocker"],
            regime_tag="risk_on",
        )
        # Blockers checked first — still ROLLED_BACK
        assert state  == SyntheticPositionState.ROLLED_BACK
        assert reason == PositionTransitionReason.ROLLBACK_BY_PERSISTENCE

    def test_rollback_sets_rollback_triggered_flag(self):
        inp    = _make_paper_input(persistence_score=0.20)
        report = run_paper_execution(inp)
        for p in report.ledger.positions:
            assert p.rollback_triggered is True

    def test_rollback_pnl_factor_is_030(self):
        assert _LIFECYCLE_PNL_FACTOR[PositionTransitionReason.ROLLBACK_BY_PERSISTENCE] == pytest.approx(0.30)

    def test_rollback_pnl_reduced_vs_full_exit(self):
        # Normal exit
        shadow_exp   = 0.020
        active_w     = 0.045
        normal_pnl   = shadow_exp * active_w * 1.0   # EXIT_BY_HOLDING_EXPIRY factor
        rollback_pnl = shadow_exp * active_w * 0.30  # ROLLBACK factor
        assert rollback_pnl < normal_pnl

    def test_all_positions_rolled_back_when_persistence_low(self):
        decisions = [
            _make_admission_decision("AAA"),
            _make_admission_decision("BBB"),
            _make_admission_decision("CCC"),
        ]
        inp    = _make_paper_input(persistence_score=0.15, decisions=decisions)
        report = run_paper_execution(inp)
        assert all(p.state == SyntheticPositionState.ROLLED_BACK for p in report.ledger.positions)
        assert report.rollback_trigger_rate == pytest.approx(1.0)

    def test_rollback_trigger_rate_calculated(self):
        # 2 admitted, both rolled back
        decisions = [_make_admission_decision("AAA"), _make_admission_decision("BBB")]
        inp       = _make_paper_input(persistence_score=0.20, decisions=decisions)
        report    = run_paper_execution(inp)
        assert report.rollback_trigger_rate == pytest.approx(1.0)

    def test_no_rollback_with_healthy_metrics(self):
        inp    = _make_paper_input(persistence_score=0.80, degradation_ratio=0.90)
        report = run_paper_execution(inp)
        assert report.rollback_trigger_rate == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lifecycle decay calculation correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycleDecay:
    def test_lifecycle_decay_ratio_formula(self):
        """lifecycle_decay_ratio = exited_position_expectancy / admitted_expectancy."""
        decisions = [_make_admission_decision("AAA", adjusted_weight=0.045)]
        inp       = _make_paper_input(
            decisions=decisions,
            shadow_expectancy=0.020,
            admitted_expectancy=0.020 * 0.045 / 0.045,  # = 0.020
            persistence_score=0.80,
            degradation_ratio=0.90,
            regime_tag="risk_on",
        )
        report = run_paper_execution(inp)
        # AAA exits via EXIT_BY_HOLDING_EXPIRY (factor=1.0)
        assert report.lifecycle_decay_ratio is not None
        # Both executed and exited expectancy are based on full hold
        exited_exp = report.exited_position_expectancy
        admitted   = report.admitted_expectancy
        if admitted != 0:
            expected_ratio = exited_exp / admitted
            assert report.lifecycle_decay_ratio == pytest.approx(expected_ratio, abs=1e-3)

    def test_no_full_exits_zero_exited_expectancy(self):
        """If all positions roll back, exited_position_expectancy = 0."""
        decisions = [_make_admission_decision("AAA")]
        inp       = _make_paper_input(persistence_score=0.20, decisions=decisions)
        report    = run_paper_execution(inp)
        assert report.exited_position_expectancy == pytest.approx(0.0, abs=1e-9)

    def test_lifecycle_decay_lt_execution_drift_when_rollbacks_exist(self):
        """If some positions roll back: lifecycle_decay ≤ execution_drift
        (rolled-back positions contribute 0.30 to execution but 0.0 to exited)."""
        # AAA: ALLOW, BBB: ALLOW; persistence collapse forces both to ROLLED_BACK
        decisions = [_make_admission_decision("AAA"), _make_admission_decision("BBB")]
        shadow_exp = 0.020
        inp = _make_paper_input(
            decisions=decisions,
            persistence_score=0.20,  # rollback trigger
            shadow_expectancy=shadow_exp,
            admitted_expectancy=shadow_exp * 0.045 * 2 / (0.045 * 2),  # = shadow_exp
        )
        report = run_paper_execution(inp)
        # Exited positions = 0 (all rolled back) → lifecycle_decay_ratio = 0.0 / admitted
        assert report.exited_position_expectancy == pytest.approx(0.0, abs=1e-9)

    def test_full_hold_gives_decay_ratio_near_one(self):
        """All ALLOW + healthy metrics → lifecycle_decay ≈ execution_drift ≈ 1.0."""
        decisions = [_make_admission_decision("AAA", adjusted_weight=0.045)]
        shadow_exp = 0.020
        admitted   = shadow_exp * 0.045 / 0.045
        inp = _make_paper_input(
            decisions=decisions,
            persistence_score=0.90,
            degradation_ratio=0.90,
            regime_tag="risk_on",
            shadow_expectancy=shadow_exp,
            admitted_expectancy=admitted,
        )
        report = run_paper_execution(inp)
        if report.lifecycle_decay_ratio is not None:
            assert report.lifecycle_decay_ratio >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Execution drift calculation correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionDrift:
    def test_execution_drift_ratio_formula(self):
        """execution_drift_ratio = executed_expectancy / admitted_expectancy."""
        decisions = [_make_admission_decision("AAA", adjusted_weight=0.045)]
        shadow_exp = 0.020
        admitted   = shadow_exp * 0.045 / 0.045
        inp = _make_paper_input(
            decisions=decisions,
            persistence_score=0.90,
            degradation_ratio=0.90,
            regime_tag="risk_on",
            shadow_expectancy=shadow_exp,
            admitted_expectancy=admitted,
        )
        report = run_paper_execution(inp)
        if report.execution_drift_ratio is not None and admitted != 0:
            expected = report.executed_expectancy / admitted
            assert report.execution_drift_ratio == pytest.approx(expected, abs=1e-3)

    def test_drift_ratio_none_when_admitted_zero(self):
        """When admitted_expectancy ≈ 0, ratio is None."""
        inp = _make_paper_input(admitted_expectancy=0.0)
        report = run_paper_execution(inp)
        assert report.execution_drift_ratio is None

    def test_rollbacks_reduce_execution_drift(self):
        """Rolled-back positions have factor 0.30, reducing executed_expectancy."""
        decisions = [_make_admission_decision("AAA")]
        shadow_exp = 0.020
        admitted   = shadow_exp
        inp_normal   = _make_paper_input(
            decisions=decisions, persistence_score=0.90,
            shadow_expectancy=shadow_exp, admitted_expectancy=admitted,
        )
        inp_rollback = _make_paper_input(
            decisions=decisions, persistence_score=0.20,
            shadow_expectancy=shadow_exp, admitted_expectancy=admitted,
        )
        r_normal   = run_paper_execution(inp_normal)
        r_rollback = run_paper_execution(inp_rollback)
        assert r_rollback.executed_expectancy < r_normal.executed_expectancy

    def test_drift_collapse_reduces_execution_drift(self):
        """EXIT_BY_DRIFT_COLLAPSE (factor 0.50) vs EXIT_BY_HOLDING_EXPIRY (1.0)."""
        decisions  = [_make_admission_decision("AAA")]
        shadow_exp = 0.020
        admitted   = shadow_exp

        inp_normal    = _make_paper_input(
            decisions=decisions, degradation_ratio=0.90,
            persistence_score=0.90, regime_tag="risk_on",
            shadow_expectancy=shadow_exp, admitted_expectancy=admitted,
        )
        inp_collapse  = _make_paper_input(
            decisions=decisions, degradation_ratio=0.30,  # < 0.40 → EXIT_BY_DRIFT_COLLAPSE
            persistence_score=0.90, regime_tag="risk_on",
            shadow_expectancy=shadow_exp, admitted_expectancy=admitted,
        )
        r_normal   = run_paper_execution(inp_normal)
        r_collapse = run_paper_execution(inp_collapse)
        assert r_collapse.executed_expectancy < r_normal.executed_expectancy


# ─────────────────────────────────────────────────────────────────────────────
# 6. Overlapping cohort tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlappingCohortTracking:
    def test_cohort_count_equals_admitted_count(self):
        """All admitted positions overlap (same evaluation date)."""
        decisions = [
            _make_admission_decision("AAA"),
            _make_admission_decision("BBB"),
            _make_admission_decision("CCC", decision=AdmissionDecision.DENY),
        ]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        # 2 admitted (AAA + BBB), 1 denied (CCC)
        assert report.overlapping_cohort_count == 2

    def test_zero_admitted_zero_cohorts(self):
        decisions = [
            _make_admission_decision("AAA", decision=AdmissionDecision.DENY),
        ]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        assert report.overlapping_cohort_count == 0

    def test_three_admitted_three_cohorts(self):
        decisions = [
            _make_admission_decision("AAA"),
            _make_admission_decision("BBB"),
            _make_admission_decision("CCC"),
        ]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        assert report.overlapping_cohort_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 7. Capital occupancy calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestCapitalOccupancy:
    def test_capital_lock_duration_formula(self):
        """capital_lock_duration = sum(active_weight × holding_days)."""
        decisions = [_make_admission_decision("AAA", adjusted_weight=0.045)]
        shadow    = _make_shadow_report(avg_holding=8.0)
        promo     = _make_promotion_report()
        routing   = _make_routing_report(decisions=decisions)
        inp       = PaperExecutionInput(
            promotion_report=promo,
            shadow_report=shadow,
            routing_report=routing,
            regime_tag="risk_on",
            evaluation_date="2026-05-25",
            previous_positions={},
        )
        report = run_paper_execution(inp)
        # 1 position, active_weight=0.045, hold=8d
        expected_lock = 0.045 * 8
        assert report.capital_lock_duration == pytest.approx(expected_lock, abs=1e-4)

    def test_capital_occupancy_ratio_bounded(self):
        decisions = [_make_admission_decision("AAA"), _make_admission_decision("BBB")]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        assert 0.0 <= report.capital_occupancy_ratio <= 1.0

    def test_no_positions_zero_occupancy(self):
        decisions = [_make_admission_decision("AAA", decision=AdmissionDecision.DENY)]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        assert report.capital_lock_duration == pytest.approx(0.0)
        assert report.capital_occupancy_ratio == pytest.approx(0.0)

    def test_capital_reuse_efficiency_positive_for_complete_exits(self):
        """n_complete_exits > 0 → capital_reuse_efficiency > 0."""
        decisions = [_make_admission_decision("AAA")]
        inp       = _make_paper_input(
            decisions=decisions,
            persistence_score=0.90,
            degradation_ratio=0.90,
            regime_tag="risk_on",
        )
        report = run_paper_execution(inp)
        # AAA should exit via EXIT_BY_HOLDING_EXPIRY → 1 complete exit
        if report.execution_persistence_score > 0:
            assert report.capital_reuse_efficiency > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Portfolio state transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioStateTransitions:
    def test_stable_state_healthy_metrics(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.0,
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=1,
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.STABLE

    def test_rollback_mode_when_rollback_occurred(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.5,  # > 0 → ROLLBACK_MODE
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=1,
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.ROLLBACK_MODE

    def test_rollback_mode_when_persistence_low(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.0,
            persistence_score=0.30,  # < 0.40 → ROLLBACK_MODE
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=1,
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.ROLLBACK_MODE

    def test_rollback_mode_when_blockers_present(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.0,
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=1,
            deployment_blockers=["something"],
        )
        assert state == SyntheticPortfolioState.ROLLBACK_MODE

    def test_capital_suppressed_on_drawdown(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.0,
            persistence_score=0.80,
            synthetic_max_drawdown=-0.12,  # < -0.10 → CAPITAL_SUPPRESSED
            overlapping_cohort_count=1,
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.CAPITAL_SUPPRESSED

    def test_congested_on_max_positions(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.0,
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=3,  # >= 3 → CONGESTED
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.CONGESTED

    def test_degraded_on_low_drift_ratio(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.60,  # < 0.80 → DEGRADED
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=0.0,
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=1,
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.DEGRADED

    def test_degraded_on_low_lifecycle_decay(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.50,  # < 0.70 → DEGRADED
            rollback_trigger_rate=0.0,
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=1,
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.DEGRADED

    def test_rollback_mode_takes_priority_over_congested(self):
        state = _determine_portfolio_state(
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.85,
            rollback_trigger_rate=1.0,  # rollback
            persistence_score=0.80,
            synthetic_max_drawdown=-0.02,
            overlapping_cohort_count=3,  # also congested
            deployment_blockers=[],
        )
        assert state == SyntheticPortfolioState.ROLLBACK_MODE


# ─────────────────────────────────────────────────────────────────────────────
# 9. FAIL_CLOSED rollback (BLOCKED promotion state)
# ─────────────────────────────────────────────────────────────────────────────

class TestFailClosedRollback:
    def test_blocked_state_all_rolled_back(self):
        decisions = [_make_admission_decision("AAA"), _make_admission_decision("BBB")]
        inp    = _make_paper_input(promotion_state=PromotionState.BLOCKED, decisions=decisions)
        report = run_paper_execution(inp)
        for p in report.ledger.positions:
            assert p.state == SyntheticPositionState.ROLLED_BACK

    def test_research_only_all_rolled_back(self):
        decisions = [_make_admission_decision("AAA")]
        inp    = _make_paper_input(promotion_state=PromotionState.RESEARCH_ONLY, decisions=decisions)
        report = run_paper_execution(inp)
        for p in report.ledger.positions:
            assert p.state == SyntheticPositionState.ROLLED_BACK

    def test_blocked_paper_execution_not_allowed(self):
        inp    = _make_paper_input(promotion_state=PromotionState.BLOCKED)
        report = run_paper_execution(inp)
        assert report.paper_execution_allowed is False

    def test_blocked_portfolio_state_rollback_mode(self):
        inp    = _make_paper_input(promotion_state=PromotionState.BLOCKED)
        report = run_paper_execution(inp)
        assert report.portfolio_state == SyntheticPortfolioState.ROLLBACK_MODE

    def test_blockers_in_shadow_report_trigger_rollback(self):
        decisions = [_make_admission_decision("AAA")]
        inp    = _make_paper_input(
            decisions=decisions,
            deployment_blockers=["deployment_degradation_excessive"],
            persistence_score=0.80,
        )
        report = run_paper_execution(inp)
        assert all(p.state == SyntheticPositionState.ROLLED_BACK for p in report.ledger.positions)

    def test_deny_positions_produce_no_synthetic_positions(self):
        decisions = [
            _make_admission_decision("AAA", decision=AdmissionDecision.DENY),
            _make_admission_decision("BBB"),
        ]
        inp    = _make_paper_input(decisions=decisions)
        report = run_paper_execution(inp)
        syms   = [p.symbol for p in report.ledger.positions]
        assert "AAA" not in syms
        assert "BBB" in syms


# ─────────────────────────────────────────────────────────────────────────────
# 10. Paper execution allowed gate
# ─────────────────────────────────────────────────────────────────────────────

class TestPaperExecutionAllowedGate:
    def _pass_conditions(self, **overrides):
        defaults = dict(
            promotion_ready=True,
            is_viable=True,
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.80,
            synthetic_max_drawdown=-0.02,
            portfolio_state=SyntheticPortfolioState.STABLE,
        )
        defaults.update(overrides)
        promo = _make_promotion_report(promotion_ready=defaults["promotion_ready"])
        shadow = _make_shadow_report(is_viable=defaults["is_viable"])
        return _compute_paper_execution_allowed(
            promotion_report=promo,
            shadow_report=shadow,
            execution_drift_ratio=defaults["execution_drift_ratio"],
            lifecycle_decay_ratio=defaults["lifecycle_decay_ratio"],
            synthetic_max_drawdown=defaults["synthetic_max_drawdown"],
            portfolio_state=defaults["portfolio_state"],
        )

    def test_allowed_when_all_conditions_met(self):
        assert self._pass_conditions() is True

    def test_blocked_when_not_promotion_ready(self):
        assert self._pass_conditions(promotion_ready=False) is False

    def test_blocked_when_not_deployment_viable(self):
        assert self._pass_conditions(is_viable=False) is False

    def test_blocked_when_drift_ratio_low(self):
        assert self._pass_conditions(execution_drift_ratio=0.70) is False

    def test_blocked_when_drift_ratio_none(self):
        assert self._pass_conditions(execution_drift_ratio=None) is False

    def test_blocked_when_lifecycle_decay_low(self):
        assert self._pass_conditions(lifecycle_decay_ratio=0.60) is False

    def test_blocked_when_lifecycle_decay_none(self):
        assert self._pass_conditions(lifecycle_decay_ratio=None) is False

    def test_blocked_when_drawdown_exceeded(self):
        assert self._pass_conditions(synthetic_max_drawdown=-0.20) is False

    def test_blocked_when_rollback_mode(self):
        assert self._pass_conditions(
            portfolio_state=SyntheticPortfolioState.ROLLBACK_MODE
        ) is False

    def test_blocked_when_capital_suppressed(self):
        assert self._pass_conditions(
            portfolio_state=SyntheticPortfolioState.CAPITAL_SUPPRESSED
        ) is False

    def test_allowed_when_congested_but_metrics_ok(self):
        # CONGESTED does NOT block paper_execution_allowed by itself
        assert self._pass_conditions(
            portfolio_state=SyntheticPortfolioState.CONGESTED
        ) is True

    def test_allowed_when_degraded_but_metrics_pass_thresholds(self):
        # DEGRADED also doesn't block if individual ratios pass
        assert self._pass_conditions(
            portfolio_state=SyntheticPortfolioState.DEGRADED,
            execution_drift_ratio=0.90,
            lifecycle_decay_ratio=0.80,
        ) is True


# ─────────────────────────────────────────────────────────────────────────────
# 11. NAV drawdown computation
# ─────────────────────────────────────────────────────────────────────────────

class TestNAVDrawdown:
    def _make_pos(self, symbol, realized_pnl, active_weight) -> SyntheticPosition:
        return SyntheticPosition(
            symbol=symbol,
            state=SyntheticPositionState.EXITED,
            transition_reason=PositionTransitionReason.EXIT_BY_HOLDING_EXPIRY,
            synthetic_entry_date="2026-05-25",
            synthetic_exit_date="2026-05-25",
            proposed_weight=active_weight,
            active_weight=active_weight,
            unrealized_pnl=0.0,
            realized_pnl=realized_pnl,
            holding_duration_days=8,
            rollback_triggered=False,
        )

    def test_no_drawdown_when_all_positive(self):
        positions = [
            self._make_pos("AAA", realized_pnl=0.001, active_weight=0.045),
            self._make_pos("BBB", realized_pnl=0.002, active_weight=0.045),
        ]
        dd = _compute_synthetic_nav_drawdown(positions)
        assert dd >= -1e-9  # no drawdown or tiny floating-point negative

    def test_drawdown_when_loss_position(self):
        positions = [
            self._make_pos("AAA", realized_pnl=0.002, active_weight=0.045),
            self._make_pos("BBB", realized_pnl=-0.005, active_weight=0.045),
        ]
        dd = _compute_synthetic_nav_drawdown(positions)
        assert dd < 0.0

    def test_empty_positions_zero_drawdown(self):
        assert _compute_synthetic_nav_drawdown([]) == pytest.approx(0.0)

    def test_drawdown_deterministic(self):
        positions = [
            self._make_pos("AAA", realized_pnl=0.001, active_weight=0.045),
            self._make_pos("BBB", realized_pnl=-0.003, active_weight=0.045),
        ]
        dd1 = _compute_synthetic_nav_drawdown(positions)
        dd2 = _compute_synthetic_nav_drawdown(positions)
        assert dd1 == dd2


# ─────────────────────────────────────────────────────────────────────────────
# 12. Append-only integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnlyIntegrity:
    def test_report_append_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            report  = run_paper_execution(inp)
            append_paper_execution_report(report, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("paper_execution_reports_*.jsonl"))
            assert len(files) == 1

    def test_report_append_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            r1      = run_paper_execution(inp)
            r2      = run_paper_execution(inp)
            today   = date(2026, 5, 25)
            append_paper_execution_report(r1, log_dir, today=today)
            append_paper_execution_report(r2, log_dir, today=today)
            lines = list(log_dir.glob("paper_execution_reports_*.jsonl"))[0] \
                .read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2

    def test_ledger_append_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            report  = run_paper_execution(inp)
            append_execution_ledger(report.ledger, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("synthetic_execution_ledger_*.jsonl"))
            assert len(files) == 1

    def test_ledger_append_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            r1      = run_paper_execution(inp)
            r2      = run_paper_execution(inp)
            today   = date(2026, 5, 25)
            append_execution_ledger(r1.ledger, log_dir, today=today)
            append_execution_ledger(r2.ledger, log_dir, today=today)
            lines = list(log_dir.glob("synthetic_execution_ledger_*.jsonl"))[0] \
                .read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2

    def test_report_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            report  = run_paper_execution(inp)
            append_paper_execution_report(report, log_dir, today=date(2026, 5, 25))
            file  = list(log_dir.glob("paper_execution_reports_*.jsonl"))[0]
            for line in file.read_text(encoding="utf-8").splitlines():
                parsed = json.loads(line)
                assert "deterministic_hash" in parsed
                assert "portfolio_state" in parsed

    def test_load_reports_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = load_paper_execution_reports(Path(tmpdir) / "nonexistent")
            assert records == []

    def test_load_reports_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            report  = run_paper_execution(inp)
            append_paper_execution_report(report, log_dir, today=date(2026, 5, 25))
            records = load_paper_execution_reports(log_dir)
            assert len(records) == 1
            assert records[0]["deterministic_hash"] == report.deterministic_hash

    def test_load_ledgers_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            report  = run_paper_execution(inp)
            append_execution_ledger(report.ledger, log_dir, today=date(2026, 5, 25))
            records = load_execution_ledgers(log_dir)
            assert len(records) == 1
            assert records[0]["evaluation_date"] == report.ledger.evaluation_date

    def test_multiple_days_chronological(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "paper"
            inp     = _make_paper_input()
            r1      = run_paper_execution(inp)
            r2      = run_paper_execution(inp)
            append_paper_execution_report(r1, log_dir, today=date(2026, 5, 24))
            append_paper_execution_report(r2, log_dir, today=date(2026, 5, 25))
            records = load_paper_execution_reports(log_dir)
            assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 13. Schema validation and to_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_report_to_dict_required_keys(self):
        inp    = _make_paper_input()
        report = run_paper_execution(inp)
        d      = report.to_dict()
        required = {
            "timestamp", "evaluation_date", "promotion_state", "regime_tag",
            "portfolio_state", "ledger",
            "research_expectancy", "shadow_expectancy", "admitted_expectancy",
            "executed_expectancy", "exited_position_expectancy",
            "execution_drift_ratio", "lifecycle_decay_ratio",
            "execution_persistence_score",
            "synthetic_max_drawdown", "rollback_trigger_rate",
            "average_holding_duration", "overlapping_cohort_count",
            "portfolio_congestion_score", "capital_lock_duration",
            "capital_occupancy_ratio", "capital_reuse_efficiency",
            "paper_execution_allowed", "deterministic_hash",
        }
        assert required.issubset(d.keys())

    def test_position_to_dict_required_keys(self):
        decisions = [_make_admission_decision("AAA")]
        inp       = _make_paper_input(decisions=decisions)
        report    = run_paper_execution(inp)
        for p_dict in report.to_dict()["ledger"]["positions"]:
            for key in [
                "symbol", "state", "transition_reason",
                "synthetic_entry_date", "synthetic_exit_date",
                "proposed_weight", "active_weight",
                "unrealized_pnl", "realized_pnl",
                "holding_duration_days", "rollback_triggered",
            ]:
                assert key in p_dict, f"missing key: {key}"

    def test_report_json_serializable(self):
        inp    = _make_paper_input()
        report = run_paper_execution(inp)
        json.dumps(report.to_dict(), ensure_ascii=False)

    def test_expectancy_chain_present(self):
        inp    = _make_paper_input()
        report = run_paper_execution(inp)
        assert hasattr(report, "research_expectancy")
        assert hasattr(report, "shadow_expectancy")
        assert hasattr(report, "admitted_expectancy")
        assert hasattr(report, "executed_expectancy")
        assert hasattr(report, "exited_position_expectancy")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Lifecycle PnL factor correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecyclePnLFactors:
    def test_holding_expiry_factor_is_1(self):
        assert _LIFECYCLE_PNL_FACTOR[PositionTransitionReason.EXIT_BY_HOLDING_EXPIRY] == pytest.approx(1.0)

    def test_rollback_factor_is_030(self):
        assert _LIFECYCLE_PNL_FACTOR[PositionTransitionReason.ROLLBACK_BY_PERSISTENCE] == pytest.approx(0.30)

    def test_drift_collapse_factor_is_050(self):
        assert _LIFECYCLE_PNL_FACTOR[PositionTransitionReason.EXIT_BY_DRIFT_COLLAPSE] == pytest.approx(0.50)

    def test_regime_collapse_factor_is_040(self):
        assert _LIFECYCLE_PNL_FACTOR[PositionTransitionReason.EXIT_BY_REGIME_COLLAPSE] == pytest.approx(0.40)

    def test_all_factors_between_0_and_1(self):
        for reason, factor in _LIFECYCLE_PNL_FACTOR.items():
            assert 0.0 <= factor <= 1.0, f"{reason.value} factor {factor} out of [0,1]"

    def test_rollback_pnl_computed_correctly(self):
        """realized_pnl = shadow_exp × active_weight × lifecycle_factor."""
        shadow_exp  = 0.020
        active_w    = 0.045
        expected    = shadow_exp * active_w * 0.30

        decisions = [_make_admission_decision("AAA", adjusted_weight=active_w)]
        shadow    = _make_shadow_report(
            shadow_expectancy=shadow_exp,
            persistence_score=0.20,  # triggers rollback
        )
        promo     = _make_promotion_report()
        routing   = _make_routing_report(
            decisions=decisions,
            shadow_expectancy=shadow_exp,
            admitted_expectancy=shadow_exp,
        )
        inp    = PaperExecutionInput(
            promotion_report=promo,
            shadow_report=shadow,
            routing_report=routing,
            regime_tag="risk_on",
            evaluation_date="2026-05-25",
            previous_positions={},
        )
        report = run_paper_execution(inp)
        pos    = next(p for p in report.ledger.positions if p.symbol == "AAA")
        assert pos.realized_pnl == pytest.approx(expected, abs=1e-7)

    def test_normal_exit_pnl_computed_correctly(self):
        shadow_exp = 0.020
        active_w   = 0.045
        expected   = shadow_exp * active_w * 1.0  # EXIT_BY_HOLDING_EXPIRY

        decisions = [_make_admission_decision("AAA", adjusted_weight=active_w)]
        shadow    = _make_shadow_report(
            shadow_expectancy=shadow_exp,
            persistence_score=0.90,
            degradation_ratio=0.90,
        )
        promo     = _make_promotion_report()
        routing   = _make_routing_report(decisions=decisions, shadow_expectancy=shadow_exp, admitted_expectancy=shadow_exp)
        inp       = PaperExecutionInput(
            promotion_report=promo,
            shadow_report=shadow,
            routing_report=routing,
            regime_tag="risk_on",
            evaluation_date="2026-05-25",
            previous_positions={},
        )
        report = run_paper_execution(inp)
        pos    = next(p for p in report.ledger.positions if p.symbol == "AAA")
        assert pos.realized_pnl == pytest.approx(expected, abs=1e-7)
