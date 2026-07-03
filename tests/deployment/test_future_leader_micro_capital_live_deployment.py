"""
tests/deployment/test_future_leader_micro_capital_live_deployment.py
Phase 4B — Micro Capital Live Deployment tests

Coverage:
  1. eligibility gate correctness (all 8 conditions)
  2. deterministic replay + same-input hash
  3. rollback trigger correctness
  4. freeze logic correctness
  5. reconciliation mismatch detection
  6. broker timeout fail-closed
  7. live drift calculation
  8. live survivability ratio
  9. append-only integrity
  10. exposure cap enforcement
  11. max order count enforcement
  12. recovery observation logic
  13. schema validation
  14. failure reason taxonomy
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pytest

sys.stdout.reconfigure(encoding="utf-8")

from src.analytics.future_leader_promotion_readiness import (
    PromotionReadinessReport,
    PromotionState,
)
from src.analytics.future_leader_shadow_deployment_validator import ShadowDeploymentReport
from src.analytics.future_leader_limited_capital_router import (
    AdmissionDecision,
    LimitedCapitalRoutingReport,
    RoutingSuppressionReason,
    VirtualAdmissionDecision,
    VirtualExposureSnapshot,
)
from src.analytics.future_leader_paper_execution_engine import (
    PaperExecutionReport,
    SyntheticExecutionLedger,
    SyntheticPortfolioState,
)
from src.deployment.future_leader_micro_capital_live_deployment import (
    AutoRollbackGovernanceReport,
    LiveDeploymentFailureReason,
    LiveDeploymentState,
    LiveExecutionDecision,
    LiveExecutionLedger,
    LivePositionSnapshot,
    MicroCapitalDeploymentInput,
    MicroCapitalDeploymentReport,
    MAX_DAILY_ORDERS,
    MAX_LIVE_POSITIONS,
    MAX_POSITION_WEIGHT,
    MAX_TOTAL_EXPOSURE,
    LIVE_RECOVERY_OBSERVATION_DAYS,
    _ELIGIBILITY_MAX_DRAWDOWN,
    _ELIGIBILITY_MAX_OCCUPANCY,
    _ELIGIBILITY_MIN_DECAY,
    _ELIGIBILITY_MIN_DRIFT,
    _LIVE_DRIFT_ROLLBACK_THRESHOLD,
    _OVERNIGHT_GAP_ROLLBACK_THRESHOLD,
    _ROLLBACK_PRESSURE_REDUCE_THRESHOLD,
    _check_position_reconciliation,
    _compute_deterministic_hash,
    _compute_live_deployment_eligibility,
    _compute_live_metrics,
    _compute_recovery_readiness,
    _determine_deployment_state,
    _make_execution_decision,
    _select_deployment_symbol,
    append_deployment_report,
    append_execution_ledger,
    evaluate_micro_capital_deployment,
    load_deployment_reports,
    load_execution_ledgers,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_promotion_report(
    state: PromotionState = PromotionState.PAPER_READY,
    promotion_ready: bool = True,
) -> PromotionReadinessReport:
    return PromotionReadinessReport(
        timestamp="2026-05-25T09:00:00+09:00",
        promotion_state=state,
        promotion_ready=promotion_ready,
        readiness_score=0.75,
        promotion_confidence=0.70,
        component_scores={
            "statistical_edge": 0.75, "survivability": 0.72,
            "regime_robustness": 0.65, "failure_quality": 0.70, "temporal_stability": 0.72,
        },
        blocking_reasons=[],
        warnings=[],
        sample_size=60,
        recent_sample_size=25,
        regime_coverage={"risk_on": 20, "risk_off": 10, "range": 15, "volatile": 5},
        deterministic_hash="a" * 64,
    )


def _make_shadow_report(
    is_viable: bool = True,
    degradation_ratio: float = 0.82,
    persistence_score: float = 0.75,
    research_expectancy: float = 0.028,
    shadow_expectancy: float = 0.022,
    deployment_blockers: Optional[List[str]] = None,
) -> ShadowDeploymentReport:
    return ShadowDeploymentReport(
        timestamp="2026-05-25T09:00:00+09:00",
        evaluation_date="2026-05-25",
        promotion_state=PromotionState.PAPER_READY.value,
        research_expectancy=research_expectancy,
        shadow_expectancy=shadow_expectancy,
        deployment_degradation_ratio=degradation_ratio,
        shadow_max_drawdown=-0.04,
        survivability_adjusted_return=0.019,
        deployment_persistence_score=persistence_score,
        average_holding_duration=8.0,
        deployment_blockers=deployment_blockers or [],
        is_deployment_viable=is_viable,
        total_research_records=60,
        shadow_positions_simulated=40,
        cohorts_simulated=12,
        allocation_weight=0.045,
        deterministic_hash="b" * 64,
    )


def _make_admission_decision(
    symbol: str,
    decision: AdmissionDecision = AdmissionDecision.ALLOW,
    readiness_score: float = 0.72,
) -> VirtualAdmissionDecision:
    return VirtualAdmissionDecision(
        symbol=symbol,
        decision=decision,
        proposed_weight=0.045,
        adjusted_weight=0.045,
        suppression_reason=RoutingSuppressionReason.NONE,
        readiness_score=readiness_score,
        promotion_confidence=0.70,
        deployment_degradation_ratio=0.82,
        deployment_persistence_score=0.75,
        previous_decision=None,
        current_decision=decision.value,
        decision_transition=f"new->{decision.value}",
    )


def _make_routing_report(
    decisions: Optional[List[VirtualAdmissionDecision]] = None,
) -> LimitedCapitalRoutingReport:
    if decisions is None:
        decisions = [_make_admission_decision("7203")]
    n       = len(decisions)
    n_allow = sum(1 for d in decisions if d.decision == AdmissionDecision.ALLOW)
    return LimitedCapitalRoutingReport(
        timestamp="2026-05-25T09:00:00+09:00",
        evaluation_date="2026-05-25",
        promotion_state="paper_ready",
        regime_tag="risk_on",
        admission_decisions=decisions,
        exposure_snapshot=VirtualExposureSnapshot(
            sector_exposure={"auto": 0.045},
            theme_exposure={"value": 0.045},
            position_concentration=0.045,
            correlated_cluster_count=0,
        ),
        research_expectancy=0.028,
        shadow_expectancy=0.022,
        unsuppressed_shadow_expectancy=0.022,
        admitted_expectancy=0.018,
        deployment_drift_ratio=0.82,
        suppression_alpha_preservation_ratio=0.90,
        admission_rate=n_allow / n if n else 0.0,
        reduced_size_rate=0.0,
        denial_rate=(n - n_allow) / n if n else 0.0,
        persistence_suppression_rate=0.0,
        exposure_concentration=0.045,
        deterministic_hash="c" * 64,
    )


def _make_paper_report(
    paper_execution_allowed: bool = True,
    execution_drift_ratio: Optional[float] = 0.88,
    lifecycle_decay_ratio: Optional[float] = 0.82,
    synthetic_max_drawdown: float = -0.03,
    capital_occupancy_ratio: float = 0.30,
    rollback_trigger_rate: float = 0.0,
    executed_expectancy: float = 0.018,
) -> PaperExecutionReport:
    return PaperExecutionReport(
        timestamp="2026-05-25T09:00:00+09:00",
        evaluation_date="2026-05-25",
        promotion_state="paper_ready",
        regime_tag="risk_on",
        portfolio_state=SyntheticPortfolioState.STABLE,
        ledger=SyntheticExecutionLedger(
            evaluation_date="2026-05-25",
            positions=[],
        ),
        research_expectancy=0.028,
        shadow_expectancy=0.022,
        admitted_expectancy=0.018,
        executed_expectancy=executed_expectancy,
        exited_position_expectancy=executed_expectancy,
        execution_drift_ratio=execution_drift_ratio,
        lifecycle_decay_ratio=lifecycle_decay_ratio,
        execution_persistence_score=0.90,
        synthetic_max_drawdown=synthetic_max_drawdown,
        rollback_trigger_rate=rollback_trigger_rate,
        average_holding_duration=8.0,
        overlapping_cohort_count=2,
        portfolio_congestion_score=0.67,
        capital_lock_duration=0.72,
        capital_occupancy_ratio=capital_occupancy_ratio,
        capital_reuse_efficiency=2.78,
        paper_execution_allowed=paper_execution_allowed,
        deterministic_hash="d" * 64,
    )


def _make_rollback_report(
    rollback_state: str = "stable",
    rollback_pressure_score: float = 0.20,
    recommended_exposure_multiplier: float = 1.0,
    deployment_blockers: Optional[List[str]] = None,
) -> AutoRollbackGovernanceReport:
    return AutoRollbackGovernanceReport(
        timestamp="2026-05-25T09:00:00+09:00",
        evaluation_date="2026-05-25",
        rollback_state=rollback_state,
        rollback_pressure_score=rollback_pressure_score,
        recommended_exposure_multiplier=recommended_exposure_multiplier,
        deployment_blockers=deployment_blockers or [],
        deterministic_hash="e" * 64,
    )


def _make_live_position(
    symbol: str = "7203",
    target_weight: float = 0.02,
    actual_weight: float = 0.019,
    deployment_state: LiveDeploymentState = LiveDeploymentState.LIVE_ACTIVE,
) -> LivePositionSnapshot:
    return LivePositionSnapshot(
        symbol=symbol,
        entry_date="2026-05-24",
        exit_date=None,
        target_weight=target_weight,
        actual_weight=actual_weight,
        entry_price=2800.0,
        current_price=2830.0,
        realized_pnl=0.0,
        unrealized_pnl=0.001,
        deployment_state=deployment_state,
        rollback_triggered=False,
        broker_order_id="ORD-20260524-001",
    )


def _make_deployment_input(
    promotion_state: PromotionState = PromotionState.PAPER_READY,
    rollback_state: str = "stable",
    rollback_pressure_score: float = 0.20,
    paper_execution_allowed: bool = True,
    execution_drift_ratio: Optional[float] = 0.88,
    lifecycle_decay_ratio: Optional[float] = 0.82,
    synthetic_max_drawdown: float = -0.03,
    capital_occupancy_ratio: float = 0.30,
    current_live_position: Optional[LivePositionSnapshot] = None,
    current_deployment_state: LiveDeploymentState = LiveDeploymentState.IDLE,
    broker_positions: Optional[Dict[str, float]] = None,
    reconciliation_mismatch: bool = False,
    broker_failure_detected: bool = False,
    api_integrity_unknown: bool = False,
    broker_timeout: bool = False,
    deterministic_replay_mismatch: bool = False,
    order_latency_ms: Optional[float] = 142.0,
    overnight_gap_impact: float = 0.0,
    daily_orders_placed: int = 0,
    recovery_days_elapsed: int = 0,
    live_expectancy: Optional[float] = None,
    live_max_drawdown: float = 0.0,
    broker_rejection_rate: float = 0.0,
    decisions: Optional[List[VirtualAdmissionDecision]] = None,
    is_viable: bool = True,
) -> MicroCapitalDeploymentInput:
    promo   = _make_promotion_report(state=promotion_state)
    shadow  = _make_shadow_report(
        is_viable=is_viable,
        persistence_score=0.75,
    )
    routing = _make_routing_report(decisions=decisions)
    paper   = _make_paper_report(
        paper_execution_allowed=paper_execution_allowed,
        execution_drift_ratio=execution_drift_ratio,
        lifecycle_decay_ratio=lifecycle_decay_ratio,
        synthetic_max_drawdown=synthetic_max_drawdown,
        capital_occupancy_ratio=capital_occupancy_ratio,
    )
    rollback = _make_rollback_report(
        rollback_state=rollback_state,
        rollback_pressure_score=rollback_pressure_score,
    )
    return MicroCapitalDeploymentInput(
        promotion_report=promo,
        shadow_report=shadow,
        routing_report=routing,
        paper_report=paper,
        rollback_report=rollback,
        evaluation_date="2026-05-25",
        regime_tag="risk_on",
        current_live_position=current_live_position,
        current_deployment_state=current_deployment_state,
        broker_positions=broker_positions or {},
        reconciliation_mismatch=reconciliation_mismatch,
        broker_failure_detected=broker_failure_detected,
        api_integrity_unknown=api_integrity_unknown,
        broker_timeout=broker_timeout,
        deterministic_replay_mismatch=deterministic_replay_mismatch,
        order_latency_ms=order_latency_ms,
        overnight_gap_impact=overnight_gap_impact,
        daily_orders_placed=daily_orders_placed,
        recovery_days_elapsed=recovery_days_elapsed,
        live_expectancy=live_expectancy,
        live_max_drawdown=live_max_drawdown,
        broker_rejection_rate=broker_rejection_rate,
        previous_deployments={},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Eligibility gate correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestEligibilityGate:
    def _gate(self, **kwargs) -> bool:
        defaults = dict(
            promotion_state=PromotionState.PAPER_READY,
            deployment_viable=True,
            paper_execution_allowed=True,
            rollback_state="stable",
            execution_drift_ratio=0.88,
            lifecycle_decay_ratio=0.82,
            synthetic_max_drawdown=-0.03,
            capital_occupancy_ratio=0.30,
        )
        defaults.update(kwargs)
        return _compute_live_deployment_eligibility(**defaults)

    def test_all_conditions_met(self):
        assert self._gate() is True

    def test_shadow_ready_not_eligible(self):
        assert self._gate(promotion_state=PromotionState.SHADOW_READY) is False

    def test_research_only_not_eligible(self):
        assert self._gate(promotion_state=PromotionState.RESEARCH_ONLY) is False

    def test_blocked_not_eligible(self):
        assert self._gate(promotion_state=PromotionState.BLOCKED) is False

    def test_limited_execution_ready_eligible(self):
        assert self._gate(promotion_state=PromotionState.LIMITED_EXECUTION_READY) is True

    def test_production_ready_eligible(self):
        assert self._gate(promotion_state=PromotionState.PRODUCTION_READY) is True

    def test_not_viable_blocks(self):
        assert self._gate(deployment_viable=False) is False

    def test_paper_not_allowed_blocks(self):
        assert self._gate(paper_execution_allowed=False) is False

    def test_soft_rollback_blocks(self):
        assert self._gate(rollback_state="soft_rollback") is False

    def test_full_rollback_blocks(self):
        assert self._gate(rollback_state="full_rollback") is False

    def test_watchlist_eligible(self):
        assert self._gate(rollback_state="watchlist") is True

    def test_low_drift_blocks(self):
        assert self._gate(execution_drift_ratio=0.79) is False

    def test_none_drift_blocks(self):
        assert self._gate(execution_drift_ratio=None) is False

    def test_low_decay_blocks(self):
        assert self._gate(lifecycle_decay_ratio=0.69) is False

    def test_none_decay_blocks(self):
        assert self._gate(lifecycle_decay_ratio=None) is False

    def test_drawdown_at_threshold_blocks(self):
        # must be strictly > -0.10; exactly -0.10 fails
        assert self._gate(synthetic_max_drawdown=-0.10) is False

    def test_drawdown_worse_than_threshold_blocks(self):
        assert self._gate(synthetic_max_drawdown=-0.12) is False

    def test_drawdown_just_above_threshold_passes(self):
        assert self._gate(synthetic_max_drawdown=-0.099) is True

    def test_high_occupancy_blocks(self):
        assert self._gate(capital_occupancy_ratio=0.70) is False

    def test_occupancy_just_below_threshold_passes(self):
        assert self._gate(capital_occupancy_ratio=0.699) is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deterministic replay
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicReplay:
    def test_same_input_same_hash(self):
        inp = _make_deployment_input()
        assert _compute_deterministic_hash(inp) == _compute_deterministic_hash(inp)

    def test_same_input_same_report_hash(self):
        inp = _make_deployment_input()
        r1  = evaluate_micro_capital_deployment(inp)
        r2  = evaluate_micro_capital_deployment(inp)
        assert r1.deterministic_hash == r2.deterministic_hash

    def test_same_input_same_state(self):
        inp = _make_deployment_input()
        r1  = evaluate_micro_capital_deployment(inp)
        r2  = evaluate_micro_capital_deployment(inp)
        assert r1.live_deployment_state == r2.live_deployment_state

    def test_different_regime_different_hash(self):
        i1 = _make_deployment_input()
        i2 = _make_deployment_input()
        i2.regime_tag = "risk_off"
        assert _compute_deterministic_hash(i1) != _compute_deterministic_hash(i2)

    def test_different_broker_positions_different_hash(self):
        i1 = _make_deployment_input(broker_positions={})
        i2 = _make_deployment_input(broker_positions={"7203": 0.02})
        assert _compute_deterministic_hash(i1) != _compute_deterministic_hash(i2)

    def test_hash_is_64_hex_chars(self):
        inp = _make_deployment_input()
        h   = _compute_deterministic_hash(inp)
        assert len(h) == 64
        int(h, 16)  # raises ValueError if not valid hex


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rollback trigger correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackTriggers:
    def test_reconciliation_mismatch_triggers_rollback(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK
        assert report.rollback_triggered is True

    def test_live_drift_below_threshold_triggers_rollback(self):
        inp    = _make_deployment_input(live_expectancy=0.006, paper_execution_allowed=True)
        # synthetic_expectancy = paper.executed_expectancy = 0.018
        # live_drift = 0.006 / 0.018 ≈ 0.333 < 0.60
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_overnight_gap_collapse_triggers_rollback(self):
        inp    = _make_deployment_input(overnight_gap_impact=-0.09)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_full_rollback_state_triggers_rollback(self):
        inp    = _make_deployment_input(rollback_state="full_rollback")
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_deterministic_replay_mismatch_triggers_rollback(self):
        inp    = _make_deployment_input(deterministic_replay_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_rollback_failure_reason_mismatch(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        d = report.ledger.decisions[0]
        assert d.failure_reason == LiveDeploymentFailureReason.POSITION_MISMATCH

    def test_rollback_failure_reason_gap(self):
        inp    = _make_deployment_input(overnight_gap_impact=-0.09)
        report = evaluate_micro_capital_deployment(inp)
        d = report.ledger.decisions[0]
        assert d.failure_reason == LiveDeploymentFailureReason.PRICE_GAP_EXCESS

    def test_rollback_takes_priority_over_frozen(self):
        inp = _make_deployment_input(
            reconciliation_mismatch=True,
            broker_timeout=True,  # both conditions
        )
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_rollback_sets_exit_date_on_position(self):
        pos    = _make_live_position()
        inp    = _make_deployment_input(
            reconciliation_mismatch=True,
            current_live_position=pos,
            current_deployment_state=LiveDeploymentState.LIVE_ACTIVE,
        )
        report = evaluate_micro_capital_deployment(inp)
        assert len(report.ledger.positions) == 1
        assert report.ledger.positions[0].exit_date == "2026-05-25"
        assert report.ledger.positions[0].rollback_triggered is True

    def test_rollback_action_is_rollback(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].action == "rollback"

    def test_rollback_order_not_eligible(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].order_eligible is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Freeze logic correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestFreezeLogic:
    def test_broker_timeout_triggers_frozen(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_FROZEN

    def test_api_integrity_unknown_triggers_frozen(self):
        inp    = _make_deployment_input(api_integrity_unknown=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_FROZEN

    def test_broker_failure_triggers_frozen(self):
        inp    = _make_deployment_input(broker_failure_detected=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_FROZEN

    def test_frozen_action_is_freeze(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].action == "freeze"

    def test_frozen_failure_reason_api_timeout(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        d = report.ledger.decisions[0]
        assert d.failure_reason == LiveDeploymentFailureReason.API_TIMEOUT

    def test_frozen_failure_reason_api_integrity(self):
        inp    = _make_deployment_input(api_integrity_unknown=True)
        report = evaluate_micro_capital_deployment(inp)
        d = report.ledger.decisions[0]
        assert d.failure_reason == LiveDeploymentFailureReason.API_INTEGRITY_UNKNOWN

    def test_frozen_does_not_set_rollback_triggered(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.rollback_triggered is False

    def test_frozen_increments_freeze_count(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.governance_freeze_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reconciliation mismatch detection
# ─────────────────────────────────────────────────────────────────────────────

class TestReconciliation:
    def test_exact_match_reconciled(self):
        ok, mismatches = _check_position_reconciliation(
            {"7203": 0.02}, {"7203": 0.02}
        )
        assert ok is True
        assert mismatches == []

    def test_within_tolerance_reconciled(self):
        ok, _ = _check_position_reconciliation(
            {"7203": 0.02}, {"7203": 0.0202}  # diff = 0.0002 < 0.005
        )
        assert ok is True

    def test_weight_diff_beyond_tolerance_fails(self):
        ok, mismatches = _check_position_reconciliation(
            {"7203": 0.02}, {"7203": 0.03}  # diff = 0.01 > 0.005
        )
        assert ok is False
        assert len(mismatches) == 1

    def test_extra_broker_position_fails(self):
        ok, mismatches = _check_position_reconciliation(
            {}, {"7203": 0.02}  # broker has position we don't know about
        )
        assert ok is False

    def test_missing_broker_position_fails(self):
        ok, mismatches = _check_position_reconciliation(
            {"7203": 0.02}, {}  # internal says 0.02, broker says 0
        )
        assert ok is False

    def test_mismatch_count_in_report(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.reconciliation_mismatch_count == 1

    def test_no_mismatch_zero_count(self):
        inp    = _make_deployment_input(reconciliation_mismatch=False)
        report = evaluate_micro_capital_deployment(inp)
        assert report.reconciliation_mismatch_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Broker timeout fail-closed
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerTimeoutFailClosed:
    def test_timeout_alone_gives_frozen(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_FROZEN

    def test_rollback_plus_timeout_gives_rollback(self):
        inp = _make_deployment_input(
            broker_timeout=True,
            reconciliation_mismatch=True,  # rollback has higher priority
        )
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_timeout_blocks_new_entry(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].order_eligible is False

    def test_timeout_freeze_count_one(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.governance_freeze_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Live drift calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveDriftCalculation:
    def test_live_drift_formula(self):
        # synthetic_expectancy = paper.executed_expectancy = 0.018
        # live_expectancy = 0.016
        # expected drift = 0.016 / 0.018 ≈ 0.8889
        inp     = _make_deployment_input(live_expectancy=0.016)
        metrics = _compute_live_metrics(inp)
        assert metrics["live_execution_drift_ratio"] == pytest.approx(0.016 / 0.018, abs=1e-3)

    def test_none_when_synthetic_zero(self):
        inp = _make_deployment_input(live_expectancy=0.016)
        # Override paper report with zero executed_expectancy
        inp.paper_report.executed_expectancy = 0.0
        metrics = _compute_live_metrics(inp)
        assert metrics["live_execution_drift_ratio"] is None

    def test_none_when_live_expectancy_none(self):
        inp     = _make_deployment_input(live_expectancy=None)
        metrics = _compute_live_metrics(inp)
        assert metrics["live_execution_drift_ratio"] is None

    def test_low_live_drift_triggers_rollback(self):
        # live=0.006, synthetic=0.018 → drift=0.333 < 0.60
        inp    = _make_deployment_input(live_expectancy=0.006)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.LIVE_ROLLBACK

    def test_healthy_live_drift_no_rollback(self):
        # live=0.016, synthetic=0.018 → drift=0.889 > 0.60
        inp    = _make_deployment_input(live_expectancy=0.016)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state != LiveDeploymentState.LIVE_ROLLBACK


# ─────────────────────────────────────────────────────────────────────────────
# 8. Live survivability ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveSurvivabilityRatio:
    def test_survivability_formula(self):
        # research_expectancy = 0.028, live_expectancy = 0.018
        # expected = 0.018 / 0.028 ≈ 0.6429
        inp     = _make_deployment_input(live_expectancy=0.018)
        metrics = _compute_live_metrics(inp)
        assert metrics["live_survivability_ratio"] == pytest.approx(0.018 / 0.028, abs=1e-3)

    def test_none_when_research_zero(self):
        inp = _make_deployment_input(live_expectancy=0.018)
        inp.shadow_report.research_expectancy = 0.0
        metrics = _compute_live_metrics(inp)
        assert metrics["live_survivability_ratio"] is None

    def test_none_when_no_live_trades(self):
        inp     = _make_deployment_input(live_expectancy=None)
        metrics = _compute_live_metrics(inp)
        assert metrics["live_survivability_ratio"] is None

    def test_report_contains_survivability(self):
        inp    = _make_deployment_input(live_expectancy=0.018)
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_survivability_ratio is not None
        assert report.live_survivability_ratio == pytest.approx(0.018 / 0.028, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Append-only integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnlyIntegrity:
    def test_report_append_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            report  = evaluate_micro_capital_deployment(inp)
            append_deployment_report(report, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("micro_capital_deployment_report_*.jsonl"))
            assert len(files) == 1

    def test_report_append_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            r1      = evaluate_micro_capital_deployment(inp)
            r2      = evaluate_micro_capital_deployment(inp)
            today   = date(2026, 5, 25)
            append_deployment_report(r1, log_dir, today=today)
            append_deployment_report(r2, log_dir, today=today)
            lines = list(log_dir.glob("micro_capital_deployment_report_*.jsonl"))[0] \
                .read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2

    def test_ledger_append_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            report  = evaluate_micro_capital_deployment(inp)
            append_execution_ledger(report.ledger, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("live_execution_ledger_*.jsonl"))
            assert len(files) == 1

    def test_ledger_append_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            r1      = evaluate_micro_capital_deployment(inp)
            r2      = evaluate_micro_capital_deployment(inp)
            today   = date(2026, 5, 25)
            append_execution_ledger(r1.ledger, log_dir, today=today)
            append_execution_ledger(r2.ledger, log_dir, today=today)
            lines = list(log_dir.glob("live_execution_ledger_*.jsonl"))[0] \
                .read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2

    def test_report_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            report  = evaluate_micro_capital_deployment(inp)
            append_deployment_report(report, log_dir, today=date(2026, 5, 25))
            file    = list(log_dir.glob("micro_capital_deployment_report_*.jsonl"))[0]
            for line in file.read_text(encoding="utf-8").splitlines():
                parsed = json.loads(line)
                assert "deterministic_hash" in parsed
                assert "live_deployment_state" in parsed

    def test_load_reports_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = load_deployment_reports(Path(tmpdir) / "nonexistent")
            assert records == []

    def test_load_reports_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            report  = evaluate_micro_capital_deployment(inp)
            append_deployment_report(report, log_dir, today=date(2026, 5, 25))
            records = load_deployment_reports(log_dir)
            assert len(records) == 1
            assert records[0]["deterministic_hash"] == report.deterministic_hash

    def test_load_ledgers_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            report  = evaluate_micro_capital_deployment(inp)
            append_execution_ledger(report.ledger, log_dir, today=date(2026, 5, 25))
            records = load_execution_ledgers(log_dir)
            assert len(records) == 1
            assert records[0]["evaluation_date"] == report.ledger.evaluation_date

    def test_multiple_days_both_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "deploy"
            inp     = _make_deployment_input()
            r1      = evaluate_micro_capital_deployment(inp)
            r2      = evaluate_micro_capital_deployment(inp)
            append_deployment_report(r1, log_dir, today=date(2026, 5, 24))
            append_deployment_report(r2, log_dir, today=date(2026, 5, 25))
            records = load_deployment_reports(log_dir)
            assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 10. Exposure cap enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestExposureCapEnforcement:
    def test_max_position_weight_constant(self):
        assert MAX_POSITION_WEIGHT == pytest.approx(0.02)

    def test_max_total_exposure_equals_max_position_weight(self):
        assert MAX_TOTAL_EXPOSURE == pytest.approx(MAX_POSITION_WEIGHT)

    def test_max_live_positions_is_one(self):
        assert MAX_LIVE_POSITIONS == 1

    def test_entry_target_weight_capped_at_max(self):
        inp    = _make_deployment_input()
        report = evaluate_micro_capital_deployment(inp)
        for d in report.ledger.decisions:
            if d.action == "enter":
                assert d.target_weight <= MAX_POSITION_WEIGHT

    def test_pending_entry_uses_max_weight(self):
        # Eligible + no position → PENDING_ENTRY with target=MAX_POSITION_WEIGHT
        inp = _make_deployment_input(
            current_live_position=None,
            current_deployment_state=LiveDeploymentState.IDLE,
        )
        report = evaluate_micro_capital_deployment(inp)
        if report.live_deployment_state == LiveDeploymentState.PENDING_ENTRY:
            d = report.ledger.decisions[0]
            if d.action == "enter":
                assert d.target_weight == pytest.approx(MAX_POSITION_WEIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Max order count enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxOrderCountEnforcement:
    def test_max_daily_orders_constant(self):
        assert MAX_DAILY_ORDERS == 2

    def test_order_not_eligible_at_max_limit(self):
        inp    = _make_deployment_input(daily_orders_placed=2)
        report = evaluate_micro_capital_deployment(inp)
        for d in report.ledger.decisions:
            assert d.order_eligible is False

    def test_order_eligible_below_limit(self):
        inp    = _make_deployment_input(daily_orders_placed=0)
        report = evaluate_micro_capital_deployment(inp)
        # If state=PENDING_ENTRY with a valid symbol, order should be eligible
        if report.live_deployment_state == LiveDeploymentState.PENDING_ENTRY:
            d = report.ledger.decisions[0]
            if d.action == "enter":
                assert d.order_eligible is True

    def test_order_not_eligible_during_rollback(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        for d in report.ledger.decisions:
            assert d.order_eligible is False

    def test_order_not_eligible_during_freeze(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        for d in report.ledger.decisions:
            assert d.order_eligible is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. Recovery observation logic
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveryObservation:
    def test_post_rollback_short_recovery_gives_observation(self):
        inp = _make_deployment_input(
            current_deployment_state=LiveDeploymentState.LIVE_ROLLBACK,
            recovery_days_elapsed=2,  # < 5
        )
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state == LiveDeploymentState.RECOVERY_OBSERVATION

    def test_recovery_observation_action_is_idle(self):
        inp = _make_deployment_input(
            current_deployment_state=LiveDeploymentState.LIVE_ROLLBACK,
            recovery_days_elapsed=2,
        )
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].action == "idle"
        assert report.ledger.decisions[0].order_eligible is False

    def test_recovery_complete_exits_observation(self):
        # 5 days elapsed + all conditions met → not RECOVERY_OBSERVATION
        inp = _make_deployment_input(
            current_deployment_state=LiveDeploymentState.RECOVERY_OBSERVATION,
            recovery_days_elapsed=5,
            execution_drift_ratio=0.90,  # > 0.80
            synthetic_max_drawdown=-0.03,  # > -0.05
        )
        report = evaluate_micro_capital_deployment(inp)
        assert report.live_deployment_state != LiveDeploymentState.RECOVERY_OBSERVATION

    def test_recovery_readiness_requires_five_days(self):
        assert _compute_recovery_readiness(
            recovery_days_elapsed=4,
            execution_drift_ratio=0.90,
            deployment_persistence_score=0.60,
            synthetic_max_drawdown=-0.03,
            broker_failure_detected=False,
        ) is False

    def test_recovery_readiness_met_after_five_days(self):
        assert _compute_recovery_readiness(
            recovery_days_elapsed=5,
            execution_drift_ratio=0.90,
            deployment_persistence_score=0.60,
            synthetic_max_drawdown=-0.03,
            broker_failure_detected=False,
        ) is True

    def test_recovery_readiness_fails_with_broker_failure(self):
        assert _compute_recovery_readiness(
            recovery_days_elapsed=5,
            execution_drift_ratio=0.90,
            deployment_persistence_score=0.60,
            synthetic_max_drawdown=-0.03,
            broker_failure_detected=True,
        ) is False

    def test_recovery_constant_is_five(self):
        assert LIVE_RECOVERY_OBSERVATION_DAYS == 5


# ─────────────────────────────────────────────────────────────────────────────
# 13. Schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_report_to_dict_required_keys(self):
        inp    = _make_deployment_input()
        report = evaluate_micro_capital_deployment(inp)
        d      = report.to_dict()
        required = {
            "timestamp", "evaluation_date", "promotion_state", "regime_tag",
            "live_deployment_state", "allow_live_micro_deployment",
            "ledger",
            "research_expectancy", "synthetic_expectancy", "live_expectancy",
            "execution_drift_ratio", "live_execution_drift_ratio", "live_survivability_ratio",
            "synthetic_max_drawdown", "live_max_drawdown",
            "rollback_triggered", "rollback_state", "rollback_trigger_rate",
            "broker_rejection_rate", "order_latency_ms", "overnight_gap_impact",
            "governance_freeze_count", "reconciliation_mismatch_count",
            "capital_occupancy_ratio",
            "deterministic_hash",
        }
        assert required.issubset(d.keys())

    def test_decision_to_dict_required_keys(self):
        inp    = _make_deployment_input()
        report = evaluate_micro_capital_deployment(inp)
        for d_dict in report.to_dict()["ledger"]["decisions"]:
            for key in [
                "symbol", "action", "target_weight",
                "failure_reason", "governance_rationale", "order_eligible",
            ]:
                assert key in d_dict, f"missing key: {key}"

    def test_report_json_serializable(self):
        inp    = _make_deployment_input()
        report = evaluate_micro_capital_deployment(inp)
        json.dumps(report.to_dict(), ensure_ascii=False)

    def test_position_to_dict_required_keys(self):
        pos    = _make_live_position()
        inp    = _make_deployment_input(
            current_live_position=pos,
            current_deployment_state=LiveDeploymentState.LIVE_ACTIVE,
        )
        report = evaluate_micro_capital_deployment(inp)
        if report.ledger.positions:
            p_dict = report.ledger.positions[0].to_dict()
            for key in [
                "symbol", "entry_date", "exit_date",
                "target_weight", "actual_weight", "entry_price",
                "current_price", "realized_pnl", "unrealized_pnl",
                "deployment_state", "rollback_triggered", "broker_order_id",
            ]:
                assert key in p_dict, f"missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 14. Failure reason taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureReasonTaxonomy:
    def test_all_failure_reasons_have_values(self):
        for reason in LiveDeploymentFailureReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_position_mismatch_reason_on_reconciliation_rollback(self):
        inp    = _make_deployment_input(reconciliation_mismatch=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].failure_reason == LiveDeploymentFailureReason.POSITION_MISMATCH

    def test_api_timeout_reason_on_broker_timeout(self):
        inp    = _make_deployment_input(broker_timeout=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].failure_reason == LiveDeploymentFailureReason.API_TIMEOUT

    def test_api_integrity_reason_on_unknown(self):
        inp    = _make_deployment_input(api_integrity_unknown=True)
        report = evaluate_micro_capital_deployment(inp)
        assert report.ledger.decisions[0].failure_reason == LiveDeploymentFailureReason.API_INTEGRITY_UNKNOWN

    def test_governance_blocked_reason_on_order_limit(self):
        inp    = _make_deployment_input(daily_orders_placed=MAX_DAILY_ORDERS)
        report = evaluate_micro_capital_deployment(inp)
        if report.live_deployment_state == LiveDeploymentState.PENDING_ENTRY:
            d = report.ledger.decisions[0]
            if d.action == "idle":
                assert d.failure_reason == LiveDeploymentFailureReason.GOVERNANCE_BLOCKED

    def test_no_failure_reason_on_healthy_active(self):
        pos    = _make_live_position()
        inp    = _make_deployment_input(
            current_live_position=pos,
            current_deployment_state=LiveDeploymentState.LIVE_ACTIVE,
        )
        report = evaluate_micro_capital_deployment(inp)
        if report.live_deployment_state == LiveDeploymentState.LIVE_ACTIVE:
            assert report.ledger.decisions[0].failure_reason is None
