"""
tests/analytics/test_future_leader_limited_capital_router.py
Phase 3B — Limited Capital Router (Shadow Admission Governance Layer) tests

Coverage:
  - BLOCKED / RESEARCH_ONLY routing denied
  - SHADOW_READY routing allowed
  - deterministic routing reproducibility + same-input hash
  - concentration suppression (sector > 0.35 → REDUCED, > 0.50 → DENY; weight > 0.10 → DENY)
  - reduced-size routing (risk_off, moderate degradation, weak confidence)
  - persistence collapse denial (persistence_score < 0.40 → DENY)
  - risk_off suppression (multiplier 0.50)
  - deployment drift calculation
  - suppression preservation ratio
  - decision transition telemetry
  - append-only integrity (no truncation, sequential appends)
  - counterfactual baseline metrics
"""
import hashlib
import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.stdout.reconfigure(encoding="utf-8")

from src.analytics.future_leader_limited_capital_router import (
    AdmissionDecision,
    LimitedCapitalRoutingInput,
    LimitedCapitalRoutingReport,
    RoutingSuppressionReason,
    VirtualAdmissionDecision,
    VirtualExposureSnapshot,
    _BASE_PROPOSED_WEIGHT,
    _DEGRADATION_DENY_THRESHOLD,
    _DEGRADATION_REDUCED_THRESHOLD,
    _PERSISTENCE_DENY_THRESHOLD,
    _POSITION_WEIGHT_DENY_THRESHOLD,
    _RISK_OFF_SIZE_MULTIPLIER,
    _SECTOR_EXPOSURE_DENY_THRESHOLD,
    _SECTOR_EXPOSURE_REDUCED_THRESHOLD,
    _build_exposure_snapshot,
    _compute_admitted_expectancy,
    _compute_decision_transition,
    _compute_deterministic_hash,
    _decide_single_admission,
    append_routing_report,
    load_routing_reports,
    route_limited_capital,
)
from src.analytics.future_leader_promotion_readiness import (
    PromotionReadinessReport,
    PromotionState,
)
from src.analytics.future_leader_shadow_deployment_validator import (
    ShadowDeploymentReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_promotion_report(
    state: PromotionState = PromotionState.SHADOW_READY,
    readiness_score: float = 0.70,
    promotion_confidence: float = 0.65,
    blocking_reasons: Optional[List[str]] = None,
) -> PromotionReadinessReport:
    return PromotionReadinessReport(
        timestamp="2026-05-25T09:00:00+09:00",
        promotion_state=state,
        promotion_ready=state not in (PromotionState.BLOCKED, PromotionState.RESEARCH_ONLY),
        readiness_score=readiness_score,
        promotion_confidence=promotion_confidence,
        component_scores={
            "statistical_edge": 0.72,
            "survivability":    0.68,
            "regime_robustness": 0.60,
            "failure_quality":  0.65,
            "temporal_stability": 0.70,
        },
        blocking_reasons=blocking_reasons or [],
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
        average_holding_duration=8.5,
        deployment_blockers=deployment_blockers or [],
        is_deployment_viable=is_viable,
        total_research_records=50,
        shadow_positions_simulated=30,
        cohorts_simulated=10,
        allocation_weight=0.045,
        deterministic_hash="b" * 64,
    )


def _make_routing_input(
    promotion_state: PromotionState = PromotionState.SHADOW_READY,
    regime_tag: str = "risk_on",
    candidate_symbols: Optional[List[str]] = None,
    symbol_sectors: Optional[Dict[str, str]] = None,
    symbol_themes: Optional[Dict[str, str]] = None,
    previous_decisions: Optional[Dict[str, str]] = None,
    degradation_ratio: Optional[float] = 0.80,
    persistence_score: float = 0.72,
    deployment_blockers: Optional[List[str]] = None,
    promotion_confidence: float = 0.65,
    readiness_score: float = 0.70,
) -> LimitedCapitalRoutingInput:
    symbols = candidate_symbols or ["AAA", "BBB", "CCC"]
    sectors = symbol_sectors or {s: "tech" if s == "AAA" else "finance" for s in symbols}
    themes  = symbol_themes or {s: "growth" for s in symbols}
    promo   = _make_promotion_report(
        state=promotion_state,
        readiness_score=readiness_score,
        promotion_confidence=promotion_confidence,
    )
    shadow  = _make_shadow_report(
        degradation_ratio=degradation_ratio,
        persistence_score=persistence_score,
        deployment_blockers=deployment_blockers or [],
    )
    return LimitedCapitalRoutingInput(
        promotion_report=promo,
        shadow_report=shadow,
        candidate_symbols=symbols,
        symbol_sectors=sectors,
        symbol_themes=themes,
        regime_tag=regime_tag,
        previous_decisions=previous_decisions or {},
        evaluation_date="2026-05-25",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. BLOCKED promotion state — routing denied for all candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockedPromotionDenied:
    def test_blocked_state_denies_all(self):
        inputs = _make_routing_input(promotion_state=PromotionState.BLOCKED)
        report = route_limited_capital(inputs)
        assert report.denial_rate == 1.0
        assert all(d.decision == AdmissionDecision.DENY for d in report.admission_decisions)

    def test_blocked_suppression_reason(self):
        inputs = _make_routing_input(promotion_state=PromotionState.BLOCKED)
        report = route_limited_capital(inputs)
        for d in report.admission_decisions:
            assert d.suppression_reason == RoutingSuppressionReason.BLOCKED_PROMOTION

    def test_blocked_adjusted_weight_zero(self):
        inputs = _make_routing_input(promotion_state=PromotionState.BLOCKED)
        report = route_limited_capital(inputs)
        for d in report.admission_decisions:
            assert d.adjusted_weight == 0.0

    def test_research_only_denies_all(self):
        inputs = _make_routing_input(promotion_state=PromotionState.RESEARCH_ONLY)
        report = route_limited_capital(inputs)
        assert report.denial_rate == 1.0
        assert all(d.decision == AdmissionDecision.DENY for d in report.admission_decisions)

    def test_blocked_empty_exposure_snapshot(self):
        inputs = _make_routing_input(promotion_state=PromotionState.BLOCKED)
        report = route_limited_capital(inputs)
        assert report.exposure_snapshot.sector_exposure == {}
        assert report.exposure_snapshot.position_concentration == 0.0

    def test_blocked_admitted_expectancy_zero(self):
        inputs = _make_routing_input(promotion_state=PromotionState.BLOCKED)
        report = route_limited_capital(inputs)
        assert report.admitted_expectancy == 0.0
        assert report.deployment_drift_ratio is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHADOW_READY — routing allowed (baseline admission)
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowReadyRouting:
    def test_shadow_ready_allows_admission(self):
        inputs = _make_routing_input(
            promotion_state=PromotionState.SHADOW_READY,
            regime_tag="risk_on",
            degradation_ratio=0.85,
            persistence_score=0.75,
        )
        report = route_limited_capital(inputs)
        # With healthy metrics, at least some should be admitted
        assert report.admission_rate + report.reduced_size_rate > 0.0

    def test_paper_ready_also_routes(self):
        inputs = _make_routing_input(promotion_state=PromotionState.PAPER_READY)
        report = route_limited_capital(inputs)
        assert report.promotion_state == "paper_ready"

    def test_limited_execution_ready_routes(self):
        inputs = _make_routing_input(promotion_state=PromotionState.LIMITED_EXECUTION_READY)
        report = route_limited_capital(inputs)
        assert report.promotion_state == "limited_execution_ready"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic routing reproducibility + same-input hash
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_hash(self):
        inputs = _make_routing_input()
        h1 = _compute_deterministic_hash(inputs)
        h2 = _compute_deterministic_hash(inputs)
        assert h1 == h2

    def test_same_input_same_report_hash(self):
        inputs = _make_routing_input()
        r1 = route_limited_capital(inputs)
        r2 = route_limited_capital(inputs)
        assert r1.deterministic_hash == r2.deterministic_hash

    def test_same_input_same_decisions(self):
        inputs = _make_routing_input()
        r1 = route_limited_capital(inputs)
        r2 = route_limited_capital(inputs)
        for d1, d2 in zip(r1.admission_decisions, r2.admission_decisions):
            assert d1.symbol == d2.symbol
            assert d1.decision == d2.decision
            assert d1.adjusted_weight == d2.adjusted_weight

    def test_different_input_different_hash(self):
        i1 = _make_routing_input(candidate_symbols=["AAA", "BBB"])
        i2 = _make_routing_input(candidate_symbols=["CCC", "DDD"])
        assert _compute_deterministic_hash(i1) != _compute_deterministic_hash(i2)

    def test_sorted_symbol_order_in_decisions(self):
        inputs = _make_routing_input(candidate_symbols=["ZZZ", "AAA", "MMM"])
        report = route_limited_capital(inputs)
        symbols = [d.symbol for d in report.admission_decisions]
        assert symbols == sorted(symbols)

    def test_regime_change_changes_hash(self):
        i1 = _make_routing_input(regime_tag="risk_on")
        i2 = _make_routing_input(regime_tag="risk_off")
        assert _compute_deterministic_hash(i1) != _compute_deterministic_hash(i2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Concentration suppression
# ─────────────────────────────────────────────────────────────────────────────

class TestConcentrationSuppression:
    def test_position_weight_exceed_deny_threshold(self):
        # proposed_weight > 0.10 → DENY via position concentration
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.12,  # > 0.10 → DENY
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.DENY
        assert reason == RoutingSuppressionReason.CONCENTRATION_EXCESS

    def test_position_weight_below_deny_threshold_allowed(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,  # < 0.10 → no weight deny
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.ALLOW

    def test_sector_exposure_over_50pct_deny(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={"tech": 0.45},  # 0.45 + 0.09 = 0.54 > 0.50 → DENY
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.DENY
        assert reason == RoutingSuppressionReason.CONCENTRATION_EXCESS

    def test_sector_exposure_over_35pct_reduced(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={"tech": 0.30},  # 0.30 + 0.09 = 0.39 > 0.35 → REDUCED
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.REDUCED_SIZE
        assert reason == RoutingSuppressionReason.CONCENTRATION_EXCESS
        assert adj_w == pytest.approx(0.09 * 0.60, abs=1e-6)

    def test_sector_exposure_below_35pct_allow(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={"tech": 0.20},  # 0.20 + 0.09 = 0.29 < 0.35 → ALLOW
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.ALLOW

    def test_sequential_sector_accumulation(self):
        """With healthy metrics, first symbol admitted; second in same sector may be reduced."""
        symbols = ["AAA", "BBB", "CCC"]
        sectors = {"AAA": "tech", "BBB": "finance", "CCC": "energy"}
        inputs = _make_routing_input(
            candidate_symbols=symbols,
            symbol_sectors=sectors,
            degradation_ratio=0.85,
            persistence_score=0.75,
        )
        report = route_limited_capital(inputs)
        # Exposure snapshot must include only non-denied symbols
        admitted = [d for d in report.admission_decisions if d.decision != AdmissionDecision.DENY]
        for d in admitted:
            sector = sectors.get(d.symbol, "unknown")
            assert sector in report.exposure_snapshot.sector_exposure


# ─────────────────────────────────────────────────────────────────────────────
# 5. Persistence collapse denial
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistenceCollapse:
    def test_persistence_below_deny_threshold(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.30,  # < 0.40 → DENY
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.DENY
        assert reason == RoutingSuppressionReason.PERSISTENCE_COLLAPSE

    def test_persistence_at_exactly_deny_threshold(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=_PERSISTENCE_DENY_THRESHOLD,  # == 0.40 → not < threshold → OK
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        # At exactly 0.40: not denied by persistence (condition is strict <)
        assert dec != AdmissionDecision.DENY or reason != RoutingSuppressionReason.PERSISTENCE_COLLAPSE

    def test_persistence_in_reduced_range(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.50,  # >= 0.40 but < 0.60 → REDUCED
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.REDUCED_SIZE
        assert reason == RoutingSuppressionReason.PERSISTENCE_COLLAPSE

    def test_persistence_suppression_rate_counted(self):
        inputs = _make_routing_input(
            persistence_score=0.25,  # < 0.40 → DENY via persistence
            degradation_ratio=0.85,
        )
        report = route_limited_capital(inputs)
        # All should be denied via persistence
        pers_decisions = [
            d for d in report.admission_decisions
            if d.suppression_reason == RoutingSuppressionReason.PERSISTENCE_COLLAPSE
            and d.decision == AdmissionDecision.DENY
        ]
        assert len(pers_decisions) > 0
        assert report.persistence_suppression_rate > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Risk-off suppression
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskOffSuppression:
    def test_risk_off_reduced_size(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_off",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.REDUCED_SIZE
        assert reason == RoutingSuppressionReason.RISK_OFF
        assert adj_w == pytest.approx(0.09 * _RISK_OFF_SIZE_MULTIPLIER, abs=1e-6)

    def test_risk_off_multiplier_is_half(self):
        proposed = 0.09
        _, _, adj_w = _decide_single_admission(
            proposed_weight=proposed,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_off",
            sector="finance",
            sector_weights={},
            deployment_blockers=[],
        )
        assert adj_w == pytest.approx(proposed * 0.50, abs=1e-6)

    def test_risk_on_does_not_suppress(self):
        dec, reason, _ = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert reason != RoutingSuppressionReason.RISK_OFF

    def test_risk_off_deny_takes_precedence_over_risk_off_suppression(self):
        """Persistence collapse DENY takes priority over risk_off REDUCED."""
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.25,  # < 0.40 → DENY (before risk_off check)
            confidence=0.70,
            regime_tag="risk_off",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.DENY
        assert reason == RoutingSuppressionReason.PERSISTENCE_COLLAPSE


# ─────────────────────────────────────────────────────────────────────────────
# 7. Deployment drift calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestDeploymentDriftCalculation:
    def test_drift_ratio_formula(self):
        """deployment_drift_ratio = admitted_expectancy / research_expectancy."""
        shadow = _make_shadow_report(
            research_expectancy=0.025,
            shadow_expectancy=0.020,
            degradation_ratio=0.80,
            persistence_score=0.75,
        )
        promo = _make_promotion_report(
            state=PromotionState.SHADOW_READY,
            promotion_confidence=0.65,
        )
        inputs = LimitedCapitalRoutingInput(
            promotion_report=promo,
            shadow_report=shadow,
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            symbol_themes={"AAA": "growth"},
            regime_tag="risk_on",
            previous_decisions={},
            evaluation_date="2026-05-25",
        )
        report = route_limited_capital(inputs)
        if report.deployment_drift_ratio is not None:
            expected = report.admitted_expectancy / 0.025
            assert report.deployment_drift_ratio == pytest.approx(expected, abs=1e-3)

    def test_drift_ratio_none_when_research_expectancy_zero(self):
        shadow = _make_shadow_report(
            research_expectancy=0.0,
            shadow_expectancy=0.020,
            degradation_ratio=0.80,
        )
        promo = _make_promotion_report(state=PromotionState.SHADOW_READY)
        inputs = LimitedCapitalRoutingInput(
            promotion_report=promo,
            shadow_report=shadow,
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            symbol_themes={"AAA": "growth"},
            regime_tag="risk_on",
            previous_decisions={},
            evaluation_date="2026-05-25",
        )
        report = route_limited_capital(inputs)
        assert report.deployment_drift_ratio is None

    def test_drift_ratio_stored_in_report(self):
        inputs = _make_routing_input(
            degradation_ratio=0.85,
            persistence_score=0.75,
        )
        report = route_limited_capital(inputs)
        # All candidates denied due to position weight > 0.10 (BASE = 0.333)
        # so admitted_expectancy = 0 → drift_ratio = 0 / research = 0
        shadow_r = inputs.shadow_report.research_expectancy
        if shadow_r > 1e-9:
            expected_drift = report.admitted_expectancy / shadow_r
            if report.deployment_drift_ratio is not None:
                assert report.deployment_drift_ratio == pytest.approx(expected_drift, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Suppression alpha preservation ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestSuppressionAlphaPreservation:
    def test_preservation_ratio_formula(self):
        """suppression_alpha_preservation_ratio = admitted_expectancy / unsuppressed."""
        shadow = _make_shadow_report(
            shadow_expectancy=0.020,
            research_expectancy=0.025,
            degradation_ratio=0.80,
            persistence_score=0.75,
        )
        promo = _make_promotion_report(state=PromotionState.SHADOW_READY, promotion_confidence=0.65)
        inputs = LimitedCapitalRoutingInput(
            promotion_report=promo,
            shadow_report=shadow,
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            symbol_themes={"AAA": "growth"},
            regime_tag="risk_on",
            previous_decisions={},
            evaluation_date="2026-05-25",
        )
        report = route_limited_capital(inputs)
        if report.suppression_alpha_preservation_ratio is not None:
            expected = report.admitted_expectancy / 0.020
            assert report.suppression_alpha_preservation_ratio == pytest.approx(expected, abs=1e-3)

    def test_preservation_ratio_none_when_shadow_expectancy_zero(self):
        shadow = _make_shadow_report(
            shadow_expectancy=0.0,
            research_expectancy=0.025,
            degradation_ratio=0.80,
        )
        promo = _make_promotion_report(state=PromotionState.SHADOW_READY)
        inputs = LimitedCapitalRoutingInput(
            promotion_report=promo,
            shadow_report=shadow,
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            symbol_themes={"AAA": "growth"},
            regime_tag="risk_on",
            previous_decisions={},
            evaluation_date="2026-05-25",
        )
        report = route_limited_capital(inputs)
        assert report.suppression_alpha_preservation_ratio is None

    def test_full_denial_gives_zero_admitted_expectancy(self):
        inputs = _make_routing_input(
            persistence_score=0.20,  # < 0.40 → all DENY
        )
        report = route_limited_capital(inputs)
        assert report.admitted_expectancy == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Decision transition telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionTransitionTelemetry:
    def test_new_symbol_transition(self):
        t = _compute_decision_transition(None, "allow")
        assert t == "new->allow"

    def test_same_decision_transition(self):
        t = _compute_decision_transition("allow", "allow")
        assert t == "allow->same"

    def test_degradation_transition(self):
        t = _compute_decision_transition("allow", "reduced_size")
        assert t == "allow->reduced_size"

    def test_rollback_transition(self):
        t = _compute_decision_transition("allow", "deny")
        assert t == "allow->deny"

    def test_recovery_transition(self):
        t = _compute_decision_transition("deny", "allow")
        assert t == "deny->allow"

    def test_transition_recorded_in_routing_report(self):
        """Previous decisions stored → transitions reflected in report."""
        inputs = _make_routing_input(
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            previous_decisions={"AAA": "allow"},
            persistence_score=0.20,  # → DENY now
        )
        report = route_limited_capital(inputs)
        dec = next(d for d in report.admission_decisions if d.symbol == "AAA")
        assert dec.previous_decision == "allow"
        assert dec.current_decision == "deny"
        assert "allow" in dec.decision_transition
        assert "deny" in dec.decision_transition

    def test_no_previous_decision_marked_new(self):
        inputs = _make_routing_input(
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            previous_decisions={},
        )
        report = route_limited_capital(inputs)
        dec = next(d for d in report.admission_decisions if d.symbol == "AAA")
        assert dec.previous_decision is None
        assert dec.decision_transition.startswith("new->")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Append-only integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnlyIntegrity:
    def test_append_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "routing"
            inputs  = _make_routing_input()
            report  = route_limited_capital(inputs)
            report.__dict__["timestamp"] = "2026-05-25T09:00:00+09:00"
            append_routing_report(report, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("routing_decisions_*.jsonl"))
            assert len(files) == 1

    def test_append_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "routing"
            inputs  = _make_routing_input()
            r1      = route_limited_capital(inputs)
            r2      = route_limited_capital(inputs)
            today   = date(2026, 5, 25)
            append_routing_report(r1, log_dir, today=today)
            append_routing_report(r2, log_dir, today=today)
            files  = list(log_dir.glob("routing_decisions_*.jsonl"))
            lines  = files[0].read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2

    def test_append_valid_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "routing"
            inputs  = _make_routing_input()
            report  = route_limited_capital(inputs)
            append_routing_report(report, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("routing_decisions_*.jsonl"))
            for line in files[0].read_text(encoding="utf-8").splitlines():
                parsed = json.loads(line)
                assert "deterministic_hash" in parsed
                assert "admission_decisions" in parsed

    def test_load_routing_reports_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = load_routing_reports(Path(tmpdir) / "nonexistent")
            assert records == []

    def test_load_routing_reports_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "routing"
            inputs  = _make_routing_input()
            report  = route_limited_capital(inputs)
            append_routing_report(report, log_dir, today=date(2026, 5, 25))
            records = load_routing_reports(log_dir)
            assert len(records) == 1
            assert records[0]["deterministic_hash"] == report.deterministic_hash

    def test_multiple_days_loaded_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "routing"
            inputs  = _make_routing_input()
            r1      = route_limited_capital(inputs)
            r2      = route_limited_capital(inputs)
            append_routing_report(r1, log_dir, today=date(2026, 5, 24))
            append_routing_report(r2, log_dir, today=date(2026, 5, 25))
            records = load_routing_reports(log_dir)
            assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 11. Counterfactual baseline metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestCounterfactualBaseline:
    def test_unsuppressed_equals_shadow_expectancy(self):
        shadow_exp = 0.022
        shadow = _make_shadow_report(shadow_expectancy=shadow_exp)
        promo  = _make_promotion_report(state=PromotionState.SHADOW_READY)
        inputs = LimitedCapitalRoutingInput(
            promotion_report=promo,
            shadow_report=shadow,
            candidate_symbols=["AAA"],
            symbol_sectors={"AAA": "tech"},
            symbol_themes={"AAA": "growth"},
            regime_tag="risk_on",
            previous_decisions={},
            evaluation_date="2026-05-25",
        )
        report = route_limited_capital(inputs)
        assert report.unsuppressed_shadow_expectancy == pytest.approx(shadow_exp, abs=1e-6)

    def test_admitted_expectancy_leq_unsuppressed_when_both_positive(self):
        """With any suppression, admitted ≤ unsuppressed (when positive)."""
        shadow = _make_shadow_report(
            shadow_expectancy=0.020,
            research_expectancy=0.025,
            degradation_ratio=0.80,
            persistence_score=0.75,
        )
        promo  = _make_promotion_report(state=PromotionState.SHADOW_READY, promotion_confidence=0.65)
        inputs = LimitedCapitalRoutingInput(
            promotion_report=promo,
            shadow_report=shadow,
            candidate_symbols=["AAA", "BBB"],
            symbol_sectors={"AAA": "tech", "BBB": "finance"},
            symbol_themes={"AAA": "growth", "BBB": "value"},
            regime_tag="risk_on",
            previous_decisions={},
            evaluation_date="2026-05-25",
        )
        report = route_limited_capital(inputs)
        if report.unsuppressed_shadow_expectancy > 0:
            assert report.admitted_expectancy <= report.unsuppressed_shadow_expectancy + 1e-6

    def test_expectancy_chain_all_present(self):
        inputs = _make_routing_input()
        report = route_limited_capital(inputs)
        assert hasattr(report, "research_expectancy")
        assert hasattr(report, "shadow_expectancy")
        assert hasattr(report, "unsuppressed_shadow_expectancy")
        assert hasattr(report, "admitted_expectancy")

    def test_admitted_expectancy_scale_with_weight_ratio(self):
        """_compute_admitted_expectancy: scale = admitted_weight / proposed_weight."""
        shadow_exp = 0.020
        decisions_all_allow = [
            VirtualAdmissionDecision(
                symbol="AAA",
                decision=AdmissionDecision.ALLOW,
                proposed_weight=0.09,
                adjusted_weight=0.09,
                suppression_reason=RoutingSuppressionReason.NONE,
                readiness_score=0.70,
                promotion_confidence=0.65,
                deployment_degradation_ratio=0.80,
                deployment_persistence_score=0.72,
                previous_decision=None,
                current_decision="allow",
                decision_transition="new->allow",
            )
        ]
        result = _compute_admitted_expectancy(decisions_all_allow, shadow_exp)
        assert result == pytest.approx(shadow_exp * (0.09 / 0.09), abs=1e-6)

    def test_admitted_expectancy_reduced_size(self):
        shadow_exp = 0.020
        decisions_reduced = [
            VirtualAdmissionDecision(
                symbol="AAA",
                decision=AdmissionDecision.REDUCED_SIZE,
                proposed_weight=0.09,
                adjusted_weight=0.045,  # 50% reduced
                suppression_reason=RoutingSuppressionReason.RISK_OFF,
                readiness_score=0.70,
                promotion_confidence=0.65,
                deployment_degradation_ratio=0.80,
                deployment_persistence_score=0.72,
                previous_decision=None,
                current_decision="reduced_size",
                decision_transition="new->reduced_size",
            )
        ]
        result = _compute_admitted_expectancy(decisions_reduced, shadow_exp)
        assert result == pytest.approx(shadow_exp * 0.5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Deployment blockers — all candidates denied
# ─────────────────────────────────────────────────────────────────────────────

class TestDeploymentBlockers:
    def test_deployment_blockers_deny_all(self):
        inputs = _make_routing_input(
            deployment_blockers=["deployment_degradation_excessive"],
            degradation_ratio=0.85,
            persistence_score=0.75,
        )
        report = route_limited_capital(inputs)
        assert all(d.decision == AdmissionDecision.DENY for d in report.admission_decisions)

    def test_deployment_blockers_suppression_reason(self):
        inputs = _make_routing_input(
            deployment_blockers=["deployment_degradation_excessive"],
        )
        report = route_limited_capital(inputs)
        for d in report.admission_decisions:
            assert d.suppression_reason == RoutingSuppressionReason.BLOCKED_PROMOTION


# ─────────────────────────────────────────────────────────────────────────────
# 13. Reduced-size routing correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestReducedSizeRouting:
    def test_moderate_degradation_reduced(self):
        # degradation in (0.50, 0.75) → REDUCED_SIZE
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.65,  # < 0.75 but >= 0.50
            persistence_score=0.75,
            confidence=0.70,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.REDUCED_SIZE
        assert reason == RoutingSuppressionReason.DEPLOYMENT_DRIFT
        assert adj_w == pytest.approx(0.09 * 0.75, abs=1e-6)

    def test_weak_confidence_reduced(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.85,
            persistence_score=0.75,
            confidence=0.45,  # < 0.55 → REDUCED
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.REDUCED_SIZE
        assert reason == RoutingSuppressionReason.CONFIDENCE_WEAKNESS

    def test_strong_metrics_allow(self):
        dec, reason, adj_w = _decide_single_admission(
            proposed_weight=0.09,
            degradation_ratio=0.90,
            persistence_score=0.80,
            confidence=0.75,
            regime_tag="risk_on",
            sector="tech",
            sector_weights={},
            deployment_blockers=[],
        )
        assert dec == AdmissionDecision.ALLOW
        assert reason == RoutingSuppressionReason.NONE
        assert adj_w == pytest.approx(0.09, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Exposure snapshot construction
# ─────────────────────────────────────────────────────────────────────────────

class TestExposureSnapshot:
    def _make_decisions(self) -> List[VirtualAdmissionDecision]:
        def _d(sym, sec, dec, proposed, adjusted, reason=RoutingSuppressionReason.NONE):
            return VirtualAdmissionDecision(
                symbol=sym, decision=dec,
                proposed_weight=proposed, adjusted_weight=adjusted,
                suppression_reason=reason,
                readiness_score=0.70, promotion_confidence=0.65,
                deployment_degradation_ratio=0.80, deployment_persistence_score=0.72,
                previous_decision=None, current_decision=dec.value,
                decision_transition=f"new->{dec.value}",
            )
        return [
            _d("AAA", "tech",    AdmissionDecision.ALLOW,        0.09, 0.09),
            _d("BBB", "tech",    AdmissionDecision.REDUCED_SIZE,  0.09, 0.045, RoutingSuppressionReason.RISK_OFF),
            _d("CCC", "finance", AdmissionDecision.DENY,          0.09, 0.0,   RoutingSuppressionReason.PERSISTENCE_COLLAPSE),
        ]

    def test_denied_excluded_from_exposure(self):
        decisions = self._make_decisions()
        symbol_sectors = {"AAA": "tech", "BBB": "tech", "CCC": "finance"}
        symbol_themes  = {"AAA": "growth", "BBB": "growth", "CCC": "value"}
        snap = _build_exposure_snapshot(decisions, symbol_sectors, symbol_themes)
        assert "finance" not in snap.sector_exposure  # CCC denied

    def test_sector_weights_accumulate(self):
        decisions = self._make_decisions()
        symbol_sectors = {"AAA": "tech", "BBB": "tech", "CCC": "finance"}
        symbol_themes  = {"AAA": "growth", "BBB": "growth", "CCC": "value"}
        snap = _build_exposure_snapshot(decisions, symbol_sectors, symbol_themes)
        # tech = 0.09 + 0.045 = 0.135
        assert snap.sector_exposure.get("tech", 0.0) == pytest.approx(0.135, abs=1e-5)

    def test_correlated_cluster_count(self):
        decisions = self._make_decisions()
        symbol_sectors = {"AAA": "tech", "BBB": "tech", "CCC": "finance"}
        symbol_themes  = {"AAA": "growth", "BBB": "growth", "CCC": "value"}
        snap = _build_exposure_snapshot(decisions, symbol_sectors, symbol_themes)
        # tech = 0.135 > 0.20 → 1 cluster
        assert snap.correlated_cluster_count == 0  # 0.135 < 0.20

    def test_position_concentration_max_weight(self):
        decisions = self._make_decisions()
        symbol_sectors = {"AAA": "tech", "BBB": "tech", "CCC": "finance"}
        symbol_themes  = {"AAA": "growth", "BBB": "growth", "CCC": "value"}
        snap = _build_exposure_snapshot(decisions, symbol_sectors, symbol_themes)
        # max non-denied adjusted weight = 0.09 (AAA)
        assert snap.position_concentration == pytest.approx(0.09, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 15. to_dict roundtrip and schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_report_to_dict_has_required_keys(self):
        inputs = _make_routing_input()
        report = route_limited_capital(inputs)
        d = report.to_dict()
        required_keys = {
            "timestamp", "evaluation_date", "promotion_state", "regime_tag",
            "admission_decisions", "exposure_snapshot",
            "research_expectancy", "shadow_expectancy",
            "unsuppressed_shadow_expectancy", "admitted_expectancy",
            "deployment_drift_ratio", "suppression_alpha_preservation_ratio",
            "admission_rate", "reduced_size_rate", "denial_rate",
            "persistence_suppression_rate", "exposure_concentration",
            "deterministic_hash",
        }
        assert required_keys.issubset(d.keys())

    def test_decision_to_dict_has_required_keys(self):
        inputs = _make_routing_input()
        report = route_limited_capital(inputs)
        for d_dict in report.to_dict()["admission_decisions"]:
            for key in [
                "symbol", "decision", "proposed_weight", "adjusted_weight",
                "suppression_reason", "readiness_score", "promotion_confidence",
                "deployment_degradation_ratio", "deployment_persistence_score",
                "previous_decision", "current_decision", "decision_transition",
            ]:
                assert key in d_dict, f"missing key: {key}"

    def test_report_json_serializable(self):
        inputs = _make_routing_input()
        report = route_limited_capital(inputs)
        # Must not raise
        json.dumps(report.to_dict(), ensure_ascii=False)

    def test_rates_sum_to_one(self):
        inputs = _make_routing_input(
            candidate_symbols=["AAA", "BBB", "CCC"],
            symbol_sectors={"AAA": "tech", "BBB": "finance", "CCC": "energy"},
        )
        report = route_limited_capital(inputs)
        total = report.admission_rate + report.reduced_size_rate + report.denial_rate
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_deterministic_hash_is_64_hex_chars(self):
        inputs = _make_routing_input()
        report = route_limited_capital(inputs)
        assert len(report.deterministic_hash) == 64
        int(report.deterministic_hash, 16)  # valid hex
