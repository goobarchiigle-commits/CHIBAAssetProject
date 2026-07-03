"""
tests/analytics/test_future_leader_shadow_deployment_validator.py
Phase 3A Shadow Deployment Validator
"""
import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.analytics.future_leader_shadow_deployment_validator import (
    MAX_SHADOW_POSITIONS,
    MAX_WEIGHT,
    _BASE_WEIGHT,
    FORWARD_WINDOW,
    DEGRADATION_THRESHOLD,
    DRAWDOWN_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    SURVIVABILITY_BREAKDOWN_RATIO,
    MIN_SHADOW_COHORTS,
    _DEPLOYMENT_ELIGIBLE_STATES,
    ShadowDeploymentInput,
    ShadowPortfolioSnapshot,
    _filter_clean_survivability,
    _build_failure_lookup,
    _compute_allocation_weight,
    _holding_days_for,
    _simulate_cohort,
    _simulate_full_portfolio,
    _compute_nav_drawdown,
    _compute_shadow_metrics,
    _check_shadow_blockers,
    _compute_deterministic_hash,
    validate_shadow_deployment,
    append_shadow_deployment_report,
    load_shadow_deployment_reports,
    format_shadow_deployment_section,
)
from src.analytics.future_leader_promotion_readiness import (
    PromotionState,
    PromotionReadinessReport,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_promo_report(
    state=PromotionState.SHADOW_READY,
    readiness_score=0.70,
    confidence=0.65,
) -> PromotionReadinessReport:
    return PromotionReadinessReport(
        timestamp="2026-05-25T09:00:00+09:00",
        promotion_state=state,
        promotion_ready=(state not in (PromotionState.BLOCKED, PromotionState.RESEARCH_ONLY)),
        readiness_score=readiness_score,
        promotion_confidence=confidence,
        component_scores={k: 0.70 for k in [
            "statistical_edge", "survivability", "regime_robustness",
            "failure_quality", "temporal_stability",
        ]},
        blocking_reasons=[],
        warnings=[],
        sample_size=30,
        recent_sample_size=10,
        regime_coverage={"risk_on": 10, "risk_off": 5, "range": 10, "volatile": 5},
        deterministic_hash="ab" * 32,
    )


def _surv(symbol="A", obs_date="2025-01-01", score=0.80, fwd10=0.03,
          dq=1.0, insufficient=False) -> dict:
    return {
        "symbol": symbol,
        "observation_date": obs_date,
        "future_leader_score": score,
        "forward_return_10d": fwd10,
        "data_quality_weight": dq,
        "insufficient_forward_data": insufficient,
        "close_price_at_observation": 1000.0,
    }


def _fail(symbol="A", obs_date="2025-01-01", failure_detected=True,
          day_offset=5, dq=1.0) -> dict:
    return {
        "symbol": symbol,
        "observation_date": obs_date,
        "data_quality_weight": dq,
        "failure_detected": failure_detected,
        "failure_day_offset": day_offset,
    }


_DATES = ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]
_BASE_SYMBOLS = [("A", 0.85), ("B", 0.75), ("C", 0.65)]


def _make_valid_surv(dates=None, fwd10=0.02):
    dates = dates or _DATES
    return [
        _surv(sym, d, sc, fwd10)
        for d in dates
        for sym, sc in _BASE_SYMBOLS
    ]


def _make_shadow_input(
    state=PromotionState.SHADOW_READY,
    readiness_score=0.70,
    confidence=0.65,
    surv_records=None,
    fail_records=None,
    eval_date="2026-05-25",
) -> ShadowDeploymentInput:
    return ShadowDeploymentInput(
        promotion_report=_make_promo_report(state, readiness_score, confidence),
        survivability_records=surv_records if surv_records is not None else _make_valid_surv(),
        failure_records=fail_records or [],
        evaluation_date=eval_date,
    )


def _snap(obs_date, portfolio_return, alloc_weight=0.10):
    return ShadowPortfolioSnapshot(
        observation_date=obs_date,
        n_positions=3,
        position_symbols=["A", "B", "C"],
        portfolio_return=portfolio_return,
        portfolio_mae=portfolio_return,
        allocation_weight=alloc_weight,
    )


# ─── TestCleanFilter ──────────────────────────────────────────────────────────

class TestCleanFilter:
    def test_keeps_clean_records(self):
        assert len(_filter_clean_survivability([_surv()])) == 1

    def test_filters_low_dq(self):
        assert _filter_clean_survivability([_surv(dq=0.9)]) == []

    def test_filters_insufficient_forward(self):
        assert _filter_clean_survivability([_surv(insufficient=True)]) == []

    def test_deduplicates_by_symbol_obs_date(self):
        recs = [
            _surv("A", "2025-01-01", fwd10=0.03),
            _surv("A", "2025-01-01", fwd10=0.05),  # duplicate
            _surv("B", "2025-01-01", fwd10=0.02),
        ]
        result = _filter_clean_survivability(recs)
        assert len(result) == 2
        assert {r["symbol"] for r in result} == {"A", "B"}

    def test_excludes_missing_symbol(self):
        recs = [{
            "observation_date": "2025-01-01", "data_quality_weight": 1.0,
            "insufficient_forward_data": False, "forward_return_10d": 0.03,
        }]
        assert _filter_clean_survivability(recs) == []

    def test_excludes_missing_obs_date(self):
        recs = [{
            "symbol": "A", "data_quality_weight": 1.0,
            "insufficient_forward_data": False, "forward_return_10d": 0.03,
        }]
        assert _filter_clean_survivability(recs) == []


# ─── TestFailureLookup ────────────────────────────────────────────────────────

class TestFailureLookup:
    def test_builds_lookup_correctly(self):
        lookup = _build_failure_lookup([_fail("A", "2025-01-01", day_offset=5)])
        assert ("A", "2025-01-01") in lookup

    def test_excludes_low_dq(self):
        assert _build_failure_lookup([_fail(dq=0.5)]) == {}

    def test_last_occurrence_wins(self):
        recs = [_fail("A", "2025-01-01", day_offset=3), _fail("A", "2025-01-01", day_offset=7)]
        lookup = _build_failure_lookup(recs)
        assert lookup[("A", "2025-01-01")]["failure_day_offset"] == 7


# ─── TestAllocationWeight ─────────────────────────────────────────────────────

class TestAllocationWeight:
    def test_formula_below_cap(self):
        report = _make_promo_report(readiness_score=0.30, confidence=0.30)
        weight = _compute_allocation_weight(report)
        expected = _BASE_WEIGHT * 0.30 * 0.30
        assert abs(weight - expected) < 1e-6

    def test_capped_at_max_weight(self):
        report = _make_promo_report(readiness_score=1.0, confidence=1.0)
        assert _compute_allocation_weight(report) == MAX_WEIGHT

    def test_scales_with_readiness(self):
        low  = _compute_allocation_weight(_make_promo_report(readiness_score=0.30, confidence=0.50))
        high = _compute_allocation_weight(_make_promo_report(readiness_score=0.60, confidence=0.50))
        assert high > low

    def test_always_nonnegative(self):
        report = _make_promo_report(readiness_score=0.0, confidence=0.0)
        assert _compute_allocation_weight(report) >= 0.0


# ─── TestHoldingDuration ─────────────────────────────────────────────────────

class TestHoldingDuration:
    def test_uses_failure_record_day_offset(self):
        lookup = _build_failure_lookup([_fail("A", "2025-01-01", failure_detected=True, day_offset=5)])
        assert _holding_days_for("A", "2025-01-01", lookup) == 5

    def test_defaults_to_forward_window_when_no_record(self):
        assert _holding_days_for("Z", "2025-01-01", {}) == FORWARD_WINDOW

    def test_defaults_when_failure_not_detected(self):
        lookup = _build_failure_lookup([_fail("A", "2025-01-01", failure_detected=False, day_offset=3)])
        assert _holding_days_for("A", "2025-01-01", lookup) == FORWARD_WINDOW

    def test_zero_offset_becomes_one(self):
        recs = [{
            "symbol": "A", "observation_date": "2025-01-01",
            "data_quality_weight": 1.0, "failure_detected": True, "failure_day_offset": 0,
        }]
        lookup = _build_failure_lookup(recs)
        assert _holding_days_for("A", "2025-01-01", lookup) == 1


# ─── TestNAVDrawdown ─────────────────────────────────────────────────────────

class TestNAVDrawdown:
    def test_empty_returns_zero(self):
        assert _compute_nav_drawdown([]) == 0.0

    def test_all_positive_returns_zero(self):
        snaps = [_snap(d, 0.05) for d in _DATES]
        assert _compute_nav_drawdown(snaps) == 0.0

    def test_single_loss_cohort(self):
        # alloc=0.10, return=-1.0 → nav=0.90 → drawdown=-10%
        dd = _compute_nav_drawdown([_snap("2025-01-01", -1.0, alloc_weight=0.10)])
        assert abs(dd - (-0.10)) < 1e-4

    def test_consecutive_losses_exceed_threshold(self):
        # Two -1.0 cohorts, alloc=0.10: nav 1.0→0.90→0.81 → max_dd=-19%
        snaps = [
            _snap("2025-01-01", -1.0, alloc_weight=0.10),
            _snap("2025-01-08", -1.0, alloc_weight=0.10),
        ]
        assert _compute_nav_drawdown(snaps) < DRAWDOWN_THRESHOLD

    def test_peak_tracked_correctly_after_gain_then_loss(self):
        # nav: 1.10 → 1.21 (peak) → 0.968
        snaps = [
            _snap("2025-01-01", 1.0, alloc_weight=0.10),   # nav=1.10
            _snap("2025-01-08", 1.0, alloc_weight=0.10),   # nav=1.21 (peak)
            _snap("2025-01-15", -2.0, alloc_weight=0.10),  # nav=1.21*0.80=0.968
        ]
        dd = _compute_nav_drawdown(snaps)
        # (0.968-1.21)/1.21 ≈ -0.20
        assert dd < 0.0
        assert abs(dd - (-0.20)) < 0.01


# ─── TestCohortSimulation ─────────────────────────────────────────────────────

class TestCohortSimulation:
    def test_selects_top_n_by_score(self):
        candidates = [
            _surv("A", "2025-01-01", 0.90, 0.05),
            _surv("B", "2025-01-01", 0.80, 0.04),
            _surv("C", "2025-01-01", 0.70, 0.03),
            _surv("D", "2025-01-01", 0.20, 0.10),  # high return but low score — excluded
        ]
        snap, positions = _simulate_cohort("2025-01-01", candidates, 0.06, 0.70, 0.65, {}, 3)
        assert snap.n_positions == 3
        assert set(snap.position_symbols) == {"A", "B", "C"}

    def test_portfolio_return_is_equal_weighted_mean(self):
        candidates = [
            _surv("A", "2025-01-01", 0.90, 0.06),
            _surv("B", "2025-01-01", 0.80, 0.02),
            _surv("C", "2025-01-01", 0.70, 0.04),
        ]
        snap, _ = _simulate_cohort("2025-01-01", candidates, 0.06, 0.70, 0.65, {}, 3)
        assert abs(snap.portfolio_return - (0.06 + 0.02 + 0.04) / 3) < 1e-6


# ─── TestShadowMetrics ────────────────────────────────────────────────────────

class TestShadowMetrics:
    def _run(self, fwd10=0.03):
        clean_surv = _make_valid_surv(fwd10=fwd10)
        snaps, pos = _simulate_full_portfolio(clean_surv, 0.06, 0.70, 0.65, {})
        return _compute_shadow_metrics(snaps, pos, clean_surv)

    def test_research_expectancy_is_unweighted_mean_of_all_fwd10(self):
        m = self._run(fwd10=0.03)
        assert abs(m["research_expectancy"] - 0.03) < 1e-6

    def test_shadow_expectancy_is_mean_of_cohort_returns(self):
        m = self._run(fwd10=0.04)
        assert abs(m["shadow_expectancy"] - 0.04) < 1e-5

    def test_degradation_ratio_none_when_research_nonpositive(self):
        m = self._run(fwd10=-0.03)
        assert m["deployment_degradation_ratio"] is None


# ─── TestDegradationRatioMath ─────────────────────────────────────────────────

class TestDegradationRatioMath:
    def test_ratio_one_when_shadow_matches_research(self):
        # All 3 records per date selected → shadow = research
        clean_surv = _make_valid_surv(fwd10=0.02)
        snaps, pos = _simulate_full_portfolio(clean_surv, 0.06, 0.70, 0.65, {})
        m = _compute_shadow_metrics(snaps, pos, clean_surv)
        assert abs(m["deployment_degradation_ratio"] - 1.0) < 0.01

    def test_ratio_below_threshold_when_unselected_inflate_research(self):
        # Top-3 (high score) have fwd10=0.01; bottom-2 not selected have fwd10=0.10
        surv = []
        for d in _DATES:
            surv += [
                _surv("A", d, 0.90, 0.01), _surv("B", d, 0.80, 0.01),
                _surv("C", d, 0.70, 0.01), _surv("D", d, 0.20, 0.10),
                _surv("E", d, 0.10, 0.10),
            ]
        snaps, pos = _simulate_full_portfolio(surv, 0.06, 0.70, 0.65, {})
        m = _compute_shadow_metrics(snaps, pos, surv)
        assert m["deployment_degradation_ratio"] < DEGRADATION_THRESHOLD

    def test_ratio_none_when_all_returns_zero(self):
        surv = [_surv("A", d, 0.80, 0.0) for d in _DATES]
        snaps, pos = _simulate_full_portfolio(surv, 0.06, 0.70, 0.65, {})
        m = _compute_shadow_metrics(snaps, pos, surv)
        assert m["deployment_degradation_ratio"] is None


# ─── TestPersistenceScore ─────────────────────────────────────────────────────

class TestPersistenceScore:
    def _run(self, fwd10=0.03):
        clean_surv = _make_valid_surv(fwd10=fwd10)
        snaps, pos = _simulate_full_portfolio(clean_surv, 0.06, 0.70, 0.65, {})
        return _compute_shadow_metrics(snaps, pos, clean_surv)

    def test_persistence_matches_formula(self):
        m = self._run(fwd10=0.03)
        ratio = m["deployment_degradation_ratio"]
        surv_frac = m["survival_fraction"]
        expected = round(max(0.0, ratio * (0.5 + 0.5 * surv_frac)), 4)
        assert abs(m["deployment_persistence_score"] - expected) < 1e-3

    def test_all_positive_returns_max_survival_fraction(self):
        m = self._run(fwd10=0.03)
        assert abs(m["survival_fraction"] - 1.0) < 1e-6

    def test_persistence_nonnegative(self):
        m = self._run(fwd10=-0.01)
        assert m["deployment_persistence_score"] >= 0.0


# ─── TestBlockers ─────────────────────────────────────────────────────────────

class TestBlockers:
    def _metrics(self, ratio=0.80, dd=0.0, shadow_mean=0.02, recent_mean=None):
        rm = recent_mean if recent_mean is not None else shadow_mean
        return {
            "deployment_degradation_ratio": ratio,
            "shadow_max_drawdown": dd,
            "shadow_expectancy": shadow_mean,
            "recent_cohort_mean": rm,
            "research_expectancy": 0.04,
        }

    def test_degradation_blocker_fires_below_threshold(self):
        blockers, _ = _check_shadow_blockers(self._metrics(ratio=0.49), _make_promo_report())
        assert "deployment_degradation_excessive" in blockers

    def test_degradation_blocker_does_not_fire_at_threshold(self):
        blockers, _ = _check_shadow_blockers(self._metrics(ratio=DEGRADATION_THRESHOLD), _make_promo_report())
        assert "deployment_degradation_excessive" not in blockers

    def test_degradation_blocker_fires_when_ratio_none(self):
        blockers, _ = _check_shadow_blockers(self._metrics(ratio=None), _make_promo_report())
        assert "deployment_degradation_excessive" in blockers

    def test_drawdown_blocker_fires_below_threshold(self):
        blockers, _ = _check_shadow_blockers(self._metrics(dd=-0.20), _make_promo_report())
        assert "shadow_drawdown_excessive" in blockers

    def test_drawdown_blocker_does_not_fire_at_threshold(self):
        blockers, _ = _check_shadow_blockers(self._metrics(dd=DRAWDOWN_THRESHOLD), _make_promo_report())
        assert "shadow_drawdown_excessive" not in blockers

    def test_survivability_breakdown_fires(self):
        # recent_mean (0.005) < 0.30 * shadow_mean (0.05) = 0.015
        m = self._metrics(shadow_mean=0.05, recent_mean=0.005)
        blockers, _ = _check_shadow_blockers(m, _make_promo_report())
        assert "survivability_breakdown" in blockers

    def test_confidence_collapse_fires(self):
        blockers, _ = _check_shadow_blockers(self._metrics(), _make_promo_report(confidence=0.30))
        assert "confidence_collapse" in blockers

    def test_no_blockers_when_all_healthy(self):
        blockers, _ = _check_shadow_blockers(
            self._metrics(ratio=0.90, dd=-0.05, shadow_mean=0.02, recent_mean=0.02),
            _make_promo_report(confidence=0.65),
        )
        assert blockers == []

    def test_multiple_blockers_fire_simultaneously(self):
        m = self._metrics(ratio=0.20, dd=-0.30)
        blockers, _ = _check_shadow_blockers(m, _make_promo_report(confidence=0.20))
        assert "deployment_degradation_excessive" in blockers
        assert "shadow_drawdown_excessive" in blockers
        assert "confidence_collapse" in blockers


# ─── TestBlockedIneligibleState ──────────────────────────────────────────────

class TestBlockedIneligibleState:
    def test_research_only_ineligible(self):
        r = validate_shadow_deployment(_make_shadow_input(state=PromotionState.RESEARCH_ONLY))
        assert not r.is_deployment_viable
        assert "ineligible_promotion_state" in r.deployment_blockers

    def test_blocked_state_ineligible(self):
        r = validate_shadow_deployment(_make_shadow_input(state=PromotionState.BLOCKED))
        assert not r.is_deployment_viable
        assert "ineligible_promotion_state" in r.deployment_blockers

    def test_ineligible_report_has_zero_simulation_metadata(self):
        r = validate_shadow_deployment(_make_shadow_input(state=PromotionState.RESEARCH_ONLY))
        assert r.total_research_records == 0
        assert r.cohorts_simulated == 0
        assert r.shadow_positions_simulated == 0

    def test_all_eligible_states_reach_simulation(self):
        for state in _DEPLOYMENT_ELIGIBLE_STATES:
            r = validate_shadow_deployment(_make_shadow_input(state=state))
            assert "ineligible_promotion_state" not in r.deployment_blockers

    def test_shadow_ready_is_eligible(self):
        r = validate_shadow_deployment(_make_shadow_input(state=PromotionState.SHADOW_READY))
        assert "ineligible_promotion_state" not in r.deployment_blockers


# ─── TestInsufficientData ────────────────────────────────────────────────────

class TestInsufficientData:
    def test_zero_clean_records_blocked(self):
        r = validate_shadow_deployment(_make_shadow_input(surv_records=[]))
        assert not r.is_deployment_viable

    def test_dirty_records_filtered_to_insufficient(self):
        surv = [_surv("A", "2025-01-01", dq=0.5), _surv("B", "2025-01-02", dq=0.0)]
        r = validate_shadow_deployment(_make_shadow_input(surv_records=surv))
        assert not r.is_deployment_viable

    def test_insufficient_forward_data_filtered(self):
        surv = [_surv(sym, d, insufficient=True) for d in _DATES for sym in ["A", "B", "C"]]
        r = validate_shadow_deployment(_make_shadow_input(surv_records=surv))
        assert not r.is_deployment_viable

    def test_exactly_min_cohorts_is_sufficient(self):
        # 3 distinct dates, 1 record each = 3 cohorts = MIN_SHADOW_COHORTS
        surv = [_surv("A", d, fwd10=0.03) for d in _DATES[:MIN_SHADOW_COHORTS]]
        r = validate_shadow_deployment(_make_shadow_input(surv_records=surv))
        assert "insufficient_survivability_records" not in r.deployment_blockers


# ─── TestDeterministicHash ────────────────────────────────────────────────────

class TestDeterministicHash:
    def test_same_inputs_same_hash(self):
        inp = _make_shadow_input()
        assert _compute_deterministic_hash(inp) == _compute_deterministic_hash(inp)

    def test_different_eval_date_different_hash(self):
        h1 = _compute_deterministic_hash(_make_shadow_input(eval_date="2026-05-25"))
        h2 = _compute_deterministic_hash(_make_shadow_input(eval_date="2026-05-26"))
        assert h1 != h2

    def test_different_surv_records_different_hash(self):
        h1 = _compute_deterministic_hash(_make_shadow_input(surv_records=_make_valid_surv(fwd10=0.02)))
        h2 = _compute_deterministic_hash(_make_shadow_input(surv_records=_make_valid_surv(fwd10=0.05)))
        assert h1 != h2

    def test_hash_is_64_char_hex(self):
        h = _compute_deterministic_hash(_make_shadow_input())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_report_hash_matches_input_hash(self):
        inp = _make_shadow_input()
        expected = _compute_deterministic_hash(inp)
        assert validate_shadow_deployment(inp).deterministic_hash == expected


# ─── TestDeterministicReplay ─────────────────────────────────────────────────

class TestDeterministicReplay:
    def test_same_inputs_same_report_metrics(self):
        inp = _make_shadow_input()
        r1 = validate_shadow_deployment(inp)
        r2 = validate_shadow_deployment(inp)
        assert r1.research_expectancy == r2.research_expectancy
        assert r1.shadow_expectancy == r2.shadow_expectancy
        assert r1.deployment_degradation_ratio == r2.deployment_degradation_ratio
        assert r1.deployment_persistence_score == r2.deployment_persistence_score
        assert r1.deployment_blockers == r2.deployment_blockers
        assert r1.is_deployment_viable == r2.is_deployment_viable

    def test_allocation_weight_deterministic(self):
        report = _make_promo_report(readiness_score=0.60, confidence=0.55)
        assert _compute_allocation_weight(report) == _compute_allocation_weight(report)


# ─── TestViableDeployment ────────────────────────────────────────────────────

class TestViableDeployment:
    def test_viable_when_all_thresholds_met(self):
        r = validate_shadow_deployment(_make_shadow_input(
            state=PromotionState.SHADOW_READY,
            readiness_score=0.70,
            confidence=0.65,
        ))
        assert r.is_deployment_viable
        assert r.deployment_blockers == []

    def test_not_viable_when_degradation_excessive(self):
        # Top-3 selected have low return; unselected inflate research_expectancy
        surv = []
        for d in _DATES:
            for sym, sc, ret in [("A", 0.90, 0.01), ("B", 0.80, 0.01), ("C", 0.70, 0.01),
                                  ("D", 0.20, 0.15), ("E", 0.10, 0.15)]:
                surv.append(_surv(sym, d, sc, ret))
        r = validate_shadow_deployment(_make_shadow_input(surv_records=surv))
        assert not r.is_deployment_viable
        assert "deployment_degradation_excessive" in r.deployment_blockers

    def test_not_viable_when_confidence_too_low(self):
        r = validate_shadow_deployment(_make_shadow_input(confidence=0.25))
        assert not r.is_deployment_viable
        assert "confidence_collapse" in r.deployment_blockers

    def test_all_required_fields_populated(self):
        r = validate_shadow_deployment(_make_shadow_input())
        assert r.timestamp
        assert r.evaluation_date == "2026-05-25"
        assert r.promotion_state
        assert r.deterministic_hash
        assert r.total_research_records > 0
        assert r.cohorts_simulated > 0
        assert r.shadow_positions_simulated > 0


# ─── TestAppendOnlyJSONL ─────────────────────────────────────────────────────

class TestAppendOnlyJSONL:
    def test_two_appends_produce_two_records(self):
        r = validate_shadow_deployment(_make_shadow_input())
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            append_shadow_deployment_report(r, log_dir, today=date(2026, 5, 25))
            append_shadow_deployment_report(r, log_dir, today=date(2026, 5, 25))
            assert len(load_shadow_deployment_reports(log_dir)) == 2

    def test_appended_record_has_expected_fields(self):
        r = validate_shadow_deployment(_make_shadow_input())
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            append_shadow_deployment_report(r, log_dir, today=date(2026, 5, 25))
            rec = load_shadow_deployment_reports(log_dir)[0]
        for field in [
            "timestamp", "evaluation_date", "promotion_state",
            "research_expectancy", "shadow_expectancy", "deployment_degradation_ratio",
            "shadow_max_drawdown", "deployment_persistence_score",
            "deployment_blockers", "is_deployment_viable", "deterministic_hash",
        ]:
            assert field in rec, f"missing field: {field}"

    def test_append_grows_monotonically(self):
        r = validate_shadow_deployment(_make_shadow_input())
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            append_shadow_deployment_report(r, log_dir, today=date(2026, 5, 25))
            count_before = len(load_shadow_deployment_reports(log_dir))
            append_shadow_deployment_report(r, log_dir, today=date(2026, 5, 25))
            count_after = len(load_shadow_deployment_reports(log_dir))
            assert count_after == count_before + 1

    def test_load_returns_empty_for_missing_dir(self):
        assert load_shadow_deployment_reports(Path("/nonexistent_xyz_9999")) == []

    def test_jsonl_filename_contains_date_suffix(self):
        r = validate_shadow_deployment(_make_shadow_input())
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            append_shadow_deployment_report(r, log_dir, today=date(2026, 5, 25))
            files = list(log_dir.glob("shadow_deployment_*.jsonl"))
            assert len(files) == 1
            assert "20260525" in files[0].name


# ─── TestFormatSection ────────────────────────────────────────────────────────

class TestFormatSection:
    def test_returns_non_empty_string(self):
        r = validate_shadow_deployment(_make_shadow_input())
        assert len(format_shadow_deployment_section(r)) > 0

    def test_contains_key_fields(self):
        r = validate_shadow_deployment(_make_shadow_input())
        section = format_shadow_deployment_section(r)
        assert "SHADOW DEPLOYMENT" in section
        assert "viable" in section.lower()
        assert "degradation" in section.lower()

    def test_shows_blocked_when_not_viable(self):
        r = validate_shadow_deployment(_make_shadow_input(state=PromotionState.RESEARCH_ONLY))
        section = format_shadow_deployment_section(r)
        assert "BLOCKED" in section

    def test_no_blocked_line_when_viable(self):
        r = validate_shadow_deployment(_make_shadow_input())
        assert r.is_deployment_viable  # sanity
        assert "[BLOCKED]" not in format_shadow_deployment_section(r)
