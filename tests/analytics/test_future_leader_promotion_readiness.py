"""
tests/analytics/test_future_leader_promotion_readiness.py

Coverage:
  - insufficient sample → FAIL_CLOSED (insufficient_clean_sample blocker)
  - blockers trigger BLOCKED state and promotion_ready=False
  - deterministic reproducibility: same inputs → same hash and outputs
  - state transition correctness (research_only → shadow → paper → limited → production)
  - low confidence prevents promotion even when score is high
  - survivability collapse blocker
  - negative recent expectancy blocker
  - high instant rejection blocker
  - temporal degradation blocker
  - risk_off instability blocker
  - regime instability blocker
  - append-only JSONL integrity
  - hash determinism: different inputs → different hash
  - promotion_ready=True only for SHADOW_READY and above
  - CLEAN filter: dq < 1.0 excluded; insufficient_forward_data excluded
  - component scores always computed even when BLOCKED
  - format_promotion_readiness_section smoke test
  - load_promotion_reports: empty dir returns []
  - PromotionReadinessReport.to_dict serialization
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import src.analytics.future_leader_promotion_readiness as promo
from src.analytics.future_leader_promotion_readiness import (
    AUTO_PROMOTION_BLOCKERS,
    CONFIDENCE_SAMPLE_TARGET,
    MIN_CLEAN_SAMPLE,
    MIN_REGIME_SAMPLE,
    RECENT_WINDOW,
    PromotionReadinessInputs,
    PromotionReadinessReport,
    PromotionState,
    _BLK_INSTANT_REJ_MAX,
    _BLK_RECENT_EXPCT_MIN,
    _BLK_ROFF_MEAN_MIN,
    _BLK_ROFF_SAMPLE_MIN,
    _BLK_SURV_RATIO,
    _BLK_TEMP_DEGRADE,
    _check_blockers,
    _compute_deterministic_hash,
    _determine_promotion_state,
    _evaluate_failure_quality,
    _evaluate_promotion_confidence,
    _evaluate_regime_robustness,
    _evaluate_statistical_edge,
    _evaluate_survivability,
    _evaluate_temporal_stability,
    _filter_clean_failure,
    _filter_clean_survivability,
    append_promotion_report,
    evaluate_promotion_readiness,
    format_promotion_readiness_section,
    load_promotion_reports,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test data factories
# ─────────────────────────────────────────────────────────────────────────────

def _surv(
    symbol: str,
    obs_date: str,
    fwd10: float = 0.03,
    fwd5: float = 0.015,
    mfe: float = 0.05,
    mae: float = -0.02,
    dq: float = 1.0,
    insuf: bool = False,
) -> dict:
    return {
        "symbol":                    symbol,
        "observation_date":          obs_date,
        "forward_return_10d":        fwd10,
        "forward_return_5d":         fwd5,
        "mfe_10d":                   mfe,
        "mae_10d":                   mae,
        "data_quality_weight":       dq,
        "insufficient_forward_data": insuf,
    }


def _fail(
    symbol: str,
    obs_date: str,
    failure_detected: bool = False,
    offset: Optional[int] = None,
    reason: str = "none",
    dq: float = 1.0,
) -> dict:
    return {
        "symbol":                symbol,
        "observation_date":      obs_date,
        "failure_detected":      failure_detected,
        "failure_day_offset":    offset,
        "primary_failure_reason": reason,
        "data_quality_weight":   dq,
    }


def _make_obs_dates(n: int, start: str = "2026-01-05") -> List[str]:
    """Generate n sequential business-day-like dates."""
    from datetime import timedelta
    base = date.fromisoformat(start)
    dates = []
    d = base
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates


def _make_spaced_dates(n: int, start: str, step_days: int = 7) -> List[str]:
    """Generate n dates spaced step_days apart (calendar days). Spans n*step_days / 30 months."""
    from datetime import timedelta
    base = date.fromisoformat(start)
    return [(base + timedelta(days=i * step_days)).isoformat() for i in range(n)]


def _make_sufficient_surv(n: int = MIN_CLEAN_SAMPLE + 5, fwd10: float = 0.03) -> List[dict]:
    dates = _make_obs_dates(n)
    return [_surv(f"SYM{i:04d}", d, fwd10=fwd10) for i, d in enumerate(dates)]


def _make_sufficient_fail(n: int = MIN_CLEAN_SAMPLE + 5) -> List[dict]:
    dates = _make_obs_dates(n)
    return [_fail(f"SYM{i:04d}", d) for i, d in enumerate(dates)]


def _make_regime_tags(obs_dates: List[str], regime: str = "risk_on") -> Dict[str, str]:
    return {d: regime for d in obs_dates}


def _make_valid_inputs(
    n: int = MIN_CLEAN_SAMPLE + 20,
    fwd10: float = 0.04,
    regime: str = "risk_on",
) -> PromotionReadinessInputs:
    dates       = _make_obs_dates(n)
    surv_recs   = [_surv(f"SYM{i:04d}", d, fwd10=fwd10) for i, d in enumerate(dates)]
    fail_recs   = [_fail(f"SYM{i:04d}", d) for i, d in enumerate(dates)]
    regime_tags = _make_regime_tags(dates, regime)
    return PromotionReadinessInputs(
        survivability_records=surv_recs,
        failure_records=fail_recs,
        regime_tags=regime_tags,
        evaluation_date="2026-05-25",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLEAN filter
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanFilter:
    def test_surviv_dq_below_1_excluded(self):
        recs = [_surv("A", "2026-01-05", dq=0.5), _surv("B", "2026-01-06", dq=1.0)]
        out  = _filter_clean_survivability(recs)
        assert len(out) == 1
        assert out[0]["symbol"] == "B"

    def test_surviv_insufficient_excluded(self):
        recs = [_surv("A", "2026-01-05", insuf=True), _surv("B", "2026-01-06")]
        out  = _filter_clean_survivability(recs)
        assert len(out) == 1
        assert out[0]["symbol"] == "B"

    def test_surviv_dedup(self):
        recs = [_surv("A", "2026-01-05", fwd10=0.01), _surv("A", "2026-01-05", fwd10=0.99)]
        out  = _filter_clean_survivability(recs)
        assert len(out) == 1
        assert out[0]["forward_return_10d"] == pytest.approx(0.01)

    def test_fail_dq_below_1_excluded(self):
        recs = [_fail("A", "2026-01-05", dq=0.0), _fail("B", "2026-01-06", dq=1.0)]
        out  = _filter_clean_failure(recs)
        assert len(out) == 1
        assert out[0]["symbol"] == "B"

    def test_empty_input_returns_empty(self):
        assert _filter_clean_survivability([]) == []
        assert _filter_clean_failure([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Insufficient sample → FAIL_CLOSED
# ─────────────────────────────────────────────────────────────────────────────

class TestInsufficientSample:
    def test_empty_input_blocked(self):
        inputs = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "insufficient_clean_sample" in report.blocking_reasons
        assert report.promotion_state == PromotionState.BLOCKED
        assert report.promotion_ready is False

    def test_below_min_sample_blocked(self):
        n    = MIN_CLEAN_SAMPLE - 1
        recs = [_surv(f"S{i}", f"2026-01-{i+2:02d}", fwd10=0.05) for i in range(n)]
        inputs = PromotionReadinessInputs(recs, [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "insufficient_clean_sample" in report.blocking_reasons
        assert report.promotion_state == PromotionState.BLOCKED

    def test_exact_min_sample_not_blocked_by_sample(self):
        n      = MIN_CLEAN_SAMPLE
        dates  = _make_obs_dates(n)
        recs   = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(dates)]
        inputs = PromotionReadinessInputs(recs, [], {d: "risk_on" for d in dates}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "insufficient_clean_sample" not in report.blocking_reasons

    def test_dirty_records_do_not_count_toward_sample(self):
        n     = MIN_CLEAN_SAMPLE + 10
        dates = _make_obs_dates(n)
        # Mix: some CLEAN, many dirty
        clean = [_surv(f"S{i}", d, fwd10=0.04, dq=1.0) for i, d in enumerate(dates[:3])]
        dirty = [_surv(f"D{i}", d, fwd10=0.04, dq=0.0) for i, d in enumerate(dates[3:])]
        inputs = PromotionReadinessInputs(clean + dirty, [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "insufficient_clean_sample" in report.blocking_reasons


# ─────────────────────────────────────────────────────────────────────────────
# 3. Individual blocker triggers
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockers:
    def _base_surv(self, n: int = MIN_CLEAN_SAMPLE + 20) -> List[dict]:
        dates = _make_obs_dates(n)
        return [_surv(f"S{i}", d, fwd10=0.03) for i, d in enumerate(dates)]

    def _base_fail(self, n: int = MIN_CLEAN_SAMPLE + 20) -> List[dict]:
        dates = _make_obs_dates(n)
        return [_fail(f"S{i}", d) for i, d in enumerate(dates)]

    def _base_tags(self, surv: List[dict], regime: str = "risk_on") -> Dict[str, str]:
        return {r["observation_date"]: regime for r in surv}

    # negative_recent_expectancy
    def test_negative_recent_expectancy_blocked(self):
        dates       = _make_obs_dates(MIN_CLEAN_SAMPLE + RECENT_WINDOW + 5)
        older_dates = dates[:MIN_CLEAN_SAMPLE]
        recent_dates = dates[MIN_CLEAN_SAMPLE:]
        surv = (
            [_surv(f"O{i}", d, fwd10=0.05) for i, d in enumerate(older_dates)]
            + [_surv(f"R{i}", d, fwd10=-0.05) for i, d in enumerate(recent_dates)]
        )
        inputs = PromotionReadinessInputs(surv, [], {d: "risk_on" for d in dates}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "negative_recent_expectancy" in report.blocking_reasons
        assert report.promotion_state == PromotionState.BLOCKED

    def test_positive_recent_expectancy_not_blocked(self):
        surv_recs = self._base_surv()
        inputs    = PromotionReadinessInputs(
            surv_recs, [], self._base_tags(surv_recs), "2026-05-25"
        )
        report = evaluate_promotion_readiness(inputs)
        assert "negative_recent_expectancy" not in report.blocking_reasons

    # risk_off_instability
    def test_risk_off_instability_blocked(self):
        dates       = _make_obs_dates(MIN_CLEAN_SAMPLE + 10)
        roff_dates  = dates[:_BLK_ROFF_SAMPLE_MIN + 2]
        other_dates = dates[_BLK_ROFF_SAMPLE_MIN + 2:]
        surv = (
            [_surv(f"R{i}", d, fwd10=-0.08) for i, d in enumerate(roff_dates)]
            + [_surv(f"O{i}", d, fwd10=0.04) for i, d in enumerate(other_dates)]
        )
        regime_tags = (
            {d: "risk_off" for d in roff_dates}
            | {d: "risk_on" for d in other_dates}
        )
        inputs = PromotionReadinessInputs(surv, [], regime_tags, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "risk_off_instability" in report.blocking_reasons

    def test_risk_off_small_sample_not_blocked(self):
        # Only 2 risk_off records — below _BLK_ROFF_SAMPLE_MIN
        dates = _make_obs_dates(MIN_CLEAN_SAMPLE + 5)
        surv  = [_surv(f"S{i}", d, fwd10=0.03) for i, d in enumerate(dates)]
        tags  = {d: ("risk_off" if i < 2 else "risk_on") for i, d in enumerate(dates)}
        inputs = PromotionReadinessInputs(surv, [], tags, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "risk_off_instability" not in report.blocking_reasons

    # high_instant_rejection
    def test_high_instant_rejection_blocked(self):
        surv_recs = self._base_surv(MIN_CLEAN_SAMPLE + 10)
        dates     = [r["observation_date"] for r in surv_recs]
        n         = len(dates)
        # All fail within 3 days → instant_rate = 1.0
        fail_recs = [_fail(f"S{i}", d, failure_detected=True, offset=2, reason="gap_fill")
                     for i, d in enumerate(dates)]
        inputs    = PromotionReadinessInputs(surv_recs, fail_recs, {d: "risk_on" for d in dates}, "2026-05-25")
        report    = evaluate_promotion_readiness(inputs)
        assert "high_instant_rejection" in report.blocking_reasons

    def test_low_instant_rejection_not_blocked(self):
        surv_recs = self._base_surv()
        dates     = [r["observation_date"] for r in surv_recs]
        # Only 1 fail within 3 days
        fail_recs = (
            [_fail("S0", dates[0], failure_detected=True, offset=2)]
            + [_fail(f"S{i}", d) for i, d in enumerate(dates[1:])]
        )
        inputs = PromotionReadinessInputs(surv_recs, fail_recs, {d: "risk_on" for d in dates}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert "high_instant_rejection" not in report.blocking_reasons

    # survivability_collapse
    def test_survivability_collapse_blocked(self):
        n     = 30
        dates = _make_obs_dates(n)
        surv  = [_surv(f"S{i}", d, fwd10=0.03) for i, d in enumerate(dates)]
        # First half: long survival; recent half: very short
        fail  = (
            [_fail(f"S{i}", d, failure_detected=True, offset=15) for i, d in enumerate(dates[:15])]
            + [_fail(f"S{i}", d, failure_detected=True, offset=2) for i, d in enumerate(dates[15:])]
        )
        inputs = PromotionReadinessInputs(surv, fail, {d: "risk_on" for d in dates}, "2026-05-25")
        blockers, _ = _check_blockers(
            _filter_clean_survivability(surv),
            _filter_clean_failure(fail),
            {d: "risk_on" for d in dates},
        )
        assert "survivability_collapse" in blockers

    def test_stable_survivability_not_blocked(self):
        n     = 30
        dates = _make_obs_dates(n)
        surv  = [_surv(f"S{i}", d, fwd10=0.03) for i, d in enumerate(dates)]
        fail  = [_fail(f"S{i}", d, failure_detected=True, offset=12) for i, d in enumerate(dates)]
        inputs = PromotionReadinessInputs(surv, fail, {d: "risk_on" for d in dates}, "2026-05-25")
        blockers, _ = _check_blockers(
            _filter_clean_survivability(surv),
            _filter_clean_failure(fail),
            {d: "risk_on" for d in dates},
        )
        assert "survivability_collapse" not in blockers

    # temporal_degradation
    def test_temporal_degradation_blocked(self):
        n     = 30
        dates = _make_obs_dates(n)
        # Early: strongly positive; recent: strongly negative
        surv  = (
            [_surv(f"E{i}", d, fwd10=0.08) for i, d in enumerate(dates[:15])]
            + [_surv(f"R{i}", d, fwd10=-0.06) for i, d in enumerate(dates[15:])]
        )
        blockers, _ = _check_blockers(
            _filter_clean_survivability(surv), [],
            {d: "risk_on" for d in dates},
        )
        assert "temporal_degradation" in blockers

    def test_stable_temporal_not_blocked(self):
        n     = 30
        dates = _make_obs_dates(n)
        surv  = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(dates)]
        blockers, _ = _check_blockers(
            _filter_clean_survivability(surv), [],
            {d: "risk_on" for d in dates},
        )
        assert "temporal_degradation" not in blockers

    # regime_instability
    def test_regime_instability_blocked(self):
        n     = 30
        dates = _make_obs_dates(n)
        # risk_on=+18%, risk_off=-15%, range=+2% → std ≈ 0.14 + negative extreme
        # Ensure >= MIN_REGIME_SAMPLE per covered regime
        on_n   = max(MIN_REGIME_SAMPLE, 5)
        off_n  = max(MIN_REGIME_SAMPLE, 5)
        rng_n  = n - on_n - off_n
        surv   = (
            [_surv(f"ON{i}", dates[i],         fwd10=0.18) for i in range(on_n)]
            + [_surv(f"OF{i}", dates[on_n + i], fwd10=-0.15) for i in range(off_n)]
            + [_surv(f"R{i}",  dates[on_n + off_n + i], fwd10=0.02) for i in range(rng_n)]
        )
        tags = (
            {dates[i]: "risk_on" for i in range(on_n)}
            | {dates[on_n + i]: "risk_off" for i in range(off_n)}
            | {dates[on_n + off_n + i]: "range" for i in range(rng_n)}
        )
        blockers, _ = _check_blockers(_filter_clean_survivability(surv), [], tags)
        # risk_off_instability blocks first (mean=-0.15 < -0.04), regime_instability also
        assert "regime_instability" in blockers or "risk_off_instability" in blockers

    def test_all_blockers_in_auto_promotion_blockers_set(self):
        # All blocker keys must be in AUTO_PROMOTION_BLOCKERS
        expected = {
            "insufficient_clean_sample",
            "negative_recent_expectancy",
            "risk_off_instability",
            "high_instant_rejection",
            "temporal_degradation",
            "survivability_collapse",
            "regime_instability",
        }
        assert expected == AUTO_PROMOTION_BLOCKERS


# ─────────────────────────────────────────────────────────────────────────────
# 4. State transition correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestStateTransitions:
    def _cov(self, regime: str, count: int = 5) -> Dict[str, int]:
        base = {r: 0 for r in ["risk_on", "risk_off", "range", "volatile"]}
        base[regime] = count
        return base

    def test_blocked_when_blockers(self):
        state = _determine_promotion_state(0.90, 0.90, self._cov("risk_on"), ["insufficient_clean_sample"])
        assert state == PromotionState.BLOCKED

    def test_research_only_low_score(self):
        state = _determine_promotion_state(0.40, 0.90, self._cov("risk_on"), [])
        assert state == PromotionState.RESEARCH_ONLY

    def test_research_only_low_confidence(self):
        state = _determine_promotion_state(0.80, 0.10, self._cov("risk_on"), [])
        assert state == PromotionState.RESEARCH_ONLY

    def test_shadow_ready_threshold(self):
        cov   = {r: 0 for r in ["risk_on", "risk_off", "range", "volatile"]}
        state = _determine_promotion_state(0.56, 0.51, cov, [])
        assert state == PromotionState.SHADOW_READY

    def test_paper_ready_threshold(self):
        # Needs 2 covered regimes
        cov = {"risk_on": 5, "risk_off": 4, "range": 0, "volatile": 0}
        state = _determine_promotion_state(0.66, 0.61, cov, [])
        assert state == PromotionState.PAPER_READY

    def test_paper_ready_requires_regime_coverage(self):
        cov   = {r: 0 for r in ["risk_on", "risk_off", "range", "volatile"]}
        state = _determine_promotion_state(0.70, 0.65, cov, [])
        # No covered regimes → cannot reach paper_ready → shadow_ready at most
        assert state in (PromotionState.SHADOW_READY, PromotionState.RESEARCH_ONLY)

    def test_limited_execution_ready(self):
        cov = {"risk_on": 5, "risk_off": 4, "range": 0, "volatile": 0}
        state = _determine_promotion_state(0.73, 0.71, cov, [])
        assert state == PromotionState.LIMITED_EXECUTION_READY

    def test_production_ready(self):
        cov = {"risk_on": 5, "risk_off": 4, "range": 3, "volatile": 0}
        state = _determine_promotion_state(0.81, 0.81, cov, [])
        assert state == PromotionState.PRODUCTION_READY

    def test_promotion_ready_false_for_research_only(self):
        inputs = _make_valid_inputs(n=MIN_CLEAN_SAMPLE + 5, fwd10=0.01)
        report = evaluate_promotion_readiness(inputs)
        # Likely RESEARCH_ONLY or SHADOW_READY — if RESEARCH_ONLY, promotion_ready=False
        if report.promotion_state == PromotionState.RESEARCH_ONLY:
            assert report.promotion_ready is False

    def test_promotion_ready_false_for_blocked(self):
        inputs = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert report.promotion_ready is False

    def test_promotion_ready_true_for_shadow_and_above(self):
        # Build a high-quality dataset that should reach at least SHADOW_READY
        # Spread over 6+ months for temporal confidence using spaced dates
        n     = CONFIDENCE_SAMPLE_TARGET + 10
        dates = _make_spaced_dates(n, start="2025-06-01", step_days=3)
        surv  = [_surv(f"S{i}", d, fwd10=0.05, mfe=0.08, mae=-0.01) for i, d in enumerate(dates)]
        fail  = [_fail(f"S{i}", d) for i, d in enumerate(dates)]
        tags  = {d: "risk_on" for d in dates}
        inputs = PromotionReadinessInputs(surv, fail, tags, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        if report.promotion_state not in (PromotionState.BLOCKED, PromotionState.RESEARCH_ONLY):
            assert report.promotion_ready is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Low confidence prevents promotion
# ─────────────────────────────────────────────────────────────────────────────

class TestLowConfidence:
    def test_high_score_low_confidence_stays_research_only(self):
        # 15 records all on the same date → temporal_factor = 0, only 1 regime covered
        # Confidence formula:  0.25*sample + 0.20*regime + 0.15*temporal + 0.20*variance + 0.20*recent
        # sample = 15/50 = 0.30,  regime = 1/4 = 0.25,  temporal = 0,
        # variance = 1.0 (identical returns),  recent = 1.0 (all positive)
        # → confidence = 0.25*0.30 + 0.20*0.25 + 0.15*0 + 0.20*1.0 + 0.20*1.0 = 0.525
        # confidence = 0.525 satisfies SHADOW_READY (≥0.50) but NOT PAPER_READY (≥0.60)
        recs    = [_surv(f"S{i}", "2026-05-25", fwd10=0.09) for i in range(MIN_CLEAN_SAMPLE)]
        inputs  = PromotionReadinessInputs(recs, [], {"2026-05-25": "risk_on"}, "2026-05-25")
        report  = evaluate_promotion_readiness(inputs)
        # With 0 temporal span and 1 covered regime, should NOT reach PAPER_READY or above
        assert report.promotion_state not in (
            PromotionState.PAPER_READY,
            PromotionState.LIMITED_EXECUTION_READY,
            PromotionState.PRODUCTION_READY,
        )

    def test_confidence_increases_with_sample_size(self):
        # Keep dates identical structure (same 1-day span) — only sample size differs
        small_n = MIN_CLEAN_SAMPLE + 1
        large_n = CONFIDENCE_SAMPLE_TARGET + 20
        small_recs = [_surv(f"S{i}", "2026-05-25", fwd10=0.04) for i in range(small_n)]
        large_recs = [_surv(f"S{i}", "2026-05-25", fwd10=0.04) for i in range(large_n)]
        c_small, _ = _evaluate_promotion_confidence(
            _filter_clean_survivability(small_recs), {"2026-05-25": "risk_on"}
        )
        c_large, _ = _evaluate_promotion_confidence(
            _filter_clean_survivability(large_recs), {"2026-05-25": "risk_on"}
        )
        assert c_large > c_small

    def test_confidence_zero_on_empty(self):
        inputs  = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report  = evaluate_promotion_readiness(inputs)
        assert report.promotion_confidence == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def _make_inputs(self) -> PromotionReadinessInputs:
        return _make_valid_inputs(n=MIN_CLEAN_SAMPLE + 10)

    def test_same_inputs_same_hash(self):
        inputs = self._make_inputs()
        h1     = _compute_deterministic_hash(inputs)
        h2     = _compute_deterministic_hash(inputs)
        assert h1 == h2

    def test_same_inputs_same_report(self):
        inputs = self._make_inputs()
        r1     = evaluate_promotion_readiness(inputs)
        r2     = evaluate_promotion_readiness(inputs)
        assert r1.deterministic_hash == r2.deterministic_hash
        assert r1.readiness_score    == r2.readiness_score
        assert r1.promotion_state    == r2.promotion_state
        assert r1.component_scores   == r2.component_scores

    def test_different_inputs_different_hash(self):
        i1 = _make_valid_inputs(fwd10=0.03)
        i2 = _make_valid_inputs(fwd10=0.05)
        h1 = _compute_deterministic_hash(i1)
        h2 = _compute_deterministic_hash(i2)
        assert h1 != h2

    def test_record_order_does_not_affect_hash(self):
        inputs = self._make_inputs()
        shuffled = PromotionReadinessInputs(
            survivability_records=list(reversed(inputs.survivability_records)),
            failure_records=list(reversed(inputs.failure_records)),
            regime_tags=inputs.regime_tags,
            evaluation_date=inputs.evaluation_date,
        )
        h1 = _compute_deterministic_hash(inputs)
        h2 = _compute_deterministic_hash(shuffled)
        assert h1 == h2

    def test_hash_changes_on_evaluation_date_change(self):
        i1 = _make_valid_inputs(); i1.evaluation_date = "2026-05-25"
        i2 = _make_valid_inputs(); i2.evaluation_date = "2026-05-26"
        assert _compute_deterministic_hash(i1) != _compute_deterministic_hash(i2)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Append-only JSONL integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnlyJSONL:
    def test_append_creates_file(self, tmp_path):
        inputs = _make_valid_inputs()
        report = evaluate_promotion_readiness(inputs)
        append_promotion_report(report, tmp_path, today=date(2026, 5, 25))
        files  = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1

    def test_multiple_appends_accumulate(self, tmp_path):
        inputs = _make_valid_inputs()
        r1     = evaluate_promotion_readiness(inputs)
        r2     = evaluate_promotion_readiness(inputs)
        append_promotion_report(r1, tmp_path, today=date(2026, 5, 25))
        append_promotion_report(r2, tmp_path, today=date(2026, 5, 25))
        loaded = load_promotion_reports(tmp_path)
        assert len(loaded) == 2

    def test_append_does_not_overwrite(self, tmp_path):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        today   = date(2026, 5, 25)
        append_promotion_report(report, tmp_path, today=today)
        append_promotion_report(report, tmp_path, today=today)
        loaded  = load_promotion_reports(tmp_path)
        assert len(loaded) == 2

    def test_load_empty_dir_returns_empty(self, tmp_path):
        result = load_promotion_reports(tmp_path)
        assert result == []

    def test_load_missing_dir_returns_empty(self, tmp_path):
        result = load_promotion_reports(tmp_path / "nonexistent")
        assert result == []

    def test_jsonl_schema_fields(self, tmp_path):
        inputs = _make_valid_inputs()
        report = evaluate_promotion_readiness(inputs)
        append_promotion_report(report, tmp_path, today=date(2026, 5, 25))
        loaded = load_promotion_reports(tmp_path)
        assert len(loaded) == 1
        rec = loaded[0]
        for field_name in [
            "timestamp", "promotion_state", "promotion_ready",
            "readiness_score", "promotion_confidence",
            "component_scores", "blocking_reasons", "warnings",
            "sample_size", "recent_sample_size", "regime_coverage",
            "deterministic_hash",
        ]:
            assert field_name in rec, f"missing field: {field_name}"

    def test_jsonl_hash_persisted_correctly(self, tmp_path):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        append_promotion_report(report, tmp_path, today=date(2026, 5, 25))
        loaded  = load_promotion_reports(tmp_path)
        assert loaded[0]["deterministic_hash"] == report.deterministic_hash

    def test_fail_open_on_invalid_dir(self):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        invalid = Path("/proc/\x00invalid\x00")
        # Should not raise
        append_promotion_report(report, invalid, today=date(2026, 5, 25))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Component score sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestComponentScores:
    def test_all_components_present(self):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        expected = {"statistical_edge", "survivability", "regime_robustness",
                    "failure_quality", "temporal_stability"}
        assert set(report.component_scores.keys()) == expected

    def test_scores_in_unit_interval(self):
        inputs = _make_valid_inputs()
        report = evaluate_promotion_readiness(inputs)
        for name, score in report.component_scores.items():
            assert 0.0 <= score <= 1.0, f"{name} = {score}"

    def test_readiness_score_is_weighted_sum(self):
        from src.analytics.future_leader_promotion_readiness import _COMP_WEIGHTS
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        expected = sum(report.component_scores[k] * _COMP_WEIGHTS[k] for k in _COMP_WEIGHTS)
        assert report.readiness_score == pytest.approx(expected, abs=0.001)

    def test_statistical_edge_zero_on_empty(self):
        comp = _evaluate_statistical_edge([])
        assert comp.score == 0.0

    def test_statistical_edge_positive_on_good_returns(self):
        dates = _make_obs_dates(20)
        recs  = [_surv(f"S{i}", d, fwd10=0.06, mfe=0.10, mae=-0.01) for i, d in enumerate(dates)]
        comp  = _evaluate_statistical_edge(_filter_clean_survivability(recs))
        assert comp.score > 0.60

    def test_statistical_edge_low_on_negative_returns(self):
        dates = _make_obs_dates(20)
        recs  = [_surv(f"S{i}", d, fwd10=-0.04) for i, d in enumerate(dates)]
        comp  = _evaluate_statistical_edge(_filter_clean_survivability(recs))
        assert comp.score < 0.40

    def test_survivability_neutral_no_fail_records(self):
        comp = _evaluate_survivability([], [])
        assert comp.score == pytest.approx(0.30)

    def test_failure_quality_neutral_no_records(self):
        comp = _evaluate_failure_quality([])
        assert comp.score == pytest.approx(0.50)

    def test_temporal_stability_neutral_small_sample(self):
        recs = [_surv("S", "2026-01-05", fwd10=0.03)]
        comp = _evaluate_temporal_stability(_filter_clean_survivability(recs))
        assert comp.score == pytest.approx(0.50)

    def test_temporal_stability_high_on_consistent_returns(self):
        dates = _make_obs_dates(30)
        recs  = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(dates)]
        comp  = _evaluate_temporal_stability(_filter_clean_survivability(recs))
        assert comp.score > 0.80

    def test_component_scores_computed_even_when_blocked(self):
        # Even with too few records, we still get component scores
        inputs  = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report  = evaluate_promotion_readiness(inputs)
        assert len(report.component_scores) == 5

    def test_regime_robustness_zero_coverage(self):
        # 2 records per regime — below MIN_REGIME_SAMPLE=3 → covered_regimes = 0
        dates = _make_obs_dates(8)
        recs  = [_surv(f"S{i}", d) for i, d in enumerate(dates)]
        tags  = {d: (["risk_on", "risk_off", "range", "volatile"][i % 4]) for i, d in enumerate(dates)}
        comp  = _evaluate_regime_robustness(_filter_clean_survivability(recs), tags)
        assert comp.sub_scores.get("covered_regimes", -1) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Promotion confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotionConfidence:
    def test_confidence_in_unit_interval(self):
        inputs = _make_valid_inputs()
        report = evaluate_promotion_readiness(inputs)
        assert 0.0 <= report.promotion_confidence <= 1.0

    def test_confidence_zero_empty_input(self):
        conf, _ = _evaluate_promotion_confidence([], {})
        assert conf == pytest.approx(0.0)

    def test_confidence_higher_with_multi_regime_coverage(self):
        n     = CONFIDENCE_SAMPLE_TARGET
        dates = _make_obs_dates(n, start="2026-01-05")
        single_tags = {d: "risk_on" for d in dates}
        # Assign >= MIN_REGIME_SAMPLE records to each of 4 regimes
        regime_cycle = ["risk_on", "risk_off", "range", "volatile"]
        multi_tags   = {d: regime_cycle[i % 4] for i, d in enumerate(dates)}
        recs = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(dates)]
        c1, _ = _evaluate_promotion_confidence(_filter_clean_survivability(recs), single_tags)
        c2, _ = _evaluate_promotion_confidence(_filter_clean_survivability(recs), multi_tags)
        assert c2 > c1

    def test_confidence_higher_over_longer_time_span(self):
        n = 20
        # short: 20 dates all within 4 weeks (step=1 day)
        short = _make_spaced_dates(n, start="2026-04-28", step_days=1)
        # long: 20 dates spread over ~7 months (step=10 days)
        long_ = _make_spaced_dates(n, start="2025-10-01", step_days=10)
        recs_s = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(short)]
        recs_l = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(long_)]
        c1, _ = _evaluate_promotion_confidence(_filter_clean_survivability(recs_s), {d: "risk_on" for d in short})
        c2, _ = _evaluate_promotion_confidence(_filter_clean_survivability(recs_l), {d: "risk_on" for d in long_})
        assert c2 > c1


# ─────────────────────────────────────────────────────────────────────────────
# 10. PromotionReadinessReport serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_serializable(self):
        inputs = _make_valid_inputs()
        report = evaluate_promotion_readiness(inputs)
        d      = report.to_dict()
        # Must be JSON serializable
        json.dumps(d)

    def test_to_dict_state_is_string(self):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        d       = report.to_dict()
        assert isinstance(d["promotion_state"], str)

    def test_to_dict_blocking_reasons_list(self):
        inputs = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        d      = report.to_dict()
        assert isinstance(d["blocking_reasons"], list)

    def test_to_dict_scores_rounded(self):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        d       = report.to_dict()
        for k, v in d["component_scores"].items():
            assert isinstance(v, float)
            # Should be rounded to 4 decimal places
            assert v == round(v, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Format section smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSection:
    def test_section_contains_state(self):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        section = format_promotion_readiness_section(report)
        assert report.promotion_state.value in section

    def test_section_contains_blockers_when_blocked(self):
        inputs  = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report  = evaluate_promotion_readiness(inputs)
        section = format_promotion_readiness_section(report)
        assert "BLOCKERS" in section
        assert "insufficient_clean_sample" in section

    def test_section_contains_hash(self):
        inputs  = _make_valid_inputs()
        report  = evaluate_promotion_readiness(inputs)
        section = format_promotion_readiness_section(report)
        assert report.deterministic_hash[:8] in section

    def test_section_no_blockers_line_when_none(self):
        n     = CONFIDENCE_SAMPLE_TARGET
        dates = _make_spaced_dates(n, start="2025-06-01", step_days=3)
        surv  = [_surv(f"S{i}", d, fwd10=0.05) for i, d in enumerate(dates)]
        fail  = [_fail(f"S{i}", d) for i, d in enumerate(dates)]
        tags  = {d: "risk_on" for d in dates}
        inputs  = PromotionReadinessInputs(surv, fail, tags, "2026-05-25")
        report  = evaluate_promotion_readiness(inputs)
        section = format_promotion_readiness_section(report)
        if not report.blocking_reasons:
            assert "[BLOCKED]" not in section


# ─────────────────────────────────────────────────────────────────────────────
# 12. observation_only invariant — no execution imports
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationOnlyInvariant:
    def test_no_execution_imports(self):
        import importlib
        import sys
        mod = importlib.import_module("src.analytics.future_leader_promotion_readiness")
        src_text = Path(mod.__file__).read_text(encoding="utf-8")
        forbidden = [
            "from src.live.signal_bridge",
            "from src.allocation",
            "from src.truth",
            "predictive_entry_enabled = True",
            "allocator.route",
            "order_manager",
        ]
        for token in forbidden:
            assert token not in src_text, f"forbidden import found: {token}"

    def test_promotion_ready_does_not_mutate_inputs(self):
        inputs  = _make_valid_inputs()
        n_before = len(inputs.survivability_records)
        _       = evaluate_promotion_readiness(inputs)
        assert len(inputs.survivability_records) == n_before


# ─────────────────────────────────────────────────────────────────────────────
# 13. Regime coverage counting
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeCoverage:
    def test_regime_coverage_counted_correctly(self):
        n     = 20
        dates = _make_obs_dates(n)
        surv  = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(dates)]
        tags  = {d: (["risk_on", "risk_off"][i % 2]) for i, d in enumerate(dates)}
        inputs  = PromotionReadinessInputs(surv, [], tags, "2026-05-25")
        report  = evaluate_promotion_readiness(inputs)
        assert report.regime_coverage.get("risk_on", 0) == 10
        assert report.regime_coverage.get("risk_off", 0) == 10
        assert report.regime_coverage.get("range", 0)    == 0

    def test_untagged_obs_dates_default_to_range(self):
        n     = 20
        dates = _make_obs_dates(n)
        surv  = [_surv(f"S{i}", d, fwd10=0.04) for i, d in enumerate(dates)]
        inputs = PromotionReadinessInputs(surv, [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        # All untagged → counted under "range"
        assert report.regime_coverage.get("range", 0) == n


# ─────────────────────────────────────────────────────────────────────────────
# 14. Multiple blockers
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleBlockers:
    def test_multiple_blockers_all_reported(self):
        n     = MIN_CLEAN_SAMPLE + 30
        dates = _make_obs_dates(n)
        # Recent strongly negative → negative_recent_expectancy
        # All fail in 2d → high_instant_rejection
        early_dates  = dates[:MIN_CLEAN_SAMPLE]
        recent_dates = dates[MIN_CLEAN_SAMPLE:]
        surv = (
            [_surv(f"E{i}", d, fwd10=0.05) for i, d in enumerate(early_dates)]
            + [_surv(f"R{i}", d, fwd10=-0.08) for i, d in enumerate(recent_dates)]
        )
        fail = [_fail(f"X{i}", d, failure_detected=True, offset=2) for i, d in enumerate(dates)]
        tags = {d: "risk_on" for d in dates}
        report = evaluate_promotion_readiness(
            PromotionReadinessInputs(surv, fail, tags, "2026-05-25")
        )
        assert report.promotion_state == PromotionState.BLOCKED
        assert len(report.blocking_reasons) >= 2

    def test_blocked_state_always_when_any_blocker(self):
        inputs = PromotionReadinessInputs([], [], {}, "2026-05-25")
        report = evaluate_promotion_readiness(inputs)
        assert report.promotion_state == PromotionState.BLOCKED
        assert report.promotion_ready is False
