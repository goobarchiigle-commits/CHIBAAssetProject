"""
tests/runtime/test_analytics_policy_bridge.py

Tests for src/runtime/policy/analytics_policy_bridge.py
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.runtime.policy.analytics_policy_bridge import (
    ALL_POLICY_TYPES,
    DEFAULT_STORE_PATH,
    POLICY_ENTRY_THROTTLING,
    POLICY_HOLD_EXTENSION,
    POLICY_PORTFOLIO_HEAT_LIMIT,
    POLICY_RISK_SCALING,
    POLICY_SECTOR_EXPOSURE_CAP,
    POLICY_SLOT_ALLOCATION,
    TRIGGER_CAPTURE_RATIO,
    TRIGGER_CONVEXITY_SCORE,
    TRIGGER_HEAT_LEAK_RATE,
    TRIGGER_LIFECYCLE_EXTENSION_SIGNAL,
    TRIGGER_SECTOR_MISS_RATE,
    TRIGGER_SKIPPED_EXPECTANCY_DELTA,
    ActivePolicy,
    ActivePolicyState,
    AnalyticsInput,
    AnalyticsPolicyBridge,
    PolicyConstraints,
    PolicyDecisionRecord,
    PolicyIntegrityReport,
    PolicyIntegrityValidator,
    PolicyViolation,
    SCHEMA_VERSION,
    _bound_policy_value,
    _clamp,
    _compute_confidence,
    _sha256,
    append_decision,
    deduplicate_decisions,
    load_decisions,
    run_policy_hook,
)

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_constraints(**kwargs) -> PolicyConstraints:
    defaults = dict(
        max_allocation_change_per_week=0.05,
        max_exposure_delta=0.10,
        cooldown_days_after_change=3,
        min_sample_size=5,
        confidence_threshold=0.50,
        max_decisions_per_type_per_week=1,
        max_total_decisions_per_week=3,
    )
    defaults.update(kwargs)
    return PolicyConstraints(**defaults)


def _make_decision(
    policy_type: str = POLICY_SLOT_ALLOCATION,
    eval_date: str = "2026-05-19",
    previous_value: float = 1.0,
    new_value: float = 1.05,
    trigger_metric: str = TRIGGER_SKIPPED_EXPECTANCY_DELTA,
    trigger_value: float = 0.042,
    confidence: float = 0.87,
    bounded: bool = True,
    rollback_eligible: bool = True,
    observation_count: int = 20,
    lookback_days: int = 20,
    timestamp_jst: str = "2026-05-19T09:00:00+09:00",
) -> PolicyDecisionRecord:
    return PolicyDecisionRecord.create(
        timestamp_jst=timestamp_jst,
        eval_date=eval_date,
        policy_type=policy_type,
        previous_value=previous_value,
        new_value=new_value,
        trigger_metric=trigger_metric,
        trigger_value=trigger_value,
        confidence=confidence,
        bounded=bounded,
        rollback_eligible=rollback_eligible,
        observation_count=observation_count,
        lookback_days=lookback_days,
    )


def _make_analytics(
    skipped_delta_5d: float = 0.05,
    skipped_win_rate: float = 0.65,
    capture_ratio: float = 0.60,
    convexity: float = 1.5,
    n_executed: int = 20,
    n_skipped: int = 10,
    capital_leak: float = 0.04,
    heat_leak: float = 0.03,
    sector_leak: float = 0.03,
    hold_extension: float = 0.05,
    eval_date: str = "2026-05-19",
) -> AnalyticsInput:
    return AnalyticsInput(
        skipped_expectancy_delta_5d=skipped_delta_5d,
        skipped_win_rate=skipped_win_rate,
        opportunity_capture_ratio=capture_ratio,
        skipped_convexity_score=convexity,
        n_executed=n_executed,
        n_skipped=n_skipped,
        capital_constraint_leakage=capital_leak,
        heat_limit_leakage=heat_leak,
        sector_exposure_leakage=sector_leak,
        lifecycle_hold_extension_signal=hold_extension,
        eval_date=eval_date,
    )


def _default_policy() -> dict:
    return {
        POLICY_SLOT_ALLOCATION:      1.0,
        POLICY_SECTOR_EXPOSURE_CAP:  0.30,
        POLICY_PORTFOLIO_HEAT_LIMIT: 0.40,
        POLICY_HOLD_EXTENSION:       0.0,
        POLICY_ENTRY_THROTTLING:     1.0,
        POLICY_RISK_SCALING:         1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PolicyConstraints
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyConstraints:
    def test_defaults(self):
        c = PolicyConstraints()
        assert c.min_sample_size == 10
        assert c.confidence_threshold == 0.70
        assert c.cooldown_days_after_change == 3
        assert c.max_decisions_per_type_per_week == 1
        assert c.max_total_decisions_per_week == 3

    def test_frozen(self):
        c = PolicyConstraints()
        with pytest.raises((AttributeError, TypeError)):
            c.min_sample_size = 99  # type: ignore

    def test_custom_construction(self):
        c = _make_constraints(min_sample_size=5, confidence_threshold=0.50)
        assert c.min_sample_size == 5
        assert c.confidence_threshold == 0.50


# ─────────────────────────────────────────────────────────────────────────────
# PolicyDecisionRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyDecisionRecord:
    def test_create_basic(self):
        r = _make_decision()
        assert len(r.decision_id) == 64
        assert r.schema_version == SCHEMA_VERSION
        assert r.bounded is True
        assert r.rollback_eligible is True

    def test_decision_id_determinism(self):
        r1 = _make_decision(eval_date="2026-05-19", policy_type=POLICY_SLOT_ALLOCATION,
                             new_value=1.05, timestamp_jst="2026-05-19T09:00:00+09:00")
        r2 = _make_decision(eval_date="2026-05-19", policy_type=POLICY_SLOT_ALLOCATION,
                             new_value=1.05, timestamp_jst="2026-05-19T09:00:00+09:00")
        assert r1.decision_id == r2.decision_id

    def test_decision_id_differs_by_policy_type(self):
        r1 = _make_decision(policy_type=POLICY_SLOT_ALLOCATION)
        r2 = _make_decision(policy_type=POLICY_SECTOR_EXPOSURE_CAP)
        assert r1.decision_id != r2.decision_id

    def test_decision_id_differs_by_new_value(self):
        r1 = _make_decision(new_value=1.05)
        r2 = _make_decision(new_value=1.10)
        assert r1.decision_id != r2.decision_id

    def test_round_trip(self):
        r = _make_decision()
        d = r.to_dict()
        r2 = PolicyDecisionRecord.from_dict(d)
        assert r2.decision_id == r.decision_id
        assert r2.policy_type == r.policy_type
        assert r2.new_value == pytest.approx(r.new_value)
        assert r2.confidence == pytest.approx(r.confidence)

    def test_frozen(self):
        r = _make_decision()
        with pytest.raises((AttributeError, TypeError)):
            r.new_value = 99.0  # type: ignore

    def test_from_dict_missing_fields(self):
        d = {
            "decision_id": "abc",
            "schema_version": 1,
            "timestamp_jst": "2026-05-19T09:00:00+09:00",
            "eval_date": "2026-05-19",
            "policy_type": POLICY_SLOT_ALLOCATION,
            "previous_value": 1.0,
            "new_value": 1.05,
            "trigger_metric": TRIGGER_SKIPPED_EXPECTANCY_DELTA,
            "trigger_value": 0.04,
            "confidence": 0.87,
            "bounded": True,
            "rollback_eligible": True,
            "observation_count": 20,
            "lookback_days": 20,
        }
        r = PolicyDecisionRecord.from_dict(d)
        assert r.bounded is True


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticsPolicyBridge — core evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticsPolicyBridge:
    def setup_method(self, tmp_path=None):
        self.c = _make_constraints()
        self.bridge = AnalyticsPolicyBridge(constraints=self.c)

    def test_produces_decisions_when_analytics_warrant(self):
        a = _make_analytics()
        decisions = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert len(decisions) > 0

    def test_all_decisions_are_bounded(self):
        a = _make_analytics()
        decisions = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        for d in decisions:
            assert d.bounded is True

    def test_all_decisions_are_rollback_eligible(self):
        a = _make_analytics()
        decisions = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        for d in decisions:
            assert d.rollback_eligible is True

    def test_no_decision_when_insufficient_sample(self):
        c = _make_constraints(min_sample_size=100)
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(n_executed=3, n_skipped=2)
        decisions = bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert len(decisions) == 0

    def test_no_decision_when_low_confidence(self):
        c = _make_constraints(confidence_threshold=0.999)
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(n_executed=5, n_skipped=3)
        decisions = bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert len(decisions) == 0

    def test_cooldown_blocks_same_type(self):
        today = "2026-05-19"
        # Decision from yesterday (within cooldown=3 days)
        history = [_make_decision(
            policy_type=POLICY_SLOT_ALLOCATION,
            eval_date="2026-05-18",
        )]
        a = _make_analytics()
        decisions = self.bridge.evaluate(a, _default_policy(), history, as_of_date=today)
        types = [d.policy_type for d in decisions]
        assert POLICY_SLOT_ALLOCATION not in types

    def test_cooldown_allows_after_window(self):
        today = "2026-05-20"
        # Decision from 8 days ago: past cooldown (3d) AND past weekly quota (7d)
        history = [_make_decision(
            policy_type=POLICY_SLOT_ALLOCATION,
            eval_date="2026-05-12",
            new_value=0.95,
            timestamp_jst="2026-05-12T09:00:00+09:00",
        )]
        # Provide a base where slot can increase (below max)
        policy = {**_default_policy(), POLICY_SLOT_ALLOCATION: 0.90}
        a = _make_analytics(capital_leak=0.05)
        decisions = self.bridge.evaluate(a, policy, history, as_of_date=today)
        types = [d.policy_type for d in decisions]
        assert POLICY_SLOT_ALLOCATION in types

    def test_weekly_quota_blocks_excess_decisions(self):
        c = _make_constraints(max_decisions_per_type_per_week=1)
        bridge = AnalyticsPolicyBridge(constraints=c)
        today = "2026-05-19"
        history = [_make_decision(
            policy_type=POLICY_SLOT_ALLOCATION,
            eval_date="2026-05-14",  # within 7-day window
        )]
        a = _make_analytics()
        decisions = bridge.evaluate(a, _default_policy(), history, as_of_date=today)
        slot_decisions = [d for d in decisions if d.policy_type == POLICY_SLOT_ALLOCATION]
        assert len(slot_decisions) == 0

    def test_max_total_decisions_per_week_capped(self):
        c = _make_constraints(max_total_decisions_per_week=1)
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics()
        decisions = bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert len(decisions) <= 1

    def test_slot_allocation_bounded_within_constraints(self):
        c = _make_constraints()
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(capital_leak=0.99)  # extreme leakage
        decisions = bridge.evaluate(a, {POLICY_SLOT_ALLOCATION: 1.0}, [], as_of_date="2026-05-19")
        slot = [d for d in decisions if d.policy_type == POLICY_SLOT_ALLOCATION]
        if slot:
            assert slot[0].new_value <= c.slot_allocation_max
            assert slot[0].new_value >= c.slot_allocation_min
            delta = abs(slot[0].new_value - slot[0].previous_value)
            assert delta <= c.max_allocation_change_per_week + 1e-6

    def test_sector_exposure_cap_bounded(self):
        c = _make_constraints()
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(sector_leak=0.99)
        decisions = bridge.evaluate(a, {POLICY_SECTOR_EXPOSURE_CAP: 0.30}, [],
                                    as_of_date="2026-05-19")
        sec = [d for d in decisions if d.policy_type == POLICY_SECTOR_EXPOSURE_CAP]
        if sec:
            assert sec[0].new_value <= c.sector_exposure_cap_max
            assert sec[0].new_value >= c.sector_exposure_cap_min

    def test_portfolio_heat_bounded(self):
        c = _make_constraints()
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(heat_leak=0.99)
        decisions = bridge.evaluate(a, {POLICY_PORTFOLIO_HEAT_LIMIT: 0.40}, [],
                                    as_of_date="2026-05-19")
        heat = [d for d in decisions if d.policy_type == POLICY_PORTFOLIO_HEAT_LIMIT]
        if heat:
            assert heat[0].new_value <= c.portfolio_heat_max

    def test_hold_extension_eligibility_is_binary(self):
        c = _make_constraints()
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(hold_extension=0.10)
        decisions = bridge.evaluate(a, {POLICY_HOLD_EXTENSION: 0.0}, [],
                                    as_of_date="2026-05-19")
        ext = [d for d in decisions if d.policy_type == POLICY_HOLD_EXTENSION]
        if ext:
            assert ext[0].new_value in (0.0, 1.0)

    def test_entry_throttling_only_increases(self):
        c = _make_constraints()
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(capture_ratio=0.5, skipped_delta_5d=0.05)
        decisions = bridge.evaluate(a, {POLICY_ENTRY_THROTTLING: 0.80}, [],
                                    as_of_date="2026-05-19")
        thr = [d for d in decisions if d.policy_type == POLICY_ENTRY_THROTTLING]
        if thr:
            assert thr[0].new_value >= thr[0].previous_value

    def test_risk_scaling_bounded(self):
        c = _make_constraints()
        bridge = AnalyticsPolicyBridge(constraints=c)
        a = _make_analytics(convexity=2.0, skipped_delta_5d=0.08)
        decisions = bridge.evaluate(a, {POLICY_RISK_SCALING: 1.0}, [],
                                    as_of_date="2026-05-19")
        rs = [d for d in decisions if d.policy_type == POLICY_RISK_SCALING]
        if rs:
            assert rs[0].new_value <= c.risk_scaling_max
            assert rs[0].new_value >= c.risk_scaling_min

    def test_no_decision_when_no_leakage(self):
        a = _make_analytics(
            capital_leak=0.0, heat_leak=0.0, sector_leak=0.0,
            skipped_delta_5d=0.0, hold_extension=0.0, convexity=0.5,
        )
        decisions = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert len(decisions) == 0

    def test_fail_closed_on_integrity_violation(self):
        # History with a future-date record → integrity violation → no decisions
        bad_history = [_make_decision(eval_date="2099-01-01")]
        a = _make_analytics()
        decisions = self.bridge.evaluate(a, _default_policy(), bad_history,
                                         as_of_date="2026-05-19")
        assert len(decisions) == 0

    def test_decisions_have_valid_confidence(self):
        a = _make_analytics()
        decisions = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        for d in decisions:
            assert 0.0 <= d.confidence <= 1.0

    def test_decisions_have_correct_observation_count(self):
        a = _make_analytics(n_executed=15, n_skipped=10)
        decisions = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        for d in decisions:
            assert d.observation_count == 25

    def test_replay_determinism(self):
        a = _make_analytics()
        d1 = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        d2 = self.bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert len(d1) == len(d2)
        for r1, r2 in zip(
            sorted(d1, key=lambda x: x.policy_type),
            sorted(d2, key=lambda x: x.policy_type),
        ):
            assert r1.policy_type == r2.policy_type
            assert r1.new_value == pytest.approx(r2.new_value)


# ─────────────────────────────────────────────────────────────────────────────
# PolicyIntegrityValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyIntegrityValidator:
    def setup_method(self):
        self.v = PolicyIntegrityValidator()
        self.c = _make_constraints()

    def test_clean_records_pass(self):
        records = [
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-10"),
            _make_decision(policy_type=POLICY_SECTOR_EXPOSURE_CAP, eval_date="2026-05-15"),
        ]
        report = self.v.validate(records, as_of_date="2026-05-20", constraints=self.c)
        assert report.passed is True
        assert report.n_violations == 0

    def test_duplicate_decision_id_fails(self):
        r = _make_decision()
        report = self.v.validate([r, r], as_of_date="2026-05-20", constraints=self.c)
        assert not report.passed
        types = [v.violation_type for v in report.violations]
        assert "duplicate_decision_id" in types

    def test_future_leakage_fails(self):
        r = _make_decision(eval_date="2099-01-01")
        report = self.v.validate([r], as_of_date="2026-05-20", constraints=self.c)
        assert not report.passed
        types = [v.violation_type for v in report.violations]
        assert "future_leakage" in types

    def test_insufficient_sample_fails(self):
        c = _make_constraints(min_sample_size=50)
        r = _make_decision(observation_count=5)
        report = self.v.validate([r], as_of_date="2026-05-20", constraints=c)
        assert not report.passed
        types = [v.violation_type for v in report.violations]
        assert "insufficient_sample" in types

    def test_invalid_confidence_flagged(self):
        import dataclasses
        r = dataclasses.replace(_make_decision(), confidence=1.5)
        report = self.v.validate([r], as_of_date="2026-05-20", constraints=self.c)
        types = [v.violation_type for v in report.violations]
        assert "invalid_confidence" in types

    def test_unbounded_decision_flagged(self):
        import dataclasses
        r = dataclasses.replace(_make_decision(), bounded=False)
        report = self.v.validate([r], as_of_date="2026-05-20", constraints=self.c)
        types = [v.violation_type for v in report.violations]
        assert "unbounded_decision" in types

    def test_rapid_recursive_adaptation_fails(self):
        c = _make_constraints(max_decisions_per_type_per_week=1)
        records = [
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-14",
                           new_value=1.02, timestamp_jst="2026-05-14T09:00:00+09:00"),
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-16",
                           new_value=1.04, timestamp_jst="2026-05-16T09:00:00+09:00"),
        ]
        report = self.v.validate(records, as_of_date="2026-05-20", constraints=c)
        assert not report.passed
        types = [v.violation_type for v in report.violations]
        assert "rapid_recursive_adaptation" in types

    def test_policy_oscillation_flagged(self):
        c = _make_constraints(cooldown_days_after_change=3)
        records = [
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-17",
                           new_value=1.05, timestamp_jst="2026-05-17T09:00:00+09:00"),
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-18",
                           new_value=1.03, timestamp_jst="2026-05-18T09:00:00+09:00"),
        ]
        report = self.v.validate(records, as_of_date="2026-05-20", constraints=c)
        types = [v.violation_type for v in report.violations]
        assert "policy_oscillation" in types

    def test_no_oscillation_outside_cooldown(self):
        c = _make_constraints(cooldown_days_after_change=3)
        records = [
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-10",
                           new_value=1.05, timestamp_jst="2026-05-10T09:00:00+09:00"),
            _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-15",
                           new_value=1.03, timestamp_jst="2026-05-15T09:00:00+09:00"),
        ]
        report = self.v.validate(records, as_of_date="2026-05-20", constraints=c)
        oscillation = [v for v in report.violations if v.violation_type == "policy_oscillation"]
        assert len(oscillation) == 0

    def test_n_records_correct(self):
        records = [_make_decision(policy_type=pt, eval_date="2026-05-10")
                   for pt in ALL_POLICY_TYPES]
        report = self.v.validate(records, as_of_date="2026-05-20", constraints=self.c)
        assert report.n_records == len(ALL_POLICY_TYPES)

    def test_empty_records_pass(self):
        report = self.v.validate([], as_of_date="2026-05-20", constraints=self.c)
        assert report.passed is True
        assert report.n_records == 0


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r1 = _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-19")
        r2 = _make_decision(policy_type=POLICY_SECTOR_EXPOSURE_CAP, eval_date="2026-05-18",
                             new_value=0.32, timestamp_jst="2026-05-18T09:00:00+09:00")
        append_decision(r1, path)
        append_decision(r2, path)
        loaded = load_decisions(path)
        assert len(loaded) == 2
        types = {r.policy_type for r in loaded}
        assert types == {POLICY_SLOT_ALLOCATION, POLICY_SECTOR_EXPOSURE_CAP}

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert load_decisions(path) == []

    def test_load_skips_corrupted_lines(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision()
        append_decision(r, path)
        with path.open("a", encoding="utf-8") as f:
            f.write("NOT_JSON\n")
        loaded = load_decisions(path)
        assert len(loaded) == 1

    def test_append_utf8(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision()
        append_decision(r, path)
        raw = path.read_text(encoding="utf-8")
        assert len(raw) > 0

    def test_serialization_sort_keys_deterministic(self, tmp_path):
        path1 = tmp_path / "a.jsonl"
        path2 = tmp_path / "b.jsonl"
        r = _make_decision()
        append_decision(r, path1)
        append_decision(r, path2)
        assert path1.read_text(encoding="utf-8") == path2.read_text(encoding="utf-8")

    def test_append_fail_open(self):
        path = Path("/nonexistent_xyz/policy.jsonl")
        r = _make_decision()
        append_decision(r, path)  # must not raise

    def test_load_preserves_all_fields(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision(confidence=0.88, observation_count=42, lookback_days=30)
        append_decision(r, path)
        loaded = load_decisions(path)
        assert loaded[0].confidence == pytest.approx(0.88)
        assert loaded[0].observation_count == 42
        assert loaded[0].lookback_days == 30


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_dedup_identical(self):
        r = _make_decision()
        result = deduplicate_decisions([r, r])
        assert len(result) == 1

    def test_dedup_distinct_kept(self):
        r1 = _make_decision(policy_type=POLICY_SLOT_ALLOCATION)
        r2 = _make_decision(policy_type=POLICY_SECTOR_EXPOSURE_CAP)
        result = deduplicate_decisions([r1, r2])
        assert len(result) == 2

    def test_stable_sort_by_eval_date(self):
        r1 = _make_decision(eval_date="2026-05-01", policy_type=POLICY_SLOT_ALLOCATION,
                             new_value=1.01, timestamp_jst="2026-05-01T09:00:00+09:00")
        r2 = _make_decision(eval_date="2026-05-02", policy_type=POLICY_SECTOR_EXPOSURE_CAP,
                             new_value=0.32, timestamp_jst="2026-05-02T09:00:00+09:00")
        result = deduplicate_decisions([r2, r1])
        assert result[0].eval_date <= result[1].eval_date


# ─────────────────────────────────────────────────────────────────────────────
# ActivePolicyState
# ─────────────────────────────────────────────────────────────────────────────

class TestActivePolicyState:
    def test_load_empty(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        state = ActivePolicyState(store_path=path)
        result = state.load(as_of_date="2026-05-20")
        assert result == {}

    def test_load_latest_per_type(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r1 = _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-10",
                             new_value=1.03, timestamp_jst="2026-05-10T09:00:00+09:00")
        r2 = _make_decision(policy_type=POLICY_SLOT_ALLOCATION, eval_date="2026-05-15",
                             new_value=1.05, timestamp_jst="2026-05-15T09:00:00+09:00")
        append_decision(r1, path)
        append_decision(r2, path)
        state = ActivePolicyState(store_path=path)
        active = state.load(as_of_date="2026-05-20")
        assert POLICY_SLOT_ALLOCATION in active
        assert active[POLICY_SLOT_ALLOCATION].value == pytest.approx(1.05)

    def test_expired_records_excluded(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        old = _make_decision(eval_date="2025-01-01",
                              timestamp_jst="2025-01-01T09:00:00+09:00")
        append_decision(old, path)
        state = ActivePolicyState(store_path=path, ttl_days=30)
        active = state.load(as_of_date="2026-05-20")
        assert POLICY_SLOT_ALLOCATION not in active

    def test_future_records_excluded(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        future = _make_decision(eval_date="2099-01-01",
                                 timestamp_jst="2099-01-01T09:00:00+09:00")
        append_decision(future, path)
        state = ActivePolicyState(store_path=path)
        active = state.load(as_of_date="2026-05-20")
        assert len(active) == 0

    def test_apply_to_runtime_params_bounded(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision(policy_type=POLICY_SECTOR_EXPOSURE_CAP,
                           new_value=0.35, eval_date="2026-05-19")
        append_decision(r, path)
        state = ActivePolicyState(store_path=path)
        active = state.load(as_of_date="2026-05-20")
        base = {POLICY_SECTOR_EXPOSURE_CAP: 0.30}
        result = state.apply_to_runtime_params(base, active)
        # Value should be changed but bounded
        assert POLICY_SECTOR_EXPOSURE_CAP in result
        c = PolicyConstraints()
        assert result[POLICY_SECTOR_EXPOSURE_CAP] <= c.sector_exposure_cap_max
        assert result[POLICY_SECTOR_EXPOSURE_CAP] >= c.sector_exposure_cap_min

    def test_apply_to_runtime_params_respects_max_delta(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        # Policy says new_value=0.90 but base=0.30, delta=0.60 >> max_allocation_change=0.05
        r = _make_decision(policy_type=POLICY_SECTOR_EXPOSURE_CAP,
                           new_value=0.90, eval_date="2026-05-19")
        append_decision(r, path)
        state = ActivePolicyState(store_path=path)
        active = state.load(as_of_date="2026-05-20")
        base = {POLICY_SECTOR_EXPOSURE_CAP: 0.30}
        result = state.apply_to_runtime_params(base, active)
        delta = abs(result[POLICY_SECTOR_EXPOSURE_CAP] - 0.30)
        assert delta <= PolicyConstraints().max_allocation_change_per_week + 1e-6

    def test_apply_does_not_mutate_base(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision(policy_type=POLICY_SLOT_ALLOCATION, new_value=1.03, eval_date="2026-05-19")
        append_decision(r, path)
        state = ActivePolicyState(store_path=path)
        active = state.load(as_of_date="2026-05-20")
        base = {POLICY_SLOT_ALLOCATION: 1.0}
        base_copy = dict(base)
        state.apply_to_runtime_params(base, active)
        assert base == base_copy

    def test_load_fail_open_on_missing_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        state = ActivePolicyState(store_path=path)
        result = state.load(as_of_date="2026-05-20")
        assert result == {}

    def test_active_policy_fields(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision(confidence=0.85, observation_count=30)
        append_decision(r, path)
        state = ActivePolicyState(store_path=path)
        active = state.load(as_of_date="2026-05-20")
        ap = active[POLICY_SLOT_ALLOCATION]
        assert ap.confidence == pytest.approx(0.85)
        assert ap.observation_count == 30
        assert ap.policy_type == POLICY_SLOT_ALLOCATION


# ─────────────────────────────────────────────────────────────────────────────
# run_policy_hook integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPolicyHook:
    def test_returns_base_when_no_decisions(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        base = {POLICY_SLOT_ALLOCATION: 1.0, POLICY_RISK_SCALING: 1.0}
        result = run_policy_hook(store_path=path, base_params=base, as_of_date="2026-05-20")
        assert result[POLICY_SLOT_ALLOCATION] == pytest.approx(1.0)
        assert result[POLICY_RISK_SCALING] == pytest.approx(1.0)

    def test_applies_active_decision(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        # sector_exposure_cap: new_value=0.33 is within [0.10, 0.50] and delta=0.03 < max_delta=0.05
        r = _make_decision(
            policy_type=POLICY_SECTOR_EXPOSURE_CAP,
            previous_value=0.30, new_value=0.33,
            eval_date="2026-05-19",
            timestamp_jst="2026-05-19T09:00:00+09:00",
        )
        append_decision(r, path)
        base = {POLICY_SECTOR_EXPOSURE_CAP: 0.30}
        result = run_policy_hook(store_path=path, base_params=base, as_of_date="2026-05-20")
        assert result[POLICY_SECTOR_EXPOSURE_CAP] != pytest.approx(0.30)

    def test_fail_open_on_error(self, tmp_path):
        base = {POLICY_SLOT_ALLOCATION: 1.0}
        bad_path = Path("/nonexistent_xyz/policy.jsonl")
        result = run_policy_hook(store_path=bad_path, base_params=base)
        assert result[POLICY_SLOT_ALLOCATION] == pytest.approx(1.0)

    def test_does_not_mutate_base_params(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        r = _make_decision(policy_type=POLICY_SLOT_ALLOCATION, new_value=1.03)
        append_decision(r, path)
        base = {POLICY_SLOT_ALLOCATION: 1.0}
        original = dict(base)
        run_policy_hook(store_path=path, base_params=base, as_of_date="2026-05-20")
        assert base == original

    def test_none_base_params_returns_empty(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        result = run_policy_hook(store_path=path, base_params=None)
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# Boundary and safety bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundsAndSafety:
    def test_bound_policy_value_slot_allocation(self):
        c = PolicyConstraints()
        assert _bound_policy_value(POLICY_SLOT_ALLOCATION, 99.0, c) == c.slot_allocation_max
        assert _bound_policy_value(POLICY_SLOT_ALLOCATION, -1.0, c) == c.slot_allocation_min

    def test_bound_policy_value_sector_cap(self):
        c = PolicyConstraints()
        assert _bound_policy_value(POLICY_SECTOR_EXPOSURE_CAP, 0.99, c) == c.sector_exposure_cap_max
        assert _bound_policy_value(POLICY_SECTOR_EXPOSURE_CAP, 0.0, c) == c.sector_exposure_cap_min

    def test_bound_policy_value_heat_limit(self):
        c = PolicyConstraints()
        assert _bound_policy_value(POLICY_PORTFOLIO_HEAT_LIMIT, 0.99, c) == c.portfolio_heat_max
        assert _bound_policy_value(POLICY_PORTFOLIO_HEAT_LIMIT, 0.0, c) == c.portfolio_heat_min

    def test_bound_policy_value_entry_throttle(self):
        c = PolicyConstraints()
        assert _bound_policy_value(POLICY_ENTRY_THROTTLING, 1.5, c) == c.entry_throttle_max
        assert _bound_policy_value(POLICY_ENTRY_THROTTLING, 0.0, c) == c.entry_throttle_min

    def test_bound_policy_value_risk_scaling(self):
        c = PolicyConstraints()
        assert _bound_policy_value(POLICY_RISK_SCALING, 9.9, c) == c.risk_scaling_max
        assert _bound_policy_value(POLICY_RISK_SCALING, 0.0, c) == c.risk_scaling_min

    def test_clamp(self):
        assert _clamp(5.0, 0.0, 1.0) == 1.0
        assert _clamp(-1.0, 0.0, 1.0) == 0.0
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_compute_confidence_below_min_sample(self):
        assert _compute_confidence(n=2, signal_strength=0.10, min_n=10) == 0.0

    def test_compute_confidence_grows_with_sample(self):
        c1 = _compute_confidence(n=10, signal_strength=0.10, min_n=10)
        c2 = _compute_confidence(n=50, signal_strength=0.10, min_n=10)
        assert c2 >= c1

    def test_compute_confidence_in_range(self):
        for n in [5, 10, 20, 100]:
            for sig in [0.01, 0.05, 0.20]:
                c = _compute_confidence(n=n, signal_strength=sig, min_n=10)
                assert 0.0 <= c <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Observational only — no strategy mutation
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationalOnlyConstraints:
    """Verify no strategy mutation occurs through the bridge."""

    def test_bridge_does_not_modify_analytics_input(self):
        a = _make_analytics()
        original_delta = a.skipped_expectancy_delta_5d
        bridge = AnalyticsPolicyBridge(constraints=_make_constraints())
        bridge.evaluate(a, _default_policy(), [], as_of_date="2026-05-19")
        assert a.skipped_expectancy_delta_5d == original_delta

    def test_bridge_does_not_modify_current_policy(self):
        current = _default_policy()
        original = dict(current)
        bridge = AnalyticsPolicyBridge(constraints=_make_constraints())
        bridge.evaluate(_make_analytics(), current, [], as_of_date="2026-05-19")
        assert current == original

    def test_bridge_does_not_modify_history(self):
        history = [_make_decision()]
        original_len = len(history)
        bridge = AnalyticsPolicyBridge(constraints=_make_constraints())
        bridge.evaluate(_make_analytics(), _default_policy(), history, as_of_date="2026-05-20")
        assert len(history) == original_len

    def test_all_decisions_bounded_true(self):
        bridge = AnalyticsPolicyBridge(constraints=_make_constraints())
        decisions = bridge.evaluate(_make_analytics(), _default_policy(), [],
                                    as_of_date="2026-05-19")
        for d in decisions:
            assert d.bounded is True, f"{d.policy_type} has bounded=False"

    def test_no_decision_written_automatically(self, tmp_path):
        """Bridge.evaluate() returns records but does NOT write to JSONL."""
        path = tmp_path / "policy.jsonl"
        bridge = AnalyticsPolicyBridge(constraints=_make_constraints(), store_path=path)
        bridge.evaluate(_make_analytics(), _default_policy(), [], as_of_date="2026-05-19")
        assert not path.exists()
