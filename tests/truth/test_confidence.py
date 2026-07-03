"""
Tests for confidence.py — truth confidence scoring.
"""
import math
import pytest
from src.truth.confidence import (
    ConfidenceScore, compute_confidence, confidence_from_truth_state,
    score_broker_factor, score_unknown_factor,
    score_staleness_factor, score_orphan_factor,
    BROKER_UNAVAILABLE_FLOOR, STALE_HALF_LIFE_SEC,
)
from src.truth.capital_truth import MIN_CONFIDENCE_TO_DEPLOY


# ── Component scorers ─────────────────────────────────────────────────────────

class TestBrokerFactor:
    def test_available_returns_one(self):
        assert score_broker_factor(True) == pytest.approx(1.0)

    def test_unavailable_returns_floor(self):
        assert score_broker_factor(False) == pytest.approx(BROKER_UNAVAILABLE_FLOOR)


class TestUnknownFactor:
    def test_no_unknowns_returns_one(self):
        assert score_unknown_factor(0, 5) == pytest.approx(1.0)

    def test_all_unknown_returns_zero(self):
        assert score_unknown_factor(5, 5) == pytest.approx(0.0)

    def test_half_unknown_returns_half(self):
        assert score_unknown_factor(3, 6) == pytest.approx(0.5)

    def test_zero_total_returns_one(self):
        assert score_unknown_factor(0, 0) == pytest.approx(1.0)

    def test_more_unknown_than_total_clamped_to_zero(self):
        # shouldn't happen, but clamp defensively
        assert score_unknown_factor(10, 5) == pytest.approx(0.0)


class TestStalenessFactor:
    def test_zero_age_returns_one(self):
        assert score_staleness_factor(0.0) == pytest.approx(1.0)

    def test_half_life_returns_half(self):
        result = score_staleness_factor(STALE_HALF_LIFE_SEC)
        assert abs(result - 0.5) < 0.001

    def test_negative_age_returns_one(self):
        assert score_staleness_factor(-10.0) == pytest.approx(1.0)

    def test_large_age_approaches_zero(self):
        result = score_staleness_factor(3600.0)  # 1 hour
        assert result < 0.01

    def test_monotone_decay(self):
        ages = [0, 60, 300, 600, 1800, 3600]
        scores = [score_staleness_factor(a) for a in ages]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestOrphanFactor:
    def test_no_orphans_returns_one(self):
        assert score_orphan_factor(0) == pytest.approx(1.0)

    def test_one_orphan_reduces(self):
        assert score_orphan_factor(1) < 1.0

    def test_more_orphans_lower_score(self):
        assert score_orphan_factor(3) < score_orphan_factor(1)

    def test_always_positive(self):
        assert score_orphan_factor(100) > 0.0


# ── Composite score ───────────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_ideal_conditions_high_score(self):
        cs = compute_confidence(
            broker_available=True,
            n_unknown=0,
            n_total_active=5,
            reconcile_age_sec=0.0,
            n_orphans=0,
        )
        assert cs.score >= 0.95

    def test_broker_unavailable_drops_score(self):
        ideal = compute_confidence(True, 0, 5, 0.0, 0)
        no_broker = compute_confidence(False, 0, 5, 0.0, 0)
        assert no_broker.score < ideal.score

    def test_high_unknown_ratio_drops_score(self):
        low_unknown = compute_confidence(True, 0, 10, 0.0, 0)
        high_unknown = compute_confidence(True, 8, 10, 0.0, 0)
        assert high_unknown.score < low_unknown.score

    def test_stale_reconcile_drops_score(self):
        fresh = compute_confidence(True, 0, 5, 0.0, 0)
        stale = compute_confidence(True, 0, 5, 1800.0, 0)  # 30 min
        assert stale.score < fresh.score

    def test_orphans_drop_score(self):
        no_orphan = compute_confidence(True, 0, 5, 0.0, 0)
        with_orphan = compute_confidence(True, 0, 5, 0.0, 3)
        assert with_orphan.score < no_orphan.score

    def test_score_always_in_unit_interval(self):
        for broker in [True, False]:
            for n_unk in [0, 3, 10]:
                for age in [0, 300, 3600]:
                    for orphans in [0, 5]:
                        cs = compute_confidence(broker, n_unk, 10, age, orphans)
                        assert 0.0 <= cs.score <= 1.0

    def test_worst_case_below_threshold(self):
        cs = compute_confidence(False, 10, 10, 7200.0, 10)
        assert cs.score < MIN_CONFIDENCE_TO_DEPLOY

    def test_decomposed_fields_populated(self):
        cs = compute_confidence(True, 2, 10, 60.0, 1)
        assert 0.0 < cs.broker_factor <= 1.0
        assert 0.0 <= cs.unknown_factor <= 1.0
        assert 0.0 < cs.staleness_factor <= 1.0
        assert 0.0 < cs.orphan_factor <= 1.0


# ── convenience wrapper ───────────────────────────────────────────────────────

class TestConfidenceFromTruthState:
    def test_no_records_no_orphans(self):
        cs = confidence_from_truth_state(
            broker_available=True,
            execution_records=[],
            orphan_records=[],
            reconcile_age_sec=0.0,
        )
        assert cs.n_total_active == 0
        assert cs.n_unknown == 0
        assert cs.score >= 0.90

    def test_unknown_record_counted(self):
        from src.truth.execution_truth import ExecutionRecord
        rec = ExecutionRecord(
            intent_id="i-001", symbol="1234", side="BUY",
            requested_qty=5, state="UNKNOWN",
        )
        cs = confidence_from_truth_state(True, [rec], [], 0.0)
        assert cs.n_unknown == 1
        assert cs.n_total_active == 1

    def test_terminal_not_counted_in_active(self):
        from src.truth.execution_truth import ExecutionRecord
        rec = ExecutionRecord(
            intent_id="i-001", symbol="1234", side="BUY",
            requested_qty=5, state="FILLED",
            submitted_qty=5, filled_qty=5,
        )
        cs = confidence_from_truth_state(True, [rec], [], 0.0)
        assert cs.n_total_active == 0
