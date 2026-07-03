"""
Tests for invariants.py — cross-layer runtime assertions.
"""
import dataclasses
import pytest
from src.truth.execution_truth import ExecutionRecord
from src.truth.position_truth import PositionRecord, OrphanRecord, PositionTruth
from src.truth.capital_truth import compute_capital_truth, CapitalTruth
from src.truth.confidence import compute_confidence
from src.truth.invariants import (
    check_execution_invariants,
    check_position_invariants,
    check_capital_layer_invariants,
    check_cross_layer_invariants,
    check_all_invariants,
    assert_no_violations,
)
from src.truth.reconciler import TruthState


def clean_record(**kwargs) -> ExecutionRecord:
    defaults = dict(
        intent_id="i-001", symbol="1234", side="BUY",
        requested_qty=10, state="FILLED",
        submitted_qty=10, filled_qty=10, remaining_qty=0,
        fill_price=1000.0, estimated_price=1000.0,
    )
    defaults.update(kwargs)
    return ExecutionRecord(**defaults)


def clean_position(**kwargs) -> PositionRecord:
    defaults = dict(symbol="1234", confirmed_lots=5,
                    avg_cost_price=1000.0, confidence=0.95)
    defaults.update(kwargs)
    return PositionRecord(**defaults)


def make_capital_truth(**kwargs) -> CapitalTruth:
    defaults = dict(
        actual_equity=1_000_000.0,
        confirmed_exposure=0.0,
        pending_exposure=0.0,
        uncertain_exposure=0.0,
        orphan_exposure=0.0,
        confidence=0.90,
    )
    defaults.update(kwargs)
    return compute_capital_truth(**defaults)


def make_truth_state(records, positions, orphans, capital, confidence_score=0.90):
    pos_truth = PositionTruth(positions=positions, orphans=orphans)
    conf = compute_confidence(True, 0, 0, 0.0, len(orphans))
    return TruthState(
        execution_records=records,
        position_truth=pos_truth,
        capital_truth=capital,
        confidence=conf,
        reconciled_at="2026-05-16T09:00:00+09:00",
        trading_day="2026-05-16",
        broker_available=True,
        stale_orders_promoted=0,
        chain_valid=True,
        chain_violations=[],
    )


# ── Execution invariants ──────────────────────────────────────────────────────

class TestExecutionInvariants:
    def test_clean_record_no_violations(self):
        rec = clean_record()
        assert check_execution_invariants([rec]) == []

    def test_filled_exceeds_submitted_violation(self):
        rec = clean_record(filled_qty=15, submitted_qty=10, remaining_qty=0)
        violations = check_execution_invariants([rec])
        assert any("filled_qty" in v for v in violations)

    def test_submitted_exceeds_requested_violation(self):
        rec = clean_record(submitted_qty=15, requested_qty=10)
        violations = check_execution_invariants([rec])
        assert any("submitted_qty" in v for v in violations)

    def test_filled_qty_nonzero_but_zero_fill_price(self):
        rec = clean_record(filled_qty=5, fill_price=0.0)
        violations = check_execution_invariants([rec])
        assert any("fill_price" in v for v in violations)

    def test_duplicate_intent_id_violation(self):
        r1 = clean_record(state="SUBMITTED")
        r2 = clean_record(state="ACKED")
        violations = check_execution_invariants([r1, r2])
        assert any("duplicate" in v for v in violations)

    def test_negative_remaining_qty_violation(self):
        rec = clean_record(remaining_qty=-1)
        violations = check_execution_invariants([rec])
        assert any("remaining_qty" in v for v in violations)


# ── Position invariants ───────────────────────────────────────────────────────

class TestPositionInvariants:
    def test_clean_position_no_violations(self):
        pos = clean_position()
        truth = PositionTruth(positions=[pos], orphans=[])
        assert check_position_invariants(truth) == []

    def test_negative_confirmed_lots_violation(self):
        pos = clean_position(confirmed_lots=-1)
        truth = PositionTruth(positions=[pos], orphans=[])
        violations = check_position_invariants(truth)
        assert any("confirmed_lots" in v for v in violations)

    def test_confirmed_lots_with_zero_avg_cost_violation(self):
        pos = clean_position(confirmed_lots=5, avg_cost_price=0.0)
        truth = PositionTruth(positions=[pos], orphans=[])
        violations = check_position_invariants(truth)
        assert any("avg_cost" in v for v in violations)

    def test_orphan_symbol_not_in_positions(self):
        # Both have same symbol → invariant violation (orphan merged)
        pos = clean_position(symbol="9999")
        orph = OrphanRecord(symbol="9999", lots=2, estimated_price=500.0)
        truth = PositionTruth(positions=[pos], orphans=[orph])
        violations = check_position_invariants(truth)
        assert any("silently merged" in v or "orphan" in v.lower() for v in violations)


# ── Capital invariants (delegates to capital_truth.check_capital_invariants) ──

class TestCapitalLayerInvariants:
    def test_clean_capital_no_violations(self):
        ct = make_capital_truth()
        violations = check_capital_layer_invariants(ct)
        assert violations == []

    def test_violations_prefixed_with_cap(self):
        ct = make_capital_truth(confidence=0.50)
        ct2 = dataclasses.replace(ct, effective_deployable=100_000.0)
        violations = check_capital_layer_invariants(ct2)
        assert all(v.startswith("[CAP]") for v in violations)


# ── Cross-layer invariants ────────────────────────────────────────────────────

class TestCrossLayerInvariants:
    def test_clean_state_no_violations(self):
        rec = clean_record()
        ct = make_capital_truth(confirmed_exposure=10 * 1000.0 * 100)
        cs = compute_confidence(True, 0, 0, 0.0, 0)
        pos_truth = PositionTruth(positions=[], orphans=[])
        violations = check_cross_layer_invariants([rec], pos_truth, ct, cs)
        assert violations == []

    def test_exposure_mismatch_detected(self):
        rec = clean_record(filled_qty=5, submitted_qty=10)
        # capital truth says confirmed = 0 but breakdown says 5×1000×100
        ct = make_capital_truth(confirmed_exposure=0.0)
        cs = compute_confidence(True, 0, 0, 0.0, 0)
        pos_truth = PositionTruth(positions=[], orphans=[])
        violations = check_cross_layer_invariants([rec], pos_truth, ct, cs)
        assert any("confirmed_value" in v for v in violations)


# ── Full check ────────────────────────────────────────────────────────────────

class TestCheckAllInvariants:
    def test_clean_truth_state_no_violations(self):
        rec = clean_record()
        ct = make_capital_truth(
            confirmed_exposure=10 * 1000.0 * 100,
        )
        pos = clean_position()
        state = make_truth_state([rec], [pos], [], ct)
        violations = check_all_invariants(state)
        assert violations == [], violations

    def test_chain_violations_included(self):
        rec = clean_record()
        ct = make_capital_truth(confirmed_exposure=10 * 1000.0 * 100)
        pos = clean_position()
        state = make_truth_state([rec], [pos], [], ct)
        state2 = dataclasses.replace(
            state, chain_valid=False,
            chain_violations=["entry 1: tampered"]
        )
        violations = check_all_invariants(state2)
        assert any("[CHAIN]" in v for v in violations)


def test_assert_no_violations_raises_on_violation():
    rec = clean_record(filled_qty=15, submitted_qty=10, remaining_qty=0)
    ct = make_capital_truth()
    pos = clean_position()
    state = make_truth_state([rec], [pos], [], ct)
    with pytest.raises(RuntimeError):
        assert_no_violations(state, raise_on_violation=True)
