"""
Tests for capital_truth.py — three deployable capital views under uncertainty.
"""
import pytest
from src.truth.capital_truth import (
    CapitalTruth, compute_capital_truth, build_capital_truth_from_components,
    check_capital_invariants, save_capital_truth, load_capital_truth,
    MIN_CONFIDENCE_TO_DEPLOY, CASH_BUFFER_DEFAULT,
)
from src.truth.position_truth import OrphanRecord


def make_ct(**kwargs) -> CapitalTruth:
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


# ── Three-view computation ────────────────────────────────────────────────────

class TestCapitalViews:
    def test_conservative_lte_optimistic(self):
        ct = make_ct(uncertain_exposure=100_000.0)
        assert ct.conservative_deployable <= ct.optimistic_deployable

    def test_uncertain_reduces_conservative_only(self):
        ct_no_uncertain = make_ct(uncertain_exposure=0.0)
        ct_with_uncertain = make_ct(uncertain_exposure=100_000.0,
                                    n_unknown_orders=1)
        # optimistic same; conservative lower
        assert ct_no_uncertain.optimistic_deployable == pytest.approx(
            ct_with_uncertain.optimistic_deployable, abs=1.0
        )
        assert ct_with_uncertain.conservative_deployable < ct_no_uncertain.conservative_deployable

    def test_cash_buffer_subtracted_from_both(self):
        ct = make_ct(actual_equity=1_000_000.0)
        expected_buffer = 1_000_000.0 * CASH_BUFFER_DEFAULT
        assert ct.cash_buffer_amount == pytest.approx(expected_buffer, rel=0.01)
        assert ct.optimistic_deployable <= 1_000_000.0 - expected_buffer

    def test_effective_in_conservative_optimistic_range(self):
        ct = make_ct(uncertain_exposure=100_000.0, confidence=0.80)
        assert ct.conservative_deployable <= ct.effective_deployable
        assert ct.effective_deployable <= ct.optimistic_deployable

    def test_effective_never_negative(self):
        ct = make_ct(confirmed_exposure=900_000.0, pending_exposure=200_000.0,
                     uncertain_exposure=100_000.0, confidence=0.85)
        assert ct.effective_deployable >= 0.0


# ── Confidence gate ───────────────────────────────────────────────────────────

class TestConfidenceGate:
    def test_below_threshold_blocks_deployment(self):
        ct = make_ct(confidence=0.50)
        assert ct.deployment_allowed is False
        assert ct.effective_deployable == 0.0

    def test_exactly_at_threshold_allowed(self):
        ct = make_ct(confidence=MIN_CONFIDENCE_TO_DEPLOY)
        assert ct.deployment_allowed is True

    def test_above_threshold_allowed(self):
        ct = make_ct(confidence=0.90)
        assert ct.deployment_allowed is True

    def test_block_reason_set_when_blocked(self):
        ct = make_ct(confidence=0.50)
        assert "confidence" in ct.deployment_block_reason

    def test_unknown_orders_force_conservative(self):
        ct = make_ct(uncertain_exposure=100_000.0, confidence=0.90,
                     n_unknown_orders=2)
        # deployment allowed but effective capped at conservative
        assert ct.deployment_allowed is True
        assert ct.effective_deployable <= ct.conservative_deployable + 1.0

    def test_zero_equity_yields_zero_deployable(self):
        ct = make_ct(actual_equity=0.0)
        assert ct.effective_deployable == 0.0
        assert ct.conservative_deployable == 0.0


# ── Invariant checker ─────────────────────────────────────────────────────────

class TestCapitalInvariants:
    def test_clean_state_no_violations(self):
        ct = make_ct()
        violations = check_capital_invariants(ct)
        assert violations == []

    def test_negative_effective_violates(self):
        ct = make_ct()
        import dataclasses
        ct2 = dataclasses.replace(ct, effective_deployable=-1.0)
        violations = check_capital_invariants(ct2)
        assert any("effective_deployable" in v for v in violations)

    def test_deployment_false_but_effective_positive_violates(self):
        ct = make_ct(confidence=0.50)
        import dataclasses
        ct2 = dataclasses.replace(ct, effective_deployable=50_000.0)
        violations = check_capital_invariants(ct2)
        assert any("deployment_allowed" in v for v in violations)

    def test_confidence_out_of_range_violates(self):
        ct = make_ct()
        import dataclasses
        ct2 = dataclasses.replace(ct, confidence=1.5)
        violations = check_capital_invariants(ct2)
        assert any("confidence" in v for v in violations)


# ── build_capital_truth_from_components ───────────────────────────────────────

class TestBuildFromComponents:
    def test_from_filled_records(self):
        from src.truth.execution_truth import ExecutionRecord
        rec = ExecutionRecord(
            intent_id="i-001", symbol="1234", side="BUY",
            requested_qty=5, state="FILLED",
            submitted_qty=5, filled_qty=5, remaining_qty=0,
            fill_price=1000.0, estimated_price=1000.0, lot_size=100,
        )
        orphans: list[OrphanRecord] = []
        ct = build_capital_truth_from_components(
            actual_equity=1_000_000.0,
            execution_records=[rec],
            orphan_records=orphans,
            confidence=0.95,
        )
        assert ct.confirmed_exposure == pytest.approx(5 * 1000.0 * 100)
        assert ct.deployment_allowed is True

    def test_orphan_exposure_consumed(self):
        from src.truth.execution_truth import ExecutionRecord
        orph = OrphanRecord(symbol="9999", lots=2, estimated_price=500.0, lot_size=100)
        ct = build_capital_truth_from_components(
            actual_equity=500_000.0,
            execution_records=[],
            orphan_records=[orph],
            confidence=0.90,
        )
        assert ct.orphan_exposure == pytest.approx(2 * 500.0 * 100)
        assert ct.conservative_deployable <= 500_000.0 - 2 * 500.0 * 100


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    ct = make_ct()
    path = tmp_path / "cap.json"
    save_capital_truth(ct, path)
    loaded = load_capital_truth(path)
    assert loaded is not None
    assert loaded.actual_equity == ct.actual_equity
    assert loaded.effective_deployable == ct.effective_deployable


def test_load_missing_returns_none(tmp_path):
    assert load_capital_truth(tmp_path / "missing.json") is None
