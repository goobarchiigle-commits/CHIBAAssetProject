"""
Tests for partial fill state reconstruction and exposure accounting.
"""
import pytest
from src.truth.execution_truth import (
    ExecutionRecord, transition_state, compute_exposure_breakdown,
)
from src.truth.position_truth import build_position_from_records
from src.truth.capital_truth import compute_capital_truth, build_capital_truth_from_components
from src.truth.position_truth import OrphanRecord


def make_record(**kwargs) -> ExecutionRecord:
    defaults = dict(
        intent_id="i-001", symbol="1234", side="BUY",
        requested_qty=10, state="ACKED",
        submitted_qty=10, filled_qty=0, remaining_qty=10,
        fill_price=0.0, estimated_price=1000.0, lot_size=100,
        trading_day="2026-05-16",
    )
    defaults.update(kwargs)
    return ExecutionRecord(**defaults)


# ── PARTIAL_FILL state transitions ────────────────────────────────────────────

class TestPartialFillTransition:
    def test_partial_fill_sets_filled_and_remaining(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        assert p.filled_qty == 4
        assert p.remaining_qty == 6

    def test_partial_fill_then_more_partial(self):
        rec = make_record()
        p1 = transition_state(rec, "PARTIAL_FILL", filled_qty=3, fill_price=990.0)
        p2 = transition_state(p1, "PARTIAL_FILL", filled_qty=3, fill_price=1010.0)
        assert p2.filled_qty == 6
        assert p2.remaining_qty == 4
        # weighted avg: (3×990 + 3×1010)/6 = 1000
        assert abs(p2.fill_price - 1000.0) < 0.01

    def test_partial_fill_then_filled(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=7, fill_price=1000.0)
        f = transition_state(p, "FILLED", filled_qty=3, fill_price=1010.0)
        assert f.state == "FILLED"
        assert f.remaining_qty == 0
        assert f.filled_qty == 10

    def test_partial_fill_then_cancelled(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        c = transition_state(p, "CANCELLED")
        assert c.state == "CANCELLED"
        assert c.is_terminal

    def test_partial_fill_then_unknown(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        u = transition_state(p, "UNKNOWN")
        assert u.state == "UNKNOWN"
        assert u.is_unknown


# ── Exposure breakdown for partial fills ──────────────────────────────────────

class TestPartialFillExposure:
    def test_partial_fill_splits_confirmed_and_pending(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        b = compute_exposure_breakdown([p])
        # confirmed: 4 lots × 1000 fill_price × 100 lot_size
        assert b.confirmed_value == pytest.approx(4 * 1000.0 * 100)
        # pending: 6 remaining × 1000 estimated × 100
        assert b.pending_value == pytest.approx(6 * 1000.0 * 100)

    def test_partial_fill_unknown_goes_to_uncertain(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        u = transition_state(p, "UNKNOWN")
        b = compute_exposure_breakdown([u])
        # confirmed: 4 lots × 1000 × 100 (already filled portion)
        assert b.confirmed_value == pytest.approx(4 * 1000.0 * 100)
        # uncertain: 6 remaining × estimated × 100
        assert b.uncertain_value == pytest.approx(6 * 1000.0 * 100)

    def test_filled_moves_all_to_confirmed(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        f = transition_state(p, "FILLED", filled_qty=6, fill_price=1010.0)
        b = compute_exposure_breakdown([f])
        assert b.confirmed_value == pytest.approx(10 * 1005.0 * 100, rel=0.01)
        assert b.pending_value == 0.0
        assert b.uncertain_value == 0.0


# ── Position reconstruction from partial fills ────────────────────────────────

class TestPositionFromPartialFill:
    def test_partial_fill_in_position(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        pos = build_position_from_records("1234", "電機", [p])
        assert pos.confirmed_lots == 4
        assert pos.pending_lots == 6
        assert pos.avg_cost_price == pytest.approx(1000.0)

    def test_multiple_partials_accumulated(self):
        rec = make_record()
        p1 = transition_state(rec, "PARTIAL_FILL", filled_qty=3, fill_price=1000.0)
        p2 = transition_state(p1, "PARTIAL_FILL", filled_qty=3, fill_price=1100.0)
        # Two separate records for same intent (last-write-wins scenario)
        pos = build_position_from_records("1234", "電機", [p2])
        assert pos.confirmed_lots == 6
        assert pos.pending_lots == 4


# ── Capital truth under partial fill ─────────────────────────────────────────

class TestCapitalTruthPartialFill:
    def test_pending_portion_reduces_deployable(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        ct = build_capital_truth_from_components(
            actual_equity=1_000_000.0,
            execution_records=[p],
            orphan_records=[],
            confidence=0.90,
        )
        # pending exposure = 6 × 1000 × 100 = 600_000
        assert ct.pending_exposure == pytest.approx(6 * 1000.0 * 100)
        # confirmed exposure = 4 × 1000 × 100 = 400_000
        assert ct.confirmed_exposure == pytest.approx(4 * 1000.0 * 100)

    def test_partial_fill_unknown_conservative_deploys_less(self):
        rec = make_record()
        p = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
        u = transition_state(p, "UNKNOWN")
        ct = build_capital_truth_from_components(
            actual_equity=1_000_000.0,
            execution_records=[u],
            orphan_records=[],
            confidence=0.90,
        )
        assert ct.conservative_deployable <= ct.optimistic_deployable
        assert ct.uncertain_exposure == pytest.approx(6 * 1000.0 * 100)
