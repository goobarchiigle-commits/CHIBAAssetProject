"""
Tests for execution_truth.py — monotonic state machine and exposure accounting.
"""
import pytest
from datetime import datetime, timezone, timedelta
from src.truth.execution_truth import (
    ExecutionRecord, ExposureBreakdown,
    InvalidTransitionError,
    VALID_TRANSITIONS, TERMINAL_STATES, UNKNOWN_STATES, PENDING_STATES,
    can_transition, transition_state, compute_exposure_breakdown,
    mark_unknown_stale, record_to_dict, record_from_dict,
    append_execution_record, load_execution_records,
)

JST = timezone(timedelta(hours=9))


def make_record(**kwargs) -> ExecutionRecord:
    defaults = dict(
        intent_id="intent-001",
        symbol="1234",
        side="BUY",
        requested_qty=10,
        state="CREATED",
        estimated_price=1000.0,
        trading_day="2026-05-16",
        created_at=datetime.now(JST).isoformat(),
        last_updated_at=datetime.now(JST).isoformat(),
    )
    defaults.update(kwargs)
    return ExecutionRecord(**defaults)


# ── State machine: valid transitions ──────────────────────────────────────────

class TestValidTransitions:
    def test_created_to_submitted(self):
        rec = make_record()
        new = transition_state(rec, "SUBMITTED", submitted_qty=10)
        assert new.state == "SUBMITTED"
        assert new.submitted_qty == 10
        assert new.remaining_qty == 10

    def test_submitted_to_acked(self):
        rec = make_record(state="SUBMITTED", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "ACKED", broker_order_id="B-001")
        assert new.state == "ACKED"
        assert new.broker_order_id == "B-001"

    def test_acked_to_filled(self):
        rec = make_record(state="ACKED", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "FILLED", filled_qty=10, fill_price=1050.0)
        assert new.state == "FILLED"
        assert new.filled_qty == 10
        assert new.fill_price == 1050.0
        assert new.remaining_qty == 0

    def test_acked_to_partial_fill(self):
        rec = make_record(state="ACKED", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1020.0)
        assert new.state == "PARTIAL_FILL"
        assert new.filled_qty == 4
        assert new.remaining_qty == 6

    def test_partial_fill_to_filled(self):
        rec = make_record(
            state="PARTIAL_FILL", submitted_qty=10, remaining_qty=6,
            filled_qty=4, fill_price=1020.0,
        )
        new = transition_state(rec, "FILLED", filled_qty=6, fill_price=1030.0)
        assert new.state == "FILLED"
        assert new.filled_qty == 10

    def test_submitted_to_unknown(self):
        rec = make_record(state="SUBMITTED", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "UNKNOWN")
        assert new.state == "UNKNOWN"
        assert new.unknown_since != ""

    def test_unknown_to_resolved_filled(self):
        rec = make_record(state="UNKNOWN", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "RESOLVED_FILLED", filled_qty=10, fill_price=1000.0)
        assert new.state == "RESOLVED_FILLED"
        assert new.is_terminal

    def test_unknown_to_resolved_cancelled(self):
        rec = make_record(state="UNKNOWN", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "RESOLVED_CANCELLED")
        assert new.state == "RESOLVED_CANCELLED"
        assert new.remaining_qty == 0

    def test_created_to_cancelled(self):
        rec = make_record()
        new = transition_state(rec, "CANCELLED")
        assert new.state == "CANCELLED"
        assert new.is_terminal

    def test_submitted_to_rejected(self):
        rec = make_record(state="SUBMITTED", submitted_qty=10, remaining_qty=10)
        new = transition_state(rec, "REJECTED")
        assert new.state == "REJECTED"
        assert new.is_terminal


# ── State machine: invalid transitions ───────────────────────────────────────

class TestInvalidTransitions:
    def test_created_to_filled_invalid(self):
        rec = make_record()
        with pytest.raises(InvalidTransitionError):
            transition_state(rec, "FILLED")

    def test_filled_is_terminal(self):
        rec = make_record(state="FILLED", submitted_qty=10, filled_qty=10)
        for state in ["SUBMITTED", "ACKED", "UNKNOWN", "CANCELLED"]:
            with pytest.raises(InvalidTransitionError):
                transition_state(rec, state)

    def test_unknown_cannot_go_to_submitted(self):
        rec = make_record(state="UNKNOWN")
        with pytest.raises(InvalidTransitionError):
            transition_state(rec, "SUBMITTED")

    def test_resolved_filled_is_terminal(self):
        rec = make_record(state="RESOLVED_FILLED")
        with pytest.raises(InvalidTransitionError):
            transition_state(rec, "UNKNOWN")

    def test_invalid_target_state(self):
        rec = make_record()
        with pytest.raises(InvalidTransitionError):
            transition_state(rec, "GHOST_STATE")


# ── Immutability ──────────────────────────────────────────────────────────────

def test_transition_immutable():
    rec = make_record()
    new = transition_state(rec, "SUBMITTED", submitted_qty=10)
    assert rec.state == "CREATED"    # original unchanged
    assert new.state == "SUBMITTED"


# ── Fill price: weighted average ──────────────────────────────────────────────

def test_weighted_avg_fill_price():
    rec = make_record(state="ACKED", submitted_qty=10, remaining_qty=10)
    p1 = transition_state(rec, "PARTIAL_FILL", filled_qty=4, fill_price=1000.0)
    p2 = transition_state(p1, "FILLED", filled_qty=6, fill_price=1100.0)
    # weighted avg: (4×1000 + 6×1100) / 10 = 10600/10 = 1060
    assert abs(p2.fill_price - 1060.0) < 0.01


# ── Derived properties ────────────────────────────────────────────────────────

def test_is_terminal():
    for state in TERMINAL_STATES:
        rec = make_record(state=state)
        assert rec.is_terminal

def test_is_unknown():
    rec = make_record(state="UNKNOWN")
    assert rec.is_unknown

def test_pending_value_only_for_pending_states():
    rec = make_record(state="ACKED", submitted_qty=10, remaining_qty=10,
                      estimated_price=500.0)
    # BUY ACKED with 10 remaining lots, lot_size=100, price=500 → 500_000
    assert rec.pending_value == 10 * 500.0 * 100

    filled_rec = make_record(state="FILLED", submitted_qty=10, filled_qty=10,
                             fill_price=500.0)
    assert filled_rec.pending_value == 0.0


# ── Exposure breakdown ────────────────────────────────────────────────────────

class TestExposureBreakdown:
    def test_filled_goes_to_confirmed(self):
        rec = make_record(state="FILLED", submitted_qty=5, filled_qty=5,
                          fill_price=2000.0)
        b = compute_exposure_breakdown([rec])
        assert b.confirmed_value == 5 * 2000.0 * 100

    def test_pending_goes_to_pending(self):
        rec = make_record(state="ACKED", submitted_qty=3, remaining_qty=3,
                          estimated_price=1500.0)
        b = compute_exposure_breakdown([rec])
        assert b.pending_value == 3 * 1500.0 * 100
        assert b.total_pending_orders == 1

    def test_unknown_goes_to_uncertain(self):
        rec = make_record(state="UNKNOWN", submitted_qty=2, remaining_qty=2,
                          estimated_price=1000.0)
        b = compute_exposure_breakdown([rec])
        assert b.uncertain_value == 2 * 1000.0 * 100
        assert b.total_uncertain_orders == 1

    def test_sell_excluded_from_exposure(self):
        rec = make_record(side="SELL", state="FILLED", submitted_qty=5,
                          filled_qty=5, fill_price=2000.0)
        b = compute_exposure_breakdown([rec])
        assert b.confirmed_value == 0.0


# ── mark_unknown_stale ────────────────────────────────────────────────────────

class TestMarkUnknownStale:
    def test_old_submitted_promoted_to_unknown(self):
        old_ts = datetime(2026, 5, 16, 8, 0, 0, tzinfo=JST).isoformat()
        rec = make_record(state="SUBMITTED", submitted_qty=5, remaining_qty=5,
                          last_updated_at=old_ts)
        now = datetime(2026, 5, 16, 9, 0, 0, tzinfo=JST)
        updated = mark_unknown_stale([rec], stale_after_minutes=30, now=now)
        assert updated[0].state == "UNKNOWN"

    def test_fresh_submitted_not_promoted(self):
        fresh_ts = datetime(2026, 5, 16, 8, 55, 0, tzinfo=JST).isoformat()
        rec = make_record(state="SUBMITTED", submitted_qty=5, remaining_qty=5,
                          last_updated_at=fresh_ts)
        now = datetime(2026, 5, 16, 9, 0, 0, tzinfo=JST)
        updated = mark_unknown_stale([rec], stale_after_minutes=30, now=now)
        assert updated[0].state == "SUBMITTED"

    def test_unknown_not_touched(self):
        old_ts = datetime(2026, 5, 16, 7, 0, 0, tzinfo=JST).isoformat()
        rec = make_record(state="UNKNOWN", submitted_qty=5, remaining_qty=5,
                          last_updated_at=old_ts)
        now = datetime(2026, 5, 16, 9, 0, 0, tzinfo=JST)
        updated = mark_unknown_stale([rec], stale_after_minutes=30, now=now)
        assert updated[0].state == "UNKNOWN"  # stays UNKNOWN, not re-promoted

    def test_terminal_not_touched(self):
        old_ts = datetime(2026, 5, 16, 7, 0, 0, tzinfo=JST).isoformat()
        rec = make_record(state="FILLED", submitted_qty=5, filled_qty=5,
                          fill_price=1000.0, last_updated_at=old_ts)
        now = datetime(2026, 5, 16, 9, 0, 0, tzinfo=JST)
        updated = mark_unknown_stale([rec], stale_after_minutes=30, now=now)
        assert updated[0].state == "FILLED"


# ── Persistence ───────────────────────────────────────────────────────────────

def test_roundtrip_record_dict():
    rec = make_record(state="PARTIAL_FILL", submitted_qty=10, filled_qty=4,
                      remaining_qty=6, fill_price=1020.0)
    d = record_to_dict(rec)
    rec2 = record_from_dict(d)
    assert rec.intent_id == rec2.intent_id
    assert rec.state == rec2.state
    assert rec.fill_price == rec2.fill_price


def test_last_write_wins(tmp_path):
    path = tmp_path / "exec.jsonl"
    r1 = make_record(state="SUBMITTED", submitted_qty=5, remaining_qty=5)
    r2 = make_record(state="ACKED", submitted_qty=5, remaining_qty=5,
                     broker_order_id="B-001")
    append_execution_record(r1, path)
    append_execution_record(r2, path)
    loaded = load_execution_records(path)
    assert len(loaded) == 1
    assert loaded[0].state == "ACKED"
