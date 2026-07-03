"""
Tests for position_truth.py — authoritative position reconstruction.
"""
import pytest
from src.truth.execution_truth import ExecutionRecord
from src.truth.position_truth import (
    PositionRecord, OrphanRecord, PositionTruth,
    build_position_from_records, merge_broker_position, build_position_truth,
    save_position_truth, load_position_truth,
)


def make_record(**kwargs) -> ExecutionRecord:
    defaults = dict(
        intent_id="i-001",
        symbol="1234",
        side="BUY",
        requested_qty=5,
        state="FILLED",
        submitted_qty=5,
        filled_qty=5,
        remaining_qty=0,
        fill_price=1000.0,
        estimated_price=1000.0,
        trading_day="2026-05-16",
    )
    defaults.update(kwargs)
    return ExecutionRecord(**defaults)


# ── build_position_from_records ───────────────────────────────────────────────

class TestBuildPositionFromRecords:
    def test_filled_buy_goes_to_confirmed_lots(self):
        rec = make_record(filled_qty=5, fill_price=1000.0)
        pos = build_position_from_records("1234", "電機", [rec])
        assert pos.confirmed_lots == 5
        assert pos.pending_lots == 0
        assert pos.uncertain_lots == 0
        assert pos.avg_cost_price == 1000.0

    def test_partial_fill_splits_confirmed_and_pending(self):
        rec = make_record(state="PARTIAL_FILL", submitted_qty=10, filled_qty=4,
                          remaining_qty=6, fill_price=1000.0)
        pos = build_position_from_records("1234", "電機", [rec])
        assert pos.confirmed_lots == 4
        assert pos.pending_lots == 6

    def test_submitted_goes_to_pending(self):
        rec = make_record(state="SUBMITTED", submitted_qty=3, filled_qty=0,
                          remaining_qty=3, fill_price=0.0)
        pos = build_position_from_records("1234", "電機", [rec])
        assert pos.pending_lots == 3

    def test_unknown_goes_to_uncertain_lots(self):
        rec = make_record(state="UNKNOWN", submitted_qty=2, remaining_qty=2,
                          filled_qty=0, fill_price=0.0)
        pos = build_position_from_records("1234", "電機", [rec])
        assert pos.uncertain_lots == 2

    def test_sell_reduces_confirmed(self):
        buy = make_record(filled_qty=5, fill_price=1000.0)
        sell = make_record(intent_id="i-002", side="SELL", requested_qty=3,
                           submitted_qty=3, filled_qty=3, remaining_qty=0,
                           fill_price=1100.0)
        pos = build_position_from_records("1234", "電機", [buy, sell])
        assert pos.confirmed_lots == 2

    def test_source_is_inferred(self):
        rec = make_record()
        pos = build_position_from_records("1234", "電機", [rec])
        assert pos.source == "inferred"
        assert pos.confidence == pytest.approx(0.40)

    def test_empty_records_returns_zero(self):
        pos = build_position_from_records("9999", "不明", [])
        assert pos.confirmed_lots == 0
        assert pos.pending_lots == 0

    def test_weighted_avg_cost(self):
        r1 = make_record(intent_id="i-001", filled_qty=4, fill_price=1000.0)
        r2 = make_record(intent_id="i-002", filled_qty=6, fill_price=1100.0,
                         requested_qty=6)
        pos = build_position_from_records("1234", "電機", [r1, r2])
        # (4×1000 + 6×1100) / 10 = 1060
        assert abs(pos.avg_cost_price - 1060.0) < 0.01


# ── merge_broker_position ─────────────────────────────────────────────────────

class TestMergeBrokerPosition:
    def test_broker_overrides_confirmed_lots(self):
        pos = build_position_from_records("1234", "電機",
                                          [make_record(filled_qty=3)])
        merged = merge_broker_position(pos, broker_lots=5,
                                       broker_avg_cost=1050.0,
                                       broker_market_price=1060.0)
        assert merged.confirmed_lots == 5
        assert merged.avg_cost_price == 1050.0

    def test_broker_source_and_confidence(self):
        pos = build_position_from_records("1234", "電機", [make_record()])
        merged = merge_broker_position(pos, broker_lots=5,
                                       broker_avg_cost=1000.0,
                                       broker_market_price=1000.0)
        assert merged.source == "broker_api"
        assert merged.confidence == pytest.approx(0.95)

    def test_pending_lots_preserved(self):
        rec = make_record(state="PARTIAL_FILL", submitted_qty=10,
                          filled_qty=4, remaining_qty=6, fill_price=1000.0)
        pos = build_position_from_records("1234", "電機", [rec])
        merged = merge_broker_position(pos, broker_lots=4,
                                       broker_avg_cost=1000.0,
                                       broker_market_price=1000.0)
        assert merged.pending_lots == 6  # kept from journal


# ── build_position_truth ──────────────────────────────────────────────────────

class TestBuildPositionTruth:
    def test_basic_build(self):
        rec = make_record()
        truth = build_position_truth(
            execution_records=[rec],
            broker_positions={},
            sector_map={"1234": "電機"},
        )
        assert len(truth.positions) == 1
        assert truth.positions[0].symbol == "1234"

    def test_orphan_detection(self):
        # broker has a symbol we have no records for
        truth = build_position_truth(
            execution_records=[],
            broker_positions={"9999": {"lots": 3, "avg_cost": 500.0,
                                        "market_price": 520.0, "sector": "食品"}},
            sector_map={},
        )
        assert len(truth.orphans) == 1
        assert truth.orphans[0].symbol == "9999"
        assert truth.orphans[0].lots == 3

    def test_no_orphan_when_intent_exists(self):
        rec = make_record()
        truth = build_position_truth(
            execution_records=[rec],
            broker_positions={"1234": {"lots": 5, "avg_cost": 1000.0,
                                        "market_price": 1050.0}},
            sector_map={"1234": "電機"},
        )
        assert len(truth.orphans) == 0
        # broker overlay applied
        assert truth.positions[0].source == "broker_api"

    def test_source_worst_case(self):
        rec = make_record()
        truth = build_position_truth([rec], {}, {"1234": "電機"})
        assert truth.source == "inferred"  # no broker overlay


# ── Properties ────────────────────────────────────────────────────────────────

def test_position_record_confirmed_value():
    pos = PositionRecord(symbol="1234", confirmed_lots=5,
                         avg_cost_price=1000.0, lot_size=100)
    assert pos.confirmed_value == 5 * 1000.0 * 100

def test_orphan_estimated_value():
    orph = OrphanRecord(symbol="9999", lots=2, estimated_price=500.0, lot_size=100)
    assert orph.estimated_value == 2 * 500.0 * 100


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    rec = make_record()
    truth = build_position_truth([rec], {}, {"1234": "電機"})
    path = tmp_path / "pos.json"
    save_position_truth(truth, path)
    loaded = load_position_truth(path)
    assert loaded is not None
    assert len(loaded.positions) == 1
    assert loaded.positions[0].symbol == "1234"


def test_load_missing_file_returns_none(tmp_path):
    assert load_position_truth(tmp_path / "missing.json") is None
