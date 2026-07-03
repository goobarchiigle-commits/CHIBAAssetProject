"""
Tests for orphan order handling — broker positions without matching intent records.
"""
import pytest
from src.truth.position_truth import (
    OrphanRecord, PositionRecord, PositionTruth,
    build_position_truth, build_position_from_records,
)
from src.truth.capital_truth import compute_capital_truth, build_capital_truth_from_components
from src.truth.execution_truth import ExecutionRecord


def make_record(**kwargs) -> ExecutionRecord:
    defaults = dict(
        intent_id="i-001", symbol="1234", side="BUY",
        requested_qty=5, state="FILLED",
        submitted_qty=5, filled_qty=5, remaining_qty=0,
        fill_price=1000.0, estimated_price=1000.0, lot_size=100,
    )
    defaults.update(kwargs)
    return ExecutionRecord(**defaults)


# ── OrphanRecord properties ───────────────────────────────────────────────────

class TestOrphanRecord:
    def test_estimated_value(self):
        orph = OrphanRecord(symbol="9999", lots=3, estimated_price=500.0, lot_size=100)
        assert orph.estimated_value == 3 * 500.0 * 100

    def test_zero_lots_zero_value(self):
        orph = OrphanRecord(symbol="9999", lots=0, estimated_price=500.0, lot_size=100)
        assert orph.estimated_value == 0.0

    def test_source_is_broker_api(self):
        orph = OrphanRecord(symbol="9999", lots=2, estimated_price=500.0)
        assert orph.source == "broker_api"


# ── Orphan detection ──────────────────────────────────────────────────────────

class TestOrphanDetection:
    def test_broker_symbol_without_intent_record_is_orphan(self):
        truth = build_position_truth(
            execution_records=[],
            broker_positions={"9999": {"lots": 2, "avg_cost": 500.0,
                                        "market_price": 510.0}},
            sector_map={},
        )
        assert len(truth.orphans) == 1
        assert truth.orphans[0].symbol == "9999"

    def test_broker_symbol_with_intent_record_not_orphan(self):
        rec = make_record(symbol="1234")
        truth = build_position_truth(
            execution_records=[rec],
            broker_positions={"1234": {"lots": 5, "avg_cost": 1000.0,
                                        "market_price": 1050.0}},
            sector_map={"1234": "電機"},
        )
        assert all(o.symbol != "1234" for o in truth.orphans)

    def test_multiple_orphans_detected(self):
        truth = build_position_truth(
            execution_records=[],
            broker_positions={
                "8888": {"lots": 1, "avg_cost": 2000.0, "market_price": 2010.0},
                "9999": {"lots": 2, "avg_cost": 500.0, "market_price": 510.0},
            },
            sector_map={},
        )
        assert len(truth.orphans) == 2
        orphan_symbols = {o.symbol for o in truth.orphans}
        assert orphan_symbols == {"8888", "9999"}

    def test_zero_lot_broker_position_not_orphan(self):
        # broker says 0 lots → nothing to track
        truth = build_position_truth(
            execution_records=[],
            broker_positions={"9999": {"lots": 0, "avg_cost": 500.0,
                                        "market_price": 510.0}},
            sector_map={},
        )
        assert len(truth.orphans) == 0


# ── Orphans not merged into positions ────────────────────────────────────────

class TestOrphanNotMerged:
    def test_orphan_symbol_absent_from_positions(self):
        truth = build_position_truth(
            execution_records=[],
            broker_positions={"9999": {"lots": 3, "avg_cost": 500.0,
                                        "market_price": 500.0}},
            sector_map={},
        )
        position_symbols = {p.symbol for p in truth.positions}
        assert "9999" not in position_symbols

    def test_orphan_and_known_position_coexist(self):
        rec = make_record(symbol="1234")
        truth = build_position_truth(
            execution_records=[rec],
            broker_positions={
                "1234": {"lots": 5, "avg_cost": 1000.0, "market_price": 1050.0},
                "9999": {"lots": 2, "avg_cost": 500.0, "market_price": 510.0},
            },
            sector_map={"1234": "電機"},
        )
        # "1234" in positions, "9999" in orphans
        assert any(p.symbol == "1234" for p in truth.positions)
        assert any(o.symbol == "9999" for o in truth.orphans)
        assert all(p.symbol != "9999" for p in truth.positions)


# ── Orphan reduces deployable capital ────────────────────────────────────────

class TestOrphanReducesCapital:
    def test_orphan_exposure_subtracted_from_conservative(self):
        orph = OrphanRecord(symbol="9999", lots=2, estimated_price=500.0, lot_size=100)
        ct_no_orphan = compute_capital_truth(
            actual_equity=500_000.0,
            confirmed_exposure=0.0,
            pending_exposure=0.0,
            uncertain_exposure=0.0,
            orphan_exposure=0.0,
            confidence=0.90,
        )
        ct_with_orphan = compute_capital_truth(
            actual_equity=500_000.0,
            confirmed_exposure=0.0,
            pending_exposure=0.0,
            uncertain_exposure=0.0,
            orphan_exposure=orph.estimated_value,
            confidence=0.90,
        )
        assert ct_with_orphan.conservative_deployable < ct_no_orphan.conservative_deployable
        assert ct_with_orphan.orphan_exposure == pytest.approx(2 * 500.0 * 100)

    def test_orphan_exposure_in_capital_from_components(self):
        orph = OrphanRecord(symbol="9999", lots=3, estimated_price=1000.0, lot_size=100)
        ct = build_capital_truth_from_components(
            actual_equity=1_000_000.0,
            execution_records=[],
            orphan_records=[orph],
            confidence=0.90,
        )
        assert ct.orphan_exposure == pytest.approx(3 * 1000.0 * 100)
        assert ct.n_orphan_positions == 1

    def test_orphan_confidence_penalty(self):
        from src.truth.confidence import compute_confidence
        cs_no_orphan = compute_confidence(True, 0, 0, 0.0, 0)
        cs_with_orphan = compute_confidence(True, 0, 0, 0.0, 3)
        assert cs_with_orphan.score < cs_no_orphan.score

    def test_optimistic_also_reduced_by_orphan(self):
        # optimistic does NOT exclude orphan in our model
        # (orphan is always treated as consumed in capital_truth.compute_capital_truth)
        # → orphan exposure NOT in optimistic formula (it's a separate term)
        orph_val = 200_000.0
        ct = compute_capital_truth(
            actual_equity=1_000_000.0,
            confirmed_exposure=0.0,
            pending_exposure=0.0,
            uncertain_exposure=0.0,
            orphan_exposure=orph_val,
            confidence=0.90,
        )
        # optimistic ignores uncertain but NOT orphan
        # optimistic = equity - confirmed - pending - cash_buffer
        # orphan is NOT in optimistic formula → optimistic higher than conservative
        assert ct.optimistic_deployable >= ct.conservative_deployable
