"""
Tests for reconciler.py — full truth pipeline orchestration.
"""
import pytest
from pathlib import Path
from src.truth.execution_truth import ExecutionRecord
from src.truth.intent_snapshot import make_snapshot, append_snapshot, GENESIS_HASH
from src.truth.reconciler import reconcile, reconcile_from_records, TruthState


def make_record(state="FILLED", **kwargs) -> ExecutionRecord:
    defaults = dict(
        intent_id="i-001",
        symbol="1234",
        side="BUY",
        requested_qty=5,
        state=state,
        submitted_qty=5,
        filled_qty=5 if state == "FILLED" else 0,
        remaining_qty=0 if state == "FILLED" else 5,
        fill_price=1000.0 if state == "FILLED" else 0.0,
        estimated_price=1000.0,
        trading_day="2026-05-16",
    )
    defaults.update(kwargs)
    return ExecutionRecord(**defaults)


def make_snapshot_chain(transitions: list[tuple[str, str]], tmp_path: Path) -> Path:
    """Helper: write intent snapshot chain to tmp_path and return file path."""
    path = tmp_path / "snaps.jsonl"
    prev_hash = GENESIS_HASH
    for i, (prev_state, new_state) in enumerate(transitions):
        snap = make_snapshot(
            intent_id="i-001",
            symbol="1234",
            side="BUY",
            requested_qty=5,
            prev_state=prev_state,
            new_state=new_state,
            reason="test",
            trading_day="2026-05-16",
            strategy="turtle",
            prev_hash=prev_hash,
        )
        append_snapshot(snap, path)
        prev_hash = snap.entry_hash
    return path


# ── reconcile (from snapshots) ────────────────────────────────────────────────

class TestReconcileFromSnapshots:
    def test_basic_reconcile(self, tmp_path):
        path = make_snapshot_chain(
            [("CREATED", "SUBMITTED"), ("SUBMITTED", "ACKED"), ("ACKED", "FILLED")],
            tmp_path,
        )
        state = reconcile(
            snapshot_path=path,
            broker_positions={},
            actual_equity=1_000_000.0,
            broker_available=False,
            sector_map={"1234": "電機"},
        )
        assert isinstance(state, TruthState)
        assert any(r.intent_id == "i-001" for r in state.execution_records)
        rec = next(r for r in state.execution_records if r.intent_id == "i-001")
        assert rec.state == "FILLED"

    def test_empty_snapshot_file_yields_empty_records(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        state = reconcile(
            snapshot_path=path,
            broker_positions={},
            actual_equity=1_000_000.0,
            broker_available=False,
            sector_map={},
        )
        assert state.execution_records == []

    def test_chain_valid_flag_set(self, tmp_path):
        path = make_snapshot_chain(
            [("CREATED", "SUBMITTED"), ("SUBMITTED", "FILLED")],
            tmp_path,
        )
        state = reconcile(path, {}, 1_000_000.0, False, {})
        assert state.chain_valid is True
        assert state.chain_violations == []

    def test_broker_overlay_applied_when_available(self, tmp_path):
        path = make_snapshot_chain(
            [("CREATED", "SUBMITTED"), ("SUBMITTED", "FILLED")],
            tmp_path,
        )
        broker_pos = {"1234": {"lots": 5, "avg_cost": 1050.0,
                                "market_price": 1060.0, "sector": "電機"}}
        state = reconcile(path, broker_pos, 1_000_000.0, True, {})
        pos = next((p for p in state.position_truth.positions
                    if p.symbol == "1234"), None)
        assert pos is not None
        assert pos.source == "broker_api"

    def test_broker_not_applied_when_unavailable(self, tmp_path):
        path = make_snapshot_chain(
            [("CREATED", "SUBMITTED"), ("SUBMITTED", "FILLED")],
            tmp_path,
        )
        broker_pos = {"1234": {"lots": 999, "avg_cost": 1050.0,
                                "market_price": 1060.0}}
        state = reconcile(path, broker_pos, 1_000_000.0, False, {"1234": "電機"})
        pos = next((p for p in state.position_truth.positions
                    if p.symbol == "1234"), None)
        if pos:
            assert pos.source == "inferred"


# ── reconcile_from_records ────────────────────────────────────────────────────

class TestReconcileFromRecords:
    def test_filled_record_deployable(self):
        rec = make_record(state="FILLED")
        state = reconcile_from_records(
            execution_records=[rec],
            broker_positions={},
            actual_equity=1_000_000.0,
            broker_available=True,
            sector_map={"1234": "電機"},
        )
        assert state.capital_truth.confirmed_exposure == pytest.approx(5 * 1000.0 * 100)
        assert state.deployment_allowed is True

    def test_orphan_detected(self):
        # No records for 9999, but broker reports it
        state = reconcile_from_records(
            execution_records=[],
            broker_positions={"9999": {"lots": 2, "avg_cost": 500.0,
                                        "market_price": 500.0}},
            actual_equity=500_000.0,
            broker_available=True,
            sector_map={},
        )
        assert state.n_orphan == 1
        assert state.position_truth.orphans[0].symbol == "9999"

    def test_stale_pending_promoted_to_unknown(self):
        from datetime import datetime, timezone, timedelta
        JST = timezone(timedelta(hours=9))
        old_ts = datetime(2026, 5, 16, 8, 0, 0, tzinfo=JST).isoformat()
        rec = make_record(
            state="SUBMITTED", submitted_qty=5, remaining_qty=5,
            filled_qty=0, fill_price=0.0, last_updated_at=old_ts,
        )
        state = reconcile_from_records(
            execution_records=[rec],
            broker_positions={},
            actual_equity=1_000_000.0,
            broker_available=True,
            sector_map={"1234": "電機"},
            stale_after_minutes=1,  # very short threshold
        )
        assert state.stale_orders_promoted >= 1
        assert any(r.state == "UNKNOWN" for r in state.execution_records)

    def test_low_confidence_blocks_deployment(self):
        state = reconcile_from_records(
            execution_records=[],
            broker_positions={},
            actual_equity=1_000_000.0,
            broker_available=False,
            sector_map={},
            reconcile_age_sec=7200.0,  # 2 hours stale
        )
        # Broker unavailable + 2h stale → confidence < 0.70
        assert state.deployment_allowed is False
        assert state.effective_deployable == 0.0

    def test_truth_state_properties(self):
        rec = make_record(state="UNKNOWN", submitted_qty=5, remaining_qty=5,
                          filled_qty=0, fill_price=0.0)
        state = reconcile_from_records(
            execution_records=[rec],
            broker_positions={},
            actual_equity=1_000_000.0,
            broker_available=True,
            sector_map={},
        )
        assert state.n_unknown == 1
        assert state.n_pending == 0

    def test_confidence_score_attached(self):
        state = reconcile_from_records([], {}, 1_000_000.0, True, {})
        assert 0.0 <= state.confidence.score <= 1.0
