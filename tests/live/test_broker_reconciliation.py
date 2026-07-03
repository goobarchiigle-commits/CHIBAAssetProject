"""
tests/live/test_broker_reconciliation.py

Tests for the broker reconciliation foundation:
  - src/live/reconciliation_engine.py
  - src/live/broker_truth_snapshot.py
  - src/live/execution_integrity_validator.py
  - src/live/replay_consistency_validator.py
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from src.live.reconciliation_engine import (
    MISMATCH_DUPLICATE_PENDING,
    MISMATCH_ORPHAN_FILL,
    MISMATCH_OVERNIGHT_CARRY,
    MISMATCH_RUNTIME_BROKER,
    MISMATCH_STALE_PENDING,
    SEVERITY_BLOCKING,
    SEVERITY_WARNING,
    ReconciliationMismatch,
    ReconciliationResult,
    _detect_duplicate_pending,
    _detect_orphan_fills,
    _detect_overnight_carry,
    _detect_runtime_broker_mismatch,
    _detect_stale_pending,
    append_reconciliation_result,
    load_reconciliation_results,
    run_reconciliation,
)
from src.live.broker_truth_snapshot import (
    BrokerTruthSnapshot,
    SNAPSHOT_STALE_SEC,
    _compute_checksum,
    append_snapshot,
    is_snapshot_stale,
    load_snapshots,
    take_snapshot,
    validate_snapshot,
)
from src.live.execution_integrity_validator import (
    VIOLATION_BAD_CHECKSUM,
    VIOLATION_BLOCKING_MISMATCH,
    VIOLATION_NO_SNAPSHOT,
    VIOLATION_STALE_SNAPSHOT,
    ExecutionIntegrityResult,
    append_integrity_result,
    load_integrity_results,
    validate_execution_integrity,
)
from src.live.replay_consistency_validator import (
    DIVERGENCE_CHECKSUM_INVALID,
    DIVERGENCE_DUPLICATE_TS,
    DIVERGENCE_MISSING_SNAPSHOT,
    DIVERGENCE_NON_MONOTONIC_TS,
    ReplayConsistencyResult,
    append_replay_result,
    load_replay_results,
    validate_replay_consistency,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(dt: datetime) -> str:
    return dt.isoformat()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _make_order(
    symbol: str,
    side: str = "BUY",
    state: str = "PENDING_SUBMIT",
    client_order_id: str = "",
    registered_at: str = "",
    trading_day: str = "",
    qty: int = 100,
) -> MagicMock:
    o = MagicMock()
    o.symbol = symbol
    o.side = side
    o.state = state
    o.client_order_id = client_order_id or f"COI-{symbol}-{state[:4]}"
    o.registered_at = registered_at or _ts(_now_utc())
    o.trading_day = trading_day or _now_utc().strftime("%Y-%m-%d")
    o.qty = qty
    return o


def _make_snapshot(
    run_id: str = "run1",
    timestamp: str = "",
    positions_count: int = 0,
    checksum_tamper: bool = False,
) -> BrokerTruthSnapshot:
    ts = timestamp or _ts(_now_utc())
    snap = BrokerTruthSnapshot(
        timestamp=ts,
        run_id=run_id,
        portfolio_mode="live",
        current_positions_count=positions_count,
        open_slots=3 - positions_count,
        available_cash=1_000_000.0,
        current_drawdown=0.0,
        positions_raw={},
        wallet_raw={},
        cb_state="NORMAL",
        checksum="",
    )
    snap.checksum = _compute_checksum(snap)
    if checksum_tamper:
        snap.checksum = "deadbeef" * 8
    return snap


def _make_recon_result(
    run_id: str = "run1",
    blocking: int = 0,
    timestamp: str = "",
) -> ReconciliationResult:
    return ReconciliationResult(
        timestamp=timestamp or _ts(_now_utc()),
        run_id=run_id,
        mismatches=[],
        broker_positions={},
        runtime_positions={},
        inflight_summary={},
        reconciliation_ok=blocking == 0,
        blocking_mismatches=blocking,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — duplicate pending detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDuplicatePending:
    def test_no_orders_returns_empty(self):
        assert _detect_duplicate_pending([]) == []

    def test_single_pending_no_duplicate(self):
        orders = [_make_order("1234.T", state="PENDING_SUBMIT")]
        assert _detect_duplicate_pending(orders) == []

    def test_two_pending_same_symbol_side_is_blocking(self):
        orders = [
            _make_order("1234.T", state="PENDING_SUBMIT", client_order_id="COI-1"),
            _make_order("1234.T", state="SUBMITTED_UNKNOWN", client_order_id="COI-2"),
        ]
        result = _detect_duplicate_pending(orders)
        assert len(result) == 1
        assert result[0].mismatch_type == MISMATCH_DUPLICATE_PENDING
        assert result[0].severity == SEVERITY_BLOCKING
        assert result[0].symbol == "1234.T"

    def test_two_pending_different_symbols_no_duplicate(self):
        orders = [
            _make_order("1234.T", state="PENDING_SUBMIT"),
            _make_order("5678.T", state="PENDING_SUBMIT"),
        ]
        assert _detect_duplicate_pending(orders) == []

    def test_terminal_states_not_counted(self):
        orders = [
            _make_order("1234.T", state="RECONCILED"),
            _make_order("1234.T", state="PENDING_SUBMIT"),
        ]
        assert _detect_duplicate_pending(orders) == []

    def test_failed_state_not_counted(self):
        orders = [
            _make_order("1234.T", state="FAILED"),
            _make_order("1234.T", state="SUBMITTED_UNKNOWN"),
        ]
        assert _detect_duplicate_pending(orders) == []

    def test_three_pending_same_symbol_still_one_mismatch(self):
        orders = [
            _make_order("1234.T", state="PENDING_SUBMIT", client_order_id="C1"),
            _make_order("1234.T", state="PENDING_SUBMIT", client_order_id="C2"),
            _make_order("1234.T", state="PENDING_SUBMIT", client_order_id="C3"),
        ]
        result = _detect_duplicate_pending(orders)
        assert len(result) == 1
        assert "count=3" in result[0].detail

    def test_different_sides_not_duplicate(self):
        orders = [
            _make_order("1234.T", side="BUY", state="PENDING_SUBMIT"),
            _make_order("1234.T", side="SELL", state="PENDING_SUBMIT"),
        ]
        # BUY:1234.T vs SELL:1234.T are different keys
        assert _detect_duplicate_pending(orders) == []


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — stale pending detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectStalePending:
    def test_fresh_order_not_stale(self):
        orders = [_make_order("1234.T", state="PENDING_SUBMIT")]
        result = _detect_stale_pending(orders, time.time(), 900.0)
        assert result == []

    def test_old_order_is_stale_warning(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=1000))
        orders = [_make_order("1234.T", state="PENDING_SUBMIT", registered_at=old_ts)]
        result = _detect_stale_pending(orders, time.time(), 900.0)
        assert len(result) == 1
        assert result[0].mismatch_type == MISMATCH_STALE_PENDING
        assert result[0].severity == SEVERITY_WARNING

    def test_terminal_order_not_stale(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=1000))
        orders = [_make_order("1234.T", state="RECONCILED", registered_at=old_ts)]
        result = _detect_stale_pending(orders, time.time(), 900.0)
        assert result == []

    def test_missing_registered_at_treated_as_stale(self):
        # Use MagicMock directly so registered_at="" isn't replaced by helper
        o = MagicMock()
        o.symbol = "1234.T"
        o.side = "BUY"
        o.state = "SUBMITTED_UNKNOWN"
        o.client_order_id = "COI-test"
        o.registered_at = ""   # explicitly empty
        result = _detect_stale_pending([o], time.time(), 900.0)
        assert len(result) == 1

    def test_acked_order_not_checked(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=1000))
        orders = [_make_order("1234.T", state="ACKED", registered_at=old_ts)]
        result = _detect_stale_pending(orders, time.time(), 900.0)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — orphan fill detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectOrphanFills:
    def test_no_orders_empty(self):
        assert _detect_orphan_fills([], "2026-05-17") == []

    def test_submitted_unknown_prior_day_is_orphan(self):
        yesterday = "2026-05-16"
        o = _make_order(
            "1234.T",
            state="SUBMITTED_UNKNOWN",
            registered_at=f"{yesterday}T09:00:00+09:00",
        )
        result = _detect_orphan_fills([o], "2026-05-17")
        assert len(result) == 1
        assert result[0].mismatch_type == MISMATCH_ORPHAN_FILL
        assert result[0].severity == SEVERITY_BLOCKING

    def test_submitted_unknown_same_day_not_orphan(self):
        today = "2026-05-17"
        o = _make_order(
            "1234.T",
            state="SUBMITTED_UNKNOWN",
            registered_at=f"{today}T09:00:00+09:00",
        )
        result = _detect_orphan_fills([o], today)
        assert result == []

    def test_pending_submit_not_orphan(self):
        yesterday = "2026-05-16"
        o = _make_order(
            "1234.T",
            state="PENDING_SUBMIT",
            registered_at=f"{yesterday}T09:00:00+09:00",
        )
        result = _detect_orphan_fills([o], "2026-05-17")
        assert result == []

    def test_reconciled_not_orphan(self):
        yesterday = "2026-05-16"
        o = _make_order(
            "1234.T",
            state="RECONCILED",
            registered_at=f"{yesterday}T09:00:00+09:00",
        )
        result = _detect_orphan_fills([o], "2026-05-17")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — runtime/broker mismatch detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectRuntimeBrokerMismatch:
    def test_empty_both_no_mismatch(self):
        assert _detect_runtime_broker_mismatch({}, {}) == []

    def test_matching_positions_no_mismatch(self):
        broker = {"1234.T": 100}
        runtime = {"1234.T": 100}
        assert _detect_runtime_broker_mismatch(broker, runtime) == []

    def test_qty_mismatch_is_warning(self):
        broker = {"1234.T": 200}
        runtime = {"1234.T": 100}
        result = _detect_runtime_broker_mismatch(broker, runtime)
        assert len(result) == 1
        assert result[0].mismatch_type == MISMATCH_RUNTIME_BROKER
        assert result[0].severity == SEVERITY_WARNING
        assert result[0].broker_qty == 200
        assert result[0].runtime_qty == 100

    def test_symbol_in_broker_not_runtime(self):
        broker = {"1234.T": 100}
        runtime = {}
        result = _detect_runtime_broker_mismatch(broker, runtime)
        assert len(result) == 1
        assert result[0].symbol == "1234.T"

    def test_symbol_in_runtime_not_broker(self):
        broker = {}
        runtime = {"5678.T": 100}
        result = _detect_runtime_broker_mismatch(broker, runtime)
        assert len(result) == 1

    def test_multiple_mismatches(self):
        broker = {"A.T": 100, "B.T": 0}
        runtime = {"A.T": 50, "B.T": 200}
        result = _detect_runtime_broker_mismatch(broker, runtime)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — overnight carry detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectOvernightCarry:
    def test_no_orders_empty(self):
        assert _detect_overnight_carry({}, [], "2026-05-17") == []

    def test_buy_for_already_held_is_warning(self):
        today = "2026-05-17"
        broker = {"1234.T": 100}
        o = _make_order("1234.T", side="BUY", state="PENDING_SUBMIT", trading_day=today)
        result = _detect_overnight_carry(broker, [o], today)
        assert len(result) == 1
        assert result[0].mismatch_type == MISMATCH_OVERNIGHT_CARRY
        assert result[0].severity == SEVERITY_WARNING

    def test_sell_for_held_not_flagged(self):
        today = "2026-05-17"
        broker = {"1234.T": 100}
        o = _make_order("1234.T", side="SELL", state="PENDING_SUBMIT", trading_day=today)
        assert _detect_overnight_carry(broker, [o], today) == []

    def test_buy_for_not_held_not_flagged(self):
        today = "2026-05-17"
        broker = {}
        o = _make_order("1234.T", side="BUY", state="PENDING_SUBMIT", trading_day=today)
        assert _detect_overnight_carry(broker, [o], today) == []

    def test_prior_day_order_not_flagged(self):
        today = "2026-05-17"
        broker = {"1234.T": 100}
        o = _make_order("1234.T", side="BUY", state="PENDING_SUBMIT", trading_day="2026-05-16")
        assert _detect_overnight_carry(broker, [o], today) == []


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — run_reconciliation (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunReconciliation:
    def test_clean_state_reconciliation_ok(self):
        result = run_reconciliation({}, {}, [], run_id="r1")
        assert result.reconciliation_ok is True
        assert result.blocking_mismatches == 0
        assert result.mismatches == []

    def test_duplicate_pending_makes_blocking(self):
        orders = [
            _make_order("1234.T", state="PENDING_SUBMIT", client_order_id="C1"),
            _make_order("1234.T", state="PENDING_SUBMIT", client_order_id="C2"),
        ]
        result = run_reconciliation({}, {}, orders, run_id="r1")
        assert result.reconciliation_ok is False
        assert result.blocking_mismatches >= 1

    def test_stale_pending_not_blocking(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=1200))
        orders = [_make_order("1234.T", state="PENDING_SUBMIT", registered_at=old_ts)]
        result = run_reconciliation({}, {}, orders, run_id="r1")
        assert result.reconciliation_ok is True   # stale = warning only
        assert len(result.mismatches) >= 1

    def test_result_has_run_id(self):
        result = run_reconciliation({}, {}, [], run_id="test-run-42")
        assert result.run_id == "test-run-42"

    def test_inflight_summary_populated(self):
        orders = [
            _make_order("A.T", state="PENDING_SUBMIT"),
            _make_order("B.T", state="ACKED"),
        ]
        result = run_reconciliation({}, {}, orders, run_id="r1")
        assert "PENDING_SUBMIT" in result.inflight_summary
        assert "ACKED" in result.inflight_summary

    def test_orphan_fill_is_blocking(self):
        yesterday = "2026-05-16"
        o = _make_order(
            "1234.T",
            state="SUBMITTED_UNKNOWN",
            registered_at=f"{yesterday}T09:00:00+09:00",
        )
        now = datetime(2026, 5, 17, 9, 0, 0, tzinfo=timezone.utc).timestamp()
        result = run_reconciliation({}, {}, [o], run_id="r1", now_sec=now)
        assert result.reconciliation_ok is False
        assert result.blocking_mismatches >= 1

    def test_never_raises_on_bad_input(self):
        # MagicMock with missing attributes
        bad = MagicMock(spec=[])
        result = run_reconciliation({}, {}, [bad], run_id="r1")
        assert isinstance(result, ReconciliationResult)

    def test_schema_version_present(self):
        result = run_reconciliation({}, {}, [])
        assert result.schema_version == 1


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEngine — persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestReconciliationPersistence:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "recon.jsonl"
        result = run_reconciliation({"A.T": 100}, {"A.T": 200}, [], run_id="rX")
        append_reconciliation_result(result, path)
        loaded = load_reconciliation_results(path)
        assert len(loaded) == 1
        assert loaded[0].run_id == "rX"
        assert loaded[0].broker_positions == {"A.T": 100}

    def test_load_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "recon.jsonl"
        assert load_reconciliation_results(path) == []

    def test_corrupted_line_skipped(self, tmp_path):
        path = tmp_path / "recon.jsonl"
        result = run_reconciliation({}, {}, [], run_id="ok")
        append_reconciliation_result(result, path)
        with path.open("a") as f:
            f.write("{bad json\n")
        loaded = load_reconciliation_results(path)
        assert len(loaded) == 1

    def test_multiple_results_preserved(self, tmp_path):
        path = tmp_path / "recon.jsonl"
        for i in range(3):
            append_reconciliation_result(
                run_reconciliation({}, {}, [], run_id=f"r{i}"), path
            )
        loaded = load_reconciliation_results(path)
        assert len(loaded) == 3


# ─────────────────────────────────────────────────────────────────────────────
# BrokerTruthSnapshot — construction and checksum
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerTruthSnapshot:
    def test_take_snapshot_returns_snapshot(self):
        ps = {
            "portfolio_mode": "live",
            "current_positions": 2,
            "open_slots": 1,
            "available_cash": 500_000.0,
            "current_drawdown": -0.02,
            "positions_api": {"source": "api"},
            "wallet_api": {"source": "api"},
            "cb_state": "NORMAL",
        }
        snap = take_snapshot(ps, run_id="r1")
        assert snap.portfolio_mode == "live"
        assert snap.current_positions_count == 2
        assert snap.checksum != ""
        assert len(snap.checksum) == 64  # sha256 hex

    def test_validate_snapshot_fresh(self):
        snap = _make_snapshot()
        assert validate_snapshot(snap) is True

    def test_validate_snapshot_tampered_checksum(self):
        snap = _make_snapshot(checksum_tamper=True)
        assert validate_snapshot(snap) is False

    def test_checksum_deterministic(self):
        snap1 = _make_snapshot(run_id="r1", timestamp="2026-05-17T09:00:00+00:00")
        snap2 = _make_snapshot(run_id="r1", timestamp="2026-05-17T09:00:00+00:00")
        assert snap1.checksum == snap2.checksum

    def test_checksum_changes_with_different_run_id(self):
        snap1 = _make_snapshot(run_id="r1", timestamp="2026-05-17T09:00:00+00:00")
        snap2 = _make_snapshot(run_id="r2", timestamp="2026-05-17T09:00:00+00:00")
        assert snap1.checksum != snap2.checksum

    def test_empty_portfolio_summary_no_raise(self):
        snap = take_snapshot({}, run_id="r1")
        assert isinstance(snap, BrokerTruthSnapshot)
        assert validate_snapshot(snap) is True

    def test_none_portfolio_summary_no_raise(self):
        snap = take_snapshot(None, run_id="r1")  # type: ignore
        assert isinstance(snap, BrokerTruthSnapshot)

    def test_is_stale_old_snapshot(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=600))
        snap = _make_snapshot(timestamp=old_ts)
        assert is_snapshot_stale(snap, time.time(), stale_sec=300.0) is True

    def test_is_not_stale_fresh_snapshot(self):
        snap = _make_snapshot()
        assert is_snapshot_stale(snap, time.time(), stale_sec=300.0) is False

    def test_malformed_timestamp_treated_as_stale(self):
        snap = _make_snapshot()
        snap.timestamp = "not-a-date"
        assert is_snapshot_stale(snap, time.time(), stale_sec=300.0) is True


# ─────────────────────────────────────────────────────────────────────────────
# BrokerTruthSnapshot — persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerTruthSnapshotPersistence:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "snapshots.jsonl"
        snap = _make_snapshot(run_id="r1")
        append_snapshot(snap, path)
        loaded = load_snapshots(path)
        assert len(loaded) == 1
        assert loaded[0].run_id == "r1"
        assert validate_snapshot(loaded[0]) is True

    def test_checksum_preserved_on_load(self, tmp_path):
        path = tmp_path / "snapshots.jsonl"
        snap = _make_snapshot(run_id="r1")
        original_checksum = snap.checksum
        append_snapshot(snap, path)
        loaded = load_snapshots(path)
        assert loaded[0].checksum == original_checksum

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "snapshots.jsonl"
        assert load_snapshots(path) == []

    def test_corrupted_line_skipped(self, tmp_path):
        path = tmp_path / "snapshots.jsonl"
        append_snapshot(_make_snapshot(run_id="r1"), path)
        with path.open("a") as f:
            f.write("CORRUPTED\n")
        loaded = load_snapshots(path)
        assert len(loaded) == 1

    def test_multiple_snapshots_preserved(self, tmp_path):
        path = tmp_path / "snapshots.jsonl"
        for i in range(3):
            append_snapshot(_make_snapshot(run_id=f"r{i}"), path)
        loaded = load_snapshots(path)
        assert len(loaded) == 3


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionIntegrityValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionIntegrityValidator:
    def test_clean_state_execution_allowed(self):
        snap = _make_snapshot()
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(recon, snap, is_live=True, now_sec=time.time())
        assert result.execution_allowed is True
        assert result.blocking_count == 0
        assert result.fail_closed_reason == ""

    def test_blocking_mismatch_blocks_live(self):
        snap = _make_snapshot()
        recon = _make_recon_result(blocking=1)
        recon.mismatches = [
            {"mismatch_type": MISMATCH_DUPLICATE_PENDING, "detail": "dup", "severity": "blocking"}
        ]
        result = validate_execution_integrity(recon, snap, is_live=True, now_sec=time.time())
        assert result.execution_allowed is False
        assert VIOLATION_BLOCKING_MISMATCH in result.fail_closed_reason or result.blocking_count > 0

    def test_blocking_mismatch_does_not_block_dry_run(self):
        snap = _make_snapshot()
        recon = _make_recon_result(blocking=1)
        recon.mismatches = [
            {"mismatch_type": MISMATCH_DUPLICATE_PENDING, "detail": "dup", "severity": "blocking"}
        ]
        result = validate_execution_integrity(recon, snap, is_live=False, now_sec=time.time())
        assert result.execution_allowed is True   # dry-run never blocked

    def test_none_snapshot_blocks_live(self):
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(recon, None, is_live=True, now_sec=time.time())
        assert result.execution_allowed is False
        assert result.snapshot_valid is False

    def test_none_snapshot_dry_run_allowed(self):
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(recon, None, is_live=False, now_sec=time.time())
        assert result.execution_allowed is True

    def test_bad_checksum_blocks_live(self):
        snap = _make_snapshot(checksum_tamper=True)
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(recon, snap, is_live=True, now_sec=time.time())
        assert result.execution_allowed is False
        assert result.snapshot_valid is False

    def test_stale_snapshot_blocks_live(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=600))
        snap = _make_snapshot(timestamp=old_ts)
        snap.checksum = _compute_checksum(snap)   # valid checksum
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(
            recon, snap, is_live=True, now_sec=time.time(), snapshot_stale_sec=300.0
        )
        assert result.execution_allowed is False

    def test_stale_snapshot_dry_run_allowed(self):
        old_ts = _ts(_now_utc() - timedelta(seconds=600))
        snap = _make_snapshot(timestamp=old_ts)
        snap.checksum = _compute_checksum(snap)
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(
            recon, snap, is_live=False, now_sec=time.time(), snapshot_stale_sec=300.0
        )
        assert result.execution_allowed is True

    def test_result_preserves_is_live_flag(self):
        snap = _make_snapshot()
        recon = _make_recon_result()
        r_live = validate_execution_integrity(recon, snap, is_live=True, now_sec=time.time())
        r_dry = validate_execution_integrity(recon, snap, is_live=False, now_sec=time.time())
        assert r_live.is_live is True
        assert r_dry.is_live is False

    def test_never_raises_on_none_recon(self):
        snap = _make_snapshot()
        result = validate_execution_integrity(None, snap, is_live=False, now_sec=time.time())  # type: ignore
        assert isinstance(result, ExecutionIntegrityResult)

    def test_multiple_violations_all_recorded(self):
        snap = _make_snapshot(checksum_tamper=True)
        recon = _make_recon_result(blocking=1)
        recon.mismatches = [
            {"mismatch_type": MISMATCH_DUPLICATE_PENDING, "detail": "dup", "severity": "blocking"}
        ]
        result = validate_execution_integrity(recon, snap, is_live=True, now_sec=time.time())
        assert len(result.integrity_violations) >= 2
        types = [v["violation_type"] for v in result.integrity_violations]
        assert VIOLATION_BLOCKING_MISMATCH in types
        assert VIOLATION_BAD_CHECKSUM in types

    def test_fail_closed_reason_set_on_block(self):
        recon = _make_recon_result(blocking=0)
        result = validate_execution_integrity(recon, None, is_live=True, now_sec=time.time())
        assert result.fail_closed_reason != ""


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionIntegrityValidator — persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionIntegrityPersistence:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "integrity.jsonl"
        snap = _make_snapshot()
        recon = _make_recon_result()
        result = validate_execution_integrity(recon, snap, is_live=True, now_sec=time.time())
        append_integrity_result(result, path)
        loaded = load_integrity_results(path)
        assert len(loaded) == 1
        assert loaded[0].execution_allowed == result.execution_allowed

    def test_load_empty_returns_empty(self, tmp_path):
        path = tmp_path / "integrity.jsonl"
        assert load_integrity_results(path) == []

    def test_corrupted_line_skipped(self, tmp_path):
        path = tmp_path / "integrity.jsonl"
        snap = _make_snapshot()
        recon = _make_recon_result()
        r = validate_execution_integrity(recon, snap, is_live=False, now_sec=time.time())
        append_integrity_result(r, path)
        with path.open("a") as f:
            f.write("{bad\n")
        loaded = load_integrity_results(path)
        assert len(loaded) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ReplayConsistencyValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayConsistencyValidator:
    def test_empty_inputs_ok(self):
        result = validate_replay_consistency([], [])
        assert result.determinism_ok is True
        assert result.lineage_ok is True
        assert result.divergence_count == 0

    def test_clean_snapshots_and_recons_ok(self):
        snap = _make_snapshot(run_id="r1", timestamp="2026-05-17T09:00:00+00:00")
        recon = _make_recon_result(run_id="r1", timestamp="2026-05-17T09:01:00+00:00")
        result = validate_replay_consistency([snap], [recon])
        assert result.determinism_ok is True
        assert result.lineage_ok is True

    def test_tampered_checksum_detected(self):
        snap = _make_snapshot(run_id="r1", checksum_tamper=True)
        result = validate_replay_consistency([snap], [])
        assert result.determinism_ok is False
        types = [d["divergence_type"] for d in result.divergences]
        assert DIVERGENCE_CHECKSUM_INVALID in types

    def test_non_monotonic_timestamps_detected(self):
        snap1 = _make_snapshot(run_id="r1", timestamp="2026-05-17T10:00:00+00:00")
        snap2 = _make_snapshot(run_id="r2", timestamp="2026-05-17T09:00:00+00:00")
        result = validate_replay_consistency([snap1, snap2], [])
        assert result.determinism_ok is False
        types = [d["divergence_type"] for d in result.divergences]
        assert DIVERGENCE_NON_MONOTONIC_TS in types

    def test_duplicate_timestamps_detected(self):
        ts = "2026-05-17T09:00:00+00:00"
        snap1 = _make_snapshot(run_id="r1", timestamp=ts)
        snap2 = _make_snapshot(run_id="r2", timestamp=ts)
        result = validate_replay_consistency([snap1, snap2], [])
        types = [d["divergence_type"] for d in result.divergences]
        assert DIVERGENCE_DUPLICATE_TS in types

    def test_missing_snapshot_for_recon_detected(self):
        snap = _make_snapshot(run_id="r1")
        recon = _make_recon_result(run_id="r2")   # r2 has no snapshot
        result = validate_replay_consistency([snap], [recon])
        assert result.lineage_ok is False
        types = [d["divergence_type"] for d in result.divergences]
        assert DIVERGENCE_MISSING_SNAPSHOT in types

    def test_recon_with_matching_snapshot_no_divergence(self):
        snap = _make_snapshot(run_id="same-run")
        recon = _make_recon_result(run_id="same-run")
        result = validate_replay_consistency([snap], [recon])
        assert DIVERGENCE_MISSING_SNAPSHOT not in [
            d["divergence_type"] for d in result.divergences
        ]

    def test_counts_match_inputs(self):
        snaps = [_make_snapshot(run_id=f"r{i}", timestamp=f"2026-05-17T0{i}:00:00+00:00")
                 for i in range(3)]
        recons = [_make_recon_result(run_id=f"r{i}") for i in range(3)]
        result = validate_replay_consistency(snaps, recons)
        assert result.total_snapshots == 3
        assert result.total_recon_results == 3

    def test_never_raises(self):
        result = validate_replay_consistency([], [MagicMock(spec=[])])  # type: ignore
        assert isinstance(result, ReplayConsistencyResult)


# ─────────────────────────────────────────────────────────────────────────────
# ReplayConsistencyValidator — persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayConsistencyPersistence:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        snap = _make_snapshot(run_id="r1")
        recon = _make_recon_result(run_id="r1")
        result = validate_replay_consistency([snap], [recon])
        append_replay_result(result, path)
        loaded = load_replay_results(path)
        assert len(loaded) == 1
        assert loaded[0].total_snapshots == 1
        assert loaded[0].determinism_ok == result.determinism_ok

    def test_load_empty_returns_empty(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        assert load_replay_results(path) == []

    def test_corrupted_line_skipped(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        r = validate_replay_consistency([], [])
        append_replay_result(r, path)
        with path.open("a") as f:
            f.write("{bad\n")
        loaded = load_replay_results(path)
        assert len(loaded) == 1
