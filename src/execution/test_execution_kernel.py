"""
src/execution/test_execution_kernel.py
Unified execution kernel tests — 11 mandatory scenarios + extras.

Scenarios:
  1. duplicate retry              — same intent_id blocked before submit
  2. timeout then ACK             — UNKNOWN → reconcile finds order → ACKED
  3. reconnect recovery           — client None → client available → UNKNOWN resolved
  4. overlap scheduler            — same symbol+side within run_id blocked
  5. partial fill recovery        — PARTIAL fill recomputes remaining_qty + alloc
  6. stale snapshot rejection     — validate_state warns on stale snapshot
  7. crash during save            — atomic write exception leaves original intact
  8. rollback detection           — generation watermark mismatch logged
  9. UNKNOWN reconciliation       — UNKNOWN + broker order found → ACKED
  10. DRY/LIVE parity             — same equity from snapshot path and loose-dict path
  11. deterministic replay        — same journal → same frames, anomalies detected

Run:
    cd C:/ai-trading
    python -m pytest src/execution/test_execution_kernel.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.execution.intent import (
    ExecutionIntent, IntentStatus, VALID_TRANSITIONS,
    ACTIVE_STATUS_VALUES, apply_transition, make_intent, make_intent_id,
)
from src.execution.journal import IntentJournal
from src.execution.dd_engine import compute_drawdown, assess_risk, MAX_DD_LIMIT
from src.execution.reconcile import ReconcileResult, reconcile_open_orders
from src.execution.replay import ReplayFrame, replay_intent_journal

JST = timezone(timedelta(hours=9))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_ts() -> str:
    return datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")


def _tmp_journal() -> tuple[IntentJournal, Path]:
    tmpdir = Path(tempfile.mkdtemp())
    journal = IntentJournal(path=tmpdir / "order_intents.jsonl")
    return journal, tmpdir


def _make_test_intent(
    symbol:         str   = "6981.T",
    side:           str   = "BUY",
    qty:            int   = 100,
    strategy:       str   = "trend_follow",
    run_id:         str   = "test_run_001",
    generation_id:  int   = 1,
    expected_price: float = 5000.0,
) -> ExecutionIntent:
    return make_intent(
        symbol         = symbol,
        side           = side,
        qty            = qty,
        strategy       = strategy,
        expected_price = expected_price,
        alloc_before   = 1_799_309.0,
        alloc_after    = 1_299_309.0,
        run_id         = run_id,
        signal_ts      = _now_ts(),
        generation_id  = generation_id,
    )


def _make_broker_client(open_orders: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.get_orders.return_value = open_orders or []
    return client


# ── 1. Duplicate Retry ───────────────────────────────────────────────────────

class TestDuplicatePrevention(unittest.TestCase):

    def test_duplicate_intent_blocked_when_active(self):
        """Same intent_id with active status → is_duplicate() returns True."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            journal.append(intent, "created")

            # Advance to SUBMITTED
            apply_transition(intent, IntentStatus.SUBMITTED)
            journal.append(intent, "submitted")

            # Duplicate check must block
            self.assertTrue(journal.is_duplicate(intent.intent_id))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_duplicate_not_blocked_after_terminal(self):
        """FILLED / CANCELLED / REJECTED are terminal — not duplicate-blocked."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            journal.append(intent, "created")
            apply_transition(intent, IntentStatus.SUBMITTED)
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.FILLED, filled_qty=100)
            journal.append(intent, "filled")

            self.assertFalse(journal.is_duplicate(intent.intent_id))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_duplicate_block_logs_warning(self):
        """is_duplicate() on active intent emits [DUPLICATE_BLOCK] warning."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            journal.append(intent, "created")
            with self.assertLogs("src.execution.journal", level="WARNING") as cm:
                journal.is_duplicate(intent.intent_id)
            self.assertTrue(any("DUPLICATE_BLOCK" in m for m in cm.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_intent_id_deterministic_across_calls(self):
        """Same inputs → same intent_id (idempotent)."""
        ts  = "2026-05-08T09:00:00+0900"
        id1 = make_intent_id("6981.T", "BUY", 100, "trend_follow", ts, 7)
        id2 = make_intent_id("6981.T", "BUY", 100, "trend_follow", ts, 7)
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 16)

    def test_intent_id_changes_with_generation(self):
        """Different generation_id → different intent_id (no cross-run collision)."""
        ts  = "2026-05-08T09:00:00+0900"
        id1 = make_intent_id("6981.T", "BUY", 100, "trend_follow", ts, 1)
        id2 = make_intent_id("6981.T", "BUY", 100, "trend_follow", ts, 2)
        self.assertNotEqual(id1, id2)


# ── 2. Timeout then ACK ───────────────────────────────────────────────────────

class TestTimeoutThenAck(unittest.TestCase):

    def test_unknown_intent_not_marked_failed_directly(self):
        """After timeout, status must be UNKNOWN, not CANCELLED/REJECTED."""
        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        # Simulate timeout: must go to UNKNOWN, not CANCELLED
        apply_transition(intent, IntentStatus.UNKNOWN)
        self.assertEqual(intent.status, "UNKNOWN")

    def test_cannot_go_directly_submitted_to_cancelled_via_unk_path(self):
        """SUBMITTED → CANCELLED is valid (explicit cancel), but UNKNOWN is preferred for timeouts."""
        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        # SUBMITTED → CANCELLED is allowed (user cancel), test it works
        apply_transition(intent, IntentStatus.CANCELLED)
        self.assertEqual(intent.status, "CANCELLED")

    def test_unknown_to_acked_via_reconcile(self):
        """UNKNOWN → ACKED transition is valid and represents recovery."""
        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.UNKNOWN)
        apply_transition(intent, IntentStatus.ACKED, filled_qty=0)
        self.assertEqual(intent.status, "ACKED")

    def test_reconcile_recovers_unknown_with_broker_match(self):
        """reconcile_open_orders() finds broker order for UNKNOWN intent → ACKED in journal."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD001")
            journal.append(intent, "submitted", broker_order_id="ORD001")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout_unknown")

            broker_client = _make_broker_client([
                {"ID": "ORD001", "Symbol": "6981", "CumQty": 0},
            ])
            result = reconcile_open_orders(journal, broker_client)

            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "ACKED")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 3. Reconnect Recovery ─────────────────────────────────────────────────────

class TestReconnectRecovery(unittest.TestCase):

    def test_client_none_leaves_unknown_unchanged(self):
        """When client=None, UNKNOWN intents cannot be reconciled → remain UNKNOWN."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD002")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "network_timeout")

            result = reconcile_open_orders(journal, client=None)
            # Without client, can't confirm → stays UNKNOWN
            self.assertEqual(journal.get_latest_status(intent.intent_id), "UNKNOWN")
            self.assertEqual(result.broker_open_count, 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reconnect_then_reconcile_resolves_unknown(self):
        """After reconnect: call reconcile with restored client → UNKNOWN → ACKED."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD003")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "network_timeout")

            # Reconnect + reconcile
            client = _make_broker_client([{"ID": "ORD003", "CumQty": 50}])
            result = reconcile_open_orders(journal, client)

            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertNotIn(intent.intent_id, result.orphan_intents)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_unknown_without_broker_order_id_becomes_orphan(self):
        """UNKNOWN intent with no broker_order_id → never submitted → CANCELLED."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED)  # no broker_order_id
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "unknown")

            client = _make_broker_client([])  # empty broker orders
            result = reconcile_open_orders(journal, client)

            self.assertIn(intent.intent_id, result.orphan_intents)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "CANCELLED")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 4. Overlap Scheduler ──────────────────────────────────────────────────────

class TestSchedulerOverlap(unittest.TestCase):

    def test_same_run_same_symbol_blocked(self):
        """Within same run_id: second BUY for same symbol → is_overlap_blocked."""
        journal, tmpdir = _tmp_journal()
        try:
            run_id = "run_20260508_abc"
            intent = _make_test_intent(run_id=run_id)
            apply_transition(intent, IntentStatus.SUBMITTED)
            journal.append(intent, "submitted")

            self.assertTrue(journal.is_overlap_blocked(
                symbol=intent.symbol, side=intent.side, run_id=run_id
            ))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_different_run_not_blocked_for_overlap(self):
        """Different run_id: same symbol+side is NOT within-run blocked."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent(run_id="run_20260507_old")
            apply_transition(intent, IntentStatus.SUBMITTED)
            journal.append(intent, "submitted")

            # Different run → not overlap-blocked
            self.assertFalse(journal.is_overlap_blocked(
                symbol=intent.symbol, side=intent.side, run_id="run_20260508_new"
            ))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_terminal_status_not_overlap_blocked(self):
        """FILLED intent in same run → not overlap-blocked (position already taken)."""
        journal, tmpdir = _tmp_journal()
        try:
            run_id = "run_20260508_xyz"
            intent = _make_test_intent(run_id=run_id)
            apply_transition(intent, IntentStatus.SUBMITTED)
            apply_transition(intent, IntentStatus.ACKED)
            apply_transition(intent, IntentStatus.FILLED, filled_qty=100)
            journal.append(intent, "filled")

            # FILLED is terminal → not active → not overlap blocked
            self.assertFalse(journal.is_overlap_blocked(
                symbol=intent.symbol, side=intent.side, run_id=run_id
            ))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 5. Partial Fill Recovery ──────────────────────────────────────────────────

class TestPartialFillRecovery(unittest.TestCase):

    def test_partial_fill_recomputes_remaining_qty(self):
        """After PARTIAL fill, remaining_qty = qty - filled_qty."""
        intent = _make_test_intent(qty=100)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=40)
        self.assertEqual(intent.remaining_qty, 60)

    def test_partial_to_filled(self):
        """PARTIAL → FILLED is a valid terminal transition."""
        intent = _make_test_intent(qty=100)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=60)
        apply_transition(intent, IntentStatus.FILLED, filled_qty=100)
        self.assertEqual(intent.status, "FILLED")
        self.assertEqual(intent.filled_qty, 100)

    def test_partial_to_cancelled(self):
        """PARTIAL → CANCELLED is valid (user cancel of remainder)."""
        intent = _make_test_intent(qty=100)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=50)
        apply_transition(intent, IntentStatus.CANCELLED)
        self.assertEqual(intent.status, "CANCELLED")

    def test_partial_journaled_correctly(self):
        """PARTIAL fill is persisted in journal with remaining_qty."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent(qty=200)
            apply_transition(intent, IntentStatus.SUBMITTED)
            apply_transition(intent, IntentStatus.PARTIAL, filled_qty=80)
            journal.append(intent, "partial_fill")

            records = journal.load_all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["remaining_qty"], 120)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 6. Stale Snapshot Rejection ───────────────────────────────────────────────

class TestStaleSnapshotRejection(unittest.TestCase):

    def test_stale_state_warns(self):
        """portfolio_state with updated_at > STALE_WARN_SECONDS emits warning."""
        from src.portfolio.state_store import validate_state, STALE_WARN_SECONDS
        old_ts = (datetime.now(JST) - timedelta(seconds=STALE_WARN_SECONDS + 120))
        state = {
            "schema_version":    2,
            "updated_at":        old_ts.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "data_source":       "test",
            "cb_state":          "NORMAL",
            "equity_peak":       3_000_000.0,
            "available_cash":    1_799_309.0,
            "position_entry_dates":  {},
            "position_entry_prices": {},
            "position_qtys":     {},
            "positions_count":   0,
            "last_equity":       3_000_000.0,
            "generation_id":     1,
        }
        vr = validate_state(state)
        self.assertTrue(vr.is_stale)
        self.assertTrue(any("stale" in w for w in vr.warnings))

    def test_very_stale_state_warns_hard(self):
        """portfolio_state > STALE_HARD_SECONDS triggers is_very_stale."""
        from src.portfolio.state_store import validate_state, STALE_HARD_SECONDS
        old_ts = (datetime.now(JST) - timedelta(seconds=STALE_HARD_SECONDS + 60))
        state = {
            "schema_version":    2,
            "updated_at":        old_ts.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "data_source":       "test",
            "cb_state":          "NORMAL",
            "equity_peak":       3_000_000.0,
            "available_cash":    1_799_309.0,
            "position_entry_dates":  {},
            "position_entry_prices": {},
            "position_qtys":     {},
            "positions_count":   0,
            "last_equity":       3_000_000.0,
            "generation_id":     1,
        }
        vr = validate_state(state)
        self.assertTrue(vr.is_very_stale)

    def test_fresh_snapshot_not_stale(self):
        """State updated 60s ago → not stale."""
        from src.portfolio.state_store import validate_state
        fresh_ts = datetime.now(JST)
        state = {
            "schema_version":    2,
            "updated_at":        fresh_ts.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "data_source":       "test",
            "cb_state":          "NORMAL",
            "equity_peak":       3_000_000.0,
            "available_cash":    1_799_309.0,
            "position_entry_dates":  {},
            "position_entry_prices": {},
            "position_qtys":     {},
            "positions_count":   0,
            "last_equity":       3_000_000.0,
            "generation_id":     1,
        }
        vr = validate_state(state)
        self.assertFalse(vr.is_stale)


# ── 7. Crash During Save ──────────────────────────────────────────────────────

class TestCrashDuringSave(unittest.TestCase):

    def test_crash_during_atomic_write_leaves_original(self):
        """Exception in json.dump → original file intact, no .tmp left."""
        from src.portfolio.state_store import atomic_write_json
        tmpdir = Path(tempfile.mkdtemp())
        try:
            p = tmpdir / "state.json"
            atomic_write_json(p, {"equity_peak": 3_000_000.0})

            call_count = [0]
            orig_dump = json.dump

            def raising_dump(obj, fp, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise OSError("simulated disk error")
                return orig_dump(obj, fp, **kwargs)

            with patch("json.dump", side_effect=raising_dump):
                try:
                    atomic_write_json(p, {"equity_peak": 9_999_999.0})
                except OSError:
                    pass

            # Original must still be intact
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["equity_peak"], 3_000_000.0)
            # No .tmp files left
            self.assertEqual(list(tmpdir.glob("*.tmp")), [])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_journal_append_fsync(self):
        """Journal append writes data durably (can be loaded after write)."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            journal.append(intent, "created")
            # Reload from disk
            records = journal.load_all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["intent_id"], intent.intent_id)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 8. Rollback Detection ─────────────────────────────────────────────────────

class TestRollbackDetection(unittest.TestCase):

    def test_generation_rollback_logged(self):
        """Loading state with lower generation than watermark logs WARNING."""
        from src.portfolio.state_store import atomic_write_json, load_portfolio_state
        from src.portfolio import state_store as _ss
        tmpdir = Path(tempfile.mkdtemp())
        try:
            # Set watermark to 99
            wm_path = tmpdir / "state_generation.txt"
            wm_path.write_text("99", encoding="utf-8")

            p = tmpdir / "state.json"
            state = {
                "schema_version":    2,
                "updated_at":        datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "data_source":       "test",
                "cb_state":          "NORMAL",
                "equity_peak":       3_000_000.0,
                "available_cash":    1_799_309.0,
                "position_entry_dates":  {},
                "position_entry_prices": {},
                "position_qtys":     {},
                "positions_count":   0,
                "last_equity":       3_000_000.0,
                "generation_id":     5,   # lower than watermark
            }
            atomic_write_json(p, state)

            orig_wm = _ss._GENERATION_WATERMARK_PATH
            _ss._GENERATION_WATERMARK_PATH = wm_path
            try:
                with self.assertLogs("src.portfolio.state_store", level="WARNING") as cm:
                    load_portfolio_state(p)
                self.assertTrue(any("rollback" in m.lower() for m in cm.output))
            finally:
                _ss._GENERATION_WATERMARK_PATH = orig_wm
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_generation_increments_monotonically(self):
        """save_portfolio_state() always increments generation_id."""
        from src.portfolio.state_store import save_portfolio_state
        tmpdir = Path(tempfile.mkdtemp())
        try:
            p = tmpdir / "state.json"
            state = {
                "schema_version": 2, "updated_at": None, "data_source": "test",
                "cb_state": "NORMAL", "equity_peak": 3_000_000.0,
                "available_cash": 1_799_309.0, "position_entry_dates": {},
                "position_entry_prices": {}, "position_qtys": {}, "positions_count": 0,
                "last_equity": 3_000_000.0, "generation_id": 10,
            }
            save_portfolio_state(state, path=p)
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["generation_id"], 11)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 9. UNKNOWN Reconciliation ─────────────────────────────────────────────────

class TestUnknownReconciliation(unittest.TestCase):

    def test_unknown_found_at_broker_becomes_acked(self):
        """UNKNOWN + broker_order_id found → ACKED in journal."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="BRK999")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout")

            client = _make_broker_client([{"ID": "BRK999", "CumQty": 0}])
            result = reconcile_open_orders(journal, client)

            self.assertEqual(journal.get_latest_status(intent.intent_id), "ACKED")
            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertEqual(len(result.orphan_intents), 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_unknown_not_at_broker_becomes_cancelled(self):
        """UNKNOWN + broker_order_id NOT found → CANCELLED (orphan)."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="GONE001")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout")

            client = _make_broker_client([])   # empty — order not found
            result = reconcile_open_orders(journal, client)

            self.assertEqual(journal.get_latest_status(intent.intent_id), "CANCELLED")
            self.assertIn(intent.intent_id, result.orphan_intents)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_phantom_broker_order_detected(self):
        """Broker has order with no matching journal entry → phantom_orders."""
        journal, tmpdir = _tmp_journal()
        try:
            client = _make_broker_client([
                {"ID": "PHANTOM_ORDER_XYZ", "Symbol": "8015", "CumQty": 0}
            ])
            result = reconcile_open_orders(journal, client)
            self.assertIn("PHANTOM_ORDER_XYZ", result.phantom_orders)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reconcile_result_log_emits_reconcile_line(self):
        """reconcile_open_orders() emits [RECONCILE] structured log."""
        journal, tmpdir = _tmp_journal()
        try:
            client = _make_broker_client([])
            with self.assertLogs("src.execution.reconcile", level="INFO") as cm:
                reconcile_open_orders(journal, client)
            self.assertTrue(any("[RECONCILE]" in m for m in cm.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 10. DRY / LIVE Parity ─────────────────────────────────────────────────────

class TestDryLiveParity(unittest.TestCase):

    def test_equity_single_snapshot_path_dry_and_live(self):
        """
        Broker-as-Sole-SSOT (2026-07-18): DRY と LIVE は同じ BrokerSnapshot 入力を
        使う限り必ず同一の equity を返す（唯一の入力・唯一の計算式のため、
        パス間の drift は構造的に発生し得ない — 旧 loose-dict フォールバック
        パスは廃止された）。
        """
        from src.portfolio.equity import compute_live_equity
        from src.portfolio.state_store import BrokerSnapshot

        cash = 1_799_309.0
        snap = BrokerSnapshot(
            cash          = cash,
            positions     = {"6981.T": 100, "8015.T": 100},
            avg_costs     = {"6981.T": 5648.0, "8015.T": 6840.0},
            market_values = {"6981.T": 5648.0, "8015.T": 6840.0},
            equity        = 0.0,
            ts            = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            source        = "broker",
            api_health    = {},
        )
        equity_dry  = compute_live_equity(snapshot=snap, mode="dry",  persist_snapshot=False)
        equity_live = compute_live_equity(snapshot=snap, mode="live", persist_snapshot=False)

        self.assertEqual(equity_dry, equity_live,
                         f"DRY equity {equity_dry} != LIVE equity {equity_live}")
        self.assertAlmostEqual(equity_dry, cash + 100 * 5648.0 + 100 * 6840.0, places=0)

    def test_compute_drawdown_canonical_single_impl(self):
        """compute_drawdown is the single implementation — produces exact result."""
        dd = compute_drawdown(2_550_000.0, 3_000_000.0)
        self.assertAlmostEqual(dd, -0.15, places=10)

    def test_cb_trigger_at_exactly_max_dd(self):
        """CB triggers at exactly MAX_DD_LIMIT (boundary condition)."""
        from src.execution.dd_engine import is_cb_trigger
        dd = compute_drawdown(2_550_000.0, 3_000_000.0)  # exactly -0.15
        self.assertTrue(is_cb_trigger(dd))

    def test_cb_not_triggered_above_threshold(self):
        """No CB when dd > MAX_DD_LIMIT (less severe drawdown)."""
        from src.execution.dd_engine import is_cb_trigger
        dd = compute_drawdown(2_700_000.0, 3_000_000.0)  # -0.10
        self.assertFalse(is_cb_trigger(dd))


# ── 11. Deterministic Replay ──────────────────────────────────────────────────

class TestDeterministicReplay(unittest.TestCase):

    def _build_journal(self, tmpdir: Path) -> tuple[IntentJournal, str]:
        journal = IntentJournal(path=tmpdir / "order_intents.jsonl")
        run_id  = "replay_test_run"

        # Intent A: CREATED → SUBMITTED → ACKED → FILLED
        a = _make_test_intent(symbol="6981.T", run_id=run_id, generation_id=5)
        journal.append(a, "created")
        apply_transition(a, IntentStatus.SUBMITTED, broker_order_id="ORD_A")
        journal.append(a, "submitted")
        apply_transition(a, IntentStatus.ACKED)
        journal.append(a, "acked")
        apply_transition(a, IntentStatus.FILLED, filled_qty=100)
        journal.append(a, "filled")

        # Intent B: CREATED → SUBMITTED → UNKNOWN (never resolved)
        b = _make_test_intent(symbol="8015.T", run_id=run_id, generation_id=5)
        journal.append(b, "created")
        apply_transition(b, IntentStatus.SUBMITTED, broker_order_id="ORD_B")
        journal.append(b, "submitted")
        apply_transition(b, IntentStatus.UNKNOWN)
        journal.append(b, "timeout")

        return journal, run_id

    def test_replay_produces_correct_frame_count(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal, _ = self._build_journal(tmpdir)
            frames = replay_intent_journal(tmpdir / "order_intents.jsonl")
            self.assertEqual(len(frames), 7)  # 4 + 3 events
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_deterministic_same_input_same_output(self):
        """Identical journal → identical replay output (ordered by ts)."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal, _ = self._build_journal(tmpdir)
            frames1 = replay_intent_journal(tmpdir / "order_intents.jsonl")
            frames2 = replay_intent_journal(tmpdir / "order_intents.jsonl")
            self.assertEqual(
                [f.to_dict() for f in frames1],
                [f.to_dict() for f in frames2],
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_detects_lingering_unknown_anomaly(self):
        """Intent that ends in UNKNOWN → logged as anomaly."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal, _ = self._build_journal(tmpdir)
            with self.assertLogs("src.execution.replay", level="WARNING") as cm:
                replay_intent_journal(tmpdir / "order_intents.jsonl")
            self.assertTrue(any("UNKNOWN" in m for m in cm.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_detects_invalid_transition(self):
        """Journal with invalid transition → anomaly flagged in ReplayFrame."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal = IntentJournal(path=tmpdir / "order_intents.jsonl")
            intent = _make_test_intent()
            journal.append(intent, "created")
            # Manually inject an invalid record (CREATED → FILLED, skipping SUBMITTED)
            bad_record = intent.to_dict()
            bad_record["status"]    = "FILLED"
            bad_record["event"]     = "injected_bad_transition"
            bad_record["ts"]        = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
            bad_record["filled_qty"] = 100
            with (tmpdir / "order_intents.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(bad_record) + "\n")

            frames = replay_intent_journal(tmpdir / "order_intents.jsonl")
            invalid_frames = [f for f in frames if f.anomalies]
            self.assertTrue(len(invalid_frames) > 0)
            self.assertTrue(any("INVALID_TRANSITION" in a
                               for f in invalid_frames for a in f.anomalies))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_empty_journal_returns_empty(self):
        """Empty journal file → empty replay."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal_path = tmpdir / "order_intents.jsonl"
            journal_path.write_text("", encoding="utf-8")
            frames = replay_intent_journal(journal_path)
            self.assertEqual(frames, [])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── State Machine Invariants ──────────────────────────────────────────────────

class TestStateMachineInvariants(unittest.TestCase):

    def test_invalid_transition_raises_value_error(self):
        """FILLED → SUBMITTED is invalid → raises ValueError."""
        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.FILLED, filled_qty=100)
        with self.assertRaises(ValueError):
            apply_transition(intent, IntentStatus.SUBMITTED)

    def test_invalid_transition_logs_critical(self):
        """Invalid transition emits CRITICAL log."""
        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.REJECTED)
        with self.assertLogs("src.execution.intent", level="CRITICAL") as cm:
            try:
                apply_transition(intent, IntentStatus.SUBMITTED)
            except ValueError:
                pass
        self.assertTrue(any("INVALID TRANSITION" in m for m in cm.output))

    def test_all_terminal_states_have_no_transitions(self):
        """FILLED / CANCELLED / REJECTED have empty transition sets."""
        from src.execution.intent import TERMINAL_STATUSES
        for s in TERMINAL_STATUSES:
            self.assertEqual(
                VALID_TRANSITIONS.get(s, frozenset()),
                frozenset(),
                f"{s} should have no outgoing transitions",
            )

    def test_created_intent_fields_are_set(self):
        """make_intent() produces intent with all required fields."""
        intent = _make_test_intent()
        self.assertEqual(len(intent.intent_id), 16)
        self.assertEqual(intent.status, "CREATED")
        self.assertEqual(intent.retry_count, 0)
        self.assertEqual(intent.remaining_qty, intent.qty)
        self.assertNotEqual(intent.created_at, "")


# ── DD Engine Invariants ──────────────────────────────────────────────────────

class TestDDEngineInvariants(unittest.TestCase):

    def test_compute_drawdown_zero_peak_returns_zero(self):
        self.assertEqual(compute_drawdown(3_000_000, 0), 0.0)

    def test_compute_drawdown_zero_equity_returns_zero(self):
        self.assertEqual(compute_drawdown(0, 3_000_000), 0.0)

    def test_compute_drawdown_at_peak_is_zero(self):
        self.assertEqual(compute_drawdown(3_000_000, 3_000_000), 0.0)

    def test_compute_drawdown_above_peak_positive(self):
        dd = compute_drawdown(3_300_000, 3_000_000)
        self.assertAlmostEqual(dd, 0.10, places=5)

    def test_assess_risk_cb_active_at_15pct(self):
        result = assess_risk(2_550_000, 3_000_000)
        self.assertEqual(result["recommendation"], "CB_ACTIVE")
        self.assertTrue(result["cb_trigger"])

    def test_assess_risk_normal_small_dd(self):
        result = assess_risk(2_900_000, 3_000_000)
        self.assertEqual(result["recommendation"], "NORMAL")
        self.assertFalse(result["cb_trigger"])

    def test_assess_risk_safe_warn_on_anomalous_peak(self):
        # current_equity=2M, peak=3M → ratio=1.5 > 1.25 → SAFE_WARN
        result = assess_risk(2_000_000, 3_000_000, cb_state="NORMAL")
        self.assertIn(result["recommendation"], ("SAFE_WARN", "CB_ACTIVE"))
        self.assertTrue(result["peak_anomaly"])

    def test_assess_risk_recovery_pending(self):
        result = assess_risk(2_600_000, 3_000_000, cb_state="CB_ACTIVE")
        self.assertEqual(result["recommendation"], "RECOVERY")

    def test_assess_risk_recovery_complete(self):
        # equity ≥ 90% of peak → back to NORMAL
        result = assess_risk(2_800_000, 3_000_000, cb_state="CB_ACTIVE")
        self.assertEqual(result["recommendation"], "NORMAL")


# ── Journal Concurrent Safety ─────────────────────────────────────────────────

class TestJournalConcurrentSafety(unittest.TestCase):

    def test_concurrent_appends_no_corruption(self):
        """8 threads appending simultaneously → journal remains valid JSONL."""
        journal, tmpdir = _tmp_journal()
        try:
            errors: list[str] = []

            def appender(i: int):
                try:
                    intent = _make_test_intent(symbol=f"600{i}.T")
                    journal.append(intent, "created")
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=appender, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            records = journal.load_all()
            self.assertEqual(len(records), 8, f"Expected 8 records, got {len(records)}")
            self.assertEqual(errors, [], f"Thread errors: {errors}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Gap 1: can_submit_intent() centralized gate ───────────────────────────────

class TestCanSubmitIntent(unittest.TestCase):

    def test_gate_allows_fresh_intent(self):
        """New intent with no journal history → allowed."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            allowed, reason = journal.can_submit_intent(
                intent.intent_id, intent.symbol, intent.side, intent.run_id
            )
            self.assertTrue(allowed)
            self.assertEqual(reason, "ok")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gate_blocks_duplicate_intent_id(self):
        """Active intent_id → gate returns allowed=False, reason contains 'duplicate'."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            journal.append(intent, "created")
            apply_transition(intent, IntentStatus.SUBMITTED)
            journal.append(intent, "submitted")

            allowed, reason = journal.can_submit_intent(
                intent.intent_id, intent.symbol, intent.side, intent.run_id
            )
            self.assertFalse(allowed)
            self.assertIn("duplicate", reason)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gate_blocks_overlap_same_run(self):
        """Different intent_id, same symbol+side in same run → overlap block."""
        journal, tmpdir = _tmp_journal()
        try:
            run_id = "run_overlap_test"
            intent = _make_test_intent(run_id=run_id)
            apply_transition(intent, IntentStatus.SUBMITTED)
            journal.append(intent, "submitted")

            # Different intent_id (different signal_ts), same symbol+side+run
            intent2 = make_intent(
                symbol="6981.T", side="BUY", qty=50, strategy="trend_follow",
                expected_price=5100.0, alloc_before=0.0, alloc_after=0.0,
                run_id=run_id, signal_ts=_now_ts(), generation_id=2,
            )
            allowed, reason = journal.can_submit_intent(
                intent2.intent_id, intent2.symbol, intent2.side, run_id
            )
            self.assertFalse(allowed)
            self.assertIn("overlap", reason)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gate_logs_intent_gate(self):
        """can_submit_intent() emits [INTENT_GATE] log for both allowed and blocked."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            with self.assertLogs("src.execution.journal", level="INFO") as cm:
                journal.can_submit_intent(
                    intent.intent_id, intent.symbol, intent.side, intent.run_id
                )
            self.assertTrue(any("INTENT_GATE" in m for m in cm.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gate_allows_after_terminal(self):
        """After FILLED (terminal), same symbol+side in same run is allowed again."""
        journal, tmpdir = _tmp_journal()
        try:
            run_id = "run_terminal_test"
            intent = _make_test_intent(run_id=run_id)
            apply_transition(intent, IntentStatus.SUBMITTED)
            apply_transition(intent, IntentStatus.ACKED)
            apply_transition(intent, IntentStatus.FILLED, filled_qty=100)
            journal.append(intent, "filled")

            intent2 = make_intent(
                symbol="6981.T", side="BUY", qty=100, strategy="trend_follow",
                expected_price=5200.0, alloc_before=0.0, alloc_after=0.0,
                run_id=run_id, signal_ts=_now_ts(), generation_id=3,
            )
            allowed, reason = journal.can_submit_intent(
                intent2.intent_id, intent2.symbol, intent2.side, run_id
            )
            self.assertTrue(allowed, f"Should be allowed after FILLED, got: {reason}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Gap 2: reconcile PARTIAL/FILLED recovery ──────────────────────────────────

class TestReconcilePartialFilled(unittest.TestCase):

    def test_reconcile_partial_fill_at_broker(self):
        """UNKNOWN + CumQty > 0 and < qty → PARTIAL in journal."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent(qty=100)
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD_PARTIAL")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout")

            # Broker shows 60/100 filled
            client = _make_broker_client([
                {"ID": "ORD_PARTIAL", "CumQty": 60, "Symbol": "6981"}
            ])
            result = reconcile_open_orders(journal, client)

            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "PARTIAL")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reconcile_full_fill_at_broker(self):
        """UNKNOWN + CumQty == qty → FILLED in journal."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent(qty=100)
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD_FULL")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout")

            client = _make_broker_client([
                {"ID": "ORD_FULL", "CumQty": 100, "Symbol": "6981"}
            ])
            result = reconcile_open_orders(journal, client)

            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "FILLED")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reconcile_zero_fill_is_acked(self):
        """UNKNOWN + CumQty == 0 → ACKED (broker accepted, not filled yet)."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent(qty=100)
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD_ZERO")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout")

            client = _make_broker_client([
                {"ID": "ORD_ZERO", "CumQty": 0, "Symbol": "6981"}
            ])
            result = reconcile_open_orders(journal, client)

            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "ACKED")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_resubmit_prohibited_when_broker_has_order(self):
        """
        If broker has the order (CumQty=0), we recover to ACKED — NOT resubmit.
        The recovered intent is in result.recovered_intents, not orphan.
        """
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD_RESUBMIT")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "timeout")

            client = _make_broker_client([
                {"ID": "ORD_RESUBMIT", "CumQty": 0}
            ])
            result = reconcile_open_orders(journal, client)

            # Must be recovered (ACKED), not orphaned
            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertNotIn(intent.intent_id, result.orphan_intents)
            # Journal says ACKED (not re-submitted)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "ACKED")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Gap 3: fill-aware allocation ──────────────────────────────────────────────

class TestFillAwareAlloc(unittest.TestCase):

    def test_partial_fill_uses_remaining_not_original(self):
        """PARTIAL intent: buying_power reduced by remaining_qty × price, not original."""
        from src.execution.alloc import compute_fill_aware_alloc

        intent = _make_test_intent(qty=100, expected_price=5000.0)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=40)
        # remaining = 60 shares

        alloc = compute_fill_aware_alloc(
            intents=[intent],
            cash=1_800_000.0,
            equity=3_000_000.0,
            market_values={"6981.T": 5000.0},
        )
        # reserved = 60 * 5000 = 300_000
        self.assertAlmostEqual(alloc.buying_power, 1_800_000.0 - 300_000.0)

    def test_exposure_includes_filled_portion_only(self):
        """Exposure from PARTIAL = filled_qty × price (not original qty × price)."""
        from src.execution.alloc import compute_fill_aware_alloc

        intent = _make_test_intent(qty=100, expected_price=5000.0)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=60)

        alloc = compute_fill_aware_alloc(
            intents=[intent],
            cash=1_000_000.0,
            equity=3_000_000.0,
            market_values={"6981.T": 5000.0},
        )
        # filled exposure = 60 * 5000 = 300_000
        self.assertAlmostEqual(alloc.total_exposure, 300_000.0)

    def test_no_partial_intents_full_buying_power(self):
        """No PARTIAL intents → buying_power equals cash (no reservation)."""
        from src.execution.alloc import compute_fill_aware_alloc

        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.ACKED)
        apply_transition(intent, IntentStatus.FILLED, filled_qty=100)

        alloc = compute_fill_aware_alloc(
            intents=[intent],
            cash=1_500_000.0,
            equity=3_000_000.0,
            market_values={"6981.T": 5000.0},
        )
        self.assertAlmostEqual(alloc.buying_power, 1_500_000.0)

    def test_alloc_recalc_log_emitted(self):
        """compute_fill_aware_alloc() emits [ALLOC_RECALC] log."""
        from src.execution.alloc import compute_fill_aware_alloc

        intent = _make_test_intent(qty=100, expected_price=5000.0)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=50)

        with self.assertLogs("src.execution.alloc", level="INFO") as cm:
            compute_fill_aware_alloc(
                intents=[intent],
                cash=1_000_000.0,
                equity=3_000_000.0,
                market_values={"6981.T": 5000.0},
            )
        self.assertTrue(any("ALLOC_RECALC" in m for m in cm.output))

    def test_partial_fills_dict_populated(self):
        """partial_fills contains correct fields for each PARTIAL intent."""
        from src.execution.alloc import compute_fill_aware_alloc

        intent = _make_test_intent(qty=200, expected_price=4000.0)
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.PARTIAL, filled_qty=80)

        alloc = compute_fill_aware_alloc(
            intents=[intent],
            cash=2_000_000.0,
            equity=3_000_000.0,
            market_values={"6981.T": 4000.0},
        )
        self.assertIn("6981.T", alloc.partial_fills)
        pf = alloc.partial_fills["6981.T"]
        self.assertEqual(pf["filled_qty"],    80)
        self.assertEqual(pf["remaining_qty"], 120)
        self.assertAlmostEqual(pf["exposure_used"],     80  * 4000.0)
        self.assertAlmostEqual(pf["exposure_reserved"], 120 * 4000.0)


# ── Gap 4: replay_execution_state ─────────────────────────────────────────────

class TestReplayExecutionState(unittest.TestCase):

    def _build_journal_with_partial(self, tmpdir: Path) -> IntentJournal:
        journal = IntentJournal(path=tmpdir / "order_intents.jsonl")

        a = _make_test_intent(symbol="6981.T", qty=100, expected_price=5000.0)
        journal.append(a, "created")
        apply_transition(a, IntentStatus.SUBMITTED, broker_order_id="ORD_A")
        journal.append(a, "submitted")
        apply_transition(a, IntentStatus.PARTIAL, filled_qty=60)
        journal.append(a, "partial_fill")
        apply_transition(a, IntentStatus.FILLED, filled_qty=100)
        journal.append(a, "filled")

        b = _make_test_intent(symbol="8015.T", qty=50, expected_price=6000.0)
        journal.append(b, "created")
        apply_transition(b, IntentStatus.SUBMITTED, broker_order_id="ORD_B")
        journal.append(b, "submitted")
        apply_transition(b, IntentStatus.UNKNOWN)
        journal.append(b, "timeout")

        return journal

    def test_replay_execution_state_returns_frames(self):
        """replay_execution_state returns same number of frames as events."""
        from src.execution.replay import replay_execution_state
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal = self._build_journal_with_partial(tmpdir)
            frames = replay_execution_state(tmpdir / "order_intents.jsonl")
            self.assertEqual(len(frames), 7)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_execution_state_deterministic(self):
        """Identical journal → identical replay_execution_state output."""
        from src.execution.replay import replay_execution_state
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal = self._build_journal_with_partial(tmpdir)
            f1 = replay_execution_state(tmpdir / "order_intents.jsonl")
            f2 = replay_execution_state(tmpdir / "order_intents.jsonl")
            self.assertEqual([f.to_dict() for f in f1], [f.to_dict() for f in f2])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_execution_state_exposure_annotated(self):
        """Frames after SUBMITTED have exposure > 0."""
        from src.execution.replay import replay_execution_state
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal = self._build_journal_with_partial(tmpdir)
            frames = replay_execution_state(tmpdir / "order_intents.jsonl")
            submitted_frames = [f for f in frames if f.status == "SUBMITTED"]
            self.assertTrue(all(f.exposure > 0 for f in submitted_frames),
                            "Submitted frames should have exposure > 0")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_execution_state_logs_replay(self):
        """replay_execution_state emits [REPLAY] run_id log."""
        from src.execution.replay import replay_execution_state
        tmpdir = Path(tempfile.mkdtemp())
        try:
            journal = self._build_journal_with_partial(tmpdir)
            with self.assertLogs("src.execution.replay", level="INFO") as cm:
                replay_execution_state(tmpdir / "order_intents.jsonl")
            self.assertTrue(any("REPLAY" in m and "run_id" in m for m in cm.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Gap 5: journal corruption scanner ────────────────────────────────────────

class TestJournalCorruptionScanner(unittest.TestCase):

    def test_scan_clean_journal_returns_empty(self):
        """Valid JSONL → no corrupted lines."""
        journal, tmpdir = _tmp_journal()
        try:
            journal.append(_make_test_intent(), "created")
            journal.append(_make_test_intent(symbol="8015.T"), "created")
            bad = journal.scan_for_corruption()
            self.assertEqual(bad, [])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scan_detects_corrupted_line(self):
        """Truncated JSON line → scan_for_corruption returns its line number."""
        journal, tmpdir = _tmp_journal()
        try:
            journal.append(_make_test_intent(), "created")
            # Inject a corrupted line
            with journal.path.open("a", encoding="utf-8") as f:
                f.write('{"intent_id": "bad", "status": TRUNCATED\n')
            journal.append(_make_test_intent(symbol="8015.T"), "created")

            bad = journal.scan_for_corruption()
            self.assertEqual(len(bad), 1)
            self.assertEqual(bad[0], 2)  # second line is corrupted
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scan_logs_journal_corrupt(self):
        """Corrupted line emits [JOURNAL_CORRUPT] log."""
        journal, tmpdir = _tmp_journal()
        try:
            with journal.path.open("w", encoding="utf-8") as f:
                f.write("not json at all\n")
            with self.assertLogs("src.execution.journal", level="ERROR") as cm:
                journal.scan_for_corruption()
            self.assertTrue(any("JOURNAL_CORRUPT" in m for m in cm.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scan_empty_journal_returns_empty(self):
        """Empty journal file → no corrupted lines."""
        journal, tmpdir = _tmp_journal()
        try:
            journal.path.write_text("", encoding="utf-8")
            bad = journal.scan_for_corruption()
            self.assertEqual(bad, [])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_all_skips_corrupted_but_keeps_valid(self):
        """load_all() skips corrupted lines, valid records still returned."""
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent()
            journal.append(intent, "created")
            # Inject corruption between two valid lines
            with journal.path.open("a", encoding="utf-8") as f:
                f.write("CORRUPT_LINE\n")
            intent2 = _make_test_intent(symbol="8015.T")
            journal.append(intent2, "created")

            records = journal.load_all()
            self.assertEqual(len(records), 2)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Gap 7 extras: timeout → reconnect → orphan broker order recovery ──────────

class TestTimeoutReconnectOrphanRecovery(unittest.TestCase):

    def test_orphan_broker_order_detected_after_restart(self):
        """
        Scheduler restart: broker has an order not in journal → phantom_orders.
        Simulates case where journal was lost but broker still has the order.
        """
        journal, tmpdir = _tmp_journal()
        try:
            # Empty journal (simulates journal loss after restart)
            client = _make_broker_client([
                {"ID": "ORD_ORPHAN", "Symbol": "6762", "CumQty": 0}
            ])
            result = reconcile_open_orders(journal, client)
            self.assertIn("ORD_ORPHAN", result.phantom_orders)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_timeout_then_reconnect_full_cycle(self):
        """
        Full cycle: SUBMITTED → UNKNOWN → reconnect → reconcile → FILLED.
        Broker shows CumQty = qty (full fill detected on reconnect).
        """
        journal, tmpdir = _tmp_journal()
        try:
            intent = _make_test_intent(qty=100)
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD_CYCLE")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "network_timeout")

            # Reconnect: broker shows fully filled
            client = _make_broker_client([
                {"ID": "ORD_CYCLE", "CumQty": 100, "Symbol": "6981"}
            ])
            result = reconcile_open_orders(journal, client)

            self.assertIn(intent.intent_id, result.recovered_intents)
            self.assertEqual(journal.get_latest_status(intent.intent_id), "FILLED")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_transition_rejected_at_gate(self):
        """FILLED → SUBMITTED attempt is rejected with ValueError."""
        intent = _make_test_intent()
        apply_transition(intent, IntentStatus.SUBMITTED)
        apply_transition(intent, IntentStatus.ACKED)
        apply_transition(intent, IntentStatus.FILLED, filled_qty=100)

        with self.assertRaises(ValueError):
            apply_transition(intent, IntentStatus.SUBMITTED)

    def test_unknown_recovery_after_scheduler_restart(self):
        """
        Simulate scheduler restart: journal has UNKNOWN intent.
        On startup, reconcile resolves it without resubmitting.
        """
        journal, tmpdir = _tmp_journal()
        try:
            # Write UNKNOWN intent (as if crashed mid-submission)
            intent = _make_test_intent()
            apply_transition(intent, IntentStatus.SUBMITTED, broker_order_id="ORD_RESTART")
            journal.append(intent, "submitted")
            apply_transition(intent, IntentStatus.UNKNOWN)
            journal.append(intent, "crash_unknown")

            # Startup reconcile
            client = _make_broker_client([
                {"ID": "ORD_RESTART", "CumQty": 50, "Symbol": "6981"}
            ])
            result = reconcile_open_orders(journal, client)

            # Recovered as PARTIAL (50/100 filled), NOT resubmitted
            self.assertIn(intent.intent_id, result.recovered_intents)
            status = journal.get_latest_status(intent.intent_id)
            self.assertIn(status, ("PARTIAL", "ACKED", "FILLED"))
            self.assertNotIn(intent.intent_id, result.orphan_intents)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
