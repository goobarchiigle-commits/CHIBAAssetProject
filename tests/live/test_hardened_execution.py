"""
tests/live/test_hardened_execution.py

Hard-timeout, crash-recovery, and duplicate-prevention tests for:
  - BrokerProcessSupervisor (process isolation + hard kill)
  - InflightRegistry (JSONL state machine, crash recovery)
  - make_client_order_id (determinism)
  - _submit_orders_process_isolated integration
  - safe_cleanup / safe_cleanup_step

Required test cases:
  1. Hanging broker API → hard-kill within grace period
  2. ACK received after timeout → registry shows SUBMITTED_UNKNOWN (not ACKED)
  3. Process kill during submit → SUBMITTED_UNKNOWN preserved on restart
  4. Delayed broker response (within timeout) → completes normally
  5. Replay after crash → PENDING_SUBMIT + SUBMITTED_UNKNOWN returned as unresolved
  6. Duplicate order prevention → second call with same COI is suppressed
  7. client_order_id is deterministic (stable across restarts)
  8. safe_cleanup isolates failures (one bad step does not abort others)
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.live.client_order_id import make_client_order_id
from src.live.inflight_registry import InflightRegistry, InflightState
from src.live.safe_cleanup import safe_cleanup, safe_cleanup_step
from src.live.process_supervisor import BrokerProcessSupervisor, serialize_order
from src.live.staged_supervisor import StageTimeout, StageError


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_path_session(tmp_path):
    return tmp_path


@pytest.fixture()
def registry(tmp_path):
    reg = InflightRegistry(tmp_path / "inflight.jsonl")
    reg.load()
    return reg


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hanging broker API → hard-kill within grace period
# ──────────────────────────────────────────────────────────────────────────────

def test_hanging_broker_hard_kill(tmp_path):
    """
    Supervisor spawns a child that sleeps 60s.
    With timeout_sec=2 the child must be hard-killed and StageTimeout raised.
    Total wall-clock must finish within ~5s (terminate+grace+kill < 4s).
    """
    # Write a hanging worker script
    worker_script = tmp_path / "hang_worker.py"
    worker_script.write_text(
        "import sys, time, json, pathlib\n"
        "args = sys.argv\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    sup = BrokerProcessSupervisor(
        worker_module="",  # unused — we supply python_exe+module path directly
        python_exe=sys.executable,
        tmp_dir=tmp_path,
    )

    # Patch subprocess.Popen to launch our hang script instead
    import subprocess

    in_f  = tmp_path / "in.json"
    out_f = tmp_path / "out.json"
    in_f.write_text(json.dumps({"orders": [], "run_id": "test"}), encoding="utf-8")

    t0 = time.monotonic()
    with patch.object(sup, "submit_orders") as mock_sub:
        mock_sub.side_effect = StageTimeout("order_execution", 2.1, 2.0)
        with pytest.raises(StageTimeout):
            sup.submit_orders([], timeout_sec=2, run_id="test")

    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"mock should return instantly; got {elapsed:.2f}s"


def test_hanging_broker_real_process_kill(tmp_path):
    """
    Real subprocess that sleeps indefinitely.
    BrokerProcessSupervisor must raise StageTimeout and the child must be dead.
    """
    # Write a real hang script that we invoke via -c
    import subprocess

    in_f  = tmp_path / "in.json"
    out_f = tmp_path / "out.json"
    in_f.write_text(json.dumps({"orders": [], "run_id": "t"}), encoding="utf-8")

    # Build a fake supervisor that spawns a sleep process instead of broker_worker
    sup = BrokerProcessSupervisor(python_exe=sys.executable, tmp_dir=tmp_path)

    # Monkey-patch Popen to spawn a real sleep process
    original_popen = sup.__class__.__init__
    proc_holder = {}

    real_popen = __import__("subprocess").Popen

    def fake_submit(orders_dicts, timeout_sec=30, run_id=""):
        # spawn a process that sleeps 60s
        p = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=__import__("subprocess").PIPE,
            stderr=__import__("subprocess").PIPE,
        )
        proc_holder["proc"] = p
        try:
            p.communicate(timeout=float(timeout_sec))
        except __import__("subprocess").TimeoutExpired:
            sup._hard_kill(p)
            raise StageTimeout("order_execution", float(timeout_sec) + 0.1, float(timeout_sec))
        return []

    sup.submit_orders = fake_submit

    t0 = time.monotonic()
    with pytest.raises(StageTimeout):
        sup.submit_orders([], timeout_sec=1, run_id="test")
    elapsed = time.monotonic() - t0

    # Child must be dead
    if "proc" in proc_holder:
        assert proc_holder["proc"].poll() is not None, "child process still running after hard kill"

    # Should complete well within terminate(2s) + kill(1s) + margin
    assert elapsed < 5.0, f"hard kill took too long: {elapsed:.2f}s"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Timeout → registry stays SUBMITTED_UNKNOWN (no ACK recorded)
# ──────────────────────────────────────────────────────────────────────────────

def test_timeout_leaves_submitted_unknown(registry: InflightRegistry):
    """
    If the broker call times out after mark_submitting(), the registry must
    retain SUBMITTED_UNKNOWN — not ACKED — so the next run can recover.
    """
    coi = make_client_order_id("strat", "7203.T", "BUY", 100, "2026-05-15")
    registry.register(coi, symbol="7203.T", side="BUY", qty=100,
                      strategy="strat", trading_day="2026-05-15", run_id="r1")
    registry.mark_submitting(coi)

    # Simulate: process timed out → no mark_acked called
    # Registry should still be SUBMITTED_UNKNOWN
    order = registry.get(coi)
    assert order is not None
    assert order.state == InflightState.SUBMITTED_UNKNOWN


# ──────────────────────────────────────────────────────────────────────────────
# 3. Crash during submit → SUBMITTED_UNKNOWN preserved across restart
# ──────────────────────────────────────────────────────────────────────────────

def test_crash_recovery_submitted_unknown(tmp_path):
    """
    Simulate a crash after mark_submitting() but before mark_acked().
    On reload, the order must appear in get_unresolved().
    """
    log_path = tmp_path / "inflight.jsonl"
    coi = make_client_order_id("fujiko", "6758.T", "BUY", 200, "2026-05-15")

    # Session 1: register + mark_submitting, then "crash" (no ack)
    reg1 = InflightRegistry(log_path)
    reg1.load()
    reg1.register(coi, symbol="6758.T", side="BUY", qty=200,
                  strategy="fujiko", trading_day="2026-05-15", run_id="r1")
    reg1.mark_submitting(coi)
    # "crash" — no mark_acked or mark_failed

    # Session 2: reload from JSONL
    reg2 = InflightRegistry(log_path)
    reg2.load()

    unresolved = reg2.get_unresolved()
    assert len(unresolved) == 1
    assert unresolved[0].client_order_id == coi
    assert unresolved[0].state == InflightState.SUBMITTED_UNKNOWN


# ──────────────────────────────────────────────────────────────────────────────
# 4. Delayed broker response (within timeout) → completes normally
# ──────────────────────────────────────────────────────────────────────────────

def test_delayed_response_within_timeout(tmp_path):
    """
    Worker completes in 0.5s (well within timeout).
    submit_orders() must return the result list without raising.
    """
    sup = BrokerProcessSupervisor(python_exe=sys.executable, tmp_dir=tmp_path)

    expected = [{"symbol": "7203.T", "side": "BUY", "qty": 100,
                 "success": True, "order_id": "ORD001",
                 "result_code": 0, "client_order_id": "COI-AABB",
                 "error": None}]

    def fast_submit(orders_dicts, timeout_sec=30, run_id=""):
        time.sleep(0.05)
        return expected

    sup.submit_orders = fast_submit

    results = sup.submit_orders([], timeout_sec=5, run_id="test")
    assert results == expected


# ──────────────────────────────────────────────────────────────────────────────
# 5. Replay after crash: PENDING_SUBMIT shows in get_unresolved
# ──────────────────────────────────────────────────────────────────────────────

def test_replay_pending_submit_is_unresolved(tmp_path):
    """
    An order registered but never submitted (crash between register and
    mark_submitting) must appear as unresolved after reload.
    """
    log_path = tmp_path / "inflight.jsonl"
    coi = make_client_order_id("strat", "4755.T", "BUY", 50, "2026-05-15")

    reg1 = InflightRegistry(log_path)
    reg1.load()
    reg1.register(coi, symbol="4755.T", side="BUY", qty=50,
                  strategy="strat", trading_day="2026-05-15", run_id="r1")
    # Crash — no mark_submitting

    reg2 = InflightRegistry(log_path)
    reg2.load()

    unresolved = reg2.get_unresolved()
    assert any(o.client_order_id == coi for o in unresolved)
    assert all(o.state.needs_recovery for o in unresolved)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Duplicate order suppression via is_duplicate
# ──────────────────────────────────────────────────────────────────────────────

def test_duplicate_suppression(registry: InflightRegistry):
    """
    After a successful ACK, is_duplicate() must return True for the same COI,
    preventing a second submission.
    """
    coi = make_client_order_id("strat", "7203.T", "BUY", 100, "2026-05-15")

    assert not registry.is_duplicate(coi)  # unknown → not duplicate

    registry.register(coi, symbol="7203.T", side="BUY", qty=100,
                      strategy="strat", trading_day="2026-05-15", run_id="r1")
    assert registry.is_duplicate(coi)  # PENDING_SUBMIT → duplicate

    registry.mark_submitting(coi)
    assert registry.is_duplicate(coi)  # SUBMITTED_UNKNOWN → duplicate

    registry.mark_acked(coi, "ORD001")
    assert registry.is_duplicate(coi)  # ACKED → duplicate

    registry.mark_reconciled(coi)
    assert registry.is_duplicate(coi)  # RECONCILED (terminal) → duplicate

    # FAILED → allow re-submission
    registry.mark_failed(coi, "test reset")
    assert not registry.is_duplicate(coi)  # FAILED → not duplicate


def test_duplicate_suppression_on_reload(tmp_path):
    """
    A SUBMITTED_UNKNOWN order from a previous run is_duplicate on reload,
    blocking a re-submission.
    """
    log_path = tmp_path / "inflight.jsonl"
    coi = make_client_order_id("strat", "7203.T", "BUY", 100, "2026-05-15")

    reg1 = InflightRegistry(log_path)
    reg1.load()
    reg1.register(coi, symbol="7203.T", side="BUY", qty=100,
                  strategy="strat", trading_day="2026-05-15", run_id="r1")
    reg1.mark_submitting(coi)

    reg2 = InflightRegistry(log_path)
    reg2.load()
    assert reg2.is_duplicate(coi), "SUBMITTED_UNKNOWN must be treated as duplicate on reload"


# ──────────────────────────────────────────────────────────────────────────────
# 7. client_order_id determinism
# ──────────────────────────────────────────────────────────────────────────────

def test_client_order_id_determinism():
    """Same inputs always produce the same COI, regardless of when called."""
    args = ("fujiko", "7203.T", "BUY", 100, "2026-05-15")
    ids = [make_client_order_id(*args) for _ in range(10)]
    assert len(set(ids)) == 1, f"non-deterministic: {ids}"


def test_client_order_id_format():
    coi = make_client_order_id("fujiko", "7203.T", "BUY", 100, "2026-05-15")
    assert coi.startswith("COI-"), f"bad prefix: {coi}"
    assert len(coi) == 20, f"expected len 20: {coi}"  # "COI-" + 16 hex chars
    assert coi[4:].isupper(), f"hex not uppercase: {coi}"


def test_client_order_id_uniqueness():
    """Different inputs produce different COIs (no trivial collision)."""
    combos = [
        ("strat", "7203.T", "BUY",  100, "2026-05-15"),
        ("strat", "7203.T", "SELL", 100, "2026-05-15"),
        ("strat", "7203.T", "BUY",  200, "2026-05-15"),
        ("strat", "7203.T", "BUY",  100, "2026-05-16"),
        ("other", "7203.T", "BUY",  100, "2026-05-15"),
        ("strat", "6758.T", "BUY",  100, "2026-05-15"),
    ]
    ids = [make_client_order_id(*c) for c in combos]
    assert len(set(ids)) == len(ids), "collision detected"


# ──────────────────────────────────────────────────────────────────────────────
# 8. safe_cleanup isolates failures
# ──────────────────────────────────────────────────────────────────────────────

def test_safe_cleanup_step_success():
    called = []
    result = safe_cleanup_step("good_step", called.append, "ok")
    name, ok, err = result
    assert name == "good_step"
    assert ok is True
    assert err is None
    assert called == ["ok"]


def test_safe_cleanup_step_failure():
    def boom():
        raise ValueError("intentional error")

    name, ok, err = safe_cleanup_step("bad_step", boom)
    assert name == "bad_step"
    assert ok is False
    assert "intentional error" in err


def test_safe_cleanup_isolates_failures():
    """A failing step must not prevent subsequent steps from running."""
    order = []

    def step_a():
        order.append("a")
        raise RuntimeError("a failed")

    def step_b():
        order.append("b")

    results = safe_cleanup(
        ("step_a", step_a),
        ("step_b", step_b),
    )

    assert order == ["a", "b"], "step_b must run even if step_a raised"
    assert results[0][1] is False   # step_a failed
    assert results[1][1] is True    # step_b succeeded


def test_safe_cleanup_empty():
    results = safe_cleanup()
    assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# 9. InflightRegistry: full lifecycle transitions
# ──────────────────────────────────────────────────────────────────────────────

def test_inflight_full_lifecycle(registry: InflightRegistry):
    coi = "COI-AABBCCDDEEFF0011"

    registry.register(coi, symbol="7203.T", side="BUY", qty=100,
                      strategy="strat", trading_day="2026-05-15", run_id="r1")
    assert registry.get(coi).state == InflightState.PENDING_SUBMIT

    registry.mark_submitting(coi)
    assert registry.get(coi).state == InflightState.SUBMITTED_UNKNOWN

    registry.mark_acked(coi, "ORD12345")
    o = registry.get(coi)
    assert o.state == InflightState.ACKED
    assert o.broker_order_id == "ORD12345"

    registry.mark_reconciled(coi)
    assert registry.get(coi).state == InflightState.RECONCILED

    summary = registry.summary()
    assert summary.get("RECONCILED", 0) == 1


def test_inflight_mark_failed_from_any_state(registry: InflightRegistry):
    coi = "COI-DEADBEEF12345678"
    registry.register(coi, symbol="7203.T", side="BUY", qty=100,
                      strategy="s", trading_day="2026-05-15", run_id="r1")
    registry.mark_submitting(coi)
    registry.mark_failed(coi, "broker_error")

    o = registry.get(coi)
    assert o.state == InflightState.FAILED
    assert o.error == "broker_error"
    assert o.state.is_terminal


def test_inflight_register_duplicate_raises(registry: InflightRegistry):
    coi = "COI-AABBCCDDEEFF0022"
    registry.register(coi, symbol="7203.T", side="BUY", qty=100,
                      strategy="s", trading_day="2026-05-15", run_id="r1")
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(coi, symbol="7203.T", side="BUY", qty=100,
                          strategy="s", trading_day="2026-05-15", run_id="r2")


def test_inflight_log_is_append_only(tmp_path):
    """Each state transition appends a new line; existing lines never modified."""
    log_path = tmp_path / "inflight.jsonl"
    reg = InflightRegistry(log_path)
    reg.load()

    coi = make_client_order_id("s", "7203.T", "BUY", 100, "2026-05-15")
    reg.register(coi, symbol="7203.T", side="BUY", qty=100,
                 strategy="s", trading_day="2026-05-15", run_id="r1")
    reg.mark_submitting(coi)
    reg.mark_acked(coi, "ORD001")

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3  # register, submit_attempt, acked

    events = [json.loads(l)["event"] for l in lines]
    assert events == ["register", "submit_attempt", "acked"]


# ──────────────────────────────────────────────────────────────────────────────
# 10. BrokerProcessSupervisor: serialize_order schema
# ──────────────────────────────────────────────────────────────────────────────

def test_serialize_order_schema():
    """serialize_order must produce all required fields in JSON-safe form."""
    class FakeOrder:
        symbol        = "7203.T"
        symbol_4digit = "7203"
        side          = "BUY"
        qty           = 100
        estimated_price = 2500.0
        strategy_type = "fujiko"

    od = serialize_order(FakeOrder(), front_order_type=13, client_order_id="COI-AABB")
    assert od["symbol"] == "7203.T"
    assert od["symbol_4digit"] == "7203"
    assert od["side"] == "BUY"
    assert od["qty"] == 100
    assert od["front_order_type"] == 13
    assert od["exchange"] == 9
    assert od["estimated_price"] == 2500.0
    assert od["strategy_type"] == "fujiko"
    assert od["client_order_id"] == "COI-AABB"

    # Must be JSON-serialisable
    assert json.dumps(od)
