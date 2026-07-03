"""
tests/live/test_staged_supervisor.py

Required tests per task spec:
  1. stale lock reclaim
  2. PID reuse protection
  3. broker timeout recovery (StageTimeout)
  4. stage timeout isolation (one stage timeout does not corrupt others)
  5. lock auto-release (finally block)
  6. duplicate execution prevention (ExecutionJournal)
  7. crash recovery replay (recover_incomplete)
  8. ExecutionJournal atomic rename transitions
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.live.staged_supervisor import (
    StagedSupervisor,
    StageError,
    StageTimeout,
    WatchdogLog,
)
from src.live.execution_journal import ExecutionJournal
from src.runtime.execution_lock import (
    ExecutionLock,
    ExecutionLockError,
    HEARTBEAT_EXPIRY_SEC,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_lock(tmp_path: Path) -> Path:
    return tmp_path / "test.lock.json"


@pytest.fixture()
def watchdog_path(tmp_path: Path) -> Path:
    return tmp_path / "watchdog.jsonl"


@pytest.fixture()
def journal_dir(tmp_path: Path) -> Path:
    d = tmp_path / "orders"
    d.mkdir()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stale lock reclaim
# ─────────────────────────────────────────────────────────────────────────────

def test_stale_lock_reclaim(tmp_lock: Path) -> None:
    """A lock with a dead PID and expired heartbeat must be reclaimed."""
    stale_data = {
        "version":             "2",
        "pid":                 999999999,    # almost certainly dead
        "process_create_time": 0.0,
        "hostname":            "ghost-host",
        "boot_id":             "0",
        "command":             "ghost",
        "mode":                "live",
        "run_id":              "stale_run",
        "started_at":          time.time() - 300,
        "heartbeat_at":        time.time() - 300,  # expired (> 30 s)
        "created_at":          time.time() - 300,
        "heartbeat_ts":        time.time() - 300,
    }
    tmp_lock.write_text(json.dumps(stale_data), encoding="utf-8")

    lock = ExecutionLock(tmp_lock, mode="live", heartbeat_expiry_sec=30.0)
    lock.acquire()   # must NOT raise — stale lock should be reclaimed
    assert lock.is_held()
    lock.release()
    assert not tmp_lock.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PID reuse protection
# ─────────────────────────────────────────────────────────────────────────────

def test_pid_reuse_protection(tmp_lock: Path) -> None:
    """Same PID but different create_time → stale (PID reused by another proc)."""
    current_pid = os.getpid()
    stale_data = {
        "version":             "2",
        "pid":                 current_pid,
        "process_create_time": 0.0,         # wrong create_time → PID reuse
        "hostname":            "localhost",
        "boot_id":             "0",
        "command":             "old-cmd",
        "mode":                "live",
        "run_id":              "old_run",
        "started_at":          time.time() - 120,
        "heartbeat_at":        time.time() - 120,  # expired
        "created_at":          time.time() - 120,
        "heartbeat_ts":        time.time() - 120,
    }
    tmp_lock.write_text(json.dumps(stale_data), encoding="utf-8")

    lock = ExecutionLock(tmp_lock, mode="live", heartbeat_expiry_sec=30.0)
    lock.acquire()   # should succeed: create_time mismatch → stale
    assert lock.is_held()
    lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Broker timeout recovery (StageTimeout)
# ─────────────────────────────────────────────────────────────────────────────

def test_broker_timeout_recovery(watchdog_path: Path) -> None:
    """A blocking broker call must raise StageTimeout within timeout_sec."""
    def slow_broker_call():
        time.sleep(5)       # blocks longer than timeout
        return "should not reach"

    t0 = time.monotonic()
    with StagedSupervisor("run_timeout", watchdog_path) as sv:
        with pytest.raises(StageTimeout) as exc_info:
            sv.run_stage("order_execution", slow_broker_call, timeout_sec=1)

    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"StageTimeout should fire within ~1s but took {elapsed:.2f}s"
    exc = exc_info.value
    assert exc.stage == "order_execution"
    assert exc.elapsed >= 1.0
    assert exc.timeout == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage timeout isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_timeout_isolation(watchdog_path: Path) -> None:
    """Timeout in one stage must not prevent subsequent stages from running."""
    results: list = []

    with StagedSupervisor("run_isolation", watchdog_path) as sv:
        with pytest.raises(StageTimeout):
            sv.run_stage("broker_sync", lambda: time.sleep(5), timeout_sec=1)
        # subsequent stage should still execute cleanly
        results.append(sv.run_stage("reconciliation", lambda: "ok", timeout_sec=5))

    assert results == ["ok"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Lock auto-release on exception
# ─────────────────────────────────────────────────────────────────────────────

def test_lock_auto_release_on_exception(tmp_lock: Path) -> None:
    """Lock must be released when the holding process exits (atexit / explicit release)."""
    lock = ExecutionLock(tmp_lock, mode="live", heartbeat_expiry_sec=30.0)
    lock.acquire()
    assert lock.is_held()
    assert tmp_lock.exists()

    # simulate crash-safe release
    try:
        raise RuntimeError("simulated crash")
    except RuntimeError:
        pass
    finally:
        lock.release()

    assert not lock.is_held()
    assert not tmp_lock.exists()

    # a new lock must be acquirable immediately
    lock2 = ExecutionLock(tmp_lock, mode="live", heartbeat_expiry_sec=30.0)
    lock2.acquire()
    assert lock2.is_held()
    lock2.release()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Duplicate execution prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_duplicate_execution_prevention(journal_dir: Path) -> None:
    """has_active() must block a second order for the same (run_id, symbol, side)."""
    journal = ExecutionJournal(journal_dir)
    run_id  = "20260515_090000"

    # first submission
    iid = journal.record_pending(run_id, "7203", "BUY", qty=100, price=2500.0)
    assert journal.has_active(run_id, "7203", "BUY")

    # second submission must be detected as duplicate
    assert journal.has_active(run_id, "7203", "BUY"), \
        "has_active must return True before the first order is filled"

    # different side is not a duplicate
    assert not journal.has_active(run_id, "7203", "SELL")

    # after failure, still visible in failed/ — not in active states
    journal.fail(iid, "pending", error="test")
    assert not journal.has_active(run_id, "7203", "BUY"), \
        "failed order must not block a re-entry"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Crash recovery replay (recover_incomplete)
# ─────────────────────────────────────────────────────────────────────────────

def test_crash_recovery_replay(journal_dir: Path) -> None:
    """Pending and submitted orders must surface in recover_incomplete()."""
    journal = ExecutionJournal(journal_dir)
    run_id  = "20260515_091500"

    iid_a = journal.record_pending(run_id, "6758", "BUY", qty=100, price=3000.0)
    iid_b = journal.record_pending(run_id, "9984", "BUY", qty=100, price=8000.0)
    journal.submit(iid_b)  # iid_b advanced to submitted

    incomplete = journal.recover_incomplete()
    intent_ids = {r["intent_id"] for r in incomplete}
    assert iid_a in intent_ids, "pending order must appear in recover_incomplete"
    assert iid_b in intent_ids, "submitted order must appear in recover_incomplete"

    # filled orders must NOT appear
    iid_c = journal.record_pending(run_id, "4502", "BUY", qty=100, price=1000.0)
    journal.submit(iid_c)
    journal.ack(iid_c, order_id="ORD-999")
    journal.fill(iid_c, fill_price=1001.0)

    incomplete2 = journal.recover_incomplete()
    ids2 = {r["intent_id"] for r in incomplete2}
    assert iid_c not in ids2, "filled order must not appear in recover_incomplete"


# ─────────────────────────────────────────────────────────────────────────────
# 8. ExecutionJournal atomic rename transitions
# ─────────────────────────────────────────────────────────────────────────────

def test_journal_atomic_transitions(journal_dir: Path) -> None:
    """Full lifecycle: pending → submitted → ack → filled."""
    journal = ExecutionJournal(journal_dir)
    run_id  = "20260515_093000"

    iid = journal.record_pending(run_id, "3382", "BUY", qty=200, price=500.0)

    assert (journal_dir / "pending" / f"{iid}.json").exists()
    assert not (journal_dir / "submitted" / f"{iid}.json").exists()

    assert journal.submit(iid)
    assert not (journal_dir / "pending" / f"{iid}.json").exists()
    assert (journal_dir / "submitted" / f"{iid}.json").exists()

    assert journal.ack(iid, order_id="ORD-12345")
    assert not (journal_dir / "submitted" / f"{iid}.json").exists()
    assert (journal_dir / "ack" / f"{iid}.json").exists()
    record = journal.get(iid)
    assert record is not None
    assert record["broker_order_id"] == "ORD-12345"
    assert record["state"] == "ack"

    assert journal.fill(iid, fill_price=501.5)
    assert (journal_dir / "filled" / f"{iid}.json").exists()
    record = journal.get(iid)
    assert record["state"] == "filled"
    assert record["fill_price"] == pytest.approx(501.5)

    # no file should remain in any intermediate directory
    assert not (journal_dir / "ack" / f"{iid}.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: WatchdogLog persists events to JSONL
# ─────────────────────────────────────────────────────────────────────────────

def test_watchdog_log_persists(watchdog_path: Path) -> None:
    wl = WatchdogLog(watchdog_path, run_id="run_wl", lock_owner=os.getpid())
    wl.emit("test_stage", "start", elapsed_sec=0.5, heartbeat_age=1.0,
            retry_count=0, stage_timeout_sec=30)
    wl.emit("test_stage", "complete", elapsed_sec=2.3, heartbeat_age=1.5,
            retry_count=0, last_progress="ok")

    assert watchdog_path.exists()
    lines = [json.loads(ln) for ln in watchdog_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["event"] == "start"
    assert lines[1]["event"] == "complete"
    assert lines[0]["stage"] == "test_stage"
    assert lines[0]["stage_timeout_sec"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: StagedSupervisor retry budget
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_retry_budget(watchdog_path: Path) -> None:
    """StageError after retry_budget exhausted."""
    call_count = {"n": 0}

    def flaky():
        call_count["n"] += 1
        raise ValueError("transient error")

    with StagedSupervisor("run_retry", watchdog_path) as sv:
        with pytest.raises(StageError) as exc_info:
            sv.run_stage("allocation", flaky, timeout_sec=5, retry_budget=2)

    # initial attempt + 2 retries = 3 total calls
    assert call_count["n"] == 3
    assert exc_info.value.stage == "allocation"
    assert isinstance(exc_info.value.cause, ValueError)
