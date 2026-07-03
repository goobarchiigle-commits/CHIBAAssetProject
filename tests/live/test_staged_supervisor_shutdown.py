"""
tests/live/test_staged_supervisor_shutdown.py

Shutdown-specific tests for StagedSupervisor daemon-thread implementation.

Verifies:
  1. _StageWorker uses daemon=True thread — no _threads_queues registration
  2. StageTimeout raises within timeout + 1s margin
  3. No non-daemon thread leak from _StageWorker after completion
  4. No non-daemon thread leak from _StageWorker after timeout
  5. StagedSupervisor context exit completes in <5s (no executor.shutdown hang)
  6. Successful stage returns value correctly through daemon thread
  7. Stage exception propagates correctly
  8. Multiple sequential stages complete without thread accumulation
  9. shutdown_audit returns expected keys
 10. WatchdogLog set_active_phase / emit_shutdown_audit do not raise
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.live.staged_supervisor import (
    StagedSupervisor,
    StageError,
    StageTimeout,
    WatchdogLog,
    _StageWorker,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _non_daemon_count() -> int:
    return sum(1 for t in threading.enumerate() if not t.daemon)


def _count_stage_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("stage_"))


# ─── _StageWorker tests ────────────────────────────────────────────────────────

class TestStageWorkerDaemon:
    def test_worker_thread_is_daemon(self) -> None:
        """_StageWorker must spawn a daemon thread."""
        spawned: list[threading.Thread] = []
        done = threading.Event()

        def _capture_self() -> int:
            # record current thread inside the worker
            current = threading.current_thread()
            spawned.append(current)
            done.wait(timeout=3)
            return 42

        worker = _StageWorker("test_daemon", _capture_self, timeout=5.0)

        # start worker in background so we can inspect while it runs
        result_holder: list[int] = []

        def _run_worker():
            result_holder.append(worker.run())

        runner = threading.Thread(target=_run_worker, daemon=True)
        runner.start()

        # wait until worker thread spawned
        for _ in range(50):
            if spawned:
                break
            time.sleep(0.05)

        assert spawned, "worker thread never started"
        assert spawned[0].daemon, "worker thread must be daemon=True"
        done.set()
        runner.join(timeout=5)
        assert result_holder == [42]

    def test_worker_not_in_threads_queues(self) -> None:
        """Worker thread must NOT appear in concurrent.futures._threads_queues."""
        import concurrent.futures.thread as _cft

        before = set(id(t) for t in _cft._threads_queues)  # type: ignore[attr-defined]

        worker = _StageWorker("test_tq", lambda: 1, timeout=5.0)
        worker.run()

        after = set(id(t) for t in _cft._threads_queues)  # type: ignore[attr-defined]
        new_ids = after - before
        assert len(new_ids) == 0, (
            f"_StageWorker registered {len(new_ids)} thread(s) in _threads_queues"
        )

    def test_worker_timeout_raises_stage_timeout(self) -> None:
        t0 = time.monotonic()
        worker = _StageWorker("slow_stage", lambda: time.sleep(60), timeout=0.5)
        with pytest.raises(StageTimeout):
            worker.run()
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"StageTimeout took too long: {elapsed:.2f}s"

    def test_worker_no_non_daemon_leak_after_success(self) -> None:
        before = _non_daemon_count()
        worker = _StageWorker("ok_stage", lambda: 99, timeout=5.0)
        assert worker.run() == 99
        time.sleep(0.05)
        after = _non_daemon_count()
        assert after <= before, f"non-daemon thread count increased: {before} → {after}"

    def test_worker_no_non_daemon_leak_after_timeout(self) -> None:
        before = _non_daemon_count()
        worker = _StageWorker("hang_stage", lambda: time.sleep(60), timeout=0.3)
        with pytest.raises(StageTimeout):
            worker.run()
        time.sleep(0.05)
        after = _non_daemon_count()
        assert after <= before, f"non-daemon thread count increased: {before} → {after}"

    def test_worker_exception_propagates(self) -> None:
        def _boom():
            raise ValueError("test-error-xyz")

        worker = _StageWorker("exc_stage", _boom, timeout=5.0)
        with pytest.raises(ValueError, match="test-error-xyz"):
            worker.run()

    def test_worker_returns_value(self) -> None:
        worker = _StageWorker("val_stage", lambda: {"key": "value"}, timeout=5.0)
        result = worker.run()
        assert result == {"key": "value"}


# ─── StagedSupervisor shutdown tests ──────────────────────────────────────────

class TestStagedSupervisorShutdown:
    def _make_supervisor(self, tmp_path: Path) -> StagedSupervisor:
        return StagedSupervisor(
            run_id="test-shutdown",
            watchdog_path=tmp_path / "watchdog.jsonl",
            lock_owner_pid=0,
        )

    def test_context_exit_completes_fast(self, tmp_path: Path) -> None:
        """StagedSupervisor __exit__ must complete in <5s (no executor.shutdown hang)."""
        sup = self._make_supervisor(tmp_path)
        t0 = time.monotonic()
        with sup:
            sup.run_stage("fast", lambda: None, timeout_sec=5)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"__exit__ blocked for {elapsed:.2f}s"

    def test_no_thread_accumulation_across_stages(self, tmp_path: Path) -> None:
        """Stage threads must not accumulate — each completes and is GC'd."""
        before = _count_stage_threads()
        sup = self._make_supervisor(tmp_path)
        with sup:
            for i in range(5):
                sup.run_stage(f"stage_{i}", lambda: time.sleep(0.01), timeout_sec=5)
        time.sleep(0.1)
        after = _count_stage_threads()
        assert after <= before + 1, (
            f"stage threads accumulated: {before} → {after}"
        )

    def test_timeout_stage_does_not_block_supervisor_exit(self, tmp_path: Path) -> None:
        """A stage that times out must not block supervisor __exit__."""
        sup = self._make_supervisor(tmp_path)
        t0 = time.monotonic()
        with pytest.raises(StageTimeout):
            with sup:
                sup.run_stage("blocker", lambda: time.sleep(60), timeout_sec=0.3)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"supervisor exit blocked after StageTimeout: {elapsed:.2f}s"

    def test_multiple_stages_return_correctly(self, tmp_path: Path) -> None:
        sup = self._make_supervisor(tmp_path)
        with sup:
            r1 = sup.run_stage("s1", lambda: 1, timeout_sec=5)
            r2 = sup.run_stage("s2", lambda: "hello", timeout_sec=5)
            r3 = sup.run_stage("s3", lambda: [1, 2, 3], timeout_sec=5)
        assert r1 == 1
        assert r2 == "hello"
        assert r3 == [1, 2, 3]


# ─── WatchdogLog shutdown audit tests ─────────────────────────────────────────

class TestWatchdogLogShutdownAudit:
    def test_set_active_phase_does_not_raise(self, tmp_path: Path) -> None:
        wlog = WatchdogLog(tmp_path / "wd.jsonl", run_id="test", lock_owner=0)
        wlog.set_active_phase("bootstrap")
        wlog.set_active_phase("signal_generation")

    def test_emit_shutdown_audit_returns_dict(self, tmp_path: Path) -> None:
        wlog = WatchdogLog(tmp_path / "wd.jsonl", run_id="test", lock_owner=0)
        wlog.set_active_phase("shutdown_cleanup")
        audit = wlog.emit_shutdown_audit()
        assert isinstance(audit, dict)
        assert "threads_alive" in audit
        assert "non_daemon_count" in audit
        assert "active_child_processes" in audit
        assert "non_daemon_threads" in audit

    def test_emit_shutdown_audit_counts_non_daemon(self, tmp_path: Path) -> None:
        wlog = WatchdogLog(tmp_path / "wd.jsonl", run_id="test", lock_owner=0)
        # MainThread is always non-daemon — count must be >= 1
        audit = wlog.emit_shutdown_audit()
        assert audit["non_daemon_count"] >= 1

    def test_emit_shutdown_audit_written_to_jsonl(self, tmp_path: Path) -> None:
        wlog = WatchdogLog(tmp_path / "wd_audit.jsonl", run_id="test", lock_owner=0)
        wlog.emit_shutdown_audit()
        path = tmp_path / "wd_audit.jsonl"
        assert path.exists()
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        import json
        rec = json.loads(lines[-1])
        assert rec.get("event") == "audit"
        assert rec.get("stage") == "shutdown_audit"
