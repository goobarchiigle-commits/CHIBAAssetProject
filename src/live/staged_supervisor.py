"""
src/live/staged_supervisor.py

Staged execution supervisor for LIVE trading runtime.

ROOT CAUSE FIX (Python 3.12 ThreadPoolExecutor hang):
  Python 3.12 ThreadPoolExecutor creates non-daemon worker threads.
  CPython's _python_exit() atexit handler calls t.join() (no timeout)
  on all registered threads.  After shutdown(wait=False), if any worker
  is still alive — even briefly — _python_exit() blocks the process
  indefinitely (observed: 600s watchdog timeout in DRY mode).

  Fix: replace ThreadPoolExecutor with daemon threading.Thread per stage.
  Daemon threads are NOT tracked in concurrent.futures.thread._threads_queues
  and are killed at process exit without blocking _python_exit().

Phase instrumentation:
  Each stage emits structured JSONL to watchdog.jsonl with:
    phase, event (start/complete/timeout/error), elapsed_ms,
    active_thread_count, run_id, ts

Shutdown audit:
  Before exit: emit SHUTDOWN_AUDIT record listing alive threads,
  active child processes, non-daemon threads, pending futures.

Stall watchdog:
  ACTIVE_PHASE / LAST_PROGRESS_TS / PHASE_ELAPSED_SEC are written
  into WatchdogLog records.  Warn if no progress >30s.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_STAGE_TIMEOUTS: Dict[str, int] = {
    "market_data":       60,
    "signal_generation": 90,
    "allocation":        20,
    "broker_sync":       20,
    "order_execution":   30,
    "reconciliation":    15,
}

# Bounded waits — no unbounded join() anywhere
_THREAD_JOIN_TIMEOUT:    float = 5.0   # per thread
_PROCESS_JOIN_TIMEOUT:   float = 10.0  # per child process
_STALL_WARN_THRESHOLD:   float = 30.0  # seconds without progress → warning


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class StageTimeout(RuntimeError):
    """Stage exceeded its timeout_sec budget."""

    def __init__(self, stage: str, elapsed: float, timeout: float) -> None:
        super().__init__(
            f"Stage '{stage}' timed out after {elapsed:.1f}s (limit={timeout}s)"
        )
        self.stage   = stage
        self.elapsed = elapsed
        self.timeout = timeout


class StageError(RuntimeError):
    """Stage raised an unhandled exception."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"Stage '{stage}' failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


# ──────────────────────────────────────────────────────────────────────────────
# WatchdogLog — structured JSONL with phase instrumentation
# ──────────────────────────────────────────────────────────────────────────────

class WatchdogLog:
    """
    Thread-safe append-only JSONL watchdog log.

    Each record includes: stage, event, elapsed_ms, elapsed_sec,
    heartbeat_age, retry_count, last_progress, lock_owner, run_id,
    stage_timeout_sec, active_thread_count, active_phase,
    last_progress_ts, phase_elapsed_sec.
    """

    def __init__(self, log_path: Path, run_id: str, lock_owner: int) -> None:
        self._path        = Path(log_path)
        self._run_id      = run_id
        self._lock_owner  = lock_owner
        self._started     = time.monotonic()
        self._mu          = threading.Lock()
        self._active_phase: str   = ""
        self._last_progress_ts: float = time.monotonic()

    def set_active_phase(self, phase: str) -> None:
        self._active_phase      = phase
        self._last_progress_ts  = time.monotonic()

    def emit(
        self,
        stage:             str,
        event:             str,
        *,
        elapsed_sec:       float = 0.0,
        heartbeat_age:     float = 0.0,
        retry_count:       int   = 0,
        last_progress:     str   = "",
        stage_timeout_sec: int   = 0,
        extra:             Optional[Dict[str, Any]] = None,
    ) -> None:
        now_mono = time.monotonic()
        phase_elapsed = now_mono - self._last_progress_ts
        n_threads = threading.active_count()

        record: Dict[str, Any] = {
            "ts":                datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id":            self._run_id,
            "stage":             stage,
            "event":             event,
            "elapsed_ms":        round((now_mono - self._started) * 1000, 1),
            "elapsed_sec":       round(elapsed_sec, 2),
            "heartbeat_age":     round(heartbeat_age, 2),
            "retry_count":       retry_count,
            "last_progress":     last_progress,
            "lock_owner":        self._lock_owner,
            "stage_timeout_sec": stage_timeout_sec,
            "active_thread_count": n_threads,
            "active_phase":        self._active_phase,
            "last_progress_ts":    round(self._last_progress_ts, 3),
            "phase_elapsed_sec":   round(phase_elapsed, 2),
        }
        if extra:
            record.update(extra)

        if phase_elapsed > _STALL_WARN_THRESHOLD:
            record["stall_warning"] = True
            logger.warning(
                "[WATCHDOG] STALL: phase=%s no progress for %.0fs",
                self._active_phase or stage, phase_elapsed,
            )

        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._mu:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception as exc:
                logger.warning("[WATCHDOG] log write failed: %s", exc)

        logger.info(
            "[WATCHDOG] run=%s stage=%s event=%s elapsed=%.1fs hb_age=%.1fs threads=%d",
            self._run_id, stage, event, elapsed_sec, heartbeat_age, n_threads,
        )

    def emit_shutdown_audit(self) -> Dict[str, Any]:
        """
        Collect and persist shutdown state: threads, processes, non-daemon threads.
        Returns the audit dict for the caller's inspection.
        """
        all_threads   = threading.enumerate()
        non_daemon    = [t for t in all_threads if not t.daemon]
        child_procs   = multiprocessing.active_children()

        non_daemon_names = [
            {"name": t.name, "alive": t.is_alive(), "daemon": t.daemon}
            for t in non_daemon
        ]

        audit: Dict[str, Any] = {
            "ts":              datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id":          self._run_id,
            "stage":           "shutdown_audit",
            "event":           "audit",
            "total_elapsed_sec": round(time.monotonic() - self._started, 2),
            "threads_alive":   len(all_threads),
            "non_daemon_count": len(non_daemon),
            "non_daemon_threads": non_daemon_names,
            "active_child_processes": len(child_procs),
            "child_process_pids": [p.pid for p in child_procs],
            "active_phase":    self._active_phase,
        }

        # Log SHUTDOWN_AUDIT to stderr for visibility
        logger.info(
            "[SHUTDOWN_AUDIT] threads_alive=%d non_daemon=%d child_procs=%d",
            len(all_threads), len(non_daemon), len(child_procs),
        )
        for nd in non_daemon_names:
            if nd["name"] != "MainThread":
                logger.warning("[SHUTDOWN_AUDIT] non-daemon thread: %s alive=%s", nd["name"], nd["alive"])
        for cp in child_procs:
            logger.warning("[SHUTDOWN_AUDIT] child process: pid=%d", cp.pid)

        line = json.dumps(audit, ensure_ascii=False) + "\n"
        with self._mu:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception:
                pass

        return audit


# ──────────────────────────────────────────────────────────────────────────────
# _StageWorker — daemon-thread-based stage execution (replaces ThreadPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────

class _StageWorker:
    """
    Run fn() in a daemon thread and wait at most timeout seconds.

    WHY daemon threads (not ThreadPoolExecutor):
      Python 3.12 ThreadPoolExecutor workers are non-daemon.  CPython's
      _python_exit() atexit handler calls t.join() (no timeout!) on every
      thread in _threads_queues.  In DRY mode, if the worker is alive
      (even briefly idle between tasks), _python_exit() blocks the process
      until that thread exits — causing the 600s watchdog timeout.

      Daemon threads are NOT tracked in _threads_queues.  They are killed
      at process exit without blocking _python_exit().

    On stage timeout, the stuck daemon thread is abandoned (it will be
    killed when the process exits).  This preserves the original behavior
    that was intended by ThreadPoolExecutor + future.result(timeout=…).
    """

    __slots__ = ("_stage", "_fn", "_timeout", "_result", "_exc", "_done")

    def __init__(self, stage: str, fn: Callable[[], Any], timeout: float) -> None:
        self._stage   = stage
        self._fn      = fn
        self._timeout = timeout
        self._result  = None
        self._exc: Optional[BaseException] = None
        self._done    = threading.Event()

    def run(self) -> Any:
        """
        Start the daemon thread, wait timeout seconds.
        Returns fn()'s value on success.
        Raises StageTimeout or re-raises fn()'s exception.
        """
        t0 = time.monotonic()
        t = threading.Thread(
            target=self._target,
            name=f"stage_{self._stage}_{int(t0 * 1000) % 100000}",
            daemon=True,   # KEY: not tracked by _python_exit()
        )
        t.start()

        completed = self._done.wait(timeout=self._timeout)
        elapsed   = time.monotonic() - t0

        if not completed:
            # Thread is alive but deadline exceeded.
            # Daemon thread → it will be killed at process exit.
            raise StageTimeout(self._stage, elapsed, self._timeout)

        if self._exc is not None:
            raise self._exc

        return self._result

    def _target(self) -> None:
        try:
            self._result = self._fn()
        except BaseException as exc:
            self._exc = exc
        finally:
            self._done.set()


# ──────────────────────────────────────────────────────────────────────────────
# StagedSupervisor
# ──────────────────────────────────────────────────────────────────────────────

class StagedSupervisor:
    """
    Executes named stages with per-stage timeouts via daemon threads.

    Each stage runs in a dedicated daemon thread; the main thread waits
    with a hard deadline.  On timeout the main thread raises StageTimeout
    and the stuck daemon thread is abandoned (killed at process exit).

    Daemon threads are not tracked by CPython's _python_exit() atexit
    handler, so process exit is never blocked by idle/stuck stage threads.

    Context manager usage::

        with StagedSupervisor(run_id, watchdog_path) as sv:
            result = sv.run_stage("market_data", fetch_fn, timeout_sec=60)
            sv.run_stage("order_execution", send_fn, timeout_sec=30)

    On any exception the supervisor emits a terminal watchdog event before
    propagating.
    """

    def __init__(
        self,
        run_id:         str,
        watchdog_path:  Path,
        lock_owner_pid: int  = 0,
        stage_timeouts: Optional[Dict[str, int]] = None,
        heartbeat_lock: Any  = None,
    ) -> None:
        self._run_id   = run_id
        self._pid      = lock_owner_pid or os.getpid()
        self._timeouts = {**DEFAULT_STAGE_TIMEOUTS, **(stage_timeouts or {})}
        self._hb_lock  = heartbeat_lock
        self._watchdog = WatchdogLog(Path(watchdog_path), run_id, self._pid)
        self._start_ts = time.monotonic()
        # Removed: ThreadPoolExecutor (see module docstring for rationale)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> "StagedSupervisor":
        self._watchdog.set_active_phase("supervisor_enter")
        self._watchdog.emit(
            "supervisor", "start",
            elapsed_sec=0.0,
            last_progress="supervisor_entered",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Emit shutdown_cleanup phase
        self._watchdog.set_active_phase("shutdown_cleanup")
        _shutdown_start = time.monotonic()

        # Shutdown audit: record alive threads/processes before exit
        audit = self._watchdog.emit_shutdown_audit()

        self._watchdog.emit(
            "shutdown_cleanup", "complete",
            elapsed_sec=time.monotonic() - _shutdown_start,
            last_progress="daemon_thread_model_no_executor_shutdown_needed",
            extra={"audit_non_daemon": audit.get("non_daemon_count", 0)},
        )
        self._watchdog.set_active_phase("")

        if exc_type is not None:
            self.emit_terminal(
                "abort",
                reason=f"{exc_type.__name__}: {exc_val}",
            )

    # ── public API ──────────────────────────────────────────────────────────

    def run_stage(
        self,
        stage:        str,
        fn:           Callable[[], Any],
        timeout_sec:  Optional[int] = None,
        retry_budget: int = 0,
    ) -> Any:
        """
        Run fn() in a daemon thread with a hard timeout.

        - timeout_sec: overrides DEFAULT_STAGE_TIMEOUTS[stage] when provided.
        - retry_budget: additional attempts on StageError (not on StageTimeout).
        - Returns fn()'s return value.
        - Raises StageTimeout or StageError on unrecoverable failure.
        """
        timeout   = timeout_sec if timeout_sec is not None \
                    else self._timeouts.get(stage, 60)
        attempt   = 0
        last_exc: Optional[Exception] = None

        while attempt <= retry_budget:
            stage_start = time.monotonic()
            self._watchdog.set_active_phase(stage)
            self._watchdog.emit(
                stage, "start",
                elapsed_sec=self._total_elapsed(),
                heartbeat_age=self._heartbeat_age(),
                retry_count=attempt,
                stage_timeout_sec=timeout,
            )
            try:
                result = self._run_with_timeout(stage, fn, timeout)
                self._watchdog.emit(
                    stage, "complete",
                    elapsed_sec=time.monotonic() - stage_start,
                    heartbeat_age=self._heartbeat_age(),
                    retry_count=attempt,
                    stage_timeout_sec=timeout,
                    last_progress="ok",
                )
                self._watchdog.set_active_phase("")
                return result

            except StageTimeout:
                self._watchdog.emit(
                    stage, "timeout",
                    elapsed_sec=time.monotonic() - stage_start,
                    heartbeat_age=self._heartbeat_age(),
                    retry_count=attempt,
                    stage_timeout_sec=timeout,
                    last_progress=f"timeout_after_{timeout}s",
                )
                self._watchdog.set_active_phase("")
                raise  # never retry a timeout

            except Exception as exc:
                last_exc = exc
                self._watchdog.emit(
                    stage, "error",
                    elapsed_sec=time.monotonic() - stage_start,
                    heartbeat_age=self._heartbeat_age(),
                    retry_count=attempt,
                    stage_timeout_sec=timeout,
                    last_progress=f"attempt_{attempt}_failed",
                    extra={"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
                )
                attempt += 1
                if attempt > retry_budget:
                    self._watchdog.set_active_phase("")
                    raise StageError(stage, exc) from exc
                logger.warning(
                    "[SUPERVISOR] stage=%s attempt=%d/%d retrying: %s",
                    stage, attempt, retry_budget, exc,
                )
                time.sleep(2)

        raise StageError(stage, last_exc or RuntimeError("retry budget exhausted"))

    def emit_terminal(self, event: str, stage: str = "supervisor", reason: str = "") -> None:
        """Persist a terminal watchdog event (abort / crash)."""
        self._watchdog.emit(
            stage, event,
            elapsed_sec=self._total_elapsed(),
            heartbeat_age=self._heartbeat_age(),
            last_progress=reason,
        )

    def get_shutdown_audit(self) -> Dict[str, Any]:
        """Return current shutdown audit without writing to log."""
        all_threads = threading.enumerate()
        non_daemon  = [t for t in all_threads if not t.daemon]
        child_procs = multiprocessing.active_children()
        return {
            "threads_alive":     len(all_threads),
            "non_daemon_count":  len(non_daemon),
            "non_daemon_names":  [t.name for t in non_daemon],
            "child_process_pids": [p.pid for p in child_procs],
            "active_phase":      self._watchdog._active_phase,
            "total_elapsed_sec": round(self._total_elapsed(), 2),
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _run_with_timeout(self, stage: str, fn: Callable[[], Any], timeout: int) -> Any:
        """
        Execute fn() in a daemon thread, wait at most timeout seconds.

        Replaces the previous ThreadPoolExecutor-based implementation.
        Daemon threads are not tracked by _python_exit(), so process exit
        is never blocked even if a stage thread is still alive.
        """
        worker = _StageWorker(stage, fn, float(timeout))
        return worker.run()

    def _total_elapsed(self) -> float:
        return time.monotonic() - self._start_ts

    def _heartbeat_age(self) -> float:
        """Seconds since last heartbeat update in the held ExecutionLock, or 0."""
        if self._hb_lock is None:
            return 0.0
        try:
            data = self._hb_lock._read()
            if data:
                hb = data.get("heartbeat_at") or data.get("heartbeat_ts")
                if hb is not None:
                    return max(0.0, time.time() - float(hb))
        except Exception:
            pass
        return 0.0
