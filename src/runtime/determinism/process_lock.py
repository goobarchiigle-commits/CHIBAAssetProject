"""
src/runtime/determinism/process_lock.py

ProcessLock: validation-phase process mutex.

Semantically separated from runtime/order_lock.json (order deduplication ledger).
Uses ONLY: runtime/execution.lock.json

Delegates liveness/staleness logic to ExecutionLock.
Adds: atexit cleanup, integrated heartbeat, monotonic internal timing.
Mutex schema validation enforced by the underlying ExecutionLock layer.
"""
from __future__ import annotations

import atexit
import logging
import time
from pathlib import Path
from typing import Optional

from src.runtime.execution_lock import ExecutionLock, ExecutionLockError
from src.runtime.heartbeat import HeartbeatThread

logger = logging.getLogger(__name__)

LOCK_PATH: Path = Path("runtime/execution.lock.json")

_HB_INTERVAL_SEC: float = 30.0
_HB_EXPIRY_SEC:   float = 120.0


class ProcessLockError(RuntimeError):
    """Raised when ProcessLock cannot be acquired."""


class ProcessLock:
    """
    Validation-phase execution mutex backed by runtime/execution.lock.json.

    Properties guaranteed:
    - atomic acquisition (via ExecutionLock write-then-verify)
    - Windows-safe atomic replace
    - PID validation + create_time staleness detection
    - heartbeat thread (daemon, auto-started on acquire)
    - stale lock auto-recovery (no manual deletion required)
    - concurrent startup safe (raises ProcessLockError if live lock exists)
    - fail-closed (any ambiguous state → error, not silent pass)
    - context manager support (__enter__/__exit__)
    - atexit cleanup registered on first acquire
    - monotonic timing for internal duration tracking

    Path contract: NEVER touches runtime/order_lock.json.
    """

    def __init__(
        self,
        lock_path:          Optional[Path] = None,
        mode:               str   = "validation",
        heartbeat_interval: float = _HB_INTERVAL_SEC,
        heartbeat_expiry:   float = _HB_EXPIRY_SEC,
    ) -> None:
        self._path             = Path(lock_path) if lock_path is not None else LOCK_PATH
        self._mode             = mode
        self._hb_interval      = heartbeat_interval
        self._hb_expiry        = heartbeat_expiry
        self._lock:            Optional[ExecutionLock]  = None
        self._heartbeat:       Optional[HeartbeatThread] = None
        self._acquired_mono:   Optional[float] = None
        self._atexit_registered = False

    # ── public API ────────────────────────────────────────────────────────────

    def acquire(self) -> None:
        if self._lock is not None and self._lock.is_held():
            raise ProcessLockError("ProcessLock already acquired by this instance")

        inner = ExecutionLock(
            lock_path=self._path,
            mode=self._mode,
            heartbeat_expiry_sec=self._hb_expiry,
        )
        try:
            inner.acquire()
        except ExecutionLockError as exc:
            raise ProcessLockError(str(exc)) from exc

        self._lock           = inner
        self._acquired_mono  = time.monotonic()

        self._heartbeat = HeartbeatThread(self._lock, interval_sec=self._hb_interval)
        self._heartbeat.start()

        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        logger.info("[PROCESS_LOCK] acquired mode=%s path=%s", self._mode, self._path)

    def release(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None
        if self._acquired_mono is not None:
            elapsed = time.monotonic() - self._acquired_mono
            logger.info("[PROCESS_LOCK] released held_sec=%.1f", elapsed)
            self._acquired_mono = None

    def is_held(self) -> bool:
        return self._lock is not None and self._lock.is_held()

    def held_duration_sec(self) -> Optional[float]:
        """Monotonic duration since acquire. None if not held."""
        if self._acquired_mono is None:
            return None
        return time.monotonic() - self._acquired_mono

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── atexit ────────────────────────────────────────────────────────────────

    def _atexit_cleanup(self) -> None:
        try:
            self.release()
        except Exception as exc:
            logger.warning("[PROCESS_LOCK] atexit cleanup error: %s", exc)
