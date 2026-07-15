"""
src/runtime/heartbeat.py
Periodic heartbeat updater for ExecutionLock.

Daemon thread that calls lock.update_heartbeat() every interval_sec.
Stops cleanly on stop() or when the owning thread exits.

Usage::

    lock = ExecutionLock(path, mode="live")
    lock.acquire()
    hb = HeartbeatThread(lock, interval_sec=30)
    hb.start()
    try:
        run_main_logic()
    finally:
        hb.stop()
        lock.release()

Context manager::

    with ExecutionLock(path, mode="live") as lock:
        with HeartbeatThread(lock, interval_sec=30):
            run_main_logic()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.runtime.execution_lock import ExecutionLock

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC: float = 30.0


class HeartbeatThread:
    """
    Daemon thread that periodically updates lock.update_heartbeat().

    interval_sec should be significantly less than lock.heartbeat_expiry_sec
    (default expiry=120s, default interval=30s → 4 heartbeats per expiry window).
    """

    def __init__(
        self,
        lock:         "ExecutionLock",
        interval_sec: float = DEFAULT_INTERVAL_SEC,
    ) -> None:
        self._lock       = lock
        self._interval   = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="HeartbeatThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[HEARTBEAT] started pid=%d interval=%.0fs expiry=%.0fs",
            self._lock.pid, self._interval, self._lock._expiry,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval + 2.0))
            self._thread = None
        logger.info("[HEARTBEAT] stopped pid=%d", self._lock.pid)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self._lock.update_heartbeat()
                logger.debug("[HEARTBEAT] tick pid=%d ts=%.0f", self._lock.pid, time.time())
            except Exception as exc:
                logger.warning("[HEARTBEAT] tick failed pid=%d: %s", self._lock.pid, exc)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "HeartbeatThread":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
