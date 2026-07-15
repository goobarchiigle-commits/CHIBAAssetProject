"""
src/runtime/execution_lock.py
Liveness-aware execution lock system.

Replaces file-existence-only lock with crash-safe deterministic recovery.
PID liveness + process create_time validation prevents PID-reuse false positives.

Recovery triggers:
  - missing process (pid gone)
  - create_time mismatch (PID reused by unrelated process)
  - expired heartbeat (process hung/zombie)
  - interrupted process state

Windows compatible (no fcntl). No async. No distributed locking.
Atomic acquisition via write-then-verify pattern.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Optional

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False

logger = logging.getLogger(__name__)

HEARTBEAT_EXPIRY_SEC: float = 120.0
LOCK_FILE_VERSION:    str   = "2"

# Fields that must be present for a file to be treated as an execution mutex.
# A dict missing these is a non-lock artifact (e.g. a duplicate-order ledger)
# and must not block execution or be deleted.
_MUTEX_SCHEMA_REQUIRED: frozenset[str] = frozenset({"pid", "heartbeat_ts"})


# ─────────────────────────────────────────────────────────────────────────────
# OS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _boot_id() -> str:
    """Windows-compatible boot-ID: system boot time as integer string."""
    if _PSUTIL_OK:
        try:
            return str(int(_psutil.boot_time()))
        except Exception:
            pass
    return "unknown"


def _process_create_time(pid: int) -> Optional[float]:
    """Return create_time for pid, or None if unavailable."""
    if _PSUTIL_OK:
        try:
            return _psutil.Process(pid).create_time()
        except (_psutil.NoSuchProcess, _psutil.AccessDenied, OSError):
            return None
    return None


def _is_process_alive(pid: int, expected_create_time: Optional[float]) -> bool:
    """
    True only if pid exists AND is the same process that wrote the lock.
    create_time mismatch → PID was reused → lock is stale.
    """
    if _PSUTIL_OK:
        try:
            proc = _psutil.Process(pid)
            actual_ct = proc.create_time()
        except (_psutil.NoSuchProcess, _psutil.AccessDenied, OSError):
            return False
        if expected_create_time is None:
            return True
        return abs(actual_ct - expected_create_time) < 2.0  # 2s tolerance
    # fallback: os.kill existence probe (Windows-safe)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False
    except Exception:
        return False


def _heartbeat_expired(heartbeat_ts: Optional[float], expiry_sec: float) -> bool:
    if heartbeat_ts is None:
        return True
    return (time.time() - heartbeat_ts) > expiry_sec


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionLockError(RuntimeError):
    """Raised when lock acquisition fails: another live process holds the lock."""
    def __init__(self, message: str, lock_data: dict) -> None:
        super().__init__(message)
        self.lock_data = lock_data


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionLock
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionLock:
    """
    Deterministic liveness-aware execution lock.

    Guarantees:
      - duplicate execution prevention (same-process + cross-process)
      - automatic stale lock recovery (no human deletion required)
      - PID reuse protection via process create_time
      - heartbeat expiry detection (default 30 s via acquire_runtime_lock)
      - atomic acquisition (write-then-verify)
      - deterministic structured logging

    Lock record fields:
      pid, process_create_time, hostname, run_id, mode,
      started_at, heartbeat_at  (+ legacy created_at / heartbeat_ts aliases)

    Usage::

        lock = ExecutionLock(ORDER_LOCK_FILE, mode="live")
        lock.acquire()          # ExecutionLockError if live process holds it
        try:
            ...
        finally:
            lock.release()

    Context manager::

        with ExecutionLock(ORDER_LOCK_FILE, mode="live"):
            ...
    """

    def __init__(
        self,
        lock_path:            Path,
        mode:                 str   = "dry",
        command:              str   = "",
        heartbeat_expiry_sec: float = HEARTBEAT_EXPIRY_SEC,
        run_id:               str   = "",
    ) -> None:
        self._path    = Path(lock_path)
        self._mode    = mode
        self._command = command or _current_command()
        self._expiry  = heartbeat_expiry_sec
        self._pid     = os.getpid()
        self._ct      = _process_create_time(self._pid)
        self._run_id  = run_id
        self._held    = False

    # ── public API ────────────────────────────────────────────────────────────

    def acquire(self) -> None:
        """
        Acquire lock.

        - Reads existing lock; recovers stale ones automatically.
        - Write-then-verify catches concurrent acquisition races.
        - Raises ExecutionLockError if a live process holds the lock.
        - Non-mutex artifacts (e.g. duplicate-order ledger) are silently ignored;
          they are never interpreted as a live lock and never deleted.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read()
        if existing:
            if not _MUTEX_SCHEMA_REQUIRED.issubset(existing.keys()):
                # Not a mutex lock record — skip entirely, don't delete
                logger.warning(
                    "[EXEC_LOCK] non-lock artifact at mutex path, ignoring "
                    "(keys=%s path=%s)",
                    sorted(existing.keys()), self._path,
                )
            elif self._is_stale(existing):
                self._recover(existing)
            else:
                raise ExecutionLockError(
                    f"[EXEC_LOCK] live process holds lock: "
                    f"pid={existing.get('pid')} mode={existing.get('mode')} "
                    f"started={existing.get('created_at')}",
                    existing,
                )
        try:
            self._write_lock_record()
        except PermissionError as exc:
            # Retry budget exhausted: another process holds the file open.
            # Semantically equivalent to a live lock — raise ExecutionLockError
            # so the exception is always typed and threads never get raw
            # PermissionError leaks.
            raise ExecutionLockError(
                f"[EXEC_LOCK] write contention after retries "
                f"(concurrent acquisition or held handle): {exc}",
                {"pid": None, "mode": self._mode, "winerror": getattr(exc, "winerror", None)},
            ) from exc
        # Verify no concurrent write overwrote ours
        written = self._read()
        if written and written.get("pid") != self._pid:
            raise ExecutionLockError(
                f"[EXEC_LOCK] concurrent acquisition: "
                f"expected pid={self._pid} found pid={written.get('pid')}",
                written,
            )
        self._held = True
        logger.info(
            "[EXEC_LOCK] acquired pid=%d mode=%s path=%s",
            self._pid, self._mode, self._path,
        )

    def release(self) -> None:
        """Release lock owned by this process. No-op if not held."""
        if not self._held:
            return
        existing = self._read()
        if existing and existing.get("pid") == self._pid:
            try:
                self._path.unlink(missing_ok=True)
                logger.info(
                    "[EXEC_LOCK] released pid=%d mode=%s",
                    self._pid, self._mode,
                )
            except Exception as exc:
                logger.warning("[EXEC_LOCK] release failed: %s", exc)
        self._held = False

    def update_heartbeat(self) -> None:
        """Update heartbeat_at (and legacy heartbeat_ts) in lock file. Called by HeartbeatThread."""
        if not self._held:
            return
        existing = self._read()
        if existing and existing.get("pid") == self._pid:
            now = time.time()
            existing["heartbeat_at"] = now   # canonical
            existing["heartbeat_ts"] = now   # legacy alias
            self._write_raw(existing)

    def is_held(self) -> bool:
        return self._held

    @property
    def pid(self) -> int:
        return self._pid

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ExecutionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ── internal ──────────────────────────────────────────────────────────────

    def _read(self) -> Optional[dict]:
        if not self._path.exists():
            return None
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _write_lock_record(self) -> None:
        now = time.time()
        data = {
            "version":             LOCK_FILE_VERSION,
            "pid":                 self._pid,
            "process_create_time": self._ct,
            "hostname":            socket.gethostname(),
            "boot_id":             _boot_id(),
            "command":             self._command,
            "mode":                self._mode,
            "run_id":              self._run_id,
            # canonical names (task spec)
            "started_at":          now,
            "heartbeat_at":        now,
            # legacy aliases preserved for backward compat
            "created_at":          now,
            "heartbeat_ts":        now,
        }
        self._write_raw(data)

    def _write_raw(self, data: dict) -> None:
        import threading
        import uuid as _uuid
        # Unique temp name per writer: eliminates handle contention on Windows
        # when multiple threads/processes call _write_raw concurrently.
        stem = self._path.stem
        unique = f"{stem}.{self._pid}.{threading.get_ident()}.{_uuid.uuid4().hex[:8]}.tmp"
        tmp = self._path.with_name(unique)
        try:
            payload = json.dumps(data, indent=2)
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            _atomic_replace(tmp, self._path)
        finally:
            # Guaranteed cleanup: remove our temp even if replace failed.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _is_stale(self, lock_data: dict) -> bool:
        """
        True if lock_data represents a dead or stale lock.

        Check order:
          1. no pid → stale
          2. process dead → stale
          3. create_time mismatch (PID reused) → stale
          4. heartbeat expired → stale
        """
        pid = lock_data.get("pid")
        if pid is None:
            logger.info("[EXEC_LOCK] stale: no pid in lock")
            return True
        pid = int(pid)
        pct = lock_data.get("process_create_time")

        alive = _is_process_alive(pid, pct)
        if not alive:
            logger.info("[EXEC_LOCK] stale: pid=%d not alive (or create_time mismatch)", pid)
            return True

        # prefer heartbeat_at (canonical); fall back to heartbeat_ts (legacy)
        hb = lock_data.get("heartbeat_at") or lock_data.get("heartbeat_ts")
        if _heartbeat_expired(hb, self._expiry):
            age = f"{time.time() - hb:.0f}s" if hb else "none"
            logger.info(
                "[EXEC_LOCK] stale: heartbeat expired pid=%d hb_age=%s expiry=%.0fs",
                pid, age, self._expiry,
            )
            return True

        return False

    def _recover(self, stale: dict) -> None:
        stale_pid  = stale.get("pid", "?")
        stale_mode = stale.get("mode", "?")
        hb         = stale.get("heartbeat_ts")
        hb_age     = f"{time.time() - hb:.0f}s" if hb else "none"
        logger.warning(
            "[EXEC_LOCK] recovering stale lock: old_pid=%s old_mode=%s hb_age=%s",
            stale_pid, stale_mode, hb_age,
        )
        _emit_recovery_event(
            lock_path        = self._path,
            stale_pid        = stale_pid,
            stale_mode       = stale_mode,
            hb_age           = hb_age,
            recovered_by_pid = self._pid,
        )
        try:
            self._path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("[EXEC_LOCK] stale removal failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic replace — Windows-safe with bounded retry
# ─────────────────────────────────────────────────────────────────────────────

_REPLACE_MAX_RETRIES: int   = 7
_REPLACE_BASE_DELAY:  float = 0.005   # 5 ms initial backoff

# Windows error codes that indicate transient file-handle contention:
#   5  = ERROR_ACCESS_DENIED    — another handle has the file open
#   32 = ERROR_SHARING_VIOLATION — explicit sharing lock held by another reader
_WIN_CONTENTION_ERRORS: frozenset[int] = frozenset({5, 32})


def _atomic_replace(src: Path, dst: Path) -> None:
    """
    Rename src → dst atomically with bounded exponential retry on Windows.

    Retries on WinError 5 (access denied) and WinError 32 (sharing violation),
    both of which occur when another thread holds a read handle on dst.
    All other exceptions propagate immediately without retry.

    Max total wait ≈ 5 ms * (2^7 - 1) ≈ 635 ms — bounded for lock writes.
    """
    delay = _REPLACE_BASE_DELAY
    for attempt in range(_REPLACE_MAX_RETRIES + 1):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:
            winerror = getattr(exc, "winerror", None)
            # Propagate immediately on non-contention errors (e.g. disk full, bad path)
            if winerror is not None and winerror not in _WIN_CONTENTION_ERRORS:
                raise
            if attempt == _REPLACE_MAX_RETRIES:
                logger.error(
                    "[EXEC_LOCK] atomic replace failed after %d retries "
                    "(WinError=%s): %s",
                    _REPLACE_MAX_RETRIES, winerror, exc,
                )
                raise
            logger.debug(
                "[EXEC_LOCK] replace contention WinError=%s attempt=%d delay=%.3fs",
                winerror, attempt, delay,
            )
            time.sleep(delay)
            delay *= 2


# ─────────────────────────────────────────────────────────────────────────────
# Recovery event logging
# ─────────────────────────────────────────────────────────────────────────────

def _emit_recovery_event(
    lock_path:        Path,
    stale_pid:        object,
    stale_mode:       str,
    hb_age:           str,
    recovered_by_pid: int,
) -> None:
    """Append stale recovery event to JSONL governance log next to lock file."""
    log_path = lock_path.parent / "lock_recovery.jsonl"
    event = {
        "ts":               time.time(),
        "event":            "stale_lock_recovered",
        "stale_pid":        stale_pid,
        "stale_mode":       stale_mode,
        "heartbeat_age":    hb_age,
        "recovered_by_pid": recovered_by_pid,
        "lock_path":        str(lock_path),
    }
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as exc:
        logger.warning("[EXEC_LOCK] recovery event log failed: %s", exc)


def read_recovery_events(lock_path: Path) -> list[dict]:
    """Load lock_recovery.jsonl adjacent to lock_path."""
    log_path = lock_path.parent / "lock_recovery.jsonl"
    if not log_path.exists():
        return []
    events = []
    try:
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception:
        pass
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _current_command() -> str:
    import sys
    argv = getattr(sys, "argv", [])
    return " ".join(str(a) for a in argv[:3]) if argv else "unknown"
