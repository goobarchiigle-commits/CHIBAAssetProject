"""
runtime_supervisor.py — Autonomous runtime health supervisor

Monitors at startup (read-only checks):
  1. Orphan lock: stale PID files from prior crash (process dead, lock not cleaned)
  2. Duplicate runtime: another live instance detected
  3. Stale heartbeat: process hung or zombie (heartbeat expired)
  4. Process create_time validation: PID reuse detection

All checks are read-only and fail-open.
Appends structured health report to audit log.
Supplements ExecutionLock — never replaces it.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

DEFAULT_HEARTBEAT_EXPIRY_SEC: float = 30.0
_MUTEX_REQUIRED_FIELDS: frozenset[str] = frozenset({"pid", "heartbeat_ts"})


@dataclass
class RuntimeHealthReport:
    """Structured runtime health snapshot at startup."""
    timestamp: str
    run_id: str
    orphan_lock_detected: bool
    orphan_lock_pid: Optional[int]
    orphan_lock_detail: str
    duplicate_runtime_detected: bool
    stale_heartbeat_detected: bool
    last_heartbeat_age_sec: Optional[float]
    process_create_time_valid: bool
    overall_healthy: bool
    schema_version: int = SCHEMA_VERSION


def _read_lock(lock_file: Path) -> Optional[dict]:
    """Read and validate lock file. Returns None if not a valid mutex lock."""
    try:
        if not lock_file.exists():
            return None
        d = json.loads(lock_file.read_text(encoding="utf-8"))
        if not _MUTEX_REQUIRED_FIELDS.issubset(d.keys()):
            return None
        return d
    except Exception:
        return None


def _process_alive(pid: int, expected_ct: Optional[float] = None) -> bool:
    """True only if pid alive and (if given) create_time within 2s tolerance."""
    if _PSUTIL_OK:
        try:
            proc = _psutil.Process(pid)
            if expected_ct is not None:
                return abs(proc.create_time() - expected_ct) < 2.0
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False
    except Exception:
        return False


def check_runtime_health(
    lock_file: Path,
    run_id: str = "",
    heartbeat_expiry_sec: float = DEFAULT_HEARTBEAT_EXPIRY_SEC,
    now_sec: Optional[float] = None,
    timestamp: str = "",
) -> RuntimeHealthReport:
    """
    Inspect the execution lock and report runtime health.
    Read-only. Never modifies lock file. Never raises.
    """
    if now_sec is None:
        now_sec = time.time()
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    orphan_lock_detected = False
    orphan_lock_pid: Optional[int] = None
    orphan_lock_detail = ""
    duplicate_runtime_detected = False
    stale_heartbeat_detected = False
    last_heartbeat_age_sec: Optional[float] = None
    process_create_time_valid = True

    lock_data = _read_lock(lock_file)
    if lock_data is not None:
        lock_pid_raw = lock_data.get("pid")
        lock_ct = lock_data.get("process_create_time")
        hb_ts = lock_data.get("heartbeat_ts") or lock_data.get("heartbeat_at")

        if hb_ts is not None:
            last_heartbeat_age_sec = round(now_sec - float(hb_ts), 2)
            if last_heartbeat_age_sec > heartbeat_expiry_sec:
                stale_heartbeat_detected = True

        if lock_pid_raw is not None:
            lock_pid = int(lock_pid_raw)
            current_pid = os.getpid()
            alive = _process_alive(lock_pid, lock_ct)

            if alive and lock_pid == current_pid:
                pass  # current process holds lock — healthy
            elif alive and lock_pid != current_pid:
                duplicate_runtime_detected = True
                orphan_lock_detail = f"live duplicate pid={lock_pid}"
            elif not alive:
                orphan_lock_detected = True
                orphan_lock_pid = lock_pid
                orphan_lock_detail = f"orphan: pid={lock_pid} not alive"
                logger.warning("[SUPERVISOR] orphan lock: %s", orphan_lock_detail)

            if lock_ct is not None and alive and _PSUTIL_OK:
                try:
                    actual_ct = _psutil.Process(lock_pid).create_time()
                    process_create_time_valid = abs(actual_ct - float(lock_ct)) < 2.0
                except Exception:
                    pass

    overall_healthy = (
        not orphan_lock_detected
        and not duplicate_runtime_detected
        and not stale_heartbeat_detected
    )

    return RuntimeHealthReport(
        timestamp=timestamp,
        run_id=run_id,
        orphan_lock_detected=orphan_lock_detected,
        orphan_lock_pid=orphan_lock_pid,
        orphan_lock_detail=orphan_lock_detail,
        duplicate_runtime_detected=duplicate_runtime_detected,
        stale_heartbeat_detected=stale_heartbeat_detected,
        last_heartbeat_age_sec=last_heartbeat_age_sec,
        process_create_time_valid=process_create_time_valid,
        overall_healthy=overall_healthy,
    )


def append_health_report(report: RuntimeHealthReport, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[RUNTIME_SUPERVISOR] write failed (%s)", exc)


def load_health_reports(path: Path) -> list[RuntimeHealthReport]:
    """Load all reports. Corrupted lines skipped."""
    if not path.exists():
        return []
    reports: list[RuntimeHealthReport] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            reports.append(RuntimeHealthReport(
                timestamp=d.get("timestamp", ""),
                run_id=d.get("run_id", ""),
                orphan_lock_detected=bool(d.get("orphan_lock_detected", False)),
                orphan_lock_pid=d.get("orphan_lock_pid"),
                orphan_lock_detail=d.get("orphan_lock_detail", ""),
                duplicate_runtime_detected=bool(d.get("duplicate_runtime_detected", False)),
                stale_heartbeat_detected=bool(d.get("stale_heartbeat_detected", False)),
                last_heartbeat_age_sec=d.get("last_heartbeat_age_sec"),
                process_create_time_valid=bool(d.get("process_create_time_valid", True)),
                overall_healthy=bool(d.get("overall_healthy", True)),
            ))
        except Exception as exc:
            logger.warning("[RUNTIME_SUPERVISOR] parse error (%s)", exc)
    return reports
