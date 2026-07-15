"""
bootstrap_recovery.py — Post-crash/reboot state recovery

After reboot or crash, performs autonomously:
  1. Crash detection: stale lock with dead PID = prior crash
  2. JSONL integrity scan: detect truncated / partially-written files
  3. JSON state file integrity scan: detect corrupted state blobs
  4. Unresolved order scan: inflight orders from prior run
  5. Allocation + capital state validation
  6. Structured recovery report appended to audit log

Never blocks execution. Fail-open.
Preserves duplicate-order guarantees (read-only on all ledgers).
Zero manual steps required after reboot or crash.
"""
from __future__ import annotations

import json
import logging
import os
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

_MUTEX_REQUIRED: frozenset[str] = frozenset({"pid", "heartbeat_ts"})


@dataclass
class BootstrapRecoveryReport:
    """Structured post-boot recovery report. Append-only."""
    timestamp: str
    run_id: str
    crash_detected: bool
    crash_detail: str
    jsonl_corrupted_files: list
    json_corrupted_files: list
    unresolved_order_count: int
    unresolved_orders: list
    allocation_state_ok: bool
    capital_state_ok: bool
    recovery_actions: list
    bootstrap_ok: bool
    schema_version: int = SCHEMA_VERSION


def _detect_crash_from_lock(lock_file: Path) -> tuple[bool, str]:
    """
    Returns (crash_detected, detail).
    Crash = lock file present with mutex fields + PID not alive.
    """
    if not lock_file.exists():
        return False, ""
    try:
        d = json.loads(lock_file.read_text(encoding="utf-8"))
        if not _MUTEX_REQUIRED.issubset(d.keys()):
            return False, ""
        pid = int(d["pid"])
        expected_ct = d.get("process_create_time")
        if _PSUTIL_OK:
            try:
                proc = _psutil.Process(pid)
                if expected_ct is not None:
                    actual_ct = proc.create_time()
                    if abs(actual_ct - float(expected_ct)) > 2.0:
                        return True, f"pid={pid} create_time_mismatch (PID reused)"
                return False, ""
            except _psutil.NoSuchProcess:
                return True, f"pid={pid} not found — prior crash detected"
            except Exception:
                pass
        # Fallback
        try:
            os.kill(pid, 0)
            return False, ""
        except (OSError, PermissionError):
            return True, f"pid={pid} not responding — prior crash detected"
    except Exception as exc:
        return False, f"lock parse error: {exc}"


def _scan_jsonl(path: Path) -> bool:
    """True if file is valid JSONL (or absent/empty). False if any line fails to parse."""
    if not path.exists():
        return True
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)
        return True
    except Exception:
        return False


def _scan_json(path: Path) -> bool:
    """True if file is valid JSON (or absent). False if corrupted."""
    if not path.exists():
        return True
    try:
        json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return True
    except Exception:
        return False


def _load_unresolved_orders(inflight_file: Path) -> list[dict]:
    """
    Scan inflight registry for non-terminal orders.
    Read-only. Returns list of unresolved order summaries.

    Delegates to InflightRegistry.load()/.get_unresolved() — the canonical
    per-client_order_id event replay (append-only JSONL: last event per
    client_order_id wins; non-terminal = PENDING_SUBMIT/SUBMITTED_UNKNOWN).
    A prior standalone re-implementation here keyed on a "status" field
    that the registry never writes (real field is "state"), so every
    event line — not every order — was counted as unresolved, e.g. one
    fully ACKED order logged across register/submit_attempt/acked events
    was reported as "3 unresolved orders" (RCA 2026-07-08).
    """
    if not inflight_file.exists():
        return []
    try:
        from src.live.inflight_registry import InflightRegistry
        registry = InflightRegistry(inflight_file)
        registry.load()
        return [
            {
                "symbol": o.symbol,
                "side": o.side,
                "status": o.state.value,
                "client_order_id": o.client_order_id,
            }
            for o in registry.get_unresolved()
        ]
    except Exception as exc:
        logger.warning("[BOOTSTRAP] inflight scan error: %s", exc)
        return []


def run_bootstrap_recovery(
    lock_file: Path,
    jsonl_files: Optional[list[Path]] = None,
    json_files: Optional[list[Path]] = None,
    inflight_file: Optional[Path] = None,
    allocation_state_files: Optional[list[Path]] = None,
    capital_state_files: Optional[list[Path]] = None,
    report_path: Optional[Path] = None,
    run_id: str = "",
    timestamp: str = "",
) -> BootstrapRecoveryReport:
    """
    Run full bootstrap recovery sequence. Never raises. Fail-open.
    Appends structured report to report_path if provided.
    """
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    recovery_actions: list[str] = []

    # 1. Crash detection
    crash_detected, crash_detail = _detect_crash_from_lock(lock_file)
    if crash_detected:
        logger.warning("[BOOTSTRAP] crash detected: %s", crash_detail)
        recovery_actions.append(f"crash_detected: {crash_detail}")

    # 2. JSONL integrity scan
    jsonl_corrupted: list[str] = []
    for p in (jsonl_files or []):
        if not _scan_jsonl(p):
            jsonl_corrupted.append(str(p))
            logger.warning("[BOOTSTRAP] corrupted JSONL: %s", p.name)
            recovery_actions.append(f"jsonl_corrupted: {p.name}")

    # 3. JSON state file integrity scan
    json_corrupted: list[str] = []
    for p in (json_files or []):
        if not _scan_json(p):
            json_corrupted.append(str(p))
            logger.warning("[BOOTSTRAP] corrupted JSON: %s", p.name)
            recovery_actions.append(f"json_corrupted: {p.name}")

    # 4. Unresolved order scan
    unresolved: list[dict] = []
    if inflight_file is not None:
        unresolved = _load_unresolved_orders(inflight_file)
        if unresolved:
            logger.warning(
                "[BOOTSTRAP] %d unresolved orders from prior run", len(unresolved),
            )
            recovery_actions.append(f"unresolved_orders: {len(unresolved)}")

    # 5. Allocation state validation
    alloc_ok = True
    for p in (allocation_state_files or []):
        if p.exists() and not _scan_json(p):
            alloc_ok = False
            recovery_actions.append(f"alloc_state_corrupted: {p.name}")

    # 6. Capital state validation
    cap_ok = True
    for p in (capital_state_files or []):
        if p.exists() and not _scan_json(p):
            cap_ok = False
            recovery_actions.append(f"capital_state_corrupted: {p.name}")

    bootstrap_ok = (
        not jsonl_corrupted
        and not json_corrupted
        and alloc_ok
        and cap_ok
    )

    report = BootstrapRecoveryReport(
        timestamp=timestamp,
        run_id=run_id,
        crash_detected=crash_detected,
        crash_detail=crash_detail,
        jsonl_corrupted_files=jsonl_corrupted,
        json_corrupted_files=json_corrupted,
        unresolved_order_count=len(unresolved),
        unresolved_orders=unresolved[:10],
        allocation_state_ok=alloc_ok,
        capital_state_ok=cap_ok,
        recovery_actions=recovery_actions,
        bootstrap_ok=bootstrap_ok,
    )

    if report_path is not None:
        _append_report(report, report_path)

    if not bootstrap_ok:
        logger.warning(
            "[BOOTSTRAP] issues detected: %d recovery actions", len(recovery_actions),
        )
    else:
        logger.info("[BOOTSTRAP] clean start (crash=%s)", crash_detected)

    return report


def _append_report(report: BootstrapRecoveryReport, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[BOOTSTRAP] report write failed (%s)", exc)


def load_bootstrap_reports(path: Path) -> list[BootstrapRecoveryReport]:
    """Load all reports. Corrupted lines skipped."""
    if not path.exists():
        return []
    reports: list[BootstrapRecoveryReport] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            reports.append(BootstrapRecoveryReport(
                timestamp=d.get("timestamp", ""),
                run_id=d.get("run_id", ""),
                crash_detected=bool(d.get("crash_detected", False)),
                crash_detail=d.get("crash_detail", ""),
                jsonl_corrupted_files=d.get("jsonl_corrupted_files", []),
                json_corrupted_files=d.get("json_corrupted_files", []),
                unresolved_order_count=int(d.get("unresolved_order_count", 0)),
                unresolved_orders=d.get("unresolved_orders", []),
                allocation_state_ok=bool(d.get("allocation_state_ok", True)),
                capital_state_ok=bool(d.get("capital_state_ok", True)),
                recovery_actions=d.get("recovery_actions", []),
                bootstrap_ok=bool(d.get("bootstrap_ok", True)),
            ))
        except Exception as exc:
            logger.warning("[BOOTSTRAP] parse error (%s)", exc)
    return reports
