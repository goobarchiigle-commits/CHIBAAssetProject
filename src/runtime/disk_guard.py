"""
disk_guard.py — Disk space and archive health guard

Detects:
  - Low disk space (< warn/critical threshold)
  - Archive directory growth anomalies
  - Write permission failures

Triggers automatic JSONL rotation when disk is low.
Emits structured warnings. Fail-open telemetry only. Never raises.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

WARN_FREE_PCT: float = 0.10
CRITICAL_FREE_PCT: float = 0.05
ARCHIVE_WARN_BYTES: int = 500 * 1024 * 1024   # 500 MB


@dataclass
class DiskGuardReport:
    """Structured disk health snapshot. Append-only."""
    timestamp: str
    check_path: str
    free_bytes: int
    total_bytes: int
    free_pct: float
    low_disk_warning: bool
    critical_disk_warning: bool
    archive_stats: list
    write_ok: bool
    rotation_triggered: bool
    rotated_files: list
    detail: str
    schema_version: int = SCHEMA_VERSION


def check_disk_space(path: Path) -> tuple[int, int]:
    """Returns (free_bytes, total_bytes). Returns (0, 0) on failure."""
    try:
        usage = shutil.disk_usage(str(path.resolve()))
        return int(usage.free), int(usage.total)
    except Exception as exc:
        logger.warning("[DISK_GUARD] disk_usage failed (%s)", exc)
        return 0, 0


def check_archive_size(archive_dir: Path) -> int:
    """Total bytes in archive_dir recursively. 0 if absent or error."""
    if not archive_dir.exists():
        return 0
    total = 0
    try:
        for p in archive_dir.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


def check_write_permission(path: Path) -> bool:
    """True if we can create+delete a temp file in path's parent."""
    try:
        test = path.parent / ".write_test_tmp"
        path.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("x", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def run_disk_guard(
    check_path: Path,
    archive_dirs: Optional[list[Path]] = None,
    jsonl_paths: Optional[list[Path]] = None,
    warn_free_pct: float = WARN_FREE_PCT,
    critical_free_pct: float = CRITICAL_FREE_PCT,
    archive_warn_bytes: int = ARCHIVE_WARN_BYTES,
    report_path: Optional[Path] = None,
    timestamp: str = "",
) -> DiskGuardReport:
    """
    Full disk guard. Optionally triggers archive rotation when disk is low.
    Never raises. Fail-open.
    """
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    free_bytes, total_bytes = check_disk_space(check_path)
    free_pct = free_bytes / total_bytes if total_bytes > 0 else 1.0

    low_disk = free_pct < warn_free_pct
    critical_disk = free_pct < critical_free_pct

    if critical_disk:
        logger.warning(
            "[DISK_GUARD] CRITICAL: %.1f%% free (%s bytes) on %s",
            free_pct * 100, f"{free_bytes:,}", check_path,
        )
    elif low_disk:
        logger.warning(
            "[DISK_GUARD] LOW DISK: %.1f%% free on %s", free_pct * 100, check_path,
        )

    archive_stats: list[dict] = []
    for ad in (archive_dirs or []):
        size = check_archive_size(ad)
        warn = size > archive_warn_bytes
        archive_stats.append({"path": str(ad), "size_bytes": size, "warning": warn})
        if warn:
            logger.warning(
                "[DISK_GUARD] archive growth: %s = %s bytes", ad.name, f"{size:,}",
            )

    write_ok = check_write_permission(check_path)
    if not write_ok:
        logger.warning("[DISK_GUARD] write permission failure: %s", check_path)

    rotation_triggered = False
    rotated_files: list[str] = []
    if low_disk and jsonl_paths:
        try:
            from src.allocation.jsonl_rotation import rotate_if_needed
            _now = datetime.now(timezone.utc)
            for jp in jsonl_paths:
                result = rotate_if_needed(jp, now=_now)
                if result is not None:
                    rotation_triggered = True
                    rotated_files.append(str(result))
                    logger.info("[DISK_GUARD] auto-rotated %s → %s", jp.name, result.name)
        except Exception as exc:
            logger.warning("[DISK_GUARD] auto-rotation failed (%s)", exc)

    detail_parts: list[str] = []
    if critical_disk:
        detail_parts.append(f"CRITICAL: {free_pct*100:.1f}% free")
    elif low_disk:
        detail_parts.append(f"LOW: {free_pct*100:.1f}% free")
    if not write_ok:
        detail_parts.append("write_permission_failed")
    if rotation_triggered:
        detail_parts.append(f"rotated {len(rotated_files)} files")
    detail = "; ".join(detail_parts) or "ok"

    report = DiskGuardReport(
        timestamp=timestamp,
        check_path=str(check_path),
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        free_pct=round(free_pct, 4),
        low_disk_warning=low_disk,
        critical_disk_warning=critical_disk,
        archive_stats=archive_stats,
        write_ok=write_ok,
        rotation_triggered=rotation_triggered,
        rotated_files=rotated_files,
        detail=detail,
    )

    if report_path is not None:
        _append_report(report, report_path)

    return report


def _append_report(report: DiskGuardReport, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[DISK_GUARD] report write failed (%s)", exc)


def load_disk_reports(path: Path) -> list[DiskGuardReport]:
    """Load all reports. Corrupted lines skipped."""
    if not path.exists():
        return []
    reports: list[DiskGuardReport] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            reports.append(DiskGuardReport(
                timestamp=d.get("timestamp", ""),
                check_path=d.get("check_path", ""),
                free_bytes=int(d.get("free_bytes", 0)),
                total_bytes=int(d.get("total_bytes", 0)),
                free_pct=float(d.get("free_pct", 1.0)),
                low_disk_warning=bool(d.get("low_disk_warning", False)),
                critical_disk_warning=bool(d.get("critical_disk_warning", False)),
                archive_stats=d.get("archive_stats", []),
                write_ok=bool(d.get("write_ok", True)),
                rotation_triggered=bool(d.get("rotation_triggered", False)),
                rotated_files=d.get("rotated_files", []),
                detail=d.get("detail", ""),
            ))
        except Exception as exc:
            logger.warning("[DISK_GUARD] parse error (%s)", exc)
    return reports
