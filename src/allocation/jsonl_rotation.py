"""
jsonl_rotation.py — Monthly JSONL archive rotation

Atomic, replay-safe, corruption-safe monthly rotation.
Never deletes data. Uses os.replace() for atomic rename.
Windows compatible (same-drive rename is atomic on NTFS).

Rotation trigger: active JSONL contains records from a prior calendar month.
Month is derived from the timestamp field of the first non-empty line.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TIMESTAMP_FIELDS: tuple[str, ...] = ("timestamp", "ts", "date", "created_at")


def _read_first_timestamp(path: Path) -> Optional[str]:
    """Read the timestamp from the first non-empty JSONL line. Returns None on any failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                for field in _TIMESTAMP_FIELDS:
                    val = d.get(field)
                    if val and isinstance(val, str):
                        return val
    except Exception:
        pass
    return None


def _parse_year_month(timestamp: str) -> Optional[tuple[int, int]]:
    """Extract (year, month) from ISO-8601 timestamp string. Returns None on failure."""
    try:
        date_part = timestamp[:10]
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        return dt.year, dt.month
    except Exception:
        return None


def _archive_filename(stem: str, year: int, month: int, suffix: str) -> str:
    return f"{stem}_{year:04d}-{month:02d}{suffix}"


def needs_rotation(path: Path, now: Optional[datetime] = None) -> bool:
    """
    True if path exists, is non-empty, and its first record is from a prior month.
    Safe to call at any time — pure read, no side effects.
    """
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
    except OSError:
        return False

    if now is None:
        now = datetime.now(timezone.utc)

    ts = _read_first_timestamp(path)
    if ts is None:
        return False
    ym = _parse_year_month(ts)
    if ym is None:
        return False
    file_year, file_month = ym
    return (file_year, file_month) < (now.year, now.month)


def rotate_monthly(
    path: Path,
    archive_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Optional[Path]:
    """
    Archive the active JSONL to a monthly archive file if rotation is needed.

    Returns the archive Path if rotated, None if not needed or on I/O failure.

    Safety guarantees:
    - os.replace(): atomic on same filesystem (POSIX and Windows NTFS)
    - Never deletes. Archive collision → skip with WARNING (no overwrite)
    - I/O failure → WARNING only, caller continues with original file
    - Replay-safe: archive filename encodes year-month for deterministic identification
    """
    if not needs_rotation(path, now=now):
        return None

    if now is None:
        now = datetime.now(timezone.utc)

    ts = _read_first_timestamp(path)
    if ts is None:
        return None
    ym = _parse_year_month(ts)
    if ym is None:
        return None
    year, month = ym

    suffix = path.suffix or ".jsonl"
    target_dir = archive_dir if archive_dir is not None else path.parent / "archive"
    archive_name = _archive_filename(path.stem, year, month, suffix)
    archive_path = target_dir / archive_name

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            logger.warning(
                "[JSONL_ROTATION] archive already exists, skipping to avoid overwrite: %s",
                archive_path,
            )
            return None
        os.replace(str(path), str(archive_path))
        logger.info("[JSONL_ROTATION] rotated %s → %s", path.name, archive_path.name)
        return archive_path
    except Exception as exc:
        logger.warning(
            "[JSONL_ROTATION] rotation failed (%s) — continuing with active file", exc
        )
        return None


def rotate_if_needed(
    path: Path,
    archive_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Optional[Path]:
    """Alias for rotate_monthly. For call-site readability."""
    return rotate_monthly(path, archive_dir=archive_dir, now=now)


def list_archives(archive_dir: Path, stem: str, suffix: str = ".jsonl") -> list[Path]:
    """
    Return sorted list of archive files matching stem pattern.
    Pattern: {stem}_{YYYY}-{MM}{suffix}
    """
    if not archive_dir.exists():
        return []
    prefix = f"{stem}_"
    return sorted(
        p for p in archive_dir.iterdir()
        if p.name.startswith(prefix) and p.suffix == suffix
    )
