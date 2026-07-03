"""
tools/archive_policy.py
Archive policy for CHIBAAssetProject.

Rules:
  - NEVER delete files automatically
  - NEVER overwrite existing files
  - Move deprecated artifacts into <parent>/archive/ with timestamp suffix
  - Preserve lineage: log every move to state/archive_log.jsonl

Usage:
    python tools/archive_policy.py --source src/backtest/old_strategy.py
    python tools/archive_policy.py --source backtests/wf_final_2026-04-04.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
STATE_DIR  = REPO_ROOT / "state"
ARCHIVE_LOG = STATE_DIR / "archive_log.jsonl"

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[ARCHIVE_POLICY] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def archive_file(
    src: Path,
    *,
    archive_dir: Path | None = None,
    dry_run: bool = False,
    reason: str = "",
) -> Path:
    """
    Move src into archive_dir with a datestamp suffix.

    Args:
        src:         File to archive (must exist).
        archive_dir: Destination directory. Defaults to src.parent/archive/.
        dry_run:     Log intent but do not move.
        reason:      Human-readable reason (logged).

    Returns:
        Destination path (may not exist if dry_run=True).

    Raises:
        FileNotFoundError: src does not exist.
        FileExistsError:   destination already exists (NEVER overwrite).
    """
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    dest_dir = archive_dir or (src.parent / "archive")
    stamp    = datetime.now(JST).strftime("%Y-%m-%d")
    dest     = _unique_dest(dest_dir, src.stem, src.suffix, stamp)

    if dry_run:
        logger.info("DRY-RUN: would archive %s → %s (reason=%s)", src, dest, reason or "none")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    logger.info("ARCHIVED: %s → %s (reason=%s)", src, dest, reason or "none")
    _append_log(src, dest, reason)
    return dest


def _unique_dest(dest_dir: Path, stem: str, suffix: str, stamp: str) -> Path:
    candidate = dest_dir / f"{stem}_{stamp}{suffix}"
    counter   = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{stamp}_{counter}{suffix}"
        counter  += 1
    return candidate


def _append_log(src: Path, dest: Path, reason: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "src":       str(src),
        "dest":      str(dest),
        "reason":    reason,
    }
    with ARCHIVE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_archive_log() -> list[dict]:
    if not ARCHIVE_LOG.exists():
        return []
    records: list[dict] = []
    with ARCHIVE_LOG.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Archive a file with timestamp suffix.")
    parser.add_argument("--source",    required=True, help="File to archive")
    parser.add_argument("--dest-dir",  default="",    help="Override archive directory")
    parser.add_argument("--reason",    default="",    help="Reason for archiving")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    src       = Path(args.source)
    dest_dir  = Path(args.dest_dir) if args.dest_dir else None
    archive_file(src, archive_dir=dest_dir, dry_run=args.dry_run, reason=args.reason)


if __name__ == "__main__":
    _cli()
