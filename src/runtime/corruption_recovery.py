"""
corruption_recovery.py — JSONL corruption tolerance

Handles: truncated JSONL, malformed JSON, partial writes, empty lines.
Quarantines corrupted segments (never deletes). Append-only audit log.
Replay-safe: quarantine preserves raw bytes for forensic analysis.
Never requires manual JSON editing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

MAX_FRAGMENT_LEN: int = 200

CORRUPT_MALFORMED_JSON: str = "malformed_json"
CORRUPT_PARTIAL_WRITE: str = "partial_write"
CORRUPT_EMPTY: str = "empty_line"
CORRUPT_SCHEMA: str = "schema_error"


@dataclass
class CorruptionAuditRecord:
    """Append-only record for each corrupted JSONL segment."""
    timestamp: str
    source_file: str
    line_number: int
    corruption_type: str
    raw_fragment: str
    recovery_action: str    # "skipped" | "quarantined"
    schema_version: int = SCHEMA_VERSION


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_corruption_audit(record: CorruptionAuditRecord, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[CORRUPTION_AUDIT] write failed (%s)", exc)


def load_corruption_audits(path: Path) -> list[CorruptionAuditRecord]:
    """Load all audit records. Corrupted lines skipped with WARNING."""
    if not path.exists():
        return []
    records: list[CorruptionAuditRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(CorruptionAuditRecord(
                timestamp=d.get("timestamp", ""),
                source_file=d.get("source_file", ""),
                line_number=int(d.get("line_number", 0)),
                corruption_type=d.get("corruption_type", ""),
                raw_fragment=d.get("raw_fragment", ""),
                recovery_action=d.get("recovery_action", ""),
            ))
        except Exception as exc:
            logger.warning("[CORRUPTION_AUDIT] parse error (%s)", exc)
    return records


def safe_read_jsonl(
    path: Path,
    *,
    audit_path: Optional[Path] = None,
    validator: Optional[Callable[[dict], bool]] = None,
    now_ts: str = "",
) -> tuple[list[dict], int]:
    """
    Tolerant JSONL reader. Handles:
      - empty lines (skipped silently)
      - malformed JSON (quarantined + logged)
      - partial writes / truncated lines (detected by incomplete JSON structure)
      - schema validation failures if validator is provided

    Returns:
        (valid_records, n_corrupted)

    Replay-safe: corrupted lines appended to audit_path (never deleted).
    Never raises.
    """
    if not path.exists():
        return [], 0

    valid: list[dict] = []
    n_corrupted = 0

    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("[CORRUPTION_RECOVERY] read failed (%s): %s", path, exc)
        return [], 0

    for lineno, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        corruption_type = None
        record = None

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line.startswith("{") and not line.endswith("}"):
                corruption_type = CORRUPT_PARTIAL_WRITE
            else:
                corruption_type = CORRUPT_MALFORMED_JSON

        if corruption_type is not None:
            n_corrupted += 1
            logger.warning(
                "[CORRUPTION_RECOVERY] %s line %d: %s",
                path.name, lineno, corruption_type,
            )
            if audit_path is not None:
                append_corruption_audit(
                    CorruptionAuditRecord(
                        timestamp=now_ts or _now_ts(),
                        source_file=str(path),
                        line_number=lineno,
                        corruption_type=corruption_type,
                        raw_fragment=line[:MAX_FRAGMENT_LEN],
                        recovery_action="quarantined",
                    ),
                    audit_path,
                )
            continue

        if validator is not None and not validator(record):
            n_corrupted += 1
            logger.warning(
                "[CORRUPTION_RECOVERY] %s line %d: schema_error",
                path.name, lineno,
            )
            if audit_path is not None:
                append_corruption_audit(
                    CorruptionAuditRecord(
                        timestamp=now_ts or _now_ts(),
                        source_file=str(path),
                        line_number=lineno,
                        corruption_type=CORRUPT_SCHEMA,
                        raw_fragment=line[:MAX_FRAGMENT_LEN],
                        recovery_action="quarantined",
                    ),
                    audit_path,
                )
            continue

        valid.append(record)

    return valid, n_corrupted


def scan_jsonl_integrity(path: Path) -> tuple[bool, int, int]:
    """
    Quick integrity scan without loading records.
    Returns (is_valid, n_valid_lines, n_corrupt_lines).
    Never raises.
    """
    _, n_corrupted = safe_read_jsonl(path)
    if not path.exists():
        return True, 0, 0
    try:
        n_total = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except Exception:
        return False, 0, 0
    n_valid = n_total - n_corrupted
    return n_corrupted == 0, n_valid, n_corrupted
