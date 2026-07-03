"""Deterministic artifact serialization, checksumming, and immutable storage.

All functions are pure and stateless. Checksum is SHA256 of canonical JSON.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_checksum(payload: dict) -> str:
    """SHA256 of canonically-serialized payload (sorted keys, no whitespace)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_checksum(payload: dict, checksum: str) -> bool:
    return compute_checksum(payload) == checksum


def stable_hash(obj: Any) -> str:
    """16-char SHA256 hash of any JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    """UTC timestamp in ISO8601 format with microseconds for uniqueness."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_immutable(path: Path, content: dict) -> None:
    """Write JSON artifact. Raises ImmutabilityError if path already exists."""
    if path.exists():
        raise ImmutabilityError(f"Artifact already exists (immutable): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, sort_keys=True, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def read_artifact(path: Path) -> dict:
    """Read and parse a JSON artifact file."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line to a .jsonl file (append-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list:
    """Read all lines from a .jsonl file. Returns [] if file absent."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


class ImmutabilityError(RuntimeError):
    """Raised when attempting to overwrite an immutable artifact."""
