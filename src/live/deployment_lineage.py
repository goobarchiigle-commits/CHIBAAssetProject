"""
deployment_lineage.py — Immutable deployment lineage chain

Each deployment cycle produces an ordered chain of steps:
  broker_snapshot → reconciliation → integrity_validation
  → authority_decision → deployment_decision
  → broker_acknowledgment → replay_artifact

Each step is cryptographically linked to the previous via SHA-256.
Append-only JSONL. Deterministic ordering. Replay-compatible.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

GENESIS_HASH = "0" * 64

STEP_BROKER_SNAPSHOT       = "broker_snapshot"
STEP_RECONCILIATION        = "reconciliation"
STEP_INTEGRITY_VALIDATION  = "integrity_validation"
STEP_AUTHORITY_DECISION    = "authority_decision"
STEP_DEPLOYMENT_DECISION   = "deployment_decision"
STEP_BROKER_ACKNOWLEDGMENT = "broker_acknowledgment"
STEP_REPLAY_ARTIFACT       = "replay_artifact"

ORDERED_STEPS: List[str] = [
    STEP_BROKER_SNAPSHOT,
    STEP_RECONCILIATION,
    STEP_INTEGRITY_VALIDATION,
    STEP_AUTHORITY_DECISION,
    STEP_DEPLOYMENT_DECISION,
    STEP_BROKER_ACKNOWLEDGMENT,
    STEP_REPLAY_ARTIFACT,
]

_STEP_INDEX: Dict[str, int] = {s: i for i, s in enumerate(ORDERED_STEPS)}


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# LineageEntry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LineageEntry:
    """Single step in the deployment lineage chain."""
    entry_id: str
    run_id: str
    step: str
    step_index: int
    content: dict
    content_hash: str
    prev_hash: str
    entry_hash: str
    timestamp: str
    schema_version: int = SCHEMA_VERSION


def _compute_content_hash(content: dict) -> str:
    return _sha256(_canonical_json(content))


def _compute_entry_hash(
    prev_hash: str,
    content_hash: str,
    step: str,
    run_id: str,
    timestamp: str,
) -> str:
    payload = _canonical_json({
        "content_hash": content_hash,
        "prev_hash": prev_hash,
        "run_id": run_id,
        "step": step,
        "timestamp": timestamp,
    })
    return _sha256(payload)


def _compute_entry_id(
    run_id: str,
    step: str,
    timestamp: str,
    prev_hash: str,
    content_hash: str,
) -> str:
    payload = _canonical_json({
        "content_hash": content_hash,
        "prev_hash": prev_hash,
        "run_id": run_id,
        "step": step,
        "timestamp": timestamp,
    })
    return _sha256(payload)


def create_lineage_entry(
    run_id: str,
    step: str,
    content: dict,
    prev_hash: str,
    timestamp: str = "",
) -> LineageEntry:
    """
    Create a new lineage entry.
    prev_hash must be the entry_hash of the preceding step (or GENESIS_HASH).
    Raises ValueError on unknown step name.
    """
    if step not in _STEP_INDEX:
        raise ValueError(f"Unknown lineage step: {step!r}. Must be one of {ORDERED_STEPS}")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    content_hash = _compute_content_hash(content)
    entry_hash = _compute_entry_hash(prev_hash, content_hash, step, run_id, timestamp)
    entry_id = _compute_entry_id(run_id, step, timestamp, prev_hash, content_hash)

    return LineageEntry(
        entry_id=entry_id,
        run_id=run_id,
        step=step,
        step_index=_STEP_INDEX[step],
        content=dict(content),
        content_hash=content_hash,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
        timestamp=timestamp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DeploymentLineage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeploymentLineage:
    """
    Complete lineage for a single deployment cycle.
    Entries are append-only and must follow ORDERED_STEPS sequence.
    """
    run_id: str
    entries: List[LineageEntry] = field(default_factory=list)

    def tail_hash(self) -> str:
        if not self.entries:
            return GENESIS_HASH
        return self.entries[-1].entry_hash

    def is_complete(self) -> bool:
        if len(self.entries) != len(ORDERED_STEPS):
            return False
        return all(e.step == ORDERED_STEPS[i] for i, e in enumerate(self.entries))

    def verify_chain(self) -> bool:
        """Walk all entries and verify hash linkage. O(n)."""
        prev = GENESIS_HASH
        for entry in self.entries:
            if entry.prev_hash != prev:
                return False
            expected_ch = _compute_content_hash(entry.content)
            if entry.content_hash != expected_ch:
                return False
            expected_eh = _compute_entry_hash(
                entry.prev_hash, entry.content_hash,
                entry.step, entry.run_id, entry.timestamp,
            )
            if entry.entry_hash != expected_eh:
                return False
            prev = entry.entry_hash
        return True

    def get_step(self, step: str) -> Optional[LineageEntry]:
        for e in self.entries:
            if e.step == step:
                return e
        return None

    def next_expected_step(self) -> Optional[str]:
        if len(self.entries) >= len(ORDERED_STEPS):
            return None
        return ORDERED_STEPS[len(self.entries)]


def build_lineage(run_id: str) -> DeploymentLineage:
    return DeploymentLineage(run_id=run_id)


def append_lineage_step(
    lineage: DeploymentLineage,
    step: str,
    content: dict,
    timestamp: str = "",
) -> LineageEntry:
    """
    Append the next step to the lineage in-place. Returns the new entry.
    Raises ValueError on step ordering violations.
    """
    expected = lineage.next_expected_step()
    if expected is not None and step != expected:
        raise ValueError(
            f"Step ordering violation for run_id={lineage.run_id!r}:"
            f" expected {expected!r} but got {step!r}"
        )
    prev_hash = lineage.tail_hash()
    entry = create_lineage_entry(
        run_id=lineage.run_id,
        step=step,
        content=content,
        prev_hash=prev_hash,
        timestamp=timestamp,
    )
    lineage.entries.append(entry)
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def persist_lineage_entry(entry: LineageEntry, path: Path) -> None:
    """Append a single lineage entry to JSONL. Fail-safe. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[LINEAGE] write failed: %s", exc)


def persist_lineage(lineage: DeploymentLineage, path: Path) -> None:
    """Append all entries in the lineage. Fail-safe. Never raises."""
    for entry in lineage.entries:
        persist_lineage_entry(entry, path)


def load_lineage_entries(path: Path) -> List[LineageEntry]:
    if not path.exists():
        return []
    entries: List[LineageEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            entries.append(LineageEntry(
                entry_id=d.get("entry_id", ""),
                run_id=d.get("run_id", ""),
                step=d.get("step", ""),
                step_index=int(d.get("step_index", 0)),
                content=d.get("content", {}),
                content_hash=d.get("content_hash", ""),
                prev_hash=d.get("prev_hash", GENESIS_HASH),
                entry_hash=d.get("entry_hash", ""),
                timestamp=d.get("timestamp", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[LINEAGE] parse error: %s", exc)
    return entries


def reconstruct_lineage(path: Path, run_id: str) -> DeploymentLineage:
    """Reconstruct lineage for a specific run_id from the JSONL log."""
    all_entries = load_lineage_entries(path)
    entries = sorted(
        [e for e in all_entries if e.run_id == run_id],
        key=lambda e: e.step_index,
    )
    return DeploymentLineage(run_id=run_id, entries=entries)
