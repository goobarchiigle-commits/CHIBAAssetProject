"""
intent_snapshot.py — Append-only intent audit log with SHA-256 hash chain

Every state transition of every ExecutionRecord is recorded as an IntentSnapshot.
Entries form a tamper-evident chain: each entry includes the hash of the previous.

Chain invariants:
  1. entry_hash = SHA-256(canonical JSON of all fields except entry_hash)
  2. prev_hash of entry N = entry_hash of entry N-1
  3. First entry: prev_hash = GENESIS_HASH
  4. Chain is forward-only: no entries can be deleted or reordered
  5. Verification walks the full chain — any gap breaks verification

Replay:
  Load all snapshots → sort by timestamp → apply in order.
  Deterministic: same sequence always produces same final state.
  UNKNOWN states are preserved exactly during replay — never auto-resolved.

Design:
  - Append-only JSONL (one entry per line)
  - sort_keys=True in JSON serialization (deterministic)
  - Verification is O(n) chain walk
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.truth.execution_truth import (
    ExecutionRecord, transition_state,
    VALID_TRANSITIONS, TERMINAL_STATES,
    record_to_dict, record_from_dict,
    InvalidTransitionError,
)

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
GENESIS_HASH = "0" * 64   # sentinel for first entry in chain


@dataclass
class IntentSnapshot:
    """
    Immutable record of a single intent state transition.
    Forms a hash-linked chain for auditability.
    """
    snapshot_id: str            # UUID or timestamp-based unique ID
    intent_id: str
    symbol: str
    side: str
    requested_qty: int
    prev_state: str
    new_state: str
    transition_reason: str
    timestamp: str              # ISO-8601 JST
    trading_day: str
    strategy: str

    # Fill data (present for FILLED/PARTIAL_FILL/RESOLVED_FILLED)
    broker_order_id: str = ""
    filled_qty: int = 0
    fill_price: float = 0.0
    remaining_qty: int = 0

    # Hash chain
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""        # computed on creation; empty before hashing

    schema_version: int = 1


def _canonical_json(snapshot: IntentSnapshot) -> str:
    """
    Deterministic JSON serialization for hashing.
    Excludes entry_hash itself (it's computed from all other fields).
    sort_keys=True ensures determinism across Python versions.
    """
    d = asdict(snapshot)
    d.pop("entry_hash", None)   # exclude from hash computation
    return json.dumps(d, sort_keys=True, ensure_ascii=True)


def _compute_hash(snapshot: IntentSnapshot) -> str:
    """Compute SHA-256 of the canonical JSON (without entry_hash field)."""
    canonical = _canonical_json(snapshot)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize_snapshot(snapshot: IntentSnapshot) -> IntentSnapshot:
    """Compute and attach entry_hash. Returns new snapshot (immutable)."""
    import dataclasses
    h = _compute_hash(snapshot)
    return dataclasses.replace(snapshot, entry_hash=h)


def make_snapshot(
    intent_id: str,
    symbol: str,
    side: str,
    requested_qty: int,
    prev_state: str,
    new_state: str,
    reason: str,
    trading_day: str,
    strategy: str,
    prev_hash: str = GENESIS_HASH,
    broker_order_id: str = "",
    filled_qty: int = 0,
    fill_price: float = 0.0,
    remaining_qty: int = 0,
) -> IntentSnapshot:
    """Create and hash a new IntentSnapshot."""
    now = datetime.now(JST).isoformat()
    snap_id = f"{trading_day}_{intent_id[:8]}_{now[-8:].replace(':', '')}"
    snap = IntentSnapshot(
        snapshot_id     = snap_id,
        intent_id       = intent_id,
        symbol          = symbol,
        side            = side,
        requested_qty   = requested_qty,
        prev_state      = prev_state,
        new_state       = new_state,
        transition_reason = reason,
        timestamp       = now,
        trading_day     = trading_day,
        strategy        = strategy,
        broker_order_id = broker_order_id,
        filled_qty      = filled_qty,
        fill_price      = fill_price,
        remaining_qty   = remaining_qty,
        prev_hash       = prev_hash,
        entry_hash      = "",
    )
    return finalize_snapshot(snap)


# ── Journal (append-only JSONL) ───────────────────────────────────────────────

def append_snapshot(snapshot: IntentSnapshot, path: Path) -> None:
    """Append one snapshot to the JSONL journal. Never overwrites existing entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(snapshot), sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_chain(path: Path) -> list[IntentSnapshot]:
    """Load all snapshots from JSONL in order (chronological)."""
    if not path.exists():
        return []
    snapshots: list[IntentSnapshot] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snap = IntentSnapshot(**{
                        k: v for k, v in data.items()
                        if k in IntentSnapshot.__dataclass_fields__
                    })
                    snapshots.append(snap)
                except Exception as _e:
                    logger.warning("intent_snapshot: line %d malformed (%s)", lineno, _e)
    except Exception as exc:
        logger.warning("intent_snapshot: load failed (%s)", exc)
    return snapshots


def verify_chain(snapshots: list[IntentSnapshot]) -> tuple[bool, list[str]]:
    """
    Verify hash chain integrity.

    Returns (is_valid, list_of_violations).
    Empty violations list = chain is valid.
    """
    violations: list[str] = []
    prev_hash = GENESIS_HASH

    for i, snap in enumerate(snapshots):
        # Verify prev_hash links correctly
        if snap.prev_hash != prev_hash:
            violations.append(
                f"entry {i} ({snap.intent_id}): "
                f"prev_hash mismatch: got={snap.prev_hash[:16]}… "
                f"expected={prev_hash[:16]}…"
            )

        # Verify entry_hash is correct for this entry's content
        expected_hash = _compute_hash(snap)
        if snap.entry_hash != expected_hash:
            violations.append(
                f"entry {i} ({snap.intent_id}): "
                f"entry_hash tampered: got={snap.entry_hash[:16]}… "
                f"expected={expected_hash[:16]}…"
            )

        # Advance chain
        prev_hash = snap.entry_hash if snap.entry_hash else expected_hash

    return len(violations) == 0, violations


# ── Replay ────────────────────────────────────────────────────────────────────

def replay_chain(snapshots: list[IntentSnapshot]) -> dict[str, ExecutionRecord]:
    """
    Deterministic replay: apply snapshot sequence to reconstruct current state.

    UNKNOWN states are preserved exactly — never auto-resolved during replay.
    Returns {intent_id: ExecutionRecord} for all intents seen.

    Replay invariant: same snapshot sequence always → same final state.
    """
    records: dict[str, ExecutionRecord] = {}

    for snap in snapshots:
        iid = snap.intent_id

        if iid not in records:
            # First entry for this intent: construct from CREATED
            records[iid] = ExecutionRecord(
                intent_id       = iid,
                symbol          = snap.symbol,
                side            = snap.side,
                requested_qty   = snap.requested_qty,
                state           = "CREATED",
                strategy        = snap.strategy,
                trading_day     = snap.trading_day,
                created_at      = snap.timestamp,
                last_updated_at = snap.timestamp,
            )

        rec = records[iid]

        # Apply the transition recorded in the snapshot
        if snap.new_state == rec.state:
            continue  # idempotent: same state, skip

        try:
            records[iid] = transition_state(
                rec,
                new_state       = snap.new_state,
                reason          = snap.transition_reason,
                broker_order_id = snap.broker_order_id,
                filled_qty      = snap.filled_qty,
                fill_price      = snap.fill_price,
                submitted_qty   = snap.requested_qty,
            )
        except InvalidTransitionError as e:
            logger.warning("replay: invalid transition in snapshot (%s) — skipping", e)

    return records


def get_tail_hash(path: Path) -> str:
    """Return the entry_hash of the last snapshot in the chain (for chaining new entries)."""
    if not path.exists():
        return GENESIS_HASH
    last_hash = GENESIS_HASH
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    last_hash = data.get("entry_hash", GENESIS_HASH)
                except Exception:
                    pass
    except Exception:
        pass
    return last_hash or GENESIS_HASH
