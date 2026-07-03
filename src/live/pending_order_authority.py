"""
pending_order_authority.py — Deterministic pending-order authority lifecycle

Tracks authority over each pending order:
  SUBMITTED → ACKNOWLEDGED → PARTIAL_FILL → FILLED
                           ↘ CANCELLED
                           ↘ REJECTED
  Any non-terminal state  → STALE  (authority TTL expired)
  Any non-terminal state  → UNKNOWN (unresolvable — broker unresponsive)

Detects:
  - stale authority: non-terminal order beyond authority TTL
  - unresolved pending: non-terminal, non-stale orders with no recent update
  - lifecycle divergence: invalid state transitions
  - duplicate authority: same order_id registered twice

Append-only JSONL. Deterministic. Fail-closed on stale/duplicate authority.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

# Authority states
AUTH_SUBMITTED    = "SUBMITTED"
AUTH_ACKNOWLEDGED = "ACKNOWLEDGED"
AUTH_PARTIAL_FILL = "PARTIAL_FILL"
AUTH_FILLED       = "FILLED"
AUTH_CANCELLED    = "CANCELLED"
AUTH_REJECTED     = "REJECTED"
AUTH_STALE        = "STALE"
AUTH_UNKNOWN      = "UNKNOWN"

TERMINAL_STATES: FrozenSet[str] = frozenset({
    AUTH_FILLED, AUTH_CANCELLED, AUTH_REJECTED, AUTH_STALE, AUTH_UNKNOWN,
})

VALID_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    AUTH_SUBMITTED:    frozenset({AUTH_ACKNOWLEDGED, AUTH_CANCELLED, AUTH_REJECTED, AUTH_STALE, AUTH_UNKNOWN}),
    AUTH_ACKNOWLEDGED: frozenset({AUTH_PARTIAL_FILL, AUTH_FILLED, AUTH_CANCELLED, AUTH_REJECTED, AUTH_STALE, AUTH_UNKNOWN}),
    AUTH_PARTIAL_FILL: frozenset({AUTH_FILLED, AUTH_CANCELLED, AUTH_STALE, AUTH_UNKNOWN}),
    AUTH_FILLED:       frozenset(),
    AUTH_CANCELLED:    frozenset(),
    AUTH_REJECTED:     frozenset(),
    AUTH_STALE:        frozenset(),
    AUTH_UNKNOWN:      frozenset(),
}

AUTHORITY_TTL_SEC: float = 900.0

DIVERGENCE_INVALID_TRANSITION  = "invalid_transition"
DIVERGENCE_STALE_AUTHORITY     = "stale_authority"
DIVERGENCE_UNRESOLVED_PENDING  = "unresolved_pending"
DIVERGENCE_DUPLICATE_AUTHORITY = "duplicate_authority"


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# PendingOrderAuthority
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PendingOrderAuthority:
    """Authority record for a single pending order."""
    authority_id: str
    order_id: str
    run_id: str
    symbol: str
    side: str
    requested_qty: int
    authority_state: str
    submitted_at: str
    last_updated_at: str
    authority_epoch_id: str
    filled_qty: int = 0
    fill_price: float = 0.0
    divergence_detected: bool = False
    divergence_reason: str = ""
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        order_id: str,
        run_id: str,
        symbol: str,
        side: str,
        requested_qty: int,
        authority_epoch_id: str,
        submitted_at: str = "",
    ) -> "PendingOrderAuthority":
        if not submitted_at:
            submitted_at = datetime.now(timezone.utc).isoformat()
        authority_id = _sha256(_canonical_json({
            "order_id": order_id,
            "run_id": run_id,
            "submitted_at": submitted_at,
        }))
        return PendingOrderAuthority(
            authority_id=authority_id,
            order_id=order_id,
            run_id=run_id,
            symbol=symbol,
            side=side,
            requested_qty=requested_qty,
            authority_state=AUTH_SUBMITTED,
            submitted_at=submitted_at,
            last_updated_at=submitted_at,
            authority_epoch_id=authority_epoch_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# AuthorityTransitionEvent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuthorityTransitionEvent:
    """Records a single state transition in authority lifecycle."""
    event_id: str
    authority_id: str
    order_id: str
    run_id: str
    prev_state: str
    new_state: str
    transition_reason: str
    timestamp: str
    schema_version: int = SCHEMA_VERSION


def transition_authority(
    authority: PendingOrderAuthority,
    new_state: str,
    reason: str,
    timestamp: str = "",
) -> Tuple[PendingOrderAuthority, AuthorityTransitionEvent]:
    """
    Transition authority state. Returns (updated_authority, event).
    Raises ValueError on invalid transition.
    """
    valid = VALID_TRANSITIONS.get(authority.authority_state, frozenset())
    if new_state not in valid:
        raise ValueError(
            f"Invalid authority transition {authority.authority_state!r} → {new_state!r}"
            f" for order_id={authority.order_id!r}"
        )
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    event_id = _sha256(_canonical_json({
        "authority_id": authority.authority_id,
        "new_state": new_state,
        "order_id": authority.order_id,
        "prev_state": authority.authority_state,
        "timestamp": timestamp,
    }))
    event = AuthorityTransitionEvent(
        event_id=event_id,
        authority_id=authority.authority_id,
        order_id=authority.order_id,
        run_id=authority.run_id,
        prev_state=authority.authority_state,
        new_state=new_state,
        transition_reason=reason,
        timestamp=timestamp,
    )
    updated = dataclasses.replace(
        authority,
        authority_state=new_state,
        last_updated_at=timestamp,
    )
    return updated, event


# ─────────────────────────────────────────────────────────────────────────────
# AuthorityDivergenceReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuthorityDivergenceReport:
    """Summary of authority divergences for a run."""
    run_id: str
    timestamp: str
    stale_authorities: List[str]
    unresolved_pending: List[str]
    invalid_transitions: List[str]
    duplicate_authorities: List[str]
    total_divergences: int
    blocks_deployment: bool
    schema_version: int = SCHEMA_VERSION


def check_stale_authority(
    authority: PendingOrderAuthority,
    now_sec: float,
    ttl_sec: float = AUTHORITY_TTL_SEC,
) -> bool:
    """True if authority is non-terminal and has exceeded TTL."""
    if authority.authority_state in TERMINAL_STATES:
        return False
    try:
        dt = datetime.fromisoformat(authority.last_updated_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now_sec - dt.timestamp()) > ttl_sec
    except Exception:
        return True


def detect_authority_divergences(
    authorities: List[PendingOrderAuthority],
    now_sec: float,
    ttl_sec: float = AUTHORITY_TTL_SEC,
) -> AuthorityDivergenceReport:
    """
    Scan all authority records. Blocks deployment on stale or duplicate authority.
    Unresolved pending (non-terminal, not stale, not duplicate) is informational.
    """
    timestamp = datetime.fromtimestamp(now_sec, tz=timezone.utc).isoformat()
    run_ids: Set[str] = {a.run_id for a in authorities}
    run_id = list(run_ids)[0] if len(run_ids) == 1 else ",".join(sorted(run_ids))

    stale: List[str] = []
    seen: Dict[str, int] = {}
    for auth in authorities:
        seen[auth.order_id] = seen.get(auth.order_id, 0) + 1
        if check_stale_authority(auth, now_sec, ttl_sec):
            stale.append(auth.order_id)

    dupes: List[str] = [oid for oid, cnt in seen.items() if cnt >= 2]

    stale_set = set(stale)
    dupe_set = set(dupes)
    unresolved: List[str] = [
        auth.order_id
        for auth in authorities
        if auth.order_id not in stale_set
        and auth.order_id not in dupe_set
        and auth.authority_state not in TERMINAL_STATES
    ]

    # Remove duplicates while preserving meaning (each unique order_id once)
    stale_unique = sorted(set(stale))
    unresolved_unique = sorted(set(unresolved))
    dupes_unique = sorted(set(dupes))

    blocks = bool(stale_unique or dupes_unique)
    total = len(stale_unique) + len(dupes_unique)

    return AuthorityDivergenceReport(
        run_id=run_id,
        timestamp=timestamp,
        stale_authorities=stale_unique,
        unresolved_pending=unresolved_unique,
        invalid_transitions=[],
        duplicate_authorities=dupes_unique,
        total_divergences=total,
        blocks_deployment=blocks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _safe_append(path: Path, record: dict, tag: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("[%s] write failed: %s", tag, exc)


def append_authority(authority: PendingOrderAuthority, path: Path) -> None:
    _safe_append(path, asdict(authority), "AUTHORITY")


def append_transition_event(event: AuthorityTransitionEvent, path: Path) -> None:
    _safe_append(path, asdict(event), "AUTHORITY_EVENT")


def append_divergence_report(report: AuthorityDivergenceReport, path: Path) -> None:
    _safe_append(path, asdict(report), "AUTHORITY_DIVERGENCE")


def load_authorities(path: Path) -> List[PendingOrderAuthority]:
    if not path.exists():
        return []
    result: List[PendingOrderAuthority] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(PendingOrderAuthority(
                authority_id=d.get("authority_id", ""),
                order_id=d.get("order_id", ""),
                run_id=d.get("run_id", ""),
                symbol=d.get("symbol", ""),
                side=d.get("side", ""),
                requested_qty=int(d.get("requested_qty", 0)),
                authority_state=d.get("authority_state", AUTH_UNKNOWN),
                submitted_at=d.get("submitted_at", ""),
                last_updated_at=d.get("last_updated_at", ""),
                authority_epoch_id=d.get("authority_epoch_id", ""),
                filled_qty=int(d.get("filled_qty", 0)),
                fill_price=float(d.get("fill_price", 0.0)),
                divergence_detected=bool(d.get("divergence_detected", False)),
                divergence_reason=d.get("divergence_reason", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[AUTHORITY] parse error: %s", exc)
    return result


def load_transition_events(path: Path) -> List[AuthorityTransitionEvent]:
    if not path.exists():
        return []
    result: List[AuthorityTransitionEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(AuthorityTransitionEvent(
                event_id=d.get("event_id", ""),
                authority_id=d.get("authority_id", ""),
                order_id=d.get("order_id", ""),
                run_id=d.get("run_id", ""),
                prev_state=d.get("prev_state", ""),
                new_state=d.get("new_state", ""),
                transition_reason=d.get("transition_reason", ""),
                timestamp=d.get("timestamp", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[AUTHORITY_EVENT] parse error: %s", exc)
    return result
