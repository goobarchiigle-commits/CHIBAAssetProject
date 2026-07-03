"""
broker_truth_snapshot.py — Immutable broker state snapshots

Takes a deterministic, checksum-validated snapshot of broker state at a
point in time. Replay-compatible: same portfolio_summary always produces
same checksum when positions are identical.

Append-only JSONL persistence. Never raises. Fail-open.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

SNAPSHOT_STALE_SEC: float = 300.0   # 5 min — stale threshold for integrity gate


@dataclass
class BrokerTruthSnapshot:
    timestamp: str
    run_id: str
    portfolio_mode: str          # "virtual" | "live" | "unknown"
    current_positions_count: int
    open_slots: int
    available_cash: Optional[float]
    current_drawdown: float
    positions_raw: dict          # full positions_api payload from portfolio_summary
    wallet_raw: dict             # full wallet_api payload from portfolio_summary
    cb_state: str                # circuit-breaker state
    checksum: str                # sha256 of canonical payload (all fields except checksum)
    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Checksum
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_payload(snapshot: BrokerTruthSnapshot) -> str:
    """Deterministic JSON string of snapshot fields excluding checksum."""
    d = {
        "timestamp": snapshot.timestamp,
        "run_id": snapshot.run_id,
        "portfolio_mode": snapshot.portfolio_mode,
        "current_positions_count": snapshot.current_positions_count,
        "open_slots": snapshot.open_slots,
        "available_cash": snapshot.available_cash,
        "current_drawdown": snapshot.current_drawdown,
        "positions_raw": snapshot.positions_raw,
        "wallet_raw": snapshot.wallet_raw,
        "cb_state": snapshot.cb_state,
        "schema_version": snapshot.schema_version,
    }
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _compute_checksum(snapshot: BrokerTruthSnapshot) -> str:
    payload = _canonical_payload(snapshot)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_snapshot(snapshot: BrokerTruthSnapshot) -> bool:
    """Return True iff stored checksum matches recomputed checksum."""
    try:
        expected = _compute_checksum(snapshot)
        return snapshot.checksum == expected
    except Exception as exc:
        logger.warning("[SNAPSHOT] checksum validation failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def take_snapshot(
    portfolio_summary: Dict[str, Any],
    run_id: str = "",
    timestamp: str = "",
) -> BrokerTruthSnapshot:
    """
    Build an immutable BrokerTruthSnapshot from portfolio_summary.
    portfolio_summary is the dict returned by bridge.run() result.portfolio_summary.
    Never raises.
    """
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    try:
        ps = portfolio_summary or {}
        snapshot = BrokerTruthSnapshot(
            timestamp=timestamp,
            run_id=run_id,
            portfolio_mode=str(ps.get("portfolio_mode", "unknown")),
            current_positions_count=int(ps.get("current_positions", 0)),
            open_slots=int(ps.get("open_slots", 0)),
            available_cash=_safe_float(ps.get("available_cash")),
            current_drawdown=float(ps.get("current_drawdown", 0.0)),
            positions_raw=_deep_copy_safe(ps.get("positions_api", {})),
            wallet_raw=_deep_copy_safe(ps.get("wallet_api", {})),
            cb_state=str(ps.get("cb_state", "NORMAL")),
            checksum="",
        )
        snapshot.checksum = _compute_checksum(snapshot)
        return snapshot
    except Exception as exc:
        logger.warning("[SNAPSHOT] take_snapshot failed (%s) — returning empty", exc)
        empty = BrokerTruthSnapshot(
            timestamp=timestamp,
            run_id=run_id,
            portfolio_mode="unknown",
            current_positions_count=0,
            open_slots=0,
            available_cash=None,
            current_drawdown=0.0,
            positions_raw={},
            wallet_raw={},
            cb_state="NORMAL",
            checksum="",
        )
        empty.checksum = _compute_checksum(empty)
        return empty


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _deep_copy_safe(obj: Any) -> Any:
    """Deep-copy via JSON round-trip. Returns {} on failure."""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return {}


def is_snapshot_stale(snapshot: BrokerTruthSnapshot, now_sec: float, stale_sec: float = SNAPSHOT_STALE_SEC) -> bool:
    """True if snapshot is older than stale_sec from now."""
    try:
        dt = datetime.fromisoformat(snapshot.timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = now_sec - dt.timestamp()
        return age > stale_sec
    except Exception:
        return True   # unknown age = treat as stale


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_snapshot(snapshot: BrokerTruthSnapshot, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[SNAPSHOT] write failed: %s", exc)


def load_snapshots(path: Path) -> List[BrokerTruthSnapshot]:
    """Load all snapshots. Corrupted lines skipped."""
    if not path.exists():
        return []
    snapshots: List[BrokerTruthSnapshot] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            snapshots.append(BrokerTruthSnapshot(
                timestamp=d.get("timestamp", ""),
                run_id=d.get("run_id", ""),
                portfolio_mode=d.get("portfolio_mode", "unknown"),
                current_positions_count=int(d.get("current_positions_count", 0)),
                open_slots=int(d.get("open_slots", 0)),
                available_cash=_safe_float(d.get("available_cash")),
                current_drawdown=float(d.get("current_drawdown", 0.0)),
                positions_raw=d.get("positions_raw", {}),
                wallet_raw=d.get("wallet_raw", {}),
                cb_state=d.get("cb_state", "NORMAL"),
                checksum=d.get("checksum", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[SNAPSHOT] parse error: %s", exc)
    return snapshots
