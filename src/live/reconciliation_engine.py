"""
reconciliation_engine.py — Broker vs runtime state reconciliation

Detects:
  - Orphan fills: inflight SUBMITTED_UNKNOWN from a prior trading day (unresolvable)
  - Missing fills: broker position present with no corresponding inflight ACKED record
  - Stale pending: PENDING_SUBMIT/SUBMITTED_UNKNOWN older than threshold
  - Duplicate pending: two non-terminal inflight records for same (symbol, side)
  - Runtime/broker mismatch: runtime qty != broker qty (broker is authoritative)
  - Overnight carry: new BUY pending for a symbol already held per broker

Broker is authoritative. No silent correction. Append-only log. Never raises.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

STALE_PENDING_SEC: float = 900.0    # 15 min — warning only
ORPHAN_FILL_DAY_THRESHOLD: int = 1  # SUBMITTED_UNKNOWN crossing a day boundary = orphan

MISMATCH_ORPHAN_FILL       = "orphan_fill"
MISMATCH_MISSING_FILL      = "missing_fill"
MISMATCH_STALE_PENDING     = "stale_pending"
MISMATCH_DUPLICATE_PENDING = "duplicate_pending"
MISMATCH_RUNTIME_BROKER    = "runtime_broker_mismatch"
MISMATCH_OVERNIGHT_CARRY   = "overnight_carry_mismatch"

SEVERITY_BLOCKING = "blocking"
SEVERITY_WARNING  = "warning"

_BLOCKING_TYPES: frozenset = frozenset({
    MISMATCH_ORPHAN_FILL,
    MISMATCH_DUPLICATE_PENDING,
    MISMATCH_MISSING_FILL,
})


@dataclass
class ReconciliationMismatch:
    mismatch_type: str
    symbol: str
    detail: str
    broker_qty: Optional[int] = None
    runtime_qty: Optional[int] = None
    inflight_state: Optional[str] = None
    client_order_id: Optional[str] = None
    severity: str = SEVERITY_WARNING


@dataclass
class ReconciliationResult:
    timestamp: str
    run_id: str
    mismatches: list   # list[dict] — ReconciliationMismatch serialised
    broker_positions: dict
    runtime_positions: dict
    inflight_summary: dict
    reconciliation_ok: bool
    blocking_mismatches: int
    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation logic
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def _trading_day(ts_str: str) -> str:
    """Extract YYYY-MM-DD from ISO timestamp string. Empty string on failure."""
    dt = _parse_ts(ts_str)
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _age_sec(ts_str: str, now_sec: float) -> float:
    """Seconds since ts_str. Returns large value on failure (conservative)."""
    dt = _parse_ts(ts_str)
    if dt is None:
        return 999999.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return now_sec - dt.timestamp()


def _detect_duplicate_pending(inflight_orders: List[Any]) -> List[ReconciliationMismatch]:
    """Detect (symbol, side) pairs with more than one non-terminal inflight record."""
    mismatches: List[ReconciliationMismatch] = []
    pending: Dict[str, List[Any]] = {}
    for order in inflight_orders:
        state = str(getattr(order, "state", "")).upper()
        if state in ("PENDING_SUBMIT", "SUBMITTED_UNKNOWN"):
            key = f"{order.symbol}:{order.side}"
            pending.setdefault(key, []).append(order)
    for key, orders in pending.items():
        if len(orders) >= 2:
            sym, side = key.split(":", 1)
            cois = ", ".join(o.client_order_id for o in orders)
            mismatches.append(ReconciliationMismatch(
                mismatch_type=MISMATCH_DUPLICATE_PENDING,
                symbol=sym,
                detail=f"duplicate non-terminal inflight: side={side} count={len(orders)} ids=[{cois}]",
                inflight_state="PENDING_SUBMIT/SUBMITTED_UNKNOWN",
                severity=SEVERITY_BLOCKING,
            ))
    return mismatches


def _detect_stale_pending(
    inflight_orders: List[Any],
    now_sec: float,
    stale_threshold_sec: float,
) -> List[ReconciliationMismatch]:
    """Detect pending orders older than stale_threshold_sec."""
    mismatches: List[ReconciliationMismatch] = []
    for order in inflight_orders:
        state = str(getattr(order, "state", "")).upper()
        if state not in ("PENDING_SUBMIT", "SUBMITTED_UNKNOWN"):
            continue
        age = _age_sec(getattr(order, "registered_at", ""), now_sec)
        if age > stale_threshold_sec:
            mismatches.append(ReconciliationMismatch(
                mismatch_type=MISMATCH_STALE_PENDING,
                symbol=order.symbol,
                detail=(
                    f"stale {state}: age={age:.0f}s > threshold={stale_threshold_sec:.0f}s"
                    f" id={order.client_order_id}"
                ),
                inflight_state=state,
                client_order_id=order.client_order_id,
                severity=SEVERITY_WARNING,
            ))
    return mismatches


def _detect_orphan_fills(
    inflight_orders: List[Any],
    today_str: str,
) -> List[ReconciliationMismatch]:
    """
    Detect SUBMITTED_UNKNOWN orders from a prior trading day — outcome unknown,
    cannot proceed without manual/broker reconciliation.
    """
    mismatches: List[ReconciliationMismatch] = []
    for order in inflight_orders:
        state = str(getattr(order, "state", "")).upper()
        if state != "SUBMITTED_UNKNOWN":
            continue
        order_day = _trading_day(getattr(order, "registered_at", ""))
        if order_day and order_day < today_str:
            mismatches.append(ReconciliationMismatch(
                mismatch_type=MISMATCH_ORPHAN_FILL,
                symbol=order.symbol,
                detail=(
                    f"SUBMITTED_UNKNOWN from prior day: registered={order_day}"
                    f" today={today_str} id={order.client_order_id}"
                ),
                inflight_state=state,
                client_order_id=order.client_order_id,
                severity=SEVERITY_BLOCKING,
            ))
    return mismatches


def _detect_runtime_broker_mismatch(
    broker_positions: Dict[str, int],
    runtime_positions: Dict[str, int],
) -> List[ReconciliationMismatch]:
    """Detect position qty discrepancies. Broker is authoritative."""
    mismatches: List[ReconciliationMismatch] = []
    all_symbols = set(broker_positions) | set(runtime_positions)
    for sym in sorted(all_symbols):
        bq = broker_positions.get(sym, 0)
        rq = runtime_positions.get(sym, 0)
        if bq != rq:
            mismatches.append(ReconciliationMismatch(
                mismatch_type=MISMATCH_RUNTIME_BROKER,
                symbol=sym,
                detail=f"runtime_qty={rq} != broker_qty={bq} (broker authoritative)",
                broker_qty=bq,
                runtime_qty=rq,
                severity=SEVERITY_WARNING,
            ))
    return mismatches


def _detect_overnight_carry(
    broker_positions: Dict[str, int],
    inflight_orders: List[Any],
    today_str: str,
) -> List[ReconciliationMismatch]:
    """Detect BUY pending for a symbol already held per broker (overnight carry risk)."""
    mismatches: List[ReconciliationMismatch] = []
    for order in inflight_orders:
        state = str(getattr(order, "state", "")).upper()
        if state not in ("PENDING_SUBMIT", "SUBMITTED_UNKNOWN"):
            continue
        side = str(getattr(order, "side", "")).upper()
        if side != "BUY":
            continue
        sym = order.symbol
        order_day = getattr(order, "trading_day", "")
        if order_day != today_str:
            continue
        if broker_positions.get(sym, 0) > 0:
            mismatches.append(ReconciliationMismatch(
                mismatch_type=MISMATCH_OVERNIGHT_CARRY,
                symbol=sym,
                detail=(
                    f"BUY pending for already-held symbol: broker_qty={broker_positions[sym]}"
                    f" id={order.client_order_id}"
                ),
                broker_qty=broker_positions[sym],
                inflight_state=state,
                client_order_id=order.client_order_id,
                severity=SEVERITY_WARNING,
            ))
    return mismatches


def run_reconciliation(
    broker_positions: Dict[str, int],
    runtime_positions: Dict[str, int],
    inflight_orders: List[Any],
    run_id: str = "",
    now_sec: Optional[float] = None,
    stale_pending_sec: float = STALE_PENDING_SEC,
    timestamp: str = "",
) -> ReconciliationResult:
    """
    Full reconciliation pass. Returns structured result. Never raises.

    Args:
        broker_positions:  symbol → qty from broker API (authoritative)
        runtime_positions: symbol → qty from runtime state
        inflight_orders:   List of InflightOrder-like objects (must have
                           .symbol, .side, .state, .client_order_id,
                           .registered_at, .trading_day attributes)
        run_id:            current run identifier
        now_sec:           epoch seconds (default: time.time())
        stale_pending_sec: age threshold for stale-pending detection
    """
    import time as _time
    if now_sec is None:
        now_sec = _time.time()
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    today_str = datetime.fromtimestamp(now_sec, tz=timezone.utc).strftime("%Y-%m-%d")

    mismatches: List[ReconciliationMismatch] = []

    try:
        mismatches += _detect_duplicate_pending(inflight_orders)
    except Exception as exc:
        logger.warning("[RECONCILE] duplicate_pending detection failed: %s", exc)

    try:
        mismatches += _detect_stale_pending(inflight_orders, now_sec, stale_pending_sec)
    except Exception as exc:
        logger.warning("[RECONCILE] stale_pending detection failed: %s", exc)

    try:
        mismatches += _detect_orphan_fills(inflight_orders, today_str)
    except Exception as exc:
        logger.warning("[RECONCILE] orphan_fill detection failed: %s", exc)

    try:
        if broker_positions or runtime_positions:
            mismatches += _detect_runtime_broker_mismatch(broker_positions, runtime_positions)
    except Exception as exc:
        logger.warning("[RECONCILE] runtime_broker_mismatch detection failed: %s", exc)

    try:
        if broker_positions:
            mismatches += _detect_overnight_carry(broker_positions, inflight_orders, today_str)
    except Exception as exc:
        logger.warning("[RECONCILE] overnight_carry detection failed: %s", exc)

    inflight_summary: Dict[str, int] = {}
    for o in inflight_orders:
        state_val = str(getattr(o, "state", "UNKNOWN"))
        inflight_summary[state_val] = inflight_summary.get(state_val, 0) + 1

    blocking = sum(
        1 for m in mismatches if m.severity == SEVERITY_BLOCKING
    )
    reconciliation_ok = blocking == 0

    if mismatches:
        logger.warning(
            "[RECONCILE] %d mismatches (%d blocking) for run_id=%s",
            len(mismatches), blocking, run_id,
        )
        for m in mismatches:
            log_fn = logger.error if m.severity == SEVERITY_BLOCKING else logger.warning
            log_fn("[RECONCILE] %s %s: %s", m.severity.upper(), m.mismatch_type, m.detail)
    else:
        logger.info("[RECONCILE] ok: 0 mismatches run_id=%s", run_id)

    return ReconciliationResult(
        timestamp=timestamp,
        run_id=run_id,
        mismatches=[asdict(m) for m in mismatches],
        broker_positions=dict(broker_positions),
        runtime_positions=dict(runtime_positions),
        inflight_summary=inflight_summary,
        reconciliation_ok=reconciliation_ok,
        blocking_mismatches=blocking,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_reconciliation_result(result: ReconciliationResult, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[RECONCILE] log write failed: %s", exc)


def load_reconciliation_results(path: Path) -> List[ReconciliationResult]:
    """Load all results from JSONL. Corrupted lines skipped."""
    if not path.exists():
        return []
    results: List[ReconciliationResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            results.append(ReconciliationResult(
                timestamp=d.get("timestamp", ""),
                run_id=d.get("run_id", ""),
                mismatches=d.get("mismatches", []),
                broker_positions=d.get("broker_positions", {}),
                runtime_positions=d.get("runtime_positions", {}),
                inflight_summary=d.get("inflight_summary", {}),
                reconciliation_ok=bool(d.get("reconciliation_ok", True)),
                blocking_mismatches=int(d.get("blocking_mismatches", 0)),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[RECONCILE] parse error: %s", exc)
    return results
