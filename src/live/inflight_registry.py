"""
src/live/inflight_registry.py

Inflight order state machine with crash-safe JSONL persistence.

State transitions:
    PENDING_SUBMIT     — recorded; broker API call not yet attempted
        ↓ (submit attempt starts)
    SUBMITTED_UNKNOWN  — API call made; no ACK received (crash/timeout)
        ↓ (broker confirms receipt)
    ACKED              — got order_id from broker
        ↓ (reconcile against live broker state)
    RECONCILED         — confirmed against broker
        → FAILED       — terminal failure at any state

Persistence:
    Append-only JSONL log.  Each line is an event record.  State is
    reconstructed on startup by replaying: the LAST event for a given
    client_order_id determines its current state.

    Append-only means no partial writes corrupt existing state — a
    crash between two appends leaves the log consistent up to the last
    complete line.

Crash recovery:
    On restart, `get_unresolved()` returns all orders in
    PENDING_SUBMIT or SUBMITTED_UNKNOWN.  The caller must reconcile
    these against the live broker state before placing new orders.

Duplicate prevention:
    `is_duplicate(client_order_id)` returns True for any order that is
    not in FAILED state.  This blocks re-submission of the same order
    even if the previous run crashed after making the API call.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


# ──────────────────────────────────────────────────────────────────────────────
# State enum
# ──────────────────────────────────────────────────────────────────────────────

class InflightState(str, Enum):
    PENDING_SUBMIT    = "PENDING_SUBMIT"
    SUBMITTED_UNKNOWN = "SUBMITTED_UNKNOWN"
    ACKED             = "ACKED"
    RECONCILED        = "RECONCILED"
    FAILED            = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in (InflightState.RECONCILED, InflightState.FAILED)

    @property
    def needs_recovery(self) -> bool:
        return self in (InflightState.PENDING_SUBMIT, InflightState.SUBMITTED_UNKNOWN)


# ──────────────────────────────────────────────────────────────────────────────
# InflightOrder (in-memory)
# ──────────────────────────────────────────────────────────────────────────────

class InflightOrder:
    """In-memory representation of a single inflight order."""

    __slots__ = (
        "client_order_id", "state", "symbol", "side", "qty", "strategy",
        "trading_day", "run_id", "registered_at", "submitted_at",
        "broker_order_id", "reconciled_at", "error",
    )

    def __init__(self, data: Dict[str, Any]) -> None:
        self.client_order_id = data["client_order_id"]
        self.state           = InflightState(data.get("state", InflightState.PENDING_SUBMIT))
        self.symbol          = data.get("symbol", "")
        self.side            = data.get("side", "")
        self.qty             = int(data.get("qty", 0))
        self.strategy        = data.get("strategy", "")
        self.trading_day     = data.get("trading_day", "")
        self.run_id          = data.get("run_id", "")
        self.registered_at   = data.get("registered_at", "")
        self.submitted_at    = data.get("submitted_at")
        self.broker_order_id = data.get("broker_order_id")
        self.reconciled_at   = data.get("reconciled_at")
        self.error           = data.get("error")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "state":           self.state.value,
            "symbol":          self.symbol,
            "side":            self.side,
            "qty":             self.qty,
            "strategy":        self.strategy,
            "trading_day":     self.trading_day,
            "run_id":          self.run_id,
            "registered_at":   self.registered_at,
            "submitted_at":    self.submitted_at,
            "broker_order_id": self.broker_order_id,
            "reconciled_at":   self.reconciled_at,
            "error":           self.error,
        }

    def __repr__(self) -> str:
        return (
            f"InflightOrder({self.client_order_id!r} state={self.state.value}"
            f" {self.side} {self.symbol} qty={self.qty})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# InflightRegistry
# ──────────────────────────────────────────────────────────────────────────────

class InflightRegistry:
    """
    Crash-safe inflight order registry.

    Thread-safe: a single threading.Lock protects the in-memory dict and
    all append operations to the JSONL log.

    Usage::

        registry = InflightRegistry(RUNTIME_DIR / "inflight_orders.jsonl")
        # On startup:
        registry.load()
        unresolved = registry.get_unresolved()  # PENDING_SUBMIT + SUBMITTED_UNKNOWN

        # Before placing an order:
        if registry.is_duplicate(client_order_id):
            logger.warning("duplicate — skipping")
        else:
            registry.register(client_order_id, symbol=..., side=..., ...)
            registry.mark_submitting(client_order_id)
            try:
                result = broker.send(...)
                registry.mark_acked(client_order_id, broker_order_id=result.order_id)
            except Exception as exc:
                registry.mark_failed(client_order_id, error=str(exc))
    """

    def __init__(self, log_path: Path) -> None:
        self._path:   Path = Path(log_path)
        self._orders: Dict[str, InflightOrder] = {}
        self._lock   = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> int:
        """
        Reconstruct in-memory state from JSONL log.
        Returns number of orders loaded.
        """
        with self._lock:
            self._orders.clear()
            if not self._path.exists():
                return 0

            loaded = 0
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            coi   = event.get("client_order_id")
                            if not coi:
                                continue
                            # Latest event wins (append-only log replay)
                            if coi not in self._orders:
                                self._orders[coi] = InflightOrder(event)
                                loaded += 1
                            else:
                                # Update state from latest event for this coi
                                existing = self._orders[coi]
                                existing.state = InflightState(
                                    event.get("state", existing.state.value)
                                )
                                for field in (
                                    "submitted_at", "broker_order_id",
                                    "reconciled_at", "error",
                                ):
                                    v = event.get(field)
                                    if v is not None:
                                        setattr(existing, field, v)
                        except (json.JSONDecodeError, KeyError, ValueError) as exc:
                            logger.warning("[INFLIGHT] log line parse error: %s", exc)
            except OSError as exc:
                logger.error("[INFLIGHT] failed to load log: %s", exc)

            logger.info(
                "[INFLIGHT] loaded %d orders from %s", loaded, self._path
            )
            return loaded

    # ── write API ──────────────────────────────────────────────────────────────

    def register(
        self,
        client_order_id: str,
        symbol:          str,
        side:            str,
        qty:             int,
        strategy:        str,
        trading_day:     str,
        run_id:          str = "",
    ) -> InflightOrder:
        """
        Register a new order in PENDING_SUBMIT state.
        Raises ValueError if client_order_id already exists and is not FAILED.
        """
        with self._lock:
            if client_order_id in self._orders:
                existing = self._orders[client_order_id]
                if not existing.state.is_terminal:
                    raise ValueError(
                        f"Duplicate client_order_id: {client_order_id} "
                        f"(state={existing.state.value})"
                    )
                # FAILED → allow re-register (re-entry after explicit failure)
                logger.info(
                    "[INFLIGHT] re-registering FAILED order: %s", client_order_id
                )

            now = self._now()
            data: Dict[str, Any] = {
                "client_order_id": client_order_id,
                "state":           InflightState.PENDING_SUBMIT.value,
                "symbol":          symbol,
                "side":            side,
                "qty":             qty,
                "strategy":        strategy,
                "trading_day":     trading_day,
                "run_id":          run_id,
                "registered_at":   now,
            }
            order = InflightOrder(data)
            self._orders[client_order_id] = order
            self._append({"event": "register", **data})
            logger.info("[INFLIGHT] registered: %r", order)
            return order

    def mark_submitting(self, client_order_id: str) -> None:
        """
        Transition PENDING_SUBMIT → SUBMITTED_UNKNOWN.
        Must be called immediately BEFORE the broker API call so that a
        crash during submission leaves the order in SUBMITTED_UNKNOWN
        (unknown outcome) rather than PENDING_SUBMIT (not yet sent).
        """
        with self._lock:
            order = self._require(client_order_id)
            order.state        = InflightState.SUBMITTED_UNKNOWN
            order.submitted_at = self._now()
            self._append({
                "event":           "submit_attempt",
                "client_order_id": client_order_id,
                "state":           InflightState.SUBMITTED_UNKNOWN.value,
                "submitted_at":    order.submitted_at,
            })
            logger.info("[INFLIGHT] submitting: %s", client_order_id)

    def mark_acked(self, client_order_id: str, broker_order_id: str) -> None:
        """Transition SUBMITTED_UNKNOWN → ACKED (broker confirmed receipt)."""
        with self._lock:
            order = self._require(client_order_id)
            order.state           = InflightState.ACKED
            order.broker_order_id = broker_order_id
            self._append({
                "event":           "acked",
                "client_order_id": client_order_id,
                "state":           InflightState.ACKED.value,
                "broker_order_id": broker_order_id,
            })
            logger.info(
                "[INFLIGHT] acked: %s broker_order_id=%s",
                client_order_id, broker_order_id,
            )

    def mark_reconciled(self, client_order_id: str) -> None:
        """Transition ACKED → RECONCILED (confirmed against live broker state)."""
        with self._lock:
            order = self._require(client_order_id)
            order.state         = InflightState.RECONCILED
            order.reconciled_at = self._now()
            self._append({
                "event":           "reconciled",
                "client_order_id": client_order_id,
                "state":           InflightState.RECONCILED.value,
                "reconciled_at":   order.reconciled_at,
            })
            logger.info("[INFLIGHT] reconciled: %s", client_order_id)

    def mark_failed(self, client_order_id: str, error: str = "") -> None:
        """Transition any state → FAILED (terminal)."""
        with self._lock:
            if client_order_id in self._orders:
                order       = self._orders[client_order_id]
                order.state = InflightState.FAILED
                order.error = error[:500]
            else:
                # Order was never registered — create a minimal record
                order = InflightOrder({
                    "client_order_id": client_order_id,
                    "state":           InflightState.FAILED.value,
                    "error":           error[:500],
                })
                self._orders[client_order_id] = order
            self._append({
                "event":           "failed",
                "client_order_id": client_order_id,
                "state":           InflightState.FAILED.value,
                "error":           error[:500],
            })
            logger.warning("[INFLIGHT] failed: %s — %s", client_order_id, error[:100])

    # ── read API ───────────────────────────────────────────────────────────────

    def is_duplicate(self, client_order_id: str) -> bool:
        """
        True if the order exists and is NOT in FAILED state.
        Must be called before register() to enforce idempotency.
        """
        with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                return False
            return not order.state.is_terminal or order.state == InflightState.RECONCILED

    def get(self, client_order_id: str) -> Optional[InflightOrder]:
        with self._lock:
            return self._orders.get(client_order_id)

    def get_unresolved(self) -> List[InflightOrder]:
        """Return all orders in PENDING_SUBMIT or SUBMITTED_UNKNOWN state."""
        with self._lock:
            return [o for o in self._orders.values() if o.state.needs_recovery]

    def all_orders(self) -> List[InflightOrder]:
        with self._lock:
            return list(self._orders.values())

    def summary(self) -> Dict[str, int]:
        """State → count dict for logging."""
        with self._lock:
            counts: Dict[str, int] = {}
            for o in self._orders.values():
                counts[o.state.value] = counts.get(o.state.value, 0) + 1
            return counts

    # ── internals ──────────────────────────────────────────────────────────────

    def _require(self, client_order_id: str) -> InflightOrder:
        order = self._orders.get(client_order_id)
        if order is None:
            raise KeyError(
                f"[INFLIGHT] unknown client_order_id: {client_order_id}"
            )
        return order

    def _append(self, record: Dict[str, Any]) -> None:
        record.setdefault("ts", self._now())
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            logger.error("[INFLIGHT] log append failed: %s", exc)

    @staticmethod
    def _now() -> str:
        return datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
