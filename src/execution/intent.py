"""
src/execution/intent.py
ExecutionIntent dataclass and deterministic state machine.

RULE: Status transitions via apply_transition() are the ONLY mutation allowed
      on ExecutionIntent after creation. All other fields are write-once.

intent_id = sha256(symbol|side|qty|strategy|signal_ts|generation_id)[:16]
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


class IntentStatus(str, Enum):
    CREATED   = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACKED     = "ACKED"
    PARTIAL   = "PARTIAL"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    UNKNOWN   = "UNKNOWN"


TERMINAL_STATUSES: frozenset[IntentStatus] = frozenset({
    IntentStatus.FILLED,
    IntentStatus.CANCELLED,
    IntentStatus.REJECTED,
})

ACTIVE_STATUSES: frozenset[IntentStatus] = frozenset({
    IntentStatus.CREATED,
    IntentStatus.SUBMITTED,
    IntentStatus.ACKED,
    IntentStatus.PARTIAL,
    IntentStatus.UNKNOWN,
})

ACTIVE_STATUS_VALUES: frozenset[str] = frozenset(s.value for s in ACTIVE_STATUSES)

# Complete transition table — invalid transitions raise ValueError + log CRITICAL
VALID_TRANSITIONS: dict[IntentStatus, frozenset[IntentStatus]] = {
    IntentStatus.CREATED:   frozenset({IntentStatus.SUBMITTED, IntentStatus.CANCELLED}),
    IntentStatus.SUBMITTED: frozenset({IntentStatus.ACKED, IntentStatus.PARTIAL,
                                        IntentStatus.REJECTED, IntentStatus.UNKNOWN,
                                        IntentStatus.CANCELLED, IntentStatus.FILLED}),
    IntentStatus.ACKED:     frozenset({IntentStatus.FILLED, IntentStatus.PARTIAL,
                                        IntentStatus.CANCELLED}),
    IntentStatus.PARTIAL:   frozenset({IntentStatus.FILLED, IntentStatus.CANCELLED,
                                        IntentStatus.ACKED}),
    IntentStatus.UNKNOWN:   frozenset({IntentStatus.ACKED, IntentStatus.PARTIAL,
                                        IntentStatus.FILLED, IntentStatus.CANCELLED,
                                        IntentStatus.SUBMITTED}),
    IntentStatus.FILLED:    frozenset(),
    IntentStatus.CANCELLED: frozenset(),
    IntentStatus.REJECTED:  frozenset(),
}


@dataclass
class ExecutionIntent:
    """
    Immutable execution intent record.

    Fields are write-once at creation. Only status, retry_count, filled_qty,
    filled_price, broker_order_id, and updated_at may change via apply_transition().
    """
    intent_id:            str
    run_id:               str
    signal_ts:            str
    symbol:               str
    side:                 str          # "BUY" | "SELL" | "SHADOW_BUY"
    qty:                  int
    strategy:             str
    expected_price:       float
    alloc_before:         float
    alloc_after:          float
    snapshot_generation:  int
    status:               str          # IntentStatus.value
    retry_count:          int   = 0
    filled_qty:           int   = 0
    filled_price:         float = 0.0
    remaining_qty:        int   = 0    # qty - filled_qty; updated on PARTIAL
    broker_order_id:      str   = ""
    created_at:           str   = ""
    updated_at:           str   = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionIntent:
        return cls(
            intent_id           = d["intent_id"],
            run_id              = d.get("run_id", ""),
            signal_ts           = d.get("signal_ts", ""),
            symbol              = d["symbol"],
            side                = d["side"],
            qty                 = int(d["qty"]),
            strategy            = d.get("strategy", ""),
            expected_price      = float(d.get("expected_price", 0)),
            alloc_before        = float(d.get("alloc_before", 0)),
            alloc_after         = float(d.get("alloc_after", 0)),
            snapshot_generation = int(d.get("snapshot_generation", 0)),
            status              = d["status"],
            retry_count         = int(d.get("retry_count", 0)),
            filled_qty          = int(d.get("filled_qty", 0)),
            filled_price        = float(d.get("filled_price", 0)),
            remaining_qty       = int(d.get("remaining_qty", int(d["qty"]) - int(d.get("filled_qty", 0)))),
            broker_order_id     = d.get("broker_order_id", ""),
            created_at          = d.get("created_at", ""),
            updated_at          = d.get("updated_at", ""),
        )


def make_intent_id(
    symbol:        str,
    side:          str,
    qty:           int,
    strategy:      str,
    signal_ts:     str,
    generation_id: int,
) -> str:
    """
    Deterministic intent ID: sha256(payload)[:16].

    Same inputs always produce the same ID — enables duplicate detection
    across process restarts and retries.
    """
    payload = f"{symbol}|{side}|{qty}|{strategy}|{signal_ts}|{generation_id}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_intent(
    *,
    symbol:         str,
    side:           str,
    qty:            int,
    strategy:       str,
    expected_price: float,
    alloc_before:   float,
    alloc_after:    float,
    run_id:         str,
    signal_ts:      str,
    generation_id:  int,
) -> ExecutionIntent:
    """Create a new ExecutionIntent in CREATED state."""
    now       = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    intent_id = make_intent_id(symbol, side, qty, strategy, signal_ts, generation_id)
    return ExecutionIntent(
        intent_id           = intent_id,
        run_id              = run_id,
        signal_ts           = signal_ts,
        symbol              = symbol,
        side                = side,
        qty                 = qty,
        strategy            = strategy,
        expected_price      = expected_price,
        alloc_before        = alloc_before,
        alloc_after         = alloc_after,
        snapshot_generation = generation_id,
        status              = IntentStatus.CREATED.value,
        remaining_qty       = qty,
        created_at          = now,
        updated_at          = now,
    )


def apply_transition(
    intent:     ExecutionIntent,
    new_status: IntentStatus,
    **updates,
) -> ExecutionIntent:
    """
    Apply a validated state transition.

    Allowed keyword updates: filled_qty, filled_price, broker_order_id, retry_count.

    Raises:
        ValueError: invalid transition (also logs CRITICAL).
    """
    current = IntentStatus(intent.status)
    allowed = VALID_TRANSITIONS.get(current, frozenset())
    if new_status not in allowed:
        msg = (
            f"[INTENT] INVALID TRANSITION {current.value} → {new_status.value} "
            f"intent={intent.intent_id} symbol={intent.symbol}"
        )
        logger.critical(msg)
        raise ValueError(msg)

    intent.status     = new_status.value
    intent.updated_at = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    for k, v in updates.items():
        if hasattr(intent, k):
            setattr(intent, k, v)

    # Recompute remaining_qty on PARTIAL
    if new_status == IntentStatus.PARTIAL:
        intent.remaining_qty = max(0, intent.qty - intent.filled_qty)

    logger.info(
        "[INTENT] id=%s state=%s symbol=%s side=%s qty=%d retry=%d filled=%d",
        intent.intent_id, intent.status, intent.symbol, intent.side,
        intent.qty, intent.retry_count, intent.filled_qty,
    )
    return intent
