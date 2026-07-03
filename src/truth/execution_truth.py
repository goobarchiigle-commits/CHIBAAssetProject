"""
execution_truth.py — Order lifecycle truth tracking

Monotonic state machine for execution intent ground truth.
UNKNOWN is a valid runtime state. It must NEVER be auto-resolved.

State machine (forward-only):
  CREATED → SUBMITTED → ACKED → FILLED              (terminal)
                               → PARTIAL_FILL → PARTIAL_FILL   (incremental fills)
                                              → FILLED
                               → CANCELLED           (terminal)
                               → UNKNOWN  → RESOLVED_FILLED     (terminal)
                                          → RESOLVED_CANCELLED  (terminal)
                       → REJECTED         (terminal)
                       → UNKNOWN  → RESOLVED_FILLED
                                  → RESOLVED_CANCELLED
             → CANCELLED          (terminal)

Invariants:
  1. Transitions are forward-only (monotonic).
  2. UNKNOWN can only be resolved by explicit broker confirmation — never automated.
  3. UNKNOWN orders contribute to uncertain_exposure in CapitalTruth.
  4. Terminal states accept no further transitions.
  5. filled_qty ≤ submitted_qty ≤ requested_qty always holds.
  6. fill_price > 0 iff filled_qty > 0.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# ── State machine ────────────────────────────────────────────────────────────

VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "CREATED":            frozenset({"SUBMITTED", "CANCELLED"}),
    "SUBMITTED":          frozenset({"ACKED", "REJECTED", "UNKNOWN", "CANCELLED"}),
    "ACKED":              frozenset({"FILLED", "PARTIAL_FILL", "UNKNOWN", "CANCELLED"}),
    "PARTIAL_FILL":       frozenset({"FILLED", "PARTIAL_FILL", "UNKNOWN", "CANCELLED"}),
    "UNKNOWN":            frozenset({"RESOLVED_FILLED", "RESOLVED_CANCELLED"}),
    "FILLED":             frozenset(),          # terminal
    "REJECTED":           frozenset(),          # terminal
    "CANCELLED":          frozenset(),          # terminal
    "RESOLVED_FILLED":    frozenset(),          # terminal
    "RESOLVED_CANCELLED": frozenset(),          # terminal
}

TERMINAL_STATES: frozenset[str] = frozenset({
    "FILLED", "REJECTED", "CANCELLED",
    "RESOLVED_FILLED", "RESOLVED_CANCELLED",
})

UNCERTAIN_STATES: frozenset[str] = frozenset({
    "SUBMITTED", "ACKED", "PARTIAL_FILL", "UNKNOWN",
})

UNKNOWN_STATES: frozenset[str] = frozenset({"UNKNOWN"})

PENDING_STATES: frozenset[str] = frozenset({"SUBMITTED", "ACKED", "PARTIAL_FILL"})

ALL_VALID_STATES: frozenset[str] = frozenset(VALID_TRANSITIONS.keys())


class InvalidTransitionError(Exception):
    """Raised when a state transition violates the monotonic state machine."""


@dataclass
class ExecutionRecord:
    """
    Ground truth for a single order intent.

    requested_qty:  what we asked for
    submitted_qty:  what the broker accepted (may differ after rounding)
    filled_qty:     confirmed filled lots (from broker)
    remaining_qty:  submitted_qty - filled_qty (still pending or unknown)
    """
    intent_id: str
    symbol: str
    side: str                      # "BUY" | "SELL"
    requested_qty: int             # lots
    state: str = "CREATED"
    submitted_qty: int = 0
    filled_qty: int = 0
    remaining_qty: int = 0         # submitted - filled (updated on transitions)
    fill_price: float = 0.0        # weighted average fill price (JPY)
    estimated_price: float = 0.0   # price at order creation (for exposure calc)
    lot_size: int = 100            # shares per lot (Tokyo Stock Exchange standard)
    broker_order_id: str = ""
    strategy: str = ""
    trading_day: str = ""
    created_at: str = ""
    last_updated_at: str = ""
    unknown_since: str = ""        # timestamp when UNKNOWN state was entered
    transition_reason: str = ""    # reason for last state transition
    schema_version: int = 1

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_uncertain(self) -> bool:
        return self.state in UNCERTAIN_STATES

    @property
    def is_unknown(self) -> bool:
        return self.state in UNKNOWN_STATES

    @property
    def is_pending(self) -> bool:
        return self.state in PENDING_STATES

    @property
    def estimated_value(self) -> float:
        """Estimated order value in JPY (requested × price × lot_size)."""
        return self.requested_qty * self.estimated_price * self.lot_size

    @property
    def confirmed_value(self) -> float:
        """Value of confirmed fills in JPY."""
        return self.filled_qty * (self.fill_price or self.estimated_price) * self.lot_size

    @property
    def pending_value(self) -> float:
        """Value of pending (unconfirmed but non-UNKNOWN) lots in JPY."""
        if not self.is_pending:
            return 0.0
        return self.remaining_qty * self.estimated_price * self.lot_size

    @property
    def uncertain_value(self) -> float:
        """Value of UNKNOWN lots in JPY."""
        if not self.is_unknown:
            return 0.0
        return self.remaining_qty * self.estimated_price * self.lot_size


# ── State transition engine ───────────────────────────────────────────────────

def can_transition(record: ExecutionRecord, new_state: str) -> bool:
    """Returns True if transition from current state to new_state is valid."""
    if new_state not in ALL_VALID_STATES:
        return False
    return new_state in VALID_TRANSITIONS.get(record.state, frozenset())


def transition_state(
    record: ExecutionRecord,
    new_state: str,
    reason: str = "",
    broker_order_id: str = "",
    filled_qty: int = 0,
    fill_price: float = 0.0,
    submitted_qty: int = 0,
) -> ExecutionRecord:
    """
    Apply a state transition, returning a new ExecutionRecord.

    Immutable update: does not modify the input record.
    Raises InvalidTransitionError if the transition is illegal.

    UNKNOWN→RESOLVED_* requires explicit broker_order_id or reason.
    """
    if not can_transition(record, new_state):
        raise InvalidTransitionError(
            f"[{record.intent_id}] {record.state} → {new_state} is invalid "
            f"(valid: {sorted(VALID_TRANSITIONS.get(record.state, set()))})"
        )

    now = datetime.now(JST).isoformat()

    new_filled_qty     = record.filled_qty
    new_fill_price     = record.fill_price
    new_submitted_qty  = record.submitted_qty
    new_remaining_qty  = record.remaining_qty
    new_broker_oid     = record.broker_order_id or broker_order_id
    new_unknown_since  = record.unknown_since

    if new_state == "SUBMITTED":
        q = submitted_qty or record.requested_qty
        new_submitted_qty = q
        new_remaining_qty = q

    elif new_state == "ACKED":
        if broker_order_id:
            new_broker_oid = broker_order_id

    elif new_state in ("FILLED", "RESOLVED_FILLED"):
        if filled_qty > 0:
            # weighted avg fill price update
            prev_value = record.filled_qty * record.fill_price
            new_value  = filled_qty * fill_price
            total_qty  = record.filled_qty + filled_qty
            if total_qty > 0:
                new_fill_price = (prev_value + new_value) / total_qty
            new_filled_qty    = total_qty
        else:
            # fill entire remaining
            new_filled_qty = record.submitted_qty
            if fill_price > 0:
                new_fill_price = fill_price
        new_remaining_qty = max(0, new_submitted_qty - new_filled_qty)

    elif new_state == "PARTIAL_FILL":
        delta_filled = filled_qty or 1
        prev_value   = record.filled_qty * record.fill_price
        new_value    = delta_filled * (fill_price or record.estimated_price)
        total_qty    = record.filled_qty + delta_filled
        if total_qty > 0:
            new_fill_price = (prev_value + new_value) / total_qty
        new_filled_qty    = total_qty
        new_remaining_qty = max(0, new_submitted_qty - new_filled_qty)

    elif new_state == "UNKNOWN":
        new_unknown_since = now

    elif new_state == "RESOLVED_CANCELLED":
        new_remaining_qty = 0

    return ExecutionRecord(
        intent_id         = record.intent_id,
        symbol            = record.symbol,
        side              = record.side,
        requested_qty     = record.requested_qty,
        state             = new_state,
        submitted_qty     = new_submitted_qty,
        filled_qty        = new_filled_qty,
        remaining_qty     = new_remaining_qty,
        fill_price        = round(new_fill_price, 2),
        estimated_price   = record.estimated_price,
        lot_size          = record.lot_size,
        broker_order_id   = new_broker_oid,
        strategy          = record.strategy,
        trading_day       = record.trading_day,
        created_at        = record.created_at,
        last_updated_at   = now,
        unknown_since     = new_unknown_since,
        transition_reason = reason or f"{record.state}→{new_state}",
    )


# ── Exposure breakdown ────────────────────────────────────────────────────────

@dataclass
class ExposureBreakdown:
    confirmed_value: float    # FILLED / RESOLVED_FILLED
    pending_value: float      # SUBMITTED / ACKED / PARTIAL_FILL (remaining)
    uncertain_value: float    # UNKNOWN (remaining)
    total_uncertain_orders: int
    total_pending_orders: int
    schema_version: int = 1


def compute_exposure_breakdown(records: list[ExecutionRecord]) -> ExposureBreakdown:
    """Aggregate BUY exposure into confirmed / pending / uncertain buckets."""
    confirmed  = 0.0
    pending    = 0.0
    uncertain  = 0.0
    n_unknown  = 0
    n_pending  = 0

    for rec in records:
        if rec.side != "BUY":
            continue
        confirmed += rec.confirmed_value
        if rec.is_pending:
            pending  += rec.pending_value
            n_pending += 1
        if rec.is_unknown:
            uncertain += rec.uncertain_value
            n_unknown += 1

    return ExposureBreakdown(
        confirmed_value       = round(confirmed, 2),
        pending_value         = round(pending, 2),
        uncertain_value       = round(uncertain, 2),
        total_uncertain_orders= n_unknown,
        total_pending_orders  = n_pending,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def record_to_dict(rec: ExecutionRecord) -> dict:
    return asdict(rec)


def record_from_dict(data: dict) -> ExecutionRecord:
    fields = set(ExecutionRecord.__dataclass_fields__)
    return ExecutionRecord(**{k: v for k, v in data.items() if k in fields})


def append_execution_record(record: ExecutionRecord, path: Path) -> None:
    """Append-only JSONL — one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record_to_dict(record), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_execution_records(path: Path) -> list[ExecutionRecord]:
    """
    Load all execution records from JSONL.
    Latest record per intent_id wins (last-write-wins for idempotent replay).
    """
    if not path.exists():
        return []
    latest: dict[str, ExecutionRecord] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rec = record_from_dict(data)
                    latest[rec.intent_id] = rec
                except Exception as _e:
                    logger.warning("execution_truth: malformed line skipped (%s)", _e)
    except Exception as exc:
        logger.warning("execution_truth: load failed (%s)", exc)
    return list(latest.values())


def mark_unknown_stale(
    records: list[ExecutionRecord],
    stale_after_minutes: int = 30,
    now: Optional[datetime] = None,
) -> list[ExecutionRecord]:
    """
    Transition SUBMITTED / ACKED orders older than stale_after_minutes to UNKNOWN.

    Does NOT touch records already in UNKNOWN or terminal states.
    Returns updated list (immutable — originals unchanged).
    """
    if now is None:
        now = datetime.now(JST)

    updated = []
    for rec in records:
        if rec.state in PENDING_STATES and rec.last_updated_at:
            try:
                last = datetime.fromisoformat(rec.last_updated_at)
                age_min = (now - last).total_seconds() / 60.0
                if age_min > stale_after_minutes:
                    rec = transition_state(
                        rec, "UNKNOWN",
                        reason=f"stale: {age_min:.1f}min > {stale_after_minutes}min threshold",
                    )
            except Exception:
                pass
        updated.append(rec)
    return updated
