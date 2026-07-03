"""
src/execution/reconcile.py
Broker ↔ intent journal reconciliation.

RULE (UNKNOWN protocol):
  Timeout/network failure → NEVER mark failed directly.
  Set UNKNOWN. Before retry: mandatory reconcile_open_orders().
  If broker order exists → recover state. Do NOT resubmit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from src.execution.intent import ExecutionIntent, IntentStatus, apply_transition
from src.execution.journal import IntentJournal

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


@dataclass
class ReconcileResult:
    run_at:               str
    broker_open_count:    int
    journal_active_count: int
    unknown_count:        int
    orphan_intents:       list[str] = field(default_factory=list)
    recovered_intents:    list[str] = field(default_factory=list)
    phantom_orders:       list[str] = field(default_factory=list)
    errors:               list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.orphan_intents or self.phantom_orders or self.errors)

    def log(self) -> None:
        logger.info(
            "[RECONCILE] broker_open=%d journal_active=%d unknown=%d "
            "orphans=%d recovered=%d phantoms=%d errors=%d",
            self.broker_open_count, self.journal_active_count,
            self.unknown_count, len(self.orphan_intents),
            len(self.recovered_intents), len(self.phantom_orders),
            len(self.errors),
        )
        for oid in self.orphan_intents:
            logger.warning("[RECONCILE] orphan intent: %s", oid)
        for oid in self.phantom_orders:
            logger.warning("[RECONCILE] phantom broker order: %s", oid)


def reconcile_open_orders(
    journal:         IntentJournal,
    client:          Any | None,
    portfolio_state: dict | None = None,
) -> ReconcileResult:
    """
    Reconcile broker open orders with intent journal.

    Protocol:
    1. Load UNKNOWN + SUBMITTED intents from journal
    2. Query broker open orders (skip gracefully if client=None)
    3. Match by broker_order_id
    4. UNKNOWN + found at broker  → ACKED  (do NOT resubmit)
    5. UNKNOWN + not found        → CANCELLED (orphan)
    6. Detect phantom broker orders (at broker but not in journal)
    7. Log [RECONCILE] structured line
    """
    now    = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    result = ReconcileResult(
        run_at               = now,
        broker_open_count    = 0,
        journal_active_count = 0,
        unknown_count        = 0,
    )

    # 1. Journal state
    latest        = {}
    for r in journal.load_all():
        latest[r.get("intent_id", "")] = r

    active_records  = [r for r in latest.values() if r.get("status") in
                       ("CREATED", "SUBMITTED", "ACKED", "PARTIAL", "UNKNOWN")]
    unknown_records = [r for r in latest.values() if r.get("status") == "UNKNOWN"]

    result.journal_active_count = len(active_records)
    result.unknown_count        = len(unknown_records)

    # 2. Broker open orders
    broker_orders: dict[str, dict] = {}
    if client is not None:
        try:
            raw = client.get_orders(only_open=True) or []
            for order in raw:
                oid = str(order.get("ID") or order.get("OrderId") or "")
                if oid:
                    broker_orders[oid] = order
            result.broker_open_count = len(broker_orders)
            logger.info("[RECONCILE] broker open orders fetched: %d", len(broker_orders))
        except Exception as e:
            err = f"broker get_orders failed: {e}"
            result.errors.append(err)
            logger.warning("[RECONCILE] %s — continuing with journal-only reconcile", err)

    # 3. Reconcile UNKNOWN intents
    for r in unknown_records:
        broker_oid = r.get("broker_order_id", "")
        intent_id  = r.get("intent_id", "")
        symbol     = r.get("symbol", "?")
        side       = r.get("side", "?")

        if not broker_oid:
            # Never got a broker order ID → was never submitted → CANCELLED
            result.orphan_intents.append(intent_id)
            _journal_transition(
                journal, r, IntentStatus.CANCELLED,
                reason="reconcile: no broker_order_id — never submitted",
            )
        elif broker_oid in broker_orders:
            # Found at broker — recover based on fill state
            broker_ord = broker_orders[broker_oid]
            filled_qty = int(broker_ord.get("CumQty", 0) or 0)
            orig_qty   = int(r.get("qty", 0))
            result.recovered_intents.append(intent_id)

            if filled_qty >= orig_qty:
                logger.info(
                    "[RECONCILE] recovered UNKNOWN %s (%s %s) broker_id=%s filled=%d → FILLED",
                    intent_id, side, symbol, broker_oid, filled_qty,
                )
                _journal_transition(
                    journal, r, IntentStatus.FILLED,
                    reason="reconcile: fully filled at broker",
                    filled_qty=filled_qty,
                )
            elif filled_qty > 0:
                logger.info(
                    "[RECONCILE] recovered UNKNOWN %s (%s %s) broker_id=%s"
                    " filled=%d/%d → PARTIAL",
                    intent_id, side, symbol, broker_oid, filled_qty, orig_qty,
                )
                _journal_transition(
                    journal, r, IntentStatus.PARTIAL,
                    reason="reconcile: partial fill at broker",
                    filled_qty=filled_qty,
                )
            else:
                logger.info(
                    "[RECONCILE] recovered UNKNOWN %s (%s %s) broker_id=%s filled=0 → ACKED",
                    intent_id, side, symbol, broker_oid,
                )
                _journal_transition(
                    journal, r, IntentStatus.ACKED,
                    reason="reconcile: found at broker (0 fill)",
                )
        elif client is not None:
            # Broker reachable + order not found → CANCELLED
            result.orphan_intents.append(intent_id)
            logger.warning(
                "[RECONCILE] UNKNOWN %s (%s %s) broker_id=%s not found → CANCELLED",
                intent_id, side, symbol, broker_oid,
            )
            _journal_transition(
                journal, r, IntentStatus.CANCELLED,
                reason="reconcile: broker_order_id not in open orders",
            )
        # else: client=None → can't confirm, leave as UNKNOWN

    # 4. Phantom broker orders
    matched_oids = {r.get("broker_order_id") for r in latest.values() if r.get("broker_order_id")}
    for oid in broker_orders:
        if oid not in matched_oids:
            result.phantom_orders.append(oid)
            logger.warning(
                "[RECONCILE] phantom broker order %s sym=%s — not in journal",
                oid, broker_orders[oid].get("Symbol", "?"),
            )

    result.log()
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _journal_transition(
    journal:    IntentJournal,
    record:     dict,
    new_status: IntentStatus,
    **extra,
) -> None:
    """Apply and journal a status transition for an intent from a journal record."""
    intent = ExecutionIntent.from_dict(record)
    allowed_updates = {k: v for k, v in extra.items()
                       if k in ("filled_qty", "filled_price", "broker_order_id", "retry_count")}
    try:
        apply_transition(intent, new_status, **allowed_updates)
    except ValueError as e:
        logger.error("[RECONCILE] transition rejected: %s", e)
        return
    reason = extra.get("reason", "reconcile")
    journal.append(intent, f"reconcile_{new_status.value.lower()}", reason=reason)
