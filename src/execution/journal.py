"""
src/execution/journal.py
Append-only intent journal backed by runtime/order_intents.jsonl.

RULES:
  - Never overwrite history (append-only).
  - fsync on every write (crash-safe).
  - Load builds latest-status view by replaying all records.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.execution.intent import ExecutionIntent, ACTIVE_STATUS_VALUES, IntentStatus

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

_BASE_DIR            = Path(os.environ.get("AI_TRADING_HOME",
                             str(Path(__file__).resolve().parents[2])))
DEFAULT_JOURNAL_PATH = _BASE_DIR / "runtime" / "order_intents.jsonl"


class IntentJournal:
    """
    Append-only execution intent journal.

    Each append():
      1. Serialises the event as JSON
      2. Writes with trailing newline
      3. flush() + fsync() — guaranteed durable before returning

    Read methods replay all records and keep the latest status per intent_id.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_JOURNAL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Write ─────────────────────────────────────────────────────────

    def append(self, intent: ExecutionIntent, event: str, **extra) -> None:
        """Append a journal event. fsync guaranteed before returning."""
        record = {
            "ts":                  datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event":               event,
            "intent_id":           intent.intent_id,
            "run_id":              intent.run_id,
            "symbol":              intent.symbol,
            "side":                intent.side,
            "qty":                 intent.qty,
            "strategy":            intent.strategy,
            "expected_price":      intent.expected_price,
            "snapshot_generation": intent.snapshot_generation,
            "status":              intent.status,
            "retry_count":         intent.retry_count,
            "filled_qty":          intent.filled_qty,
            "remaining_qty":       intent.remaining_qty,
            "broker_order_id":     intent.broker_order_id,
        }
        record.update(extra)

        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

        logger.info(
            "[INTENT] id=%s state=%s event=%s symbol=%s qty=%d broker_id=%s",
            intent.intent_id, intent.status, event,
            intent.symbol, intent.qty, intent.broker_order_id or "-",
        )

    # ── Read ──────────────────────────────────────────────────────────

    def load_all(self) -> list[dict]:
        """Return all records in append order."""
        if not self.path.exists():
            return []
        records: list[dict] = []
        with self.path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("[JOURNAL] line %d corrupted, skipped: %s", i + 1, e)
        return records

    def _latest_per_intent(self) -> dict[str, dict]:
        """Return the most recent record for each intent_id."""
        latest: dict[str, dict] = {}
        for r in self.load_all():
            iid = r.get("intent_id", "")
            if iid:
                latest[iid] = r
        return latest

    def get_latest_status(self, intent_id: str) -> str | None:
        """Most recent status for intent_id, or None if not found."""
        result = None
        for r in self.load_all():
            if r.get("intent_id") == intent_id:
                result = r.get("status")
        return result

    # ── Corruption scanner ────────────────────────────────────────────

    def scan_for_corruption(self) -> list[int]:
        """
        Scan journal for non-parseable lines.

        Returns 1-indexed line numbers that failed JSON decode.
        Logs [JOURNAL_CORRUPT] for each bad line.
        """
        if not self.path.exists():
            return []
        bad: list[int] = []
        with self.path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as e:
                    bad.append(i)
                    logger.error(
                        "[JOURNAL_CORRUPT] line=%d path=%s error=%s",
                        i, self.path.name, e,
                    )
        if bad:
            logger.warning(
                "[JOURNAL_CORRUPT] %d corrupted line(s) in %s: %s",
                len(bad), self.path.name, bad,
            )
        return bad

    # ── Duplicate / overlap prevention ────────────────────────────────

    def can_submit_intent(
        self,
        intent_id:    str,
        symbol:       str,
        side:         str,
        run_id:       str,
        window_hours: float = 24.0,
    ) -> tuple[bool, str]:
        """
        Centralized submit gate. ALL submit paths must call this.

        Returns (allowed: bool, reason: str).
        Logs [INTENT_GATE] intent=... allowed=... reason=...

        Blocks if:
          1. intent_id already active (duplicate)
          2. same symbol+side already active within this run (overlap)
        """
        status = self.get_latest_status(intent_id)
        if status in ACTIVE_STATUS_VALUES:
            reason = f"duplicate intent_id={intent_id} status={status}"
            logger.warning(
                "[INTENT_GATE] intent=%s symbol=%s side=%s allowed=False reason=%s",
                intent_id, symbol, side, reason,
            )
            return False, reason

        active = self.find_active_by_symbol_side(
            symbol, side, run_id=run_id, window_hours=window_hours
        )
        if active:
            reason = (
                f"overlap symbol={symbol} side={side} "
                f"run={run_id} active_count={len(active)}"
            )
            logger.warning(
                "[INTENT_GATE] intent=%s symbol=%s side=%s allowed=False reason=%s",
                intent_id, symbol, side, reason,
            )
            return False, reason

        logger.info(
            "[INTENT_GATE] intent=%s symbol=%s side=%s allowed=True reason=ok",
            intent_id, symbol, side,
        )
        return True, "ok"

    def is_duplicate(self, intent_id: str) -> bool:
        """
        True if intent_id exists in the journal with an active status.

        Prefer can_submit_intent() for submit-path enforcement.
        """
        status = self.get_latest_status(intent_id)
        dup = status in ACTIVE_STATUS_VALUES
        if dup:
            logger.warning(
                "[DUPLICATE_BLOCK] intent_id=%s already active with status=%s",
                intent_id, status,
            )
        return dup

    def find_active_by_symbol_side(
        self,
        symbol:       str,
        side:         str,
        run_id:       str | None = None,
        window_hours: float      = 24.0,
    ) -> list[dict]:
        """
        Active intents for symbol+side within window_hours.

        Optionally filter to a single run_id for within-run overlap protection.
        """
        cutoff = (
            datetime.now(JST) - timedelta(hours=window_hours)
        ).strftime("%Y-%m-%dT%H:%M:%S%z")

        latest = self._latest_per_intent()
        return [
            r for r in latest.values()
            if r.get("symbol") == symbol
            and r.get("side") == side
            and r.get("status") in ACTIVE_STATUS_VALUES
            and r.get("ts", "") >= cutoff
            and (run_id is None or r.get("run_id") == run_id)
        ]

    def is_overlap_blocked(self, symbol: str, side: str, run_id: str) -> bool:
        """
        Within-run overlap protection: same symbol+side already active in this run.
        """
        active = self.find_active_by_symbol_side(symbol, side, run_id=run_id)
        if active:
            logger.warning(
                "[OVERLAP_BLOCK] run=%s symbol=%s side=%s already active (%d records)",
                run_id, symbol, side, len(active),
            )
            return True
        return False

    # ── Query helpers ─────────────────────────────────────────────────

    def get_unknown_intents(self) -> list[dict]:
        """All intents currently in UNKNOWN state."""
        return [
            r for r in self._latest_per_intent().values()
            if r.get("status") == IntentStatus.UNKNOWN.value
        ]

    def get_run_intents(self, run_id: str) -> list[dict]:
        """Latest-status records for a given run_id."""
        return [
            r for r in self._latest_per_intent().values()
            if r.get("run_id") == run_id
        ]

    def get_active_intents(self) -> list[dict]:
        """All intents with active status (across all runs)."""
        return [
            r for r in self._latest_per_intent().values()
            if r.get("status") in ACTIVE_STATUS_VALUES
        ]
