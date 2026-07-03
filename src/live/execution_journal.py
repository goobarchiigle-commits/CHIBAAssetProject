"""
src/live/execution_journal.py

Filesystem-based execution journal for live orders.
Provides crash recovery and duplicate-order prevention via atomic
rename transitions between state directories.

State directories (created on demand):
  runtime/orders/pending/    — intent recorded before broker submission
  runtime/orders/submitted/  — handed to broker API
  runtime/orders/ack/        — broker accepted (order_id received)
  runtime/orders/filled/     — confirmed filled
  runtime/orders/failed/     — terminal failure

Each order is stored as a JSON file named by intent_id:
  {run_id}_{symbol}_{side}.json

Transition = atomic rename(src_dir/file, dst_dir/file).
On Windows NTFS, Path.replace() is atomic for files on the same volume.
A temp file is written to the destination directory first to ensure the
rename is intra-volume.

Crash recovery: on restart, scan pending/ and submitted/ for orders that
survived a crash without reaching ack/.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

ORDER_STATES = ("pending", "submitted", "ack", "filled", "failed")


class ExecutionJournal:
    """
    File-system order journal with atomic rename transitions.

    Designed to survive process crashes between any two operations.
    Every write uses write-to-temp + replace so no partial state is
    ever visible in the target directory.

    Typical flow::

        journal = ExecutionJournal(RUNTIME_DIR / "orders")
        intent_id = journal.record_pending(run_id, symbol, "BUY", qty, price)
        journal.submit(intent_id)
        journal.ack(intent_id, order_id=broker_order_id)
        journal.fill(intent_id, fill_price=actual_price)

    On failure::

        journal.fail(intent_id, from_state="submitted", error=str(exc))
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        for state in ORDER_STATES:
            (self._base / state).mkdir(parents=True, exist_ok=True)

    # ── public write API ─────────────────────────────────────────────────────

    def record_pending(
        self,
        run_id:   str,
        symbol:   str,
        side:     str,
        qty:      int,
        price:    float,
        strategy: str = "",
    ) -> str:
        """
        Create a new order record in pending/.
        Returns intent_id (used as the filename stem for all transitions).
        Idempotent: if the file already exists, returns the same intent_id.
        """
        intent_id = self._make_intent_id(run_id, symbol, side)
        dest = self._base / "pending" / f"{intent_id}.json"
        if dest.exists():
            logger.info("[JOURNAL] pending already exists (idempotent): %s", intent_id)
            return intent_id

        data: Dict[str, Any] = {
            "intent_id":  intent_id,
            "run_id":     run_id,
            "symbol":     symbol,
            "side":       side,
            "qty":        qty,
            "price":      price,
            "strategy":   strategy,
            "state":      "pending",
            "created_at": self._now(),
        }
        self._write_atomic("pending", intent_id, data)
        logger.info(
            "[JOURNAL] pending: %s %s %s qty=%d price=%.2f",
            intent_id, side, symbol, qty, price,
        )
        return intent_id

    def submit(self, intent_id: str) -> bool:
        """Transition pending → submitted (order handed to broker API)."""
        return self._transition(intent_id, "pending", "submitted")

    def ack(self, intent_id: str, order_id: str = "") -> bool:
        """Transition submitted → ack (broker accepted, got order_id)."""
        return self._transition(
            intent_id, "submitted", "ack",
            extra={"broker_order_id": order_id},
        )

    def fill(self, intent_id: str, fill_price: float = 0.0) -> bool:
        """Transition ack → filled."""
        return self._transition(
            intent_id, "ack", "filled",
            extra={"fill_price": fill_price},
        )

    def fail(self, intent_id: str, from_state: str, error: str = "") -> bool:
        """Transition any live state → failed (terminal)."""
        if from_state not in ORDER_STATES:
            logger.warning("[JOURNAL] fail: unknown from_state=%s", from_state)
            return False
        return self._transition(
            intent_id, from_state, "failed",
            extra={"error": error[:500]},
        )

    # ── public read API ──────────────────────────────────────────────────────

    def list_state(self, state: str) -> List[Dict[str, Any]]:
        """Return all order records in the given state directory."""
        results: List[Dict[str, Any]] = []
        for f in (self._base / state).glob("*.json"):
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("[JOURNAL] read failed %s: %s", f.name, exc)
        return results

    def get(self, intent_id: str) -> Optional[Dict[str, Any]]:
        """Find order record across all state directories. Returns None if not found."""
        for state in ORDER_STATES:
            path = self._base / state / f"{intent_id}.json"
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return None
        return None

    def has_active(self, run_id: str, symbol: str, side: str) -> bool:
        """
        True if a non-failed order for this (symbol, side) exists in this run.
        Used for duplicate-order prevention before submitting.
        """
        intent_id = self._make_intent_id(run_id, symbol, side)
        for state in ("pending", "submitted", "ack", "filled"):
            if (self._base / state / f"{intent_id}.json").exists():
                return True
        return False

    def recover_incomplete(self) -> List[Dict[str, Any]]:
        """
        Return orders in pending/ and submitted/ that survived a crash.
        Caller should reconcile these against the broker before placing
        new orders to avoid duplicates.
        """
        incomplete: List[Dict[str, Any]] = []
        for state in ("pending", "submitted"):
            incomplete.extend(self.list_state(state))
        return incomplete

    def make_intent_id(self, run_id: str, symbol: str, side: str) -> str:
        """Public access to intent_id construction (for callers that need the key)."""
        return self._make_intent_id(run_id, symbol, side)

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_intent_id(run_id: str, symbol: str, side: str) -> str:
        safe = run_id.replace(":", "").replace("/", "_")
        return f"{safe}_{symbol}_{side}"

    @staticmethod
    def _now() -> str:
        return datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")

    def _transition(
        self,
        intent_id:  str,
        from_state: str,
        to_state:   str,
        extra:      Optional[Dict[str, Any]] = None,
    ) -> bool:
        src     = self._base / from_state / f"{intent_id}.json"
        dst_dir = self._base / to_state
        dst     = dst_dir / f"{intent_id}.json"

        if not src.exists():
            logger.warning(
                "[JOURNAL] transition src not found: %s/%s",
                from_state, intent_id,
            )
            return False

        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("[JOURNAL] read failed %s: %s", intent_id, exc)
            return False

        data["state"]              = to_state
        data[f"{to_state}_at"]    = self._now()
        data["prev_state"]        = from_state
        if extra:
            data.update(extra)

        tmp = dst_dir / f"{intent_id}.tmp"
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(dst)           # atomic on NTFS (same volume)
            src.unlink(missing_ok=True)
            logger.info(
                "[JOURNAL] %s → %s: %s",
                from_state, to_state, intent_id,
            )
            return True
        except Exception as exc:
            logger.error(
                "[JOURNAL] transition %s → %s failed: %s",
                from_state, to_state, exc,
            )
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def _write_atomic(self, state: str, intent_id: str, data: Dict[str, Any]) -> None:
        dst_dir = self._base / state
        dst     = dst_dir / f"{intent_id}.json"
        tmp     = dst_dir / f"{intent_id}.tmp"
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(dst)
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise exc
