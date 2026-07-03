"""
src/live/client_order_id.py

Deterministic client_order_id generation.

ID is a stable hash of (strategy, symbol, side, qty, trading_day).
Same inputs always produce the same ID across restarts, allowing the
InflightRegistry to detect and suppress duplicate broker submissions.

Format: COI-<16-char uppercase hex>
Example: COI-A1B2C3D4E5F60718

Design constraints:
  - No timestamp component (stability across restarts)
  - No random component (deterministic)
  - Human-readable prefix for audit logs
  - 16-hex suffix → 64-bit collision space (negligible risk for 1-3 orders/day)
"""
from __future__ import annotations

import hashlib


def make_client_order_id(
    strategy:    str,
    symbol:      str,
    side:        str,
    qty:         int,
    trading_day: str,   # ISO date: "2026-05-15"
) -> str:
    """
    Return a stable, content-addressed client_order_id.

    Replay/restart safety: the same order (same strategy, symbol, side,
    qty, day) always maps to the same ID.  InflightRegistry uses this to
    block duplicate broker submissions even after a crash.

    Args:
        strategy:    strategy identifier ("fujiko", "mean_rev", "unknown")
        symbol:      ticker including suffix if any ("7203.T")
        side:        "BUY" or "SELL"
        qty:         order quantity in shares
        trading_day: ISO date string (e.g. "2026-05-15")

    Returns:
        "COI-" + 16 uppercase hex chars
    """
    key = f"{trading_day}|{symbol}|{side.upper()}|{qty}|{strategy.lower()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return "COI-" + digest[:16].upper()


def parse_client_order_id(coi: str) -> dict:
    """Extract prefix and hex suffix for logging/debugging."""
    if not coi.startswith("COI-"):
        raise ValueError(f"Not a valid client_order_id: {coi!r}")
    return {"prefix": "COI", "hex": coi[4:]}
