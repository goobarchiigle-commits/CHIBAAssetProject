"""
research_candidate/atr_extension.py — Study40 ATR Extension post-filter.
Defers RSR-triggered SELL orders when the position is profitable and price
remains within 1×ATR20 of the highest close. Max deferral: 7 calendar days.
FAIL_OPEN: returns original order_objects unchanged on any error.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

MAX_DEFER_CALENDAR_DAYS: int = 7   # ~5 business days
ATR_MULT: float              = 1.0  # defer if close > highest_close - mult×ATR20

# Reason substring that identifies RSR-triggered SELL orders in signal_bridge
_RSR_REASON_TAG = "SELL[多層RSR]"


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict, state_path: Path) -> None:
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("[ATR_EXT] state save failed: %s", e)


def _purge_exited(state: dict, held_syms: set) -> dict:
    return {k: v for k, v in state.items() if k in held_syms}


def filter_atr_extension_sells(
    order_objects: list,
    result_signals: list[dict],
    portfolio_state: dict,
    state_path: Path,
    today: str,
    atr_mult: float  = ATR_MULT,
    max_defer_days: int = MAX_DEFER_CALENDAR_DAYS,
) -> tuple[list, int]:
    """
    Post-filter SELL orders from bridge.run() for ATR Extension deferral.

    Rules:
      - Only RSR-triggered SELL orders (reason contains "SELL[多層RSR]")
      - Profitable position required (unrealized_pnl_pct > 0)
      - Deferral condition: close_today > highest_close - atr_mult × entry_ATR20
      - Max deferral: max_defer_days calendar days from first deferral

    Returns (filtered_order_objects, n_deferred).
    FAIL_OPEN: returns (order_objects, 0) on any error.
    """
    try:
        today_date = datetime.strptime(today, "%Y-%m-%d").date()
        held_syms  = {s.get("symbol", "") for s in result_signals if s.get("currently_holding")}
        state      = _load_state(state_path)
        state      = _purge_exited(state, held_syms)

        pos_highest = portfolio_state.get("position_highest_closes", {})
        pos_atrs    = portfolio_state.get("position_entry_atrs", {})
        pos_prices  = portfolio_state.get("position_entry_prices", {})
        sig_map     = {s.get("symbol", ""): s for s in result_signals}

        passed     = []
        n_deferred = 0

        for order in order_objects:
            sym    = getattr(order, "symbol", "")
            side   = getattr(order, "side",   "")
            reason = getattr(order, "reason", "") or ""

            # Only filter RSR-triggered SELL orders
            if side != "SELL" or _RSR_REASON_TAG not in reason:
                passed.append(order)
                continue

            # Check deferral expiry first
            sym_state = state.get(sym, {})
            if sym_state:
                expire_str = sym_state.get("defer_expires", "")
                if expire_str:
                    expire_date = datetime.strptime(expire_str, "%Y-%m-%d").date()
                    if today_date >= expire_date:
                        logger.info(
                            "[ATR_EXT] %s: deferral expired (%s) → SELL proceeds", sym, expire_str
                        )
                        passed.append(order)
                        state.pop(sym, None)
                        continue

            # Compute close_today from signal
            sig         = sig_map.get(sym, {})
            entry_price = float(pos_prices.get(sym, sig.get("entry_price", 0.0)))
            pnl_pct     = float(sig.get("unrealized_pnl_pct", 0.0))
            close_today = entry_price * (1.0 + pnl_pct) if entry_price > 0 else 0.0

            # Require profitable position
            if pnl_pct <= 0 or close_today <= 0:
                passed.append(order)
                continue

            highest_close = float(pos_highest.get(sym, 0.0))
            entry_atr     = float(pos_atrs.get(sym, 0.0))

            if highest_close <= 0 or entry_atr <= 0:
                # Missing portfolio_state data → pass through (fail-open)
                passed.append(order)
                continue

            threshold = highest_close - atr_mult * entry_atr
            if close_today > threshold:
                # ATR Extension condition met → defer
                if not sym_state:
                    expire_date = today_date + timedelta(days=max_defer_days)
                    state[sym] = {
                        "deferred_since":  today,
                        "defer_expires":   expire_date.strftime("%Y-%m-%d"),
                        "highest_close":   round(highest_close, 2),
                        "entry_atr":       round(entry_atr, 2),
                        "close_at_defer":  round(close_today, 2),
                        "threshold":       round(threshold, 2),
                    }
                n_deferred += 1
                logger.info(
                    "[ATR_EXT] %s: DEFERRED RSR exit — close=%.0f > threshold=%.0f"
                    " (highest=%.0f - %.1f×ATR=%.0f)",
                    sym, close_today, threshold,
                    highest_close, atr_mult, entry_atr,
                )
                # Intentionally NOT appended to `passed` → SELL suppressed
            else:
                # Below threshold → allow SELL
                passed.append(order)
                state.pop(sym, None)

        _save_state(state, state_path)
        return passed, n_deferred

    except Exception as e:
        logger.warning("[ATR_EXT] filter failed (%s) — returning original orders", e)
        return order_objects, 0
