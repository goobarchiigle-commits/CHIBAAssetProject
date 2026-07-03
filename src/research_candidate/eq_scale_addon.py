"""
research_candidate/eq_scale_addon.py — Study45 D_EQ_SCALE live addon.
Triggers a BUY addon when unrealized_gain >= 1×entry_ATR20.
One addon per position lifecycle; state persisted to runtime JSON.
FAIL_OPEN: returns empty list on any error.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

UNIT_SHARES: int = 100  # Japanese unit (1単元)


@dataclass
class EqScaleAddonOrder:
    symbol: str
    qty: int
    estimated_price: float
    estimated_cost: float
    reason: str


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
        logger.warning("[EQ_SCALE_ADDON] state save failed: %s", e)


def _purge_exited(state: dict, held_syms: set) -> dict:
    """Remove positions no longer held (lifecycle ended)."""
    return {k: v for k, v in state.items() if k in held_syms}


def generate_eq_scale_addon_orders(
    held_signals: list[dict],
    portfolio_state: dict,
    available_cash: float,
    portfolio_equity: float,
    max_single_weight: float,
    state_path: Path,
    today: str,
    run_id: str = "",
    atr_mult: float = 1.0,
    size_frac: float = 0.25,
) -> list[EqScaleAddonOrder]:
    """
    Evaluate held positions for D_EQ_SCALE addon eligibility.

    Trigger:  unrealized_gain_yen >= atr_mult × entry_ATR20
    Size:     available_cash × size_frac, rounded down to unit (100 shares)
    Guards:
      - One addon per position lifecycle (tracked by entry_date key)
      - qty >= 1 unit (100 shares)
      - addon_weight = addon_value / portfolio_equity <= max_single_weight × 1.5
    FAIL_OPEN: returns [] on any error.
    """
    try:
        held_syms     = {s.get("symbol", "") for s in held_signals if s.get("currently_holding")}
        state         = _load_state(state_path)
        state         = _purge_exited(state, held_syms)

        pos_prices    = portfolio_state.get("position_entry_prices", {})
        pos_atrs      = portfolio_state.get("position_entry_atrs", {})
        pos_entry_dts = portfolio_state.get("position_entry_dates", {})

        orders: list[EqScaleAddonOrder] = []

        for sig in held_signals:
            if not sig.get("currently_holding"):
                continue
            sym = sig.get("symbol", "")
            if not sym:
                continue

            entry_date = str(pos_entry_dts.get(sym, ""))
            sym_state  = state.get(sym, {})

            # Clear stale state when position has been re-opened
            if sym_state.get("entry_date", "") != entry_date:
                sym_state = {}

            if sym_state.get("addon_done", False):
                continue

            entry_price = float(pos_prices.get(sym, sig.get("entry_price", 0.0)))
            pnl_pct     = float(sig.get("unrealized_pnl_pct", 0.0))
            cur_price   = entry_price * (1.0 + pnl_pct) if entry_price > 0 else 0.0

            if cur_price <= 0 or entry_price <= 0:
                continue

            entry_atr = float(pos_atrs.get(sym, 0.0))
            if entry_atr <= 0:
                continue

            unrealized_gain = cur_price - entry_price
            trigger         = atr_mult * entry_atr

            if unrealized_gain < trigger:
                logger.debug(
                    "[EQ_SCALE_ADDON] %s: gain=¥%.0f < trigger=¥%.0f — skip",
                    sym, unrealized_gain, trigger,
                )
                continue

            # Size: available_cash × size_frac, rounded to lot
            raw_qty = int((available_cash * size_frac) / cur_price)
            qty     = (raw_qty // UNIT_SHARES) * UNIT_SHARES
            if qty < UNIT_SHARES:
                logger.info("[EQ_SCALE_ADDON] %s: cash insufficient for 1 lot — skip", sym)
                continue

            # Concentration gate: addon alone <= max_single_weight × 1.5
            addon_value  = qty * cur_price
            addon_weight = addon_value / portfolio_equity if portfolio_equity > 0 else 1.0
            cap          = max_single_weight * 1.5
            if addon_weight > cap:
                logger.info(
                    "[EQ_SCALE_ADDON] %s: addon_weight=%.1f%% > cap=%.1f%% — skip",
                    sym, addon_weight * 100, cap * 100,
                )
                continue

            reason = (
                f"EQ_SCALE_D: gain=¥{unrealized_gain:.0f}"
                f"≥{atr_mult:.1f}×ATR=¥{trigger:.0f}"
                f" qty={qty} size_frac={size_frac:.0%}"
            )
            orders.append(EqScaleAddonOrder(
                symbol=sym,
                qty=qty,
                estimated_price=cur_price,
                estimated_cost=addon_value,
                reason=reason,
            ))
            state[sym] = {
                "addon_done":   True,
                "addon_date":   today,
                "entry_date":   entry_date,
                "qty_added":    qty,
                "trigger_gain": round(unrealized_gain, 2),
                "entry_atr":    round(entry_atr, 2),
                "run_id":       run_id,
            }
            logger.info(
                "[EQ_SCALE_ADDON] %s: APPROVED qty=%d @ ¥%.0f — %s",
                sym, qty, cur_price, reason,
            )

        _save_state(state, state_path)
        return orders

    except Exception as e:
        logger.warning("[EQ_SCALE_ADDON] generate failed (%s) — returning []", e)
        return []
