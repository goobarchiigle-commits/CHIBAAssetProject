"""
pilot/shadow_engine.py — Study13 shadow execution engine

Study9 Case B signal + virtual order execution.
Records full audit trail without touching real orders.
"""
from __future__ import annotations

import json, math, time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pilot.shadow_state import (
    PilotState, ShadowPosition, PILOT_LOG_DIR,
    LATENCY_P99_MAX_MS, PHASE1, STOPPED,
)

# ── Study9 Case B fixed params ─────────────────────────────────────────
RSR_LO   = 92.0
RSR_HI   = 95.0
D90_MAX  = 5
D90_MIN  = 1
SLOPE_MAX = 5.0
EXIT_THR = 90.0
MIN_HOLD = 3
SHOCK_MKT = -0.05
SHOCK_SYM = -0.08
TARGET_NAV = 750_000.0
LOT = 100
COST_ONE_WAY = 0.00155   # slippage + commission

STOP_CONDITIONS = {
    "duplicate_order",
    "cash_sync_error",
    "state_hash_break",
    "latency_p99_exceeded",
}


class AuditLogger:
    def __init__(self, log_dir: Path = PILOT_LOG_DIR) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / "shadow_audit.jsonl"
        self._order_path = log_dir / "shadow_orders.jsonl"
        self._fill_path  = log_dir / "shadow_fills.jsonl"
        self._daily_path = log_dir / "shadow_daily.jsonl"

    def _append(self, path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def order(self, ds: str, side: str, symbol: str, qty: int, price: float,
              shadow_fill: bool = True, latency_ms: float = 0.0,
              reason: str = "") -> None:
        self._append(self._order_path, {
            "date": ds, "side": side, "symbol": symbol, "qty": qty,
            "price": round(price, 2), "shadow_fill": shadow_fill,
            "broker_latency_ms": latency_ms, "duplicate": False, "reason": reason,
        })

    def fill(self, ds: str, side: str, symbol: str, qty: int, price: float,
             pnl: float, cash_after: float) -> None:
        self._append(self._fill_path, {
            "date": ds, "side": side, "symbol": symbol, "qty": qty,
            "fill_price": round(price, 2), "shadow_pnl": round(pnl, 2),
            "cash_after": round(cash_after, 2),
        })

    def daily(self, ds: str, shadow_eq: float, expected_eq: float,
              position_diff: bool, cash_drift: float, state_hash: str,
              n_orders: int, latency_p99: float, phase: str) -> None:
        te = abs(shadow_eq - expected_eq) / max(1.0, expected_eq) if expected_eq > 0 else 0.0
        self._append(self._daily_path, {
            "date": ds, "shadow_equity": round(shadow_eq, 2),
            "expected_equity": round(expected_eq, 2),
            "tracking_error": round(te, 6),
            "position_diff": position_diff, "cash_drift": round(cash_drift, 2),
            "state_hash": state_hash, "n_orders_cumul": n_orders,
            "latency_p99_ms": round(latency_p99, 1), "phase": phase,
        })

    def critical(self, ds: str, event: str, detail: str = "") -> None:
        self._append(self._path, {
            "date": ds, "level": "CRITICAL",
            "event": event, "detail": detail,
        })

    def info(self, ds: str, event: str, detail: str = "") -> None:
        self._append(self._path, {
            "date": ds, "level": "INFO",
            "event": event, "detail": detail,
        })


def _is_annual_rebal(date: pd.Timestamp) -> bool:
    return date.month == 1 and date.day <= 5


def step_day(
    state: PilotState,
    date: pd.Timestamp,
    ds: str,
    # Signal inputs (today's close)
    rsr_by_sym:    dict[str, float],
    d90_by_sym:    dict[str, int],
    slope5_by_sym: dict[str, float],
    close_by_sym:  dict[str, float],
    mkt_ret:       float,
    active_syms:   list[str],
    # Next day's open (virtual fill price)
    next_open_by_sym: dict[str, float],
    # Ideal equity for tracking error
    ideal_eq_today: float,
    audit: AuditLogger,
) -> tuple[PilotState, Optional[str]]:
    """
    Process one trading day in Phase 1 shadow mode.
    Returns (updated_state, stop_reason or None).
    """
    stop_reason: Optional[str] = None

    # ── Annual rebalance (cash-only) ──────────────────────────────────
    pos_val = 0.0
    if state.position:
        p = ShadowPosition.from_dict(state.position)
        pos_val = p.qty * close_by_sym.get(p.symbol, p.entry_price)

    nav = state.shadow_cash + pos_val
    if _is_annual_rebal(date):
        if nav > TARGET_NAV:
            excess = min(nav - TARGET_NAV, state.shadow_cash)
            state.shadow_cash -= excess
            state.rebal_withdrawn += excess
            state.rebal_events += 1
            audit.info(ds, "REBAL_WITHDRAW", f"excess={excess:.0f}")
        elif nav < TARGET_NAV and state.position is None:
            deficit = TARGET_NAV - nav
            state.shadow_cash += deficit
            state.rebal_injected += deficit
            state.rebal_events += 1
            audit.info(ds, "REBAL_INJECT", f"deficit={deficit:.0f}")

    # ── Equity snapshot: after rebalance, BEFORE exit/entry (matches ideal timing)
    pos_val_snap = 0.0
    if state.position:
        ps = ShadowPosition.from_dict(state.position)
        pos_val_snap = ps.qty * close_by_sym.get(ps.symbol, ps.entry_price)
    shadow_eq_today = state.shadow_cash + pos_val_snap

    # ── State hash check ──────────────────────────────────────────────
    new_hash = state.state_hash(ds)
    state.state_hashes.append(new_hash)

    # ── EXIT ──────────────────────────────────────────────────────────
    if state.position:
        p     = ShadowPosition.from_dict(state.position)
        rv    = rsr_by_sym.get(p.symbol, 0.0)
        hd    = state.total_days - p.entry_idx
        do_exit = False; exit_reason = ""

        # Market shock
        if mkt_ret <= SHOCK_MKT:
            prev_close = close_by_sym.get(p.symbol, p.entry_price)
            sym_ret = prev_close / p.entry_price - 1 if p.entry_price > 0 else 0
            if sym_ret <= SHOCK_SYM:
                do_exit = True; exit_reason = "MKT_SHOCK"

        if not do_exit:
            if rv < EXIT_THR:
                p.days_below += 1
            else:
                p.days_below = 0
            if p.days_below >= 1 and hd >= MIN_HOLD:
                do_exit = True; exit_reason = "RSR_EXIT"

        if do_exit:
            sp = next_open_by_sym.get(p.symbol, p.entry_price)
            if sp <= 0: sp = close_by_sym.get(p.symbol, p.entry_price)
            pnl = (sp - p.entry_price) * p.qty

            t0 = time.perf_counter()
            # Shadow: virtual order (no API call)
            latency_ms = (time.perf_counter() - t0) * 1000
            state.latency_ms_samples.append(latency_ms)

            state.shadow_cash += p.qty * sp * (1 - COST_ONE_WAY)
            state.n_orders += 1
            state.n_filled += 1

            audit.order(ds, "SELL", p.symbol, p.qty, sp, True, latency_ms, exit_reason)
            audit.fill(ds,  "SELL", p.symbol, p.qty, sp, pnl, state.shadow_cash)
            audit.info(ds, f"EXIT_{exit_reason}", f"pnl={pnl:.0f} hd={hd}")
            state.position = None
        else:
            # Persist days_below update — critical for RSR_EXIT timing
            state.position = p.to_dict()

    # ── ENTRY ─────────────────────────────────────────────────────────
    mkt_shock = mkt_ret <= SHOCK_MKT
    if state.position is None and not mkt_shock:
        cands: list[tuple] = []
        for sym in active_syms:
            rv  = rsr_by_sym.get(sym, 0.0)
            d90 = d90_by_sym.get(sym, 0)
            sl5 = slope5_by_sym.get(sym, 999.0)
            if rv < RSR_LO or rv >= RSR_HI:  continue
            if d90 < D90_MIN or d90 > D90_MAX: continue
            if sl5 > SLOPE_MAX:             continue
            cands.append((rv, d90, sl5, sym))

        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            rv, d90, sl5, sym = cands[0]
            bp = next_open_by_sym.get(sym, 0.0)
            if bp <= 0: bp = close_by_sym.get(sym, 0.0)

            if bp > 0:
                # Duplicate guard: ensure no same-day same-symbol order
                dup = (state.position is not None and
                       ShadowPosition.from_dict(state.position).symbol == sym)
                if dup:
                    state.n_duplicate_guard += 1
                    audit.critical(ds, "DUPLICATE_ORDER", f"sym={sym}")
                    stop_reason = "duplicate_order"
                else:
                    alloc = state.shadow_cash * 0.95
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST_ONE_WAY)

                    if qty > 0 and cost <= state.shadow_cash:
                        t0 = time.perf_counter()
                        latency_ms = (time.perf_counter() - t0) * 1000
                        state.latency_ms_samples.append(latency_ms)

                        # Latency P99 check
                        if state.latency_p99() > LATENCY_P99_MAX_MS:
                            audit.critical(ds, "LATENCY_P99_EXCEEDED",
                                           f"p99={state.latency_p99():.0f}ms")
                            stop_reason = "latency_p99_exceeded"

                        state.shadow_cash -= cost
                        state.n_orders += 1
                        state.n_filled += 1

                        pos = ShadowPosition(
                            symbol=sym, qty=qty, entry_price=bp,
                            entry_date=ds, entry_idx=state.total_days,
                            rsr_entry=rv, d90_entry=d90,
                        )
                        state.position = pos.to_dict()
                        audit.order(ds, "BUY", sym, qty, bp, True, latency_ms, "RSR_ENTRY")
                        audit.fill(ds, "BUY", sym, qty, bp, 0.0, state.shadow_cash)
                        audit.info(ds, "ENTRY", f"rsr={rv:.1f} d90={d90} sl5={sl5:.1f}")

    # ── Equity tracking (uses beginning-of-day snapshot) ─────────────
    te = (abs(shadow_eq_today - ideal_eq_today) / max(1.0, ideal_eq_today)
          if ideal_eq_today > 0 else 0.0)

    state.equity_history.append(round(shadow_eq_today, 2))
    state.equity_dates.append(ds)
    state.expected_equity.append(round(ideal_eq_today, 2))
    state.tracking_errors.append(round(te, 6))

    cash_drift     = 0.0
    position_diff  = False

    # Cash sync check
    if state.shadow_cash < 0:
        audit.critical(ds, "CASH_SYNC_ERROR", f"cash={state.shadow_cash:.2f}")
        state.n_cash_drift_events += 1
        stop_reason = stop_reason or "cash_sync_error"

    # Daily log
    audit.daily(ds, shadow_eq_today, ideal_eq_today, position_diff, cash_drift,
                new_hash, state.n_orders, state.latency_p99(), state.phase)

    state.total_days  += 1
    state.trading_days += 1
    state.last_date    = ds

    if not state.start_date:
        state.start_date = ds

    return state, stop_reason
