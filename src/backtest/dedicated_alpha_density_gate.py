"""
src/backtest/dedicated_alpha_density_gate.py
Study 8 Opportunity Density Gate WF

背景:
  Concentration Relief WF で全Case WF=3/5。
  Fold3(2023) で trig=32-43/yr → sleeve が base を希薄化。
  → signal density 問題か？ density gate で入場制限を付けて WF 改善を試みる。

固定:
  ENTRY  RSR∈[92,95), days_cross90≤5
  EXIT   RSR<90 (即座)
  BASE   2-slot 70/30 no-refill (Concentration Relief Case E)
  CAP    equity×20%

Gate Cases (入場ゲートのみ変更 / exit・sizing・refill変更禁止):
  A  gate なし                       (baseline = Concentration Relief Case E)
  B  rolling 60d trigger ≤ 8
  C  rolling 60d trigger ≤ 6
  D  rolling 60d trigger ≤ 4
  E  base active positions ≤ 2
  F  combined cash idle ≥ 20%
  G  combined rolling exposure ≤ 85%

採用条件: WF≥4/5, ΔCAGR≥+0.3pp, ΔDD≤+1.5pp, alpha_ret≥85%

出力: reports/dedicated_alpha_density_gate.md

Run: cd C:/ai-trading && python src/backtest/dedicated_alpha_density_gate.py
"""
from __future__ import annotations

import sys, time, math, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy,
    Position, _take, calc_metrics,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR  = Path("reports")
IS_START     = "2018-01-01"
STALL_THR    = 0.15

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Sleeve config (fixed = Concentration Relief Case E) ───────────────
SLEEVE_MAX     = 2
SL_WEIGHTS     = [0.70, 0.30]
NO_REFILL      = True
SLEEVE_RSR_LO  = 92.0
SLEEVE_RSR_HI  = 95.0
SLEEVE_D90_MAX = 5
SLEEVE_CAP_FR  = 0.20
SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08
TRIG_WINDOW    = 60   # rolling trading days for trigger gate

GATE_SPECS: dict[str, dict] = {
    "A": {"type": "none",         "params": {},              "desc": "gate なし (Conc.Relief Case E)"},
    "B": {"type": "rolling_trig", "params": {"max": 8},      "desc": "rolling60d trigger ≤ 8"},
    "C": {"type": "rolling_trig", "params": {"max": 6},      "desc": "rolling60d trigger ≤ 6"},
    "D": {"type": "rolling_trig", "params": {"max": 4},      "desc": "rolling60d trigger ≤ 4"},
    "E": {"type": "base_active",  "params": {"max": 2},      "desc": "base active positions ≤ 2"},
    "F": {"type": "cash_idle",    "params": {"min": 0.20},   "desc": "combined cash idle ≥ 20%"},
    "G": {"type": "rolling_exp",  "params": {"max": 0.85},   "desc": "combined rolling exposure ≤ 85%"},
}

# Adoption
ADOPT_WF_MIN    = 4
ADOPT_DCAGR     = 0.3
ADOPT_DDD       = 1.5
ADOPT_ALPHA_RET = 85.0


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def classify_state(s5: float, s20: float) -> str:
    if np.isnan(s5) or np.isnan(s20): return "UNKNOWN"
    if abs(s5) < STALL_THR:            return "STALL"
    if s20 > 0:
        if s5 > s20: return "EARLY_UP"
        if s5 > 0:   return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"

STATE_SCORE_MAP = {
    "EARLY_UP": 3, "STEADY_UP": 2, "FLAT": 1,
    "STALL": 1, "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0,
}

def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    n_dates, n_syms = rsr_mat.shape
    out     = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = running
        above  = rsr_mat[i] >= threshold
        running[above] += 1
        running[~above]  = 0
    return out

def compute_slope_mats(rsr_mat: np.ndarray):
    s5  = np.zeros_like(rsr_mat, dtype=np.float32)
    s20 = np.zeros_like(rsr_mat, dtype=np.float32)
    s5[5:]   = (rsr_mat[5:]  - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:] - rsr_mat[:-20]) / 20.0
    return s5, s20

def regime_label(topix_close: pd.Series, year: int) -> str:
    yr = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ─────────────────────────────────────────────────────────────────────
#  GATE CHECK
# ─────────────────────────────────────────────────────────────────────

def check_gate(
    gate_spec: dict, i: int,
    sl_exit_indices: list[int],
    base_pos: dict,
    base_cash: float, sl_cash: float,
    b_eq: float, s_eq: float,
    b_inv: float, s_inv: float,
) -> tuple[bool, str]:
    gtype  = gate_spec["type"]
    params = gate_spec["params"]

    if gtype == "none":
        return True, "none"

    elif gtype == "rolling_trig":
        max_t = params["max"]
        t0    = i - TRIG_WINDOW
        cnt   = sum(1 for idx in sl_exit_indices if t0 <= idx <= i)
        if cnt >= max_t:
            return False, f"rolling60={cnt}>={max_t}"
        return True, "none"

    elif gtype == "base_active":
        max_a = params["max"]
        n_a   = len(base_pos)
        if n_a > max_a:
            return False, f"base_active={n_a}>{max_a}"
        return True, "none"

    elif gtype == "cash_idle":
        min_idle   = params["min"]
        total_eq   = b_eq + s_eq
        total_cash = base_cash + sl_cash
        idle_r     = total_cash / max(1.0, total_eq)
        if idle_r < min_idle:
            return False, f"idle={idle_r:.1%}<{min_idle:.0%}"
        return True, "none"

    elif gtype == "rolling_exp":
        max_exp  = params["max"]
        total_eq = b_eq + s_eq
        total_inv = b_inv + s_inv
        exp_r    = total_inv / max(1.0, total_eq)
        if exp_r > max_exp:
            return False, f"exp={exp_r:.1%}>{max_exp:.0%}"
        return True, "none"

    return True, "none"


# ─────────────────────────────────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────────────────────────────────

def run_gate(
    open_mat, close_mat,
    sig_mat, sig_ready, rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, trade_syms, cfg,
    common_dates,
    gate_spec: dict,
) -> dict:
    capital      = float(cfg.portfolio.capital)
    base_max_pos = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)

    rc            = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W  = float(rc.sector_cap)  if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True))  if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0))  if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    # ── Base state ────────────────────────────────────────────────────
    base_cash   = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak   = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    # ── Sleeve state ──────────────────────────────────────────────────
    sl_slots: list[Position | None] = [None] * SLEEVE_MAX
    sl_cash  = float(capital * SLEEVE_CAP_FR)
    sl_trades: list[dict] = []
    sl_exit_indices: list[int] = []   # rolling trigger tracking (exit day indices)

    # ── Audit log ─────────────────────────────────────────────────────
    gate_log: list[dict] = []       # blocked entries
    n_cand_found   = 0              # total candidates found (gate-check level)
    n_gate_blocked = 0              # blocked by gate
    n_accepted     = 0              # entered

    # ── Daily arrays ──────────────────────────────────────────────────
    n_dates = len(common_dates)
    base_eq  = np.zeros(n_dates, dtype=np.float64)
    sl_eq    = np.zeros(n_dates, dtype=np.float64)
    base_inv = np.zeros(n_dates, dtype=np.float64)
    sl_inv   = np.zeros(n_dates, dtype=np.float64)
    daily_overlap = np.full(n_dates, np.nan, dtype=np.float64)
    daily_trig60  = np.zeros(n_dates, dtype=np.int32)  # rolling trigger count per day
    daily_gate_ok = np.zeros(n_dates, dtype=np.int8)   # 1=gate open, 0=gate closed

    sl_held_set   = [set() for _ in range(n_dates)]
    base_held_set = [set() for _ in range(n_dates)]

    for i, date in enumerate(common_dates):
        date_str = str(date.date())

        b_inv_v = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                      for s, p in base_pos.items())
        s_inv_v = sum(
            sl_slots[k].qty * float(close_mat[i, sym_to_i[sl_slots[k].symbol]])
            for k in range(SLEEVE_MAX) if sl_slots[k] is not None
        )
        b_eq_v  = base_cash + b_inv_v
        s_eq_v  = sl_cash + s_inv_v

        base_eq[i]  = b_eq_v
        sl_eq[i]    = s_eq_v
        base_inv[i] = b_inv_v
        sl_inv[i]   = s_inv_v

        n_occ = sum(1 for k in range(SLEEVE_MAX) if sl_slots[k] is not None)
        n_all = len(base_pos) + n_occ
        daily_overlap[i] = (n_occ / max(n_all, 1)) if n_all > 0 else 0.0

        t60 = max(0, i - TRIG_WINDOW)
        daily_trig60[i] = sum(1 for idx in sl_exit_indices if t60 <= idx <= i)

        sl_held_set[i]   = {sl_slots[k].symbol for k in range(SLEEVE_MAX) if sl_slots[k] is not None}
        base_held_set[i] = set(base_pos.keys())

        # CB check (base only)
        if b_eq_v > base_peak: base_peak = b_eq_v
        dd = (b_eq_v - base_peak) / base_peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0

        gross_cap = gross_normal
        if gross_enabled and topix_ret20 is not None:
            r20 = float(topix_ret20[i]) if i < len(topix_ret20) else 0.0
            r60 = float(topix_ret60[i]) if i < len(topix_ret60) else 0.0
            if   r20 < -0.05: gross_cap = gross_dd5
            elif r60 < -0.08: gross_cap = gross_dd8

        is_bear     = bool(bear_arr[i]) if bear_arr is not None else False
        sec_cap_eff = 0.18 if is_bear else MAX_SECTOR_W
        mkt_shock   = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n_dates: break
        next_i = i + 1

        # ── BASE: sells ───────────────────────────────────────────────
        base_sell: list[tuple] = []
        base_buy:  list[tuple] = []

        for sym in active_syms:
            si_sym     = sym_to_i[sym]
            is_holding = sym in base_pos
            hold_idx   = (i - base_pos[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si_sym])
            close_t    = float(close_mat[i, si_sym])

            if mkt_shock and is_holding:
                if i > 0:
                    prev_c = float(close_mat[i-1, si_sym])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= SHOCK_SYM_THR:
                        base_sell.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not is_holding: continue
            if is_holding and max_hold is not None and hold_idx > max_hold:
                base_sell.append((sym, "TIME_STOP")); continue
            if is_holding and rsr_val < rsr_exit_thr and hold_idx >= min_hold:
                base_sell.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si_sym]) if sig_ready[si_sym] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                base_sell.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si_sym]) < 0.5:
                    continue
                base_buy.append((rsr_val, sym))

        for sym, reason in base_sell:
            if sym not in base_pos: continue
            pos     = base_pos[sym]
            sell_px = float(open_mat[next_i, sym_to_i[sym]])
            pnl     = (sell_px - pos.entry_price) * pos.qty
            base_cash += pos.qty * sell_px * (1 - COST_ONE_WAY)
            base_trades.append({
                "side": "SELL", "symbol": sym, "pnl": pnl,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "entry_idx": pos.entry_idx,
                "exit_idx": i, "reason": reason, "date": date_str,
            })
            del base_pos[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not cb_active and base_buy:
            base_buy.sort(key=lambda x: -x[0])
            for rsr_v, sym in base_buy:
                si_sym     = sym_to_i[sym]
                open_slots = base_max_pos - len(base_pos)
                if open_slots <= 0: break
                buy_px = float(open_mat[next_i, si_sym])
                if buy_px <= 0: continue
                if not _sector_ok(sym, base_pos, close_mat, i, sym_to_i, trade_syms,
                                   capital, sec_cap_eff): continue
                if gross_enabled:
                    cur_g = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in base_pos.values()) / max(1.0, capital)
                    if cur_g + buy_px * LOT / max(1.0, capital) > gross_cap: continue
                alloc = capital / base_max_pos
                qty   = int(alloc / buy_px / LOT) * LOT
                if qty <= 0: continue
                if qty * buy_px * (1 + COST_ONE_WAY) > base_cash: continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms,
                              base_trades, base_pos, rsr_v)
                base_cash -= qty * buy_px * (1 + COST_ONE_WAY)
                base_trades[-1]["date"] = date_str

        # ── SLEEVE: exits ─────────────────────────────────────────────
        for k in range(SLEEVE_MAX):
            pos = sl_slots[k]
            if pos is None: continue
            si_sym  = sym_to_i[pos.symbol]
            rsr_val = float(rsr_mat[i, si_sym])
            close_t = float(close_mat[i, si_sym])

            do_exit, reason = False, ""
            if mkt_shock and i > 0:
                prev_c = float(close_mat[i-1, si_sym])
                if prev_c > 0 and (close_t / prev_c - 1.0) <= SHOCK_SYM_THR:
                    do_exit, reason = True, "MARKET_SHOCK_EXIT"
            if not do_exit and rsr_val < SLEEVE_RSR_LO:
                do_exit, reason = True, "RSR_EXIT_LO"

            if do_exit:
                sell_px = float(open_mat[next_i, si_sym])
                pnl     = (sell_px - pos.entry_price) * pos.qty
                sl_cash += pos.qty * sell_px * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side": "SELL", "symbol": pos.symbol, "pnl": pnl,
                    "entry": pos.entry_price, "exit": sell_px,
                    "qty": pos.qty, "entry_idx": pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": date_str,
                    "slot": k, "weight": SL_WEIGHTS[k],
                    "fwd10": np.nan, "fwd20": np.nan,
                })
                sl_exit_indices.append(i)
                sl_slots[k] = None

        # ── SLEEVE: entries ───────────────────────────────────────────
        if not mkt_shock:
            n_occ_now = sum(1 for k in range(SLEEVE_MAX) if sl_slots[k] is not None)
            n_open    = SLEEVE_MAX - n_occ_now

            # no-refill: wait for full clearance
            if NO_REFILL and 0 < n_occ_now < SLEEVE_MAX:
                pass
            elif n_open > 0:
                # Gate check (before finding candidates — gate is global per day)
                gate_ok, gate_reason = check_gate(
                    gate_spec, i, sl_exit_indices, base_pos,
                    base_cash, sl_cash, b_eq_v, s_eq_v, b_inv_v, s_inv_v,
                )
                daily_gate_ok[i] = 1 if gate_ok else 0

                # Find candidates regardless of gate (for missed_alpha log)
                sl_syms = {sl_slots[k].symbol for k in range(SLEEVE_MAX) if sl_slots[k] is not None}
                cands: list[tuple] = []
                for sym in active_syms:
                    if sym in base_pos: continue
                    if sym in sl_syms:  continue
                    si_sym = sym_to_i[sym]
                    rsr_v  = float(rsr_mat[i, si_sym])
                    if not (SLEEVE_RSR_LO <= rsr_v < SLEEVE_RSR_HI): continue
                    d90 = int(cross90_mat[i, si_sym])
                    if d90 > SLEEVE_D90_MAX: continue
                    if sym_active_mat is not None and float(sym_active_mat[i, si_sym]) < 0.5:
                        continue
                    st = classify_state(float(slope5_mat[i, si_sym]),
                                        float(slope20_mat[i, si_sym]))
                    sr = STATE_SCORE_MAP.get(st, 0)
                    cands.append((rsr_v, d90, sr, sym))

                if cands:
                    cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                    n_cand_found += len(cands)

                    if not gate_ok:
                        n_gate_blocked += min(len(cands), n_open)
                        # Log top-1 candidate as missed alpha
                        rsr_v, d90, sr, sym = cands[0]
                        gate_log.append({
                            "day_idx":    i,
                            "date_str":   date_str,
                            "symbol":     sym,
                            "rsr":        rsr_v,
                            "gate_reason":gate_reason,
                            "trig60":     daily_trig60[i],
                            "base_active":len(base_pos),
                            "fwd10":      np.nan,
                            "fwd20":      np.nan,
                        })
                    else:
                        # Enter
                        cand_idx = 0
                        for k in range(SLEEVE_MAX):
                            if sl_slots[k] is not None: continue
                            if cand_idx >= len(cands): break
                            rsr_v, d90, sr, sym = cands[cand_idx]; cand_idx += 1
                            si_sym  = sym_to_i[sym]
                            buy_px  = float(open_mat[next_i, si_sym])
                            if buy_px <= 0: continue
                            target  = b_eq_v * SLEEVE_CAP_FR * SL_WEIGHTS[k]
                            alloc   = min(sl_cash, target)
                            qty     = int(alloc / buy_px / LOT) * LOT
                            cost    = qty * buy_px * (1 + COST_ONE_WAY)
                            if qty <= 0 or cost > sl_cash: continue
                            sl_cash -= cost
                            sl_slots[k] = Position(sym, trade_syms.get(sym, ""),
                                                    qty, buy_px, next_i, rsr_v)
                            sl_trades.append({
                                "side": "BUY", "symbol": sym,
                                "entry": buy_px, "exit": None,
                                "qty": qty, "pnl": None,
                                "entry_idx": next_i, "exit_idx": None,
                                "reason": f"RSR={rsr_v:.1f} d90={d90} slot={k}",
                                "date": date_str, "rsr_entry": rsr_v,
                                "slot": k, "weight": SL_WEIGHTS[k],
                                "fwd10": np.nan, "fwd20": np.nan,
                            })
                            n_accepted += 1

    # ── Post-hoc: fwd10/fwd20 on SELL trades ─────────────────────────
    for t in sl_trades:
        if t["side"] != "SELL": continue
        si = sym_to_i.get(t["symbol"], -1)
        if si < 0: continue
        ei = t["exit_idx"]
        for fk, fn in [("fwd10", 10), ("fwd20", 20)]:
            if ei + fn < n_dates:
                c0 = float(close_mat[ei, si])
                cn = float(close_mat[ei + fn, si])
                t[fk] = round((cn / c0 - 1) * 100, 2) if c0 > 0 and cn > 0 else float("nan")

    # ── Post-hoc: missed alpha for gate log ───────────────────────────
    for g in gate_log:
        si = sym_to_i.get(g["symbol"], -1)
        if si < 0: continue
        oi = g["day_idx"] + 1   # would-be open at next day
        for fk, fn in [("fwd10", 10), ("fwd20", 20)]:
            if oi + fn < n_dates:
                c0 = float(close_mat[oi, si])
                cn = float(close_mat[oi + fn, si])
                g[fk] = round((cn / c0 - 1) * 100, 2) if c0 > 0 and cn > 0 else float("nan")

    return {
        "base_eq":        base_eq,
        "sl_eq":          sl_eq,
        "base_inv":       base_inv,
        "sl_inv":         sl_inv,
        "base_trades":    base_trades,
        "sl_trades":      sl_trades,
        "sl_held_set":    sl_held_set,
        "base_held_set":  base_held_set,
        "gate_log":       gate_log,
        "n_cand_found":   n_cand_found,
        "n_gate_blocked": n_gate_blocked,
        "n_accepted":     n_accepted,
        "daily_overlap":  daily_overlap,
        "daily_trig60":   daily_trig60,
        "daily_gate_ok":  daily_gate_ok,
        "sl_exit_indices":sl_exit_indices,
    }


# ─────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    sim: dict,
    close_mat: np.ndarray,
    common_dates,
    fold: dict,
    capital: float,
) -> dict:
    oos_s = fold["oos_start"]
    oos_e = fold["oos_end"]

    date_strs   = [str(d.date()) for d in common_dates]
    oos_indices = [i for i, ds in enumerate(date_strs) if oos_s <= ds <= oos_e]
    if not oos_indices: return {}
    si = oos_indices[0]; ei = oos_indices[-1] + 1

    base_eq  = sim["base_eq"]
    sl_eq    = sim["sl_eq"]
    base_inv = sim["base_inv"]
    sl_inv   = sim["sl_inv"]

    base_oos  = base_eq[si:ei].tolist()
    sl_oos    = sl_eq[si:ei].tolist()
    comb_oos  = (base_eq + sl_eq)[si:ei].tolist()
    b_inv_o   = base_inv[si:ei].tolist()
    s_inv_o   = sl_inv[si:ei].tolist()
    oos_dates = list(common_dates[si:ei])
    n_days    = len(oos_dates)
    n_years   = n_days / 252.0

    if not base_oos or len(base_oos) < 5: return {}

    base_t  = [t for t in sim["base_trades"] if oos_s <= t.get("date","") <= oos_e]
    sl_t    = [t for t in sim["sl_trades"]   if oos_s <= t.get("date","") <= oos_e]
    sl_sell = [t for t in sl_t if t["side"] == "SELL"]
    comb_t  = base_t + sl_t

    base_exp_list = [bi / max(1, be) for bi, be in zip(b_inv_o, base_oos)]
    base_m = calc_metrics(base_oos, base_t, base_exp_list, base_oos[0], oos_dates)

    comb_exp_list = [(bi + si_) / max(1, be + se)
                     for bi, si_, be, se in zip(b_inv_o, s_inv_o, base_oos, sl_oos)]
    comb_m = calc_metrics(comb_oos, comb_t, comb_exp_list, comb_oos[0], oos_dates)

    sl_start   = sl_oos[0]
    unused_oos = [be + sl_start for be in base_oos]
    unused_m   = calc_metrics(unused_oos, base_t,
                               [bi / max(1, u) for bi, u in zip(b_inv_o, unused_oos)],
                               unused_oos[0], oos_dates)

    delta_cagr   = comb_m.get("cagr", 0)   - base_m.get("cagr", 0)
    comb_dd      = comb_m.get("max_dd", 0)
    base_dd      = base_m.get("max_dd", 0)
    delta_dd     = -(comb_dd - base_dd)
    delta_calmar = comb_m.get("calmar", 0) - base_m.get("calmar", 0)

    sl_cagr = sl_calmar = 0.0
    if len(sl_oos) >= 5 and sl_oos[0] > 0:
        sl_exp = [si_ / max(1, se) for si_, se in zip(s_inv_o, sl_oos)]
        sl_m   = calc_metrics(sl_oos, sl_t, sl_exp, sl_oos[0], oos_dates)
        sl_cagr   = sl_m.get("cagr", 0)
        sl_calmar = sl_m.get("calmar", 0)

    n_triggers       = len(sl_sell)
    trigger_per_year = n_triggers / max(0.01, n_years)

    hold_days_list = [t.get("exit_idx", 0) - t.get("entry_idx", 0) for t in sl_sell
                      if t.get("exit_idx", 0) > t.get("entry_idx", 0)]
    avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0.0

    invested_days = int(sum(1 for v in s_inv_o if v > 0))
    idle_days     = n_days - invested_days
    capital_util  = invested_days / max(1, n_days) * 100.0

    sl_wins = [t for t in sl_sell if (t.get("pnl") or 0) > 0]
    sl_loss = [t for t in sl_sell if (t.get("pnl") or 0) <= 0]
    hit_rate = len(sl_wins) / max(1, len(sl_sell)) * 100.0
    gp = sum(t["pnl"] for t in sl_wins) if sl_wins else 0.0
    gl = abs(sum(t["pnl"] for t in sl_loss)) if sl_loss else 0.0
    pf = gp / max(1.0, gl)

    fwd10v = [t.get("fwd10", float("nan")) for t in sl_sell
              if not math.isnan(t.get("fwd10", float("nan")))]
    fwd20v = [t.get("fwd20", float("nan")) for t in sl_sell
              if not math.isnan(t.get("fwd20", float("nan")))]
    med_fwd10 = float(np.median(fwd10v)) if fwd10v else float("nan")
    med_fwd20 = float(np.median(fwd20v)) if fwd20v else float("nan")

    # Gate-specific: rejection count and missed alpha in OOS window
    gate_log_oos = [g for g in sim["gate_log"] if oos_s <= g.get("date_str","") <= oos_e]
    n_blocked_oos = len(gate_log_oos)

    n_buy_oos = len([t for t in sl_t if t["side"] == "BUY"])
    rejection_rate = (n_blocked_oos / max(1, n_blocked_oos + n_buy_oos)) * 100.0

    missed_fwd10v = [g.get("fwd10", float("nan")) for g in gate_log_oos
                     if not math.isnan(g.get("fwd10", float("nan")))]
    missed_fwd20v = [g.get("fwd20", float("nan")) for g in gate_log_oos
                     if not math.isnan(g.get("fwd20", float("nan")))]
    missed_alpha_10 = float(np.mean(missed_fwd10v)) if missed_fwd10v else float("nan")
    missed_alpha_20 = float(np.mean(missed_fwd20v)) if missed_fwd20v else float("nan")

    # avg_alpha_overlap for OOS window
    overlap_oos = sim["daily_overlap"][si:ei]
    avg_overlap = float(np.nanmean(overlap_oos))

    # avg rolling trig60 for OOS window
    trig60_oos = sim["daily_trig60"][si:ei]
    avg_trig60 = float(np.mean(trig60_oos))

    # Gate openness rate
    gate_ok_oos = sim["daily_gate_ok"][si:ei]
    gate_open_rate = float(np.mean(gate_ok_oos)) * 100.0

    # Base-sleeve correlation
    base_oos_arr = np.array(base_oos)
    sl_oos_arr   = np.array(sl_oos)
    bdr  = np.diff(base_oos_arr) / np.maximum(base_oos_arr[:-1], 1)
    sdr  = np.diff(sl_oos_arr)   / np.maximum(sl_oos_arr[:-1],   1)
    corr_to_base = 0.0
    if len(bdr) > 10 and np.std(sdr) > 1e-10:
        corr_to_base = float(np.corrcoef(bdr, sdr)[0, 1])

    held_syms  = sim["sl_held_set"][si:ei]
    base_sets  = sim["base_held_set"][si:ei]
    overlap_days = sum(1 for hs, bs in zip(held_syms, base_sets) if hs and hs & bs)

    fold_pass = (delta_cagr > ADOPT_DCAGR and delta_dd <= ADOPT_DDD)

    return {
        "base_cagr":     base_m.get("cagr", 0),
        "base_max_dd":   base_dd,
        "base_calmar":   base_m.get("calmar", 0),
        "comb_cagr":     comb_m.get("cagr", 0),
        "comb_max_dd":   comb_dd,
        "comb_calmar":   comb_m.get("calmar", 0),
        "unused_cagr":   unused_m.get("cagr", 0),
        "delta_cagr":    round(delta_cagr,   2),
        "delta_dd":      round(delta_dd,      2),
        "delta_calmar":  round(delta_calmar,  3),
        "sl_cagr":       round(sl_cagr,       2),
        "sl_calmar":     round(sl_calmar,      3),
        "n_triggers":    n_triggers,
        "trigger_per_year": round(trigger_per_year, 1),
        "avg_hold":      round(avg_hold,       1),
        "idle_days":     idle_days,
        "capital_util":  round(capital_util,   1),
        "hit_rate":      round(hit_rate,       1),
        "profit_factor": round(pf,             3),
        "med_fwd10":     med_fwd10,
        "med_fwd20":     med_fwd20,
        "n_blocked":     n_blocked_oos,
        "rejection_rate":round(rejection_rate, 1),
        "missed_alpha_10": missed_alpha_10,
        "missed_alpha_20": missed_alpha_20,
        "avg_overlap":   round(avg_overlap,    3),
        "avg_trig60":    round(avg_trig60,     1),
        "gate_open_rate":round(gate_open_rate, 1),
        "corr_to_base":  round(corr_to_base,   3),
        "overlap_days":  overlap_days,
        "fold_pass":     fold_pass,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATOR
# ─────────────────────────────────────────────────────────────────────

def evaluate_wf(fold_metrics: list[dict], sl_cagr_a: float | None = None) -> dict:
    valid = [f for f in fold_metrics if f]
    if not valid: return {}

    n_pass      = sum(1 for f in valid if f.get("fold_pass"))
    avg_dc      = float(np.mean([f["delta_cagr"]       for f in valid]))
    avg_dd      = float(np.mean([f["delta_dd"]         for f in valid]))
    avg_dcal    = float(np.mean([f["delta_calmar"]     for f in valid]))
    avg_trig    = float(np.mean([f["trigger_per_year"] for f in valid]))
    avg_hold    = float(np.mean([f["avg_hold"]         for f in valid]))
    avg_util    = float(np.mean([f["capital_util"]     for f in valid]))
    avg_slcagr  = float(np.mean([f["sl_cagr"]         for f in valid]))
    avg_reject  = float(np.mean([f["rejection_rate"]  for f in valid]))
    avg_idle    = float(np.mean([f["idle_days"]        for f in valid]))
    avg_olap    = float(np.mean([f["avg_overlap"]      for f in valid]))
    avg_gopen   = float(np.mean([f["gate_open_rate"]   for f in valid]))

    missed10v = [f["missed_alpha_10"] for f in valid
                 if not math.isnan(f.get("missed_alpha_10", float("nan")))]
    missed20v = [f["missed_alpha_20"] for f in valid
                 if not math.isnan(f.get("missed_alpha_20", float("nan")))]
    fwd10v = [f["med_fwd10"] for f in valid if not math.isnan(f.get("med_fwd10", float("nan")))]
    fwd20v = [f["med_fwd20"] for f in valid if not math.isnan(f.get("med_fwd20", float("nan")))]

    # regime_dependency = std of fold ΔCAGR
    fold_dcs = [f["delta_cagr"] for f in valid]
    regime_dep = float(np.std(fold_dcs))

    alpha_retention = (avg_slcagr / sl_cagr_a * 100) if sl_cagr_a and sl_cagr_a > 0 else float("nan")

    adopted = (
        n_pass >= ADOPT_WF_MIN
        and avg_dc > ADOPT_DCAGR
        and avg_dd <= ADOPT_DDD
        and (math.isnan(alpha_retention) or alpha_retention >= ADOPT_ALPHA_RET)
    )

    return {
        "n_pass":         n_pass,
        "avg_dc":         round(avg_dc,     2),
        "avg_dd":         round(avg_dd,     2),
        "avg_dcal":       round(avg_dcal,   3),
        "avg_trig":       round(avg_trig,   1),
        "avg_hold":       round(avg_hold,   1),
        "avg_util":       round(avg_util,   1),
        "avg_slcagr":     round(avg_slcagr, 2),
        "alpha_retention":round(alpha_retention, 1) if not math.isnan(alpha_retention) else float("nan"),
        "avg_reject":     round(avg_reject, 1),
        "avg_idle":       round(avg_idle,   1),
        "avg_overlap":    round(avg_olap,   3),
        "gate_open_rate": round(avg_gopen,  1),
        "regime_dep":     round(regime_dep, 3),
        "missed_alpha_10":round(float(np.mean(missed10v)), 2) if missed10v else float("nan"),
        "missed_alpha_20":round(float(np.mean(missed20v)), 2) if missed20v else float("nan"),
        "med_fwd10":      round(float(np.median(fwd10v)), 2) if fwd10v else float("nan"),
        "med_fwd20":      round(float(np.median(fwd20v)), 2) if fwd20v else float("nan"),
        "adopted":        adopted,
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def _f(v, fmt=".2f") -> str:
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:{fmt}}"

def _pp(v) -> str:
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:+.2f}pp"


def write_report(
    gate_results: dict,
    topix_close: pd.Series | None,
    output_path: Path,
    cagr_per_trigger: dict,
) -> None:
    L = []; w = L.append

    adopted = [c for c in GATE_SPECS if gate_results[c]["wf"].get("adopted")]
    first_c = next(iter(gate_results.values()))
    bl_cagrs = [f["base_cagr"] for f in first_c["folds"] if f]
    bl_avg   = float(np.mean(bl_cagrs)) if bl_cagrs else 0.0

    verdict = ("✅ GO LIVE (shadow 30d 必須)" if adopted
               else "🔬 KEEP RESEARCH" if any(
                   gate_results[c]["wf"].get("avg_dc", 0) > 0.0 for c in GATE_SPECS)
               else "❌ REJECT")

    w("# Dedicated Alpha Density Gate WF — Study 8")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n固定: ENTRY=RSR∈[92,95) d90≤5 / EXIT=RSR<90 / BASE=2-slot 70/30 no-refill / CAP=20%")
    w(f"\n採用条件: WF≥{ADOPT_WF_MIN}/5, ΔCAGR>+{ADOPT_DCAGR}pp, ΔDD≤+{ADOPT_DDD}pp, "
      f"alpha_ret≥{ADOPT_ALPHA_RET}%\n")
    w(f"**採用Gate**: {len(adopted)}件  |  最終判定: **{verdict}**\n")

    # ── 1. Executive Summary ──────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    w(f"- ベースライン avg CAGR: **{bl_avg:+.1f}%** (5-fold avg)")
    w(f"- 仮説: Fold3(2023) の trig=32-43/yr が signal density 問題で WF失敗の原因")
    w(f"- Gate 7種: rolling trig / base_active / cash_idle / rolling_exp")
    if adopted:
        best_c = max(adopted, key=lambda c: gate_results[c]["wf"]["avg_dc"])
        bwf    = gate_results[best_c]["wf"]
        w(f"- 最良採用: **{best_c}** ({GATE_SPECS[best_c]['desc']}) — "
          f"ΔCAGR={bwf['avg_dc']:+.2f}pp, ΔDD={bwf['avg_dd']:+.2f}pp, WF={bwf['n_pass']}/5")

    # ── 2. WF Summary ────────────────────────────────────────────────
    w("\n---\n## 2. WF サマリ (7 Gate × 5 Fold)\n")
    w("| Gate | 説明 | WF | ΔCAGR | ΔDD | sl_CAGR | α_ret | reject% | gate_open% | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for case, spec in GATE_SPECS.items():
        wf   = gate_results[case]["wf"]
        mark = "**✅**" if wf.get("adopted") else "❌"
        ar   = (_f(wf.get("alpha_retention", float("nan")), ".1f") + "%"
                if not math.isnan(wf.get("alpha_retention", float("nan"))) else "—")
        rj   = f"{wf.get('avg_reject', 0):.1f}%"
        go   = f"{wf.get('gate_open_rate', 100):.1f}%"
        w(f"| {case} | {spec['desc']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_dc',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_slcagr',0):+.1f}% "
          f"| {ar} "
          f"| {rj} "
          f"| {go} "
          f"| {mark} |")

    # ── 3. Trigger Reduction Analysis ────────────────────────────────
    w("\n---\n## 3. Trigger Reduction Analysis\n")
    wf_a = gate_results["A"]["wf"]
    trig_a = wf_a.get("avg_trig", 0)
    slcagr_a = wf_a.get("avg_slcagr", 0)
    dc_a = wf_a.get("avg_dc", 0)

    w("| Gate | trig/yr | trig削減 | sl_CAGR | α損失 | ΔCAGR | ΔCAGR回復 | density_penalty |")
    w("|---|---|---|---|---|---|---|---|")
    for case in GATE_SPECS:
        wf = gate_results[case]["wf"]
        trig   = wf.get("avg_trig", 0)
        slcagr = wf.get("avg_slcagr", 0)
        dc     = wf.get("avg_dc", 0)
        trig_red  = trig_a - trig
        sl_loss   = slcagr_a - slcagr
        dc_recov  = dc - dc_a
        penalty   = sl_loss / max(1.0, trig_red) if trig_red > 0 else float("nan")
        pen_str   = f"{penalty:.3f}" if not math.isnan(penalty) else "—"
        rec_str   = f"{dc_recov:+.2f}pp" if not math.isnan(dc_recov) else "—"
        w(f"| {case} | {trig:.1f} | {trig_red:+.1f} | {slcagr:+.1f}% "
          f"| {sl_loss:+.1f}% | {dc:+.2f}pp | {rec_str} | {pen_str} |")

    # ── 4. Density Analysis ───────────────────────────────────────────
    w("\n---\n## 4. Density / Regime Analysis\n")
    w("| Gate | regime_dep(std_Δ) | missed_α10 | missed_α20 | avg_overlap | idle_days/fold |")
    w("|---|---|---|---|---|---|")
    for case in GATE_SPECS:
        wf = gate_results[case]["wf"]
        rd  = _f(wf.get("regime_dep", float("nan")), ".3f")
        ma10 = (_f(wf.get("missed_alpha_10", float("nan")), ".2f") + "%"
                if not math.isnan(wf.get("missed_alpha_10", float("nan"))) else "—")
        ma20 = (_f(wf.get("missed_alpha_20", float("nan")), ".2f") + "%"
                if not math.isnan(wf.get("missed_alpha_20", float("nan"))) else "—")
        ov  = _f(wf.get("avg_overlap", float("nan")), ".3f")
        idl = f"{wf.get('avg_idle', 0):.1f}d"
        w(f"| {case} | {rd} | {ma10} | {ma20} | {ov} | {idl} |")

    # ── 5. Fold Detail ────────────────────────────────────────────────
    w("\n---\n## 5. Fold 詳細\n")
    w("| Gate | Fold | OOS年 | Regime | base | comb | ΔCAGR | ΔDD | sl_CAGR | trig/yr | reject% | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in GATE_SPECS:
        for fold, fm in zip(FOLDS, gate_results[case]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr) if topix_close is not None else "N/A"
            ok  = "✅" if fm.get("fold_pass") else "❌"
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['base_cagr']:+.1f}% "
              f"| {fm['comb_cagr']:+.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['trigger_per_year']:.1f} "
              f"| {fm['rejection_rate']:.1f}% "
              f"| {ok} |")

    # ── 6. Gate Audit ─────────────────────────────────────────────────
    w("\n---\n## 6. Gate Audit — Rejection & Missed Alpha\n")
    w("> missed_α = 入場禁止された候補の平均 fwd10d リターン\n")
    w("| Gate | reject/fold | reject_rate | missed_α10 | missed_α20 | gate_open% | idle_days |")
    w("|---|---|---|---|---|---|---|")
    for case in GATE_SPECS:
        folds = [fm for fm in gate_results[case]["folds"] if fm]
        avg_n_blocked = float(np.mean([f.get("n_blocked", 0) for f in folds])) if folds else 0.0
        wf = gate_results[case]["wf"]
        ma10 = (_f(wf.get("missed_alpha_10", float("nan")), ".2f") + "%"
                if not math.isnan(wf.get("missed_alpha_10", float("nan"))) else "—")
        ma20 = (_f(wf.get("missed_alpha_20", float("nan")), ".2f") + "%"
                if not math.isnan(wf.get("missed_alpha_20", float("nan"))) else "—")
        w(f"| {case} | {avg_n_blocked:.1f} "
          f"| {wf.get('avg_reject', 0):.1f}% "
          f"| {ma10} | {ma20} "
          f"| {wf.get('gate_open_rate', 100):.1f}% "
          f"| {wf.get('avg_idle', 0):.1f}d |")

    # ── 7. Hypothesis Tests ────────────────────────────────────────────
    w("\n---\n## 7. Hypothesis Analysis\n")

    # H1: density hypothesis
    w("### H1: Density仮説 — trigger低下とmissed_alpha増加が比例しない？\n")
    w("> 支持: trigger削減 → ΔCAGR改善 かつ missed_alpha が低い → 削除したのは低品質trigger\n")
    w("> 棄却: trigger削減 → ΔCAGR悪化 かつ missed_alpha が高い → 良質なtriggerを削除\n")
    w("| Gate | trig削減 | ΔCAGR変化 | missed_α10 | 判定 |")
    w("|---|---|---|---|---|")
    for case in GATE_SPECS:
        wf = gate_results[case]["wf"]
        trig_red = trig_a - wf.get("avg_trig", trig_a)
        dc_chg   = wf.get("avg_dc", 0) - dc_a
        ma10 = wf.get("missed_alpha_10", float("nan"))
        if math.isnan(ma10) or trig_red <= 0:
            verdict_h1 = "N/A"
        elif dc_chg > 0 and ma10 < 3.0:
            verdict_h1 = "✅ 仮説支持"
        elif dc_chg < 0 and ma10 > 5.0:
            verdict_h1 = "❌ 仮説棄却"
        else:
            verdict_h1 = "△ 中立"
        ma10_str = f"{ma10:.2f}%" if not math.isnan(ma10) else "—"
        w(f"| {case} | {trig_red:+.1f} | {dc_chg:+.2f}pp | {ma10_str} | {verdict_h1} |")

    # H2: Fold3 specific
    w("\n### H2: Fold3(2023) のみ改善 → regime interaction 仮説\n")
    w("| Gate | Fold3 ΔCAGR (baseline) | Fold3 ΔCAGR (gate) | Fold3改善? |")
    w("|---|---|---|---|")
    f3_base = None
    for case in GATE_SPECS:
        folds = gate_results[case]["folds"]
        f3    = folds[2] if len(folds) > 2 else {}   # Fold3 = index 2
        dc3   = f3.get("delta_cagr", float("nan")) if f3 else float("nan")
        if case == "A":
            f3_base = dc3
        if f3_base is not None and not math.isnan(dc3) and not math.isnan(f3_base):
            delta_f3 = dc3 - f3_base
            ok = "✅ 改善" if delta_f3 > 0.3 else ("❌ 悪化" if delta_f3 < -0.1 else "△ 変化なし")
        else:
            delta_f3 = float("nan"); ok = "—"
        base_str = f"{f3_base:+.2f}pp" if f3_base is not None and not math.isnan(f3_base) else "—"
        dc3_str  = f"{dc3:+.2f}pp" if not math.isnan(dc3) else "—"
        w(f"| {case} | {base_str} | {dc3_str} | {ok} |")

    # H3: overlap interaction
    w("\n### H3: overlap上昇時のみ失敗 → position interaction\n")
    w("| Gate | avg_overlap | WF失敗fold | 考察 |")
    w("|---|---|---|---|")
    for case in GATE_SPECS:
        wf    = gate_results[case]["wf"]
        olap  = wf.get("avg_overlap", 0)
        nfail = 5 - wf.get("n_pass", 0)
        comment = "高overlap = 失敗多い" if olap > 0.30 and nfail >= 2 else "相関不明瞭"
        w(f"| {case} | {olap:.3f} | {nfail}/5失敗 | {comment} |")

    # ── 8. CAGR recovered per suppressed trigger ──────────────────────
    w("\n---\n## 8. CAGR Recovered per Suppressed Trigger\n")
    w("> (Gate ΔCAGR − Baseline ΔCAGR) / max(1, trigger削減数)\n")
    w("> 正 = trigger抑制がCAGR改善に貢献 / 負 = 抑制がCAGR悪化を招く\n")
    w("| Gate | Gate ΔCAGR | Base ΔCAGR | ΔCAGR回復 | trigger抑制 | **CAGR/trigger** |")
    w("|---|---|---|---|---|---|")
    for case, row in cagr_per_trigger.items():
        dc_g  = row["dc_gate"]
        dc_b  = row["dc_base"]
        recov = dc_g - dc_b
        supp  = row["trig_suppressed"]
        per_t = row["cagr_per_trigger"]
        per_str = f"{per_t:+.4f}pp/trigger" if not math.isnan(per_t) else "—"
        w(f"| {case} | {dc_g:+.2f}pp | {dc_b:+.2f}pp "
          f"| {recov:+.2f}pp | {supp:+.1f} | **{per_str}** |")

    # ── 9. Failure Analysis ───────────────────────────────────────────
    w("\n---\n## 9. Failure Analysis\n")
    for case, spec in GATE_SPECS.items():
        wf   = gate_results[case]["wf"]
        fails = []
        if wf.get("n_pass", 0) < ADOPT_WF_MIN:
            fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF_MIN}")
        if wf.get("avg_dc", 0) <= ADOPT_DCAGR:
            fails.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp ≤ +{ADOPT_DCAGR}pp")
        if wf.get("avg_dd", 0) > ADOPT_DDD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp > +{ADOPT_DDD}pp")
        ar = wf.get("alpha_retention", float("nan"))
        if not math.isnan(ar) and ar < ADOPT_ALPHA_RET:
            fails.append(f"alpha_ret={ar:.1f}% < {ADOPT_ALPHA_RET}%")
        if not fails:
            w(f"\n**{case}** ✅ 全基準クリア")
        else:
            w(f"\n**{case}** ({spec['desc']}): REJECT — " + " / ".join(fails))

    # ── 10. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 10. Final Recommendation\n")
    w(f"## `{verdict}`\n")

    best_c  = max(GATE_SPECS, key=lambda c: gate_results[c]["wf"].get("avg_dc", 0))
    bwf     = gate_results[best_c]["wf"]
    w(f"**最良: Gate {best_c}** — {GATE_SPECS[best_c]['desc']}\n")
    w(f"- WF: {bwf.get('n_pass',0)}/5  ΔCAGR: {bwf.get('avg_dc',0):+.2f}pp  "
      f"ΔDD: {bwf.get('avg_dd',0):+.2f}pp")
    w(f"- sl_CAGR: {bwf.get('avg_slcagr',0):+.1f}%  "
      f"trig/yr: {bwf.get('avg_trig',0):.1f}  "
      f"reject: {bwf.get('avg_reject',0):.1f}%")

    ar_val = bwf.get("alpha_retention", float("nan"))
    if not math.isnan(ar_val):
        w(f"- alpha_retention: {ar_val:.1f}%")

    if adopted:
        w(f"\n実装必須 (ASK_FIRST):")
        w(f"- signal_bridge.py に density gate ロジック追加")
        w(f"- gate_state.json でトリガー履歴管理")
        w(f"- live shadow 検証 ≥ 30日")
    else:
        # Density hypothesis verdict
        any_improved_fold3 = any(
            gate_results[c]["folds"][2].get("delta_cagr", 0) >
            gate_results["A"]["folds"][2].get("delta_cagr", -99)
            for c in GATE_SPECS if c != "A" and len(gate_results[c]["folds"]) > 2 and gate_results[c]["folds"][2]
        )
        w(f"\n**density 仮説評価**: {'一部支持 → Fold3改善ケースあり' if any_improved_fold3 else '棄却 → gate によるFold3改善なし'}")
        w(f"\n**次ステップ候補**:")
        w(f"- density gate + BASE entry filter (Fold3 の 2023年 base low-CAGR 期間の特定)")
        w(f"- CAP=15% で ΔDD+ΔCAGR トレードオフ改善確認")
        w(f"- Fold3(2023) RCA: base portfolio の低リターン期がスリーブ alpha に影響するメカニズム分析")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3, "max_positions must be 3 (PARAMS_LOCKED)"

    print("=" * 68)
    print("  Study 8 Opportunity Density Gate WF")
    print("  Base: 2-slot 70/30 no-refill (Conc.Relief Case E)")
    print("  Gates A-G: rolling_trig / base_active / cash_idle / rolling_exp")
    print("=" * 68 + "\n")

    # ── Data load ────────────────────────────────────────────────────
    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: idx for idx, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(IS_START)) &
        (common_dates <= pd.Timestamp("2025-12-31"))
    ]
    n_dates = len(common_dates)
    print(f"  共通日数: {n_dates}, 銘柄数: {n_syms}")

    # ── Build matrices ───────────────────────────────────────────────
    print("[2/4] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

        rule  = SECTOR_STRATEGY.get(trade_syms.get(sym, ""), "fujiko")
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        st    = (MeanReversionStrategy(**MR_PARAMS) if rule == "mean_rev"
                 else FujikoStrategy(
                     min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                     rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                     mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                     use_turtle_entry=cfg.fujiko.use_turtle_entry))
        required = 252 + getattr(st, "mom_period", 21) + 2
        if hasattr(st, "precompute_signals") and len(df_src) >= required:
            sig_mat[:, si] = st.precompute_signals(df_src).to_numpy(dtype=np.int8)[row_idx]
            sig_ready[si]  = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                             dtype=np.float32, fill_value=1.0))

    mkt_ret1 = topix_ret20 = topix_ret60 = bear_arr = None
    if topix_close is not None:
        mkt_ret1    = _take(topix_close.pct_change(),    common_dates,
                             dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20), common_dates,
                             dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60), common_dates,
                             dtype=np.float32, fill_value=0.0)
        ma200    = topix_close.rolling(200, min_periods=100).mean()
        bear_arr = ((topix_close < ma200)
                    .reindex(pd.DatetimeIndex(common_dates), method="ffill")
                    .fillna(False).values.astype(bool))

    cross90_mat            = compute_cross_mat(rsr_mat, SLEEVE_RSR_LO)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)

    # ── WF Simulation ────────────────────────────────────────────────
    print(f"\n[3/4] WF Simulation ({len(GATE_SPECS)} gate cases × 5 folds)...\n")

    capital = float(cfg.portfolio.capital)
    gate_results: dict[str, dict] = {}
    sl_cagr_a: float | None = None

    for gate_id, gate_spec in GATE_SPECS.items():
        print(f"  [{gate_id}] {gate_spec['desc']}")
        sim = run_gate(
            open_mat, close_mat,
            sig_mat, sig_ready, rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, trade_syms, cfg,
            common_dates, gate_spec,
        )

        fold_metrics: list[dict] = []
        for fold in FOLDS:
            fm = compute_fold_metrics(sim, close_mat, common_dates, fold, capital)
            fold_metrics.append(fm)
            if fm:
                ok  = "✅" if fm.get("fold_pass") else "❌"
                rj  = f"rj={fm['rejection_rate']:.0f}%"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"Δ={fm['delta_cagr']:+.2f}pp  ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"sl={fm['sl_cagr']:+.1f}%  trig={fm['trigger_per_year']:.1f}  "
                      f"{rj}  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")

        wf = evaluate_wf(fold_metrics, sl_cagr_a)
        if gate_id == "A":
            sl_cagr_a = wf.get("avg_slcagr")

        gate_results[gate_id] = {"folds": fold_metrics, "wf": wf, "sim": sim}

        ar_str = (f"  α_ret={wf.get('alpha_retention',0):.1f}%"
                  if not math.isnan(wf.get("alpha_retention", float("nan"))) else "")
        ok_str = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"trig={wf.get('avg_trig',0):.1f}  "
              f"reject={wf.get('avg_reject',0):.1f}%{ar_str}  {ok_str}\n")

    # Re-evaluate non-A gates with Case A sl_CAGR baseline
    if sl_cagr_a is not None:
        for gate_id in list(GATE_SPECS)[1:]:
            gate_results[gate_id]["wf"] = evaluate_wf(gate_results[gate_id]["folds"], sl_cagr_a)

    # ── Summary ───────────────────────────────────────────────────────
    adopted = [c for c in GATE_SPECS if gate_results[c]["wf"].get("adopted")]
    print(f"{'='*68}")
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for case in GATE_SPECS:
        wf = gate_results[case]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        ar = wf.get("alpha_retention", float("nan"))
        ar_s = f"  α_ret={ar:.1f}%" if not math.isnan(ar) else ""
        print(f"  {ok} Gate {case}: WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"trig={wf.get('avg_trig',0):.1f}  "
              f"reject={wf.get('avg_reject',0):.1f}%{ar_s}")
    print(f"{'='*68}\n")

    # ── CAGR recovered per suppressed trigger ─────────────────────────
    wf_a     = gate_results["A"]["wf"]
    trig_a   = wf_a.get("avg_trig",  0)
    dc_a     = wf_a.get("avg_dc",    0)

    cagr_per_trigger: dict[str, dict] = {}
    print("  CAGR recovered per suppressed trigger:")
    for case in GATE_SPECS:
        wf      = gate_results[case]["wf"]
        dc_g    = wf.get("avg_dc",   0)
        trig_g  = wf.get("avg_trig", 0)
        supp    = trig_a - trig_g
        recov   = dc_g - dc_a
        per_t   = recov / max(1.0, abs(supp)) if supp != 0 else float("nan")
        cagr_per_trigger[case] = {
            "dc_gate":          dc_g,
            "dc_base":          dc_a,
            "trig_suppressed":  supp,
            "cagr_per_trigger": round(per_t, 4) if not math.isnan(per_t) else float("nan"),
        }
        per_str = f"{per_t:+.4f}pp/trigger" if not math.isnan(per_t) else "—"
        print(f"    Gate {case}: supp={supp:+.1f} recov={recov:+.2f}pp → {per_str}")

    # ── Report ───────────────────────────────────────────────────────
    print("\n[4/4] レポート生成...")
    write_report(
        gate_results, topix_close,
        REPORTS_DIR / "dedicated_alpha_density_gate.md",
        cagr_per_trigger,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
