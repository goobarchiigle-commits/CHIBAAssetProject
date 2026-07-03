"""
src/backtest/study8_cap_state_regime.py
Study 8 CAP State Regime WF — State Attribution

目的: adaptive CAP 状態変数が将来DD悪化を予測できるかを検証
      predictive_power 確認により adaptive研究継続可否を判定

Cases:
  A: 固定20% (anchor)
  B: rolling_dd > 7%
  C: rolling_dd>7% OR (rolling_dd>4% AND avg_hold>5d)
  D: rolling_dd>7% OR (rolling_dd>4% AND unrealized_pnl<0)
  E: rolling_dd>7% OR (rolling_dd>4% AND slot_hhi>0.6)
  F: rolling_dd>7% OR (rolling_dd>4% AND trailing_pf3<1.0)
  G: composite score>=2 → CAP=15% / recovery=score==0

Recovery: B-F=5営業日 / G=score==0 (即時)
Adoption: WF>=4/5, ΔCAGR>=+0.3pp, ΔDD<=+1.5pp, α_ret>=90%
Stop: (best ΔCAGR - anchor ΔCAGR) < 0.5pp OR state_precision < 55%

Run: cd C:/ai-trading && python src/backtest/study8_cap_state_regime.py
"""
from __future__ import annotations
import sys, math, time, warnings
from collections import deque
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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

REPORTS_DIR    = Path("reports")
IS_START       = "2018-01-01"
SLEEVE_RSR_LO  = 92.0
SLEEVE_RSR_HI  = 95.0
SLEEVE_D90_MAX = 5
SLEEVE_MAX     = 2
SL_WEIGHTS     = [0.70, 0.30]
NO_REFILL      = True
SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08
STALL_THR      = 0.15

CAP_HI         = 0.20
CAP_LO         = 0.15
RECOVERY_DAYS  = 5    # business days (Cases B-F)

# State thresholds
DD_BASE_THR    = 0.07   # primary rolling_dd trigger (B-F, G partial)
DD_SEC_THR     = 0.04   # secondary compound trigger (C-F)
DD_COMP_THR    = 0.05   # composite score DD component
HOLD_THR       = 5.0    # avg hold days (Case C)
COMP_HOLD_THR  = 4.0    # composite hold days
HHI_THR        = 0.60   # sleeve concentration (Case E, composite)
SCORE_ON       = 2      # composite: activate
SCORE_OFF      = 1      # composite: recover when score < SCORE_OFF

# State diagnostic thresholds
BAD_DROP       = 0.015  # 1.5% sl_eq drop in next 10d = "bad period"
BAD_HORIZON    = 10
LATENCY_HORIZON = 20

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

CASE_SPECS: dict[str, dict] = {
    "A": {"type": "fixed",      "desc": "固定20% (anchor)"},
    "B": {"type": "rolling_dd", "desc": "rolling_dd>7%"},
    "C": {"type": "dd_hold",    "desc": "rolling_dd + hold_days"},
    "D": {"type": "dd_upnl",    "desc": "rolling_dd + unrealized_pnl"},
    "E": {"type": "dd_hhi",     "desc": "rolling_dd + slot_concentration"},
    "F": {"type": "dd_pf3",     "desc": "rolling_dd + trailing_pf3"},
    "G": {"type": "composite",  "desc": "composite score>=2"},
}

ADOPT_WF = 4
ADOPT_DC = 0.3
ADOPT_DD = 1.5
ADOPT_AR = 90.0
STOP_MARGIN   = 0.5   # adaptive must beat anchor by ≥ 0.5pp
STOP_PREC     = 0.55  # state_precision minimum


# ─── helpers ──────────────────────────────────────────────────────────

def classify_state(s5, s20):
    if np.isnan(s5) or np.isnan(s20): return "UNKNOWN"
    if abs(s5) < STALL_THR: return "STALL"
    if s20 > 0:
        if s5 > s20:  return "EARLY_UP"
        if s5 > 0:    return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"

STATE_SCORE = {"EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1,
               "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0}

def cross_mat(rsr_mat, thr):
    n_d, n_s = rsr_mat.shape
    out  = np.zeros((n_d, n_s), dtype=np.int32)
    run  = np.zeros(n_s, dtype=np.int32)
    for i in range(n_d):
        out[i] = run
        ab = rsr_mat[i] >= thr
        run[ab] += 1; run[~ab] = 0
    return out

def slope_mats(rsr_mat):
    s5  = np.zeros_like(rsr_mat, dtype=np.float32)
    s20 = np.zeros_like(rsr_mat, dtype=np.float32)
    s5[5:]   = (rsr_mat[5:]  - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:] - rsr_mat[:-20]) / 20.0
    return s5, s20


# ─── AdaptiveCap ──────────────────────────────────────────────────────

class AdaptiveCap:
    """Multi-signal CAP state machine.

    step() accepts auxiliary state inputs (avg_hold, upnl, hhi, pf3)
    computed from the current sleeve portfolio state.
    """

    def __init__(self, spec: dict):
        self.ctype          = spec["type"]
        self.sl_eq_peak     = 0.0
        self.in_reduced     = False
        self.reduced_since  = -9999
        self.switches_total = 0
        # Full-simulation history
        self.cap_hist:   list[float] = []
        self.state_hist: list[bool]  = []
        self.rdd_hist:   list[float] = []
        self.score_hist: list[int]   = []

    def step(self, day_i: int, sl_eq: float,
             avg_hold: float = 0.0,
             upnl:     float = 0.0,
             hhi:      float = 0.0,
             pf3:      float = 1.0) -> float:
        # Update sleeve equity peak
        if sl_eq > self.sl_eq_peak:
            self.sl_eq_peak = sl_eq
        rdd = max(0.0, (self.sl_eq_peak - sl_eq) / max(1.0, self.sl_eq_peak))
        self.rdd_hist.append(rdd)

        if self.ctype == "fixed":
            self.cap_hist.append(CAP_HI)
            self.state_hist.append(False)
            self.score_hist.append(0)
            return CAP_HI

        # ── Compute trigger & score ───────────────────────────────────
        if self.ctype == "rolling_dd":
            trigger = rdd > DD_BASE_THR
            score   = int(trigger)

        elif self.ctype == "dd_hold":
            t1 = rdd > DD_BASE_THR
            t2 = (rdd > DD_SEC_THR) and (avg_hold > HOLD_THR)
            trigger = t1 or t2
            score   = int(t1) + int(t2)

        elif self.ctype == "dd_upnl":
            t1 = rdd > DD_BASE_THR
            t2 = (rdd > DD_SEC_THR) and (upnl < 0)
            trigger = t1 or t2
            score   = int(t1) + int(t2)

        elif self.ctype == "dd_hhi":
            t1 = rdd > DD_BASE_THR
            t2 = (rdd > DD_SEC_THR) and (hhi > HHI_THR)
            trigger = t1 or t2
            score   = int(t1) + int(t2)

        elif self.ctype == "dd_pf3":
            t1 = rdd > DD_BASE_THR
            t2 = (rdd > DD_SEC_THR) and (pf3 < 1.0)
            trigger = t1 or t2
            score   = int(t1) + int(t2)

        elif self.ctype == "composite":
            score = (
                (1 if rdd        > DD_COMP_THR    else 0) +
                (1 if avg_hold   > COMP_HOLD_THR  else 0) +
                (1 if upnl       < 0              else 0) +
                (1 if hhi        > HHI_THR        else 0) +
                (1 if pf3        < 1.0            else 0)
            )
            trigger = score >= SCORE_ON

        else:
            trigger = False
            score   = 0

        self.score_hist.append(score)

        # ── State machine ────────────────────────────────────────────
        if not self.in_reduced:
            if trigger:
                self.in_reduced    = True
                self.reduced_since = day_i
                self.switches_total += 1
        else:
            if self.ctype == "composite":
                # Score-based recovery (immediate when score == 0)
                if score < SCORE_OFF:
                    self.in_reduced = False
            else:
                # Time-based: 5 business days AND no trigger
                elapsed = day_i - self.reduced_since
                if not trigger and elapsed >= RECOVERY_DAYS:
                    self.in_reduced = False

        self.state_hist.append(self.in_reduced)
        cap = CAP_LO if self.in_reduced else CAP_HI
        self.cap_hist.append(cap)
        return cap


# ─── state inputs helper ──────────────────────────────────────────────

def get_state_inputs(sl_slots, recent_pnls: deque,
                     close_mat, sym_to_i, i: int):
    """Compute auxiliary state variables from current sleeve portfolio."""
    active = [sl_slots[k] for k in range(SLEEVE_MAX) if sl_slots[k] is not None]

    # avg hold days of current sleeve positions
    avg_hold = (float(np.mean([max(0, i - p.entry_idx) for p in active]))
                if active else 0.0)

    # total unrealized PnL of sleeve
    upnl = 0.0
    values: list[float] = []
    for p in active:
        si_ = sym_to_i.get(p.symbol, -1)
        if si_ < 0: continue
        cur = float(close_mat[i, si_])
        upnl  += (cur - p.entry_price) * p.qty
        values.append(cur * p.qty)

    # HHI of sleeve by current market value
    total_v = sum(values)
    if total_v > 0 and len(values) > 0:
        hhi = sum((v / total_v) ** 2 for v in values)
    elif len(values) == 1:
        hhi = 1.0
    else:
        hhi = 0.0

    # Trailing profit factor (last 3 completed sleeve sells)
    gp = sum(p for p in recent_pnls if p > 0)
    gl = abs(sum(p for p in recent_pnls if p <= 0))
    pf3 = gp / max(1.0, gl) if recent_pnls else 1.0

    return avg_hold, upnl, hhi, pf3


# ─── state diagnostics ────────────────────────────────────────────────

def compute_state_diagnostics(state_hist: list[bool],
                               sl_eq_arr:  list[float],
                               n_days:     int) -> dict:
    """Post-hoc diagnostic for state predictive power.
    bad_period[i] = True if sl_eq drops >1.5% in next 10 days.
    """
    sl = np.array(sl_eq_arr[:n_days], dtype=np.float64)
    state_on = np.array(state_hist[:n_days], dtype=bool)

    # Forward-looking bad periods
    bad = np.zeros(n_days, dtype=bool)
    for i in range(n_days - BAD_HORIZON):
        if sl[i] > 0:
            worst = float(np.min(sl[i+1 : i+1+BAD_HORIZON]))
            if worst / sl[i] < (1.0 - BAD_DROP):
                bad[i] = True

    # Precision: P(bad | state_on)
    tp       = int(np.sum(state_on & bad))
    n_on     = int(np.sum(state_on))
    precision = tp / max(1, n_on)

    # Recall: P(state_on | bad)
    n_bad    = int(np.sum(bad))
    recall   = tp / max(1, n_bad)

    # Latency: for each state activation, argmin of sl_eq in next 20d
    latencies: list[int] = []
    for j in range(1, n_days):
        if state_on[j] and not state_on[j-1]:
            end = min(n_days, j + LATENCY_HORIZON)
            if end > j + 1:
                worst_rel = int(np.argmin(sl[j:end]))
                latencies.append(worst_rel)

    avg_latency   = float(np.mean(latencies)) if latencies else 0.0
    n_activations = len(latencies)

    return {
        "precision":     round(precision,   3),
        "recall":        round(recall,      3),
        "avg_latency":   round(avg_latency, 1),
        "n_activations": n_activations,
        "n_bad":         n_bad,
        "n_on":          n_on,
    }


# ─── simulation ───────────────────────────────────────────────────────

def run_case(
    spec: dict,
    open_mat, close_mat, sig_mat, sig_ready,
    rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, trade_syms, cfg, common_dates,
) -> dict:
    capital      = float(cfg.portfolio.capital)
    base_max_pos = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)

    rc       = getattr(cfg, "risk_controls", None)
    MSEC     = float(rc.sector_cap)  if rc else 0.25
    gross_en = bool(getattr(rc, "gross_exposure_enabled", True))  if rc else True
    g_norm   = float(getattr(rc, "gross_cap_normal",       1.0))  if rc else 1.0
    g_dd5    = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    g_dd8    = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    acap         = AdaptiveCap(spec)
    recent_pnls  = deque(maxlen=3)  # last 3 completed sleeve sell PnLs

    base_cash    = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak    = float(capital)
    cb_active    = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    sl_slots: list[Position | None] = [None] * SLEEVE_MAX
    sl_cash  = float(capital * CAP_LO)
    sl_trades: list[dict] = []

    n_dates = len(common_dates)
    base_eq  = np.zeros(n_dates)
    sl_eq    = np.zeros(n_dates)
    base_inv = np.zeros(n_dates)
    sl_inv   = np.zeros(n_dates)
    cap_arr  = np.zeros(n_dates, dtype=np.float32)

    for i, date in enumerate(common_dates):
        ds = str(date.date())

        b_inv_v = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                      for s, p in base_pos.items())
        s_inv_v = sum(
            sl_slots[k].qty * float(close_mat[i, sym_to_i[sl_slots[k].symbol]])
            for k in range(SLEEVE_MAX) if sl_slots[k] is not None
        )
        b_eq_v = base_cash + b_inv_v
        s_eq_v = sl_cash  + s_inv_v
        base_eq[i]  = b_eq_v
        sl_eq[i]    = s_eq_v
        base_inv[i] = b_inv_v
        sl_inv[i]   = s_inv_v

        # Compute state inputs and get effective CAP
        avg_hold, upnl, hhi, pf3 = get_state_inputs(
            sl_slots, recent_pnls, close_mat, sym_to_i, i)
        eff_cap = acap.step(i, s_eq_v,
                            avg_hold=avg_hold, upnl=upnl, hhi=hhi, pf3=pf3)
        cap_arr[i] = eff_cap

        # Base portfolio circuit-breaker
        if b_eq_v > base_peak: base_peak = b_eq_v
        dd = (b_eq_v - base_peak) / base_peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0

        gross_cap = g_norm
        if gross_en and topix_ret20 is not None:
            r20 = float(topix_ret20[i]) if i < len(topix_ret20) else 0.0
            r60 = float(topix_ret60[i]) if i < len(topix_ret60) else 0.0
            if r20 < -0.05:   gross_cap = g_dd5
            elif r60 < -0.08: gross_cap = g_dd8

        is_bear   = bool(bear_arr[i]) if bear_arr is not None else False
        sec_eff   = 0.18 if is_bear else MSEC
        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n_dates: break
        ni = i + 1

        # ── Base portfolio sells ──────────────────────────────────────
        b_sell, b_buy = [], []
        for sym in active_syms:
            si_ = sym_to_i[sym]
            hld = sym in base_pos
            hix = (i - base_pos[sym].entry_idx) if hld else 0
            rv  = float(rsr_mat[i, si_])
            cl  = float(close_mat[i, si_])

            if mkt_shock and hld:
                if i > 0:
                    pc = float(close_mat[i-1, si_])
                    if pc > 0 and (cl/pc - 1.0) <= SHOCK_SYM_THR:
                        b_sell.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not hld: continue
            if hld and max_hold and hix > max_hold:
                b_sell.append((sym, "TIME_STOP")); continue
            if hld and rv < rsr_exit_thr and hix >= min_hold:
                b_sell.append((sym, "RSR_EXIT")); continue
            sig = int(sig_mat[i, si_]) if sig_ready[si_] else 0
            if sig == -1 and hld and hix >= min_hold:
                b_sell.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not hld:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si_]) < 0.5: continue
                b_buy.append((rv, sym))

        for sym, reason in b_sell:
            if sym not in base_pos: continue
            pos = base_pos[sym]
            sp  = float(open_mat[ni, sym_to_i[sym]])
            base_cash += pos.qty * sp * (1 - COST_ONE_WAY)
            base_trades.append({
                "side": "SELL", "symbol": sym,
                "pnl":  (sp - pos.entry_price) * pos.qty,
                "entry": pos.entry_price, "exit": sp, "qty": pos.qty,
                "entry_idx": pos.entry_idx, "exit_idx": i,
                "reason": reason, "date": ds,
            })
            del base_pos[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not cb_active and b_buy:
            b_buy.sort(key=lambda x: -x[0])
            for rv, sym in b_buy:
                si_ = sym_to_i[sym]
                if base_max_pos - len(base_pos) <= 0: break
                bp = float(open_mat[ni, si_])
                if bp <= 0: continue
                if not _sector_ok(sym, base_pos, close_mat, i, sym_to_i,
                                  trade_syms, capital, sec_eff): continue
                if gross_en:
                    cg = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                             for p in base_pos.values()) / max(1.0, capital)
                    if cg + bp * LOT / max(1.0, capital) > gross_cap: continue
                qty = int(capital / base_max_pos / bp / LOT) * LOT
                if qty <= 0 or qty * bp * (1 + COST_ONE_WAY) > base_cash: continue
                _execute_buy(sym, bp, qty, i, ni, trade_syms,
                             base_trades, base_pos, rv)
                base_cash -= qty * bp * (1 + COST_ONE_WAY)
                base_trades[-1]["date"] = ds

        # ── Sleeve exits ──────────────────────────────────────────────
        for k in range(SLEEVE_MAX):
            pos = sl_slots[k]
            if pos is None: continue
            si_     = sym_to_i[pos.symbol]
            rv      = float(rsr_mat[i, si_])
            cl      = float(close_mat[i, si_])
            do_exit = False; reason = ""
            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si_])
                if pc > 0 and (cl/pc - 1.0) <= SHOCK_SYM_THR:
                    do_exit = True; reason = "MARKET_SHOCK_EXIT"
            if not do_exit and rv < SLEEVE_RSR_LO:
                do_exit = True; reason = "RSR_EXIT_LO"
            if do_exit:
                sp  = float(open_mat[ni, si_])
                pnl = (sp - pos.entry_price) * pos.qty
                sl_cash += pos.qty * sp * (1 - COST_ONE_WAY)
                recent_pnls.append(pnl)
                sl_trades.append({
                    "side": "SELL", "symbol": pos.symbol, "pnl": pnl,
                    "entry": pos.entry_price, "exit": sp, "qty": pos.qty,
                    "entry_idx": pos.entry_idx, "exit_idx": i,
                    "hold_days": i - pos.entry_idx, "reason": reason,
                    "date": ds, "slot": k, "cap_at_entry": pos.rsr_at_entry,
                    "fwd10": np.nan, "fwd20": np.nan,
                })
                sl_slots[k] = None

        # ── Sleeve entries ────────────────────────────────────────────
        if not mkt_shock:
            n_occ = sum(1 for k in range(SLEEVE_MAX) if sl_slots[k] is not None)
            if NO_REFILL and 0 < n_occ < SLEEVE_MAX:
                pass
            elif n_occ < SLEEVE_MAX:
                sl_syms = {sl_slots[k].symbol
                           for k in range(SLEEVE_MAX) if sl_slots[k] is not None}
                cands = []
                for sym in active_syms:
                    if sym in base_pos or sym in sl_syms: continue
                    si_ = sym_to_i[sym]
                    rv  = float(rsr_mat[i, si_])
                    if not (SLEEVE_RSR_LO <= rv < SLEEVE_RSR_HI): continue
                    d90 = int(cross90_mat[i, si_])
                    if d90 > SLEEVE_D90_MAX: continue
                    if sym_active_mat is not None and float(sym_active_mat[i, si_]) < 0.5:
                        continue
                    st = classify_state(float(slope5_mat[i, si_]),
                                        float(slope20_mat[i, si_]))
                    cands.append((rv, d90, STATE_SCORE.get(st, 0), sym))
                if cands:
                    cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                    ci = 0
                    for k in range(SLEEVE_MAX):
                        if sl_slots[k] is not None: continue
                        if ci >= len(cands): break
                        rv, d90, _, sym = cands[ci]; ci += 1
                        si_ = sym_to_i[sym]
                        bp  = float(open_mat[ni, si_])
                        if bp <= 0: continue
                        target = b_eq_v * eff_cap * SL_WEIGHTS[k]
                        alloc  = min(sl_cash, target)
                        qty    = int(alloc / bp / LOT) * LOT
                        cost   = qty * bp * (1 + COST_ONE_WAY)
                        if qty <= 0 or cost > sl_cash: continue
                        sl_cash -= cost
                        sl_slots[k] = Position(sym, trade_syms.get(sym, ""),
                                               qty, bp, ni, eff_cap)
                        sl_trades.append({
                            "side": "BUY", "symbol": sym, "slot": k,
                            "weight": SL_WEIGHTS[k], "entry": bp, "qty": qty,
                            "cap_at_entry": eff_cap, "entry_idx": ni, "date": ds,
                        })

    # fwd returns for completed sleeve sells
    for t in sl_trades:
        if t["side"] != "SELL": continue
        si_ = sym_to_i.get(t["symbol"], -1)
        if si_ < 0: continue
        ei = t["exit_idx"]
        for fk, fn in [("fwd10", 10), ("fwd20", 20)]:
            if ei + fn < n_dates:
                c0 = float(close_mat[ei, si_])
                cn = float(close_mat[ei+fn, si_])
                t[fk] = round((cn/c0 - 1)*100, 2) if c0 > 0 and cn > 0 else float("nan")

    return {
        "base_eq":    base_eq,
        "sl_eq":      sl_eq,
        "base_inv":   base_inv,
        "sl_inv":     sl_inv,
        "base_trades": base_trades,
        "sl_trades":  sl_trades,
        "cap_arr":    cap_arr,
        "acap":       acap,
    }


# ─── fold metrics ─────────────────────────────────────────────────────

def compute_fold(sim, close_mat, common_dates, fold, capital,
                 base_sl_cagr: float | None) -> dict:
    oos_s, oos_e = fold["oos_start"], fold["oos_end"]
    ds_all   = [str(d.date()) for d in common_dates]
    oos_idx  = [i for i, ds in enumerate(ds_all) if oos_s <= ds <= oos_e]
    if not oos_idx: return {}
    si, ei   = oos_idx[0], oos_idx[-1] + 1
    n_days   = ei - si
    n_years  = n_days / 252.0
    oos_dates = list(common_dates[si:ei])

    base_eq_o = sim["base_eq"][si:ei].tolist()
    sl_eq_o   = sim["sl_eq"][si:ei].tolist()
    comb_eq_o = (sim["base_eq"] + sim["sl_eq"])[si:ei].tolist()
    b_inv_o   = sim["base_inv"][si:ei].tolist()
    s_inv_o   = sim["sl_inv"][si:ei].tolist()
    cap_oos   = sim["cap_arr"][si:ei]
    acap      = sim["acap"]

    base_t = [t for t in sim["base_trades"]
               if oos_s <= t.get("date", "") <= oos_e]
    sl_t   = [t for t in sim["sl_trades"]
               if oos_s <= t.get("date", "") <= oos_e]
    sl_sell = [t for t in sl_t if t["side"] == "SELL"]

    b_exp  = [bi / max(1, be) for bi, be in zip(b_inv_o, base_eq_o)]
    bm     = calc_metrics(base_eq_o, base_t, b_exp, base_eq_o[0], oos_dates)
    ce     = [(bi+si_) / max(1, be+se)
              for bi, si_, be, se in zip(b_inv_o, s_inv_o, base_eq_o, sl_eq_o)]
    cm     = calc_metrics(comb_eq_o, base_t + sl_t, ce, comb_eq_o[0], oos_dates)

    se     = [si_ / max(1, se) for si_, se in zip(s_inv_o, sl_eq_o)]
    slm    = calc_metrics(sl_eq_o, sl_t, se, sl_eq_o[0], oos_dates)

    delta_cagr = cm.get("cagr", 0) - bm.get("cagr", 0)
    delta_dd   = -(cm.get("max_dd", 0) - bm.get("max_dd", 0))
    sl_cagr    = slm.get("cagr", 0)

    n_trig  = len(sl_sell)
    avg_cap = float(np.mean(cap_oos)) if len(cap_oos) > 0 else CAP_LO

    # CAP state in OOS
    st_oos    = acap.state_hist[si:ei]
    days_lo   = int(sum(1 for s in st_oos if s))
    switches  = int(sum(1 for j in range(1, len(st_oos))
                        if st_oos[j] and not st_oos[j-1]))
    state_dur = days_lo / max(1, switches)
    trans_yr  = switches / max(0.01, n_years)

    # State diagnostics (post-hoc, OOS slice)
    diag = compute_state_diagnostics(st_oos, sl_eq_o, n_days)

    alpha_pres = (sl_cagr / base_sl_cagr
                  if base_sl_cagr and base_sl_cagr > 0 else float("nan"))

    fold_pass = (delta_cagr > ADOPT_DC and delta_dd <= ADOPT_DD)

    return {
        "base_cagr":   round(bm.get("cagr", 0), 2),
        "sl_cagr":     round(sl_cagr, 2),
        "comb_cagr":   round(cm.get("cagr", 0), 2),
        "delta_cagr":  round(delta_cagr, 2),
        "delta_dd":    round(delta_dd, 2),
        "n_trig":  n_trig,
        "avg_cap": round(avg_cap, 4),
        "days_lo": days_lo,
        "switches": switches,
        "state_duration":    round(state_dur, 1),
        "state_transition":  round(trans_yr, 1),
        "state_precision":   diag["precision"],
        "state_recall":      diag["recall"],
        "state_latency":     diag["avg_latency"],
        "n_activations":     diag["n_activations"],
        "n_bad":             diag["n_bad"],
        "alpha_preservation": round(alpha_pres, 3) if not math.isnan(alpha_pres) else float("nan"),
        "fold_pass": fold_pass,
    }


# ─── WF aggregate ─────────────────────────────────────────────────────

def agg_wf(folds: list[dict], sl_cagr_ref: float | None) -> dict:
    v = [f for f in folds if f]
    if not v: return {}
    n_pass    = sum(1 for f in v if f["fold_pass"])
    avg_dc    = float(np.mean([f["delta_cagr"]  for f in v]))
    avg_dd    = float(np.mean([f["delta_dd"]    for f in v]))
    avg_sl    = float(np.mean([f["sl_cagr"]     for f in v]))
    avg_cap   = float(np.mean([f["avg_cap"]     for f in v]))
    avg_sw    = float(np.mean([f["switches"]    for f in v]))
    avg_lo    = float(np.mean([f["days_lo"]     for f in v]))
    avg_prec  = float(np.mean([f["state_precision"]  for f in v]))
    avg_rec   = float(np.mean([f["state_recall"]     for f in v]))
    avg_lat   = float(np.mean([f["state_latency"]    for f in v]))
    avg_sd    = float(np.mean([f["state_duration"]   for f in v]))
    avg_tr    = float(np.mean([f["state_transition"] for f in v]))

    alpha_pres_vals = [f["alpha_preservation"] for f in v
                       if not math.isnan(f.get("alpha_preservation", float("nan")))]
    avg_ap = float(np.mean(alpha_pres_vals)) if alpha_pres_vals else float("nan")

    ar = (avg_sl / sl_cagr_ref * 100
          if sl_cagr_ref and sl_cagr_ref > 0 else float("nan"))
    adopted = (n_pass >= ADOPT_WF
               and avg_dc > ADOPT_DC
               and avg_dd <= ADOPT_DD
               and (math.isnan(ar) or ar >= ADOPT_AR))
    return {
        "n_pass": n_pass,
        "avg_dc": round(avg_dc, 2), "avg_dd": round(avg_dd, 2),
        "avg_sl": round(avg_sl, 2), "avg_cap": round(avg_cap, 4),
        "avg_sw": round(avg_sw, 1), "avg_lo":  round(avg_lo,  1),
        "avg_prec": round(avg_prec, 3), "avg_rec": round(avg_rec, 3),
        "avg_lat":  round(avg_lat,  1),
        "avg_sd":   round(avg_sd,   1),  # state_duration
        "avg_tr":   round(avg_tr,   1),  # state_transition (per yr)
        "avg_ap":   round(avg_ap,   3) if not math.isnan(avg_ap) else float("nan"),
        "alpha_ret": round(ar, 1) if not math.isnan(ar) else float("nan"),
        "adopted": adopted,
    }


# ─── report ───────────────────────────────────────────────────────────

def _f(v, fmt=".2f"):
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:{fmt}}"

def regime_label(tc, yr):
    if tc is None: return "N/A"
    s = tc[tc.index.year == yr]
    if len(s) < 2: return "N/A"
    r = float(s.iloc[-1] / s.iloc[0] - 1)
    return f"{'Bull' if r > 0 else 'Bear'} ({r*100:+.1f}%)"

def write_report(results, topix_close, output_path,
                 stop_met: bool, stop_reason: str, sl_cagr_ref: float | None):
    L = []; w = L.append
    adopted = [c for c in CASE_SPECS if results[c]["wf"].get("adopted")]

    w("# Study8 CAP State Regime WF — State Attribution")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  解析専用 / 実装変更禁止")
    w(f"\n固定: 2-slot 70/30 no-refill / RSR[92,95) d90≤5 / EXIT RSR<90 / "
      f"CAP_HI={CAP_HI*100:.0f}% / CAP_LO={CAP_LO*100:.0f}%")
    w(f"\n復帰条件: B-F=5営業日 / G=composite score==0")
    w(f"\n採用条件: WF≥{ADOPT_WF}/5, ΔCAGR>+{ADOPT_DC}pp, ΔDD≤+{ADOPT_DD}pp, "
      f"α_ret≥{ADOPT_AR}%")
    w(f"\n停止: (best ΔCAGR - anchor ΔCAGR) < {STOP_MARGIN}pp  OR  "
      f"state_precision < {STOP_PREC*100:.0f}%\n")

    if stop_met:
        w(f"🛑 **停止条件発動**: {stop_reason}")
        w("**→ adaptive CAP 研究終了**\n")

    w(f"**採用**: {len(adopted)}件  "
      f"| 判定: **{'✅ ADOPT' if adopted else '🛑 STOP' if stop_met else '🔬 CONTINUE'}**\n")

    # ── 1. WF Summary ────────────────────────────────────────────────
    w("---\n## 1. WF サマリ\n")
    w("| Case | 説明 | WF | ΔCAGR | ΔDD | sl_CAGR | α_ret | avg_cap | precision | recall | latency | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid, spec in CASE_SPECS.items():
        wf = results[cid]["wf"]
        ar = (_f(wf.get("alpha_ret", float("nan")), ".1f") + "%"
              if not math.isnan(wf.get("alpha_ret", float("nan"))) else "—")
        ok = "**✅**" if wf.get("adopted") else "❌"
        w(f"| {cid} | {spec['desc']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_dc',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_sl',0):+.1f}% "
          f"| {ar} "
          f"| {wf.get('avg_cap',0)*100:.1f}% "
          f"| {wf.get('avg_prec',0)*100:.1f}% "
          f"| {wf.get('avg_rec',0)*100:.1f}% "
          f"| {wf.get('avg_lat',0):.1f}d "
          f"| {ok} |")

    # ── 2. State Attribution Detail ───────────────────────────────────
    w("\n---\n## 2. State Attribution Detail (OOS avg)\n")
    w("| Case | state_precision | state_recall | latency(d) | state_duration(d) | "
      "trans/yr | alpha_pres | n_acts |")
    w("|---|---|---|---|---|---|---|---|")
    for cid in CASE_SPECS:
        wf = results[cid]["wf"]
        ap = _f(wf.get("avg_ap", float("nan")), ".3f")
        w(f"| {cid} "
          f"| **{wf.get('avg_prec',0)*100:.1f}%** "
          f"| {wf.get('avg_rec',0)*100:.1f}% "
          f"| {wf.get('avg_lat',0):.1f} "
          f"| {wf.get('avg_sd',0):.1f} "
          f"| {wf.get('avg_tr',0):.1f} "
          f"| {ap} "
          f"| {sum(f.get('n_activations',0) for f in results[cid]['folds'] if f)} |")

    # ── 3. Fold Detail ───────────────────────────────────────────────
    w("\n---\n## 3. Fold 詳細\n")
    w("| Case | Fold | 年 | Regime | ΔCAGR | ΔDD | sl_CAGR | cap | prec | recall | lat | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in CASE_SPECS:
        for fold, fm in zip(FOLDS, results[cid]["folds"]):
            if not fm: continue
            yr = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm["fold_pass"] else "❌"
            w(f"| {cid} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['avg_cap']*100:.0f}% "
              f"| {fm['state_precision']*100:.0f}% "
              f"| {fm['state_recall']*100:.0f}% "
              f"| {fm['state_latency']:.1f}d "
              f"| {ok} |")

    # ── 4. Predictive Power Matrix ────────────────────────────────────
    w("\n---\n## 4. Predictive Power Matrix\n")
    w("> state_precision: P(bad_period | state_on)  — state が活性の時に実際にDD悪化が起きる確率")
    w("> state_recall: P(state_on | bad_period)  — DD悪化期間にstateが活性だった割合")
    w(f"> bad_period定義: 次{BAD_HORIZON}日以内にsleeve equity が{BAD_DROP*100:.1f}%以上下落\n")
    w("| Case | 説明 | avg precision | avg recall | avg latency | predictive? |")
    w("|---|---|---|---|---|---|")
    for cid, spec in CASE_SPECS.items():
        wf = results[cid]["wf"]
        prec = wf.get("avg_prec", 0)
        rec  = wf.get("avg_rec",  0)
        lat  = wf.get("avg_lat",  0)
        pred = prec >= STOP_PREC and lat > 0
        label = "✅ 有効" if pred else ("🔶 部分的" if prec >= 0.45 else "❌ 無効")
        w(f"| {cid} | {spec['desc']} "
          f"| **{prec*100:.1f}%** "
          f"| {rec*100:.1f}% "
          f"| {lat:.1f}d "
          f"| {label} |")

    # ── 5. Fold3 / Fold5 Barrier ─────────────────────────────────────
    w("\n---\n## 5. WF Barrier Analysis (Fold3/Fold5)\n")
    for fidx, fname in [(2, "Fold3 (2023)"), (4, "Fold5 (2025)")]:
        w(f"\n### {fname}\n")
        w("| Case | ΔCAGR | ΔDD | sl_CAGR | cap | precision | latency | pass |")
        w("|---|---|---|---|---|---|---|---|")
        for cid in CASE_SPECS:
            fm = results[cid]["folds"][fidx]
            if not fm: continue
            ok = "✅" if fm["fold_pass"] else "❌"
            w(f"| {cid} {ok} "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['avg_cap']*100:.0f}% "
              f"| {fm['state_precision']*100:.0f}% "
              f"| {fm['state_latency']:.1f}d |")

    # ── 6. Stop Condition ─────────────────────────────────────────────
    w("\n---\n## 6. Stop Condition Assessment\n")
    wf_a = results["A"]["wf"]
    adapt = [c for c in CASE_SPECS if CASE_SPECS[c]["type"] != "fixed"]
    best_c = max(adapt, key=lambda c: results[c]["wf"].get("avg_dc", -99))
    best_dc = results[best_c]["wf"].get("avg_dc", -99)
    margin = best_dc - wf_a.get("avg_dc", 0)
    best_prec = results[best_c]["wf"].get("avg_prec", 0)

    w(f"Case A (固定20%): WF={wf_a.get('n_pass',0)}/5  "
      f"ΔCAGR={wf_a.get('avg_dc',0):+.2f}pp  ΔDD={wf_a.get('avg_dd',0):+.2f}pp\n")
    w(f"Best adaptive: Case {best_c} — ΔCAGR={best_dc:+.2f}pp  "
      f"margin over anchor={margin:+.2f}pp  precision={best_prec*100:.1f}%\n")
    w("| 停止条件 | 判定値 | 閾値 | 発動? |")
    w("|---|---|---|---|")
    w(f"| (best ΔCAGR - anchor ΔCAGR) < {STOP_MARGIN}pp "
      f"| {margin:+.2f}pp | {STOP_MARGIN}pp | "
      f"{'**⚠ 発動**' if margin < STOP_MARGIN else '未発動'} |")
    w(f"| state_precision < {STOP_PREC*100:.0f}% "
      f"| {best_prec*100:.1f}% | {STOP_PREC*100:.0f}% | "
      f"{'**⚠ 発動**' if best_prec < STOP_PREC else '未発動'} |")
    w(f"\n**結果**: {'**停止条件発動 → adaptive CAP研究終了**' if stop_met else '未発動 → 継続可'}")

    # ── 7. Failure Analysis ───────────────────────────────────────────
    w("\n---\n## 7. Failure Analysis\n")
    for cid, spec in CASE_SPECS.items():
        wf    = results[cid]["wf"]
        fails = []
        if wf.get("n_pass", 0) < ADOPT_WF:
            fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF}")
        if wf.get("avg_dc", 0) <= ADOPT_DC:
            fails.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp ≤ {ADOPT_DC}")
        if wf.get("avg_dd", 0) > ADOPT_DD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp > {ADOPT_DD}")
        ar = wf.get("alpha_ret", float("nan"))
        if not math.isnan(ar) and ar < ADOPT_AR:
            fails.append(f"α_ret={ar:.1f}% < {ADOPT_AR}%")
        if not fails:
            w(f"\n**{cid}** ✅ 全基準クリア")
        else:
            w(f"\n**{cid}** REJECT: " + " / ".join(fails))

    # ── 8. Final Output ───────────────────────────────────────────────
    w("\n---\n## 8. Final Output\n")

    # best_state
    w("### best_state\n")
    adapt_rank = sorted(adapt,
                        key=lambda c: (results[c]["wf"].get("n_pass", 0),
                                       results[c]["wf"].get("avg_dc", 0),
                                       -results[c]["wf"].get("avg_dd", 99)))
    best_state = adapt_rank[-1]
    bwf = results[best_state]["wf"]
    w(f"| 項目 | 値 |")
    w("|---|---|")
    w(f"| **best case** | **{best_state}** — {CASE_SPECS[best_state]['desc']} |")
    w(f"| WF | {bwf.get('n_pass',0)}/5 |")
    w(f"| ΔCAGR | {bwf.get('avg_dc',0):+.2f}pp |")
    w(f"| ΔDD | {bwf.get('avg_dd',0):+.2f}pp |")
    w(f"| avg_cap | {bwf.get('avg_cap',0)*100:.1f}% |")
    w(f"| state_precision | {bwf.get('avg_prec',0)*100:.1f}% |")
    w(f"| state_recall | {bwf.get('avg_rec',0)*100:.1f}% |")
    w(f"| state_latency | {bwf.get('avg_lat',0):.1f}d |")
    w(f"| alpha_preservation | {_f(bwf.get('avg_ap',float('nan')),'.3f')} |")

    # predictive_power summary
    w("\n### predictive_power\n")
    all_prec   = {c: results[c]["wf"].get("avg_prec", 0) for c in adapt}
    all_recall = {c: results[c]["wf"].get("avg_rec",  0) for c in adapt}
    best_pred  = max(adapt, key=lambda c: all_prec[c])
    w(f"| 指標 | Case {best_pred} | 全adaptive平均 |")
    w("|---|---|---|")
    w(f"| state_precision "
      f"| {all_prec[best_pred]*100:.1f}% "
      f"| {float(np.mean(list(all_prec.values())))*100:.1f}% |")
    w(f"| state_recall "
      f"| {all_recall[best_pred]*100:.1f}% "
      f"| {float(np.mean(list(all_recall.values())))*100:.1f}% |")
    w(f"| threshold | {STOP_PREC*100:.0f}% | — |")
    pp_verdict = (f"**✅ 予測力あり** (precision={all_prec[best_pred]*100:.1f}% ≥ {STOP_PREC*100:.0f}%)"
                  if all_prec[best_pred] >= STOP_PREC
                  else f"**❌ 予測力不足** (best precision={all_prec[best_pred]*100:.1f}% < {STOP_PREC*100:.0f}%)")
    w(f"\n予測力判定: {pp_verdict}")

    # recommend next study
    w("\n### recommend next study\n")
    if stop_met:
        w("**🛑 停止条件発動 → adaptive CAP研究終了**\n")
        if stop_reason.startswith("precision"):
            w("**根本原因**: rolling_dd は将来DD悪化の予測子として機能しない")
            w("- state_precision < 55% = 状態が活性でも実際のDD悪化は半数以下")
            w("- 信号はノイズ。CAP切替は無益 or 有害\n")
            w("**次研究推奨**: adaptive CAP軸を廃棄し、別軸に転換")
            w("- 候補A: RSR exit 閾値動的化 (RSR<90 → RSR<85 in high_dd期)")
            w("  スクリプト: `src/backtest/study8_exit_dynamic.py`")
            w("- 候補B: Fold3 barrier 専用解析 (base_CAGR=+2.1% 根本原因特定)")
            w("  スクリプト: `src/backtest/study8_fold3_analysis.py`")
        else:
            w("**根本原因**: best adaptive ΔCAGR が anchor ΔCAGR + 0.5pp 未満")
            w("- CAP切替のメリット < コスト = CAP研究に追加改善余地なし\n")
            w("**次研究推奨**: CAP sweep 終結、最大CAP固定値を採用")
            w("- スクリプト: `src/backtest/study8_cap_final.py`")
            w("- 固定 CAP=20% (Case A) をそのまま production に組み込み")
            w("- 変更禁止: ENTRY/EXIT/GATE/PARAMS_LOCKED")
    elif adopted:
        best_ad = adopted[0]
        spec_a  = CASE_SPECS[best_ad]
        w(f"**✅ 採用確定: Case {best_ad}** — {spec_a['desc']}\n")
        w(f"- signal_bridge.py に adaptive CAP ロジック追加")
        w(f"- recovery_days={RECOVERY_DAYS}d / CAP_HI=20% / CAP_LO=15%")
        w(f"- cap_state.json で状態管理 (idempotent)")
        w(f"- 変更禁止: ENTRY/EXIT/GATE/PARAMS_LOCKED")
    else:
        # WF 3/5 border case
        near = [c for c in adapt
                if results[c]["wf"].get("n_pass", 0) >= 3
                and results[c]["wf"].get("avg_dd", 99) <= 2.0]
        if near:
            best_n   = max(near, key=lambda c: results[c]["wf"].get("avg_dc", 0))
            best_prec_n = results[best_n]["wf"].get("avg_prec", 0)
            w(f"**WF 3/5 ボーダー残存**  予測力={best_prec_n*100:.1f}%\n")
            w(f"最良 adaptive: Case {best_n} ({CASE_SPECS[best_n]['desc']})")
            if best_prec_n >= STOP_PREC:
                w(f"\n**次研究推奨**: CAP_LO sweep (予測力は確認済み、CAP_LO最適値探索)")
                w(f"- スクリプト: `src/backtest/study8_cap_sweep.py`")
                w(f"- CAP_LO=[10,11,12,13,14,15]% × WF5fold")
                w(f"- 目標: Fold5 ΔDD≤+1.5pp AND ΔCAGR>+0.3pp の同時達成")
            else:
                w(f"\n**次研究推奨**: state 変数の改善探索 (精度={best_prec_n*100:.1f}%は境界)")
                w(f"- 試案: RSR slope decline + DD 複合条件")
                w(f"- スクリプト: `src/backtest/study8_cap_state_v2.py`")
        else:
            w("**全Case WF失敗・precision低水準 → adaptive CAP軸の限界**\n")
            w("次研究推奨: ENTRY/EXIT 改善軸に転換")
            w("- Study9: RSR exit 閾値動的化 or Fold3 barrier 分離解析")

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─── main ─────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Study8 CAP State Regime WF — State Attribution")
    print(f"  CAP_HI={CAP_HI*100:.0f}% / CAP_LO={CAP_LO*100:.0f}% "
          f"/ recovery_BF={RECOVERY_DAYS}d / recovery_G=score==0")
    print("=" * 68 + "\n")

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
        rule   = SECTOR_STRATEGY.get(trade_syms.get(sym, ""), "fujiko")
        rsr_s  = rsr_df[sym] if sym in rsr_df.columns else None
        st = (MeanReversionStrategy(**MR_PARAMS) if rule == "mean_rev"
              else FujikoStrategy(
                  min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                  rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                  mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                  use_turtle_entry=cfg.fujiko.use_turtle_entry))
        req = 252 + getattr(st, "mom_period", 21) + 2
        if hasattr(st, "precompute_signals") and len(df_src) >= req:
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
        topix_ret20 = _take(topix_close.pct_change(20),  common_dates,
                            dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60),  common_dates,
                            dtype=np.float32, fill_value=0.0)
        ma200    = topix_close.rolling(200, min_periods=100).mean()
        bear_arr = ((topix_close < ma200)
                    .reindex(pd.DatetimeIndex(common_dates), method="ffill")
                    .fillna(False).values.astype(bool))

    cross90_mat            = cross_mat(rsr_mat, SLEEVE_RSR_LO)
    slope5_mat, slope20_mat = slope_mats(rsr_mat)

    print(f"\n[3/4] WF Simulation ({len(CASE_SPECS)} cases × 5 folds)...\n")
    capital       = float(cfg.portfolio.capital)
    all_results: dict[str, dict] = {}
    sl_cagr_ref: float | None = None  # Case A sl_CAGR → alpha_ret denominator

    for case_id, spec in CASE_SPECS.items():
        print(f"  [{case_id}] {spec['desc']}")
        sim = run_case(
            spec, open_mat, close_mat, sig_mat, sig_ready,
            rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, trade_syms, cfg, common_dates,
        )
        folds = []
        for fold in FOLDS:
            fm = compute_fold(sim, close_mat, common_dates, fold, capital, sl_cagr_ref)
            folds.append(fm)
            if fm:
                prec_s = f"prec={fm['state_precision']*100:.0f}%  lat={fm['state_latency']:.1f}d"
                ok = "✅" if fm["fold_pass"] else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"Δ={fm['delta_cagr']:+.2f}pp  ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"sl={fm['sl_cagr']:+.1f}%  cap={fm['avg_cap']*100:.0f}%  "
                      f"{prec_s}  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")

        wf = agg_wf(folds, sl_cagr_ref)
        if case_id == "A":
            sl_cagr_ref = wf.get("avg_sl")  # use Case A sl_CAGR as reference

        all_results[case_id] = {"folds": folds, "wf": wf, "sim": sim}
        ar = wf.get("alpha_ret", float("nan"))
        ar_s = f"  α_ret={ar:.1f}%" if not math.isnan(ar) else ""
        ok = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"prec={wf.get('avg_prec',0)*100:.1f}%  "
              f"lat={wf.get('avg_lat',0):.1f}d{ar_s}  {ok}\n")

    # Re-compute with self-reference (Case A as denominator)
    if sl_cagr_ref is not None:
        for cid in list(CASE_SPECS)[1:]:
            all_results[cid]["wf"] = agg_wf(all_results[cid]["folds"], sl_cagr_ref)

    # ── Stop condition ────────────────────────────────────────────────
    wf_a    = all_results["A"]["wf"]
    adapt   = [c for c in CASE_SPECS if CASE_SPECS[c]["type"] != "fixed"]
    best_c  = max(adapt, key=lambda c: all_results[c]["wf"].get("avg_dc", -99))
    best_dc = all_results[best_c]["wf"].get("avg_dc", -99)
    margin  = best_dc - wf_a.get("avg_dc", 0)
    best_prec = all_results[best_c]["wf"].get("avg_prec", 0)

    stop_margin_hit = margin < STOP_MARGIN
    stop_prec_hit   = best_prec < STOP_PREC
    stop_met = stop_margin_hit or stop_prec_hit

    stop_reason = ""
    if stop_margin_hit and stop_prec_hit:
        stop_reason = (f"margin={margin:+.2f}pp < {STOP_MARGIN}pp AND "
                       f"precision={best_prec*100:.1f}% < {STOP_PREC*100:.0f}%")
    elif stop_margin_hit:
        stop_reason = f"margin={margin:+.2f}pp < {STOP_MARGIN}pp"
    elif stop_prec_hit:
        stop_reason = f"precision={best_prec*100:.1f}% < {STOP_PREC*100:.0f}%"

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 68)
    adopted = [c for c in CASE_SPECS if all_results[c]["wf"].get("adopted")]
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for cid in CASE_SPECS:
        wf = all_results[cid]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        ar = wf.get("alpha_ret", float("nan"))
        ar_s = f"  α_ret={ar:.1f}%" if not math.isnan(ar) else ""
        prec = wf.get("avg_prec", 0)
        w_lat = wf.get("avg_lat", 0)
        print(f"  {ok} [{cid}] WF={wf.get('n_pass',0)}/5  "
              f"Δ={wf.get('avg_dc',0):+.2f}pp  DD={wf.get('avg_dd',0):+.2f}pp  "
              f"prec={prec*100:.1f}%  lat={w_lat:.1f}d{ar_s}")

    print(f"\n  停止条件: {'発動 (' + stop_reason + ')' if stop_met else '未発動'}")
    print(f"  best adaptive: Case {best_c}  margin={margin:+.2f}pp  prec={best_prec*100:.1f}%")
    print("=" * 68 + "\n")

    print("[4/4] レポート生成...")
    write_report(all_results, topix_close,
                 REPORTS_DIR / "study8_cap_state_regime.md",
                 stop_met, stop_reason, sl_cagr_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
