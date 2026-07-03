"""
src/backtest/study8_adaptive_risk_budget.py
Study8 Adaptive Risk Budget WF — State-Driven CAP

固定: 2-slot 70/30 no-refill (Concentration Relief Case E)
     ENTRY RSR[92,95) d90<=5 / EXIT RSR<90

Cases:
  A  fixed 20%  (baseline)
  B  fixed 15%  (anchor)
  C  rolling_pnl: last5_trade_pnl<0 → CAP=10% else 20%
  D  rolling_dd:  sleeve_rolling_dd>5% → CAP=10% else 20%
  E  loss_streak: consecutive_loss>=2 → CAP=10% else 20%
  F  composite:   score>=2 → CAP=10% else 20%
     (each of: last5<0, rolling_dd>5%, streak>=2 → +1)

Recovery (adaptive only): rolling_pnl>0 OR 10 business days elapsed

採用: WF>=4/5, ΔCAGR>=+0.3pp, ΔDD<=+1.5pp, alpha_ret>=85%
停止: fixed 15% (Case B) が adaptive 全群を上回れば adaptive 研究終了

Run: cd C:/ai-trading && python src/backtest/study8_adaptive_risk_budget.py
"""
from __future__ import annotations
import sys, math, time, warnings
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

CAP_HIGH = 0.20
CAP_LOW  = 0.10
CAP_B    = 0.15
DD_THR   = 0.05   # sleeve rolling_dd threshold
STREAK_THR = 2    # consecutive losses
RECOVERY_DAYS = 10

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

CASE_SPECS = {
    "A": {"type": "fixed",       "cap": CAP_HIGH, "desc": "固定 20% (baseline)"},
    "B": {"type": "fixed",       "cap": CAP_B,    "desc": "固定 15% (anchor)"},
    "C": {"type": "rolling_pnl", "cap_hi": CAP_HIGH, "cap_lo": CAP_LOW,
          "desc": "rolling_pnl: last5<0 → 10%"},
    "D": {"type": "rolling_dd",  "cap_hi": CAP_HIGH, "cap_lo": CAP_LOW,
          "desc": "rolling_dd: dd>5% → 10%"},
    "E": {"type": "loss_streak", "cap_hi": CAP_HIGH, "cap_lo": CAP_LOW,
          "desc": "loss_streak: streak>=2 → 10%"},
    "F": {"type": "composite",   "cap_hi": CAP_HIGH, "cap_lo": CAP_LOW,
          "desc": "composite: score>=2 → 10%"},
}

ADOPT_WF   = 4
ADOPT_DC   = 0.3
ADOPT_DD   = 1.5
ADOPT_AR   = 85.0


# ─── helpers ─────────────────────────────────────────────────────────

def classify_state(s5, s20):
    if np.isnan(s5) or np.isnan(s20): return "UNKNOWN"
    if abs(s5) < STALL_THR: return "STALL"
    if s20 > 0:
        if s5 > s20: return "EARLY_UP"
        if s5 > 0:   return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"

STATE_SCORE = {"EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1, "DOWN": 0,
               "EARLY_ROLL": -1, "UNKNOWN": 0}

def cross_mat(rsr_mat, thr):
    n_d, n_s = rsr_mat.shape
    out = np.zeros((n_d, n_s), dtype=np.int32)
    run = np.zeros(n_s, dtype=np.int32)
    for i in range(n_d):
        out[i] = run
        ab = rsr_mat[i] >= thr
        run[ab] += 1; run[~ab] = 0
    return out

def slopes(rsr_mat):
    s5  = np.zeros_like(rsr_mat, dtype=np.float32)
    s20 = np.zeros_like(rsr_mat, dtype=np.float32)
    s5[5:]   = (rsr_mat[5:]  - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:] - rsr_mat[:-20]) / 20.0
    return s5, s20


# ─── adaptive CAP state machine ──────────────────────────────────────

class AdaptiveCap:
    """Determines effective sleeve CAP based on sleeve state signals.
    No lookahead — all signals computed from completed trades only."""

    def __init__(self, spec: dict):
        self.spec          = spec
        self.ctype         = spec["type"]
        self.cap_hi        = spec.get("cap", spec.get("cap_hi", CAP_HIGH))
        self.cap_lo        = spec.get("cap_lo", CAP_LOW)
        # state
        self.trade_pnls: deque[float] = deque(maxlen=5)
        self.consec_loss   = 0
        self.sl_eq_peak    = 0.0
        self.reduced_since = -999   # day index when CAP was reduced
        self.in_reduced    = False
        self.switches      = 0
        self.cap_history: list[float] = []

    def record_trade(self, pnl: float) -> None:
        self.trade_pnls.append(pnl)
        if pnl < 0:
            self.consec_loss += 1
        else:
            self.consec_loss = 0

    def update_eq_peak(self, sl_eq: float) -> None:
        if sl_eq > self.sl_eq_peak:
            self.sl_eq_peak = sl_eq

    def rolling_dd(self, sl_eq: float) -> float:
        if self.sl_eq_peak <= 0: return 0.0
        return max(0.0, (self.sl_eq_peak - sl_eq) / self.sl_eq_peak)

    def _recovery_met(self, day_i: int) -> bool:
        pnl_ok = len(self.trade_pnls) > 0 and sum(self.trade_pnls) > 0
        days_ok = (day_i - self.reduced_since) >= RECOVERY_DAYS
        return pnl_ok or days_ok

    def get_cap(self, day_i: int, sl_eq: float) -> float:
        if self.ctype == "fixed":
            cap = self.cap_hi
            self.cap_history.append(cap)
            return cap

        self.update_eq_peak(sl_eq)
        rdd = self.rolling_dd(sl_eq)
        last5_pnl = sum(self.trade_pnls)
        streak = self.consec_loss

        # Compute trigger
        if self.ctype == "rolling_pnl":
            trigger = (last5_pnl < 0 and len(self.trade_pnls) >= 1)
        elif self.ctype == "rolling_dd":
            trigger = (rdd > DD_THR)
        elif self.ctype == "loss_streak":
            trigger = (streak >= STREAK_THR)
        elif self.ctype == "composite":
            score = int(last5_pnl < 0 and len(self.trade_pnls) >= 1) \
                  + int(rdd > DD_THR) \
                  + int(streak >= STREAK_THR)
            trigger = (score >= 2)
        else:
            trigger = False

        # State transitions
        if not self.in_reduced and trigger:
            self.in_reduced    = True
            self.reduced_since = day_i
            self.switches     += 1
        elif self.in_reduced and not trigger and self._recovery_met(day_i):
            self.in_reduced = False

        cap = self.cap_lo if self.in_reduced else self.cap_hi
        self.cap_history.append(cap)
        return cap

    def avg_cap(self) -> float:
        return float(np.mean(self.cap_history)) if self.cap_history else self.cap_hi

    def state_ratio(self) -> float:
        if not self.cap_history: return 0.0
        return sum(1 for c in self.cap_history if c < self.cap_hi) / len(self.cap_history)


# ─── simulation ──────────────────────────────────────────────────────

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

    rc           = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W = float(rc.sector_cap)  if rc else 0.25
    gross_en     = bool(getattr(rc, "gross_exposure_enabled", True))  if rc else True
    g_normal     = float(getattr(rc, "gross_cap_normal",       1.0))  if rc else 1.0
    g_dd5        = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    g_dd8        = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    acap = AdaptiveCap(spec)

    base_cash  = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak  = float(capital)
    cb_active  = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    sl_slots: list[Position | None] = [None] * SLEEVE_MAX
    sl_cash  = float(capital * acap.cap_hi)  # initial sleeve budget
    sl_trades: list[dict] = []

    n_dates  = len(common_dates)
    base_eq  = np.zeros(n_dates)
    sl_eq    = np.zeros(n_dates)
    base_inv = np.zeros(n_dates)
    sl_inv   = np.zeros(n_dates)
    cap_arr  = np.zeros(n_dates, dtype=np.float32)  # effective CAP per day

    for i, date in enumerate(common_dates):
        ds = str(date.date())

        b_inv_v = sum(p.qty * float(close_mat[i, sym_to_i[s]]) for s, p in base_pos.items())
        s_inv_v = sum(
            sl_slots[k].qty * float(close_mat[i, sym_to_i[sl_slots[k].symbol]])
            for k in range(SLEEVE_MAX) if sl_slots[k] is not None
        )
        b_eq_v = base_cash + b_inv_v
        s_eq_v = sl_cash  + s_inv_v
        base_eq[i] = b_eq_v; sl_eq[i] = s_eq_v
        base_inv[i] = b_inv_v; sl_inv[i] = s_inv_v

        # Determine today's effective CAP
        eff_cap = acap.get_cap(i, s_eq_v)
        cap_arr[i] = eff_cap

        if b_eq_v > base_peak: base_peak = b_eq_v
        dd = (b_eq_v - base_peak) / base_peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05: cb_active = False; cb_days = 0

        gross_cap = g_normal
        if gross_en and topix_ret20 is not None:
            r20 = float(topix_ret20[i]) if i < len(topix_ret20) else 0.0
            r60 = float(topix_ret60[i]) if i < len(topix_ret60) else 0.0
            if r20 < -0.05: gross_cap = g_dd5
            elif r60 < -0.08: gross_cap = g_dd8

        is_bear   = bool(bear_arr[i]) if bear_arr is not None else False
        sec_eff   = 0.18 if is_bear else MAX_SECTOR_W
        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n_dates: break
        ni = i + 1

        # ── Base sells ────────────────────────────────────────────────
        b_sell, b_buy = [], []
        for sym in active_syms:
            si_sym  = sym_to_i[sym]
            holding = sym in base_pos
            hidx    = (i - base_pos[sym].entry_idx) if holding else 0
            rsr_v   = float(rsr_mat[i, si_sym])
            cl_t    = float(close_mat[i, si_sym])
            if mkt_shock and holding:
                if i > 0:
                    pc = float(close_mat[i-1, si_sym])
                    if pc > 0 and (cl_t/pc - 1.0) <= SHOCK_SYM_THR:
                        b_sell.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not holding: continue
            if holding and max_hold and hidx > max_hold:
                b_sell.append((sym, "TIME_STOP")); continue
            if holding and rsr_v < rsr_exit_thr and hidx >= min_hold:
                b_sell.append((sym, "RSR_EXIT")); continue
            sig = int(sig_mat[i, si_sym]) if sig_ready[si_sym] else 0
            if sig == -1 and holding and hidx >= min_hold:
                b_sell.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not holding:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si_sym]) < 0.5: continue
                b_buy.append((rsr_v, sym))

        for sym, reason in b_sell:
            if sym not in base_pos: continue
            pos = base_pos[sym]
            sp  = float(open_mat[ni, sym_to_i[sym]])
            base_cash += pos.qty * sp * (1 - COST_ONE_WAY)
            base_trades.append({"side":"SELL","symbol":sym,"pnl":(sp-pos.entry_price)*pos.qty,
                                 "entry":pos.entry_price,"exit":sp,"qty":pos.qty,
                                 "entry_idx":pos.entry_idx,"exit_idx":i,"reason":reason,"date":ds})
            del base_pos[sym]
            if reason == "TIME_STOP": reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not cb_active and b_buy:
            b_buy.sort(key=lambda x: -x[0])
            for rsr_v, sym in b_buy:
                si_sym = sym_to_i[sym]
                if base_max_pos - len(base_pos) <= 0: break
                bp = float(open_mat[ni, si_sym])
                if bp <= 0: continue
                if not _sector_ok(sym, base_pos, close_mat, i, sym_to_i, trade_syms, capital, sec_eff): continue
                if gross_en:
                    cg = sum(p.qty*float(close_mat[i,sym_to_i[p.symbol]]) for p in base_pos.values())/max(1.0,capital)
                    if cg + bp*LOT/max(1.0,capital) > gross_cap: continue
                qty = int(capital / base_max_pos / bp / LOT) * LOT
                if qty <= 0 or qty*bp*(1+COST_ONE_WAY) > base_cash: continue
                _execute_buy(sym, bp, qty, i, ni, trade_syms, base_trades, base_pos, rsr_v)
                base_cash -= qty*bp*(1+COST_ONE_WAY)
                base_trades[-1]["date"] = ds

        # ── Sleeve exits ──────────────────────────────────────────────
        for k in range(SLEEVE_MAX):
            pos = sl_slots[k]
            if pos is None: continue
            si_sym  = sym_to_i[pos.symbol]
            rsr_v   = float(rsr_mat[i, si_sym])
            cl_t    = float(close_mat[i, si_sym])
            do_exit = False; reason = ""
            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si_sym])
                if pc > 0 and (cl_t/pc - 1.0) <= SHOCK_SYM_THR:
                    do_exit = True; reason = "MARKET_SHOCK_EXIT"
            if not do_exit and rsr_v < SLEEVE_RSR_LO:
                do_exit = True; reason = "RSR_EXIT_LO"
            if do_exit:
                sp  = float(open_mat[ni, si_sym])
                pnl = (sp - pos.entry_price) * pos.qty
                sl_cash += pos.qty * sp * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side":"SELL","symbol":pos.symbol,"pnl":pnl,
                    "entry":pos.entry_price,"exit":sp,"qty":pos.qty,
                    "entry_idx":pos.entry_idx,"exit_idx":i,"hold_days":i-pos.entry_idx,
                    "reason":reason,"date":ds,"slot":k,"weight":SL_WEIGHTS[k],
                    "fwd10":np.nan,"fwd20":np.nan,
                    "cap_at_entry": pos.rsr_at_entry,  # store effective cap at entry
                })
                acap.record_trade(pnl)
                sl_slots[k] = None

        # ── Sleeve entries ─────────────────────────────────────────────
        if not mkt_shock:
            n_occ = sum(1 for k in range(SLEEVE_MAX) if sl_slots[k] is not None)
            if NO_REFILL and 0 < n_occ < SLEEVE_MAX:
                pass
            elif n_occ < SLEEVE_MAX:
                sl_syms = {sl_slots[k].symbol for k in range(SLEEVE_MAX) if sl_slots[k] is not None}
                cands = []
                for sym in active_syms:
                    if sym in base_pos or sym in sl_syms: continue
                    si_sym = sym_to_i[sym]
                    rsr_v  = float(rsr_mat[i, si_sym])
                    if not (SLEEVE_RSR_LO <= rsr_v < SLEEVE_RSR_HI): continue
                    d90 = int(cross90_mat[i, si_sym])
                    if d90 > SLEEVE_D90_MAX: continue
                    if sym_active_mat is not None and float(sym_active_mat[i, si_sym]) < 0.5: continue
                    st = classify_state(float(slope5_mat[i, si_sym]), float(slope20_mat[i, si_sym]))
                    cands.append((rsr_v, d90, STATE_SCORE.get(st,0), sym))
                if cands:
                    cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                    ci = 0
                    for k in range(SLEEVE_MAX):
                        if sl_slots[k] is not None: continue
                        if ci >= len(cands): break
                        rsr_v, d90, _, sym = cands[ci]; ci += 1
                        si_sym = sym_to_i[sym]
                        bp     = float(open_mat[ni, si_sym])
                        if bp <= 0: continue
                        # Sizing uses effective CAP from b_eq (base equity)
                        target = b_eq_v * eff_cap * SL_WEIGHTS[k]
                        alloc  = min(sl_cash, target)
                        qty    = int(alloc / bp / LOT) * LOT
                        cost   = qty * bp * (1 + COST_ONE_WAY)
                        if qty <= 0 or cost > sl_cash: continue
                        sl_cash -= cost
                        # store eff_cap in rsr_at_entry field
                        sl_slots[k] = Position(sym, trade_syms.get(sym,""), qty, bp, ni, eff_cap)
                        sl_trades.append({
                            "side":"BUY","symbol":sym,"slot":k,"weight":SL_WEIGHTS[k],
                            "rsr_entry":rsr_v,"d90_entry":d90,"entry_idx":ni,
                            "date":ds,"entry":bp,"qty":qty,"cap_at_entry":eff_cap,
                        })

    # fwd returns
    for t in sl_trades:
        if t["side"] != "SELL": continue
        si = sym_to_i.get(t["symbol"], -1)
        if si < 0: continue
        ei = t["exit_idx"]
        for fk, fn in [("fwd10",10),("fwd20",20)]:
            if ei + fn < n_dates:
                c0 = float(close_mat[ei, si]); cn = float(close_mat[ei+fn, si])
                t[fk] = round((cn/c0-1)*100,2) if c0>0 and cn>0 else float("nan")

    return {
        "base_eq": base_eq, "sl_eq": sl_eq,
        "base_inv": base_inv, "sl_inv": sl_inv,
        "base_trades": base_trades, "sl_trades": sl_trades,
        "cap_arr": cap_arr, "acap": acap,
    }


# ─── fold metrics ─────────────────────────────────────────────────────

def compute_fold(sim, close_mat, common_dates, fold, capital):
    oos_s, oos_e = fold["oos_start"], fold["oos_end"]
    ds_all  = [str(d.date()) for d in common_dates]
    oos_idx = [i for i, ds in enumerate(ds_all) if oos_s <= ds <= oos_e]
    if not oos_idx: return {}
    si, ei  = oos_idx[0], oos_idx[-1]+1
    n_days  = ei - si; n_years = n_days / 252.0

    base_eq  = sim["base_eq"][si:ei].tolist()
    sl_eq    = sim["sl_eq"][si:ei].tolist()
    comb_eq  = (sim["base_eq"] + sim["sl_eq"])[si:ei].tolist()
    b_inv_o  = sim["base_inv"][si:ei].tolist()
    s_inv_o  = sim["sl_inv"][si:ei].tolist()
    cap_oos  = sim["cap_arr"][si:ei]
    oos_dates= list(common_dates[si:ei])

    base_t = [t for t in sim["base_trades"] if oos_s <= t.get("date","") <= oos_e]
    sl_t   = [t for t in sim["sl_trades"]   if oos_s <= t.get("date","") <= oos_e]
    sl_sell= [t for t in sl_t if t["side"]=="SELL"]

    b_exp = [bi/max(1,be) for bi,be in zip(b_inv_o,base_eq)]
    bm    = calc_metrics(base_eq, base_t, b_exp, base_eq[0], oos_dates)
    ce    = [(bi+si_)/max(1,be+se) for bi,si_,be,se in zip(b_inv_o,s_inv_o,base_eq,sl_eq)]
    cm    = calc_metrics(comb_eq, base_t+sl_t, ce, comb_eq[0], oos_dates)

    sl0    = sl_eq[0]
    unused = [be+sl0 for be in base_eq]
    um     = calc_metrics(unused, base_t, b_exp, unused[0], oos_dates)

    delta_cagr = cm.get("cagr",0) - bm.get("cagr",0)
    comb_dd    = cm.get("max_dd",0)
    delta_dd   = -(comb_dd - bm.get("max_dd",0))

    se  = [si_/max(1,se) for si_,se in zip(s_inv_o,sl_eq)]
    slm = calc_metrics(sl_eq, sl_t, se, sl_eq[0], oos_dates)
    sl_cagr = slm.get("cagr",0)

    n_trig  = len(sl_sell)
    tpy     = n_trig / max(0.01, n_years)
    holds   = [t["hold_days"] for t in sl_sell if t.get("hold_days",0)>0]
    avg_hold= float(np.mean(holds)) if holds else 0.0

    s_inv_arr = np.array(s_inv_o)
    inv_days  = int(np.sum(s_inv_arr > 0))
    cap_util  = inv_days / max(1,n_days) * 100

    wins = [t for t in sl_sell if (t.get("pnl") or 0) > 0]
    loss = [t for t in sl_sell if (t.get("pnl") or 0) <= 0]
    pf   = sum(t["pnl"] for t in wins) / max(1.0, abs(sum(t["pnl"] for t in loss))) if wins else 0.0

    avg_cap      = float(np.mean(cap_oos)) if len(cap_oos) > 0 else CAP_HIGH
    cap_lo_days  = int(np.sum(cap_oos < CAP_HIGH))
    state_ratio  = cap_lo_days / max(1, n_days)

    # cap_switch_count in OOS
    switches = 0
    prev = None
    for c in cap_oos:
        if prev is not None and c != prev:
            switches += 1
        prev = c

    # fold_state_transition: (hi→lo, lo→hi)
    hi_to_lo = lo_to_hi = 0
    for j in range(1, len(cap_oos)):
        if cap_oos[j-1] >= CAP_HIGH and cap_oos[j] < CAP_HIGH:
            hi_to_lo += 1
        elif cap_oos[j-1] < CAP_HIGH and cap_oos[j] >= CAP_HIGH:
            lo_to_hi += 1

    fwd10v = [t.get("fwd10",np.nan) for t in sl_sell if not math.isnan(t.get("fwd10",float("nan")))]
    med_fwd10 = float(np.median(fwd10v)) if fwd10v else float("nan")

    fold_pass = (delta_cagr > ADOPT_DC and delta_dd <= ADOPT_DD)

    return {
        "base_cagr":    round(bm.get("cagr",0),   2),
        "sl_cagr":      round(sl_cagr,             2),
        "comb_cagr":    round(cm.get("cagr",0),    2),
        "unused_cagr":  round(um.get("cagr",0),    2),
        "delta_cagr":   round(delta_cagr,           2),
        "delta_dd":     round(delta_dd,             2),
        "comb_dd":      round(comb_dd,              3),
        "n_trig":       n_trig, "tpy": round(tpy,1),
        "avg_hold":     round(avg_hold,1),
        "cap_util":     round(cap_util,1),
        "avg_cap":      round(avg_cap,4),
        "state_ratio":  round(state_ratio,3),
        "switches":     switches,
        "hi_to_lo":     hi_to_lo, "lo_to_hi": lo_to_hi,
        "pf":           round(pf,3),
        "med_fwd10":    round(med_fwd10,2) if not math.isnan(med_fwd10) else float("nan"),
        "fold_pass":    fold_pass,
    }


# ─── WF aggregate ────────────────────────────────────────────────────

def aggregate_wf(folds: list[dict], sl_cagr_a: float | None) -> dict:
    v = [f for f in folds if f]
    if not v: return {}
    n_pass    = sum(1 for f in v if f["fold_pass"])
    avg_dc    = float(np.mean([f["delta_cagr"] for f in v]))
    avg_dd    = float(np.mean([f["delta_dd"]   for f in v]))
    avg_sl    = float(np.mean([f["sl_cagr"]    for f in v]))
    avg_cap   = float(np.mean([f["avg_cap"]    for f in v]))
    avg_sr    = float(np.mean([f["state_ratio"]for f in v]))
    avg_sw    = float(np.mean([f["switches"]   for f in v]))
    avg_tpy   = float(np.mean([f["tpy"]        for f in v]))

    alpha_ret = (avg_sl / sl_cagr_a * 100) if sl_cagr_a and sl_cagr_a > 0 else float("nan")
    eff_cap_red = CAP_HIGH - avg_cap  # how much cap was reduced on average

    adopted = (n_pass >= ADOPT_WF and avg_dc > ADOPT_DC
               and avg_dd <= ADOPT_DD
               and (math.isnan(alpha_ret) or alpha_ret >= ADOPT_AR))
    return {
        "n_pass": n_pass, "avg_dc": round(avg_dc,2), "avg_dd": round(avg_dd,2),
        "avg_sl": round(avg_sl,2), "avg_cap": round(avg_cap,4),
        "avg_sr": round(avg_sr,3), "avg_sw": round(avg_sw,1),
        "avg_tpy": round(avg_tpy,1),
        "alpha_ret": round(alpha_ret,1) if not math.isnan(alpha_ret) else float("nan"),
        "eff_cap_red": round(eff_cap_red,4),
        "adopted": adopted,
    }


# ─── report ──────────────────────────────────────────────────────────

def _f(v, fmt=".2f") -> str:
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:{fmt}}"

def regime_label(tc, yr):
    if tc is None: return "N/A"
    s = tc[tc.index.year == yr]
    if len(s) < 2: return "N/A"
    r = float(s.iloc[-1]/s.iloc[0]-1)
    return f"{'Bull' if r>0 else 'Bear'} ({r*100:+.1f}%)"

def write_report(results: dict, topix_close, output_path: Path,
                 sl_cagr_a: float, stop_condition_met: bool,
                 risk_elasticity: dict, dd_saved_alpha_lost: dict,
                 cap_efficiency: dict) -> None:
    L = []; w = L.append

    adopted = [c for c in CASE_SPECS if results[c]["wf"].get("adopted")]

    w("# Study8 Adaptive Risk Budget WF — State-Driven CAP")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  解析専用 / 実装変更禁止")
    w(f"\n固定: 2-slot 70/30 no-refill / ENTRY RSR[92,95) d90≤5 / EXIT RSR<90")
    w(f"\n採用条件: WF≥{ADOPT_WF}/5, ΔCAGR>+{ADOPT_DC}pp, ΔDD≤+{ADOPT_DD}pp, α_ret≥{ADOPT_AR}%\n")
    if stop_condition_met:
        w("⚠ **停止条件発動**: 固定15%(Case B)がadaptive全群を上回る → adaptive研究終了\n")
    w(f"**採用Case**: {len(adopted)}件  "
      f"| 判定: **{'✅ ADOPT' if adopted else '🔬 RESEARCH' if not stop_condition_met else '🛑 STOP'}**\n")

    # ── 1. WF Summary ─────────────────────────────────────────────────
    w("---\n## 1. WF サマリ\n")
    w("| Case | 説明 | WF | ΔCAGR | ΔDD | sl_CAGR | α_ret | avg_cap | state_ratio | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for case, spec in CASE_SPECS.items():
        wf = results[case]["wf"]
        ar = (_f(wf.get("alpha_ret",float("nan")),".1f")+"%"
              if not math.isnan(wf.get("alpha_ret",float("nan"))) else "—")
        ok = "**✅**" if wf.get("adopted") else "❌"
        w(f"| {case} | {spec['desc']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_dc',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_sl',0):+.1f}% "
          f"| {ar} "
          f"| {wf.get('avg_cap',0)*100:.1f}% "
          f"| {wf.get('avg_sr',0)*100:.1f}% "
          f"| {ok} |")

    # ── 2. Fold Detail ─────────────────────────────────────────────────
    w("\n---\n## 2. Fold 詳細\n")
    w("| Case | Fold | OOS | Regime | ΔCAGR | ΔDD | sl_CAGR | trig/yr | avg_cap | state_ratio | switches | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in CASE_SPECS:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm["fold_pass"] else "❌"
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['tpy']:.1f} "
              f"| {fm['avg_cap']*100:.1f}% "
              f"| {fm['state_ratio']*100:.1f}% "
              f"| {fm['switches']} "
              f"| {ok} |")

    # ── 3. Risk Audit ──────────────────────────────────────────────────
    w("\n---\n## 3. Risk Audit\n")
    w("| Case | risk_elasticity (ΔDD/cap_red) | dd_saved/alpha_lost | cap_efficiency (ΔCAGR/avg_cap) | avg_switches/fold |")
    w("|---|---|---|---|---|")
    for case in CASE_SPECS:
        re   = risk_elasticity.get(case, float("nan"))
        dsal = dd_saved_alpha_lost.get(case, float("nan"))
        ce   = cap_efficiency.get(case, float("nan"))
        sw   = results[case]["wf"].get("avg_sw", 0)
        re_s   = f"{re:+.3f}" if not math.isnan(re) else "—"
        dsal_s = f"{dsal:+.3f}" if not math.isnan(dsal) else "—"
        ce_s   = f"{ce:+.3f}" if not math.isnan(ce) else "—"
        w(f"| {case} | {re_s} | {dsal_s} | {ce_s} | {sw:.1f} |")

    # ── 4. Fold3 / Fold5 Deep Dive ─────────────────────────────────────
    w("\n---\n## 4. Fold3 / Fold5 — Target Fold Analysis\n")
    for fold_idx, fold_name in [(2,"Fold3 (2023)"), (4,"Fold5 (2025)")]:
        w(f"\n### {fold_name}\n")
        w("| Case | ΔCAGR | ΔDD | sl_CAGR | avg_cap | state_ratio | switches |")
        w("|---|---|---|---|---|---|---|")
        for case in CASE_SPECS:
            fm = results[case]["folds"][fold_idx]
            if not fm: continue
            ok = "✅" if fm["fold_pass"] else "❌"
            w(f"| {case} {ok} "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['avg_cap']*100:.1f}% "
              f"| {fm['state_ratio']*100:.1f}% "
              f"| {fm['switches']} |")

    # ── 5. State Analysis ──────────────────────────────────────────────
    w("\n---\n## 5. State Signal Analysis\n")
    w("| Case | state_driver | avg_days_reduced/fold | avg_cap | "
      "ΔDD_at_reduced | ΔDD_at_normal | cap_red_effectiveness |")
    w("|---|---|---|---|---|---|---|")
    for case, spec in CASE_SPECS.items():
        wf = results[case]["wf"]
        if spec["type"] == "fixed":
            w(f"| {case} | fixed | — | {wf.get('avg_cap',0)*100:.1f}% | — | — | — |")
            continue
        avg_sr    = wf.get("avg_sr", 0)
        avg_cap_v = wf.get("avg_cap", CAP_HIGH)
        re_v      = risk_elasticity.get(case, float("nan"))
        # Avg reduced days per fold
        red_days = [int(fm.get("state_ratio",0)*max(1,fm.get("n_trig",1)*10))
                    for fm in results[case]["folds"] if fm]
        avg_red_d = float(np.mean(red_days)) if red_days else 0.0
        eff_str   = f"{re_v:+.3f}" if not math.isnan(re_v) else "—"
        w(f"| {case} | {spec['type']} "
          f"| {avg_sr*252:.0f}d/yr "
          f"| {avg_cap_v*100:.1f}% "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {results['A']['wf'].get('avg_dd',0):+.2f}pp "
          f"| {eff_str} |")

    # ── 6. Failure Analysis ────────────────────────────────────────────
    w("\n---\n## 6. Failure Analysis\n")
    for case, spec in CASE_SPECS.items():
        wf    = results[case]["wf"]
        fails = []
        if wf.get("n_pass",0) < ADOPT_WF:
            fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF}")
        if wf.get("avg_dc",0) <= ADOPT_DC:
            fails.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp ≤ +{ADOPT_DC}pp")
        if wf.get("avg_dd",0) > ADOPT_DD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp > +{ADOPT_DD}pp")
        ar = wf.get("alpha_ret", float("nan"))
        if not math.isnan(ar) and ar < ADOPT_AR:
            fails.append(f"α_ret={ar:.1f}% < {ADOPT_AR}%")
        if not fails:
            w(f"\n**{case}** ✅ 全基準クリア")
        else:
            w(f"\n**{case}** REJECT: " + " / ".join(fails))

    # ── 7. Stop Condition Assessment ──────────────────────────────────
    w("\n---\n## 7. Stop Condition Assessment\n")
    wf_b  = results["B"]["wf"]
    adapt = [c for c in CASE_SPECS if CASE_SPECS[c]["type"] != "fixed"]
    b_dc  = wf_b.get("avg_dc",0)
    b_dd  = wf_b.get("avg_dd",0)
    b_wf  = wf_b.get("n_pass",0)
    w(f"Case B (固定15%): WF={b_wf}/5  ΔCAGR={b_dc:+.2f}pp  ΔDD={b_dd:+.2f}pp\n")
    w("| adaptive Case | ΔCAGR vs B | ΔDD vs B | WF vs B | B wins? |")
    w("|---|---|---|---|---|")
    for case in adapt:
        wf = results[case]["wf"]
        dc_diff = wf.get("avg_dc",0) - b_dc
        dd_diff = wf.get("avg_dd",0) - b_dd
        wf_diff = wf.get("n_pass",0) - b_wf
        # B wins if it has better or equal on all dimensions
        b_wins = (b_dc >= wf.get("avg_dc",0) and b_dd <= wf.get("avg_dd",0)
                  and b_wf >= wf.get("n_pass",0))
        w(f"| {case} "
          f"| {dc_diff:+.2f}pp "
          f"| {dd_diff:+.2f}pp "
          f"| {wf_diff:+d} "
          f"| {'⚠ Yes' if b_wins else 'No'} |")
    w(f"\n**停止条件**: {'**発動** → adaptive研究終了' if stop_condition_met else '未発動 → 継続検討余地あり'}")

    # ── 8. Final Output (必須) ────────────────────────────────────────
    w("\n---\n## 8. Final Output\n")

    # best_state_driver
    adapt_results = [(c, results[c]["wf"]) for c in adapt]
    best_adaptive = max(adapt_results, key=lambda x: (
        x[1].get("n_pass",0), x[1].get("avg_dc",0), -x[1].get("avg_dd",99)
    ))
    best_c, best_wf = best_adaptive

    w("### best_state_driver\n")
    w(f"| 項目 | 値 |")
    w(f"|---|---|")
    w(f"| name | **{best_c}** — {CASE_SPECS[best_c]['desc']} |")
    w(f"| WF | {best_wf.get('n_pass',0)}/5 |")
    w(f"| ΔCAGR | {best_wf.get('avg_dc',0):+.2f}pp |")
    w(f"| ΔDD | {best_wf.get('avg_dd',0):+.2f}pp |")
    w(f"| sl_CAGR | {best_wf.get('avg_sl',0):+.1f}% |")
    ar_v = best_wf.get("alpha_ret", float("nan"))
    w(f"| alpha_ret | {ar_v:.1f}% |" if not math.isnan(ar_v) else "| alpha_ret | — |")
    w(f"| avg_cap | {best_wf.get('avg_cap',0)*100:.1f}% |")
    w(f"| state_ratio | {best_wf.get('avg_sr',0)*100:.1f}% |")
    w(f"| adopted | {'✅ Yes' if best_wf.get('adopted') else '❌ No'} |")

    # risk_elasticity
    w("\n### risk_elasticity\n")
    w("| Case | ΔDD/effective_cap_reduction | interpretation |")
    w("|---|---|---|")
    for case in CASE_SPECS:
        re = risk_elasticity.get(case, float("nan"))
        if math.isnan(re):
            interp = "fixed / N/A"
        elif re < 0:
            interp = "cap削減 → ΔDD改善 (期待通り)"
        elif re < 1.0:
            interp = "弱い改善"
        else:
            interp = "cap削減 → ΔDD悪化 (逆効果)"
        re_s = f"{re:+.3f}" if not math.isnan(re) else "—"
        w(f"| {case} | {re_s} | {interp} |")

    # recommend next study
    w("\n### recommend next study\n")
    if stop_condition_met:
        w("**停止条件発動 → adaptive研究終了**\n")
        w("固定15%が adaptive 全群を上回った。CAP削減の改善余地は state-driven でなく固定削減にある。")
        w("\n**次研究推奨**: Study8 CAP Sweep (固定 10/12.5/15%) の WF による最適固定CAP探索")
        w("- スクリプト: `src/backtest/study8_cap_sweep.py`")
        w("- 出力: `reports/study8_cap_sweep.md`")
        w("- 理由: adaptive gain がゼロなら固定最適値を確定するのが最効率")
    elif adopted:
        best_for_next = max(adopted, key=lambda c: results[c]["wf"].get("avg_dc",0))
        w(f"**採用 Case {best_for_next} を production path に統合**")
        w(f"\n- shadow 30d 必須")
        w(f"- signal_bridge.py に {CASE_SPECS[best_for_next]['type']} CAP ロジック追加")
        w(f"- cap_state.json で状態管理 (idempotent)")
        w(f"- EXIT/ENTRY/GATE 変更禁止")
    else:
        # partial improvement
        w(f"**best_state_driver = {best_c}** ({CASE_SPECS[best_c]['desc']}): WF不足だが改善兆候あり")
        w(f"\n**次研究推奨**: CAP threshold 探索")
        w(f"- {best_c} の閾値パラメータ (DD_THR / STREAK_THR) をWFで最適化")
        w(f"- または SLEEVE_CAP_FR=15% 固定 (Study8 failure attribution 推奨) を先行")
        w(f"- スクリプト: `src/backtest/study8_cap_reduction.py`")
        w(f"- 禁止: gate追加 / exit変更 / entry変更")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─── main ────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Study8 Adaptive Risk Budget WF — State-Driven CAP")
    print("  Base: 2-slot 70/30 no-refill | CAP_HI=20% CAP_LO=10%")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms   = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms  = list(trade_syms.keys())
    sym_to_i     = {s: idx for idx, s in enumerate(active_syms)}
    n_syms       = len(active_syms)

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
        rule  = SECTOR_STRATEGY.get(trade_syms.get(sym,""), "fujiko")
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        st    = (MeanReversionStrategy(**MR_PARAMS) if rule == "mean_rev"
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
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                             dtype=np.float32, fill_value=1.0))

    mkt_ret1 = topix_ret20 = topix_ret60 = bear_arr = None
    if topix_close is not None:
        mkt_ret1    = _take(topix_close.pct_change(),    common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20), common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60), common_dates, dtype=np.float32, fill_value=0.0)
        ma200    = topix_close.rolling(200, min_periods=100).mean()
        bear_arr = ((topix_close < ma200)
                    .reindex(pd.DatetimeIndex(common_dates), method="ffill")
                    .fillna(False).values.astype(bool))

    cross90_mat         = cross_mat(rsr_mat, SLEEVE_RSR_LO)
    slope5_mat, slope20_mat = slopes(rsr_mat)

    print(f"\n[3/4] WF Simulation ({len(CASE_SPECS)} cases × 5 folds)...\n")
    capital = float(cfg.portfolio.capital)
    all_results: dict[str, dict] = {}
    sl_cagr_a: float | None = None

    for case_id, spec in CASE_SPECS.items():
        print(f"  [{case_id}] {spec['desc']}")
        sim = run_case(
            spec, open_mat, close_mat, sig_mat, sig_ready,
            rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, trade_syms, cfg, common_dates,
        )
        fold_metrics = []
        for fold in FOLDS:
            fm = compute_fold(sim, close_mat, common_dates, fold, capital)
            fold_metrics.append(fm)
            if fm:
                ok = "✅" if fm["fold_pass"] else "❌"
                cap_s = f"cap={fm['avg_cap']*100:.0f}%  sr={fm['state_ratio']*100:.0f}%"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"Δ={fm['delta_cagr']:+.2f}pp  ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"sl={fm['sl_cagr']:+.1f}%  {cap_s}  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")

        wf = aggregate_wf(fold_metrics, sl_cagr_a)
        if case_id == "A":
            sl_cagr_a = wf.get("avg_sl")

        all_results[case_id] = {"folds": fold_metrics, "wf": wf, "sim": sim}
        ar_s = (f"  α_ret={wf.get('alpha_ret',0):.1f}%"
                if not math.isnan(wf.get("alpha_ret", float("nan"))) else "")
        ok = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"avg_cap={wf.get('avg_cap',0)*100:.1f}%  "
              f"sr={wf.get('avg_sr',0)*100:.1f}%{ar_s}  {ok}\n")

    # Re-aggregate with sl_cagr_a baseline
    if sl_cagr_a is not None:
        for case_id in list(CASE_SPECS)[1:]:
            all_results[case_id]["wf"] = aggregate_wf(
                all_results[case_id]["folds"], sl_cagr_a)

    # ── Risk audit metrics ────────────────────────────────────────────
    wf_a = all_results["A"]["wf"]
    dd_a = wf_a.get("avg_dd", 0)
    dc_a = wf_a.get("avg_dc", 0)

    risk_elasticity:    dict[str, float] = {}
    dd_saved_alpha_lost:dict[str, float] = {}
    cap_efficiency:     dict[str, float] = {}

    for case_id in CASE_SPECS:
        wf      = all_results[case_id]["wf"]
        avg_cap = wf.get("avg_cap", CAP_HIGH)
        cap_red = CAP_HIGH - avg_cap          # reduction vs baseline
        dd_chg  = wf.get("avg_dd", 0) - dd_a  # negative = DD improved
        dc_chg  = wf.get("avg_dc", 0) - dc_a

        if cap_red > 1e-6:
            risk_elasticity[case_id]     = round(dd_chg / cap_red, 3)
            alpha_lost                   = max(0, -dc_chg)   # positive if ΔCAGR fell
            dd_saved                     = max(0, -dd_chg)   # positive if ΔDD fell
            dd_saved_alpha_lost[case_id] = round(dd_saved / max(0.001, alpha_lost), 3)
        else:
            risk_elasticity[case_id]     = float("nan")
            dd_saved_alpha_lost[case_id] = float("nan")

        cap_eff_v = wf.get("avg_dc", 0) / max(0.001, avg_cap)
        cap_efficiency[case_id] = round(cap_eff_v, 3)

    # Stop condition: Case B beats ALL adaptive cases
    wf_b  = all_results["B"]["wf"]
    adapt_ids = [c for c in CASE_SPECS if CASE_SPECS[c]["type"] != "fixed"]
    stop_condition_met = all(
        wf_b.get("n_pass",0) >= all_results[c]["wf"].get("n_pass",0)
        and wf_b.get("avg_dc",0) >= all_results[c]["wf"].get("avg_dc",0)
        and wf_b.get("avg_dd",0) <= all_results[c]["wf"].get("avg_dd",0)
        for c in adapt_ids
    )

    # Print summary
    print("=" * 68)
    adopted = [c for c in CASE_SPECS if all_results[c]["wf"].get("adopted")]
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for case_id in CASE_SPECS:
        wf = all_results[case_id]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        ar = wf.get("alpha_ret", float("nan"))
        ar_s = f"  α_ret={ar:.1f}%" if not math.isnan(ar) else ""
        print(f"  {ok} [{case_id}] WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"cap={wf.get('avg_cap',0)*100:.1f}%{ar_s}")

    print(f"\n  停止条件: {'発動' if stop_condition_met else '未発動'}")
    print(f"\n  risk_elasticity:")
    for c, re in risk_elasticity.items():
        re_s = f"{re:+.3f}" if not math.isnan(re) else "N/A"
        print(f"    [{c}] {re_s}")
    print("=" * 68 + "\n")

    print("[4/4] レポート生成...")
    write_report(
        all_results, topix_close,
        REPORTS_DIR / "study8_adaptive_risk_budget.md",
        sl_cagr_a or 0.0,
        stop_condition_met,
        risk_elasticity, dd_saved_alpha_lost, cap_efficiency,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
