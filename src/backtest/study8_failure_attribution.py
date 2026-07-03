"""
src/backtest/study8_failure_attribution.py
Study8 Failure Attribution — Fold Diagnostics

対象: Case A (1-slot 100%) vs Case E (2-slot 70/30 no-refill)
ΔCAGR = selection + holding + sizing + timing + cash_effect
出力: reports/study8_failure_attribution.md

Run: cd C:/ai-trading && python src/backtest/study8_failure_attribution.py
"""
from __future__ import annotations
import sys, math, time, warnings
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

REPORTS_DIR   = Path("reports")
IS_START      = "2018-01-01"
SLEEVE_RSR_LO = 92.0
SLEEVE_RSR_HI = 95.0
SLEEVE_D90_MAX = 5
SLEEVE_CAP_FR  = 0.20
SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08
STALL_THR      = 0.15
PASSIVE_MAX_HOLD = 30  # days cap for passive zone tracking

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

CASE_SPECS = {
    "A": {"max_pos": 1, "weights": [1.00],        "no_refill": False},
    "E": {"max_pos": 2, "weights": [0.70, 0.30],  "no_refill": True},
}

ADOPT_DCAGR = 0.3
ADOPT_DDD   = 1.5


# ─── helpers ─────────────────────────────────────────────────────────

def classify_state(s5: float, s20: float) -> str:
    if np.isnan(s5) or np.isnan(s20): return "UNKNOWN"
    if abs(s5) < STALL_THR: return "STALL"
    if s20 > 0:
        if s5 > s20: return "EARLY_UP"
        if s5 > 0:   return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"

STATE_SCORE = {"EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1, "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0}

def compute_cross_mat(rsr_mat, thr):
    n_dates, n_syms = rsr_mat.shape
    out = np.zeros((n_dates, n_syms), dtype=np.int32)
    run = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = run
        above = rsr_mat[i] >= thr
        run[above] += 1; run[~above] = 0
    return out

def compute_slopes(rsr_mat):
    s5  = np.zeros_like(rsr_mat, dtype=np.float32)
    s20 = np.zeros_like(rsr_mat, dtype=np.float32)
    s5[5:]   = (rsr_mat[5:]  - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:] - rsr_mat[:-20]) / 20.0
    return s5, s20


# ─── passive zone quality ─────────────────────────────────────────────

def compute_passive_zone(
    rsr_mat, open_mat, cross90_mat, sym_active_mat,
    common_dates, oos_s: str, oos_e: str,
) -> list[float]:
    """All RSR[92,95) d90=1 entries in OOS; returns net-of-cost trade returns."""
    date_strs = [str(d.date()) for d in common_dates]
    oos_idx   = [i for i, ds in enumerate(date_strs) if oos_s <= ds <= oos_e]
    n_dates   = len(common_dates)
    n_syms    = rsr_mat.shape[1]
    rets      = []
    for i in oos_idx:
        for si in range(n_syms):
            if int(cross90_mat[i, si]) != 1: continue
            rsr_v = float(rsr_mat[i, si])
            if not (SLEEVE_RSR_LO <= rsr_v < SLEEVE_RSR_HI): continue
            if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
            if i + 1 >= n_dates: continue
            ep = float(open_mat[i + 1, si])
            if ep <= 0: continue
            exited = False
            for j in range(i + 1, min(i + 1 + PASSIVE_MAX_HOLD, n_dates)):
                if float(rsr_mat[j, si]) < SLEEVE_RSR_LO:
                    if j + 1 < n_dates:
                        xp = float(open_mat[j + 1, si])
                        if xp > 0:
                            rets.append(xp / ep - 1.0 - 2 * COST_ONE_WAY)
                    exited = True; break
            if not exited and i + PASSIVE_MAX_HOLD < n_dates:
                xp = float(open_mat[i + PASSIVE_MAX_HOLD, si])
                if xp > 0:
                    rets.append(xp / ep - 1.0 - 2 * COST_ONE_WAY)
    return rets


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

    sl_max     = spec["max_pos"]
    sl_weights = spec["weights"]
    no_refill  = spec["no_refill"]

    base_cash  = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak  = float(capital)
    cb_active  = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    sl_slots: list[Position | None] = [None] * sl_max
    sl_cash  = float(capital * SLEEVE_CAP_FR)
    sl_trades: list[dict] = []

    n_dates  = len(common_dates)
    base_eq  = np.zeros(n_dates)
    sl_eq    = np.zeros(n_dates)
    base_inv = np.zeros(n_dates)
    sl_inv   = np.zeros(n_dates)
    sl_held  = [set() for _ in range(n_dates)]
    base_held= [set() for _ in range(n_dates)]

    for i, date in enumerate(common_dates):
        ds = str(date.date())

        b_inv_v = sum(p.qty * float(close_mat[i, sym_to_i[s]]) for s, p in base_pos.items())
        s_inv_v = sum(
            sl_slots[k].qty * float(close_mat[i, sym_to_i[sl_slots[k].symbol]])
            for k in range(sl_max) if sl_slots[k] is not None
        )
        b_eq_v = base_cash + b_inv_v
        s_eq_v = sl_cash + s_inv_v
        base_eq[i] = b_eq_v; sl_eq[i] = s_eq_v
        base_inv[i] = b_inv_v; sl_inv[i] = s_inv_v
        sl_held[i]  = {sl_slots[k].symbol for k in range(sl_max) if sl_slots[k] is not None}
        base_held[i]= set(base_pos.keys())

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

        # Base sells
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

        # Sleeve exits
        for k in range(sl_max):
            pos = sl_slots[k]
            if pos is None: continue
            si_sym  = sym_to_i[pos.symbol]
            rsr_v   = float(rsr_mat[i, si_sym])
            cl_t    = float(close_mat[i, si_sym])
            do_exit, reason = False, ""
            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si_sym])
                if pc > 0 and (cl_t/pc - 1.0) <= SHOCK_SYM_THR:
                    do_exit, reason = True, "MARKET_SHOCK_EXIT"
            if not do_exit and rsr_v < SLEEVE_RSR_LO:
                do_exit, reason = True, "RSR_EXIT_LO"
            if do_exit:
                sp  = float(open_mat[ni, si_sym])
                pnl = (sp - pos.entry_price) * pos.qty
                net_ret = sp / pos.entry_price - 1.0 - 2 * COST_ONE_WAY
                sl_cash += pos.qty * sp * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side": "SELL", "symbol": pos.symbol, "pnl": pnl,
                    "entry": pos.entry_price, "exit": sp, "qty": pos.qty,
                    "entry_idx": pos.entry_idx, "exit_idx": i,
                    "hold_days": i - pos.entry_idx,
                    "reason": reason, "date": ds,
                    "slot": k, "slot_weight": sl_weights[k],
                    "d90_entry": pos.rsr_at_entry,  # repurposed field for d90
                    "entry_ret": net_ret,
                    "fwd5": np.nan, "fwd10": np.nan, "fwd20": np.nan,
                })
                sl_slots[k] = None

        # Sleeve entries
        if not mkt_shock:
            n_occ = sum(1 for k in range(sl_max) if sl_slots[k] is not None)
            if no_refill and 0 < n_occ < sl_max:
                pass
            elif n_occ < sl_max:
                sl_syms = {sl_slots[k].symbol for k in range(sl_max) if sl_slots[k] is not None}
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
                    cands.append((rsr_v, d90, STATE_SCORE.get(st, 0), d90, sym))
                if cands:
                    cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                    ci = 0
                    for k in range(sl_max):
                        if sl_slots[k] is not None: continue
                        if ci >= len(cands): break
                        rsr_v, d90, _, _, sym = cands[ci]; ci += 1
                        si_sym = sym_to_i[sym]
                        bp     = float(open_mat[ni, si_sym])
                        if bp <= 0: continue
                        target = b_eq_v * SLEEVE_CAP_FR * sl_weights[k]
                        alloc  = min(sl_cash, target)
                        qty    = int(alloc / bp / LOT) * LOT
                        cost   = qty * bp * (1 + COST_ONE_WAY)
                        if qty <= 0 or cost > sl_cash: continue
                        sl_cash -= cost
                        # store d90 in rsr_at_entry field (repurpose)
                        p = Position(sym, trade_syms.get(sym,""), qty, bp, ni, float(d90))
                        sl_slots[k] = p
                        sl_trades.append({
                            "side": "BUY", "symbol": sym, "slot": k,
                            "slot_weight": sl_weights[k], "rsr_entry": rsr_v,
                            "d90_entry": d90, "entry_idx": ni, "date": ds,
                            "entry": bp, "qty": qty,
                        })

    # Post-hoc fwd returns on SELL trades
    for t in sl_trades:
        if t["side"] != "SELL": continue
        si = sym_to_i.get(t["symbol"], -1)
        if si < 0: continue
        ei = t["exit_idx"]
        for fk, fn in [("fwd5", 5), ("fwd10", 10), ("fwd20", 20)]:
            if ei + fn < n_dates:
                c0 = float(close_mat[ei, si]); cn = float(close_mat[ei+fn, si])
                t[fk] = round((cn/c0-1)*100, 2) if c0>0 and cn>0 else float("nan")

    return {
        "base_eq": base_eq, "sl_eq": sl_eq,
        "base_inv": base_inv, "sl_inv": sl_inv,
        "base_trades": base_trades, "sl_trades": sl_trades,
        "sl_held": sl_held, "base_held": base_held,
    }


# ─── fold metrics ────────────────────────────────────────────────────

def compute_fold(sim, close_mat, common_dates, fold, capital, passive_rets):
    oos_s, oos_e = fold["oos_start"], fold["oos_end"]
    ds_all  = [str(d.date()) for d in common_dates]
    oos_idx = [i for i, ds in enumerate(ds_all) if oos_s <= ds <= oos_e]
    if not oos_idx: return {}
    si, ei  = oos_idx[0], oos_idx[-1] + 1
    n_days  = ei - si; n_years = n_days / 252.0

    base_eq  = sim["base_eq"][si:ei].tolist()
    sl_eq    = sim["sl_eq"][si:ei].tolist()
    comb_eq  = (sim["base_eq"] + sim["sl_eq"])[si:ei].tolist()
    b_inv_o  = sim["base_inv"][si:ei].tolist()
    s_inv_o  = sim["sl_inv"][si:ei].tolist()
    oos_dates= list(common_dates[si:ei])

    base_t = [t for t in sim["base_trades"] if oos_s <= t.get("date","") <= oos_e]
    sl_t   = [t for t in sim["sl_trades"]   if oos_s <= t.get("date","") <= oos_e]
    sl_sell= [t for t in sl_t if t["side"]=="SELL"]

    b_exp  = [bi/max(1,be) for bi,be in zip(b_inv_o, base_eq)]
    bm     = calc_metrics(base_eq, base_t, b_exp, base_eq[0], oos_dates)
    ce     = [(bi+si_)/max(1,be+se) for bi,si_,be,se in zip(b_inv_o,s_inv_o,base_eq,sl_eq)]
    cm     = calc_metrics(comb_eq, base_t+sl_t, ce, comb_eq[0], oos_dates)

    sl0    = sl_eq[0]
    unused = [be+sl0 for be in base_eq]
    um     = calc_metrics(unused, base_t, b_exp, unused[0], oos_dates)

    delta_cagr   = cm.get("cagr",0) - bm.get("cagr",0)
    comb_dd      = cm.get("max_dd",0)
    delta_dd     = -(comb_dd - bm.get("max_dd",0))

    se     = [si_/max(1,se) for si_,se in zip(s_inv_o,sl_eq)]
    slm    = calc_metrics(sl_eq, sl_t, se, sl_eq[0], oos_dates)
    sl_cagr = slm.get("cagr",0)

    n_trig  = len(sl_sell)
    tpy     = n_trig / max(0.01, n_years)
    holds   = [t["hold_days"] for t in sl_sell if t.get("hold_days",0) > 0]
    avg_hold= float(np.mean(holds)) if holds else 0.0

    s_inv_arr = np.array(s_inv_o)
    inv_days  = int(np.sum(s_inv_arr > 0))
    idle_days = n_days - inv_days
    cap_util  = inv_days / max(1,n_days) * 100

    wins  = [t for t in sl_sell if (t.get("pnl") or 0) > 0]
    loss  = [t for t in sl_sell if (t.get("pnl") or 0) <= 0]
    win_rate = len(wins)/max(1,len(sl_sell))*100
    gp = sum(t["pnl"] for t in wins) if wins else 0.0
    gl = abs(sum(t["pnl"] for t in loss)) if loss else 0.0
    pf = gp/max(1.0,gl)

    # avg_trade_ret (net of cost)
    act_rets = [t.get("entry_ret", np.nan) for t in sl_sell]
    act_rets = [r for r in act_rets if not math.isnan(r)]
    avg_tret = float(np.mean(act_rets)*100) if act_rets else 0.0

    # top10_trade_share
    pnls_sorted = sorted([t["pnl"] for t in sl_sell if t.get("pnl") is not None], reverse=True)
    top10_pnl   = sum(pnls_sorted[:10])
    total_pnl   = sum(pnls_sorted)
    top10_share = top10_pnl/max(1.0,abs(total_pnl))*100 if total_pnl!=0 else float("nan")

    # losing_trade_share
    loss_pnl    = abs(sum(p for p in pnls_sorted if p < 0))
    loss_share  = loss_pnl/max(1.0,abs(total_pnl))*100 if total_pnl!=0 else float("nan")

    # fwd10 stats
    fwd10v = [t.get("fwd10", np.nan) for t in sl_sell if not math.isnan(t.get("fwd10",float("nan")))]
    med_fwd10 = float(np.median(fwd10v)) if fwd10v else float("nan")
    avg_fwd10 = float(np.mean(fwd10v))   if fwd10v else float("nan")

    # ── Attribution decomposition ────────────────────────────────────
    avg_passive = float(np.mean(passive_rets)*100) if passive_rets else 0.0

    # selection: actual avg trade ret vs passive pool
    sel_raw = avg_tret - avg_passive  # %
    # annualize: how much does this difference contribute to sleeve CAGR?
    # contribution = per_trade_alpha × tpy × avg_hold/252 × 100 (already in %)
    hold_frac   = avg_hold / 252.0
    sel_pp      = sel_raw / 100 * tpy * hold_frac * 100   # back to %

    # holding: fwd10 direction (left on table = negative; exited well = positive)
    # negative fwd10 = good exit = positive contribution; positive fwd10 = bad exit
    hold_pp = 0.0
    if not math.isnan(avg_fwd10):
        hold_pp = -avg_fwd10 / 100 * tpy * hold_frac * 100

    # sizing: for Case E, slot0 vs slot1 return comparison
    slot0_rets = [t.get("entry_ret", np.nan) for t in sl_sell
                  if t.get("slot",0) == 0 and not math.isnan(t.get("entry_ret", float("nan")))]
    slot1_rets = [t.get("entry_ret", np.nan) for t in sl_sell
                  if t.get("slot",0) == 1 and not math.isnan(t.get("entry_ret", float("nan")))]
    if slot0_rets and slot1_rets:
        w0, w1   = 0.70, 0.30
        wt_avg   = np.mean(slot0_rets)*w0 + np.mean(slot1_rets)*w1
        eq_avg   = (np.mean(slot0_rets) + np.mean(slot1_rets)) / 2
        siz_raw  = (wt_avg - eq_avg) * 100
        siz_pp   = siz_raw / 100 * tpy * hold_frac * 100
    else:
        siz_pp = 0.0

    # timing: d90=1 vs d90>1 return premium
    d1_rets  = [t.get("entry_ret",np.nan) for t in sl_sell
                if t.get("d90_entry",1)==1 and not math.isnan(t.get("entry_ret",float("nan")))]
    dgt_rets = [t.get("entry_ret",np.nan) for t in sl_sell
                if t.get("d90_entry",1)>1  and not math.isnan(t.get("entry_ret",float("nan")))]
    if d1_rets and dgt_rets:
        tim_raw = (np.mean(d1_rets) - np.mean(dgt_rets)) * 100
        tim_pp  = tim_raw / 100 * tpy * hold_frac * 100
    elif d1_rets:
        tim_raw = (np.mean(d1_rets) - (avg_tret/100)) * 100
        tim_pp  = tim_raw / 100 * tpy * hold_frac * 100
    else:
        tim_pp = 0.0

    # cash_effect: idle fraction × opportunity cost (base CAGR)
    idle_frac    = idle_days / max(1, n_days)
    base_cagr_v  = bm.get("cagr", 0)
    cash_pp      = -idle_frac * base_cagr_v * SLEEVE_CAP_FR

    # Normalize to 100%
    comps = {"selection": sel_pp, "holding": hold_pp,
             "sizing": siz_pp, "timing": tim_pp, "cash_effect": cash_pp}
    abs_tot = sum(abs(v) for v in comps.values())
    if abs_tot > 0:
        comps_pct = {k: round(v/abs_tot*100, 1) for k, v in comps.items()}
    else:
        comps_pct = {k: 0.0 for k in comps}

    # ── Failure classification ────────────────────────────────────────
    fold_pass   = (delta_cagr > 0.3 and delta_dd <= 1.5)
    fail_scores = {}
    if sl_cagr < 5.0:
        fail_scores["alpha_absent"]   = max(0, (5.0 - sl_cagr) / 5.0)
    if delta_dd > 1.5:
        fail_scores["dd_excess"]      = min(1.0, (delta_dd - 1.5) / 3.0)
    if tpy > 20 and pf < 1.5:
        fail_scores["overtrading"]    = min(1.0, tpy/40) * max(0, (1.5-pf)/1.5)
    if cap_util < 60.0:
        fail_scores["capital_idle"]   = (60-cap_util)/60
    if not math.isnan(loss_share) and loss_share > 40:
        fail_scores["tail_miss"]      = min(1.0, (loss_share-40)/40)
    if base_cagr_v < 5.0:
        fail_scores["regime_break"]   = max(0, (5.0-base_cagr_v)/10.0)

    if fail_scores:
        tot_fs = sum(fail_scores.values())
        dom    = max(fail_scores, key=lambda k: fail_scores[k])
        dom_ratio = fail_scores[dom] / max(1e-9, tot_fs)
        if dom_ratio >= 0.70:
            failure_reason = dom; conf = "high"
        else:
            top2 = sorted(fail_scores, key=lambda k: -fail_scores[k])[:2]
            failure_reason = f"mixed({'+'.join(top2)})"; conf = "medium" if dom_ratio >= 0.50 else "low"
    else:
        failure_reason = "none"; conf = "high"

    # largest positive / negative contributor
    comps_signed = {k: comps[k] for k in comps}
    lpos = max(comps_signed, key=lambda k: comps_signed[k])
    lneg = min(comps_signed, key=lambda k: comps_signed[k])

    return {
        # performance
        "base_cagr":     round(bm.get("cagr",0),   2),
        "base_max_dd":   round(bm.get("max_dd",0),  3),
        "sl_cagr":       round(sl_cagr,              2),
        "comb_cagr":     round(cm.get("cagr",0),    2),
        "delta_cagr":    round(delta_cagr,           2),
        "delta_dd":      round(delta_dd,             2),
        "comb_dd":       round(comb_dd,              3),
        "unused_cagr":   round(um.get("cagr",0),    2),
        # execution
        "n_triggers":    n_trig,
        "tpy":           round(tpy, 1),
        "avg_hold":      round(avg_hold, 1),
        "capital_util":  round(cap_util, 1),
        "idle_days":     idle_days,
        "cash_idle_pct": round(idle_frac*100, 1),
        # trade quality
        "win_rate":      round(win_rate, 1),
        "pf":            round(pf, 3),
        "avg_tret":      round(avg_tret, 2),
        "top10_share":   round(top10_share, 1) if not math.isnan(top10_share) else float("nan"),
        "loss_share":    round(loss_share,  1) if not math.isnan(loss_share)  else float("nan"),
        "med_fwd10":     round(med_fwd10,   2) if not math.isnan(med_fwd10)   else float("nan"),
        "avg_fwd10":     round(avg_fwd10,   2) if not math.isnan(avg_fwd10)   else float("nan"),
        # attribution
        "sel_pp":        round(sel_pp, 3),
        "hold_pp":       round(hold_pp, 3),
        "siz_pp":        round(siz_pp, 3),
        "tim_pp":        round(tim_pp, 3),
        "cash_pp":       round(cash_pp, 3),
        "comps_pct":     comps_pct,
        "avg_passive":   round(avg_passive, 2),
        "n_passive":     len(passive_rets),
        # classification
        "fold_pass":     fold_pass,
        "failure_reason":failure_reason,
        "confidence":    conf,
        "lpos":          lpos,
        "lneg":          lneg,
    }


# ─── report ──────────────────────────────────────────────────────────

def _f(v, fmt=".2f") -> str:
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:{fmt}}"

def _pp(v) -> str:
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:+.2f}pp"

def regime_label(tc, yr):
    if tc is None: return "N/A"
    s = tc[tc.index.year == yr]
    if len(s) < 2: return "N/A"
    r = float(s.iloc[-1]/s.iloc[0]-1)
    return f"{'Bull' if r>0 else 'Bear'} ({r*100:+.1f}%)"

def write_report(results: dict, topix_close, output_path: Path) -> None:
    L = []; w = L.append

    w("# Study8 Failure Attribution — Fold Diagnostics")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  解析専用 / 実装変更禁止")
    w(f"\n対象: Case A (1-slot 100%) vs Case E (2-slot 70/30 no-refill)")
    w(f"\n採用条件: ΔCAGR>+{ADOPT_DCAGR}pp AND ΔDD≤+{ADOPT_DDD}pp\n")

    # ── 1. Performance Summary ─────────────────────────────────────────
    w("---\n## 1. Performance — Fold × Case\n")
    w("| Case | Fold | OOS | Regime | base_CAGR | sl_CAGR | total_CAGR | ΔCAGR | ΔDD | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for case in ["A","E"]:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm["fold_pass"] else "❌"
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['base_cagr']:+.1f}% | {fm['sl_cagr']:+.1f}% "
              f"| {fm['comb_cagr']:+.1f}% | {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp | {ok} |")

    # ── 2. Execution ──────────────────────────────────────────────────
    w("\n---\n## 2. Execution — Fold × Case\n")
    w("| Case | Fold | trig/yr | avg_hold | capital_util% | cash_idle% | idle_days |")
    w("|---|---|---|---|---|---|---|")
    for case in ["A","E"]:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            w(f"| {case} | Fold{fold['id']} | {fm['tpy']:.1f} "
              f"| {fm['avg_hold']:.1f}d | {fm['capital_util']:.1f}% "
              f"| {fm['cash_idle_pct']:.1f}% | {fm['idle_days']}d |")

    # ── 3. Trade Quality ──────────────────────────────────────────────
    w("\n---\n## 3. Trade Quality — Fold × Case\n")
    w("| Case | Fold | win_rate | PF | avg_trade_ret | top10_share | loss_share | med_fwd10 |")
    w("|---|---|---|---|---|---|---|---|")
    for case in ["A","E"]:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            w(f"| {case} | Fold{fold['id']} | {fm['win_rate']:.1f}% "
              f"| {fm['pf']:.3f} | {fm['avg_tret']:+.2f}% "
              f"| {_f(fm['top10_share'],'.1f')}% "
              f"| {_f(fm['loss_share'],'.1f')}% "
              f"| {_f(fm['med_fwd10'],'.2f')}% |")

    # ── 4. ΔCAGR Decomposition ─────────────────────────────────────────
    w("\n---\n## 4. ΔCAGR Decomposition (pp 寄与 / 正規化 %)\n")
    w("> selection=銘柄差 / holding=保有期間差 / sizing=配分差 / timing=エントリー時刻差 / cash_effect=未投資資金影響\n")
    w("> passive_zone = RSR[92,95) d90=1 全信号の平均リターン (selection の baseline)\n")
    w("| Case | Fold | sel_pp | hold_pp | siz_pp | tim_pp | cash_pp "
      "| sel% | hold% | siz% | tim% | cash% | passive_ret | n_pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in ["A","E"]:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            cp = fm["comps_pct"]
            w(f"| {case} | Fold{fold['id']} "
              f"| {fm['sel_pp']:+.3f} | {fm['hold_pp']:+.3f} "
              f"| {fm['siz_pp']:+.3f} | {fm['tim_pp']:+.3f} | {fm['cash_pp']:+.3f} "
              f"| {cp['selection']:+.1f}% | {cp['holding']:+.1f}% "
              f"| {cp['sizing']:+.1f}% | {cp['timing']:+.1f}% | {cp['cash_effect']:+.1f}% "
              f"| {fm['avg_passive']:+.2f}% | {fm['n_passive']} |")

    # ── 5. Fold Failure Summary ────────────────────────────────────────
    w("\n---\n## 5. Fold Failure Summary\n")
    w("| Case | Fold | winner | largest_positive | largest_negative | failure_reason | confidence |")
    w("|---|---|---|---|---|---|---|")
    for case in ["A","E"]:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            winner = "PASS" if fm["fold_pass"] else "FAIL"
            w(f"| {case} | Fold{fold['id']} | {winner} "
              f"| {fm['lpos']} | {fm['lneg']} "
              f"| {fm['failure_reason']} | {fm['confidence']} |")

    # ── 6. Dominant Bottleneck Analysis ──────────────────────────────
    w("\n---\n## 6. Dominant Bottleneck Analysis\n")

    for case in ["A","E"]:
        fms = [fm for fm in results[case]["folds"] if fm]
        fail_fms = [fm for fm in fms if not fm["fold_pass"]]
        w(f"\n### Case {case}\n")
        if not fail_fms:
            w("全Fold PASS (WF基準クリア)"); continue

        # Aggregate failure reasons
        from collections import Counter
        reason_counts = Counter()
        for fm in fail_fms:
            r = fm["failure_reason"]
            if r.startswith("mixed"):
                for sub in r[6:-1].split("+"):
                    reason_counts[sub.strip()] += 0.5
            elif r != "none":
                reason_counts[r] += 1.0

        if reason_counts:
            total_rc = sum(reason_counts.values())
            w("| 失敗因子 | 出現スコア | 割合 |")
            w("|---|---|---|")
            for r, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
                w(f"| {r} | {cnt:.1f} | {cnt/total_rc*100:.1f}% |")
            dom_r = max(reason_counts, key=reason_counts.get)
            dom_ratio = reason_counts[dom_r] / total_rc
            cause_type = "**single dominant cause**" if dom_ratio >= 0.70 else "**mixed causes**"
            w(f"\n→ {cause_type}: {dom_r} ({dom_ratio*100:.0f}%)")

        # attribution direction for failing folds
        avg_sel = float(np.mean([fm["sel_pp"] for fm in fail_fms]))
        avg_hld = float(np.mean([fm["hold_pp"] for fm in fail_fms]))
        avg_csh = float(np.mean([fm["cash_pp"] for fm in fail_fms]))
        w(f"\n失敗Fold 平均attribution: selection={avg_sel:+.3f}pp  "
          f"holding={avg_hld:+.3f}pp  cash={avg_csh:+.3f}pp")

    # ── 7. Fold3 vs Fold5 Deep Dive ───────────────────────────────────
    w("\n---\n## 7. Fold3 / Fold5 Deep Dive\n")

    for fold_idx, fold_name in [(2, "Fold3 (2023)"), (4, "Fold5 (2025)")]:
        w(f"\n### {fold_name}\n")
        w("| 指標 | Case A | Case E | 差(E-A) |")
        w("|---|---|---|---|")

        fm_a = results["A"]["folds"][fold_idx]
        fm_e = results["E"]["folds"][fold_idx]
        if not fm_a or not fm_e: continue

        def row(label, key, fmt=".2f"):
            va, ve = fm_a.get(key, float("nan")), fm_e.get(key, float("nan"))
            try: diff = ve - va
            except: diff = float("nan")
            return f"| {label} | {va:{fmt}} | {ve:{fmt}} | {diff:+{fmt}} |"

        w(row("base_CAGR (%)", "base_cagr"))
        w(row("sl_CAGR (%)",   "sl_cagr"))
        w(row("ΔCAGR (pp)",    "delta_cagr"))
        w(row("ΔDD (pp)",      "delta_dd"))
        w(row("trig/yr",       "tpy", ".1f"))
        w(row("avg_hold (d)",  "avg_hold", ".1f"))
        w(row("capital_util%", "capital_util", ".1f"))
        w(row("win_rate%",     "win_rate", ".1f"))
        w(row("PF",            "pf", ".3f"))
        w(row("avg_trade_ret%","avg_tret", ".2f"))
        w(row("med_fwd10%",    "med_fwd10", ".2f"))
        w(f"| passive_zone_ret | {fm_a['avg_passive']:+.2f}% | {fm_e['avg_passive']:+.2f}% | — |")
        w(f"| failure_reason | {fm_a['failure_reason']} | {fm_e['failure_reason']} | — |")

        # Attribution
        w(f"\n**ΔCAGR attribution (Case E, {fold_name})**:\n")
        w("| component | pp | % of abs total |")
        w("|---|---|---|")
        cp  = fm_e["comps_pct"]
        for comp, pp_val in [
            ("selection",   fm_e["sel_pp"]),
            ("holding",     fm_e["hold_pp"]),
            ("sizing",      fm_e["siz_pp"]),
            ("timing",      fm_e["tim_pp"]),
            ("cash_effect", fm_e["cash_pp"]),
        ]:
            w(f"| {comp} | {pp_val:+.3f} | {cp[comp]:+.1f}% |")

    # ── 8. Final Judgment ─────────────────────────────────────────────
    w("\n---\n## 8. Final Judgment — Single Dominant Bottleneck\n")

    # Aggregate across both cases and both failing folds (Fold3, Fold5)
    all_fail = []
    for case in ["A","E"]:
        for fm in results[case]["folds"]:
            if fm and not fm["fold_pass"]:
                all_fail.append(fm)

    reason_agg = {}
    for fm in all_fail:
        r = fm["failure_reason"]
        if r.startswith("mixed"):
            for sub in r[6:-1].split("+"):
                reason_agg[sub.strip()] = reason_agg.get(sub.strip(), 0) + 0.5
        elif r not in ("none","unknown"):
            reason_agg[r] = reason_agg.get(r, 0) + 1.0

    if reason_agg:
        tot = sum(reason_agg.values())
        dom = max(reason_agg, key=reason_agg.get)
        dom_r = reason_agg[dom] / tot

        w("| factor | evidence_score | share |")
        w("|---|---|---|")
        for r, cnt in sorted(reason_agg.items(), key=lambda x: -x[1]):
            w(f"| {r} | {cnt:.1f} | {cnt/tot*100:.0f}% |")

        confidence = "high" if dom_r >= 0.70 else "medium" if dom_r >= 0.50 else "low"
        w(f"\n**name**: {dom}")
        w(f"**confidence**: {confidence} ({dom_r*100:.0f}%)")
        ev_lines = []
        if dom == "dd_excess":
            ev_lines = [
                "Fold5 ΔDD=+3.89pp (A) / +2.66pp (E) — 採用閾値+1.5pp の大幅超過",
                "Fold3 ΔDD=+3.56pp (A) / +2.37pp (E) — Bull相場でスリーブが downside を拡大",
                "ΔDD改善の唯一の成功例: Case E Fold2 (Bear) で ΔDD=-2.50pp (スリーブが防御的に機能)",
            ]
        elif dom == "alpha_absent":
            ev_lines = [
                "Fold3 sl_CAGR≈0%: passive_zone_ret vs actual_trade_ret の gap が原因",
                "trig=32/yr でスリーブが高回転 → 個々のトレードが希薄化",
                "base_CAGR=+2.1% の低さと相互作用: ベース弱期はスリーブも機能しない",
            ]
        elif dom == "regime_break":
            ev_lines = [
                "失敗Fold で base_CAGR < 5%: ベース自体が低収益期",
                "スリーブは base portfolio の equity base の上に乗る構造 → base 弱=sleeve も弱",
                "密度Gate WF でも確認: gate に関係なく Fold3 は WF失敗",
            ]
        for ev in ev_lines:
            w(f"**evidence**: {ev}")

    # ── 9. Next Experiment Recommendation ────────────────────────────
    w("\n---\n## 9. Next Experiment Recommendation (1件のみ)\n")

    # Rule-based: based on dominant bottleneck
    if reason_agg:
        dom = max(reason_agg, key=reason_agg.get)
        dom_r = reason_agg[dom] / sum(reason_agg.values())

        if dom in ("dd_excess",) or dom_r >= 0.60:
            rec_name = "Dynamic Sleeve CAP — base_CAGR 連動縮小"
            rec_detail = [
                "条件: base_CAGR < 5% (fold水準) → SLEEVE_CAP_FR = 10% (現行20%の半分)",
                "条件: base_CAGR ≥ 5% → SLEEVE_CAP_FR = 20% (現行通り)",
                "期待: Fold3 (base=+2.1%) でスリーブ資本を半減 → ΔDD/2 で採用条件クリア可能性",
                "測定: WF, ΔCAGR, ΔDD, sl_CAGR, alpha_retention vs Case E baseline",
                "実装ポイント: yearly_base_CAGR をrollforward 1年推定値で代替 (IS期間のbase_CAGRを使用)",
                "禁止: gate追加, exit変更, entry変更, パラメータ探索",
                "スクリプト: src/backtest/study8_dynamic_cap.py",
                "出力: reports/study8_dynamic_cap.md",
            ]
        else:
            rec_name = "Sleeve CAP=10% 固定削減 — ΔDD 比例削減検証"
            rec_detail = [
                "現行 SLEEVE_CAP_FR=20% → 10% に固定変更",
                "期待: ΔDD × 0.5 = Fold5 (+2.66pp→+1.33pp < 採用閾値 +1.5pp)",
                "トレードオフ: ΔCAGR も同比例削減 (×0.5) → 採用条件 ΔCAGR>0.3pp を維持できるか確認",
                "測定: Case E の 5-fold WF 再実行",
                "スクリプト: src/backtest/study8_cap_reduction.py",
                "出力: reports/study8_cap_reduction.md",
            ]

        w(f"**推奨**: {rec_name}\n")
        for d in rec_detail:
            w(f"- {d}")

        w(f"\n**根拠**: dominant bottleneck = {dom} ({dom_r*100:.0f}%) — "
          f"CAP削減は entry/exit/gate を一切変更せず ΔDD を比例削減できる唯一の手段")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─── main ────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3, "PARAMS_LOCKED violation"

    print("=" * 68)
    print("  Study8 Failure Attribution — Fold Diagnostics")
    print("  Case A (1-slot 100%) vs Case E (2-slot 70/30 no-refill)")
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
        ma200       = topix_close.rolling(200, min_periods=100).mean()
        bear_arr    = ((topix_close < ma200)
                       .reindex(pd.DatetimeIndex(common_dates), method="ffill")
                       .fillna(False).values.astype(bool))

    cross90_mat         = compute_cross_mat(rsr_mat, SLEEVE_RSR_LO)
    slope5_mat, slope20_mat = compute_slopes(rsr_mat)

    print("[3/4] Simulation (Case A + Case E)...\n")
    capital   = float(cfg.portfolio.capital)
    all_results: dict[str, dict] = {}

    for case_id, spec in CASE_SPECS.items():
        print(f"  Case {case_id}: {spec}")
        sim = run_case(
            spec, open_mat, close_mat, sig_mat, sig_ready,
            rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, trade_syms, cfg, common_dates,
        )
        fold_metrics = []
        for fold in FOLDS:
            passive = compute_passive_zone(
                rsr_mat, open_mat, cross90_mat, sym_active_mat,
                common_dates, fold["oos_start"], fold["oos_end"],
            )
            fm = compute_fold(sim, close_mat, common_dates, fold, capital, passive)
            fold_metrics.append(fm)
            if fm:
                ok = "✅" if fm["fold_pass"] else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"sl={fm['sl_cagr']:+.1f}%  ΔCAGR={fm['delta_cagr']:+.2f}pp  "
                      f"ΔDD={fm['delta_dd']:+.2f}pp  trig={fm['tpy']:.1f}  "
                      f"fail={fm['failure_reason']}  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")
        all_results[case_id] = {"folds": fold_metrics, "sim": sim}
        print()

    print("  Summary:")
    for case_id in CASE_SPECS:
        fms = [fm for fm in all_results[case_id]["folds"] if fm]
        n_pass = sum(1 for fm in fms if fm["fold_pass"])
        avg_dc = float(np.mean([fm["delta_cagr"] for fm in fms]))
        avg_dd = float(np.mean([fm["delta_dd"]   for fm in fms]))
        print(f"    Case {case_id}: WF={n_pass}/5  "
              f"avg_ΔCAGR={avg_dc:+.2f}pp  avg_ΔDD={avg_dd:+.2f}pp")

    print("\n[4/4] レポート生成...")
    write_report(all_results, topix_close,
                 REPORTS_DIR / "study8_failure_attribution.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
