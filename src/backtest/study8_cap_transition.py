"""
src/backtest/study8_cap_transition.py
Study8 Adaptive CAP Transition WF — Persistence Isolation

目的: CAP_LO=10% の過剰削減原因を「切替頻度」vs「縮小深度」で分離
固定: CAP_LO=15% / state=rolling_dd / recovery=10営業日のみ

Cases:
  A  固定15% (anchor)
  B  DD>5%  persist=0d (即時)
  C  DD>7%  persist=0d (即時)
  D  DD>5%  persist=3d
  E  DD>7%  persist=3d
  F  DD>5%  persist=5d
  G  DD>7%  persist=5d

Recovery: DD<threshold AND 10営業日経過 (両条件必須)
採用: WF>=4/5, ΔCAGR>=+0.3pp, ΔDD<=+1.5pp, alpha_ret>=90%
停止: Case A (固定15%) が adaptive 全群を上回れば研究終了

Run: cd C:/ai-trading && python src/backtest/study8_cap_transition.py
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

CAP_HI       = 0.20
CAP_LO       = 0.15  # fixed per spec
RECOVERY_DAYS = 10

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

CASE_SPECS: dict[str, dict] = {
    "A": {"type": "fixed",   "dd_thr": None, "persist": 0,
          "desc": "固定15% (anchor)"},
    "B": {"type": "rolling_dd", "dd_thr": 0.05, "persist": 0,
          "desc": "DD>5%  即時"},
    "C": {"type": "rolling_dd", "dd_thr": 0.07, "persist": 0,
          "desc": "DD>7%  即時"},
    "D": {"type": "rolling_dd", "dd_thr": 0.05, "persist": 3,
          "desc": "DD>5%  3d継続"},
    "E": {"type": "rolling_dd", "dd_thr": 0.07, "persist": 3,
          "desc": "DD>7%  3d継続"},
    "F": {"type": "rolling_dd", "dd_thr": 0.05, "persist": 5,
          "desc": "DD>5%  5d継続"},
    "G": {"type": "rolling_dd", "dd_thr": 0.07, "persist": 5,
          "desc": "DD>7%  5d継続"},
}

ADOPT_WF = 4
ADOPT_DC = 0.3
ADOPT_DD = 1.5
ADOPT_AR = 90.0


# ─── helpers ──────────────────────────────────────────────────────────

def classify_state(s5, s20):
    if np.isnan(s5) or np.isnan(s20): return "UNKNOWN"
    if abs(s5) < STALL_THR: return "STALL"
    if s20 > 0:
        if s5 > s20:   return "EARLY_UP"
        if s5 > 0:     return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"

STATE_SCORE = {"EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1,
               "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0}

def cross_mat(rsr_mat, thr):
    n_d, n_s = rsr_mat.shape
    out = np.zeros((n_d, n_s), dtype=np.int32)
    run = np.zeros(n_s, dtype=np.int32)
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


# ─── adaptive CAP state machine ───────────────────────────────────────

class AdaptiveCap:
    """rolling_dd driven CAP with configurable threshold & persistence.
    Recovery: DD below threshold AND 10 business days elapsed since switch."""

    def __init__(self, spec: dict):
        self.ctype        = spec["type"]
        self.dd_thr       = spec.get("dd_thr", 0.05)
        self.persist_req  = spec.get("persist", 0)    # days DD must exceed thr
        self.sl_eq_peak   = 0.0
        self.in_reduced   = False
        self.reduced_since= -9999
        self.trigger_streak = 0                        # consecutive days DD>thr
        self.switches_total = 0
        # per-day history (full simulation length)
        self.cap_hist: list[float] = []
        self.dd_hist:  list[float] = []
        self.state_hist: list[bool] = []  # True=in_reduced

    def step(self, day_i: int, sl_eq: float) -> float:
        if self.ctype == "fixed":
            cap = CAP_LO
            self.cap_hist.append(cap)
            self.dd_hist.append(0.0)
            self.state_hist.append(False)
            return cap

        if sl_eq > self.sl_eq_peak:
            self.sl_eq_peak = sl_eq
        rdd = max(0.0, (self.sl_eq_peak - sl_eq) / max(1.0, self.sl_eq_peak))
        self.dd_hist.append(rdd)

        trigger = rdd > self.dd_thr

        # Streak counter
        if trigger:
            self.trigger_streak += 1
        else:
            self.trigger_streak = 0

        # Entry into reduced
        if not self.in_reduced:
            if self.persist_req == 0:
                enters = trigger
            else:
                enters = self.trigger_streak >= self.persist_req
            if enters:
                self.in_reduced    = True
                self.reduced_since = day_i
                self.switches_total += 1
        else:
            # Recovery: NOT trigger AND 10+ days elapsed
            elapsed = day_i - self.reduced_since
            if not trigger and elapsed >= RECOVERY_DAYS:
                self.in_reduced = False

        cap = CAP_LO if self.in_reduced else CAP_HI
        self.cap_hist.append(cap)
        self.state_hist.append(self.in_reduced)
        return cap


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

    rc       = getattr(cfg, "risk_controls", None)
    MSEC     = float(rc.sector_cap)  if rc else 0.25
    gross_en = bool(getattr(rc, "gross_exposure_enabled", True))  if rc else True
    g_norm   = float(getattr(rc, "gross_cap_normal",       1.0))  if rc else 1.0
    g_dd5    = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    g_dd8    = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    acap = AdaptiveCap(spec)

    base_cash  = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak  = float(capital)
    cb_active  = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    sl_slots: list[Position | None] = [None] * SLEEVE_MAX
    sl_cash  = float(capital * CAP_LO)  # init to lower bound
    sl_trades: list[dict] = []

    n_dates = len(common_dates)
    base_eq = np.zeros(n_dates)
    sl_eq   = np.zeros(n_dates)
    base_inv= np.zeros(n_dates)
    sl_inv  = np.zeros(n_dates)
    cap_arr = np.zeros(n_dates, dtype=np.float32)

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

        eff_cap = acap.step(i, s_eq_v)
        cap_arr[i] = eff_cap

        if b_eq_v > base_peak: base_peak = b_eq_v
        dd = (b_eq_v - base_peak) / base_peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05: cb_active = False; cb_days = 0

        gross_cap = g_norm
        if gross_en and topix_ret20 is not None:
            r20 = float(topix_ret20[i]) if i < len(topix_ret20) else 0.0
            r60 = float(topix_ret60[i]) if i < len(topix_ret60) else 0.0
            if r20 < -0.05: gross_cap = g_dd5
            elif r60 < -0.08: gross_cap = g_dd8

        is_bear   = bool(bear_arr[i]) if bear_arr is not None else False
        sec_eff   = 0.18 if is_bear else MSEC
        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n_dates: break
        ni = i + 1

        # Base sells
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
                    if pc > 0 and (cl/pc-1.0) <= SHOCK_SYM_THR:
                        b_sell.append((sym,"MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not hld: continue
            if hld and max_hold and hix > max_hold:
                b_sell.append((sym,"TIME_STOP")); continue
            if hld and rv < rsr_exit_thr and hix >= min_hold:
                b_sell.append((sym,"RSR_EXIT")); continue
            sig = int(sig_mat[i, si_]) if sig_ready[si_] else 0
            if sig == -1 and hld and hix >= min_hold:
                b_sell.append((sym,"STRATEGY_EXIT"))
            elif sig == 1 and not hld:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i,si_]) < 0.5: continue
                b_buy.append((rv, sym))

        for sym, reason in b_sell:
            if sym not in base_pos: continue
            pos = base_pos[sym]
            sp  = float(open_mat[ni, sym_to_i[sym]])
            base_cash += pos.qty * sp * (1 - COST_ONE_WAY)
            base_trades.append({"side":"SELL","symbol":sym,"pnl":(sp-pos.entry_price)*pos.qty,
                                 "entry":pos.entry_price,"exit":sp,"qty":pos.qty,
                                 "entry_idx":pos.entry_idx,"exit_idx":i,"reason":reason,"date":ds})
            del base_pos[sym]
            if reason=="TIME_STOP": reentry_ban[sym]=i+1+REENTRY_COOL

        if not cb_active and b_buy:
            b_buy.sort(key=lambda x:-x[0])
            for rv, sym in b_buy:
                si_ = sym_to_i[sym]
                if base_max_pos-len(base_pos)<=0: break
                bp = float(open_mat[ni, si_])
                if bp<=0: continue
                if not _sector_ok(sym,base_pos,close_mat,i,sym_to_i,trade_syms,capital,sec_eff): continue
                if gross_en:
                    cg = sum(p.qty*float(close_mat[i,sym_to_i[p.symbol]]) for p in base_pos.values())/max(1.0,capital)
                    if cg+bp*LOT/max(1.0,capital)>gross_cap: continue
                qty = int(capital/base_max_pos/bp/LOT)*LOT
                if qty<=0 or qty*bp*(1+COST_ONE_WAY)>base_cash: continue
                _execute_buy(sym,bp,qty,i,ni,trade_syms,base_trades,base_pos,rv)
                base_cash -= qty*bp*(1+COST_ONE_WAY)
                base_trades[-1]["date"] = ds

        # Sleeve exits
        for k in range(SLEEVE_MAX):
            pos = sl_slots[k]
            if pos is None: continue
            si_ = sym_to_i[pos.symbol]
            rv  = float(rsr_mat[i, si_])
            cl  = float(close_mat[i, si_])
            do_exit=False; reason=""
            if mkt_shock and i>0:
                pc=float(close_mat[i-1,si_])
                if pc>0 and (cl/pc-1.0)<=SHOCK_SYM_THR:
                    do_exit=True; reason="MARKET_SHOCK_EXIT"
            if not do_exit and rv<SLEEVE_RSR_LO:
                do_exit=True; reason="RSR_EXIT_LO"
            if do_exit:
                sp  = float(open_mat[ni, si_])
                pnl = (sp-pos.entry_price)*pos.qty
                sl_cash += pos.qty*sp*(1-COST_ONE_WAY)
                sl_trades.append({
                    "side":"SELL","symbol":pos.symbol,"pnl":pnl,
                    "entry":pos.entry_price,"exit":sp,"qty":pos.qty,
                    "entry_idx":pos.entry_idx,"exit_idx":i,
                    "hold_days":i-pos.entry_idx,"reason":reason,"date":ds,
                    "slot":k,"cap_at_entry":pos.rsr_at_entry,
                    "fwd10":np.nan,"fwd20":np.nan,
                })
                sl_slots[k]=None

        # Sleeve entries
        if not mkt_shock:
            n_occ=sum(1 for k in range(SLEEVE_MAX) if sl_slots[k] is not None)
            if NO_REFILL and 0<n_occ<SLEEVE_MAX:
                pass
            elif n_occ<SLEEVE_MAX:
                sl_syms={sl_slots[k].symbol for k in range(SLEEVE_MAX) if sl_slots[k] is not None}
                cands=[]
                for sym in active_syms:
                    if sym in base_pos or sym in sl_syms: continue
                    si_=sym_to_i[sym]
                    rv=float(rsr_mat[i,si_])
                    if not (SLEEVE_RSR_LO<=rv<SLEEVE_RSR_HI): continue
                    d90=int(cross90_mat[i,si_])
                    if d90>SLEEVE_D90_MAX: continue
                    if sym_active_mat is not None and float(sym_active_mat[i,si_])<0.5: continue
                    st=classify_state(float(slope5_mat[i,si_]),float(slope20_mat[i,si_]))
                    cands.append((rv,d90,STATE_SCORE.get(st,0),sym))
                if cands:
                    cands.sort(key=lambda x:(-x[0],x[1],-x[2]))
                    ci=0
                    for k in range(SLEEVE_MAX):
                        if sl_slots[k] is not None: continue
                        if ci>=len(cands): break
                        rv,d90,_,sym=cands[ci]; ci+=1
                        si_=sym_to_i[sym]
                        bp=float(open_mat[ni,si_])
                        if bp<=0: continue
                        target=b_eq_v*eff_cap*SL_WEIGHTS[k]
                        alloc=min(sl_cash,target)
                        qty=int(alloc/bp/LOT)*LOT
                        cost=qty*bp*(1+COST_ONE_WAY)
                        if qty<=0 or cost>sl_cash: continue
                        sl_cash-=cost
                        sl_slots[k]=Position(sym,trade_syms.get(sym,""),qty,bp,ni,eff_cap)
                        sl_trades.append({"side":"BUY","symbol":sym,"slot":k,"weight":SL_WEIGHTS[k],
                                           "entry":bp,"qty":qty,"cap_at_entry":eff_cap,
                                           "entry_idx":ni,"date":ds})

    # fwd returns
    for t in sl_trades:
        if t["side"]!="SELL": continue
        si_=sym_to_i.get(t["symbol"],-1)
        if si_<0: continue
        ei=t["exit_idx"]
        for fk,fn in [("fwd10",10),("fwd20",20)]:
            if ei+fn<n_dates:
                c0=float(close_mat[ei,si_]); cn=float(close_mat[ei+fn,si_])
                t[fk]=round((cn/c0-1)*100,2) if c0>0 and cn>0 else float("nan")

    return {
        "base_eq":base_eq,"sl_eq":sl_eq,"base_inv":base_inv,"sl_inv":sl_inv,
        "base_trades":base_trades,"sl_trades":sl_trades,
        "cap_arr":cap_arr,"acap":acap,
    }


# ─── fold metrics ─────────────────────────────────────────────────────

def compute_fold(sim, close_mat, common_dates, fold, capital, spec):
    oos_s, oos_e = fold["oos_start"], fold["oos_end"]
    ds_all  = [str(d.date()) for d in common_dates]
    oos_idx = [i for i,ds in enumerate(ds_all) if oos_s<=ds<=oos_e]
    if not oos_idx: return {}
    si, ei  = oos_idx[0], oos_idx[-1]+1
    n_days  = ei-si; n_years=n_days/252.0

    base_eq = sim["base_eq"][si:ei].tolist()
    sl_eq   = sim["sl_eq"][si:ei].tolist()
    comb_eq = (sim["base_eq"]+sim["sl_eq"])[si:ei].tolist()
    b_inv_o = sim["base_inv"][si:ei].tolist()
    s_inv_o = sim["sl_inv"][si:ei].tolist()
    cap_oos = sim["cap_arr"][si:ei]
    oos_dates=list(common_dates[si:ei])
    dd_hist = sim["acap"].dd_hist[si:ei]

    base_t = [t for t in sim["base_trades"] if oos_s<=t.get("date","")<=oos_e]
    sl_t   = [t for t in sim["sl_trades"]   if oos_s<=t.get("date","")<=oos_e]
    sl_sell= [t for t in sl_t if t["side"]=="SELL"]

    b_exp=[bi/max(1,be) for bi,be in zip(b_inv_o,base_eq)]
    bm=calc_metrics(base_eq,base_t,b_exp,base_eq[0],oos_dates)
    ce=[(bi+si_)/max(1,be+se) for bi,si_,be,se in zip(b_inv_o,s_inv_o,base_eq,sl_eq)]
    cm=calc_metrics(comb_eq,base_t+sl_t,ce,comb_eq[0],oos_dates)

    sl0   = sl_eq[0]
    unused= [be+sl0 for be in base_eq]
    um=calc_metrics(unused,base_t,b_exp,unused[0],oos_dates)

    delta_cagr=cm.get("cagr",0)-bm.get("cagr",0)
    comb_dd=cm.get("max_dd",0)
    delta_dd=-(comb_dd-bm.get("max_dd",0))

    se=[si_/max(1,se) for si_,se in zip(s_inv_o,sl_eq)]
    slm=calc_metrics(sl_eq,sl_t,se,sl_eq[0],oos_dates)
    sl_cagr=slm.get("cagr",0)

    n_trig=len(sl_sell)
    tpy=n_trig/max(0.01,n_years)
    holds=[t["hold_days"] for t in sl_sell if t.get("hold_days",0)>0]
    avg_hold=float(np.mean(holds)) if holds else 0.0

    avg_cap=float(np.mean(cap_oos)) if len(cap_oos)>0 else CAP_LO
    days_lo=int(np.sum(cap_oos<CAP_HI))

    # switches in OOS
    switches=0; prev=None
    for c in cap_oos:
        if prev is not None and c!=prev: switches+=1
        prev=c

    # state_persistence = fraction of days with DD > threshold
    dd_thr = spec.get("dd_thr", 0.0)
    if dd_hist and dd_thr:
        state_persist = sum(1 for d in dd_hist if d>dd_thr)/max(1,len(dd_hist))
    else:
        state_persist = float(days_lo)/max(1,n_days)

    wins=[t for t in sl_sell if (t.get("pnl") or 0)>0]
    loss=[t for t in sl_sell if (t.get("pnl") or 0)<=0]
    gp=sum(t["pnl"] for t in wins) if wins else 0.0
    gl=abs(sum(t["pnl"] for t in loss)) if loss else 0.0
    pf=gp/max(1.0,gl)
    wr=len(wins)/max(1,len(sl_sell))*100

    fold_pass=(delta_cagr>ADOPT_DC and delta_dd<=ADOPT_DD)
    return {
        "base_cagr":    round(bm.get("cagr",0),2),
        "sl_cagr":      round(sl_cagr,2),
        "comb_cagr":    round(cm.get("cagr",0),2),
        "delta_cagr":   round(delta_cagr,2),
        "delta_dd":     round(delta_dd,2),
        "n_trig":n_trig,"tpy":round(tpy,1),
        "avg_hold":round(avg_hold,1),
        "avg_cap":round(avg_cap,4),
        "days_lo":days_lo,"switches":switches,
        "state_persist":round(state_persist,3),
        "wr":round(wr,1),"pf":round(pf,3),
        "fold_pass":fold_pass,
    }


# ─── WF aggregate ─────────────────────────────────────────────────────

def agg_wf(folds: list[dict], sl_cagr_ref: float | None) -> dict:
    v=[f for f in folds if f]
    if not v: return {}
    n_pass=sum(1 for f in v if f["fold_pass"])
    avg_dc=float(np.mean([f["delta_cagr"] for f in v]))
    avg_dd=float(np.mean([f["delta_dd"]   for f in v]))
    avg_sl=float(np.mean([f["sl_cagr"]    for f in v]))
    avg_cap=float(np.mean([f["avg_cap"]   for f in v]))
    avg_sw=float(np.mean([f["switches"]   for f in v]))
    avg_lo=float(np.mean([f["days_lo"]    for f in v]))
    avg_sp=float(np.mean([f["state_persist"] for f in v]))
    alpha_ret=(avg_sl/sl_cagr_ref*100) if sl_cagr_ref and sl_cagr_ref>0 else float("nan")
    cap_red=CAP_HI-avg_cap
    adopted=(n_pass>=ADOPT_WF and avg_dc>ADOPT_DC and avg_dd<=ADOPT_DD
             and (math.isnan(alpha_ret) or alpha_ret>=ADOPT_AR))
    return {
        "n_pass":n_pass,"avg_dc":round(avg_dc,2),"avg_dd":round(avg_dd,2),
        "avg_sl":round(avg_sl,2),"avg_cap":round(avg_cap,4),
        "avg_sw":round(avg_sw,1),"avg_lo":round(avg_lo,1),
        "avg_sp":round(avg_sp,3),
        "alpha_ret":round(alpha_ret,1) if not math.isnan(alpha_ret) else float("nan"),
        "cap_red":round(cap_red,4),"adopted":adopted,
    }


# ─── audit metrics ────────────────────────────────────────────────────

def compute_audit(results: dict, ref_wf: dict) -> dict:
    """Compute cross-case audit: alpha_loss_per_switch, dd_saved_per_switch,
    switch_half_life, transition_efficiency."""
    dc_ref = ref_wf.get("avg_dc", 0)
    dd_ref = ref_wf.get("avg_dd", 0)
    out = {}
    for case_id in CASE_SPECS:
        wf = results[case_id]["wf"]
        sw = wf.get("avg_sw", 0)
        lo = wf.get("avg_lo", 0)
        dc = wf.get("avg_dc", dc_ref)
        dd = wf.get("avg_dd", dd_ref)
        alpha_loss = max(0, dc_ref - dc)      # ΔCAGR lost vs Case A (fixed 20% ref)
        dd_saved   = max(0, dd_ref - dd)      # ΔDD saved vs Case A
        alpha_lps  = alpha_loss / max(1, sw)
        dd_sps     = dd_saved  / max(1, sw)
        half_life  = lo / max(1, sw)
        t_eff      = dd_sps / max(0.01, alpha_lps)
        out[case_id] = {
            "alpha_loss_ps": round(alpha_lps, 4),
            "dd_saved_ps":   round(dd_sps,   4),
            "half_life":     round(half_life, 1),
            "t_eff":         round(t_eff,     3),
        }
    return out


# ─── report ────────────────────────────────────────────────────────────

def _f(v, fmt=".2f"):
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:{fmt}}"

def regime_label(tc, yr):
    if tc is None: return "N/A"
    s=tc[tc.index.year==yr]
    if len(s)<2: return "N/A"
    r=float(s.iloc[-1]/s.iloc[0]-1)
    return f"{'Bull' if r>0 else 'Bear'} ({r*100:+.1f}%)"

def write_report(results, topix_close, output_path, audit,
                 stop_met, sl_cagr_ref):
    L=[]; w=L.append
    adopted=[c for c in CASE_SPECS if results[c]["wf"].get("adopted")]

    w("# Study8 Adaptive CAP Transition WF — Persistence Isolation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  解析専用 / 実装変更禁止")
    w(f"\n固定: 2-slot 70/30 no-refill / ENTRY RSR[92,95) d90≤5 / EXIT RSR<90")
    w(f"\nCAP_HI={CAP_HI*100:.0f}% / CAP_LO={CAP_LO*100:.0f}% / 復帰条件=DD解除+10d経過")
    w(f"\n採用条件: WF≥{ADOPT_WF}/5, ΔCAGR>+{ADOPT_DC}pp, ΔDD≤+{ADOPT_DD}pp, α_ret≥{ADOPT_AR}%\n")
    if stop_met:
        w("🛑 **停止条件発動**: 固定15%(Case A)がadaptive全群を上回る → adaptive研究終了\n")
    w(f"**採用**: {len(adopted)}件  "
      f"| 判定: **{'✅ ADOPT' if adopted else '🛑 STOP' if stop_met else '🔬 RESEARCH'}**\n")

    # 1. WF Summary
    w("---\n## 1. WF サマリ\n")
    w("| Case | 説明 | WF | ΔCAGR | ΔDD | sl_CAGR | α_ret | avg_cap | avg_sw | state_persist | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for case, spec in CASE_SPECS.items():
        wf=results[case]["wf"]
        ar=(_f(wf.get("alpha_ret",float("nan")),".1f")+"%"
            if not math.isnan(wf.get("alpha_ret",float("nan"))) else "—")
        ok="**✅**" if wf.get("adopted") else "❌"
        w(f"| {case} | {spec['desc']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_dc',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_sl',0):+.1f}% "
          f"| {ar} | {wf.get('avg_cap',0)*100:.1f}% "
          f"| {wf.get('avg_sw',0):.1f} "
          f"| {wf.get('avg_sp',0)*100:.1f}% "
          f"| {ok} |")

    # 2. Fold Detail
    w("\n---\n## 2. Fold 詳細\n")
    w("| Case | Fold | OOS | Regime | ΔCAGR | ΔDD | sl_CAGR | avg_cap | days_lo | switches | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in CASE_SPECS:
        for fold, fm in zip(FOLDS, results[case]["folds"]):
            if not fm: continue
            yr=int(fold["oos_start"][:4])
            reg=regime_label(topix_close,yr)
            ok="✅" if fm["fold_pass"] else "❌"
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['avg_cap']*100:.1f}% "
              f"| {fm['days_lo']}d "
              f"| {fm['switches']} "
              f"| {ok} |")

    # 3. Persistence Analysis
    w("\n---\n## 3. Persistence vs Threshold Analysis\n")
    w("> 切替条件の「閾値」×「継続日数」マトリクス\n")
    w("| | DD>5% 即時(B) | DD>5% 3d(D) | DD>5% 5d(F) |")
    w("|---|---|---|---|")
    for metric, label in [("avg_dc","avg ΔCAGR"), ("avg_dd","avg ΔDD"),
                           ("avg_sw","avg switches"), ("n_pass","WF")]:
        row = []
        for case in ["B","D","F"]:
            wf=results[case]["wf"]
            v=wf.get(metric,0)
            row.append(f"{v:+.2f}pp" if metric in ("avg_dc","avg_dd") and isinstance(v,float)
                       else f"{v:.1f}" if isinstance(v,float) else str(v))
        w(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")

    w("\n| | DD>7% 即時(C) | DD>7% 3d(E) | DD>7% 5d(G) |")
    w("|---|---|---|---|")
    for metric, label in [("avg_dc","avg ΔCAGR"), ("avg_dd","avg ΔDD"),
                           ("avg_sw","avg switches"), ("n_pass","WF")]:
        row = []
        for case in ["C","E","G"]:
            wf=results[case]["wf"]
            v=wf.get(metric,0)
            row.append(f"{v:+.2f}pp" if metric in ("avg_dc","avg_dd") and isinstance(v,float)
                       else f"{v:.1f}" if isinstance(v,float) else str(v))
        w(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")

    # 4. Transition Efficiency Audit
    w("\n---\n## 4. Transition Efficiency Audit\n")
    w("| Case | alpha_loss/switch | dd_saved/switch | switch_half_life | transition_eff |")
    w("|---|---|---|---|---|")
    for case in CASE_SPECS:
        a=audit[case]
        w(f"| {case} "
          f"| {a['alpha_loss_ps']:+.4f}pp "
          f"| {a['dd_saved_ps']:+.4f}pp "
          f"| {a['half_life']:.1f}d "
          f"| {a['t_eff']:.3f} |")

    # 5. Fold3 / Fold5 Target
    w("\n---\n## 5. Fold3 / Fold5 — WF Barrier Analysis\n")
    for fidx, fname in [(2,"Fold3 (2023)"),(4,"Fold5 (2025)")]:
        w(f"\n### {fname}\n")
        w("| Case | ΔCAGR | ΔDD | sl_CAGR | avg_cap | days_lo | switches | pass |")
        w("|---|---|---|---|---|---|---|---|")
        for case in CASE_SPECS:
            fm=results[case]["folds"][fidx]
            if not fm: continue
            ok="✅" if fm["fold_pass"] else "❌"
            w(f"| {case} {ok} "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['avg_cap']*100:.1f}% "
              f"| {fm['days_lo']}d "
              f"| {fm['switches']} |")

    # 6. Stop Condition
    w("\n---\n## 6. Stop Condition Assessment\n")
    wf_a=results["A"]["wf"]
    adapt=[c for c in CASE_SPECS if CASE_SPECS[c]["type"]!="fixed"]
    w(f"Case A (固定15%): WF={wf_a.get('n_pass',0)}/5  "
      f"ΔCAGR={wf_a.get('avg_dc',0):+.2f}pp  ΔDD={wf_a.get('avg_dd',0):+.2f}pp\n")
    w("| adaptive Case | ΔCAGR vs A | ΔDD vs A | WF vs A | A wins? |")
    w("|---|---|---|---|---|")
    for case in adapt:
        wf=results[case]["wf"]
        dc_d=wf.get("avg_dc",0)-wf_a.get("avg_dc",0)
        dd_d=wf.get("avg_dd",0)-wf_a.get("avg_dd",0)
        wf_d=wf.get("n_pass",0)-wf_a.get("n_pass",0)
        a_wins=(wf_a.get("avg_dc",0)>=wf.get("avg_dc",0)
                and wf_a.get("avg_dd",0)<=wf.get("avg_dd",0)
                and wf_a.get("n_pass",0)>=wf.get("n_pass",0))
        w(f"| {case} | {dc_d:+.2f}pp | {dd_d:+.2f}pp | {wf_d:+d} | {'⚠ Yes' if a_wins else 'No'} |")
    w(f"\n**停止条件**: {'**発動 → adaptive研究終了**' if stop_met else '未発動 → 継続可'}")

    # 7. Failure Analysis
    w("\n---\n## 7. Failure Analysis\n")
    for case,spec in CASE_SPECS.items():
        wf=results[case]["wf"]
        fails=[]
        if wf.get("n_pass",0)<ADOPT_WF: fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF}")
        if wf.get("avg_dc",0)<=ADOPT_DC: fails.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp")
        if wf.get("avg_dd",0)>ADOPT_DD: fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp > {ADOPT_DD}")
        ar=wf.get("alpha_ret",float("nan"))
        if not math.isnan(ar) and ar<ADOPT_AR: fails.append(f"α_ret={ar:.1f}%<{ADOPT_AR}%")
        if not fails:
            w(f"\n**{case}** ✅ 全基準クリア")
        else:
            w(f"\n**{case}** REJECT: "+" / ".join(fails))

    # 8. Final Output
    w("\n---\n## 8. Final Output\n")

    # optimal switch frequency
    adapt_wfs=[(c,results[c]["wf"]) for c in adapt]
    best_c,best_wf=max(adapt_wfs,
                       key=lambda x:(x[1].get("n_pass",0),
                                     x[1].get("avg_dc",0),
                                     -x[1].get("avg_dd",99)))
    w("### optimal switch frequency\n")
    w(f"| 項目 | 値 |")
    w("|---|---|")
    w(f"| best case | **{best_c}** — {CASE_SPECS[best_c]['desc']} |")
    w(f"| avg_switches/fold | {best_wf.get('avg_sw',0):.1f} |")
    w(f"| avg_days_lo/fold | {best_wf.get('avg_lo',0):.1f}d |")
    w(f"| switch_half_life | {audit[best_c]['half_life']:.1f}d |")
    w(f"| state_persistence | {best_wf.get('avg_sp',0)*100:.1f}% |")
    w(f"| WF | {best_wf.get('n_pass',0)}/5 |")
    w(f"| ΔCAGR | {best_wf.get('avg_dc',0):+.2f}pp |")
    w(f"| ΔDD | {best_wf.get('avg_dd',0):+.2f}pp |")

    # best persistence (by threshold × persistence combination)
    w("\n### best persistence\n")
    # Sort adaptive by transition_efficiency
    sorted_by_eff=sorted(adapt,key=lambda c:-audit[c]["t_eff"])
    best_persist=sorted_by_eff[0]
    w(f"| 項目 | 値 |")
    w("|---|---|")
    w(f"| case | **{best_persist}** — {CASE_SPECS[best_persist]['desc']} |")
    w(f"| transition_efficiency | {audit[best_persist]['t_eff']:.3f} |")
    w(f"| dd_saved/switch | {audit[best_persist]['dd_saved_ps']:+.4f}pp |")
    w(f"| alpha_loss/switch | {audit[best_persist]['alpha_loss_ps']:+.4f}pp |")
    w(f"| half_life | {audit[best_persist]['half_life']:.1f}d |")

    # threshold effect summary
    w("\n**閾値比較** (5% vs 7%, 即時):\n")
    b_wf=results["B"]["wf"]; c_wf=results["C"]["wf"]
    w(f"- 5% 即時(B): ΔCAGR={b_wf.get('avg_dc',0):+.2f}pp  ΔDD={b_wf.get('avg_dd',0):+.2f}pp  sw={b_wf.get('avg_sw',0):.1f}")
    w(f"- 7% 即時(C): ΔCAGR={c_wf.get('avg_dc',0):+.2f}pp  ΔDD={c_wf.get('avg_dd',0):+.2f}pp  sw={c_wf.get('avg_sw',0):.1f}")

    w("\n**継続日数比較** (5% threshold):\n")
    for persist_case in ["B","D","F"]:
        wf=results[persist_case]["wf"]
        ps=CASE_SPECS[persist_case]["persist"]
        w(f"- persist={ps}d (Case {persist_case}): ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
          f"ΔDD={wf.get('avg_dd',0):+.2f}pp  sw={wf.get('avg_sw',0):.1f}")

    # recommend next study
    w("\n### recommend next study\n")
    if stop_met:
        w("**🛑 停止条件発動 → adaptive研究終了**\n")
        w("固定15%が全adaptive を上回る。state-driven CAP に追加改善余地なし。")
        w("\n**次研究推奨**: Study8 CAP固定最適値確定 (10〜15% sweep WF)")
        w("- スクリプト: `src/backtest/study8_cap_sweep.py`")
        w("- 内容: SLEEVE_CAP_FR = [0.10, 0.12, 0.14, 0.15] × 5-fold WF")
        w("- 目標: ΔDD≤+1.5pp かつ α_ret≥90% を満たす最大 CAP を特定")
        w("- 採用後: production に SLEEVE_CAP_FR を変更 (ASK_FIRST)")
    elif adopted:
        w(f"**✅ 採用確定: Case {adopted[0]}** — shadow 30日後 production 組み込み\n")
        w(f"- signal_bridge.py に rolling_dd CAP ロジック追加 (DD_THR={CASE_SPECS[adopted[0]].get('dd_thr',0)*100:.0f}%, persist={CASE_SPECS[adopted[0]].get('persist',0)}d)")
        w(f"- cap_state.json で状態管理 (idempotent)")
        w(f"- 変更禁止: ENTRY / EXIT / GATE / BASE PARAMS")
    else:
        # Near-threshold guidance
        near=[c for c in adapt if results[c]["wf"].get("n_pass",0)>=3
              and results[c]["wf"].get("avg_dd",99)<=2.0]
        if near:
            best_near=max(near,key=lambda c:results[c]["wf"].get("avg_dc",0))
            bwf_near=results[best_near]["wf"]
            w(f"**WF 3/5 ボーダー残存 — 追加分離研究を推奨**\n")
            w(f"最良 adaptive: Case {best_near} ({CASE_SPECS[best_near]['desc']})")
            w(f"- ΔCAGR={bwf_near.get('avg_dc',0):+.2f}pp  ΔDD={bwf_near.get('avg_dd',0):+.2f}pp  WF={bwf_near.get('n_pass',0)}/5")
            w(f"\n**次研究推奨**: Fold3/Fold5 barrier の分離")
            w(f"- Fold3(2023): alpha_absent が binding → CAP削減は効果限定的")
            w(f"- Fold5(2025): ΔDD excess → Case {best_near} の ΔDD改善を確認")
            w(f"- 仮説: Case {best_near} で Fold5 ΔDD ≤ +1.5pp に収束するかを検証")
            w(f"- スクリプト: `src/backtest/study8_cap_transition_v2.py`")
            w(f"- 変更禁止: ENTRY / EXIT / GATE / PARAMS_LOCKED")
        else:
            w("**全Case WF失敗 — adaptive CAP アプローチの根本限界**\n")
            w("推奨: Study8 研究を ENTRY/EXIT 改善軸に転換")
            w("- RSR exit 閾値の動的調整 (RSR<90 → RSR<85 in DD期) などを新研究軸として検討")

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─── main ──────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("="*68)
    print("  Study8 Adaptive CAP Transition WF — Persistence Isolation")
    print(f"  CAP_HI={CAP_HI*100:.0f}% → CAP_LO={CAP_LO*100:.0f}% | recovery={RECOVERY_DAYS}d")
    print("="*68+"\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s:v for s,v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s:idx for idx,s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    common_dates=None
    for sym in active_syms:
        idx=universe_raw[sym]["df"].index
        common_dates=idx if common_dates is None else common_dates.intersection(idx)
    common_dates=common_dates.sort_values()
    common_dates=common_dates[
        (common_dates>=pd.Timestamp(IS_START))&
        (common_dates<=pd.Timestamp("2025-12-31"))
    ]
    n_dates=len(common_dates)
    print(f"  共通日数: {n_dates}, 銘柄数: {n_syms}")

    print("[2/4] マトリクス構築...")
    open_mat  = np.full((n_dates,n_syms),np.nan,dtype=np.float32)
    close_mat = np.full((n_dates,n_syms),np.nan,dtype=np.float32)
    sig_mat   = np.zeros((n_dates,n_syms),dtype=np.int8)
    sig_ready = np.zeros(n_syms,dtype=bool)

    for si,sym in enumerate(active_syms):
        df_src =universe_raw[sym]["df"]
        row_idx=df_src.index.get_indexer(common_dates)
        if np.any(row_idx<0): continue
        open_mat[:,si] =df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:,si]=df_src["Close"].to_numpy(dtype=np.float32)[row_idx]
        rule =SECTOR_STRATEGY.get(trade_syms.get(sym,""),"fujiko")
        rsr_s=rsr_df[sym] if sym in rsr_df.columns else None
        st=(MeanReversionStrategy(**MR_PARAMS) if rule=="mean_rev"
            else FujikoStrategy(
                min_rsr=cfg.fujiko.min_rsr,turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s,min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period,turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry))
        req=252+getattr(st,"mom_period",21)+2
        if hasattr(st,"precompute_signals") and len(df_src)>=req:
            sig_mat[:,si]=st.precompute_signals(df_src).to_numpy(dtype=np.int8)[row_idx]
            sig_ready[si]=True

    rsr_mat=np.nan_to_num(
        _take(rsr_df,common_dates,active_syms,dtype=np.float32,fill_value=np.nan),nan=0.0)
    sym_active_mat=(None if sym_active_df is None else
                    _take(sym_active_df,common_dates,active_syms,dtype=np.float32,fill_value=1.0))

    mkt_ret1=topix_ret20=topix_ret60=bear_arr=None
    if topix_close is not None:
        mkt_ret1   =_take(topix_close.pct_change(),   common_dates,dtype=np.float32,fill_value=0.0)
        topix_ret20=_take(topix_close.pct_change(20), common_dates,dtype=np.float32,fill_value=0.0)
        topix_ret60=_take(topix_close.pct_change(60), common_dates,dtype=np.float32,fill_value=0.0)
        ma200=topix_close.rolling(200,min_periods=100).mean()
        bear_arr=((topix_close<ma200)
                  .reindex(pd.DatetimeIndex(common_dates),method="ffill")
                  .fillna(False).values.astype(bool))

    cross90_mat        =cross_mat(rsr_mat,SLEEVE_RSR_LO)
    slope5_mat,slope20_mat=slope_mats(rsr_mat)

    print(f"\n[3/4] WF Simulation ({len(CASE_SPECS)} cases × 5 folds)...\n")
    capital=float(cfg.portfolio.capital)
    all_results:dict[str,dict]={}
    sl_cagr_ref:float|None=None  # Case A fixed 20% reference for alpha_ret

    # Reference: Case A of PREVIOUS study (fixed 20%) = +36.7% sl_CAGR
    # We'll compute it from this run's equivalent (we don't have it here,
    # so we use Case A of this run as self-reference for alpha_ret)
    # alpha_ret = sl_cagr / ref_sl_cagr ; ref = first case computed

    for case_id,spec in CASE_SPECS.items():
        print(f"  [{case_id}] {spec['desc']}")
        sim=run_case(
            spec,open_mat,close_mat,sig_mat,sig_ready,
            rsr_mat,sym_active_mat,
            mkt_ret1,topix_ret20,topix_ret60,bear_arr,
            cross90_mat,slope5_mat,slope20_mat,
            active_syms,sym_to_i,trade_syms,cfg,common_dates,
        )
        folds=[]
        for fold in FOLDS:
            fm=compute_fold(sim,close_mat,common_dates,fold,capital,spec)
            folds.append(fm)
            if fm:
                ok="✅" if fm["fold_pass"] else "❌"
                cap_s=f"cap={fm['avg_cap']*100:.0f}%  lo={fm['days_lo']}d  sw={fm['switches']}"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"Δ={fm['delta_cagr']:+.2f}pp  ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"sl={fm['sl_cagr']:+.1f}%  {cap_s}  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")

        wf=agg_wf(folds,sl_cagr_ref)
        if case_id=="A":
            sl_cagr_ref=wf.get("avg_sl")  # fixed-15% sl_CAGR as reference

        all_results[case_id]={"folds":folds,"wf":wf,"sim":sim}
        ar_s=(f"  α_ret={wf.get('alpha_ret',0):.1f}%"
              if not math.isnan(wf.get("alpha_ret",float("nan"))) else "")
        ok="✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"cap={wf.get('avg_cap',0)*100:.1f}%  "
              f"sw={wf.get('avg_sw',0):.1f}  "
              f"lo={wf.get('avg_lo',0):.0f}d{ar_s}  {ok}\n")

    # Re-aggregate with self-reference
    if sl_cagr_ref is not None:
        for cid in list(CASE_SPECS)[1:]:
            all_results[cid]["wf"]=agg_wf(all_results[cid]["folds"],sl_cagr_ref)

    # Stop condition: Case A beats ALL adaptive cases
    wf_a=all_results["A"]["wf"]
    adapt=[c for c in CASE_SPECS if CASE_SPECS[c]["type"]!="fixed"]
    stop_met=all(
        wf_a.get("n_pass",0)>=all_results[c]["wf"].get("n_pass",0)
        and wf_a.get("avg_dc",0)>=all_results[c]["wf"].get("avg_dc",0)
        and wf_a.get("avg_dd",0)<=all_results[c]["wf"].get("avg_dd",0)
        for c in adapt
    )

    # Compute audit metrics (vs Case A fixed-15% baseline)
    audit=compute_audit(all_results,wf_a)

    # Summary
    print("="*68)
    adopted=[c for c in CASE_SPECS if all_results[c]["wf"].get("adopted")]
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for cid in CASE_SPECS:
        wf=all_results[cid]["wf"]
        ok="✅" if wf.get("adopted") else "❌"
        ar=wf.get("alpha_ret",float("nan"))
        ar_s=f"  α_ret={ar:.1f}%" if not math.isnan(ar) else ""
        print(f"  {ok} [{cid}] WF={wf.get('n_pass',0)}/5  "
              f"Δ={wf.get('avg_dc',0):+.2f}pp  DD={wf.get('avg_dd',0):+.2f}pp  "
              f"cap={wf.get('avg_cap',0)*100:.1f}%  sw={wf.get('avg_sw',0):.1f}{ar_s}")

    print(f"\n  停止条件: {'発動' if stop_met else '未発動'}")
    print(f"\n  transition efficiency (dd_saved/alpha_loss per switch):")
    for cid in adapt:
        print(f"    [{cid}] t_eff={audit[cid]['t_eff']:.3f}  "
              f"dd_saved={audit[cid]['dd_saved_ps']:+.4f}pp  "
              f"alpha_loss={audit[cid]['alpha_loss_ps']:+.4f}pp  "
              f"half_life={audit[cid]['half_life']:.1f}d")
    print("="*68+"\n")

    print("[4/4] レポート生成...")
    write_report(all_results,topix_close,
                 REPORTS_DIR/"study8_cap_transition.md",
                 audit,stop_met,sl_cagr_ref)
    return 0


if __name__=="__main__":
    sys.exit(main())
