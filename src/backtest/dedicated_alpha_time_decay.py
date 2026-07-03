"""
backtest/dedicated_alpha_time_decay.py  —  Study 8D

固定:
  ENTRY = RSR[92,95) d90≤5
  EXIT  = RSR<90 3日連続 (Study 8C Case C)

変更:  allocation decay のみ (6 cases)
  A: 25% fixed (baseline)
  B: 25%→15% after hold≥5d
  C: 25%→10% after hold≥5d
  D: 25%→15% after hold≥3d
  E: 25%→10% after hold≥3d
  F: 20% fixed

追加計測:
  time_weighted_exposure (早期/後期 exposure 分離)
  late_period_drawdown   (hold≥decay 日の DD)
  ret_after_day5         (5日保有以降のリターン)

採用:
  WF≥4/5, ΔCAGR≥+0.3pp, ΔDD≤+1.5pp, alpha_retention≥90%

出力: reports/dedicated_alpha_time_decay.md

Run:
    cd C:/ai-trading
    python src/backtest/dedicated_alpha_time_decay.py
"""

from __future__ import annotations

import sys, time, math, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy,
    Position, _take, calc_metrics,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"
STALL_THR   = 0.15

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Fixed conditions ──────────────────────────────────────────────────
ENTRY_RSR_LO    = 92.0
ENTRY_RSR_HI    = 95.0
ENTRY_D90_MAX   = 5
EXIT_THR        = 90.0   # RSR<90
EXIT_CONSEC     = 3      # 3日連続
SHOCK_MKT_THR   = -0.05
SHOCK_SYM_THR   = -0.08

LATE_BOUNDARY   = 5      # days: "late" starts at hold_day >= 5

STATE_SCORE_MAP = {
    "EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1,
    "FLAT": 1, "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0,
}

# ── Decay cases ───────────────────────────────────────────────────────
@dataclass
class DecaySpec:
    label:        str
    initial_frac: float       # initial alloc / base_equity
    target_frac:  float       # post-decay alloc / base_equity
    decay_days:   int | None  # hold days threshold; None = no decay

DECAY_CASES: dict[str, DecaySpec] = {
    "A": DecaySpec("25% fixed",            0.25, 0.25, None),
    "B": DecaySpec("25%→15% after hold≥5d", 0.25, 0.15,    5),
    "C": DecaySpec("25%→10% after hold≥5d", 0.25, 0.10,    5),
    "D": DecaySpec("25%→15% after hold≥3d", 0.25, 0.15,    3),
    "E": DecaySpec("25%→10% after hold≥3d", 0.25, 0.10,    3),
    "F": DecaySpec("20% fixed",            0.20, 0.20, None),
}

ADOPT_WF          = 4
ADOPT_DCAGR       = 0.3    # pp
ADOPT_DDD         = 1.5    # pp
ADOPT_ALPHA_RET   = 0.90   # fraction of Case A sl_CAGR


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    n_dates, n_syms = rsr_mat.shape
    out = np.zeros((n_dates, n_syms), dtype=np.int32)
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


def classify_state(s5: float, s20: float) -> str:
    if np.isnan(s5) or np.isnan(s20): return "UNKNOWN"
    if abs(s5) < STALL_THR:            return "STALL"
    if s20 > 0:
        if s5 > s20: return "EARLY_UP"
        if s5 > 0:   return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"


def regime_label(topix_close: pd.Series | None, year: int) -> str:
    if topix_close is None: return "N/A"
    yr  = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ─────────────────────────────────────────────────────────────────────
#  BASE SIMULATION
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BaseState:
    equity:   np.ndarray
    invested: np.ndarray
    held:     list
    trades:   list


def run_base(
    open_mat, close_mat,
    sig_mat, sig_ready, rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    active_syms, sym_to_i, trade_syms, cfg, common_dates,
) -> BaseState:
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    rc   = getattr(cfg, "risk_controls", None)
    MS   = float(rc.sector_cap) if rc else 0.25
    gen  = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gn   = float(getattr(rc, "gross_cap_normal", 1.0)) if rc else 1.0
    gd5  = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gd8  = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    cash = float(capital); pos: dict[str, Position] = {}
    peak = float(capital); cb_active = False; cb_days = 0
    reentry_ban: dict[str, int] = {}; trades: list = []
    n   = len(common_dates)
    eq  = np.zeros(n, dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)
    held = [set() for _ in range(n)]

    for i, date in enumerate(common_dates):
        ds    = str(date.date())
        b_inv = sum(p.qty*float(close_mat[i,sym_to_i[s]]) for s,p in pos.items())
        b_eq  = cash + b_inv
        eq[i] = b_eq; inv[i] = b_inv; held[i] = set(pos.keys())

        if b_eq > peak: peak = b_eq
        dd = (b_eq - peak) / peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05: cb_active = False; cb_days = 0

        gc = gn
        if gen and topix_ret20 is not None:
            r20 = float(topix_ret20[i]); r60 = float(topix_ret60[i])
            if r20 < -0.05: gc = gd5
            elif r60 < -0.08: gc = gd8

        bear  = bool(bear_arr[i]) if bear_arr is not None else False
        scap  = 0.18 if bear else MS
        shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n: break
        nxt = i + 1
        sells, buys = [], []

        for sym in active_syms:
            si = sym_to_i[sym]; h = sym in pos
            hd = (i - pos[sym].entry_idx) if h else 0
            rv = float(rsr_mat[i,si]); ct = float(close_mat[i,si])

            if shock and h:
                if i > 0:
                    pc = float(close_mat[i-1,si])
                    if pc > 0 and (ct/pc-1) <= SHOCK_SYM_THR:
                        sells.append((sym,"MARKET_SHOCK_EXIT")); continue
            if shock and not h: continue
            if h and max_hold and hd > max_hold: sells.append((sym,"TIME_STOP")); continue
            if h and rv < rsr_exit_thr and hd >= min_hold: sells.append((sym,"RSR_EXIT")); continue

            sig = int(sig_mat[i,si]) if sig_ready[si] else 0
            if sig == -1 and h and hd >= min_hold: sells.append((sym,"STRATEGY_EXIT"))
            elif sig == 1 and not h:
                if i < reentry_ban.get(sym,-1): continue
                if sym_active_mat is not None and float(sym_active_mat[i,si]) < 0.5: continue
                buys.append((rv,sym))

        for sym, reason in sells:
            if sym not in pos: continue
            p = pos[sym]; si = sym_to_i[sym]
            sp = float(open_mat[nxt,si])
            cash += p.qty*sp*(1-COST_ONE_WAY)
            trades.append({"side":"SELL","symbol":sym,"pnl":(sp-p.entry_price)*p.qty,
                           "entry":p.entry_price,"exit":sp,"qty":p.qty,
                           "entry_idx":p.entry_idx,"exit_idx":i,"reason":reason,"date":ds})
            del pos[sym]
            if reason=="TIME_STOP": reentry_ban[sym] = i+1+REENTRY_COOL

        if not cb_active and buys:
            buys.sort(key=lambda x:-x[0])
            for rv,sym in buys:
                si = sym_to_i[sym]
                if max_pos-len(pos) <= 0: break
                bp = float(open_mat[nxt,si])
                if bp <= 0: continue
                if not _sector_ok(sym,pos,close_mat,i,sym_to_i,trade_syms,capital,scap): continue
                if gen:
                    cg = sum(p.qty*float(close_mat[i,sym_to_i[p.symbol]]) for p in pos.values())/max(1.,capital)
                    if cg+bp*LOT/max(1.,capital) > gc: continue
                alloc = capital/max_pos
                qty   = int(alloc/bp/LOT)*LOT
                if qty<=0 or qty*bp*(1+COST_ONE_WAY)>cash: continue
                _execute_buy(sym,bp,qty,i,nxt,trade_syms,trades,pos,rv)
                cash -= qty*bp*(1+COST_ONE_WAY)
                trades[-1]["date"] = ds

    return BaseState(equity=eq,invested=inv,held=held,trades=trades)


# ─────────────────────────────────────────────────────────────────────
#  SLEEVE POSITION
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:       str
    qty:          int
    entry_price:  float
    entry_idx:    int
    rsr_entry:    float
    days_below:   int  = 0
    decayed:      bool = False
    close_day_n:  float= 0.0   # close price at decay boundary day (for ret_after split)
    price_day_n:  float= 0.0   # open price used to enter (for return calc)


# ─────────────────────────────────────────────────────────────────────
#  SLEEVE SIMULATION
# ─────────────────────────────────────────────────────────────────────

def run_sleeve(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, common_dates,
    base_st: BaseState,
    spec: DecaySpec,
) -> dict:
    n = len(common_dates)
    sl_cash = float(base_st.equity[0]) * spec.initial_frac
    sl_pos: SlPos | None = None
    sl_trades: list[dict] = []
    sl_eq  = np.zeros(n, dtype=np.float64)
    sl_inv = np.zeros(n, dtype=np.float64)
    sl_sym = [None] * n

    # Per-day arrays for diagnostics
    hold_age   = np.full(n, -1, dtype=np.int32)  # age of held position (-1=idle)
    comb_eq_d  = np.zeros(n, dtype=np.float64)   # combined equity (for late DD)

    for i, date in enumerate(common_dates):
        ds    = str(date.date())
        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        s_eq  = sl_cash + s_inv
        sl_eq[i]  = s_eq
        sl_inv[i] = s_inv
        sl_sym[i] = sl_pos.symbol if sl_pos else None
        hold_age[i] = (i - sl_pos.entry_idx) if sl_pos else -1
        comb_eq_d[i] = base_st.equity[i] + s_eq

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n: break
        nxt = i + 1

        # ── DECAY: partial trim at boundary ──────────────────────────
        if sl_pos and not sl_pos.decayed and spec.decay_days is not None:
            hd = i - sl_pos.entry_idx
            if hd >= spec.decay_days:
                si      = sym_to_i[sl_pos.symbol]
                ct      = float(close_mat[i, si])
                trim_px = float(open_mat[nxt, si])
                b_eq    = float(base_st.equity[i])
                target_val  = b_eq * spec.target_frac
                new_qty     = int(target_val / max(1.0, trim_px) / LOT) * LOT
                sell_qty    = sl_pos.qty - new_qty

                if sell_qty >= LOT and trim_px > 0:
                    pnl_trim = (trim_px - sl_pos.entry_price) * sell_qty
                    sl_cash += sell_qty * trim_px * (1 - COST_ONE_WAY)
                    sl_trades.append({
                        "side": "TRIM", "symbol": sl_pos.symbol,
                        "pnl": pnl_trim, "entry": sl_pos.entry_price,
                        "exit": trim_px, "qty": sell_qty,
                        "entry_idx": sl_pos.entry_idx, "exit_idx": i,
                        "reason": f"DECAY_d{hd}", "date": ds,
                        "hold_days": hd, "is_trim": True,
                    })
                    sl_pos.qty = new_qty

                sl_pos.decayed   = True
                sl_pos.close_day_n = ct   # record close at decay day

        # ── EXIT: RSR<90 3日連続 OR market shock ─────────────────────
        if sl_pos:
            si    = sym_to_i[sl_pos.symbol]
            rv    = float(rsr_mat[i, si])
            ct    = float(close_mat[i, si])
            hd    = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si])
                if pc > 0 and (ct/pc - 1) <= SHOCK_SYM_THR:
                    do_exit = True; reason = "MARKET_SHOCK"

            if not do_exit:
                below = rv < EXIT_THR
                if below: sl_pos.days_below += 1
                else:     sl_pos.days_below  = 0
                if sl_pos.days_below >= EXIT_CONSEC:
                    do_exit = True; reason = "RSR_EXIT_90_3D"

            if do_exit:
                sp   = float(open_mat[nxt, si])
                pnl  = (sp - sl_pos.entry_price) * sl_pos.qty
                hd_f = i - sl_pos.entry_idx

                # ret_after_dayN: return from decay boundary to exit
                if sl_pos.decayed and sl_pos.close_day_n > 0:
                    ret_after = (sp / sl_pos.close_day_n - 1) * 100
                else:
                    ret_after = float("nan")

                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": sl_pos.entry_price, "exit": sp,
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": ds,
                    "hold_days": hd_f, "decayed": sl_pos.decayed,
                    "ret_after_dayN": ret_after, "is_trim": False,
                })
                sl_pos = None

        # ── ENTRY ─────────────────────────────────────────────────────
        if sl_pos is None and not mkt_shock:
            base_held = base_st.held[i]
            cands: list[tuple] = []
            for sym in active_syms:
                if sym in base_held: continue
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                if not (ENTRY_RSR_LO <= rv < ENTRY_RSR_HI): continue
                d90 = int(cross90_mat[i, si])
                if d90 > ENTRY_D90_MAX: continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                    continue
                st = classify_state(float(slope5_mat[i, si]),
                                    float(slope20_mat[i, si]))
                sr = STATE_SCORE_MAP.get(st, 0)
                cands.append((rv, d90, sr, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                rv, d90, sr, sym = cands[0]
                si   = sym_to_i[sym]
                bp   = float(open_mat[nxt, si])
                if bp > 0:
                    b_eq  = base_st.equity[i]
                    alloc = min(sl_cash, b_eq * spec.initial_frac)
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos = SlPos(
                            symbol=sym, qty=qty,
                            entry_price=bp, entry_idx=nxt, rsr_entry=rv,
                        )
                        sl_trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "exit": None,
                            "qty": qty, "pnl": None,
                            "entry_idx": nxt, "exit_idx": None,
                            "reason": f"RSR={rv:.1f} d90={d90}",
                            "date": ds, "rsr_entry": rv,
                            "hold_days": None, "decayed": False,
                            "ret_after_dayN": float("nan"), "is_trim": False,
                        })

    return {
        "sl_eq":    sl_eq,
        "sl_inv":   sl_inv,
        "sl_sym":   sl_sym,
        "sl_trades":sl_trades,
        "hold_age": hold_age,
        "comb_eq":  comb_eq_d,
    }


# ─────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    base_st: BaseState,
    sleeve: dict,
    common_dates,
    fold: dict,
    capital: float,
    sl_cagr_baseline: float,
) -> dict:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    date_strs   = [str(d.date()) for d in common_dates]
    oos_idx     = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not oos_idx: return {}

    si_oos, ei_oos = oos_idx[0], oos_idx[-1] + 1
    oos_dates = list(common_dates[si_oos:ei_oos])
    n_days    = len(oos_dates)
    n_years   = n_days / 252.0

    b_eq  = base_st.equity[si_oos:ei_oos].tolist()
    s_eq  = sleeve["sl_eq"][si_oos:ei_oos].tolist()
    b_inv = base_st.invested[si_oos:ei_oos].tolist()
    s_inv = sleeve["sl_inv"][si_oos:ei_oos].tolist()
    c_eq  = (base_st.equity + sleeve["sl_eq"])[si_oos:ei_oos].tolist()
    h_age = sleeve["hold_age"][si_oos:ei_oos]

    if len(b_eq) < 5: return {}

    base_t = [t for t in base_st.trades if ds_s <= t.get("date","") <= ds_e]
    sl_t   = [t for t in sleeve["sl_trades"] if ds_s <= t.get("date","") <= ds_e]
    sl_sell= [t for t in sl_t if t["side"] == "SELL"]
    sl_trim= [t for t in sl_t if t.get("is_trim")]

    # ── Core metrics ──────────────────────────────────────────────────
    b_exp_l = [bi/max(1,be) for bi,be in zip(b_inv, b_eq)]
    c_exp_l = [(bi+si)/max(1,be+se) for bi,si,be,se in zip(b_inv,s_inv,b_eq,s_eq)]
    base_m  = calc_metrics(b_eq, base_t, b_exp_l, b_eq[0], oos_dates)
    comb_m  = calc_metrics(c_eq, base_t+sl_t, c_exp_l, c_eq[0], oos_dates)
    sl_m    = {}
    if s_eq[0] > 0:
        sl_m = calc_metrics(s_eq, sl_t,
                             [si/max(1,se) for si,se in zip(s_inv,s_eq)],
                             s_eq[0], oos_dates)

    delta_cagr   = comb_m.get("cagr",0) - base_m.get("cagr",0)
    delta_dd     = -(comb_m.get("max_dd",0) - base_m.get("max_dd",0))
    delta_calmar = comb_m.get("calmar",0) - base_m.get("calmar",0)
    sl_cagr      = sl_m.get("cagr",0)

    # ── Activity ──────────────────────────────────────────────────────
    n_exits   = len(sl_sell)
    trig_yr   = n_exits / max(0.01, n_years)
    hold_list = [t["hold_days"] for t in sl_sell if (t.get("hold_days") or 0) > 0]
    avg_hold  = float(np.mean(hold_list)) if hold_list else 0.0

    invested_days = sum(1 for v in s_inv if v > 0)
    cap_util      = invested_days / max(1, n_days)

    # ── Time-weighted exposure ────────────────────────────────────────
    early_exp_days, late_exp_days = [], []
    for j, age in enumerate(h_age):
        if age < 0: continue   # idle
        exp_val = s_inv[j] / max(1, s_eq[j])
        if age < LATE_BOUNDARY: early_exp_days.append(exp_val)
        else:                   late_exp_days.append(exp_val)

    avg_early_exp = float(np.mean(early_exp_days)) if early_exp_days else 0.0
    avg_late_exp  = float(np.mean(late_exp_days))  if late_exp_days  else 0.0

    # ── Late-period drawdown ──────────────────────────────────────────
    # Max drawdown of combined equity on days where sleeve is in "late" phase
    late_ceq = [c_eq[j] for j, age in enumerate(h_age) if age >= LATE_BOUNDARY]
    if len(late_ceq) >= 2:
        peak_l = late_ceq[0]; late_dd = 0.0
        for v in late_ceq:
            if v > peak_l: peak_l = v
            dd = (v - peak_l) / max(1, peak_l)
            if dd < late_dd: late_dd = dd
        late_period_dd = -late_dd * 100   # positive = worse
    else:
        late_period_dd = 0.0

    # ── ret_after_dayN ────────────────────────────────────────────────
    rat = [t["ret_after_dayN"] for t in sl_sell
           if not math.isnan(t.get("ret_after_dayN", float("nan")))]
    avg_ret_after = float(np.mean(rat)) if rat else float("nan")
    n_decayed     = sum(1 for t in sl_sell if t.get("decayed"))

    # ── Alpha retention ───────────────────────────────────────────────
    alpha_ret = sl_cagr / max(0.01, sl_cagr_baseline) if sl_cagr_baseline > 0 else 0.0

    # ── Fold pass ─────────────────────────────────────────────────────
    fold_pass = (
        delta_cagr > ADOPT_DCAGR
        and delta_dd <= ADOPT_DDD
        and alpha_ret >= ADOPT_ALPHA_RET
    )

    return {
        "base_cagr":    base_m.get("cagr",0),
        "base_dd":      base_m.get("max_dd",0),
        "comb_cagr":    comb_m.get("cagr",0),
        "comb_dd":      comb_m.get("max_dd",0),
        "comb_calmar":  comb_m.get("calmar",0),
        "sl_cagr":      sl_cagr,
        "sl_calmar":    sl_m.get("calmar",0),
        "delta_cagr":   round(delta_cagr,2),
        "delta_dd":     round(delta_dd,2),
        "delta_calmar": round(delta_calmar,3),
        "trig_yr":      round(trig_yr,1),
        "avg_hold":     round(avg_hold,1),
        "cap_util":     round(cap_util*100,1),
        "avg_early_exp":round(avg_early_exp*100,1),
        "avg_late_exp": round(avg_late_exp*100,1),
        "late_period_dd":round(late_period_dd,2),
        "avg_ret_after":avg_ret_after,
        "n_decayed":    n_decayed,
        "n_trims":      len(sl_trim),
        "alpha_ret":    round(alpha_ret,3),
        "fold_pass":    fold_pass,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF AGGREGATE
# ─────────────────────────────────────────────────────────────────────

def aggregate_wf(folds: list[dict]) -> dict:
    valid = [f for f in folds if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(k): return float(np.mean([f[k] for f in valid]))

    rat_vals = [f["avg_ret_after"] for f in valid
                if not math.isnan(f.get("avg_ret_after", float("nan")))]

    adopted = (
        n_pass >= ADOPT_WF
        and avg("delta_cagr") > ADOPT_DCAGR
        and avg("delta_dd")   <= ADOPT_DDD
        and avg("alpha_ret")  >= ADOPT_ALPHA_RET
    )

    return {
        "n_pass":        n_pass,
        "avg_sl_cagr":   round(avg("sl_cagr"),1),
        "avg_dcagr":     round(avg("delta_cagr"),2),
        "avg_dd":        round(avg("delta_dd"),2),
        "avg_dcalmar":   round(avg("delta_calmar"),3),
        "avg_trig":      round(avg("trig_yr"),1),
        "avg_hold":      round(avg("avg_hold"),1),
        "avg_util":      round(avg("cap_util"),1),
        "avg_early_exp": round(avg("avg_early_exp"),1),
        "avg_late_exp":  round(avg("avg_late_exp"),1),
        "avg_late_dd":   round(avg("late_period_dd"),2),
        "avg_ret_after": round(float(np.mean(rat_vals)),2) if rat_vals else float("nan"),
        "avg_alpha_ret": round(avg("alpha_ret"),3),
        "adopted":       adopted,
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    topix_close: pd.Series | None,
    sl_cagr_baseline: float,
    path: Path,
) -> None:
    L = []; w = L.append

    adopted = [c for c in DECAY_CASES if case_results[c]["wf"].get("adopted")]
    verdict = "✅ GO LIVE" if adopted else "🔬 KEEP RESEARCH"

    def ee(v): return f"{v:+.2f}" if not math.isnan(float(v if v is not None else float("nan"))) else "—"

    w("# Dedicated Alpha Sleeve — Time Decay Allocation WF  (Study 8D)")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n固定: ENTRY=RSR[{ENTRY_RSR_LO:.0f},{ENTRY_RSR_HI:.0f}) d90≤{ENTRY_D90_MAX}"
      f"  EXIT=RSR<{EXIT_THR:.0f} {EXIT_CONSEC}日連続")
    w(f"\n採用条件: WF≥{ADOPT_WF}/5, ΔCAGR>+{ADOPT_DCAGR}pp, "
      f"ΔDD≤+{ADOPT_DDD}pp, alpha_retention≥{ADOPT_ALPHA_RET*100:.0f}%\n")
    w(f"**採用Case**: {len(adopted)}件  |  最終判定: **{verdict}**\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    wf_a = case_results["A"]["wf"]
    w(f"- ベースライン (Case A, 25%固定): sl_CAGR={wf_a.get('avg_sl_cagr',0):.1f}%  "
      f"ΔDD={wf_a.get('avg_dd',0):+.2f}pp  WF={wf_a.get('n_pass',0)}/5")
    w(f"- 検証軸: time decay alloc で late-phase DD を抑制しながら alpha 維持")
    w(f"- 採用: **{len(adopted)}件**{' — ' + ', '.join(adopted) if adopted else ''}\n")

    # ── S2. WF Summary ────────────────────────────────────────────────
    w("---\n## 2. WF サマリ (6 Allocation Cases)\n")
    w("| Case | Allocation | WF | sl_CAGR | ΔCAGR | ΔDD | Calmar | α_ret | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c, spec in DECAY_CASES.items():
        wf   = case_results[c]["wf"]
        mark = "**✅**" if wf.get("adopted") else "❌"
        w(f"| {c} | {spec.label} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_sl_cagr',0):.1f}% "
          f"| {wf.get('avg_dcagr',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_dcalmar',0):+.3f} "
          f"| {wf.get('avg_alpha_ret',0)*100:.0f}% "
          f"| {mark} |")

    # ── S3. Time-weighted Exposure ────────────────────────────────────
    w("\n---\n## 3. Time-weighted Exposure Decomposition\n")
    w(f"> 「早期」= hold < {LATE_BOUNDARY}日  「後期」= hold ≥ {LATE_BOUNDARY}日\n")
    w("| Case | early_exp% | late_exp% | late_exp削減 | late_period_DD | ret_after_d5 | n_decayed |")
    w("|---|---|---|---|---|---|---|")
    wf_a_early = case_results["A"]["wf"].get("avg_early_exp",0)
    wf_a_late  = case_results["A"]["wf"].get("avg_late_exp",0)
    for c in DECAY_CASES:
        wf  = case_results[c]["wf"]
        el  = wf.get("avg_early_exp",0)
        lat = wf.get("avg_late_exp",0)
        d_lat = lat - wf_a_late
        rat   = wf.get("avg_ret_after", float("nan"))
        rat_s = f"{rat:+.1f}%" if not math.isnan(float(rat if rat is not None else float("nan"))) else "—"
        nd    = sum(f.get("n_decayed",0) for f in case_results[c]["folds"] if f)
        w(f"| {c} | {el:.1f}% | {lat:.1f}% | {d_lat:+.1f}pp | "
          f"{wf.get('avg_late_dd',0):+.2f}pp | {rat_s} | {nd} |")

    # ── S4. Fold Detail ───────────────────────────────────────────────
    w("\n---\n## 4. Fold 詳細\n")
    w("| Case | Fold | Regime | sl_CAGR | ΔCAGR | ΔDD | late_DD | α_ret | pass |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in DECAY_CASES:
        for fold, fm in zip(FOLDS, case_results[c]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm.get("fold_pass") else "❌"
            w(f"| {c} | Fold{fold['id']} | {reg} "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['late_period_dd']:+.2f}pp "
              f"| {fm['alpha_ret']*100:.0f}% "
              f"| {ok} |")

    # ── S5. DD Reduction Analysis ─────────────────────────────────────
    w("\n---\n## 5. ΔDD 削減分析\n")
    w("> Case A ΔDD avg から各 Case がどれだけ ΔDD を削減したか\n")
    w("| Case | avg_ΔDD | ΔDD削減 vs A | sl_CAGR削減 | CAGR/DD trade-off |")
    w("|---|---|---|---|---|")
    dd_a  = case_results["A"]["wf"].get("avg_dd",0)
    sc_a  = case_results["A"]["wf"].get("avg_sl_cagr",0)
    for c in DECAY_CASES:
        wf    = case_results[c]["wf"]
        dd_x  = wf.get("avg_dd",0)
        sc_x  = wf.get("avg_sl_cagr",0)
        d_dd  = dd_a - dd_x   # positive = ΔDD reduced
        d_sc  = sc_x - sc_a   # positive = sl_CAGR improved
        if d_dd > 0.05:
            ratio = d_sc / d_dd
            r_str = f"{ratio:+.2f}pp_sl_CAGR / pp_ΔDD"
        else:
            ratio = None; r_str = "N/A (DD増)"
        w(f"| {c} | {dd_x:+.2f}pp | {d_dd:+.2f}pp | {d_sc:+.1f}pp | {r_str} |")

    # ── S6. Failure Analysis ──────────────────────────────────────────
    w("\n---\n## 6. Failure Analysis\n")
    for c, spec in DECAY_CASES.items():
        wf = case_results[c]["wf"]
        fails = []
        if wf.get("n_pass",0) < ADOPT_WF:
            fails.append(f"WF={wf['n_pass']}/5")
        if wf.get("avg_dcagr",0) <= ADOPT_DCAGR:
            fails.append(f"ΔCAGR={wf['avg_dcagr']:+.2f}pp")
        if wf.get("avg_dd",0) > ADOPT_DDD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp")
        if wf.get("avg_alpha_ret",0) < ADOPT_ALPHA_RET:
            fails.append(f"α_ret={wf['avg_alpha_ret']*100:.0f}%<{ADOPT_ALPHA_RET*100:.0f}%")
        if not fails:
            w(f"\n**{c}** ({spec.label}): ✅ 全基準クリア")
        else:
            w(f"\n**{c}** ({spec.label}): REJECT — " + " / ".join(fails))

    # ── S7. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 7. Final Recommendation\n")
    w(f"## `{verdict}`\n")

    if adopted:
        best = max(adopted, key=lambda c: case_results[c]["wf"].get("avg_dcagr",0))
        bwf  = case_results[best]["wf"]
        bsp  = DECAY_CASES[best]
        w(f"**採用推奨: Case {best}** — {bsp.label}\n")
        w(f"- sl_CAGR: {bwf['avg_sl_cagr']:.1f}%  (baseline={sl_cagr_baseline:.1f}%  "
          f"retention={bwf['avg_alpha_ret']*100:.0f}%)")
        w(f"- ΔCAGR: {bwf['avg_dcagr']:+.2f}pp  ΔDD: {bwf['avg_dd']:+.2f}pp  "
          f"ΔCalmar: {bwf['avg_dcalmar']:+.3f}")
        w(f"- WF: {bwf['n_pass']}/5  trig: {bwf['avg_trig']:.1f}/yr")
        w(f"- late_exp: {bwf['avg_late_exp']:.1f}%  late_DD: {bwf['avg_late_dd']:+.2f}pp\n")
        w("実装優先度 (ASK_FIRST 必須):")
        w("1. signal_bridge.py スリーブ注文ロジック追加")
        w("2. TRIM (partial sell) の注文ロジック実装")
        w("3. sleeve_state.json でポジション age 追跡")
        w("4. live dry-run 30日以上 → 本番移行")
        w("\n**次ステップ**: Study8 全研究の統合判断 → 採用条件最終確認")
    else:
        best = max(DECAY_CASES, key=lambda c: case_results[c]["wf"].get("avg_dcagr",0))
        bwf  = case_results[best]["wf"]
        bsp  = DECAY_CASES[best]
        w(f"全6 Case 採用基準未達。最良: **Case {best}** ({bsp.label})\n")
        w(f"- sl_CAGR={bwf['avg_sl_cagr']:.1f}%  "
          f"ΔCAGR={bwf['avg_dcagr']:+.2f}pp  ΔDD={bwf['avg_dd']:+.2f}pp  "
          f"WF={bwf['n_pass']}/5\n")
        fails = []
        if bwf.get("n_pass",0) < ADOPT_WF:     fails.append(f"WF={bwf['n_pass']}/5")
        if bwf.get("avg_dd",0) > ADOPT_DDD:     fails.append(f"ΔDD={bwf['avg_dd']:+.2f}pp")
        if bwf.get("avg_dcagr",0) <= ADOPT_DCAGR: fails.append(f"ΔCAGR={bwf['avg_dcagr']:+.2f}pp")
        if bwf.get("avg_alpha_ret",0) < ADOPT_ALPHA_RET:
            fails.append(f"α_ret={bwf['avg_alpha_ret']*100:.0f}%")
        w("**バインディング制約**: " + " / ".join(fails) + "\n")
        w("**構造的発見:**")
        w(f"- Decay alloc は late_exp を削減するが、sl_CAGR も同時に低下")
        dd_a = case_results["A"]["wf"].get("avg_dd",0)
        dd_best = bwf.get("avg_dd",0)
        w(f"- Case A ΔDD={dd_a:+.2f}pp → best Case {best} ΔDD={dd_best:+.2f}pp")
        w(f"  削減幅={(dd_a-dd_best):+.2f}pp に対する alpha_retention={bwf['avg_alpha_ret']*100:.0f}%")
        w("\n次ステップ候補:")
        w("- Study 8E: Case C (3日連続 EXIT) + 25% 固定 でポートフォリオ集中度を下げる別アプローチ")
        w("- ΔDD の根本原因は sleeve 単体の volatility: max_pos=1 の single-stock concentration")
        w("- 解決策: sleeve max_pos=2 で集中度分散 or cap=15% 固定 (decay なし)")
        w("- Study 8 シリーズの総括: RSR90-94 の alpha は実在。実装可能な risk control が課題")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Study 8D: Alpha Sleeve Time Decay Allocation WF")
    print(f"  ENTRY RSR[{ENTRY_RSR_LO:.0f},{ENTRY_RSR_HI:.0f}) d90≤{ENTRY_D90_MAX}"
          f"  EXIT RSR<{EXIT_THR:.0f} {EXIT_CONSEC}日連続")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms   = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms  = list(trade_syms.keys())
    sym_to_i     = {s: i for i, s in enumerate(active_syms)}
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
        st    = (MeanReversionStrategy(**MR_PARAMS) if rule=="mean_rev" else
                 FujikoStrategy(
                     min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                     rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                     mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                     use_turtle_entry=cfg.fujiko.use_turtle_entry))
        required = 252 + getattr(st,"mom_period",21) + 2
        if hasattr(st,"precompute_signals") and len(df_src) >= required:
            sig_mat[:,si] = st.precompute_signals(df_src).to_numpy(dtype=np.int8)[row_idx]
            sig_ready[si] = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
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

    cross90_mat             = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)

    print("[3/4] Base シミュレーション (1回)...")
    base_st = run_base(
        open_mat, close_mat,
        sig_mat, sig_ready, rsr_mat, sym_active_mat,
        mkt_ret1, topix_ret20, topix_ret60, bear_arr,
        active_syms, sym_to_i, trade_syms, cfg, common_dates,
    )
    capital = float(cfg.portfolio.capital)

    print("[4/4] Decay Case シミュレーション (6 cases)...\n")
    case_results: dict = {}
    sl_cagr_baseline: float = 0.0

    for case, spec in DECAY_CASES.items():
        print(f"  [{case}] {spec.label}")
        sleeve = run_sleeve(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, common_dates, base_st, spec,
        )
        bl = sl_cagr_baseline  # use Case A baseline (0.0 on first pass → recomputed)
        folds_m = []
        for fold in FOLDS:
            fm = compute_fold_metrics(base_st, sleeve, common_dates, fold, capital, bl)
            folds_m.append(fm)

        wf = aggregate_wf(folds_m)

        # Set baseline from Case A
        if case == "A":
            sl_cagr_baseline = wf.get("avg_sl_cagr", 0.0)
            # Recompute with correct baseline
            folds_m = []
            for fold in FOLDS:
                fm = compute_fold_metrics(
                    base_st, sleeve, common_dates, fold, capital, sl_cagr_baseline)
                folds_m.append(fm)
            wf = aggregate_wf(folds_m)

        case_results[case] = {"folds": folds_m, "wf": wf}

        for fold, fm in zip(FOLDS, folds_m):
            if fm:
                ok = "✅" if fm.get("fold_pass") else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"sl={fm['sl_cagr']:+.1f}%  "
                      f"Δ={fm['delta_cagr']:+.2f}pp  "
                      f"ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"late_DD={fm['late_period_dd']:+.2f}pp  "
                      f"α_ret={fm['alpha_ret']*100:.0f}%  {ok}")
        ok_s = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"sl={wf.get('avg_sl_cagr',0):.1f}%  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"late_DD={wf.get('avg_late_dd',0):+.2f}pp  "
              f"α_ret={wf.get('avg_alpha_ret',0)*100:.0f}%  {ok_s}\n")

    adopted = [c for c in DECAY_CASES if case_results[c]["wf"].get("adopted")]
    print(f"{'='*68}")
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for c in DECAY_CASES:
        wf = case_results[c]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        print(f"  {ok} Case {c}: WF={wf.get('n_pass',0)}/5  "
              f"sl={wf.get('avg_sl_cagr',0):.1f}%  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"α_ret={wf.get('avg_alpha_ret',0)*100:.0f}%")
    print(f"{'='*68}\n")

    write_report(case_results, topix_close, sl_cagr_baseline,
                 REPORTS_DIR / "dedicated_alpha_time_decay.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
