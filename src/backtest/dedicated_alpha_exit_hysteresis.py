"""
backtest/dedicated_alpha_exit_hysteresis.py  —  Study 8C

固定:  ENTRY=RSR[92,95) d90≤5, CAP=25%, max_pos=1
変化:  EXIT 条件のみ (Case A–G)

追加監査:
  bounce_exit_rate     exit後10日以内にRSR>=92復帰率
  false_exit_rate      exit後20日でmax_return>10%の率
  state_flip_rate      保有中にRSR90クロス頻度
  trigger_elasticity   Δtrigger / Δavg_hold  (vs Case A)
  Calmar_per_trigger   ΔCalmar / trigger_removed  (vs Case A)
  alpha_retention      sl_CAGR / sl_CAGR_baseline_A

採用:  WF≥4/5, ΔCAGR≥+0.3pp, ΔDD≤+1.5pp, trigger≤8/yr, alpha_retention≥85%

出力:  reports/dedicated_alpha_exit_hysteresis.md

Run:
    cd C:/ai-trading
    python src/backtest/dedicated_alpha_exit_hysteresis.py
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

# ── Fixed entry ───────────────────────────────────────────────────────
ENTRY_RSR_LO   = 92.0
ENTRY_RSR_HI   = 95.0
ENTRY_D90_MAX  = 5
SLEEVE_CAP_FR  = 0.25
SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08

# ── Exit cases ────────────────────────────────────────────────────────
EXIT_CASES = {
    "A": {"label": "RSR<90 即exit",           "thr": 90.0, "consec": 1, "min_hold": 0},
    "B": {"label": "RSR<90 2日連続",           "thr": 90.0, "consec": 2, "min_hold": 0},
    "C": {"label": "RSR<90 3日連続",           "thr": 90.0, "consec": 3, "min_hold": 0},
    "D": {"label": "RSR<89 即exit",           "thr": 89.0, "consec": 1, "min_hold": 0},
    "E": {"label": "RSR<89 2日連続",           "thr": 89.0, "consec": 2, "min_hold": 0},
    "F": {"label": "RSR<90 AND hold≥5d",      "thr": 90.0, "consec": 1, "min_hold": 5},
    "G": {"label": "RSR<90 AND hold≥3d",      "thr": 90.0, "consec": 1, "min_hold": 3},
}

# ── Adoption thresholds ───────────────────────────────────────────────
ADOPT_WF          = 4
ADOPT_DCAGR       = 0.3    # pp
ADOPT_DDD         = 1.5    # pp
ADOPT_TRIG_MAX    = 8.0    # /yr
ADOPT_ALPHA_RET   = 0.85   # fraction of baseline sl_CAGR

# RSR thresholds for diagnostics
BOUNCE_RSR_THR    = 92.0   # bounce = RSR returns to ≥92 within 10d
FALSE_EXIT_RET    = 0.10   # false exit = max_ret > 10% within 20d

STATE_SCORE_MAP = {
    "EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1,
    "FLAT": 1, "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0,
}


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    n_dates, n_syms = rsr_mat.shape
    out     = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i]  = running
        above   = rsr_mat[i] >= threshold
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
    yr = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ─────────────────────────────────────────────────────────────────────
#  BASE SIMULATION  (identical to Study 8B)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BaseState:
    equity:   np.ndarray
    invested: np.ndarray
    held:     list           # list[set[str]]
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
    rc            = getattr(cfg, "risk_controls", None)
    MAX_SEC       = float(rc.sector_cap)  if rc else 0.25
    gross_en      = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    g_norm        = float(getattr(rc, "gross_cap_normal",        1.0)) if rc else 1.0
    g_dd5         = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    g_dd8         = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    cash = float(capital); pos: dict[str, Position] = {}
    peak = float(capital); cb_active = False; cb_days = 0
    reentry_ban: dict[str, int] = {}; trades: list = []
    n = len(common_dates)
    eq  = np.zeros(n, dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)
    held = [set() for _ in range(n)]

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        b_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]]) for s, p in pos.items())
        b_eq  = cash + b_inv
        eq[i]  = b_eq; inv[i] = b_inv; held[i] = set(pos.keys())

        if b_eq > peak: peak = b_eq
        dd = (b_eq - peak) / peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05: cb_active = False; cb_days = 0

        gc = g_norm
        if gross_en and topix_ret20 is not None:
            r20 = float(topix_ret20[i]); r60 = float(topix_ret60[i])
            if r20 < -0.05: gc = g_dd5
            elif r60 < -0.08: gc = g_dd8

        bear       = bool(bear_arr[i]) if bear_arr is not None else False
        sec_cap    = 0.18 if bear else MAX_SEC
        mkt_shock  = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR

        if i + 1 >= n: break
        nxt = i + 1
        sells, buys = [], []

        for sym in active_syms:
            si = sym_to_i[sym]; holding = sym in pos
            hd = (i - pos[sym].entry_idx) if holding else 0
            rv = float(rsr_mat[i, si]); ct = float(close_mat[i, si])

            if mkt_shock and holding:
                if i > 0:
                    pc = float(close_mat[i-1, si])
                    if pc > 0 and (ct/pc-1) <= SHOCK_SYM_THR:
                        sells.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not holding: continue
            if holding and max_hold and hd > max_hold:
                sells.append((sym, "TIME_STOP")); continue
            if holding and rv < rsr_exit_thr and hd >= min_hold:
                sells.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and holding and hd >= min_hold:
                sells.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not holding:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
                buys.append((rv, sym))

        for sym, reason in sells:
            if sym not in pos: continue
            p = pos[sym]; si = sym_to_i[sym]
            sp = float(open_mat[nxt, si])
            cash += p.qty * sp * (1 - COST_ONE_WAY)
            trades.append({"side":"SELL","symbol":sym,"pnl":(sp-p.entry_price)*p.qty,
                           "entry":p.entry_price,"exit":sp,"qty":p.qty,
                           "entry_idx":p.entry_idx,"exit_idx":i,"reason":reason,"date":ds})
            del pos[sym]
            if reason == "TIME_STOP": reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not cb_active and buys:
            buys.sort(key=lambda x: -x[0])
            for rv, sym in buys:
                si = sym_to_i[sym]
                if max_pos - len(pos) <= 0: break
                bp = float(open_mat[nxt, si])
                if bp <= 0: continue
                if not _sector_ok(sym, pos, close_mat, i, sym_to_i, trade_syms, capital, sec_cap): continue
                if gross_en:
                    cg = sum(p.qty*float(close_mat[i,sym_to_i[p.symbol]]) for p in pos.values())/max(1.,capital)
                    if cg + bp*LOT/max(1.,capital) > gc: continue
                alloc = capital / max_pos
                qty   = int(alloc / bp / LOT) * LOT
                if qty <= 0 or qty*bp*(1+COST_ONE_WAY) > cash: continue
                _execute_buy(sym, bp, qty, i, nxt, trade_syms, trades, pos, rv)
                cash -= qty * bp * (1 + COST_ONE_WAY)
                trades[-1]["date"] = ds

    return BaseState(equity=eq, invested=inv, held=held, trades=trades)


# ─────────────────────────────────────────────────────────────────────
#  SLEEVE POSITION TRACKER
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:       str
    qty:          int
    entry_price:  float
    entry_idx:    int
    rsr_entry:    float
    days_below:   int  = 0       # consecutive days below exit threshold
    flip_count:   int  = 0       # RSR 90.0 crossings during hold
    last_above90: bool = True    # for flip detection


# ─────────────────────────────────────────────────────────────────────
#  SLEEVE SIMULATION
# ─────────────────────────────────────────────────────────────────────

def run_sleeve(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, common_dates,
    base_st: BaseState,
    exit_thr: float, consec: int, min_hold_exit: int,
) -> dict:
    """Fixed entry RSR[92,95) d90≤5; exit varies by parameters."""
    n        = len(common_dates)
    sl_cash  = float(base_st.equity[0]) * SLEEVE_CAP_FR
    sl_pos: SlPos | None = None
    sl_trades: list[dict] = []
    sl_eq    = np.zeros(n, dtype=np.float64)
    sl_inv   = np.zeros(n, dtype=np.float64)
    sl_sym   = [None] * n

    for i, date in enumerate(common_dates):
        ds       = str(date.date())
        s_inv    = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                    if sl_pos else 0.0)
        s_eq     = sl_cash + s_inv
        sl_eq[i]  = s_eq
        sl_inv[i] = s_inv
        sl_sym[i] = sl_pos.symbol if sl_pos else None

        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR)

        if i + 1 >= n: break
        nxt = i + 1

        # ── RSR flip tracking for held position ───────────────────────
        if sl_pos:
            si        = sym_to_i[sl_pos.symbol]
            above90   = float(rsr_mat[i, si]) >= 90.0
            if above90 != sl_pos.last_above90:
                sl_pos.flip_count += 1
            sl_pos.last_above90 = above90

        # ── EXIT ──────────────────────────────────────────────────────
        if sl_pos:
            si      = sym_to_i[sl_pos.symbol]
            rsr_v   = float(rsr_mat[i, si])
            ct      = float(close_mat[i, si])
            hd      = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            # Market shock
            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si])
                if pc > 0 and (ct/pc-1) <= SHOCK_SYM_THR:
                    do_exit = True; reason = "MARKET_SHOCK"

            if not do_exit:
                below_thr = rsr_v < exit_thr
                if below_thr:
                    sl_pos.days_below += 1
                else:
                    sl_pos.days_below = 0

                triggered = sl_pos.days_below >= consec
                hold_ok   = hd >= min_hold_exit

                if triggered and hold_ok:
                    do_exit = True
                    reason  = (f"RSR<{exit_thr:.0f}"
                               + (f"_{consec}D" if consec > 1 else "")
                               + (f"_HOLD{min_hold_exit}" if min_hold_exit > 0 else ""))

            if do_exit:
                sp  = float(open_mat[nxt, si])
                pnl = (sp - sl_pos.entry_price) * sl_pos.qty
                hd_final = i - sl_pos.entry_idx
                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": sl_pos.entry_price, "exit": sp,
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": ds,
                    "hold_days": hd_final, "rsr_entry": sl_pos.rsr_entry,
                    "flip_count": sl_pos.flip_count,
                    "bounce10": None,   # filled post-hoc
                    "false_exit20": None,
                })
                sl_pos = None

        # ── ENTRY ─────────────────────────────────────────────────────
        if sl_pos is None and not mkt_shock:
            base_held = base_st.held[i]
            cands: list[tuple] = []
            for sym in active_syms:
                if sym in base_held: continue
                si    = sym_to_i[sym]
                rv    = float(rsr_mat[i, si])
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
                    alloc = min(sl_cash, b_eq * SLEEVE_CAP_FR)
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos = SlPos(
                            symbol=sym, qty=qty,
                            entry_price=bp, entry_idx=nxt, rsr_entry=rv,
                            last_above90=(rv >= 90.0),
                        )
                        sl_trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "exit": None,
                            "qty": qty, "pnl": None,
                            "entry_idx": nxt, "exit_idx": None,
                            "reason": f"RSR={rv:.1f} d90={d90}",
                            "date": ds, "rsr_entry": rv,
                            "hold_days": None, "flip_count": None,
                            "bounce10": None, "false_exit20": None,
                        })

    # ── Post-hoc: bounce10 & false_exit20 ────────────────────────────
    for t in sl_trades:
        if t["side"] != "SELL": continue
        ei  = t["exit_idx"]
        si  = sym_to_i.get(t["symbol"], -1)
        if si < 0: continue

        # bounce: RSR >= BOUNCE_RSR_THR within 10d after exit
        bounce = False
        for fwd in range(1, 11):
            if ei + fwd < n:
                if float(rsr_mat[ei + fwd, si]) >= BOUNCE_RSR_THR:
                    bounce = True; break
        t["bounce10"] = bounce

        # false exit: max_return > 10% within 20d after exit
        c0 = float(close_mat[ei, si])
        max_ret = 0.0
        for fwd in range(1, 21):
            if ei + fwd < n:
                cn = float(close_mat[ei + fwd, si])
                if c0 > 0 and cn > 0:
                    max_ret = max(max_ret, cn / c0 - 1)
        t["false_exit20"] = max_ret > FALSE_EXIT_RET

    return {
        "sl_eq": sl_eq, "sl_inv": sl_inv,
        "sl_sym": sl_sym, "sl_trades": sl_trades,
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
    dates_str   = [str(d.date()) for d in common_dates]
    oos_idx     = [i for i, s in enumerate(dates_str) if ds_s <= s <= ds_e]
    if not oos_idx: return {}

    si_oos, ei_oos = oos_idx[0], oos_idx[-1] + 1
    oos_dates       = list(common_dates[si_oos:ei_oos])
    n_days          = len(oos_dates)
    n_years         = n_days / 252.0

    b_eq  = base_st.equity[si_oos:ei_oos].tolist()
    s_eq  = sleeve["sl_eq"][si_oos:ei_oos].tolist()
    b_inv = base_st.invested[si_oos:ei_oos].tolist()
    s_inv = sleeve["sl_inv"][si_oos:ei_oos].tolist()
    c_eq  = (base_st.equity + sleeve["sl_eq"])[si_oos:ei_oos].tolist()

    if len(b_eq) < 5: return {}

    base_t = [t for t in base_st.trades if ds_s <= t.get("date","") <= ds_e]
    sl_t   = [t for t in sleeve["sl_trades"] if ds_s <= t.get("date","") <= ds_e]
    sl_sell= [t for t in sl_t if t["side"] == "SELL"]

    # ── Core metrics ──────────────────────────────────────────────────
    b_exp_l = [bi/max(1,be) for bi,be in zip(b_inv, b_eq)]
    c_exp_l = [(bi+si)/max(1,be+se) for bi,si,be,se in zip(b_inv,s_inv,b_eq,s_eq)]
    base_m  = calc_metrics(b_eq, base_t, b_exp_l, b_eq[0], oos_dates)
    comb_m  = calc_metrics(c_eq, base_t+sl_t, c_exp_l, c_eq[0], oos_dates)
    sl_m: dict = {}
    if s_eq[0] > 0:
        s_exp_l = [si/max(1,se) for si,se in zip(s_inv,s_eq)]
        sl_m = calc_metrics(s_eq, sl_t, s_exp_l, s_eq[0], oos_dates)

    delta_cagr   = comb_m.get("cagr",0) - base_m.get("cagr",0)
    delta_dd     = -(comb_m.get("max_dd",0) - base_m.get("max_dd",0))
    delta_calmar = comb_m.get("calmar",0) - base_m.get("calmar",0)

    # ── Activity ──────────────────────────────────────────────────────
    n_exits   = len(sl_sell)
    trig_yr   = n_exits / max(0.01, n_years)
    hold_list = [t["hold_days"] for t in sl_sell if (t.get("hold_days") or 0) > 0]
    avg_hold  = float(np.mean(hold_list)) if hold_list else 0.0
    med_hold  = float(np.median(hold_list)) if hold_list else 0.0

    invested  = sum(1 for v in s_inv if v > 0)
    idle_rate = 1.0 - invested / max(1, n_days)

    # ── Exit reason mix ───────────────────────────────────────────────
    reason_counts: dict[str,int] = defaultdict(int)
    for t in sl_sell: reason_counts[t.get("reason","??")] += 1

    # ── State flip rate (avg flips per hold day) ──────────────────────
    flip_vals = [t["flip_count"] / max(1, t["hold_days"])
                 for t in sl_sell
                 if t.get("flip_count") is not None and (t.get("hold_days") or 0) > 0]
    state_flip_rate = float(np.mean(flip_vals)) if flip_vals else 0.0

    # ── Bounce & false exit ───────────────────────────────────────────
    bounce_exits = [t for t in sl_sell if t.get("bounce10")]
    false_exits  = [t for t in sl_sell if t.get("false_exit20")]
    bounce_rate  = len(bounce_exits) / max(1, n_exits)
    false_rate   = len(false_exits)  / max(1, n_exits)

    # ── Alpha retention ───────────────────────────────────────────────
    sl_cagr = sl_m.get("cagr", 0)
    alpha_retention = sl_cagr / max(0.01, sl_cagr_baseline) if sl_cagr_baseline > 0 else 0.0

    # ── Quality ───────────────────────────────────────────────────────
    wins = [t for t in sl_sell if (t.get("pnl") or 0) > 0]
    loss = [t for t in sl_sell if (t.get("pnl") or 0) <= 0]
    gp   = sum(t["pnl"] for t in wins) if wins else 0.0
    gl   = abs(sum(t["pnl"] for t in loss)) if loss else 0.0

    # ── Fold pass  (alpha_retention = aggregate check のみ) ──────────
    fold_pass = (
        delta_cagr > ADOPT_DCAGR
        and delta_dd <= ADOPT_DDD
        and trig_yr  <= ADOPT_TRIG_MAX
    )

    return {
        "base_cagr":     base_m.get("cagr",0),
        "base_dd":       base_m.get("max_dd",0),
        "comb_cagr":     comb_m.get("cagr",0),
        "comb_dd":       comb_m.get("max_dd",0),
        "comb_calmar":   comb_m.get("calmar",0),
        "sl_cagr":       sl_cagr,
        "sl_calmar":     sl_m.get("calmar",0),
        "delta_cagr":    round(delta_cagr,2),
        "delta_dd":      round(delta_dd,2),
        "delta_calmar":  round(delta_calmar,3),
        "trig_yr":       round(trig_yr,1),
        "avg_hold":      round(avg_hold,1),
        "med_hold":      round(med_hold,1),
        "idle_rate":     round(idle_rate*100,1),
        "n_exits":       n_exits,
        "reason_counts": dict(reason_counts),
        "state_flip_rate": round(state_flip_rate,4),
        "bounce_rate":   round(bounce_rate,3),
        "false_rate":    round(false_rate,3),
        "alpha_retention": round(alpha_retention,3),
        "hit_rate":      round(len(wins)/max(1,n_exits)*100,1),
        "profit_factor": round(gp/max(1.,gl),3),
        "fold_pass":     fold_pass,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF AGGREGATE
# ─────────────────────────────────────────────────────────────────────

def aggregate_wf(folds: list[dict]) -> dict:
    valid = [f for f in folds if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(k): return float(np.mean([f[k] for f in valid]))

    # aggregate reason counts
    all_reasons: dict[str,int] = defaultdict(int)
    for f in valid:
        for r, cnt in f.get("reason_counts",{}).items():
            all_reasons[r] += cnt
    total_exits = sum(all_reasons.values())
    reason_pct  = {r: round(cnt/max(1,total_exits)*100,1)
                   for r, cnt in sorted(all_reasons.items(), key=lambda x:-x[1])}

    adopted = (
        n_pass >= ADOPT_WF
        and avg("delta_cagr")      > ADOPT_DCAGR
        and avg("delta_dd")       <= ADOPT_DDD
        and avg("trig_yr")        <= ADOPT_TRIG_MAX
        and avg("alpha_retention") >= ADOPT_ALPHA_RET
    )

    return {
        "n_pass":          n_pass,
        "avg_sl_cagr":     round(avg("sl_cagr"),1),
        "avg_dcagr":       round(avg("delta_cagr"),2),
        "avg_dd":          round(avg("delta_dd"),2),
        "avg_dcalmar":     round(avg("delta_calmar"),3),
        "avg_trig":        round(avg("trig_yr"),1),
        "avg_hold":        round(avg("avg_hold"),1),
        "avg_med_hold":    round(avg("med_hold"),1),
        "avg_idle":        round(avg("idle_rate"),1),
        "avg_flip":        round(avg("state_flip_rate"),4),
        "avg_bounce":      round(avg("bounce_rate"),3),
        "avg_false":       round(avg("false_rate"),3),
        "avg_alpha_ret":   round(avg("alpha_retention"),3),
        "avg_hit":         round(avg("hit_rate"),1),
        "avg_pf":          round(avg("profit_factor"),3),
        "reason_pct":      reason_pct,
        "adopted":         adopted,
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

    adopted = [c for c in EXIT_CASES if case_results[c]["wf"].get("adopted")]
    verdict = "✅ GO LIVE" if adopted else "🔬 KEEP RESEARCH"

    w("# Dedicated Alpha Sleeve — Exit Persistence WF  (Study 8C)")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n固定ENTRY: RSR[{ENTRY_RSR_LO:.0f},{ENTRY_RSR_HI:.0f}) d90≤{ENTRY_D90_MAX}"
      f", cap=equity×{SLEEVE_CAP_FR*100:.0f}%, max_pos=1")
    w(f"\n採用条件: WF≥{ADOPT_WF}/5, ΔCAGR>+{ADOPT_DCAGR}pp, "
      f"ΔDD≤+{ADOPT_DDD}pp, trigger≤{ADOPT_TRIG_MAX}/yr, "
      f"alpha_retention≥{ADOPT_ALPHA_RET*100:.0f}%\n")
    w(f"**採用Case**: {len(adopted)}件  |  最終判定: **{verdict}**\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    bl = case_results["A"]["wf"]
    w(f"- ベースライン (Case A): sl_CAGR={bl.get('avg_sl_cagr',0):.1f}%  "
      f"trig={bl.get('avg_trig',0):.1f}/yr  WF={bl.get('n_pass',0)}/5")
    w(f"- 検証軸: EXIT 条件変更による trig/yr 削減 + alpha_retention ≥85% 維持")
    w(f"- 採用: **{len(adopted)}件**\n")

    # ── S2. WF Summary ────────────────────────────────────────────────
    w("---\n## 2. WF サマリ (7 Exit Case)\n")
    w("| Case | EXIT条件 | WF | sl_CAGR | ΔCAGR | ΔDD | trig/yr "
      "| α_ret | bounce | false | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for c, ep in EXIT_CASES.items():
        wf   = case_results[c]["wf"]
        mark = "**✅**" if wf.get("adopted") else "❌"
        w(f"| {c} | {ep['label']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_sl_cagr',0):.1f}% "
          f"| {wf.get('avg_dcagr',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_trig',0):.1f} "
          f"| {wf.get('avg_alpha_ret',0)*100:.0f}% "
          f"| {wf.get('avg_bounce',0)*100:.0f}% "
          f"| {wf.get('avg_false',0)*100:.0f}% "
          f"| {mark} |")

    # ── S3. Trigger Elasticity & Calmar per Trigger ───────────────────
    w("\n---\n## 3. Trigger Elasticity & Calmar per Trigger Removed\n")
    w("> Δtrig = trig_A − trig_X (正 = trigger 削減)  "
      "ΔCalmar/Δtrig = Calmar 改善 per trigger 削除\n")
    w("| Case | trig/yr | Δtrig vs A | Δavg_hold | elast. | ΔCalmar | Cal/trig | 評価 |")
    w("|---|---|---|---|---|---|---|---|")
    wf_a      = case_results["A"]["wf"]
    trig_a    = wf_a.get("avg_trig",0)
    dcal_a    = wf_a.get("avg_dcalmar",0)
    hold_a    = wf_a.get("avg_hold",0)
    for c in EXIT_CASES:
        wf     = case_results[c]["wf"]
        trig_x = wf.get("avg_trig",0)
        dcal_x = wf.get("avg_dcalmar",0)
        hold_x = wf.get("avg_hold",0)
        d_trig = trig_a - trig_x         # removed (pos = fewer triggers)
        d_hold = hold_x - hold_a         # extra hold days
        d_dcal = dcal_x - dcal_a         # calmar gain vs A
        if d_trig <= 0.05:
            elast = "N/A"; cpt = "N/A"; ev = "同等/悪化"
        else:
            elast = f"{d_trig/max(0.01,d_hold):.2f} trig/d" if d_hold > 0.1 else "hold不変"
            cpt   = f"{d_dcal/d_trig:+.4f}"
            ev    = "✅" if d_dcal >= 0 else f"❌ -{abs(d_dcal/d_trig):.4f}/trig"
        w(f"| {c} | {trig_x:.1f} | {d_trig:+.1f} | {d_hold:+.1f}d "
          f"| {elast} | {d_dcal:+.3f} | {cpt} | {ev} |")

    # ── S4. Exit Diagnostic ───────────────────────────────────────────
    w("\n---\n## 4. Exit Diagnostics\n")
    w("| Case | avg_hold | med_hold | idle% | flip/d | bounce% | false% | reason (top2) |")
    w("|---|---|---|---|---|---|---|---|")
    for c in EXIT_CASES:
        wf = case_results[c]["wf"]
        top2 = list(wf.get("reason_pct",{}).items())[:2]
        r2str = "  ".join(f"{r}:{p:.0f}%" for r, p in top2)
        w(f"| {c} "
          f"| {wf.get('avg_hold',0):.1f}d "
          f"| {wf.get('avg_med_hold',0):.1f}d "
          f"| {wf.get('avg_idle',0):.1f}% "
          f"| {wf.get('avg_flip',0):.3f} "
          f"| {wf.get('avg_bounce',0)*100:.0f}% "
          f"| {wf.get('avg_false',0)*100:.0f}% "
          f"| {r2str} |")

    # ── S5. Alpha Retention ───────────────────────────────────────────
    w("\n---\n## 5. Alpha Retention (vs Case A baseline)\n")
    w(f"> 基準 sl_CAGR (Case A, avg 5fold) = **{sl_cagr_baseline:.1f}%**  "
      f"採用基準 ≥{ADOPT_ALPHA_RET*100:.0f}%\n")
    w("| Case | sl_CAGR | α_retention | 採用基準 |")
    w("|---|---|---|---|")
    for c in EXIT_CASES:
        wf  = case_results[c]["wf"]
        sc  = wf.get("avg_sl_cagr",0)
        ar  = wf.get("avg_alpha_ret",0)
        ok  = "✅" if ar >= ADOPT_ALPHA_RET else "❌"
        w(f"| {c} | {sc:.1f}% | {ar*100:.0f}% | {ok} |")

    # ── S6. Fold Detail ───────────────────────────────────────────────
    w("\n---\n## 6. Fold 詳細\n")
    w("| Case | Fold | Regime | sl_CAGR | ΔCAGR | ΔDD | trig/yr | hold | bounce | false | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in EXIT_CASES:
        for fold, fm in zip(FOLDS, case_results[c]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm.get("fold_pass") else "❌"
            w(f"| {c} | Fold{fold['id']} | {reg} "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['trig_yr']:.1f} "
              f"| {fm['avg_hold']:.1f}d "
              f"| {fm['bounce_rate']*100:.0f}% "
              f"| {fm['false_rate']*100:.0f}% "
              f"| {ok} |")

    # ── S7. Structural Analysis ───────────────────────────────────────
    w("\n---\n## 7. Structural Analysis\n")
    bounce_a = case_results["A"]["wf"].get("avg_bounce",0)
    false_a  = case_results["A"]["wf"].get("avg_false",0)
    flip_a   = case_results["A"]["wf"].get("avg_flip",0)
    w(f"**Case A (baseline) bounce_rate={bounce_a*100:.0f}%  "
      f"false_exit_rate={false_a*100:.0f}%  flip/day={flip_a:.3f}**\n")
    w("- bounce_rate: 即exit後10日以内にRSR≥92に復帰する割合 → 高いほど早期 exit が多い")
    w("- false_exit_rate: exit後20日maxリターン>10% → 高いほど持続 alpha を切っている")
    w("- flip/day: 保有中のRSR90クロス頻度 → 高いほどRSRがbounce帯で振動\n")

    # Explain trigger persistence
    trig_vals = {c: case_results[c]["wf"].get("avg_trig",0) for c in EXIT_CASES}
    min_c = min(trig_vals, key=trig_vals.get)
    w(f"最低 trigger Case: **{min_c}** ({EXIT_CASES[min_c]['label']}) "
      f"trig={trig_vals[min_c]:.1f}/yr\n")
    w("**trigger/yr 感度まとめ:**")
    for c in EXIT_CASES:
        wf = case_results[c]["wf"]
        w(f"  {c}: {EXIT_CASES[c]['label']:35s} trig={wf.get('avg_trig',0):5.1f}/yr  "
          f"hold={wf.get('avg_hold',0):5.1f}d  α_ret={wf.get('avg_alpha_ret',0)*100:.0f}%")

    # ── S8. Failure Analysis ──────────────────────────────────────────
    w("\n---\n## 8. Failure Analysis\n")
    for c, ep in EXIT_CASES.items():
        wf = case_results[c]["wf"]
        fails = []
        if wf.get("n_pass",0) < ADOPT_WF:
            fails.append(f"WF={wf['n_pass']}/5")
        if wf.get("avg_dcagr",0) <= ADOPT_DCAGR:
            fails.append(f"ΔCAGR={wf['avg_dcagr']:+.2f}pp")
        if wf.get("avg_dd",0) > ADOPT_DDD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp")
        if wf.get("avg_trig",0) > ADOPT_TRIG_MAX:
            fails.append(f"trig={wf['avg_trig']:.1f}/yr")
        if wf.get("avg_alpha_ret",0) < ADOPT_ALPHA_RET:
            fails.append(f"α_ret={wf['avg_alpha_ret']*100:.0f}%<{ADOPT_ALPHA_RET*100:.0f}%")
        if not fails:
            w(f"\n**{c}** ({ep['label']}): ✅ 全基準クリア")
        else:
            w(f"\n**{c}** ({ep['label']}): REJECT — " + " / ".join(fails))

    # ── S9. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 9. Final Recommendation\n")
    w(f"## `{verdict}`\n")

    if adopted:
        best = max(adopted, key=lambda c: case_results[c]["wf"].get("avg_dcagr",0))
        bwf  = case_results[best]["wf"]
        w(f"**採用推奨: Case {best}** — {EXIT_CASES[best]['label']}\n")
        w(f"- sl_CAGR: {bwf['avg_sl_cagr']:.1f}%  ΔCAGR: {bwf['avg_dcagr']:+.2f}pp  "
          f"ΔDD: {bwf['avg_dd']:+.2f}pp")
        w(f"- trig/yr: {bwf['avg_trig']:.1f}  WF: {bwf['n_pass']}/5  "
          f"alpha_retention: {bwf['avg_alpha_ret']*100:.0f}%")
        w(f"- bounce: {bwf['avg_bounce']*100:.0f}%  false_exit: {bwf['avg_false']*100:.0f}%\n")
        w("実装優先度 (ASK_FIRST 必須):")
        w("1. signal_bridge.py スリーブ注文ロジック追加")
        w("2. EXIT 持続カウンタ実装 (days_below_thr per position)")
        w("3. 独立 sleeve_state.json 管理")
        w("4. live dry-run 30日以上 → 本番移行")
    else:
        best = max(EXIT_CASES, key=lambda c: case_results[c]["wf"].get("avg_dcagr",0))
        bwf  = case_results[best]["wf"]
        w(f"全7 Case 採用基準未達。最良: **Case {best}** ({EXIT_CASES[best]['label']})\n")
        w(f"- sl_CAGR={bwf['avg_sl_cagr']:.1f}%  "
          f"ΔCAGR={bwf['avg_dcagr']:+.2f}pp  trig={bwf['avg_trig']:.1f}/yr  "
          f"WF={bwf['n_pass']}/5\n")
        fails = []
        if bwf.get("n_pass",0) < ADOPT_WF:     fails.append(f"WF={bwf['n_pass']}/5")
        if bwf.get("avg_dd",0) > ADOPT_DDD:     fails.append(f"ΔDD={bwf['avg_dd']:+.2f}pp")
        if bwf.get("avg_trig",0) > ADOPT_TRIG_MAX: fails.append(f"trig={bwf['avg_trig']:.1f}/yr")
        if bwf.get("avg_alpha_ret",0) < ADOPT_ALPHA_RET:
            fails.append(f"α_ret={bwf['avg_alpha_ret']*100:.0f}%")
        w("**バインディング制約**: " + " / ".join(fails) + "\n")

        # Structural insight
        w("**構造的発見:**")
        trig_A = case_results["A"]["wf"].get("avg_trig",0)
        trig_best = bwf.get("avg_trig",0)
        trig_min  = min(case_results[c]["wf"].get("avg_trig",0) for c in EXIT_CASES)
        w(f"- Case A (即exit): trig={trig_A:.1f}/yr")
        w(f"- 最低 trig: {trig_min:.1f}/yr — EXIT 持続化でも 8/yr 未達")
        bounce_a = case_results["A"]["wf"].get("avg_bounce",0)
        w(f"- bounce_rate={bounce_a*100:.0f}%: 約{bounce_a*100:.0f}% の exit が10日以内に RSR≥92 復帰")
        w(f"  → RSR<90 threshold 自体が市場ノイズに対して敏感すぎる可能性")
        w("\n次ステップ候補:")
        w("- Study 8D: EXIT = ATR-based stop (threshold-free, noise-robust)")
        w("- Study 8E: capital 25%→15% で ΔDD 削減 + current best EXIT 組み合わせ")
        w("- Study 8F: RSR 70 exit (regime-aware 既採用済み, Study 7 知見) を sleeve に適用")

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
    print("  Study 8C: Alpha Sleeve Exit Persistence WF")
    print(f"  ENTRY固定 RSR[{ENTRY_RSR_LO:.0f},{ENTRY_RSR_HI:.0f}) d90≤{ENTRY_D90_MAX}"
          f" / EXIT 7 cases")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
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
        rule  = SECTOR_STRATEGY.get(trade_syms.get(sym,""), "fujiko")
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        st    = (MeanReversionStrategy(**MR_PARAMS) if rule == "mean_rev" else
                 FujikoStrategy(
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

    print("[4/4] Exit Case シミュレーション (7 cases)...\n")
    case_results = {}
    sl_cagr_a: float = 0.0

    for case, ep in EXIT_CASES.items():
        print(f"  [{case}] {ep['label']}")
        sleeve = run_sleeve(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, common_dates, base_st,
            ep["thr"], ep["consec"], ep["min_hold"],
        )

        # alpha_retention baseline = Case A sl_CAGR (computed on first pass)
        baseline_cagr = sl_cagr_a if case != "A" else 0.0

        folds_m: list[dict] = []
        for fold in FOLDS:
            fm = compute_fold_metrics(
                base_st, sleeve, common_dates, fold, capital, baseline_cagr)
            folds_m.append(fm)

        wf = aggregate_wf(folds_m)

        # Set Case A as alpha_retention baseline
        if case == "A":
            sl_cagr_a = wf.get("avg_sl_cagr", 0.0)
            # Recompute fold metrics with correct baseline
            folds_m = []
            for fold in FOLDS:
                fm = compute_fold_metrics(
                    base_st, sleeve, common_dates, fold, capital, sl_cagr_a)
                folds_m.append(fm)
            wf = aggregate_wf(folds_m)

        case_results[case] = {"folds": folds_m, "wf": wf}

        ok = "✅ ADOPTED" if wf.get("adopted") else "❌"
        for fold, fm in zip(FOLDS, folds_m):
            if fm:
                fp = "✅" if fm.get("fold_pass") else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"sl={fm['sl_cagr']:+.1f}%  "
                      f"Δ={fm['delta_cagr']:+.2f}pp  "
                      f"ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"trig={fm['trig_yr']:.1f}/yr  "
                      f"hold={fm['avg_hold']:.1f}d  "
                      f"bounce={fm['bounce_rate']*100:.0f}%  {fp}")
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"sl={wf.get('avg_sl_cagr',0):.1f}%  "
              f"trig={wf.get('avg_trig',0):.1f}/yr  "
              f"hold={wf.get('avg_hold',0):.1f}d  "
              f"α_ret={wf.get('avg_alpha_ret',0)*100:.0f}%  {ok}\n")

    # Recompute all cases with correct baseline (except A already done)
    for case in list(EXIT_CASES.keys())[1:]:
        ep = EXIT_CASES[case]
        sleeve = run_sleeve(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, common_dates, base_st,
            ep["thr"], ep["consec"], ep["min_hold"],
        )
        folds_m = []
        for fold in FOLDS:
            fm = compute_fold_metrics(
                base_st, sleeve, common_dates, fold, capital, sl_cagr_a)
            folds_m.append(fm)
        wf = aggregate_wf(folds_m)
        case_results[case] = {"folds": folds_m, "wf": wf}

    adopted = [c for c in EXIT_CASES if case_results[c]["wf"].get("adopted")]
    print(f"{'='*68}")
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for c in EXIT_CASES:
        wf = case_results[c]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        print(f"  {ok} Case {c}: WF={wf.get('n_pass',0)}/5  "
              f"sl={wf.get('avg_sl_cagr',0):.1f}%  "
              f"trig={wf.get('avg_trig',0):.1f}/yr  "
              f"α_ret={wf.get('avg_alpha_ret',0)*100:.0f}%")
    print(f"{'='*68}\n")

    write_report(case_results, topix_close, sl_cagr_a,
                 REPORTS_DIR / "dedicated_alpha_exit_hysteresis.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
