"""
backtest/dedicated_alpha_entry_quality.py  —  Study 8B

固定:  EXIT=RSR<90, SLEEVE_CAP_FR=25%, max_pos=1
変化:  Entry RSR band × d90 threshold (Case A–F)
追加:  entry_edge = median(fwd20_entry) − median(universe_fwd20_on_entry_dates)
報告:  Alpha loss per trigger removed (vs Case A)

Run:
    cd C:/ai-trading
    python src/backtest/dedicated_alpha_entry_quality.py
"""

from __future__ import annotations

import sys, time, math, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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

# ── Entry cases (EXIT fixed = RSR < 90) ───────────────────────────────
ENTRY_CASES = {
    "A": {"rsr_lo": 90.0, "rsr_hi": 95.0, "d90_max": 5, "label": "RSR[90,95) d90≤5"},
    "B": {"rsr_lo": 92.0, "rsr_hi": 95.0, "d90_max": 5, "label": "RSR[92,95) d90≤5"},
    "C": {"rsr_lo": 90.0, "rsr_hi": 95.0, "d90_max": 3, "label": "RSR[90,95) d90≤3"},
    "D": {"rsr_lo": 92.0, "rsr_hi": 95.0, "d90_max": 3, "label": "RSR[92,95) d90≤3"},
    "E": {"rsr_lo": 93.0, "rsr_hi": 95.0, "d90_max": 2, "label": "RSR[93,95) d90≤2"},
    "F": {"rsr_lo": 92.0, "rsr_hi": 95.0, "d90_max": 2, "label": "RSR[92,95) d90≤2"},
}

SLEEVE_CAP_FR   = 0.25
SLEEVE_MAX_POS  = 1
SHOCK_MKT_THR   = -0.05
SHOCK_SYM_THR   = -0.08

# Adoption thresholds
ADOPT_WF_MIN    = 4
ADOPT_SL_CAGR   = 30.0   # %
ADOPT_DCAGR     = 0.3    # pp
ADOPT_DDD       = 1.5    # pp
ADOPT_TRIG_MAX  = 8.0    # /yr

STATE_SCORE_MAP = {
    "EARLY_UP": 3, "STEADY_UP": 2, "STALL": 1,
    "FLAT": 1, "DOWN": 0, "EARLY_ROLL": -1, "UNKNOWN": 0,
}


# ─────────────────────────────────────────────────────────────────────
#  HELPER MATRICES
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
    yr  = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ─────────────────────────────────────────────────────────────────────
#  BASE-ONLY SIMULATION  (run once, reuse across all entry cases)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BaseState:
    equity:    np.ndarray        # daily base equity [n_dates]
    invested:  np.ndarray        # daily base invested [n_dates]
    held:      list              # list[set[str]] per day
    trades:    list              # trade records
    close_mat: np.ndarray        # kept for fwd-return lookup


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
    MAX_SECTOR_W  = float(rc.sector_cap)  if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",        1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    cash     = float(capital)
    pos: dict[str, Position] = {}
    peak     = float(capital)
    cb_active= False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    trades: list[dict] = []

    n_dates = len(common_dates)
    eq_arr  = np.zeros(n_dates, dtype=np.float64)
    inv_arr = np.zeros(n_dates, dtype=np.float64)
    held    = [set() for _ in range(n_dates)]

    for i, date in enumerate(common_dates):
        date_str = str(date.date())
        b_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]]) for s, p in pos.items())
        b_eq  = cash + b_inv
        eq_arr[i]  = b_eq
        inv_arr[i] = b_inv
        held[i]    = set(pos.keys())

        if b_eq > peak: peak = b_eq
        dd = (b_eq - peak) / peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0

        gross_cap   = gross_normal
        if gross_enabled and topix_ret20 is not None:
            r20 = float(topix_ret20[i]) if i < len(topix_ret20) else 0.0
            r60 = float(topix_ret60[i]) if i < len(topix_ret60) else 0.0
            if r20 < -0.05:   gross_cap = gross_dd5
            elif r60 < -0.08: gross_cap = gross_dd8

        is_bear     = bool(bear_arr[i]) if bear_arr is not None else False
        sec_cap_eff = 0.18 if is_bear else MAX_SECTOR_W
        mkt_shock   = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR)

        if i + 1 >= n_dates: break
        next_i = i + 1

        sells, buys = [], []
        for sym in active_syms:
            si      = sym_to_i[sym]
            holding = sym in pos
            hold_idx= (i - pos[sym].entry_idx) if holding else 0
            rsr_v   = float(rsr_mat[i, si])
            close_t = float(close_mat[i, si])

            if mkt_shock and holding:
                if i > 0:
                    prev_c = float(close_mat[i - 1, si])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= SHOCK_SYM_THR:
                        sells.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not holding: continue

            if holding and max_hold is not None and hold_idx > max_hold:
                sells.append((sym, "TIME_STOP")); continue
            if holding and rsr_v < rsr_exit_thr and hold_idx >= min_hold:
                sells.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and holding and hold_idx >= min_hold:
                sells.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not holding:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
                buys.append((rsr_v, sym))

        for sym, reason in sells:
            if sym not in pos: continue
            p       = pos[sym]
            sell_px = float(open_mat[next_i, sym_to_i[sym]])
            pnl     = (sell_px - p.entry_price) * p.qty
            cash   += p.qty * sell_px * (1 - COST_ONE_WAY)
            trades.append({"side": "SELL", "symbol": sym, "pnl": pnl,
                           "entry": p.entry_price, "exit": sell_px,
                           "qty": p.qty, "entry_idx": p.entry_idx,
                           "exit_idx": i, "reason": reason, "date": date_str})
            del pos[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not cb_active and buys:
            buys.sort(key=lambda x: -x[0])
            for rsr_v, sym in buys:
                si      = sym_to_i[sym]
                if max_pos - len(pos) <= 0: break
                buy_px  = float(open_mat[next_i, si])
                if buy_px <= 0: continue
                if not _sector_ok(sym, pos, close_mat, i, sym_to_i, trade_syms,
                                   capital, sec_cap_eff): continue
                if gross_enabled:
                    cur_g = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in pos.values()) / max(1.0, capital)
                    if cur_g + buy_px * LOT / max(1.0, capital) > gross_cap: continue
                alloc = capital / max_pos
                qty   = int(alloc / buy_px / LOT) * LOT
                if qty <= 0: continue
                if qty * buy_px * (1 + COST_ONE_WAY) > cash: continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms,
                              trades, pos, rsr_v)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                trades[-1]["date"] = date_str

    return BaseState(
        equity=eq_arr, invested=inv_arr,
        held=held, trades=trades, close_mat=close_mat,
    )


# ─────────────────────────────────────────────────────────────────────
#  SLEEVE-ONLY SIMULATION
# ─────────────────────────────────────────────────────────────────────

def run_sleeve(
    open_mat, close_mat,
    rsr_mat, sym_active_mat,
    mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, common_dates,
    base_state: BaseState,
    rsr_lo: float, rsr_hi: float, d90_max: int,
) -> dict:
    """Sleeve simulation with fixed EXIT=RSR<90."""
    n_dates   = len(common_dates)
    sl_cash   = float(base_state.equity[0]) * SLEEVE_CAP_FR  # initial sleeve capital
    sl_pos: dict[str, Position] = {}
    sl_trades: list[dict] = []
    sl_eq     = np.zeros(n_dates, dtype=np.float64)
    sl_inv    = np.zeros(n_dates, dtype=np.float64)
    sl_sym    = [None] * n_dates

    for i, date in enumerate(common_dates):
        date_str = str(date.date())
        s_inv    = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                       for s, p in sl_pos.items())
        s_eq     = sl_cash + s_inv
        sl_eq[i]  = s_eq
        sl_inv[i] = s_inv
        sl_sym[i] = list(sl_pos.keys())[0] if sl_pos else None

        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT_THR)

        if i + 1 >= n_dates: break
        next_i = i + 1

        # ── EXIT: RSR < rsr_lo  OR  market shock ─────────────────────
        if sl_pos:
            sym = list(sl_pos.keys())[0]
            p   = sl_pos[sym]
            si  = sym_to_i[sym]
            rsr_v   = float(rsr_mat[i, si])
            close_t = float(close_mat[i, si])

            do_exit, reason = False, ""
            if mkt_shock and i > 0:
                prev_c = float(close_mat[i - 1, si])
                if prev_c > 0 and (close_t / prev_c - 1.0) <= SHOCK_SYM_THR:
                    do_exit, reason = True, "MARKET_SHOCK_EXIT"
            if not do_exit and rsr_v < rsr_lo:
                do_exit, reason = True, "RSR_EXIT_LO"

            if do_exit:
                sell_px = float(open_mat[next_i, si])
                pnl     = (sell_px - p.entry_price) * p.qty
                hold_d  = i - p.entry_idx
                sl_cash += p.qty * sell_px * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side": "SELL", "symbol": sym, "pnl": pnl,
                    "entry": p.entry_price, "exit": sell_px,
                    "qty": p.qty, "entry_idx": p.entry_idx, "exit_idx": i,
                    "reason": reason, "date": date_str, "hold_days": hold_d,
                    "rsr_entry": p.rsr_at_entry,
                    "fwd20_entry": np.nan,   # filled post-hoc from ENTRY date
                    "universe_fwd20": np.nan,
                })
                del sl_pos[sym]

        # ── ENTRY ─────────────────────────────────────────────────────
        if not sl_pos and not mkt_shock:
            base_held = base_state.held[i]
            cands: list[tuple] = []
            for sym in active_syms:
                if sym in base_held: continue
                si    = sym_to_i[sym]
                rsr_v = float(rsr_mat[i, si])
                if not (rsr_lo <= rsr_v < rsr_hi): continue
                d90 = int(cross90_mat[i, si])
                if d90 > d90_max: continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                    continue
                st = classify_state(float(slope5_mat[i, si]),
                                    float(slope20_mat[i, si]))
                sr = STATE_SCORE_MAP.get(st, 0)
                cands.append((rsr_v, d90, sr, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                rsr_v, d90, sr, sym = cands[0]
                si     = sym_to_i[sym]
                buy_px = float(open_mat[next_i, si])
                if buy_px > 0:
                    b_eq  = base_state.equity[i]
                    alloc = min(sl_cash, b_eq * SLEEVE_CAP_FR)
                    qty   = int(alloc / buy_px / LOT) * LOT
                    cost  = qty * buy_px * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos[sym] = Position(sym, "", qty, buy_px, next_i, rsr_v)
                        sl_trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": buy_px, "exit": None,
                            "qty": qty, "pnl": None,
                            "entry_idx": next_i, "exit_idx": None,
                            "reason": f"RSR={rsr_v:.1f} d90={d90}",
                            "date": date_str, "rsr_entry": rsr_v, "hold_days": None,
                            "fwd20_entry": np.nan, "universe_fwd20": np.nan,
                        })

    # ── Post-hoc: fwd20 from ENTRY date ──────────────────────────────
    buy_by_idx = {t["entry_idx"]: t for t in sl_trades if t["side"] == "BUY"}
    for t in sl_trades:
        if t["side"] != "SELL": continue
        ei  = t["entry_idx"]
        si  = sym_to_i.get(t["symbol"], -1)
        if si < 0: continue

        # fwd20 of this stock from entry date
        if ei + 20 < n_dates:
            c0 = float(close_mat[ei, si])
            cn = float(close_mat[ei + 20, si])
            t["fwd20_entry"] = (cn / c0 - 1) * 100 if c0 > 0 and cn > 0 else np.nan

        # universe fwd20: median fwd20 of all active_syms from entry date
        uni_fwd = []
        for s2 in active_syms:
            si2 = sym_to_i[s2]
            if sym_active_mat is not None and float(sym_active_mat[ei, si2]) < 0.5:
                continue
            if ei + 20 < n_dates:
                c0 = float(close_mat[ei, si2])
                cn = float(close_mat[ei + 20, si2])
                if c0 > 0 and cn > 0: uni_fwd.append((cn / c0 - 1) * 100)
        t["universe_fwd20"] = float(np.median(uni_fwd)) if uni_fwd else np.nan

    return {
        "sl_eq":   sl_eq,
        "sl_inv":  sl_inv,
        "sl_sym":  sl_sym,
        "sl_trades": sl_trades,
    }


# ─────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ─────────────────────────────────────────────────────────────────────

def fold_metrics(
    base_st: BaseState,
    sleeve: dict,
    common_dates,
    fold: dict,
    capital: float,
) -> dict:
    date_strs   = [str(d.date()) for d in common_dates]
    oos_mask    = [(fold["oos_start"] <= ds <= fold["oos_end"]) for ds in date_strs]
    oos_idx     = [i for i, m in enumerate(oos_mask) if m]
    if not oos_idx: return {}

    si, ei = oos_idx[0], oos_idx[-1] + 1
    oos_dates   = list(common_dates[si:ei])
    n_days      = len(oos_dates)
    n_years     = n_days / 252.0

    base_oos    = base_st.equity[si:ei].tolist()
    sl_oos      = sleeve["sl_eq"][si:ei].tolist()
    comb_oos    = (base_st.equity + sleeve["sl_eq"])[si:ei].tolist()
    base_inv_o  = base_st.invested[si:ei].tolist()
    sl_inv_o    = sleeve["sl_inv"][si:ei].tolist()

    if len(base_oos) < 5: return {}

    ds_s, ds_e = fold["oos_start"], fold["oos_end"]

    base_t = [t for t in base_st.trades if ds_s <= t.get("date","") <= ds_e]
    sl_t   = [t for t in sleeve["sl_trades"] if ds_s <= t.get("date","") <= ds_e]
    sl_sell= [t for t in sl_t if t["side"] == "SELL"]
    comb_t = base_t + sl_t

    # ── Core metrics ──────────────────────────────────────────────────
    base_exp_l = [bi / max(1, be) for bi, be in zip(base_inv_o, base_oos)]
    comb_exp_l = [(bi + si) / max(1, be + se)
                  for bi, si, be, se in zip(base_inv_o, sl_inv_o, base_oos, sl_oos)]

    base_m = calc_metrics(base_oos, base_t, base_exp_l, base_oos[0], oos_dates)
    comb_m = calc_metrics(comb_oos, comb_t, comb_exp_l, comb_oos[0], oos_dates)

    sl_m: dict = {}
    if sl_oos[0] > 0 and len(sl_oos) >= 5:
        sl_exp_l = [si / max(1, se) for si, se in zip(sl_inv_o, sl_oos)]
        sl_m = calc_metrics(sl_oos, sl_t, sl_exp_l, sl_oos[0], oos_dates)

    # ── Incremental ───────────────────────────────────────────────────
    delta_cagr   = comb_m.get("cagr", 0) - base_m.get("cagr", 0)
    delta_dd     = -(comb_m.get("max_dd", 0) - base_m.get("max_dd", 0))  # pos=worse
    delta_calmar = comb_m.get("calmar", 0) - base_m.get("calmar", 0)

    # ── Exposure efficiency ───────────────────────────────────────────
    base_avg_exp = float(np.mean(base_exp_l))
    comb_avg_exp = float(np.mean(comb_exp_l))
    delta_exp    = comb_avg_exp - base_avg_exp
    base_eff     = base_m.get("cagr", 0) / max(0.001, base_avg_exp)
    sleeve_eff   = delta_cagr / delta_exp if delta_exp > 0.001 else 0.0
    exp_eff_pass = sleeve_eff > base_eff and delta_exp > 0.001

    # ── Activity ──────────────────────────────────────────────────────
    n_entries        = sum(1 for t in sl_t if t["side"] == "BUY")
    n_exits          = len(sl_sell)
    trigger_yr       = n_exits / max(0.01, n_years)

    hold_days_list   = [t["hold_days"] for t in sl_sell
                        if t.get("hold_days") is not None and t["hold_days"] > 0]
    avg_hold         = float(np.mean(hold_days_list)) if hold_days_list else 0.0
    days_to_exit_rsr = [t["hold_days"] for t in sl_sell
                        if t.get("reason") == "RSR_EXIT_LO" and t.get("hold_days", 0) > 0]
    avg_days_exit    = float(np.mean(days_to_exit_rsr)) if days_to_exit_rsr else 0.0

    invested_days    = sum(1 for v in sl_inv_o if v > 0)
    cash_util        = invested_days / max(1, n_days)

    # ── Overlap ───────────────────────────────────────────────────────
    sl_syms   = sleeve["sl_sym"][si:ei]
    base_held = base_st.held[si:ei]
    overlap   = sum(1 for h, bs in zip(sl_syms, base_held)
                    if h is not None and h in bs)

    # ── Sleeve quality ────────────────────────────────────────────────
    wins = [t for t in sl_sell if (t.get("pnl") or 0) > 0]
    loss = [t for t in sl_sell if (t.get("pnl") or 0) <= 0]
    gp   = sum(t["pnl"] for t in wins) if wins else 0.0
    gl   = abs(sum(t["pnl"] for t in loss)) if loss else 0.0
    pf   = gp / max(1.0, gl)

    # ── Entry edge ────────────────────────────────────────────────────
    fwd20_entry = [t["fwd20_entry"]   for t in sl_sell
                   if not math.isnan(t.get("fwd20_entry", float("nan")))]
    uni_fwd20   = [t["universe_fwd20"] for t in sl_sell
                   if not math.isnan(t.get("universe_fwd20", float("nan")))]
    med_fwd20_entry = float(np.median(fwd20_entry)) if fwd20_entry else float("nan")
    med_uni_fwd20   = float(np.median(uni_fwd20))   if uni_fwd20   else float("nan")
    entry_edge = (med_fwd20_entry - med_uni_fwd20
                  if not math.isnan(med_fwd20_entry) and not math.isnan(med_uni_fwd20)
                  else float("nan"))

    # ── Fold pass ─────────────────────────────────────────────────────
    sl_cagr = sl_m.get("cagr", 0)
    fold_pass = (
        sl_cagr >= ADOPT_SL_CAGR
        and delta_cagr > ADOPT_DCAGR
        and delta_dd <= ADOPT_DDD
        and trigger_yr <= ADOPT_TRIG_MAX
    )

    return {
        "base_cagr":    base_m.get("cagr", 0),
        "base_max_dd":  base_m.get("max_dd", 0),
        "comb_cagr":    comb_m.get("cagr", 0),
        "comb_max_dd":  comb_m.get("max_dd", 0),
        "comb_calmar":  comb_m.get("calmar", 0),
        "sl_cagr":      sl_cagr,
        "sl_calmar":    sl_m.get("calmar", 0),
        "delta_cagr":   round(delta_cagr,   2),
        "delta_dd":     round(delta_dd,     2),
        "delta_calmar": round(delta_calmar, 3),
        "trigger_yr":   round(trigger_yr,   1),
        "avg_hold":     round(avg_hold,     1),
        "days_to_exit": round(avg_days_exit,1),
        "cash_util":    round(cash_util * 100, 1),
        "entry_count":  n_entries,
        "overlap":      overlap,
        "profit_factor":round(pf, 3),
        "hit_rate":     round(len(wins) / max(1, n_exits) * 100, 1),
        "med_fwd20_entry": med_fwd20_entry,
        "med_uni_fwd20":   med_uni_fwd20,
        "entry_edge":      entry_edge,
        "base_avg_exp": round(base_avg_exp * 100, 1),
        "comb_avg_exp": round(comb_avg_exp * 100, 1),
        "delta_exp":    round(delta_exp * 100, 1),
        "exp_eff_pass": exp_eff_pass,
        "fold_pass":    fold_pass,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF AGGREGATE
# ─────────────────────────────────────────────────────────────────────

def aggregate_wf(folds: list[dict]) -> dict:
    valid = [f for f in folds if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(key): return float(np.mean([f[key] for f in valid]))

    fwd20_vals = [f["entry_edge"] for f in valid
                  if not math.isnan(f.get("entry_edge", float("nan")))]

    adopted = (
        n_pass >= ADOPT_WF_MIN
        and avg("sl_cagr")    >= ADOPT_SL_CAGR
        and avg("delta_cagr") >  ADOPT_DCAGR
        and avg("delta_dd")   <= ADOPT_DDD
        and avg("trigger_yr") <= ADOPT_TRIG_MAX
    )

    return {
        "n_pass":        n_pass,
        "avg_sl_cagr":   round(avg("sl_cagr"),    1),
        "avg_dcagr":     round(avg("delta_cagr"),  2),
        "avg_dd":        round(avg("delta_dd"),    2),
        "avg_dcalmar":   round(avg("delta_calmar"),3),
        "avg_trig":      round(avg("trigger_yr"),  1),
        "avg_hold":      round(avg("avg_hold"),    1),
        "avg_days_exit": round(avg("days_to_exit"),1),
        "avg_util":      round(avg("cash_util"),   1),
        "avg_entries":   round(avg("entry_count"), 1),
        "avg_overlap":   round(avg("overlap"),     1),
        "avg_pf":        round(avg("profit_factor"),3),
        "avg_hit":       round(avg("hit_rate"),    1),
        "avg_entry_edge":round(float(np.mean(fwd20_vals)), 2) if fwd20_vals else float("nan"),
        "eff_passes":    sum(1 for f in valid if f.get("exp_eff_pass")),
        "adopted":       adopted,
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    topix_close: pd.Series | None,
    path: Path,
) -> None:
    L = []; w = L.append

    def pp(v): return f"{v:+.2f}pp" if not math.isnan(float(v)) else "—"
    def pct(v): return f"{v:.1f}%"  if not math.isnan(float(v)) else "—"
    def ee(v):  return f"{v:+.2f}pp" if not math.isnan(float(v)) else "—"

    adopted = [c for c in ENTRY_CASES if case_results[c]["wf"].get("adopted")]
    verdict = "✅ GO LIVE" if adopted else "🔬 KEEP RESEARCH"

    w("# Dedicated Alpha Sleeve — Entry Quality Compression  (Study 8B)")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n固定: EXIT=RSR<90, cap=equity×25%, max_pos=1")
    w(f"\n採用条件: WF≥{ADOPT_WF_MIN}/5, sl_CAGR≥{ADOPT_SL_CAGR}%, "
      f"ΔCAGR>+{ADOPT_DCAGR}pp, ΔDD≤+{ADOPT_DDD}pp, trigger≤{ADOPT_TRIG_MAX}/yr\n")
    w(f"**採用Case**: {len(adopted)}件  |  最終判定: **{verdict}**\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    bl_cagrs = [f["base_cagr"] for f in case_results["A"]["folds"] if f]
    w(f"- ベースライン avg CAGR: **{float(np.mean(bl_cagrs)):+.1f}%** (production P0)")
    w(f"- Case A = Study8 Case B (baseline: RSR[90,95) d90≤5, EXIT=RSR<90)")
    w(f"- 採用: **{len(adopted)}件**{' — ' + ', '.join(adopted) if adopted else ''}")
    wf_a = case_results["A"]["wf"]
    w(f"- Case A 参照: sl_CAGR={wf_a.get('avg_sl_cagr',0):.1f}%  "
      f"ΔCAGR={wf_a.get('avg_dcagr',0):+.2f}pp  "
      f"trig={wf_a.get('avg_trig',0):.1f}/yr  WF={wf_a.get('n_pass',0)}/5\n")

    # ── S2. WF Summary Table ──────────────────────────────────────────
    w("---\n## 2. WF サマリ (6 Entry Case)\n")
    w("| Case | Entry条件 | WF | sl_CAGR | ΔCAGR | ΔDD | trig/yr | entry_edge | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for case, ep in ENTRY_CASES.items():
        wf   = case_results[case]["wf"]
        mark = "**✅**" if wf.get("adopted") else "❌"
        eedge = ee(wf.get("avg_entry_edge", float("nan")))
        w(f"| {case} | {ep['label']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_sl_cagr',0):.1f}% "
          f"| {wf.get('avg_dcagr',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_trig',0):.1f} "
          f"| {eedge} "
          f"| {mark} |")

    # ── S3. Alpha Loss per Trigger Removed ───────────────────────────
    w("\n---\n## 3. Alpha Loss per Trigger Removed (vs Case A)\n")
    w("> 基準 Case A の trigger/yr, ΔCAGR からどれだけ alpha を失うか。")
    w("> alpha_loss = ΔΔCalmar / Δtrigger（Calmar の純変化 / 削除された trigger 数）\n")
    w("| Case | Δtrig/yr vs A | ΔΔCalmar vs A | alpha_loss / trigger | 評価 |")
    w("|---|---|---|---|---|")
    wf_a = case_results["A"]["wf"]
    trig_a  = wf_a.get("avg_trig",    0)
    dcal_a  = wf_a.get("avg_dcalmar", 0)
    for case in ENTRY_CASES:
        wf      = case_results[case]["wf"]
        trig_x  = wf.get("avg_trig",    0)
        dcal_x  = wf.get("avg_dcalmar", 0)
        d_trig  = trig_a - trig_x          # triggers removed (positive = fewer)
        d_dcal  = dcal_x - dcal_a          # Calmar change from A (positive = better than A)
        if d_trig <= 0.01:
            loss_str = "N/A (trigger増)"
            eval_str = "参考外"
        else:
            loss_per = -d_dcal / d_trig    # positive = Calmar lost per trigger removed
            loss_str = f"{loss_per:+.4f} Calmar/trigger"
            eval_str = "✅ 効率改善" if d_dcal >= 0 else f"❌ -{loss_per:.4f} Calmar loss"
        w(f"| {case} | {d_trig:+.1f} | {d_dcal:+.3f} | {loss_str} | {eval_str} |")

    # ── S4. Entry Quality Audit ───────────────────────────────────────
    w("\n---\n## 4. Entry Quality Audit\n")
    w("| Case | sl_CAGR | sl_Calmar | hit_rate | PF | avg_hold | days_to_exit "
      "| cash_util | entries | overlap | entry_edge |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for case, ep in ENTRY_CASES.items():
        wf = case_results[case]["wf"]
        ee_v = ee(wf.get("avg_entry_edge", float("nan")))
        w(f"| {case} "
          f"| {wf.get('avg_sl_cagr',0):.1f}% "
          f"| {wf.get('avg_dcalmar',0):+.3f} "
          f"| {wf.get('avg_hit',0):.1f}% "
          f"| {wf.get('avg_pf',0):.3f} "
          f"| {wf.get('avg_hold',0):.1f}d "
          f"| {wf.get('avg_days_exit',0):.1f}d "
          f"| {wf.get('avg_util',0):.1f}% "
          f"| {wf.get('avg_entries',0):.0f} "
          f"| {wf.get('avg_overlap',0):.0f}d "
          f"| {ee_v} |")

    # ── S5. Fold Detail ───────────────────────────────────────────────
    w("\n---\n## 5. Fold 詳細\n")
    w("| Case | Fold | OOS年 | Regime | base | sl_CAGR | ΔCAGR | ΔDD | trig/yr "
      "| entry_edge | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in ENTRY_CASES:
        for fold, fm in zip(FOLDS, case_results[case]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm.get("fold_pass") else "❌"
            ee_v = ee(fm.get("entry_edge", float("nan")))
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['base_cagr']:+.1f}% "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['trigger_yr']:.1f} "
              f"| {ee_v} "
              f"| {ok} |")

    # ── S6. Failure Analysis ──────────────────────────────────────────
    w("\n---\n## 6. Failure Analysis\n")
    for case, ep in ENTRY_CASES.items():
        wf    = case_results[case]["wf"]
        fails = []
        if wf.get("n_pass",0) < ADOPT_WF_MIN:
            fails.append(f"WF={wf['n_pass']}/5")
        if wf.get("avg_sl_cagr",0) < ADOPT_SL_CAGR:
            fails.append(f"sl_CAGR={wf['avg_sl_cagr']:.1f}% < {ADOPT_SL_CAGR}%")
        if wf.get("avg_dcagr",0) <= ADOPT_DCAGR:
            fails.append(f"ΔCAGR={wf['avg_dcagr']:+.2f}pp")
        if wf.get("avg_dd",0) > ADOPT_DDD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp")
        if wf.get("avg_trig",0) > ADOPT_TRIG_MAX:
            fails.append(f"trig={wf['avg_trig']:.1f}/yr")
        if not fails:
            w(f"\n**{case}** ({ep['label']}): ✅ 全基準クリア")
        else:
            w(f"\n**{case}** ({ep['label']}): REJECT — " + " / ".join(fails))

    # ── S7. Compression Efficiency ────────────────────────────────────
    w("\n---\n## 7. Compression Efficiency サマリ\n")
    w("> d90 と RSR 下限をどの方向で絞ると alpha 損失が最小か。\n")

    d90_axis   = [(c, case_results[c]["wf"]) for c in ["A","C","F","E"] if case_results[c]["wf"]]
    rsr_lo_axis= [(c, case_results[c]["wf"]) for c in ["A","B","D","F"] if case_results[c]["wf"]]

    w("**d90 絞り (RSR≥90, d90: 5→3→2):**")
    for c, wf in [("A", case_results["A"]["wf"]),
                  ("C", case_results["C"]["wf"]),
                  ("F", case_results["F"]["wf"])]:
        w(f"  {c}: trig={wf.get('avg_trig',0):.1f}  sl_CAGR={wf.get('avg_sl_cagr',0):.1f}%  "
          f"entry_edge={ee(wf.get('avg_entry_edge', float('nan')))}")

    w("\n**RSR下限絞り (d90≤5, RSR: 90→92→93):**")
    for c, wf in [("A", case_results["A"]["wf"]),
                  ("B", case_results["B"]["wf"]),
                  ("E", case_results["E"]["wf"])]:
        w(f"  {c}: trig={wf.get('avg_trig',0):.1f}  sl_CAGR={wf.get('avg_sl_cagr',0):.1f}%  "
          f"entry_edge={ee(wf.get('avg_entry_edge', float('nan')))}")

    # ── S8. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 8. Final Recommendation\n")
    w(f"## `{verdict}`\n")

    if adopted:
        best = max(adopted, key=lambda c: case_results[c]["wf"].get("avg_dcagr", 0))
        bwf  = case_results[best]["wf"]
        w(f"**採用推奨: Case {best}** — {ENTRY_CASES[best]['label']}\n")
        w(f"- sl_CAGR: {bwf['avg_sl_cagr']:.1f}%  ΔCAGR: {bwf['avg_dcagr']:+.2f}pp  "
          f"ΔDD: {bwf['avg_dd']:+.2f}pp")
        w(f"- trigger/yr: {bwf['avg_trig']:.1f}  WF: {bwf['n_pass']}/5")
        w(f"- entry_edge: {ee(bwf.get('avg_entry_edge', float('nan')))}\n")
        w("次ステップ (ASK_FIRST必須):")
        w("1. signal_bridge.py へのスリーブ注文追加")
        w("2. 独立 sleeve_state.json の設計")
        w("3. live dry-run 30日以上")
    else:
        best = max(ENTRY_CASES, key=lambda c: case_results[c]["wf"].get("avg_dcagr", 0))
        bwf  = case_results[best]["wf"]
        w(f"全6 Case が採用基準未達。最良: **Case {best}** ({ENTRY_CASES[best]['label']})\n")
        w(f"- sl_CAGR={bwf['avg_sl_cagr']:.1f}%  "
          f"ΔCAGR={bwf['avg_dcagr']:+.2f}pp  "
          f"trig={bwf['avg_trig']:.1f}/yr  WF={bwf['n_pass']}/5\n")

        # Identify binding constraint
        fails: list[str] = []
        if bwf.get("n_pass",0)      < ADOPT_WF_MIN:  fails.append(f"WF={bwf['n_pass']}/5")
        if bwf.get("avg_sl_cagr",0) < ADOPT_SL_CAGR: fails.append(f"sl_CAGR={bwf['avg_sl_cagr']:.1f}%")
        if bwf.get("avg_dcagr",0)   <= ADOPT_DCAGR:   fails.append(f"ΔCAGR={bwf['avg_dcagr']:+.2f}pp")
        if bwf.get("avg_dd",0)      > ADOPT_DDD:      fails.append(f"ΔDD={bwf['avg_dd']:+.2f}pp")
        if bwf.get("avg_trig",0)    > ADOPT_TRIG_MAX: fails.append(f"trig={bwf['avg_trig']:.1f}/yr")
        w("**バインディング制約**: " + " / ".join(fails) + "\n")

        w("継続研究候補:")
        w("- d90 + RSR 絞りでは alpha 損失が先に来る → 別次元の filter 探索が必要")
        w("- Study 8C: ΔDD を減らす capital ratio 削減 (25%→15%)")
        w("- Study 8D: entry filter に state=EARLY_UP 必須を追加")
        w("- Study 8E: Bear フォールド (2022) を除外した Bull 時限定採用の再評価")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Study 8B: Alpha Sleeve Entry Quality Compression WF")
    print("  EXIT=RSR<90 固定 / Entry: RSR band × d90 (6 cases)")
    print("=" * 68 + "\n")

    # ── Data ─────────────────────────────────────────────────────────
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

    # ── Matrices ─────────────────────────────────────────────────────
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

    cross90_mat            = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)

    # ── Base simulation (once) ────────────────────────────────────────
    print("[3/4] Base シミュレーション (1回)...")
    base_st = run_base(
        open_mat, close_mat,
        sig_mat, sig_ready, rsr_mat, sym_active_mat,
        mkt_ret1, topix_ret20, topix_ret60, bear_arr,
        active_syms, sym_to_i, trade_syms, cfg, common_dates,
    )
    capital = float(cfg.portfolio.capital)

    # Base fold check
    base_cagrs = []
    for fold in FOLDS:
        ds = [str(d.date()) for d in common_dates]
        idx = [i for i, s in enumerate(ds)
               if fold["oos_start"] <= s <= fold["oos_end"]]
        if idx:
            eq  = base_st.equity[idx[0]:idx[-1]+1]
            dts = list(common_dates[idx[0]:idx[-1]+1])
            if len(eq) > 5:
                m = calc_metrics(eq.tolist(), [], [], eq[0], dts)
                base_cagrs.append(m.get("cagr", 0))
    print(f"  Base avg CAGR: {float(np.mean(base_cagrs)):+.1f}%\n")

    # ── Sleeve simulation (6 cases) ───────────────────────────────────
    print("[4/4] Entry Case シミュレーション...")
    case_results = {}

    for case, ep in ENTRY_CASES.items():
        print(f"  [{case}] {ep['label']}")
        sleeve = run_sleeve(
            open_mat, close_mat,
            rsr_mat, sym_active_mat,
            mkt_ret1, cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, common_dates,
            base_st, ep["rsr_lo"], ep["rsr_hi"], ep["d90_max"],
        )

        folds_m = []
        for fold in FOLDS:
            fm = fold_metrics(base_st, sleeve, common_dates, fold, capital)
            folds_m.append(fm)
            if fm:
                ok = "✅" if fm.get("fold_pass") else "❌"
                ee_v = (f"{fm['entry_edge']:+.1f}pp"
                        if not math.isnan(fm.get("entry_edge", float("nan")))
                        else "—")
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"sl={fm['sl_cagr']:+.1f}%  "
                      f"Δ={fm['delta_cagr']:+.2f}pp  "
                      f"trig={fm['trigger_yr']:.1f}/yr  "
                      f"ee={ee_v}  {ok}")

        wf = aggregate_wf(folds_m)
        case_results[case] = {"folds": folds_m, "wf": wf}

        ok_str = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"sl_CAGR={wf.get('avg_sl_cagr',0):.1f}%  "
              f"ΔCAGR={wf.get('avg_dcagr',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"trig={wf.get('avg_trig',0):.1f}/yr  {ok_str}\n")

    # Summary
    adopted = [c for c in ENTRY_CASES if case_results[c]["wf"].get("adopted")]
    print(f"{'='*68}")
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for c in ENTRY_CASES:
        wf = case_results[c]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        print(f"  {ok} Case {c}: WF={wf.get('n_pass',0)}/5  "
              f"sl={wf.get('avg_sl_cagr',0):.1f}%  "
              f"Δ={wf.get('avg_dcagr',0):+.2f}pp  "
              f"trig={wf.get('avg_trig',0):.1f}/yr")
    print(f"{'='*68}\n")

    write_report(case_results, topix_close,
                 REPORTS_DIR / "dedicated_alpha_entry_quality.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
