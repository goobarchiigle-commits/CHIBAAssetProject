"""
backtest/dedicated_alpha_sleeve.py  —  Study 8

目的: RSR90–94帯のみを対象とした独立スリーブ（equity×25%）が
      ポートフォリオ全体の効率改善をもたらすか検証。
      単なる露出増加ではなく exposure-efficiency 改善が必要。

禁止: 既存3ポジ変更/entry-exit変更/sizing変更/leverage変更/MSW変更/risk変更

スリーブ:
  capital   = base_equity × 25%（動的上限）
  max_pos   = 1
  entry     = RSR ∈ [90,95) AND days_cross90 ≤ 5
  priority  = ①高RSR ②低d90 ③高state_rank
  exit      = 5 case WF比較 (A/B/C/D/E)

採用条件:
  WF ≥ 4/5  AND  ΔCAGR > +0.3pp  AND  ΔCalmar > 0
  AND  ΔDD ≤ +1.5pp  AND  trigger/yr ≤ 8
  AND  exposure_efficiency_new > base_CAGR/base_exp

出力:
  reports/dedicated_alpha_sleeve.md

Run:
    cd C:/ai-trading
    python src/backtest/dedicated_alpha_sleeve.py
"""

from __future__ import annotations

import sys, time, math, warnings
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
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
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

EXIT_CASES = {
    "A": "RSR > 95",
    "B": "RSR < 90",
    "C": "RSR > 95 OR hold ≥ 10d",
    "D": "RSR < 90 OR hold ≥ 10d",
    "E": "RSR下降2日 OR hold ≥ 10d",
}

SLEEVE_RSR_LO  = 90.0
SLEEVE_RSR_HI  = 95.0    # [90, 95)
SLEEVE_D90_MAX = 5
SLEEVE_MAX_POS = 1
SLEEVE_CAP_FR  = 0.25    # equity × 25%
SLEEVE_TIME    = 10      # max hold days for cases with time exit
SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08

# Adoption criteria
ADOPT_WF_MIN       = 4
ADOPT_DCAGR        = 0.3   # pp
ADOPT_DCALMAR      = 0.0
ADOPT_DDD          = 1.5   # pp max increase
ADOPT_TRIG_MAX     = 8.0   # per year

STATE_SCORE_MAP = {
    "EARLY_UP": 3, "STEADY_UP": 2, "FLAT": 1,
    "STALL": 1,    "DOWN": 0,      "EARLY_ROLL": -1, "UNKNOWN": 0,
}


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
    s5[5:]   = (rsr_mat[5:]   - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:]  - rsr_mat[:-20]) / 20.0
    return s5, s20


def regime_label(topix_close: pd.Series, year: int) -> str:
    yr = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


def should_sleeve_exit(rsr: float, hold_days: int, i: int,
                        si: int, rsr_mat: np.ndarray,
                        exit_case: str) -> tuple[bool, str]:
    if exit_case == "A":
        if rsr > 95.0: return True, "RSR_GRAD_UP"
    elif exit_case == "B":
        if rsr < SLEEVE_RSR_LO: return True, "RSR_EXIT_LO"
    elif exit_case == "C":
        if rsr > 95.0:               return True, "RSR_GRAD_UP"
        if hold_days >= SLEEVE_TIME: return True, "TIME_STOP"
    elif exit_case == "D":
        if rsr < SLEEVE_RSR_LO:      return True, "RSR_EXIT_LO"
        if hold_days >= SLEEVE_TIME: return True, "TIME_STOP"
    elif exit_case == "E":
        if hold_days >= SLEEVE_TIME: return True, "TIME_STOP"
        if i >= 2:
            r0 = float(rsr_mat[i, si])
            r1 = float(rsr_mat[i - 1, si])
            r2 = float(rsr_mat[i - 2, si])
            if r0 < r1 and r1 < r2:  return True, "RSR_DECLINE_2D"
    return False, ""


# ─────────────────────────────────────────────────────────────────────
#  MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────

def run_exit_case(
    open_mat, close_mat,
    sig_mat, sig_ready, rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, trade_syms, cfg,
    common_dates, exit_case: str,
) -> dict:
    """
    Run full 2018-2025 simulation (base + sleeve in parallel).
    Returns raw curves and trade lists; metrics computed per-fold by caller.
    """
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)

    rc = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W  = float(rc.sector_cap)  if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    sleeve_capital = capital * SLEEVE_CAP_FR   # 750K initial

    # ── Base state ────────────────────────────────────────────────────
    base_cash   = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak   = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    # ── Sleeve state ──────────────────────────────────────────────────
    sl_cash = float(sleeve_capital)
    sl_pos: dict[str, Position] = {}   # max 1 entry
    sl_trades: list[dict] = []

    # ── Daily arrays ──────────────────────────────────────────────────
    n_dates   = len(common_dates)
    base_eq   = np.zeros(n_dates, dtype=np.float64)
    sl_eq     = np.zeros(n_dates, dtype=np.float64)
    base_inv  = np.zeros(n_dates, dtype=np.float64)
    sl_inv    = np.zeros(n_dates, dtype=np.float64)

    # date-level tracking for overlap / invested-day analysis
    sl_held_sym  = [None] * n_dates   # which symbol sleeve holds each day
    base_held_set = [set()] * n_dates # set of base symbols held each day

    for i, date in enumerate(common_dates):
        date_str = str(date.date())

        b_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                    for s, p in base_pos.items())
        s_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                    for s, p in sl_pos.items())

        b_eq  = base_cash + b_inv
        s_eq  = sl_cash + s_inv

        base_eq[i]  = b_eq
        sl_eq[i]    = s_eq
        base_inv[i] = b_inv
        sl_inv[i]   = s_inv
        sl_held_sym[i]   = list(sl_pos.keys())[0] if sl_pos else None
        base_held_set[i] = set(base_pos.keys())

        # CB (base only)
        if b_eq > base_peak: base_peak = b_eq
        dd = (b_eq - base_peak) / base_peak
        if not cb_active:
            if dd <= -max_dd_limit: cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0

        # Gross cap
        gross_cap = gross_normal
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

        # ── BASE: collect sell/buy signals ────────────────────────────
        base_sell: list[tuple] = []
        base_buy:  list[tuple] = []   # (rsr_val, sym)

        for sym in active_syms:
            si_sym     = sym_to_i[sym]
            is_holding = sym in base_pos
            hold_idx   = (i - base_pos[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si_sym])
            close_t    = float(close_mat[i, si_sym])

            if mkt_shock and is_holding:
                if i > 0:
                    prev_c = float(close_mat[i - 1, si_sym])
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

        # ── BASE: execute sells ───────────────────────────────────────
        for sym, reason in base_sell:
            if sym not in base_pos: continue
            pos     = base_pos[sym]
            sell_px = float(open_mat[next_i, sym_to_i[sym]])
            pnl     = (sell_px - pos.entry_price) * pos.qty
            base_cash += pos.qty * sell_px * (1 - COST_ONE_WAY)
            base_trades.append({"side": "SELL", "symbol": sym, "pnl": pnl,
                                 "entry": pos.entry_price, "exit": sell_px,
                                 "qty": pos.qty, "entry_idx": pos.entry_idx,
                                 "exit_idx": i, "reason": reason, "date": date_str})
            del base_pos[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        # ── BASE: execute buys ────────────────────────────────────────
        if not cb_active and base_buy:
            base_buy.sort(key=lambda x: -x[0])
            for rsr_v, sym in base_buy:
                si_sym     = sym_to_i[sym]
                open_slots = max_pos - len(base_pos)
                if open_slots <= 0: break
                buy_px = float(open_mat[next_i, si_sym])
                if buy_px <= 0: continue
                if not _sector_ok(sym, base_pos, close_mat, i, sym_to_i, trade_syms,
                                   capital, sec_cap_eff): continue
                if gross_enabled:
                    cur_g = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in base_pos.values()) / max(1.0, capital)
                    if cur_g + buy_px * LOT / max(1.0, capital) > gross_cap: continue
                alloc = capital / max_pos
                qty   = int(alloc / buy_px / LOT) * LOT
                if qty <= 0: continue
                if qty * buy_px * (1 + COST_ONE_WAY) > base_cash: continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms,
                              base_trades, base_pos, rsr_v)
                base_cash -= qty * buy_px * (1 + COST_ONE_WAY)
                base_trades[-1]["date"] = date_str

        # ── SLEEVE: exit ──────────────────────────────────────────────
        if sl_pos:
            sym = list(sl_pos.keys())[0]
            pos = sl_pos[sym]
            si_sym     = sym_to_i[sym]
            hold_days  = i - pos.entry_idx
            rsr_val    = float(rsr_mat[i, si_sym])
            close_t    = float(close_mat[i, si_sym])

            # Sleeve market shock exit
            do_exit, reason = False, ""
            if mkt_shock and i > 0:
                prev_c = float(close_mat[i - 1, si_sym])
                if prev_c > 0 and (close_t / prev_c - 1.0) <= SHOCK_SYM_THR:
                    do_exit, reason = True, "MARKET_SHOCK_EXIT"

            if not do_exit:
                do_exit, reason = should_sleeve_exit(rsr_val, hold_days, i, si_sym,
                                                      rsr_mat, exit_case)
            if do_exit:
                sell_px = float(open_mat[next_i, si_sym])
                pnl     = (sell_px - pos.entry_price) * pos.qty
                sl_cash += pos.qty * sell_px * (1 - COST_ONE_WAY)
                sl_trades.append({
                    "side": "SELL", "symbol": sym, "pnl": pnl,
                    "entry": pos.entry_price, "exit": sell_px,
                    "qty": pos.qty, "entry_idx": pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": date_str,
                    "fwd10": np.nan, "fwd20": np.nan,
                })
                del sl_pos[sym]

        # ── SLEEVE: entry ─────────────────────────────────────────────
        if not sl_pos and not mkt_shock:
            cands: list[tuple] = []   # (rsr, d90, state_rank, sym)
            for sym in active_syms:
                if sym in base_pos: continue   # conflict prohibition
                si_sym  = sym_to_i[sym]
                rsr_v   = float(rsr_mat[i, si_sym])
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
                # Priority: ①高RSR ②低d90 ③高state_rank
                cands.sort(key=lambda x: (-x[0], x[1], -x[2]))
                rsr_v, d90, sr, sym = cands[0]
                si_sym  = sym_to_i[sym]
                buy_px  = float(open_mat[next_i, si_sym])
                if buy_px > 0:
                    alloc = min(sl_cash, b_eq * SLEEVE_CAP_FR)
                    qty   = int(alloc / buy_px / LOT) * LOT
                    cost  = qty * buy_px * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos[sym] = Position(sym, trade_syms.get(sym, ""),
                                               qty, buy_px, next_i, rsr_v)
                        sl_trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": buy_px, "exit": None,
                            "qty": qty, "pnl": None,
                            "entry_idx": next_i, "exit_idx": None,
                            "reason": f"RSR={rsr_v:.1f} d90={d90}",
                            "date": date_str, "rsr_entry": rsr_v,
                            "fwd10": np.nan, "fwd20": np.nan,
                        })

    # ── Post-hoc: fwd10/fwd20 on SELL trades ─────────────────────────
    buy_map: dict[str, dict] = {}  # entry_idx → buy trade
    for t in sl_trades:
        if t["side"] == "BUY":
            buy_map[t["entry_idx"]] = t

    for t in sl_trades:
        if t["side"] != "SELL": continue
        si  = sym_to_i.get(t["symbol"], -1)
        if si < 0: continue
        ei  = t["exit_idx"]
        # fwd10/fwd20 from EXIT date (missed alpha diagnostic)
        for fwd_k, fwd_n in [("fwd10", 10), ("fwd20", 20)]:
            if ei + fwd_n < n_dates:
                c0 = float(close_mat[ei, si])
                cn = float(close_mat[ei + fwd_n, si])
                t[fwd_k] = round((cn / c0 - 1) * 100, 2) if c0 > 0 and cn > 0 else float("nan")

    return {
        "base_eq":       base_eq,
        "sl_eq":         sl_eq,
        "base_inv":      base_inv,
        "sl_inv":        sl_inv,
        "base_trades":   base_trades,
        "sl_trades":     sl_trades,
        "sl_held_sym":   sl_held_sym,
        "base_held_set": base_held_set,
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
    sleeve_capital: float,
) -> dict:
    oos_start_str = fold["oos_start"]
    oos_end_str   = fold["oos_end"]

    # Date range for OOS
    date_strs   = [str(d.date()) for d in common_dates]
    oos_mask    = [(oos_start_str <= ds <= oos_end_str) for ds in date_strs]
    oos_indices = [i for i, m in enumerate(oos_mask) if m]
    if not oos_indices:
        return {}

    oos_si = oos_indices[0]
    oos_ei = oos_indices[-1] + 1

    base_eq  = sim["base_eq"]
    sl_eq    = sim["sl_eq"]
    base_inv = sim["base_inv"]
    sl_inv   = sim["sl_inv"]

    base_oos   = base_eq[oos_si:oos_ei].tolist()
    sl_oos     = sl_eq[oos_si:oos_ei].tolist()
    comb_oos   = (base_eq + sl_eq)[oos_si:oos_ei].tolist()
    base_inv_o = base_inv[oos_si:oos_ei].tolist()
    sl_inv_o   = sl_inv[oos_si:oos_ei].tolist()
    oos_dates  = list(common_dates[oos_si:oos_ei])

    # Unused-cash control: sleeve stays at OOS-start value (0% return)
    sl_start    = sl_oos[0]
    unused_oos  = [base_oos[j] + sl_start for j in range(len(base_oos))]
    total_start = base_oos[0] + sl_start

    # Trade filter
    base_t  = [t for t in sim["base_trades"] if oos_start_str <= t.get("date","") <= oos_end_str]
    sl_t    = [t for t in sim["sl_trades"]   if oos_start_str <= t.get("date","") <= oos_end_str]
    sl_sell = [t for t in sl_t if t["side"] == "SELL"]
    sl_buy  = [t for t in sl_t if t["side"] == "BUY"]
    comb_t  = base_t + sl_t

    n_oos_days = len(oos_dates)
    n_years    = n_oos_days / 252.0

    if not base_oos or len(base_oos) < 5:
        return {}

    # ── Core metrics ──────────────────────────────────────────────────
    base_m = calc_metrics(base_oos, base_t,
                           [bi / max(1, be) for bi, be in zip(base_inv_o, base_oos)],
                           base_oos[0], oos_dates)

    comb_exp_list = [(bi + si) / max(1, be + se)
                     for bi, si, be, se in zip(base_inv_o, sl_inv_o, base_oos, sl_oos)]
    comb_m = calc_metrics(comb_oos, comb_t, comb_exp_list, comb_oos[0], oos_dates)

    unused_m = calc_metrics(unused_oos, base_t,
                             [bi / max(1, u) for bi, u in zip(base_inv_o, unused_oos)],
                             total_start, oos_dates)

    # ── Incremental ───────────────────────────────────────────────────
    delta_cagr   = comb_m.get("cagr", 0)   - base_m.get("cagr", 0)
    comb_dd      = comb_m.get("max_dd", 0)
    base_dd      = base_m.get("max_dd", 0)
    delta_dd     = -(comb_dd - base_dd)     # positive = worse
    delta_calmar = comb_m.get("calmar", 0) - base_m.get("calmar", 0)

    # ── Exposure ──────────────────────────────────────────────────────
    base_avg_exp  = float(np.mean([bi / max(1, be) for bi, be in zip(base_inv_o, base_oos)]))
    comb_avg_exp  = float(np.mean(comb_exp_list))
    delta_exp     = comb_avg_exp - base_avg_exp
    base_cagr_val = base_m.get("cagr", 0)
    base_eff      = base_cagr_val / max(0.001, base_avg_exp)
    if delta_exp > 0.001:
        sleeve_eff     = delta_cagr / delta_exp
        exp_eff_pass   = sleeve_eff > base_eff
    else:
        sleeve_eff   = 0.0
        exp_eff_pass = False

    # ── Sleeve activity ───────────────────────────────────────────────
    n_triggers       = len(sl_sell)
    trigger_per_year = n_triggers / max(0.01, n_years)
    invested_days    = int(sum(1 for v in sl_inv_o if v > 0))
    cash_util_rate   = invested_days / max(1, n_oos_days)

    # Average hold days from BUY-SELL pairs
    hold_days_list: list[float] = []
    sell_map = {t.get("entry_idx", -1): t for t in sl_sell}  # keyed by entry_idx
    for t in sl_sell:
        hold_d = t.get("exit_idx", 0) - t.get("entry_idx", 0)
        if hold_d > 0: hold_days_list.append(float(hold_d))
    avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0.0

    # ── Sleeve quality ────────────────────────────────────────────────
    sl_wins = [t for t in sl_sell if (t.get("pnl") or 0) > 0]
    sl_loss = [t for t in sl_sell if (t.get("pnl") or 0) <= 0]
    hit_rate = len(sl_wins) / max(1, len(sl_sell))
    gp = sum(t["pnl"] for t in sl_wins) if sl_wins else 0.0
    gl = abs(sum(t["pnl"] for t in sl_loss)) if sl_loss else 0.0
    pf = gp / max(1.0, gl)

    fwd10_vals = [t.get("fwd10", float("nan")) for t in sl_sell
                  if not math.isnan(t.get("fwd10", float("nan")))]
    fwd20_vals = [t.get("fwd20", float("nan")) for t in sl_sell
                  if not math.isnan(t.get("fwd20", float("nan")))]
    median_fwd10 = float(np.median(fwd10_vals)) if fwd10_vals else float("nan")
    median_fwd20 = float(np.median(fwd20_vals)) if fwd20_vals else float("nan")

    # Sleeve standalone CAGR (on sleeve capital)
    sl_cagr   = sl_m_calmar = 0.0
    if len(sl_oos) >= 5 and sl_oos[0] > 0:
        sl_m = calc_metrics(sl_oos, sl_t,
                             [si / max(1, se) for si, se in zip(sl_inv_o, sl_oos)],
                             sl_oos[0], oos_dates)
        sl_cagr = sl_m.get("cagr", 0)
        sl_m_calmar = sl_m.get("calmar", 0)

    # ── Interaction ───────────────────────────────────────────────────
    base_dr  = np.diff(base_oos) / np.maximum(np.array(base_oos[:-1]), 1)
    sl_dr    = np.diff(sl_oos)   / np.maximum(np.array(sl_oos[:-1]),   1)
    if len(base_dr) > 10 and np.std(sl_dr) > 1e-10:
        corr = float(np.corrcoef(base_dr, sl_dr)[0, 1])
    else:
        corr = 0.0

    held_syms   = sim["sl_held_sym"][oos_si:oos_ei]
    base_sets   = sim["base_held_set"][oos_si:oos_ei]
    overlap_days = sum(1 for h, bs in zip(held_syms, base_sets) if h is not None and h in bs)

    # ── WF fold pass ─────────────────────────────────────────────────
    fold_pass = (
        delta_cagr > 0
        and delta_calmar > 0
        and delta_dd <= ADOPT_DDD
        and trigger_per_year <= ADOPT_TRIG_MAX
        and exp_eff_pass
    )

    return {
        # Performance
        "base_cagr":    base_m.get("cagr", 0),
        "base_max_dd":  base_dd,
        "base_calmar":  base_m.get("calmar", 0),
        "base_sharpe":  base_m.get("sharpe", 0),
        "base_ntrades": base_m.get("n_trades", 0),
        "comb_cagr":    comb_m.get("cagr", 0),
        "comb_max_dd":  comb_dd,
        "comb_calmar":  comb_m.get("calmar", 0),
        "comb_sharpe":  comb_m.get("sharpe", 0),
        "unused_cagr":  unused_m.get("cagr", 0),
        # Incremental
        "delta_cagr":   round(delta_cagr, 2),
        "delta_dd":     round(delta_dd, 2),
        "delta_calmar": round(delta_calmar, 3),
        # Activity
        "trigger_per_year": round(trigger_per_year, 1),
        "cash_util_rate":   round(cash_util_rate * 100, 1),
        "avg_hold_days":    round(avg_hold, 1),
        "n_sl_trades":      n_triggers,
        # Sleeve quality
        "sl_cagr":       round(sl_cagr, 2),
        "sl_calmar":     round(sl_m_calmar, 3),
        "hit_rate":      round(hit_rate * 100, 1),
        "profit_factor": round(pf, 3),
        "median_fwd10":  median_fwd10,
        "median_fwd20":  median_fwd20,
        # Exposure
        "base_avg_exp":  round(base_avg_exp * 100, 1),
        "comb_avg_exp":  round(comb_avg_exp * 100, 1),
        "delta_exp":     round(delta_exp * 100, 1),
        "base_eff":      round(base_eff, 3),
        "sleeve_eff":    round(sleeve_eff, 3),
        "exp_eff_pass":  exp_eff_pass,
        # Interaction
        "corr_to_base":  round(corr, 3),
        "overlap_days":  overlap_days,
        # WF
        "fold_pass":     fold_pass,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATOR PER EXIT CASE
# ─────────────────────────────────────────────────────────────────────

def evaluate_exit_case(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    n_pass       = sum(1 for f in fold_metrics if f.get("fold_pass"))
    avg_dc       = float(np.mean([f["delta_cagr"]   for f in fold_metrics]))
    avg_dd       = float(np.mean([f["delta_dd"]     for f in fold_metrics]))
    avg_dcal     = float(np.mean([f["delta_calmar"] for f in fold_metrics]))
    avg_trig     = float(np.mean([f["trigger_per_year"] for f in fold_metrics]))
    avg_hold     = float(np.mean([f["avg_hold_days"] for f in fold_metrics]))
    avg_util     = float(np.mean([f["cash_util_rate"] for f in fold_metrics]))
    eff_passes   = sum(1 for f in fold_metrics if f.get("exp_eff_pass"))
    avg_corr     = float(np.mean([f["corr_to_base"] for f in fold_metrics]))
    avg_ov       = float(np.mean([f["overlap_days"] for f in fold_metrics]))
    avg_hit      = float(np.mean([f["hit_rate"] for f in fold_metrics]))
    avg_pf       = float(np.mean([f["profit_factor"] for f in fold_metrics]))

    fwd10_vals = [f["median_fwd10"] for f in fold_metrics
                  if not math.isnan(f.get("median_fwd10", float("nan")))]
    fwd20_vals = [f["median_fwd20"] for f in fold_metrics
                  if not math.isnan(f.get("median_fwd20", float("nan")))]

    adopted = (
        n_pass >= ADOPT_WF_MIN
        and avg_dc > ADOPT_DCAGR
        and avg_dcal > ADOPT_DCALMAR
        and avg_dd <= ADOPT_DDD
        and avg_trig <= ADOPT_TRIG_MAX
        and eff_passes >= ADOPT_WF_MIN
    )

    return {
        "n_pass":    n_pass,
        "avg_dc":    round(avg_dc, 2),
        "avg_dd":    round(avg_dd, 2),
        "avg_dcal":  round(avg_dcal, 3),
        "avg_trig":  round(avg_trig, 1),
        "avg_hold":  round(avg_hold, 1),
        "avg_util":  round(avg_util, 1),
        "eff_passes":eff_passes,
        "avg_corr":  round(avg_corr, 3),
        "avg_overlap":round(avg_ov, 1),
        "avg_hit":   round(avg_hit, 1),
        "avg_pf":    round(avg_pf, 3),
        "med_fwd10": round(float(np.median(fwd10_vals)), 2) if fwd10_vals else float("nan"),
        "med_fwd20": round(float(np.median(fwd20_vals)), 2) if fwd20_vals else float("nan"),
        "adopted":   adopted,
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def _pp(v: float) -> str:
    return f"{v:+.2f}pp" if not math.isnan(v) else "—"

def _pct(v: float) -> str:
    return f"{v:+.1f}%" if not math.isnan(v) else "—"


def write_report(
    case_results: dict,   # exit_case -> {"wf": ..., "folds": [...], "sim": sim_dict}
    topix_close: pd.Series | None,
    output_path: Path,
) -> None:
    L = []; w = L.append

    adopted_cases = [c for c in EXIT_CASES if case_results[c]["wf"].get("adopted")]

    w("# Dedicated Alpha Sleeve (4th Capital Pocket) — Study 8")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\nスリーブ: RSR ∈ [90,95), days_cross90 ≤ 5, max_pos=1, capital=equity×25%")
    w(f"\n採用条件: WF≥{ADOPT_WF_MIN}/5, ΔCAGR>+{ADOPT_DCAGR}pp, ΔCalmar>0, "
      f"ΔDD≤+{ADOPT_DDD}pp, trigger/yr≤{ADOPT_TRIG_MAX}, exposure_efficiency_pass\n")

    verdict = "✅ GO LIVE" if adopted_cases else \
              ("🔬 KEEP RESEARCH" if any(
                  case_results[c]["wf"].get("avg_dc", 0) > 0.1 for c in EXIT_CASES
              ) else "❌ REJECT")
    w(f"**採用Exit Case**: {len(adopted_cases)}件  |  最終判定: **{verdict}**\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    first_case = next(iter(case_results.values()))
    bl_cagrs = [f["base_cagr"] for f in first_case["folds"]]
    bl_avg   = float(np.mean(bl_cagrs))
    w(f"- ベースライン avg CAGR: **{bl_avg:+.1f}%** (production-faithful P0)")
    w(f"- スリーブ候補: RSR [90,95) で days_cross90 ≤ 5 の銘柄のみ")
    w(f"- Exit Case 5種 (A–E) を WF 5-fold で比較")
    w(f"- 採用: **{len(adopted_cases)}件**")
    if adopted_cases:
        best_c  = max(adopted_cases, key=lambda c: case_results[c]["wf"]["avg_dc"])
        best_wf = case_results[best_c]["wf"]
        w(f"- 最良: **{best_c}** ({EXIT_CASES[best_c]}) — "
          f"ΔCAGR={best_wf['avg_dc']:+.2f}pp, WF={best_wf['n_pass']}/5")

    # ── S2. WF Table ─────────────────────────────────────────────────
    w("\n---\n## 2. WF サマリ (5 Exit Case × 5 Fold)\n")
    w("| Exit | 説明 | WF | ΔCAGR | ΔCalmar | ΔDD | trig/yr | eff_pass | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for case, desc in EXIT_CASES.items():
        wf   = case_results[case]["wf"]
        mark = "**✅**" if wf.get("adopted") else "❌"
        ep   = f"{wf.get('eff_passes',0)}/5"
        w(f"| {case} | {desc} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_dc',0):+.2f}pp "
          f"| {wf.get('avg_dcal',0):+.3f} "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_trig',0):.1f} "
          f"| {ep} "
          f"| {mark} |")

    # ── S3. Exposure Audit ────────────────────────────────────────────
    w("\n---\n## 3. Exposure Efficiency Audit\n")
    w("> 判定: ΔCAGR/Δexp > base_CAGR/base_exp で PASS\n")
    w("| Exit | Fold | base_CAGR | base_exp | comb_CAGR | comb_exp "
      "| ΔCAGR | Δexp | base_eff | sl_eff | PASS |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in EXIT_CASES:
        for fold, fm in zip(FOLDS, case_results[case]["folds"]):
            if not fm: continue
            ok = "✅" if fm.get("exp_eff_pass") else "❌"
            w(f"| {case} | {fold['oos_start'][:4]} "
              f"| {fm['base_cagr']:+.1f}% "
              f"| {fm['base_avg_exp']:.1f}% "
              f"| {fm['comb_cagr']:+.1f}% "
              f"| {fm['comb_avg_exp']:.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_exp']:+.1f}pp "
              f"| {fm['base_eff']:.3f} "
              f"| {fm['sleeve_eff']:.3f} "
              f"| {ok} |")

    # ── S4. Alpha Sleeve Diagnostics ─────────────────────────────────
    w("\n---\n## 4. Alpha Sleeve Diagnostics\n")
    w("| Exit | sl_CAGR | sl_Calmar | hit_rate | PF | med_fwd10 | med_fwd20 | avg_hold | util% | corr | overlap |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in EXIT_CASES:
        wf = case_results[case]["wf"]
        f10 = f"{wf['med_fwd10']:+.1f}%" if not math.isnan(wf.get("med_fwd10", float("nan"))) else "—"
        f20 = f"{wf['med_fwd20']:+.1f}%" if not math.isnan(wf.get("med_fwd20", float("nan"))) else "—"
        folds = case_results[case]["folds"]
        avg_sl_cagr = float(np.mean([f.get("sl_cagr", 0) for f in folds if f]))
        avg_sl_cal  = float(np.mean([f.get("sl_calmar", 0) for f in folds if f]))
        w(f"| {case} "
          f"| {avg_sl_cagr:+.1f}% "
          f"| {avg_sl_cal:.3f} "
          f"| {wf.get('avg_hit',0):.1f}% "
          f"| {wf.get('avg_pf',0):.3f} "
          f"| {f10} | {f20} "
          f"| {wf.get('avg_hold',0):.1f}d "
          f"| {wf.get('avg_util',0):.1f}% "
          f"| {wf.get('avg_corr',0):.3f} "
          f"| {wf.get('avg_overlap',0):.0f}d |")

    # ── S5. Fold-level detail for each case ───────────────────────────
    w("\n---\n## 5. Fold 詳細\n")
    w("| Exit | Fold | OOS年 | Regime | base_CAGR | comb_CAGR | unused_CAGR "
      "| ΔCAGR | ΔDD | ΔCalmar | trig/yr | exp_eff | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in EXIT_CASES:
        for fold, fm in zip(FOLDS, case_results[case]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr) if topix_close is not None else "N/A"
            ok  = "✅" if fm.get("fold_pass") else "❌"
            ep  = "✅" if fm.get("exp_eff_pass") else "❌"
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['base_cagr']:+.1f}% "
              f"| {fm['comb_cagr']:+.1f}% "
              f"| {fm['unused_cagr']:+.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['delta_calmar']:+.3f} "
              f"| {fm['trigger_per_year']:.1f} "
              f"| {ep} | {ok} |")

    # ── S6. Failure Analysis ──────────────────────────────────────────
    w("\n---\n## 6. Failure Analysis\n")
    fail_reasons: dict[str, list] = {}
    for case in EXIT_CASES:
        fails = []
        wf    = case_results[case]["wf"]
        if wf.get("n_pass", 0) < ADOPT_WF_MIN:
            fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF_MIN}")
        if wf.get("avg_dc", 0) <= ADOPT_DCAGR:
            fails.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp ≤ +{ADOPT_DCAGR}pp")
        if wf.get("avg_dcal", 0) <= ADOPT_DCALMAR:
            fails.append(f"ΔCalmar={wf['avg_dcal']:+.3f} ≤ 0")
        if wf.get("avg_dd", 0) > ADOPT_DDD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp > +{ADOPT_DDD}pp")
        if wf.get("avg_trig", 0) > ADOPT_TRIG_MAX:
            fails.append(f"trigger/yr={wf['avg_trig']:.1f} > {ADOPT_TRIG_MAX}")
        if wf.get("eff_passes", 0) < ADOPT_WF_MIN:
            fails.append(f"exp_eff_pass={wf['eff_passes']}/5 < {ADOPT_WF_MIN}")
        fail_reasons[case] = fails

    for case, fails in fail_reasons.items():
        if not fails:
            w(f"\n**{case}** ({EXIT_CASES[case]}): ✅ 全基準クリア")
        else:
            w(f"\n**{case}** ({EXIT_CASES[case]}): REJECT — " + " / ".join(fails))

    # ── S7. Unused-cash control comparison ───────────────────────────
    w("\n---\n## 7. Unused-Cash Control 比較\n")
    w("> 同資本でスリーブを未運用（0%）とした場合の dilution 効果。")
    w("> スリーブ CAGR > unused_CAGR ならスリーブに付加価値あり。\n")
    w("| Exit | Fold | Base CAGR | +Sleeve CAGR | +Unused CAGR | Sleeve優位? |")
    w("|---|---|---|---|---|---|")
    for case in EXIT_CASES:
        for fold, fm in zip(FOLDS, case_results[case]["folds"]):
            if not fm: continue
            advantage = "✅" if fm["comb_cagr"] > fm["unused_cagr"] else "❌"
            w(f"| {case} | {fold['oos_start'][:4]} "
              f"| {fm['base_cagr']:+.1f}% "
              f"| {fm['comb_cagr']:+.1f}% "
              f"| {fm['unused_cagr']:+.1f}% "
              f"| {advantage} |")

    # ── S8. Adopted case detail ───────────────────────────────────────
    if adopted_cases:
        w("\n---\n## 8. 採用 Case 詳細\n")
        for case in adopted_cases:
            wf = case_results[case]["wf"]
            w(f"\n### ✅ Case {case}: {EXIT_CASES[case]}\n")
            w(f"- ΔCAGR: **{wf['avg_dc']:+.2f}pp**  WF: **{wf['n_pass']}/5**")
            w(f"- ΔCalmar: {wf['avg_dcal']:+.3f}  ΔDD: {wf['avg_dd']:+.2f}pp")
            w(f"- trigger/yr: {wf['avg_trig']:.1f}  avg_hold: {wf['avg_hold']:.1f}d  "
              f"util: {wf['avg_util']:.1f}%")
            w(f"- corr_to_base: {wf['avg_corr']:.3f}  overlap_days: {wf['avg_overlap']:.0f}\n")
            w("実装前必須事項 (ASK_FIRST):")
            w("- signal_bridge.py にスリーブ注文ロジック追加")
            w("- スリーブ用 portfolio_state (sleeve_state.json) の設計")
            w("- 独立 CB 判断ロジックの定義")
            w("- 追加 live shadow 検証 ≥30日")

    # ── S9. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 9. Final Recommendation\n")
    w(f"## `{verdict}`\n")

    if verdict.startswith("✅"):
        best_c = max(adopted_cases, key=lambda c: case_results[c]["wf"]["avg_dc"])
        bwf    = case_results[best_c]["wf"]
        w(f"**採用推奨: Exit Case {best_c}** — {EXIT_CASES[best_c]}\n")
        w(f"- ΔCAGR avg: **{bwf['avg_dc']:+.2f}pp**")
        w(f"- ΔCalmar: {bwf['avg_dcal']:+.3f}  ΔDD: {bwf['avg_dd']:+.2f}pp")
        w(f"- trigger/yr: {bwf['avg_trig']:.1f}  hit_rate: {bwf['avg_hit']:.1f}%\n")
        w("実装優先度 (ASK_FIRST必須):")
        w("1. signal_bridge.py スリーブ注文追加")
        w("2. 独立 sleeve_state.json 管理")
        w("3. live dry-run 30日 → 本番移行")
    elif verdict.startswith("🔬"):
        best_c  = max(EXIT_CASES, key=lambda c: case_results[c]["wf"].get("avg_dc", 0))
        bwf     = case_results[best_c]["wf"]
        w(f"仮説 A (RSR90-94高期待値) / 仮説 B (ポートフォリオ効率改善) は部分的に支持。\n")
        w(f"- 最良 Case {best_c}: ΔCAGR={bwf['avg_dc']:+.2f}pp, WF={bwf['n_pass']}/5")
        w(f"- REJECT 主因: " + " / ".join(fail_reasons.get(best_c, ["不明"])))
        w(f"\n継続研究候補:")
        w(f"- exposure_efficiency が {bwf['eff_passes']}/5 のみ PASS → d90 ≤ 3 に絞り込み")
        w(f"- trigger/yr {bwf['avg_trig']:.1f} → max_hold 短縮または入場基準追加で削減")
        w(f"- Fold別 regime 解析: Bull/Bear で効果分離確認")
    else:
        w("**REJECT: 全 Exit Case が採用基準未達。**\n")
        best_c = max(EXIT_CASES, key=lambda c: case_results[c]["wf"].get("avg_dc", 0))
        bwf    = case_results[best_c]["wf"]
        w(f"- 最良 Case {best_c}: ΔCAGR={bwf['avg_dc']:+.2f}pp, WF={bwf['n_pass']}/5")
        w(f"\n根本的制約:")
        w(f"- RSR [90,95) + d90≤5 の OOS 信号は希少 → portfolio-level CAGR への寄与が小さい")
        w(f"- スリーブ自体は短期 alpha を持つが、ポートフォリオ規模に対して金額効果が軽微")
        w(f"- exposure_efficiency が不安定 → 単なる露出増と区別困難")
        w(f"\n次ステップ候補:")
        w(f"- G: max_hold=60 の時間停止を撤廃した場合の baseline 改善余地")
        w(f"- H: スリーブ capital 比率変更 (25% → 40%) での再実験")
        w(f"- I: d90 ≤ 2 (さらに新鮮なクロスのみ) での信号品質向上確認")

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
    print("  Study 8: Dedicated Alpha Sleeve WF")
    print("  RSR [90,95) × d90≤5 × equity×25% × 5 exit cases × 5-fold WF")
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

    cross90_mat            = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)

    # Phase 1 signal count
    n_sleeve_sigs = sum(
        1 for i in range(n_dates)
        if str(common_dates[i].date()) >= "2021-01-01"
        for si in range(n_syms)
        if (sig_ready[si]
            and SLEEVE_RSR_LO <= float(rsr_mat[i, si]) < SLEEVE_RSR_HI
            and int(cross90_mat[i, si]) <= SLEEVE_D90_MAX
            and (sym_active_mat is None or float(sym_active_mat[i, si]) >= 0.5))
    )
    print(f"  OOS sleeve candidate signals: {n_sleeve_sigs}")

    # ── WF Simulation ────────────────────────────────────────────────
    print(f"\n[3/4] WF Simulation ({len(EXIT_CASES)} exit cases × 5 folds)...\n")

    case_results: dict[str, dict] = {}
    capital = float(cfg.portfolio.capital)
    sleeve_capital = capital * SLEEVE_CAP_FR

    for exit_case, desc in EXIT_CASES.items():
        print(f"  [{exit_case}] {desc}")
        sim = run_exit_case(
            open_mat, close_mat,
            sig_mat, sig_ready, rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, trade_syms, cfg,
            common_dates, exit_case,
        )

        fold_metrics: list[dict] = []
        for fold in FOLDS:
            fm = compute_fold_metrics(sim, close_mat, common_dates, fold,
                                       capital, sleeve_capital)
            fold_metrics.append(fm)
            if fm:
                ok = "✅" if fm.get("fold_pass") else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"base={fm['base_cagr']:+.1f}%  "
                      f"comb={fm['comb_cagr']:+.1f}%  "
                      f"Δ={fm['delta_cagr']:+.2f}pp  "
                      f"trig={fm['trigger_per_year']:.1f}/yr  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")

        wf = evaluate_exit_case(fold_metrics)
        case_results[exit_case] = {"folds": fold_metrics, "wf": wf, "sim": sim}

        ok_str = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔCalmar={wf.get('avg_dcal',0):+.3f}  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"trig/yr={wf.get('avg_trig',0):.1f}  "
              f"eff_pass={wf.get('eff_passes',0)}/5  {ok_str}\n")

    # Summary
    adopted = [c for c in EXIT_CASES if case_results[c]["wf"].get("adopted")]
    print(f"{'='*68}")
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for c in EXIT_CASES:
        wf = case_results[c]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        print(f"  {ok} Case {c}: WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"eff={wf.get('eff_passes',0)}/5")
    print(f"{'='*68}\n")

    # ── Report ───────────────────────────────────────────────────────
    print("[4/4] レポート生成...")
    write_report(case_results, topix_close,
                 REPORTS_DIR / "dedicated_alpha_sleeve.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
