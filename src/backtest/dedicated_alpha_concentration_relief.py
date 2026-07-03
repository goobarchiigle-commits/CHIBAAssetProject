"""
src/backtest/dedicated_alpha_concentration_relief.py
Study 8 Concentration Relief — 6-Case Position Structure WF

背景:
  Study 8/8B/8C/8D で RSR90-94 alpha (sl_CAGR≈40%) は実在と確認。
  ΔDD > +1.5pp の根本因 = Bull期 single-stock concentration。
  → sleeve max_pos 変更で concentration を分散し ΔDD を改善できるか検証。

固定:
  ENTRY  RSR ∈ [92,95), days_cross90 ≤ 5
  EXIT   RSR < 90 (即座)
  CAP    equity × 20%
  選択順: ①高RSR ②低d90 ③高state_rank
  制約: 既存3baseポジと競合禁止 / 空き枠時のみ新規追加 / 既存保有の途中減額禁止

Cases:
  A  max_pos=1  [100%]               (concentration baseline)
  B  max_pos=2  [70/30]              (dominant leader)
  C  max_pos=2  [60/40]              (moderate split)
  D  max_pos=2  [50/50]              (equal weight)
  E  max_pos=2  [70/30] no-refill    (pair-synchronized: wait for all clear)
  F  max_pos=3  [50/30/20]           (3-slot diversification)

採用条件:
  WF ≥ 4/5, ΔCAGR ≥ +0.3pp, ΔDD ≤ +1.5pp, sl_CAGR ≥ 35%, alpha_retention ≥ 90%

出力:
  reports/dedicated_alpha_concentration_relief.md

Run: cd C:/ai-trading && python src/backtest/dedicated_alpha_concentration_relief.py
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

CASE_SPECS: dict[str, dict] = {
    "A": {"max_pos": 1, "weights": [1.00],              "no_refill": False,
          "desc": "max_pos=1  100%          (concentration baseline)"},
    "B": {"max_pos": 2, "weights": [0.70, 0.30],        "no_refill": False,
          "desc": "max_pos=2  70/30         (dominant leader)"},
    "C": {"max_pos": 2, "weights": [0.60, 0.40],        "no_refill": False,
          "desc": "max_pos=2  60/40         (moderate split)"},
    "D": {"max_pos": 2, "weights": [0.50, 0.50],        "no_refill": False,
          "desc": "max_pos=2  equal         (equal weight)"},
    "E": {"max_pos": 2, "weights": [0.70, 0.30],        "no_refill": True,
          "desc": "max_pos=2  70/30 no-refill (pair-synchronized)"},
    "F": {"max_pos": 3, "weights": [0.50, 0.30, 0.20], "no_refill": False,
          "desc": "max_pos=3  50/30/20      (3-slot diversification)"},
}

SLEEVE_RSR_LO  = 92.0
SLEEVE_RSR_HI  = 95.0
SLEEVE_D90_MAX = 5
SLEEVE_CAP_FR  = 0.20
SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08

ADOPT_WF_MIN      = 4
ADOPT_DCAGR       = 0.3
ADOPT_DDD         = 1.5
ADOPT_SL_CAGR_MIN = 35.0
ADOPT_ALPHA_RET   = 90.0


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
    "STALL": 1,    "DOWN": 0,      "EARLY_ROLL": -1, "UNKNOWN": 0,
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


def hhi_from_weights(weights: list[float]) -> float:
    return sum(w**2 for w in weights)


# ─────────────────────────────────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────────────────────────────────

def run_case(
    open_mat, close_mat,
    sig_mat, sig_ready, rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    cross90_mat, slope5_mat, slope20_mat,
    active_syms, sym_to_i, trade_syms, cfg,
    common_dates,
    case_spec: dict,
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

    sleeve_max = case_spec["max_pos"]
    sl_weights = case_spec["weights"]
    no_refill  = case_spec["no_refill"]

    # ── Base state ────────────────────────────────────────────────────
    base_cash   = float(capital)
    base_pos: dict[str, Position] = {}
    base_peak   = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    base_trades: list[dict] = []

    # ── Sleeve state ──────────────────────────────────────────────────
    sl_slots: list[Position | None] = [None] * sleeve_max
    sl_cash  = float(capital * SLEEVE_CAP_FR)
    sl_trades: list[dict] = []

    # ── Daily arrays ──────────────────────────────────────────────────
    n_dates = len(common_dates)
    base_eq  = np.zeros(n_dates, dtype=np.float64)
    sl_eq    = np.zeros(n_dates, dtype=np.float64)
    base_inv = np.zeros(n_dates, dtype=np.float64)
    sl_inv   = np.zeros(n_dates, dtype=np.float64)

    # Slot daily returns for intra-sleeve correlation
    slot_ret_arr = np.full((n_dates, sleeve_max), np.nan, dtype=np.float64)
    # Daily HHI (actual occupied weights)
    daily_hhi_arr = np.full(n_dates, np.nan, dtype=np.float64)

    sl_held_set   = [set() for _ in range(n_dates)]
    base_held_set = [set() for _ in range(n_dates)]

    for i, date in enumerate(common_dates):
        date_str = str(date.date())

        b_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                    for s, p in base_pos.items())
        s_inv = sum(
            sl_slots[k].qty * float(close_mat[i, sym_to_i[sl_slots[k].symbol]])
            for k in range(sleeve_max) if sl_slots[k] is not None
        )

        b_eq = base_cash + b_inv
        s_eq = sl_cash + s_inv
        base_eq[i]  = b_eq
        sl_eq[i]    = s_eq
        base_inv[i] = b_inv
        sl_inv[i]   = s_inv

        # Track slot daily returns
        if i > 0:
            for k in range(sleeve_max):
                if sl_slots[k] is not None:
                    si_k = sym_to_i[sl_slots[k].symbol]
                    c0 = float(close_mat[i-1, si_k])
                    c1 = float(close_mat[i,   si_k])
                    if c0 > 0 and c1 > 0:
                        slot_ret_arr[i, k] = c1 / c0 - 1.0

        # Daily HHI
        occ = [(k, sl_weights[k]) for k in range(sleeve_max) if sl_slots[k] is not None]
        if occ:
            tw = sum(w for _, w in occ)
            daily_hhi_arr[i] = sum((w / tw) ** 2 for _, w in occ)

        sl_held_set[i]   = {sl_slots[k].symbol for k in range(sleeve_max) if sl_slots[k] is not None}
        base_held_set[i] = set(base_pos.keys())

        # CB check (base only)
        if b_eq > base_peak: base_peak = b_eq
        dd = (b_eq - base_peak) / base_peak
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
        for k in range(sleeve_max):
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
                    "slot": k, "weight": sl_weights[k],
                    "fwd10": np.nan, "fwd20": np.nan,
                })
                sl_slots[k] = None

        # ── SLEEVE: entries ───────────────────────────────────────────
        if not mkt_shock:
            n_occupied = sum(1 for k in range(sleeve_max) if sl_slots[k] is not None)
            n_open     = sleeve_max - n_occupied

            # no_refill: when partially filled, wait for full clearance
            if no_refill and 0 < n_occupied < sleeve_max:
                pass
            elif n_open > 0:
                sl_syms = {sl_slots[k].symbol for k in range(sleeve_max) if sl_slots[k] is not None}
                cands: list[tuple] = []
                for sym in active_syms:
                    if sym in base_pos: continue   # conflict prohibition
                    if sym in sl_syms:  continue   # already in sleeve
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
                    cand_idx = 0
                    for k in range(sleeve_max):
                        if sl_slots[k] is not None: continue
                        if cand_idx >= len(cands): break
                        rsr_v, d90, sr, sym = cands[cand_idx]; cand_idx += 1
                        si_sym  = sym_to_i[sym]
                        buy_px  = float(open_mat[next_i, si_sym])
                        if buy_px <= 0: continue
                        target  = b_eq * SLEEVE_CAP_FR * sl_weights[k]
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
                            "slot": k, "weight": sl_weights[k],
                            "fwd10": np.nan, "fwd20": np.nan,
                        })

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

    return {
        "base_eq":       base_eq,
        "sl_eq":         sl_eq,
        "base_inv":      base_inv,
        "sl_inv":        sl_inv,
        "base_trades":   base_trades,
        "sl_trades":     sl_trades,
        "sl_held_set":   sl_held_set,
        "base_held_set": base_held_set,
        "slot_ret_arr":  slot_ret_arr,
        "daily_hhi_arr": daily_hhi_arr,
    }


# ─────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ─────────────────────────────────────────────────────────────────────

def _intra_corr(slot_ret_arr: np.ndarray, si: int, ei: int, sleeve_max: int) -> float:
    if sleeve_max < 2: return float("nan")
    corrs = []
    for k1 in range(sleeve_max):
        for k2 in range(k1+1, sleeve_max):
            r1 = slot_ret_arr[si:ei, k1]
            r2 = slot_ret_arr[si:ei, k2]
            mask = ~(np.isnan(r1) | np.isnan(r2))
            if mask.sum() < 10: continue
            c = float(np.corrcoef(r1[mask], r2[mask])[0, 1])
            if not math.isnan(c): corrs.append(c)
    return float(np.mean(corrs)) if corrs else float("nan")


def compute_fold_metrics(
    sim: dict,
    close_mat: np.ndarray,
    common_dates,
    fold: dict,
    capital: float,
    sleeve_max: int,
    sl_weights: list[float],
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

    base_t = [t for t in sim["base_trades"] if oos_s <= t.get("date","") <= oos_e]
    sl_t   = [t for t in sim["sl_trades"]   if oos_s <= t.get("date","") <= oos_e]
    sl_sell = [t for t in sl_t if t["side"] == "SELL"]
    comb_t  = base_t + sl_t

    # ── Core metrics ──────────────────────────────────────────────────
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
    delta_dd     = -(comb_dd - base_dd)    # positive = ΔDD worse
    delta_calmar = comb_m.get("calmar", 0) - base_m.get("calmar", 0)

    # ── Sleeve standalone metrics ─────────────────────────────────────
    sl_cagr = sl_calmar = 0.0
    if len(sl_oos) >= 5 and sl_oos[0] > 0:
        sl_exp = [si_ / max(1, se) for si_, se in zip(s_inv_o, sl_oos)]
        sl_m   = calc_metrics(sl_oos, sl_t, sl_exp, sl_oos[0], oos_dates)
        sl_cagr    = sl_m.get("cagr", 0)
        sl_calmar  = sl_m.get("calmar", 0)

    # ── Activity metrics ─────────────────────────────────────────────
    n_triggers       = len(sl_sell)
    trigger_per_year = n_triggers / max(0.01, n_years)

    hold_days_list = [t.get("exit_idx", 0) - t.get("entry_idx", 0) for t in sl_sell
                      if t.get("exit_idx", 0) > t.get("entry_idx", 0)]
    avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0.0

    invested_days = int(sum(1 for v in s_inv_o if v > 0))
    capital_util  = invested_days / max(1, n_days) * 100.0

    # ── Quality metrics ───────────────────────────────────────────────
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

    # ── Position concentration ────────────────────────────────────────
    avg_hhi = float(np.nanmean(sim["daily_hhi_arr"][si:ei]))

    # ── Intra-sleeve correlation ──────────────────────────────────────
    avg_corr = _intra_corr(sim["slot_ret_arr"], si, ei, sleeve_max)

    # ── top1_pnl_share / tail_capture ────────────────────────────────
    sorted_pnl = sorted([t["pnl"] for t in sl_sell if (t.get("pnl") or 0) > 0], reverse=True)
    total_pos_pnl = sum(sorted_pnl) if sorted_pnl else 0.0
    top1_pnl_share  = (sorted_pnl[0] / total_pos_pnl * 100) if sorted_pnl and total_pos_pnl > 0 else float("nan")
    tail_capture    = (sum(sorted_pnl[:10]) / total_pos_pnl * 100) if len(sorted_pnl) >= 1 and total_pos_pnl > 0 else float("nan")

    # ── Base-sleeve correlation & overlap ─────────────────────────────
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

    # ── Fold pass ─────────────────────────────────────────────────────
    fold_pass = (delta_cagr > ADOPT_DCAGR and delta_dd <= ADOPT_DDD)

    return {
        "base_cagr":    base_m.get("cagr", 0),
        "base_max_dd":  base_dd,
        "base_calmar":  base_m.get("calmar", 0),
        "base_ntrades": base_m.get("n_trades", 0),
        "comb_cagr":    comb_m.get("cagr", 0),
        "comb_max_dd":  comb_dd,
        "comb_calmar":  comb_m.get("calmar", 0),
        "unused_cagr":  unused_m.get("cagr", 0),
        "delta_cagr":   round(delta_cagr,   2),
        "delta_dd":     round(delta_dd,      2),
        "delta_calmar": round(delta_calmar,  3),
        "sl_cagr":      round(sl_cagr,       2),
        "sl_calmar":    round(sl_calmar,      3),
        "n_triggers":   n_triggers,
        "trigger_per_year": round(trigger_per_year, 1),
        "avg_hold":     round(avg_hold,       1),
        "capital_util": round(capital_util,   1),
        "hit_rate":     round(hit_rate,       1),
        "profit_factor":round(pf,             3),
        "med_fwd10":    med_fwd10,
        "med_fwd20":    med_fwd20,
        "avg_hhi":      round(avg_hhi, 3) if not math.isnan(avg_hhi) else float("nan"),
        "avg_corr":     round(avg_corr, 3) if not math.isnan(avg_corr) else float("nan"),
        "top1_pnl_share": round(top1_pnl_share,  1) if not math.isnan(top1_pnl_share) else float("nan"),
        "tail_capture":   round(tail_capture,    1) if not math.isnan(tail_capture) else float("nan"),
        "corr_to_base": round(corr_to_base, 3),
        "overlap_days": overlap_days,
        "fold_pass":    fold_pass,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATOR
# ─────────────────────────────────────────────────────────────────────

def evaluate_wf(fold_metrics: list[dict], sl_cagr_a: float | None = None) -> dict:
    if not fold_metrics: return {}
    valid = [f for f in fold_metrics if f]
    if not valid: return {}

    n_pass     = sum(1 for f in valid if f.get("fold_pass"))
    avg_dc     = float(np.mean([f["delta_cagr"]    for f in valid]))
    avg_dd     = float(np.mean([f["delta_dd"]      for f in valid]))
    avg_dcal   = float(np.mean([f["delta_calmar"]  for f in valid]))
    avg_trig   = float(np.mean([f["trigger_per_year"] for f in valid]))
    avg_hold   = float(np.mean([f["avg_hold"]      for f in valid]))
    avg_util   = float(np.mean([f["capital_util"]  for f in valid]))
    avg_slcagr = float(np.mean([f["sl_cagr"]       for f in valid]))
    avg_corr_b = float(np.mean([f["corr_to_base"]  for f in valid]))
    avg_ovlp   = float(np.mean([f["overlap_days"]  for f in valid]))
    avg_hit    = float(np.mean([f["hit_rate"]       for f in valid]))
    avg_pf     = float(np.mean([f["profit_factor"]  for f in valid]))

    hhi_vals  = [f["avg_hhi"]  for f in valid if not math.isnan(f.get("avg_hhi", float("nan")))]
    corr_vals = [f["avg_corr"] for f in valid if not math.isnan(f.get("avg_corr", float("nan")))]
    top1_vals = [f["top1_pnl_share"] for f in valid if not math.isnan(f.get("top1_pnl_share", float("nan")))]
    tail_vals = [f["tail_capture"]   for f in valid if not math.isnan(f.get("tail_capture",   float("nan")))]

    fwd10v = [f["med_fwd10"] for f in valid if not math.isnan(f.get("med_fwd10", float("nan")))]
    fwd20v = [f["med_fwd20"] for f in valid if not math.isnan(f.get("med_fwd20", float("nan")))]

    alpha_retention = (avg_slcagr / sl_cagr_a * 100) if sl_cagr_a and sl_cagr_a > 0 else float("nan")

    adopted = (
        n_pass >= ADOPT_WF_MIN
        and avg_dc > ADOPT_DCAGR
        and avg_dd <= ADOPT_DDD
        and avg_slcagr >= ADOPT_SL_CAGR_MIN
        and (math.isnan(alpha_retention) or alpha_retention >= ADOPT_ALPHA_RET)
    )

    return {
        "n_pass":          n_pass,
        "avg_dc":          round(avg_dc,    2),
        "avg_dd":          round(avg_dd,    2),
        "avg_dcal":        round(avg_dcal,  3),
        "avg_trig":        round(avg_trig,  1),
        "avg_hold":        round(avg_hold,  1),
        "avg_util":        round(avg_util,  1),
        "avg_slcagr":      round(avg_slcagr, 2),
        "alpha_retention": round(alpha_retention, 1) if not math.isnan(alpha_retention) else float("nan"),
        "avg_corr_base":   round(avg_corr_b, 3),
        "avg_overlap":     round(avg_ovlp,   1),
        "avg_hit":         round(avg_hit,    1),
        "avg_pf":          round(avg_pf,     3),
        "avg_hhi":         round(float(np.mean(hhi_vals)),  3) if hhi_vals  else float("nan"),
        "avg_intra_corr":  round(float(np.mean(corr_vals)), 3) if corr_vals else float("nan"),
        "avg_top1_share":  round(float(np.mean(top1_vals)), 1) if top1_vals else float("nan"),
        "avg_tail_cap":    round(float(np.mean(tail_vals)),  1) if tail_vals else float("nan"),
        "med_fwd10":       round(float(np.median(fwd10v)), 2) if fwd10v else float("nan"),
        "med_fwd20":       round(float(np.median(fwd20v)), 2) if fwd20v else float("nan"),
        "adopted":         adopted,
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

def _pct(v) -> str:
    if isinstance(v, float) and math.isnan(v): return "—"
    return f"{v:+.1f}%"


def write_report(
    case_results: dict,
    topix_close: pd.Series | None,
    output_path: Path,
    marginal_dd: dict,
) -> None:
    L = []; w = L.append

    adopted_cases = [c for c in CASE_SPECS if case_results[c]["wf"].get("adopted")]
    first_c       = next(iter(case_results.values()))
    bl_cagrs      = [f["base_cagr"] for f in first_c["folds"] if f]
    bl_avg        = float(np.mean(bl_cagrs)) if bl_cagrs else 0.0

    verdict = ("✅ GO LIVE (shadow 30d 必須)" if adopted_cases
               else "🔬 KEEP RESEARCH" if any(case_results[c]["wf"].get("avg_dc", 0) > 0.0
                                               for c in CASE_SPECS)
               else "❌ REJECT")

    w("# Dedicated Alpha Concentration Relief — Study 8 (Concentration WF)")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n固定: ENTRY=RSR∈[92,95) d90≤5 / EXIT=RSR<90 / CAP=equity×20%")
    w(f"\n採用条件: WF≥{ADOPT_WF_MIN}/5, ΔCAGR>+{ADOPT_DCAGR}pp, "
      f"ΔDD≤+{ADOPT_DDD}pp, sl_CAGR≥{ADOPT_SL_CAGR_MIN}%, alpha_retention≥{ADOPT_ALPHA_RET}%\n")
    w(f"**採用Case**: {len(adopted_cases)}件  |  最終判定: **{verdict}**\n")

    # ── 1. Executive Summary ──────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    w(f"- ベースライン avg CAGR: **{bl_avg:+.1f}%**")
    w(f"- スリーブ: RSR[92,95) × d90≤5 / EXIT=RSR<90(即座) / CAP=20%")
    w(f"- 仮説: single-stock concentration (HHI=1.0) → max_pos=2/3 で ΔDD を削減")
    if adopted_cases:
        best_c = max(adopted_cases, key=lambda c: case_results[c]["wf"]["avg_dc"])
        bwf    = case_results[best_c]["wf"]
        w(f"- 最良採用: **{best_c}** ({CASE_SPECS[best_c]['desc']}) — "
          f"ΔCAGR={bwf['avg_dc']:+.2f}pp, ΔDD={bwf['avg_dd']:+.2f}pp, "
          f"WF={bwf['n_pass']}/5")

    # ── 2. WF Summary ────────────────────────────────────────────────
    w("\n---\n## 2. WF サマリ (6 Case × 5 Fold)\n")
    w("| Case | 説明 | WF | ΔCAGR | ΔDD | ΔCalmar | sl_CAGR | α_ret | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for case, spec in CASE_SPECS.items():
        wf   = case_results[case]["wf"]
        mark = "**✅**" if wf.get("adopted") else "❌"
        ar   = (_f(wf.get("alpha_retention", float("nan")), ".1f") + "%"
                if not math.isnan(wf.get("alpha_retention", float("nan"))) else "—")
        w(f"| {case} | {spec['desc']} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_dc',0):+.2f}pp "
          f"| {wf.get('avg_dd',0):+.2f}pp "
          f"| {wf.get('avg_dcal',0):+.3f} "
          f"| {wf.get('avg_slcagr',0):+.1f}% "
          f"| {ar} "
          f"| {mark} |")

    # ── 3. Concentration Metrics ──────────────────────────────────────
    w("\n---\n## 3. Concentration Analysis\n")
    w("| Case | HHI(理論) | HHI(実測avg) | avg_intra_corr | top1_pnl_share | tail_capture | capital_util |")
    w("|---|---|---|---|---|---|---|")
    for case, spec in CASE_SPECS.items():
        wf      = case_results[case]["wf"]
        hhi_th  = round(hhi_from_weights(spec["weights"]), 3)
        hhi_act = _f(wf.get("avg_hhi", float("nan")), ".3f")
        corr_i  = _f(wf.get("avg_intra_corr", float("nan")), ".3f")
        top1    = (_f(wf.get("avg_top1_share", float("nan")), ".1f") + "%"
                   if not math.isnan(wf.get("avg_top1_share", float("nan"))) else "—")
        tail    = (_f(wf.get("avg_tail_cap", float("nan")), ".1f") + "%"
                   if not math.isnan(wf.get("avg_tail_cap", float("nan"))) else "—")
        util    = f"{wf.get('avg_util', 0):.1f}%"
        w(f"| {case} | {hhi_th} | {hhi_act} | {corr_i} | {top1} | {tail} | {util} |")

    # ── 4. Diversification Efficiency ────────────────────────────────
    w("\n---\n## 4. Diversification Efficiency Audit\n")
    w("> diversification_efficiency = DD削減量 / alpha損失量 (Case A比)\n")
    wf_a = case_results["A"]["wf"]
    dd_a = wf_a.get("avg_dd", 0)
    sc_a = wf_a.get("avg_slcagr", 0)
    w("| Case | ΔDD(avg) | ΔDD_A−ΔDD | sl_CAGR | sl_CAGR損失 | div_efficiency |")
    w("|---|---|---|---|---|---|")
    for case in CASE_SPECS:
        wf = case_results[case]["wf"]
        dd = wf.get("avg_dd", 0)
        sc = wf.get("avg_slcagr", 0)
        dd_saved  = dd_a - dd
        sl_loss   = sc_a - sc
        if sl_loss > 0.01:
            de = round(dd_saved / sl_loss, 3)
            de_str = f"{de:.3f}"
        elif sl_loss <= 0 and dd_saved > 0:
            de_str = "∞ (no alpha loss)"
        else:
            de_str = "—"
        w(f"| {case} | {dd:+.2f}pp | {dd_saved:+.2f}pp "
          f"| {sc:+.1f}% | {sl_loss:+.1f}% | {de_str} |")

    # ── 5. Fold Detail ────────────────────────────────────────────────
    w("\n---\n## 5. Fold 詳細\n")
    w("| Case | Fold | OOS年 | Regime | base_CAGR | comb_CAGR | unused_CAGR "
      "| ΔCAGR | ΔDD | sl_CAGR | trig/yr | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in CASE_SPECS:
        for fold, fm in zip(FOLDS, case_results[case]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr) if topix_close is not None else "N/A"
            ok  = "✅" if fm.get("fold_pass") else "❌"
            w(f"| {case} | Fold{fold['id']} | {yr} | {reg} "
              f"| {fm['base_cagr']:+.1f}% "
              f"| {fm['comb_cagr']:+.1f}% "
              f"| {fm['unused_cagr']:+.1f}% "
              f"| {fm['delta_cagr']:+.2f}pp "
              f"| {fm['delta_dd']:+.2f}pp "
              f"| {fm['sl_cagr']:+.1f}% "
              f"| {fm['trigger_per_year']:.1f} "
              f"| {ok} |")

    # ── 6. Failure Analysis ───────────────────────────────────────────
    w("\n---\n## 6. Failure / Accept Analysis\n")
    for case, spec in CASE_SPECS.items():
        wf    = case_results[case]["wf"]
        fails = []
        if wf.get("n_pass", 0) < ADOPT_WF_MIN:
            fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF_MIN}")
        if wf.get("avg_dc", 0) <= ADOPT_DCAGR:
            fails.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp ≤ +{ADOPT_DCAGR}pp")
        if wf.get("avg_dd", 0) > ADOPT_DDD:
            fails.append(f"ΔDD={wf['avg_dd']:+.2f}pp > +{ADOPT_DDD}pp")
        if wf.get("avg_slcagr", 0) < ADOPT_SL_CAGR_MIN:
            fails.append(f"sl_CAGR={wf['avg_slcagr']:.1f}% < {ADOPT_SL_CAGR_MIN}%")
        ar = wf.get("alpha_retention", float("nan"))
        if not math.isnan(ar) and ar < ADOPT_ALPHA_RET:
            fails.append(f"alpha_ret={ar:.1f}% < {ADOPT_ALPHA_RET}%")
        if not fails:
            w(f"\n**{case}** ✅ 全基準クリア")
        else:
            w(f"\n**{case}** REJECT — " + " / ".join(fails))

    # ── 7. Marginal DD Saved Per Slot ─────────────────────────────────
    w("\n---\n## 7. Marginal DD Saved Per Added Slot\n")
    w("> ΔDD = positive → DD worsening vs base (smaller = better)\n")
    w("| 比較 | ΔDD(from) | ΔDD(to) | DD削減量 | 追加スロット数 | DD削減/slot |")
    w("|---|---|---|---|---|---|")
    for comp_key, row in marginal_dd.items():
        fr_dd   = row["from_dd"]
        to_dd   = row["to_dd"]
        dd_save = fr_dd - to_dd
        n_slots = row["n_slots"]
        per_slot = dd_save / n_slots if n_slots > 0 else float("nan")
        per_slot_str = f"{per_slot:+.3f}pp/slot" if not math.isnan(per_slot) else "—"
        w(f"| {comp_key} "
          f"| {fr_dd:+.2f}pp "
          f"| {to_dd:+.2f}pp "
          f"| {dd_save:+.2f}pp "
          f"| {n_slots} "
          f"| **{per_slot_str}** |")

    w(f"\n**ΔDD/slot 解釈**: 値が負(マイナス) = スロット追加でΔDD悪化。正 = 改善。")

    # ── 8. Final Recommendation ───────────────────────────────────────
    w("\n---\n## 8. Final Recommendation\n")
    w(f"## `{verdict}`\n")

    if adopted_cases:
        best_c = max(adopted_cases, key=lambda c: case_results[c]["wf"]["avg_dc"])
        bwf    = case_results[best_c]["wf"]
        w(f"**採用推奨: Case {best_c}** — {CASE_SPECS[best_c]['desc']}\n")
        w(f"- ΔCAGR avg: **{bwf['avg_dc']:+.2f}pp**  WF: **{bwf['n_pass']}/5**")
        w(f"- ΔDD: {bwf['avg_dd']:+.2f}pp  ΔCalmar: {bwf['avg_dcal']:+.3f}")
        w(f"- sl_CAGR: {bwf['avg_slcagr']:+.1f}%  alpha_retention: "
          f"{bwf.get('alpha_retention', float('nan')):.1f}%")
        w(f"- capital_util: {bwf['avg_util']:.1f}%  avg_intra_corr: "
          f"{_f(bwf.get('avg_intra_corr', float('nan')), '.3f')}\n")
        w("実装必須 (ASK_FIRST):")
        w("- signal_bridge.py にスリーブ注文ロジック追加 (multi-slot)")
        w("- sleeve_state.json でスロット状態管理")
        w("- 独立 CB 判断ロジックの定義")
        w("- live shadow 検証 ≥ 30日")
    else:
        best_c  = max(CASE_SPECS, key=lambda c: case_results[c]["wf"].get("avg_dc", 0))
        bwf     = case_results[best_c]["wf"]
        w(f"最良 Case {best_c}: ΔCAGR={bwf['avg_dc']:+.2f}pp, WF={bwf['n_pass']}/5, "
          f"ΔDD={bwf['avg_dd']:+.2f}pp, sl_CAGR={bwf['avg_slcagr']:+.1f}%\n")
        fail_map = {}
        for case in CASE_SPECS:
            wf = case_results[case]["wf"]
            fls = []
            if wf.get("n_pass", 0) < ADOPT_WF_MIN:       fls.append(f"WF={wf['n_pass']}/5")
            if wf.get("avg_dc", 0) <= ADOPT_DCAGR:        fls.append(f"ΔCAGR={wf['avg_dc']:+.2f}pp")
            if wf.get("avg_dd", 0) > ADOPT_DDD:           fls.append(f"ΔDD={wf['avg_dd']:+.2f}pp")
            if wf.get("avg_slcagr", 0) < ADOPT_SL_CAGR_MIN: fls.append(f"sl_CAGR={wf['avg_slcagr']:.1f}%")
            ar = wf.get("alpha_retention", float("nan"))
            if not math.isnan(ar) and ar < ADOPT_ALPHA_RET: fls.append(f"α_ret={ar:.1f}%")
            fail_map[case] = fls
        w("**主要 REJECT 理由:**")
        for case, fls in fail_map.items():
            if fls: w(f"- {case}: {' / '.join(fls)}")
        w(f"\n**次ステップ候補:**")
        w(f"- max_pos制約をさらに変更: Case F (3-slot) が効果あれば max_pos=4 も検討")
        w(f"- SLEEVE_CAP_FR=15% で ΔDD の比例削減確認")
        w(f"- alpha_retention 問題: リーダー集中度を高め (80/20) alpha 保存性を改善")

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
    print("  Study 8 Concentration Relief WF")
    print("  RSR[92,95) × d90≤5 × EXIT RSR<90 × CAP 20%")
    print("  Cases A-F (1-3 slots, variable weights)")
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

    # Sleeve signal count (OOS period)
    n_sleeve_sigs = sum(
        1 for i in range(n_dates)
        if str(common_dates[i].date()) >= "2021-01-01"
        for j in range(n_syms)
        if (sig_ready[j]
            and SLEEVE_RSR_LO <= float(rsr_mat[i, j]) < SLEEVE_RSR_HI
            and int(cross90_mat[i, j]) <= SLEEVE_D90_MAX
            and (sym_active_mat is None or float(sym_active_mat[i, j]) >= 0.5))
    )
    print(f"  OOS sleeve candidate signals: {n_sleeve_sigs}")

    # ── WF Simulation ────────────────────────────────────────────────
    print(f"\n[3/4] WF Simulation ({len(CASE_SPECS)} cases × 5 folds)...\n")

    capital = float(cfg.portfolio.capital)
    case_results: dict[str, dict] = {}
    sl_cagr_a: float | None = None  # Case A baseline for alpha_retention

    for case_id, spec in CASE_SPECS.items():
        print(f"  [{case_id}] {spec['desc']}")
        sim = run_case(
            open_mat, close_mat,
            sig_mat, sig_ready, rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            cross90_mat, slope5_mat, slope20_mat,
            active_syms, sym_to_i, trade_syms, cfg,
            common_dates, spec,
        )

        fold_metrics: list[dict] = []
        for fold in FOLDS:
            fm = compute_fold_metrics(sim, close_mat, common_dates, fold,
                                       capital, spec["max_pos"], spec["weights"])
            fold_metrics.append(fm)
            if fm:
                ok = "✅" if fm.get("fold_pass") else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"base={fm['base_cagr']:+.1f}%  "
                      f"comb={fm['comb_cagr']:+.1f}%  "
                      f"Δ={fm['delta_cagr']:+.2f}pp  "
                      f"ΔDD={fm['delta_dd']:+.2f}pp  "
                      f"sl_CAGR={fm['sl_cagr']:+.1f}%  "
                      f"trig={fm['trigger_per_year']:.1f}/yr  {ok}")
            else:
                print(f"    Fold{fold['id']}: データ不足")

        wf = evaluate_wf(fold_metrics, sl_cagr_a)
        if case_id == "A":
            sl_cagr_a = wf.get("avg_slcagr")

        case_results[case_id] = {"folds": fold_metrics, "wf": wf, "sim": sim}

        ar_str = (f"α_ret={wf.get('alpha_retention',0):.1f}%"
                  if not math.isnan(wf.get("alpha_retention", float("nan"))) else "")
        ok_str = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"sl_CAGR={wf.get('avg_slcagr',0):+.1f}%  "
              f"{ar_str}  {ok_str}\n")

    # Re-evaluate B-F with Case A baseline
    if sl_cagr_a is not None:
        for case_id in list(CASE_SPECS)[1:]:
            folds = case_results[case_id]["folds"]
            case_results[case_id]["wf"] = evaluate_wf(folds, sl_cagr_a)

    # ── Summary ───────────────────────────────────────────────────────
    adopted = [c for c in CASE_SPECS if case_results[c]["wf"].get("adopted")]
    print(f"{'='*68}")
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for case in CASE_SPECS:
        wf = case_results[case]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        ar = wf.get("alpha_retention", float("nan"))
        ar_str = f"  α_ret={ar:.1f}%" if not math.isnan(ar) else ""
        print(f"  {ok} Case {case}: WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dc',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_dd',0):+.2f}pp  "
              f"sl_CAGR={wf.get('avg_slcagr',0):+.1f}%{ar_str}")
    print(f"{'='*68}\n")

    # ── Marginal DD analysis ─────────────────────────────────────────
    def avg_dd_for(case_id: str) -> float:
        return case_results[case_id]["wf"].get("avg_dd", float("nan"))

    dd_A = avg_dd_for("A")
    dd_B = avg_dd_for("B")
    dd_C = avg_dd_for("C")
    dd_D = avg_dd_for("D")
    dd_E = avg_dd_for("E")
    dd_F = avg_dd_for("F")

    # Best 2-slot case by lowest ΔDD
    best_2slot = min(["B","C","D","E"], key=lambda c: avg_dd_for(c))
    dd_best2   = avg_dd_for(best_2slot)

    marginal_dd: dict[str, dict] = {
        f"A(1slot) → B(2slot 70/30)":      {"from_dd": dd_A, "to_dd": dd_B, "n_slots": 1},
        f"A(1slot) → C(2slot 60/40)":      {"from_dd": dd_A, "to_dd": dd_C, "n_slots": 1},
        f"A(1slot) → D(2slot equal)":      {"from_dd": dd_A, "to_dd": dd_D, "n_slots": 1},
        f"A(1slot) → E(2slot no-refill)":  {"from_dd": dd_A, "to_dd": dd_E, "n_slots": 1},
        f"A(1slot) → F(3slot 50/30/20)":   {"from_dd": dd_A, "to_dd": dd_F, "n_slots": 2},
        f"{best_2slot}(2slot) → F(3slot)": {"from_dd": dd_best2, "to_dd": dd_F, "n_slots": 1},
    }

    print("  Marginal ΔDD saved per slot:")
    for k, v in marginal_dd.items():
        saved = v["from_dd"] - v["to_dd"]
        per   = saved / v["n_slots"] if v["n_slots"] > 0 else float("nan")
        print(f"    {k}: {saved:+.3f}pp total → {per:+.3f}pp/slot")

    # ── Report ───────────────────────────────────────────────────────
    print("\n[4/4] レポート生成...")
    write_report(case_results, topix_close,
                 REPORTS_DIR / "dedicated_alpha_concentration_relief.md",
                 marginal_dd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
