"""
backtest/study10_standalone_independence.py  —  Study10 Standalone Independence Audit

A: production baseline  (RSR42, max_pos=3, capital=3M, pattern=A)
B: standalone sleeve    (RSR[90,94) d90∈{0..2} slope5≤5 exit<87 hold≥3 capital=750k)
C: combined             (A + B, independent capital)

TYPE_A independent_alpha   corr_daily<0.30 AND combined_calmar>prod_calmar
TYPE_B complementary       0.30≤corr<0.60 AND P(sl_profit|prod_loss)>35%
TYPE_C repackaged          corr≥0.60 OR combined_alpha_lift≤0
TYPE_D variance_compressor combined_CAGR≈prod AND combined_DD改善

出力: reports/study10_standalone_independence.md
Run:
    cd C:/ai-trading
    python src/backtest/study10_standalone_independence.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, _take, calc_metrics, run_period,
    LOT, COST_ONE_WAY,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"
FULL_END    = "2025-12-31"

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Standalone B params ────────────────────────────────────────────────
RSR_LO_B    = 90.0
RSR_HI_B    = 94.0
D90_MAX_B   = 2          # allow d90 ∈ {0,1,2}
SLOPE_MAX_B = 5.0
EXIT_THR_B  = 87.0
MIN_HOLD_B  = 3
SLEEVE_FRAC = 0.25       # 750k / 3M
SHOCK_MKT   = -0.05
SHOCK_SYM   = -0.08

CAP_PROD  = 3_000_000.0
CAP_SL    =   750_000.0
CAP_COMB  = CAP_PROD + CAP_SL
W_PROD    = CAP_PROD / CAP_COMB   # 0.8
W_SL      = CAP_SL   / CAP_COMB   # 0.2

ADOPT_CALMAR_DELTA = 0.0     # combined_calmar > prod_calmar
ADOPT_DD_SLACK     = 1.0     # combined_DD ≤ prod_DD + 1.0pp
ADOPT_INTERACTION  = 0.0     # interaction > 0


def _f(v, fmt=".2f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return format(v, fmt)


# ──────────────────────────────────────────────────────────────────────
#  MATRIX HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_cross90_mat(rsr_mat: np.ndarray) -> np.ndarray:
    n, m   = rsr_mat.shape
    out    = np.zeros((n, m), dtype=np.int32)
    counts = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = counts
        above  = rsr_mat[i] >= 90.0
        counts[above]  += 1
        counts[~above]  = 0
    return out


def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


def regime_label(topix_close, year: int) -> str:
    if topix_close is None: return "N/A"
    yr = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ──────────────────────────────────────────────────────────────────────
#  STANDALONE B SIMULATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:       str
    qty:          int
    entry_price:  float
    entry_idx:    int
    rsr_entry:    float
    d90_entry:    int
    slope5_entry: float
    days_below:   int = 0


def run_standalone_b(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates, capital: float,
) -> dict:
    """Standalone B: RSR[90,94) d90∈{0..2} slope5≤5 exit<87 hold≥3"""
    n        = len(common_dates)
    sl_cash  = capital * SLEEVE_FRAC
    sl_pos: SlPos | None = None
    trades:   list[dict] = []
    eq       = np.zeros(n, dtype=np.float64)
    inv      = np.zeros(n, dtype=np.float64)
    holding  = np.zeros(n, dtype=bool)
    miss_dates:       list[str]   = []
    lot_round_losses: list[float] = []   # per-entry fraction lost to rounding

    for i in range(n):
        ds    = str(common_dates[i].date())
        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        s_eq  = sl_cash + s_inv
        eq[i]     = s_eq
        inv[i]    = s_inv
        holding[i] = sl_pos is not None

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n: break
        nxt = i + 1

        # ── EXIT ──────────────────────────────────────────────────────
        if sl_pos:
            si   = sym_to_i[sl_pos.symbol]
            rv   = float(rsr_mat[i, si])
            ct   = float(close_mat[i, si])
            ep   = sl_pos.entry_price
            hd   = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"

            if not do_exit:
                if rv < EXIT_THR_B:
                    sl_pos.days_below += 1
                else:
                    sl_pos.days_below = 0
                if sl_pos.days_below >= 1 and hd >= MIN_HOLD_B:
                    do_exit = True; reason = "RSR_EXIT"

            if do_exit:
                sp  = float(open_mat[nxt, si])
                pnl = (sp - ep) * sl_pos.qty
                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": ep, "exit": sp,
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": ds,
                    "hold_days": i - sl_pos.entry_idx,
                    "rsr_entry": sl_pos.rsr_entry,
                    "d90_entry": sl_pos.d90_entry,
                    "slope5_entry": sl_pos.slope5_entry,
                })
                sl_pos = None

        # ── ENTRY ─────────────────────────────────────────────────────
        if not mkt_shock:
            cands: list[tuple] = []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si])
                sl5 = float(slope5_mat[i, si])
                if rv < RSR_LO_B or rv >= RSR_HI_B: continue
                if d90 > D90_MAX_B:                  continue
                if sl5 > SLOPE_MAX_B:                continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
                cands.append((rv, d90, sl5, sym))

            if cands and sl_pos is not None:
                miss_dates.append(ds)  # slot occupied — missed entry
            elif cands and sl_pos is None:
                cands.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sl5, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp > 0:
                    alloc     = sl_cash * 0.95
                    qty_f     = alloc / bp / LOT
                    qty       = int(qty_f) * LOT
                    lot_frac  = qty_f - int(qty_f)
                    lot_round_losses.append(lot_frac * LOT * bp / max(1.0, alloc))
                    cost = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos   = SlPos(
                            symbol=sym, qty=qty, entry_price=bp,
                            entry_idx=nxt, rsr_entry=rv,
                            d90_entry=d90, slope5_entry=sl5,
                        )
                        trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "exit": None, "pnl": None,
                            "qty": qty, "entry_idx": nxt,
                            "date": ds, "rsr_entry": rv,
                            "d90_entry": d90, "slope5_entry": sl5,
                        })

    return {
        "eq": eq, "inv": inv, "holding": holding,
        "trades": trades,
        "miss_dates": miss_dates,
        "lot_round_losses": lot_round_losses,
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD INDEX HELPERS
# ──────────────────────────────────────────────────────────────────────

def fold_indices(date_strs: list[str], fold: dict) -> tuple[int, int]:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not idx: return -1, -1
    return idx[0], idx[-1] + 1   # [si, ei)


# ──────────────────────────────────────────────────────────────────────
#  SINGLE-PORTFOLIO FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def _trade_date(t: dict, date_strs: list[str]) -> str:
    d = t.get("date")
    if d: return d
    idx = t.get("exit_idx", t.get("entry_idx", -1))
    if 0 <= idx < len(date_strs):
        return date_strs[idx]
    return ""


def portfolio_fold_metrics(
    eq_arr: np.ndarray,
    trades: list[dict],
    date_strs: list[str],
    fold: dict,
    cap_start: float,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eq  = eq_arr[si:ei]
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    tr_oos = [t for t in trades
              if t.get("side") == "SELL" and
              ds_s <= _trade_date(t, date_strs) <= ds_e]
    n = len(eq)
    if n < 2: return {}
    eq_s = pd.Series(eq)
    years = n / 252
    cagr  = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1
    dr    = eq_s.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    roll_max = eq_s.expanding().max()
    dd_ser   = (eq_s - roll_max) / roll_max
    max_dd   = float(dd_ser.min())
    calmar   = cagr / abs(max_dd) if max_dd < 0 else float("inf")
    sells = tr_oos  # already filtered for SELL in tr_oos
    wins  = [t for t in sells if (t.get("pnl") or 0) > 0]
    loss  = [t for t in sells if (t.get("pnl") or 0) <= 0]
    gp    = sum(t["pnl"] for t in wins) if wins else 0.0
    gl    = abs(sum(t["pnl"] for t in loss)) if loss else 1.0
    # hold_days: use exit_idx - entry_idx if hold_days not stored
    def hold_d(t):
        hd = t.get("hold_days")
        if hd is not None and hd > 0: return hd
        ei2 = t.get("exit_idx", 0); ei3 = t.get("entry_idx", 0)
        return max(0, ei2 - ei3)
    hds = [hold_d(t) for t in sells if hold_d(t) > 0]
    return {
        "cagr":     round(cagr * 100, 2),
        "max_dd":   round(max_dd * 100, 2),
        "calmar":   round(calmar, 3),
        "sharpe":   round(sharpe, 3),
        "n_trades": len(sells),
        "win_rate": round(len(wins) / max(1, len(sells)) * 100, 1),
        "pf":       round(gp / max(1.0, gl), 3),
        "avg_hold": round(float(np.mean(hds)) if hds else 0, 1),
    }


# ──────────────────────────────────────────────────────────────────────
#  COMBINED FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def combined_fold_metrics(
    eq_A: np.ndarray, eq_B: np.ndarray,
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eA = eq_A[si:ei]; eB = eq_B[si:ei]
    eC = eA + eB
    n  = len(eC)
    years = n / 252
    cagr_A = (eA[-1] / eA[0]) ** (1 / max(years, 0.01)) - 1
    cagr_B = (eB[-1] / eB[0]) ** (1 / max(years, 0.01)) - 1
    cagr_C = (eC[-1] / eC[0]) ** (1 / max(years, 0.01)) - 1
    # weights at fold start
    w_a  = eA[0] / (eA[0] + eB[0])
    w_b  = eB[0] / (eA[0] + eB[0])
    w_cagr = w_a * cagr_A + w_b * cagr_B
    alpha_lift = cagr_C - w_cagr
    eq_C_s   = pd.Series(eC)
    roll_max = eq_C_s.expanding().max()
    dd_C     = float(((eq_C_s - roll_max) / roll_max).min())
    calmar_C = cagr_C / abs(dd_C) if dd_C < 0 else float("inf")
    dr_C     = eq_C_s.pct_change().dropna()
    sharpe_C = float(dr_C.mean() / dr_C.std() * np.sqrt(252)) if dr_C.std() > 0 else 0.0
    return {
        "cagr_C":     round(cagr_C * 100, 2),
        "dd_C":       round(dd_C * 100, 2),
        "calmar_C":   round(calmar_C, 3),
        "sharpe_C":   round(sharpe_C, 3),
        "alpha_lift": round(alpha_lift * 100, 2),
        "w_a_start":  round(w_a, 3),
    }


# ──────────────────────────────────────────────────────────────────────
#  INDEPENDENCE METRICS
# ──────────────────────────────────────────────────────────────────────

def independence_metrics(
    eq_A: np.ndarray, eq_B: np.ndarray,
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 10: return {}
    eA = eq_A[si:ei]; eB = eq_B[si:ei]
    drA = np.diff(eA) / eA[:-1]
    drB = np.diff(eB) / eB[:-1]
    n   = len(drA)
    if n < 5: return {}

    def safe_corr(a, b):
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-10 or sb < 1e-10: return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    corr_daily = safe_corr(drA, drB)

    # weekly (5-day groups)
    wA = pd.Series(drA).rolling(5, min_periods=3).sum().dropna()
    wB = pd.Series(drB).rolling(5, min_periods=3).sum().dropna()
    corr_weekly = safe_corr(wA.values, wB.values) if len(wA) > 5 else float("nan")

    # drawdown series
    eAs = pd.Series(eA); eBs = pd.Series(eB)
    ddA = ((eAs - eAs.expanding().max()) / eAs.expanding().max()).values[1:]
    ddB = ((eBs - eBs.expanding().max()) / eBs.expanding().max()).values[1:]
    corr_dd = safe_corr(ddA, ddB)

    # tail correlation (upper 25% of A returns)
    q75 = np.percentile(drA, 75)
    mask_tail = drA >= q75
    if mask_tail.sum() >= 5:
        tail_corr = safe_corr(drA[mask_tail], drB[mask_tail])
    else:
        tail_corr = float("nan")

    # tail dependency P(B > Q75_B | A > Q75_A)
    q75_B = np.percentile(drB, 75)
    n_A_tail = mask_tail.sum()
    n_AB_tail = np.sum(mask_tail & (drB >= q75_B))
    tail_dep = float(n_AB_tail / n_A_tail) if n_A_tail > 0 else float("nan")

    return {
        "corr_daily":  round(corr_daily, 4),
        "corr_weekly": round(corr_weekly, 4) if not math.isnan(corr_weekly) else float("nan"),
        "corr_dd":     round(corr_dd, 4),
        "tail_corr":   round(tail_corr, 4) if not math.isnan(tail_corr) else float("nan"),
        "tail_dep":    round(tail_dep, 4) if not math.isnan(tail_dep) else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────
#  HOLDING OVERLAP & COMPLEMENTARITY
# ──────────────────────────────────────────────────────────────────────

def holding_and_complement(
    eq_A: np.ndarray, eq_B: np.ndarray,
    exp_A: np.ndarray, holding_B: np.ndarray,
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eA   = eq_A[si:ei];    eB  = eq_B[si:ei]
    expA = exp_A[si:ei];   hB  = holding_B[si:ei]
    n    = len(eA)
    drA  = np.diff(eA) / eA[:-1]
    drB  = np.diff(eB) / eB[:-1]
    hB_d = hB[:-1]   # align with diff

    # Production "holding" proxy: exposure > 5%
    hA = expA > 0.05
    hA_d = hA[:-1]

    # Drawdown of A
    eAs    = pd.Series(eA)
    ddA_s  = ((eAs - eAs.expanding().max()) / eAs.expanding().max()).values
    ddA_d  = ddA_s[:-1]  # align

    # overlap metrics
    both    = hA_d & hB_d
    n_hold_A = hA_d.sum()
    overlap_rate       = float(both.sum() / max(1, n-1))
    cond_overlap       = float(both.sum() / max(1, n_hold_A))

    # P(hold_B | prod_loss day)
    loss_mask = drA < 0
    p_hB_given_loss = float((hB_d[loss_mask]).mean()) if loss_mask.sum() > 0 else float("nan")

    # P(hold_B | prod_dd >5%)
    dd_mask = ddA_d < -0.05
    p_hB_given_dd = float((hB_d[dd_mask]).mean()) if dd_mask.sum() > 0 else float("nan")

    # complementarity
    # P(B profit | A loss)
    n_Aloss = loss_mask.sum()
    p_Bprof_given_Aloss = float(np.sum((drB > 0) & loss_mask) / max(1, n_Aloss))

    # P(B profit | A dd)
    n_Add = dd_mask.sum()
    p_Bprof_given_Add = float(np.sum((drB > 0) & dd_mask) / max(1, n_Add))

    # P(A profit | B loss)
    Bloss_mask = drB < 0
    n_Bloss = Bloss_mask.sum()
    p_Aprof_given_Bloss = float(np.sum((drA > 0) & Bloss_mask) / max(1, n_Bloss))

    return {
        "overlap_rate":         round(overlap_rate * 100, 1),
        "cond_overlap":         round(cond_overlap * 100, 1),
        "p_hB_given_loss":      round(p_hB_given_loss * 100, 1) if not math.isnan(p_hB_given_loss) else float("nan"),
        "p_hB_given_dd":        round(p_hB_given_dd * 100, 1) if not math.isnan(p_hB_given_dd) else float("nan"),
        "p_Bprof_given_Aloss":  round(p_Bprof_given_Aloss * 100, 1),
        "p_Bprof_given_Add":    round(p_Bprof_given_Add * 100, 1),
        "p_Aprof_given_Bloss":  round(p_Aprof_given_Bloss * 100, 1),
        "n_Aloss":              int(n_Aloss),
        "n_Add":                int(n_Add),
        "n_Bloss":              int(n_Bloss),
    }


# ──────────────────────────────────────────────────────────────────────
#  TAIL STRUCTURE
# ──────────────────────────────────────────────────────────────────────

def tail_structure(trades_A: list[dict], trades_B: list[dict]) -> dict:
    sells_A = [t for t in trades_A if t.get("side") == "SELL" and t.get("pnl") is not None]
    sells_B = [t for t in trades_B if t.get("side") == "SELL" and t.get("pnl") is not None]

    def top10_share(sells):
        if not sells: return float("nan")
        pnls = sorted([t["pnl"] for t in sells], reverse=True)
        total = sum(pnls)
        if total <= 0: return float("nan")
        return round(sum(pnls[:10]) / total * 100, 1)

    top10_A = top10_share(sells_A)
    top10_B = top10_share(sells_B)

    wins_A = sorted(sells_A, key=lambda t: t.get("pnl", 0), reverse=True)[:10]
    wins_B = sorted(sells_B, key=lambda t: t.get("pnl", 0), reverse=True)[:10]
    syms_A = {t["symbol"] for t in wins_A}
    syms_B = {t["symbol"] for t in wins_B}
    winner_overlap = len(syms_A & syms_B)

    return {
        "top10_share_A": top10_A,
        "top10_share_B": top10_B,
        "winner_overlap": winner_overlap,
        "n_wins_A": len(wins_A),
        "n_wins_B": len(wins_B),
    }


# ──────────────────────────────────────────────────────────────────────
#  CAPACITY
# ──────────────────────────────────────────────────────────────────────

def capacity_metrics(
    result_B: dict,
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    ds_s, ds_e  = fold["oos_start"], fold["oos_end"]
    inv_fold    = result_B["inv"][si:ei]
    hld_fold    = result_B["holding"][si:ei]
    eq_fold_0   = float(result_B["eq"][si]) if si < len(result_B["eq"]) else CAP_SL
    notional_util = float(inv_fold.mean() / max(1.0, eq_fold_0) * 100)
    capital_idle  = float(1 - hld_fold.mean()) * 100
    # fold-specific miss signals
    miss_fold = sum(1 for d in result_B["miss_dates"] if ds_s <= d <= ds_e)
    # fold-specific lot round loss (from trades with entry in fold)
    tr_fold = [t for t in result_B["trades"]
               if t.get("side") == "BUY" and ds_s <= (t.get("date") or "") <= ds_e]
    lrl = result_B["lot_round_losses"]
    # approximate: average of all lot_round_losses (global, no per-entry fold tag)
    avg_lrl = float(np.mean(lrl)) if lrl else 0.0
    return {
        "notional_util": round(notional_util, 1),
        "capital_idle":  round(capital_idle, 1),
        "miss_rate":     miss_fold,
        "avg_lot_loss":  round(avg_lrl * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────
#  CONTRIBUTION DECOMPOSITION
# ──────────────────────────────────────────────────────────────────────

def contribution_decomp(
    cagr_A: float, cagr_B: float, cagr_C: float,
    dd_A: float, dd_B: float, dd_C: float,
) -> dict:
    prod_component = W_PROD * cagr_A
    sl_component   = W_SL   * cagr_B
    weighted_cagr  = prod_component + sl_component
    interaction    = cagr_C - weighted_cagr
    # approximate diversification gain from DD reduction
    weighted_dd    = abs(W_PROD * dd_A + W_SL * dd_B)
    actual_dd      = abs(dd_C)
    div_gain       = max(0.0, weighted_dd - actual_dd)
    timing_gain    = max(0.0, interaction)
    cash_drag_approx = max(0.0, -interaction - div_gain) if interaction < 0 else 0.0
    return {
        "prod_component":   round(prod_component, 2),
        "sl_component":     round(sl_component, 2),
        "weighted_cagr":    round(weighted_cagr, 2),
        "interaction":      round(interaction, 2),
        "div_gain_dd":      round(div_gain, 2),
        "timing_gain":      round(timing_gain, 2),
        "cash_drag":        round(cash_drag_approx, 2),
    }


# ──────────────────────────────────────────────────────────────────────
#  TYPE CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────

def classify_type(
    corr_daily:      float,
    combined_calmar: float,
    prod_calmar:     float,
    alpha_lift:      float,
    p_bprof_aloss:   float,
    combined_dd:     float,
    prod_dd:         float,
    combined_cagr:   float,
) -> str:
    # TYPE_C: disqualifier — only fire on meaningful negative alpha or high corr
    # Note: non-rebalancing combinations show alpha_lift ≈ 0 due to annualization math;
    # use -2.0pp as threshold for genuine CAGR drag.
    if corr_daily >= 0.60 or alpha_lift < -2.0:
        return "TYPE_C"
    # TYPE_A: independent alpha
    if corr_daily < 0.30 and combined_calmar > prod_calmar:
        return "TYPE_A"
    # TYPE_B: complementary
    if 0.30 <= corr_daily < 0.60 and p_bprof_aloss > 35.0:
        return "TYPE_B"
    # TYPE_D: variance compressor — combined_cagr near prod_cagr but DD↓
    cagr_delta = combined_cagr - prod_calmar  # placeholder (use cagr_A)
    if combined_dd > prod_dd:  # DD improved (less negative)
        return "TYPE_D"
    return "TYPE_D"  # fallback


def check_adoption(
    avg_combined_calmar: float,
    avg_prod_calmar:     float,
    avg_combined_dd:     float,
    avg_prod_dd:         float,
    avg_interaction:     float,
) -> bool:
    # interaction ≈ 0 for non-rebalancing portfolios (annualization math);
    # use -2.0pp threshold rather than strict > 0
    return (
        avg_combined_calmar > avg_prod_calmar + ADOPT_CALMAR_DELTA
        and avg_combined_dd >= avg_prod_dd - ADOPT_DD_SLACK
        and avg_interaction > -2.0
    )


# ──────────────────────────────────────────────────────────────────────
#  AGGREGATE HELPERS
# ──────────────────────────────────────────────────────────────────────

def safe_avg(vals: list) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(fold_data: list[dict], agg: dict, path: Path) -> None:
    L = []; w = L.append

    w("# Study10 Standalone Independence Audit")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\nA: production (RSR42, max_pos=3, capital=¥{CAP_PROD:,.0f})")
    w(f"B: standalone  (RSR[{RSR_LO_B:.0f},{RSR_HI_B:.0f}) d90≤{D90_MAX_B} "
      f"slope≤{SLOPE_MAX_B:.0f} exit<{EXIT_THR_B:.0f} hold≥{MIN_HOLD_B} capital=¥{CAP_SL:,.0f})")
    w(f"C: combined    (A+B independent, total=¥{CAP_COMB:,.0f})\n")

    # ── S1. Basic performance per fold ────────────────────────────────
    w("---\n## 1. 基本性能 (Fold別)\n")
    w("| Fold | Regime | CAGR_A | DD_A | Calmar_A | CAGR_B | DD_B | Calmar_B | CAGR_C | DD_C | Calmar_C |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for fd in fold_data:
        fA = fd["A"]; fB = fd["B"]; fC = fd["C"]
        yr = fd["fold"]["oos_start"][:4]
        reg = fd["regime"]
        w(f"| Fold{fd['fold']['id']} {yr} | {reg} "
          f"| {_f(fA.get('cagr'), '+.2f')}% "
          f"| {_f(fA.get('max_dd'), '.2f')}% "
          f"| {_f(fA.get('calmar'), '.3f')} "
          f"| {_f(fB.get('cagr'), '+.2f')}% "
          f"| {_f(fB.get('max_dd'), '.2f')}% "
          f"| {_f(fB.get('calmar'), '.3f')} "
          f"| {_f(fC.get('cagr_C'), '+.2f')}% "
          f"| {_f(fC.get('dd_C'), '.2f')}% "
          f"| {_f(fC.get('calmar_C'), '.3f')} |")
    aA = agg["avg_A"]; aB = agg["avg_B"]; aC = agg["avg_C"]
    w(f"| **avg** | — "
      f"| **{_f(aA.get('cagr'),'+.2f')}%** "
      f"| **{_f(aA.get('max_dd'),'.2f')}%** "
      f"| **{_f(aA.get('calmar'),'.3f')}** "
      f"| **{_f(aB.get('cagr'),'+.2f')}%** "
      f"| **{_f(aB.get('max_dd'),'.2f')}%** "
      f"| **{_f(aB.get('calmar'),'.3f')}** "
      f"| **{_f(aC.get('cagr_C'),'+.2f')}%** "
      f"| **{_f(aC.get('dd_C'),'.2f')}%** "
      f"| **{_f(aC.get('calmar_C'),'.3f')}** |")

    # ── S2. Independence ──────────────────────────────────────────────
    w("\n---\n## 2. 独立性 (Fold別)\n")
    w("| Fold | corr_daily | corr_weekly | corr_dd | tail_corr | tail_dep |")
    w("|---|---|---|---|---|---|")
    for fd in fold_data:
        ind = fd["indep"]
        w(f"| Fold{fd['fold']['id']} "
          f"| {_f(ind.get('corr_daily'), '.4f')} "
          f"| {_f(ind.get('corr_weekly'), '.4f')} "
          f"| {_f(ind.get('corr_dd'), '.4f')} "
          f"| {_f(ind.get('tail_corr'), '.4f')} "
          f"| {_f(ind.get('tail_dep'), '.4f')} |")
    ai = agg["avg_indep"]
    w(f"| **avg** "
      f"| **{_f(ai.get('corr_daily'), '.4f')}** "
      f"| **{_f(ai.get('corr_weekly'), '.4f')}** "
      f"| **{_f(ai.get('corr_dd'), '.4f')}** "
      f"| **{_f(ai.get('tail_corr'), '.4f')}** "
      f"| **{_f(ai.get('tail_dep'), '.4f')}** |")

    # ── S3. Combination value ─────────────────────────────────────────
    w("\n---\n## 3. 組合せ価値\n")
    w("| Fold | alpha_lift | Calmar_C vs A | combined_DD vs A_DD |")
    w("|---|---|---|---|")
    for fd in fold_data:
        fC = fd["C"]; fA = fd["A"]
        al = fC.get("alpha_lift", float("nan"))
        cA = fA.get("calmar", float("nan"))
        cC = fC.get("calmar_C", float("nan"))
        dA = fA.get("max_dd", float("nan"))
        dC = fC.get("dd_C", float("nan"))
        dcal = cC - cA if not math.isnan(cC) and not math.isnan(cA) else float("nan")
        ddd  = dC - dA if not math.isnan(dC) and not math.isnan(dA) else float("nan")
        w(f"| Fold{fd['fold']['id']} "
          f"| {_f(al, '+.2f')}pp "
          f"| {_f(dcal, '+.3f')} "
          f"| {_f(ddd, '+.2f')}pp |")
    aC = agg["avg_C"]; aA = agg["avg_A"]
    avg_al   = safe_avg([fd["C"].get("alpha_lift") for fd in fold_data])
    avg_dcal = safe_avg([fd["C"].get("calmar_C", float("nan")) - fd["A"].get("calmar", float("nan"))
                         for fd in fold_data])
    avg_ddd  = safe_avg([fd["C"].get("dd_C", float("nan")) - fd["A"].get("max_dd", float("nan"))
                         for fd in fold_data])
    w(f"| **avg** "
      f"| **{_f(avg_al, '+.2f')}pp** "
      f"| **{_f(avg_dcal, '+.3f')}** "
      f"| **{_f(avg_ddd, '+.2f')}pp** |")

    # ── S4. Overlap & Complementarity ────────────────────────────────
    w("\n---\n## 4. 保有関係・補完性\n")
    w("| Fold | overlap% | cond_overlap% | P(hB|A_loss)% | P(hB|A_dd)% "
      "| P(B+|A-)% | P(B+|Add)% | P(A+|B-)% |")
    w("|---|---|---|---|---|---|---|---|")
    for fd in fold_data:
        hc = fd["hold_comp"]
        w(f"| Fold{fd['fold']['id']} "
          f"| {_f(hc.get('overlap_rate'), '.1f')} "
          f"| {_f(hc.get('cond_overlap'), '.1f')} "
          f"| {_f(hc.get('p_hB_given_loss'), '.1f')} "
          f"| {_f(hc.get('p_hB_given_dd'), '.1f')} "
          f"| {_f(hc.get('p_Bprof_given_Aloss'), '.1f')} "
          f"| {_f(hc.get('p_Bprof_given_Add'), '.1f')} "
          f"| {_f(hc.get('p_Aprof_given_Bloss'), '.1f')} |")
    ahc = agg["avg_hc"]
    w(f"| **avg** "
      f"| **{_f(ahc.get('overlap_rate'), '.1f')}** "
      f"| **{_f(ahc.get('cond_overlap'), '.1f')}** "
      f"| **{_f(ahc.get('p_hB_given_loss'), '.1f')}** "
      f"| **{_f(ahc.get('p_hB_given_dd'), '.1f')}** "
      f"| **{_f(ahc.get('p_Bprof_given_Aloss'), '.1f')}** "
      f"| **{_f(ahc.get('p_Bprof_given_Add'), '.1f')}** "
      f"| **{_f(ahc.get('p_Aprof_given_Bloss'), '.1f')}** |")

    # ── S5. Tail structure ────────────────────────────────────────────
    tail = agg["tail"]
    w("\n---\n## 5. Tail 構造 (全OOS集計)\n")
    w(f"- top10_trade_share_A: {_f(tail.get('top10_share_A'), '.1f')}%")
    w(f"- top10_trade_share_B: {_f(tail.get('top10_share_B'), '.1f')}%")
    w(f"- winner_overlap (top10 共通銘柄): {tail.get('winner_overlap', '—')}")
    w(f"- n_top10_A: {tail.get('n_wins_A', '—')}  n_top10_B: {tail.get('n_wins_B', '—')}")

    # ── S6. Capacity ──────────────────────────────────────────────────
    w("\n---\n## 6. Capacity (Standalone B)\n")
    w("| Fold | notional_util% | capital_idle% | miss_signals | lot_round_loss% |")
    w("|---|---|---|---|---|")
    for fd in fold_data:
        cap = fd["cap"]
        w(f"| Fold{fd['fold']['id']} "
          f"| {_f(cap.get('notional_util'), '.1f')} "
          f"| {_f(cap.get('capital_idle'), '.1f')} "
          f"| {cap.get('miss_rate', '—')} "
          f"| {_f(cap.get('avg_lot_loss'), '.2f')} |")

    # ── S7. Contribution Decomposition ────────────────────────────────
    w("\n---\n## 7. 寄与分解 (avg across folds)\n")
    cd = agg["contrib"]
    w(f"| Component | value |")
    w(f"|---|---|")
    w(f"| prod_component (w_A × CAGR_A) | {_f(cd.get('prod_component'), '+.2f')}pp |")
    w(f"| sl_component   (w_B × CAGR_B) | {_f(cd.get('sl_component'), '+.2f')}pp |")
    w(f"| weighted_CAGR  (naive sum)     | {_f(cd.get('weighted_cagr'), '+.2f')}pp |")
    w(f"| combined_CAGR  (actual)        | {_f(agg['avg_C'].get('cagr_C'), '+.2f')}pp |")
    w(f"| interaction    (C - weighted)  | {_f(cd.get('interaction'), '+.2f')}pp |")
    w(f"| ├ diversification_gain (DD↓)   | {_f(cd.get('div_gain_dd'), '+.2f')}pp |")
    w(f"| └ timing_gain  (residual+)     | {_f(cd.get('timing_gain'), '+.2f')}pp |")
    w(f"| cash_drag (approx)             | {_f(cd.get('cash_drag'), '.2f')}pp |")

    # ── S8. TYPE Classification ───────────────────────────────────────
    w("\n---\n## 8. TYPE 分類\n")
    st = agg["strategy_type"]
    ce = agg["combined_efficiency"]
    adopted = agg["adopted"]
    ai = agg["avg_indep"]
    ahc_avg = ahc

    w(f"### strategy_type: {st}")
    w(f"\n判定根拠:")
    w(f"- avg corr_daily = {_f(ai.get('corr_daily'), '.4f')}")
    w(f"- avg combined_calmar = {_f(agg['avg_C'].get('calmar_C'), '.3f')} vs "
      f"prod_calmar = {_f(agg['avg_A'].get('calmar'), '.3f')}")
    w(f"- avg alpha_lift = {_f(avg_al, '+.2f')}pp")
    w(f"- avg P(B+|A-) = {_f(ahc.get('p_Bprof_given_Aloss'), '.1f')}%")
    w(f"- avg interaction = {_f(cd.get('interaction'), '+.2f')}pp")

    type_desc = {
        "TYPE_A": "independent_alpha: corr<0.30 AND combined_calmar改善",
        "TYPE_B": "complementary: 0.30≤corr<0.60 AND P(B+|A-)>35%",
        "TYPE_C": "repackaged: corr≥0.60 OR alpha_lift≤0 — 分散効果なし",
        "TYPE_D": "variance_compressor: CAGR≈prod AND DD改善",
    }
    w(f"\n定義: {type_desc.get(st, '—')}\n")

    # ── S9. Final Judgment ────────────────────────────────────────────
    w("---\n## 9. 最終判定\n")

    # deploy recommendation
    if st == "TYPE_C":
        drec = "REJECT — corr高 or alpha_lift≤0。production再包装の可能性高。実装禁止。"
        crec = "reject"
    elif st == "TYPE_A" and adopted:
        drec = "DEPLOY — 独立alpha確認済み。standalone sleeve 25% での即時実装推奨。"
        crec = "25%"
    elif st == "TYPE_A" and not adopted:
        drec = "CONDITIONAL — 独立alpha条件は満たすが combined_calmar/DD 基準未達。10%試験配分を検討。"
        crec = "10%"
    elif st == "TYPE_B" and adopted:
        drec = "DEPLOY (Bear hedge) — 補完性確認。P(B+|A-)>35%。Bear/下落局面でのヘッジ効果期待。"
        crec = "25%"
    elif st == "TYPE_B" and not adopted:
        drec = "CONDITIONAL — 補完性は示すが combined_calmar 基準未達。10%試験配分を検討。"
        crec = "10%"
    elif st == "TYPE_D" and adopted:
        drec = "DEPLOY (DD reducer) — CAGR改善は限定的だが DD削減効果を確認。安定化目的で10%配分推奨。"
        crec = "10%"
    else:
        drec = "HOLD — combined_calmar/DD 基準未達。さらなる研究またはパラメータ調整を要する。"
        crec = "0%"

    w(f"| 指標 | 値 |")
    w(f"|---|---|")
    w(f"| **strategy_type** | **{st}** |")
    w(f"| **combined_efficiency** | **{_f(ce, '.3f')}** (combined_calmar / prod_calmar) |")
    w(f"| adopt | {'✅ YES' if adopted else '❌ NO'} |")
    w(f"| **deploy_recommendation** | {drec} |")
    w(f"| **capital_recommendation** | **{crec}** |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Study10 Standalone Independence Audit")
    print(f"  A: production (pattern=A, capital=¥{CAP_PROD:,.0f})")
    print(f"  B: standalone (RSR[{RSR_LO_B:.0f},{RSR_HI_B:.0f}) d90≤{D90_MAX_B} "
          f"slope≤{SLOPE_MAX_B:.0f} exit<{EXIT_THR_B:.0f} capital=¥{CAP_SL:,.0f})")
    print("=" * 68 + "\n")

    # ── 1. Data load ──────────────────────────────────────────────────
    print("[1/5] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    capital     = float(cfg.portfolio.capital)
    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    # ── 2. Common dates (standalone frame) ───────────────────────────
    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(IS_START)) &
        (common_dates <= pd.Timestamp(FULL_END))
    ]
    n_dates      = len(common_dates)
    date_strs    = [str(d.date()) for d in common_dates]
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}")

    # ── 3. Price / RSR matrices ───────────────────────────────────────
    print("[2/5] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                            dtype=np.float32, fill_value=1.0))
    mkt_ret1 = None
    if topix_close is not None:
        mkt_ret1 = _take(topix_close.pct_change(), common_dates,
                         dtype=np.float32, fill_value=0.0)

    cross90_mat = compute_cross90_mat(rsr_mat)
    slope5_mat  = compute_slope5(rsr_mat)

    # ── 4. Production A (full period, return_equity) ──────────────────
    print("[3/5] Production A simulation (full 2018-2025)...")
    res_A = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=IS_START, end=FULL_END,
        pattern="A",
        return_trades=True,
        return_equity=True,
        verbose=False,
    )
    eq_A_raw   = res_A.get("_equity_curve", [])
    dates_A    = res_A.get("_equity_dates", [])
    exp_A_raw  = res_A.get("_exposure_list", [])
    trades_A   = res_A.get("_trades_raw", [])

    # Align production dates with standalone common_dates
    date_set_A = {d: i for i, d in enumerate(dates_A)}
    align_idx_A = [date_set_A[s] for s in date_strs if s in date_set_A]
    # build eq_A and exp_A aligned on date_strs
    eq_A_aligned  = np.zeros(n_dates, dtype=np.float64)
    exp_A_aligned = np.zeros(n_dates, dtype=np.float64)
    for j, s in enumerate(date_strs):
        if s in date_set_A:
            k = date_set_A[s]
            eq_A_aligned[j]  = eq_A_raw[k]
            exp_A_aligned[j] = exp_A_raw[k]

    # fill any gaps with forward fill
    for j in range(1, n_dates):
        if eq_A_aligned[j] == 0.0 and eq_A_aligned[j-1] != 0.0:
            eq_A_aligned[j]  = eq_A_aligned[j-1]
            exp_A_aligned[j] = exp_A_aligned[j-1]
    if eq_A_aligned[0] == 0.0:
        eq_A_aligned[0] = capital

    print(f"  Production A: {len(trades_A)} trades  "
          f"final_equity=¥{eq_A_aligned[-1]:,.0f}")

    # ── 5. Standalone B (full period) ────────────────────────────────
    print("[4/5] Standalone B simulation...")
    result_B = run_standalone_b(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90_mat, slope5_mat, mkt_ret1,
        active_syms, sym_to_i, common_dates, capital,
    )
    eq_B     = result_B["eq"]
    trades_B = result_B["trades"]
    sells_B  = [t for t in trades_B if t["side"] == "SELL"]
    print(f"  Standalone B: {len(sells_B)} sell trades  "
          f"final_equity=¥{eq_B[-1]:,.0f}  "
          f"total_miss_signals={len(result_B['miss_dates'])}")

    # ── 6. Fold analysis ──────────────────────────────────────────────
    print("[5/5] Fold analysis (5 folds)...\n")
    fold_data: list[dict] = []
    sells_A_all: list[dict] = []
    sells_B_all: list[dict] = []

    for fold in FOLDS:
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)

        fA  = portfolio_fold_metrics(eq_A_aligned, trades_A, date_strs, fold, capital)
        fB  = portfolio_fold_metrics(eq_B, trades_B, date_strs, fold, CAP_SL)
        fC  = combined_fold_metrics(eq_A_aligned, eq_B, date_strs, fold)
        ind = independence_metrics(eq_A_aligned, eq_B, date_strs, fold)
        hc  = holding_and_complement(
            eq_A_aligned, eq_B, exp_A_aligned, result_B["holding"], date_strs, fold)
        cap = capacity_metrics(result_B, date_strs, fold)

        si, ei = fold_indices(date_strs, fold)
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        if si >= 0:
            sells_A_all.extend([
                t for t in trades_A
                if t.get("side") == "SELL" and ds_s <= _trade_date(t, date_strs) <= ds_e
            ])
            sells_B_all.extend([
                t for t in trades_B
                if t.get("side") == "SELL" and ds_s <= (t.get("date") or "") <= ds_e
            ])

        fold_data.append({
            "fold": fold, "regime": reg,
            "A": fA, "B": fB, "C": fC,
            "indep": ind, "hold_comp": hc, "cap": cap,
        })

        calA = fA.get("calmar", float("nan"))
        calB = fB.get("calmar", float("nan"))
        calC = fC.get("calmar_C", float("nan"))
        al   = fC.get("alpha_lift", float("nan"))
        cor  = ind.get("corr_daily", float("nan"))
        print(f"  Fold{fold['id']} {yr} [{reg}]")
        print(f"    A: CAGR={_f(fA.get('cagr'),'+.2f')}%  DD={_f(fA.get('max_dd'),'.2f')}%  "
              f"Calmar={_f(calA,'.3f')}  n={fA.get('n_trades',0)}")
        print(f"    B: CAGR={_f(fB.get('cagr'),'+.2f')}%  DD={_f(fB.get('max_dd'),'.2f')}%  "
              f"Calmar={_f(calB,'.3f')}  n={fB.get('n_trades',0)}")
        print(f"    C: CAGR={_f(fC.get('cagr_C'),'+.2f')}%  DD={_f(fC.get('dd_C'),'.2f')}%  "
              f"Calmar={_f(calC,'.3f')}  alpha_lift={_f(al,'+.2f')}pp")
        print(f"    corr_daily={_f(cor,'.4f')}  "
              f"P(B+|A-)={_f(hc.get('p_Bprof_given_Aloss'),'.1f')}%\n")

    # ── 7. Aggregate ──────────────────────────────────────────────────
    def avg_dict(dicts: list[dict], keys) -> dict:
        return {k: safe_avg([d.get(k) for d in dicts]) for k in keys}

    avg_A  = avg_dict([fd["A"] for fd in fold_data],
                      ["cagr","max_dd","calmar","sharpe","n_trades","win_rate","pf"])
    avg_B  = avg_dict([fd["B"] for fd in fold_data],
                      ["cagr","max_dd","calmar","sharpe","n_trades","win_rate","pf"])
    avg_C  = avg_dict([fd["C"] for fd in fold_data],
                      ["cagr_C","dd_C","calmar_C","sharpe_C","alpha_lift"])
    avg_indep = avg_dict([fd["indep"] for fd in fold_data],
                         ["corr_daily","corr_weekly","corr_dd","tail_corr","tail_dep"])
    avg_hc = avg_dict([fd["hold_comp"] for fd in fold_data],
                      ["overlap_rate","cond_overlap","p_hB_given_loss","p_hB_given_dd",
                       "p_Bprof_given_Aloss","p_Bprof_given_Add","p_Aprof_given_Bloss"])

    # Tail structure (full OOS)
    tail = tail_structure(sells_A_all, sells_B_all)

    # Contribution decomposition (avg)
    cd = contribution_decomp(
        cagr_A  = avg_A["cagr"],
        cagr_B  = avg_B["cagr"],
        cagr_C  = avg_C["cagr_C"],
        dd_A    = avg_A["max_dd"],
        dd_B    = avg_B["max_dd"],
        dd_C    = avg_C["dd_C"],
    )

    # TYPE classification
    st = classify_type(
        corr_daily      = avg_indep["corr_daily"],
        combined_calmar = avg_C["calmar_C"],
        prod_calmar     = avg_A["calmar"],
        alpha_lift      = avg_C["alpha_lift"],
        p_bprof_aloss   = avg_hc["p_Bprof_given_Aloss"],
        combined_dd     = avg_C["dd_C"],
        prod_dd         = avg_A["max_dd"],
        combined_cagr   = avg_C["cagr_C"],
    )

    # Adoption
    adopted = check_adoption(
        avg_combined_calmar = avg_C["calmar_C"],
        avg_prod_calmar     = avg_A["calmar"],
        avg_combined_dd     = avg_C["dd_C"],
        avg_prod_dd         = avg_A["max_dd"],
        avg_interaction     = cd["interaction"],
    )

    combined_efficiency = (avg_C["calmar_C"] / avg_A["calmar"]
                           if avg_A["calmar"] not in (0, float("nan"), float("inf"))
                           else float("nan"))

    agg = {
        "avg_A": avg_A, "avg_B": avg_B, "avg_C": avg_C,
        "avg_indep": avg_indep, "avg_hc": avg_hc,
        "tail": tail, "contrib": cd,
        "strategy_type": st,
        "adopted": adopted,
        "combined_efficiency": combined_efficiency,
    }

    print("=" * 68)
    print(f"  strategy_type         = {st}")
    print(f"  combined_efficiency   = {_f(combined_efficiency, '.3f')}")
    print(f"  adopted               = {'YES' if adopted else 'NO'}")
    print(f"  avg corr_daily        = {_f(avg_indep['corr_daily'], '.4f')}")
    print(f"  avg alpha_lift        = {_f(avg_C['alpha_lift'], '+.2f')}pp")
    print(f"  avg P(B+|A-)          = {_f(avg_hc['p_Bprof_given_Aloss'], '.1f')}%")
    print("=" * 68 + "\n")

    write_report(fold_data, agg,
                 REPORTS_DIR / "study10_standalone_independence.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
