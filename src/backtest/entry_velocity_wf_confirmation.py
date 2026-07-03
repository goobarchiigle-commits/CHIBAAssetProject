"""
backtest/entry_velocity_wf_confirmation.py  —  Entry Velocity WF Confirmation

目的: Study8 系列の改善源が position sizing でなく entry quality か検証

Cases:
  A  baseline      既存 production 条件 (no slope filter)
  B  GLOBAL_FILTER entry に rsr_slope_5d<=5 を追加
  C  SOFT_PENALTY  ranking score = RSR − λ*max(0,slope5−5)  (発注ブロックなし)
  D  MONITOR_ONLY  発注変更なし、B 適用時の仮想監査のみ

判定 (PROMOTE):
  WF>=4/5 AND ΔCAGR>=+0.3pp AND ΔDD<=+1.5pp
  AND selection_gain>=1.10
  AND bad_trade_removed >= 1.5×opportunity_loss

CAUTION:
  winner_removed_rate > 30%

最終出力:
  filter_precision / filter_recall / best_filter
  recommend_promotion / recommend_live_shadow (PASS/FAIL)

出力: reports/entry_velocity_wf_confirmation.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_velocity_wf_confirmation.py
"""
from __future__ import annotations

import sys, math, time, warnings
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

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

SHOCK_MKT_THR = -0.05
SHOCK_SYM_THR = -0.08
SLOPE_FILTER  = 5.0   # Case B: hard threshold
LAMBDA        = 2.0   # Case C: penalty coefficient
FWD_DAYS      = 20    # for forward return attribution

# ── Promotion criteria ─────────────────────────────────────────────────
ADOPT_WF          = 4
ADOPT_DCAGR       = 0.3
ADOPT_DDD         = 1.5
ADOPT_SEL_GAIN    = 1.10
ADOPT_BTR_RATIO   = 1.5   # bad_trade_removed >= 1.5 × opportunity_loss
CAUTION_WRR       = 0.30   # winner_removed_rate > 30%


# ──────────────────────────────────────────────────────────────────────
#  CASE SPEC
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CaseSpec:
    label:        str
    slope_filter: float | None   # B: 5.0 (hard block)
    slope_lambda: float | None   # C: 2.0 (soft penalty)
    monitor_only: bool = False   # D: True


CASES: dict[str, CaseSpec] = {
    "A": CaseSpec("baseline (no filter)",                  None,         None,  False),
    "B": CaseSpec(f"GLOBAL_FILTER slope≤{SLOPE_FILTER:.0f}", SLOPE_FILTER, None,  False),
    "C": CaseSpec(f"SOFT_PENALTY λ={LAMBDA:.1f}",           None,         LAMBDA, False),
    "D": CaseSpec("MONITOR_ONLY",                           None,         None,  True),
}

CASE_ORDER = ["A", "B", "C", "D"]


# ──────────────────────────────────────────────────────────────────────
#  SLOPE MATRIX
# ──────────────────────────────────────────────────────────────────────

def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


def regime_label_for(topix_close, year: int) -> str:
    if topix_close is None: return "N/A"
    yr  = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ──────────────────────────────────────────────────────────────────────
#  PORTFOLIO SIMULATION
# ──────────────────────────────────────────────────────────────────────

def run_portfolio(
    open_mat, close_mat,
    sig_mat, sig_ready, rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    slope5_mat,
    active_syms, sym_to_i, trade_syms, cfg, common_dates,
    spec: CaseSpec,
) -> dict:
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    rc   = getattr(cfg, "risk_controls", None)
    MS   = float(rc.sector_cap)            if rc else 0.25
    gen  = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gn   = float(getattr(rc, "gross_cap_normal", 1.0))         if rc else 1.0
    gd5  = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6))  if rc else 0.6
    gd8  = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4))  if rc else 0.4

    cash = float(capital); pos: dict[str, Position] = {}
    peak = float(capital); cb_active = False; cb_days = 0
    reentry_ban: dict[str, int] = {}; trades: list[dict] = []
    n   = len(common_dates)
    eq  = np.zeros(n, dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)

    for i, date in enumerate(common_dates):
        ds    = str(date.date())
        b_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]]) for s, p in pos.items())
        b_eq  = cash + b_inv
        eq[i] = b_eq; inv[i] = b_inv

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
            rv = float(rsr_mat[i, si]); ct = float(close_mat[i, si])
            sl5 = float(slope5_mat[i, si])

            if shock and h:
                if i > 0:
                    pc = float(close_mat[i - 1, si])
                    if pc > 0 and (ct / pc - 1) <= SHOCK_SYM_THR:
                        sells.append((sym, "MKT_SHOCK")); continue
            if shock and not h: continue
            if h and max_hold and hd > max_hold: sells.append((sym, "TIME_STOP")); continue
            if h and rv < rsr_exit_thr and hd >= min_hold: sells.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and h and hd >= min_hold:
                sells.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not h:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue

                # ── Case-specific filter ───────────────────────────────
                if spec.monitor_only:
                    # D: same as A (no filter), just track slope
                    score = rv
                elif spec.slope_filter is not None:
                    # B: hard block
                    if sl5 > spec.slope_filter: continue
                    score = rv
                elif spec.slope_lambda is not None:
                    # C: soft penalty
                    score = rv - spec.slope_lambda * max(0.0, sl5 - SLOPE_FILTER)
                else:
                    # A: pure RSR ranking
                    score = rv

                buys.append((score, rv, sl5, sym))

        for sym, reason in sells:
            if sym not in pos: continue
            p = pos[sym]; si = sym_to_i[sym]
            sp = float(open_mat[nxt, si])
            pnl = (sp - p.entry_price) * p.qty
            cash += p.qty * sp * (1 - COST_ONE_WAY)
            trades.append({
                "side": "SELL", "symbol": sym, "pnl": pnl,
                "entry": p.entry_price, "exit": sp, "qty": p.qty,
                "entry_idx": p.entry_idx, "exit_idx": i,
                "reason": reason, "date": ds,
                "hold_days": i - p.entry_idx,
                "rsr_at_entry": getattr(p, "rsr_at_entry", 0.0),
            })
            del pos[sym]
            if reason == "TIME_STOP": reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not cb_active and buys:
            buys.sort(key=lambda x: -x[0])   # descending score
            for score, rv, sl5, sym in buys:
                si = sym_to_i[sym]
                if max_pos - len(pos) <= 0: break
                bp = float(open_mat[nxt, si])
                if bp <= 0: continue
                if not _sector_ok(sym, pos, close_mat, i, sym_to_i, trade_syms, capital, scap): continue
                if gen:
                    cg = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                             for p in pos.values()) / max(1.0, capital)
                    if cg + bp * LOT / max(1.0, capital) > gc: continue
                alloc = capital / max_pos
                qty   = int(alloc / bp / LOT) * LOT
                cost  = qty * bp * (1 + COST_ONE_WAY)
                if qty <= 0 or cost > cash: continue
                _execute_buy(sym, bp, qty, i, nxt, trade_syms, trades, pos, rv)
                cash -= cost
                trades[-1]["date"]        = ds
                trades[-1]["slope5"]      = sl5
                trades[-1]["sig_date_idx"]= i

    return {"eq": eq, "inv": inv, "trades": trades}


# ──────────────────────────────────────────────────────────────────────
#  ATTRIBUTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def build_trade_key(t: dict) -> str:
    # entry_idx = execution day (nxt), consistent across BUY and SELL records
    return f"{t['symbol']}@{t.get('entry_idx', 0)}"


def compute_attribution(
    trades_A: list[dict],
    trades_B: list[dict],
    close_mat: np.ndarray,
    sym_to_i: dict,
    n_dates: int,
    date_s: str, date_e: str,
) -> dict:
    """Compare A vs B within [date_s, date_e]. Returns attribution metrics."""
    sells_A = {build_trade_key(t): t
               for t in trades_A
               if t["side"] == "SELL" and date_s <= t["date"] <= date_e}
    sells_B = {build_trade_key(t): t
               for t in trades_B
               if t["side"] == "SELL" and date_s <= t["date"] <= date_e}

    # Also track BUY entries
    buys_A = {build_trade_key(t): t
              for t in trades_A
              if t["side"] == "BUY" and date_s <= t["date"] <= date_e}
    buys_B = {build_trade_key(t): t
              for t in trades_B
              if t["side"] == "BUY" and date_s <= t["date"] <= date_e}

    # "removed" = BUY entries in A not present in B (blocked by filter)
    removed_keys = set(buys_A.keys()) - set(buys_B.keys())
    removed      = [buys_A[k] for k in removed_keys]

    n_removed     = len(removed)
    if n_removed == 0:
        return {
            "n_removed": 0, "loss_removed": 0, "profit_removed": 0,
            "bad_trade_removed": float("nan"), "opportunity_loss": float("nan"),
            "selection_gain": float("nan"), "winner_removed_rate": float("nan"),
            "blocked_ret20": float("nan"), "accepted_ret20": float("nan"),
            "counterfactual_ret20": float("nan"),
        }

    # Forward returns for removed trades (from entry open, post-hoc)
    def fwd20(t: dict) -> float:
        sym  = t["symbol"]; si = sym_to_i.get(sym)
        ei   = t.get("entry_idx") or t.get("sig_date_idx")
        ep   = t.get("entry", 0)
        if si is None or ei is None or ep <= 0: return float("nan")
        fi = min(ei + FWD_DAYS, n_dates - 1)
        fp = float(close_mat[fi, si])
        return (fp / ep - 1) * 100 if fp > 0 else float("nan")

    removed_fwd = [fwd20(t) for t in removed]
    removed_fwd_clean = [v for v in removed_fwd if not math.isnan(v)]

    # Classify removed trades: loss vs profit (using forward return as proxy)
    loss_removed   = sum(1 for v in removed_fwd_clean if v < 0)
    profit_removed = sum(1 for v in removed_fwd_clean if v >= 0)
    n_clean        = max(1, len(removed_fwd_clean))

    bad_trade_removed = loss_removed   / n_clean
    opportunity_loss  = profit_removed / n_clean

    # PF comparison for selection_gain
    def pf(trades):
        sells = [t for t in trades if t["side"] == "SELL"]
        gp = sum(t["pnl"] for t in sells if t.get("pnl", 0) > 0)
        gl = abs(sum(t["pnl"] for t in sells if t.get("pnl", 0) <= 0))
        return gp / max(1.0, gl)
    pf_A = pf([t for t in trades_A if date_s <= t.get("date","") <= date_e])
    pf_B = pf([t for t in trades_B if date_s <= t.get("date","") <= date_e])
    selection_gain = pf_B / max(0.001, pf_A) if pf_A > 0 else float("nan")

    # Winner removed rate: of top-20 profits in A, how many were removed?
    sells_A_sorted = sorted(
        [t for t in trades_A
         if t["side"] == "SELL" and date_s <= t.get("date","") <= date_e
         and t.get("pnl",0) > 0],
        key=lambda x: -x.get("pnl", 0),
    )
    top20_keys = {build_trade_key(t) for t in sells_A_sorted[:20]}
    top20_removed = sum(1 for k in top20_keys if k in removed_keys)
    winner_removed_rate = top20_removed / max(1, len(top20_keys))

    # Distribution shift
    accepted_keys  = set(buys_A.keys()) & set(buys_B.keys())
    accepted_fwd   = [fwd20(buys_A[k]) for k in accepted_keys]
    accepted_fwd_c = [v for v in accepted_fwd if not math.isnan(v)]
    counterfactual = removed_fwd_clean  # if we hadn't removed these

    return {
        "n_removed":          n_removed,
        "loss_removed":       loss_removed,
        "profit_removed":     profit_removed,
        "bad_trade_removed":  round(bad_trade_removed, 4),
        "opportunity_loss":   round(opportunity_loss, 4),
        "selection_gain":     round(selection_gain, 4) if not math.isnan(selection_gain) else float("nan"),
        "winner_removed_rate":round(winner_removed_rate, 4),
        "blocked_ret20":      round(float(np.mean(removed_fwd_clean)), 2) if removed_fwd_clean else float("nan"),
        "accepted_ret20":     round(float(np.mean(accepted_fwd_c)), 2) if accepted_fwd_c else float("nan"),
        "counterfactual_ret20": round(float(np.mean(counterfactual)), 2) if counterfactual else float("nan"),
        "removed_symbols":    [t["symbol"] for t in removed],
        "removed_slopes":     [t.get("slope5", 0) for t in removed],
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    result_A: dict, result_X: dict,
    trades_B: list[dict],
    common_dates, fold: dict, capital: float,
    sym_to_i: dict, close_mat: np.ndarray, n_dates: int,
) -> dict:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    date_strs  = [str(d.date()) for d in common_dates]
    oos_idx    = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not oos_idx: return {}
    si, ei    = oos_idx[0], oos_idx[-1] + 1
    oos_dates = list(common_dates[si:ei])
    n_days    = len(oos_dates); n_years = n_days / 252.0

    def period_metrics(result):
        eq  = result["eq"][si:ei].tolist()
        inv = result["inv"][si:ei].tolist()
        if len(eq) < 5: return {}
        t   = [t for t in result["trades"] if ds_s <= (t.get("date","")) <= ds_e]
        exp = [vi / max(1.0, ve) for vi, ve in zip(inv, eq)]
        return calc_metrics(eq, t, exp, eq[0], oos_dates)

    mA = period_metrics(result_A)
    mX = period_metrics(result_X)
    if not mA or not mX: return {}

    sells_A = [t for t in result_A["trades"]
               if t["side"]=="SELL" and ds_s <= t.get("date","") <= ds_e]
    sells_X = [t for t in result_X["trades"]
               if t["side"]=="SELL" and ds_s <= t.get("date","") <= ds_e]

    d_cagr   = mX.get("cagr", 0)    - mA.get("cagr", 0)
    d_dd     = abs(mX.get("max_dd", 0)) - abs(mA.get("max_dd", 0))
    d_calmar = mX.get("calmar", 0)  - mA.get("calmar", 0)

    # attribution for this fold
    attr = compute_attribution(
        result_A["trades"], result_X["trades"],
        close_mat, sym_to_i, n_dates, ds_s, ds_e,
    )

    sel_gain = attr.get("selection_gain", float("nan"))
    btr      = attr.get("bad_trade_removed", float("nan"))
    opl      = attr.get("opportunity_loss", float("nan"))
    btr_ok   = (not math.isnan(btr) and not math.isnan(opl) and
                btr >= ADOPT_BTR_RATIO * opl)
    sg_ok    = not math.isnan(sel_gain) and sel_gain >= ADOPT_SEL_GAIN

    fold_pass = (
        d_cagr >= ADOPT_DCAGR and
        d_dd   <= ADOPT_DDD   and
        sg_ok                 and
        btr_ok
    )

    return {
        "cagr_A":    mA.get("cagr", 0),   "cagr_X":  mX.get("cagr", 0),
        "dd_A":      mA.get("max_dd", 0), "dd_X":    mX.get("max_dd", 0),
        "calmar_A":  mA.get("calmar", 0), "calmar_X": mX.get("calmar", 0),
        "pf_A":      mA.get("profit_factor", 0), "pf_X": mX.get("profit_factor", 0),
        "win_A":     mA.get("win_rate", 0), "win_X":  mX.get("win_rate", 0),
        "n_A":       len(sells_A),         "n_X":    len(sells_X),
        "d_cagr":    round(d_cagr, 2),
        "d_dd":      round(d_dd, 2),
        "d_calmar":  round(d_calmar, 3),
        "sel_gain":  round(sel_gain, 4) if not math.isnan(sel_gain) else float("nan"),
        "btr":       round(btr, 4) if not math.isnan(btr) else float("nan"),
        "opl":       round(opl, 4) if not math.isnan(opl) else float("nan"),
        "n_removed": attr.get("n_removed", 0),
        "blocked_ret20": attr.get("blocked_ret20", float("nan")),
        "accepted_ret20":attr.get("accepted_ret20", float("nan")),
        "fold_pass": fold_pass,
    }


def aggregate_wf(fold_results: list[dict]) -> dict:
    valid = [f for f in fold_results if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(k):
        vals = [f[k] for f in valid if not (isinstance(f[k], float) and math.isnan(f[k]))]
        return float(np.mean(vals)) if vals else float("nan")

    sel_vals   = [f["sel_gain"] for f in valid if not math.isnan(f.get("sel_gain", float("nan")))]
    btr_vals   = [f["btr"]      for f in valid if not math.isnan(f.get("btr", float("nan")))]
    opl_vals   = [f["opl"]      for f in valid if not math.isnan(f.get("opl", float("nan")))]
    avg_btr    = float(np.mean(btr_vals))   if btr_vals else float("nan")
    avg_opl    = float(np.mean(opl_vals))   if opl_vals else float("nan")
    avg_sel    = float(np.mean(sel_vals))   if sel_vals else float("nan")
    btr_ok     = not math.isnan(avg_btr) and not math.isnan(avg_opl) and avg_btr >= ADOPT_BTR_RATIO * avg_opl
    sg_ok      = not math.isnan(avg_sel) and avg_sel >= ADOPT_SEL_GAIN

    promote = (
        n_pass >= ADOPT_WF and
        avg("d_cagr") >= ADOPT_DCAGR and
        avg("d_dd") <= ADOPT_DDD and
        sg_ok and btr_ok
    )
    return {
        "n_pass":      n_pass,
        "avg_dcagr":   round(avg("d_cagr"), 2),
        "avg_ddd":     round(avg("d_dd"), 2),
        "avg_dcalmar": round(avg("d_calmar"), 3),
        "avg_pf_A":    round(avg("pf_A"), 3),
        "avg_pf_X":    round(avg("pf_X"), 3),
        "avg_win_A":   round(avg("win_A"), 1),
        "avg_win_X":   round(avg("win_X"), 1),
        "avg_sel_gain":round(avg_sel, 4),
        "avg_btr":     round(avg_btr, 4),
        "avg_opl":     round(avg_opl, 4),
        "avg_removed": round(avg("n_removed"), 1),
        "avg_blocked_ret20":  round(avg("blocked_ret20"), 2),
        "avg_accepted_ret20": round(avg("accepted_ret20"), 2),
        "promote":     promote,
    }


# ──────────────────────────────────────────────────────────────────────
#  MONITOR-ONLY ANALYSIS (Case D)
# ──────────────────────────────────────────────────────────────────────

def monitor_only_analysis(trades_A: list[dict]) -> dict:
    """Post-process A's trades to find what B would have blocked."""
    buys_A = [t for t in trades_A if t["side"] == "BUY"]
    would_block = [t for t in buys_A if t.get("slope5", 0) > SLOPE_FILTER]
    would_keep  = [t for t in buys_A if t.get("slope5", 0) <= SLOPE_FILTER]

    def sym_mix(trades):
        counts = {}
        for t in trades: counts[t["symbol"]] = counts.get(t["symbol"], 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:5])

    return {
        "n_would_block":    len(would_block),
        "n_would_keep":     len(would_keep),
        "block_rate":       round(len(would_block) / max(1, len(buys_A)), 4),
        "slope_dist_block": {
            "min":  min((t.get("slope5", 0) for t in would_block), default=0),
            "max":  max((t.get("slope5", 0) for t in would_block), default=0),
            "mean": round(float(np.mean([t.get("slope5", 0) for t in would_block])), 2)
                    if would_block else float("nan"),
        },
        "sym_mix_block": sym_mix(would_block),
    }


# ──────────────────────────────────────────────────────────────────────
#  FILTER QUALITY METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_filter_quality(trades_A, trades_B, close_mat, sym_to_i, n_dates) -> dict:
    """Global (full period) filter quality: precision, recall."""
    all_buys_A = [t for t in trades_A if t["side"] == "BUY"]
    all_buys_B = {build_trade_key(t) for t in trades_B if t["side"] == "BUY"}

    def fwd20(t) -> float:
        sym = t["symbol"]; si = sym_to_i.get(sym)
        ei  = t.get("entry_idx") or t.get("sig_date_idx")
        ep  = t.get("entry", 0)
        if si is None or ei is None or ep <= 0: return float("nan")
        fi = min(ei + FWD_DAYS, n_dates - 1)
        fp = float(close_mat[fi, si])
        return (fp / ep - 1) * 100 if fp > 0 else float("nan")

    # Every buy in A: accepted (in B) or removed (not in B)
    accepted, removed = [], []
    for t in all_buys_A:
        ret = fwd20(t)
        if math.isnan(ret): continue
        if build_trade_key(t) in all_buys_B:
            accepted.append(ret)
        else:
            removed.append(ret)

    # precision: among removed, fraction that are losses
    n_rm  = len(removed)
    prec  = sum(1 for r in removed if r < 0) / max(1, n_rm)
    # recall: of all losses in A, fraction removed
    all_rets = accepted + removed
    all_loss = sum(1 for r in all_rets if r < 0)
    rem_loss = sum(1 for r in removed   if r < 0)
    recall   = rem_loss / max(1, all_loss)

    # top-20 winner impact
    all_buys_sorted = sorted(zip(all_rets, [True if build_trade_key(t) in all_buys_B
                                              else False for t in all_buys_A
                                              if not math.isnan(fwd20(t))]),
                             key=lambda x: -x[0])
    top20_in_B = sum(1 for _, in_b in all_buys_sorted[:20] if in_b)
    winner_removed_rate_global = (20 - top20_in_B) / 20.0

    return {
        "filter_precision":      round(prec, 4),
        "filter_recall":         round(recall, 4),
        "n_total_buys_A":        len(all_buys_A),
        "n_removed_global":      n_rm,
        "winner_removed_rate":   round(winner_removed_rate_global, 4),
        "blocked_ret20_global":  round(float(np.mean(removed)), 2) if removed else float("nan"),
        "accepted_ret20_global": round(float(np.mean(accepted)), 2) if accepted else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    filt_qual: dict,
    mon_D: dict,
    best_filter: str,
    topix_close,
    path: Path,
) -> None:
    L = []; w = L.append

    prec  = filt_qual.get("filter_precision", float("nan"))
    rec   = filt_qual.get("filter_recall", float("nan"))
    wrr   = filt_qual.get("winner_removed_rate", float("nan"))
    caution = not math.isnan(wrr) and wrr > CAUTION_WRR

    wf_B  = case_results["B"]["wf"]
    wf_C  = case_results["C"]["wf"]
    prom_B = wf_B.get("promote", False)
    prom_C = wf_C.get("promote", False)
    rec_prom = "PROMOTE" if (prom_B or prom_C) else "REJECT"
    rec_live = "PASS" if (prom_B or prom_C) and not caution else "FAIL"

    def ff(v, fmt="+.2f"):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
        return format(v, fmt)
    def fp2(v): return ff(v, ".2f")
    def fp3(v): return ff(v, ".3f")
    def pct(v): return (ff(v, "+.2f") + "%") if not (v is None or (isinstance(v, float) and math.isnan(v))) else "—"

    w("# Entry Velocity WF Confirmation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n仮説: 高速度エントリーが低品質 trade を混入させ DD と CAP 制約を生んでいた")
    w(f"\n採用条件: WF>={ADOPT_WF}/5 AND ΔCAGR≥+{ADOPT_DCAGR}pp AND ΔDD≤+{ADOPT_DDD}pp")
    w(f"         AND selection_gain≥{ADOPT_SEL_GAIN:.2f} AND bad_trade_removed≥{ADOPT_BTR_RATIO:.1f}×opportunity_loss")
    w(f"\nCAUTION: winner_removed_rate>{CAUTION_WRR:.0%}\n")
    w(f"**recommend_promotion: {rec_prom}**  |  "
      f"**recommend_live_shadow: {rec_live}**  |  "
      f"best_filter: **{best_filter}**\n")

    # ── S1. Final Metrics ─────────────────────────────────────────────
    w("---\n## 1. Filter Quality\n")
    w("| 指標 | 値 | 解釈 |")
    w("|---|---|---|")
    w(f"| filter_precision | {ff(prec,'.4f')} | removed trades のうち損失の割合 |")
    w(f"| filter_recall    | {ff(rec,'.4f')}  | 全損失 trades のうち捕捉した割合 |")
    w(f"| winner_removed_rate | {ff(wrr,'.2%')} | top20利益 trade のうち除外率 |")
    w(f"| recommend_promotion | **{rec_prom}** | PROMOTE基準 |")
    w(f"| recommend_live_shadow | **{rec_live}** | CAUTION考慮後 |")
    if caution:
        w(f"\n⚠ CAUTION: winner_removed_rate={wrr:.1%} > {CAUTION_WRR:.0%} "
          f"→ 高パフォーマンス trade を除外している可能性\n")

    # ── S2. WF Summary ────────────────────────────────────────────────
    w("\n---\n## 2. WF サマリ (Case A vs B vs C)\n")
    w("| Case | WF | ΔCAGR | ΔDD | ΔCalmar | avg_PF_X | sel_gain | BTR | OPL | PROMOTE |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in ["B", "C"]:
        wf   = case_results[c]["wf"]
        mark = "**PROMOTE**" if wf.get("promote") else "REJECT"
        btr  = wf.get("avg_btr", float("nan"))
        opl  = wf.get("avg_opl", float("nan"))
        w(f"| {c} ({CASES[c].label}) "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {ff(wf.get('avg_dcagr'))}pp "
          f"| {ff(wf.get('avg_ddd'))}pp "
          f"| {ff(wf.get('avg_dcalmar'),'+.3f')} "
          f"| {fp3(wf.get('avg_pf_X'))} "
          f"| {ff(wf.get('avg_sel_gain'),'.4f')} "
          f"| {ff(btr,'.4f')} "
          f"| {ff(opl,'.4f')} "
          f"| {mark} |")

    # ── S3. Fold Detail (Case B) ──────────────────────────────────────
    w("\n---\n## 3. Fold 詳細 (Case B: GLOBAL_FILTER)\n")
    w("| Fold | Regime | CAGR_A | CAGR_B | ΔCAGR | ΔDD | sel_gain | BTR | n_removed | blocked_ret20 | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for fold, fm in zip(FOLDS, case_results["B"]["folds"]):
        if not fm: continue
        yr  = int(fold["oos_start"][:4])
        reg = regime_label_for(topix_close, yr)
        ok  = "✅" if fm.get("fold_pass") else "❌"
        w(f"| Fold{fold['id']} | {reg} "
          f"| {fm['cagr_A']:+.2f}% "
          f"| {fm['cagr_X']:+.2f}% "
          f"| {fm['d_cagr']:+.2f}pp "
          f"| {fm['d_dd']:+.2f}pp "
          f"| {ff(fm.get('sel_gain'),'.4f')} "
          f"| {ff(fm.get('btr'),'.4f')} "
          f"| {fm.get('n_removed',0)} "
          f"| {pct(fm.get('blocked_ret20'))} "
          f"| {ok} |")

    # ── S4. Fold Detail (Case C) ──────────────────────────────────────
    w("\n---\n## 4. Fold 詳細 (Case C: SOFT_PENALTY λ={:.1f})\n".format(LAMBDA))
    w("| Fold | Regime | CAGR_A | CAGR_C | ΔCAGR | ΔDD | sel_gain | BTR | n_removed | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for fold, fm in zip(FOLDS, case_results["C"]["folds"]):
        if not fm: continue
        yr  = int(fold["oos_start"][:4])
        reg = regime_label_for(topix_close, yr)
        ok  = "✅" if fm.get("fold_pass") else "❌"
        w(f"| Fold{fold['id']} | {reg} "
          f"| {fm['cagr_A']:+.2f}% "
          f"| {fm['cagr_X']:+.2f}% "
          f"| {fm['d_cagr']:+.2f}pp "
          f"| {fm['d_dd']:+.2f}pp "
          f"| {ff(fm.get('sel_gain'),'.4f')} "
          f"| {ff(fm.get('btr'),'.4f')} "
          f"| {fm.get('n_removed',0)} "
          f"| {ok} |")

    # ── S5. Attribution Detail ────────────────────────────────────────
    w("\n---\n## 5. Attribution 詳細\n")
    w("> blocked_ret20: 除外された entry の fwd20 return (損失なら仮説支持)\n")
    w("| Case | blocked_ret20 | accepted_ret20 | bad_trade_removed | opportunity_loss | BTR/OPL ratio |")
    w("|---|---|---|---|---|---|")
    for c in ["B", "C"]:
        wf  = case_results[c]["wf"]
        btr = wf.get("avg_btr", float("nan"))
        opl = wf.get("avg_opl", float("nan"))
        ratio = round(btr / max(0.001, opl), 2) if not (math.isnan(btr) or math.isnan(opl)) else float("nan")
        w(f"| {c} "
          f"| {pct(wf.get('avg_blocked_ret20'))} "
          f"| {pct(wf.get('avg_accepted_ret20'))} "
          f"| {ff(btr,'.4f')} "
          f"| {ff(opl,'.4f')} "
          f"| {ff(ratio,'.2f')} |")

    # Hypothesis check
    w("")
    blk_B = case_results["B"]["wf"].get("avg_blocked_ret20", float("nan"))
    acc_B = case_results["B"]["wf"].get("avg_accepted_ret20", float("nan"))
    if not (math.isnan(blk_B) or math.isnan(acc_B)):
        gap = acc_B - blk_B
        if gap > 2.0:
            w(f"**仮説支持**: accepted_ret20({acc_B:+.2f}%) > blocked_ret20({blk_B:+.2f}%)  差={gap:+.2f}pp\n")
            w("→ slope>5 entry は slope≤5 entry より fwd return が低い = loss removal 確認\n")
        elif gap > 0:
            w(f"→ 弱支持: accepted vs blocked gap={gap:+.2f}pp (軽微)\n")
        else:
            w(f"→ 仮説否定: blocked entry の方が return が高い\n")

    # ── S6. Monitor-Only (D) ──────────────────────────────────────────
    w("\n---\n## 6. Case D: MONITOR_ONLY 仮想監査\n")
    w(f"| 指標 | 値 |")
    w(f"|---|---|")
    w(f"| would_block count | {mon_D.get('n_would_block',0)} |")
    w(f"| would_keep count  | {mon_D.get('n_would_keep',0)} |")
    w(f"| block_rate        | {mon_D.get('block_rate',0):.1%} |")
    sd = mon_D.get("slope_dist_block", {})
    w(f"| blocked slope min/mean/max | {sd.get('min',0):.1f} / {sd.get('mean',0):.1f} / {sd.get('max',0):.1f} |")
    sym_mix = mon_D.get("sym_mix_block", {})
    w(f"| 仮想除外シンボル top5 | {', '.join(f'{s}({n})' for s,n in sym_mix.items())} |")

    # ── S7. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 7. Final Recommendation\n")
    w(f"| 指標 | 値 |")
    w(f"|---|---|")
    w(f"| filter_precision | {ff(prec,'.4f')} ({prec:.1%}) |")
    w(f"| filter_recall    | {ff(rec,'.4f')} ({rec:.1%}) |")
    w(f"| best_filter      | **{best_filter}** |")
    w(f"| recommend_promotion | **{rec_prom}** |")
    w(f"| recommend_live_shadow | **{rec_live}** |")
    if caution:
        w(f"| CAUTION | winner_removed_rate={wrr:.1%} > {CAUTION_WRR:.0%} |")
    w("")

    if rec_prom == "PROMOTE":
        best_wf = case_results[best_filter]["wf"]
        w(f"### PROMOTE — {CASES[best_filter].label}\n")
        w(f"仮説確認: 高速度エントリー除外により entry quality が改善。")
        w(f"改善源は alpha 強化でなく **loss removal**。\n")
        w(f"- ΔCAGR: {best_wf.get('avg_dcagr',0):+.2f}pp  "
          f"ΔDD: {best_wf.get('avg_ddd',0):+.2f}pp  "
          f"WF: {best_wf.get('n_pass',0)}/5")
        w(f"- selection_gain: {ff(best_wf.get('avg_sel_gain'),'.4f')}  "
          f"BTR: {ff(best_wf.get('avg_btr'),'.4f')}")
        w(f"- filter_precision: {prec:.1%}  filter_recall: {rec:.1%}\n")
        if rec_live == "PASS":
            w("**実装 (ASK_FIRST)**: signal_bridge.py に `rsr_slope_5d <= 5` 追加")
            w("→ live shadow 30日 → production 移行")
        else:
            w(f"⚠ CAUTION のため live shadow 延長推奨 (winner_removed_rate={wrr:.1%})")
    else:
        w(f"### REJECT\n")
        fails = []
        best_wf = case_results["B"]["wf"]
        if best_wf.get("n_pass", 0) < ADOPT_WF:
            fails.append(f"WF={best_wf['n_pass']}/5 < {ADOPT_WF}")
        if best_wf.get("avg_dcagr", 0) < ADOPT_DCAGR:
            fails.append(f"ΔCAGR={best_wf.get('avg_dcagr',0):+.2f}pp < +{ADOPT_DCAGR}pp")
        if best_wf.get("avg_ddd", 0) > ADOPT_DDD:
            fails.append(f"ΔDD={best_wf.get('avg_ddd',0):+.2f}pp > +{ADOPT_DDD}pp")
        if not math.isnan(best_wf.get("avg_sel_gain", float("nan"))):
            if best_wf.get("avg_sel_gain", 0) < ADOPT_SEL_GAIN:
                fails.append(f"sel_gain={best_wf.get('avg_sel_gain',0):.4f} < {ADOPT_SEL_GAIN:.2f}")
        w("バインディング制約: " + " / ".join(fails) if fails else "全基準未達")
        w(f"\n保守推奨: slope≤5 フィルターを採用済み研究 (standalone) で維持")
        w("production 統合は再評価後")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Entry Velocity WF Confirmation")
    print(f"  slope_filter={SLOPE_FILTER:.0f}  lambda={LAMBDA:.1f}  FWD={FWD_DAYS}d")
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
        rule  = SECTOR_STRATEGY.get(trade_syms.get(sym, ""), "fujiko")
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

    slope5_mat = compute_slope5(rsr_mat)
    capital    = float(cfg.portfolio.capital)

    print("[3/4] Portfolio シミュレーション (4 cases)...\n")
    case_results: dict = {}
    result_A: dict | None = None

    for c in CASE_ORDER:
        spec = CASES[c]
        if spec.monitor_only:
            # D: re-use A, post-process
            result_D = result_A  # same as A
            mon_D    = monitor_only_analysis(result_A["trades"])
            folds_D  = []
            for fold in FOLDS:
                fm = compute_fold_metrics(result_A, result_A, [],
                                          common_dates, fold, capital,
                                          sym_to_i, close_mat, n_dates)
                folds_D.append(fm)
            wf_D = aggregate_wf(folds_D)
            case_results["D"] = {"folds": folds_D, "wf": wf_D,
                                  "monitor": mon_D, "trades": result_A["trades"]}
            print(f"  [D] MONITOR_ONLY — {mon_D.get('n_would_block',0)} would-block  "
                  f"({mon_D.get('block_rate',0):.1%} of all entries)")
            continue

        print(f"  [{c}] {spec.label}")
        res = run_portfolio(
            open_mat, close_mat,
            sig_mat, sig_ready, rsr_mat, sym_active_mat,
            mkt_ret1, topix_ret20, topix_ret60, bear_arr,
            slope5_mat, active_syms, sym_to_i, trade_syms, cfg, common_dates,
            spec,
        )
        if c == "A":
            result_A = res

        folds_m = []
        for fold in FOLDS:
            fm = compute_fold_metrics(
                result_A, res, result_A["trades"] if c == "A" else [],
                common_dates, fold, capital, sym_to_i, close_mat, n_dates,
            )
            folds_m.append(fm)

        wf = aggregate_wf(folds_m)
        case_results[c] = {"folds": folds_m, "wf": wf, "trades": res["trades"]}

        def _fmt(v, fmt="+.2f"):
            if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
            return format(v, fmt)
        for fold, fm in zip(FOLDS, folds_m):
            if fm and c != "A":
                ok = "✅" if fm.get("fold_pass") else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"ΔCAGR={fm['d_cagr']:+.2f}pp  ΔDD={fm['d_dd']:+.2f}pp  "
                      f"sel={_fmt(fm.get('sel_gain'),'.4f')}  "
                      f"BTR={_fmt(fm.get('btr'),'.4f')}  "
                      f"n_rm={fm.get('n_removed',0)}  {ok}")
        if c != "A":
            ok_s = "✅ PROMOTE" if wf.get("promote") else "❌ REJECT"
            print(f"    → WF={wf.get('n_pass',0)}/5  "
                  f"ΔCAGR={wf.get('avg_dcagr',0):+.2f}pp  "
                  f"ΔDD={wf.get('avg_ddd',0):+.2f}pp  "
                  f"sel_gain={_fmt(wf.get('avg_sel_gain'),'.4f')}  {ok_s}\n")

    print("[4/4] Filter quality 計算...")
    filt_qual = compute_filter_quality(
        result_A["trades"], case_results["B"]["trades"],
        close_mat, sym_to_i, n_dates,
    )
    # D monitor
    mon_D = case_results.get("D", {}).get("monitor", {})

    # best_filter
    prom_B = case_results["B"]["wf"].get("promote", False)
    prom_C = case_results["C"]["wf"].get("promote", False)
    if prom_B and prom_C:
        best_filter = ("B" if case_results["B"]["wf"].get("avg_dcagr", 0) >=
                              case_results["C"]["wf"].get("avg_dcagr", 0) else "C")
    elif prom_B:
        best_filter = "B"
    elif prom_C:
        best_filter = "C"
    else:
        best_filter = ("B" if case_results["B"]["wf"].get("avg_dcagr", 0) >=
                              case_results["C"]["wf"].get("avg_dcagr", 0) else "C")

    prec = filt_qual.get("filter_precision", float("nan"))
    rec  = filt_qual.get("filter_recall", float("nan"))
    wrr  = filt_qual.get("winner_removed_rate", float("nan"))
    prom_B = case_results["B"]["wf"].get("promote", False)
    prom_C = case_results["C"]["wf"].get("promote", False)
    rec_prom = "PROMOTE" if (prom_B or prom_C) else "REJECT"
    rec_live = "PASS" if (prom_B or prom_C) and not (not math.isnan(wrr) and wrr > CAUTION_WRR) else "FAIL"

    def _fmtg(v, fmt=".4f"):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
        return format(v, fmt)
    print(f"\n  filter_precision:      {_fmtg(prec)}")
    print(f"  filter_recall:         {_fmtg(rec)}")
    print(f"  winner_removed_rate:   {_fmtg(wrr,'.2%')}")
    print(f"  best_filter:           {best_filter}")
    print(f"  recommend_promotion:   {rec_prom}")
    print(f"  recommend_live_shadow: {rec_live}")
    print("=" * 68 + "\n")

    write_report(
        case_results, filt_qual, mon_D, best_filter,
        topix_close,
        REPORTS_DIR / "entry_velocity_wf_confirmation.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
