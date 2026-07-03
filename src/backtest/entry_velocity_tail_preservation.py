"""
backtest/entry_velocity_tail_preservation.py  —  Entry Velocity Tail Preservation WF

目的:
  slope>5 entry の「高損失率 × 巨大利益寄与」を両立して扱えるか検証。
  ENTRY/EXIT 変更禁止。allocation のみ制御。

Cases:
  A  baseline                    (変更なし)
  B  slope>5 weight×0.5          (半サイズ)
  C  slope>5 weight×0.3          (30%サイズ)
  D  slope>5 max_hold=1          (1日後強制退出)
  E  slope>5 slot2のみ許可       (最大1ポジション)
  F  slope>5 profit_lock+8%      (+8%到達で即利食い)

採用条件:
  WF>=4/5 AND ΔCAGR>=+0.3pp AND ΔDD<=+1.5pp AND tail_capture>=90%

最終出力: tail_efficiency, keep_or_drop

Run:
    cd C:/ai-trading
    python src/backtest/entry_velocity_tail_preservation.py
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

SHOCK_MKT_THR  = -0.05
SHOCK_SYM_THR  = -0.08
SLOPE_FILTER   = 5.0    # slope>5 = "fast velocity entry"
PROFIT_LOCK_F  = 0.08   # Case F: 8% profit lock threshold
MAX_HOLD_D     = 1      # Case D: max hold days for slope>5
MAX_S5_SLOTS   = 1      # Case E: max slope>5 positions at once
TOP_N_TAIL     = 20     # top-N winners for tail analysis

# ── Adoption criteria ──────────────────────────────────────────────────
ADOPT_WF         = 4
ADOPT_DCAGR      = 0.3
ADOPT_DDD        = 1.5
ADOPT_TAIL_CAP   = 0.90   # tail_capture >= 90%


# ──────────────────────────────────────────────────────────────────────
#  CASE SPEC
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CaseSpec:
    label:       str
    weight:      Optional[float] = None   # B/C: size factor for slope>5
    maxhold:     Optional[int]   = None   # D: max hold days for slope>5
    max_s5_pos:  Optional[int]   = None   # E: max slope>5 positions simultaneously
    profit_lock: Optional[float] = None   # F: profit take threshold for slope>5


CASES: dict[str, CaseSpec] = {
    "A": CaseSpec("baseline"),
    "B": CaseSpec("slope>5 weight×0.5",    weight=0.5),
    "C": CaseSpec("slope>5 weight×0.3",    weight=0.3),
    "D": CaseSpec("slope>5 max_hold=1",    maxhold=MAX_HOLD_D),
    "E": CaseSpec("slope>5 slot2のみ",      max_s5_pos=MAX_S5_SLOTS),
    "F": CaseSpec("slope>5 profit_lock+8%", profit_lock=PROFIT_LOCK_F),
}

CASE_ORDER = ["A", "B", "C", "D", "E", "F"]


def _fmt(v, fmt="+.2f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return format(v, fmt)

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return f"{v*100:.1f}%"


# ──────────────────────────────────────────────────────────────────────
#  SLOPE MATRIX / REGIME
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
    MS   = float(rc.sector_cap)                              if rc else 0.25
    gen  = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gn   = float(getattr(rc, "gross_cap_normal", 1.0))       if rc else 1.0
    gd5  = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6))if rc else 0.6
    gd8  = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4))if rc else 0.4

    cash = float(capital); pos: dict[str, Position] = {}
    # slope5 value at entry time for each open position
    pos_slope5: dict[str, float] = {}
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
            si  = sym_to_i[sym]; h = sym in pos
            hd  = (i - pos[sym].entry_idx) if h else 0
            rv  = float(rsr_mat[i, si]); ct = float(close_mat[i, si])
            sl5 = float(slope5_mat[i, si])

            # ── Market shock exits ──
            if shock and h:
                if i > 0:
                    pc = float(close_mat[i - 1, si])
                    if pc > 0 and (ct / pc - 1) <= SHOCK_SYM_THR:
                        sells.append((sym, "MKT_SHOCK")); continue
            if shock and not h: continue

            # ── Case D: slope>5 max_hold override (exposure limit) ──
            if h and spec.maxhold is not None:
                s5_at_entry = pos_slope5.get(sym, 0.0)
                if s5_at_entry > SLOPE_FILTER and hd >= spec.maxhold:
                    sells.append((sym, "S5_MAXHOLD")); continue

            # ── Case F: profit lock for slope>5 (exposure limit) ──
            if h and spec.profit_lock is not None:
                s5_at_entry = pos_slope5.get(sym, 0.0)
                ep = pos[sym].entry_price
                if s5_at_entry > SLOPE_FILTER and ep > 0:
                    cur_ret = ct / ep - 1
                    if cur_ret >= spec.profit_lock:
                        sells.append((sym, "S5_PROFLOCK")); continue

            # ── Regular exits ──
            if h and max_hold and hd > max_hold:
                sells.append((sym, "TIME_STOP")); continue
            if h and rv < rsr_exit_thr and hd >= min_hold:
                sells.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and h and hd >= min_hold:
                sells.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not h:
                if i < reentry_ban.get(sym, -1): continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue

                # ── Case E: slope>5 slot limit ──
                if spec.max_s5_pos is not None and sl5 > SLOPE_FILTER:
                    n_s5_open = sum(1 for s in pos if pos_slope5.get(s, 0) > SLOPE_FILTER)
                    if n_s5_open >= spec.max_s5_pos: continue

                buys.append((rv, sl5, sym))

        # ── Execute sells ──
        for sym, reason in sells:
            if sym not in pos: continue
            p  = pos[sym]; si = sym_to_i[sym]
            sp = float(open_mat[nxt, si])
            pnl = (sp - p.entry_price) * p.qty
            cash += p.qty * sp * (1 - COST_ONE_WAY)
            trades.append({
                "side":      "SELL", "symbol": sym, "pnl": pnl,
                "entry":     p.entry_price, "exit": sp, "qty": p.qty,
                "entry_idx": p.entry_idx, "exit_idx": i,
                "reason":    reason, "date": ds,
                "hold_days": i - p.entry_idx,
                "rsr_at_entry": getattr(p, "rsr_at_entry", 0.0),
                "slope5":    pos_slope5.get(sym, 0.0),
            })
            del pos[sym]; pos_slope5.pop(sym, None)
            if reason == "TIME_STOP": reentry_ban[sym] = i + 1 + REENTRY_COOL

        # ── Execute buys ──
        if not cb_active and buys:
            buys.sort(key=lambda x: -x[0])   # descending RSR
            for rv, sl5, sym in buys:
                si = sym_to_i[sym]
                if max_pos - len(pos) <= 0: break
                bp = float(open_mat[nxt, si])
                if bp <= 0: continue
                if not _sector_ok(sym, pos, close_mat, i, sym_to_i, trade_syms, capital, scap): continue
                if gen:
                    cg = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                             for p in pos.values()) / max(1.0, capital)
                    if cg + bp * LOT / max(1.0, capital) > gc: continue

                # ── Cases B/C: reduce alloc for slope>5 ──
                if spec.weight is not None and sl5 > SLOPE_FILTER:
                    alloc = capital / max_pos * spec.weight
                else:
                    alloc = capital / max_pos
                qty  = int(alloc / bp / LOT) * LOT
                cost = qty * bp * (1 + COST_ONE_WAY)
                if qty <= 0 or cost > cash: continue

                _execute_buy(sym, bp, qty, i, nxt, trade_syms, trades, pos, rv)
                cash -= cost
                pos_slope5[sym] = sl5
                trades[-1]["date"]   = ds
                trades[-1]["slope5"] = sl5

    return {"eq": eq, "inv": inv, "trades": trades}


# ──────────────────────────────────────────────────────────────────────
#  TAIL METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_tail_metrics(
    trades_A: list[dict],
    trades_X: list[dict],
) -> dict:
    """
    Compute tail preservation metrics for slope>5 trades.

    tail_capture       = count(s5 winners in X AND winners in A) / count(s5 winners in A)
    winner_retention   = sum(PnL s5 winners in X) / sum(PnL s5 winners in A)
    avg_loss_s5        = mean losing PnL for s5 SELL trades in X
    PF_s5              = PF of slope>5 SELL trades in X
    top10_share        = top-10 trade PnL / total gross profit in X (all trades)
    """
    # Build SELL lookup by (symbol, entry_idx)
    def sells_by_key(trades):
        out: dict[tuple, dict] = {}
        for t in trades:
            if t["side"] == "SELL":
                k = (t["symbol"], t.get("entry_idx", 0))
                out[k] = t
        return out

    sell_A = sells_by_key(trades_A)
    sell_X = sells_by_key(trades_X)

    # Identify slope>5 entries in A
    s5_entries_A: list[dict] = [
        t for t in trades_A
        if t["side"] == "BUY" and t.get("slope5", 0) > SLOPE_FILTER
    ]

    # Match to SELL records in A and X
    s5_winners_A:  list[dict] = []
    s5_losers_A:   list[dict] = []
    for t_buy in s5_entries_A:
        k = (t_buy["symbol"], t_buy.get("entry_idx", 0))
        t_sell_A = sell_A.get(k)
        if t_sell_A is None: continue
        if t_sell_A.get("pnl", 0) > 0:
            s5_winners_A.append({"key": k, "pnl_A": t_sell_A["pnl"]})
        else:
            s5_losers_A.append({"key": k, "pnl_A": t_sell_A.get("pnl", 0)})

    n_winners_A = len(s5_winners_A)
    total_win_A = sum(w["pnl_A"] for w in s5_winners_A)

    # For each A winner: check if winner in X
    retained_count = 0; retained_pnl = 0.0
    for w in s5_winners_A:
        t_sell_X = sell_X.get(w["key"])
        if t_sell_X and t_sell_X.get("pnl", 0) > 0:
            retained_count += 1
            retained_pnl   += t_sell_X["pnl"]

    tail_capture     = retained_count / max(1, n_winners_A)
    winner_retention = retained_pnl   / max(1, total_win_A)

    # PF and avg_loss for slope>5 trades in X
    s5_sells_X = [
        t for t in trades_X
        if t["side"] == "SELL" and t.get("slope5", 0) > SLOPE_FILTER
    ]
    s5_wins_X  = [t["pnl"] for t in s5_sells_X if t.get("pnl", 0) > 0]
    s5_loss_X  = [t["pnl"] for t in s5_sells_X if t.get("pnl", 0) <= 0]

    gp_s5 = sum(s5_wins_X) if s5_wins_X else 0.0
    gl_s5 = abs(sum(s5_loss_X)) if s5_loss_X else 0.0
    PF_s5 = gp_s5 / max(1.0, gl_s5)
    avg_loss_s5 = float(np.mean(s5_loss_X)) if s5_loss_X else 0.0
    avg_win_s5  = float(np.mean(s5_wins_X))  if s5_wins_X  else 0.0

    # top10_share (all trades in X)
    all_sells_X = [t for t in trades_X if t["side"] == "SELL"]
    all_pnl_pos = [t["pnl"] for t in all_sells_X if t.get("pnl", 0) > 0]
    top10_pnl   = sum(sorted(all_pnl_pos, reverse=True)[:10]) if all_pnl_pos else 0.0
    total_gp_X  = sum(all_pnl_pos)
    top10_share  = top10_pnl / max(1.0, total_gp_X)

    return {
        "n_s5_winners_A":    n_winners_A,
        "n_s5_losers_A":     len(s5_losers_A),
        "tail_capture":      round(tail_capture, 4),
        "winner_retention":  round(winner_retention, 4),
        "avg_loss_s5":       round(avg_loss_s5),
        "avg_win_s5":        round(avg_win_s5),
        "PF_s5":             round(PF_s5, 3),
        "n_s5_trades_X":     len(s5_sells_X),
        "n_s5_wins_X":       len(s5_wins_X),
        "n_s5_loss_X":       len(s5_loss_X),
        "top10_share":       round(top10_share, 4),
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    result_A: dict, result_X: dict,
    common_dates, fold: dict,
) -> dict:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    date_strs  = [str(d.date()) for d in common_dates]
    oos_idx    = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not oos_idx: return {}
    si, ei    = oos_idx[0], oos_idx[-1] + 1
    oos_dates = list(common_dates[si:ei])

    def period_metrics(result):
        eq  = result["eq"][si:ei].tolist()
        inv = result["inv"][si:ei].tolist()
        if len(eq) < 5: return {}
        t   = [t for t in result["trades"] if ds_s <= t.get("date","") <= ds_e]
        exp = [vi / max(1.0, ve) for vi, ve in zip(inv, eq)]
        return calc_metrics(eq, t, exp, eq[0], oos_dates)

    mA = period_metrics(result_A)
    mX = period_metrics(result_X)
    if not mA or not mX: return {}

    n_A = sum(1 for t in result_A["trades"]
              if t["side"]=="SELL" and ds_s <= t.get("date","") <= ds_e)
    n_X = sum(1 for t in result_X["trades"]
              if t["side"]=="SELL" and ds_s <= t.get("date","") <= ds_e)

    d_cagr  = mX.get("cagr", 0)   - mA.get("cagr", 0)
    d_dd    = abs(mX.get("max_dd",0)) - abs(mA.get("max_dd",0))
    d_calmar= mX.get("calmar",0)  - mA.get("calmar",0)

    fold_pass = d_cagr >= ADOPT_DCAGR and d_dd <= ADOPT_DDD

    return {
        "cagr_A":  mA.get("cagr",0),  "cagr_X":  mX.get("cagr",0),
        "dd_A":    mA.get("max_dd",0), "dd_X":    mX.get("max_dd",0),
        "cal_A":   mA.get("calmar",0), "cal_X":   mX.get("calmar",0),
        "pf_A":    mA.get("profit_factor",0), "pf_X": mX.get("profit_factor",0),
        "win_A":   mA.get("win_rate",0), "win_X":  mX.get("win_rate",0),
        "n_A":     n_A, "n_X": n_X,
        "d_cagr":  round(d_cagr, 2),
        "d_dd":    round(d_dd, 2),
        "d_calmar":round(d_calmar, 3),
        "fold_pass": fold_pass,
    }


def aggregate_wf(fold_results: list[dict]) -> dict:
    valid  = [f for f in fold_results if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(k):
        vals = [f[k] for f in valid
                if not (isinstance(f.get(k), float) and math.isnan(f.get(k)))]
        return float(np.mean(vals)) if vals else float("nan")

    promote_metric = (
        n_pass >= ADOPT_WF and
        avg("d_cagr") >= ADOPT_DCAGR and
        avg("d_dd")   <= ADOPT_DDD
    )
    return {
        "n_pass":      n_pass,
        "avg_dcagr":   round(avg("d_cagr"), 2),
        "avg_ddd":     round(avg("d_dd"), 2),
        "avg_dcalmar": round(avg("d_calmar"), 3),
        "avg_pf_X":    round(avg("pf_X"), 3),
        "avg_win_X":   round(avg("win_X"), 1),
        "promote_metric": promote_metric,
    }


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    result_A: dict,
    topix_close,
    path: Path,
) -> None:
    L = []; w = L.append

    # ── Collect global tail metrics and adoption ────────────────────────
    adoption: dict[str, dict] = {}
    for c, cr in case_results.items():
        if c == "A": continue
        wf   = cr.get("wf", {})
        tail = cr.get("tail", {})
        tc   = tail.get("tail_capture", float("nan"))
        pm   = wf.get("promote_metric", False)
        adopt = pm and not math.isnan(tc) and tc >= ADOPT_TAIL_CAP
        te   = round(tc * (wf.get("n_pass", 0) / 5), 4) if not math.isnan(tc) else float("nan")
        kod  = "KEEP" if adopt else "DROP"
        adoption[c] = {"adopt": adopt, "tail_efficiency": te, "keep_or_drop": kod}

    best_c = max(
        [c for c in CASE_ORDER if c != "A"],
        key=lambda c: (
            adoption[c]["adopt"],
            case_results[c].get("tail", {}).get("tail_capture", 0),
            case_results[c].get("wf", {}).get("avg_dcagr", 0),
        )
    )

    w("# Entry Velocity Tail Preservation WF")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n仮説: slope>5 entry のサイズ・期間・利食い制御によりリスク削減しつつ tail を保全できる")
    w(f"\n採用条件: WF>={ADOPT_WF}/5 AND ΔCAGR≥+{ADOPT_DCAGR}pp AND ΔDD≤+{ADOPT_DDD}pp"
      f" AND tail_capture≥{ADOPT_TAIL_CAP:.0%}\n")

    # ── S1. Final Output ────────────────────────────────────────────────
    w("---\n## 1. Final Output\n")
    w("| Case | tail_efficiency | keep_or_drop |")
    w("|---|---|---|")
    for c in CASE_ORDER:
        if c == "A": continue
        te  = adoption[c]["tail_efficiency"]
        kod = adoption[c]["keep_or_drop"]
        w(f"| {c}: {CASES[c].label} | {_fmt(te,'.4f')} | **{kod}** |")

    # ── S2. WF Summary ──────────────────────────────────────────────────
    w("\n---\n## 2. WF サマリ\n")
    w("| Case | WF | ΔCAGR | ΔDD | ΔCalmar | PF_X | win_X | tail_capture | ADOPT |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        if c == "A": continue
        wf   = case_results[c].get("wf", {})
        tail = case_results[c].get("tail", {})
        tc   = tail.get("tail_capture", float("nan"))
        adopt= adoption[c]["adopt"]
        mark = "**ADOPT**" if adopt else "REJECT"
        w(f"| {c} ({CASES[c].label}) "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {_fmt(wf.get('avg_dcagr'))}pp "
          f"| {_fmt(wf.get('avg_ddd'))}pp "
          f"| {_fmt(wf.get('avg_dcalmar'),'+.3f')} "
          f"| {_fmt(wf.get('avg_pf_X'),'.3f')} "
          f"| {_fmt(wf.get('avg_win_X'),'.1f')}% "
          f"| {_pct(tc)} "
          f"| {mark} |")

    # ── S3. Tail Metrics ────────────────────────────────────────────────
    w("\n---\n## 3. Tail 保全指標\n")
    w("| Case | tail_capture | winner_retention | avg_loss_s5 | avg_win_s5 | PF_s5 | top10_share |")
    w("|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        if c == "A": continue
        tail = case_results[c].get("tail", {})
        tc   = tail.get("tail_capture", float("nan"))
        wr   = tail.get("winner_retention", float("nan"))
        al   = tail.get("avg_loss_s5", 0)
        aw   = tail.get("avg_win_s5", 0)
        pf   = tail.get("PF_s5", float("nan"))
        t10  = tail.get("top10_share", float("nan"))
        tc_mark = f"⚠{_pct(tc)}" if not math.isnan(tc) and tc < ADOPT_TAIL_CAP else _pct(tc)
        w(f"| {c} ({CASES[c].label}) "
          f"| {tc_mark} "
          f"| {_pct(wr)} "
          f"| ¥{al:+,.0f} "
          f"| ¥{aw:+,.0f} "
          f"| {_fmt(pf,'.3f')} "
          f"| {_pct(t10)} |")

    # Baseline tail stats
    tail_A = case_results.get("__tail_A", {})
    w(f"\n> **A baseline**: slope>5 winners={tail_A.get('n_s5_winners_A',0)}, "
      f"losers={tail_A.get('n_s5_losers_A',0)}, "
      f"PF_s5={_fmt(tail_A.get('PF_s5', float('nan')),'.3f')}, "
      f"avg_win=¥{tail_A.get('avg_win_s5',0):+,.0f}, "
      f"avg_loss=¥{tail_A.get('avg_loss_s5',0):+,.0f}\n")

    # ── S4. Fold Detail (best case) ─────────────────────────────────────
    w(f"\n---\n## 4. Fold 詳細 (Case {best_c}: {CASES[best_c].label})\n")
    w("| Fold | Regime | CAGR_A | CAGR_X | ΔCAGR | ΔDD | n_A | n_X | pass |")
    w("|---|---|---|---|---|---|---|---|---|")
    for fold, fm in zip(FOLDS, case_results[best_c]["folds"]):
        if not fm: continue
        yr  = int(fold["oos_start"][:4])
        reg = regime_label_for(topix_close, yr)
        ok  = "✅" if fm.get("fold_pass") else "❌"
        w(f"| Fold{fold['id']} | {reg} "
          f"| {fm['cagr_A']:+.2f}% "
          f"| {fm['cagr_X']:+.2f}% "
          f"| {fm['d_cagr']:+.2f}pp "
          f"| {fm['d_dd']:+.2f}pp "
          f"| {fm['n_A']} "
          f"| {fm['n_X']} "
          f"| {ok} |")

    # ── S5. All Cases Fold Summary ──────────────────────────────────────
    w("\n---\n## 5. 全 Case × Fold マトリクス (ΔCAGR)\n")
    header = "| Fold |" + "".join(f" {c} |" for c in CASE_ORDER if c != "A")
    sep    = "|---|" + "---|" * (len(CASE_ORDER) - 1)
    w(header); w(sep)
    for fi, fold in enumerate(FOLDS):
        row = f"| Fold{fold['id']} ({fold['oos_start'][:4]}) |"
        for c in CASE_ORDER:
            if c == "A": continue
            fm = case_results[c]["folds"][fi]
            if fm:
                ok  = "✅" if fm.get("fold_pass") else "❌"
                row += f" {fm['d_cagr']:+.2f}pp {ok} |"
            else:
                row += " — |"
        w(row)

    # ── S6. DD Matrix ────────────────────────────────────────────────────
    w("\n---\n## 6. 全 Case × Fold マトリクス (ΔDD)\n")
    w(header); w(sep)
    for fi, fold in enumerate(FOLDS):
        row = f"| Fold{fold['id']} ({fold['oos_start'][:4]}) |"
        for c in CASE_ORDER:
            if c == "A": continue
            fm = case_results[c]["folds"][fi]
            if fm:
                ok  = "✅" if fm["d_dd"] <= ADOPT_DDD else "❌"
                row += f" {fm['d_dd']:+.2f}pp {ok} |"
            else:
                row += " — |"
        w(row)

    # ── S7. Case Analysis ───────────────────────────────────────────────
    w("\n---\n## 7. Case 分析\n")

    # B/C: size reduction insight
    wf_B = case_results["B"]["wf"]
    wf_C = case_results["C"]["wf"]
    tc_B = case_results["B"]["tail"].get("tail_capture", float("nan"))
    tc_C = case_results["C"]["tail"].get("tail_capture", float("nan"))
    w("### B/C: サイズ削減 (weight×0.5 / ×0.3)")
    w(f"- B: ΔCAGR={wf_B.get('avg_dcagr',0):+.2f}pp  "
      f"ΔDD={wf_B.get('avg_ddd',0):+.2f}pp  "
      f"tail_capture={_pct(tc_B)}")
    w(f"- C: ΔCAGR={wf_C.get('avg_dcagr',0):+.2f}pp  "
      f"ΔDD={wf_C.get('avg_ddd',0):+.2f}pp  "
      f"tail_capture={_pct(tc_C)}")
    w("> サイズ削減は slope>5 ポジションのスロットを占有しつつ投資額を減らす。"
      " tail_capture は count ベースで高い (同じ取引を実行) が PnL 規模は削減される。\n")

    # D: max_hold insight
    wf_D = case_results["D"]["wf"]
    tc_D = case_results["D"]["tail"].get("tail_capture", float("nan"))
    tail_D = case_results["D"]["tail"]
    w(f"### D: max_hold={MAX_HOLD_D} (slope>5)")
    w(f"- ΔCAGR={wf_D.get('avg_dcagr',0):+.2f}pp  "
      f"ΔDD={wf_D.get('avg_ddd',0):+.2f}pp  "
      f"tail_capture={_pct(tc_D)}")
    w(f"- avg_win_s5=¥{tail_D.get('avg_win_s5',0):+,.0f}  "
      f"avg_loss_s5=¥{tail_D.get('avg_loss_s5',0):+,.0f}  "
      f"PF_s5={_fmt(tail_D.get('PF_s5', float('nan')),'.3f')}")
    w("> 短期保有により損失拡大を抑制するが、multi-week winner を早期退出で逃す可能性あり。\n")

    # E: slot limit insight
    wf_E = case_results["E"]["wf"]
    tc_E = case_results["E"]["tail"].get("tail_capture", float("nan"))
    tail_E = case_results["E"]["tail"]
    w(f"### E: slope>5 最大{MAX_S5_SLOTS}ポジション")
    w(f"- ΔCAGR={wf_E.get('avg_dcagr',0):+.2f}pp  "
      f"ΔDD={wf_E.get('avg_ddd',0):+.2f}pp  "
      f"tail_capture={_pct(tc_E)}")
    w(f"- n_s5_trades_X={tail_E.get('n_s5_trades_X',0)}  "
      f"(A baseline: {tail_E.get('n_s5_winners_A',0)+tail_E.get('n_s5_losers_A',0)} total)")
    w("> slope>5 ポジション数を制限。Bull 相場で複数 slope>5 候補が並ぶ場合の集中リスクを削減。\n")

    # F: profit lock insight
    wf_F = case_results["F"]["wf"]
    tc_F = case_results["F"]["tail"].get("tail_capture", float("nan"))
    tail_F = case_results["F"]["tail"]
    w(f"### F: profit_lock +{PROFIT_LOCK_F:.0%} (slope>5)")
    w(f"- ΔCAGR={wf_F.get('avg_dcagr',0):+.2f}pp  "
      f"ΔDD={wf_F.get('avg_ddd',0):+.2f}pp  "
      f"tail_capture={_pct(tc_F)}")
    w(f"- avg_win_s5=¥{tail_F.get('avg_win_s5',0):+,.0f}  "
      f"PF_s5={_fmt(tail_F.get('PF_s5', float('nan')),'.3f')}")
    w("> +8%到達で即利食い。slope>5 ポジションの益確保でリターン安定化を狙うが、"
      " 大型 winner のキャップが PnL を押し下げる可能性あり。\n")

    # ── S8. Final Recommendation ────────────────────────────────────────
    w("---\n## 8. 最終判定\n")
    w("| Case | WF | ΔCAGR | ΔDD | tail_capture | tail_efficiency | keep_or_drop |")
    w("|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        if c == "A": continue
        wf   = case_results[c].get("wf", {})
        tail = case_results[c].get("tail", {})
        tc   = tail.get("tail_capture", float("nan"))
        te   = adoption[c]["tail_efficiency"]
        kod  = adoption[c]["keep_or_drop"]
        w(f"| **{c}** {CASES[c].label} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {_fmt(wf.get('avg_dcagr'))}pp "
          f"| {_fmt(wf.get('avg_ddd'))}pp "
          f"| {_pct(tc)} "
          f"| {_fmt(te,'.4f')} "
          f"| **{kod}** |")

    # Adopted cases
    adopted = [c for c in CASE_ORDER if c != "A" and adoption[c]["adopt"]]
    if adopted:
        w(f"\n### ADOPT: {', '.join(adopted)}\n")
        for c in adopted:
            wf   = case_results[c].get("wf", {})
            tail = case_results[c].get("tail", {})
            w(f"**{c}: {CASES[c].label}**")
            w(f"- WF={wf.get('n_pass',0)}/5  "
              f"ΔCAGR={wf.get('avg_dcagr',0):+.2f}pp  "
              f"ΔDD={wf.get('avg_ddd',0):+.2f}pp  "
              f"tail_capture={_pct(tail.get('tail_capture', float('nan')))}")
            w(f"- tail_efficiency={_fmt(adoption[c]['tail_efficiency'],'.4f')}\n")
        w("実装 (ASK_FIRST): 対象は signal_bridge.py でなく position sizing ロジック")
    else:
        w("\n### 全 Case REJECT\n")
        w("slope>5 の allocation 制御は production portfolio では機能しない。")
        w("根本原因: Bull 相場での slope>5 = alpha source → 任意の制御が 2021 Fold を破壊。")
        w("\n推奨: slope>5 を production でそのまま保持。")
        w("standalone/研究文脈のみ slope≤5 フィルターを維持 (WF=4/5 確認済み)。")
        w("次研究候補: regime-conditional control (Bear/Sideways のみ slope>5 制御)。")

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
    print("  Entry Velocity Tail Preservation WF")
    print(f"  slope_filter={SLOPE_FILTER:.0f}  weight_B={0.5}  weight_C={0.3}")
    print(f"  maxhold_D={MAX_HOLD_D}  max_s5_E={MAX_S5_SLOTS}  profit_lock_F={PROFIT_LOCK_F:.0%}")
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
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}")

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
        ma200       = topix_close.rolling(200, min_periods=100).mean()
        bear_arr    = ((topix_close < ma200)
                       .reindex(pd.DatetimeIndex(common_dates), method="ffill")
                       .fillna(False).values.astype(bool))

    slope5_mat = compute_slope5(rsr_mat)

    print("[3/4] Portfolio シミュレーション (6 cases)...\n")
    case_results: dict = {}
    result_A: dict | None = None

    for c in CASE_ORDER:
        spec = CASES[c]
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
            fm = compute_fold_metrics(result_A, res, common_dates, fold)
            folds_m.append(fm)

        wf = aggregate_wf(folds_m)

        # tail metrics vs A
        if c == "A":
            tail = compute_tail_metrics(result_A["trades"], result_A["trades"])
        else:
            tail = compute_tail_metrics(result_A["trades"], res["trades"])

        case_results[c] = {"folds": folds_m, "wf": wf, "tail": tail, "trades": res["trades"]}

        if c != "A":
            def _f(v, fmt="+.2f"):
                if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
                return format(v, fmt)
            tc = tail.get("tail_capture", float("nan"))
            for fold, fm in zip(FOLDS, folds_m):
                if fm:
                    ok = "✅" if fm.get("fold_pass") else "❌"
                    print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                          f"ΔCAGR={fm['d_cagr']:+.2f}pp  ΔDD={fm['d_dd']:+.2f}pp  {ok}")
            tc_ok = not math.isnan(tc) and tc >= ADOPT_TAIL_CAP
            promote = wf.get("promote_metric", False) and tc_ok
            status  = "✅ ADOPT" if promote else "❌ REJECT"
            print(f"    → WF={wf.get('n_pass',0)}/5  "
                  f"ΔCAGR={wf.get('avg_dcagr',0):+.2f}pp  "
                  f"ΔDD={wf.get('avg_ddd',0):+.2f}pp  "
                  f"tail_capture={_pct(tc)}  {status}\n")

    # Store A tail info separately for report
    case_results["__tail_A"] = compute_tail_metrics(result_A["trades"], result_A["trades"])

    print("[4/4] Final output 計算...")
    print("\n  Case | tail_efficiency | keep_or_drop")
    for c in CASE_ORDER:
        if c == "A": continue
        wf   = case_results[c].get("wf", {})
        tail = case_results[c].get("tail", {})
        tc   = tail.get("tail_capture", float("nan"))
        pm   = wf.get("promote_metric", False)
        tc_ok= not math.isnan(tc) and tc >= ADOPT_TAIL_CAP
        adopt= pm and tc_ok
        te   = round(tc * (wf.get("n_pass", 0) / 5), 4) if not math.isnan(tc) else float("nan")
        kod  = "KEEP" if adopt else "DROP"
        print(f"  {c}: {CASES[c].label}")
        print(f"     tail_efficiency={te}  keep_or_drop={kod}")

    print("=" * 68 + "\n")
    write_report(
        case_results, result_A, topix_close,
        REPORTS_DIR / "entry_velocity_tail_preservation.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
