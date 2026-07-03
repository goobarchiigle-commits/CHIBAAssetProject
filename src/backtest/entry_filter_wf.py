"""
backtest/entry_filter_wf.py

研究専用 / 実装変更禁止

Oracle 改善(+5.38pp) のうち、未来情報なしで何pp回収できるか。
単独 → 組み合わせ順に WF 評価。

候補ルール:
A: days_since_RSR_cross70 <= N  (N=5/10/15/20/25/30)
B: state != EARLY_ROLL
C: ATR20_pct >= X (X = IS期間の下位 10/20/30/40% パーセンタイル)

WF 5-fold: OOS 2021/2022/2023/2024/2025

採用条件: WF≥4/5 AND ΔCAGR≥+1pp AND ΔDD≤+2pp

出力:
  reports/entry_filter_wf.md
  reports/entry_filter_wf_detail.csv

Run:
    cd C:/ai-trading
    python src/backtest/entry_filter_wf.py
"""

from __future__ import annotations

import sys, time, warnings, csv, itertools
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy, Position, _take, calc_metrics,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR   = Path("reports")
IS_START      = "2018-01-01"
ORACLE_PP     = 5.38   # Oracle上限 (entry_zone_audit.py より)

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31", "is_end": "2020-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31", "is_end": "2021-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31", "is_end": "2022-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31", "is_end": "2023-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31", "is_end": "2024-12-31"},
]

A_PARAMS  = [5, 10, 15, 20, 25, 30]
C_PCTS    = [0.10, 0.20, 0.30, 0.40]
STALL_THR = 0.15

# 採用条件
ADOPT_WF_MIN   = 4
ADOPT_DCAGR    = 1.0   # pp
ADOPT_DMAXDD   = 2.0   # pp (positive = worse)


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def classify_entry_state(s5: float, s20: float) -> str:
    if np.isnan(s5) or np.isnan(s20):
        return "UNKNOWN"
    if abs(s5) < STALL_THR:
        return "STALL"
    if s20 > 0:
        if s5 > s20:   return "EARLY_UP"
        if s5 > 0:     return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"


def check_filter(fp: dict, d70: int, state: str, atr: float) -> bool:
    """True = passes all active rules."""
    if fp.get("rule_a_n") is not None:
        if d70 > fp["rule_a_n"]:
            return False
    if fp.get("rule_b"):
        if state == "EARLY_ROLL":
            return False
    c_thr = fp.get("rule_c_threshold")
    if c_thr is not None:
        if atr <= 0 or atr < c_thr:
            return False
    return True


def compute_cross70_mat(rsr_mat: np.ndarray) -> np.ndarray:
    """Cross70 running days: cross70_mat[i,si] = consecutive days RSR>=70 up to (not including) day i."""
    n_dates, n_syms = rsr_mat.shape
    out     = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = running
        above  = rsr_mat[i] >= 70.0
        running[above]  += 1
        running[~above]  = 0
    return out


def compute_slope_mats(rsr_mat: np.ndarray):
    """Return slope5_mat, slope20_mat as float32."""
    s5  = np.zeros_like(rsr_mat)
    s20 = np.zeros_like(rsr_mat)
    s5[5:]  = (rsr_mat[5:]  - rsr_mat[:-5])  / 5.0
    s20[20:]= (rsr_mat[20:] - rsr_mat[:-20]) / 20.0
    return s5, s20


def compute_atr20_mat(high_mat: np.ndarray, low_mat: np.ndarray,
                      close_mat: np.ndarray, lookback: int = 20) -> np.ndarray:
    """ATR20 as % of close. Vectorized O(n_dates × n_syms)."""
    n_dates, n_syms = high_mat.shape
    # True Range (day 1 onward)
    tr = np.zeros_like(high_mat)
    tr[1:] = np.maximum(
        np.maximum(high_mat[1:] - low_mat[1:],
                   np.abs(high_mat[1:] - close_mat[:-1])),
        np.abs(low_mat[1:] - close_mat[:-1])
    )
    # Rolling mean via cumsum (no per-date loop)
    cumsum = np.zeros((n_dates + 1, n_syms), dtype=np.float64)
    cumsum[1:] = np.cumsum(tr, axis=0)
    atr_raw = np.zeros_like(high_mat, dtype=np.float64)
    atr_raw[lookback:] = (
        cumsum[lookback + 1:] - cumsum[1: n_dates - lookback + 1]
    ) / lookback
    safe_close = np.where(close_mat > 0, close_mat, np.nan)
    pct = np.where(close_mat > 0, atr_raw / safe_close * 100.0, 0.0)
    return pct.astype(np.float32)


def compute_atr_threshold_for_fold(
    atr20_mat: np.ndarray, sig_mat: np.ndarray, sig_ready: np.ndarray,
    common_dates, is_end_str: str, percentile: float,
) -> float:
    """Compute ATR Xth-percentile from IS buy signals for Rule C calibration."""
    atr_vals = []
    for i, d in enumerate(common_dates):
        if str(d.date()) > is_end_str:
            break
        sigs = sig_mat[i]
        for si in range(len(sig_ready)):
            if not sig_ready[si]:
                continue
            if int(sigs[si]) == 1:
                v = float(atr20_mat[i, si])
                if v > 0:
                    atr_vals.append(v)
    if not atr_vals:
        return 0.0
    return float(np.percentile(atr_vals, percentile * 100))


# ─────────────────────────────────────────────────────────────────────
#  CORE BACKTEST FOLD (with optional filter)
# ─────────────────────────────────────────────────────────────────────

def run_fold(
    open_mat, close_mat, sig_mat, sig_ready, rsr_mat,
    cross70_mat, slope5_mat, slope20_mat, atr20_mat,
    active_syms, sym_to_i, trade_syms, cfg,
    common_dates,
    oos_start_str: str, oos_end_str: str,
    fp: dict,                  # filter params
) -> dict:
    """
    Run one WF fold (IS + OOS) with filter fp.
    Returns OOS-only metrics dict + pass_rate.
    fp = {rule_a_n, rule_b, rule_c_threshold}  — None = disabled
    """
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)

    shock_market_thr = -0.05; shock_sym_thr = -0.08

    rc           = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W = float(rc.sector_cap) if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    n_syms = len(active_syms)

    # ── Topix-based bear / gross lookups from pre-computed matrices ───
    # (Embedded via extra arrays passed externally — simplified version)
    # We skip bear/gross_cap for this study (same as baseline)

    cash        = float(capital)
    positions: dict[str, Position] = {}
    peak_equity = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    trades: list[dict] = []

    equity_curve:  list[float] = []
    exposure_list: list[float] = []
    date_list:     list = []

    n_filter_passed = 0; n_filter_blocked = 0
    oos_start_i = None; oos_end_i = None

    n_dates  = len(common_dates)
    mkt_rets = None  # not pre-loaded here for simplicity

    for i, date in enumerate(common_dates):
        date_str = str(date.date())

        if date_str >= oos_start_str and oos_start_i is None:
            oos_start_i = i
        if date_str > oos_end_str and oos_end_i is None:
            oos_end_i = i

        # Equity snapshot
        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested
        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))
        date_list.append(date)

        # CB logic
        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0
        cb_scale = CB_SCALE if cb_active else 1.0

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        sell_sigs: list[tuple] = []
        buy_cands: list[tuple] = []

        for sym in active_syms:
            si_sym     = sym_to_i[sym]
            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si_sym])
            close_t    = float(close_mat[i, si_sym])

            if is_holding:
                if max_hold is not None and hold_idx > max_hold:
                    sell_sigs.append((sym, "TIME_STOP")); continue
                if rsr_val < rsr_exit_thr and hold_idx >= min_hold:
                    sell_sigs.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si_sym]) if sig_ready[si_sym] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_sigs.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1):
                    continue

                # ── Feature lookup (O(1) from pre-computed matrices) ──
                d70   = int(cross70_mat[i, si_sym])
                s5    = float(slope5_mat[i, si_sym])
                s20   = float(slope20_mat[i, si_sym])
                state = classify_entry_state(s5, s20)
                atr   = float(atr20_mat[i, si_sym])

                # ── Filter check ──────────────────────────────────────
                in_oos = (oos_start_i is not None and i >= oos_start_i
                          and (oos_end_i is None or i < oos_end_i))
                passes = check_filter(fp, d70, state, atr)
                if in_oos:
                    if passes:
                        n_filter_passed += 1
                    else:
                        n_filter_blocked += 1

                if not passes:
                    continue

                buy_cands.append((rsr_val, sym))

        # ── Sell ──────────────────────────────────────────────────────
        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos      = positions[sym]
            sell_px  = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({
                "side": "SELL", "symbol": sym, "pnl": pnl,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "entry_idx": pos.entry_idx, "exit_idx": i,
                "date": date_str,
            })
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not buy_cands:
            continue
        buy_cands.sort(key=lambda x: -x[0])

        # ── Buy ───────────────────────────────────────────────────────
        if cb_active:
            continue

        for rsr_val, sym in buy_cands:
            si_sym     = sym_to_i[sym]
            open_slots = max_pos - len(positions)
            if open_slots <= 0:
                break

            buy_px = float(open_mat[next_i, si_sym])
            if buy_px <= 0:
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, MAX_SECTOR_W):
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_normal:
                    continue

            n_rem  = sum(1 for _, s in buy_cands if s not in positions)
            alloc  = (cash / max(1, min(open_slots, n_rem))) * cb_scale
            alloc  = min(alloc, capital * MAX_SINGLE_W)
            qty    = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue

            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            trades[-1].update({"date": date_str, "entry_idx": next_i})

    # ── Extract OOS metrics ───────────────────────────────────────────
    oos_si = oos_start_i or 0
    oos_ei = oos_end_i   or len(equity_curve)

    oos_eq     = equity_curve[oos_si:oos_ei]
    oos_exp    = exposure_list[oos_si:oos_ei]
    oos_dates  = list(common_dates[oos_si:oos_ei])
    oos_trades = [t for t in trades
                  if oos_start_str <= t.get("date", "") <= oos_end_str]

    if not oos_eq:
        return {"cagr": 0, "max_dd": 0, "calmar": 0, "sharpe": 0,
                "pass_rate": 0, "n_passed": 0}

    # Compute metrics starting from beginning of OOS equity
    capital_at_oos = oos_eq[0]
    m = calc_metrics(oos_eq, oos_trades, oos_exp, capital_at_oos, oos_dates)

    n_total = n_filter_passed + n_filter_blocked
    pass_rate = n_filter_passed / max(1, n_total) * 100

    return {
        "cagr":      m.get("cagr", 0),      # already in % (e.g. 22.5)
        "max_dd":    m.get("max_dd", 0),    # already negative % (e.g. -18.0)
        "calmar":    m.get("calmar", 0),
        "sharpe":    m.get("sharpe", 0),
        "pass_rate": round(pass_rate, 1),
        "n_passed":  n_filter_passed,
        "n_blocked": n_filter_blocked,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATOR
# ─────────────────────────────────────────────────────────────────────

def evaluate_wf(
    baseline_folds: list[dict],
    filtered_folds: list[dict],
) -> dict:
    """Compute WF statistics across 5 folds."""
    dcagrs = []; ddds = []; dcalmars = []; pass_rates = []
    n_pass = 0

    for bf, ff in zip(baseline_folds, filtered_folds):
        dc  = ff["cagr"]  - bf["cagr"]                 # pp (already %)
        dd  = -(ff["max_dd"] - bf["max_dd"])           # positive = worse (already %)
        dcal= ff["calmar"]   - bf["calmar"]
        pr  = ff["pass_rate"]

        dcagrs.append(dc); ddds.append(dd); dcalmars.append(dcal)
        pass_rates.append(pr)

        # Fold pass: ΔCAGR >= 0 AND ΔDD <= 2pp
        if dc >= 0 and dd <= ADOPT_DMAXDD:
            n_pass += 1

    avg_dc    = float(np.mean(dcagrs))
    avg_dd    = float(np.mean(ddds))
    avg_dcal  = float(np.mean(dcalmars))
    avg_pr    = float(np.mean(pass_rates))

    adopted = (n_pass >= ADOPT_WF_MIN
               and avg_dc >= ADOPT_DCAGR
               and avg_dd <= ADOPT_DMAXDD)
    recovery = round(avg_dc / ORACLE_PP * 100, 1) if ORACLE_PP > 0 else 0.0

    return {
        "wf_score":   n_pass,
        "avg_dcagr":  round(avg_dc, 2),
        "avg_ddd":    round(avg_dd, 2),
        "avg_dcalmar":round(avg_dcal, 3),
        "avg_pass_rate": round(avg_pr, 1),
        "fold_dcagrs": [round(x, 2) for x in dcagrs],
        "fold_ddds":   [round(x, 2) for x in ddds],
        "adopted":     adopted,
        "recovery_pct":recovery,
    }


# ─────────────────────────────────────────────────────────────────────
#  COMBO BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_combos() -> list[dict]:
    combos = [{"name": "BASELINE", "rule_a_n": None, "rule_b": False, "rule_c_pct": None}]

    # Single: A
    for N in A_PARAMS:
        combos.append({"name": f"A(N={N})", "rule_a_n": N, "rule_b": False, "rule_c_pct": None})

    # Single: B
    combos.append({"name": "B", "rule_a_n": None, "rule_b": True, "rule_c_pct": None})

    # Single: C
    for pct in C_PCTS:
        combos.append({"name": f"C(bot{int(pct*100)}%)", "rule_a_n": None, "rule_b": False, "rule_c_pct": pct})

    # A+B
    for N in A_PARAMS:
        combos.append({"name": f"A({N})+B", "rule_a_n": N, "rule_b": True, "rule_c_pct": None})

    # A+C
    for N in A_PARAMS:
        for pct in C_PCTS:
            combos.append({"name": f"A({N})+C(bot{int(pct*100)}%)", "rule_a_n": N, "rule_b": False, "rule_c_pct": pct})

    # B+C
    for pct in C_PCTS:
        combos.append({"name": f"B+C(bot{int(pct*100)}%)", "rule_a_n": None, "rule_b": True, "rule_c_pct": pct})

    # A+B+C
    for N in A_PARAMS:
        for pct in C_PCTS:
            combos.append({"name": f"A({N})+B+C(bot{int(pct*100)}%)", "rule_a_n": N, "rule_b": True, "rule_c_pct": pct})

    return combos


# ─────────────────────────────────────────────────────────────────────
#  REPORT WRITER
# ─────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], output_path: Path) -> None:
    L = []; w = L.append

    w("# エントリーフィルタ WF 研究")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\nOracle 上限: **+{ORACLE_PP}pp**  "
      f"|  採用条件: WF≥{ADOPT_WF_MIN}/5, ΔCAGR≥+{ADOPT_DCAGR}pp, ΔDD≤+{ADOPT_DMAXDD}pp\n")

    adopted  = [r for r in results if r["wf"].get("adopted")]
    rejected = [r for r in results if not r["wf"].get("adopted") and r["name"] != "BASELINE"]

    w(f"**採用**: {len(adopted)}件  |  **不採用**: {len(rejected)}件\n")

    # ── Section 1: ベースライン ─────────────────────────────────────
    w("---\n## 1. ベースライン (フィルタなし)\n")
    bl = next(r for r in results if r["name"] == "BASELINE")
    w("| Fold | CAGR | MaxDD | Calmar |")
    w("|---|---|---|---|")
    for i, fd in enumerate(bl["fold_results"]):
        w(f"| Fold{i+1} | {fd['cagr']:+.1f}% | {abs(fd['max_dd']):.1f}% | {fd['calmar']:.2f} |")
    avg_cagr = np.mean([fd["cagr"] for fd in bl["fold_results"]])
    avg_dd   = np.mean([abs(fd["max_dd"]) for fd in bl["fold_results"]])
    avg_cal  = np.mean([fd["calmar"] for fd in bl["fold_results"]])
    w(f"| **avg** | **{avg_cagr:+.1f}%** | **{avg_dd:.1f}%** | **{avg_cal:.2f}** |")

    # ── Section 2: 全結果サマリ ────────────────────────────────────
    w("\n---\n## 2. 全 Filter 結果サマリ\n")
    w("| Filter | WF | ΔCAGR | ΔDD | ΔCalmar | pass_rate | recovery | 採用 |")
    w("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["name"] == "BASELINE":
            continue
        wf = r["wf"]
        mark = "**✅**" if wf["adopted"] else "❌"
        w(f"| {r['name']} "
          f"| {wf['wf_score']}/5 "
          f"| {wf['avg_dcagr']:+.2f}pp "
          f"| {wf['avg_ddd']:+.2f}pp "
          f"| {wf['avg_dcalmar']:+.3f} "
          f"| {wf['avg_pass_rate']:.1f}% "
          f"| {wf['recovery_pct']}% "
          f"| {mark} |")

    # ── Section 3: 採用フィルタ詳細 ──────────────────────────────
    if adopted:
        w("\n---\n## 3. 採用フィルタ 詳細\n")
        for r in sorted(adopted, key=lambda x: -x["wf"]["avg_dcagr"]):
            wf = r["wf"]
            w(f"\n### ✅ {r['name']}\n")
            w(f"- WF: **{wf['wf_score']}/5**")
            w(f"- avg ΔCAGR: **{wf['avg_dcagr']:+.2f}pp**")
            w(f"- avg ΔDD: **{wf['avg_ddd']:+.2f}pp**")
            w(f"- avg ΔCalmar: **{wf['avg_dcalmar']:+.3f}**")
            w(f"- 通過率: **{wf['avg_pass_rate']:.1f}%**")
            w(f"- Oracle 回収率: **{wf['recovery_pct']}%**")
            fold_tbl = "| Fold | ΔCAGR | ΔDD | 判定 |"
            w(f"\n{fold_tbl}")
            w("|---|---|---|---|")
            for i, (dc, dd) in enumerate(zip(wf["fold_dcagrs"], wf["fold_ddds"])):
                ok = "✅" if dc >= 0 and dd <= ADOPT_DMAXDD else "❌"
                w(f"| Fold{i+1} | {dc:+.2f}pp | {dd:+.2f}pp | {ok} |")

    # ── Section 4: 単独ルール比較 ─────────────────────────────────
    w("\n---\n## 4. 単独ルール比較\n")

    single_names = (
        [f"A(N={N})" for N in A_PARAMS]
        + ["B"]
        + [f"C(bot{int(p*100)}%)" for p in C_PCTS]
    )
    single_results = [r for r in results if r["name"] in single_names]
    w("| Filter | WF | ΔCAGR | ΔDD | pass_rate | 採用 |")
    w("|---|---|---|---|---|---|")
    for r in single_results:
        wf = r["wf"]
        mark = "✅" if wf["adopted"] else "❌"
        w(f"| {r['name']} | {wf['wf_score']}/5 | {wf['avg_dcagr']:+.2f}pp "
          f"| {wf['avg_ddd']:+.2f}pp | {wf['avg_pass_rate']:.1f}% | {mark} |")

    # ── Section 5: Rule A N 感度 ──────────────────────────────────
    w("\n---\n## 5. Rule A: N 感度分析\n")
    w("| N | WF | ΔCAGR | ΔDD | pass_rate |")
    w("|---|---|---|---|---|")
    for N in A_PARAMS:
        r = next((x for x in results if x["name"] == f"A(N={N})"), None)
        if not r: continue
        wf = r["wf"]
        w(f"| {N} | {wf['wf_score']}/5 | {wf['avg_dcagr']:+.2f}pp "
          f"| {wf['avg_ddd']:+.2f}pp | {wf['avg_pass_rate']:.1f}% |")

    # ── Section 6: Rule C パーセンタイル感度 ─────────────────────
    w("\n---\n## 6. Rule C: ATR パーセンタイル感度\n")
    w("| bottom% | WF | ΔCAGR | ΔDD | pass_rate |")
    w("|---|---|---|---|---|")
    for pct in C_PCTS:
        r = next((x for x in results if x["name"] == f"C(bot{int(pct*100)}%)"), None)
        if not r: continue
        wf = r["wf"]
        w(f"| bot{int(pct*100)}% | {wf['wf_score']}/5 | {wf['avg_dcagr']:+.2f}pp "
          f"| {wf['avg_ddd']:+.2f}pp | {wf['avg_pass_rate']:.1f}% |")

    # ── Section 7: Oracle 回収サマリ ─────────────────────────────
    w("\n---\n## 7. Oracle 回収サマリ\n")
    w(f"Oracle 上限: **+{ORACLE_PP}pp**\n")
    if adopted:
        best = max(adopted, key=lambda x: x["wf"]["recovery_pct"])
        w(f"最高回収: **{best['wf']['recovery_pct']}%** ({best['name']})")
        w(f"最高 ΔCAGR: **{best['wf']['avg_dcagr']:+.2f}pp**\n")
        w("| Filter | recovery_pct | ΔCAGR |")
        w("|---|---|---|")
        for r in sorted(adopted, key=lambda x: -x["wf"]["recovery_pct"]):
            w(f"| {r['name']} | **{r['wf']['recovery_pct']}%** | {r['wf']['avg_dcagr']:+.2f}pp |")
    else:
        w("採用ルールなし。改善の多くは未来情報に依存。\n")

    # ── Section 8: 結論 ───────────────────────────────────────────
    w("\n---\n## 8. 結論・次ステップ\n")
    if adopted:
        best = max(adopted, key=lambda x: x["wf"]["avg_dcagr"])
        w(f"**採用推奨: {best['name']}**\n")
        w(f"- ΔCAGR: {best['wf']['avg_dcagr']:+.2f}pp / Oracle {ORACLE_PP}pp = "
          f"{best['wf']['recovery_pct']}% 回収")
        w(f"- WF: {best['wf']['wf_score']}/5  |  通過率: {best['wf']['avg_pass_rate']:.1f}%")
        w(f"- ΔCalmar: {best['wf']['avg_dcalmar']:+.3f}  |  ΔDD: {best['wf']['avg_ddd']:+.2f}pp")
        w("\n**実装前に必要なステップ (ASK_FIRST):**")
        w("- signal_bridge.py 発注ロジック変更に該当 → 事前確認必須")
        w("- IS ATR閾値を毎月または四半期更新する仕組みが必要 (Rule C 使用時)")
        w("- 本番投入前の追加 OOS 期間検証を推奨")
    else:
        w("採用ルールなし。")
        w(f"- Oracle +{ORACLE_PP}pp の改善は主に未来情報によるもの")
        w("- 最良 ΔCAGR = "
          + str(max((r["wf"]["avg_dcagr"] for r in results if r["name"] != "BASELINE"), default=0))
          + "pp (採用基準未満)")
        w("- 次: 異なる特徴量 (days_since_cross90, sector_filter) の探索を検討")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  MD レポート: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  エントリーフィルタ WF 研究")
    print(f"  Oracle 上限 {ORACLE_PP}pp の回収率測定")
    print("=" * 68 + "\n")

    print("[1/5] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    # ── Matrix setup ─────────────────────────────────────────────────
    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: idx for idx, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts = pd.Timestamp(IS_START)
    te = pd.Timestamp("2025-12-31")
    common_dates = common_dates[(common_dates >= ts) & (common_dates <= te)]
    n_dates  = len(common_dates)

    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    high_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    low_mat   = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0):
            continue
        open_mat[:, si]  = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]
        high_mat[:, si]  = df_src["High"].to_numpy(dtype=np.float32)[row_idx]
        low_mat[:, si]   = df_src["Low"].to_numpy(dtype=np.float32)[row_idx]

        strats_map = {}
        sector = trade_syms.get(sym, "")
        rule   = SECTOR_STRATEGY.get(sector, "fujiko")
        rsr_s  = rsr_df[sym] if sym in rsr_df.columns else None
        if rule == "mean_rev":
            st = MeanReversionStrategy(**MR_PARAMS)
        else:
            st = FujikoStrategy(
                min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
            )
        required = 252 + getattr(st, "mom_period", 21) + 2
        if hasattr(st, "precompute_signals") and len(df_src) >= required:
            sig_series = st.precompute_signals(df_src)
            sig_mat[:, si] = sig_series.to_numpy(dtype=np.int8)[row_idx]
            sig_ready[si]  = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )

    print("[2/5] 特徴量マトリクス計算...")
    t0 = time.time()
    cross70_mat          = compute_cross70_mat(rsr_mat)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)
    atr20_mat            = compute_atr20_mat(high_mat, low_mat, close_mat)
    print(f"  完了 ({time.time()-t0:.1f}s)")

    # ── Build all filter combos ──────────────────────────────────────
    combos = build_combos()
    print(f"\n[3/5] {len(combos)-1} フィルタ組み合わせ × 5 fold を評価中...")
    print(f"  (内訳: A×6 / B×1 / C×4 / A+B×6 / A+C×24 / B+C×4 / A+B+C×24)\n")

    # Pre-compute ATR thresholds for all folds × percentiles
    atr_thresholds: dict[tuple, float] = {}
    for fold in FOLDS:
        for pct in C_PCTS:
            key = (fold["id"], pct)
            atr_thresholds[key] = compute_atr_threshold_for_fold(
                atr20_mat, sig_mat, sig_ready, common_dates, fold["is_end"], pct
            )

    # ── Run WF evaluations ───────────────────────────────────────────
    results = []
    t_total = time.time()
    n_combos = len(combos)

    for ci, combo in enumerate(combos):
        fold_results = []

        for fold in FOLDS:
            # Build filter params
            fp = {
                "rule_a_n":          combo["rule_a_n"],
                "rule_b":            combo["rule_b"],
                "rule_c_threshold":  None,
            }
            if combo.get("rule_c_pct") is not None:
                key = (fold["id"], combo["rule_c_pct"])
                fp["rule_c_threshold"] = atr_thresholds.get(key, 0.0)

            fd = run_fold(
                open_mat, close_mat, sig_mat, sig_ready, rsr_mat,
                cross70_mat, slope5_mat, slope20_mat, atr20_mat,
                active_syms, sym_to_i, trade_syms, cfg,
                common_dates,
                oos_start_str=fold["oos_start"],
                oos_end_str=fold["oos_end"],
                fp=fp,
            )
            fold_results.append(fd)

        # Store fold results for baseline
        if combo["name"] == "BASELINE":
            baseline_folds = fold_results
            results.append({
                "name": "BASELINE",
                "combo": combo,
                "fold_results": fold_results,
                "wf": {},
            })
            # Print baseline
            bl_cagrs = [fd["cagr"] for fd in fold_results]
            print(f"  BASELINE avg CAGR: {np.mean(bl_cagrs):+.1f}%  "
                  f"({'/'.join(f'{x:+.0f}' for x in bl_cagrs)})")
            continue

        wf_stats = evaluate_wf(baseline_folds, fold_results)
        results.append({
            "name": combo["name"],
            "combo": combo,
            "fold_results": fold_results,
            "wf": wf_stats,
        })

        # Progress print
        if wf_stats["adopted"]:
            print(f"  ✅ {combo['name']:<35} WF={wf_stats['wf_score']}/5 "
                  f"ΔCAGR={wf_stats['avg_dcagr']:+.2f}pp  "
                  f"recovery={wf_stats['recovery_pct']}%")
        elif (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{n_combos}] {combo['name']:<30} WF={wf_stats['wf_score']}/5 "
                  f"ΔCAGR={wf_stats['avg_dcagr']:+.2f}pp")

    elapsed = time.time() - t_total
    print(f"\n  完了: {n_combos}件  ({elapsed:.1f}s, {elapsed/max(n_combos,1):.2f}s/combo)\n")

    # ── CSV output ───────────────────────────────────────────────────
    print("[4/5] CSV 出力...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "entry_filter_wf_detail.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["name", "wf_score", "avg_dcagr", "avg_ddd", "avg_dcalmar",
                      "avg_pass_rate", "recovery_pct", "adopted",
                      "f1_dcagr", "f2_dcagr", "f3_dcagr", "f4_dcagr", "f5_dcagr",
                      "f1_ddd",   "f2_ddd",   "f3_ddd",   "f4_ddd",   "f5_ddd"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r["name"] == "BASELINE":
                continue
            wf = r["wf"]
            row = {
                "name":         r["name"],
                "wf_score":     wf.get("wf_score", ""),
                "avg_dcagr":    wf.get("avg_dcagr", ""),
                "avg_ddd":      wf.get("avg_ddd", ""),
                "avg_dcalmar":  wf.get("avg_dcalmar", ""),
                "avg_pass_rate":wf.get("avg_pass_rate", ""),
                "recovery_pct": wf.get("recovery_pct", ""),
                "adopted":      int(wf.get("adopted", False)),
            }
            for fi, (dc, dd) in enumerate(
                zip(wf.get("fold_dcagrs", []), wf.get("fold_ddds", []))
            ):
                row[f"f{fi+1}_dcagr"] = dc
                row[f"f{fi+1}_ddd"]   = dd
            writer.writerow(row)
    print(f"  CSV: {csv_path}")

    print("[5/5] レポート生成...")
    write_report(results, REPORTS_DIR / "entry_filter_wf.md")

    # ── Console summary ──────────────────────────────────────────────
    adopted = [r for r in results if r["wf"].get("adopted")]
    print(f"\n{'='*68}")
    print(f"  Oracle: +{ORACLE_PP}pp  |  採用: {len(adopted)}件")
    if adopted:
        best = max(adopted, key=lambda x: x["wf"]["avg_dcagr"])
        print(f"  最高回収: {best['wf']['recovery_pct']}% ({best['name']})")
        print(f"  最高 ΔCAGR: {best['wf']['avg_dcagr']:+.2f}pp "
              f"WF={best['wf']['wf_score']}/5")
    else:
        print("  採用ルールなし")
    print(f"{'='*68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
