"""
backtest/study9_standalone_validation.py  —  Study9 Standalone Validation WF

baseline: standalone sleeve (Task1 Case A)
  Entry:  RSR[92,95) d90≤5  (no slope filter)
  Exit:   RSR<90 AND hold≥3
  Capital: SLEEVE_FRAC=0.25 (固定)
  MaxPos:  1 スロット (固定)
  Regime:  なし (禁止)

Case B: +slope5≤5
Case C: +slope5≤5  +exit thr 90→87
Case D: +slope5≤5  +profit_lock+8%
Case E: +slope5≤5  +exit thr 90→87  +profit_lock+8%

WF fold pass: calmar_X >= calmar_A
Adopt: WF>=4/5 AND avg_calmar > avg_calmar_A

出力: reports/study9_standalone_validation.md

Run:
    cd C:/ai-trading
    python src/backtest/study9_standalone_validation.py
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
from src.backtest.capital_allocation_abc import load_data, _take, calc_metrics, LOT, COST_ONE_WAY

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Fixed params (全Case共通) ──────────────────────────────────────────
RSR_LO       = 92.0    # entry RSR lower bound (inclusive)
RSR_HI       = 95.0    # entry RSR upper bound (exclusive)
D90_MAX      = 5       # max days since RSR crossed 90
MIN_HOLD     = 3       # min hold before ANY exit fires
SLEEVE_FRAC  = 0.25    # capital fraction for sleeve
SHOCK_MKT    = -0.05
SHOCK_SYM    = -0.08

# ── Adoption ──────────────────────────────────────────────────────────
ADOPT_WF     = 4


# ──────────────────────────────────────────────────────────────────────
#  CASE SPEC
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CaseSpec:
    label:       str
    slope_max:   Optional[float] = None   # B/C/D/E: 5.0; A: None
    exit_thr:    float           = 90.0   # A/B/D: 90; C/E: 87
    profit_lock: Optional[float] = None   # D/E: 0.08


CASES: dict[str, CaseSpec] = {
    "A": CaseSpec("baseline  (exit<90)",                     None, 90.0, None),
    "B": CaseSpec("+slope≤5  (exit<90)",                      5.0, 90.0, None),
    "C": CaseSpec("+slope≤5  +exit<87",                       5.0, 87.0, None),
    "D": CaseSpec("+slope≤5  +profit_lock+8%",                5.0, 90.0, 0.08),
    "E": CaseSpec("+slope≤5  +exit<87  +profit_lock+8%",      5.0, 87.0, 0.08),
}

CASE_ORDER = ["A", "B", "C", "D", "E"]


def _f(v, fmt="+.2f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return format(v, fmt)


# ──────────────────────────────────────────────────────────────────────
#  MATRIX HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_cross90_mat(rsr_mat: np.ndarray) -> np.ndarray:
    """days-since-RSR-crossed-90: 0 = not above 90, 1 = first day above, ..."""
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


def regime_label_for(topix_close, year: int) -> str:
    if topix_close is None: return "N/A"
    yr  = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ──────────────────────────────────────────────────────────────────────
#  STANDALONE SIMULATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:      str
    qty:         int
    entry_price: float
    entry_idx:   int
    rsr_entry:   float
    d90_entry:   int
    slope5_entry:float
    days_below:  int = 0


def run_standalone(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i,
    common_dates, capital: float,
    spec: CaseSpec,
) -> dict:
    n       = len(common_dates)
    sl_cash = capital * SLEEVE_FRAC
    sl_pos: SlPos | None = None
    trades: list[dict]   = []
    eq  = np.zeros(n, dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)

    for i, date in enumerate(common_dates):
        ds    = str(date.date())
        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        s_eq  = sl_cash + s_inv
        eq[i]  = s_eq
        inv[i] = s_inv

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

            # 1. Market shock (always fires, ignores min_hold)
            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"

            # 2. Profit lock (fires before RSR exit, no min_hold requirement)
            if not do_exit and spec.profit_lock is not None and ep > 0:
                if (ct / ep - 1) >= spec.profit_lock:
                    do_exit = True; reason = "PROFIT_LOCK"

            # 3. RSR exit (requires min_hold)
            if not do_exit:
                if rv < spec.exit_thr:
                    sl_pos.days_below += 1
                else:
                    sl_pos.days_below = 0
                if sl_pos.days_below >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = f"RSR_EXIT_{spec.exit_thr:.0f}"

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
        if sl_pos is None and not mkt_shock:
            cands: list[tuple] = []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si])
                sl5 = float(slope5_mat[i, si])

                if rv < RSR_LO or rv >= RSR_HI: continue
                if d90 > D90_MAX or d90 < 1:    continue   # must be above 90 already
                if spec.slope_max is not None and sl5 > spec.slope_max: continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue

                cands.append((rv, d90, sl5, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))   # high RSR, low d90
                rv, d90, sl5, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp > 0:
                    alloc = sl_cash * 0.95
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST_ONE_WAY)
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

    return {"eq": eq, "inv": inv, "trades": trades}


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    result: dict, common_dates, fold: dict,
    capital: float, calmar_A: float,
) -> dict:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    date_strs  = [str(d.date()) for d in common_dates]
    oos_idx    = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not oos_idx: return {}
    si, ei    = oos_idx[0], oos_idx[-1] + 1
    oos_dates = list(common_dates[si:ei])

    s_eq  = result["eq"][si:ei].tolist()
    s_inv = result["inv"][si:ei].tolist()
    if len(s_eq) < 5: return {}

    trades_oos = [t for t in result["trades"]
                  if ds_s <= (t.get("date") or "") <= ds_e]
    exp_list   = [vi / max(1.0, ve) for vi, ve in zip(s_inv, s_eq)]
    m = calc_metrics(s_eq, trades_oos, exp_list, s_eq[0], oos_dates)

    sells = [t for t in trades_oos if t["side"] == "SELL"]
    hds   = [t["hold_days"] for t in sells if (t.get("hold_days") or 0) > 0]
    n_pl  = sum(1 for t in sells if t.get("reason","").startswith("PROFIT_LOCK"))
    n_rsr = sum(1 for t in sells if t.get("reason","").startswith("RSR_EXIT"))
    cap_util = sum(1 for v in s_inv if v > 0) / max(1, len(s_eq)) * 100

    calmar    = m.get("calmar", 0.0)
    fold_pass = calmar >= calmar_A

    return {
        "cagr":      m.get("cagr", 0.0),
        "max_dd":    m.get("max_dd", 0.0),
        "calmar":    calmar,
        "pf":        m.get("profit_factor", 0.0),
        "win_rate":  m.get("win_rate", 0.0),
        "sharpe":    m.get("sharpe", 0.0),
        "n_trades":  len(sells),
        "avg_hold":  round(float(np.mean(hds)) if hds else 0.0, 1),
        "cap_util":  round(cap_util, 1),
        "n_pl":      n_pl,
        "n_rsr":     n_rsr,
        "fold_pass": fold_pass,
        "calmar_A":  calmar_A,
    }


def aggregate_wf(fold_results: list[dict]) -> dict:
    valid  = [f for f in fold_results if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(k):
        vals = [f[k] for f in valid if not math.isnan(f.get(k, 0))]
        return float(np.mean(vals)) if vals else float("nan")

    calmar_vals = [f["calmar"]   for f in valid]
    calmar_A_v  = [f["calmar_A"] for f in valid]

    adopted = (
        n_pass >= ADOPT_WF and
        avg("calmar") > avg("calmar_A")
    )
    return {
        "n_pass":       n_pass,
        "avg_cagr":     round(avg("cagr"), 2),
        "avg_dd":       round(avg("max_dd"), 2),
        "avg_calmar":   round(avg("calmar"), 3),
        "avg_calmar_A": round(avg("calmar_A"), 3),
        "calmar_lift":  round(avg("calmar") - avg("calmar_A"), 3),
        "calmar_std":   round(float(np.std(calmar_vals)), 3),
        "avg_pf":       round(avg("pf"), 3),
        "avg_win":      round(avg("win_rate"), 1),
        "avg_hold":     round(avg("avg_hold"), 1),
        "avg_util":     round(avg("cap_util"), 1),
        "avg_n_pl":     round(avg("n_pl"), 1),
        "avg_n_rsr":    round(avg("n_rsr"), 1),
        "adopted":      adopted,
    }


# ──────────────────────────────────────────────────────────────────────
#  EXIT BREAKDOWN
# ──────────────────────────────────────────────────────────────────────

def exit_breakdown(trades: list[dict]) -> dict:
    sells = [t for t in trades if t["side"] == "SELL"]
    n     = max(1, len(sells))
    n_pl  = sum(1 for t in sells if t.get("reason","").startswith("PROFIT_LOCK"))
    n_rsr = sum(1 for t in sells if t.get("reason","").startswith("RSR_EXIT"))
    n_mkt = sum(1 for t in sells if "SHOCK" in t.get("reason",""))

    wins  = [t["pnl"] for t in sells if t.get("pnl",0) > 0]
    loss  = [t["pnl"] for t in sells if t.get("pnl",0) <= 0]
    hds   = [t.get("hold_days",0) for t in sells if (t.get("hold_days") or 0) > 0]

    # slope5 breakdown
    s5_sells = [t for t in sells if t.get("slope5_entry", 0) > 5.0]
    s5_wins  = sum(1 for t in s5_sells if t.get("pnl",0) > 0)

    return {
        "n_total":   len(sells),
        "n_pl":      n_pl,
        "n_rsr":     n_rsr,
        "n_mkt":     n_mkt,
        "pl_rate":   round(n_pl / n * 100, 1),
        "rsr_rate":  round(n_rsr / n * 100, 1),
        "win_rate":  round(len(wins) / n * 100, 1),
        "avg_win":   round(float(np.mean(wins)) if wins else 0, 0),
        "avg_loss":  round(float(np.mean(loss)) if loss else 0, 0),
        "avg_hold":  round(float(np.mean(hds)) if hds else 0, 1),
        "n_s5":      len(s5_sells),
        "s5_win_rate": round(s5_wins / max(1, len(s5_sells)) * 100, 1),
    }


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    topix_close,
    path: Path,
) -> None:
    L = []; w = L.append

    # determine best adopted case
    adopted_cases = [c for c in CASE_ORDER if c != "A"
                     and case_results[c]["wf"].get("adopted")]
    if adopted_cases:
        best = max(adopted_cases,
                   key=lambda c: case_results[c]["wf"].get("calmar_lift", 0))
    else:
        best = max([c for c in CASE_ORDER if c != "A"],
                   key=lambda c: case_results[c]["wf"].get("calmar_lift", 0))

    w("# Study9 Standalone Validation WF")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\nEntry: RSR[{RSR_LO:.0f},{RSR_HI:.0f}) d90≤{D90_MAX}  "
      f"Capital: SLEEVE×{SLEEVE_FRAC:.2f}  MaxPos: 1  Regime: なし")
    w(f"\n採用条件: WF>={ADOPT_WF}/5 AND avg_calmar > avg_calmar_A\n")

    # ── S1. WF Summary ──────────────────────────────────────────────────
    w("---\n## 1. WF サマリ\n")
    w("| Case | WF | avg_CAGR | avg_DD | avg_Calmar | Calmar_A | Calmar_lift | avg_PF | avg_hold | ADOPT |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        wf  = case_results[c]["wf"]
        if c == "A":
            w(f"| **A** {CASES[c].label} "
              f"| —/5 "
              f"| {_f(wf.get('avg_cagr'))}% "
              f"| {_f(wf.get('avg_dd'))}% "
              f"| {_f(wf.get('avg_calmar'),'.3f')} "
              f"| — "
              f"| — "
              f"| {_f(wf.get('avg_pf'),'.3f')} "
              f"| {_f(wf.get('avg_hold'),'.1f')}d "
              f"| baseline |")
        else:
            mark = "**ADOPT**" if wf.get("adopted") else "REJECT"
            w(f"| {c}: {CASES[c].label} "
              f"| {wf.get('n_pass',0)}/5 "
              f"| {_f(wf.get('avg_cagr'))}% "
              f"| {_f(wf.get('avg_dd'))}% "
              f"| {_f(wf.get('avg_calmar'),'.3f')} "
              f"| {_f(wf.get('avg_calmar_A'),'.3f')} "
              f"| {_f(wf.get('calmar_lift'),'+.3f')} "
              f"| {_f(wf.get('avg_pf'),'.3f')} "
              f"| {_f(wf.get('avg_hold'),'.1f')}d "
              f"| {mark} |")

    # ── S2. ΔCAGR Fold Matrix ────────────────────────────────────────────
    w("\n---\n## 2. Fold × Case マトリクス (Calmar)\n")
    w("> Fold pass: calmar_X ≥ calmar_A (同 Fold)\n")
    header = "| Fold | Regime | A (calmar) |" + "".join(
        f" {c}: calmar | pass |" for c in CASE_ORDER if c != "A")
    w(header)
    sep = "|---|---|---|" + "---|---|" * (len(CASE_ORDER) - 1)
    w(sep)
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label_for(topix_close, yr)
        fm_A = case_results["A"]["folds"][fi]
        cal_A_str = _f(fm_A.get("calmar"), ".3f") if fm_A else "—"
        row = f"| Fold{fold['id']} | {reg} | {cal_A_str} |"
        for c in CASE_ORDER:
            if c == "A": continue
            fm = case_results[c]["folds"][fi]
            if fm:
                ok  = "✅" if fm.get("fold_pass") else "❌"
                row += f" {_f(fm.get('calmar'),'.3f')} | {ok} |"
            else:
                row += " — | — |"
        w(row)

    # ── S3. ΔCAGR raw ───────────────────────────────────────────────────
    w("\n---\n## 3. Fold × Case マトリクス (CAGR%)\n")
    header2 = "| Fold | Regime | CAGR_A |" + "".join(
        f" CAGR_{c} | ΔCAGR |" for c in CASE_ORDER if c != "A")
    w(header2)
    sep2 = "|---|---|---|" + "---|---|" * (len(CASE_ORDER) - 1)
    w(sep2)
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label_for(topix_close, yr)
        fm_A = case_results["A"]["folds"][fi]
        cagr_A = fm_A.get("cagr", 0) if fm_A else 0
        row = f"| Fold{fold['id']} | {reg} | {cagr_A:+.2f}% |"
        for c in CASE_ORDER:
            if c == "A": continue
            fm = case_results[c]["folds"][fi]
            if fm:
                cx  = fm.get("cagr", 0)
                row += f" {cx:+.2f}% | {cx-cagr_A:+.2f}pp |"
            else:
                row += " — | — |"
        w(row)

    # ── S4. Exit Breakdown ───────────────────────────────────────────────
    w("\n---\n## 4. Exit 内訳\n")
    w("| Case | n_trades | PROFIT_LOCK | RSR_EXIT | win_rate | avg_win | avg_loss | avg_hold | s5_win% |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        eb = case_results[c]["exit_bd"]
        pl_s = f"{eb['n_pl']}({eb['pl_rate']:.0f}%)" if eb['n_pl'] > 0 else "—"
        w(f"| {c}: {CASES[c].label} "
          f"| {eb['n_total']} "
          f"| {pl_s} "
          f"| {eb['n_rsr']}({eb['rsr_rate']:.0f}%) "
          f"| {eb['win_rate']:.0f}% "
          f"| ¥{eb['avg_win']:+,.0f} "
          f"| ¥{eb['avg_loss']:+,.0f} "
          f"| {eb['avg_hold']:.1f}d "
          f"| {eb['s5_win_rate']:.0f}% |")

    # ── S5. Case Analysis ────────────────────────────────────────────────
    w("\n---\n## 5. Case 分析\n")

    wf_A = case_results["A"]["wf"]
    wf_B = case_results["B"]["wf"]
    wf_C = case_results["C"]["wf"]
    wf_D = case_results["D"]["wf"]
    wf_E = case_results["E"]["wf"]

    w("### B vs A: slope≤5 フィルター効果")
    w(f"- Calmar lift: {_f(wf_B.get('calmar_lift'),'+.3f')}  "
      f"WF: {wf_B.get('n_pass',0)}/5  "
      f"avg_hold: {_f(wf_B.get('avg_hold'),'.1f')}d")
    w(f"- 解釈: slope>5 entry 除外の純効果 (exit 変更なし)\n")

    w("### C vs B: exit 87 効果 (slope≤5 維持)")
    lift_CB = (wf_C.get("avg_calmar",0) - wf_B.get("avg_calmar",0))
    w(f"- Calmar vs B: {lift_CB:+.3f}  "
      f"WF: {wf_C.get('n_pass',0)}/5  "
      f"avg_hold: {_f(wf_C.get('avg_hold'),'.1f')}d")
    w(f"- 解釈: exit thr 引下げ(90→87) による保有延長効果\n")

    w("### D vs B: profit_lock+8% 効果 (slope≤5 維持、exit<90)")
    lift_DB = (wf_D.get("avg_calmar",0) - wf_B.get("avg_calmar",0))
    w(f"- Calmar vs B: {lift_DB:+.3f}  "
      f"WF: {wf_D.get('n_pass',0)}/5  "
      f"avg_hold: {_f(wf_D.get('avg_hold'),'.1f')}d  "
      f"avg PROFIT_LOCK: {_f(wf_D.get('avg_n_pl'),'.1f')}/fold")
    w(f"- 解釈: +8%到達で即利食い → avg_win 変化、DD 影響\n")

    w("### E vs B: exit87 + profit_lock 複合効果")
    lift_EB = (wf_E.get("avg_calmar",0) - wf_B.get("avg_calmar",0))
    w(f"- Calmar vs B: {lift_EB:+.3f}  "
      f"WF: {wf_E.get('n_pass',0)}/5")
    w(f"- 解釈: 両方組み合わせが相乗効果を生むか、または干渉するか\n")

    # ── S6. Final Recommendation ─────────────────────────────────────────
    w("---\n## 6. 最終判定\n")
    w("| Case | WF | Calmar_lift | ADOPT |")
    w("|---|---|---|---|")
    for c in CASE_ORDER:
        if c == "A": continue
        wf   = case_results[c]["wf"]
        mark = "**ADOPT**" if wf.get("adopted") else "REJECT"
        w(f"| {c}: {CASES[c].label} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {_f(wf.get('calmar_lift'),'+.3f')} "
          f"| {mark} |")

    if adopted_cases:
        w(f"\n### ADOPT: {', '.join(adopted_cases)}\n")
        w(f"best = **{best}**: {CASES[best].label}")
        wf_best = case_results[best]["wf"]
        w(f"- WF={wf_best.get('n_pass',0)}/5  "
          f"Calmar_lift={_f(wf_best.get('calmar_lift'),'+.3f')}  "
          f"avg_Calmar={_f(wf_best.get('avg_calmar'),'.3f')}")
        w(f"- 実装推奨 (ASK_FIRST): standalone sleeve にこの設定を適用")
    else:
        w(f"\n### 全 Case REJECT\n")
        w(f"best candidate: {best} (最大 Calmar_lift={_f(case_results[best]['wf'].get('calmar_lift'),'+.3f')})")
        w("standalone slope≤5 フィルターは WF=4/5 維持。exit/profit_lock 追加は未採用。")
        w("次研究: profit_lock 閾値スイープ (6%/10%/12%) または exit_thr スイープ (85/88/90)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Study9 Standalone Validation WF")
    print(f"  entry RSR[{RSR_LO:.0f},{RSR_HI:.0f}) d90≤{D90_MAX}  "
          f"sleeve={SLEEVE_FRAC:.2f}  min_hold={MIN_HOLD}")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
    n_syms      = len(active_syms)
    capital     = float(cfg.portfolio.capital)

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

    print("[3/4] シミュレーション (5 cases)...\n")
    case_results: dict = {}
    calmar_A_by_fold: list[float] = [0.0] * 5

    for c in CASE_ORDER:
        spec = CASES[c]
        print(f"  [{c}] {spec.label}")
        res  = run_standalone(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            cross90_mat, slope5_mat, mkt_ret1,
            active_syms, sym_to_i, common_dates, capital, spec,
        )
        eb = exit_breakdown(res["trades"])

        folds_m: list[dict] = []
        for fi, fold in enumerate(FOLDS):
            cal_A = calmar_A_by_fold[fi] if c != "A" else 0.0
            fm = compute_fold_metrics(res, common_dates, fold, capital, cal_A)
            if c == "A" and fm:
                calmar_A_by_fold[fi] = fm.get("calmar", 0.0)
            folds_m.append(fm)

        # Re-run fold metrics for A with correct calmar_A (self-comparison gives pass=True always)
        # For A: store fold calmars for use by B/C/D/E
        wf = aggregate_wf(folds_m)
        case_results[c] = {"folds": folds_m, "wf": wf, "trades": res["trades"], "exit_bd": eb}

        if c != "A":
            for fold, fm in zip(FOLDS, folds_m):
                if fm:
                    ok = "✅" if fm.get("fold_pass") else "❌"
                    print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                          f"Calmar={fm.get('calmar',0):.3f}  "
                          f"(A={fm.get('calmar_A',0):.3f})  "
                          f"CAGR={fm.get('cagr',0):+.2f}%  {ok}")
            status = "✅ ADOPT" if wf.get("adopted") else "❌ REJECT"
            print(f"    → WF={wf.get('n_pass',0)}/5  "
                  f"Calmar_lift={wf.get('calmar_lift',0):+.3f}  "
                  f"avg_hold={wf.get('avg_hold',0):.1f}d  {status}\n")
        else:
            sells_A = [t for t in res["trades"] if t["side"] == "SELL"]
            n_A = len(sells_A)
            slope_A = [t.get("slope5_entry",0) for t in res["trades"] if t["side"]=="BUY"]
            n_s5 = sum(1 for v in slope_A if v > 5.0)
            print(f"    n_trades={n_A}  slope>5={n_s5}({n_s5/max(1,n_A)*100:.0f}%)"
                  f"  avg_hold={eb.get('avg_hold',0):.1f}d"
                  f"  win_rate={eb.get('win_rate',0):.0f}%\n")

    print("[4/4] レポート出力...")
    write_report(
        case_results, topix_close,
        REPORTS_DIR / "study9_standalone_validation.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
