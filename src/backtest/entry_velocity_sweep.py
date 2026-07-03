"""
backtest/entry_velocity_sweep.py  —  Entry Velocity Sweep WF

目的: slope5<=6 が偶然の閾値か構造的改善かを判定

固定:
  ENTRY = RSR[92,94]  d90<=5
  EXIT  = RSR<90 (即時)  MIN_HOLD=3d

比較 (slope_max のみ変化):
  A: slope<=4
  B: slope<=5
  C: slope<=6   (前回採用)
  D: slope<=7
  E: slope<=8
  F: no slope   (unfiltered baseline)

追加監査:
  entry_accept_rate    有効候補のうちフィルタ通過率
  marginal_calmar_gain 隣接Case間 Calmar差分
  gain_per_rejected_entry  rejected 1件あたりのCalmar gain

停止条件:
  best − neighbor < 0.05 → slope研究終了

出力: reports/entry_velocity_sweep.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_velocity_sweep.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, _take, calc_metrics,
    LOT, COST_ONE_WAY,
)

REPORTS_DIR  = Path("reports")
IS_START     = "2018-01-01"

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Fixed params ───────────────────────────────────────────────────────
RSR_LO     = 92.0
RSR_HI     = 94.0   # inclusive
D90_MAX    = 5
EXIT_THR   = 90.0
MIN_HOLD   = 3
SLEEVE_FRAC = 0.25
SHOCK_MKT  = -0.05
SHOCK_SYM  = -0.08

ADOPT_WF   = 4
STOP_GAP   = 0.05   # best − neighbor < STOP_GAP → 停止条件


# ──────────────────────────────────────────────────────────────────────
#  CASE SPEC
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlopeSpec:
    label:     str
    slope_max: float | None   # None = no slope filter


CASES: dict[str, SlopeSpec] = {
    "A": SlopeSpec("slope≤4",  4.0),
    "B": SlopeSpec("slope≤5",  5.0),
    "C": SlopeSpec("slope≤6",  6.0),
    "D": SlopeSpec("slope≤7",  7.0),
    "E": SlopeSpec("slope≤8",  8.0),
    "F": SlopeSpec("no slope", None),
}

CASE_ORDER = ["A", "B", "C", "D", "E", "F"]


# ──────────────────────────────────────────────────────────────────────
#  PRE-COMPUTE
# ──────────────────────────────────────────────────────────────────────

def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    n_dates, n_syms = rsr_mat.shape
    out     = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = running
        above = rsr_mat[i] >= threshold
        running[above] += 1
        running[~above] = 0
    return out


def compute_rsr_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


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


def run_standalone(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i,
    common_dates, capital: float,
    spec: SlopeSpec,
) -> dict:
    n       = len(common_dates)
    sl_cash = capital * SLEEVE_FRAC
    sl_pos: SlPos | None = None
    trades: list[dict]   = []
    eq  = np.zeros(n, dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)
    # candidate tracking for accept_rate
    cand_total   = 0
    cand_accepted = 0

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
            hd   = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"

            if not do_exit and rv < EXIT_THR and hd >= MIN_HOLD:
                do_exit = True; reason = "RSR_EXIT"

            if do_exit:
                sp  = float(open_mat[nxt, si])
                pnl = (sp - sl_pos.entry_price) * sl_pos.qty
                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": sl_pos.entry_price, "exit": sp,
                    "qty": sl_pos.qty,
                    "entry_idx": sl_pos.entry_idx, "exit_idx": i,
                    "reason": reason, "date": ds,
                    "hold_days": i - sl_pos.entry_idx,
                    "rsr_entry": sl_pos.rsr_entry,
                })
                sl_pos = None

        # ── ENTRY ─────────────────────────────────────────────────────
        if sl_pos is None and not mkt_shock:
            cands_pass, cands_all = [], []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si])
                sl5 = float(slope5_mat[i, si])
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                    continue
                # base RSR+d90 filter
                if not (RSR_LO <= rv <= RSR_HI and d90 <= D90_MAX):
                    continue
                cands_all.append((rv, d90, sl5, sym))
                # slope filter
                if spec.slope_max is None or sl5 <= spec.slope_max:
                    cands_pass.append((rv, d90, sym))

            cand_total    += len(cands_all)
            cand_accepted += len(cands_pass)

            if cands_pass:
                cands_pass.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sym = cands_pass[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp > 0:
                    qty  = int(sl_cash * 0.95 / bp / LOT) * LOT
                    cost = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos = SlPos(sym, qty, bp, nxt, rv)
                        trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "exit": None, "pnl": None,
                            "qty": qty, "entry_idx": nxt, "exit_idx": None,
                            "reason": f"RSR={rv:.1f} d90={d90}", "date": ds,
                            "hold_days": None, "rsr_entry": rv,
                        })

    accept_rate = (cand_accepted / max(1, cand_total)) if cand_total > 0 else 1.0
    return {
        "eq": eq, "inv": inv, "trades": trades,
        "cand_total": cand_total, "cand_accepted": cand_accepted,
        "accept_rate": accept_rate,
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    result: dict, common_dates, fold: dict, capital: float,
) -> dict:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    date_strs  = [str(d.date()) for d in common_dates]
    oos_idx    = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not oos_idx: return {}

    si, ei    = oos_idx[0], oos_idx[-1] + 1
    oos_dates = list(common_dates[si:ei])
    n_days    = len(oos_dates)
    n_years   = n_days / 252.0

    s_eq  = result["eq"][si:ei].tolist()
    s_inv = result["inv"][si:ei].tolist()
    if len(s_eq) < 5: return {}

    trades_oos = [t for t in result["trades"]
                  if ds_s <= (t.get("date") or "") <= ds_e]
    exp_list = [vi / max(1.0, ve) for vi, ve in zip(s_inv, s_eq)]
    m = calc_metrics(s_eq, trades_oos, exp_list, s_eq[0], oos_dates)

    sells   = [t for t in trades_oos if t["side"] == "SELL"]
    n_exits = len(sells)
    hds     = [t["hold_days"] for t in sells if (t.get("hold_days") or 0) > 0]

    return {
        "cagr":     m.get("cagr", 0.0),
        "max_dd":   m.get("max_dd", 0.0),
        "calmar":   m.get("calmar", 0.0),
        "pf":       m.get("profit_factor", 0.0),
        "hit_rate": m.get("win_rate", 0.0),
        "n_trades": n_exits,
        "trig_yr":  round(n_exits / max(0.01, n_years), 1),
        "avg_hold": round(float(np.mean(hds)) if hds else 0.0, 1),
        "cap_util": round(sum(1 for v in s_inv if v > 0) / max(1, n_days) * 100, 1),
    }


def aggregate_wf(fold_results: list[dict], calmar_F_folds: list[float]) -> dict:
    valid = [f for f in fold_results if f]
    if not valid: return {}

    def avg(k): return float(np.mean([f[k] for f in valid]))
    calmar_vals = [f["calmar"] for f in valid]
    # fold pass: Calmar >= Case F (unfiltered baseline) per fold
    n_pass = sum(1 for f, cf in zip(valid, calmar_F_folds) if f.get("calmar", 0) >= cf)

    adopted = n_pass >= ADOPT_WF

    return {
        "n_pass":     n_pass,
        "avg_cagr":   round(avg("cagr"), 2),
        "avg_dd":     round(avg("max_dd"), 2),
        "avg_calmar": round(avg("calmar"), 3),
        "calmar_std": round(float(np.std(calmar_vals)), 3),
        "avg_pf":     round(avg("pf"), 3),
        "avg_hit":    round(avg("hit_rate"), 1),
        "avg_trig":   round(avg("trig_yr"), 1),
        "avg_hold":   round(avg("avg_hold"), 1),
        "avg_util":   round(avg("cap_util"), 1),
        "adopted":    adopted,
    }


# ──────────────────────────────────────────────────────────────────────
#  MARGINAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def compute_marginal(case_results: dict, n_trades_F: int) -> dict:
    calmar = {c: case_results[c]["wf"].get("avg_calmar", 0.0) for c in CASE_ORDER}
    n_acc  = {c: case_results[c]["cand_accepted"] for c in CASE_ORDER}
    accept = {c: case_results[c]["accept_rate"]   for c in CASE_ORDER}

    # marginal gain vs neighbor (adjacent slope)
    ordered_filtered = ["A", "B", "C", "D", "E"]  # excludes F
    marginal = {}
    for i, c in enumerate(ordered_filtered):
        prev = ordered_filtered[i - 1] if i > 0 else "F"
        marginal[c] = round(calmar[c] - calmar[prev], 4)

    # best - neighbor stop condition
    best_c  = max(CASE_ORDER, key=lambda c: calmar[c])
    best_v  = calmar[best_c]
    neighbors = []
    idx = CASE_ORDER.index(best_c)
    if idx > 0:            neighbors.append(calmar[CASE_ORDER[idx - 1]])
    if idx < len(CASE_ORDER) - 1: neighbors.append(calmar[CASE_ORDER[idx + 1]])
    neighbor_best = max(neighbors) if neighbors else best_v
    gap = round(best_v - neighbor_best, 4)
    stop_triggered = gap < STOP_GAP

    # gain_per_rejected_entry:
    # for each filtered case vs F: Δcalmar / (n_rejected = n_F_accepted - n_c_accepted)
    gain_per_rej = {}
    n_F = n_acc.get("F", 1)
    for c in ordered_filtered:
        n_rej = max(1, n_F - n_acc.get(c, 0))
        gain_per_rej[c] = round((calmar[c] - calmar["F"]) / n_rej * 100, 4)

    return {
        "calmar_by_case":   {c: calmar[c] for c in CASE_ORDER},
        "accept_rate":      accept,
        "marginal_gain":    marginal,
        "best_case":        best_c,
        "best_calmar":      round(best_v, 3),
        "neighbor_calmar":  round(neighbor_best, 3),
        "gap":              gap,
        "stop_triggered":   stop_triggered,
        "gain_per_rej":     gain_per_rej,
    }


# ──────────────────────────────────────────────────────────────────────
#  REGIME LABEL
# ──────────────────────────────────────────────────────────────────────

def regime_label(topix_close, year: int) -> str:
    if topix_close is None: return "N/A"
    yr  = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret * 100:+.1f}%)"


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    marginal: dict,
    topix_close,
    path: Path,
) -> None:
    L = []; w = L.append
    stop = marginal["stop_triggered"]
    best = marginal["best_case"]
    adopted = [c for c in CASE_ORDER if case_results[c]["wf"].get("adopted")]
    verdict = "SLOPE_STRUCTURAL" if not stop else "SLOPE_COINCIDENCE"

    def fp(v, fmt=".3f"):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
        return format(v, fmt)

    w("# Entry Velocity Sweep WF")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n固定: ENTRY=RSR[{RSR_LO:.0f},{RSR_HI:.0f}] d90≤{D90_MAX}  "
      f"EXIT=RSR<{EXIT_THR:.0f}(即時)  MIN_HOLD={MIN_HOLD}d")
    w(f"\n採用条件: WF>={ADOPT_WF}/5  Calmar最大  fold_std最小")
    w(f"\n停止条件: best−neighbor<{STOP_GAP} → slope研究終了\n")
    w(f"**最終判定: {verdict}**  "
      f"best=Case{best}(Calmar={marginal['best_calmar']:.3f})  "
      f"neighbor={marginal['neighbor_calmar']:.3f}  "
      f"gap={marginal['gap']:+.4f}  "
      f"stop={'YES' if stop else 'NO'}\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    wf_F = case_results["F"]["wf"]
    wf_C = case_results["C"]["wf"]
    wf_B = case_results[best]["wf"]
    w(f"- Baseline (Case F, no slope): CAGR={wf_F['avg_cagr']:+.2f}%  "
      f"Calmar={wf_F['avg_calmar']:.3f}  Calmar_std={wf_F['calmar_std']:.3f}")
    w(f"- 前回採用 (Case C, slope≤6): CAGR={wf_C['avg_cagr']:+.2f}%  "
      f"Calmar={wf_C['avg_calmar']:.3f}  WF={wf_C['n_pass']}/5")
    w(f"- Best this sweep (Case {best}): CAGR={wf_B['avg_cagr']:+.2f}%  "
      f"Calmar={wf_B['avg_calmar']:.3f}  WF={wf_B['n_pass']}/5  "
      f"Calmar_std={wf_B['calmar_std']:.3f}")
    w(f"- 停止条件: gap={marginal['gap']:+.4f}  → **{'TRIGGERED' if stop else 'NOT triggered'}**\n")

    # ── S2. WF Summary ────────────────────────────────────────────────
    w("---\n## 2. WF サマリ (slope sweep)\n")
    w("| Case | slope_max | WF | CAGR | DD | Calmar | Calmar_std | PF | hit% | trig/yr | accept% | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        spec = CASES[c]
        wf   = case_results[c]["wf"]
        ar   = case_results[c]["accept_rate"] * 100
        sl   = f"≤{spec.slope_max:.0f}" if spec.slope_max is not None else "—"
        mark = "**ADOPT**" if wf.get("adopted") else "reject"
        w(f"| {c} | {sl} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_cagr',0):+.2f}% "
          f"| {wf.get('avg_dd',0):.2f}% "
          f"| {fp(wf.get('avg_calmar'))} "
          f"| {fp(wf.get('calmar_std'))} "
          f"| {fp(wf.get('avg_pf'),'.3f')} "
          f"| {fp(wf.get('avg_hit'),'.1f')} "
          f"| {fp(wf.get('avg_trig'),'.1f')} "
          f"| {ar:.1f}% "
          f"| {mark} |")

    # ── S3. Marginal Calmar Gain ──────────────────────────────────────
    w("\n---\n## 3. Marginal Calmar Gain\n")
    w("> 隣接Case間 Calmar差分  |  gain_per_rej = ΔCalmar / n_rejected × 100\n")
    w("| Case | slope | Calmar | ΔCalmar_vs_prev | ΔCalmar_vs_F | gain_per_rej | accept% |")
    w("|---|---|---|---|---|---|---|")
    calmar_F_val = case_results["F"]["wf"].get("avg_calmar", 0.0)
    for c in CASE_ORDER:
        spec = CASES[c]
        wf   = case_results[c]["wf"]
        sl   = f"≤{spec.slope_max:.0f}" if spec.slope_max is not None else "—"
        cal  = wf.get("avg_calmar", 0.0)
        d_f  = round(cal - calmar_F_val, 4)
        d_p  = marginal["marginal_gain"].get(c, 0.0) if c != "F" else 0.0
        gpr  = marginal["gain_per_rej"].get(c, float("nan"))
        ar   = case_results[c]["accept_rate"] * 100
        w(f"| {c} | {sl} "
          f"| {fp(cal)} "
          f"| {d_p:+.4f} "
          f"| {d_f:+.4f} "
          f"| {fp(gpr, '.4f')} "
          f"| {ar:.1f}% |")

    w(f"\n**Best Case {best}**: Calmar={marginal['best_calmar']:.3f}  "
      f"Neighbor={marginal['neighbor_calmar']:.3f}  "
      f"Gap={marginal['gap']:+.4f}")
    if stop:
        w(f"\n→ Gap < {STOP_GAP}: **slope研究終了**  "
          f"slope閾値の選択に構造的根拠なし\n")
    else:
        w(f"\n→ Gap >= {STOP_GAP}: **slope改善は構造的**  "
          f"Case {best} を推奨閾値として採用\n")

    # ── S4. Fold Detail ───────────────────────────────────────────────
    w("\n---\n## 4. Fold 詳細\n")
    w("| Case | Fold | Regime | CAGR | DD | Calmar | PF | hit% | avg_hold | trig/yr |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        for fold, fm in zip(FOLDS, case_results[c]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            w(f"| {c} | Fold{fold['id']} | {reg} "
              f"| {fm['cagr']:+.2f}% "
              f"| {fm['max_dd']:.2f}% "
              f"| {fp(fm['calmar'])} "
              f"| {fp(fm['pf'],'.3f')} "
              f"| {fm['hit_rate']:.1f}% "
              f"| {fm['avg_hold']:.1f}d "
              f"| {fm['trig_yr']:.1f} |")

    # ── S5. Calmar Dispersion ─────────────────────────────────────────
    w("\n---\n## 5. Calmar 分散 vs Case F\n")
    w("| Case | avg_Calmar | std_Calmar | std改善 | ΔCalmar_F | WF |")
    w("|---|---|---|---|---|---|")
    std_F = case_results["F"]["wf"].get("calmar_std", 9999)
    for c in CASE_ORDER:
        wf  = case_results[c]["wf"]
        std = wf.get("calmar_std", 0)
        imp = "✅" if std < std_F else ("=" if abs(std - std_F) < 0.001 else "❌")
        d_f = round(wf.get("avg_calmar", 0) - calmar_F_val, 4)
        w(f"| {c} "
          f"| {fp(wf.get('avg_calmar'))} "
          f"| {fp(std)} "
          f"| {imp} "
          f"| {d_f:+.4f} "
          f"| {wf.get('n_pass',0)}/5 |")

    # ── S6. Equivalence Note ─────────────────────────────────────────
    w("\n---\n## 6. 等価ケース分析\n")
    # find groups of identical calmar
    calmar_by_c = marginal["calmar_by_case"]
    groups: dict[float, list] = {}
    for c in CASE_ORDER:
        v = round(calmar_by_c[c], 4)
        groups.setdefault(v, []).append(c)
    identical = {k: v for k, v in groups.items() if len(v) > 1}
    if identical:
        w("> 以下のケースは完全同一 (slope範囲内に実データなし)\n")
        for v, grp in identical.items():
            w(f"- **{', '.join(grp)}** → Calmar={v:.4f}  accept={case_results[grp[0]]['accept_rate']*100:.1f}%")
        w("")
    else:
        w("> 全ケース異なる結果\n")

    # ── S7. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 7. Final Recommendation\n")
    w(f"### {verdict}\n")

    # best ADOPTED case (WF>=4/5)
    adopted_cases = [c for c in CASE_ORDER if case_results[c]["wf"].get("adopted")]
    best_adopted  = (max(adopted_cases,
                         key=lambda c: case_results[c]["wf"].get("avg_calmar", 0))
                     if adopted_cases else None)

    if stop:
        w(f"**停止条件発動**: best(Case{best}) − neighbor gap = {marginal['gap']:+.4f} < {STOP_GAP}\n")
        w("slope閾値の選択に統計的有意差なし。slope研究終了。\n")
        w("推奨:")
        w(f"- slope フィルターを採用する場合は **Case {best}** (Calmar最大) で打ち止め")
        w(f"- ただし隣接との差分が軽微なため、slope=6 (前回採用) を維持でもよい")
        w(f"- 次ステップ: d90パラメータ or RSR上限の掘り下げへ移行\n")
    else:
        w(f"**slope改善は構造的**: gap = {marginal['gap']:+.4f} >= {STOP_GAP}\n")
        if best_adopted:
            wf_ba = case_results[best_adopted]["wf"]
            spec_ba = CASES[best_adopted]
            w(f"**推奨: Case {best_adopted}** (slope_max={spec_ba.slope_max})  "
              f"← WF採用済み最良\n")
            w(f"- CAGR={wf_ba['avg_cagr']:+.2f}%  DD={wf_ba['avg_dd']:.2f}%  "
              f"Calmar={wf_ba['avg_calmar']:.3f}  WF={wf_ba['n_pass']}/5")
            w(f"- accept_rate={case_results[best_adopted]['accept_rate']*100:.1f}%  "
              f"Calmar_std={wf_ba['calmar_std']:.3f}")
            w(f"- ΔCalmar vs F: {round(wf_ba['avg_calmar']-calmar_F_val,4):+.4f}\n")
            if best != best_adopted:
                wf_b = case_results[best]["wf"]
                w(f"⚠ raw best = Case {best} (Calmar={wf_b['avg_calmar']:.3f}) "
                  f"だが WF={wf_b['n_pass']}/5 < {ADOPT_WF} で不採用\n")
        # note if B=C=D equivalent
        if identical:
            for v, grp in identical.items():
                if best_adopted in grp and len(grp) > 1:
                    slopevals = [CASES[c].slope_max for c in grp if CASES[c].slope_max is not None]
                    if slopevals:
                        lo, hi = min(slopevals), max(slopevals)
                        w(f"**分布発見**: slope範囲({lo:.0f},{hi:.0f}]に実エントリーデータなし "
                          f"→ slope≤{lo:.0f} が実質的な境界\n")
        w("実装優先度 (ASK_FIRST 必須):")
        w("1. signal_bridge.py エントリー条件に slope5 フィルター追加")
        w("2. live dry-run 30日 → 本番移行")

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
    print("  Entry Velocity Sweep WF  (slope5 threshold sweep)")
    print(f"  ENTRY=RSR[{RSR_LO:.0f},{RSR_HI:.0f}] d90≤{D90_MAX}  "
          f"EXIT=RSR<{EXIT_THR:.0f}  MIN_HOLD={MIN_HOLD}d")
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

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                             dtype=np.float32, fill_value=1.0))

    mkt_ret1 = None
    if topix_close is not None:
        mkt_ret1 = _take(topix_close.pct_change(), common_dates,
                          dtype=np.float32, fill_value=0.0)

    cross90_mat = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat  = compute_rsr_slope5(rsr_mat)
    capital     = float(cfg.portfolio.capital)

    print("[3/4] Case F (no slope) ベースライン...")
    res_F = run_standalone(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90_mat, slope5_mat, mkt_ret1,
        active_syms, sym_to_i, common_dates, capital, CASES["F"],
    )
    folds_F_m = [compute_fold_metrics(res_F, common_dates, f, capital) for f in FOLDS]
    calmar_F_per_fold = [fm.get("calmar", 0.0) if fm else 0.0 for fm in folds_F_m]

    print("[4/4] Slope sweep (6 cases)...\n")
    case_results: dict = {}

    for c, spec in CASES.items():
        print(f"  [{c}] {spec.label}")
        if c == "F":
            res = res_F
        else:
            res = run_standalone(
                open_mat, close_mat, rsr_mat, sym_active_mat,
                cross90_mat, slope5_mat, mkt_ret1,
                active_syms, sym_to_i, common_dates, capital, spec,
            )

        folds_m = [
            compute_fold_metrics(res, common_dates, f, capital)
            for f in FOLDS
        ]
        wf = aggregate_wf(folds_m, calmar_F_per_fold)

        case_results[c] = {
            "folds":         folds_m,
            "wf":            wf,
            "accept_rate":   res["accept_rate"],
            "cand_accepted": res["cand_accepted"],
        }

        for fold, fm in zip(FOLDS, folds_m):
            if fm:
                cf = calmar_F_per_fold[FOLDS.index(fold)]
                ok = "✅" if fm.get("calmar", 0) >= cf else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"CAGR={fm['cagr']:+.2f}%  Calmar={fm['calmar']:.3f}"
                      f"(F={cf:.3f})  {ok}")
        ok_s = "✅ ADOPTED" if wf.get("adopted") else "❌"
        print(f"    → WF={wf.get('n_pass',0)}/5  "
              f"Calmar={wf.get('avg_calmar',0):.3f}  "
              f"std={wf.get('calmar_std',0):.3f}  "
              f"accept={res['accept_rate']*100:.1f}%  {ok_s}\n")

    # ── Marginal analysis ──────────────────────────────────────────────
    marginal = compute_marginal(case_results, res_F["cand_accepted"])

    best  = marginal["best_case"]
    gap   = marginal["gap"]
    stop  = marginal["stop_triggered"]

    print("=" * 68)
    print(f"  slope sweep 結果:")
    for c in CASE_ORDER:
        wf   = case_results[c]["wf"]
        mark = "✅" if wf.get("adopted") else "❌"
        mg   = marginal["marginal_gain"].get(c, 0.0)
        print(f"  {mark} Case{c}(slope={CASES[c].slope_max}): "
              f"Calmar={wf['avg_calmar']:.3f}  "
              f"std={wf['calmar_std']:.3f}  "
              f"Δprev={mg:+.4f}  "
              f"accept={case_results[c]['accept_rate']*100:.1f}%")
    print(f"\n  Best=Case{best}  neighbor={marginal['neighbor_calmar']:.3f}  "
          f"gap={gap:+.4f}")
    print(f"  停止条件: {'TRIGGERED' if stop else 'NOT triggered'}  "
          f"→ {'slope研究終了' if stop else 'slope改善は構造的'}")
    print("=" * 68 + "\n")

    write_report(
        case_results, marginal, topix_close,
        REPORTS_DIR / "entry_velocity_sweep.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
