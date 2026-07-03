"""
backtest/entry_velocity_attribution.py  —  Entry Velocity Attribution

目的: slope>5 の悪化が普遍的か特定条件集中かを判定

方法: 全シグナル候補(RSR[92,94] d90<=5)を列挙し、
      slope<=5 (group1) vs slope>5 (group2) に分割して
      forward return・失敗集中度・条件付き確率を計測

判定:
  GLOBAL_FILTER   slope>5 は全 regime で悪化
  REGIME_FILTER   slope>5 は特定 regime のみ悪化
  DATA_SPARSE     group2 が n<20 → 結論不能

出力: reports/entry_velocity_attribution.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_velocity_attribution.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    from scipy.stats import ks_2samp as _ks2
    _SCIPY = True
except ImportError:
    _SCIPY = False

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, _take,
    LOT, COST_ONE_WAY,
)

REPORTS_DIR  = Path("reports")
IS_START     = "2018-01-01"

YEARS = [2021, 2022, 2023, 2024, 2025]

# ── Fixed entry params ─────────────────────────────────────────────────
RSR_LO      = 92.0
RSR_HI      = 94.0   # inclusive
D90_MAX     = 5
SLOPE_SPLIT = 5.0    # group1 = slope<=5, group2 = slope>5

# ── Regime thresholds (TOPIX 20d return) ──────────────────────────────
BULL_THR    =  0.03
BEAR_THR    = -0.03

# ── Verdict thresholds ────────────────────────────────────────────────
DATA_SPARSE_N = 20     # group2 < N → DATA_SPARSE


# ──────────────────────────────────────────────────────────────────────
#  PRE-COMPUTE HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_cross_mat(rsr_mat: np.ndarray, thr: float) -> np.ndarray:
    n, m    = rsr_mat.shape
    out     = np.zeros((n, m), dtype=np.int32)
    running = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = running
        above   = rsr_mat[i] >= thr
        running[above] += 1
        running[~above] = 0
    return out


def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


# ──────────────────────────────────────────────────────────────────────
#  CANDIDATE ENUMERATION
# ──────────────────────────────────────────────────────────────────────

def enumerate_candidates(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, topix_ret20,
    active_syms, sym_to_i, sym_sector_map,
    common_dates,
) -> list[dict]:
    """
    RSR[92,94] d90<=5 の全シグナル候補を列挙。
    forward return は open[nxt] から計測 (entry価格と同一)。
    """
    n      = len(common_dates)
    cands  = []
    date_strs = [str(d.date()) for d in common_dates]
    year_arr  = np.array([d.year for d in common_dates])

    for i in range(n - 21):   # need 20 fwd days
        ds   = date_strs[i]
        nxt  = i + 1
        yr   = int(year_arr[i])

        r20 = float(topix_ret20[i]) if topix_ret20 is not None else 0.0
        regime = ("bull" if r20 > BULL_THR else
                  "bear" if r20 < BEAR_THR else "sideways")

        for sym in active_syms:
            si  = sym_to_i[sym]
            rv  = float(rsr_mat[i, si])
            d90 = int(cross90_mat[i, si])
            sl5 = float(slope5_mat[i, si])

            if not (RSR_LO <= rv <= RSR_HI and d90 <= D90_MAX):
                continue
            if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                continue

            bp = float(open_mat[nxt, si])
            if bp <= 0 or math.isnan(bp):
                continue

            # forward returns from entry open (analytics only — no lookahead in live)
            fwd10 = float(close_mat[min(i + 10, n - 1), si]) / bp - 1
            fwd20 = float(close_mat[min(i + 20, n - 1), si]) / bp - 1
            if math.isnan(fwd10): fwd10 = 0.0
            if math.isnan(fwd20): fwd20 = 0.0

            group = 1 if sl5 <= SLOPE_SPLIT else 2
            sector = sym_sector_map.get(sym, "不明")

            cands.append({
                "date":    ds,
                "year":    yr,
                "symbol":  sym,
                "sector":  sector,
                "slope5":  round(sl5, 2),
                "rsr":     round(rv, 1),
                "d90":     d90,
                "fwd10":   round(fwd10 * 100, 2),
                "fwd20":   round(fwd20 * 100, 2),
                "regime":  regime,
                "group":   group,
            })

    return cands


# ──────────────────────────────────────────────────────────────────────
#  STATISTICS
# ──────────────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2: return 0.0
    n1, n2 = len(a), len(b)
    var1   = np.var(a, ddof=1); var2 = np.var(b, ddof=1)
    pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return round((float(np.mean(a)) - float(np.mean(b))) / pooled, 4) if pooled > 0 else 0.0


def distribution_overlap(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    if len(a) == 0 or len(b) == 0: return 0.0
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi == lo: return 1.0
    h1, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    h2, _     = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    width = edges[1] - edges[0]
    return round(float(np.sum(np.minimum(h1, h2)) * width), 4)


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if _SCIPY and len(a) >= 2 and len(b) >= 2:
        stat, pval = _ks2(a, b)
        return round(float(stat), 4), round(float(pval), 6)
    # manual: max |F1(x) - F2(x)| over merged sorted values
    if len(a) < 2 or len(b) < 2: return (0.0, 1.0)
    all_x = np.sort(np.concatenate([a, b]))
    f1 = np.searchsorted(np.sort(a), all_x, side="right") / len(a)
    f2 = np.searchsorted(np.sort(b), all_x, side="right") / len(b)
    stat = float(np.max(np.abs(f1 - f2)))
    n    = len(a) * len(b) / (len(a) + len(b))
    pval = math.exp(-2 * n * stat ** 2)  # approximate
    return round(stat, 4), round(pval, 6)


def profit_factor(rets: np.ndarray) -> float:
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    g = float(np.sum(wins))
    l = float(np.abs(np.sum(losses)))
    return round(g / l, 3) if l > 0 else float("inf")


def group_stats(rets20: np.ndarray, rets10: np.ndarray) -> dict:
    if len(rets20) == 0:
        return {k: float("nan") for k in
                ["n", "avg_ret10", "avg_ret20", "win_rate", "pf",
                 "max_dd_proxy", "ret_std"]}
    win = np.sum(rets20 > 0)
    return {
        "n":          len(rets20),
        "avg_ret10":  round(float(np.mean(rets10)), 2),
        "avg_ret20":  round(float(np.mean(rets20)), 2),
        "win_rate":   round(float(win / len(rets20)) * 100, 1),
        "pf":         profit_factor(rets20),
        "max_dd_proxy": round(float(np.percentile(rets20, 5)), 2),   # P5 as proxy
        "ret_std":    round(float(np.std(rets20)), 2),
    }


def regime_stats(cands: list[dict], group: int) -> dict:
    g    = [c for c in cands if c["group"] == group]
    out  = {}
    for reg in ["bull", "bear", "sideways"]:
        r20 = np.array([c["fwd20"] for c in g if c["regime"] == reg])
        r10 = np.array([c["fwd10"] for c in g if c["regime"] == reg])
        if len(r20) >= 2:
            out[reg] = group_stats(r20, r10)
        else:
            out[reg] = {"n": len(r20), "avg_ret20": float("nan")}
    return out


# ──────────────────────────────────────────────────────────────────────
#  FAILURE CONCENTRATION
# ──────────────────────────────────────────────────────────────────────

def failure_concentration(g2: list[dict]) -> dict:
    """Group2 (slope>5) の損失集中度分析"""
    losses = [c for c in g2 if c["fwd20"] < 0]
    if not losses:
        return {"top10_loss_share": 0.0, "hhi_year": 0.0, "hhi_regime": 0.0}

    total_loss = sum(c["fwd20"] for c in losses)   # sum of negatives
    sorted_loss = sorted(losses, key=lambda x: x["fwd20"])  # worst first
    top10 = sorted_loss[:10]
    top10_share = round(sum(c["fwd20"] for c in top10) / total_loss, 4) if total_loss != 0 else 0.0

    # HHI by year
    year_counts = defaultdict(int)
    for c in losses: year_counts[c["year"]] += 1
    total = sum(year_counts.values())
    hhi_yr = round(sum((v / total) ** 2 for v in year_counts.values()), 4)

    # HHI by regime
    reg_counts = defaultdict(int)
    for c in losses: reg_counts[c["regime"]] += 1
    hhi_reg = round(sum((v / total) ** 2 for v in reg_counts.values()), 4)

    year_breakdown = {str(yr): year_counts.get(yr, 0) for yr in YEARS}
    reg_breakdown  = {r: reg_counts.get(r, 0) for r in ["bull", "bear", "sideways"]}

    return {
        "n_losses":        len(losses),
        "total_loss_ret":  round(total_loss, 2),
        "top10_loss_share": top10_share,
        "hhi_year":        hhi_yr,
        "hhi_regime":      hhi_reg,
        "year_breakdown":  year_breakdown,
        "regime_breakdown":reg_breakdown,
        "top3_worst":      [{"sym": c["symbol"], "fwd20": c["fwd20"],
                             "regime": c["regime"], "year": c["year"],
                             "slope5": c["slope5"]}
                            for c in sorted_loss[:3]],
    }


# ──────────────────────────────────────────────────────────────────────
#  CONDITIONAL PROBABILITIES
# ──────────────────────────────────────────────────────────────────────

def conditional_probs(cands: list[dict]) -> dict:
    total  = len(cands)
    g1     = [c for c in cands if c["group"] == 1]
    g2     = [c for c in cands if c["group"] == 2]
    bears  = [c for c in cands if c["regime"] == "bear"]
    g2_bear = [c for c in g2 if c["regime"] == "bear"]

    def safe_rate(num, den):
        return round(len(num) / max(1, len(den)) * 100, 1)

    p_g2            = safe_rate(g2, cands)
    p_g2_bear       = safe_rate(g2_bear, bears) if bears else float("nan")
    p_loss_g2       = safe_rate([c for c in g2 if c["fwd20"] < 0], g2)
    p_loss_g2_bear  = safe_rate([c for c in g2_bear if c["fwd20"] < 0], g2_bear)
    p_loss_g1       = safe_rate([c for c in g1 if c["fwd20"] < 0], g1)

    # per-year P(group2)
    py_g2 = {}
    for yr in YEARS:
        yr_all = [c for c in cands if c["year"] == yr]
        yr_g2  = [c for c in yr_all  if c["group"] == 2]
        py_g2[str(yr)] = safe_rate(yr_g2, yr_all) if yr_all else float("nan")

    return {
        "P(g2)":             p_g2,
        "P(g2|bear)":        p_g2_bear,
        "P(loss|g2)":        p_loss_g2,
        "P(loss|g2,bear)":   p_loss_g2_bear,
        "P(loss|g1)":        p_loss_g1,
        "P(g2|fold)":        py_g2,
        "n_total":           total,
        "n_g1":              len(g1),
        "n_g2":              len(g2),
    }


# ──────────────────────────────────────────────────────────────────────
#  SECTOR & MONTHLY DENSITY
# ──────────────────────────────────────────────────────────────────────

def sector_analysis(g1: list[dict], g2: list[dict]) -> dict:
    def sector_dist(g):
        counts = defaultdict(int)
        for c in g: counts[c["sector"]] += 1
        total = max(1, sum(counts.values()))
        return {s: round(v / total * 100, 1) for s, v in
                sorted(counts.items(), key=lambda x: -x[1])}
    return {"g1": sector_dist(g1), "g2": sector_dist(g2)}


def monthly_density(cands: list[dict]) -> dict:
    g1_mo = defaultdict(int)
    g2_mo = defaultdict(int)
    for c in cands:
        mo = c["date"][5:7]   # "01".."12"
        if c["group"] == 1: g1_mo[mo] += 1
        else:                g2_mo[mo] += 1
    months = [f"{m:02d}" for m in range(1, 13)]
    g1_tot = max(1, sum(g1_mo.values()))
    g2_tot = max(1, sum(g2_mo.values()))
    return {
        "g1": {m: round(g1_mo[m] / g1_tot * 100, 1) for m in months},
        "g2": {m: round(g2_mo[m] / g2_tot * 100, 1) for m in months},
    }


# ──────────────────────────────────────────────────────────────────────
#  VERDICT
# ──────────────────────────────────────────────────────────────────────

def compute_verdict(
    g1_st: dict, g2_st: dict,
    reg1: dict, reg2: dict,
    cprob: dict,
    n_g2: int,
) -> dict:
    if n_g2 < DATA_SPARSE_N:
        return {"verdict": "DATA_SPARSE", "reason": f"n_g2={n_g2} < {DATA_SPARSE_N}"}

    avg1 = g1_st.get("avg_ret20", 0.0)
    avg2 = g2_st.get("avg_ret20", 0.0)
    g2_worse_overall = avg2 < avg1

    # check each regime
    regimes_g2_worse = []
    for reg in ["bull", "bear", "sideways"]:
        r1 = reg1.get(reg, {}).get("avg_ret20", float("nan"))
        r2 = reg2.get(reg, {}).get("avg_ret20", float("nan"))
        n2 = reg2.get(reg, {}).get("n", 0)
        if n2 < 5 or math.isnan(r1) or math.isnan(r2): continue
        if r2 < r1:
            regimes_g2_worse.append(reg)

    if len(regimes_g2_worse) >= 2:
        verdict = "GLOBAL_FILTER"
        reason  = f"group2 worse in {len(regimes_g2_worse)}/3 regimes: {regimes_g2_worse}"
    elif len(regimes_g2_worse) == 1:
        verdict = "REGIME_FILTER"
        reason  = f"group2 worse only in {regimes_g2_worse[0]}"
    else:
        verdict = "DATA_SPARSE"
        reason  = "group2 not worse in any regime (insufficient evidence)"

    return {"verdict": verdict, "reason": reason,
            "regimes_g2_worse": regimes_g2_worse}


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    g1_st: dict, g2_st: dict,
    reg1: dict, reg2: dict,
    year_g1: dict, year_g2: dict,
    ks_stat: float, ks_pval: float,
    overlap: float,
    effect_d: float,
    fc: dict,
    cprob: dict,
    secs: dict,
    monthly: dict,
    verdict_info: dict,
    path: Path,
) -> None:
    L = []; w = L.append
    verd  = verdict_info["verdict"]
    g1_n  = g1_st.get("n", 0)
    g2_n  = g2_st.get("n", 0)
    total = g1_n + g2_n

    def ff(v, fmt="+.2f"):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
        return format(v, fmt)
    def pct(v): return ff(v, "+.2f") + "%" if not (v is None or (isinstance(v, float) and math.isnan(v))) else "—"
    def f3(v):  return ff(v, ".3f")

    w("# Entry Velocity Attribution")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n対象: RSR[{RSR_LO:.0f},{RSR_HI:.0f}] d90≤{D90_MAX}  分割: slope≤{SLOPE_SPLIT:.0f} vs slope>{SLOPE_SPLIT:.0f}")
    w(f"\nN_total={total}  N_group1(slope≤5)={g1_n}  N_group2(slope>5)={g2_n}\n")
    w(f"**判定: {verd}**  — {verdict_info['reason']}\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    w(f"| 指標 | group1 (slope≤5) | group2 (slope>5) | 差分 |")
    w(f"|---|---|---|---|")
    for k, label in [("n","N"), ("avg_ret10","avg_ret10"),
                     ("avg_ret20","avg_ret20"), ("win_rate","win_rate%"),
                     ("pf","PF"), ("max_dd_proxy","P5_ret(DD proxy)"),
                     ("ret_std","ret_std")]:
        v1 = g1_st.get(k, float("nan"))
        v2 = g2_st.get(k, float("nan"))
        diff = ""
        if not (math.isnan(float(v1)) if isinstance(v1,float) else False) and \
           not (math.isnan(float(v2)) if isinstance(v2,float) else False):
            try: diff = f"{float(v2)-float(v1):+.2f}"
            except: diff = "—"
        w(f"| {label} | {v1} | {v2} | {diff} |")

    w(f"\n統計: KS={ks_stat:.4f}(p={ks_pval:.4f})  Cohen_d={effect_d:+.4f}  overlap={overlap:.4f}")
    if ks_pval < 0.05:
        w(f"  → 両群の分布は有意に異なる (p<0.05)\n")
    else:
        w(f"  → 有意差なし (p={ks_pval:.4f} ≥ 0.05)\n")

    # ── S2. Regime Breakdown ─────────────────────────────────────────
    w("---\n## 2. Regime別 avg_ret20\n")
    w("| Regime | g1_n | g1_ret20 | g2_n | g2_ret20 | 差分 | g2悪化 |")
    w("|---|---|---|---|---|---|---|")
    for reg in ["bull", "bear", "sideways"]:
        r1 = reg1.get(reg, {}); r2 = reg2.get(reg, {})
        v1 = r1.get("avg_ret20", float("nan"))
        v2 = r2.get("avg_ret20", float("nan"))
        n1 = r1.get("n", 0);   n2 = r2.get("n", 0)
        diff = round(v2 - v1, 2) if not (math.isnan(v1) or math.isnan(v2)) else float("nan")
        worse = "✅" if not math.isnan(diff) and diff < 0 else ""
        w(f"| {reg} | {n1} | {pct(v1)} | {n2} | {pct(v2)} "
          f"| {ff(diff,'+.2f')+('%' if not math.isnan(diff) else '')} | {worse} |")

    # ── S3. Year Breakdown ────────────────────────────────────────────
    w("\n---\n## 3. 年別 avg_ret20 & count\n")
    w("| year | g1_n | g1_ret20 | g1_win% | g2_n | g2_ret20 | g2_win% | 差分 |")
    w("|---|---|---|---|---|---|---|---|")
    for yr in YEARS:
        ys1 = year_g1.get(yr, {}); ys2 = year_g2.get(yr, {})
        v1  = ys1.get("avg_ret20", float("nan"))
        v2  = ys2.get("avg_ret20", float("nan"))
        d   = round(v2 - v1, 2) if not (math.isnan(v1) or math.isnan(v2)) else float("nan")
        w(f"| {yr} | {ys1.get('n',0)} | {pct(v1)} | {ff(ys1.get('win_rate',0),'.1f')}% "
          f"| {ys2.get('n',0)} | {pct(v2)} | {ff(ys2.get('win_rate',0),'.1f')}% "
          f"| {ff(d,'+.2f')+('%' if not math.isnan(d) else '')} |")

    # ── S4. Distribution Analysis ─────────────────────────────────────
    w("\n---\n## 4. 分布解析\n")
    w(f"| 指標 | 値 |")
    w(f"|---|---|")
    w(f"| KS statistic | {ks_stat:.4f} |")
    w(f"| KS p-value | {ks_pval:.6f} |")
    w(f"| Cohen's d (g1−g2) | {effect_d:+.4f} |")
    w(f"| distribution overlap | {overlap:.4f} |")
    interp = ("large" if abs(effect_d) >= 0.8 else
              "medium" if abs(effect_d) >= 0.5 else
              "small" if abs(effect_d) >= 0.2 else "negligible")
    w(f"| effect size 解釈 | {interp} |")
    w(f"| 有意差 (p<0.05) | {'YES' if ks_pval < 0.05 else 'NO'} |")

    # ── S5. Failure Concentration ─────────────────────────────────────
    w("\n---\n## 5. 失敗集中度 (group2)\n")
    w(f"| 指標 | 値 |")
    w(f"|---|---|")
    w(f"| n_losses (group2) | {fc.get('n_losses',0)} |")
    w(f"| top10_loss_share | {fc.get('top10_loss_share',0):.2%} |")
    w(f"| HHI 年別集中度 | {fc.get('hhi_year',0):.4f} |")
    w(f"| HHI regime別集中度 | {fc.get('hhi_regime',0):.4f} |")
    yr_bk = fc.get("year_breakdown", {})
    reg_bk = fc.get("regime_breakdown", {})
    w(f"| 損失年別内訳 | {', '.join(f'{k}:{v}件' for k,v in yr_bk.items() if v>0)} |")
    w(f"| 損失regime別内訳 | {', '.join(f'{k}:{v}件' for k,v in reg_bk.items() if v>0)} |")
    w(f"\n**Top-3 最大損失 (group2)**\n")
    w("| rank | symbol | slope5 | regime | year | fwd20 |")
    w("|---|---|---|---|---|---|")
    for i, t in enumerate(fc.get("top3_worst", []), 1):
        w(f"| {i} | {t['sym']} | {t['slope5']:.1f} | {t['regime']} "
          f"| {t['year']} | {t['fwd20']:+.2f}% |")

    # ── S6. Conditional Probabilities ────────────────────────────────
    w("\n---\n## 6. 条件付き確率\n")
    w("| 確率 | 値 |")
    w("|---|---|")
    for k in ["P(g2)", "P(g2|bear)", "P(loss|g2)", "P(loss|g2,bear)", "P(loss|g1)"]:
        v = cprob.get(k, float("nan"))
        w(f"| {k} | {ff(v,'.1f')}% |")
    w(f"\n**P(group2 | fold)**\n")
    w("| fold | P(g2|fold)% |")
    w("|---|---|")
    for yr, v in cprob.get("P(g2|fold)", {}).items():
        w(f"| {yr} | {ff(v,'.1f')}% |")

    # ── S7. Sector Mix ────────────────────────────────────────────────
    w("\n---\n## 7. Sector Mix\n")
    all_sectors = sorted(set(list(secs["g1"].keys()) + list(secs["g2"].keys())))
    w("| sector | g1% | g2% | 偏差 |")
    w("|---|---|---|---|")
    for s in all_sectors:
        v1 = secs["g1"].get(s, 0.0)
        v2 = secs["g2"].get(s, 0.0)
        w(f"| {s} | {v1:.1f}% | {v2:.1f}% | {v2-v1:+.1f}pp |")

    # ── S8. Monthly Density ───────────────────────────────────────────
    w("\n---\n## 8. 月別エントリー密度\n")
    month_names = {
        "01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"May","06":"Jun",
        "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec",
    }
    w("| month | g1% | g2% | 差 |")
    w("|---|---|---|---|")
    for m in [f"{i:02d}" for i in range(1, 13)]:
        v1 = monthly["g1"].get(m, 0.0)
        v2 = monthly["g2"].get(m, 0.0)
        w(f"| {month_names[m]} | {v1:.1f}% | {v2:.1f}% | {v2-v1:+.1f}pp |")

    # ── S9. Verdict ───────────────────────────────────────────────────
    w("\n---\n## 9. 最終判定\n")
    w(f"### **{verd}**\n")
    w(f"{verdict_info['reason']}\n")

    if verd == "GLOBAL_FILTER":
        w("**slope>5 エントリーは全 regime で劣化 → 普遍的フィルター確認**\n")
        w(f"- group2 avg_ret20={g2_st.get('avg_ret20',0):+.2f}%  "
          f"vs group1={g1_st.get('avg_ret20',0):+.2f}%  "
          f"差={g2_st.get('avg_ret20',0)-g1_st.get('avg_ret20',0):+.2f}pp")
        w(f"- KS p={ks_pval:.4f}  Cohen_d={effect_d:+.4f}({interp})")
        w(f"- P(loss|g2)={cprob.get('P(loss|g2)',0):.1f}%  "
          f"P(loss|g1)={cprob.get('P(loss|g1)',0):.1f}%\n")
        w("推奨: **slope≤5 を恒常フィルターとして実装** (ASK_FIRST)")
        w("実装: signal_bridge.py エントリー条件 `rsr_slope_5d <= 5`")

    elif verd == "REGIME_FILTER":
        bad_reg = verdict_info.get("regimes_g2_worse", [])
        w(f"**slope>5 の悪化は {bad_reg} のみ → regime条件付きフィルター推奨**\n")
        for reg in bad_reg:
            r1v = reg1.get(reg, {}).get("avg_ret20", float("nan"))
            r2v = reg2.get(reg, {}).get("avg_ret20", float("nan"))
            w(f"- {reg}: g1={pct(r1v)}  g2={pct(r2v)}  "
              f"差={r2v-r1v:+.2f}pp")
        w(f"\n- KS p={ks_pval:.4f}  Cohen_d={effect_d:+.4f}({interp})\n")
        w("推奨: Bear相場のみ slope≤5 フィルター適用 (ASK_FIRST)")

    else:  # DATA_SPARSE or no-worse
        w(f"n_g2={g2_n}  → {'データ不足、結論不能' if verd=='DATA_SPARSE' else 'slope>5 の影響が判定困難'}\n")
        w("推奨: slope フィルターの効果不明確 → 保守的に slope≤5 維持 (前回採用値)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Entry Velocity Attribution")
    print(f"  RSR[{RSR_LO:.0f},{RSR_HI:.0f}] d90≤{D90_MAX}  "
          f"split=slope≤{SLOPE_SPLIT:.0f} vs slope>{SLOPE_SPLIT:.0f}")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms     = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms    = list(trade_syms.keys())
    sym_to_i       = {s: i for i, s in enumerate(active_syms)}
    sym_sector_map = dict(trade_syms)
    n_syms         = len(active_syms)

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

    topix_ret20 = None
    if topix_close is not None:
        topix_ret20 = _take(topix_close.pct_change(20), common_dates,
                             dtype=np.float32, fill_value=0.0)

    cross90_mat = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat  = compute_slope5(rsr_mat)

    print("[3/4] 候補列挙中 (全 RSR[92,94] d90≤5 シグナル)...")
    cands = enumerate_candidates(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90_mat, slope5_mat, topix_ret20,
        active_syms, sym_to_i, sym_sector_map,
        common_dates,
    )

    g1 = [c for c in cands if c["group"] == 1]
    g2 = [c for c in cands if c["group"] == 2]
    print(f"  全候補: {len(cands)}件  group1(slope≤5): {len(g1)}件  "
          f"group2(slope>5): {len(g2)}件")
    print(f"  accept% group2: {len(g2)/max(1,len(cands))*100:.1f}%")

    print("[4/4] 統計計算中...")
    r1_20 = np.array([c["fwd20"] for c in g1])
    r1_10 = np.array([c["fwd10"] for c in g1])
    r2_20 = np.array([c["fwd20"] for c in g2])
    r2_10 = np.array([c["fwd10"] for c in g2])

    g1_st = group_stats(r1_20, r1_10)
    g2_st = group_stats(r2_20, r2_10)

    reg1 = regime_stats(cands, 1)
    reg2 = regime_stats(cands, 2)

    # per-year stats
    def year_stats(g):
        out = {}
        for yr in YEARS:
            sub = [c for c in g if c["year"] == yr]
            r20 = np.array([c["fwd20"] for c in sub])
            r10 = np.array([c["fwd10"] for c in sub])
            out[yr] = group_stats(r20, r10)
        return out
    year_g1 = year_stats(g1)
    year_g2 = year_stats(g2)

    # distribution tests
    ks_stat, ks_pval = ks_test(r1_20, r2_20)
    overlap          = distribution_overlap(r1_20, r2_20)
    effect_d         = cohens_d(r1_20, r2_20)

    fc     = failure_concentration(g2)
    cprob  = conditional_probs(cands)
    secs   = sector_analysis(g1, g2)
    monthly = monthly_density(cands)
    verdict_info = compute_verdict(g1_st, g2_st, reg1, reg2, cprob, len(g2))

    print(f"\n  group1: avg_ret20={g1_st['avg_ret20']:+.2f}%  "
          f"win={g1_st['win_rate']:.1f}%  PF={g1_st['pf']:.3f}")
    print(f"  group2: avg_ret20={g2_st['avg_ret20']:+.2f}%  "
          f"win={g2_st['win_rate']:.1f}%  PF={g2_st['pf']:.3f}")
    print(f"  KS={ks_stat:.4f}(p={ks_pval:.4f})  "
          f"Cohen_d={effect_d:+.4f}  overlap={overlap:.4f}")
    print(f"  P(loss|g2)={cprob['P(loss|g2)']:.1f}%  "
          f"P(loss|g1)={cprob['P(loss|g1)']:.1f}%")
    print(f"\n  判定: **{verdict_info['verdict']}**  {verdict_info['reason']}")

    print("=" * 68 + "\n")

    write_report(
        g1_st, g2_st, reg1, reg2, year_g1, year_g2,
        ks_stat, ks_pval, overlap, effect_d,
        fc, cprob, secs, monthly,
        verdict_info,
        REPORTS_DIR / "entry_velocity_attribution.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
