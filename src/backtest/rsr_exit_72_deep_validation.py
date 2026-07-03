"""
backtest/rsr_exit_72_deep_validation.py

RSR Exit 72 Deep Validation
A=75(Baseline) / B=70 / C=72 の5-Fold WF + 詳細分析

分析1: 5-Fold WF (CAGR/Sharpe/MaxDD/Calmar/PF/Exposure)
分析2: 2022年詳細解析 (保有日数/Exit理由/PnL分布)
分析3: Exit Event Audit (70→72変更で影響された取引)
分析4: Counterfactual Attribution (銘柄別寄与度)
分析5: Stability Analysis (Fold間分散比較)

判定基準:
  72採用: avgCalmar(72)>=avgCalmar(70) AND 2022Calmar(72)>2022Calmar(70) AND Calmar分散(72)<分散(70)
  それ以外: 70採用

Run:
    cd C:/ai-trading
    python src/backtest/rsr_exit_72_deep_validation.py
"""

from __future__ import annotations

import sys, time, warnings, statistics
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

REPORTS_DIR = Path("reports")

# ── 比較閾値 ─────────────────────────────────────────────────────────
THR_A = 75  # Baseline
THR_B = 70  # 前回sweep最良
THR_C = 72  # 今回検証対象

FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022弱気"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

FOLD2_START = "2022-01-01"
FOLD2_END   = "2022-12-31"
FOLD2_IS    = ("2018-01-01", "2021-12-31")


# ── helpers ───────────────────────────────────────────────────────────

def _g(m, key, default=0.0):
    return m.get(key, default)


def _pf_from_trades(trades: list) -> float:
    sells = [t for t in trades if t["side"] == "SELL"]
    wins  = [t for t in sells if (t.get("pnl") or 0) > 0]
    loss  = [t for t in sells if (t.get("pnl") or 0) <= 0]
    gp = sum(t["pnl"] for t in wins) if wins else 0.0
    gl = abs(sum(t["pnl"] for t in loss)) if loss else 0.0
    return round(gp / max(1.0, gl), 3)


def _avg_hold(trades: list) -> float:
    sells = [t for t in trades if t["side"] == "SELL" and "exit_idx" in t]
    h = [t["exit_idx"] - t["entry_idx"] for t in sells]
    return float(np.mean(h)) if h else 0.0


def _median_pnl(trades: list) -> float:
    sells = [t for t in trades if t["side"] == "SELL"]
    pnls  = [t.get("pnl", 0) or 0 for t in sells]
    return float(np.median(pnls)) if pnls else 0.0


def run_fold(fi, thr, universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
             topix_close, rsr_syms, cfg,
             return_trades=False, is_period=False) -> dict:
    fold = FOLDS[fi]
    if is_period:
        start, end = fold[1], fold[2]
    else:
        start, end = fold[3], fold[4]
    return run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=start, end=end,
        pattern="A",
        rsr_exit_override=float(thr),
        return_trades=return_trades,
    )


def run_all(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg) -> dict:
    """
    Returns: results[thr][fi] = {"IS": ..., "Test": ...}
    Also: results2022[thr] = full metrics for 2022 with return_trades=True
    """
    thrs   = [THR_A, THR_B, THR_C]
    res    = {t: [] for t in thrs}
    res22  = {}

    total = len(thrs) * len(FOLDS)
    done  = 0

    for fi in range(len(FOLDS)):
        fold_name = FOLDS[fi][0]
        char      = FOLDS[fi][5]
        print(f"\n── {fold_name} (Test={FOLDS[fi][3][:4]} / {char}) ──")
        for thr in thrs:
            row = {}
            for period, is_p in [("IS", True), ("Test", False)]:
                done += 1
                tag = f"[{done:2d}/{total*2}] thr={thr} {fold_name} {period}"
                print(f"  {tag}...", end=" ", flush=True)
                t0 = time.time()
                m = run_fold(fi, thr, universe_raw, rsr_df, alpha_df, sym_active_df,
                             regime_df, topix_close, rsr_syms, cfg,
                             return_trades=False, is_period=is_p)
                print(f"CAGR={m.get('cagr',0):+.1f}%  "
                      f"Sharpe={m.get('sharpe',0):.3f}  "
                      f"Calmar={m.get('calmar',0):.3f}  "
                      f"PF={m.get('profit_factor',0):.3f}  "
                      f"Exp={m.get('avg_exposure',0):.1f}%  "
                      f"({time.time()-t0:.1f}s)")
                row[period] = m
            res[thr].append(row)

    # 2022単独 (with return_trades)
    print("\n── 2022単独 詳細分析 (return_trades=True) ──")
    for thr in thrs:
        print(f"  thr={thr}...", end=" ", flush=True)
        t0 = time.time()
        m = run_period(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=FOLD2_START, end=FOLD2_END,
            pattern="A",
            rsr_exit_override=float(thr),
            return_trades=True,
        )
        print(f"  CAGR={m.get('cagr',0):+.1f}%  trades={m.get('n_trades',0)}  ({time.time()-t0:.1f}s)")
        res22[thr] = m

    return res, res22


def _agg(res, thr) -> dict:
    rows    = res[thr]
    cagrs   = [_g(r["Test"], "cagr")          for r in rows]
    sharpes = [_g(r["Test"], "sharpe")         for r in rows]
    maxdds  = [_g(r["Test"], "max_dd")         for r in rows]
    calmars = [_g(r["Test"], "calmar")         for r in rows]
    pfs     = [_g(r["Test"], "profit_factor")  for r in rows]
    exps    = [_g(r["Test"], "avg_exposure")   for r in rows]
    return {
        "avg_cagr":   float(np.mean(cagrs)),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_maxdd":  float(np.mean(maxdds)),
        "avg_calmar": float(np.mean(calmars)),
        "avg_pf":     float(np.mean(pfs)),
        "avg_exp":    float(np.mean(exps)),
        "std_calmar": float(np.std(calmars, ddof=1)) if len(calmars) > 1 else 0.0,
        "std_sharpe": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        "std_cagr":   float(np.std(cagrs,   ddof=1)) if len(cagrs)   > 1 else 0.0,
        "cagrs":   cagrs,
        "sharpes": sharpes,
        "maxdds":  maxdds,
        "calmars": calmars,
        "pfs":     pfs,
        "exps":    exps,
    }


# ── Exit Event Audit ──────────────────────────────────────────────────

def _match_trades(trades_b: list, trades_c: list) -> dict:
    """
    Return {sym: {"b": trade, "c": trade}} for matched sell trades.
    Match by (symbol, entry_idx).
    """
    def index_sells(trades):
        idx = {}
        for t in trades:
            if t["side"] != "SELL":
                continue
            key = (t["symbol"], t["entry_idx"])
            idx[key] = t
        return idx

    ib = index_sells(trades_b)
    ic = index_sells(trades_c)
    keys = set(ib.keys()) | set(ic.keys())
    out = {}
    for key in keys:
        sym = key[0]
        if key in ib and key in ic:
            out[sym + f"_ei{key[1]}"] = {"b": ib[key], "c": ic[key], "key": key}
    return out


def exit_event_audit(res22_b: dict, res22_c: dict, dates_b: list, dates_c: list) -> dict:
    """Classify matched trades into categories."""
    trades_b = res22_b.get("_trades_raw", [])
    trades_c = res22_c.get("_trades_raw", [])

    matched = _match_trades(trades_b, trades_c)

    added_hold   = []   # c held longer (exit_idx_c > exit_idx_b)
    early_exit   = []   # c exited earlier (exit_idx_c < exit_idx_b) — prevented later loss
    unchanged    = []
    loss_expand  = []   # c held longer AND pnl_c < pnl_b (loss expanded)

    for uid, pair in matched.items():
        tb = pair["b"]; tc = pair["c"]
        ei_b = tb["exit_idx"]; ei_c = tc["exit_idx"]
        pnl_b = tb.get("pnl") or 0.0
        pnl_c = tc.get("pnl") or 0.0
        ret_b = (tb["exit"] / tb["entry"] - 1) * 100 if tb["entry"] > 0 else 0.0
        ret_c = (tc["exit"] / tc["entry"] - 1) * 100 if tc["entry"] > 0 else 0.0

        sym  = tb["symbol"]
        entry_date = dates_b[min(tb["entry_idx"], len(dates_b)-1)] if dates_b else "?"
        exit_b_date = dates_b[min(ei_b, len(dates_b)-1)] if dates_b else "?"
        exit_c_date = dates_c[min(ei_c, len(dates_c)-1)] if dates_c else "?"

        record = {
            "ticker": sym,
            "entry_date": entry_date,
            "exit_b": exit_b_date, "exit_c": exit_c_date,
            "exit_idx_b": ei_b, "exit_idx_c": ei_c,
            "reason_b": tb.get("reason","?"), "reason_c": tc.get("reason","?"),
            "ret_b": round(ret_b, 2), "ret_c": round(ret_c, 2),
            "pnl_b": round(pnl_b, 0), "pnl_c": round(pnl_c, 0),
            "delta_ret": round(ret_c - ret_b, 2),
            "delta_pnl": round(pnl_c - pnl_b, 0),
        }

        if abs(ei_b - ei_c) <= 1:
            unchanged.append(record)
        elif ei_c < ei_b:
            early_exit.append(record)
            if pnl_c < pnl_b:
                loss_expand.append(record)
        else:
            added_hold.append(record)
            if pnl_c < pnl_b:
                loss_expand.append(record)

    return {
        "added_hold": sorted(added_hold, key=lambda x: x["delta_ret"]),
        "early_exit": sorted(early_exit, key=lambda x: -x["delta_ret"]),
        "loss_expand": sorted(loss_expand, key=lambda x: x["delta_ret"]),
        "unchanged":   unchanged,
    }


# ── Counterfactual Attribution ────────────────────────────────────────

def attribution(res22_b: dict, res22_c: dict) -> list:
    """Rank symbols by contribution of (PnL_C - PnL_B) on matched sells."""
    trades_b = res22_b.get("_trades_raw", [])
    trades_c = res22_c.get("_trades_raw", [])

    matched = _match_trades(trades_b, trades_c)

    contrib: dict[str, float] = {}
    for uid, pair in matched.items():
        sym = pair["b"]["symbol"]
        pnl_b = pair["b"].get("pnl") or 0.0
        pnl_c = pair["c"].get("pnl") or 0.0
        contrib[sym] = contrib.get(sym, 0.0) + (pnl_c - pnl_b)

    ranked = sorted(contrib.items(), key=lambda x: -x[1])
    total  = sum(abs(v) for v in contrib.values()) or 1.0
    return [{"symbol": sym, "delta_pnl": round(v, 0),
             "pct": round(v / total * 100, 1)} for sym, v in ranked]


# ── Report Writer ─────────────────────────────────────────────────────

def write_report(res, res22, output_path: Path) -> None:
    agg_a = _agg(res, THR_A)
    agg_b = _agg(res, THR_B)
    agg_c = _agg(res, THR_C)

    fold_years = [f[3][:4] for f in FOLDS]
    fold_chars = [f[5]     for f in FOLDS]

    L = []; w = L.append

    w("# RSR Exit 72 Deep Validation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  A=75(Base) / B=70 / C=72")
    w("\n**採用判定基準**: avgCalmar(C)≥avgCalmar(B) AND 2022Calmar(C)>2022Calmar(B) AND Calmar分散(C)<Calmar分散(B)")
    w("")

    # ─────────────────────────────────────────────
    # 分析1: 5-Fold WF サマリー
    # ─────────────────────────────────────────────
    w("---\n## 分析1: 5-Fold Walk-Forward サマリー\n")

    # 1-a avgメトリクス表
    w("### 1-a. 平均メトリクス比較\n")
    w("| 閾値 | avgCAGR | avgSharpe | avgMaxDD | avgCalmar | avgPF | avgExp |")
    w("|---|---|---|---|---|---|---|")
    for label, ag in [("A(75) Baseline", agg_a), ("B(70)", agg_b), ("C(72)", agg_c)]:
        w(f"| {label} | {ag['avg_cagr']:+.2f}% | {ag['avg_sharpe']:.3f} "
          f"| {ag['avg_maxdd']:.1f}% | {ag['avg_calmar']:.3f} "
          f"| {ag['avg_pf']:.3f} | {ag['avg_exp']:.1f}% |")

    # 1-b Fold別 Test Calmar
    w("\n### 1-b. Fold別 Test Calmar\n")
    w("| 閾値 | " + " | ".join(fold_years) + " | avg | std |")
    w("|---|" + "---|" * (len(FOLDS) + 2))
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{c:.3f}" for c in ag["calmars"])
        w(f"| {label} | {vals} | {ag['avg_calmar']:.3f} | {ag['std_calmar']:.3f} |")

    # 1-c Fold別 Test CAGR
    w("\n### 1-c. Fold別 Test CAGR\n")
    w("| 閾値 | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{c:+.1f}%" for c in ag["cagrs"])
        w(f"| {label} | {vals} | {ag['avg_cagr']:+.2f}% |")

    # 1-d Fold別 Test Sharpe
    w("\n### 1-d. Fold別 Test Sharpe\n")
    w("| 閾値 | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{s:.3f}" for s in ag["sharpes"])
        w(f"| {label} | {vals} | {ag['avg_sharpe']:.3f} |")

    # 1-e Fold別 Test PF
    w("\n### 1-e. Fold別 Test Profit Factor\n")
    w("| 閾値 | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{p:.3f}" for p in ag["pfs"])
        w(f"| {label} | {vals} | {ag['avg_pf']:.3f} |")

    # 1-f Fold別 Test Exposure
    w("\n### 1-f. Fold別 Test Exposure\n")
    w("| 閾値 | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{e:.1f}%" for e in ag["exps"])
        w(f"| {label} | {vals} | {ag['avg_exp']:.1f}% |")

    # ─────────────────────────────────────────────
    # 分析2: 2022年詳細解析
    # ─────────────────────────────────────────────
    w("\n---\n## 分析2: 2022年詳細解析\n")
    w("2022年 = 金利ショック弱気相場。RSR緩和の方向性が逆効果になる代表年。\n")

    for thr_label, thr in [("A(75) Baseline", THR_A), ("B(70)", THR_B), ("C(72)", THR_C)]:
        m22 = res22[thr]
        trades_22 = m22.get("_trades_raw", [])
        sells_22  = [t for t in trades_22 if t["side"] == "SELL"]
        w(f"### {thr_label}\n")
        w("| 指標 | 値 |")
        w("|---|---|")
        w(f"| CAGR | {_g(m22,'cagr'):+.1f}% |")
        w(f"| Sharpe | {_g(m22,'sharpe'):.3f} |")
        w(f"| MaxDD | {_g(m22,'max_dd'):.1f}% |")
        w(f"| Calmar | {_g(m22,'calmar'):.3f} |")
        w(f"| Profit Factor | {_g(m22,'profit_factor'):.3f} |")
        w(f"| Avg Exposure | {_g(m22,'avg_exposure'):.1f}% |")
        w(f"| 取引数 | {_g(m22,'n_trades',0):.0f} |")
        w(f"| 平均保有日数 | {_avg_hold(trades_22):.1f}d |")
        w(f"| 平均PnL | ¥{float(np.mean([t.get('pnl',0) or 0 for t in sells_22])):,.0f} |" if sells_22 else "| 平均PnL | N/A |")
        w(f"| 中央値PnL | ¥{_median_pnl(trades_22):,.0f} |")

        # Exit理由別件数
        reasons = defaultdict(int)
        for t in sells_22:
            reasons[t.get("reason","?")] += 1
        w("\n**Exit理由別件数**:\n")
        w("| Exit理由 | 件数 | 割合 |")
        w("|---|---|---|")
        total_r = sum(reasons.values())
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            w(f"| {r} | {cnt} | {cnt/max(1,total_r)*100:.0f}% |")
        w("")

    # B vs C 2022比較
    w("### B(70) vs C(72) 2022直接比較\n")
    m22b = res22[THR_B]; m22c = res22[THR_C]
    tb22 = [t for t in m22b.get("_trades_raw",[]) if t["side"]=="SELL"]
    tc22 = [t for t in m22c.get("_trades_raw",[]) if t["side"]=="SELL"]
    w("| 指標 | B(70) | C(72) | Δ(C-B) |")
    w("|---|---|---|---|")
    rows_2cmp = [
        ("CAGR",        "cagr",          "%"),
        ("Sharpe",      "sharpe",        ""),
        ("MaxDD",       "max_dd",        "%"),
        ("Calmar",      "calmar",        ""),
        ("PF",          "profit_factor", ""),
        ("Exposure",    "avg_exposure",  "%"),
    ]
    for label, key, unit in rows_2cmp:
        vb = _g(m22b, key); vc = _g(m22c, key)
        delta = vc - vb
        w(f"| {label} | {vb:.3f}{unit} | {vc:.3f}{unit} | {delta:+.3f}{unit} |")
    # Hold days
    hold_b = _avg_hold(m22b.get("_trades_raw",[])); hold_c = _avg_hold(m22c.get("_trades_raw",[]))
    w(f"| 平均保有日数 | {hold_b:.1f}d | {hold_c:.1f}d | {hold_c-hold_b:+.1f}d |")
    # RSR_EXIT
    rsr_b = m22b.get("exit_reasons",{}).get("RSR_EXIT",0)
    rsr_c = m22c.get("exit_reasons",{}).get("RSR_EXIT",0)
    w(f"| RSR_EXIT件数 | {rsr_b} | {rsr_c} | {rsr_c-rsr_b:+d} |")
    # Avg PnL
    avg_b = float(np.mean([t.get("pnl",0) or 0 for t in tb22])) if tb22 else 0.0
    avg_c = float(np.mean([t.get("pnl",0) or 0 for t in tc22])) if tc22 else 0.0
    w(f"| 平均PnL | ¥{avg_b:,.0f} | ¥{avg_c:,.0f} | ¥{avg_c-avg_b:,.0f} |")
    med_b = _median_pnl(m22b.get("_trades_raw",[])); med_c = _median_pnl(m22c.get("_trades_raw",[]))
    w(f"| 中央値PnL | ¥{med_b:,.0f} | ¥{med_c:,.0f} | ¥{med_c-med_b:,.0f} |")

    # ─────────────────────────────────────────────
    # 分析3: Exit Event Audit
    # ─────────────────────────────────────────────
    w("\n---\n## 分析3: Exit Event Audit (70→72変更で影響された取引 / 2022年)\n")

    dates_b = res22[THR_B].get("_common_dates", [])
    dates_c = res22[THR_C].get("_common_dates", [])
    audit = exit_event_audit(res22[THR_B], res22[THR_C], dates_b, dates_c)

    # 早期退出 (C=72が先にexit → 損失防止)
    w("### 3-a. 早期退出: 72が先にexit（損失防止候補）\n")
    w("C(72)がB(70)より先にexitした取引。ΔRet=C-B。正=72が有利。\n")
    if audit["early_exit"]:
        w("| ticker | entry | exit_B(70) | exit_C(72) | ret_B | ret_C | ΔRet | reason_B | reason_C | ΔPNL |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for r in audit["early_exit"][:20]:
            w(f"| {r['ticker']} | {r['entry_date']} | {r['exit_b']} | {r['exit_c']} "
              f"| {r['ret_b']:+.1f}% | {r['ret_c']:+.1f}% | {r['delta_ret']:+.2f}% "
              f"| {r['reason_b']} | {r['reason_c']} | ¥{r['delta_pnl']:,.0f} |")
    else:
        w("*該当なし*")

    # 追加保有 (C=72が後にexit)
    w("\n### 3-b. 追加保有: 72がより長く保有した取引\n")
    w("C(72)がB(70)より遅くexitした取引。\n")
    if audit["added_hold"]:
        w("| ticker | entry | exit_B(70) | exit_C(72) | ret_B | ret_C | ΔRet | reason_B | reason_C | ΔPNL |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for r in audit["added_hold"][:20]:
            w(f"| {r['ticker']} | {r['entry_date']} | {r['exit_b']} | {r['exit_c']} "
              f"| {r['ret_b']:+.1f}% | {r['ret_c']:+.1f}% | {r['delta_ret']:+.2f}% "
              f"| {r['reason_b']} | {r['reason_c']} | ¥{r['delta_pnl']:,.0f} |")
    else:
        w("*該当なし*")

    # 損失拡大
    w("\n### 3-c. 損失拡大: 変更によりPnL悪化した取引\n")
    if audit["loss_expand"]:
        w("| ticker | entry | exit_B | exit_C | ret_B | ret_C | ΔRet | ΔPNL |")
        w("|---|---|---|---|---|---|---|---|")
        for r in audit["loss_expand"][:10]:
            w(f"| {r['ticker']} | {r['entry_date']} | {r['exit_b']} | {r['exit_c']} "
              f"| {r['ret_b']:+.1f}% | {r['ret_c']:+.1f}% | {r['delta_ret']:+.2f}% | ¥{r['delta_pnl']:,.0f} |")
    else:
        w("*該当なし*")

    # サマリー
    total_delta = sum(r["delta_pnl"] for r in audit["early_exit"]) + \
                  sum(r["delta_pnl"] for r in audit["added_hold"])
    w(f"\n**Auditサマリー**:")
    w(f"- 早期退出件数: {len(audit['early_exit'])} 件")
    w(f"- 追加保有件数: {len(audit['added_hold'])} 件")
    w(f"- 損失拡大件数: {len(audit['loss_expand'])} 件")
    w(f"- 変更なし件数: {len(audit['unchanged'])} 件")
    w(f"- 合計ΔPNL: ¥{total_delta:,.0f}")

    # ─────────────────────────────────────────────
    # 分析4: Counterfactual Attribution
    # ─────────────────────────────────────────────
    w("\n---\n## 分析4: Counterfactual Attribution (2022年, 銘柄別寄与度)\n")
    w("各銘柄の ΔPNL(C=72) - ΔPNL(B=70) を集計。正=72が有利。\n")

    att = attribution(res22[THR_B], res22[THR_C])
    total_att = sum(r["delta_pnl"] for r in att)

    w("### 上位寄与銘柄（72が有利）\n")
    w("| 順位 | 銘柄 | ΔPNL | 寄与度% |")
    w("|---|---|---|---|")
    positives = [r for r in att if r["delta_pnl"] > 0]
    for i, r in enumerate(positives[:10], 1):
        w(f"| {i} | {r['symbol']} | ¥{r['delta_pnl']:,.0f} | +{r['pct']:.1f}% |")

    w("\n### 下位寄与銘柄（70が有利）\n")
    w("| 順位 | 銘柄 | ΔPNL | 寄与度% |")
    w("|---|---|---|---|")
    negatives = [r for r in att if r["delta_pnl"] < 0]
    for i, r in enumerate(sorted(negatives, key=lambda x: x["delta_pnl"])[:10], 1):
        w(f"| {i} | {r['symbol']} | ¥{r['delta_pnl']:,.0f} | {r['pct']:.1f}% |")

    w(f"\n**Attribution合計ΔPNL**: ¥{total_att:,.0f}")

    # 2022年改善の原因特定
    w("\n### 2022年改善原因の特定\n")
    n_early_positive = sum(1 for r in audit["early_exit"] if r["delta_ret"] > 0)
    n_added_positive = sum(1 for r in audit["added_hold"] if r["delta_ret"] > 0)
    total_early_pnl  = sum(r["delta_pnl"] for r in audit["early_exit"])
    total_added_pnl  = sum(r["delta_pnl"] for r in audit["added_hold"])

    if total_early_pnl > 0 and abs(total_early_pnl) > abs(total_added_pnl):
        cause = "**早期退出による損失防止** が主因: 72が先に退出することで下落をカット"
    elif total_added_pnl > 0 and abs(total_added_pnl) > abs(total_early_pnl):
        cause = "**追加保有による利益伸長** が主因: 72がより長く保有して利益を延長"
    else:
        cause = "早期退出/追加保有の両方が複合的に寄与"

    w(f"- 早期退出 ΔPNL合計: ¥{total_early_pnl:,.0f} ({n_early_positive}/{len(audit['early_exit'])}件で有利)")
    w(f"- 追加保有 ΔPNL合計: ¥{total_added_pnl:,.0f} ({n_added_positive}/{len(audit['added_hold'])}件で有利)")
    w(f"\n**結論**: {cause}")

    # ─────────────────────────────────────────────
    # 分析5: Stability Analysis
    # ─────────────────────────────────────────────
    w("\n---\n## 分析5: Stability Analysis (Fold間分散比較)\n")
    w("分散(std)が小さい = Fold間で安定したパフォーマンス。\n")

    w("### 5-a. Calmar 安定性\n")
    w("| 閾値 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | avg | std | CV(std/avg) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        cals = ag["calmars"]
        avg  = ag["avg_calmar"]
        std  = ag["std_calmar"]
        cv   = std / max(abs(avg), 0.001)
        vals = " | ".join(f"{c:.3f}" for c in cals)
        w(f"| {label} | {vals} | {avg:.3f} | {std:.3f} | {cv:.2f} |")

    w("\n### 5-b. Sharpe 安定性\n")
    w("| 閾値 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | avg | std |")
    w("|---|---|---|---|---|---|---|---|")
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{s:.3f}" for s in ag["sharpes"])
        w(f"| {label} | {vals} | {ag['avg_sharpe']:.3f} | {ag['std_sharpe']:.3f} |")

    w("\n### 5-c. CAGR 安定性\n")
    w("| 閾値 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | avg | std |")
    w("|---|---|---|---|---|---|---|---|")
    for label, thr, ag in [("A(75)", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        vals = " | ".join(f"{c:+.1f}%" for c in ag["cagrs"])
        w(f"| {label} | {vals} | {ag['avg_cagr']:+.2f}% | {ag['std_cagr']:.2f}pp |")

    w("\n### 5-d. 安定性サマリー\n")
    w("| 閾値 | Calmar_std | Sharpe_std | CAGR_std | 判定 |")
    w("|---|---|---|---|---|")
    for label, thr, ag in [("A(75) Baseline", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        # Lower std = more stable
        min_calmar_std = min(agg_a["std_calmar"], agg_b["std_calmar"], agg_c["std_calmar"])
        stable = "✅ 最安定" if abs(ag["std_calmar"] - min_calmar_std) < 0.001 else "—"
        w(f"| {label} | {ag['std_calmar']:.3f} | {ag['std_sharpe']:.3f} | {ag['std_cagr']:.2f}pp | {stable} |")

    # ─────────────────────────────────────────────
    # 最終判定
    # ─────────────────────────────────────────────
    w("\n---\n## 最終判定\n")

    # 3条件評価
    cal_b = agg_b["avg_calmar"]; cal_c = agg_c["avg_calmar"]
    cal22_b = _g(res22[THR_B], "calmar"); cal22_c = _g(res22[THR_C], "calmar")
    std_b = agg_b["std_calmar"]; std_c = agg_c["std_calmar"]

    crit1 = cal_c >= cal_b          # avgCalmar(72) >= avgCalmar(70)
    crit2 = cal22_c > cal22_b       # 2022Calmar(72) > 2022Calmar(70)
    crit3 = std_c < std_b           # Calmar分散(72) < 分散(70)

    w("### 採用条件チェック\n")
    w("| 条件 | 値B(70) | 値C(72) | 判定 |")
    w("|---|---|---|---|")
    w(f"| avgCalmar(C)≥avgCalmar(B) | {cal_b:.3f} | {cal_c:.3f} | {'✅' if crit1 else '❌'} |")
    w(f"| 2022Calmar(C)>2022Calmar(B) | {cal22_b:.3f} | {cal22_c:.3f} | {'✅' if crit2 else '❌'} |")
    w(f"| Calmar分散(C)<Calmar分散(B) | std={std_b:.3f} | std={std_c:.3f} | {'✅' if crit3 else '❌'} |")

    all_pass = crit1 and crit2 and crit3

    w(f"\n### **最終判定: {'B: 72採用' if all_pass else 'A: 70採用'}**\n")

    if all_pass:
        w("**rsr_exit = 72 を採用する。**\n")
        w("根拠:")
        w(f"- avgCalmar改善: {cal_b:.3f} → {cal_c:.3f} (+{cal_c-cal_b:.4f})")
        w(f"- 2022年Calmar改善: {cal22_b:.3f} → {cal22_c:.3f} (+{cal22_c-cal22_b:.4f})")
        w(f"- Calmar安定性向上: std {std_b:.3f} → {std_c:.3f} (Δ{std_c-std_b:+.3f})")
        w(f"\n**変更内容 (ASK_FIRST_ON_CHANGE適用):**")
        w("- `rsr_exit_thr` を 70.0 → 72.0 に変更 (exit条件のみ)")
        w("- entry条件 `min_rsr=75` は変更しない")
    else:
        w("**rsr_exit = 70 を採用する。**\n")
        w("根拠:")
        failed = []
        if not crit1: failed.append(f"avgCalmar(72)={cal_c:.3f} < avgCalmar(70)={cal_b:.3f}")
        if not crit2: failed.append(f"2022Calmar(72)={cal22_c:.3f} ≤ 2022Calmar(70)={cal22_b:.3f}")
        if not crit3: failed.append(f"Calmar分散(72)={std_c:.3f} ≥ 分散(70)={std_b:.3f}")
        for f in failed:
            w(f"- ❌ {f}")
        w(f"\n**変更内容 (ASK_FIRST_ON_CHANGE適用):**")
        w("- `rsr_exit_thr` を 75.0 → 70.0 に変更 (exit条件のみ, Bull限定)")
        w("- entry条件 `min_rsr=75` は変更しない")
        w("- Bear時は rsr_exit=75 を維持")

    w("\n---\n*生成: src/backtest/rsr_exit_72_deep_validation.py*")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  RSR Exit 72 Deep Validation")
    print(f"  A={THR_A}(Base) / B={THR_B} / C={THR_C}")
    print(f"  5-Fold WF + 2022詳細 + Audit + Attribution + Stability")
    print("=" * 72)

    print("\n[1/2] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 全Fold × 全閾値 実行中...")
    res, res22 = run_all(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
    )

    out_path = REPORTS_DIR / "rsr_exit_72_deep_validation.md"
    write_report(res, res22, out_path)

    # ── Console summary ───────────────────────────────────────────────
    agg_a = _agg(res, THR_A); agg_b = _agg(res, THR_B); agg_c = _agg(res, THR_C)
    cal22_b = _g(res22[THR_B], "calmar"); cal22_c = _g(res22[THR_C], "calmar")

    crit1 = agg_c["avg_calmar"] >= agg_b["avg_calmar"]
    crit2 = cal22_c > cal22_b
    crit3 = agg_c["std_calmar"] < agg_b["std_calmar"]
    verdict = "B: 72採用" if (crit1 and crit2 and crit3) else "A: 70採用"

    print("\n" + "=" * 72)
    print("  ★ Deep Validation Summary")
    print("=" * 72)
    print(f"\n  {'閾値':<12} {'avgCAGR':>9} {'avgSharpe':>10} {'avgCalmar':>10} "
          f"{'CalStd':>8} {'2022Cal':>8} {'2022CAGR':>9}")
    print("  " + "-" * 68)
    for label, thr, ag in [("A(75) Base", THR_A, agg_a), ("B(70)", THR_B, agg_b), ("C(72)", THR_C, agg_c)]:
        cal22 = _g(res22[thr], "calmar")
        cagr22 = _g(res22[thr], "cagr")
        print(f"  {label:<12} {ag['avg_cagr']:>+8.2f}% {ag['avg_sharpe']:>10.3f} {ag['avg_calmar']:>10.3f} "
              f"{ag['std_calmar']:>8.3f} {cal22:>8.3f} {cagr22:>+8.1f}%")
    print(f"\n  判定: {verdict}")
    print(f"  条件1 avgCalmar(C)≥(B): {'✅' if crit1 else '❌'}  ({agg_b['avg_calmar']:.3f} vs {agg_c['avg_calmar']:.3f})")
    print(f"  条件2 2022Calmar(C)>(B): {'✅' if crit2 else '❌'}  ({cal22_b:.3f} vs {cal22_c:.3f})")
    print(f"  条件3 CalStd(C)<(B):     {'✅' if crit3 else '❌'}  ({agg_b['std_calmar']:.3f} vs {agg_c['std_calmar']:.3f})")
    print(f"\n  レポート → {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
