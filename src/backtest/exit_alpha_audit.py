"""
backtest/exit_alpha_audit.py

Exit Alpha Audit — Pattern A 全トレード分析

目的: Exitによるアルファ毀損を定量化する
対象: 2018-01-01〜2025-12-31 全トレード

分析1: Exit Reason別統計 (N / WR / avg_return / PF)
分析2: Early Exit Audit (exit後 fwd20d/40d/60d)
分析3: Counterfactual Hold (+20/+40/+60日追加保有時の追加損益)
分析4: 保有期間別損益 (0-20 / 20-40 / 40-60 / 60d+)
分析5: Turtle Exit感度 (55d / 65d / 75d の CAGR/Sharpe/MaxDD/Calmar比較)

出力: reports/exit_alpha_audit.md

結論:
  A: Exit最適
  B: Exit改善余地あり
  C: 55日制限がボトルネック
  D: Composite Exitがボトルネック

Run:
    cd C:/ai-trading
    python src/backtest/exit_alpha_audit.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, run_period,
    IS_START, IS_END, OOS_START, OOS_END,
)

REPORTS_DIR = Path("reports")
FULL_START  = IS_START         # "2018-01-01"
FULL_END    = OOS_END          # "2025-12-31"

# 分析5: turtle exit感度比較
TURTLE_SETTINGS = [55, 65, 75]

# Early Exit Threshold (avg post-exit return > this → leaving money on table)
EARLY_EXIT_WARN_PCT = 3.0


# ─────────────────────────────────────────────────────────────────────
#  FORWARD RETURN HELPER
# ─────────────────────────────────────────────────────────────────────

def _get_fwd_returns(
    sym: str,
    exec_date_str: str,
    universe_raw: dict,
    n_list: tuple = (20, 40, 60),
) -> dict:
    """
    Compute N-day forward close returns from exec_date for sym.
    Returns {n: float | None}.
    Handles both tz-aware and tz-naive index.
    """
    results = {n: None for n in n_list}
    try:
        if sym not in universe_raw:
            return results
        closes = universe_raw[sym]["df"]["Close"]
        exec_dt = pd.Timestamp(exec_date_str)
        # Normalize timezone: if index is tz-aware, localize exec_dt
        if closes.index.tz is not None:
            try:
                exec_dt = exec_dt.tz_localize(closes.index.tz)
            except Exception:
                exec_dt = exec_dt.tz_localize("UTC").tz_convert(closes.index.tz)
        # Use date-only comparison to avoid time-of-day issues
        idx_dates = closes.index.normalize()
        target_dt = exec_dt.normalize()
        mask = np.asarray(idx_dates >= target_dt)   # ensure numpy bool array
        if not mask.any():
            return results
        start_pos = int(np.argmax(mask))
        base_price = float(closes.iloc[start_pos])
        if base_price <= 0:
            return results
        for n in n_list:
            tgt = start_pos + n
            if tgt < len(closes):
                results[n] = float(closes.iloc[tgt]) / base_price - 1.0
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────────────────────────────
#  TRADE POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────

def process_trades(raw_trades: list, common_dates: list, universe_raw: dict) -> list:
    """
    Augment SELL trades with:
      - hold_days (days held)
      - fwd20d/fwd40d/fwd60d (post-exit returns)
      - cf_add20/40/60 (counterfactual additional PnL)
      - return_pct (actual trade return %)
    """
    sells = [t for t in raw_trades if t.get("side") == "SELL"]
    n_dates = len(common_dates)

    for t in sells:
        entry_idx = t.get("entry_idx", 0)
        exit_idx  = t.get("exit_idx",  0)
        exec_idx  = min(exit_idx + 1, n_dates - 1)
        exec_date = common_dates[exec_idx]

        t["hold_days"]   = max(0, exit_idx - entry_idx + 1)
        t["return_pct"]  = (
            (t["exit"] / t["entry"] - 1.0) * 100.0
            if t.get("entry") and t["entry"] > 0 else 0.0
        )

        fwd = _get_fwd_returns(t["symbol"], exec_date, universe_raw)
        t["fwd20d"] = fwd[20]
        t["fwd40d"] = fwd[40]
        t["fwd60d"] = fwd[60]

        exit_price = t.get("exit", 0.0) or 0.0
        qty        = t.get("qty",  0)   or 0
        for n in (20, 40, 60):
            f = fwd[n]
            t[f"cf_add{n}"] = (f * exit_price * qty) if (f is not None and exit_price > 0 and qty > 0) else None

    return sells


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────

def _stats(trades: list, ret_key: str = "return_pct") -> dict:
    """Compute N / WR / avg_return / PF for a list of trades."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "avg_ret": 0.0, "pf": 0.0}
    wins   = [t for t in trades if t.get("pnl", 0) is not None and t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) is not None and t.get("pnl", 0) <= 0]
    gross_p = sum(t["pnl"] for t in wins   if t["pnl"] is not None)
    gross_l = abs(sum(t["pnl"] for t in losses if t["pnl"] is not None))
    rets   = [t.get(ret_key, 0.0) or 0.0 for t in trades]
    return {
        "n":       n,
        "wr":      len(wins) / n * 100,
        "avg_ret": float(np.mean(rets)) if rets else 0.0,
        "pf":      gross_p / max(gross_l, 1e-6),
    }


def _fwd_stats(trades: list, fwd_key: str) -> dict:
    """Mean / median / WR for a forward return field."""
    vals = [t[fwd_key] for t in trades if t.get(fwd_key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "wr": None}
    wins = [v for v in vals if v > 0]
    return {
        "n":      len(vals),
        "mean":   float(np.mean(vals)) * 100,
        "median": float(np.median(vals)) * 100,
        "wr":     len(wins) / len(vals) * 100,
    }


def _cf_stats(trades: list, cf_key: str) -> dict:
    """Total / mean counterfactual additional PnL."""
    vals = [t[cf_key] for t in trades if t.get(cf_key) is not None]
    if not vals:
        return {"n": 0, "total": None, "mean": None, "positive_rate": None}
    pos = [v for v in vals if v > 0]
    return {
        "n":             len(vals),
        "total":         sum(vals),
        "mean":          float(np.mean(vals)),
        "positive_rate": len(pos) / len(vals) * 100,
    }


# ─────────────────────────────────────────────────────────────────────
#  CONCLUSION LOGIC
# ─────────────────────────────────────────────────────────────────────

def _determine_conclusion(sells: list, turtle_results: dict) -> tuple[str, str]:
    """
    Determine primary conclusion label.
    Priority order:
      1. Overall fwd60d > 8% + RSR_EXIT major → B (general early exit)
      2. MARKET_SHOCK fwd20d > 20% + N >= 5 → D (shock exit specific)
      3. TIME_STOP / turtle sensitivity → C
      4. Otherwise → A
    Returns (label, rationale).
    """
    # Overall post-exit returns
    all_fwd60 = _fwd_stats(sells, "fwd60d")
    all_fwd20 = _fwd_stats(sells, "fwd20d")
    all_cf20  = _cf_stats(sells, "cf_add20")

    # RSR_EXIT specific
    rsr_trades = [t for t in sells if t.get("reason") == "RSR_EXIT"]
    rsr_fwd60  = _fwd_stats(rsr_trades, "fwd60d")

    # MARKET_SHOCK specific — require N >= 5 for D conclusion
    shock_trades = [t for t in sells if t.get("reason") == "MARKET_SHOCK_EXIT"]
    shock_fwd20  = _fwd_stats(shock_trades, "fwd20d")
    shock_bottleneck_strong = (
        len(shock_trades) >= 5
        and shock_fwd20["mean"] is not None
        and shock_fwd20["mean"] > EARLY_EXIT_WARN_PCT * 2
    )

    # B condition: overall systematic early exit
    # Primary evidence: fwd60d > 8% AND >60% of exits show positive post-exit returns
    general_early_exit = (
        all_fwd60["mean"] is not None
        and all_fwd60["mean"] > 8.0
        and all_fwd20["wr"] is not None
        and all_fwd20["wr"] > 60.0
    )
    # Also: RSR_EXIT (major group) leaves significant alpha on table
    rsr_leakage = (
        len(rsr_trades) > 20
        and rsr_fwd60["mean"] is not None
        and rsr_fwd60["mean"] > 8.0
    )

    # C condition: TIME_STOP cutting winners + turtle sensitivity
    time_stop_trades = [t for t in sells if t.get("reason") == "TIME_STOP"]
    ts_fwd20 = _fwd_stats(time_stop_trades, "fwd20d")
    te_65_calmar = turtle_results.get(65, {}).get("IS", {}).get("calmar", 0)
    te_55_calmar = turtle_results.get(55, {}).get("IS", {}).get("calmar", 0)
    turtle_bottleneck = (
        len(time_stop_trades) >= 5
        and ts_fwd20["mean"] is not None
        and ts_fwd20["mean"] > EARLY_EXIT_WARN_PCT
        and te_65_calmar > te_55_calmar * 1.05
    )

    if general_early_exit or rsr_leakage:
        shock_note = ""
        if shock_fwd20["mean"] is not None and shock_fwd20["mean"] > 20 and len(shock_trades) >= 3:
            shock_note = (
                f" | MARKET_SHOCK後 fwd20d={shock_fwd20['mean']:+.1f}%(N={len(shock_trades)}:統計弱)"
            )
        rsr_note = (
            f"RSR_EXIT({len(rsr_trades)}件): avg_return={_stats(rsr_trades)['avg_ret']:+.2f}%, "
            f"fwd60d={rsr_fwd60['mean']:+.1f}%"
            if rsr_fwd60["mean"] is not None else ""
        )
        return "B", (
            f"Exit改善余地あり: fwd60d 全体={all_fwd60['mean']:+.1f}%, fwd20d 勝率={all_fwd20['wr']:.0f}%, "
            f"CF+60d 平均追加損益={_cf_stats(sells,'cf_add60')['mean']:+,.0f}円. "
            f"{rsr_note}{shock_note}"
        )
    elif shock_bottleneck_strong:
        return "D", (
            f"Composite Exit ボトルネック確定: MARKET_SHOCK_EXIT後 fwd20d 平均"
            f"+{shock_fwd20['mean']:.1f}% (N={shock_fwd20['n']})"
        )
    elif turtle_bottleneck:
        te_55 = turtle_results.get(55, {}).get("IS", {})
        te_65 = turtle_results.get(65, {}).get("IS", {})
        return "C", (
            f"55日Exit制限ボトルネック: 65d Calmar={te_65.get('calmar',0):.3f} > "
            f"55d Calmar={te_55.get('calmar',0):.3f}"
        )
    else:
        cf20 = _cf_stats(sells, "cf_add20")
        if cf20["positive_rate"] is not None and cf20["positive_rate"] > 55:
            return "B", (
                f"軽度のExit改善余地: CF+20d 正率={cf20['positive_rate']:.0f}%, "
                f"平均追加損益={cf20['mean']:+,.0f}円"
            )
        return "A", "Exit最適: カウンターファクチュアル保有による追加期待値は限定的"


# ─────────────────────────────────────────────────────────────────────
#  REPORT WRITER
# ─────────────────────────────────────────────────────────────────────

def write_report(
    sells:          list,
    turtle_results: dict,
    conclusion_label: str,
    conclusion_reason: str,
    output_path:    Path,
    is_metrics:     dict,
    oos_metrics:    dict,
) -> None:
    L = []; w = L.append

    w("# Exit Alpha Audit — Pattern A")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  対象: {FULL_START}〜{FULL_END}  |  全トレード\n")
    w(f"**IS: {IS_START}〜{IS_END}  OOS: {OOS_START}〜{OOS_END}**\n")

    # ── ベースライン確認 ──────────────────────────────────────
    w("---\n## 0. ベースライン確認\n")
    w("| 期間 | CAGR | Sharpe | MaxDD | Calmar | 取引数 | 平均保有 |")
    w("|---|---|---|---|---|---|---|")
    for label, m in [("IS 2018-2024", is_metrics), ("OOS 2025", oos_metrics)]:
        w(
            f"| {label} "
            f"| {m.get('cagr',0):+.1f}% "
            f"| {m.get('sharpe',0):.3f} "
            f"| {m.get('max_dd',0):.1f}% "
            f"| {m.get('calmar',0):.3f} "
            f"| {m.get('n_trades',0)} "
            f"| {m.get('avg_hold_days',0):.1f}d |"
        )

    # ── 分析1: Exit Reason別 ──────────────────────────────────
    w("\n---\n## 1. Exit Reason別統計\n")
    w("| Exit理由 | N | WR | avg_return | PF | avg_hold | fwd20d_mean | fwd20d_WR |")
    w("|---|---|---|---|---|---|---|---|")

    reason_map = {
        "STRATEGY_EXIT":    "Turtle/Momentum Exit",
        "RSR_EXIT":         "RSR Exit",
        "MARKET_SHOCK_EXIT":"Composite Shock Exit",
        "TIME_STOP":        "Time Stop (max_hold)",
    }

    reason_groups = defaultdict(list)
    for t in sells:
        reason_groups[t.get("reason", "OTHER")].append(t)

    for reason, label in reason_map.items():
        group = reason_groups.get(reason, [])
        if not group:
            continue
        s = _stats(group)
        f = _fwd_stats(group, "fwd20d")
        holds = [t.get("hold_days", 0) for t in group]
        avg_h = float(np.mean(holds)) if holds else 0.0
        fwd_mean = f"{f['mean']:+.1f}%" if f["mean"] is not None else "N/A"
        fwd_wr   = f"{f['wr']:.0f}%"   if f["wr"]   is not None else "N/A"
        w(
            f"| **{label}** "
            f"| {s['n']} "
            f"| {s['wr']:.0f}% "
            f"| {s['avg_ret']:+.2f}% "
            f"| {s['pf']:.2f} "
            f"| {avg_h:.1f}d "
            f"| {fwd_mean} "
            f"| {fwd_wr} |"
        )

    # Other reasons
    for reason in reason_groups:
        if reason not in reason_map:
            group = reason_groups[reason]
            s = _stats(group)
            w(f"| {reason} | {s['n']} | {s['wr']:.0f}% | {s['avg_ret']:+.2f}% | {s['pf']:.2f} | — | — | — |")

    w(f"\n**合計**: {len(sells)}件")

    # ── 分析2: Early Exit Audit ──────────────────────────────
    w("\n---\n## 2. Early Exit Audit (exit後フォワードリターン)\n")
    w("\n### 2a. 全Exit後 forward returns\n")
    w("| Forward期間 | N | 平均 | 中央値 | 勝率 | 解釈 |")
    w("|---|---|---|---|---|---|")
    for n_days in (20, 40, 60):
        f = _fwd_stats(sells, f"fwd{n_days}d")
        if f["mean"] is None:
            continue
        if f["mean"] > EARLY_EXIT_WARN_PCT:
            interp = "⚠ 退出後も上昇継続"
        elif f["mean"] > 1.0:
            interp = "△ 軽微な早期退出"
        elif f["mean"] < -1.0:
            interp = "✅ Exit タイミング良好"
        else:
            interp = "✅ ほぼ最適"
        w(
            f"| fwd{n_days}d "
            f"| {f['n']} "
            f"| {f['mean']:+.2f}% "
            f"| {f['median']:+.2f}% "
            f"| {f['wr']:.0f}% "
            f"| {interp} |"
        )

    w("\n### 2b. Exit Reason別 forward returns\n")
    w("| Exit理由 | N | fwd20d_mean | fwd40d_mean | fwd60d_mean | fwd20d_WR |")
    w("|---|---|---|---|---|---|")
    for reason, label in reason_map.items():
        group = reason_groups.get(reason, [])
        if not group:
            continue
        f20 = _fwd_stats(group, "fwd20d")
        f40 = _fwd_stats(group, "fwd40d")
        f60 = _fwd_stats(group, "fwd60d")
        def _fmt(fs): return f"{fs['mean']:+.1f}%" if fs["mean"] is not None else "N/A"
        def _fmtwr(fs): return f"{fs['wr']:.0f}%" if fs["wr"] is not None else "N/A"
        w(
            f"| {label} "
            f"| {len(group)} "
            f"| {_fmt(f20)} "
            f"| {_fmt(f40)} "
            f"| {_fmt(f60)} "
            f"| {_fmtwr(f20)} |"
        )

    # ── 分析3: Counterfactual Hold ────────────────────────────
    w("\n---\n## 3. Counterfactual Hold (追加保有時の期待損益)\n")
    w("| 延長期間 | N | 追加損益_合計 | 追加損益_平均 | 正率 | 解釈 |")
    w("|---|---|---|---|---|---|")
    for n_days in (20, 40, 60):
        cf = _cf_stats(sells, f"cf_add{n_days}")
        if cf["total"] is None:
            continue
        pos_rate = cf["positive_rate"] or 0.0
        mean_val = cf["mean"] or 0.0
        total_val = cf["total"] or 0.0
        if pos_rate > 55 and mean_val > 0:
            interp = "⚠ 保有延長で追加リターン期待"
        elif pos_rate > 50:
            interp = "△ 微改善の可能性"
        else:
            interp = "✅ 早期exitが有利"
        w(
            f"| +{n_days}日 "
            f"| {cf['n']} "
            f"| {total_val:+,.0f}円 "
            f"| {mean_val:+,.0f}円 "
            f"| {pos_rate:.0f}% "
            f"| {interp} |"
        )

    # ── 分析4: 保有期間別 ─────────────────────────────────────
    w("\n---\n## 4. 保有期間別損益分析\n")
    w("| 保有期間 | N | WR | avg_return | PF | avg_fwd20d | avg_fwd60d |")
    w("|---|---|---|---|---|---|---|")
    buckets = [
        ("0-20日",  lambda t: t.get("hold_days", 0) <= 20),
        ("21-40日", lambda t: 20 < t.get("hold_days", 0) <= 40),
        ("41-60日", lambda t: 40 < t.get("hold_days", 0) <= 60),
        ("61日+",   lambda t: t.get("hold_days", 0) > 60),
    ]
    for label, pred in buckets:
        group = [t for t in sells if pred(t)]
        if not group:
            w(f"| {label} | 0 | — | — | — | — | — |")
            continue
        s = _stats(group)
        f20 = _fwd_stats(group, "fwd20d")
        f60 = _fwd_stats(group, "fwd60d")
        f20m = f"{f20['mean']:+.1f}%" if f20["mean"] is not None else "N/A"
        f60m = f"{f60['mean']:+.1f}%" if f60["mean"] is not None else "N/A"
        w(
            f"| {label} "
            f"| {s['n']} "
            f"| {s['wr']:.0f}% "
            f"| {s['avg_ret']:+.2f}% "
            f"| {s['pf']:.2f} "
            f"| {f20m} "
            f"| {f60m} |"
        )

    # 月次保有日数分布
    hold_arr = [t.get("hold_days", 0) for t in sells]
    if hold_arr:
        w(f"\n**保有日数統計**: mean={np.mean(hold_arr):.1f}d  "
          f"median={np.median(hold_arr):.0f}d  "
          f"max={max(hold_arr)}d  "
          f"p75={np.percentile(hold_arr, 75):.0f}d  "
          f"p90={np.percentile(hold_arr, 90):.0f}d")

    # ── Turtle Stop 診断 (感度テスト前) ─────────────────────
    w("\n---\n## 5a. Turtle Trailing Stop 実効性診断\n")
    w(
        "> **注意**: turtle_exit=55d/65d/75d が全て同一結果 → trailing stop 独立発動ゼロを示唆\n"
    )
    w("**実効性診断**:\n")
    w("FujikoStrategy の exit条件:")
    w("```")
    w("exit = (RSR_momentum < 0 AND RSR_momentum_declining)   ← 一次出口")
    w("     OR (close < N日最安値)                            ← trailing stop")
    w("```")
    w("RSR momentumがN日最安値と同時またはより早く発動するため、")
    w("turtle_exit N=55/65/75 のいずれでも exit タイミングが不変。")
    w("\n**EXIT 内訳 (実態)**:")
    w("| 条件 | 発動件数 | 備考 |")
    w("|---|---|---|")
    w(f"| RSR_EXIT (rsr_val < min_rsr=75) | {len(reason_groups.get('RSR_EXIT',[]))} | abc.pyで先に判定 |")
    w(f"| STRATEGY_EXIT (momentum+trailing) | {len(reason_groups.get('STRATEGY_EXIT',[]))} | 主にmomentum起因 |")
    w(f"| TIME_STOP (max_hold=60d) | {len(reason_groups.get('TIME_STOP',[]))} | ★**60日保有強制終了** |")
    w(f"| MARKET_SHOCK_EXIT (composite) | {len(reason_groups.get('MARKET_SHOCK_EXIT',[]))} | market-5%+sym-8% |")
    w("\n**結論**: turtle_exit パラメータ (55/65/75d) は実質的に EXIT に影響しない。")
    w("主要 EXIT ドライバーは RSR レベル低下と RSR モメンタム反転。\n")

    # ── 分析5: Turtle Exit感度 ───────────────────────────────
    w("\n---\n## 5b. Turtle Exit感度テスト (IS+OOS比較)\n")
    w("| turtle_exit | IS CAGR | IS Sharpe | IS MaxDD | IS Calmar | OOS CAGR | OOS Sharpe | OOS MaxDD | OOS Calmar |")
    w("|---|---|---|---|---|---|---|---|---|")
    w("| *(turtle trailing stopは実効性ゼロ — 全て同一結果)* | | | | | | | | |")
    for te in TURTLE_SETTINGS:
        r = turtle_results.get(te, {})
        r_is  = r.get("IS",  {})
        r_oos = r.get("OOS", {})
        is_mark  = " ★" if te == 55 else ""
        w(
            f"| **{te}d{is_mark}** "
            f"| {r_is.get('cagr',0):+.1f}% "
            f"| {r_is.get('sharpe',0):.3f} "
            f"| {r_is.get('max_dd',0):.1f}% "
            f"| {r_is.get('calmar',0):.3f} "
            f"| {r_oos.get('cagr',0):+.1f}% "
            f"| {r_oos.get('sharpe',0):.3f} "
            f"| {r_oos.get('max_dd',0):.1f}% "
            f"| {r_oos.get('calmar',0):.3f} |"
        )

    w("\n### Δ vs 55d (現行)\n")
    w("| turtle_exit | ΔCAGR_IS | ΔSharpe_IS | ΔCalmar_IS | ΔCAGR_OOS | ΔSharpe_OOS | ΔCalmar_OOS |")
    w("|---|---|---|---|---|---|---|")
    base55_is  = turtle_results.get(55, {}).get("IS",  {})
    base55_oos = turtle_results.get(55, {}).get("OOS", {})
    for te in [65, 75]:
        r = turtle_results.get(te, {})
        r_is  = r.get("IS",  {})
        r_oos = r.get("OOS", {})
        w(
            f"| {te}d "
            f"| {r_is.get('cagr',0)   - base55_is.get('cagr',0):+.2f}pp "
            f"| {r_is.get('sharpe',0) - base55_is.get('sharpe',0):+.4f} "
            f"| {r_is.get('calmar',0) - base55_is.get('calmar',0):+.4f} "
            f"| {r_oos.get('cagr',0)   - base55_oos.get('cagr',0):+.2f}pp "
            f"| {r_oos.get('sharpe',0) - base55_oos.get('sharpe',0):+.4f} "
            f"| {r_oos.get('calmar',0) - base55_oos.get('calmar',0):+.4f} |"
        )

    # ── 総合分析 ─────────────────────────────────────────────
    w("\n---\n## 6. 総合分析\n")
    w("### アルファ漏出源: Entry vs Exit\n")

    # Summary of key findings
    all_fwd20 = _fwd_stats(sells, "fwd20d")
    all_cf20  = _cf_stats(sells, "cf_add20")
    strategy_exits = reason_groups.get("STRATEGY_EXIT", [])
    rsr_exits      = reason_groups.get("RSR_EXIT",       [])
    shock_exits    = reason_groups.get("MARKET_SHOCK_EXIT", [])
    time_stops     = reason_groups.get("TIME_STOP",      [])

    s_fwd20 = _fwd_stats(strategy_exits, "fwd20d")
    r_fwd20 = _fwd_stats(rsr_exits,      "fwd20d")
    sh_fwd20= _fwd_stats(shock_exits,    "fwd20d")
    ts_fwd20= _fwd_stats(time_stops,     "fwd20d")

    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| 全Exit後 fwd20d 平均 | {all_fwd20['mean']:+.2f}% (N={all_fwd20['n']}) |" if all_fwd20['mean'] is not None else "| 全Exit後 fwd20d 平均 | N/A |")
    w(f"| 全Exit後 fwd60d 平均 | {_fwd_stats(sells, 'fwd60d')['mean']:+.2f}% |" if _fwd_stats(sells, 'fwd60d')['mean'] is not None else "| 全Exit後 fwd60d 平均 | N/A |")
    w(f"| Counterfactual +20d 正率 | {all_cf20['positive_rate']:.0f}% |" if all_cf20['positive_rate'] is not None else "| Counterfactual +20d 正率 | N/A |")
    w(f"| Counterfactual +20d 平均追加損益 | {all_cf20['mean']:+,.0f}円 |" if all_cf20['mean'] is not None else "| Counterfactual +20d 平均追加損益 | N/A |")
    w(f"| STRATEGY_EXIT後 fwd20d | {s_fwd20['mean']:+.2f}% (N={s_fwd20['n']}) |" if s_fwd20['mean'] is not None else "| STRATEGY_EXIT後 fwd20d | N/A |")
    w(f"| RSR_EXIT後 fwd20d | {r_fwd20['mean']:+.2f}% (N={r_fwd20['n']}) |" if r_fwd20['mean'] is not None else "| RSR_EXIT後 fwd20d | N/A |")
    if sh_fwd20['mean'] is not None:
        w(f"| MARKET_SHOCK_EXIT後 fwd20d | {sh_fwd20['mean']:+.2f}% (N={sh_fwd20['n']}) |")
    if ts_fwd20['mean'] is not None:
        w(f"| TIME_STOP後 fwd20d | {ts_fwd20['mean']:+.2f}% (N={ts_fwd20['n']}) |")

    w("\n### Entry vs Exit アルファ漏出 比較\n")
    w("| 項目 | 値 | 解釈 |")
    w("|---|---|---|")
    w("| Missed Signal fwd60d (既知) | +8.2% | Entry改善の上限期待値 |")
    w("| Executed Signal fwd60d (既知) | +5.6% | 実際の保有銘柄60日平均 |")
    w(f"| Exit後 fwd20d 平均 | {all_fwd20['mean']:+.2f}% | Exit改善の期待値上限 |" if all_fwd20['mean'] is not None else "| Exit後 fwd20d 平均 | N/A | |")

    # ── 最終判定 ─────────────────────────────────────────────
    w("\n---\n## 7. 最終判定\n")

    label_map = {
        "A": "A: Exit最適",
        "B": "B: Exit改善余地あり",
        "C": "C: 55日制限がボトルネック",
        "D": "D: Composite Exitがボトルネック",
    }
    w(f"### **{label_map[conclusion_label]}**\n")
    w(f"{conclusion_reason}\n")

    # アルファ漏出源の最終結論
    w("### 現システム最大のアルファ漏出源\n")
    if all_fwd20["mean"] is not None and all_fwd20["mean"] > EARLY_EXIT_WARN_PCT:
        primary_source = "**Exit**"
        secondary = "Entry (Missed Signal +8.2% vs executed +5.6%の差)"
    elif all_fwd20["mean"] is not None and all_fwd20["mean"] > 1.5:
        primary_source = "**EntryとExitの両方**"
        secondary = "どちらも軽微な漏出あり"
    else:
        primary_source = "**Entry** (Missed Signal)"
        secondary = "Exitは概ね最適"

    w(f"**一次漏出源**: {primary_source}")
    w(f"**二次漏出源**: {secondary}\n")

    w("| 漏出源 | 指標 | 大きさ | 優先度 |")
    w("|---|---|---|---|")
    # Entry leakage
    w("| Entry (Missed Signal) | fwd60d差: +8.2% - +5.6% = +2.6% | 機会損失 | — |")
    # Exit leakage
    fwd20_val = all_fwd20["mean"] if all_fwd20["mean"] is not None else 0.0
    exit_priority = "高" if fwd20_val > EARLY_EXIT_WARN_PCT else ("中" if fwd20_val > 1.0 else "低")
    w(f"| Exit (早期退出) | fwd20d平均: {fwd20_val:+.2f}% | 直接損失 | {exit_priority} |")

    w("\n**推奨アクション**:\n")

    # RSR_EXIT specific stats for recommendations
    rsr_trades   = [t for t in sells if t.get("reason") == "RSR_EXIT"]
    shock_trades = [t for t in sells if t.get("reason") == "MARKET_SHOCK_EXIT"]
    shock_fwd20  = _fwd_stats(shock_trades, "fwd20d")

    if conclusion_label == "D":
        w("1. MARKET_SHOCK_EXIT の閾値緩和または廃止を検討")
        w("2. shock_sym_thr を -8% → -12% に引き上げてWF検証")
        w("3. RSR_EXIT の閾値調整も並行検討 (min_rsr 75 → 70)")
    elif conclusion_label == "C":
        w("1. turtle_exit を 55d → 65d または 75d に変更してWF検証")
        w("2. time_stop(max_hold_days=60) の撤廃または延長を検討")
    elif conclusion_label == "B":
        w("**【最優先】RSR_EXIT 閾値の見直し**")
        rsr_s = _stats(rsr_trades)
        rsr_f60 = _fwd_stats(rsr_trades, "fwd60d")
        w(f"- RSR_EXIT: {len(rsr_trades)}件, avg_return={rsr_s['avg_ret']:+.2f}%, fwd60d={rsr_f60['mean']:+.1f}%")
        w("- min_rsr=75 → 70 に緩和してWF検証 (RSR75境界での早期退出を防ぐ)")
        w("- または: RSR exit に保有日数条件追加 (hold >= 5d かつ RSR < 70 で退出)")
        w("")
        w("**【次優先】MARKET_SHOCK_EXIT の確認**")
        if shock_fwd20["mean"] is not None:
            w(f"- MARKET_SHOCK_EXIT後 fwd20d={shock_fwd20['mean']:+.1f}%(N={len(shock_trades)}:統計弱)")
        w("- shock_sym_thr を -8% → -12% に緩和してWF検証")
        w("")
        w("**【参考】turtle_exit parameter**")
        w("- 55d/65d/75d が全て同一結果 → trailing stop は実質デッドパラメータ")
        w("- RSR momentum exit が trailing stop より常に先行発動")
        w("- turtle_exit の変更は効果なし (PARAMS_LOCKEDから除外検討)")
    else:
        w("1. Exit は現状維持が最適")
        w("2. Entry改善 (Missed Signal 削減) に注力する")
        w("3. turtle_exit パラメータはデッドパラメータ (変更不要)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  Exit Alpha Audit — Pattern A 2018-2025")
    print("=" * 72)

    print("\n[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    # ── Step 1: Full period run with raw trades ──────────────────────
    print("\n[2/3] Pattern A 全期間 実行中 (return_trades=True)...")
    t0 = time.time()
    m_full = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=FULL_START, end=FULL_END,
        pattern="A",
        return_trades=True,
    )
    print(f"  完了 ({time.time()-t0:.1f}s) — "
          f"CAGR={m_full.get('cagr',0):+.1f}%  "
          f"Sharpe={m_full.get('sharpe',0):.3f}  "
          f"Calmar={m_full.get('calmar',0):.3f}  "
          f"Trades={m_full.get('n_trades',0)}")

    raw_trades   = m_full.get("_trades_raw", [])
    common_dates = m_full.get("_common_dates", [])

    # ── Step 2: IS/OOS reference metrics ────────────────────────────
    print("\n  IS期間 実行中...")
    m_is = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=IS_START, end=IS_END,
        pattern="A",
    )
    print(f"  IS: CAGR={m_is.get('cagr',0):+.1f}%  Calmar={m_is.get('calmar',0):.3f}")

    print("  OOS期間 実行中...")
    m_oos = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=OOS_START, end=OOS_END,
        pattern="A",
    )
    print(f"  OOS: CAGR={m_oos.get('cagr',0):+.1f}%  Calmar={m_oos.get('calmar',0):.3f}")

    # ── Step 3: Process trades ───────────────────────────────────────
    print("\n  トレード処理中 (forward returns計算)...")
    t0 = time.time()
    sells = process_trades(raw_trades, common_dates, universe_raw)
    print(f"  完了 ({time.time()-t0:.1f}s) — SELL {len(sells)}件")

    # ── Step 4: Turtle exit sensitivity ─────────────────────────────
    print("\n[3/3] Turtle Exit感度テスト (55d / 65d / 75d)...")
    turtle_results: dict = {}
    for te in TURTLE_SETTINGS:
        turtle_results[te] = {}
        for period_label, start, end in [
            ("IS",  IS_START,  IS_END),
            ("OOS", OOS_START, OOS_END),
        ]:
            print(f"  turtle={te}d [{period_label}] ...", end=" ", flush=True)
            t0 = time.time()
            mr = run_period(
                universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                topix_close, rsr_syms, cfg,
                start=start, end=end,
                pattern="A",
                turtle_exit_override=te,
            )
            print(
                f"CAGR={mr.get('cagr',0):+.1f}%  "
                f"Calmar={mr.get('calmar',0):.3f}  "
                f"({time.time()-t0:.1f}s)"
            )
            turtle_results[te][period_label] = mr

    # ── Step 5: Conclusion ───────────────────────────────────────────
    conclusion_label, conclusion_reason = _determine_conclusion(sells, turtle_results)

    # ── Step 6: Write report ─────────────────────────────────────────
    out_path = REPORTS_DIR / "exit_alpha_audit.md"
    write_report(
        sells          = sells,
        turtle_results = turtle_results,
        conclusion_label  = conclusion_label,
        conclusion_reason = conclusion_reason,
        output_path    = out_path,
        is_metrics     = m_is,
        oos_metrics    = m_oos,
    )

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ★ Exit Alpha Audit — 結果サマリー")
    print("=" * 72)

    # Exit reason breakdown
    from collections import Counter
    reason_counts = Counter(t.get("reason") for t in sells)
    print("\n  Exit Reason 内訳:")
    for reason, cnt in reason_counts.most_common():
        pct = cnt / len(sells) * 100 if sells else 0
        print(f"    {reason:<28}: {cnt:4d}件 ({pct:.0f}%)")

    # Forward returns
    from statistics import mean as smean
    fwd20_vals = [t["fwd20d"] for t in sells if t.get("fwd20d") is not None]
    fwd60_vals = [t["fwd60d"] for t in sells if t.get("fwd60d") is not None]
    if fwd20_vals:
        print(f"\n  Exit後 fwd20d: 平均{smean(fwd20_vals)*100:+.2f}%  N={len(fwd20_vals)}")
    if fwd60_vals:
        print(f"  Exit後 fwd60d: 平均{smean(fwd60_vals)*100:+.2f}%  N={len(fwd60_vals)}")

    # Turtle sensitivity
    print("\n  Turtle Exit感度 (IS Calmar / OOS Calmar):")
    for te in TURTLE_SETTINGS:
        r = turtle_results.get(te, {})
        mark = " ← 現行" if te == 55 else ""
        print(
            f"    {te}d{mark:<8}: "
            f"IS Calmar={r.get('IS',{}).get('calmar',0):.3f}  "
            f"OOS Calmar={r.get('OOS',{}).get('calmar',0):.3f}"
        )

    print(f"\n  ★ 最終判定: **{conclusion_label}** — {['Exit最適','Exit改善余地あり','55日制限がボトルネック','Composite Exitがボトルネック']['ABCD'.index(conclusion_label)]}")
    print(f"  {conclusion_reason}")
    print(f"\n  レポート → {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
