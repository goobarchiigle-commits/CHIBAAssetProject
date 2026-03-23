"""
backtest/turtle_entry_rolling_oos.py

Turtle エントリー期間のウォークフォワード OOS 検証（Step 1: 検証精度向上）

目的: IS差 +0.06 Sharpe（turtle 10日 vs 20日）は小さく、レジーム依存で逆転する可能性がある。
     rolling OOS で中央値 Sharpe / fold別安定性を確認して誤判断を防ぐ。

設計:
  - Equity curve を年別にスライスしてfold-level指標を計算
  - 比較: turtle_entry = 20（現行） vs 15 vs 10
  - Universe: RSR42（全期間固定）+ RSR42 コンテキスト
  - Weight: max_single_weight=0.15（確定設計）

見たい指標（平均でなく中央値）:
  - median Sharpe / median CAGR
  - max DD の安定性（fold間のブレ幅）
  - 負け年の数

実行:
  python -m backtest.turtle_entry_rolling_oos
  python -m backtest.turtle_entry_rolling_oos --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from backtest.live_equivalent import (
    run_backtest,
    _load_rsr_universe_42,
    START, END,
)


def _fold_metrics(eq: pd.Series, capital: float) -> list[dict]:
    """Equity curve を年単位に分割してfold-level指標を計算する。"""
    folds = []
    years = eq.index.year.unique()

    for yr in years:
        yr_eq = eq[eq.index.year == yr]
        if len(yr_eq) < 20:
            continue
        start_val = float(yr_eq.iloc[0])
        end_val   = float(yr_eq.iloc[-1])
        cagr_yr   = end_val / start_val - 1   # 1年なのでそのまま

        dr = yr_eq.pct_change().dropna()
        sharpe_yr = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

        roll_max = yr_eq.expanding().max()
        dd_ser   = (yr_eq - roll_max) / roll_max
        max_dd_yr = float(dd_ser.min())

        folds.append({
            "year":   yr,
            "cagr":   cagr_yr,
            "sharpe": sharpe_yr,
            "max_dd": max_dd_yr,
        })
    return folds


def run_turtle_comparison(start: str = START, end: str = END) -> None:
    rsr42 = _load_rsr_universe_42()

    turtle_entries = [20, 15, 10]
    all_results: dict[int, dict] = {}

    for te in turtle_entries:
        print(f"[turtle_entry={te}] 実行中...")
        m = run_backtest(
            start=start, end=end,
            min_rsr=75.0,
            max_single_weight=0.15,
            trade_universe=rsr42,
            turtle_entry=te,
            verbose=False,
        )
        folds = _fold_metrics(m["equity_curve"], capital=2_000_000)
        m["folds"] = folds
        all_results[te] = m

    _print_comparison(all_results)


def _print_comparison(results: dict[int, dict]) -> None:
    print("\n" + "=" * 72)
    print("  Turtle Entry 期間 OOS 検証（fold-level 中央値）")
    print("=" * 72)

    # ---- 集計指標比較 ----
    print(f"\n  {'指標':<24} {'turtle=20':>12} {'turtle=15':>12} {'turtle=10':>12}")
    print("  " + "-" * 60)

    te_list = [20, 15, 10]

    def fmt_row(label, key, pct=False):
        row = f"  {label:<24}"
        for te in te_list:
            folds = results[te]["folds"]
            vals  = [f[key] for f in folds]
            med   = float(np.median(vals))
            row  += f" {med:>12.2%}" if pct else f" {med:>12.3f}"
        return row

    print(fmt_row("中央値 CAGR (年次)",    "cagr",   True))
    print(fmt_row("中央値 Sharpe (年次)",   "sharpe", False))
    print(fmt_row("中央値 MaxDD (年次)",    "max_dd", True))

    # ---- 負け年カウント ----
    row = f"  {'負け年 / 全年':<24}"
    for te in te_list:
        folds   = results[te]["folds"]
        n_loss  = sum(1 for f in folds if f["cagr"] < 0)
        n_total = len(folds)
        row    += f" {n_loss:>5}/{n_total:<6}"
    print(row)

    # ---- DD 安定性（fold間ブレ幅） ----
    row = f"  {'MaxDD 最悪 fold':<24}"
    for te in te_list:
        folds  = results[te]["folds"]
        worst  = min(f["max_dd"] for f in folds)
        row   += f" {worst:>12.2%}"
    print(row)

    row = f"  {'MaxDD std (fold間)':<24}"
    for te in te_list:
        folds = results[te]["folds"]
        std   = float(np.std([f["max_dd"] for f in folds]))
        row  += f" {std:>12.3f}"
    print(row)

    # ---- fold-level 詳細 ----
    print(f"\n  ── fold 詳細（年次）──")
    header = f"  {'年':<6}"
    for te in te_list:
        header += f" {'te='+str(te)+' CAGR':>10} {'Sharpe':>7} {'MaxDD':>8}"
    print(header)
    print("  " + "-" * (6 + 3 * 27))

    all_years = sorted({f["year"] for te in te_list for f in results[te]["folds"]})
    for yr in all_years:
        row = f"  {yr:<6}"
        for te in te_list:
            folds = {f["year"]: f for f in results[te]["folds"]}
            if yr in folds:
                f = folds[yr]
                row += f" {f['cagr']:>10.2%} {f['sharpe']:>7.3f} {f['max_dd']:>8.2%}"
            else:
                row += f" {'N/A':>10} {'':>7} {'':>8}"
        print(row)

    # ---- 推奨判断 ----
    print(f"\n  ── 判断 ──")
    for te in [10, 15]:
        base = results[20]
        comp = results[te]

        base_med_sh = float(np.median([f["sharpe"] for f in base["folds"]]))
        comp_med_sh = float(np.median([f["sharpe"] for f in comp["folds"]]))
        diff_sh     = comp_med_sh - base_med_sh

        base_loss = sum(1 for f in base["folds"] if f["cagr"] < 0)
        comp_loss = sum(1 for f in comp["folds"] if f["cagr"] < 0)

        print(f"\n  turtle={te} vs 20:")
        print(f"    中央値 Sharpe差: {diff_sh:+.3f}", end="  ")
        if diff_sh >= 0.05:
            print("→ ✅ OOS でも有意な改善 → 採用推奨")
        elif diff_sh >= 0.0:
            print("→ △ 微改善（IS同等以上）→ 観察継続")
        else:
            print("→ ✗ OOS で劣化 → 不採用")

        if comp_loss > base_loss:
            print(f"    ⚠ 負け年増加: {base_loss}→{comp_loss} (リスク増)")
        elif comp_loss < base_loss:
            print(f"    ✅ 負け年減少: {base_loss}→{comp_loss}")

    print("\n" + "=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description="Turtle Entry 期間 Rolling OOS 検証")
    p.add_argument("--start", default=START)
    p.add_argument("--end",   default=END)
    args = p.parse_args()

    print("=" * 72)
    print("  Turtle Entry 期間 Rolling OOS 検証")
    print(f"  期間: {args.start} 〜 {args.end}")
    print(f"  RSR42 universe / min_rsr=75 / weight=0.15")
    print("=" * 72)

    run_turtle_comparison(start=args.start, end=args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
