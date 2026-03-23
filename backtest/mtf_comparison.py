"""
backtest/mtf_comparison.py

マルチタイムフレーム（MTF）フィルター比較検証

BaselineのRSR+Turtle20に対して週足トレンドフィルターを2種類追加して比較する。

パターンA: weekly_close > weekly_MA20  （遅いが安定）
パターンB: weekly_MA10  > weekly_MA30  （初動に強い）

主要KPI:
  - Sharpe / CAGR / MaxDD / Calmar
  - false_breakout_rate（entry後5日以内に -2ATR 到達した割合）← 最重要
  - trade_count / win_rate

実行:
  python -m backtest.mtf_comparison
  python -m backtest.mtf_comparison --start 2018-01-01 --end 2024-12-31
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

from backtest.live_equivalent import (
    run_backtest,
    _load_rsr_universe_42,
    START, END,
)


def run_mtf_comparison(start: str = START, end: str = END) -> None:
    rsr42 = _load_rsr_universe_42()

    configs = [
        ("Baseline",  None),
        ("MTF-A: ma20", "ma20"),
        ("MTF-B: cross","cross"),
    ]
    results = {}

    for label, mtf_filter in configs:
        print(f"[{label}] 実行中...")
        m = run_backtest(
            start              = start,
            end                = end,
            min_rsr            = 75.0,
            max_single_weight  = 0.15,
            trade_universe     = rsr42,
            turtle_entry       = 20,
            weekly_trend_filter= mtf_filter,
            verbose            = False,
        )
        results[label] = m

    _print_report(results)


def _print_report(results: dict) -> None:
    labels = list(results.keys())

    print("\n" + "=" * 76)
    print("  MTF フィルター比較（Baseline vs weekly MA20 vs weekly cross）")
    print("=" * 76)

    metrics = [
        ("CAGR",               "cagr",               True),
        ("Sharpe",             "sharpe",              False),
        ("MaxDD",              "max_dd",              True),
        ("Calmar",             "calmar",              False),
        ("False Breakout率",   "false_breakout_rate", True),
        ("Win Rate",           "win_rate",            True),
        ("avg Exposure",       "avg_exposure",        True),
        ("Trade Count",        "trade_count",         False),
    ]

    # ヘッダー
    print(f"\n  {'指標':<22}", end="")
    for label in labels:
        print(f" {label:>20}", end="")
    print()
    print("  " + "-" * (22 + 21 * len(labels)))

    for display, key, is_pct in metrics:
        print(f"  {display:<22}", end="")
        for label in labels:
            v = results[label].get(key, 0)
            if key == "trade_count":
                print(f" {int(v):>20}", end="")
            elif is_pct:
                print(f" {v:>20.2%}", end="")
            else:
                print(f" {v:>20.3f}", end="")
        print()

    # 判断
    print(f"\n  ── 判断 ──")
    baseline_sh   = results["Baseline"]["sharpe"]
    baseline_fb   = results["Baseline"]["false_breakout_rate"]

    for label in labels[1:]:
        sh_diff = results[label]["sharpe"]           - baseline_sh
        fb_diff = results[label]["false_breakout_rate"] - baseline_fb
        print(f"\n  {label} vs Baseline:")
        print(f"    Sharpe差:          {sh_diff:+.3f}", end="  ")
        if sh_diff >= 0.05:
            print("→ ✅ 有意な改善")
        elif sh_diff >= 0.0:
            print("→ △ 微改善（変化なし相当）")
        else:
            print("→ ✗ 悪化")
        print(f"    False Breakout率差: {fb_diff:+.3f}", end="  ")
        if fb_diff <= -0.02:
            print("→ ✅ フォールスブレイク削減（MTF導入効果あり）")
        elif fb_diff <= 0.01:
            print("→ △ 誤差範囲")
        else:
            print("→ ✗ フォールスブレイク増加（逆効果）")

    print("\n  ── 導入判断基準 ──")
    print("  MTF採用条件（両方満たす）:")
    print("    1. Sharpe差 ≥ 0.0（悪化しない）")
    print("    2. false_breakout_rate差 ≤ -0.02（フォールスブレイク2%以上削減）")

    print("\n" + "=" * 76)


def main() -> int:
    p = argparse.ArgumentParser(description="MTFフィルター比較検証")
    p.add_argument("--start", default=START)
    p.add_argument("--end",   default=END)
    args = p.parse_args()

    print("=" * 76)
    print("  MTF フィルター比較検証")
    print(f"  期間: {args.start} 〜 {args.end}  /  RSR42ユニバース")
    print("=" * 76)

    run_mtf_comparison(start=args.start, end=args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
