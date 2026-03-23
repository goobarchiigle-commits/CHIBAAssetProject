"""
backtest/topk_sweep.py

top_k sweep バックテスト（k=3/4/5 の性能比較）

topk_live_equivalent.py の run_backtest() を呼び出す。
ライブ構造と同じ経路で検証するため、portfolio_v2.py との混在は行わない。

判定基準:
  Sharpe 差 < 0.15 → 戦略は k 変化に対して堅牢
  avg_exposure が k によって大きく変わる場合 → exposure-driven 設計を確認

実行:
  python backtest/topk_sweep.py
  python backtest/topk_sweep.py --k 3 4 5 --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from backtest.topk_live_equivalent import run_backtest, START, END, CAPITAL


def main() -> int:
    p = argparse.ArgumentParser(description="top_k sweep バックテスト")
    p.add_argument("--k",     type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--start", default=START)
    p.add_argument("--end",   default=END)
    args = p.parse_args()

    print("=" * 72)
    print(f"  top_k sweep バックテスト（ライブアーキテクチャ完全再現）")
    print(f"  期間: {args.start} 〜 {args.end}  /  初期資本: ¥{CAPITAL:,}")
    print(f"  sweep: k = {args.k}")
    print("=" * 72)

    results = []
    for k in args.k:
        print(f"\n  ── k={k} 実行中... ──")
        r = run_backtest(top_k=k, start=args.start, end=args.end, verbose=False)
        results.append(r)
        print(
            f"  k={k}: CAGR={r['cagr']:+.2%}  Sharpe={r['sharpe']:.3f}"
            f"  MaxDD={r['max_dd']:.2%}  Calmar={r['calmar']:.3f}"
            f"  exposure={r['avg_exposure']:.1%}"
        )

    # ---- 比較テーブル ----
    rows = []
    for r in results:
        rows.append({
            "top_k":         r["top_k"],
            "CAGR%":         f"{r['cagr']*100:+.2f}",
            "Sharpe":        f"{r['sharpe']:.3f}",
            "MaxDD%":        f"{r['max_dd']*100:.2f}",
            "Calmar":        f"{r['calmar']:.3f}",
            "avg_exp%":      f"{r['avg_exposure']*100:.1f}",
            "avg_pos":       f"{r['avg_positions']:.2f}",
            "zero_exp%":     f"{r['zero_exp_rate']:.1f}",
            "trades":        r["trade_count"],
            "win_rate%":     f"{r['win_rate']*100:.1f}",
        })

    df = pd.DataFrame(rows).set_index("top_k")

    print("\n" + "=" * 72)
    print("  比較結果")
    print("=" * 72)
    print(df.to_string())

    # ---- 感度分析 ----
    if len(results) >= 2:
        sharpes  = [r["sharpe"]  for r in results]
        exposures = [r["avg_exposure"] for r in results]
        sharpe_range  = max(sharpes)  - min(sharpes)
        exposure_range = max(exposures) - min(exposures)

        print("\n  感度分析:")
        print(f"    Sharpe  最大差: {sharpe_range:.3f}  "
              f"{'✅ < 0.15 → 堅牢' if sharpe_range < 0.15 else '⚠️  >= 0.15 → k に感度あり'}")
        print(f"    exposure 最大差: {exposure_range:.1%}")

    print("\n  注意: RSR コンテキストは TEMPORAL24 内。"
          "ライブ（TOPIX100 コンテキスト）との差異は許容範囲。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
