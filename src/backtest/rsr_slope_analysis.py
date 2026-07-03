"""
backtest/rsr_slope_analysis.py

RSR スロープフィルターの影響測定（Step 1: supply 改善 - 戦略変更前の測定）

「RSR slope を入れたら候補銘柄がどれだけ増えるか」を測定する。

分析する4つのフィルター条件:
  A（現行）  : RSR ≥ 75
  B（厳格化）: RSR ≥ 75 AND slope > 0  （今の候補をさらに絞る）
  C（拡張）  : RSR ≥ 60 AND slope > 0  （閾値引き下げ + トレンド確認）
  D（slope単独）: slope > 0 のみ  （RSR閾値を廃止）

さらに breakout 期間比較を RSR context 統一で計算:
  turtle_20 / turtle_15 / turtle_10 の通過数の差

実行:
  python -m backtest.rsr_slope_analysis
  python -m backtest.rsr_slope_analysis --start 2022-01-01 --end 2024-12-31
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

from backtest.rsr            import calc_universe_rsr
from backtest.universe_builder import download_universe
from backtest.live_equivalent  import _load_rsr_universe_42, START, END


SLOPE_WINDOW = 5   # RSR slope 計算期間（5日）


def run_analysis(start: str = START, end: str = END) -> dict:
    # ---- データ取得 ----
    rsr42 = _load_rsr_universe_42()
    print(f"[1/3] データ取得中（{len(rsr42)}銘柄）...")
    universe_raw = download_universe(rsr42, start=start, end=end, verbose=False)
    print(f"  取得完了: {len(universe_raw)} 銘柄")

    # ---- RSR 計算（42銘柄コンテキスト）----
    print("[2/3] RSR 計算中...")
    prices = {sym: universe_raw[sym]["df"]["Close"] for sym in universe_raw}
    rsr_df = calc_universe_rsr(prices)
    print(f"  RSR matrix: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")

    # ---- スロープ計算 ----
    rsr_slope_df = rsr_df.diff(SLOPE_WINDOW)   # 5日差分

    # ---- 日次集計 ----
    print("[3/3] フィルター影響を集計中...")
    rows = []

    for date in rsr_df.index:
        rsr_row   = rsr_df.loc[date]
        slope_row = rsr_slope_df.loc[date]

        # 取引可能銘柄（NaN 除外）
        valid = rsr_row.dropna().index

        # フィルター条件 A〜D の通過数
        pass_a = sum(1 for s in valid if rsr_row[s] >= 75)
        pass_b = sum(1 for s in valid if rsr_row[s] >= 75 and slope_row.get(s, np.nan) > 0)
        pass_c = sum(1 for s in valid if rsr_row[s] >= 60 and slope_row.get(s, np.nan) > 0)
        pass_d = sum(1 for s in valid if slope_row.get(s, np.nan) > 0)

        # Turtle breakout 通過数（20/15/10日高値）
        bo20 = bo15 = bo10 = 0
        for sym in valid:
            if sym not in universe_raw:
                continue
            df_sym = universe_raw[sym]["df"]
            if date not in df_sym.index:
                continue
            close_s   = df_sym["Close"]
            loc_i     = close_s.index.get_loc(date)
            price_now = float(close_s.iloc[loc_i])
            if loc_i >= 21:
                high_20 = float(close_s.iloc[loc_i-20:loc_i].max())
                if price_now > high_20: bo20 += 1
            if loc_i >= 16:
                high_15 = float(close_s.iloc[loc_i-15:loc_i].max())
                if price_now > high_15: bo15 += 1
            if loc_i >= 11:
                high_10 = float(close_s.iloc[loc_i-10:loc_i].max())
                if price_now > high_10: bo10 += 1

        rows.append({
            "date":    date,
            "pass_a":  pass_a,   # RSR≥75（現行）
            "pass_b":  pass_b,   # RSR≥75 + slope>0
            "pass_c":  pass_c,   # RSR≥60 + slope>0
            "pass_d":  pass_d,   # slope>0 only
            "bo20":    bo20,
            "bo15":    bo15,
            "bo10":    bo10,
        })

    df_result = pd.DataFrame(rows).set_index("date")
    return df_result


def print_report(df: pd.DataFrame) -> None:
    n = len(df)
    print("\n" + "=" * 64)
    print("  RSR スロープ影響測定 / Turtle期間比較")
    print("=" * 64)

    # ---- RSR フィルター比較 ----
    print("\n  ── RSR フィルター: 平均通過銘柄数/日 ──")
    fmt = "  {:<32} {:>8.2f}/日  {:>7}"
    a = df["pass_a"].mean()
    b = df["pass_b"].mean()
    c = df["pass_c"].mean()
    d = df["pass_d"].mean()

    print(fmt.format("A: RSR≥75（現行）",           a, "基準"))
    print(fmt.format("B: RSR≥75 AND slope>0",       b, f"{b/max(a,0.01):.0%}（絞り込み）"))
    print(fmt.format("C: RSR≥60 AND slope>0（拡張）", c, f"{c/max(a,0.01):.0%}（{c-a:+.2f}/日）"))
    print(fmt.format("D: slope>0 のみ",              d, f"{d/max(a,0.01):.0%}（{d-a:+.2f}/日）"))

    # ---- Turtle期間比較 ----
    print("\n  ── Turtle ブレイクアウト: 平均通過銘柄数/日 ──")
    t20 = df["bo20"].mean()
    t15 = df["bo15"].mean()
    t10 = df["bo10"].mean()
    print(fmt.format("Turtle 20日（現行）", t20, "基準"))
    print(fmt.format("Turtle 15日",        t15, f"{t15/max(t20,0.01):.0%}（{t15-t20:+.2f}/日）"))
    print(fmt.format("Turtle 10日",        t10, f"{t10/max(t20,0.01):.0%}（{t10-t20:+.2f}/日）"))

    # ---- 零候補日の比較 ----
    print("\n  ── 候補ゼロ日の比率（supply枯渇率）──")
    zero_a = (df["pass_a"] == 0).mean() * 100
    zero_b = (df["pass_b"] == 0).mean() * 100
    zero_c = (df["pass_c"] == 0).mean() * 100
    print(f"  A（現行）     : {zero_a:.1f}% の日で RSR候補=0")
    print(f"  B（slope追加）: {zero_b:.1f}% の日で RSR候補=0  ({'悪化' if zero_b > zero_a else '改善'})")
    print(f"  C（RSR60+sl） : {zero_c:.1f}% の日で RSR候補=0  ({'悪化' if zero_c > zero_a else '改善'})")

    # ---- 結論 ----
    print("\n  ── 判断基準 ──")
    c_vs_a = c - a
    t15_vs_t20 = t15 - t20
    t10_vs_t20 = t10 - t20

    print(f"\n  RSRスロープ拡張（C: RSR≥60 + slope>0）: {c_vs_a:+.2f}/日")
    if c_vs_a >= a * 0.4:
        print("  → ✅ +40%超: RSR閾値引き下げ + slopeフィルター採用を推奨")
    elif c_vs_a >= a * 0.2:
        print("  → △ +20-40%: 効果あり。OOSバックテストで検証してから採用")
    else:
        print("  → ✗ +20%未満: 効果限定的。RSRスロープ単独では改善困難")

    print(f"\n  Turtle 15日 vs 20日: {t15_vs_t20:+.2f}/日")
    if t15_vs_t20 >= t20 * 0.3:
        print("  → ✅ +30%超: turtle_entry=15への変更を推奨（バックテスト検証後）")
    elif t15_vs_t20 >= t20 * 0.1:
        print("  → △ +10-30%: 効果あり。turtle_entry=15のバックテスト比較を推奨")
    else:
        print("  → ✗ +10%未満: 期間短縮の効果限定的")

    print(f"\n  Turtle 10日 vs 20日: {t10_vs_t20:+.2f}/日")
    if t10_vs_t20 >= t20 * 0.5:
        print("  → ⚠ +50%超: 候補増加だが偽シグナルも増える可能性")
    print("\n" + "=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description="RSRスロープ影響測定 / Turtle期間比較")
    p.add_argument("--start", default=START)
    p.add_argument("--end",   default=END)
    args = p.parse_args()

    print("=" * 64)
    print("  RSRスロープ影響測定 / Turtle期間比較")
    print(f"  期間: {args.start} 〜 {args.end}  /  RSR42ユニバース")
    print("=" * 64)

    df = run_analysis(start=args.start, end=args.end)
    print_report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
