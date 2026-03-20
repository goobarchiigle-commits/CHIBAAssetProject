"""
diagnostics/exposure_report.py
Top-k 実稼働率診断スクリプト

OOS バックテストの equity ログから以下を測定する:
  - avg_exposure    : 平均投資比率（invested_capital / equity）
  - median_exposure : 中央値投資比率
  - under_50pct_ratio: 投資比率 < 50% の日の割合（キャッシュ比率高すぎ問題）
  - topk_fill_rate  : ポジション数 >= k の日の割合（Top-k 充足率）
  - exposure_hist   : 投資比率の分布（10%刻み）

ログ形式（logs/backtest_equity.csv）:
  date,equity,invested_capital,positions

使い方:
  python diagnostics/exposure_report.py
  python diagnostics/exposure_report.py --log logs/backtest_equity.csv --k 3
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# Phase 2 評価基準（推奨ライン）
# ------------------------------------------------------------------ #
CRITERIA = {
    "avg_exposure":      (">=", 0.65, "平均投資比率"),
    "topk_fill_rate":   (">=", 0.55, "Top-k 充足率"),
    "under_50pct_ratio": ("<=", 0.25, "キャッシュ過多比率"),
}


# ------------------------------------------------------------------ #
# ロード
# ------------------------------------------------------------------ #
def load_equity_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    required = {"date", "equity", "invested_capital", "positions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必須カラムが不足: {missing}")
    return df


# ------------------------------------------------------------------ #
# 計算
# ------------------------------------------------------------------ #
def compute_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exposure"] = df["invested_capital"] / df["equity"].replace(0, float("nan"))
    df["exposure"] = df["exposure"].clip(0, 1)   # 0-100% に丸める
    return df


def generate_report(df: pd.DataFrame, k: int) -> dict:
    exp = df["exposure"].dropna()
    pos = df["positions"].dropna()

    # 投資比率分布（10%刻みのヒストグラム）
    bins   = np.arange(0, 1.1, 0.1)
    hist, edges = np.histogram(exp, bins=bins)
    hist_pct = {
        f"{int(edges[i]*100)}-{int(edges[i+1]*100)}%": int(hist[i])
        for i in range(len(hist))
    }

    # ポジション数分布
    pos_dist = pos.value_counts().sort_index().to_dict()
    pos_dist = {int(k): int(v) for k, v in pos_dist.items()}

    return {
        "period_start":      str(df["date"].min().date()),
        "period_end":        str(df["date"].max().date()),
        "total_days":        len(df),
        "avg_exposure":      float(exp.mean()),
        "median_exposure":   float(exp.median()),
        "under_50pct_ratio": float((exp < 0.5).mean()),
        "topk":              k,
        "topk_fill_rate":    float((pos >= k).mean()),
        "avg_positions":     float(pos.mean()),
        "exposure_hist":     hist_pct,
        "positions_dist":    pos_dist,
    }


# ------------------------------------------------------------------ #
# 出力
# ------------------------------------------------------------------ #
def print_report(report: dict) -> None:
    print()
    print("=" * 50)
    print("  Top-k Exposure Report")
    print("=" * 50)
    print(f"  期間: {report['period_start']} 〜 {report['period_end']}")
    print(f"  営業日数: {report['total_days']} 日")
    print()
    print(f"  平均投資比率   : {report['avg_exposure']:.1%}")
    print(f"  中央値投資比率 : {report['median_exposure']:.1%}")
    print(f"  投資50%未満の日: {report['under_50pct_ratio']:.1%}  ← キャッシュ多すぎ注意")
    print()
    print(f"  Top-k（k={report['topk']}）充足率: {report['topk_fill_rate']:.1%}")
    print(f"  平均保有銘柄数            : {report['avg_positions']:.2f} 銘柄")
    print()
    print("  投資比率 分布（10%刻み）:")
    for band, count in report["exposure_hist"].items():
        bar  = "█" * (count // 5 + 1) if count > 0 else ""
        print(f"    {band:>10s}: {count:4d}日  {bar}")
    print()
    print("  ポジション数 分布:")
    for n, count in sorted(report["positions_dist"].items()):
        pct = count / report["total_days"] * 100
        print(f"    {n}銘柄: {count:4d}日 ({pct:.1f}%)")
    print()
    print("  Phase 2 評価基準（推奨ライン）:")
    all_pass = True
    for key, (op, threshold, label) in CRITERIA.items():
        val = report.get(key, float("nan"))
        if op == ">=":
            ok = val >= threshold
        else:
            ok = val <= threshold
        mark = "✅" if ok else "❌"
        if not ok:
            all_pass = False
        print(f"    {mark} {label}: {val:.1%} (推奨 {op} {threshold:.0%})")
    print()
    print("  総合判定:", "✅ 推奨ライン達成" if all_pass else "❌ 稼働率改善が必要")
    print("=" * 50)
    print()


# ------------------------------------------------------------------ #
# エントリーポイント
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Top-k 実稼働率診断")
    parser.add_argument(
        "--log", default="logs/backtest_equity.csv",
        help="equity ログ CSV のパス"
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Top-k の k（max_positions と合わせる）"
    )
    parser.add_argument(
        "--out", default=None,
        help="JSON レポートの出力先（省略時は標準出力のみ）"
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] ログファイルが見つかりません: {log_path}", file=sys.stderr)
        print(
            "バックテストスクリプト側で以下を出力するように修正してください:\n"
            "  date,equity,invested_capital,positions",
            file=sys.stderr,
        )
        sys.exit(1)

    df     = load_equity_log(log_path)
    df     = compute_exposure(df)
    report = generate_report(df, k=args.k)

    print_report(report)

    if args.out:
        import json
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] レポートを保存: {out_path}")


if __name__ == "__main__":
    main()
