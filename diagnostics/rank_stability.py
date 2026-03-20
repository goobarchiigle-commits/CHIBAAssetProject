"""
diagnostics/rank_stability.py
Top-k ランキングの安定性（回転率）を診断するツール

【測る指標】
  rank_overlap_ratio  : 前日 Top-k と今日 Top-k の重複率
                        = len(prev_topk ∩ today_topk) / k
  daily_turnover      : 1 - rank_overlap_ratio（入れ替わり率）
  avg_rsr_spread      : Top-k 内の RSR 最大/最小差（分散の目安）

【評価基準（MEMORY.md の知見）】
  overlap 0.3-0.6  : 理想的（適度な回転）
  overlap > 0.8    : RSR 偏り疑い（同じ銘柄が居座り続ける）
  overlap < 0.3    : 不安定すぎ（ノイズトレード増加）

【入力ログ形式（logs/backtest_topk.csv）】
  date         : 日付
  rank1 〜 rank_k: その日の Top-k 銘柄コード（列名は rank1, rank2, rank3...）

  または

  date, topk_symbols: カンマ区切り文字列（例: "8035.T,9101.T,6857.T"）

【使い方】
  python diagnostics/rank_stability.py --log logs/backtest_topk.csv --k 3
  python diagnostics/rank_stability.py --log logs/backtest_topk.csv --k 3 --out results/rank_stability.json
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# 評価基準
# ------------------------------------------------------------------ #
IDEAL_OVERLAP_LOW  = 0.30
IDEAL_OVERLAP_HIGH = 0.60
STICKY_THRESHOLD   = 0.80   # これ以上 → RSR 偏り疑い
UNSTABLE_THRESHOLD = 0.30   # これ以下 → 不安定


# ------------------------------------------------------------------ #
# ログ読み込み
# ------------------------------------------------------------------ #
def _load_topk_log(path: Path, k: int) -> pd.DataFrame:
    """
    Top-k ログを読み込み、各日の top-k セットを list[set] で返す。

    対応フォーマット:
      A) date, rank1, rank2, rank3 列
      B) date, topk_symbols 列（カンマ区切り文字列）
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # フォーマット判定
    rank_cols = [c for c in df.columns if c.startswith("rank")]
    if rank_cols:
        # フォーマット A
        df["topk_set"] = df[rank_cols[:k]].apply(
            lambda row: frozenset(v for v in row if pd.notna(v)), axis=1
        )
    elif "topk_symbols" in df.columns:
        # フォーマット B
        df["topk_set"] = df["topk_symbols"].apply(
            lambda s: frozenset(x.strip() for x in str(s).split(",") if x.strip())
        )
    else:
        raise ValueError(
            "rank1/rank2.../rankN 列か topk_symbols 列が必要です。"
        )

    return df[["date", "topk_set"]]


# ------------------------------------------------------------------ #
# 診断
# ------------------------------------------------------------------ #
class RankStabilityAnalyzer:

    def __init__(self, df: pd.DataFrame, k: int):
        self.df = df.copy()
        self.k  = k

    def _compute_overlap(self) -> pd.Series:
        """連続する 2 日間の Top-k 重複率を計算する。"""
        overlaps = []
        sets = self.df["topk_set"].tolist()
        for i in range(1, len(sets)):
            if len(sets[i]) == 0:
                overlaps.append(float("nan"))
                continue
            overlap = len(sets[i - 1] & sets[i]) / self.k
            overlaps.append(overlap)
        return pd.Series([float("nan")] + overlaps, index=self.df.index)

    def _symbol_frequency(self) -> pd.Series:
        """各銘柄が Top-k に入った日数の頻度（居座り銘柄の特定に使う）。"""
        from collections import Counter
        counter: Counter = Counter()
        for s in self.df["topk_set"]:
            counter.update(s)
        total = len(self.df)
        freq = pd.Series({sym: cnt / total for sym, cnt in counter.items()})
        return freq.sort_values(ascending=False)

    def _longest_streak(self) -> dict[str, int]:
        """各銘柄の最長連続 Top-k 在位日数。"""
        from collections import defaultdict
        streak:     defaultdict[str, int] = defaultdict(int)
        max_streak: defaultdict[str, int] = defaultdict(int)
        prev_set: frozenset = frozenset()

        for row in self.df["topk_set"]:
            appeared = row
            for sym in appeared:
                if sym in prev_set:
                    streak[sym] += 1
                else:
                    streak[sym] = 1
                max_streak[sym] = max(max_streak[sym], streak[sym])
            for sym in prev_set - appeared:
                streak[sym] = 0
            prev_set = appeared

        return dict(max_streak)

    def summary(self) -> dict:
        overlap = self._compute_overlap().dropna()
        freq    = self._symbol_frequency()
        streak  = self._longest_streak()

        avg_overlap    = float(overlap.mean())
        avg_turnover   = 1.0 - avg_overlap
        sticky_symbols = freq[freq > 0.80].index.tolist()   # 80%超の日に Top-k に入る銘柄

        report = {
            "period_start":         str(self.df["date"].min().date()),
            "period_end":           str(self.df["date"].max().date()),
            "total_days":           int(len(self.df)),
            "k":                    self.k,
            # 主要指標
            "avg_overlap_ratio":    avg_overlap,
            "avg_daily_turnover":   avg_turnover,
            "median_overlap_ratio": float(overlap.median()),
            "overlap_gt_80pct_ratio": float((overlap > 0.80).mean()),
            "overlap_lt_30pct_ratio": float((overlap < 0.30).mean()),
            # 銘柄別
            "symbol_frequency":     freq.head(10).round(3).to_dict(),
            "sticky_symbols":       sticky_symbols,
            "longest_streak":       dict(sorted(streak.items(),
                                                key=lambda x: -x[1])[:10]),
            # 診断
            "diagnosis": self._diagnose(avg_overlap, sticky_symbols),
        }
        return report

    def _diagnose(self, avg_overlap: float, sticky: list[str]) -> str:
        if avg_overlap > STICKY_THRESHOLD:
            sticky_str = ", ".join(sticky[:5]) if sticky else "特定銘柄"
            return (
                f"RSR 偏り疑い: overlap={avg_overlap:.2f} > {STICKY_THRESHOLD}。"
                f" {sticky_str} が居座り続けています。"
                f" min_rsr 閾値の引き上げ・ユニバース見直しを検討してください。"
            )
        if avg_overlap < UNSTABLE_THRESHOLD:
            return (
                f"不安定: overlap={avg_overlap:.2f} < {UNSTABLE_THRESHOLD}。"
                f" ランキング変動が大きすぎます。"
                f" RSR 計算期間を延ばすか、ランキング変動のダンピング追加を検討してください。"
            )
        return (
            f"正常範囲: overlap={avg_overlap:.2f}（理想 {IDEAL_OVERLAP_LOW}-{IDEAL_OVERLAP_HIGH}）。"
            f" 適度な回転率が維持されています。"
        )


# ------------------------------------------------------------------ #
# 出力
# ------------------------------------------------------------------ #
def print_report(report: dict) -> None:
    SEP = "=" * 54
    print()
    print(SEP)
    print("  Rank Stability Diagnostics")
    print(SEP)
    print(f"  期間: {report['period_start']} 〜 {report['period_end']}")
    print(f"  日数: {report['total_days']} 日  /  k = {report['k']}")
    print()

    ov = report["avg_overlap_ratio"]
    if ov > STICKY_THRESHOLD:
        flag = "⚠ RSR 偏り疑い"
    elif ov < UNSTABLE_THRESHOLD:
        flag = "⚠ 不安定"
    else:
        flag = "✅ 正常"

    print(f"  ── 回転率 ────────────────────────────────────")
    print(f"  平均 overlap ratio   : {report['avg_overlap_ratio']:.3f}  {flag}")
    print(f"  平均 daily turnover  : {report['avg_daily_turnover']:.3f}")
    print(f"  中央値 overlap       : {report['median_overlap_ratio']:.3f}")
    print(f"  overlap > 80% の日率 : {report['overlap_gt_80pct_ratio']:.1%}")
    print(f"  overlap < 30% の日率 : {report['overlap_lt_30pct_ratio']:.1%}")
    print(f"  参考: 理想範囲 = {IDEAL_OVERLAP_LOW}-{IDEAL_OVERLAP_HIGH}")
    print()

    print("  ── 銘柄別 Top-k 在位頻度（上位 10 銘柄）─────")
    for sym, freq in report["symbol_frequency"].items():
        bar  = "█" * int(freq * 20)
        mark = " ⚠" if freq > 0.80 else ""
        print(f"    {sym:<12}: {freq:.1%}  {bar}{mark}")
    print()

    if report["sticky_symbols"]:
        print(f"  ⚠ 居座り銘柄（>80%）: {report['sticky_symbols']}")
        print()

    print("  ── 最長連続 Top-k 在位日数 ───────────────────")
    for sym, days in list(report["longest_streak"].items())[:8]:
        print(f"    {sym:<12}: {days} 日連続")
    print()

    print("  ── 診断 ──────────────────────────────────────")
    print(f"  → {report['diagnosis']}")
    print(SEP)
    print()


# ------------------------------------------------------------------ #
# エントリーポイント
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Top-k ランキング安定性診断")
    parser.add_argument("--log", default="logs/backtest_topk.csv",
                        help="Top-k ログ CSV（rank1,rank2,...またはtopk_symbols列）")
    parser.add_argument("--k",   type=int, default=3, help="Top-k の k")
    parser.add_argument("--out", default=None, help="JSON 出力先")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] ログが見つかりません: {log_path}", file=sys.stderr)
        print(
            "\nバックテスト側で以下のどちらかを出力してください:\n"
            "  A) date, rank1, rank2, rank3 （rank1 = RSR 1位銘柄）\n"
            "  B) date, topk_symbols        （例: '8035.T,9101.T,6857.T'）",
            file=sys.stderr,
        )
        sys.exit(1)

    df     = _load_topk_log(log_path, k=args.k)
    report = RankStabilityAnalyzer(df, k=args.k).summary()
    print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] レポート保存: {out_path}")


if __name__ == "__main__":
    main()
