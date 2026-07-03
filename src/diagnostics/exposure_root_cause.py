"""
diagnostics/exposure_root_cause.py
Exposure 低下の根本原因を5因子に分解する診断ツール

【分解する5因子】
  1. candidate_shortage  : 候補銘柄数 < k（universe が薄い）
  2. topk_not_filled     : positions < k（エントリー条件が通らない）
  3. size_constraint     : 単元株・株価制約で actual_weight < target_weight の乖離
  4. weight_cap          : max_single_weight 上限による切り捨て
  5. cb_block            : サーキットブレーカー発動中

【入力ログ（logs/backtest_daily_state.csv）】

  必須カラム:
    date             : 日付
    equity           : ポートフォリオ評価額
    invested_capital : 実際に投資済みの金額
    positions        : 実際の保有銘柄数
    topk_target      : Top-k の k（max_positions と同値が通常）
    candidates       : エントリー候補数（BUY シグナルが出た銘柄数）
    blocked_by_cb    : CB 発動中フラグ（0/1）

  拡張カラム（あるとより詳細に分析できる）:
    target_weight         : 目標配分比率（1/k など）
    actual_weight         : 実際の平均配分比率
    skipped_due_to_price  : 株価高すぎでスキップされた件数
    skipped_due_to_lot    : 単元株制約でスキップされた件数
    skipped_due_to_cash   : 資金不足でスキップされた件数
    entry_signals         : 当日の BUY シグナル総数

【使い方】
  python diagnostics/exposure_root_cause.py --log logs/backtest_daily_state.csv --k 3
  python diagnostics/exposure_root_cause.py --log logs/backtest_daily_state.csv --k 3 --out results/root_cause.json
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
# 判定閾値
# ------------------------------------------------------------------ #
THRESHOLDS = {
    "candidate_shortage_ratio": 0.20,   # 20%超 → 候補銘柄不足が主因
    "topk_not_filled_ratio":    0.20,   # 20%超 → エントリー条件が主因
    "avg_weight_gap":           0.10,   # 10%pt超 → サイズ制限が主因
    "cb_block_ratio":           0.10,   # 10%超 → CB が主因
    "price_skip_ratio":         0.15,   # 15%超 → 株価制約が主因
}


# ------------------------------------------------------------------ #
# 診断クラス
# ------------------------------------------------------------------ #
class ExposureDiagnostics:

    def __init__(self, df: pd.DataFrame, k: int):
        self.df = df.copy()
        self.k  = k
        self._validate()

    def _validate(self) -> None:
        required = {"date", "equity", "invested_capital", "positions",
                    "topk_target", "candidates", "blocked_by_cb"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"必須カラムが不足しています: {sorted(missing)}\n"
                f"バックテスト側でこれらの列を出力してください。"
            )

    # ── 因子1: candidate 不足 ──────────────────────────────────────
    def _candidate_shortage(self) -> float:
        """candidates < k の日の割合"""
        return float((self.df["candidates"] < self.k).mean())

    # ── 因子2: Top-k 未充足 ────────────────────────────────────────
    def _topk_not_filled(self) -> float:
        """実保有数 < k の日の割合（候補はあるのに埋まらない）"""
        return float((self.df["positions"] < self.k).mean())

    # ── 因子3: サイズ制限（target vs actual の乖離） ──────────────
    def _size_constraint(self) -> dict:
        has_weight = ("target_weight" in self.df.columns
                      and "actual_weight" in self.df.columns)
        if not has_weight:
            return {"avg_weight_gap": None, "size_gap_severity": None}

        gap = self.df["target_weight"] - self.df["actual_weight"]
        avg_gap = float(gap.mean())
        # gap > 10%pt の日の割合（= 「かなりショートしている日」）
        severe_ratio = float((gap > 0.10).mean())
        return {
            "avg_weight_gap":     avg_gap,
            "size_gap_severity":  severe_ratio,
        }

    # ── 因子4: max_single_weight キャップ ──────────────────────────
    def _weight_cap(self) -> float | None:
        """actual_weight が 1/k の 70% 未満の日の割合（過小配分）"""
        if "actual_weight" not in self.df.columns:
            return None
        threshold = (1.0 / self.k) * 0.70
        return float((self.df["actual_weight"] < threshold).mean())

    # ── 因子5: CB ブロック ─────────────────────────────────────────
    def _cb_block(self) -> float:
        return float(self.df["blocked_by_cb"].mean())

    # ── 拡張: スキップ内訳 ─────────────────────────────────────────
    def _skip_breakdown(self) -> dict:
        out: dict = {}
        for col, label in [
            ("skipped_due_to_price", "price_skip_ratio"),
            ("skipped_due_to_lot",   "lot_skip_ratio"),
            ("skipped_due_to_cash",  "cash_skip_ratio"),
            ("entry_signals",        "avg_entry_signals"),
        ]:
            if col not in self.df.columns:
                out[label] = None
                continue
            if col == "entry_signals":
                out[label] = float(self.df[col].mean())
            else:
                # シグナルがあった日のうち、スキップ > 0 の割合
                has_signal = self.df.get("entry_signals", pd.Series(1, index=self.df.index)) > 0
                denom = has_signal.sum()
                out[label] = float((self.df.loc[has_signal, col] > 0).sum() / denom) if denom > 0 else 0.0
        return out

    # ── 統合 ──────────────────────────────────────────────────────
    def summary(self) -> dict:
        self.df["exposure"] = (
            self.df["invested_capital"] / self.df["equity"].replace(0, float("nan"))
        ).clip(0, 1)

        sc = self._size_constraint()

        report: dict = {
            # 基礎
            "period_start":              str(self.df["date"].min().date()),
            "period_end":                str(self.df["date"].max().date()),
            "total_days":                int(len(self.df)),
            "k":                         self.k,
            # Exposure
            "avg_exposure":              float(self.df["exposure"].mean()),
            "median_exposure":           float(self.df["exposure"].median()),
            # 因子1: candidate 不足
            "candidate_shortage_ratio":  self._candidate_shortage(),
            "avg_candidates":            float(self.df["candidates"].mean()),
            # 因子2: Top-k 未充足
            "topk_not_filled_ratio":     self._topk_not_filled(),
            "avg_positions":             float(self.df["positions"].mean()),
            # 因子3: サイズ制限
            "avg_weight_gap":            sc["avg_weight_gap"],
            "size_gap_severity_ratio":   sc["size_gap_severity"],
            # 因子4: weight キャップ
            "weight_cap_ratio":          self._weight_cap(),
            # 因子5: CB
            "cb_block_ratio":            self._cb_block(),
        }
        # 拡張カラム
        report.update(self._skip_breakdown())

        # 主因判定
        report["root_cause"] = self._identify_root_cause(report)

        return report

    def _identify_root_cause(self, r: dict) -> str:
        causes = []

        if r["candidate_shortage_ratio"] > THRESHOLDS["candidate_shortage_ratio"]:
            causes.append(f"候補銘柄不足（{r['candidate_shortage_ratio']:.0%}の日でcandidates<{self.k}）")

        if r["cb_block_ratio"] > THRESHOLDS["cb_block_ratio"]:
            causes.append(f"CB発動（{r['cb_block_ratio']:.0%}の日でBUY停止）")

        # サイズ制限は weight_gap と price_skip の両方を見る
        if (r["avg_weight_gap"] is not None
                and r["avg_weight_gap"] > THRESHOLDS["avg_weight_gap"]):
            causes.append(f"ポジションサイズ制限（target比 {r['avg_weight_gap']:.1%}pt 乖離）")

        if (r.get("price_skip_ratio") is not None
                and r["price_skip_ratio"] > THRESHOLDS["price_skip_ratio"]):
            causes.append(f"株価制約（{r['price_skip_ratio']:.0%}の日でBUY候補を価格理由でスキップ）")

        if (r["topk_not_filled_ratio"] > THRESHOLDS["topk_not_filled_ratio"]
                and not causes):
            causes.append(f"エントリー条件の厳しさ（{r['topk_not_filled_ratio']:.0%}の日でpositions<{self.k}）")

        if not causes:
            return "複合要因（各因子が閾値未満。バックテストログの拡張カラムを追加すると精度向上）"

        return " / ".join(causes)


# ------------------------------------------------------------------ #
# 出力
# ------------------------------------------------------------------ #
def print_report(report: dict) -> None:
    SEP = "=" * 54

    print()
    print(SEP)
    print("  Exposure Root Cause Diagnostics")
    print(SEP)
    print(f"  期間  : {report['period_start']} 〜 {report['period_end']}")
    print(f"  日数  : {report['total_days']} 日  /  k = {report['k']}")
    print()

    print("  ── 概要 ──────────────────────────────────────")
    print(f"  平均 exposure        : {report['avg_exposure']:.1%}")
    print(f"  中央値 exposure      : {report['median_exposure']:.1%}")
    print()

    # 因子別
    def row(label, val, fmt="pct", threshold=None):
        if val is None:
            v_str = "N/A（ログなし）"
            flag  = ""
        elif fmt == "pct":
            v_str = f"{val:.1%}"
            flag  = "  ⚠" if (threshold and val > threshold) else ""
        elif fmt == "val":
            v_str = f"{val:.3f}"
            flag  = "  ⚠" if (threshold and val > threshold) else ""
        elif fmt == "num":
            v_str = f"{val:.2f}"
            flag  = ""
        else:
            v_str = str(val)
            flag  = ""
        print(f"  {label:<28}: {v_str}{flag}")

    print("  ── 因子1: 候補銘柄不足 ──────────────────────")
    row("候補不足率 (candidates<k)", report["candidate_shortage_ratio"],
        threshold=THRESHOLDS["candidate_shortage_ratio"])
    row("平均候補数", report["avg_candidates"], fmt="num")
    print()

    print("  ── 因子2: Top-k 未充足 ──────────────────────")
    row("Top-k 未充足率 (pos<k)", report["topk_not_filled_ratio"],
        threshold=THRESHOLDS["topk_not_filled_ratio"])
    row("平均保有銘柄数", report["avg_positions"], fmt="num")
    print()

    print("  ── 因子3: サイズ制限 (target vs actual) ────")
    row("平均 weight 乖離", report["avg_weight_gap"],
        fmt="pct", threshold=THRESHOLDS["avg_weight_gap"])
    row("乖離10%pt超の日の割合", report["size_gap_severity_ratio"])
    print()

    print("  ── 因子4: weight キャップ ───────────────────")
    row("過小配分率 (<1/k×70%)", report["weight_cap_ratio"])
    print()

    print("  ── 因子5: CB ブロック ───────────────────────")
    row("CB ブロック率", report["cb_block_ratio"],
        threshold=THRESHOLDS["cb_block_ratio"])
    print()

    # 拡張カラム（存在する場合）
    ext_keys = ["price_skip_ratio", "lot_skip_ratio", "cash_skip_ratio", "avg_entry_signals"]
    if any(report.get(k) is not None for k in ext_keys):
        print("  ── 拡張: スキップ内訳 ───────────────────────")
        row("株価制約スキップ率",    report.get("price_skip_ratio"),
            threshold=THRESHOLDS["price_skip_ratio"])
        row("単元株制約スキップ率",  report.get("lot_skip_ratio"))
        row("資金不足スキップ率",    report.get("cash_skip_ratio"))
        row("平均 BUY シグナル数",   report.get("avg_entry_signals"), fmt="num")
        print()

    print("  ── 主因判定 ─────────────────────────────────")
    print(f"  → {report['root_cause']}")
    print(SEP)
    print()


# ------------------------------------------------------------------ #
# エントリーポイント
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Exposure 低下の根本原因診断")
    parser.add_argument("--log", default="logs/backtest_daily_state.csv",
                        help="バックテスト日次ログ CSV")
    parser.add_argument("--k",   type=int, default=3,
                        help="Top-k の k（max_positions と合わせる）")
    parser.add_argument("--out", default=None,
                        help="JSON レポートの出力先（省略時は標準出力のみ）")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] ログが見つかりません: {log_path}", file=sys.stderr)
        print(
            "\nバックテスト側で以下のカラムを出力してください:\n"
            "  date, equity, invested_capital, positions,\n"
            "  topk_target, candidates, blocked_by_cb\n"
            "\n拡張カラム（より精密な分析に）:\n"
            "  target_weight, actual_weight,\n"
            "  entry_signals, skipped_due_to_price,\n"
            "  skipped_due_to_lot, skipped_due_to_cash",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(log_path, parse_dates=["date"])

    diag   = ExposureDiagnostics(df, k=args.k)
    report = diag.summary()

    print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # None を "N/A" に変換してから保存
        out_path.write_text(
            json.dumps(
                {k: (v if v is not None else "N/A") for k, v in report.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[INFO] レポート保存: {out_path}")


if __name__ == "__main__":
    main()
