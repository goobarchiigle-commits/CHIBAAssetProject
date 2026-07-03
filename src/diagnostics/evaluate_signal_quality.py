"""
evaluate_signal_quality.py
orders_detail ログ × trades_log でシグナル品質を評価する。

実行:
    python src/diagnostics/evaluate_signal_quality.py

出力:
    1. シグナル日カバレッジ
    2. BUY/SELL 分布
    3. シグナル日 vs 非シグナル日リターン比較
    4. score 相関（score が null の間はスキップ・警告）
    5. score バケット別ヒットレート（同上）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

EVENTS_LOG = Path("logs/live_events.jsonl")
TRADES_LOG = Path("data/trades_log.csv")
EXEC_LOG   = Path("logs/execution_log.csv")

SEP = "─" * 60


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def load_orders_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"events log not found: {path}")

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") != "orders_detail":
                continue
            date = rec["ts"][:10]
            for o in rec["payload"].get("orders", []):
                rows.append({
                    "date":   date,
                    "symbol": o.get("symbol"),
                    "side":   o.get("side"),
                    "qty":    o.get("qty"),
                    "score":  o.get("score"),
                })

    if not rows:
        raise ValueError("orders_detail イベントが logs/live_events.jsonl に存在しません")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trades log not found: {path}")
    df = pd.read_csv(path)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def load_exec_log(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────
# Section 1: シグナル日カバレッジ・分布
# ─────────────────────────────────────────────────────────────

def report_signal_coverage(orders: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    print(SEP)
    print("1. シグナル日カバレッジ")
    print(SEP)

    signal_dates = set(orders["date"].dt.normalize().unique())
    trade_dates  = set(trades["entry_date"].dt.normalize().unique())
    overlap      = signal_dates & trade_dates

    print(f"  シグナル発生日数     : {len(signal_dates)}")
    print(f"  トレード記録日数     : {len(trade_dates)}")
    print(f"  重複（結合可能）日数 : {len(overlap)}")
    print(f"  カバレッジ          : {len(overlap)/max(len(signal_dates),1):.1%}")

    daily = (
        orders.groupby(orders["date"].dt.normalize())
        .agg(
            n_orders=("symbol", "count"),
            n_buy=("side", lambda x: (x == "BUY").sum()),
            n_sell=("side", lambda x: (x == "SELL").sum()),
        )
        .reset_index()
        .rename(columns={"date": "signal_date"})
    )

    print(f"\n  注文数/日  mean={daily['n_orders'].mean():.1f}  "
          f"max={daily['n_orders'].max()}  min={daily['n_orders'].min()}")
    print(f"  BUY 比率  : {orders['side'].eq('BUY').mean():.1%}")
    print(f"  SELL 比率 : {orders['side'].eq('SELL').mean():.1%}")

    return daily


# ─────────────────────────────────────────────────────────────
# Section 2: シグナル日 vs 非シグナル日リターン
# ─────────────────────────────────────────────────────────────

def report_signal_vs_nosignal_return(
    daily_signals: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    print()
    print(SEP)
    print("2. シグナル日 vs 非シグナル日  平均リターン比較")
    print(SEP)

    trades_d = trades.copy()
    trades_d["date"] = trades_d["entry_date"].dt.normalize()
    trades_d["on_signal"] = trades_d["date"].isin(
        daily_signals["signal_date"].dt.normalize()
    )

    for flag, label in [(True, "シグナル日"), (False, "非シグナル日")]:
        sub = trades_d[trades_d["on_signal"] == flag]["ret"]
        if sub.empty:
            print(f"  {label}: データなし")
            continue
        print(f"  {label} (n={len(sub)}): "
              f"mean={sub.mean():.4f}  std={sub.std():.4f}  "
              f"hit={( sub > 0).mean():.1%}  median={sub.median():.4f}")

    sig  = trades_d[trades_d["on_signal"]]["ret"].dropna()
    nosig = trades_d[~trades_d["on_signal"]]["ret"].dropna()
    if len(sig) >= 2 and len(nosig) >= 2:
        t_stat, p_val = stats.ttest_ind(sig, nosig, equal_var=False)
        print(f"\n  Welch t-test: t={t_stat:.3f}  p={p_val:.4f}  "
              f"{'有意差あり (p<0.05)' if p_val < 0.05 else '有意差なし'}")


# ─────────────────────────────────────────────────────────────
# Section 3: score 相関 / score=null 時は execution-based fallback
# ─────────────────────────────────────────────────────────────

def _execution_based_evaluation(orders: pd.DataFrame, trades: pd.DataFrame) -> None:
    """
    score=null 時のフォールバック。
    side 方向精度と qty を conviction プロキシとして使う。
    qty が全銘柄同値の場合はその旨を警告してスキップする。
    """
    print("  [FALLBACK] execution-based evaluation (score=null)")

    orders_d = orders.copy()
    orders_d["date_norm"] = orders_d["date"].dt.normalize()

    trades_d = trades.copy()
    trades_d["date_norm"] = trades_d["entry_date"].dt.normalize()

    # ── qty 分散チェック ──────────────────────────────────────
    qty_unique = orders_d["qty"].nunique()
    if qty_unique == 1:
        print(f"  [WARN] qty 全銘柄同値 ({orders_d['qty'].iloc[0]:,}) "
              "— conviction プロキシ無効")
        qty_proxy = False
    else:
        print(f"  qty ユニーク数: {qty_unique} → conviction プロキシ使用")
        qty_proxy = True

    # ── サイド方向精度 ────────────────────────────────────────
    side_by_date = (
        orders_d.groupby("date_norm")
        .agg(
            dominant_side=("side", lambda x: "BUY" if (x == "BUY").mean() >= 0.5 else "SELL"),
            buy_ratio=("side", lambda x: (x == "BUY").mean()),
        )
        .reset_index()
    )

    merged = side_by_date.merge(trades_d[["date_norm", "ret"]], on="date_norm", how="inner")
    if merged.empty:
        print("  [WARN] 日次結合結果が空 — trades_log と日付が重複していません")
        return

    merged["direction_correct"] = (
        ((merged["dominant_side"] == "BUY")  & (merged["ret"] > 0)) |
        ((merged["dominant_side"] == "SELL") & (merged["ret"] < 0))
    )
    hit = merged["direction_correct"].mean()
    n_buy_days  = merged["dominant_side"].eq("BUY").sum()
    n_sell_days = merged["dominant_side"].eq("SELL").sum()
    print(f"  サイド方向精度 : {hit:.1%}  "
          f"(n={len(merged)}, BUY日={n_buy_days}, SELL日={n_sell_days})")

    # ── BUY 比率 × リターン相関 ───────────────────────────────
    if len(merged) >= 3:
        r, p = stats.spearmanr(merged["buy_ratio"], merged["ret"])
        print(f"  BUY比率×リターン: Spearman r={r:.4f}  p={p:.4f}  "
              f"{'有意 (p<0.05)' if p < 0.05 else '非有意'}")

    # ── qty プロキシ相関（qty が多様な場合のみ） ───────────────
    if qty_proxy:
        qty_by_date = (
            orders_d.groupby("date_norm")["qty"].mean().reset_index()
        )
        merged_qty = qty_by_date.merge(
            trades_d[["date_norm", "ret"]], on="date_norm", how="inner"
        )
        if len(merged_qty) >= 3:
            r_q, p_q = stats.spearmanr(merged_qty["qty"], merged_qty["ret"])
            print(f"  qty×リターン    : Spearman r={r_q:.4f}  p={p_q:.4f}  "
                  f"{'有意 (p<0.05)' if p_q < 0.05 else '非有意'}")


def report_score_correlation(orders: pd.DataFrame, trades: pd.DataFrame) -> None:
    print()
    print(SEP)
    print("3. score × リターン 相関")
    print(SEP)

    n_nonnull = orders["score"].notna().sum()
    n_total   = len(orders)

    if n_nonnull == 0:
        print(f"  score が全 {n_total} 件 null")
        _execution_based_evaluation(orders, trades)
        return

    print(f"  score 非 null: {n_nonnull}/{n_total} ({n_nonnull/n_total:.1%})")

    orders_scored = orders[orders["score"].notna()].copy()
    orders_scored["date_norm"] = orders_scored["date"].dt.normalize()

    trades_d = trades.copy()
    trades_d["date_norm"] = trades_d["entry_date"].dt.normalize()

    merged = orders_scored.merge(
        trades_d[["date_norm", "ret"]], on="date_norm", how="inner"
    )
    if merged.empty:
        print("  [WARN] score × return の結合結果が空です")
        return

    r, p = stats.spearmanr(merged["score"], merged["ret"])
    print(f"  Spearman r = {r:.4f}  p = {p:.4f}  "
          f"{'有意 (p<0.05)' if p < 0.05 else '非有意'}")
    print(f"  n = {len(merged)}")


# ─────────────────────────────────────────────────────────────
# Section 4: score バケット別ヒットレート / fallback: qty バケット
# ─────────────────────────────────────────────────────────────

def report_score_buckets(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    n_buckets: int = 4,
) -> None:
    print()
    print(SEP)
    print(f"4. score バケット別ヒットレート (n_buckets={n_buckets})")
    print(SEP)

    trades_d = trades.copy()
    trades_d["date_norm"] = trades_d["entry_date"].dt.normalize()

    if orders["score"].isna().all():
        # ── qty バケットフォールバック ─────────────────────────
        if orders["qty"].nunique() == 1:
            print("  [SKIP] score=null かつ qty も一様 — バケット分析不可")
            print("  → generate_orders で score を付与してください")
            return

        print("  [FALLBACK] qty バケット分析 (score=null)")
        orders_q = orders.copy()
        orders_q["date_norm"] = orders_q["date"].dt.normalize()
        orders_q["bucket"] = pd.qcut(
            orders_q["qty"], q=n_buckets, labels=False, duplicates="drop"
        )
        merged = orders_q.merge(trades_d[["date_norm", "ret"]], on="date_norm", how="inner")
        if merged.empty:
            print("  [WARN] 結合結果が空です")
            return
        summary = (
            merged.groupby("bucket")["ret"]
            .agg(n="count", mean_ret="mean", hit_rate=lambda x: (x > 0).mean())
        )
        print(summary.to_string())
        return

    # ── score バケット（通常パス） ─────────────────────────────
    orders_s = orders[orders["score"].notna()].copy()
    orders_s["date_norm"] = orders_s["date"].dt.normalize()
    orders_s["bucket"] = pd.qcut(
        orders_s["score"], q=n_buckets, labels=False, duplicates="drop"
    )
    merged = orders_s.merge(trades_d[["date_norm", "ret"]], on="date_norm", how="inner")
    if merged.empty:
        print("  [WARN] 結合結果が空です")
        return
    summary = (
        merged.groupby("bucket")["ret"]
        .agg(n="count", mean_ret="mean", hit_rate=lambda x: (x > 0).mean())
    )
    print(summary.to_string())


# ─────────────────────────────────────────────────────────────
# Section 5: スリッページ分析（execution_log.csv が存在する場合）
# ─────────────────────────────────────────────────────────────

def report_slippage(exec_log: pd.DataFrame | None) -> None:
    print()
    print(SEP)
    print("5. スリッページ分析 (execution_log.csv)")
    print(SEP)

    if exec_log is None:
        print("  [SKIP] logs/execution_log.csv が存在しません")
        print("  → live 実行後に自動生成されます")
        return

    slippage = exec_log["slippage"].dropna()
    if slippage.empty:
        print("  [WARN] slippage 列に有効値なし")
        return

    print(f"  mean  : {slippage.mean():.6f}")
    print(f"  std   : {slippage.std():.6f}")
    print(f"  min   : {slippage.min():.6f}")
    print(f"  max   : {slippage.max():.6f}")

    by_side = exec_log.groupby("side")["slippage"].mean()
    print(f"  BUY  avg slippage: {by_side.get('BUY', float('nan')):.6f}")
    print(f"  SELL avg slippage: {by_side.get('SELL', float('nan')):.6f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("  Signal Quality Evaluation Report")
    print(SEP)

    try:
        orders = load_orders_detail(EVENTS_LOG)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[FATAL] orders_detail 読み込み失敗: {e}")

    try:
        trades = load_trades(TRADES_LOG)
    except FileNotFoundError as e:
        sys.exit(f"[FATAL] trades_log 読み込み失敗: {e}")

    exec_log = load_exec_log(EXEC_LOG)

    daily_signals = report_signal_coverage(orders, trades)
    report_signal_vs_nosignal_return(daily_signals, trades)
    report_score_correlation(orders, trades)
    report_score_buckets(orders, trades)
    report_slippage(exec_log)

    print()
    print(SEP)
    print("  完了")
    print(SEP)


if __name__ == "__main__":
    main()
