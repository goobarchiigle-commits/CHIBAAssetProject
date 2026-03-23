"""
research/weekly_report.py

週次・月次レビューレポート（意思決定支援）

出力:
  - 週次: 取引数 / 勝率 / 期待値 / 平均リターン / signals_per_week
  - 月次: 資産変化 / MaxDD / regime別成績（bull/neutral/bear）
  - 供給診断: rsr_pass / near_breakout / rsr_dispersion の推移

データソース:
  - logs/trades.jsonl       : BUY/SELL 実績（update_state_after_execution が書き込む）
  - logs/diagnostics/metrics.jsonl : 日次診断メトリクス（regime / supply）

実行:
  python research/weekly_report.py
  python research/weekly_report.py --weeks 4
  python research/weekly_report.py --since 2026-04-01
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from datetime import datetime, date, timedelta, timezone, tzinfo

import numpy as np
import pandas as pd


JST            = timezone(timedelta(hours=9))
TRADES_PATH    = Path("logs/trades.jsonl")
METRICS_PATH   = Path("logs/diagnostics/metrics.jsonl")


# ── データロード ──────────────────────────────────────────────────────

def _load_trades() -> pd.DataFrame:
    if not TRADES_PATH.exists():
        return pd.DataFrame()
    lines = [l for l in TRADES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in lines])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    lines = [l for l in METRICS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # 同日複数実行 → 最新を残す
    df = df.sort_values("run_at").groupby("date").last().reset_index()
    return df


# ── 週次集計 ──────────────────────────────────────────────────────────

def _weekly_trade_stats(trades: pd.DataFrame, since: pd.Timestamp | None) -> None:
    if trades.empty or "side" not in trades.columns:
        print("  取引実績なし（SELLレコードが0件）")
        _signals_per_week_from_metrics()
        return
    sells = trades[trades["side"] == "SELL"].copy()
    if since is not None:
        sells = sells[sells["date"] >= since]

    if sells.empty:
        print("  取引実績なし（SELLレコードが0件）")
        return

    sells["week"] = sells["date"].dt.to_period("W")
    weekly = sells.groupby("week").agg(
        trades   = ("pnl", "count"),
        total_pnl= ("pnl", "sum"),
        wins     = ("pnl", lambda x: (x > 0).sum()),
    ).reset_index()
    weekly["win_rate"]   = weekly["wins"] / weekly["trades"]
    weekly["expectancy"] = weekly["total_pnl"] / weekly["trades"]

    print(f"\n  {'週':>12} {'取引数':>6} {'合計PnL':>10} {'勝率':>7} {'期待値/取引':>12}")
    print("  " + "-" * 52)
    for _, row in weekly.tail(8).iterrows():
        sign = "+" if row["total_pnl"] >= 0 else ""
        print(
            f"  {str(row['week']):>12} {row['trades']:>6.0f}"
            f" {sign}¥{row['total_pnl']:>8,.0f}"
            f" {row['win_rate']:>7.1%}"
            f" {sign}¥{row['expectancy']:>10,.0f}"
        )

    # 直近5営業日のシグナル密度
    print()
    _signals_per_week_from_metrics()


def _signals_per_week_from_metrics() -> None:
    metrics = _load_metrics()
    if metrics.empty:
        return
    col = "candidate_count" if "candidate_count" in metrics.columns else None
    if col is None:
        return
    last5 = metrics.sort_values("date").tail(5)
    spw = int(last5[col].sum())
    avg = float(last5[col].mean())
    print(f"  直近5営業日 BUY候補合計: {spw}件  (平均 {avg:.1f}/日)  目安: 2〜6/週")


# ── 月次集計 ──────────────────────────────────────────────────────────

def _monthly_stats(trades: pd.DataFrame, metrics: pd.DataFrame, since: pd.Timestamp | None) -> None:
    sells = trades[trades["side"] == "SELL"].copy() if (not trades.empty and "side" in trades.columns) else pd.DataFrame()
    if since is not None and not sells.empty:
        sells = sells[sells["date"] >= since]

    # --- 月次PnL ---
    print(f"\n  {'月':>8} {'取引':>5} {'勝率':>7} {'PnL合計':>10} {'平均%':>8} {'期待値':>10}")
    print("  " + "-" * 54)
    if not sells.empty and "pnl" in sells.columns:
        sells["month"] = sells["date"].dt.to_period("M")
        monthly = sells.groupby("month").agg(
            trades   = ("pnl", "count"),
            total_pnl= ("pnl", "sum"),
            wins     = ("pnl", lambda x: (x > 0).sum()),
            avg_pct  = ("pnl_pct", "mean"),
        ).reset_index()
        monthly["win_rate"]   = monthly["wins"] / monthly["trades"]
        monthly["expectancy"] = monthly["total_pnl"] / monthly["trades"]
        for _, row in monthly.tail(6).iterrows():
            sign = "+" if row["total_pnl"] >= 0 else ""
            avg_pct = row.get("avg_pct", 0) or 0
            print(
                f"  {str(row['month']):>8} {row['trades']:>5.0f}"
                f" {row['win_rate']:>7.1%}"
                f" {sign}¥{row['total_pnl']:>8,.0f}"
                f" {avg_pct:>+8.1%}"
                f" {sign}¥{row['expectancy']:>8,.0f}"
            )
    else:
        print("  データなし")

    # --- Regime別成績 ---
    print(f"\n  ── Regime別成績 ──")
    if not sells.empty and "entry_regime" in sells.columns and "pnl" in sells.columns:
        regime_grp = sells.dropna(subset=["pnl"]).groupby("entry_regime").agg(
            trades   = ("pnl", "count"),
            total_pnl= ("pnl", "sum"),
            wins     = ("pnl", lambda x: (x > 0).sum()),
            avg_pct  = ("pnl_pct", "mean"),
        ).reset_index()
        regime_grp["win_rate"] = regime_grp["wins"] / regime_grp["trades"]
        print(f"  {'regime':>10} {'取引':>5} {'勝率':>7} {'PnL合計':>10} {'平均%':>8}")
        print("  " + "-" * 44)
        for _, row in regime_grp.iterrows():
            sign    = "+" if row["total_pnl"] >= 0 else ""
            avg_pct = row.get("avg_pct", 0) or 0
            print(
                f"  {str(row['entry_regime']):>10} {row['trades']:>5.0f}"
                f" {row['win_rate']:>7.1%}"
                f" {sign}¥{row['total_pnl']:>8,.0f}"
                f" {avg_pct:>+8.1%}"
            )
    else:
        print("  データなし（regime情報は新しい取引から付与されます）")


# ── 供給診断トレンド ──────────────────────────────────────────────────

def _supply_trend(metrics: pd.DataFrame, since: pd.Timestamp | None) -> None:
    if metrics.empty:
        print("  メトリクスデータなし")
        return

    df = metrics.copy()
    if since is not None:
        df = df[df["date"] >= since]
    df = df.sort_values("date").tail(20)

    # フィールド名の互換性（旧: candidate_count, 新: rsr_pass_count）
    rsr_pass_col = "rsr_pass_count" if "rsr_pass_count" in df.columns else (
                   "candidate_count" if "candidate_count" in df.columns else None)
    cand_col     = "candidate_count"  # 最終BUY候補（新形式）

    print(f"\n  {'日付':>12} {'RSR通過':>7} {'候補':>5} {'近接':>5} {'分散':>7} {'Regime':>8} {'TOPIX%':>8}")
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        rsr_pass      = int(row.get(rsr_pass_col, 0)) if rsr_pass_col else "-"
        cands         = int(row.get(cand_col, 0)) if cand_col in row else "-"
        near_bo       = row.get("near_breakout_count", "-")
        near_bo_str   = f"{int(near_bo):>5}" if near_bo != "-" and pd.notna(near_bo) else "    -"
        rsr_disp      = row.get("rsr_dispersion", None)
        disp_str      = f"{rsr_disp:>7.1f}" if pd.notna(rsr_disp) else "      -"
        regime        = row.get("trend_market", "-") or "-"
        topix_pct     = row.get("topix_vs_ma200_pct", None)
        topix_str     = f"{topix_pct:>+8.1f}%" if pd.notna(topix_pct) else "       -"
        print(
            f"  {str(row['date'].date()):>12}"
            f" {rsr_pass:>7}"
            f" {cands:>5}"
            f"{near_bo_str}"
            f"{disp_str}"
            f" {regime:>8}"
            f"{topix_str}"
        )

    # 要約統計
    if rsr_pass_col:
        avg_pass = df[rsr_pass_col].mean()
        print(f"\n  RSR通過 平均: {avg_pass:.1f}/日  目安: ≥3/日（supply十分）")
    if "near_breakout_count" in df.columns:
        avg_near = df["near_breakout_count"].mean()
        print(f"  近接銘柄 平均: {avg_near:.1f}/日  ＞3なら近く候補が増える見込み")
    if "rsr_dispersion" in df.columns:
        avg_disp = df["rsr_dispersion"].dropna().mean()
        state = "強トレンド相場" if avg_disp > 10 else ("普通" if avg_disp >= 6 else "横ばい相場")
        print(f"  RSR分散 平均: {avg_disp:.1f}  → {state}  (目安: >10=強い / <5=横ばい)")


# ── メイン ────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="週次・月次レビューレポート")
    p.add_argument("--weeks", type=int, default=8, help="直近何週分表示するか（default 8）")
    p.add_argument("--since", default=None, help="集計開始日 YYYY-MM-DD（省略時: --weeks に従う）")
    args = p.parse_args()

    since: pd.Timestamp | None = None
    if args.since:
        since = pd.Timestamp(args.since)
    else:
        since = pd.Timestamp.now() - pd.Timedelta(weeks=args.weeks)

    trades  = _load_trades()
    metrics = _load_metrics()

    print("=" * 68)
    print("  フジコ法 週次・月次レビューレポート")
    since_str = since.strftime("%Y-%m-%d") if since else "全期間"
    print(f"  集計期間: {since_str} 〜 {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    closed = trades[trades["side"] == "SELL"] if (not trades.empty and "side" in trades.columns) else pd.DataFrame()
    print(f"  クローズ済み取引: {len(closed)} 件 / データソース: {TRADES_PATH}")
    print("=" * 68)

    # 1. 週次トレード統計
    print("\n━━ 週次トレード統計 ━━")
    _weekly_trade_stats(trades, since)

    # 2. 月次統計 + regime別
    print("\n━━ 月次統計 ━━")
    _monthly_stats(trades, metrics, since)

    # 3. 供給診断
    print("\n━━ 供給診断（直近20日）━━")
    _supply_trend(metrics, since)

    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
