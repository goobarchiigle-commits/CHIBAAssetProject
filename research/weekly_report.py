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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from paths import LOGS_DIR

import json
from pathlib import Path
from datetime import datetime, date, timedelta, timezone, tzinfo

import numpy as np
import pandas as pd


JST            = timezone(timedelta(hours=9))
TRADES_PATH    = LOGS_DIR / "trades.jsonl"
METRICS_PATH   = LOGS_DIR / "diagnostics" / "metrics.jsonl"


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

    print(f"\n  {'日付':>12} {'RSR通過':>7} {'候補':>5} {'近接':>5} {'分散':>7} {'Regime':>8} {'強度':>7} {'半減期':>7} {'MTF率':>7}")
    print("  " + "-" * 72)
    for _, row in df.iterrows():
        rsr_pass      = int(row.get(rsr_pass_col, 0)) if rsr_pass_col else "-"
        cands         = int(row.get(cand_col, 0)) if cand_col in row else "-"
        near_bo       = row.get("near_breakout_count", "-")
        near_bo_str   = f"{int(near_bo):>5}" if near_bo != "-" and pd.notna(near_bo) else "    -"
        rsr_disp      = row.get("rsr_dispersion", None)
        disp_str      = f"{rsr_disp:>7.1f}" if pd.notna(rsr_disp) else "      -"
        regime        = row.get("trend_market", "-") or "-"
        ts            = row.get("trend_strength", None)
        ts_str        = f"{ts:>+7.3f}" if pd.notna(ts) else "      -"
        hl            = row.get("rsr_leader_half_life", None)
        hl_str        = f"{hl:>7.1f}" if pd.notna(hl) else "      -"
        mtf_r         = row.get("mtf_filter_rate", None)
        mtf_str       = f"{mtf_r:>7.2f}" if pd.notna(mtf_r) else "      -"
        print(
            f"  {str(row['date'].date()):>12}"
            f" {rsr_pass:>7}"
            f" {cands:>5}"
            f"{near_bo_str}"
            f"{disp_str}"
            f" {regime:>8}"
            f"{ts_str}"
            f"{hl_str}"
            f"{mtf_str}"
        )

    # 要約統計
    if rsr_pass_col:
        avg_pass = df[rsr_pass_col].mean()
        print(f"\n  RSR通過 平均: {avg_pass:.1f}/日  目安: ≥3/日（supply十分）")
    if "near_breakout_count" in df.columns:
        avg_near = df["near_breakout_count"].mean()
        avg_raw  = df["bo_pressure_raw"].mean() if "bo_pressure_raw" in df.columns else avg_near
        print(f"  近接銘柄(bo_pressure_raw) 平均: {avg_raw:.1f}/日  ＞3なら近く候補が増える見込み（bo_rateより先行）")
    if "rsr_pass_tradeable_ratio" in df.columns:
        tr_latest = df["rsr_pass_tradeable_ratio"].dropna()
        if not tr_latest.empty:
            tr_v = float(tr_latest.iloc[-1])
            tr_label = "正常" if tr_v >= 0.6 else "⚠ 構造的ブロック（強い銘柄が価格/流動性で除外）"
            print(f"  RSR通過→売買可能割合（最新）: {tr_v:.1%}  → {tr_label}  (目安: ≥60%)")
    if "rsr_dispersion" in df.columns:
        avg_disp = df["rsr_dispersion"].dropna().mean()
        state = "強トレンド相場" if avg_disp > 10 else ("普通" if avg_disp >= 6 else "横ばい相場")
        print(f"  RSR分散 平均: {avg_disp:.1f}  → {state}  (目安: >10=強い / <5=横ばい)")
    if "trend_strength" in df.columns:
        ts_latest = df["trend_strength"].dropna()
        if not ts_latest.empty:
            ts_v = float(ts_latest.iloc[-1])
            ts_label = "強トレンド" if ts_v > 0.05 else ("通常" if ts_v > 0.02 else ("横ばい" if ts_v > -0.02 else "下落"))
            print(f"  トレンド強度（最新）: {ts_v:+.3f}  → {ts_label}  (目安: >0.05=強 / <-0.02=下落)")
    if "rsr_leader_half_life" in df.columns:
        hl_latest = df["rsr_leader_half_life"].dropna()
        if not hl_latest.empty:
            hl_v = float(hl_latest.iloc[-1])
            # R²で信頼性を表示（<0.2 = half-lifeは信頼不可）
            r2_v = None
            if "rsr_leader_hl_r2" in df.columns:
                r2_s = df["rsr_leader_hl_r2"].dropna()
                r2_v = float(r2_s.iloc[-1]) if not r2_s.empty else None
            hl_label = "強トレンド" if hl_v > 20 else ("通常" if hl_v >= 8 else "回転相場")
            r2_str = f"  R²={r2_v:.2f}{'（信頼不可）' if r2_v is not None and r2_v < 0.2 else ''}" if r2_v is not None else ""
            print(f"  RSRリーダー半減期: {hl_v:.1f}日  → {hl_label}  (目安: >12=持続 / <8=回転){r2_str}")
    if "mtf_filter_rate" in df.columns:
        mtf_latest = df["mtf_filter_rate"].dropna()
        if not mtf_latest.empty:
            mtf_v = float(mtf_latest.iloc[-1])
            mtf_label = "MTF有効" if 0.2 <= mtf_v <= 0.4 else ("MTF過剰フィルター" if mtf_v > 0.4 else "MTF意味なし（週足整合）")
            print(f"  MTFフィルター率: {mtf_v:.2f}  → {mtf_label}  (目安: 0.2〜0.4=理想 / <0.05=意味なし)")

    # ── 構造的ボトルネック（リーダー集中相場・売買不可理由）──
    print()
    if "blocked_leaders_count" in df.columns:
        bl_v   = float(df["blocked_leaders_count"].dropna().iloc[-1]) if not df["blocked_leaders_count"].dropna().empty else 0
        blw_v  = float(df["blocked_leaders_weight"].dropna().iloc[-1]) if "blocked_leaders_weight" in df.columns and not df["blocked_leaders_weight"].dropna().empty else 0.0
        bl_label = "⚠ 期待値低下リスク" if blw_v > 0.2 else ("注意" if blw_v > 0.1 else "正常")
        print(f"  RSR Top10 売買不可: {int(bl_v)}銘柄  重み={blw_v:.1%}  → {bl_label}  (目安: >20%で期待値低下)")
    if "rsr_top10_tradeable_ratio" in df.columns:
        tr10_s = df["rsr_top10_tradeable_ratio"].dropna()
        if not tr10_s.empty:
            tr10_v = float(tr10_s.iloc[-1])
            tr10_label = "リーダー集中相場（高価格株主導）" if tr10_v < 0.5 else ("やや集中" if tr10_v < 0.7 else "正常")
            print(f"  RSR Top10 売買可能割合: {tr10_v:.0%}  → {tr10_label}  (目安: ≥50%)")
    if "blocked_by_price" in df.columns:
        bp_s = df["blocked_by_price"].dropna()
        bl_s = df["blocked_by_liquidity"].dropna() if "blocked_by_liquidity" in df.columns else pd.Series(dtype=float)
        br_s = df["blocked_by_risk"].dropna() if "blocked_by_risk" in df.columns else pd.Series(dtype=float)
        bp_v  = int(bp_s.iloc[-1])  if not bp_s.empty  else 0
        bl_v2 = int(bl_s.iloc[-1])  if not bl_s.empty  else 0
        br_v  = int(br_s.iloc[-1])  if not br_s.empty  else 0
        note  = "→ ポジションサイズ設計を見直す" if bp_v > 0 else ""
        print(f"  ブロック理由: 価格={bp_v}銘柄  流動性={bl_v2}銘柄  リスク管理={br_v}銘柄  {note}")


# ── メイン ────────────────────────────────────────────────────────────

def _judgment_auto(metrics: pd.DataFrame) -> None:
    """4/8 判断ロジック（直近データから自動判定）"""
    if metrics.empty:
        print("  メトリクスデータ不足（判断不可）")
        return

    df = metrics.sort_values("date")
    last = df.iloc[-1]  # 最新1行

    # 直近10日平均
    last10 = df.tail(10)
    rsr_pass_col = "rsr_pass_count" if "rsr_pass_count" in df.columns else (
                   "candidate_count" if "candidate_count" in df.columns else None)

    rsr_pass_avg  = float(last10[rsr_pass_col].mean())  if rsr_pass_col else 0.0
    bo_rate_avg   = float(last10["breakout_opportunity_rate"].dropna().mean()) if "breakout_opportunity_rate" in last10.columns else None
    rsr_disp_avg  = float(last10["rsr_dispersion"].dropna().mean())   if "rsr_dispersion" in last10.columns else None
    fb_rate_avg   = float(last10["failed_breakout_rate"].dropna().mean()) if "failed_breakout_rate" in last10.columns else None
    ts_latest     = float(last.get("trend_strength", 0) or 0)
    hl_latest     = float(last.get("rsr_leader_half_life", 0) or 0)
    mtf_r_latest  = float(last.get("mtf_filter_rate", 0) or 0)
    regime        = str(last.get("trend_market", "unknown") or "unknown")

    print(f"\n  直近10日平均:")
    print(f"    rsr_pass_count            = {rsr_pass_avg:.1f}  (目安 ≥4)")
    print(f"    breakout_opportunity_rate = {bo_rate_avg:.2f}  (目安 ≥0.25)" if bo_rate_avg is not None else "    breakout_opportunity_rate = N/A")
    print(f"    failed_breakout_rate      = {fb_rate_avg:.3f}  (ベースライン記録中)" if fb_rate_avg is not None else "    failed_breakout_rate      = N/A")
    print(f"  最新:")
    print(f"    trend_strength     = {ts_latest:+.3f}  (目安 >0 / <-0.02=下落)")
    print(f"    rsr_leader_hl      = {hl_latest:.1f}日  (目安 >12=持続 / <8=回転)")
    print(f"    mtf_filter_rate    = {mtf_r_latest:.2f}  (目安 0.2〜0.4=MTF有効帯)")
    print(f"    trend_market       = {regime}")

    # ── 判断 ──
    print(f"\n  ── 4/8 判断 ──")

    # ケースC（相場停滞）を最優先チェック
    if ts_latest < -0.02 and (rsr_disp_avg is not None and rsr_disp_avg < 5):
        print("  ⚠ ケースC: 相場停滞 → 何もしない")
        print(f"    trend_strength={ts_latest:+.3f} < -0.02 かつ rsr_dispersion={rsr_disp_avg:.1f} < 5")
        print("    【推奨】観察継続。戦略パラメータ変更は禁止。")
        return

    if regime == "bear" and ts_latest < 0:
        print("  ⚠ bear相場中 → 新規改善の導入は見送り")
        print("    【推奨】既存ポジションの管理のみ。MTF等の導入は次の bull/neutral まで延期。")
        return

    # ケースA: ブレイクアウト相場の4条件（同時成立で高精度）
    #   ① トレンド持続（指数ベース）
    #   ② RSRリーダー固定化（EMA平滑化半減期 > 12日）
    #   ③ MTFフィルターが有効帯（週足で整合した銘柄が多い）
    #   ④ ブレイク前圧力が十分（RSR通過銘柄の25%以上が高値3%以内）
    case_a = (
        ts_latest > 0
        and hl_latest > 12
        and mtf_r_latest > 0.2
        and (bo_rate_avg is not None and bo_rate_avg >= 0.25)
    )
    # ケースB: 供給不足（RSR通過は多いが候補が出ない）
    cand_avg = float(last10["candidate_count"].mean()) if "candidate_count" in last10.columns else None
    case_b = rsr_pass_avg >= 6 and (cand_avg is not None and cand_avg < 1.0)

    if case_a:
        print("  ✅ ケースA: ブレイクアウト相場 → MTF導入を検討")
        print(f"    trend_strength={ts_latest:+.3f}>0 / hl={hl_latest:.1f}>12 / mtf_r={mtf_r_latest:.2f}>0.2 / bo_rate={bo_rate_avg:.2f}≥0.25")
        print("    【推奨】`python -m backtest.mtf_comparison` を実行してMTF導入効果を確認。")
    elif case_b:
        print("  △ ケースB: 供給不足 → breakout条件の調整を検討")
        print(f"    rsr_pass={rsr_pass_avg:.1f}≥6 だが候補ゼロが続いている")
        print("    【候補】Donchian hybrid（turtle_entry=15 → mtf_comparison で検証後）")
    else:
        n_ok = sum([
            ts_latest > 0,
            hl_latest > 12,
            mtf_r_latest > 0.2,
            bo_rate_avg is not None and bo_rate_avg >= 0.25,
        ])
        print(f"  △ 条件未満（{n_ok}/4 満たす）→ 観察継続")
        if ts_latest <= 0:
            print(f"    trend_strength={ts_latest:+.3f} ≤ 0 → トレンド確認待ち")
        if hl_latest <= 12:
            print(f"    rsr_leader_hl={hl_latest:.1f} ≤ 12 → リーダー固定化待ち（データ蓄積中の可能性あり）")
        if mtf_r_latest <= 0.2:
            print(f"    mtf_filter_rate={mtf_r_latest:.2f} ≤ 0.2 → 週足未整合が少ない（MTFが効かない相場 or データ不足）")
        if bo_rate_avg is not None and bo_rate_avg < 0.25:
            print(f"    breakout_opportunity_rate={bo_rate_avg:.2f} < 0.25 → ブレイク前圧力不足")
        elif bo_rate_avg is None:
            print(f"    breakout_opportunity_rate = N/A → データ蓄積待ち")
        print("    【推奨】4/8以降も同条件で継続観察")


# ── シグナルメトリクス集計（研究PDCA用） ────────────────────────────

def summarize_signal_metrics(metrics: pd.DataFrame, since: pd.Timestamp | None) -> dict:
    """
    シグナル生成の健全性を集計する（取引ゼロでも機能する研究PDCA用）。

    Returns dict with:
        avg_universe, avg_rsr_pass, avg_candidates,
        execution_rate, filter_efficiency,
        avg_blocked_by_rsr, avg_blocked_by_breakout, avg_blocked_by_price,
        avg_market_leader_block_rate, dominant_bottleneck
    """
    if metrics.empty:
        return {}

    df = metrics.copy()
    if since is not None:
        df = df[df["date"] >= since]
    df = df.sort_values("date")
    if df.empty:
        return {}

    rsr_pass_col = "rsr_pass_count" if "rsr_pass_count" in df.columns else (
                   "candidate_count" if "candidate_count" in df.columns else None)

    avg_universe   = float(df.get("universe_size",        pd.Series([0])).mean())
    avg_rsr_pass   = float(df[rsr_pass_col].mean())                 if rsr_pass_col else 0.0
    avg_candidates = float(df.get("candidate_count",      pd.Series([0])).mean())
    avg_blocked_rsr  = float(df.get("blocked_by_rsr",     pd.Series([0])).mean())
    avg_blocked_bo   = float(df.get("blocked_by_breakout",pd.Series([0])).mean())
    avg_blocked_px   = float(df.get("blocked_by_price",   pd.Series([0])).mean())
    avg_bl_rate      = float(df.get("market_leader_block_rate", pd.Series([0])).dropna().mean())
    avg_tradeable    = float(df.get("rsr_pass_tradeable_ratio", pd.Series([1])).dropna().mean())

    # 実行率 = 最終BUY候補 / RSR通過数
    execution_rate = avg_candidates / max(avg_rsr_pass, 0.01)

    # ボトルネック特定（どのフィルターが最も多く候補を削っているか）
    bottlenecks = {
        "RSRフィルター": avg_blocked_rsr,
        "Breakoutフィルター": avg_blocked_bo,
        "価格フィルター": avg_blocked_px,
    }
    dominant_bottleneck = max(bottlenecks, key=bottlenecks.get)

    return {
        "avg_universe":             round(avg_universe, 1),
        "avg_rsr_pass":             round(avg_rsr_pass, 1),
        "avg_candidates":           round(avg_candidates, 2),
        "execution_rate":           round(execution_rate, 3),
        "avg_blocked_by_rsr":       round(avg_blocked_rsr, 1),
        "avg_blocked_by_breakout":  round(avg_blocked_bo, 1),
        "avg_blocked_by_price":     round(avg_blocked_px, 1),
        "avg_market_leader_block":  round(avg_bl_rate, 2),
        "avg_tradeable_ratio":      round(avg_tradeable, 2),
        "dominant_bottleneck":      dominant_bottleneck,
    }


def _signal_health_report(metrics: pd.DataFrame, since: pd.Timestamp | None) -> None:
    """シグナル生成健全性レポート（研究PDCA用・取引ゼロでも有効）"""
    summary = summarize_signal_metrics(metrics, since)
    if not summary:
        print("  シグナルメトリクスデータなし")
        return

    exec_rate = summary["execution_rate"]
    exec_label = (
        "正常" if exec_rate >= 0.3 else
        ("低下" if exec_rate >= 0.1 else
         "枯渇（signal starvation）")
    )
    bl_rate = summary["avg_market_leader_block"]
    bl_label = (
        "正常" if bl_rate < 0.3 else
        ("注意" if bl_rate < 0.5 else
         "リーダー集中相場（高RSR銘柄が価格で買えない）")
    )
    tr_ratio = summary["avg_tradeable_ratio"]

    print(f"\n  {'指標':<35} {'値':>8}  {'判定'}")
    print("  " + "-" * 65)
    print(f"  {'ユニバース平均銘柄数':<35} {summary['avg_universe']:>8.1f}")
    print(f"  {'RSR通過平均（/日）':<35} {summary['avg_rsr_pass']:>8.1f}  目安: >=4/日")
    print(f"  {'最終BUY候補平均（/日）':<35} {summary['avg_candidates']:>8.2f}  目安: >=1/日")
    print(f"  {'実行率（候補/RSR通過）':<35} {exec_rate:>8.1%}  {exec_label}")
    print(f"  {'RSRブロック平均':<35} {summary['avg_blocked_by_rsr']:>8.1f}")
    print(f"  {'Breakoutブロック平均':<35} {summary['avg_blocked_by_breakout']:>8.1f}")
    print(f"  {'価格ブロック平均':<35} {summary['avg_blocked_by_price']:>8.1f}")
    print(f"  {'RSRリーダーブロック率':<35} {bl_rate:>8.1%}  {bl_label}")
    print(f"  {'RSR通過→売買可能率':<35} {tr_ratio:>8.1%}  目安: >=60%")
    print(f"\n  主なボトルネック: {summary['dominant_bottleneck']}")

    # 改善示唆
    print()
    if exec_rate < 0.1:
        print("  [要対応] 実行率 < 10% — signal starvation 状態")
        if summary["avg_blocked_by_rsr"] > summary["avg_rsr_pass"] * 3:
            print("    -> RSRフィルター(min_rsr)の緩和を検討（現状: バックテスト確認要）")
        if bl_rate > 0.5:
            print("    -> 価格上限に引っかかる高RSR銘柄が多い。capital拡大 or ユニバース見直し")
    elif exec_rate < 0.3:
        print("  [注意] 実行率 30%未満 — Breakoutフィルターの調整を検討")
        print("       `python -m backtest.turtle_entry_rolling_oos` で turtle_entry=15 vs 20 を比較")


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

    # 0. シグナル健全性（研究PDCA用・最優先）
    print("\n━━ シグナル生成健全性 ━━")
    _signal_health_report(metrics, since)

    # 1. 週次トレード統計
    print("\n━━ 週次トレード統計 ━━")
    _weekly_trade_stats(trades, since)

    # 2. 月次統計 + regime別
    print("\n━━ 月次統計 ━━")
    _monthly_stats(trades, metrics, since)

    # 3. 供給診断
    print("\n━━ 供給診断（直近20日）━━")
    _supply_trend(metrics, since)

    # 4/8 判断ロジック
    print("\n━━ 4/8 判断ロジック（自動） ━━")
    _judgment_auto(metrics)

    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
