"""
src/run_f4_tp50_weekly_report.py
F4 TP50/T15 -- Weekly performance report (read-only, no order, no state write).

2026-08-20 全面刷新（.claude/skills/notification-design.md準拠）: 従来の「日別シグナル件数
ダイジェスト」から、「週間の戦略成績・リスク・Benchmark比較」を主眼とするレポートへ再設計。
旧仕様（各日signal件数+score一覧）は日別詳細セクションとしてMarkdownファイル側に残すが、
メール本文（スマートフォンで読む主経路）はPERFORMANCE/BENCHMARK/TRADING/TOP CONTRIBUTORS/
TOP LOSERS/CURRENT HOLDINGS/EXIT BREAKDOWN/SCORE REPLACEMENT/RISK/SYSTEM HEALTHの
9セクション構成に統一する。

実現損益・保有銘柄のEntry/Current価格は、logs/live/の当日ログに記録された値（すでに
src.run_live_signal_f4_tp50側でbroker実約定ベースへ統一済み。2026-08-20 9344インシデント
SSOT修正、commit 9db0ab7参照）に加え、本スクリプト自身もkabu API GET /orders から実約定を
再取得する（唯一のSource of Truth。理論値では絶対に代用しない）。GET専用・sendorderは
一切呼ばない。

TOPIX比較には database/market/index/prices/0000.parquet を使う（J-Quants取得済み）。
日経平均・S&P500の日次価格系列は本プロジェクトに未取得のため「データ未取得」と明示する
（database/market/index/README.md参照。捏造しない）。

NEVER imports src.f4_tp50.executor. NEVER calls send_order. NEVER writes to
portfolio_state.json or the Replacement sidecar. NEVER changes
SCORE_REPLACEMENT_ENABLED.

Usage:
    python -m src.run_f4_tp50_weekly_report              # last 7 days, email sent
    python -m src.run_f4_tp50_weekly_report --days 14    # custom window
    python -m src.run_f4_tp50_weekly_report --no-email   # file only, no email
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from src.f4_tp50 import entry_pipeline as ep  # noqa: E402
from src.f4_tp50 import score as f4_score  # noqa: E402
from src.paths import LIVE_LOG_DIR  # noqa: E402

JST = timezone(timedelta(hours=9))
REPORT_DIR = BASE_DIR / "reports" / "f4_tp50_weekly"
TOPIX_PARQUET = BASE_DIR / "database" / "market" / "index" / "prices" / "0000.parquet"
TOP_N = 5

LOGIC_NAME = "F4 TP50/T15 (+ Score Replacement layer)"

_EXIT_REASON_LABEL = {
    "trailing_gap_open": "T15 トレーリングSTOP", "trailing_touch": "T15 トレーリングSTOP",
    "target_gap_open": "TP50 利確", "target_touch": "TP50 利確",
}


def _exit_reason_bucket(reason: str) -> str:
    if reason in ("trailing_gap_open", "trailing_touch"):
        return "T15 STOP"
    if reason in ("target_gap_open", "target_touch"):
        return "TP50 TARGET"
    if reason == "score_replacement":
        return "REPLACEMENT"
    return "Other"


def _fmt_yen(x, signed: bool = False) -> str:
    if x is None:
        return "N/A"
    sign = "+" if (signed and x >= 0) else ""
    return f"{sign}¥{x:,.0f}"


def _fmt_pct(x, signed: bool = True) -> str:
    if x is None:
        return "N/A"
    sign = "+" if (signed and x >= 0) else ""
    return f"{sign}{x:.2%}"


# ── kabu API 実約定取得（GET専用・sendorder禁止・read-only） ────────────────
def _fetch_actual_fill(client, order_id: str | None) -> tuple[float | None, float | None]:
    """(avg_price, filled_qty) を kabu API GET /orders から取得。取得不能ならNone。
    理論値へのフォールバックは行わない（実約定が取れない取引はP/L集計から除外する）。"""
    if client is None or not order_id or order_id in ("DRY_RUN_SIMULATED",):
        return None, None
    try:
        orders = client.get_orders(only_open=False)
    except Exception:
        return None, None
    for o in orders:
        if o.get("ID") != order_id:
            continue
        fills = [(float(d["Price"]), float(d["Qty"])) for d in (o.get("Details") or [])
                 if d.get("RecType") == 8 and d.get("Price") and d.get("Qty")]
        if not fills:
            return None, None
        total_qty = sum(q for _, q in fills)
        avg_price = sum(p * q for p, q in fills) / total_qty
        return avg_price, total_qty
    return None, None


_MANUAL_DELAYED_THRESHOLD_MIN = 20


def _run_was_scheduler_delayed(run: dict) -> bool:
    """run_started_at が scheduled_trigger_hhmm(通常08:49)より一定分以上遅い場合、
    Task Scheduler自動発火なしの手動遅延実行とみなす
    （src.run_live_signal_f4_tp50._is_manual_delayed_run()と同一ロジック。
    2026-08-20 9344インシデントの背景となったscheduler未発火の週次集計用）。"""
    started_at = run.get("run_started_at")
    scheduled_hhmm = run.get("scheduled_trigger_hhmm")
    if not started_at or not scheduled_hhmm:
        return False
    try:
        started = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        sched_h, sched_m = (int(x) for x in scheduled_hhmm.split(":"))
        scheduled = started.replace(hour=sched_h, minute=sched_m, second=0, microsecond=0)
        return (started - scheduled).total_seconds() / 60.0 > _MANUAL_DELAYED_THRESHOLD_MIN
    except (ValueError, TypeError):
        return False


def _symbol_name(client, code5: str) -> str:
    if client is None:
        return code5
    try:
        board = client.get_board(code5[:4])
        return board.symbol_name or code5
    except Exception:
        return code5


# ── 1. Collect daily logs（既存ロジック維持） ───────────────────────────
def _run_ts_from_filename(p: Path) -> datetime | None:
    stem = p.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        date_part, time_part = parts[-2], parts[-1]
        return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(tzinfo=JST)
    except ValueError:
        return None


def collect_daily_runs(days: int) -> dict[str, list[dict]]:
    """Returns {YYYY-MM-DD: [run_summary, ...]} for the past `days` days
    (JST calendar days, inclusive of today), sorted chronologically within each day."""
    cutoff = datetime.now(JST) - timedelta(days=days)
    by_date: dict[str, list[dict]] = {}
    for p in sorted(LIVE_LOG_DIR.glob("f4_tp50_f4_tp50_*.json")):
        ts = _run_ts_from_filename(p)
        if ts is None or ts < cutoff:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        date_key = ts.strftime("%Y-%m-%d")
        data["_run_ts"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        by_date.setdefault(date_key, []).append(data)
    for k in by_date:
        by_date[k].sort(key=lambda d: d["_run_ts"])
    return dict(sorted(by_date.items()))


def pick_representative_run(runs: list[dict]) -> dict:
    """1日に複数run（手動再実行・dry-run+live）がある場合、最後のlive runを優先、
    無ければ最後のdry-runを使う。"""
    live_runs = [r for r in runs if r.get("live")]
    return live_runs[-1] if live_runs else runs[-1]


# ── 2. Weekly trade aggregation（新設: 実現損益・Exit内訳・Top銘柄） ────────
def collect_weekly_trades(day_rows: list[dict], client) -> dict:
    """週内の全LIVE SELL/BUYを実約定ベースで集計する。
    古い形式のログ（entry_price/exits_detail未収録・2026-08-20修正前）は
    可能な範囲でorder_submission_resultsから復元し、entry_priceが取得できない
    取引はP/L集計から除外する（推測値で埋めない）。"""
    closed_trades: list[dict] = []   # 週内に確定したSELLのうちP/L算出できたもの
    sell_success_count = 0           # 週内に成功した全SELL（P/L算出可否に関わらず、EXIT BREAKDOWNと一致させる）
    buy_count = 0
    replacement_count = 0
    exit_bucket_counts: dict[str, int] = {"T15 STOP": 0, "TP50 TARGET": 0, "REPLACEMENT": 0, "Other": 0}
    order_fail_count = 0
    scheduler_failures = 0
    metadata_warning_count = 0
    api_error_count = 0

    for r in day_rows:
        run = r["_raw"]
        if not run.get("live"):
            continue
        results = run.get("order_submission_results") or []
        exits_detail = {e["code"]: e for e in (run.get("exits_detail") or [])}
        decisions = (run.get("score_replacement") or {}).get("decisions") or []
        replace_codes = {d.get("sold_code") for d in decisions
                          if d.get("decision") in ("REPLACE_SIMULATED", "BUY_FILLED")}
        replacement_count += len(replace_codes)
        order_fail_count += sum(1 for o in results if not o.get("success"))
        if _run_was_scheduler_delayed(run):
            scheduler_failures += 1
        metadata_warning_count += len(run.get("metadata_warnings") or [])
        if run.get("cash_source") not in ("broker_live", None):
            api_error_count += 1

        for o in results:
            if not o.get("success"):
                continue
            code = o.get("symbol")
            if o.get("side") == "BUY":
                buy_count += 1
                continue
            if o.get("side") != "SELL":
                continue
            sell_success_count += 1
            e = exits_detail.get(code)
            entry_price = e.get("entry_price") if e else None
            exit_reason = e.get("exit_reason") if e else ("score_replacement" if code in replace_codes else None)
            bucket = "REPLACEMENT" if code in replace_codes else _exit_reason_bucket(exit_reason or "")
            exit_bucket_counts[bucket] = exit_bucket_counts.get(bucket, 0) + 1

            fill_price, filled_qty = _fetch_actual_fill(client, o.get("order_id"))
            if fill_price is None or entry_price is None:
                continue  # 実約定 or entry_price を確認できない取引はP/L集計から除外（推測しない）
            qty = filled_qty or o.get("qty") or 0
            pnl = (fill_price - entry_price) * qty
            closed_trades.append({
                "code": code, "entry_price": entry_price, "exit_price": fill_price,
                "qty": qty, "pnl": pnl, "pnl_pct": (fill_price - entry_price) / entry_price,
                "reason_label": _EXIT_REASON_LABEL.get(exit_reason, "Score入替" if bucket == "REPLACEMENT" else "その他"),
                "date": r["date"],
            })

    return {
        "closed_trades": closed_trades, "sell_success_count": sell_success_count, "buy_count": buy_count,
        "replacement_count": replacement_count, "exit_bucket_counts": exit_bucket_counts,
        "order_fail_count": order_fail_count, "scheduler_failures": scheduler_failures,
        "metadata_warning_count": metadata_warning_count, "api_error_count": api_error_count,
    }


def _week_equity_bounds(day_rows: list[dict]) -> tuple[float | None, float | None, float | None]:
    """(start_equity, end_equity, end_dd) を、last_equity フィールドを持つ最古/最新の
    runから取る。2026-08-20以前のログにはこのフィールドが無いため、無ければNoneを返す
    （捏造しない — 呼び出し側は「データ不足」と明示する）。"""
    with_equity = [r for r in day_rows if r["_raw"].get("last_equity") is not None]
    if not with_equity:
        return None, None, None
    start = with_equity[0]["_raw"]["last_equity"]
    end_run = with_equity[-1]["_raw"]
    end = end_run.get("last_equity")
    dd = (end_run.get("risk_gate") or {}).get("dd")
    return start, end, dd


def _topix_return(start_date: str, end_date: str) -> float | None:
    """週初〜週末（レポート対象期間内で実際にログが存在する最古日〜最新日）のTOPIXリターン。
    データが無ければNone（捏造しない）。"""
    if not TOPIX_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(TOPIX_PARQUET)
        sub = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        if len(sub) < 1:
            return None
        start_close = sub["Close"].iloc[0]
        end_close = sub["Close"].iloc[-1]
        return (end_close - start_close) / start_close
    except Exception:
        return None


def _current_holdings(client) -> list[dict]:
    """kabu API GET /positions から現在保有中のTP50建玉を取得する（read-only）。
    strategy tagはportfolio_state.jsonから引く（他戦略の建玉を混同しないため）。"""
    if client is None:
        return []
    try:
        from src.portfolio.state_store import load_portfolio_state
        state, _ = load_portfolio_state()
        strategy_types = state.get("position_strategy_types", {})
        entry_dates = state.get("position_entry_dates", {})
        entry_prices = state.get("position_entry_prices", {})
        positions = client.get_positions()
    except Exception:
        return []
    holdings = []
    for p in positions:
        sym4 = p.get("Symbol")
        matched_key = next((k for k in strategy_types if k[:4] == sym4 and strategy_types[k] == "f4_tp50"), None)
        if matched_key is None:
            continue
        qty = float(p.get("LeavesQty") or 0)
        if qty <= 0:
            continue
        entry_price = entry_prices.get(matched_key)
        current_price = p.get("CurrentPrice")
        pnl_pct = ((current_price - entry_price) / entry_price) if (entry_price and current_price) else None
        holdings.append({
            "code": matched_key, "name": p.get("SymbolName") or matched_key,
            "entry_date": entry_dates.get(matched_key), "entry_price": entry_price,
            "current_price": current_price, "qty": qty, "pnl_pct": pnl_pct,
        })
    return holdings


# ── 3. Report building ───────────────────────────────────────────────────
def _fmt_run(run: dict) -> dict:
    sb = run.get("sizing_breakdown") or {}
    osr = run.get("order_submission_results")
    n_success = sum(1 for r in osr if r.get("success")) if osr else None
    n_fail = sum(1 for r in osr if not r.get("success")) if osr else None
    sr = run.get("score_replacement") or {}
    return {
        "mode": "LIVE" if run.get("live") else "dry-run",
        "run_ts": run.get("_run_ts"),
        "signal_date": run.get("signal_date"),
        "entry_date": run.get("entry_date"),
        "exits": run.get("exits_intended"),
        "funded": sb.get("funded_total"),
        "capital_exhausted": sb.get("capital_exhausted_skip"),
        "already_held_skip": sb.get("already_held_skip"),
        "order_success": n_success, "order_fail": n_fail,
        "sr_enabled": sr.get("enabled", False),
        "sr_replacements": len(sr.get("decisions") or []),
        "entry_freeze": run.get("entry_freeze_enabled"),
        "risk_gate": (run.get("risk_gate") or {}).get("recommendation"),
        "_raw": run,
    }


def build_report(daily_runs: dict[str, list[dict]], days: int, client=None, score_map_ctx=None) -> tuple[str, str, dict]:
    """Returns (full_markdown, email_plaintext, stats)."""
    day_rows = []
    for date_key, runs in daily_runs.items():
        rep = _fmt_run(pick_representative_run(runs))
        rep["date"] = date_key
        rep["n_runs"] = len(runs)
        day_rows.append(rep)

    week = collect_weekly_trades(day_rows, client)
    closed = week["closed_trades"]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    realized_pnl = sum(t["pnl"] for t in closed)
    win_rate = (len(wins) / len(closed)) if closed else None
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    start_equity, end_equity, end_dd = _week_equity_bounds(day_rows)
    week_return = ((end_equity - start_equity) / start_equity) if (start_equity and end_equity) else None

    date_keys = list(daily_runs.keys())
    topix_ret = _topix_return(date_keys[0], date_keys[-1]) if date_keys else None
    relative_vs_topix = (week_return - topix_ret) if (week_return is not None and topix_ret is not None) else None

    holdings = _current_holdings(client)

    top_contributors = sorted(closed, key=lambda t: t["pnl"], reverse=True)[:TOP_N]
    top_losers = sorted(closed, key=lambda t: t["pnl"])[:TOP_N]

    sep = "━" * 22
    period_label = f"{date_keys[0].replace('-', '/')} - {date_keys[-1].replace('-', '/')}" if date_keys else "(データなし)"

    # ── メール本文（.claude/skills/notification-design.md §5準拠の9セクション） ──
    E: list[str] = [sep, "CHIBA F4 TP50", "WEEKLY REPORT", period_label, sep, ""]

    E.append("【PERFORMANCE】")
    if week_return is not None:
        E.append(f"週間損益       {_fmt_yen(end_equity - start_equity, signed=True)}")
        E.append(f"週間Return     {_fmt_pct(week_return)}")
        E.append(f"開始資産       {_fmt_yen(start_equity)}")
        E.append(f"終了資産       {_fmt_yen(end_equity)}")
    else:
        E.append("週間損益（実現分のみ） " + _fmt_yen(realized_pnl, signed=True))
        E.append("※週次の日次資産スナップショットが不足しているため、Return%・開始/終了資産は算出不可")
        E.append("　（2026-08-20以降のログから利用可能になります）")
    E.append("")

    E.append(sep)
    E.append("【BENCHMARK】")
    E.append(sep)
    E.append("")
    if week_return is not None and topix_ret is not None:
        E.append(f"F4 TP50        {_fmt_pct(week_return)}")
    E.append(f"TOPIX          {_fmt_pct(topix_ret) if topix_ret is not None else 'N/A'}")
    E.append("日経平均       データ未取得")
    E.append("S&P500         データ未取得")
    if relative_vs_topix is not None:
        E.append(f"対TOPIX        {_fmt_pct(relative_vs_topix)}pt")
    E.append("")

    E.append(sep)
    E.append("【TRADING】")
    E.append(sep)
    E.append("")
    E.append(f"BUY             {week['buy_count']}")
    E.append(f"SELL            {week['sell_success_count']}")
    E.append(f"SCORE REPLACE   {week['replacement_count']}")
    E.append("")
    E.append(f"勝ち            {len(wins)}")
    E.append(f"負け            {len(losses)}")
    E.append(f"Win Rate        {_fmt_pct(win_rate, signed=False) if win_rate is not None else 'N/A'}")
    E.append(f"Profit Factor   {f'{profit_factor:.2f}' if profit_factor is not None else 'N/A'}")
    E.append(f"Realized P/L    {_fmt_yen(realized_pnl, signed=True)}")
    if week["sell_success_count"] > len(closed):
        E.append(f"※{week['sell_success_count'] - len(closed)}件は実約定/Entry価格未確認のためP/L集計対象外")
    E.append("")

    E.append(sep)
    E.append("【TOP CONTRIBUTORS】")
    E.append(sep)
    E.append("")
    if not top_contributors:
        E.append("なし")
    else:
        for i, t in enumerate(top_contributors, 1):
            name = _symbol_name(client, t["code"])
            E.append(f"{i} {t['code']} {name}")
            E.append(f"{_fmt_yen(t['pnl'], signed=True)}（{_fmt_pct(t['pnl_pct'])}）")
            E.append("")

    E.append(sep)
    E.append("【TOP LOSERS】")
    E.append(sep)
    E.append("")
    if not top_losers:
        E.append("なし")
    else:
        for i, t in enumerate(top_losers, 1):
            name = _symbol_name(client, t["code"])
            E.append(f"{i} {t['code']} {name}")
            E.append(f"{_fmt_yen(t['pnl'], signed=True)}（{_fmt_pct(t['pnl_pct'])}）")
            E.append("")

    E.append(sep)
    E.append("【CURRENT HOLDINGS】")
    E.append(sep)
    E.append("")
    if not holdings:
        E.append("なし（またはkabu API未接続）")
    else:
        for h in holdings:
            E.append(f"{h['code']} {h['name']}")
            E.append(f"Entry {_fmt_yen(h['entry_price'])}")
            E.append(f"Current {_fmt_yen(h['current_price'])}")
            E.append(f"P/L {_fmt_pct(h['pnl_pct']) if h['pnl_pct'] is not None else 'N/A'}")
            E.append("")

    E.append(sep)
    E.append("【EXIT BREAKDOWN】")
    E.append(sep)
    E.append("")
    for label in ("T15 STOP", "TP50 TARGET", "REPLACEMENT", "Other"):
        E.append(f"{label:14s} {week['exit_bucket_counts'].get(label, 0)}")
    E.append("")

    E.append(sep)
    E.append("【SCORE REPLACEMENT】")
    E.append(sep)
    E.append("")
    E.append(f"Replacement       {week['replacement_count']}")
    E.append("")

    E.append(sep)
    E.append("【RISK】")
    E.append(sep)
    E.append("")
    E.append(f"Current DD       {_fmt_pct(end_dd) if end_dd is not None else 'N/A'}")
    E.append(f"Positions        {len(holdings)}")
    E.append("")

    E.append(sep)
    E.append("【SYSTEM HEALTH】")
    E.append(sep)
    E.append("")
    E.append(f"Live executions        {sum(1 for r in day_rows if r['mode'] == 'LIVE')}")
    E.append(f"Order failures         {week['order_fail_count']}")
    E.append(f"API errors             {week['api_error_count']}")
    E.append(f"Scheduler failures     {week['scheduler_failures']}")
    E.append(f"Metadata warnings      {week['metadata_warning_count']}")
    E.append("")
    E.append(sep)
    E.append("END OF WEEK")
    E.append(sep)
    email_text = "\n".join(E)

    # ── フルMarkdown（既存の日別詳細を維持・ファイル保存用） ──
    L = [f"# {LOGIC_NAME} — Weekly Report — {datetime.now(JST).strftime('%Y-%m-%d')}",
         f"\n対象期間: {period_label}（JST）\n", email_text, "\n---\n## 日別詳細\n"]
    for r in day_rows:
        L.append(f"### {r['date']}（{r['mode']}, {r['n_runs']}回実行, run@{r['run_ts']}）")
        L.append(f"- SignalDate={r['signal_date']} / EntryDate={r['entry_date']}")
        L.append(f"- Exit={r['exits']} / Funded={r['funded']} / "
                  f"CapitalExhausted={r['capital_exhausted']} / AlreadyHeldSkip={r['already_held_skip']}")
        if r["order_success"] is not None:
            L.append(f"- 発注結果: success={r['order_success']} fail={r['order_fail']}")
        L.append(f"- Score Replacement={'ON' if r['sr_enabled'] else 'OFF'} / "
                  f"Replacement={r['sr_replacements']}件 / entry_freeze={r['entry_freeze']} / risk_gate={r['risk_gate']}")
        L.append("")
    full_md = "\n".join(L)

    stats = {
        "total_days": len(day_rows), "closed_trades": len(closed), "realized_pnl": realized_pnl,
        "win_rate": win_rate, "profit_factor": profit_factor, "week_return": week_return,
        "topix_return": topix_ret,
    }
    return full_md, email_text, stats


# ── 4. main ──────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="F4 TP50/T15 weekly performance report (read-only)")
    parser.add_argument("--days", type=int, default=7, help="lookback window in days (default 7)")
    parser.add_argument("--no-email", action="store_true", help="skip sending the email digest")
    args = parser.parse_args()

    print(f"[WEEKLY_REPORT] collecting past {args.days} days of logs from {LIVE_LOG_DIR} ...")
    daily_runs = collect_daily_runs(args.days)
    print(f"[WEEKLY_REPORT] {len(daily_runs)} trading day(s) with logs found.")

    client = None
    try:
        from src.kabusapi.client import KabuClient
        client = KabuClient()
        client.fetch_token()
        print("[WEEKLY_REPORT] kabu API接続OK（GET専用・sendorderは呼びません）。")
    except Exception as exc:
        print(f"[WEEKLY_REPORT][WARNING] kabu API接続不可: {exc} — 実約定/保有銘柄/銘柄名が一部N/Aになります。")

    print("[WEEKLY_REPORT] building report ...")
    full_md, email_text, stats = build_report(daily_runs, args.days, client=client)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"f4_tp50_weekly_{datetime.now(JST).strftime('%Y-%m-%d')}.md"
    out_path.write_text(full_md, encoding="utf-8")
    print(f"[WEEKLY_REPORT] saved: {out_path}")

    print("\n" + email_text)

    if not args.no_email:
        from src.notifier import notify_success, wait_pending
        notify_success(email_text, subject_suffix="[F4 TP50 Weekly Report]")
        # notify_*()はdaemon threadでfire-and-forget送信のため、ここでブロックして
        # SMTP送信完了を待つ（2026-08-20修正: 待たないとthreadごと消え未送信になる、
        # run_live_signal_f4_tp50.pyと同一のバグパターン）。
        wait_pending(timeout=15.0)
        print("[WEEKLY_REPORT] email dispatched and send completion awaited.")

    print(f"\n[WEEKLY_REPORT] stats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
