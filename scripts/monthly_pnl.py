"""
scripts/monthly_pnl.py
Phase 2 損益評価スクリプト

【データソース（優先順）】
  1. kabuステーション API の約定照会（実際の約定価格・数量）
  2. logs/live/YYYYMMDD_orders.json の estimated_price（API未接続時フォールバック）

【評価基準（Phase 2）】
  合格: 月利益 ≥ ¥10,000 × 3ヶ月連続

【P&L計算】
  実現P&L : SELL受取 − 対応BUYコスト（FIFO）− 手数料（0.055%×2）
  未実現P&L: 保有中ポジションの時価評価

【使い方】
  python scripts/monthly_pnl.py             # 月次 + Phase 2 判定
  python scripts/monthly_pnl.py --weekly    # 週次異常検知を追加
  python scripts/monthly_pnl.py --summary   # 判定サマリーのみ
  python scripts/monthly_pnl.py --source log  # ログベース強制（API使わない）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import LIVE_LOG_DIR  # .env 読み込みも paths.py が担う

import pandas as pd

JST          = timezone(timedelta(hours=9))
COMMISSION   = 0.00055   # auカブコム 現物手数料率（片道）
LOT_SIZE     = 100
FILLS_DIR    = LIVE_LOG_DIR / "fills"

# Phase 2 判定基準
PHASE2_MONTHLY_MIN   = 10_000    # 月次最低利益（円）
PHASE2_STREAK_MONTHS = 3         # 連続月数


# ================================================================
# データ取得
# ================================================================

def fetch_fills_from_api() -> list[dict]:
    """
    kabuステーション API から約定済み注文を取得し
    正規化した fill レコードのリストを返す。

    返り値: [{date, symbol, side, qty, price, amount, fee, source}, ...]
    """
    from src.kabusapi.client import KabuClient
    client = KabuClient()
    client.fetch_token()
    raw_orders = client.get_filled_orders()

    fills = []
    for order in raw_orders:
        sym_raw = order.get("Symbol", "")
        sym     = f"{sym_raw}.T" if not sym_raw.endswith(".T") else sym_raw
        side    = "BUY" if order.get("Side") == "2" else "SELL"

        for detail in order.get("Details", []):
            # 約定明細ごとにレコードを作成
            price = float(detail.get("Price", 0) or 0)
            qty   = int(detail.get("Qty", 0) or 0)
            if price <= 0 or qty <= 0:
                continue

            ts_raw  = detail.get("Timestamp", "") or order.get("RecvTime", "")
            try:
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(JST)
            except (ValueError, AttributeError):
                dt = datetime.now(JST)

            amount = price * qty
            fee    = amount * COMMISSION

            fills.append({
                "date":     dt.date().isoformat(),
                "symbol":   sym,
                "side":     side,
                "qty":      qty,
                "price":    price,
                "amount":   amount,
                "fee":      round(fee, 0),
                "order_id": order.get("OrderId", ""),
                "source":   "api",
            })

    return fills


def fetch_fills_from_logs() -> list[dict]:
    """
    logs/live/YYYYMMDD_orders.json から推定価格ベースの fill を構築する。
    成功した発注のみ対象（send_results.success == True）。

    注意: estimated_price を使うため実際の約定価格とは異なる場合がある。
    """
    fills = []
    if not LIVE_LOG_DIR.exists():
        return fills

    for json_file in sorted(LIVE_LOG_DIR.glob("*_orders.json")):
        try:
            runs = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        for run in runs:
            run_at   = run.get("run_at", "")
            orders   = run.get("orders", [])
            results  = {r["symbol"]: r for r in run.get("send_results", [])}

            try:
                dt = datetime.fromisoformat(run_at).astimezone(JST)
                date_str = dt.date().isoformat()
            except (ValueError, AttributeError):
                # ファイル名から日付を推定
                date_str = json_file.stem[:8]
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            for order in orders:
                sym  = order.get("symbol", "")
                res  = results.get(sym, {})
                if not res.get("success"):
                    continue   # 発注失敗はスキップ

                price  = float(order.get("estimated_price", 0) or 0)
                qty    = int(order.get("qty", 0) or 0)
                side   = order.get("side", "BUY")
                if price <= 0 or qty <= 0:
                    continue

                amount = price * qty
                fee    = amount * COMMISSION

                fills.append({
                    "date":     date_str,
                    "symbol":   sym,
                    "side":     side,
                    "qty":      qty,
                    "price":    price,
                    "amount":   amount,
                    "fee":      round(fee, 0),
                    "order_id": res.get("order_id", ""),
                    "source":   "log_estimate",
                })

    return fills


def save_fills_cache(fills: list[dict]) -> None:
    """API取得した fill データをキャッシュとして保存（後で参照・監査用）。"""
    if not fills:
        return
    FILLS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(JST).strftime("%Y%m%d")
    out   = FILLS_DIR / f"{today}_fills.json"
    out.write_text(json.dumps(fills, ensure_ascii=False, indent=2), encoding="utf-8")


# ================================================================
# P&L 計算
# ================================================================

def calc_pnl(fills: list[dict]) -> dict:
    """
    FIFO マッチングで実現P&Lを計算する。

    Returns:
        {
          "realized_by_date": {date: pnl},
          "open_positions":   {symbol: [(qty, cost_price), ...]},
          "total_realized":   float,
          "total_fees":       float,
        }
    """
    # BUY キュー: {symbol: deque([(qty, price), ...])}
    buy_queue: dict[str, deque] = defaultdict(deque)

    realized_by_date: dict[str, float] = defaultdict(float)
    total_fees = 0.0

    for fill in sorted(fills, key=lambda f: f["date"]):
        sym   = fill["symbol"]
        side  = fill["side"]
        qty   = fill["qty"]
        price = fill["price"]
        fee   = fill["fee"]
        date  = fill["date"]

        total_fees += fee

        if side == "BUY":
            buy_queue[sym].append((qty, price))
            realized_by_date[date] -= fee   # BUY手数料は発生日に計上

        elif side == "SELL":
            sell_proceeds = price * qty - fee  # SELL手数料控除
            cost = 0.0
            remaining = qty

            while remaining > 0 and buy_queue[sym]:
                bq_qty, bq_price = buy_queue[sym][0]
                matched = min(remaining, bq_qty)
                cost += matched * bq_price
                remaining -= matched
                if matched == bq_qty:
                    buy_queue[sym].popleft()
                else:
                    buy_queue[sym][0] = (bq_qty - matched, bq_price)

            # 対応する BUY がない場合（ログ取得前の既存ポジション）
            # → コストゼロとして処理（保守的に警告を出す）
            if remaining > 0:
                print(f"  ⚠ {sym}: SELL {qty}株 に対応する BUY が {remaining}株 不足（既存ポジション？）")

            realized_pnl = sell_proceeds - cost
            realized_by_date[date] += realized_pnl

    # 未実現ポジション（残BUYキュー）
    open_positions = {sym: list(q) for sym, q in buy_queue.items() if q}
    total_realized = sum(realized_by_date.values())

    return {
        "realized_by_date": dict(realized_by_date),
        "open_positions":   open_positions,
        "total_realized":   total_realized,
        "total_fees":       total_fees,
    }


def calc_unrealized_pnl(open_positions: dict) -> dict[str, float]:
    """
    保有中ポジションの未実現P&Lを現在株価（yfinance）で計算する。
    API未接続・取得失敗時はゼロを返す。
    """
    if not open_positions:
        return {}

    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf

    syms = list(open_positions.keys())
    try:
        raw = yf.download(syms, period="3d", progress=False, group_by="ticker")
    except Exception:
        return {}

    unrealized: dict[str, float] = {}
    for sym, queue in open_positions.items():
        try:
            if len(syms) == 1:
                current_price = float(raw["Close"].dropna().iloc[-1])
            else:
                current_price = float(raw[sym]["Close"].dropna().iloc[-1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        total_qty  = sum(q for q, _ in queue)
        total_cost = sum(q * p for q, p in queue)
        unrealized[sym] = current_price * total_qty - total_cost

    return unrealized


# ================================================================
# 集計
# ================================================================

def to_monthly(realized_by_date: dict) -> pd.Series:
    """日次P&Lを月次に集計する。"""
    if not realized_by_date:
        return pd.Series(dtype=float)
    df = pd.DataFrame([
        {"date": pd.Timestamp(d), "pnl": v}
        for d, v in realized_by_date.items()
    ])
    df.set_index("date", inplace=True)
    return df["pnl"].resample("ME").sum().rename(lambda x: x.to_period("M"))


def to_weekly(realized_by_date: dict) -> pd.Series:
    """日次P&Lを週次に集計する（月曜始まり）。"""
    if not realized_by_date:
        return pd.Series(dtype=float)
    df = pd.DataFrame([
        {"date": pd.Timestamp(d), "pnl": v}
        for d, v in realized_by_date.items()
    ])
    df.set_index("date", inplace=True)
    return df["pnl"].resample("W-FRI").sum()


def phase2_pass(monthly_pnl: pd.Series) -> tuple[bool, int]:
    """
    Phase 2 合格判定。

    Returns:
        (passed, max_streak): passed=合格, max_streak=最長連続月数
    """
    streak = 0
    max_streak = 0
    for v in monthly_pnl:
        if v >= PHASE2_MONTHLY_MIN:
            streak += 1
            max_streak = max(max_streak, streak)
            if streak >= PHASE2_STREAK_MONTHS:
                return True, max_streak
        else:
            streak = 0
    return False, max_streak


# ================================================================
# 表示
# ================================================================

def print_monthly(monthly: pd.Series, unrealized_total: float) -> None:
    print("\n  === 月次損益 ===")
    print(f"  {'月':<10} {'実現P&L':>12} {'判定':>8}")
    print("  " + "-" * 34)
    for period, val in monthly.items():
        mark = "✅" if val >= PHASE2_MONTHLY_MIN else ("⚠" if val >= 0 else "❌")
        print(f"  {str(period):<10} ¥{val:>10,.0f}   {mark}")
    total = monthly.sum()
    print("  " + "-" * 34)
    print(f"  {'合計':<10} ¥{total:>10,.0f}")
    if unrealized_total:
        print(f"  {'未実現':<10} ¥{unrealized_total:>10,.0f}  （参考）")
        print(f"  {'合算':<10} ¥{total + unrealized_total:>10,.0f}  （参考）")


def print_weekly(weekly: pd.Series) -> None:
    print("\n  === 週次損益（異常検知用）===")
    print(f"  {'週末':<12} {'P&L':>12} {'異常':>6}")
    print("  " + "-" * 34)
    if weekly.empty:
        print("  データなし")
        return
    mean = weekly.mean()
    std  = weekly.std() if len(weekly) > 1 else 0
    for date, val in weekly.items():
        # 標準偏差2σ超を異常フラグ
        flag = " ⚠ 2σ超" if std > 0 and abs(val - mean) > 2 * std else ""
        print(f"  {str(date.date()):<12} ¥{val:>10,.0f}{flag}")


def print_phase2(monthly: pd.Series) -> None:
    passed, streak = phase2_pass(monthly)
    print("\n" + "=" * 50)
    print("  Phase 2 判定")
    print("=" * 50)
    print(f"  基準  : 月¥{PHASE2_MONTHLY_MIN:,} 以上 × {PHASE2_STREAK_MONTHS}ヶ月連続")
    print(f"  状態  : 最長連続 {streak}ヶ月")
    if passed:
        print("  結果  : ✅ Phase 2 クリア → Phase 3 移行検討")
    elif streak >= PHASE2_STREAK_MONTHS - 1:
        print("  結果  : ⚠  あと1ヶ月で達成圏")
    elif monthly.empty:
        print("  結果  : ⏳ データ蓄積中（実運用開始後 3ヶ月待機）")
    else:
        months_done = len(monthly)
        print(f"  結果  : ❌ 未達成（{months_done}ヶ月データ / 目標 {PHASE2_STREAK_MONTHS}ヶ月連続）")
    print("=" * 50)


# ================================================================
# MAIN
# ================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 損益評価スクリプト")
    p.add_argument("--weekly",  action="store_true", help="週次異常検知を表示")
    p.add_argument("--summary", action="store_true", help="Phase 2 判定サマリーのみ")
    p.add_argument("--source",  choices=["api", "log", "auto"], default="auto",
                   help="データソース（auto=API優先→ログフォールバック）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 50)
    print("  Phase 2 損益評価スクリプト")
    print(f"  実行日時: {datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')}")
    print("=" * 50)

    # ---- データ取得 ----
    fills: list[dict] = []
    source_label = ""

    if args.source in ("api", "auto"):
        try:
            fills = fetch_fills_from_api()
            save_fills_cache(fills)
            source_label = f"kabuステーション API（{len(fills)}件）"
            print(f"\n  データソース: {source_label}")
        except Exception as e:
            if args.source == "api":
                print(f"  [ERROR] API取得失敗: {e}", file=sys.stderr)
                return 1
            print(f"  API未接続（{e}）→ ログフォールバック")

    if not fills and args.source in ("log", "auto"):
        fills = fetch_fills_from_logs()
        source_label = f"ログ推定価格（{len(fills)}件・参考値）"
        print(f"\n  データソース: {source_label}")
        if fills:
            print("  ⚠ 推定価格使用中。実際の約定価格とは異なる場合があります。")

    if not fills:
        print("\n  データなし。実運用開始後に再実行してください。")
        print_phase2(pd.Series(dtype=float))
        return 0

    # ---- P&L計算 ----
    pnl_data = calc_pnl(fills)
    unrealized = calc_unrealized_pnl(pnl_data["open_positions"])
    unrealized_total = sum(unrealized.values())

    monthly = to_monthly(pnl_data["realized_by_date"])

    # ---- 表示 ----
    if not args.summary:
        print(f"\n  取引件数  : {len(fills)}件")
        print(f"  実現P&L   : ¥{pnl_data['total_realized']:,.0f}")
        print(f"  支払手数料: ¥{pnl_data['total_fees']:,.0f}")

        if pnl_data["open_positions"]:
            print(f"\n  保有中ポジション:")
            for sym, queue in pnl_data["open_positions"].items():
                total_qty  = sum(q for q, _ in queue)
                avg_price  = sum(q * p for q, p in queue) / total_qty
                unreal     = unrealized.get(sym, 0)
                print(f"    {sym:<10} {total_qty:>5}株  平均¥{avg_price:,.0f}  "
                      f"未実現¥{unreal:+,.0f}")

        print_monthly(monthly, unrealized_total)

        if args.weekly:
            weekly = to_weekly(pnl_data["realized_by_date"])
            print_weekly(weekly)

    print_phase2(monthly)
    return 0


if __name__ == "__main__":
    sys.exit(main())
