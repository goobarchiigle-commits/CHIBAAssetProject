"""
sync_positions.py
kabuステーション API から現在のポジションを取得し、
portfolio_state.json に同期するユーティリティスクリプト。

【使い方】
  # API 接続確認 + 同期（インタラクティブ）
  python src/scripts/sync_positions.py

  # --force: 確認プロンプトをスキップ
  python src/scripts/sync_positions.py --force

【前提】
  - kabuステーション起動・ログイン済みであること
  - src/.env に正しい KABU_API_PASSWORD が設定済みであること

【動作】
  1. API /positions から実際のポジションを取得
  2. 既存の portfolio_state.json を読み込む
  3. position_entry_dates / position_entry_prices / equity を更新
  4. 上書き保存

【注意】
  - entry_date は API から取れないため、取得当日 (today) を使う
    （min_hold チェックが正しく動作するよう、実際の建て日が分かれば --entry-date で指定）
  - このスクリプトは絶対に API に発注しない（読み取り専用）
"""

import sys
import json
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import RUNTIME_DIR
from src.kabusapi.client import KabuClient

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_positions")

PORTFOLIO_STATE_FILE = RUNTIME_DIR / "portfolio_state.json"


def load_state() -> dict:
    from src.portfolio.state_store import load_portfolio_state
    state, vr = load_portfolio_state(PORTFOLIO_STATE_FILE)
    return state


def save_state(state: dict) -> None:
    from src.portfolio.state_store import save_portfolio_state
    save_portfolio_state(state, path=PORTFOLIO_STATE_FILE, data_source="sync_positions")
    logger.info("portfolio_state.json を保存しました: %s", PORTFOLIO_STATE_FILE)


def main() -> int:
    parser = argparse.ArgumentParser(description="kabu API ポジション → portfolio_state.json 同期")
    parser.add_argument("--force", action="store_true", help="確認プロンプトをスキップ")
    parser.add_argument(
        "--entry-date",
        default=None,
        help="ポジションの建て日を指定（YYYY-MM-DD）。未指定の場合は本日を使用",
    )
    args = parser.parse_args()

    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    entry_date = args.entry_date or today_str

    # 1. API接続
    print("=" * 60)
    print("  kabu API ポジション同期スクリプト")
    print(f"  実行日時: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 60)

    try:
        client = KabuClient()
        client.fetch_token()
        logger.info("API トークン取得成功")
    except Exception as e:
        print(f"\n❌ API 接続失敗: {e}")
        print("\n対処法:")
        print("  1. kabuステーションが起動・ログイン済みか確認")
        print("  2. kabuステーション → ツール → API → APIパスワードを確認")
        print("  3. src/.env の KABU_API_PASSWORD を正しい値に更新")
        return 1

    # 2. ポジション取得
    try:
        raw_positions = client.get_positions()
    except Exception as e:
        print(f"\n❌ ポジション取得失敗: {e}")
        return 1

    print(f"\n✅ API 接続成功 | ポジション数: {len(raw_positions)}")
    if not raw_positions:
        print("  現在保有ポジションなし")
        # ポジションなしで state をクリア
        if not args.force:
            ans = input("\n  portfolio_state.json のポジション情報をクリアしますか？ [y/N] > ")
            if ans.strip().lower() != "y":
                print("キャンセルしました。")
                return 0
        state = load_state()
        state["position_entry_dates"]    = {}
        state["position_entry_prices"]   = {}
        state["position_entry_atrs"]     = {}
        state["position_highest_closes"] = {}
        save_state(state)
        print("  portfolio_state.json をクリアしました。")
        return 0

    # 3. パース・表示（qty <= 0 は除外）
    from src.common.position_normalizer import filter_live_positions
    live_raw = filter_live_positions(raw_positions)
    zero_qty_syms = [
        f"{p.get('Symbol','')}.T" for p in raw_positions
        if p not in live_raw
    ]

    print(f"\n{'銘柄':<10} {'保有株数':>8} {'平均単価':>10} {'評価額':>12}")
    print("-" * 46)
    total_value   = 0.0
    api_positions = []

    if zero_qty_syms:
        for sym in zero_qty_syms:
            print(f"  {sym:<10} {'0':>8}株  (保有なし — スキップ)")

    for p in live_raw:
        code  = p.get("Symbol", "")
        sym   = f"{code}.T" if not code.endswith(".T") else code
        qty   = p.get("LeavesQty") or 0
        price = p.get("Price", 0.0) or 0.0

        value = qty * price
        total_value += value
        api_positions.append({"symbol": sym, "qty": qty, "avg_price": price, "value": value})
        print(f"  {sym:<10} {qty:>8}株  ¥{price:>9,.0f}  ¥{value:>11,.0f}")

    if zero_qty_syms:
        print(f"\n  ⚠ qty=0 として除外: {zero_qty_syms}")

    print("-" * 46)
    print(f"  {'合計評価額':>30}  ¥{total_value:>11,.0f}")

    # 4. 余力取得
    available_cash = None
    try:
        wallet = client.get_wallet_cash()
        available_cash = float(wallet.get("StockAccountWallet", 0))
        print(f"  {'現物余力':>30}  ¥{available_cash:>11,.0f}")
        from src.portfolio.equity import compute_live_equity as _cle
        _disp_pos = {p["symbol"]: {"qty": p["qty"], "avg_price": p["avg_price"]}
                     for p in api_positions}
        estimated_equity = _cle(available_cash, _disp_pos, mode="display",
                                persist_snapshot=False)
        print(f"  {'推定総資産':>30}  ¥{estimated_equity:>11,.0f}")
    except Exception as e:
        logger.warning("余力取得失敗: %s", e)

    # 5. 確認
    if not args.force:
        print(f"\n  建て日として '{entry_date}' を使用します。")
        print("  （実際の建て日が異なる場合は --entry-date YYYY-MM-DD で指定してください）")
        ans = input("\n  portfolio_state.json を上記内容で更新しますか？ [y/N] > ")
        if ans.strip().lower() != "y":
            print("キャンセルしました。")
            return 0

    # 6. state 更新
    state = load_state()

    # ── 6a. qty=0 銘柄をすべての state キーから削除 ──────────────────
    # GHOST_POSITION_FIX (2026-07-03): 判定元を position_entry_dates だけでなく
    # position_qtys とも突き合わせる（2026-07-03 実例: 5301.T は position_qtys に
    # qty=300 で残留していたが position_entry_dates には既に存在せず、旧ロジックの
    # _stale 集合が検出できなかった）。削除対象キーにも position_qtys /
    # position_current_prices / position_unrealized_pnl / position_unrealized_pct /
    # position_missing_streak を追加し、equity 計算に効く数量系フィールドを
    # 確実に一掃する。
    _all_zero = set(zero_qty_syms)
    # API に存在しない既存保有もゼロ扱いでクリーンアップ
    _api_syms = {p["symbol"] for p in api_positions}
    _known_syms = (
        set(state.get("position_entry_dates", {}).keys())
        | set(state.get("position_qtys", {}).keys())
    )
    _stale = _known_syms - _api_syms
    _to_remove = _all_zero | _stale
    if _to_remove:
        for _key in (
            "position_entry_dates", "position_entry_prices",
            "position_entry_atrs", "position_highest_closes",
            "position_strategy_types", "shadow_positions",
            "position_qtys", "position_current_prices",
            "position_unrealized_pnl", "position_unrealized_pct",
            "position_missing_streak",
        ):
            for sym in _to_remove:
                state.get(_key, {}).pop(sym, None)
        state["positions_count"] = sum(
            1 for q in state.get("position_qtys", {}).values() if int(q) > 0
        )
        logger.info("state から除去（qty=0 または API 未検出）: %s", sorted(_to_remove))

    # ── 6b. 保有中銘柄を書き込み ────────────────────────────────────
    new_entry_dates  = {}
    new_entry_prices = {}
    for p in api_positions:
        sym = p["symbol"]
        existing_date    = state.get("position_entry_dates", {}).get(sym)
        new_entry_dates[sym]  = existing_date or entry_date
        new_entry_prices[sym] = p["avg_price"]

    state["position_entry_dates"]  = new_entry_dates
    state["position_entry_prices"] = new_entry_prices
    # highest_close は avg_price で初期化（未設定の場合のみ）
    # 保有中ポジションの highest_close は絶対に avg_price で上書きしない（trailing stop 基準が壊れる）
    for p in api_positions:
        sym = p["symbol"]
        existing_hc = state.get("position_highest_closes", {}).get(sym)
        if existing_hc is None:
            state.setdefault("position_highest_closes", {})[sym] = p["avg_price"]
        elif p["avg_price"] > 0 and p["avg_price"] < existing_hc:
            logger.info(
                "highest_close 保護: %s avg_price=¥%,.0f < highest_close=¥%,.0f → 上書きしない",
                sym, p["avg_price"], existing_hc,
            )

    # last_equity のみ更新（正規計算路を経由）。
    # equity_peak は _update_cb_state() 以外からの書き込みを禁止する
    # (EQUITY_PEAK_HARDENING, 2026-07-03)。sync_positions.py は観測ログのみ出力し、
    # 実際の peak 更新（broker整合性チェック + candidate_peak ステージング込み）は
    # 次回 bridge.run() の _update_cb_state() に委ねる。
    if available_cash is not None:
        from src.portfolio.equity import compute_live_equity
        _pos_for_equity = {
            p["symbol"]: {"qty": p["qty"], "avg_price": p["avg_price"]}
            for p in api_positions
        }
        estimated_equity = compute_live_equity(
            live_cash    = available_cash,
            positions    = _pos_for_equity,
            universe_raw = None,   # avg_price = API current price を使用
            mode         = "reconcile",
            equity_peak  = float(state.get("equity_peak", 0)),
        )
        _old_peak = float(state.get("equity_peak", 0))
        if estimated_equity > _old_peak:
            logger.warning(
                "[EQUITY_PEAK_OBSERVED_HIGHER] estimated_equity=¥%s > persisted_peak=¥%s "
                "— sync_positions は peak を更新しません（次回 bridge 実行で反映）。",
                f"{estimated_equity:,.0f}", f"{_old_peak:,.0f}",
            )
        state["last_equity"] = round(estimated_equity, 0)

    save_state(state)

    print("\n✅ portfolio_state.json を更新しました。")
    print(f"   建て日: {entry_date}")
    print(f"   ポジション数: {len(api_positions)}")
    for sym, d in new_entry_dates.items():
        price = new_entry_prices.get(sym, 0)
        print(f"   {sym}: entry_date={d} avg_price=¥{price:,.0f}")

    # 7. 不整合原因のメモをログに記録
    cause_log = PORTFOLIO_STATE_FILE.parent / "sync_log.jsonl"
    import json as _j
    event = {
        "date":        today_str,
        "event":       "manual_sync",
        "symbols":     [p["symbol"] for p in api_positions],
        "entry_date":  entry_date,
        "total_value": round(total_value, 0),
        "cash":        round(available_cash, 0) if available_cash else None,
    }
    with cause_log.open("a", encoding="utf-8") as f:
        f.write(_j.dumps(event, ensure_ascii=False) + "\n")
    logger.info("同期ログ記録: %s", cause_log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
