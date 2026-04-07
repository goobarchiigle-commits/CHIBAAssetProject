"""
run_morning_signal.py
毎朝のシグナル生成 + 発注スクリプト

【使い方】
  # ドライラン（発注しない・口座不要）
  python run_morning_signal.py

  # 発注確認あり（実際にAPIへ送信）
  python run_morning_signal.py --live

  # 確認プロンプトをスキップして自動送信（cron/タスクスケジューラ用）
  python run_morning_signal.py --live --yes

  # シグナルのみ表示（JSON保存しない）
  python run_morning_signal.py --no-save

【推奨実行タイミング】
  8:30〜8:55（前場寄付き 9:00 の前）
  → 前日終値データが yfinance に反映されていることを確認してから実行

【Windowsタスクスケジューラへの登録例】
  タスク名    : 朝のシグナル確認
  トリガー    : 毎営業日 08:30
  操作        : python run_morning_signal.py
  作業ディレクトリ: (プロジェクトルートを設定すること)

【CLAUDE.md ルール3 確認】
  .env ファイルに KABU_API_PASSWORD / KABU_TRADE_PASSWORD が設定されていること。
  このスクリプト自体に認証情報を記述しないこと。
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))           # C:/ai-trading/src/ → backtest.xxx imports
sys.path.insert(0, str(_here.parent))    # C:/ai-trading/     → src.xxx imports
sys.stdout.reconfigure(encoding="utf-8")

from src.config_loader import load_strategy_config
from src.paths import SIGNALS_DIR, LIVE_UNIVERSE_FILE, ORDER_LOCK_FILE  # .env の読み込みも paths.py が担う
from src.utils.morning_smoke_test import run_morning_smoke_test

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("morning_signal")
cfg = load_strategy_config()

# token reuse DEBUG 確認用（確認後は削除またはコメントアウト可）
import logging as _logging
_logging.getLogger("src.kabusapi.client").setLevel(_logging.DEBUG)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="フジコ法 朝のシグナル生成スクリプト")
    p.add_argument(
        "--live",
        action="store_true",
        help="kabuステーション API に実際に発注する（デフォルト: ドライラン）",
    )
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="発注確認プロンプトをスキップ（--live と併用）",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="JSON シグナルファイルを保存しない",
    )
    p.add_argument(
        "--output-dir",
        default=str(SIGNALS_DIR),
        help="シグナル JSON の保存先ディレクトリ",
    )
    return p.parse_args()


def print_banner(live: bool) -> None:
    mode = "🔴 LIVE（実発注）" if live else "🟡 DRY RUN（発注なし）"
    print("=" * 60)
    print(f"  フジコ法 朝のシグナル確認")
    print(f"  実行日時 : {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"  モード   : {mode}")
    print("=" * 60)


def print_signals(result) -> None:
    """シグナルを人間が読みやすい形式で表示する。"""
    orders = result.orders

    if not orders:
        print("\n📭 本日の注文なし（全銘柄 HOLD / 条件不成立）")
    else:
        print(f"\n📋 発注予定: {len(orders)} 件")
        print(f"  {'銘柄':<10} {'売買':<6} {'数量':>6} {'参考価格':>10} {'参考金額':>12}  理由")
        print("  " + "-" * 72)
        for o in orders:
            side_str = "🟢 BUY " if o["side"] == "BUY" else "🔴 SELL"
            print(
                f"  {o['symbol']:<10} {side_str} {o['qty']:>6}株 "
                f"¥{o['estimated_price']:>9,.0f} "
                f"¥{o['estimated_amount']:>11,.0f}  "
                f"{o['reason']}"
            )

    if result.warnings:
        print("\n⚠ 警告:")
        for w in result.warnings:
            print(f"  - {w}")

    # RSR トップ10 をサマリー表示
    top_signals = [s for s in result.signals if s["signal"] in (1, 0)][:10]
    if top_signals:
        print(f"\n📊 RSR ランキング（ユニバース上位10銘柄）")
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<6} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  シグナル")
        print("  " + "-" * 66)
        for s in top_signals:
            sig_str   = "✅ BUY" if s["signal"] == 1 else "  -  "
            strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<6} "
                f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  {sig_str}"
            )
    # 平均回帰 BUY シグナルを別枠で表示
    mr_buys = [s for s in result.signals if s["signal"] == 1 and s.get("strategy_type") == "mean_rev"]
    if mr_buys:
        print(f"\n📊 平均回帰 BUY シグナル（{len(mr_buys)}件）")
        print(f"  {'銘柄':<10} {'セクター':<10}  理由")
        print("  " + "-" * 60)
        for s in mr_buys:
            print(f"  {s['symbol']:<10} {s['sector']:<10}  {s['reason']}")


def assert_live_mode_env() -> None:
    """LIVE_MODE 環境変数が 'true' でない限り --live フラグを無効化して終了する。"""
    live_mode_env = os.getenv("LIVE_MODE", "false").lower()
    if live_mode_env != "true":
        print(f"[GUARD] LIVE_MODE={live_mode_env!r}. --live フラグが指定されましたが LIVE_MODE=true でないため終了します。")
        print("  発注を行うには: export LIVE_MODE=true または .env に LIVE_MODE=true を設定してください。")
        sys.exit(0)  # エラーではなく正常終了（スクリプト誤実行の保護）


def confirm_live_orders(orders: list) -> bool:
    """発注前に確認プロンプトを表示する。"""
    buy_orders  = [o for o in orders if o["side"] == "BUY"]
    sell_orders = [o for o in orders if o["side"] == "SELL"]
    total_buy   = sum(o["estimated_amount"] for o in buy_orders)

    print("\n" + "=" * 60)
    print("  ⚠  実際の発注を行います。内容を確認してください。")
    print("=" * 60)
    print(f"  BUY  : {len(buy_orders)} 件  （推定合計: ¥{total_buy:,.0f}）")
    print(f"  SELL : {len(sell_orders)} 件")
    print("=" * 60)

    ans = input("  発注を実行しますか？ [y/N] > ").strip().lower()
    return ans == "y"


def save_signal_json(result, output_dir: str) -> Path:
    """シグナル JSON をタイムスタンプ付きファイルに保存する。"""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    file_path = dir_path / f"signal_{ts}.json"
    file_path.write_text(result.to_json(), encoding="utf-8")
    return file_path


def already_ordered_today(symbol: str) -> bool:
    """当日すでに発注済みか確認する（同一銘柄・同日の重複発注防止）。"""
    today = datetime.now(JST).date().isoformat()
    if not ORDER_LOCK_FILE.exists():
        return False
    try:
        lock = json.loads(ORDER_LOCK_FILE.read_text(encoding="utf-8"))
        return bool(lock.get(today, {}).get(symbol))
    except Exception:
        return False


def mark_ordered(symbol: str, side: str) -> None:
    """発注成功後にロックを書き込む（再実行時の二重発注防止）。"""
    today = datetime.now(JST).date().isoformat()
    lock: dict = {}
    if ORDER_LOCK_FILE.exists():
        try:
            lock = json.loads(ORDER_LOCK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    ORDER_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock.setdefault(today, {})[symbol] = side
    ORDER_LOCK_FILE.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("idempotency lock: %s %s を記録", side, symbol)


def check_api_connection() -> bool:
    """
    kabuステーションAPI ping（fetch_token で疎通確認）。
    接続成功なら True、失敗なら False を返す（例外は握りつぶす）。
    注: fetch_token() は毎回新規トークンを取得するためトークン失効問題は発生しない。
    """
    try:
        from src.kabusapi.client import KabuClient
        KabuClient().fetch_token()
        logger.info("kabuステーション API: 接続OK ✅")
        return True
    except Exception as exc:
        logger.warning("kabuステーション API: 未接続 → %s", exc)
        return False


def sync_portfolio_from_api() -> "dict | None":
    """
    kabuステーション API から現在の保有ポジションを取得し、
    portfolio_state.json と突き合わせて同期する。

    - broker にあるが portfolio_state にない銘柄 → 追加（entry_date=今日、entry_price=broker avg_price）
    - portfolio_state にあるが broker にない銘柄 → 削除（実際に決済済みとみなす）
    - API 未接続の場合は何もしない（ドライランでも安全に動作）
    """
    from pathlib import Path as _Path
    import json as _json

    PORTFOLIO_STATE_FILE = _Path(__file__).resolve().parent.parent / "runtime" / "portfolio_state.json"
    TODAY = datetime.now(JST).strftime("%Y-%m-%d")

    try:
        from src.kabusapi.client import KabuClient
        client = KabuClient()
        client.fetch_token()
        raw_positions = client.get_positions()
    except Exception as exc:
        logger.info("ポジション同期スキップ（API未接続）: %s", exc)
        return None

    # broker から保有銘柄を収集（.T サフィックス付きに正規化）
    broker_positions: dict[str, float] = {}
    for p in raw_positions:
        sym_code = p.get("Symbol", "")
        if sym_code:
            sym = f"{sym_code}.T" if not sym_code.endswith(".T") else sym_code
            avg_price = float(p.get("Price", 0.0))
            qty = int(p.get("LeavesQty", 0))
            if qty > 0:
                broker_positions[sym] = avg_price

    # portfolio_state.json 読み込み
    default_state: dict = {
        "cb_state": "NORMAL",
        "equity_peak": 3000000,
        "cb_cooldown_end_date": None,
        "recovery_threshold": None,
        "position_entry_dates": {},
        "position_entry_prices": {},
        "position_entry_atrs": {},
        "position_highest_closes": {},
        "reentry_blocked": {},
        "last_updated": None,
        "shadow_virtual_positions": {},
    }
    if PORTFOLIO_STATE_FILE.exists():
        try:
            state = _json.loads(PORTFOLIO_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = default_state.copy()
    else:
        state = default_state.copy()

    entry_dates  = state.setdefault("position_entry_dates",   {})
    entry_prices = state.setdefault("position_entry_prices",  {})
    highest_cls  = state.setdefault("position_highest_closes", {})
    changed = False

    # broker にあるが portfolio_state にない → 追加
    for sym, avg_price in broker_positions.items():
        if sym not in entry_dates:
            logger.warning("ポジション同期: %s を portfolio_state に追加（entry_date=%s, price=%.1f）", sym, TODAY, avg_price)
            entry_dates[sym]  = TODAY
            entry_prices[sym] = avg_price
            highest_cls[sym]  = avg_price
            changed = True

    # portfolio_state にあるが broker にない → 削除
    stale = [sym for sym in list(entry_dates.keys()) if sym not in broker_positions]
    for sym in stale:
        logger.warning("ポジション同期: %s を portfolio_state から削除（broker に保有なし）", sym)
        entry_dates.pop(sym, None)
        entry_prices.pop(sym, None)
        highest_cls.pop(sym, None)
        state.get("position_entry_atrs", {}).pop(sym, None)
        changed = True

    if changed:
        state["last_updated"] = TODAY
        PORTFOLIO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PORTFOLIO_STATE_FILE.write_text(
            _json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("portfolio_state.json を API から同期しました（broker: %d銘柄）", len(broker_positions))
    else:
        logger.info("portfolio_state.json は最新（差分なし・broker: %d銘柄）", len(broker_positions))

    return broker_positions


def load_live_universe() -> dict[str, str]:
    """
    LIVE_UNIVERSE_FILE（configs/universe/rsr42_trading.json）から
    {symbol: sector} を読み込む。ファイル不在・銘柄数不足は RuntimeError。
    """
    if not LIVE_UNIVERSE_FILE.exists():
        raise RuntimeError(
            f"ユニバースファイルが見つかりません: {LIVE_UNIVERSE_FILE}\n"
            "LIVE_UNIVERSE_FILE 環境変数または configs/universe/ を確認してください。"
        )
    data = json.loads(LIVE_UNIVERSE_FILE.read_text(encoding="utf-8"))
    symbols: dict[str, str] = data.get("symbols", {})
    if len(symbols) < 10:
        raise RuntimeError(
            f"ユニバース銘柄数 {len(symbols)} < 最小要件 10。"
            "ファイルが壊れている可能性があります。"
        )
    logger.info(
        "ユニバース読み込み: %d銘柄 (%s)",
        len(symbols), LIVE_UNIVERSE_FILE.name,
    )
    return symbols


def _write_shadow_log(record: dict) -> None:
    """仮想約定ログを runtime/shadow_orders.jsonl に追記する。"""
    path = Path("runtime/shadow_orders.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[SHADOW] Write failed: %s", e)


def main() -> int:
    args = parse_args()
    print_banner(args.live)

    import warnings
    warnings.filterwarnings("ignore")

    # ---- ユニバース読み込み（旧: backtest.topix100_backtest → 現: configs/universe/） ----
    try:
        universe_tickers = load_live_universe()
    except RuntimeError as exc:
        logger.error("ユニバース読み込み失敗: %s", exc)
        return 1

    # ---- kabuステーション API ping ----
    api_connected = check_api_connection()
    if not api_connected:
        logger.error("API unavailable - skip all signal generation")
        return 1
    api_ok = True  # downstream の sync_portfolio_from_api() 参照用

    # ---- ポジション同期（API 接続時のみ）----
    # portfolio_state.json と broker の保有ポジションを突き合わせ、乖離を修正する。
    # これにより「発注成功したのに state が更新されていない」状態を自動修復する。
    if api_ok:
        positions = sync_portfolio_from_api()
        try:
            from src.kabusapi.client import KabuClient
            cash_info = KabuClient().get_wallet_cash()
        except Exception:
            cash_info = None
        if cash_info is None or positions is None:
            logger.error(
                "Broker state unavailable (cash=%s, positions=%s) - halt execution",
                cash_info,
                positions,
            )
            raise RuntimeError("broker state unavailable")

    from src.kabusapi.signal_bridge import SignalBridge

    # ---- ブリッジ初期化 ----
    try:
        bridge = SignalBridge(
            universe_tickers   = universe_tickers,
            fujiko_params      = {
                "min_sepa":         cfg.fujiko.min_sepa,
                "min_rsr":          cfg.fujiko.min_rsr,
                "mom_period":       cfg.fujiko.mom_period,
                "turtle_entry":     cfg.fujiko.turtle_entry,
                "turtle_exit":      cfg.fujiko.turtle_exit,
                "use_turtle_entry": cfg.fujiko.use_turtle_entry,
            },
            capital            = cfg.portfolio.capital,
            max_positions      = cfg.portfolio.max_positions,
            max_dd_limit       = 0.15,
            min_sectors        = 1,
            live               = args.live,
            min_hold_days      = cfg.risk.min_hold_days,
            emergency_exit_pct = cfg.risk.emergency_exit_pct,
            cfg                = cfg,
        )
    except Exception as exc:
        logger.error("SignalBridge 初期化失敗: %s", exc)
        return 1

    # ---- シグナル生成 ----
    logger.info("シグナル生成開始...")
    result, order_objects = bridge.run()

    # ---- 朝イチ smoke test（bridge.run() 完了後・発注前） ----
    # bear_filter_log.jsonl から当日の bear 判定情報を自動取得。
    # active_count < 3 のみ WARNING を出すが、発注は止めない。
    try:
        run_morning_smoke_test(
            today=datetime.now(JST).date(),
            universe_tickers=universe_tickers,  # bull 時の全銘柄（active_syms 推定用）
        )
    except Exception as _smoke_exc:
        logger.warning("smoke test 実行失敗（発注は継続）: %s", _smoke_exc)

    # ---- 表示 ----
    print(f"\n  データ基準日 : {result.data_as_of}")
    print(f"  ユニバース   : {result.n_universe} 銘柄")
    print(
        f"  ポートフォリオ: 保有 {result.portfolio_summary['current_positions']} / "
        f"最大 {result.portfolio_summary['max_positions']} 銘柄  "
        f"空きスロット: {result.portfolio_summary['open_slots']}  "
        f"余力: ¥{result.portfolio_summary['available_cash']:,.0f}"
    )

    print_signals(result)

    # ---- JSON 保存 ----
    if not args.no_save:
        saved_path = save_signal_json(result, args.output_dir)
        print(f"\n💾 シグナル保存: {saved_path}")

    # ---- 発注 ----
    if not args.live:
        print("\n" + "=" * 60)
        print("  ドライランのため発注は行いません。")
        print("  実際に発注するには --live オプションを付けて実行してください。")
        print("=" * 60)
        return 0

    # ---- P0-3: LIVE_MODE 環境変数ガード（--live 指定時のみ） ----
    assert_live_mode_env()

    # ---- Shadow mode 設定読み込み ----
    SHADOW_MODE: bool = cfg.live_execution.shadow_mode

    # ---- OrderLedger 初期化（重複チェック・当日カウント） ----
    from src.live.order_ledger import OrderLedger
    today = datetime.now(JST).date()
    ledger = OrderLedger(trade_date=today)

    # ---- 発注直前 hard guard ----
    from src.live.execution_guard import check_execution_preconditions
    guard_result = check_execution_preconditions(
        active_syms=list(universe_tickers.keys()),
        signals=result.orders,
        current_positions={},
        daily_order_count=ledger.daily_count(),
    )
    if guard_result["status"] == "BLOCKED":
        logger.error(
            "[GUARD] Execution BLOCKED: reason=%s, detail=%s",
            guard_result["reason"],
            guard_result.get("detail", ""),
        )
        return 1
    if guard_result["status"] == "NO_SIGNAL":
        logger.info("[GUARD] No signals today. Skipping execution.")
        return 0

    # ---- P0-1: 起動時 broker positions 取得 + drift チェック ----
    from src.live.position_sync import sync_and_validate_state
    sync_result = sync_and_validate_state(bridge)
    if sync_result["drift_detected"] and sync_result["halt_trading"]:
        logger.critical(
            "[STARTUP] State drift detected. HALTING. "
            "broker_syms=%s, local_syms=%s",
            sync_result["broker_syms"],
            sync_result["local_syms"],
        )
        sys.exit(1)

    if not order_objects:
        print("\n発注なし。終了します。")
        return 0

    # 確認プロンプト
    order_dicts = result.orders
    if not args.yes:
        confirmed = confirm_live_orders(order_dicts)
        if not confirmed:
            print("発注をキャンセルしました。")
            return 0

    # ---- idempotency チェック（OrderLedger による重複発注防止）----
    deduped = []
    for o in order_objects:
        check = ledger.check_and_record(
            o.symbol, o.side,
            qty=getattr(o, "qty", 0),
            price=getattr(o, "estimated_price", 0.0),
        )
        if not check["allowed"]:
            logger.warning("重複発注スキップ: %s（%s）", o.symbol, check["reason"])
            print(f"  ⚠ {o.symbol}: 発注スキップ（{check['reason']}）")
        else:
            deduped.append(o)
    order_objects = deduped

    if not order_objects:
        print("\n発注可能な注文なし（重複除外後）。終了します。")
        return 0

    # ---- 執行前二重ガード（research層capとは独立した実行直前チェック）----
    exec_checked = []
    for _o in order_objects:
        if _o.side != "BUY" or bridge.pre_trade_risk_check(_o):
            exec_checked.append(_o)
        else:
            logger.warning(
                "RISK_CHECK_REJECT (execution layer): %s %s をスキップ",
                _o.side, _o.symbol,
            )
            print(f"  ⚠ {_o.symbol}: 執行前リスクチェック不合格（sector/symbol cap）→ スキップ")
    order_objects = exec_checked
    if not order_objects:
        print("\n発注可能な注文なし（二重ガード除外後）。終了します。")
        return 0

    # ---- Shadow mode: 実発注せずログのみ ----
    if SHADOW_MODE:
        print("\n[SHADOW MODE] 実発注をスキップしてログ記録のみ行います。")
        print(f"  shadow_mode_reason: {cfg.live_execution.shadow_mode_reason}")
        for o in order_objects:
            shadow_record = {
                "shadow_order": {
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": getattr(o, "qty", 0),
                    "price": getattr(o, "estimated_price", 0.0),
                    "reason": "shadow_mode=true",
                }
            }
            logger.info("[SHADOW] %s", json.dumps(shadow_record, ensure_ascii=False))
            _write_shadow_log(shadow_record)
            print(f"  [SHADOW] {o.side} {o.symbol} qty={getattr(o, 'qty', 0)}")
        print("\n  shadow_orders.jsonl に記録しました。")
        print("  本番発注には strategy.yaml の live_execution.shadow_mode: false が必要です。")
        return 0

    # 発注実行
    print("\n発注中...")
    # P1-4: idempotency ロックは OrderLedger.check_and_record() が発注前に記録済み。
    send_results = bridge._send_orders(order_objects)

    # 発注結果表示
    print("\n=== 発注結果 ===")
    for r in send_results:
        status = "✅ 成功" if r.get("success") else "❌ 失敗"
        order_id = r.get("order_id", r.get("error", ""))
        print(f"  {r['side']} {r['symbol']} {r['qty']}株 → {status}  ({order_id})")

    # ── ポートフォリオ状態更新（約定確認後）────────────────────────────
    # 発注成功した銘柄のエントリー日・価格・ATR を portfolio_state.json に記録する。
    # これがないと次回起動時に「ポジションなし」と誤認され二重発注が起きる。
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    bridge.update_state_after_execution(send_results, today_str)

    # 結果を JSON に追記して再保存
    result_dict = json.loads(result.to_json())
    result_dict["send_results"] = send_results
    if not args.no_save:
        saved_path = Path(args.output_dir) / f"signal_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}_executed.json"
        saved_path.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n💾 発注結果保存: {saved_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
