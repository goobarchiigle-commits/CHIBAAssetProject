"""
run_live_signal.py
最適化済みパラメータ（V2推奨設定 Calmar 2.656達成）での朝のシグナル＆発注スクリプト

【設定根拠】
  - バックテスト: 2018-2024、初期資本200万円
  - V2シナリオ（RSR>70 + 均等ウェイト + 29銘柄）:
      CAGR=+16.26% / MaxDD=-6.12% / Calmar=2.656 / Sharpe=1.693
  - 2026-03-19 TEMPORAL選定に更新（銘柄選択バイアス除去）:
      Sharpe=1.070 / CAGR=+9.98% / MaxDD=-10.62%（真の性能推定値）

【パラメータ（V2）】
  - 宇宙   : LIVE_UNIVERSE_FILE で指定（configs/universe/2026Q1_temporal24.json）
  - min_rsr : 70（旧75から緩和）
  - max_pos : 3
  - min_sec : 1（セクター制約なし）
  - ウェイト: 均等（vol_target=0、IDM無効）
  - 決算保護: 無効（逆効果のため採用しない）

【使い方】
  # ドライラン（発注しない）
  python run_live_signal.py

  # 実発注（kabuステーション起動・ログイン後）
  python run_live_signal.py --live

  # 確認スキップ（自動化用）
  python run_live_signal.py --live --yes

【CLAUDE.md ルール3 確認】
  .env に KABU_API_PASSWORD / KABU_TRADE_PASSWORD / LIVE_UNIVERSE_FILE が設定されていること。
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("live_signal")

# ------------------------------------------------------------------ #
# .env 読み込み（python-dotenv があれば使用、なければ手動パース）
# ------------------------------------------------------------------ #
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

# ------------------------------------------------------------------ #
# 安全設計定数（取引所の過剰発注監視対策）
# ------------------------------------------------------------------ #
MAX_DAILY_ORDERS   = 20   # 1日の発注上限（BUY + SELL 合計）
MAX_SYMBOL_ORDERS  = 2    # 1銘柄あたりの1日の発注上限
MAX_OPEN_POSITIONS = 10   # ポートフォリオ最大保有銘柄数（ハードキャップ）
MIN_UNIVERSE_SIZE  = 10   # ユニバース最小銘柄数（これを下回ったら起動しない）

# ------------------------------------------------------------------ #
# ユニバースファイルのロード（LIVE_UNIVERSE_FILE 環境変数から）
# ------------------------------------------------------------------ #
def load_universe() -> tuple[dict[str, str], dict]:
    """
    環境変数 LIVE_UNIVERSE_FILE で指定された JSON ファイルからユニバースを読み込む。

    Returns:
        (tickers, meta): tickers = {symbol: sector}, meta = JSONのメタ情報

    Raises:
        RuntimeError: 環境変数未設定・ファイル不存在・銘柄数不足
    """
    universe_file = os.environ.get("LIVE_UNIVERSE_FILE")
    if not universe_file:
        raise RuntimeError(
            "LIVE_UNIVERSE_FILE が設定されていません。\n"
            ".env に LIVE_UNIVERSE_FILE=configs/universe/2026Q1_temporal24.json を追加してください。"
        )

    file_path = Path(universe_file)
    if not file_path.exists():
        raise RuntimeError(
            f"ユニバースファイルが見つかりません: {file_path}\n"
            "パスを確認してください。"
        )

    data = json.loads(file_path.read_text(encoding="utf-8"))

    symbols: dict[str, str] = data.get("symbols", {})
    if len(symbols) < MIN_UNIVERSE_SIZE:
        raise RuntimeError(
            f"ユニバース銘柄数 {len(symbols)} < 最小要件 {MIN_UNIVERSE_SIZE}。\n"
            "ファイルが壊れている可能性があります。起動を中止します。"
        )

    meta = {k: v for k, v in data.items() if k != "symbols"}
    return symbols, meta


# ------------------------------------------------------------------ #
# 株価上限フィルター（シグナル生成前に適用）
# ------------------------------------------------------------------ #
MAX_ALLOCATION = int(os.environ.get("MAX_POSITION_YEN", 500_000))
LOT_SIZE       = 100   # 東証標準単元株数

def filter_universe_by_price(
    tickers:      dict[str, str],
    max_alloc:    int,
    held_symbols: set[str],
) -> tuple[dict[str, str], list[tuple[str, float, float]]]:
    """
    現在株価に基づきBUY不可能な銘柄をユニバースから除外する。

    Args:
        tickers:      {symbol: sector} のユニバース辞書
        max_alloc:    1銘柄への最大配分額（円）
        held_symbols: 保有中銘柄のset — SELL シグナルが必要なため価格問わず残す

    Returns:
        (filtered, skipped):
            filtered = {symbol: sector}（価格フィルター通過分）
            skipped  = [(symbol, price, cost), ...]（除外分）

    注意:
        API 負荷を抑えるため period="3d" の軽量取得のみ行う。
        価格取得失敗時は保守的に残す（シグナル生成側でスキップされる）。
    """
    import yfinance as yf
    import warnings
    warnings.filterwarnings("ignore")

    syms = list(tickers.keys())
    # 3日分だけ取得（終値だけ欲しい。600日取得は bridge 内で行う）
    raw = yf.download(syms, period="3d", progress=False, group_by="ticker")

    filtered: dict[str, str] = {}
    skipped:  list[tuple[str, float, float]] = []

    for sym, sector in tickers.items():
        # 保有中は価格関係なく残す（SELL シグナルを殺さない）
        if sym in held_symbols:
            filtered[sym] = sector
            continue

        # 最新終値を取得
        try:
            if len(syms) == 1:
                price = float(raw["Close"].dropna().iloc[-1])
            else:
                price = float(raw[sym]["Close"].dropna().iloc[-1])
        except (KeyError, IndexError, TypeError, ValueError):
            filtered[sym] = sector   # 取得失敗 → 保守的に残す
            continue

        cost = price * LOT_SIZE
        if cost <= max_alloc:
            filtered[sym] = sector
        else:
            skipped.append((sym, price, cost))

    return filtered, skipped


# ------------------------------------------------------------------ #
# 1. 二重発注防止 — オーダーロックファイル
# ------------------------------------------------------------------ #
ORDER_LOCK_FILE = Path("runtime/order_lock.json")

def _load_order_lock() -> dict:
    if not ORDER_LOCK_FILE.exists():
        return {}
    try:
        return json.loads(ORDER_LOCK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_order_lock(data: dict) -> None:
    ORDER_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    ORDER_LOCK_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def already_ordered_today(symbol: str) -> bool:
    """当日すでに発注済みか確認する（BUY/SELL 問わず）。"""
    today = datetime.now(JST).date().isoformat()
    return _load_order_lock().get(today, {}).get(symbol, False)

def mark_ordered(symbol: str, side: str) -> None:
    """発注成功後にロックを書き込む。"""
    today = datetime.now(JST).date().isoformat()
    lock = _load_order_lock()
    lock.setdefault(today, {})[symbol] = side   # "BUY" / "SELL" を記録
    _save_order_lock(lock)


# ------------------------------------------------------------------ #
# 当日の発注件数カウント（MAX_DAILY_ORDERS / MAX_SYMBOL_ORDERS チェック用）
# ------------------------------------------------------------------ #
def _count_today_orders(signal_dir: str) -> tuple[int, dict[str, int]]:
    """
    data/signals/ 内の当日 executed ファイルを集計し
    (total_orders, per_symbol_count) を返す。
    """
    today_str    = datetime.now(JST).strftime("%Y%m%d")
    signal_path  = Path(signal_dir)
    total        = 0
    per_symbol: dict[str, int] = {}

    if not signal_path.exists():
        return 0, {}

    for f in signal_path.glob(f"signal_{today_str}*_executed.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for o in data.get("orders", []):
                sym = o.get("symbol", "")
                total += 1
                per_symbol[sym] = per_symbol.get(sym, 0) + 1
        except Exception:
            pass  # 壊れたファイルはスキップ

    return total, per_symbol


# ------------------------------------------------------------------ #
# 2. API障害耐性 — リトライ付き発注
# ------------------------------------------------------------------ #
MAX_RETRY   = 3
RETRY_SLEEP = 3   # 秒

def _send_orders_with_retry(bridge, orders: list) -> list[dict]:
    """
    _send_orders() を最大 MAX_RETRY 回リトライする。
    成功した発注のロックマークも書き込む。

    Returns:
        send_results: [{symbol, side, qty, order_id, success, ...}, ...]
    """
    import time
    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            results = bridge._send_orders(orders)
            # 成功した注文をロックファイルに記録
            for r in results:
                if r.get("success"):
                    mark_ordered(r["symbol"], r["side"])
            return results
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRY:
                logger.warning(
                    "発注エラー（試行 %d/%d）: %s — %d秒後にリトライ",
                    attempt, MAX_RETRY, e, RETRY_SLEEP,
                )
                time.sleep(RETRY_SLEEP)
            else:
                logger.error("発注失敗（%d回試行後）: %s", MAX_RETRY, e)
    raise RuntimeError(f"発注 {MAX_RETRY}回失敗: {last_exc}") from last_exc


# ------------------------------------------------------------------ #
# 3. ライブ運用監視ログ
# ------------------------------------------------------------------ #
LIVE_LOG_DIR = Path("logs/live")

def save_live_logs(run_id: str, result, send_results: list) -> None:
    """
    logs/live/YYYYMMDD_signals.json  — 全銘柄シグナル
    logs/live/YYYYMMDD_orders.json   — 発注+約定結果

    同日に複数回実行された場合はリスト末尾に追記（上書きしない）。
    """
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- signals ---
    sig_path = LIVE_LOG_DIR / f"{run_id}_signals.json"
    runs: list = []
    if sig_path.exists():
        try:
            runs = json.loads(sig_path.read_text(encoding="utf-8"))
        except Exception:
            runs = []
    runs.append({
        "run_at":  datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "signals": result.signals,
    })
    sig_path.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- orders ---
    ord_path = LIVE_LOG_DIR / f"{run_id}_orders.json"
    runs_ord: list = []
    if ord_path.exists():
        try:
            runs_ord = json.loads(ord_path.read_text(encoding="utf-8"))
        except Exception:
            runs_ord = []
    runs_ord.append({
        "run_at":       datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "orders":       result.orders,
        "send_results": send_results,
    })
    ord_path.write_text(json.dumps(runs_ord, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("ライブログ保存: %s / %s", sig_path, ord_path)

# G29_UNIVERSE は廃止。load_universe() で configs/universe/*.json から読み込む。
# 後方互換用として参照のみ残す（直接使用禁止）
_LEGACY_G29_UNIVERSE_REMOVED = True

FUJIKO_PARAMS = dict(
    min_sepa         = 6,
    min_rsr          = 70.0,   # V2: 75.0 → 70.0 に緩和（Calmar改善確認済み）
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 10,
    use_turtle_entry = True,
)

CAPITAL      = 2_000_000
MAX_POS      = 3
MIN_SECTORS  = 1   # セクター制約なし（D_IDM_sec1 設定）
MAX_DD_LIMIT = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="フジコ法 最適化済み朝のシグナル生成スクリプト")
    p.add_argument("--live",   action="store_true", help="kabuステーション API に実際に発注する")
    p.add_argument("--yes","-y", action="store_true", help="発注確認プロンプトをスキップ")
    p.add_argument("--no-save", action="store_true", help="JSON シグナルファイルを保存しない")
    p.add_argument("--output-dir", default="data/signals", help="シグナルJSONの保存先")
    return p.parse_args()


def print_banner(live: bool, universe: dict[str, str], universe_meta: dict) -> None:
    mode    = "LIVE（実発注）" if live else "DRY RUN（発注なし）"
    version = universe_meta.get("version", "unknown")
    created = universe_meta.get("created_at", "?")
    print("=" * 64)
    print("  フジコ法シグナル確認スクリプト")
    print(f"  実行日時       : {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"  モード         : {mode}")
    print(f"  ユニバース     : {len(universe)}銘柄 (v={version}, created={created})")
    print(f"  ポートフォリオ : max_pos={MAX_POS} / min_sectors={MIN_SECTORS} / min_rsr=70")
    print(f"  安全設計       : MAX_DAILY={MAX_DAILY_ORDERS} / MAX_PER_SYM={MAX_SYMBOL_ORDERS} / MAX_OPEN={MAX_OPEN_POSITIONS}")
    print("=" * 64)


def print_signals(result) -> None:
    orders = result.orders
    if not orders:
        print("\n📭 本日の注文なし（全銘柄 HOLD / 条件不成立）")
    else:
        print(f"\n📋 発注予定: {len(orders)} 件")
        print(f"  {'銘柄':<10} {'売買':<6} {'数量':>6} {'参考価格':>10} {'参考金額':>12}  理由")
        print("  " + "-" * 74)
        for o in orders:
            side_str = "🟢 BUY " if o["side"] == "BUY" else "🔴 SELL"
            print(
                f"  {o['symbol']:<10} {side_str} {o['qty']:>6}株 "
                f"¥{o['estimated_price']:>9,.0f} "
                f"¥{o['estimated_amount']:>11,.0f}  "
                f"{o['reason'][:40]}"
            )

    if result.warnings:
        print("\n⚠ 警告:")
        for w in result.warnings:
            print(f"  - {w}")

    # BUYシグナル銘柄一覧
    buy_sigs  = [s for s in result.signals if s["signal"] == 1]
    sell_sigs = [s for s in result.signals if s["signal"] == -1]

    if buy_sigs:
        print(f"\n📊 BUYシグナル銘柄 ({len(buy_sigs)}件):")
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  理由")
        print("  " + "-" * 68)
        for s in sorted(buy_sigs, key=lambda x: x["rsr"], reverse=True):
            strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
                f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  "
                f"{s['reason'][:30]}"
            )

    if sell_sigs:
        print(f"\n📊 SELLシグナル銘柄 ({len(sell_sigs)}件):")
        for s in sell_sigs:
            print(f"  {s['symbol']} ({s['sector']}) — {s['reason'][:50]}")

    # RSRトップ10
    print(f"\n📊 RSRランキング上位10銘柄:")
    print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  シグナル")
    print("  " + "-" * 66)
    for s in sorted(result.signals, key=lambda x: x["rsr"], reverse=True)[:10]:
        sig_str   = "✅ BUY" if s["signal"] == 1 else ("🔴 SELL" if s["signal"] == -1 else "  -  ")
        strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
        print(
            f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
            f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  {sig_str}"
        )


def confirm_live_orders(orders: list) -> bool:
    buy_orders  = [o for o in orders if o["side"] == "BUY"]
    sell_orders = [o for o in orders if o["side"] == "SELL"]
    total_buy   = sum(o["estimated_amount"] for o in buy_orders)
    print("\n" + "=" * 64)
    print("  ⚠  実際の発注を行います。内容を確認してください。")
    print("=" * 64)
    print(f"  BUY  : {len(buy_orders)} 件  （推定合計: ¥{total_buy:,.0f}）")
    print(f"  SELL : {len(sell_orders)} 件")
    print("=" * 64)
    ans = input("  発注を実行しますか？ [y/N] > ").strip().lower()
    return ans == "y"


def save_signal_json(result, output_dir: str) -> Path:
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    file_path = dir_path / f"signal_{ts}.json"
    file_path.write_text(result.to_json(), encoding="utf-8")
    return file_path


def main() -> int:
    args = parse_args()

    # ---- ユニバースロード（起動時チェック） ----
    try:
        LIVE_UNIVERSE, universe_meta = load_universe()
    except RuntimeError as e:
        print(f"[FATAL] ユニバースロードエラー:\n  {e}", file=sys.stderr)
        return 1

    print_banner(args.live, LIVE_UNIVERSE, universe_meta)

    import warnings
    warnings.filterwarnings("ignore")

    # ---- 株価上限フィルター（シグナル生成前） ----
    # 保有中銘柄は API 接続前のため空集合で保守的に処理する。
    # 将来買った高額株が値上がりした場合への備えだが、
    # 高額株はこのフィルターで BUY 除外されるため事実上発生しない。
    logger.info("株価上限フィルター適用中（上限: ¥%s/単元）...", f"{MAX_ALLOCATION:,}")
    LIVE_UNIVERSE, price_skipped = filter_universe_by_price(
        LIVE_UNIVERSE, MAX_ALLOCATION, held_symbols=set()
    )
    if price_skipped:
        print(f"\n[価格フィルター] {len(price_skipped)}銘柄を除外（¥{MAX_ALLOCATION:,}/単元超）:")
        for sym, price, cost in price_skipped:
            print(f"  ✗ {sym:<8} ¥{price:>8,.0f}/株  1単元=¥{cost:>9,.0f}  > 上限¥{MAX_ALLOCATION:,}")
    logger.info("フィルター後ユニバース: %d銘柄", len(LIVE_UNIVERSE))

    from kabusapi.signal_bridge import SignalBridge

    bridge = SignalBridge(
        universe_tickers = LIVE_UNIVERSE,
        fujiko_params    = FUJIKO_PARAMS,
        capital          = CAPITAL,
        max_positions    = min(MAX_POS, MAX_OPEN_POSITIONS),  # ハードキャップ適用
        max_dd_limit     = MAX_DD_LIMIT,
        min_sectors      = MIN_SECTORS,
        live             = args.live,
    )

    logger.info("シグナル生成開始...")
    result, order_objects = bridge.run()

    print(f"\n  データ基準日 : {result.data_as_of}")
    print(f"  ユニバース   : {result.n_universe} 銘柄")
    print(
        f"  ポートフォリオ: 保有 {result.portfolio_summary['current_positions']} / "
        f"最大 {result.portfolio_summary['max_positions']} 銘柄  "
        f"空きスロット: {result.portfolio_summary['open_slots']}  "
        f"余力: ¥{result.portfolio_summary['available_cash']:,.0f}"
    )

    print_signals(result)

    if not args.no_save:
        saved_path = save_signal_json(result, args.output_dir)
        print(f"\n💾 シグナル保存: {saved_path}")

    if not args.live:
        print("\n" + "=" * 64)
        print("  ドライランのため発注は行いません。")
        print("  実際に発注するには --live オプションを付けて実行してください。")
        print("=" * 64)
        return 0

    if not order_objects:
        print("\n発注なし。終了します。")
        return 0

    run_id = datetime.now(JST).strftime("%Y%m%d")

    # ----------------------------------------------------------------
    # 発注前安全チェック（3層）
    # 1. オーダーロック（銘柄単位の当日重複チェック）
    # 2. MAX_SYMBOL_ORDERS / MAX_DAILY_ORDERS（数量上限）
    # ----------------------------------------------------------------
    today_total, today_per_sym = _count_today_orders(args.output_dir)
    blocked = []
    filtered_orders = []

    for o in order_objects:
        sym = o.symbol
        # 層1: ロックファイルチェック（スクリプト再起動による二重発注を防ぐ）
        if already_ordered_today(sym):
            blocked.append(f"{sym}: 当日発注済み（ロックファイルで確認）")
            continue
        # 層2: 件数上限チェック
        sym_count = today_per_sym.get(sym, 0)
        if today_total + len(filtered_orders) >= MAX_DAILY_ORDERS:
            blocked.append(f"{sym}: 本日の発注上限({MAX_DAILY_ORDERS}件)に到達")
        elif sym_count >= MAX_SYMBOL_ORDERS:
            blocked.append(
                f"{sym}: 銘柄別上限({MAX_SYMBOL_ORDERS}件/日)に到達（本日既に{sym_count}件）"
            )
        else:
            filtered_orders.append(o)

    if blocked:
        print("\n[安全設計] 以下の注文は除外されました:")
        for msg in blocked:
            print(f"  ⚠ {msg}")
        if not filtered_orders:
            print("発注可能な注文がありません。終了します。")
            # シグナルログは発注なしでも保存
            if not args.no_save:
                save_live_logs(run_id, result, [])
            return 0
        order_objects = filtered_orders

    if today_total > 0:
        print(f"\n[安全設計] 本日の発注履歴: 合計{today_total}件 / 上限{MAX_DAILY_ORDERS}件")

    order_dicts = result.orders
    if not args.yes:
        confirmed = confirm_live_orders(order_dicts)
        if not confirmed:
            print("発注をキャンセルしました。")
            return 0

    # ── 取引時間チェック: 09:00未満なら待機 ──────────────────────────
    import time
    now = datetime.now(JST)
    market_open = now.replace(hour=9, minute=0, second=5, microsecond=0)
    if now < market_open:
        wait_sec = (market_open - now).total_seconds()
        print(f"\n⏰ 取引開始まで {wait_sec:.0f}秒待機中... (09:00:05 発注予定)")
        time.sleep(wait_sec)
        print("  → 待機完了。発注を開始します。")
    # ──────────────────────────────────────────────────────────────────

    # ----------------------------------------------------------------
    # 発注実行（リトライ付き）
    # ----------------------------------------------------------------
    print(f"\n発注中（最大{MAX_RETRY}回リトライ）...")
    try:
        send_results = _send_orders_with_retry(bridge, order_objects)
    except RuntimeError as e:
        logger.error("発注中断: %s", e)
        print(f"\n[FATAL] 発注失敗: {e}")
        if not args.no_save:
            save_live_logs(run_id, result, [{"error": str(e)}])
        return 1

    print("\n=== 発注結果 ===")
    for r in send_results:
        status   = "✅ 成功" if r.get("success") else "❌ 失敗"
        order_id = r.get("order_id", r.get("error", ""))
        print(f"  {r['side']} {r['symbol']} {r['qty']}株 → {status}  ({order_id})")

    # ----------------------------------------------------------------
    # ログ保存（data/signals/ + logs/live/）
    # ----------------------------------------------------------------
    result_dict = json.loads(result.to_json())
    result_dict["send_results"] = send_results
    if not args.no_save:
        # 既存: data/signals/*_executed.json（audit trail）
        saved_path = (
            Path(args.output_dir)
            / f"signal_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}_executed.json"
        )
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n💾 発注結果保存: {saved_path}")
        # 新規: logs/live/YYYYMMDD_*.json（分析用構造化ログ）
        save_live_logs(run_id, result, send_results)
        print(f"📋 ライブログ: logs/live/{run_id}_orders.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
