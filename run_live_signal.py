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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

# .env 読み込み + パス定数 + ライブ安全設定（paths.py が一括管理）
from paths import (
    SIGNALS_DIR, ORDER_LOCK_FILE, LIVE_LOG_DIR,
    PHASE2_METRICS_FILE, RSR_UNIVERSE_FILE,
    LIVE_UNIVERSE_FILE, SHADOW_UNIVERSE_FILE,
    LIVE_MODE, MAX_ORDERS_PER_DAY, KABUS_PORT,
    assert_live_ready, assert_execution_context,
    assert_kabus_connection, verify_dataset_integrity,
    acquire_runtime_lock, release_runtime_lock,
    enforce_order_rate_limit, record_order_sent,
)

# ── 実行コンテキスト検証（最優先: モジュール読み込み直後）──────────────────
# research / backtest スクリプトが LIVE_MODE=true のまま呼ばれた場合にブロック
assert_execution_context()

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("live_signal")

# ------------------------------------------------------------------ #
# 安全設計定数（取引所の過剰発注監視対策）
# MAX_DAILY_ORDERS は paths.py の MAX_ORDERS_PER_DAY（.env で制御）
# ------------------------------------------------------------------ #
MAX_DAILY_ORDERS   = MAX_ORDERS_PER_DAY  # .env の MAX_ORDERS_PER_DAY= で変更可（デフォルト20）
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
    file_path = LIVE_UNIVERSE_FILE
    if not file_path.exists():
        raise RuntimeError(
            f"ユニバースファイルが見つかりません: {file_path}\n"
            "LIVE_UNIVERSE_FILE 環境変数または configs/universe/ を確認してください。"
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
# 資本連動パラメータ導出（固定値廃止・資本比率化）
# ------------------------------------------------------------------ #
def derive_risk_params(capital: int) -> dict:
    """
    総資本からポジション制約を比率で導出する。
    固定値（MAX_ALLOCATION=60万など）を廃止し、資本増加が自動的に
    ユニバース価格上限・ポジションサイズに反映されるようにする。

    Args:
        capital: 総資本（円）

    Returns:
        risk_per_trade  : 1トレード最大リスク（capital × 1%）
        max_position    : 1銘柄最大配分（capital × 20%）
        leader_slot     : リーダースロット配分（capital × 35%）
        max_allocation  : ユニバース価格上限 = 1単元コスト上限（capital × 30%）

    資本別の挙動:
        200万 → max_allocation ¥600,000  (¥6,000/株まで)
        350万 → max_allocation ¥1,050,000 (¥10,500/株まで)
        480万 → max_allocation ¥1,440,000 (¥14,400/株まで ≈ 川崎重工クラス)
        500万 → max_allocation ¥1,500,000 (¥15,000/株まで)
        800万 → max_allocation ¥2,400,000 (¥24,000/株まで ≈ アドバンテストクラス)
    """
    return {
        "risk_per_trade":  capital * 0.01,
        "max_position":    capital * 0.20,
        "leader_slot":     capital * 0.35,
        "max_allocation":  capital * 0.30,
    }


def recommended_universe_size(capital: int) -> int:
    """
    資本に応じた推奨観測ユニバースサイズ（advisory / 自動変更しない）。
    実際の変更は configs/universe/*.json の手動更新 + ユーザー確認が必要。
    """
    if capital < 3_000_000:
        return 42
    elif capital < 5_000_000:
        return 50
    elif capital < 8_000_000:
        return 60
    else:
        return 80


# ------------------------------------------------------------------ #
# 株価上限フィルター（シグナル生成前に適用）
# ------------------------------------------------------------------ #
LOT_SIZE = 100   # 東証標準単元株数

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
# 1. 二重発注防止 — オーダーロックファイル（paths.py からインポート済み）
# ------------------------------------------------------------------ #

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
# 3. ライブ運用監視ログ（LIVE_LOG_DIR / PHASE2_METRICS_FILE は paths.py からインポート済み）
# ------------------------------------------------------------------ #

def log_phase2_metrics(result) -> None:
    """
    Phase2 日次運用メトリクスを logs/phase2_live_metrics.jsonl に追記する。

    フィールド:
        date         : 実行日（JST）
        equity       : 現在の推定資産総額（円）
        drawdown     : 現在のドローダウン（0.0〜-1.0）
        positions    : 保有銘柄リスト
        breadth_50   : RSR>=50 比率（市場参加の広がり）
        breadth_75   : RSR>=75 比率（エントリー閾値以上の銘柄割合）
        signal_count : BUY シグナル数（当日）

    breadth_50 - breadth_75 はトレンド成熟度の代理変数として使用できる。
    差が小さい（全体が高RSR）= 成熟相場。差が大きい = 局所的な強さ。
    """
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    pf = result.portfolio_summary

    holding = [s["symbol"] for s in result.signals if s.get("currently_holding")]
    rsr_vals = [s["rsr"] for s in result.signals]
    n = len(rsr_vals)
    breadth_50 = sum(1 for r in rsr_vals if r >= 50) / n if n else 0.0
    breadth_75 = sum(1 for r in rsr_vals if r >= 75) / n if n else 0.0
    signal_count = sum(1 for s in result.signals if s.get("signal") == 1)

    entry = {
        "date":         today_str,
        "equity":       round(pf.get("current_equity", 0), 0),
        "drawdown":     round(pf.get("current_drawdown", 0.0), 4),
        "positions":    holding,
        "breadth_50":   round(breadth_50, 3),
        "breadth_75":   round(breadth_75, 3),
        "signal_count": signal_count,
    }

    PHASE2_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PHASE2_METRICS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Phase2メトリクス保存: %s", PHASE2_METRICS_FILE)


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
    min_rsr          = 75.0,   # research 確定値（filter-first アーキテクチャ）
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 55,   # 2026-03-31: exit感度テストで20→55に変更
    use_turtle_entry = True,
)
logger.info(
    "戦略パラメータ: turtle_entry=%d exit_lookback=%d min_sepa=%d",
    FUJIKO_PARAMS["turtle_entry"],
    FUJIKO_PARAMS["turtle_exit"],
    FUJIKO_PARAMS["min_sepa"],
)

CAPITAL              = int(os.environ.get("CAPITAL", 2_000_000))  # .env の CAPITAL= で上書き可
_RISK_PARAMS         = derive_risk_params(CAPITAL)               # 資本連動パラメータ（全体で一貫して使用）
MAX_SINGLE_WEIGHT    = 0.30     # 1銘柄最大ウェイト（capital * 30% = ¥600,000/200万）
# 変更履歴: 0.20 → 0.30 (2026-03-25)
# 理由: RSR上位銘柄（高時価総額）が単元価格¥400,000超で構造的排除されていた。
#       日本株RSR戦略では equity×30%（単元許可型）が最も安定。
MAX_POS              = 3        # top_k と一致させる（確定設計 2026-03-23）
MIN_SECTORS          = 1        # セクター制約なし
MAX_DD_LIMIT         = 0.15
TOP_K                = 3        # RSR 上位 k 銘柄のみ BUY 対象（確定設計 2026-03-23）
MAX_HOLD_DAYS        = 60       # 最大保有営業日数（OOS MaxDD -13.98% 確認済み）
MIN_HOLD_DAYS        = int(os.environ.get("MIN_HOLD_DAYS", 5))          # 最低保有日数（RSR exit抑制 / .envで上書き可）
EMERGENCY_EXIT_PCT   = float(os.environ.get("EMERGENCY_EXIT_PCT", -0.08)) # 緊急 exit 閾値（-8%でmin_hold無視）
MAX_NEW_POS_PER_DAY  = 1        # 1回の実行で生成する新規 BUY 上限（過剰発注防止・初期運用安定化）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="フジコ法 最適化済み朝のシグナル生成スクリプト")
    p.add_argument("--live",   action="store_true", help="kabuステーション API に実際に発注する")
    p.add_argument("--yes","-y", action="store_true", help="発注確認プロンプトをスキップ")
    p.add_argument("--no-save", action="store_true", help="JSON シグナルファイルを保存しない")
    p.add_argument("--output-dir", default=str(SIGNALS_DIR), help="シグナルJSONの保存先")
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
    print(f"  ポートフォリオ : max_pos={MAX_POS} / top_k={TOP_K} / max_hold={MAX_HOLD_DAYS}d"
          f" / min_hold={MIN_HOLD_DAYS}d / emg_exit={EMERGENCY_EXIT_PCT:.0%}")
    print(f"  安全設計       : MAX_DAILY={MAX_DAILY_ORDERS} / MAX_PER_SYM={MAX_SYMBOL_ORDERS} / MAX_OPEN={MAX_OPEN_POSITIONS}")
    _rp = _RISK_PARAMS
    _rec_uni = recommended_universe_size(CAPITAL)
    print(f"  資本設定       : ¥{CAPITAL:,}  max_alloc=¥{int(_rp['max_allocation']):,}(×30%)  max_pos=¥{int(_rp['max_position']):,}(×20%)")
    if _rec_uni > len(universe):
        print(f"  [推奨] ユニバース拡張: 現在{len(universe)}銘柄 → 推奨{_rec_uni}銘柄（資本¥{CAPITAL:,}に対応）")
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
            if o["side"] == "BUY":
                side_str = "🟢 BUY  "
            elif o["side"] == "SHADOW_BUY":
                side_str = "🔵 SHDW "
            else:
                side_str = "🔴 SELL "
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
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'rank':>5} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  理由")
        print("  " + "-" * 74)
        for s in sorted(buy_sigs, key=lambda x: x.get("rsr_rank", 99)):
            strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
                f"{s.get('rsr_rank', 0):>5} {s['rsr']:>6.1f} {s['sepa_score']:>5}"
                f" {s['rsr_momentum']:>+7.2f}  {s['reason'][:30]}"
            )

    if sell_sigs:
        print(f"\n📊 SELLシグナル銘柄 ({len(sell_sigs)}件):")
        for s in sell_sigs:
            print(f"  {s['symbol']} ({s['sector']}) — {s['reason'][:50]}")

    # RSR ランキング上位10銘柄
    print(f"\n📊 RSR ランキング上位10銘柄:")
    print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'rank':>5} {'RSR':>6} {'SEPA':>5} {'Mom':>7} {'保有日':>5}  シグナル")
    print("  " + "-" * 78)
    for s in sorted(result.signals, key=lambda x: x.get("rsr_rank", 99))[:10]:
        sig_str   = "✅ BUY" if s["signal"] == 1 else ("🔴 SELL" if s["signal"] == -1 else "  -  ")
        strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
        hold_str  = f"{s.get('hold_days', 0):>5}d" if s.get("currently_holding") else "     -"
        print(
            f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
            f"{s.get('rsr_rank', 0):>5} {s['rsr']:>6.1f} {s['sepa_score']:>5}"
            f" {s['rsr_momentum']:>+7.2f} {hold_str}  {sig_str}"
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

    # ── Step2: 二重起動防止（最初に取得・atexit で自動解放）──────────────────
    if args.live:
        try:
            acquire_runtime_lock()
        except RuntimeError as _le:
            print(f"[FATAL] {_le}", file=sys.stderr)
            return 1

    # ── LIVE_MODE 二重ガード（paths.py 経由）────────────────────────────────
    # --live フラグ AND .env の LIVE_MODE=true の両方が必要。
    if args.live:
        try:
            assert_live_ready()          # LIVE_MODE / ファイル存在 / 上限値を一括チェック
            assert_kabus_connection()    # kabuStation API 疎通確認（未起動ならここで止まる）
        except RuntimeError as _e:
            release_runtime_lock()
            print(f"[FATAL] {_e}", file=sys.stderr)
            return 1

    # ── データ整合性チェック ──────────────────────────────────────────────────
    try:
        verify_dataset_integrity(os.environ.get("DATA_VERSION", ""))
    except RuntimeError as _de:
        logger.warning("データ整合性チェック警告（ライブは継続）: %s", _de)

    # ── ユニバースロード（起動時チェック）────────────────────────────────────
    try:
        LIVE_UNIVERSE, universe_meta = load_universe()
    except RuntimeError as e:
        print(f"[FATAL] ユニバースロードエラー:\n  {e}", file=sys.stderr)
        return 1

    print_banner(args.live, LIVE_UNIVERSE, universe_meta)

    # ── Step3: 本番モード明示ログ ──────────────────────────────────────────
    if args.live:
        logger.warning(
            "★ LIVE TRADING ENABLED"
            " | max_orders=%d/day | cooldown=%ss | kabu_port=%s"
            " | universe=%d銘柄 | universe_ver=%s | dataset=%s",
            MAX_ORDERS_PER_DAY,
            os.environ.get("ORDER_COOLDOWN_SECONDS", "5"),
            KABUS_PORT,
            len(LIVE_UNIVERSE),
            universe_meta.get("version", "unknown"),
            os.environ.get("DATA_VERSION", "live_yfinance"),
        )

    import warnings
    warnings.filterwarnings("ignore")

    # ---- RSRユニバース（62銘柄コンテキスト: 42 live + 20 shadow）を設定 ----
    # research / live / backtest で同一の母集団を使い RSR percentile を統一する。
    # RSR は cross-sectional factor なので母集団が変わると別指標になる。
    # 2026-03-24 バックテスト検証で RSR62 の Calmar +28% / Sharpe +9% を確認 → 採用。
    # RSR_UNIVERSE_FILE は paths.py / 環境変数で管理（絶対パス保証済み）
    import pandas as _pd_rsr
    _rsr_df = _pd_rsr.read_csv(RSR_UNIVERSE_FILE)
    RSR_UNIVERSE: dict[str, str] = {
        row["symbol"]: row.get("sector", "不明")
        for _, row in _rsr_df.iterrows()
    }

    logger.info(
        "RSR context size=%d trade_universe=%d",
        len(RSR_UNIVERSE),
        len(LIVE_UNIVERSE),
    )

    # ---- 株価上限フィルター（資本連動 = capital × 30%）----
    # 保有中銘柄は API 接続前のため空集合で保守的に処理する。
    _max_alloc = int(_RISK_PARAMS["max_allocation"])
    logger.info(
        "株価上限フィルター適用中（上限: ¥%s/単元 = capital×30%%）", f"{_max_alloc:,}"
    )
    LIVE_UNIVERSE, price_skipped = filter_universe_by_price(
        LIVE_UNIVERSE, _max_alloc, held_symbols=set()
    )
    if price_skipped:
        print(f"\n[価格フィルター] {len(price_skipped)}銘柄を除外（¥{_max_alloc:,}/単元超 = capital×30%）:")
        for sym, price, cost in price_skipped:
            print(f"  ✗ {sym:<8} ¥{price:>8,.0f}/株  1単元=¥{cost:>9,.0f}  > 上限¥{_max_alloc:,}")
    logger.info(
        "RSRユニバース: %d銘柄（TOPIX100固定） / 売買ユニバース: %d銘柄（価格フィルター後）",
        len(RSR_UNIVERSE), len(LIVE_UNIVERSE),
    )

    # ---- Shadow Universe（監視専用・RSR42母集団は変更しない）----
    # SHADOW_UNIVERSE_FILE 未設定でも動作する（省略可能）。
    # SHADOW_UNIVERSE_FILE は paths.py / 環境変数で管理（絶対パス保証済み）
    SHADOW_UNIVERSE: dict[str, str] = {}
    try:
        _shadow_path = SHADOW_UNIVERSE_FILE
        if _shadow_path.exists():
            _shadow_data = json.loads(_shadow_path.read_text(encoding="utf-8"))
            SHADOW_UNIVERSE = _shadow_data.get("symbols", {})
            logger.info("Shadow Universe: %d銘柄（監視専用）", len(SHADOW_UNIVERSE))
    except Exception as _se:
        logger.warning("Shadow Universe読み込みスキップ: %s", _se)

    # RSR62コンテキスト = 42 live + shadow（重複なし）
    RSR_UNIVERSE_62: dict[str, str] = {**RSR_UNIVERSE, **SHADOW_UNIVERSE}
    logger.info(
        "RSRコンテキスト: %d銘柄（42 live + %d shadow）/ 売買ユニバース: %d銘柄",
        len(RSR_UNIVERSE_62), len(SHADOW_UNIVERSE), len(LIVE_UNIVERSE),
    )

    from kabusapi.signal_bridge import SignalBridge

    bridge = SignalBridge(
        universe_tickers          = LIVE_UNIVERSE,
        rsr_universe_tickers      = RSR_UNIVERSE_62,  # RSR62コンテキスト（42+shadow20）
        shadow_universe_tickers   = SHADOW_UNIVERSE or None,  # 監視用（RSR計算兼用）
        fujiko_params             = FUJIKO_PARAMS,
        capital                   = CAPITAL,
        max_positions             = min(MAX_POS, MAX_OPEN_POSITIONS),
        max_single_weight         = MAX_SINGLE_WEIGHT,
        max_dd_limit              = MAX_DD_LIMIT,
        min_sectors               = MIN_SECTORS,
        live                      = args.live,
        top_k                     = TOP_K,
        max_hold_days             = MAX_HOLD_DAYS,
        min_hold_days             = MIN_HOLD_DAYS,
        emergency_exit_pct        = EMERGENCY_EXIT_PCT,
        max_new_positions_per_day = MAX_NEW_POS_PER_DAY,
    )

    logger.info("シグナル生成開始...")
    result, order_objects = bridge.run()

    print(f"\n  データ基準日 : {result.data_as_of}")
    print(f"  ユニバース   : {result.n_universe} 銘柄")
    ps = result.portfolio_summary
    cb_str = ps.get("cb_state", "NORMAL")
    if cb_str != "NORMAL":
        cooldown = ps.get("cb_cooldown_end") or ""
        print(f"  ⚠ CB状態    : {cb_str}（クールダウン終了: {cooldown}）")
    dd_pct = ps.get("current_drawdown", 0.0) * 100
    _eq_max = ps.get("equity_based_max_pos", ps["max_positions"])
    _eq_note = f" [資本連動→{_eq_max}]" if _eq_max != ps["max_positions"] else ""
    print(
        f"  ポートフォリオ: 保有 {ps['current_positions']} / "
        f"最大 {ps['max_positions']} 銘柄{_eq_note}  "
        f"空きスロット: {ps['open_slots']}  "
        f"余力: ¥{ps['available_cash']:,.0f}  "
        f"DD: {dd_pct:+.1f}%"
    )
    print(f"  Top-{TOP_K}銘柄  : {', '.join(result.top_k_symbols)}")

    print_signals(result)

    if not args.no_save:
        saved_path = save_signal_json(result, args.output_dir)
        print(f"\n💾 シグナル保存: {saved_path}")

    # Phase2 日次メトリクス記録（DRY_RUN / LIVE 共通）
    try:
        log_phase2_metrics(result)
    except Exception as _e:
        logger.warning("Phase2メトリクス記録失敗（無視）: %s", _e)

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

    # ── ギャップダウン追加チェック（9:00 直後・board 価格で再評価）────
    # 前日終値では検出できなかったギャップダウンを寄り付き直前の板で検出し、
    # 必要なら SELL 注文を追加する。API 未接続は自動スキップ。
    logger.info("ギャップダウンチェック中（board 取得）...")
    try:
        order_objects = bridge.check_gap_stops(order_objects)
    except Exception as _gap_e:
        logger.warning("ギャップダウンチェック失敗（スキップ）: %s", _gap_e)
    # ──────────────────────────────────────────────────────────────────

    # ----------------------------------------------------------------
    # 発注実行（リトライ付き）
    # ----------------------------------------------------------------
    print(f"\n発注中（最大{MAX_RETRY}回リトライ）...")
    try:
        enforce_order_rate_limit()          # レートリミット確認（kabu API BAN防止）
        send_results = _send_orders_with_retry(bridge, order_objects)
        record_order_sent()                 # タイムスタンプ記録
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

    # ── ポートフォリオ状態更新（約定確認後） ────────────────────────
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    bridge.update_state_after_execution(send_results, today_str)

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
