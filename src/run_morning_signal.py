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

import copy
import os
import sys
import argparse
import logging
import json
import pickle
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))           # C:/ai-trading/src/ → backtest.xxx imports
sys.path.insert(0, str(_here.parent))    # C:/ai-trading/     → src.xxx imports
sys.stdout.reconfigure(encoding="utf-8")

from src.config_loader import load_strategy_config
from src.paths import SIGNALS_DIR, LIVE_UNIVERSE_FILE, ORDER_LOCK_FILE, RUNTIME_DIR, LOGS_DIR
from src.utils.morning_smoke_test import run_morning_smoke_test
from src.portfolio.capital_efficiency import CapitalEfficiencyModule
from src.portfolio.ce_compare_logger import CECompareLogger

JST = timezone(timedelta(hours=9))

_CE_STATE_FILE = RUNTIME_DIR / "ce_state.pkl"
_ce_logger     = CECompareLogger(runtime_dir=RUNTIME_DIR, logs_dir=LOGS_DIR)


def _load_ce() -> CapitalEfficiencyModule:
    """Load persisted CE state from disk, or create a fresh instance."""
    if _CE_STATE_FILE.exists():
        try:
            with open(_CE_STATE_FILE, "rb") as f:
                ce = pickle.load(f)
            if isinstance(ce, CapitalEfficiencyModule):
                return ce
        except Exception as exc:
            logging.getLogger("morning_signal").warning("CE state load failed, fresh instance: %s", exc)
    return CapitalEfficiencyModule()


def _save_ce(ce: CapitalEfficiencyModule) -> None:
    """Persist CE state to disk for next run."""
    try:
        _CE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CE_STATE_FILE, "wb") as f:
            pickle.dump(ce, f)
    except Exception as exc:
        logging.getLogger("morning_signal").warning("CE state save failed: %s", exc)


def _apply_ce_to_orders(
    order_objects: list,
    result,
    capital: float,
    ce: CapitalEfficiencyModule,
    date_idx: int,
) -> tuple[list, dict[str, dict]]:
    """
    Apply Capital Efficiency adjustments to BUY orders.

    Returns
    -------
    (adjusted_orders, ce_meta)
      ce_meta: {symbol: {ea, confidence, sample_size, regime_state,
                         size_scale, base_qty, new_qty,
                         base_weight, adjusted_weight}}

    Adjustments applied to each BUY:
      adjusted_weight = base_weight * (1 + ea)
      new_qty         = old_qty * clip(1 + ea * 5, 0.3, 1.0)
    """
    _ce_log    = logging.getLogger("morning_signal")
    sig_by_sym = {s["symbol"]: s for s in result.signals}
    adjusted:  list     = []
    ce_meta:   dict     = {}

    for order in order_objects:
        if order.side != "BUY":
            adjusted.append(order)
            continue

        sig       = sig_by_sym.get(order.symbol, {})
        score     = float(sig.get("rsr", 75.0))
        gap_pct   = 0.0   # morning signal: open price unknown
        fill_prob = 0.9   # market-open order

        ea, confidence, regime_state = ce.estimate_expected_alpha(
            date_idx, score, gap_pct, fill_prob
        )
        sample_size     = len(ce._buf)
        size_scale      = float(np.clip(1.0 + ea * 5.0, 0.3, 1.0))
        base_qty        = order.qty
        base_weight     = (base_qty * order.estimated_price) / max(1.0, capital)
        adjusted_weight = base_weight * (1.0 + ea)

        new_qty_raw = int(base_qty * size_scale)
        new_qty     = max(100, (new_qty_raw // 100) * 100)

        _ce_log.info(
            "[CE] %s | expected_alpha=%.4f confidence=%.2f sample_size=%d regime=%s"
            " | base_weight=%.4f adjusted_weight=%.4f"
            " | qty_before=%d qty_after=%d size_scale=%.3f",
            order.symbol, ea, confidence, sample_size, regime_state,
            base_weight, adjusted_weight, base_qty, new_qty, size_scale,
        )
        print(
            f"  [CE] {order.symbol}: ea={ea:.4f} conf={confidence:.2f} n={sample_size}"
            f" regime={regime_state} | base_w={base_weight:.4f} adj_w={adjusted_weight:.4f}"
            f" | qty {base_qty}→{new_qty} (scale={size_scale:.3f})"
        )

        order.qty              = new_qty
        order.estimated_amount = new_qty * order.estimated_price
        adjusted.append(order)

        ce_meta[order.symbol] = {
            "ea":              ea,
            "confidence":      confidence,
            "sample_size":     sample_size,
            "regime_state":    regime_state,
            "size_scale":      size_scale,
            "base_qty":        base_qty,
            "new_qty":         new_qty,
            "base_weight":     base_weight,
            "adjusted_weight": adjusted_weight,
        }

    return adjusted, ce_meta


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
    p.add_argument(
        "--reconcile",
        action="store_true",
        help="不確定発注（pending/confirmed 不一致）をブローカーと照合して診断する（発注なし）",
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

    # 保有継続銘柄（signal==1 だが holding_excluded = 新規発注対象外）
    held_signals = [s for s in result.signals if s.get("currently_holding")]

    if not orders:
        if held_signals:
            held_syms = ", ".join(s["symbol"] for s in held_signals)
            print(f"\n📭 新規注文なし（保有継続: {held_syms}）")
        else:
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

    # 保有継続セクション（ランキング前に表示）
    if held_signals:
        print(f"\n📌 保有継続")
        for s in held_signals:
            ep  = s.get("entry_price", 0)
            stp = s.get("trailing_stop", 0)
            pnl = s.get("unrealized_pnl_pct", 0.0)
            pnl_str = f"{pnl:+.1%}" if ep > 0 else "---"
            ep_str  = f"{ep:,.0f}" if ep > 0 else "---"
            stp_str = f"{stp:,.0f}" if stp > 0 else "---"
            print(f"    {s['symbol']:<10}  avg={ep_str}  last_stop={stp_str}  pnl={pnl_str}")

    # RSR トップ10 をサマリー表示
    top_signals = [s for s in result.signals if s["signal"] in (1, 0)][:10]
    if top_signals:
        print(f"\n📊 RSR ランキング（ユニバース上位10銘柄）")
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<6} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  シグナル")
        print("  " + "-" * 66)
        for s in top_signals:
            if s["signal"] == 1 and s.get("currently_holding"):
                sig_str = "📌 HOLD"   # 保有継続: 新規BUY対象外
            elif s["signal"] == 1:
                sig_str = "✅ BUY "   # 新規BUY候補
            else:
                sig_str = "  -   "
            strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<6} "
                f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  {sig_str}"
            )
    # 平均回帰 BUY シグナルを別枠で表示（新規のみ）
    mr_buys = [
        s for s in result.signals
        if s["signal"] == 1
        and s.get("strategy_type") == "mean_rev"
        and not s.get("currently_holding")
    ]
    if mr_buys:
        print(f"\n📊 平均回帰 BUY シグナル（{len(mr_buys)}件）")
        print(f"  {'銘柄':<10} {'セクター':<10}  理由")
        print("  " + "-" * 60)
        for s in mr_buys:
            print(f"  {s['symbol']:<10} {s['sector']:<10}  {s['reason']}")


def assert_live_mode_env() -> None:
    """LIVE_MODE 環境変数が 'true' でない限り --live フラグを無効化して終了する。"""
    from src.kabusapi.signal_bridge import AbortError

    live_mode_env = os.getenv("LIVE_MODE", "false").lower()
    if live_mode_env != "true":
        raise AbortError(
            "live_mode_disabled",
            "LIVE_MODE guard blocked --live execution. "
            f"LIVE_MODE={live_mode_env!r}",
        )


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


def _fetch_broker_position_qty() -> dict[str, int]:
    """broker の現在保有数量を {symbol: qty} で返す。"""
    from src.kabusapi.client import KabuClient

    from src.common.position_normalizer import filter_live_positions
    client = KabuClient()
    client.fetch_token()
    raw_positions = client.get_positions()
    broker_qty: dict[str, int] = {}
    for pos in filter_live_positions(raw_positions):
        sym_code = str(pos.get("Symbol", "")).strip()
        if not sym_code:
            continue
        symbol = sym_code if sym_code.endswith(".T") else f"{sym_code}.T"
        qty = int(pos.get("LeavesQty", 0) or 0)
        broker_qty[symbol] = broker_qty.get(symbol, 0) + qty
    return broker_qty


def _apply_live_order_guards(
    order_objects: list,
    broker_qty_map: dict[str, int] | None = None,
) -> list:
    """
    live 発注直前ガード。
    - 当日 one-shot lock 済み銘柄はスキップ
    - SELL は broker 保有数量で再照合し、0株ならスキップ
    - SELL 数量は min(signal_qty, broker_qty) に丸める
    """
    guarded = []
    broker_qty_map = broker_qty_map or {}

    for order in order_objects:
        if already_ordered_today(order.symbol):
            logger.warning("ONE_SHOT_LOCK: %s %s をスキップ", order.side, order.symbol)
            print(f"  ⚠ {order.symbol}: 発注スキップ（one_shot_lock）")
            continue

        if order.side != "SELL":
            guarded.append(order)
            continue

        broker_qty = int(broker_qty_map.get(order.symbol, 0))
        if broker_qty <= 0:
            logger.warning("SELL_SKIP_NO_POSITION: %s broker_qty=0", order.symbol)
            print(f"  ⚠ {order.symbol}: 発注スキップ（broker_position=0）")
            continue

        signal_qty = int(getattr(order, "qty", 0) or 0)
        adjusted_qty = min(signal_qty, broker_qty)
        if adjusted_qty <= 0:
            logger.warning(
                "SELL_SKIP_INVALID_QTY: %s signal_qty=%s broker_qty=%s",
                order.symbol, signal_qty, broker_qty,
            )
            print(f"  ⚠ {order.symbol}: 発注スキップ（adjusted_qty=0）")
            continue
        if adjusted_qty != signal_qty:
            logger.warning(
                "SELL_QTY_ADJUSTED: %s signal_qty=%d broker_qty=%d adjusted=%d",
                order.symbol, signal_qty, broker_qty, adjusted_qty,
            )
            print(f"  ⚠ {order.symbol}: SELL数量を {signal_qty} → {adjusted_qty} に調整")
            order.qty = adjusted_qty

        guarded.append(order)

    return guarded


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

    from src.kabusapi.signal_bridge import AbortError

    PORTFOLIO_STATE_FILE = _Path(__file__).resolve().parent.parent / "runtime" / "portfolio_state.json"
    TODAY = datetime.now(JST).strftime("%Y-%m-%d")

    try:
        from src.kabusapi.client import KabuClient
        client = KabuClient()
        client.fetch_token()
        raw_positions = client.get_positions()
    except AbortError:
        raise
    except Exception as exc:
        logger.info("ポジション同期スキップ（API未接続）: %s", exc)
        return None

    # broker から保有銘柄を収集（.T サフィックス付きに正規化）
    from src.common.position_normalizer import filter_live_positions
    broker_positions: dict[str, float] = {}
    for p in filter_live_positions(raw_positions):
        sym_code = p.get("Symbol", "")
        if sym_code:
            sym = f"{sym_code}.T" if not sym_code.endswith(".T") else sym_code
            avg_price = float(p.get("Price", 0.0))
            qty = int(p.get("LeavesQty", 0) or 0)
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
    if not PORTFOLIO_STATE_FILE.exists():
        raise AbortError(
            "portfolio_state_missing",
            f"portfolio state file is missing: {PORTFOLIO_STATE_FILE}",
        )
    try:
        state = _json.loads(PORTFOLIO_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AbortError(
            "portfolio_state_missing",
            f"portfolio state file is unreadable: {PORTFOLIO_STATE_FILE}: {exc}",
        ) from exc

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
        from src.portfolio.state_store import save_portfolio_state as _sps
        _sps(state, path=PORTFOLIO_STATE_FILE, data_source="morning_sync")
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
    path = RUNTIME_DIR / "shadow_orders.jsonl"  # FIXED: was cwd-relative
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[SHADOW] Write failed: %s", e)


def _reconcile_uncertain_orders() -> int:
    """pending-without-confirmed な不確定発注をブローカーポジションと照合して診断する。"""
    print("=== RECONCILE MODE (診断のみ・発注なし) ===")
    trade_day = datetime.now(JST).date().isoformat()
    pending_file = RUNTIME_DIR / "send_idem_pending.json"
    confirm_file = RUNTIME_DIR / "send_idem_confirmed.json"

    def _load(path: Path) -> set:
        try:
            if path.exists():
                return set(json.loads(path.read_text(encoding="utf-8")).get(trade_day, []))
        except Exception as exc:
            print(f"  ERROR loading {path.name}: {exc}")
        return set()

    uncertain = _load(pending_file) - _load(confirm_file)
    if not uncertain:
        print("  不確定発注なし。")
        return 0

    print(f"  不確定発注: {len(uncertain)} 件")

    try:
        broker_qty = _fetch_broker_position_qty()
    except Exception as exc:
        print(f"  ERROR ブローカー照会失敗: {exc}")
        broker_qty = {}

    for key in sorted(uncertain):
        parts = key.split(":", 2)
        if len(parts) != 3:
            print(f"  MALFORMED_KEY: {key}")
            continue
        _, symbol, side = parts
        held_qty = broker_qty.get(symbol, 0)
        if side == "BUY":
            verdict = "LIKELY_FILLED" if held_qty > 0 else "LIKELY_NOT_FILLED"
        else:
            verdict = "LIKELY_FILLED" if held_qty == 0 else "LIKELY_NOT_FILLED"
        print(f"  {verdict}: {side} {symbol}  broker_qty={held_qty}")
        logger.critical(
            "UNCERTAIN_ORDER_RECONCILE: key=%s verdict=%s broker_qty=%s",
            key, verdict, held_qty,
        )

    return 0


def main() -> int:
    from src.kabusapi.signal_bridge import AbortError, SignalBridge

    current_positions: dict = {}
    pending_orders: list = []

    try:
        args = parse_args()
        if args.reconcile:
            return _reconcile_uncertain_orders()
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
            current_positions = positions or {}
            try:
                from src.kabusapi.client import KabuClient
                cash_info = KabuClient().get_wallet_cash()
            except AbortError:
                raise
            except Exception:
                cash_info = None
            if cash_info is None or positions is None:
                logger.error(
                    "Broker state unavailable (cash=%s, positions=%s) - halt execution",
                    cash_info,
                    positions,
                )
                raise RuntimeError("broker state unavailable")

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
        except AbortError:
            raise
        except Exception as exc:
            logger.error("SignalBridge 初期化失敗: %s", exc)
            return 1

        # ---- シグナル生成 ----
        logger.info("シグナル生成開始...")
        result, order_objects = bridge.run()
        pending_orders = order_objects

        # ---- Capital Efficiency (CE) 調整 + 比較ログ ----
        # kNN-based expected_alpha でポジションサイズを調整する。
        # バッファが min_samples(10) 未満の場合は ea=0 でスケール不変（安全フォールバック）。
        _ce          = _load_ce()
        _ce_date_idx = datetime.now(JST).toordinal()  # stable integer per calendar day
        _today       = datetime.now(JST).date()
        _today_str   = _today.isoformat()
        _ce.on_day_open(_ce_date_idx, market_return=0.0)  # open unknown at signal time

        # --- CE compare: forward-fill yesterday's pending records (no lookahead) ---
        print("\n[CE_COMPARE] 前日レコードの forward return 埋め込み...")
        _fwd_filled = _ce_logger.try_fill_forward_returns(_today)
        if _fwd_filled:
            logger.info("[CE_COMPARE] forward returns filled: %d record(s)", _fwd_filled)
            _ce_logger.flush_daily_csv(_today)
        else:
            logger.info("[CE_COMPARE] no pending records to fill today")

        # --- missed trade logging (スロット不足で除外された RSR通過銘柄) ---
        buy_syms = {o.symbol for o in order_objects if o.side == "BUY"}
        for sig in result.signals:
            if sig["signal"] == 1 and not sig.get("currently_holding") and sig["symbol"] not in buy_syms:
                _ce.log_missed_trade(
                    _ce_date_idx,
                    symbol=sig["symbol"],
                    score=float(sig.get("rsr", 75.0)),
                    gap_pct=0.0,
                    fill_prob=0.9,
                )
                logger.info("[CE] missed_trade logged: %s rsr=%.1f", sig["symbol"], sig.get("rsr", 0.0))

        # --- Dual order capture: save base (pre-CE) quantities ---
        _orders_base_by_sym = {
            o.symbol: copy.copy(o)
            for o in order_objects if o.side == "BUY"
        }

        # --- Apply CE ---
        print("\n[CE] Capital Efficiency 調整")
        order_objects, _ce_meta = _apply_ce_to_orders(
            order_objects, result, cfg.portfolio.capital, _ce, _ce_date_idx
        )
        _save_ce(_ce)
        logger.info("[CE] state saved: buffer_size=%d", len(_ce._buf))

        # --- CE compare: record trade-level comparison entries ---
        for sym, meta in _ce_meta.items():
            base_o = _orders_base_by_sym.get(sym)
            _ce_logger.record_order(
                date_str    = _today_str,
                symbol      = sym,
                side        = "BUY",
                qty_base    = base_o.qty if base_o else meta["base_qty"],
                qty_ce      = meta["new_qty"],
                fill_price  = base_o.estimated_price if base_o else 0.0,
                ea          = meta["ea"],
                confidence  = meta["confidence"],
                sample_size = meta["sample_size"],
            )

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

        _MAX_DAILY_ORDERS = int(os.environ.get("MAX_DAILY_ORDERS", 50))
        if ledger.daily_count() >= _MAX_DAILY_ORDERS:
            raise AbortError("over_order_limit", "daily order limit exceeded")

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
        sync_and_validate_state(bridge)

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

        # ---- Shadow mode: 実発注せずログのみ ----
        if SHADOW_MODE:
            print("\n[SHADOW MODE] 実発注をスキップしてログ記録のみ行います。")
            print(f"  shadow_mode_reason: {cfg.live_execution.shadow_mode_reason}")
            for o in order_objects:
                ledger.record_shadow(
                    o.symbol,
                    o.side,
                    qty=getattr(o, "qty", 0),
                    price=getattr(o, "estimated_price", 0.0),
                )
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

        sell_count = sum(1 for o in order_objects if o.side == "SELL")
        if sell_count > 0:
            broker_qty_map = _fetch_broker_position_qty()
            order_objects = _apply_live_order_guards(order_objects, broker_qty_map)
            if not order_objects:
                print("\n発注可能な注文なし（live guard 除外後）。終了します。")
                return 0

        # ---- idempotency チェック（OrderLedger による重複発注防止）----
        deduped = []
        for o in order_objects:
            ledger.check_and_record(
                o.symbol, o.side,
                qty=getattr(o, "qty", 0),
                price=getattr(o, "estimated_price", 0.0),
            )
            deduped.append(o)
        order_objects = deduped

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

        # 発注実行（partial execution guard）
        print("\n発注中...")
        # P1-4: idempotency ロックは OrderLedger.check_and_record() が発注前に記録済み。
        _ORDER_SEND_INTERVAL = float(os.environ.get("ORDER_SEND_INTERVAL", 0.2))
        _trade_day = today.isoformat()
        _IDEM_PENDING_FILE = RUNTIME_DIR / "send_idem_pending.json"
        _IDEM_CONFIRM_FILE = RUNTIME_DIR / "send_idem_confirmed.json"

        def _load_idem_file(path: Path) -> dict:
            try:
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
            return {}

        def _save_idem(path: Path, data: dict, keys: set, *, after_send: bool = False) -> None:
            try:
                data[_trade_day] = sorted(keys)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as _ie:
                # pending-without-confirmed on disk is the durable uncertain_sent marker
                raise AbortError(
                    "idem_persist_failure_after_send" if after_send else "idem_persist_failure",
                    f"failed to persist idem keys to {path}: {_ie}",
                ) from _ie

        _pending_data = _load_idem_file(_IDEM_PENDING_FILE)
        _confirm_data = _load_idem_file(_IDEM_CONFIRM_FILE)
        _pending_keys: set = set(_pending_data.get(_trade_day, []))
        _confirm_keys: set = set(_confirm_data.get(_trade_day, []))

        sent_orders: list = []
        send_results: list = []
        for o in order_objects:
            idem_key = f"{_trade_day}:{o.symbol}:{o.side}"
            if idem_key in _confirm_keys:
                logger.warning("IDEM_SKIP: %s %s already confirmed today", o.side, o.symbol)
                continue
            if idem_key in _pending_keys:
                logger.critical(
                    "UNCERTAIN_ORDER_STATE: %s %s pending without confirm — skipping to prevent double-send",
                    o.side, o.symbol,
                )
                continue
            if sent_orders:
                time.sleep(_ORDER_SEND_INTERVAL)
            # Write-ahead: persist pending before API call
            _pending_keys.add(idem_key)
            _save_idem(_IDEM_PENDING_FILE, _pending_data, _pending_keys)
            try:
                results = bridge._send_orders([o])
            except Exception as exc:
                logger.critical(
                    "PARTIAL_EXECUTION: sent=%s failed_on=%s error=%s",
                    [s.symbol for s in sent_orders], o.symbol, exc,
                )
                raise AbortError(
                    "partial_execution_failure",
                    f"failed after {len(sent_orders)} orders: {exc}",
                ) from exc
            r = results[0]
            if not r.get("success"):
                logger.critical(
                    "PARTIAL_EXECUTION: sent=%s failed_on=%s error=%s",
                    [s.symbol for s in sent_orders], o.symbol,
                    r.get("error") or f"result_code={r.get('result_code')}",
                )
                raise AbortError(
                    "partial_execution_failure",
                    f"order rejected after {len(sent_orders)} successful orders: {o.symbol}",
                )
            # Confirm after success; failure here means uncertain send state
            _confirm_keys.add(idem_key)
            _save_idem(_IDEM_CONFIRM_FILE, _confirm_data, _confirm_keys, after_send=True)
            send_results.append(r)
            sent_orders.append(o)

        # 発注結果表示
        print("\n=== 発注結果 ===")
        for r in send_results:
            status = "✅ 成功" if r.get("success") else "❌ 失敗"
            order_id = r.get("order_id", r.get("error", ""))
            print(f"  {r['side']} {r['symbol']} {r['qty']}株 → {status}  ({order_id})")
            if r.get("success"):
                mark_ordered(r["symbol"], r["side"])
                ledger.mark_submitted(r["symbol"], r["side"], r.get("order_id", ""))
            else:
                ledger.mark_failed(
                    r["symbol"],
                    r["side"],
                    error=(
                        r.get("error")
                        or f"result_code={r.get('result_code')}"
                    ),
                )

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
    except AbortError as exc:
        logger.critical(
            "EMERGENCY_STOP: reason=%s detail=%s positions=%s orders=%s",
            exc.reason,
            exc,
            current_positions,
            pending_orders,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
