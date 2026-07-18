"""
run_live_signal.py
最適化済みパラメータ（V2推奨設定 Calmar 2.656達成）での朝のシグナル＆発注スクリプト

【設定根拠】
  - バックテスト: 2018-2024、初期資本300万円
  - V2シナリオ（RSR>75 + 均等ウェイト + 29銘柄）:
      CAGR=+16.26% / MaxDD=-6.12% / Calmar=2.656 / Sharpe=1.693
  - 2026-03-19 TEMPORAL選定に更新（銘柄選択バイアス除去）:
      Sharpe=1.070 / CAGR=+9.98% / MaxDD=-10.62%（真の性能推定値）

【パラメータ（V2）】
  - 宇宙   : LIVE_UNIVERSE_FILE で指定（configs/universe/2026Q1_temporal24.json）
  - min_rsr : 75
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

import os
os.environ.setdefault("PYTHONUTF8", "1")  # must precede sys import so child processes inherit UTF-8 mode

import sys
import argparse
import locale
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import threading
import time as _time_module

import pandas as pd

if os.name == "nt":
    locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Runtime phase instrumentation (JSONL) ─────────────────────────────────────
# Tracks every major phase start/complete with elapsed_ms and thread count.
# Written to runtime/phase_log.jsonl (separate from watchdog.jsonl).
_PHASE_LOG_PATH: "Path | None" = None   # resolved after RUNTIME_DIR import
_PHASE_LOG_LOCK = threading.Lock()
_ACTIVE_PHASE:   str   = ""
_LAST_PROGRESS_TS: float = _time_module.monotonic()
_PROCESS_START_TS: float = _time_module.monotonic()


def _emit_phase(phase: str, event: str, *, run_id: str = "", extra: "dict | None" = None) -> None:
    """
    Emit one JSONL record to runtime/phase_log.jsonl.
    Fail-open: any I/O error is silently suppressed.
    """
    global _ACTIVE_PHASE, _LAST_PROGRESS_TS
    now_mono = _time_module.monotonic()
    if event == "start":
        _ACTIVE_PHASE      = phase
        _LAST_PROGRESS_TS  = now_mono

    elapsed_ms    = round((now_mono - _PROCESS_START_TS) * 1000, 1)
    phase_elapsed = round(now_mono - _LAST_PROGRESS_TS, 3)
    n_threads     = threading.active_count()

    record: dict = {
        "ts":                datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase":             phase,
        "event":             event,
        "elapsed_ms":        elapsed_ms,
        "active_thread_count": n_threads,
        "phase_elapsed_sec": phase_elapsed,
        "active_phase":      _ACTIVE_PHASE,
    }
    if run_id:
        record["run_id"] = run_id
    if extra:
        record.update(extra)

    if phase_elapsed > 30.0 and event != "start":
        logger.warning("[PHASE] stall detected: phase=%s event=%s phase_elapsed=%.0fs",
                       phase, event, phase_elapsed)

    if _PHASE_LOG_PATH is not None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _PHASE_LOG_LOCK:
            try:
                _PHASE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with _PHASE_LOG_PATH.open("a", encoding="utf-8") as _fh:
                    _fh.write(line)
            except Exception:
                pass


def _shutdown_audit(run_id: str = "") -> dict:
    """
    Collect and log shutdown state immediately before process termination.
    Returns audit dict. Fail-open.
    """
    import multiprocessing as _mp
    try:
        all_threads   = threading.enumerate()
        non_daemon    = [t for t in all_threads if not t.daemon]
        child_procs   = _mp.active_children()

        nd_info = [
            {"name": t.name, "alive": t.is_alive(), "daemon": t.daemon}
            for t in non_daemon
        ]
        audit = {
            "threads_alive":          len(all_threads),
            "non_daemon_count":       len(non_daemon),
            "non_daemon_threads":     nd_info,
            "active_child_processes": len(child_procs),
            "child_process_pids":     [p.pid for p in child_procs],
            "active_phase":           _ACTIVE_PHASE,
            "total_elapsed_sec":      round(_time_module.monotonic() - _PROCESS_START_TS, 2),
        }

        # Structured stderr output for watchdog visibility
        print(f"\n[SHUTDOWN_AUDIT] threads_alive={audit['threads_alive']}",
              file=sys.stderr)
        for nd in nd_info:
            if nd["name"] != "MainThread":
                print(f"  * {nd['name']} daemon={nd['daemon']}", file=sys.stderr)
        if child_procs:
            print(f"  child_processes: {audit['child_process_pids']}", file=sys.stderr)

        _emit_phase("final_exit", "audit", run_id=run_id, extra=audit)
        return audit
    except Exception as _ae:
        logger.warning("[SHUTDOWN_AUDIT] failed: %s", _ae)
        return {}

from src.config_loader import load_strategy_config

# .env 読み込み + パス定数 + ライブ安全設定（paths.py が一括管理）
from src.paths import (
    ALLOW_YFINANCE_NETWORK,
    BACKTEST_DATASET_DIR, CACHE_DIR,
    DEFAULT_DATA_VERSION,
    LOGS_DIR,
    STRATEGY_CONFIG_FILE,
    SIGNALS_DIR, ORDER_LOCK_FILE, LIVE_LOG_DIR,
    PHASE2_METRICS_FILE, RSR_UNIVERSE_FILE,
    LIVE_UNIVERSE_FILE, SHADOW_UNIVERSE_FILE,
    LIVE_MODE, MAX_ORDERS_PER_DAY, KABUS_PORT,
    RUNTIME_DIR, WATCHDOG_LOG_FILE, ORDERS_JOURNAL_DIR, INFLIGHT_REGISTRY_FILE,
    assert_live_ready, assert_execution_context,
    assert_kabus_connection, verify_dataset_integrity,
    acquire_runtime_lock, release_runtime_lock,
    enforce_order_rate_limit, record_order_sent,
    # Execution lineage + epoch governance (Task 1)
    EXECUTION_EPOCH_FILE, DEPLOYMENT_EPOCH_FILE, AUTHORITY_CHAIN_FILE,
    DEPLOYMENT_LINEAGE_FILE, DEPLOYMENT_MANIFEST_FILE,
    # Deployable-alpha leakage analytics (Task 2)
    SKIPPED_OPPORTUNITY_FILE, ALPHA_METRICS_FILE, DAILY_LEAKAGE_FILE,
    # Research priority summary
    RESEARCH_PRIORITY_FILE, RESEARCH_PRIORITY_REPORT_DIR,
    # Deployable universe governance layer
    UNIVERSE_RUNTIME_DIR, UNIVERSE_MANIFEST_FILE, UNIVERSE_PROMOTION_FILE,
    UNIVERSE_LINEAGE_FILE, UNIVERSE_DIAGNOSTICS_FILE, UNIVERSE_GOVERNANCE_REPORT_DIR,
    # Predictive expansion layer
    PREDICTIVE_SCORES_FILE, PREDICTIVE_REPORTS_DIR,
    # Intraday expansion + shadow observation + candidate ranking
    SHADOW_ENTRY_LOG_FILE, SHADOW_STATS_FILE, PREDICTIVE_CANDIDATE_LOG_FILE,
    # Future Leader shadow observer layer
    FUTURE_LEADER_CANDIDATES_FILE, FUTURE_LEADER_REPORTS_DIR,
    FUTURE_LEADER_INTEGRITY_DIR,
    FUTURE_LEADER_SURVIVABILITY_DIR,
    FUTURE_LEADER_FAILURE_DIR,
    FUTURE_LEADER_REGIME_DIR,
    FUTURE_LEADER_ARCHETYPE_DIR,
    FUTURE_LEADER_TRANSITION_DIR,
    # Universe determinism audit
    UNIVERSE_DETERMINISM_AUDIT_DIR,
)
_PHASE_LOG_PATH = RUNTIME_DIR / "phase_log.jsonl"

from src.runtime.heartbeat import HeartbeatThread
from src.live.staged_supervisor import StagedSupervisor, StageTimeout, StageError
from src.live.execution_journal import ExecutionJournal
from src.live.inflight_registry import InflightRegistry
from src.live.process_supervisor import BrokerProcessSupervisor, serialize_order, compute_front_order_type
from src.live.client_order_id import make_client_order_id
from src.live.safe_cleanup import safe_cleanup, safe_cleanup_step

from src.execution import (
    ExecutionIntent, IntentJournal, IntentStatus,
    make_intent, make_intent_id,
    apply_transition,
    reconcile_open_orders,
)

# ── 実行コンテキスト検証（最優先: モジュール読み込み直後）──────────────────
# research / backtest スクリプトが LIVE_MODE=true のまま呼ばれた場合にブロック
assert_execution_context()

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("live_signal")
logger.info(
    "[ENCODING] stdout=%s stderr=%s preferred=%s PYTHONUTF8=%s",
    getattr(sys.stdout, "encoding", "?"),
    getattr(sys.stderr, "encoding", "?"),
    locale.getpreferredencoding(False),
    os.environ.get("PYTHONUTF8", "0"),
)
cfg = load_strategy_config()

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


def _latest_local_close(symbol: str) -> float | None:
    """cache/backtest snapshot からローカル終値を返す。取得不可なら None。"""
    candidates = []
    if DEFAULT_DATA_VERSION:
        candidates.append(BACKTEST_DATASET_DIR / DEFAULT_DATA_VERSION / f"{symbol}.parquet")
    candidates.append(Path("cache") / "ohlcv" / f"{symbol}.parquet")
    candidates.append(Path("src") / "cache" / "ohlcv" / f"{symbol}.parquet")

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[close_col].dropna()
            if not series.empty:
                return float(series.iloc[-1])
        except Exception:
            continue
    return None


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
    syms = list(tickers.keys())
    raw = None
    if ALLOW_YFINANCE_NETWORK and syms:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore")
        raw = yf.download(syms, period="3d", progress=False, group_by="ticker")

    filtered: dict[str, str] = {}
    skipped:  list[tuple[str, float, float]] = []

    for sym, sector in tickers.items():
        # 保有中は価格関係なく残す（SELL シグナルを殺さない）
        if sym in held_symbols:
            filtered[sym] = sector
            continue

        # 最新終値を取得
        price = _latest_local_close(sym)
        if price is None and raw is not None:
            try:
                if len(syms) == 1:
                    price = float(raw["Close"].dropna().iloc[-1])
                else:
                    price = float(raw[sym]["Close"].dropna().iloc[-1])
            except (KeyError, IndexError, TypeError, ValueError):
                price = None

        if price is None:
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

def _send_orders_with_retry(
    bridge,
    orders: list,
    journal: "IntentJournal | None" = None,
    run_id: str = "",
    broker_client=None,
) -> list[dict]:
    """
    _send_orders() を最大 MAX_RETRY 回リトライする。
    OrderLedger で pending → submitted / failed を明示管理し、
    成功した発注のみ order_lock に記録する。

    IntentJournal (journal) が渡された場合:
      - 発注前に CREATED → 重複/重複チェック
      - 送信時に SUBMITTED
      - 成功時に ACKED (broker_order_id 記録)
      - 例外時に UNKNOWN（直接 CANCELLED にしない）
      - リトライ前に reconcile_open_orders() 実行

    journal=None の場合はジャーナル処理をスキップ（DRY 互換）。

    ledger ステート遷移:
        check_and_record(pending) → API成功 → mark_submitted(order_id)
                                  → API失敗 → mark_failed(error)

    Returns:
        send_results: [{symbol, side, qty, order_id, success, ...}, ...]
    """
    import time
    from src.live.order_ledger import OrderLedger

    ledger = OrderLedger()

    # ── IntentJournal: CREATED + centralized gate ─────────────────────
    intent_map: dict[str, ExecutionIntent] = {}  # symbol+side → intent
    now_ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    if journal is not None:
        filtered_by_journal: list = []
        for o in orders:
            key = f"{o.symbol}|{o.side}"
            try:
                intent = make_intent(
                    symbol         = o.symbol,
                    side           = o.side,
                    qty            = o.qty,
                    strategy       = getattr(o, "strategy_type", "unknown"),
                    expected_price = getattr(o, "estimated_price", 0.0),
                    run_id         = run_id,
                    signal_ts      = now_ts,
                    snapshot_generation = 0,
                    alloc_before   = 0.0,
                    alloc_after    = 0.0,
                )
                allowed, reason = journal.can_submit_intent(
                    intent.intent_id, o.symbol, o.side, run_id
                )
                if not allowed:
                    logger.warning("[INTENT] gate blocked: %s — %s", key, reason)
                    continue
                journal.append(intent, "created")
                intent_map[key] = intent
                filtered_by_journal.append(o)
            except Exception as _je:
                logger.warning("[INTENT] journal CREATED failed, continuing: %s", _je)
                filtered_by_journal.append(o)
        orders = filtered_by_journal

    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        # ── IntentJournal: SUBMITTED ──────────────────────────────────
        if journal is not None:
            for o in orders:
                key = f"{o.symbol}|{o.side}"
                intent = intent_map.get(key)
                if intent is not None:
                    try:
                        apply_transition(intent, IntentStatus.SUBMITTED)
                        journal.append(intent, "submitted")
                    except Exception as _je:
                        logger.warning("[INTENT] SUBMITTED journal failed: %s", _je)

        try:
            results = bridge._send_orders(orders)
            # 成功 → submitted 記録 / 失敗 → failed 記録
            for r in results:
                sym  = r["symbol"]
                side = r["side"]
                if r.get("success"):
                    mark_ordered(sym, side)                           # ORDER_LOCK_FILE（重複防止）
                    ledger.mark_submitted(sym, side, r.get("order_id", ""))  # ledger audit trail
                    # IntentJournal: ACKED
                    if journal is not None:
                        key = f"{sym}|{side}"
                        intent = intent_map.get(key)
                        if intent is not None:
                            try:
                                broker_oid = str(r.get("order_id", ""))
                                apply_transition(intent, IntentStatus.ACKED,
                                                 broker_order_id=broker_oid)
                                journal.append(intent, "acked")
                            except Exception as _je:
                                logger.warning("[INTENT] ACKED journal failed: %s", _je)
                else:
                    err = r.get("error", f"result_code={r.get('result_code', '?')}")
                    ledger.mark_failed(sym, side, err)
            return results
        except Exception as e:
            last_exc = e
            # ── IntentJournal: UNKNOWN (timeout/network failure) ──────
            if journal is not None:
                for o in orders:
                    key = f"{o.symbol}|{o.side}"
                    intent = intent_map.get(key)
                    if intent is not None and intent.status == IntentStatus.SUBMITTED.value:
                        try:
                            apply_transition(intent, IntentStatus.UNKNOWN)
                            journal.append(intent, "unknown", reason=str(e))
                        except Exception as _je:
                            logger.warning("[INTENT] UNKNOWN journal failed: %s", _je)

            if attempt < MAX_RETRY:
                logger.warning(
                    "発注エラー（試行 %d/%d）: %s — %d秒後にリトライ",
                    attempt, MAX_RETRY, e, RETRY_SLEEP,
                )
                # ── IntentJournal: reconcile before retry ─────────────
                if journal is not None:
                    try:
                        reconcile_open_orders(journal, broker_client)
                    except Exception as _re:
                        logger.warning("[INTENT] reconcile failed: %s", _re)
                time.sleep(RETRY_SLEEP)
            else:
                logger.error("発注失敗（%d回試行後）: %s", MAX_RETRY, e)
    raise RuntimeError(f"発注 {MAX_RETRY}回失敗: {last_exc}") from last_exc


def _submit_orders_process_isolated(
    orders_objs: list,
    registry: "InflightRegistry | None",
    trading_day: str,
    run_id: str,
    proc_supervisor: "BrokerProcessSupervisor",
    timeout_sec: int = 25,
) -> list:
    """
    Submit orders via child-process isolation (BrokerProcessSupervisor).

    Lifecycle per order:
        register → mark_submitting → BrokerProcessSupervisor.submit_orders()
        → mark_acked (success) | mark_failed (error/timeout)

    Duplicate suppression: orders whose client_order_id already exists in the
    registry (and is not FAILED) are silently skipped.

    Returns list of result dicts (same schema as broker_worker output):
        {symbol, side, qty, success, order_id, result_code, client_order_id, error}
    """
    fot, skip = compute_front_order_type()
    if skip:
        logger.warning("[PROC_ORDER] market closed — all orders skipped")
        return []

    orders_dicts: list = []
    # SAFETY FIX (2026-07-07 follow-up incident): broker_worker.py's result
    # schema only carries {symbol, side, qty, client_order_id, success,
    # order_id, result_code, error} — it has no estimated_price/atr20/reason/
    # sector/strategy_type. Those fields were silently lost between here and
    # update_state_after_execution(), which then fell back to 0.0 and wrote
    # corrupted entry_price/highest_close/missing ATR into portfolio_state
    # (found on 5301.T). The parent process already holds the full order
    # object, so keep a coi→order lookup and restore these fields onto the
    # child's result below, instead of round-tripping them through the worker.
    _coi_to_order: dict[str, object] = {}
    for o in orders_objs:
        coi = make_client_order_id(
            strategy    = getattr(o, "strategy_type", "unknown"),
            symbol      = o.symbol,
            side        = o.side,
            qty         = int(o.qty),
            trading_day = trading_day,
        )
        if registry is not None and registry.is_duplicate(coi):
            logger.warning(
                "[PROC_ORDER] duplicate suppressed: %s %s %s qty=%d",
                coi, o.symbol, o.side, int(o.qty),
            )
            continue
        od = serialize_order(o, fot, coi)
        orders_dicts.append(od)
        _coi_to_order[coi] = o
        if registry is not None:
            try:
                registry.register(
                    coi,
                    symbol      = o.symbol,
                    side        = o.side,
                    qty         = int(o.qty),
                    strategy    = getattr(o, "strategy_type", "unknown"),
                    trading_day = trading_day,
                    run_id      = run_id,
                )
            except ValueError as _ve:
                logger.warning("[PROC_ORDER] register: %s", _ve)

    if not orders_dicts:
        return []

    # Mark submitting BEFORE broker API call — crash leaves SUBMITTED_UNKNOWN not PENDING_SUBMIT
    if registry is not None:
        for od in orders_dicts:
            safe_cleanup_step(
                f"mark_submitting_{od['client_order_id'][:8]}",
                registry.mark_submitting,
                od["client_order_id"],
            )

    raw_results = proc_supervisor.submit_orders(orders_dicts, timeout_sec=timeout_sec, run_id=run_id)

    # ── entry metadata restoration (SAFETY FIX) ──────────────────────────
    for r in raw_results:
        _src_order = _coi_to_order.get(r.get("client_order_id", ""))
        if _src_order is None:
            continue
        r.setdefault("estimated_price", float(getattr(_src_order, "estimated_price", 0.0)))
        r.setdefault("atr20",           float(getattr(_src_order, "atr20", 0.0)))
        r.setdefault("reason",          getattr(_src_order, "reason", ""))
        r.setdefault("sector",          getattr(_src_order, "sector", "不明"))
        r.setdefault("strategy_type",   getattr(_src_order, "strategy_type", ""))

    # Update registry state from results
    if registry is not None:
        for r in raw_results:
            coi = r.get("client_order_id", "")
            if not coi:
                continue
            if r.get("success") and r.get("order_id"):
                safe_cleanup_step(f"mark_acked_{coi[:8]}", registry.mark_acked, coi, str(r["order_id"]))
            else:
                err = str(r.get("error") or f"result_code={r.get('result_code', '?')}")
                safe_cleanup_step(f"mark_failed_{coi[:8]}", registry.mark_failed, coi, err)

    return raw_results


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
    # Build enriched order log: merge signal-level data with send_result execution quality
    _sig_map = {s["symbol"]: s for s in result.signals}
    _enriched_results = []
    for _r in send_results:
        _sym  = _r.get("symbol", "")
        _sig  = _sig_map.get(_sym, {})
        _enriched_results.append({
            **_r,
            # execution quality fields (present in send_results from signal_bridge)
            "planned_entry_price": _r.get("planned_entry_price"),
            "actual_entry_price":  _r.get("actual_entry_price"),
            "slippage_pct":        _r.get("slippage_pct"),
            "gap_pct":             _r.get("gap_pct"),
            "fill_status":         _r.get("fill_status"),
            "order_submit_time":   _r.get("order_submit_time"),
            "fill_time":           _r.get("fill_time"),
            # RSR percentile from signal (observation-only)
            "rsr_pct_raw":         _sig.get("rsr_pct_raw"),
            "rsr_pct_smooth":      _sig.get("rsr_pct_smooth"),
            "entry_signal_time":   _sig.get("entry_signal_time"),
        })
    runs_ord.append({
        "run_at":       datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "orders":       result.orders,
        "send_results": _enriched_results,
    })
    ord_path.write_text(json.dumps(runs_ord, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("ライブログ保存: %s / %s", sig_path, ord_path)

# G29_UNIVERSE は廃止。load_universe() で configs/universe/*.json から読み込む。
# 後方互換用として参照のみ残す（直接使用禁止）
_LEGACY_G29_UNIVERSE_REMOVED = True

logger.info(
    "戦略パラメータ: min_rsr=%.1f turtle_entry=%d exit_lookback=%d min_sepa=%d",
    cfg.fujiko.min_rsr,
    cfg.fujiko.turtle_entry,
    cfg.fujiko.turtle_exit,
    cfg.fujiko.min_sepa,
)

CAPITAL             = cfg.portfolio.capital
_RISK_PARAMS         = derive_risk_params(CAPITAL)               # 資本連動パラメータ（全体で一貫して使用）
MAX_SINGLE_WEIGHT    = cfg.portfolio.max_single_weight
MAX_POS              = cfg.portfolio.max_positions
MIN_SECTORS          = cfg.portfolio.min_sectors
MAX_DD_LIMIT         = cfg.portfolio.max_dd_limit
TOP_K                = MAX_POS
MAX_HOLD_DAYS        = cfg.risk.max_hold_days
MIN_HOLD_DAYS        = cfg.risk.min_hold_days
EMERGENCY_EXIT_PCT   = cfg.risk.emergency_exit_pct
MAX_NEW_POS_PER_DAY  = 2        # 1回の実行で生成する新規 BUY 上限（過剰発注防止・初期運用安定化）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="フジコ法 最適化済み朝のシグナル生成スクリプト")
    p.add_argument("--live",   action="store_true", help="kabuステーション API に実際に発注する")
    p.add_argument("--dry-run", dest="live", action="store_false", help="ドライランを明示する")
    p.add_argument(
        "--allow-no-broker", action="store_true", default=False,
        help="broker snapshot取得失敗時にAbortErrorで停止せず、明示的に省略モードで続行する"
        "（手動検証・研究用途専用。watchdog_runner.py からは絶対に付与しない）。",
    )
    p.add_argument("--yes","-y", action="store_true", help="発注確認プロンプトをスキップ")
    p.add_argument("--no-save", action="store_true", help="JSON シグナルファイルを保存しない")
    p.add_argument("--output-dir", default=str(SIGNALS_DIR), help="シグナルJSONの保存先")
    return p.parse_args()


def print_banner(
    live: bool,
    universe: dict[str, str],
    universe_meta: dict,
    *,
    actual_equity: "float | None" = None,
    eff_capital: "int | None" = None,
) -> None:
    """
    起動バナーを表示する。

    actual_equity / eff_capital が渡された場合（capital layer 確定後）は
    実資産ベースの情報を表示する。省略時は config CAPITAL にフォールバック。
    """
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

    # 実資産情報（capital layer 確定後に呼ばれた場合のみ表示）
    _disp_cap = eff_capital if eff_capital is not None else CAPITAL
    _rp = derive_risk_params(_disp_cap)
    _rec_uni = recommended_universe_size(_disp_cap)
    if actual_equity is not None and eff_capital is not None:
        print(f"  実口座資産     : ¥{actual_equity:,.0f}")
        _delta = int(actual_equity) - eff_capital
        _rate_label = f"  （差額=¥{_delta:,} ← rate-limit適用中）" if abs(_delta) > 10_000 else ""
        print(
            f"  有効資本       : ¥{eff_capital:,}"
            f"  max_alloc=¥{int(_rp['max_allocation']):,}(×30%)"
            f"  max_pos=¥{int(_rp['max_position']):,}(×20%)"
            + _rate_label
        )
    else:
        print(f"  資本設定       : ¥{_disp_cap:,}  max_alloc=¥{int(_rp['max_allocation']):,}(×30%)  max_pos=¥{int(_rp['max_position']):,}(×20%)")

    if _rec_uni > len(universe):
        logger.info(
            "[RECOMMENDATION] capital=%d target_positions=%d suggested_live_universe=%d "
            "current_live_universe=%d reason=capital_based_sizing",
            _disp_cap, MAX_POS, _rec_uni, len(universe),
        )
        print(f"  [RECOMMENDATION] capital=¥{_disp_cap:,} suggested_live_universe={_rec_uni}"
              f" current={len(universe)} reason=capital_based_sizing")
    print("=" * 64)


def _format_hold_days(s: dict) -> str:
    """
    保有日数の表示文字列を返す（2026-07-15 entry metadata SSOT修正）。

    entry_date_known=False（position_entry_dates欠損）の場合は"0d"へ
    フォールバック表示しない — "Unknown"を返す。当日新規建ての真の0dとの
    誤認を防ぐ（signal_bridge.py::StockSignal.entry_date_known参照）。
    """
    if not s.get("entry_date_known", True):
        return "Unknown"
    return f"{s.get('hold_days', 0):>4}d"


def print_signals(result) -> None:
    orders = result.orders

    # ── 保有継続（HOLD_CONTINUE）銘柄を先に抽出 ───────────────────────
    # signal==1 かつ currently_holding → 新規発注ではなく「保有継続」
    hold_continue_sigs = [
        s for s in result.signals
        if s["signal"] == 1 and s.get("currently_holding")
    ]
    new_buy_sigs = [
        s for s in result.signals
        if s["signal"] == 1 and not s.get("currently_holding")
    ]
    sell_sigs = [s for s in result.signals if s["signal"] == -1]

    # ── 発注予定（実行可能な注文のみ）──────────────────────────────────
    if not orders:
        if hold_continue_sigs:
            hold_syms = ", ".join(s["symbol"] for s in hold_continue_sigs)
            print(f"\n📭 新規注文なし  （保有継続: {hold_syms}）")
        else:
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

    # ── 保有継続ポジション（HOLD_CONTINUE）─────────────────────────────
    if hold_continue_sigs:
        print(f"\n📌 保有継続 ({len(hold_continue_sigs)}件):")
        print(f"  {'銘柄':<10} {'セクター':<10} {'rank':>5} {'RSR':>6} {'保有日':>5} {'trailing_stop':>14}")
        print("  " + "-" * 65)
        for s in sorted(hold_continue_sigs, key=lambda x: x.get("rsr_rank", 99)):
            stop_str = f"¥{s['stop_price']:,.0f}" if s.get("stop_price", 0) > 0 else "   —"
            hold_str = _format_hold_days(s)
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} "
                f"{s.get('rsr_rank', 0):>5} {s['rsr']:>6.1f} "
                f"{hold_str:>6} {stop_str:>14}"
            )

    # ── 新規BUY候補（発注可能・未保有）──────────────────────────────────
    if new_buy_sigs:
        print(f"\n📊 新規BUY候補 ({len(new_buy_sigs)}件):")
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'rank':>5} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  理由")
        print("  " + "-" * 74)
        for s in sorted(new_buy_sigs, key=lambda x: x.get("rsr_rank", 99)):
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

    # ── RSR ランキング上位10銘柄 ─────────────────────────────────────
    print(f"\n📊 RSR ランキング上位10銘柄:")
    print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'rank':>5} {'RSR':>6} {'SEPA':>5} {'Mom':>7} {'保有日':>5}  シグナル")
    print("  " + "-" * 78)
    for s in sorted(result.signals, key=lambda x: x.get("rsr_rank", 99))[:10]:
        if s["signal"] == 1 and s.get("currently_holding"):
            sig_str = "📌 HOLD"   # 保有継続（新規発注なし）
        elif s["signal"] == 1:
            sig_str = "✅ BUY "   # 新規エントリー候補
        elif s["signal"] == -1:
            sig_str = "🔴 SELL"
        else:
            sig_str = "  -   "
        strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
        hold_str  = _format_hold_days(s) if s.get("currently_holding") else "     -"
        print(
            f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
            f"{s.get('rsr_rank', 0):>5} {s['rsr']:>6.1f} {s['sepa_score']:>5}"
            f" {s['rsr_momentum']:>+7.2f} {hold_str}  {sig_str}"
        )


from src.live.preview import print_live_preview  # noqa: E402 — placed after module-level init


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

    # run_id をメイン冒頭で確定（lock / supervisor / journal で共有）
    run_id = datetime.now(JST).strftime("%Y%m%d_%H%M%S")

    # ── ENTRY FREEZE 起動時ログ（資産保全・2026-07-17・defense-in-depth）──
    _ef_startup = cfg.entry_freeze
    logger.warning(
        "[ENTRY_FREEZE_STATE] entry_freeze_enabled=%s reason=%s mode=%s run_id=%s",
        _ef_startup.enabled, _ef_startup.reason,
        "LIVE" if args.live else "DRY", run_id,
    )

    # ── Step2: 二重起動防止（最初に取得・atexit で自動解放）──────────────────
    _lock_instance = None
    _heartbeat    = None
    if args.live:
        try:
            _lock_instance = acquire_runtime_lock(run_id=run_id)
        except RuntimeError as _le:
            print(f"[FATAL] {_le}", file=sys.stderr)
            return 1
        # Heartbeat: 5 s interval keeps lock fresh; stale threshold = 30 s
        _heartbeat = HeartbeatThread(_lock_instance, interval_sec=5.0)
        _heartbeat.start()

    # ── Inflight registry: load + crash-recovery check ───────────────────────
    _inflight_registry: InflightRegistry | None = None
    if args.live:
        _inflight_registry = InflightRegistry(INFLIGHT_REGISTRY_FILE)
        _n_loaded = _inflight_registry.load()
        _unresolved = _inflight_registry.get_unresolved()
        if _unresolved:
            logger.warning(
                "[INFLIGHT] %d unresolved orders from previous run — verify broker state before trading:",
                len(_unresolved),
            )
            for _u in _unresolved:
                logger.warning("  %r", _u)
        else:
            logger.info("[INFLIGHT] registry ok: %d orders loaded, 0 unresolved", _n_loaded)

    # ── Phase instrumentation: bootstrap ─────────────────────────────────────
    _emit_phase("bootstrap", "start", run_id=run_id)

    # ── Autonomous recovery layer: bootstrap / supervisor / disk / rotation ──
    try:
        from src.runtime.bootstrap_recovery import run_bootstrap_recovery
        from src.runtime.runtime_supervisor import check_runtime_health, append_health_report
        from src.runtime.disk_guard import run_disk_guard
        from src.allocation.jsonl_rotation import rotate_if_needed as _rot
        from src.paths import (
            EXECUTION_LOCK_FILE,
            BOOTSTRAP_RECOVERY_FILE, RUNTIME_HEALTH_FILE, DISK_GUARD_FILE,
            SKIPPED_TRADE_FILE, SKIPPED_OUTCOMES_FILE,
            ALLOCATION_OUTCOMES_FILE, ALLOCATION_INTENT_FILE,
            STALE_VISIBILITY_FILE, OUTCOMES_ARCHIVE_DIR,
            EXPOSURE_STATE_FILE, STABILITY_WINDOW_FILE,
            CAPITAL_STATE_FILE, DEPLOYMENT_RAMP_STATE_FILE,
        )
        import time as _time_boot
        from datetime import datetime as _dt_boot, timezone as _tz_boot

        # Bootstrap: crash detection + file integrity + unresolved orders
        _boot_report = run_bootstrap_recovery(
            lock_file=EXECUTION_LOCK_FILE,
            jsonl_files=[
                SKIPPED_TRADE_FILE, ALLOCATION_OUTCOMES_FILE,
                ALLOCATION_INTENT_FILE, STALE_VISIBILITY_FILE,
            ],
            json_files=[
                CAPITAL_STATE_FILE, DEPLOYMENT_RAMP_STATE_FILE,
                EXPOSURE_STATE_FILE, STABILITY_WINDOW_FILE,
            ],
            inflight_file=INFLIGHT_REGISTRY_FILE,
            allocation_state_files=[EXPOSURE_STATE_FILE, STABILITY_WINDOW_FILE],
            capital_state_files=[CAPITAL_STATE_FILE],
            report_path=BOOTSTRAP_RECOVERY_FILE,
            run_id=run_id,
        )
        if _boot_report.crash_detected:
            logger.warning("[BOOTSTRAP] prior crash: %s", _boot_report.crash_detail)
        if _boot_report.unresolved_order_count > 0:
            logger.warning(
                "[BOOTSTRAP] %d unresolved orders from prior run",
                _boot_report.unresolved_order_count,
            )
        if not _boot_report.bootstrap_ok:
            logger.warning("[BOOTSTRAP] recovery issues: %s", _boot_report.recovery_actions)

        # Runtime supervisor: orphan lock / stale heartbeat / duplicate detection
        _rt_health = check_runtime_health(EXECUTION_LOCK_FILE, run_id=run_id)
        append_health_report(_rt_health, RUNTIME_HEALTH_FILE)
        if _rt_health.orphan_lock_detected:
            logger.warning("[SUPERVISOR] orphan lock: %s", _rt_health.orphan_lock_detail)
        if _rt_health.duplicate_runtime_detected:
            logger.warning("[SUPERVISOR] duplicate runtime: %s", _rt_health.orphan_lock_detail)

        # Disk guard: space check + auto-rotation on low disk
        _disk_report = run_disk_guard(
            check_path=RUNTIME_DIR,
            archive_dirs=[OUTCOMES_ARCHIVE_DIR],
            jsonl_paths=[
                SKIPPED_TRADE_FILE, ALLOCATION_OUTCOMES_FILE,
                STALE_VISIBILITY_FILE, ALLOCATION_INTENT_FILE,
            ],
            report_path=DISK_GUARD_FILE,
        )
        if _disk_report.low_disk_warning:
            logger.warning("[DISK_GUARD] %s", _disk_report.detail)

        # Monthly JSONL rotation for all allocation telemetry files
        _rot_now = _dt_boot.now(_tz_boot.utc)
        for _rp in [
            SKIPPED_TRADE_FILE, SKIPPED_OUTCOMES_FILE,
            ALLOCATION_OUTCOMES_FILE, ALLOCATION_INTENT_FILE,
            STALE_VISIBILITY_FILE,
        ]:
            try:
                _rot(_rp, archive_dir=OUTCOMES_ARCHIVE_DIR, now=_rot_now)
            except Exception:
                pass

        # entry_price/entry_atr/highest_close 欠落リカバリ（2026-07-07 follow-up incident）
        # process-isolated経路のsend_resultにprice/ATRが乗らず0.0で書き込まれた
        # 既存ポジションを、当日の注文ログ(logs/live/*_orders.json)から復元する。
        # 復元できない銘柄は自動売買を止めず entry_metadata_missing に記録するのみ。
        try:
            from src.live.entry_metadata_recovery import (
                recover_missing_entry_metadata, recover_missing_entry_rsr,
            )
            from src.portfolio.state_store import load_portfolio_state, save_portfolio_state
            from src.paths import SIGNALS_DIR as _pmr_signals_dir
            _pmr_state_file = RUNTIME_DIR / "portfolio_state.json"
            _pmr_state, _pmr_vr = load_portfolio_state(_pmr_state_file)
            _pmr_result = recover_missing_entry_metadata(
                _pmr_state,
                logs_live_dir=LOGS_DIR / "live",
                audit_log_path=LOGS_DIR / "entry_metadata_recovery_audit.jsonl",
            )
            # entry_rsr 欠落リカバリ（2026-07-08 RCA）: signal_rsr_map が
            # update_state_after_execution() に渡らなかった経路(run_morning_signal.py 等)
            # のBUYで position_entry_rsrs が欠落し、Quality Replacement Engine が
            # current RSR を proxy 使用し続けていた既存ポジションを日次シグナルJSONから復元する。
            _rsr_result = recover_missing_entry_rsr(
                _pmr_state,
                signals_dir=_pmr_signals_dir,
                audit_log_path=LOGS_DIR / "entry_rsr_recovery_audit.jsonl",
            )
            if _pmr_result["recovered"] or _pmr_result["unrecoverable"] or _rsr_result["recovered"]:
                save_portfolio_state(_pmr_state, path=_pmr_state_file, data_source="internal")
                logger.warning(
                    "[ENTRY_METADATA_RECOVERY] recovered=%d unrecoverable=%d",
                    len(_pmr_result["recovered"]), len(_pmr_result["unrecoverable"]),
                )
            if _rsr_result["recovered"] or _rsr_result["unrecoverable"]:
                logger.warning(
                    "[ENTRY_RSR_RECOVERY] recovered=%d unrecoverable=%d",
                    len(_rsr_result["recovered"]), len(_rsr_result["unrecoverable"]),
                )
        except Exception as _pmr_err:
            logger.warning("[ENTRY_METADATA_RECOVERY] startup recovery failed (%s) — continuing", _pmr_err)

        logger.info(
            "[RECOVERY] bootstrap=%s supervisor_healthy=%s disk_ok=%s",
            _boot_report.bootstrap_ok,
            _rt_health.overall_healthy,
            not _disk_report.low_disk_warning,
        )
    except Exception as _recovery_err:
        logger.warning("[RECOVERY] startup checks failed (%s) — continuing", _recovery_err)

    # ── Phase 2.5: Materialize forward returns from prior promotion decisions ──
    # Runs in bootstrap so OHLCV data is fresh before today's signal generation.
    # FAIL_OPEN: never blocks execution.
    try:
        from src.universe.outcome_materializer import (
            materialize_forward_returns as _mat_fwd,
            materialize_rotation_returns as _mat_rot,
        )
        from src.paths import (
            UNIVERSE_OUTCOME_DIR as _mat_outcome_dir,
            OHLCV_DIR as _mat_ohlcv_dir,
        )
        _mat_summary = _mat_fwd(_mat_outcome_dir, _mat_ohlcv_dir)
        _mat_rot_n   = _mat_rot(_mat_outcome_dir, _mat_ohlcv_dir)
        if _mat_summary.newly_materialized > 0 or _mat_rot_n > 0:
            logger.info(
                "[MATERIALIZER] promotions_newly_done=%d rotation_newly_done=%d "
                "no_data=%d no_file=%d",
                _mat_summary.newly_materialized, _mat_rot_n,
                _mat_summary.skipped_no_data, _mat_summary.skipped_no_file,
            )
    except Exception as _mat_err:
        logger.warning("[MATERIALIZER] forward return materialization failed (FAIL_OPEN): %s", _mat_err)
        _emit_phase("bootstrap", "materialization_failed", run_id=run_id,
                    extra={"materialization_error": str(_mat_err)})

    _emit_phase("bootstrap", "complete", run_id=run_id)

    # ── LIVE_MODE 二重ガード（paths.py 経由）────────────────────────────────
    # --live フラグ AND .env の LIVE_MODE=true の両方が必要。
    if args.live:
        try:
            assert_live_ready()          # LIVE_MODE / ファイル存在 / 上限値を一括チェック
            assert_kabus_connection()    # kabuStation API 疎通確認（未起動ならここで止まる）
        except RuntimeError as _e:
            if _heartbeat:
                _heartbeat.stop()
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

    # バナーは capital layer 確定後（下方）に実資産つきで表示する

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

    # ─── CAPITAL LAYER: load adaptive growth states ───────────────────────────
    # Phase 5B.1: moved here (before universe price filter + governance) so both
    # use live effective_capital rather than the static config CAPITAL.
    # Falls back to static CAPITAL if state files are missing or corrupted.
    # Failure here NEVER aborts the trading pipeline.

    # ── Step 0: 診断ログ（config 資本の確認）──────────────────────────────────
    logger.info(
        "[CAPITAL_DIAG] config_capital=¥%s source=strategy.yaml"
        " — live equity fetch will follow",
        f"{float(CAPITAL):,.0f}",
    )

    # ── Step 2: ブローカーAPIから実資産を取得（FAIL_OPEN）──────────────────
    # LIVE / DRY のみ。バックテスト系スクリプトはここに到達しない。
    _live_snap = None
    try:
        from src.broker.live_equity_fetcher import fetch_live_equity as _fetch_live_equity
        from src.kabusapi.client import KabuClient as _KabuClientForEquity
        _eq_client = _KabuClientForEquity()
        _eq_client.fetch_token()
        _live_snap = _fetch_live_equity(_eq_client)
        # _live_snap が None の場合は両 API 失敗 → state file フォールバックへ進む
    except Exception as _live_eq_err:
        logger.warning("[LIVE_EQUITY] client/fetch initialization failed (FAIL_OPEN): %s", _live_eq_err)

    _eff_capital: int = CAPITAL
    _cap_state_loaded = False
    _cap_state = _ramp_state = _freeze_state = None
    _aggression_state = _edge_model_state = _deploy_state_rec = None
    _multihorizon_conf = _reflexivity_state = _exploration_state = None
    _opp_cost_state = None
    try:
        from src.capital import (
            load_capital_state, save_capital_state,
            load_deployment_ramp, save_deployment_ramp,
            load_freeze_state, save_freeze_state,
            load_aggression_state, save_aggression_state,
            load_edge_model, save_edge_model,
            load_deployment_state_record, save_deployment_state_record,
            load_multihorizon, save_multihorizon,
            load_concentration_metrics, save_concentration_metrics,
            load_reflexivity_state, save_reflexivity_state,
            load_exploration_state, save_exploration_state,
            load_opportunity_cost_state,
            build_telemetry_entry, append_telemetry,
        )
        from src.paths import (
            CAPITAL_STATE_FILE, DEPLOYMENT_RAMP_STATE_FILE, CAPITAL_FREEZE_STATE_FILE,
            CAPITAL_TELEMETRY_FILE, AGGRESSION_STATE_FILE, EDGE_MODEL_STATE_FILE,
            DEPLOYMENT_STATE_RECORD_FILE, MULTIHORIZON_STATE_FILE,
            CONCENTRATION_METRICS_FILE, EXPLORATION_STATE_FILE,
            REFLEXIVITY_STATE_FILE, OPPORTUNITY_COST_LOG_FILE,
        )
        _cap_state       = load_capital_state(CAPITAL_STATE_FILE)
        _ramp_state      = load_deployment_ramp(DEPLOYMENT_RAMP_STATE_FILE)
        _freeze_state    = load_freeze_state(CAPITAL_FREEZE_STATE_FILE)
        _aggression_state = load_aggression_state(AGGRESSION_STATE_FILE)
        _edge_model_state = load_edge_model(EDGE_MODEL_STATE_FILE)
        _deploy_state_rec = load_deployment_state_record(DEPLOYMENT_STATE_RECORD_FILE)
        _multihorizon_conf = load_multihorizon(MULTIHORIZON_STATE_FILE)
        _reflexivity_state = load_reflexivity_state(REFLEXIVITY_STATE_FILE)
        _exploration_state = load_exploration_state(EXPLORATION_STATE_FILE)
        _opp_cost_state   = load_opportunity_cost_state(OPPORTUNITY_COST_LOG_FILE)

        if _cap_state and _cap_state.risk_adjusted_capital > 0:
            _eff_capital = int(_cap_state.risk_adjusted_capital)
        _cap_state_loaded = True
        if _cap_state is not None:
            # actual= は Step 2 で取得済みの broker 実資産(_live_snap)を表示する
            # （disk-loaded _cap_state.actual_equity は前回run終了時点の値で1日ずれるため）。
            # effective/risk_adj は意図的に pre-sync（今回のrate-limit適用前）の値のまま
            # ロードした状態を監査する行として残す。sync後の確定値は [CAPITAL_SYNC] を参照。
            _loaded_actual_display = (
                _live_snap.actual_equity if _live_snap is not None else _cap_state.actual_equity
            )
            logger.info(
                "[CAPITAL_LOADED] actual=¥%s effective=¥%s risk_adj=¥%s "
                "ramp=%s freeze=%s deploy=%s aggression_ema=%.3f edge=%.3f",
                f"{_loaded_actual_display:,.0f}", f"{_cap_state.effective_capital:,.0f}",
                f"{_cap_state.risk_adjusted_capital:,.0f}",
                _ramp_state.mode if _ramp_state else "N/A",
                _freeze_state.is_frozen if _freeze_state else "N/A",
                _deploy_state_rec.state if _deploy_state_rec else "N/A",
                _aggression_state.ema_score if _aggression_state else 0.0,
                _edge_model_state.composite_edge_score if _edge_model_state else 0.0,
            )
        else:
            logger.info(
                "[CAPITAL_FALLBACK] capital_state.json missing — "
                "using static CAPITAL=¥%s (deploy live will bootstrap on first success)",
                f"{float(CAPITAL):,.0f}",
            )
    except Exception as _cap_err:
        logger.warning(
            "[CAPITAL] state load failed (%s) — using static CAPITAL=¥%s",
            _cap_err, f"{CAPITAL:,}",
        )
        # RC2 FIX: construct a deterministic default CapitalState so that
        # _cap_state is never None. A None critical state causes the registry
        # to raise RuntimeContractError and blocks LIVE execution entirely.
        if _cap_state is None:
            try:
                from src.capital import CapitalState as _CapitalState
                _cap_state = _CapitalState.initial(float(CAPITAL))
                logger.info(
                    "[CAPITAL] default CapitalState created: risk_adj=¥%s",
                    f"{_cap_state.risk_adjusted_capital:,.0f}",
                )
            except Exception as _cap_init_err:
                logger.warning("[CAPITAL] CapitalState.initial() failed (%s) — cap_state stays None", _cap_init_err)

    # ── Step 3: 実資産で capital_state を更新（FAIL_OPEN）────────────────────
    # _live_snap が取得できていた場合のみ実行する。
    # effective_capital の rate-limit (1.5%/日) は維持する（過剰発注防止）。
    if _live_snap is not None and _cap_state is not None:
        try:
            from src.capital import (
                update_capital_state as _update_cap_state,
                save_capital_state as _save_cap_state,
            )
            from src.paths import CAPITAL_STATE_FILE as _cap_state_file
            _growth_limit = float(getattr(
                getattr(cfg, "capital_scaling", None),
                "effective_capital_growth_limit_daily",
                0.015,
            ))
            _synced_cap = _update_cap_state(
                _cap_state,
                _live_snap.actual_equity,
                daily_growth_limit=_growth_limit,
            )
            logger.info(
                "[CAPITAL_SYNC] actual=¥%s effective=¥%s deployable=¥%s risk_adjusted=¥%s",
                f"{_synced_cap.actual_equity:,.0f}",
                f"{_synced_cap.effective_capital:,.0f}",
                f"{_synced_cap.deployable_capital:,.0f}",
                f"{_synced_cap.risk_adjusted_capital:,.0f}",
            )
            _cap_state = _synced_cap
            _eff_capital = int(_synced_cap.risk_adjusted_capital)
            # bridge 構築前に確定値を保存（次回実行時の state file として使われる）
            _save_cap_state(_cap_state, _cap_state_file)
            logger.info("[CAPITAL_SYNC] capital_state pre-saved (bridge construction uses fresh values)")
        except Exception as _csync_err:
            logger.warning("[CAPITAL_SYNC] update failed (FAIL_OPEN): %s", _csync_err)
    elif _live_snap is not None and _cap_state is None:
        # state file がない初回起動: live_snap から CapitalState を bootstrap する
        try:
            from src.capital import CapitalState as _CapStateForInit, save_capital_state as _save_cap_state
            from src.paths import CAPITAL_STATE_FILE as _cap_state_file
            _cap_state = _CapStateForInit.initial(_live_snap.actual_equity)
            _eff_capital = int(_cap_state.risk_adjusted_capital)
            _save_cap_state(_cap_state, _cap_state_file)
            logger.info(
                "[CAPITAL_SYNC] initial bootstrap from live_equity actual=¥%s risk_adj=¥%s",
                f"{_live_snap.actual_equity:,.0f}", f"{_cap_state.risk_adjusted_capital:,.0f}",
            )
        except Exception as _boot_err:
            logger.warning("[CAPITAL_SYNC] initial bootstrap failed (FAIL_OPEN): %s", _boot_err)

    # deployable_capital for CapitalDeploymentOS dynamic_max_positions (Phase 5B.1)
    _cdos_deployable: float = (
        float(_cap_state.deployable_capital)
        if (_cap_state is not None and _cap_state.deployable_capital > 0)
        else float(_eff_capital)
    )

    # ── Step 4: バナー表示（実資産確定後）───────────────────────────────────
    # 優先順位: live_snap (broker) > capital_state (state file) > portfolio_state (last run) > None (config fallback)
    if _live_snap is not None:
        _banner_actual = float(_live_snap.actual_equity)
        _banner_eff    = _eff_capital
    elif _cap_state is not None:
        # broker API 不通だが state file あり → state file の actual_equity をフォールバック表示
        _banner_actual = float(_cap_state.actual_equity)
        _banner_eff    = _eff_capital
        logger.info("[CAPITAL_FALLBACK] banner using capital_state: actual=¥%s eff=¥%s",
                    f"{_banner_actual:,.0f}", f"{_banner_eff:,}")
    else:
        # 完全フォールバック: portfolio_state.json の last_equity を参照
        _banner_actual = None
        _banner_eff    = None
        try:
            import json as _json_banner
            _ps_file_banner = RUNTIME_DIR / "portfolio_state.json"
            if _ps_file_banner.exists():
                _ps_data = _json_banner.loads(_ps_file_banner.read_text(encoding="utf-8"))
                _ps_last_eq = float(_ps_data.get("last_equity", 0))
                if _ps_last_eq > 0:
                    _banner_actual = _ps_last_eq
                    _banner_eff    = _eff_capital
                    logger.info(
                        "[CAPITAL_FALLBACK] banner using portfolio_state.last_equity=¥%s"
                        " (broker API unavailable — displayed value may be stale)",
                        f"{_banner_actual:,.0f}",
                    )
        except Exception as _banner_fb_err:
            logger.debug("[CAPITAL_FALLBACK] portfolio_state fallback failed: %s", _banner_fb_err)

    print_banner(args.live, LIVE_UNIVERSE, universe_meta,
                 actual_equity=_banner_actual, eff_capital=_banner_eff)

    # ---- 株価上限フィルター（資本連動 = capital × 30%）----
    _max_alloc = int(_eff_capital * 0.30)  # live capital (was: static CAPITAL via _RISK_PARAMS)
    logger.info(
        "株価上限フィルター適用中（上限: ¥%s/単元 = capital×30%%）", f"{_max_alloc:,}"
    )
    # 保有中銘柄は price 問わず除外しない（broker API 接続前に portfolio_state から取得）
    _pf_held_syms: set[str] = set()
    try:
        _ps_pf_path = RUNTIME_DIR / "portfolio_state.json"
        if _ps_pf_path.exists():
            _ps_pf_raw = json.loads(_ps_pf_path.read_text(encoding="utf-8"))
            _pf_held_syms = set(_ps_pf_raw.get("position_qtys", {}).keys())
            if _pf_held_syms:
                logger.info(
                    "[PRICE_FILTER] 保有銘柄をフィルター除外対象から保護: %s",
                    sorted(_pf_held_syms),
                )
    except Exception as _pf_held_err:
        logger.debug("[PRICE_FILTER] portfolio_state 読み込みスキップ (FAIL_OPEN): %s", _pf_held_err)
    LIVE_UNIVERSE, price_skipped = filter_universe_by_price(
        LIVE_UNIVERSE, _max_alloc, held_symbols=_pf_held_syms
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

    # ── Universe determinism audit: post_price_filter snapshot ───────────────
    try:
        from src.runtime.universe_determinism_audit import record_universe_snapshot as _uda_record
        _uda_record(
            audit_dir         = UNIVERSE_DETERMINISM_AUDIT_DIR,
            mode              = "LIVE" if args.live else "DRY",
            snapshot_stage    = "post_price_filter",
            live_symbols      = LIVE_UNIVERSE.keys(),
            shadow_symbols    = SHADOW_UNIVERSE.keys(),
            tradeable_symbols = LIVE_UNIVERSE.keys(),
        )
    except Exception as _uda_err:
        logger.warning("[UNIVERSE_AUDIT] post_price_filter record failed (FAIL_OPEN): %s", _uda_err)

    # ── Deployable Universe Governance (after shadow eval, before signal gen) ──
    # Phase 5B.1: _eff_capital and _cap_state now loaded above (before price filter),
    # so governance affordability uses live effective_capital instead of static CAPITAL.
    # AUTO_PROMOTE_SAFE: evaluates shadow candidates and auto-promotes qualifying
    # symbols into the live universe. FAIL_OPEN for promotion failures.
    # FAIL_CLOSED for manifest corruption / replay inconsistency.
    _emit_phase("governance", "start", run_id=run_id)
    _gov_result = None
    try:
        from src.universe.deployable_universe_governance import (
            run_universe_governance,
            AUTO_PROMOTE_SAFE,
        )

        # ── RSR 取得: A=Snapshot → B=MTF fallback → C=block (FAIL_CLOSED) ────────
        # Governance runs BEFORE SignalBridge computes today's RSR (runtime ordering).
        # Yesterday's snapshot is deterministic, replayable, and audit-compatible.
        # MTF fallback is used only when snapshot is unavailable.
        # When both are unavailable, rsr_scores=None → promotions BLOCKED (FAIL_CLOSED).
        _gov_rsr_scores: dict[str, float] | None = None
        _gov_rsr_source: str = "missing"
        try:
            from src.paths import RSR_SNAPSHOT_DIR as _rsr_snap_dir
            if _rsr_snap_dir.exists():
                _snap_files = sorted(_rsr_snap_dir.glob("*.json"))
                if _snap_files:
                    _snap_data = json.loads(_snap_files[-1].read_text(encoding="utf-8"))
                    _loaded_snap = _snap_data.get("scores") or None
                    if _loaded_snap:
                        _gov_rsr_scores = _loaded_snap
                        _gov_rsr_source = "snapshot"
                        logger.info(
                            "[GOV] RSR snapshot=%s  %d symbols",
                            _snap_files[-1].name, len(_gov_rsr_scores),
                        )
        except Exception as _rsr_snap_err:
            _gov_rsr_scores = None
            logger.warning("[GOV] RSR snapshot load failed: %s", _rsr_snap_err)

        # MTF fallback: only when snapshot unavailable
        if _gov_rsr_scores is None:
            try:
                from src.paths import CACHE_DIR as _mtf_cache_dir
                _mtf_files = sorted(_mtf_cache_dir.glob("mtf_state_*.json"))
                if _mtf_files:
                    _mtf_data = json.loads(_mtf_files[-1].read_text(encoding="utf-8"))
                    _mtf_rsr = _mtf_data.get("rsr_weekly") or None
                    if _mtf_rsr:
                        _gov_rsr_scores = {str(k): float(v) for k, v in _mtf_rsr.items()}
                        _gov_rsr_source = "mtf"
                        logger.info(
                            "[GOV] RSR mtf fallback=%s  %d symbols",
                            _mtf_files[-1].name, len(_gov_rsr_scores),
                        )
            except Exception as _mtf_err:
                logger.warning("[GOV] RSR mtf fallback failed: %s", _mtf_err)

        if _gov_rsr_scores is None:
            logger.warning("[GOV] RSR unavailable (source=missing) — shadow promotions BLOCKED")

        # ── Phase 2: Universe EV Engine — replacement_delta gate ─────────────
        # Scores live and shadow by expected CAGR contribution, not RSR rank alone.
        # FAIL_OPEN: EV computation failure does not block promotion (reverts to
        # deployability-only scoring, same as before Phase 2).
        _candidate_ev_scores: dict[str, float] | None = None
        _live_ev_floor: float | None = None
        # Phase 2.5: initialize here so outcome tracker can access even if EV engine fails mid-run.
        _shadow_ev: dict = {}
        _live_ev: dict = {}
        try:
            from src.universe.universe_ev_engine import UniverseEVEngine
            from src.paths import RSR_SNAPSHOT_DIR as _ev_rsr_dir, OHLCV_DIR as _ev_ohlcv_dir
            _ev_engine = UniverseEVEngine(
                ohlcv_dir        = _ev_ohlcv_dir,
                trades_jsonl     = LOGS_DIR / "trades.jsonl",
                rsr_snapshot_dir = _ev_rsr_dir,
            )
            _shadow_ev = _ev_engine.compute_all_scores(SHADOW_UNIVERSE)
            _live_ev   = _ev_engine.compute_all_scores(LIVE_UNIVERSE)

            # Pass composite scores + sample_n (encoded as __n_{sym} key) to governance
            _candidate_ev_scores = {sym: s.composite for sym, s in _shadow_ev.items()}
            for sym, s in _shadow_ev.items():
                _candidate_ev_scores[f"__n_{sym}"] = float(s.sample_n)

            _live_ev_floor = _ev_engine.live_ev_floor(_live_ev)
            logger.info(
                "[GOV_EV] live_ev_floor=%.3f  shadow_candidates=%d  live=%d",
                _live_ev_floor, len(_shadow_ev), len(_live_ev),
            )
        except Exception as _ev_err:
            _candidate_ev_scores = None
            _live_ev_floor       = None
            logger.warning("[GOV_EV] EV engine failed (FAIL_OPEN, using deployability only): %s", _ev_err)

        _gov_result = run_universe_governance(
            live_universe       = LIVE_UNIVERSE,
            shadow_universe     = SHADOW_UNIVERSE,
            capital             = _eff_capital,
            run_id              = run_id,
            universe_dir        = UNIVERSE_RUNTIME_DIR,
            universe_file       = LIVE_UNIVERSE_FILE,      # updated atomically on promotion
            price_lookup        = _latest_local_close,     # uses local cache (no extra API call)
            rsr_scores          = _gov_rsr_scores,
            mode                = AUTO_PROMOTE_SAFE,
            candidate_ev_scores = _candidate_ev_scores,
            live_ev_floor       = _live_ev_floor,
            rsr_source          = _gov_rsr_source,
        )

        try:
            from src.universe.decision_context_capture import DecisionContextCapture as _DCC_promo
            from src.paths import DECISION_CONTEXT_HISTORY_FILE as _dcc_file
            _dcc_promo = _DCC_promo(history_file=_dcc_file)
            _dcc_promo.capture(
                "PROMOTION",
                universe_context={"live_universe_size": len(LIVE_UNIVERSE), "shadow_universe_size": len(SHADOW_UNIVERSE), "pending_exit_count": None},
                decision_inputs={"mode": str(AUTO_PROMOTE_SAFE), "candidate_count": len(SHADOW_UNIVERSE), "ev_floor": _live_ev_floor if "_live_ev_floor" in dir() else None},
                decision_result={"promoted_symbols": list(_gov_result.promoted_symbols) if _gov_result.promoted_symbols else [], "universe_size_after": _gov_result.governance_report.universe_size_after},
            )
        except Exception as _dcc_promo_err:
            logger.debug("[DCC] PROMOTION capture failed (FAIL_OPEN): %s", _dcc_promo_err)

        if _gov_result.promoted_symbols:
            LIVE_UNIVERSE = _gov_result.updated_live_universe   # effective this run
            logger.info(
                "[UNIVERSE_GOV] %d symbol(s) promoted: %s  universe=%d→%d  file_updated=%s",
                len(_gov_result.promoted_symbols),
                _gov_result.promoted_symbols,
                _gov_result.governance_report.universe_size_before,
                _gov_result.governance_report.universe_size_after,
                _gov_result.universe_file_updated,
            )
            print(
                f"\n[ユニバースガバナンス] {len(_gov_result.promoted_symbols)}銘柄 プロモーション: "
                f"{_gov_result.promoted_symbols}  "
                f"({_gov_result.governance_report.universe_size_before}→"
                f"{_gov_result.governance_report.universe_size_after}銘柄)"
            )
        else:
            logger.info(
                "[UNIVERSE_GOV] no promotions  breadth=%.2f  affordable_shadow=%d  high_price_excl=%d",
                _gov_result.governance_report.deployable_breadth_score,
                len([d for d in _gov_result.decisions if d.action == "OBSERVE"]),
                _gov_result.governance_report.high_price_exclusion_count,
            )
    except RuntimeError as _gov_fatal:
        # FAIL_CLOSED: manifest corruption / replay inconsistency
        logger.error("[UNIVERSE_GOV] FATAL governance error: %s", _gov_fatal)
        if args.live:
            print(f"\n[FATAL] ユニバースガバナンス整合性エラー: {_gov_fatal}", file=sys.stderr)
            return 1
        print(f"\n[UNIVERSE_GOV] ガバナンスエラー（ドライラン継続）: {_gov_fatal}")
    except Exception as _gov_err:
        # FAIL_OPEN: promotion errors do not abort execution
        logger.warning("[UNIVERSE_GOV] governance failed (%s) — proceeding with original universe", _gov_err)

    _emit_phase("governance", "complete", run_id=run_id)

    # ── Phase 2.5: Outcome Tracking — promotion / rotation / exploration ──────
    # Writes append-only JSONL records for each PROMOTE decision.
    # Forward returns are null at record time; materialized the next morning.
    # FAIL_OPEN: never blocks governance or execution.
    try:
        from src.universe.outcome_tracker import record_governance_outcomes as _rec_outcomes
        from src.paths import UNIVERSE_OUTCOME_DIR as _outcome_dir
        if _gov_result is not None and _gov_result.promoted_symbols:
            _rec_outcomes(
                gov_result       = _gov_result,
                shadow_ev_scores = _shadow_ev,
                live_ev_scores   = _live_ev,
                live_ev_floor    = _live_ev_floor,
                outcome_dir      = _outcome_dir,
                run_id           = run_id,
            )
    except Exception as _outcome_err:
        logger.warning("[OUTCOME] tracking failed (FAIL_OPEN): %s", _outcome_err)
        _emit_phase("governance", "outcome_tracking_failed", run_id=run_id,
                    extra={"outcome_tracking_error": str(_outcome_err)})

    # ── Universe determinism audit: post_governance snapshot ─────────────────
    try:
        from src.runtime.universe_determinism_audit import record_universe_snapshot as _uda_gov
        _uda_gov(
            audit_dir         = UNIVERSE_DETERMINISM_AUDIT_DIR,
            mode              = "LIVE" if args.live else "DRY",
            snapshot_stage    = "post_governance",
            live_symbols      = LIVE_UNIVERSE.keys(),
            shadow_symbols    = SHADOW_UNIVERSE.keys(),
            tradeable_symbols = LIVE_UNIVERSE.keys(),
        )
    except Exception as _uda_gov_err:
        logger.warning("[UNIVERSE_AUDIT] post_governance record failed (FAIL_OPEN): %s", _uda_gov_err)

    # ── Data Freshness Guard ──────────────────────────────────────────────────
    # LIVE: stale RSR (> 3 calendar days) → DataStalenessError → abort
    # DRY:  stale RSR → WARNING only, execution continues
    try:
        from src.universe.data_freshness_guard import (
            check_rsr_freshness as _check_rsr_fresh,
            DataStalenessError as _DataStalenessError,
        )
        from src.paths import RSR_SNAPSHOT_DIR as _fresh_rsr_dir
        _check_rsr_fresh(_fresh_rsr_dir, is_live=args.live)
    except ImportError:
        pass
    except Exception as _fresh_err:
        from src.universe.data_freshness_guard import DataStalenessError as _DSE
        if isinstance(_fresh_err, _DSE):
            raise   # LIVE stale-data abort — re-raise without wrapping
        logger.warning("[FRESHNESS] guard error (FAIL_OPEN): %s", _fresh_err)

    # ── Phase 2: Demotion Pressure Engine ─────────────────────────────────────
    # Scores held positions by capital occupancy inefficiency.
    # Output is a telemetry report only — no automatic demotion.
    # FAIL_OPEN: pressure computation failure does not affect execution.
    _dem_pressures: dict = {}
    try:
        from src.universe.demotion_pressure_engine import DemotionPressureEngine
        from src.paths import (
            RSR_SNAPSHOT_DIR as _dem_rsr_dir,
            OHLCV_DIR as _dem_ohlcv_dir,
            DEMOTION_PRESSURE_FILE as _dem_out,
        )
        _dem_engine = DemotionPressureEngine(
            portfolio_state_file = RUNTIME_DIR / "portfolio_state.json",
            trades_jsonl         = LOGS_DIR / "trades.jsonl",
            rsr_snapshot_dir     = _dem_rsr_dir,
            ohlcv_dir            = _dem_ohlcv_dir,
        )
        _dem_pressures = _dem_engine.compute_all()
        if _dem_pressures:
            _dem_engine.write_report(_dem_pressures, _dem_out)
            _dem_cands = [sym for sym, p in _dem_pressures.items() if p.is_candidate]
            if _dem_cands:
                logger.warning(
                    "[DEMOTION] capital occupancy pressure candidates: %s", _dem_cands
                )
                print(f"\n[資本効率警告] 保有効率低下候補: {_dem_cands}")
            else:
                logger.info("[DEMOTION] no demotion pressure candidates (all held positions healthy)")
    except Exception as _dem_err:
        logger.warning("[DEMOTION] pressure engine failed (FAIL_OPEN): %s", _dem_err)

    # ── Universe Shrink Engine ────────────────────────────────────────────────
    # Evaluates non-held live symbols for removal when live_size > 42.
    # Uses RSR persistence + Future Leader + Predictive Expansion (2+ sources).
    # Output merged into _dem_pressures → DemotionActuationEngine handles lifecycle.
    # FAIL_OPEN: any error leaves _shrink_pressures empty (no effect).
    _shrink_pressures: dict = {}
    try:
        from src.universe.universe_shrink_engine import UniverseShrinkEngine as _USE
        from src.paths import (
            RSR_SNAPSHOT_DIR              as _shrink_rsr_dir,
            FUTURE_LEADER_CANDIDATES_FILE as _shrink_fl_file,
            PREDICTIVE_SCORES_FILE        as _shrink_pe_file,
        )
        _held_syms = frozenset(
            _dem_act_ps.get("position_entry_dates", {}).keys()
            if "_dem_act_ps" in dir() else []
        )
        _shrink_engine = _USE(
            rsr_snapshot_dir   = _shrink_rsr_dir,
            fl_candidates_file = _shrink_fl_file,
            pe_scores_file     = _shrink_pe_file,
        )
        from src.universe.rotation_planner import TARGET_UNIVERSE_SIZE as _SHRINK_TARGET
        _shrink_pressures = _shrink_engine.compute(
            live_universe = LIVE_UNIVERSE,
            held_symbols  = _held_syms,
            target_size   = _SHRINK_TARGET,
        )
        _shrink_cands = [s for s, p in _shrink_pressures.items() if p.is_candidate]
        if _shrink_cands:
            logger.warning(
                "[SHRINK] live=%d target=%d  removal candidates: %s",
                len(LIVE_UNIVERSE), _SHRINK_TARGET, _shrink_cands,
            )
            print(f"\n[Universe縮退候補] {_shrink_cands}  (live={len(LIVE_UNIVERSE)}/target={_SHRINK_TARGET})")
    except Exception as _shrink_err:
        logger.warning("[SHRINK] engine failed (FAIL_OPEN): %s", _shrink_err)

    # ── Phase 3C enforcement: load prior health state before pipeline ────────────
    # Enforcement is derived from PREVIOUS run's governance_health.json.
    # Failure → WARNING enforcement (same as NORMAL: all True). FAIL_OPEN.
    _gov_enforce_allow_new_demotions     = True
    _gov_enforce_allow_demotion_updates  = True
    _gov_enforce_allow_rotations         = True
    _gov_enforce_allow_promotions        = True
    try:
        from src.universe.governance_health_engine import GovernanceHealthEngine as _GHE_enf
        from src.paths import (
            GOVERNANCE_HEALTH_FILE        as _ghf_enf,
            GOVERNANCE_MODE_HISTORY_FILE  as _ghmf_enf,
            GOVERNANCE_OVERRIDE_FILE      as _gov_override_enf,
        )
        _ghe_enf = _GHE_enf(
            governance_health_file=_ghf_enf,
            mode_history_file=_ghmf_enf,
            governance_override_file=_gov_override_enf,
        )
        _enf = _ghe_enf.load_enforcement()
        _gov_enforce_allow_new_demotions    = _enf.allow_new_demotions
        _gov_enforce_allow_demotion_updates = _enf.allow_demotion_state_updates
        _gov_enforce_allow_rotations        = _enf.allow_rotations
        _gov_enforce_allow_promotions       = _enf.allow_promotions
        if not _gov_enforce_allow_new_demotions:
            logger.warning(
                "[GOVERNANCE_HEALTH] enforcement: new_demotions=BLOCKED rotations=%s promotions=%s",
                _gov_enforce_allow_rotations, _gov_enforce_allow_promotions,
            )
    except Exception as _enf_err:
        logger.warning("[GOVERNANCE_HEALTH] enforcement load failed (FAIL_OPEN): %s", _enf_err)

    # ── Phase 3A: Demotion Actuation Engine ───────────────────────────────────
    # Drives demotion lifecycle: DEMOTION_CANDIDATE→PENDING_EXIT→EXIT_CONFIRMED→REMOVED
    # Universe exclusion only. No sell orders. FAIL_OPEN.
    try:
        from src.universe.demotion_actuation import DemotionActuationEngine as _DemActEngine
        from src.paths import (
            PENDING_DEMOTION_FILE as _dem_act_state_file,
            DEMOTION_ACTUATION_METRICS_FILE as _dem_act_metrics_file,
        )
        _dem_act_ps: dict = {}
        _dem_act_ps_path = RUNTIME_DIR / "portfolio_state.json"
        if _dem_act_ps_path.exists():
            try:
                _dem_act_ps = json.loads(_dem_act_ps_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        _dem_act_engine = _DemActEngine(
            pending_demotion_file=_dem_act_state_file,
            metrics_file=_dem_act_metrics_file,
        )
        _dem_act_result = _dem_act_engine.run(
            pressures={**_dem_pressures, **_shrink_pressures},   # merge: shrink supplements position pressure
            portfolio_state=_dem_act_ps,
            shadow_universe=SHADOW_UNIVERSE,
            live_universe=LIVE_UNIVERSE,
            ev_scores=_candidate_ev_scores if "_candidate_ev_scores" in dir() else None,
            allow_new_demotions=_gov_enforce_allow_new_demotions,
        )
        _dem_act_engine.write_state(_dem_act_result.transitions)
        _dem_act_engine.write_metrics(_dem_act_result)
        if _dem_act_result.symbols_to_remove:
            for _rm_sym in _dem_act_result.symbols_to_remove:
                LIVE_UNIVERSE.pop(_rm_sym, None)
            logger.info(
                "[DEMOTION_ACT] removed from live universe: %s",
                _dem_act_result.symbols_to_remove,
            )
            print(f"\n[降格確定] Universeから除外: {_dem_act_result.symbols_to_remove}")
            # Persist removal to LIVE_UNIVERSE_FILE (atomic write)
            try:
                _rm_live_data: dict = {}
                if LIVE_UNIVERSE_FILE.exists():
                    _rm_live_data = json.loads(LIVE_UNIVERSE_FILE.read_text(encoding="utf-8"))
                _rm_live_data.update({
                    "symbols":              {s: v for s, v in sorted(LIVE_UNIVERSE.items())},
                    "n_stocks":             len(LIVE_UNIVERSE),
                    "last_shrink_update":   datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "last_shrink_removed":  sorted(_dem_act_result.symbols_to_remove),
                })
                _rm_tmp = LIVE_UNIVERSE_FILE.with_suffix(".tmp")
                _rm_tmp.write_text(
                    json.dumps(_rm_live_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _rm_tmp.replace(LIVE_UNIVERSE_FILE)
                logger.info(
                    "[DEMOTION_ACT] universe file updated: %d symbols remain",
                    len(LIVE_UNIVERSE),
                )
            except Exception as _rm_persist_err:
                logger.warning(
                    "[DEMOTION_ACT] universe file persist failed (FAIL_OPEN): %s",
                    _rm_persist_err,
                )
        if _dem_act_result.demotion_candidates or _dem_act_result.pending_exit_count:
            logger.info(
                "[DEMOTION_ACT] candidates=%s pending_exit=%d exit_confirmed=%d recovered=%d",
                _dem_act_result.demotion_candidates,
                _dem_act_result.pending_exit_count,
                _dem_act_result.exit_confirmed_count,
                _dem_act_result.recovered_count,
            )
    except Exception as _dem_act_err:
        logger.warning("[DEMOTION_ACT] actuation engine failed (FAIL_OPEN): %s", _dem_act_err)

    try:
        from src.universe.decision_context_capture import DecisionContextCapture as _DCC_dem
        from src.paths import DECISION_CONTEXT_HISTORY_FILE as _dcc_file_dem
        _dcc_dem = _DCC_dem(history_file=_dcc_file_dem)
        _dem_r = _dem_act_result if "_dem_act_result" in dir() else None
        _dcc_dem.capture(
            "DEMOTION",
            universe_context={"live_universe_size": len(LIVE_UNIVERSE), "shadow_universe_size": len(SHADOW_UNIVERSE), "pending_exit_count": _dem_r.pending_exit_count if _dem_r else None},
            decision_inputs={"allow_new_demotions": _gov_enforce_allow_new_demotions, "live_universe_size": len(LIVE_UNIVERSE)},
            decision_result={"demotion_candidates": list(_dem_r.demotion_candidates) if _dem_r else [], "symbols_to_remove": list(_dem_r.symbols_to_remove) if _dem_r else [], "recovered_count": _dem_r.recovered_count if _dem_r else 0},
        )
    except Exception as _dcc_dem_err:
        logger.debug("[DCC] DEMOTION capture failed (FAIL_OPEN): %s", _dcc_dem_err)

    # ── Phase 3B: Rotation Planner ────────────────────────────────────────────
    # Builds deterministic rotation plan to maintain TARGET_UNIVERSE_SIZE.
    # Rejects stale candidates. Lifecycle: PROPOSED → APPROVED → EXECUTED.
    # Writes rotation_plan.json + rotation_plan_history.jsonl (outcome_status=PENDING).
    # FAIL_OPEN: any error logged, execution continues with original universe.
    # Enforcement gate: skipped when allow_rotations=False (SAFE_MODE/FREEZE).
    if _gov_enforce_allow_rotations:
        try:
            from src.universe.rotation_planner import RotationPlanner as _RotPlan
            from src.paths import (
                ROTATION_PLAN_FILE as _rot_plan_file,
                ROTATION_PLAN_HISTORY_FILE as _rot_history_file,
            )
            _rot_removes = (
                _dem_act_result.symbols_to_remove
                if "_dem_act_result" in dir()
                else []
            )
            # Derive candidate_scored_at from RSR snapshot filename (YYYY-MM-DD.json)
            _rot_scored_at: dict = {}
            try:
                from src.paths import RSR_SNAPSHOT_DIR as _rot_rsr_dir
                if _rot_rsr_dir.exists():
                    _rot_snaps = sorted(_rot_rsr_dir.glob("*.json"))
                    if _rot_snaps:
                        _rot_snap_date = _rot_snaps[-1].stem  # filename without .json
                        _rot_scored_iso = f"{_rot_snap_date}T09:00:00+09:00"
                        for _rsym in SHADOW_UNIVERSE:
                            _rot_scored_at[_rsym] = _rot_scored_iso
            except Exception:
                pass

            _rot_planner = _RotPlan(
                rotation_plan_file=_rot_plan_file,
                rotation_plan_history_file=_rot_history_file,
            )
            _rot_plan = _rot_planner.plan(
                live_universe=LIVE_UNIVERSE,
                shadow_universe=SHADOW_UNIVERSE,
                symbols_to_remove=_rot_removes,
                ev_scores=_candidate_ev_scores if "_candidate_ev_scores" in dir() else None,
                rsr_scores=_gov_rsr_scores if "_gov_rsr_scores" in dir() else None,
                candidate_scored_at=_rot_scored_at or None,
            )
            _rot_planner.write_plan(_rot_plan)
            _rot_planner.write_history(_rot_plan)
            if _rot_plan.symbols_to_add:
                logger.info(
                    "[ROTATION_PLANNER] plan=%s status=%s deficit=%d add=%s remove=%s score=%.3f",
                    _rot_plan.plan_id[:12], _rot_plan.plan_status,
                    _rot_plan.deficit, _rot_plan.symbols_to_add,
                    _rot_plan.symbols_to_remove, _rot_plan.rotation_score,
                )
                print(
                    f"\n[ローテーション計画] 追加候補: {_rot_plan.symbols_to_add}"
                    f"  (不足: {_rot_plan.deficit}スロット, score={_rot_plan.rotation_score:.3f})"
                )

            # ── Auto-approve + execute rotation plan ─────────────────────────────
            # PROPOSED → APPROVED → EXECUTED within the same run.
            # Guard: already inside _gov_enforce_allow_rotations=True.
            # Execution: add symbols_to_add to LIVE_UNIVERSE (in-memory + file).
            # FAIL_OPEN: any error logs and continues with original universe.
            try:
                from src.universe.rotation_planner import (
                    PLAN_STATUS_APPROVED as _ROTST_APPROVED,
                    PLAN_STATUS_EXECUTED as _ROTST_EXECUTED,
                )
                if _rot_planner.update_plan_status(_rot_plan.plan_id, _ROTST_APPROVED):
                    _rot_added: list = []
                    for _rot_sym in _rot_plan.symbols_to_add:
                        if _rot_sym not in LIVE_UNIVERSE and _rot_sym not in SHADOW_UNIVERSE:
                            continue
                        _rot_sec = SHADOW_UNIVERSE.get(_rot_sym, LIVE_UNIVERSE.get(_rot_sym, ""))
                        LIVE_UNIVERSE[_rot_sym] = _rot_sec
                        _rot_added.append(_rot_sym)
                    if _rot_added:
                        # Persist to LIVE_UNIVERSE_FILE atomically
                        import json as _rot_json
                        _rot_live_data: dict = {}
                        if LIVE_UNIVERSE_FILE.exists():
                            _rot_live_data = _rot_json.loads(
                                LIVE_UNIVERSE_FILE.read_text(encoding="utf-8")
                            )
                        _rot_base_ver = _rot_live_data.get("version", "rsr42_v1")
                        if "_rot" in _rot_base_ver:
                            _rot_base_ver = _rot_base_ver.split("_rot")[0]
                        _rot_live_data.update({
                            "version": f"{_rot_base_ver}_rot{len(_rot_added)}",
                            "symbols": {s: v for s, v in sorted(LIVE_UNIVERSE.items())},
                            "n_stocks": len(LIVE_UNIVERSE),
                            "last_rotation_update": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "last_rotation_added": sorted(_rot_added),
                        })
                        _rot_tmp = LIVE_UNIVERSE_FILE.with_suffix(".tmp")
                        _rot_tmp.write_text(
                            _rot_json.dumps(_rot_live_data, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        _rot_tmp.replace(LIVE_UNIVERSE_FILE)
                        logger.info(
                            "[ROTATION_PLANNER] EXECUTED plan=%s: added=%s",
                            _rot_plan.plan_id[:12], _rot_added,
                        )
                        print(f"\n[ローテーション実行] Live Universe追加: {_rot_added}")
                    _rot_planner.update_plan_status(_rot_plan.plan_id, _ROTST_EXECUTED)
            except Exception as _rot_exec_err:
                logger.warning("[ROTATION_PLANNER] approve/execute failed (FAIL_OPEN): %s", _rot_exec_err)

        except Exception as _rot_err:
            logger.warning("[ROTATION_PLANNER] planner failed (FAIL_OPEN): %s", _rot_err)

    # ── Shadow Universe Auto Refill ───────────────────────────────────────────
    # Replenishes shadow_universe.json when shadow-only count < MIN_SHADOW_SIZE.
    # Candidates: symbols in RSR/PE data but absent from live universe.
    # Ranked by predictive_alpha_score, filtered by RSR >= RSR_VIABILITY_MIN.
    # FAIL_OPEN: any error leaves shadow unchanged.
    try:
        from src.universe.shadow_refill import ShadowRefillEngine as _SRE
        from src.paths import (
            SHADOW_UNIVERSE_FILE  as _shadow_file_refill,
            PREDICTIVE_SCORES_FILE as _shadow_pe_file,
            RSR_SNAPSHOT_DIR       as _shadow_rsr_dir,
        )
        _sre = _SRE()
        if _sre.should_refill(SHADOW_UNIVERSE, LIVE_UNIVERSE):
            _refill_cands = _sre.compute_refill(
                shadow_universe  = SHADOW_UNIVERSE,
                live_universe    = LIVE_UNIVERSE,
                rsr_scores       = _gov_rsr_scores if "_gov_rsr_scores" in dir() else None,
                pe_scores_file   = _shadow_pe_file,
                rsr_snapshot_dir = _shadow_rsr_dir,
            )
            if _refill_cands:
                _refill_added = _sre.execute_refill(_shadow_file_refill, _refill_cands)
                if _refill_added:
                    for _radd_sym, _radd_sec in _refill_added:
                        SHADOW_UNIVERSE[_radd_sym] = _radd_sec
                    _shadow_only_new = sum(1 for s in SHADOW_UNIVERSE if s not in LIVE_UNIVERSE)
                    logger.info(
                        "[SHADOW_REFILL] added=%s  shadow_only=%d",
                        [s for s, _ in _refill_added], _shadow_only_new,
                    )
                    print(
                        f"\n[Shadow補充] 追加: {[s for s,_ in _refill_added]}"
                        f"  → shadow専用={_shadow_only_new}銘柄"
                    )
        else:
            _shadow_only_cnt = sum(1 for s in SHADOW_UNIVERSE if s not in LIVE_UNIVERSE)
            logger.info("[SHADOW_REFILL] shadow sufficient: shadow_only=%d", _shadow_only_cnt)
    except Exception as _sre_err:
        logger.warning("[SHADOW_REFILL] failed (FAIL_OPEN): %s", _sre_err)

    if _gov_enforce_allow_rotations:
        try:
            from src.universe.decision_context_capture import DecisionContextCapture as _DCC_rot
            from src.paths import DECISION_CONTEXT_HISTORY_FILE as _dcc_file_rot
            _dcc_rot = _DCC_rot(history_file=_dcc_file_rot)
            _rot_r = _rot_plan if "_rot_plan" in dir() else None
            _dcc_rot.capture(
                "ROTATION",
                universe_context={"live_universe_size": len(LIVE_UNIVERSE), "shadow_universe_size": len(SHADOW_UNIVERSE), "pending_exit_count": None},
                decision_inputs={"deficit": _rot_r.deficit if _rot_r else None, "candidate_count": len(SHADOW_UNIVERSE)},
                decision_result={"plan_id": _rot_r.plan_id if _rot_r else None, "symbols_to_add": list(_rot_r.symbols_to_add) if _rot_r else [], "rotation_score": _rot_r.rotation_score if _rot_r else None},
            )
        except Exception as _dcc_rot_err:
            logger.debug("[DCC] ROTATION capture failed (FAIL_OPEN): %s", _dcc_rot_err)
    else:
        logger.warning("[GOVERNANCE_HEALTH] rotation planner skipped: allow_rotations=False")

    # ── Phase 3C-A: Auto Freeze Engine ───────────────────────────────────────
    # Outcome-based freeze trigger (promotion/rotation/exploration alpha collapse).
    # Produces auto_freeze_mode passed to GovernanceHealthEngine.
    # FAIL_OPEN: engine errors → NORMAL auto_freeze_mode.
    _auto_freeze_state = None
    try:
        from src.universe.auto_freeze_engine import AutoFreezeEngine as _AFE
        from src.paths import AUTO_FREEZE_STATE_FILE as _afe_state_file
        _afe = _AFE(state_file=_afe_state_file)
        # OutcomeSummary inputs are None until Phase-3C data pipeline is wired.
        # Engine returns NORMAL mode when all inputs are None.
        _auto_freeze_state = _afe.run(
            promotion_outcomes=None,
            rotation_outcomes=None,
            exploration_outcomes=None,
            turnover_stats=None,
        )
        _afe.write_state(_auto_freeze_state)
        if _auto_freeze_state.triggered_checks:
            logger.warning(
                "[AUTO_FREEZE] mode=%s score=%.0f triggered=%s",
                _auto_freeze_state.auto_freeze_mode,
                _auto_freeze_state.degradation_score,
                _auto_freeze_state.triggered_checks,
            )
        else:
            logger.info("[AUTO_FREEZE] mode=%s no_triggers", _auto_freeze_state.auto_freeze_mode)
    except Exception as _afe_err:
        logger.warning("[AUTO_FREEZE] engine failed (FAIL_OPEN): %s", _afe_err)

    try:
        from src.universe.decision_context_capture import DecisionContextCapture as _DCC_afe
        from src.paths import DECISION_CONTEXT_HISTORY_FILE as _dcc_file_afe
        _dcc_afe = _DCC_afe(history_file=_dcc_file_afe)
        _af_r = _auto_freeze_state if "_auto_freeze_state" in dir() else None
        _dcc_afe.capture(
            "AUTO_FREEZE",
            universe_context={"live_universe_size": len(LIVE_UNIVERSE), "shadow_universe_size": len(SHADOW_UNIVERSE), "pending_exit_count": None},
            decision_inputs={"promotion_outcomes": None, "rotation_outcomes": None, "exploration_outcomes": None},
            decision_result={"auto_freeze_mode": _af_r.auto_freeze_mode if _af_r else None, "degradation_score": _af_r.degradation_score if _af_r else None, "triggered_checks": list(_af_r.triggered_checks) if _af_r else []},
        )
    except Exception as _dcc_afe_err:
        logger.debug("[DCC] AUTO_FREEZE capture failed (FAIL_OPEN): %s", _dcc_afe_err)

    # effective_mode が実行経路で消費されていないことを常時 warning で可視化する。
    # この警告は設計上の配線漏れを示す（動作変更なし）。
    logger.warning(
        "[AUTO_FREEZE_INACTIVE] effective_mode has no execution consumers "
        "(computed by GovernanceHealthEngine but not wired to SignalBridge). "
        "Auto Freeze does not affect order generation."
    )

    # ── Phase 3C: Governance Health Engine ────────────────────────────────────
    # Monitors governance pipeline health and manages operational mode ladder.
    # FAIL_OPEN: health engine errors never affect execution.
    _gov_health_state = None
    try:
        from src.universe.governance_health_engine import GovernanceHealthEngine as _GHE
        from src.paths import (
            GOVERNANCE_HEALTH_FILE       as _ghf,
            GOVERNANCE_MODE_HISTORY_FILE as _ghmf,
            GOVERNANCE_OVERRIDE_FILE     as _gov_override_file,
        )
        _ghe = _GHE(
            governance_health_file=_ghf,
            mode_history_file=_ghmf,
            governance_override_file=_gov_override_file,
        )
        _gov_health_state = _ghe.run(
            live_universe=LIVE_UNIVERSE,
            shadow_universe=SHADOW_UNIVERSE,
            demotion_result=_dem_act_result if "_dem_act_result" in dir() else None,
            rotation_plan=_rot_plan if "_rot_plan" in dir() else None,
            auto_freeze_state=_auto_freeze_state if "_auto_freeze_state" in dir() else None,
        )
        _ghe.write_state(_gov_health_state)
        logger.info(
            "[GOVERNANCE_HEALTH] mode=%s score=%.1f confirmed=%s pending=%s",
            _gov_health_state.governance_mode,
            _gov_health_state.health_score,
            _gov_health_state.mode_confirmed,
            _gov_health_state.pending_mode,
        )
        if _gov_health_state.governance_mode not in ("NORMAL", "WARNING"):
            print(
                f"\n[ガバナンス健全性] mode={_gov_health_state.governance_mode}"
                f"  score={_gov_health_state.health_score:.1f}"
                f"  degraded={[k for k, v in _gov_health_state.checks.items() if v.status == 'DEGRADED']}"
            )
    except Exception as _ghe_err:
        # Import or constructor failure: engine.run() handles internal errors itself.
        # If we get here, even the engine instantiation failed → WARNING (NORMAL禁止).
        logger.warning("[GOVERNANCE_HEALTH] engine import/init failed (WARNING): %s", _ghe_err)
        try:
            from src.universe.governance_health_engine import (
                GovernanceHealthEngine as _GHE_warn,
                MODE_WARNING as _ghe_warn_mode,
            )
            _ghe_warn = _GHE_warn.__new__(_GHE_warn)
            _gov_health_state = _ghe_warn._warning_fallback(str(_ghe_err))
            # Best-effort write to governance_health.json
            _ghe_warn._health_file = _ghf if "_ghf" in dir() else None
            if _ghe_warn._health_file:
                _ghe_warn.write_state(_gov_health_state)
        except Exception:
            pass

    try:
        from src.universe.decision_context_capture import DecisionContextCapture as _DCC_gov
        from src.paths import DECISION_CONTEXT_HISTORY_FILE as _dcc_file_gov
        _dcc_gov = _DCC_gov(history_file=_dcc_file_gov)
        _gh_r = _gov_health_state if "_gov_health_state" in dir() else None
        _dcc_gov.capture(
            "GOVERNANCE",
            universe_context={"live_universe_size": len(LIVE_UNIVERSE), "shadow_universe_size": len(SHADOW_UNIVERSE), "pending_exit_count": None},
            governance_context={"health_score": _gh_r.health_score if _gh_r else None, "governance_mode": _gh_r.governance_mode if _gh_r else None, "auto_freeze_mode": _gh_r.auto_freeze_mode if _gh_r else None, "effective_mode": _gh_r.effective_mode if _gh_r else None},
            decision_inputs={"checks_run": list(_gh_r.checks.keys()) if _gh_r and _gh_r.checks else []},
            decision_result={"governance_mode": _gh_r.governance_mode if _gh_r else None, "effective_mode": _gh_r.effective_mode if _gh_r else None, "mode_confirmed": _gh_r.mode_confirmed if _gh_r else None, "health_score": _gh_r.health_score if _gh_r else None},
        )
    except Exception as _dcc_gov_err:
        logger.debug("[DCC] GOVERNANCE capture failed (FAIL_OPEN): %s", _dcc_gov_err)

    # ── AUTO_PROMOTE_SAFE_V2: Probation-based deployment gate ─────────────────
    # Checks shadow candidates against 3-condition gate (RSR/predictive/ignition).
    # Qualifying symbols enter probation universe with 0.25x allocation cap.
    # Also runs daily outcome observation for existing probation symbols.
    # FAIL_OPEN: any error is logged and execution proceeds with original universe.
    _probation_active_symbols: set = set()
    try:
        from src.universe.auto_promote_safe_v2 import (
            run_probation_gate,
            run_probation_outcome_observation,
            load_latest_predictive_scores as _v2_load_pred,
            load_latest_fl_candidates as _v2_load_fl,
        )
        from src.paths import PROBATION_PROMOTIONS_FILE, PROBATION_OUTCOMES_FILE, PROBATION_REJECTION_FILE

        _v2_pred_scores: dict = {}
        try:
            _v2_pred_scores = _v2_load_pred(PREDICTIVE_SCORES_FILE)
        except Exception as _v2_pe:
            logger.warning("[V2] predictive scores load failed: %s", _v2_pe)

        _v2_fl_cands: dict = {}
        try:
            _v2_fl_cands = _v2_load_fl(FUTURE_LEADER_CANDIDATES_FILE)
        except Exception as _v2_fle:
            logger.warning("[V2] FL candidates load failed: %s", _v2_fle)

        _v2_cb_state = "NORMAL"
        try:
            _v2_ps_file = RUNTIME_DIR / "portfolio_state.json"
            if _v2_ps_file.exists():
                _v2_cb_state = json.loads(
                    _v2_ps_file.read_text(encoding="utf-8")
                ).get("cb_state", "NORMAL")
        except Exception:
            pass

        run_probation_outcome_observation(
            probation_path=PROBATION_PROMOTIONS_FILE,
            outcomes_path=PROBATION_OUTCOMES_FILE,
            rsr_scores=_gov_rsr_scores or {},
            run_id=run_id,
        )

        _probation_active_symbols, _probation_newly = run_probation_gate(
            shadow_universe=SHADOW_UNIVERSE,
            live_universe=LIVE_UNIVERSE,
            cb_state=_v2_cb_state,
            rsr_scores=_gov_rsr_scores or {},
            predictive_scores=_v2_pred_scores,
            fl_candidates=_v2_fl_cands,
            probation_path=PROBATION_PROMOTIONS_FILE,
            outcomes_path=PROBATION_OUTCOMES_FILE,
            run_id=run_id,
            rejection_path=PROBATION_REJECTION_FILE,
        )

        if _probation_active_symbols:
            for _ps in _probation_active_symbols:
                if _ps not in LIVE_UNIVERSE and _ps in SHADOW_UNIVERSE:
                    LIVE_UNIVERSE[_ps] = SHADOW_UNIVERSE[_ps]
            logger.info(
                "[V2] probation in-memory: %d symbol(s): %s",
                len(_probation_active_symbols), sorted(_probation_active_symbols),
            )
            if _probation_newly:
                print(
                    f"\n[AUTO_PROMOTE_V2] 試用昇格: {sorted(_probation_newly)}"
                    f"  (試用中合計: {len(_probation_active_symbols)}/{3}枠)"
                )
    except Exception as _v2_err:
        logger.warning("[V2] probation gate failed (FAIL_OPEN): %s", _v2_err)

    # ── AUTO_PROMOTE_SAFE_V2: graduated → LIVE_UNIVERSE_FILE ─────────────────
    # Graduated symbols (STATUS_GRADUATED) are not in active_probation_symbols.
    # They must be persisted to LIVE_UNIVERSE_FILE to survive the next run.
    # FAIL_OPEN: write errors logged; next run will retry (record stays GRADUATED).
    try:
        from src.universe.auto_promote_safe_v2 import get_graduated_symbols as _v2_get_grads
        _grad_syms = _v2_get_grads(PROBATION_PROMOTIONS_FILE, live_universe=LIVE_UNIVERSE)
        if _grad_syms:
            _grad_added: list = []
            for _g_sym in sorted(_grad_syms):
                _g_sec = SHADOW_UNIVERSE.get(_g_sym, LIVE_UNIVERSE.get(_g_sym, ""))
                LIVE_UNIVERSE[_g_sym] = _g_sec
                _grad_added.append(_g_sym)
            if _grad_added:
                _grad_live_data: dict = {}
                if LIVE_UNIVERSE_FILE.exists():
                    _grad_live_data = json.loads(LIVE_UNIVERSE_FILE.read_text(encoding="utf-8"))
                _grad_base_ver = _grad_live_data.get("version", "rsr42_v1")
                if "_grad" in _grad_base_ver:
                    _grad_base_ver = _grad_base_ver.split("_grad")[0]
                _grad_live_data.update({
                    "version": f"{_grad_base_ver}_grad{len(_grad_added)}",
                    "symbols": {s: v for s, v in sorted(LIVE_UNIVERSE.items())},
                    "n_stocks": len(LIVE_UNIVERSE),
                    "last_graduation_update": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "last_graduated_symbols": sorted(_grad_added),
                })
                _grad_tmp = LIVE_UNIVERSE_FILE.with_suffix(".tmp")
                _grad_tmp.write_text(
                    json.dumps(_grad_live_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _grad_tmp.replace(LIVE_UNIVERSE_FILE)
                logger.info("[V2_GRAD] graduated→LIVE persisted: %s", _grad_added)
                print(f"\n[AUTO_PROMOTE_V2] 卒業→LIVE昇格(永続化): {_grad_added}")
    except Exception as _v2_grad_err:
        logger.warning("[V2_GRAD] graduation→LIVE failed (FAIL_OPEN): %s", _v2_grad_err)

    from src.kabusapi.signal_bridge import SignalBridge

    # ─── CAPITAL LAYER ─────────────────────────────────────────────────────────
    # Phase 5B.1: capital state loaded above (before universe price filter and
    # governance). All state variables (_cap_state, _ramp_state, etc.) already set.

    # ── Phase 5A: load allocation states ─────────────────────────────────────
    _alloc_state_loaded = False
    _exposure_state = None
    _stability_window_state = None
    _lifecycle_map: dict = {}
    _alloc_state_load_time = 0.0
    try:
        import time as _time_alloc
        from src.allocation import (
            load_exposure_state, load_stability_window, build_lifecycle_map,
        )
        from src.paths import (
            EXPOSURE_STATE_FILE, STABILITY_WINDOW_FILE, ALPHA_LIFECYCLE_DIR,
            ALLOCATION_INTENT_FILE,
        )
        _exposure_state = load_exposure_state(EXPOSURE_STATE_FILE)
        _stability_window_state = load_stability_window(STABILITY_WINDOW_FILE)
        _lifecycle_map = build_lifecycle_map(ALPHA_LIFECYCLE_DIR, list(LIVE_UNIVERSE.keys()))
        _alloc_state_load_time = _time_alloc.time()
        _alloc_state_loaded = True
        logger.info(
            "[ALLOC] state loaded: exposure_symbols=%d stability_obs=%d lifecycle=%d",
            len(_exposure_state.return_buffers),
            len(_stability_window_state.rank_snapshots),
            len(_lifecycle_map),
        )
    except Exception as _alloc_load_err:
        logger.warning("[ALLOC] state load failed (%s) — fallback sizing active", _alloc_load_err)

    # ── Stale visibility: emit warnings for stale allocator state ────────────
    try:
        import time as _time_sv
        from src.allocation.stale_visibility import emit_stale_warnings as _emit_sv
        from src.paths import STALE_VISIBILITY_FILE as _sv_file
        _sv_now = _time_sv.time()
        _sv_ts = datetime.now(JST).isoformat()
        _sv_load_time = _alloc_state_load_time if _alloc_state_loaded else 0.0
        _sv_fallback = not _alloc_state_loaded
        _sv_reason = "alloc_state_load_failed" if not _alloc_state_loaded else ""
        _sv_emitted = _emit_sv(
            exposure_load_time=_sv_load_time,
            lifecycle_load_time=_sv_load_time,
            stability_load_time=_sv_load_time,
            now_sec=_sv_now,
            timestamp=_sv_ts,
            path=_sv_file,
            fallback_mode=_sv_fallback,
            fallback_reason=_sv_reason,
        )
        if _sv_emitted:
            logger.warning(
                "[STALE_VIS] %d warning(s): %s",
                len(_sv_emitted),
                [w.warning_type for w in _sv_emitted],
            )
    except Exception as _sv_err:
        logger.warning("[STALE_VIS] check failed (%s) — continuing", _sv_err)

    # ── Analytics policy bridge: apply bounded runtime policy adjustments ────────
    # Fail-open: any error leaves runtime params unchanged.
    # No strategy mutation — observational analytics → bounded overlays only.
    try:
        from src.runtime.policy.analytics_policy_bridge import run_policy_hook
        from src.paths import POLICY_DECISIONS_FILE
        _policy_params = run_policy_hook(
            store_path=POLICY_DECISIONS_FILE,
            base_params={
                "slot_allocation":           1.0,
                "sector_exposure_cap":       float(MAX_SINGLE_WEIGHT),
                "portfolio_heat_limit":      float(MAX_DD_LIMIT),
                "hold_extension_eligibility": 0.0,
                "entry_throttling":          1.0,
                "risk_scaling":              1.0,
            },
        )
        logger.info("[POLICY_BRIDGE] effective_params=%s", _policy_params)
    except Exception as _policy_err:
        logger.warning("[POLICY_BRIDGE] hook failed (%s) — using base params", _policy_err)
        _policy_params = {}

    # ── [RC] D_VOL_ADJ: dynamic max_positions from TOPIX 20d volatility ─────────
    # Study41 production candidate. Expands max_positions to 4 on calm days.
    # FAIL_OPEN: defaults to MAX_POS=3 on error. Default OFF (strategy.yaml).
    _rc_max_pos = MAX_POS
    try:
        _rc_vol_cfg = getattr(getattr(cfg, "research_candidates", None), "vol_adj", None)
        if _rc_vol_cfg and getattr(_rc_vol_cfg, "enabled", False):
            from src.research_candidate.vol_adj import compute_effective_max_positions
            from src.paths import VOL_ADJ_STATE_FILE as _va_state_f, BACKTEST_DATASET_DIR as _va_data
            from src.paths import DEFAULT_DATA_VERSION as _va_ver
            _va_data_dir = _va_data / (_va_ver or "")
            _rc_max_pos = compute_effective_max_positions(
                data_dir       = _va_data_dir,
                default_max_pos = int(MAX_POS),
                calm_max_pos   = int(getattr(_rc_vol_cfg, "calm_max_positions", 4)),
                vol_threshold  = float(getattr(_rc_vol_cfg, "topix_vol_threshold", 0.008)),
                state_path     = _va_state_f,
            )
            logger.info("[RC_VOL_ADJ] effective_max_pos=%d (base=%d)", _rc_max_pos, MAX_POS)
    except Exception as _rc_va_err:
        logger.warning("[RC_VOL_ADJ] failed (%s) — using MAX_POS=%d", _rc_va_err, MAX_POS)
        _rc_max_pos = MAX_POS

    try:
        bridge = SignalBridge(
            universe_tickers          = LIVE_UNIVERSE,
            rsr_universe_tickers      = RSR_UNIVERSE_62,  # RSR62コンテキスト（42+shadow20）
            shadow_universe_tickers   = SHADOW_UNIVERSE or None,  # 監視用（RSR計算兼用）
            fujiko_params             = {
                "min_sepa": cfg.fujiko.min_sepa,
                "min_rsr": cfg.fujiko.min_rsr,           # entry 専用 (75, 変更禁止)
                "rsr_exit": cfg.fujiko.rsr_exit,          # exit 専用 (70)
                "mom_period": cfg.fujiko.mom_period,
                "turtle_entry": cfg.fujiko.turtle_entry,
                "turtle_exit": cfg.fujiko.turtle_exit,
                "use_turtle_entry": cfg.fujiko.use_turtle_entry,
            },
            capital                   = _eff_capital,
            max_positions             = min(_rc_max_pos, MAX_OPEN_POSITIONS),
            max_single_weight         = MAX_SINGLE_WEIGHT,
            max_dd_limit              = MAX_DD_LIMIT,
            min_sectors               = MIN_SECTORS,
            live                      = args.live,
            top_k                     = TOP_K,
            max_hold_days             = MAX_HOLD_DAYS,
            min_hold_days             = MIN_HOLD_DAYS,
            emergency_exit_pct        = EMERGENCY_EXIT_PCT,
            max_new_positions_per_day = MAX_NEW_POS_PER_DAY,
            shock_exit_mode           = cfg.risk_controls.shock_exit_mode,
            regime_sizing             = cfg.risk_controls.regime_sizing,
            bear_scale                = cfg.risk_controls.bear_scale,
            deployable_capital        = _cdos_deployable,
            entry_freeze_enabled      = cfg.entry_freeze.enabled,
            entry_freeze_reason       = cfg.entry_freeze.reason,
            require_broker            = not args.allow_no_broker,
        )
    except Exception as exc:
        print(f"[FATAL] SignalBridge 初期化失敗: {exc}", file=sys.stderr)
        return 1

    # ── StagedSupervisor: per-stage timeouts & watchdog JSONL ────────────────
    # 各ステージを ThreadPoolExecutor でラップし無期限ブロックを排除する。
    # bridge.run() は yfinance / kabu API を内部で呼ぶため 150 s でタイムアウト。
    _supervisor = StagedSupervisor(
        run_id=run_id,
        watchdog_path=WATCHDOG_LOG_FILE,
        lock_owner_pid=os.getpid(),
        heartbeat_lock=_lock_instance,
    )
    _exec_journal = ExecutionJournal(ORDERS_JOURNAL_DIR)

    logger.info("シグナル生成開始...")
    _exit_code = 0
    _emit_phase("signal_generation", "start", run_id=run_id)
    try:
        with _supervisor:
            # ── Stage: market_data (combined fetch + signal generation) ──────
            try:
                result, order_objects = _supervisor.run_stage(
                    "market_data",
                    bridge.run,
                    timeout_sec=150,   # 60 s data fetch + 90 s signal compute
                    retry_budget=0,
                )
            except StageTimeout as _ste:
                print(f"\n[FATAL] シグナル生成タイムアウト: {_ste}", file=sys.stderr)
                _exit_code = 1
                _supervisor.emit_terminal("abort", reason=str(_ste))
                return _exit_code
            except StageError as _se:
                from src.kabusapi.signal_bridge import AbortError as _AbortError
                if isinstance(_se.cause, _AbortError):
                    # Broker-as-Sole-SSOT (2026-07-18): bridge.run() が broker
                    # snapshot取得失敗等でAbortErrorをraiseした場合、
                    # run_morning_signal.py（廃止済み）が持っていたEMERGENCY_STOP
                    # パターンと同じ構造化ログ+exit 1で停止する。
                    logger.critical(
                        "EMERGENCY_STOP: reason=%s detail=%s",
                        _se.cause.reason, _se.cause,
                    )
                    print(f"\n[EMERGENCY_STOP] reason={_se.cause.reason}: {_se.cause}", file=sys.stderr)
                else:
                    print(f"\n[FATAL] シグナル生成失敗: {_se}", file=sys.stderr)
                _exit_code = 1
                return _exit_code

            # ── [RC] ATR Extension: defer RSR exits near highest close ────────
            # Study40 production candidate. Post-filters SELL orders from bridge.run().
            # FAIL_OPEN: leaves order_objects unchanged on error. Default OFF.
            try:
                _rc_atr_cfg = getattr(getattr(cfg, "research_candidates", None), "atr_extension", None)
                if _rc_atr_cfg and getattr(_rc_atr_cfg, "enabled", False):
                    from src.research_candidate.atr_extension import filter_atr_extension_sells
                    from src.paths import ATR_EXT_DEFER_STATE_FILE as _atr_state_f
                    _ps_atr: dict = {}
                    try:
                        import json as _json_atr
                        _ps_atr_path = RUNTIME_DIR / "portfolio_state.json"
                        if _ps_atr_path.exists():
                            _ps_atr = _json_atr.loads(_ps_atr_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                    order_objects, _rc_n_deferred = filter_atr_extension_sells(
                        order_objects  = list(order_objects),
                        result_signals = list(result.signals),
                        portfolio_state = _ps_atr,
                        state_path     = _atr_state_f,
                        today          = _trading_day,
                        atr_mult       = float(getattr(_rc_atr_cfg, "atr_mult", 1.0)),
                        max_defer_days = int(getattr(_rc_atr_cfg, "max_defer_calendar_days", 7)),
                    )
                    if _rc_n_deferred > 0:
                        logger.info("[RC_ATR_EXT] %d RSR exit(s) deferred", _rc_n_deferred)
                        print(f"\n[ATR Extension] {_rc_n_deferred}件のRSR出口を延期しました")
            except Exception as _rc_atr_err:
                logger.warning("[RC_ATR_EXT] failed (%s) — continuing without filter", _rc_atr_err)

            # ── Market data guard: check freshness of received data ──────────
            try:
                from src.runtime.market_data_guard import run_market_data_guard, append_guard_report
                from src.paths import MARKET_DATA_GUARD_FILE as _mdg_file
                import time as _time_mdg
                _mdg_symbols = list(LIVE_UNIVERSE.keys())
                _mdg_received = list(getattr(result, "received_symbols", None) or [])
                _mdg_report = run_market_data_guard(
                    expected_symbols=_mdg_symbols,
                    received_symbols=_mdg_received if _mdg_received else None,
                    now_sec=_time_mdg.time(),
                )
                append_guard_report(_mdg_report, _mdg_file)
                if _mdg_report.halt_recommended:
                    logger.warning("[MARKET_DATA_GUARD] %s", _mdg_report.halt_reason)
            except Exception as _mdg_err:
                logger.warning("[MARKET_DATA_GUARD] check failed (%s) — continuing", _mdg_err)

            # ── Broker reconciliation: snapshot + integrity gate ─────────────
            # Fail-open for snapshot/reconciliation; fail-closed gate is OUTSIDE
            # the try/except so live execution is blocked on integrity failure.
            _integrity_result = None
            _run_manifest = None   # deployment authority manifest (Task 1)
            try:
                from src.live.broker_truth_snapshot import take_snapshot, append_snapshot
                from src.live.reconciliation_engine import run_reconciliation, append_reconciliation_result
                from src.live.execution_integrity_validator import (
                    validate_execution_integrity, append_integrity_result,
                )
                from src.paths import (
                    BROKER_SNAPSHOT_FILE as _bsf,
                    RECONCILIATION_LOG_FILE as _rcf,
                    EXECUTION_INTEGRITY_FILE as _eif,
                )
                import time as _time_recon

                # Snapshot: immutable broker state at this moment
                _ps_recon = result.portfolio_summary or {}
                _broker_snapshot = take_snapshot(_ps_recon, run_id=run_id)
                append_snapshot(_broker_snapshot, _bsf)

                # Extract broker positions best-effort (symbol → qty from API)
                _broker_pos: dict = {}
                _pos_api = _ps_recon.get("positions_api", {})
                if isinstance(_pos_api, dict):
                    # Try common keys: "holdings", "positions", "items"
                    for _pos_key in ("holdings", "positions", "items", "data"):
                        _holdings = _pos_api.get(_pos_key)
                        if isinstance(_holdings, dict):
                            _broker_pos = {
                                str(k): int(v) for k, v in _holdings.items()
                                if str(v).lstrip("-").isdigit()
                            }
                            break
                        if isinstance(_holdings, list):
                            for _h in _holdings:
                                if isinstance(_h, dict) and "symbol" in _h and "qty" in _h:
                                    try:
                                        _broker_pos[str(_h["symbol"])] = int(_h["qty"])
                                    except (TypeError, ValueError):
                                        pass
                            if _broker_pos:
                                break

                # Runtime positions: symbols currently held (from signals)
                _runtime_pos: dict = {
                    _s.get("symbol", ""): 1
                    for _s in result.signals
                    if _s.get("currently_holding") and _s.get("symbol")
                }

                # Inflight orders: all from registry (empty list if no registry)
                _all_inflight = (
                    _inflight_registry.all_orders()
                    if _inflight_registry is not None else []
                )

                _recon_result = run_reconciliation(
                    broker_positions=_broker_pos,
                    runtime_positions=_runtime_pos,
                    inflight_orders=_all_inflight,
                    run_id=run_id,
                    now_sec=_time_recon.time(),
                )
                append_reconciliation_result(_recon_result, _rcf)

                # RC4 FIX: Purge phantom positions — symbols where runtime
                # reports holding (qty=1) but broker reports flat (qty=0) with
                # no inflight SELL orders.  Broker is authoritative; stale
                # portfolio_state.json entries cause permanent reconciliation
                # failures and false signals.  Only purge when broker data is
                # actually available (non-empty _broker_pos from API).
                if _broker_pos:
                    try:
                        _inflight_syms_sell: set = {
                            getattr(_o, "symbol", "")
                            for _o in _all_inflight
                            if str(getattr(_o, "side", "")).upper() == "SELL"
                            and str(getattr(_o, "state", "")).upper()
                               in ("PENDING_SUBMIT", "SUBMITTED_UNKNOWN")
                        }
                        _phantom_syms = [
                            sym for sym, rq in _runtime_pos.items()
                            if rq > 0
                            and _broker_pos.get(sym, 0) == 0
                            and sym not in _inflight_syms_sell
                        ]
                        if _phantom_syms:
                            logger.warning(
                                "[RECON] phantom positions detected — purging from state: %s",
                                _phantom_syms,
                            )
                            bridge.overwrite_local_positions(_broker_pos)
                            logger.info(
                                "[RECON] portfolio_state reconciled to broker truth: "
                                "held=%s purged=%s",
                                sorted(_broker_pos.keys()), _phantom_syms,
                            )
                    except Exception as _phantom_err:
                        logger.warning("[RECON] phantom purge failed (%s) — continuing", _phantom_err)

                _integrity_result = validate_execution_integrity(
                    _recon_result,
                    _broker_snapshot,
                    is_live=args.live,
                    now_sec=_time_recon.time(),
                )
                append_integrity_result(_integrity_result, _eif)

                # ── Execution lineage + epoch governance (Task 1) ─────────────
                try:
                    from src.live.execution_epoch import (
                        ReconciliationEpoch as _RE, DeploymentEpoch as _DE,
                        build_authority_chain as _build_chain,
                        append_reconciliation_epoch as _app_re,
                        append_deployment_epoch as _app_de,
                        append_authority_chain as _app_chain,
                        AUTHORITY_LEVEL_FULL as _AL_FULL,
                        AUTHORITY_LEVEL_BLOCKED as _AL_BLOCKED,
                    )
                    from src.live.deployment_lineage import (
                        build_lineage as _build_lin,
                        append_lineage_step as _app_step,
                        persist_lineage as _persist_lin,
                        STEP_BROKER_SNAPSHOT as _ST_BS,
                        STEP_RECONCILIATION as _ST_RC,
                        STEP_INTEGRITY_VALIDATION as _ST_IV,
                        STEP_AUTHORITY_DECISION as _ST_AD,
                        STEP_DEPLOYMENT_DECISION as _ST_DD,
                    )
                    from src.live.deployment_authority_manifest import (
                        build_manifest as _build_mf,
                        append_manifest as _app_mf,
                    )

                    _bs_hash = _broker_snapshot.checksum
                    _integrity_ok = _integrity_result.execution_allowed
                    _authority_level = _AL_FULL if _integrity_ok else _AL_BLOCKED

                    _recon_epoch = _RE.create(
                        run_id=run_id,
                        broker_snapshot_hash=_bs_hash,
                        reconciliation_ok=_recon_result.reconciliation_ok,
                        blocking_mismatches=_recon_result.blocking_mismatches,
                    )
                    _app_re(_recon_epoch, EXECUTION_EPOCH_FILE)

                    _dep_epoch = _DE.create(
                        run_id=run_id,
                        reconciliation_epoch_id=_recon_epoch.epoch_id,
                        broker_snapshot_hash=_bs_hash,
                        deployment_allowed=_integrity_ok,
                        authority_level=_authority_level,
                    )
                    _app_de(_dep_epoch, DEPLOYMENT_EPOCH_FILE)

                    _epoch_chain = _build_chain(_recon_epoch, _dep_epoch)
                    _app_chain(_epoch_chain, AUTHORITY_CHAIN_FILE)

                    _run_lineage = _build_lin(run_id)
                    _app_step(_run_lineage, _ST_BS, {
                        "snapshot_checksum": _bs_hash, "run_id": run_id,
                    })
                    _app_step(_run_lineage, _ST_RC, {
                        "reconciliation_ok": _recon_result.reconciliation_ok,
                        "blocking_mismatches": _recon_result.blocking_mismatches,
                    })
                    _app_step(_run_lineage, _ST_IV, {
                        "execution_allowed": _integrity_ok,
                        "violation_count": len(_integrity_result.integrity_violations),
                    })
                    _app_step(_run_lineage, _ST_AD, {
                        "authority_level": _authority_level,
                        "chain_hash": _epoch_chain.chain_hash,
                    })
                    _app_step(_run_lineage, _ST_DD, {
                        "deployment_status": "authorized" if _integrity_ok else "blocked",
                        "epoch_id": _dep_epoch.epoch_id,
                    })
                    _persist_lin(_run_lineage, DEPLOYMENT_LINEAGE_FILE)

                    _run_manifest = _build_mf(
                        run_id=run_id,
                        reconciliation_epoch_id=_recon_epoch.epoch_id,
                        deployment_epoch_id=_dep_epoch.epoch_id,
                        broker_snapshot_hash=_bs_hash,
                        integrity_ok=_integrity_ok,
                        integrity_violation_count=len(_integrity_result.integrity_violations),
                        is_quarantined=False,
                        quarantine_id=None,
                        pending_divergences=0,
                        lineage_entry_hash=_run_lineage.tail_hash(),
                    )
                    _app_mf(_run_manifest, DEPLOYMENT_MANIFEST_FILE)
                    logger.info(
                        "[LINEAGE] manifest=%s status=%s chain=%s run_id=%s",
                        _run_manifest.manifest_id[:12],
                        _run_manifest.deployment_status,
                        _epoch_chain.chain_hash[:12],
                        run_id,
                    )
                except Exception as _lineage_err:
                    logger.warning("[LINEAGE] build failed (%s) — continuing", _lineage_err)

            except Exception as _recon_err:
                logger.warning("[RECONCILE] broker reconciliation failed (%s) — continuing", _recon_err)

            # Fail-closed gate: block live execution on integrity violation.
            # For dry-run: log and continue (observable per requirement).
            if _integrity_result is not None and not _integrity_result.execution_allowed:
                _fail_reason = _integrity_result.fail_closed_reason or "reconciliation_failure"
                logger.error("[INTEGRITY] execution blocked: %s", _fail_reason)
                if args.live:
                    print(
                        f"\n[FATAL] 整合性チェック失敗 — 発注ブロック: {_fail_reason}",
                        file=sys.stderr,
                    )
                    return 1
                # dry-run: log but do not block
                print(f"\n[整合性チェック] ライブなら発注ブロック: {_fail_reason}")
            elif _integrity_result is not None and _integrity_result.execution_allowed:
                logger.info(
                    "[INTEGRITY] ok violations=%d run_id=%s",
                    len(_integrity_result.integrity_violations),
                    run_id,
                )

            # ── Manifest gate (Task 1): fail-closed for live, warn for dry-run ──
            # In live mode: block if no authorized manifest exists.
            # In dry-run:   log status only — execution is never attempted.
            if args.live:
                try:
                    from src.live.deployment_authority_manifest import (
                        require_manifest as _require_mf,
                    )
                    _require_mf(DEPLOYMENT_MANIFEST_FILE, run_id)
                    logger.info("[MANIFEST] gate passed run_id=%s", run_id)
                except RuntimeError as _mf_gate_err:
                    logger.error("[MANIFEST] execution blocked: %s", _mf_gate_err)
                    print(
                        f"\n[FATAL] マニフェスト認証失敗 — 発注ブロック: {_mf_gate_err}",
                        file=sys.stderr,
                    )
                    return 1
            elif _run_manifest is not None:
                logger.info(
                    "[MANIFEST] dry-run: status=%s run_id=%s",
                    _run_manifest.deployment_status, run_id,
                )

            # ── Runtime integration stabilization gate ───────────────────────
            # Validates: state contracts, broker/runtime consistency, allocator
            # integration, stage ordering. Fail-closed for live; observable for DR.
            try:
                from src.live.runtime_integration import (
                    validate_runtime_integration as _validate_integration,
                    RuntimeContractError as _RCE,
                    RuntimeStateRegistry as _RSR,
                    IntegrationDependencyGraph as _IDG,
                    StagePreconditionValidator as _SPV,
                    STAGE_RECONCILIATION as _ST_RECON,
                )
                from src.paths import INTEGRATION_DIR as _integ_dir

                _integ_registry = _RSR()
                _integ_registry.register("cap_state",       _cap_state,       is_critical=True)
                _integ_registry.register("exposure_state",  _exposure_state,  is_critical=False)
                _integ_registry.register("ramp_state",      _ramp_state,      is_critical=False)
                _integ_registry.register("freeze_state",    _freeze_state,    is_critical=False)
                _integ_registry.register("inflight_registry", _inflight_registry, is_critical=True)

                _integ_dep = _IDG()
                _integ_dep.declare("capital",    [])
                _integ_dep.declare("allocation", ["capital"])
                _integ_dep.declare("broker",     ["capital"])
                _integ_dep.declare("execution",  ["allocation", "broker"])
                # RC3 FIX: "capital" is always available — either from the
                # persisted state file OR from the static CAPITAL fallback.
                # _eff_capital > 0 is the correct initialization predicate;
                # using _cap_state_loaded caused allocation/broker to appear
                # unresolved on every first-run (file absent) and blocked LIVE.
                if _eff_capital > 0:
                    _integ_dep.mark_initialized("capital")
                if _alloc_state_loaded:
                    _integ_dep.mark_initialized("allocation")
                if _broker_pos is not None:
                    _integ_dep.mark_initialized("broker")

                _integ_pre = _SPV()
                _integ_pre.record_stage_complete("api_check")
                _integ_pre.record_stage_complete("bootstrap")
                _integ_pre.record_stage_complete("broker_sync")
                _integ_pre.record_stage_complete("market_data")
                _integ_pre.record_stage_complete("signal_generation")
                _integ_pre.record_stage_complete("reconciliation")

                _integ_result = _validate_integration(
                    run_id=run_id,
                    stage=_ST_RECON,
                    registry=_integ_registry,
                    dep_graph=_integ_dep,
                    precondition_val=_integ_pre,
                    broker_positions=_broker_pos,
                    runtime_positions=_runtime_pos,
                    cap_state=_cap_state,
                    exposure_state=_exposure_state,
                    artifact_dir=_integ_dir,
                    is_live=args.live,
                )
                if not _integ_result.execution_allowed:
                    logger.warning(
                        "[INTEGRATION] %d blocking violations — dry-run continues",
                        _integ_result.blocking_count,
                    )
                else:
                    logger.info("[INTEGRATION] contracts ok run_id=%s", run_id)
            except _RCE as _ic_err:
                logger.error("[INTEGRATION] contract violation: %s", _ic_err)
                if args.live:
                    print(
                        f"\n[FATAL] ランタイム整合性契約違反 — 発注ブロック: {_ic_err}",
                        file=sys.stderr,
                    )
                    return 1
            except Exception as _integ_err:
                logger.warning("[INTEGRATION] validation error (%s) — continuing", _integ_err)

            # ── Compression Continuation: phase-score held positions ──────────────
            # Injects compression_score / compression_phase / suppression_eligible
            # into signal dicts BEFORE exit orchestrator reads them.
            # Fail-open: any error leaves signals unchanged.
            _cc_result: dict = {}
            try:
                from src.analytics.compression_continuation_hook import (
                    run_compression_continuation_hook as _cc_hook,
                )
                from src.paths import COMPRESSION_CONTINUATION_LOG_FILE as _cc_log
                result.signals, _cc_result = _cc_hook(
                    signals=result.signals,
                    backtest_dataset_dir=BACKTEST_DATASET_DIR,
                    default_data_version=DEFAULT_DATA_VERSION or "",
                    cache_dir=CACHE_DIR,
                    log_path=_cc_log,
                    run_id=run_id,
                )
                if _cc_result.get("n_scored", 0) > 0:
                    _cc_eligible = [
                        s for s, r in _cc_result.get("results", {}).items()
                        if r.get("suppression_eligible")
                    ]
                    _cc_distrib = [
                        s for s, r in _cc_result.get("results", {}).items()
                        if r.get("is_distribution")
                    ]
                    logger.info(
                        "[COMPRESSION] scored=%d eligible=%s distrib=%s",
                        _cc_result["n_scored"], _cc_eligible, _cc_distrib,
                    )
            except Exception as _cc_err:
                logger.warning("[COMPRESSION] hook failed (%s) — continuing", _cc_err)

            # ── Exit orchestration: autonomous exit behavior from analytics ────────
            # Fail-open: any error leaves signals and orders unchanged.
            # Modifies held-position signals (suppress_exit / force_exit) and order_objects.
            try:
                from src.runtime.policy.runtime_exit_orchestrator import run_exit_orchestration_hook
                from src.paths import EXIT_POLICY_DECISIONS_FILE as _exit_pol_file
                _regime_info_eo: dict = {}
                try:
                    from src.paths import DEPLOYMENT_STATE_RECORD_FILE as _dsr_file
                    import json as _json_eo
                    if _dsr_file.exists():
                        _dsr = _json_eo.loads(_dsr_file.read_text(encoding="utf-8"))
                        _regime_info_eo = {
                            "regime": _dsr.get("regime", "unknown"),
                            "deteriorating": _dsr.get("regime_deteriorating", False),
                            "rs_collapse": _dsr.get("rs_collapse_detected", False),
                            "confidence": float(_dsr.get("regime_confidence", 0.5)),
                        }
                except Exception:
                    pass
                result.signals, order_objects = run_exit_orchestration_hook(
                    signals=result.signals,
                    order_objects=order_objects,
                    store_path=_exit_pol_file,
                    portfolio_summary=result.portfolio_summary,
                    broker_positions=_broker_pos,
                    regime_info=_regime_info_eo,
                )
                logger.info("[EXIT_ORCH] hook complete: %d signals, %d orders",
                            len(result.signals), len(order_objects))
            except Exception as _exit_orch_err:
                logger.warning("[EXIT_ORCH] hook failed (%s) — continuing", _exit_orch_err)

            # ── Holding Expectancy Telemetry: daily snapshot + 3d return materialization ──
            # observation_only: never modifies signals. Fail-open.
            try:
                from src.analytics.holding_expectancy_telemetry import (
                    run_holding_expectancy_hook as _het_hook,
                )
                from src.paths import (
                    HOLDING_EXPECTANCY_SNAPSHOT_FILE     as _het_snap,
                    HOLDING_EXPECTANCY_MATERIALIZED_FILE as _het_mat,
                )
                import pandas as _pd_het

                def _het_ohlcv_loader(sym: str):
                    for _p in [
                        BACKTEST_DATASET_DIR / (DEFAULT_DATA_VERSION or "") / f"{sym}.parquet"
                        if DEFAULT_DATA_VERSION else None,
                        CACHE_DIR / "ohlcv" / f"{sym}.parquet",
                    ]:
                        if _p and _p.exists():
                            try:
                                return _pd_het.read_parquet(_p)
                            except Exception:
                                pass
                    return None

                _het_result = _het_hook(
                    signals=result.signals,
                    snapshot_path=_het_snap,
                    materialized_path=_het_mat,
                    ohlcv_loader=_het_ohlcv_loader,
                    regime_info=_regime_info_eo,
                )
                logger.info(
                    "[HET] written=%d materialized=%d",
                    _het_result.get("n_written", 0),
                    _het_result.get("n_materialized", 0),
                )
            except Exception as _het_err:
                logger.warning("[HET] hook failed (%s) — continuing", _het_err)

            # ── Suppression Outcome Telemetry: attribution for suppressed exits ──────────────
            # Detects suppress_exit events from exit orchestrator, tracks lifecycle,
            # materializes return delta and MFE/MAE on exit.
            # observation_only: never modifies signals. Fail-open.
            try:
                from src.analytics.suppression_outcome_telemetry import (
                    run_suppression_outcome_hook as _sot_hook,
                )
                from src.paths import (
                    SUPPRESSION_ACTIVE_STATE_FILE     as _sot_active,
                    SUPPRESSION_OUTCOME_LOG_FILE      as _sot_outcomes,
                    SUPPRESSION_PHASE_TRANSITION_FILE as _sot_trans,
                    COMPRESSION_CONTINUATION_LOG_FILE as _sot_cc_snaps,
                )
                import pandas as _pd_sot

                def _sot_ohlcv_loader(sym: str):
                    for _p in [
                        (
                            BACKTEST_DATASET_DIR / (DEFAULT_DATA_VERSION or "") / f"{sym}.parquet"
                            if DEFAULT_DATA_VERSION else None
                        ),
                        CACHE_DIR / "ohlcv" / f"{sym}.parquet",
                    ]:
                        if _p and _p.exists():
                            try:
                                return _pd_sot.read_parquet(_p)
                            except Exception:
                                pass
                    return None

                _sot_result = _sot_hook(
                    signals=result.signals,
                    active_state_path=_sot_active,
                    outcome_path=_sot_outcomes,
                    transition_path=_sot_trans,
                    compression_snapshots_path=_sot_cc_snaps,
                    ohlcv_loader=_sot_ohlcv_loader,
                )
                logger.info(
                    "[SOT] starts=%d exits=%d transitions=%d active=%d",
                    _sot_result.get("n_suppression_starts", 0),
                    _sot_result.get("n_exits_resolved", 0),
                    _sot_result.get("n_transitions", 0),
                    _sot_result.get("n_active", 0),
                )
            except Exception as _sot_err:
                logger.warning("[SOT] hook failed (%s) — continuing", _sot_err)

            # ── Portfolio intelligence: unified cross-position analytics → runtime policies ──
            # Fail-open: any error leaves signals and orders unchanged.
            try:
                from src.runtime.policy.portfolio_intelligence_engine import (
                    run_portfolio_intelligence_hook as _pi_hook_fn,
                )
                from src.paths import PI_DECISIONS_FILE as _pi_file
                result.signals, order_objects, _pi_overlay = _pi_hook_fn(
                    signals=result.signals,
                    order_objects=order_objects,
                    store_path=_pi_file,
                    portfolio_summary=result.portfolio_summary,
                    broker_positions=_broker_pos,
                    regime_info=_regime_info_eo,
                    exposure_state=_exposure_state,
                )
                logger.info(
                    "[PI_HOOK] complete: corr=%.3f surv=%.3f slot=%.2f gate=%s decisions=%d",
                    _pi_overlay.overall_corr_score, _pi_overlay.survivability_score,
                    _pi_overlay.slot_scale, _pi_overlay.survivability_gate,
                    _pi_overlay.n_decisions,
                )
            except Exception as _pi_err:
                logger.warning("[PI_HOOK] hook failed (%s) — continuing", _pi_err)

            # ── Exit Intelligence OS: observation capture + velocity scoring ─────
            # Fail-open: any error leaves signals unchanged.
            # Observes open positions + computes ExitVelocityScore per symbol.
            try:
                from src.analytics.exit_intelligence.hook import run_exit_intelligence_hook
                from src.paths import (
                    EXIT_INTEL_OBSERVATIONS_FILE as _ei_obs,
                    EXIT_INTEL_PATH_SNAPSHOTS_FILE as _ei_paths,
                    EXIT_INTEL_EXEC_OBSERVATIONS_FILE as _ei_exec,
                    EXIT_INTEL_PORTFOLIO_PRESSURE_FILE as _ei_port,
                    EXIT_INTEL_VELOCITY_SCORES_FILE as _ei_vel,
                    EXIT_INTEL_LIFECYCLE_FILE as _ei_lc,
                    EXIT_INTEL_DATASET_FILE as _ei_ds,
                    EXIT_INTEL_REPORTS_DIR as _ei_rpt_dir,
                )
                import json as _json_ei
                # Build market_data from broker positions + signals
                _md_ei: dict = {}
                try:
                    _ei_ps = result.portfolio_summary
                    for _ei_pos in (_broker_pos or []):
                        _sym = str(_ei_pos.get("Symbol") or _ei_pos.get("symbol", ""))
                        if not _sym:
                            continue
                        _md_ei[_sym] = {
                            "entry_price": float(_ei_pos.get("AvgPrice", _ei_pos.get("avg_price", 0))),
                            "close": float(_ei_pos.get("CurrentPrice", _ei_pos.get("current_price", 0))),
                            "peak_price": float(_ei_pos.get("PeakPrice", _ei_pos.get("peak_price",
                                               _ei_pos.get("CurrentPrice", 0)))),
                            "entry_date": str(_ei_pos.get("EntryDate", _ei_pos.get("entry_date", ""))),
                            "day_number": int(_ei_pos.get("HoldingDays", _ei_pos.get("holding_days", 1))),
                            "rs_rank": float(_ei_pos.get("RSRank", _ei_pos.get("rs_rank", 50.0))),
                            "atr_pct": float(_ei_pos.get("ATRPct", _ei_pos.get("atr_pct", 0.02))),
                            "momentum": float(_ei_pos.get("Momentum", _ei_pos.get("momentum", 0.0))),
                            "momentum_prev": float(_ei_pos.get("MomentumPrev", 0.0)),
                            "volume_ratio": float(_ei_pos.get("VolumeRatio", 1.0)),
                            "volatility_compression": float(_ei_pos.get("VolatilityCompression", 0.5)),
                            "upper_shadow_ratio": float(_ei_pos.get("UpperShadowRatio", 0.3)),
                            "exit_signal_count": int(_ei_pos.get("ExitSignalCount", 0)),
                            "trend_persistence": float(_ei_pos.get("TrendPersistence", 0.5)),
                            "breadth_score": float(_regime_info_eo.get("breadth", 0.5)),
                        }
                except Exception:
                    pass
                _ei_result = run_exit_intelligence_hook(
                    portfolio_summary=result.portfolio_summary,
                    broker_positions=_broker_pos,
                    regime_info=_regime_info_eo,
                    market_data=_md_ei if _md_ei else None,
                    obs_path=_ei_obs,
                    path_snapshots_path=_ei_paths,
                    exec_obs_path=_ei_exec,
                    portfolio_pressure_path=_ei_port,
                    velocity_scores_path=_ei_vel,
                    lifecycle_path=_ei_lc,
                    dataset_path=_ei_ds,
                    reports_dir=_ei_rpt_dir,
                )
                _ei_hp = _ei_result.get("high_pressure_count", 0)
                if _ei_hp > 0:
                    logger.warning(
                        "[EXIT_INTEL] %d high-pressure position(s) detected (score≥5.0)",
                        _ei_hp,
                    )
                logger.info(
                    "[EXIT_INTEL] hook complete: scored=%d positions heat=%.2f%%",
                    len(_ei_result.get("velocity_scores", {})),
                    _ei_result.get("portfolio_heat", 0.0) * 100,
                )
            except Exception as _ei_err:
                logger.warning("[EXIT_INTEL] hook failed (%s) — continuing", _ei_err)

            # ── Predictive Expansion hook (fail-open analytics) ───────────────────
            _predictive_result: dict = {}
            try:
                _pe_cfg = getattr(cfg, "predictive_expansion", None) or {}
                _pe_enabled = _pe_cfg.get("enabled", True) if isinstance(_pe_cfg, dict) else True
                if _pe_enabled:
                    from src.analytics.predictive_expansion_hook import run_predictive_expansion_hook
                    _predictive_result = run_predictive_expansion_hook(
                        universe             = RSR_UNIVERSE_62,   # full RSR context (live + shadow)
                        backtest_dataset_dir = BACKTEST_DATASET_DIR,
                        default_data_version = DEFAULT_DATA_VERSION or "",
                        cache_dir            = CACHE_DIR,
                        scores_path          = PREDICTIVE_SCORES_FILE,
                        reports_dir          = PREDICTIVE_REPORTS_DIR,
                        run_id               = run_id,
                        print_summary        = True,
                    )
                    logger.info(
                        "[PREDICTIVE] hook complete: top3=%s igniting=%d followers=%d",
                        [s for s, _ in _predictive_result.get("top_candidates", [])[:3]],
                        len(_predictive_result.get("sector_igniting_symbols", [])),
                        len(_predictive_result.get("followers", [])),
                    )
            except Exception as _pe_err:
                logger.warning("[PREDICTIVE] hook failed (%s) — continuing", _pe_err)

            _emit_phase("predictive_hook", "complete", run_id=run_id)
            _emit_phase("signal_generation", "complete", run_id=run_id)

            # ── Universe determinism audit: post_signal snapshot ──────────────────
            try:
                from src.runtime.universe_determinism_audit import record_universe_snapshot as _uda_sig
                _ustats_for_audit = getattr(result, "universe_stats", {})
                _uda_sig(
                    audit_dir         = UNIVERSE_DETERMINISM_AUDIT_DIR,
                    mode              = "LIVE" if args.live else "DRY",
                    snapshot_stage    = "post_signal",
                    live_symbols      = LIVE_UNIVERSE.keys(),
                    shadow_symbols    = SHADOW_UNIVERSE.keys(),
                    tradeable_symbols = (
                        list(_ustats_for_audit.get("tradeable_symbols", LIVE_UNIVERSE.keys()))
                        if _ustats_for_audit else LIVE_UNIVERSE.keys()
                    ),
                )
            except Exception as _uda_sig_err:
                logger.warning("[UNIVERSE_AUDIT] post_signal record failed (FAIL_OPEN): %s", _uda_sig_err)

            # ── display results (within supervisor context) ───────────────────────
            print(f"\n  データ基準日 : {result.data_as_of}")
            _ustats = result.universe_stats
            print(f"  LIVE_UNIVERSE    : {_ustats.get('live', result.n_universe)} 銘柄")
            print(f"  SHADOW_UNIVERSE  : {_ustats.get('shadow', 0)} 銘柄")
            print(f"  RSR_CONTEXT      : {_ustats.get('rsr_context', result.n_universe)} 銘柄")
            print(
                f"  TRADEABLE_UNIVERSE: {_ustats.get('tradeable', result.n_universe)} 銘柄"
                f"  (price除外={_ustats.get('filtered_price', 0)}"
                f" risk除外={_ustats.get('filtered_risk', 0)})"
            )
            ps = result.portfolio_summary
            portfolio_label = "仮想ポートフォリオ" if ps.get("portfolio_mode") == "virtual" else "ポートフォリオ"
            positions_api = ps.get("positions_api", {})
            wallet_api = ps.get("wallet_api", {})
            cb_str = ps.get("cb_state", "NORMAL")
            if cb_str != "NORMAL":
                cooldown = ps.get("cb_cooldown_end") or ""
                print(f"  ⚠ CB状態    : {cb_str}（クールダウン終了: {cooldown}）")
            dd_pct = ps.get("current_drawdown", 0.0) * 100
            _eq_max = ps.get("equity_based_max_pos", ps["max_positions"])
            _eq_note = f" [資本連動→{_eq_max}]" if _eq_max != ps["max_positions"] else ""
            available_cash = ps.get("available_cash")
            cash_display = f"¥{available_cash:,.0f}" if available_cash is not None else "不明"
            print(
                f"  {portfolio_label}: 保有 {ps['current_positions']} / "
                f"最大 {ps['max_positions']} 銘柄{_eq_note}  "
                f"空きスロット: {ps['open_slots']}  "
                f"余力: {cash_display}  "
                f"DD: {dd_pct:+.1f}%"
            )
            print(
                f"  API状態       : positions={positions_api.get('source', 'unknown')}"
                f" / wallet={wallet_api.get('source', 'unknown')}"
            )
            print(f"  Top-{TOP_K}銘柄  : {', '.join(result.top_k_symbols)}")

            # ── Capital Deployment Runtime Visibility ──────────────────────────
            # Observation-only: computes from existing variables; no new state files.
            try:
                from src.live.capital_deployment_os import dynamic_max_positions as _dyn_max_fn
                _cdep_dyn_max = _dyn_max_fn(_cdos_deployable)
                _cdep_target_size = _cdos_deployable / _cdep_dyn_max if _cdep_dyn_max > 0 else 0.0
                _cdep_equity = float(_cap_state.actual_equity) if _cap_state is not None else float(_eff_capital)
                _cdep_cash = float(available_cash) if available_cash is not None else 0.0
                _cdep_utilization = (
                    (_cdep_equity - _cdep_cash) / _cdep_equity * 100.0
                    if _cdep_equity > 0 else 0.0
                )
                print(
                    f"\n[CAPITAL_DEPLOYMENT]"
                    f"  eff=¥{_eff_capital:,.0f}"
                    f"  deployable=¥{_cdos_deployable:,.0f}"
                    f"  dyn_max_pos={_cdep_dyn_max}"
                    f"  target_size=¥{_cdep_target_size:,.0f}"
                    f"  utilization={_cdep_utilization:.1f}%"
                )
            except Exception as _cdep_err:
                logger.debug("[CAPITAL_DEPLOYMENT] visibility block failed: %s", _cdep_err)

            # ── POSITION_SIZING_AUDIT DCC ──────────────────────────────────
            # configured/effective/dynamic max_positions + per-day throttle を毎 run 保存。
            # filter block counts は metrics.jsonl の blocked_by_price/liquidity と同値。
            try:
                from src.universe.decision_context_capture import (
                    DecisionContextCapture as _DCC_psz,
                )
                from src.paths import DECISION_CONTEXT_HISTORY_FILE as _dcc_file_psz
                _dcc_psz = _DCC_psz(history_file=_dcc_file_psz)
                _psz_dyn = _cdep_dyn_max if "_cdep_dyn_max" in dir() else None
                _psz_eff = result.portfolio_summary.get("equity_based_max_pos")
                _psz_raw = result.portfolio_summary.get("cdos_dyn_raw")
                _dcc_psz.capture(
                    "POSITION_SIZING_AUDIT",
                    decision_inputs={
                        "configured_max_positions":  MAX_POS,
                        "dynamic_max_positions":     _psz_raw,
                        "effective_max_positions":   _psz_eff,
                        "max_new_positions_per_day": MAX_NEW_POS_PER_DAY,
                    },
                    decision_result={
                        "cdos_clamped":               (_psz_raw or 0) > MAX_POS,
                        "price_filter_block_count":   result.portfolio_summary.get("universe_stats", {}).get("filtered_price"),
                        "liquidity_filter_block_count": None,  # in metrics.jsonl: blocked_by_liquidity
                    },
                )
            except Exception as _dcc_psz_err:
                logger.debug("[DCC] POSITION_SIZING_AUDIT failed (FAIL_OPEN): %s", _dcc_psz_err)

            print_signals(result)

            if not args.no_save:
                saved_path = save_signal_json(result, args.output_dir)
                print(f"\n💾 シグナル保存: {saved_path}")

            # Phase2 日次メトリクス記録（DRY_RUN / LIVE 共通）
            try:
                log_phase2_metrics(result)
            except Exception as _e:
                logger.warning("Phase2メトリクス記録失敗（無視）: %s", _e)

            # ── Feature forward expectancy snapshot (fail-open) ───────────────
            try:
                from src.analytics.feature_forward_expectancy import write_signal_snapshots as _write_snaps
                from src.paths import ENTRY_FEATURES_FILE as _eff_path
                _ps_ffe = result.portfolio_summary
                _rsr_ffe = [s.get("rsr", 0) for s in result.signals]
                _n_ffe = len(_rsr_ffe)
                _b50_ffe = sum(1 for r in _rsr_ffe if r >= 50) / _n_ffe if _n_ffe else 0.0
                _b75_ffe = sum(1 for r in _rsr_ffe if r >= 75) / _n_ffe if _n_ffe else 0.0
                _eq_ffe = float(_ps_ffe.get("current_equity") or 0)
                _cash_ffe = float(_ps_ffe.get("available_cash") or 0)
                _cash_ratio_ffe = _cash_ffe / _eq_ffe if _eq_ffe > 0 else 1.0
                _max_pos_ffe = int(_ps_ffe.get("max_positions") or 3)
                _cur_pos_ffe = int(_ps_ffe.get("current_positions") or 0)
                _slot_ffe = _cur_pos_ffe / _max_pos_ffe if _max_pos_ffe > 0 else 0.0
                _regime_ffe = (
                    "trend_persistent" if _b50_ffe >= 0.50
                    else ("mixed" if _b50_ffe >= 0.30 else "weak")
                )
                _write_snaps(
                    signals=result.signals,
                    run_id=run_id,
                    cash_ratio=_cash_ratio_ffe,
                    slot_utilization=_slot_ffe,
                    breadth_50=_b50_ffe,
                    breadth_75=_b75_ffe,
                    market_regime=_regime_ffe,
                    snapshots_path=_eff_path,
                )
            except Exception as _ffe_err:
                logger.warning("FeatureSnapshot write failed (ignored): %s", _ffe_err)

            # ── Phase 5A: score candidates + update exposure state ────────────
            _efficiency_scores: dict = {}   # symbol → float
            _phase5a_effective_n = 0.0
            if _alloc_state_loaded:
                try:
                    from src.allocation import (
                        CandidateSignal, rank_candidates,
                        compute_exposure_report, update_exposure_state,
                        save_exposure_state,
                    )
                    _daily_returns: dict = {}
                    for _s5 in result.signals:
                        _sym5 = _s5.get("symbol", "")
                        _ret5 = float(_s5.get("day_return", _s5.get("return", 0.0)))
                        if _sym5:
                            _daily_returns[_sym5] = _ret5
                    if _daily_returns:
                        _exposure_state = update_exposure_state(_exposure_state, _daily_returns)
                    # RC5 FIX: compute_exposure_report() requires symbols,
                    # sector_map, and rsr_rank_map in addition to state.
                    # Build them from the current signal set before calling.
                    _held_syms = [
                        _s5.get("symbol", "") for _s5 in result.signals
                        if _s5.get("currently_holding") and _s5.get("symbol")
                    ]
                    _exp_sector_map: dict = {
                        _s5.get("symbol", ""): _s5.get("sector", "不明")
                        for _s5 in result.signals if _s5.get("symbol")
                    }
                    _exp_rsr_map: dict = {
                        _s5.get("symbol", ""): float(_s5.get("rsr", 50.0))
                        for _s5 in result.signals if _s5.get("symbol")
                    }
                    _exp_report = compute_exposure_report(
                        _exposure_state,
                        symbols=_held_syms,
                        sector_map=_exp_sector_map,
                        rsr_rank_map=_exp_rsr_map,
                        n_universe=result.n_universe or 42,
                    )
                    _phase5a_effective_n = _exp_report.effective_independent_n
                    _buy_candidates = []
                    for _s5 in result.signals:
                        if _s5.get("action") != "BUY":
                            continue
                        _sym5 = _s5.get("symbol", "")
                        _lc5 = _lifecycle_map.get(_sym5)
                        _buy_candidates.append(CandidateSignal(
                            symbol=_sym5,
                            rsr_rank=float(_s5.get("rsr", 50.0)),
                            post_cost_alpha_bps=float(_s5.get("post_cost_alpha_bps", 5.0)),
                            avg_daily_volume_yen=float(_s5.get("volume_yen", 1_000_000_000.0)),
                            sector=_s5.get("sector", "不明"),
                            estimated_symbol_dd=abs(float(_s5.get("current_drawdown", 0.0))),
                        ))
                    if _buy_candidates:
                        _ranked5 = rank_candidates(
                            _buy_candidates, _held_syms, _exposure_state,
                        )
                        _efficiency_scores = {
                            r.symbol: r.capital_efficiency_score for r in _ranked5
                        }
                    logger.info(
                        "[ALLOC] exposure: effective_n=%.2f collapse=%s scored=%d",
                        _phase5a_effective_n,
                        _exp_report.collapse_alert,
                        len(_efficiency_scores),
                    )
                except Exception as _alloc_score_err:
                    logger.warning("[ALLOC] scoring failed (%s)", _alloc_score_err)

            # ── Winner confirmation for add-on eligibility ────────────────────
            # Pure computation (no I/O). Runs in both DRY and LIVE paths.
            # Fail-open: any error leaves _winner_confirmations empty (no add-ons).
            _winner_confirmations: list = []
            try:
                from src.addon import check_winner
                from src.addon.extension_filter import ExtensionInput as _ExtInput
                _wc_summary = result.portfolio_summary or {}
                _wc_dd      = float(_wc_summary.get("current_drawdown", 0.0))
                # Regime bear: portfolio drawdown < -5% from peak
                _wc_bear    = _wc_dd < -0.05
                _wc_held    = [
                    _ws for _ws in result.signals
                    if _ws.get("currently_holding") and _ws.get("signal") != -1
                ]
                for _ws in _wc_held:
                    _wsym = _ws.get("symbol", "")
                    if not _wsym:
                        continue
                    _wep   = float(_ws.get("entry_price", 0.0))
                    _wpnl  = float(_ws.get("unrealized_pnl_pct", 0.0))
                    _wcur  = _wep * (1.0 + _wpnl) if _wep > 0 else 0.0
                    # rsr_momentum key used in result.signals dict (see signal_bridge line 4491)
                    _wmom  = float(_ws.get("rsr_momentum", 0.0))
                    # deterioration proxy: signal==1 (still BUY) → low; signal==0 (hold) → moderate
                    _wdetr = 0.0 if _ws.get("signal") == 1 else 0.15
                    _wstop = float(_ws.get("stop_price", 0.0))
                    _wext  = _ExtInput(
                        symbol=_wsym,
                        unrealized_pnl_pct=_wpnl,
                        hold_days=int(_ws.get("hold_days", 0)),
                        rsr=float(_ws.get("rsr", 0.0)),
                        rsr_rank=int(_ws.get("rsr_rank", 99)),
                        rsr_momentum=_wmom,
                        entry_price=_wep,
                        current_price=_wcur,
                        trailing_stop_price=_wstop,
                        regime_bear=_wc_bear,
                        portfolio_pnl_pct=_wc_dd,
                    )
                    _wconf = check_winner(
                        symbol=_wsym,
                        unrealized_pnl_pct=_wpnl,
                        hold_days=int(_ws.get("hold_days", 0)),
                        rsr_rank=int(_ws.get("rsr_rank", 99)),
                        rsr=float(_ws.get("rsr", 0.0)),
                        trend_quality=_wmom,
                        deterioration_score=_wdetr,
                        regime_bear=_wc_bear,
                        current_price=_wcur,
                        entry_price=_wep,
                        extension_input=_wext,
                    )
                    _winner_confirmations.append(_wconf)
                    # [ADDON_EXT] extension filter diagnostics — observation-only, no behavioral change
                    try:
                        from src.addon.extension_filter import check_extension as _chk_ext
                        _ext_result = _chk_ext(_wext)
                        _ext_reasons = [r for r in _wconf.fail_reasons if r.startswith("EXT:")]
                        logger.info(
                            "[ADDON_EXT] symbol=%s blocked=%s reasons=%s pqs=%.4f",
                            _wsym,
                            _ext_result.extension_blocked,
                            _ext_reasons,
                            _ext_result.persistence_quality_score,
                        )
                    except Exception as _ext_log_err:
                        logger.debug("[ADDON_EXT] diagnostic failed for %s: %s", _wsym, _ext_log_err)
                logger.info(
                    "[ADDON] winner_check: held=%d confirmed=%d",
                    len(_wc_held),
                    sum(1 for _c in _winner_confirmations if _c.confirmed),
                )
            except Exception as _wc_err:
                logger.warning("[ADDON] winner_confirmation failed (%s) — no add-ons", _wc_err)

            # ── Continuation breakout boost (compression→breakout phase priority) ──
            # Boosts confidence_score for continuation_breakout confirmed winners.
            # Affects ordering only — sizing (100 shares) unchanged.
            try:
                from src.addon.winner_confirmation import apply_continuation_breakout_boost as _cb_boost
                _sig_by_sym_cb = {str(s.get("symbol", "")): s for s in result.signals}
                _winner_confirmations = [
                    _cb_boost(
                        _wc,
                        continuation_breakout=bool(
                            _sig_by_sym_cb.get(str(_wc.symbol), {}).get("continuation_breakout", False)
                        ),
                        compression_score=float(
                            _sig_by_sym_cb.get(str(_wc.symbol), {}).get("compression_score", 0.0)
                        ),
                        volume_ratio_5d=float(
                            _sig_by_sym_cb.get(str(_wc.symbol), {}).get("compression_volume_ratio", 1.0)
                        ),
                        rsr_momentum=float(
                            _sig_by_sym_cb.get(str(_wc.symbol), {}).get("rsr_momentum", 0.0)
                        ),
                    )[0]
                    for _wc in _winner_confirmations
                ]
                _n_cb_boosted = sum(
                    1 for _wc in _winner_confirmations
                    if _wc.confirmed and _wc.continuation_breakout
                )
                if _n_cb_boosted:
                    logger.info("[ADDON] continuation_breakout boost applied: %d winner(s)", _n_cb_boosted)
            except Exception as _cb_err:
                logger.warning("[ADDON] continuation_breakout boost failed (%s) — continuing", _cb_err)

            # ── Breakout Quality Intelligence: modulate continuation boost ────────
            # Classifies candle quality (healthy/extended/weak/failed) and adjusts
            # continuation_boost_applied for each confirmed winner.
            # healthy→1.15 (unchanged), extended/weak→1.00, failed→0.90
            # Only confidence_score ordering is affected; position size unchanged.
            # FAIL_OPEN: any error preserves original boost and continues.
            _bq_results_by_sym: dict = {}   # populated below; reused by CP hook
            try:
                from src.analytics.breakout_quality import (
                    extract_bq_inputs_from_df as _bq_extract,
                    compute_breakout_quality   as _bq_compute,
                    get_addon_boost_multiplier as _bq_boost_mult,
                    append_bq_telemetry        as _bq_telem,
                    materialize_bq_returns     as _bq_mat,
                )
                from src.paths import BREAKOUT_QUALITY_TELEMETRY_FILE as _bq_tpath
                from dataclasses import replace as _bq_dc_replace
                import pandas as _pd_bq

                _sig_map_bq = {str(_s.get("symbol", "")): _s for _s in result.signals}
                _wc_bq: list = []

                for _wc in _winner_confirmations:
                    if not (_wc.confirmed and _wc.continuation_breakout):
                        _wc_bq.append(_wc)
                        continue
                    _bq_sym = str(_wc.symbol)
                    try:
                        _bq_p = CACHE_DIR / "ohlcv" / f"{_bq_sym}.parquet"
                        _bq_df = _pd_bq.read_parquet(_bq_p)
                        if len(_bq_df) < 2:
                            _wc_bq.append(_wc)
                            continue
                        _bq_sig = _sig_map_bq.get(_bq_sym, {})
                        _bq_inp = _bq_extract(
                            symbol=_bq_sym,
                            df=_bq_df,
                            signal=_bq_sig,
                            rsr=float(_wc.rsr),
                            rsr_momentum=float(_wc.trend_quality),
                        )
                        _bq_res        = _bq_compute(_bq_inp)
                        _bq_results_by_sym[_bq_sym] = _bq_res
                        _boost_before  = float(_wc.continuation_boost_applied)
                        _boost_after   = _bq_boost_mult(_bq_res.phase_type)

                        if abs(_boost_after - _boost_before) > 1e-6:
                            _pre_score = _wc.confidence_score / max(_boost_before, 1e-6)
                            _wc = _bq_dc_replace(
                                _wc,
                                confidence_score=round(min(1.0, _pre_score * _boost_after), 4),
                                continuation_boost_applied=_boost_after,
                            )
                            logger.info(
                                "[BQ] %s phase=%s boost %.2f→%.2f",
                                _bq_sym, _bq_res.phase_type, _boost_before, _boost_after,
                            )

                        try:
                            _bq_telem(_bq_res, _boost_before, _boost_after, _bq_tpath)
                        except Exception:
                            pass
                    except Exception as _bq_sym_e:
                        logger.debug("[BQ] skip %s: %s", _bq_sym, _bq_sym_e)
                    _wc_bq.append(_wc)

                _winner_confirmations = _wc_bq
                _n_bq_healthy = sum(
                    1 for _wc in _winner_confirmations
                    if _wc.confirmed and _wc.continuation_breakout
                    and getattr(_wc, "continuation_boost_applied", 1.0) >= 1.15
                )
                logger.info(
                    "[BQ] quality applied: cb=%d healthy_boost=%d",
                    sum(1 for _wc in _winner_confirmations if _wc.confirmed and _wc.continuation_breakout),
                    _n_bq_healthy,
                )
                # Materialize past forward returns (fail-open, non-blocking)
                try:
                    def _bq_ohlcv_loader(_s: str):
                        _bp = CACHE_DIR / "ohlcv" / f"{_s}.parquet"
                        return _pd_bq.read_parquet(_bp) if _bp.exists() else None
                    _bq_mat(_bq_tpath, _bq_ohlcv_loader)
                except Exception:
                    pass

            except Exception as _bq_hook_err:
                logger.warning("[BQ] breakout_quality hook failed (%s) — continuing", _bq_hook_err)

            # ── Addon continuation telemetry hook ─────────────────────────────
            try:
                from src.analytics.addon_continuation_telemetry import run_addon_continuation_hook as _act_hook
                from src.paths import ADDON_CONTINUATION_LOG_FILE as _act_log
                import pandas as _pd_act

                def _act_ohlcv_loader(sym: str):
                    try:
                        _p = CACHE_DIR / "ohlcv" / f"{sym}.parquet"
                        return _pd_act.read_parquet(_p) if _p.exists() else None
                    except Exception:
                        return None

                _act_sig_map = {str(s.get("symbol", "")): s for s in result.signals}
                _act_result = _act_hook(
                    confirmations=_winner_confirmations,
                    signals_by_sym=_act_sig_map,
                    log_path=_act_log,
                    ohlcv_loader=_act_ohlcv_loader,
                )
                logger.info(
                    "[ACT] recorded=%d mat_3d=%d mat_5d=%d",
                    _act_result.get("n_recorded", 0),
                    _act_result.get("n_materialized_3d", 0),
                    _act_result.get("n_materialized_5d", 0),
                )
            except Exception as _act_err:
                logger.warning("[ACT] addon_continuation telemetry failed (%s) — continuing", _act_err)

            # ── Continuation Priority Intelligence hook ───────────────────────
            # Ranks held positions by capital efficiency (0-100).
            # Observation-only: no execution or sizing changes.
            # Uses BQ scores already computed above; falls back to parquet
            # computation for positions not tagged continuation_breakout.
            # FAIL_OPEN: any error leaves _cp_results empty → reports skipped.
            _cp_results: list = []
            try:
                from src.analytics.continuation_priority import (
                    ContinuationPriorityInput         as _CpInp,
                    compute_continuation_priority     as _cp_compute,
                    append_cp_telemetry               as _cp_telem,
                    materialize_cp_returns            as _cp_mat,
                )
                from src.analytics.breakout_quality import (
                    extract_bq_inputs_from_df as _cp_bq_extract,
                    compute_breakout_quality  as _cp_bq_compute,
                )
                from src.paths import CONTINUATION_PRIORITY_TELEMETRY_FILE as _cp_tpath
                import pandas as _pd_cp

                _cp_sig_map = {str(_s.get("symbol", "")): _s for _s in result.signals}
                _cp_today   = datetime.now(JST).strftime("%Y-%m-%d")

                for _wc in _winner_confirmations:
                    if not _wc.confirmed:
                        continue
                    _cp_sym = str(_wc.symbol)
                    _cp_sig = _cp_sig_map.get(_cp_sym, {})

                    # BQ score: reuse from continuation_breakout pass if available
                    if _cp_sym in _bq_results_by_sym:
                        _cp_bq_score = float(_bq_results_by_sym[_cp_sym].quality_score)
                        _cp_phase    = str(_bq_results_by_sym[_cp_sym].phase_type)
                    else:
                        try:
                            _cp_bq_p  = CACHE_DIR / "ohlcv" / f"{_cp_sym}.parquet"
                            _cp_bq_df = _pd_cp.read_parquet(_cp_bq_p)
                            if len(_cp_bq_df) >= 2:
                                _cp_bq_inp2 = _cp_bq_extract(
                                    symbol=_cp_sym, df=_cp_bq_df, signal=_cp_sig,
                                    rsr=float(_wc.rsr),
                                    rsr_momentum=float(_wc.trend_quality),
                                )
                                _cp_bq_r2    = _cp_bq_compute(_cp_bq_inp2)
                                _cp_bq_score = float(_cp_bq_r2.quality_score)
                                _cp_phase    = str(_cp_bq_r2.phase_type)
                            else:
                                _cp_bq_score = 50.0
                                _cp_phase    = str(_cp_sig.get("compression_phase", "neutral"))
                        except Exception:
                            _cp_bq_score = 50.0
                            _cp_phase    = str(_cp_sig.get("compression_phase", "neutral"))

                    _cp_inp = _CpInp(
                        symbol=_cp_sym,
                        hold_days=int(_cp_sig.get("hold_days", 0)),
                        unrealized_pnl_pct=float(_cp_sig.get("unrealized_pnl_pct", 0.0)),
                        compression_score=float(_cp_sig.get("compression_score", 50.0)),
                        breakout_quality_score=_cp_bq_score,
                        rsr=float(_wc.rsr),
                        rsr_momentum=float(_wc.trend_quality),
                        mfe_pct=max(0.0, float(_cp_sig.get("unrealized_pnl_pct", 0.0))),
                        giveback_ratio=0.0,
                        current_phase=_cp_phase,
                        suppression_eligible=bool(_cp_sig.get("suppression_eligible", False)),
                    )
                    _cp_res = _cp_compute(_cp_inp)
                    _cp_results.append((_cp_sig, _cp_res))
                    try:
                        _cp_telem(_cp_res, _cp_inp, _cp_tpath, date_str=_cp_today)
                    except Exception:
                        pass

                _cp_results.sort(key=lambda _t: _t[1].priority_score, reverse=True)

                try:
                    def _cp_ohlcv_loader(_s: str):
                        _p = CACHE_DIR / "ohlcv" / f"{_s}.parquet"
                        return _pd_cp.read_parquet(_p) if _p.exists() else None
                    _cp_mat(_cp_tpath, _cp_ohlcv_loader)
                except Exception:
                    pass

                logger.info("[CP] priority scored: %d position(s)", len(_cp_results))

            except Exception as _cp_hook_err:
                logger.warning("[CP] continuation_priority hook failed (%s) — continuing", _cp_hook_err)

            # ── Opportunity Cost Intelligence (DRY + LIVE) ────────────────────
            # Observation-only: records portfolio-full rejections vs held priority.
            # FAIL-OPEN: any error is logged and ignored.
            try:
                from src.analytics.opportunity_cost import (
                    make_rejected_candidate_from_signal as _oc_make_cand,
                    make_held_summary_from_cp_result    as _oc_make_held,
                    record_opportunity_cost_events      as _oc_record,
                    materialize_outcomes                as _oc_materialize,
                )
                from src.paths import (
                    OPPORTUNITY_COST_EVENTS_FILE       as _oc_events_file,
                    OPPORTUNITY_COST_MATERIALIZED_FILE as _oc_mat_file,
                )
                _oc_today = datetime.now(JST).strftime("%Y-%m-%d")

                # Identify rejected BUY signals: appeared in signals but not orders
                _oc_executed_syms = {
                    str(o.get("symbol", o.get("Symbol", "")))
                    for o in (result.orders or [])
                    if str(o.get("side", o.get("Side", ""))).upper() == "BUY"
                }
                _oc_buy_signals = [
                    s for s in result.signals
                    if s.get("signal") == 1 and not s.get("currently_holding", False)
                ]
                _oc_rejected_raw = [
                    s for s in _oc_buy_signals
                    if str(s.get("symbol", "")) not in _oc_executed_syms
                ]

                # Portfolio-full detection
                _oc_n_held = sum(
                    1 for s in result.signals if s.get("currently_holding", False)
                )
                _oc_portfolio_full = _oc_n_held >= MAX_POS

                if _oc_rejected_raw and _cp_results and _oc_portfolio_full:
                    _oc_rejected_cands = [
                        _oc_make_cand(s) for s in _oc_rejected_raw
                    ]
                    _oc_held_summaries = [
                        h for h in (
                            _oc_make_held(t) for t in _cp_results
                        ) if h is not None
                    ]
                    _oc_recorded = _oc_record(
                        rejected_candidates=_oc_rejected_cands,
                        held_positions=_oc_held_summaries,
                        portfolio_full=True,
                        date=_oc_today,
                        events_file=_oc_events_file,
                    )
                    if _oc_recorded:
                        logger.info(
                            "[OC] %d opportunity cost event(s) recorded"
                            " (severe=%d moderate=%d)",
                            len(_oc_recorded),
                            sum(1 for r in _oc_recorded if "severe" in r.opportunity_cost_severity),
                            sum(1 for r in _oc_recorded if "moderate" in r.opportunity_cost_severity),
                        )

                # Materialize pending events (5-business-day forward returns)
                try:
                    import pandas as _pd_oc
                    def _oc_price_fetcher(sym: str, dt: str) -> "float | None":
                        _oc_p = CACHE_DIR / "ohlcv" / f"{sym}.parquet"
                        if not _oc_p.exists():
                            return None
                        try:
                            _oc_df = _pd_oc.read_parquet(_oc_p)
                            _oc_col = "Adj Close" if "Adj Close" in _oc_df.columns else "Close"
                            _oc_df.index = _pd_oc.to_datetime(_oc_df.index)
                            _oc_dt = _pd_oc.Timestamp(dt)
                            _oc_mask = _oc_df.index >= _oc_dt
                            if not _oc_mask.any():
                                return None
                            return float(_oc_df.loc[_oc_mask, _oc_col].iloc[0])
                        except Exception:
                            return None
                    _oc_n_mat = _oc_materialize(
                        _oc_events_file, _oc_mat_file, _oc_price_fetcher, _oc_today,
                    )
                    if _oc_n_mat > 0:
                        logger.info("[OC] materialized %d event(s)", _oc_n_mat)
                except Exception as _oc_mat_err:
                    logger.debug("[OC] materialization skipped: %s", _oc_mat_err)

            except Exception as _oc_hook_err:
                logger.warning("[OC] opportunity_cost hook failed (%s) — continuing", _oc_hook_err)

            # ── Opportunity Capture: forward-return enrichment (2026-07-08 fix) ──
            # enrich_forward_returns() existed but was never invoked anywhere —
            # every skipped_opportunities.jsonl record stayed enrichment_status
            # ="pending" forever, so missed_alpha_score/forward_return in the
            # weekly report could never be computed from real data.
            try:
                import pandas as _pd_sor
                from src.analytics.skipped_opportunity_analytics import enrich_and_rewrite_store as _sor_enrich_rw

                def _sor_price_fetcher(sym: str, iso_date: str) -> "list[float] | None":
                    _p = CACHE_DIR / "ohlcv" / f"{sym}.parquet"
                    if not _p.exists():
                        return None
                    try:
                        _df = _pd_sor.read_parquet(_p)
                        _df.index = _pd_sor.to_datetime(_df.index)
                        _mask = _df.index >= _pd_sor.Timestamp(iso_date)
                        if not _mask.any():
                            return None
                        return _df.loc[_mask, "Close"].tolist()[:6]
                    except Exception:
                        return None

                _sor_n_enriched = _sor_enrich_rw(SKIPPED_OPPORTUNITY_FILE, _sor_price_fetcher)
                if _sor_n_enriched:
                    logger.info("[SKIPPED_OPP] enriched %d record(s)", _sor_n_enriched)
            except Exception as _sor_enrich_err:
                logger.warning("[SKIPPED_OPP] enrichment hook failed (%s) — continuing", _sor_enrich_err)

            # ── EVS: Executed vs Skipped Expectancy — context (DRY + LIVE) ───────
            # SCHEMA_VERSION 2 (2026-07-08 integrity fix). The old inline hook
            # here determined "executed" from result.orders (the pre-send
            # candidate list) — architecturally incapable of ever being True
            # for a LIVE run, since real order confirmation doesn't exist yet
            # at this point in the pipeline. DRY-mode records are written
            # immediately below (DRY never sends, so send_results=[] is
            # correct). LIVE-mode records are written later, after real
            # send_results are available (see "[EVS] LIVE record" below).
            _evs_today       = datetime.now(JST).strftime("%Y-%m-%d")
            _evs_run_ts      = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
            _evs_ps          = result.portfolio_summary or {}
            _evs_cash        = float(_evs_ps.get("available_cash") or 0)
            _evs_cap         = float(_eff_capital) if _eff_capital else 1.0
            _evs_cap_avail_pct = _evs_cash / _evs_cap if _evs_cap > 0 else 0.0
            _evs_heat        = float(_evs_ps.get("portfolio_heat", 0.0))
            try:
                _evs_regime = _regime_info_eo.get("regime", "unknown")
            except NameError:
                _evs_regime = "unknown"

            # ── CE Shadow Mode（2026-07-15 SSOT統合） — DRY/LIVE共通で記録 ──
            try:
                from src.live.ce_shadow_tracking import run_ce_shadow_tracking as _run_ce_shadow
                _run_ce_shadow(result.signals, order_objects)
            except Exception as _ce_import_err:
                logger.warning("[CE_SHADOW] module unavailable (FAIL_OPEN): %s", _ce_import_err)

            if not args.live:
                try:
                    from src.analytics.executed_vs_skipped_expectancy import (
                        build_opportunity_records as _evs_build,
                        append_opportunity        as _evs_append,
                        DEFAULT_STORE_PATH        as _evs_path,
                    )
                    _evs_recs = _evs_build(
                        signals=result.signals,
                        stage_audit=getattr(bridge, "_last_stage_audit", []),
                        send_results=[],
                        run_id=run_id,
                        run_timestamp=_evs_run_ts,
                        mode="DRY",
                        source_script="run_live_signal.py",
                        capital_available_pct=_evs_cap_avail_pct,
                        portfolio_heat=_evs_heat,
                        market_regime=_evs_regime,
                        max_positions=int(MAX_POS),
                    )
                    for _evs_rec in _evs_recs:
                        _evs_append(_evs_rec, _evs_path)
                    if _evs_recs:
                        logger.info(
                            "[EVS] DRY recorded %d opportunity records (executed=%d skipped=%d)",
                            len(_evs_recs),
                            sum(1 for r in _evs_recs if r.executed),
                            sum(1 for r in _evs_recs if not r.executed),
                        )
                except Exception as _evs_err:
                    logger.warning("[EVS] DRY build_opportunity_records failed (%s) — continuing", _evs_err)

                # ── Opportunity Capture integrity fix (2026-07-08) ────────────
                # Same run_id/run_timestamp as the EVS block above so
                # Stage Audit / EVS / Opportunity Capture join cleanly.
                try:
                    from src.analytics.skipped_opportunity_analytics import (
                        build_skipped_opportunity_records as _sor_build,
                        append_skipped_opportunity         as _sor_append,
                    )
                    _sor_recs = _sor_build(
                        signals=result.signals,
                        stage_audit=getattr(bridge, "_last_stage_audit", []),
                        send_results=[],
                        run_id=run_id,
                        run_timestamp=_evs_run_ts,
                        mode="DRY",
                        source_script="run_live_signal.py",
                        available_cash=_evs_cash,
                    )
                    for _sor_rec in _sor_recs:
                        _sor_append(_sor_rec, SKIPPED_OPPORTUNITY_FILE)
                    if _sor_recs:
                        logger.info("[SKIPPED_OPP] DRY recorded %d record(s)", len(_sor_recs))
                except Exception as _sor_err:
                    logger.warning("[SKIPPED_OPP] DRY build failed (%s) — continuing", _sor_err)
                try:
                    print_live_preview(result, order_objects)
                except Exception as _prev_err:
                    logger.warning("[LIVE_PREVIEW] failed (FAIL_OPEN): %s", _prev_err)

                print("\n" + "=" * 64)
                print("  ドライランのため発注は行いません。")
                print("  実際に発注するには --live オプションを付けて実行してください。")
                print("=" * 64)

                _emit_phase("report_generation", "start", run_id=run_id)
                # ── Research priority summary (dry-run) ───────────────────────
                try:
                    from src.analytics.research_priority_summary import run_research_priority_summary as _run_rps
                    from src.analytics.daily_leakage_summary import load_daily_summaries as _load_ls_hist
                    _rps_date = datetime.now(JST).strftime("%Y-%m-%d")
                    _rps_hist = _load_ls_hist(DAILY_LEAKAGE_FILE)
                    _rps_summary, _rps_path = _run_rps(
                        date=_rps_date,
                        run_id=run_id,
                        is_live=False,
                        summary_jsonl_path=RESEARCH_PRIORITY_FILE,
                        report_dir=RESEARCH_PRIORITY_REPORT_DIR,
                        historical_leakage=_rps_hist,
                        integrity_result=_integrity_result,
                    )
                    if _rps_path:
                        print(f"\n📊 研究優先レポート: {_rps_path}")
                    if _rps_summary:
                        print(f"   ボトルネック: {_rps_summary.dominant_bottleneck}"
                              f" | 優先領域: {_rps_summary.primary_research_category}"
                              f" | 信頼度: {_rps_summary.confidence_level}")
                except Exception as _rps_dr_err:
                    logger.warning("[RESEARCH_PRI] dry-run summary failed (%s)", _rps_dr_err)

                # ── Winner add-on candidates (dry-run report) ─────────────────
                try:
                    if _winner_confirmations:
                        _dry_confirmed = [_c for _c in _winner_confirmations if _c.confirmed]
                        if _dry_confirmed:
                            print(f"\n📈 アドオン候補 ({len(_dry_confirmed)}件) ─ DRY RUN（実発注なし）:")
                            print(f"  {'銘柄':<10} {'score':>6} {'pnl':>7} {'RSR':>6} {'rank':>5} {'保有日':>5}")
                            print("  " + "-" * 50)
                            for _dc in sorted(_dry_confirmed, key=lambda c: c.confidence_score, reverse=True):
                                print(
                                    f"  {_dc.symbol:<10}"
                                    f" {_dc.confidence_score:>6.3f}"
                                    f" {_dc.unrealized_pnl_pct:>+7.1%}"
                                    f" {_dc.rsr:>6.1f}"
                                    f" {_dc.rsr_rank:>5}"
                                    f" {_dc.hold_days:>4}d"
                                )
                        else:
                            _not_confirmed = [_c for _c in _winner_confirmations if not _c.confirmed]
                            print("\n📈 アドオン候補なし（保有中銘柄の確認条件不成立）")
                            for _nc in _not_confirmed:
                                logger.info(
                                    "[ADDON] not_confirmed: %s reasons=%s",
                                    _nc.symbol, _nc.fail_reasons,
                                )
                except Exception as _dry_addon_err:
                    logger.warning("[ADDON] dry-run report failed (%s)", _dry_addon_err)

                # ── Opportunity Cost report (dry-run display) ─────────────────
                try:
                    from src.analytics.opportunity_cost import (
                        format_opportunity_cost_report as _oc_fmt_report,
                    )
                    from src.paths import (
                        OPPORTUNITY_COST_EVENTS_FILE       as _oc_ef_dr,
                        OPPORTUNITY_COST_MATERIALIZED_FILE as _oc_mf_dr,
                    )
                    _oc_report_str = _oc_fmt_report(
                        events_file=_oc_ef_dr,
                        materialized_file=_oc_mf_dr,
                        date=datetime.now(JST).strftime("%Y-%m-%d"),
                    )
                    if _oc_report_str:
                        print("\n" + _oc_report_str)
                except Exception as _oc_dr_err:
                    logger.warning("[OC] dry-run report failed (%s)", _oc_dr_err)

                # ── OC 5D Materialization (Phase 5D) ──────────────────────────
                try:
                    from src.analytics.opportunity_cost_materializer import (
                        materialize_oc_events_5d        as _oc5d_mat,
                        format_oc_materialization_report as _oc5d_fmt,
                    )
                    from src.paths import (
                        OPPORTUNITY_COST_EVENTS_FILE as _oc5d_ev,
                        OC_5D_ENRICHED_FILE          as _oc5d_enr,
                    )
                    _oc5d_today = datetime.now(JST).strftime("%Y-%m-%d")
                    _oc5d_mat(
                        events_file=_oc5d_ev,
                        enriched_file=_oc5d_enr,
                        price_fetcher=lambda _sym, _dt: None,
                        today=_oc5d_today,
                        regime_provider=None,
                    )
                    _oc5d_rpt = _oc5d_fmt(_oc5d_enr, events_file=_oc5d_ev)
                    if _oc5d_rpt:
                        print("\n" + _oc5d_rpt)
                except Exception as _oc5d_err:
                    logger.warning("[OC5D] materialization report failed: %s", _oc5d_err)

                # ── Slot Pressure Forecast (Phase 5E) ──────────────────────────
                try:
                    from src.analytics.slot_pressure import (
                        build_slot_pressure_input_from_oc_data as _sp_build,
                        compute_slot_pressure                   as _sp_compute,
                        record_slot_pressure                    as _sp_record,
                        format_slot_pressure_report             as _sp_fmt,
                    )
                    from src.paths import (
                        OPPORTUNITY_COST_EVENTS_FILE as _sp_oc_evf,
                        SLOT_PRESSURE_FILE           as _sp_file,
                    )
                    _sp_today = datetime.now(JST).strftime("%Y-%m-%d")
                    _sp_held_list = [r for (_, r) in _cp_results] if _cp_results else []
                    _sp_inp = _sp_build(
                        current_positions=result.portfolio_summary.get(
                            "current_positions", len(_sp_held_list)
                        ),
                        max_positions=MAX_POS,
                        cp_results=_sp_held_list,
                        oc_events_file=_sp_oc_evf,
                        today=_sp_today,
                    )
                    _sp_result = _sp_compute(_sp_inp)
                    _sp_record(_sp_inp, _sp_today, _sp_file)
                    _sp_rpt = _sp_fmt(_sp_inp, _sp_result, _sp_today)
                    if _sp_rpt:
                        print("\n" + _sp_rpt)
                except Exception as _sp_err:
                    logger.warning("[SP] slot_pressure hook failed: %s", _sp_err)

                # ── Forward Continuation Outcome Intelligence (DRY) ───────────
                # Cross-telemetry: which signals best predict 5d forward return?
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.forward_continuation_outcome import (
                        run_fco_analysis   as _fco_run,
                        format_fco_report  as _fco_fmt,
                    )
                    from src.paths import (
                        CONTINUATION_PRIORITY_TELEMETRY_FILE as _fco_cp_file,
                        BREAKOUT_QUALITY_TELEMETRY_FILE      as _fco_bq_file,
                        FORWARD_CONTINUATION_OUTCOME_FILE    as _fco_out_file,
                    )
                    _fco_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _fco_result = _fco_run(
                        cp_file=_fco_cp_file,
                        bq_file=_fco_bq_file,
                        output_file=_fco_out_file,
                        today=_fco_today,
                    )
                    _fco_rpt = _fco_fmt(_fco_result)
                    if _fco_rpt:
                        print("\n" + _fco_rpt)
                    logger.info(
                        "[FCO] analysis: n=%d materialized=%d top=%s IR=%.4f",
                        _fco_result.n_records_total,
                        _fco_result.n_materialized,
                        _fco_result.top_signal,
                        _fco_result.top_signal_ir,
                    )
                except Exception as _fco_err:
                    logger.warning("[FCO] forward_continuation_outcome hook failed: %s", _fco_err)

                # ── Signal Attribution Intelligence (DRY) ─────────────────────
                # Quantifies which signals explain future 5d return.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.signal_attribution import (
                        run_signal_attribution   as _sa_run,
                        format_attribution_report as _sa_fmt,
                    )
                    from src.paths import (
                        CONTINUATION_PRIORITY_TELEMETRY_FILE as _sa_cp_file,
                        SLOT_PRESSURE_FILE                   as _sa_sp_file,
                        SIGNAL_ATTRIBUTION_FILE              as _sa_out_file,
                    )
                    _sa_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _sa_result = _sa_run(
                        cp_file=_sa_cp_file,
                        sp_file=_sa_sp_file,
                        output_file=_sa_out_file,
                        today=_sa_today,
                    )
                    _sa_rpt = _sa_fmt(_sa_result)
                    if _sa_rpt:
                        print("\n" + _sa_rpt)
                    logger.info(
                        "[SA] attribution: n=%d materialized=%d top=%s IR=%.4f",
                        _sa_result.n_records_total,
                        _sa_result.n_materialized,
                        _sa_result.top_signal,
                        _sa_result.top_signal_ir,
                    )
                except Exception as _sa_err:
                    logger.warning("[SA] signal_attribution hook failed: %s", _sa_err)

                # ── Capital Concentration Shadow Intelligence (DRY) ───────────
                # Observes whether priority-proportional weighting outperforms
                # equal-weight using materialized 5d returns.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.capital_concentration_shadow import (
                        run_concentration_shadow    as _ccs_run,
                        format_concentration_report as _ccs_fmt,
                    )
                    from src.paths import (
                        CONTINUATION_PRIORITY_TELEMETRY_FILE  as _ccs_cp_file,
                        CAPITAL_CONCENTRATION_SHADOW_FILE      as _ccs_out_file,
                    )
                    _ccs_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _ccs_result = _ccs_run(
                        cp_file=_ccs_cp_file,
                        output_file=_ccs_out_file,
                        today=_ccs_today,
                    )
                    _ccs_rpt = _ccs_fmt(_ccs_result)
                    if _ccs_rpt:
                        print("\n" + _ccs_rpt)
                    logger.info(
                        "[CCS] n=%d materialized=%d dates=%d alpha=%.4f pos_rate=%.1f%%",
                        _ccs_result.n_records_total,
                        _ccs_result.n_materialized,
                        _ccs_result.n_dates_analyzed,
                        _ccs_result.mean_concentration_alpha,
                        _ccs_result.positive_alpha_rate * 100,
                    )
                except Exception as _ccs_err:
                    logger.warning("[CCS] capital_concentration_shadow hook failed: %s", _ccs_err)

                # ── Priority Calibration Intelligence (DRY) ───────────────────
                # Verifies monotonicity of priority_score vs future return.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.priority_calibration import (
                        run_priority_calibration   as _pci_run,
                        format_calibration_report  as _pci_fmt,
                    )
                    from src.paths import (
                        CONTINUATION_PRIORITY_TELEMETRY_FILE as _pci_cp_file,
                        PRIORITY_CALIBRATION_FILE            as _pci_out_file,
                    )
                    _pci_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _pci_result = _pci_run(
                        cp_file=_pci_cp_file,
                        output_file=_pci_out_file,
                        today=_pci_today,
                    )
                    _pci_rpt = _pci_fmt(_pci_result)
                    if _pci_rpt:
                        print("\n" + _pci_rpt)
                    logger.info(
                        "[PCI] n=%d mat=%d buckets=%d mono=%.4f calib_err=%.4f",
                        _pci_result.n_records_total,
                        _pci_result.n_materialized,
                        _pci_result.n_buckets_populated,
                        _pci_result.priority_monotonicity,
                        _pci_result.priority_calibration_error,
                    )
                except Exception as _pci_err:
                    logger.warning("[PCI] priority_calibration hook failed: %s", _pci_err)

                # ── Hold Duration Calibration Intelligence (DRY) ─────────────
                # Quantifies hold_days vs future return relationship.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.hold_duration_calibration import (
                        run_hold_duration_calibration as _hdc_run,
                        format_duration_report        as _hdc_fmt,
                    )
                    from src.paths import (
                        CONTINUATION_PRIORITY_TELEMETRY_FILE as _hdc_cp_file,
                        HOLD_DURATION_CALIBRATION_FILE       as _hdc_out_file,
                    )
                    _hdc_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _hdc_result = _hdc_run(
                        cp_file=_hdc_cp_file,
                        output_file=_hdc_out_file,
                        today=_hdc_today,
                    )
                    _hdc_rpt = _hdc_fmt(_hdc_result)
                    if _hdc_rpt:
                        print("\n" + _hdc_rpt)
                    logger.info(
                        "[HDC] n=%d mat=%d buckets=%d best=%s spread=%.4f sup_delta=%.4f",
                        _hdc_result.n_records_total,
                        _hdc_result.n_materialized,
                        _hdc_result.n_buckets_populated,
                        _hdc_result.best_duration_bucket,
                        _hdc_result.duration_ev_spread,
                        _hdc_result.suppression_return_delta,
                    )
                except Exception as _hdc_err:
                    logger.warning("[HDC] hold_duration_calibration hook failed: %s", _hdc_err)

                # ── Evidence Promotion Engine (DRY) ───────────────────────────
                # Reads latest KPI records from HDC/CCS/PCI/SA/FCO and evaluates
                # which observation modules have crossed promotion thresholds.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.evidence_promotion import (
                        run_evidence_promotion   as _ep_run,
                        format_promotion_report  as _ep_fmt,
                    )
                    from src.paths import (
                        HOLD_DURATION_CALIBRATION_FILE       as _ep_hdc_file,
                        CAPITAL_CONCENTRATION_SHADOW_FILE    as _ep_ccs_file,
                        PRIORITY_CALIBRATION_FILE            as _ep_pci_file,
                        SIGNAL_ATTRIBUTION_FILE              as _ep_sa_file,
                        FORWARD_CONTINUATION_OUTCOME_FILE    as _ep_fco_file,
                        EVIDENCE_PROMOTION_FILE              as _ep_out_file,
                    )
                    _ep_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _ep_result = _ep_run(
                        hdc_file=_ep_hdc_file,
                        ccs_file=_ep_ccs_file,
                        pci_file=_ep_pci_file,
                        sa_file =_ep_sa_file,
                        fco_file=_ep_fco_file,
                        output_file=_ep_out_file,
                        today=_ep_today,
                    )
                    _ep_rpt = _ep_fmt(_ep_result)
                    if _ep_rpt:
                        print("\n" + _ep_rpt)
                    logger.info(
                        "[EP] evidence_promotion: promotable=%d observe=%d insufficient=%d top=%s",
                        _ep_result.n_promotable,
                        _ep_result.n_observe,
                        _ep_result.n_insufficient,
                        _ep_result.top_candidate or "none",
                    )
                except Exception as _ep_err:
                    logger.warning("[EP] evidence_promotion hook failed: %s", _ep_err)

                # ── Shadow Recommendation Engine (DRY) ────────────────────────
                # Maps PROMOTABLE→SHADOW_READY, OBSERVE→CONTINUE_OBSERVE, etc.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.shadow_recommendation import (
                        run_shadow_recommendation       as _sr_run,
                        format_recommendation_report    as _sr_fmt,
                    )
                    from src.paths import (
                        EVIDENCE_PROMOTION_FILE   as _sr_ep_file,
                        SHADOW_RECOMMENDATION_FILE as _sr_out_file,
                    )
                    _sr_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _sr_result = _sr_run(
                        ep_file=_sr_ep_file,
                        output_file=_sr_out_file,
                        today=_sr_today,
                    )
                    _sr_rpt = _sr_fmt(_sr_result)
                    if _sr_rpt:
                        print("\n" + _sr_rpt)
                    logger.info(
                        "[SR] shadow_recommendation: ready=%d observe=%d wait=%d top=%s",
                        _sr_result.n_shadow_ready,
                        _sr_result.n_continue_observe,
                        _sr_result.n_wait_data,
                        _sr_result.top_shadow_candidate or "none",
                    )
                except Exception as _sr_err:
                    logger.warning("[SR] shadow_recommendation hook failed: %s", _sr_err)

                # ── Shadow Outcome Tracker (DRY) ──────────────────────────────
                # Evaluates matured SHADOW_READY recs against current KPIs.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.shadow_outcome_tracker import (
                        run_shadow_outcome_tracker  as _sot_run,
                        format_outcome_report       as _sot_fmt,
                    )
                    from src.paths import (
                        SHADOW_RECOMMENDATION_FILE as _sot_sr_file,
                        EVIDENCE_PROMOTION_FILE    as _sot_ep_file,
                        SHADOW_OUTCOME_FILE         as _sot_out_file,
                    )
                    _sot_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _sot_result = _sot_run(
                        sr_file=_sot_sr_file,
                        ep_file=_sot_ep_file,
                        output_file=_sot_out_file,
                        today=_sot_today,
                    )
                    _sot_rpt = _sot_fmt(_sot_result.summary)
                    if _sot_rpt:
                        print("\n" + _sot_rpt)
                    if _sot_result.new_records:
                        logger.info(
                            "[SOT] shadow_outcome: new=%d completed=%d success_rate=%.1f%%",
                            len(_sot_result.new_records),
                            _sot_result.summary.n_completed,
                            _sot_result.summary.success_rate * 100,
                        )
                except Exception as _sot_err:
                    logger.warning("[SOT] shadow_outcome hook failed: %s", _sot_err)
                    print("[SHADOW_OUTCOME] skipped")

                # ── Promotion Readiness Engine (DRY) ──────────────────────────
                # Evaluates promotion readiness per candidate from outcome history.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.promotion_readiness import (
                        run_promotion_readiness  as _pr_run,
                        format_readiness_report  as _pr_fmt,
                    )
                    from src.paths import (
                        EVIDENCE_PROMOTION_FILE    as _pr_ep_file,
                        SHADOW_RECOMMENDATION_FILE as _pr_sr_file,
                        SHADOW_OUTCOME_FILE         as _pr_oc_file,
                        PROMOTION_READINESS_FILE    as _pr_out_file,
                    )
                    _pr_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _pr_result = _pr_run(
                        ep_file=_pr_ep_file,
                        sr_file=_pr_sr_file,
                        outcome_file=_pr_oc_file,
                        output_file=_pr_out_file,
                        today=_pr_today,
                    )
                    _pr_rpt = _pr_fmt(_pr_result.snapshot)
                    if _pr_rpt:
                        print("\n" + _pr_rpt)
                    if _pr_result.summary.promotion_ready_count > 0:
                        logger.info(
                            "[PR] promotion_readiness: ready=%d top=%s avg_score=%.1f",
                            _pr_result.summary.promotion_ready_count,
                            _pr_result.summary.top_ready_candidate or "none",
                            _pr_result.summary.avg_readiness_score,
                        )
                except Exception as _pr_err:
                    logger.warning("[PR] promotion_readiness hook failed: %s", _pr_err)
                    print("[PROMOTION_READINESS] skipped")

                # ── Promotion Candidate Registry (DRY) ───────────────────────
                # Ledger of PROMOTION_READY candidates for weekly governance.
                # Observation-only. FAIL_OPEN.
                try:
                    from src.analytics.promotion_candidate_registry import (
                        run_promotion_candidate_registry  as _pcr_run,
                        format_registry_report            as _pcr_fmt,
                    )
                    from src.paths import (
                        PROMOTION_READINESS_FILE            as _pcr_pr_file,
                        PROMOTION_CANDIDATE_REGISTRY_FILE   as _pcr_out_file,
                    )
                    _pcr_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _pcr_result = _pcr_run(
                        pr_file=_pcr_pr_file,
                        output_file=_pcr_out_file,
                        today=_pcr_today,
                    )
                    _pcr_rpt = _pcr_fmt(
                        _pcr_result.active_registry,
                        _pcr_result.summary,
                        _pcr_today,
                    )
                    if _pcr_rpt:
                        print("\n" + _pcr_rpt)
                    if _pcr_result.new_records:
                        logger.info(
                            "[PCR] registry: new=%d active=%d top=%s",
                            len(_pcr_result.new_records),
                            _pcr_result.summary.active_candidates,
                            _pcr_result.summary.top_candidate or "none",
                        )
                except Exception as _pcr_err:
                    logger.warning("[PCR] promotion_registry hook failed: %s", _pcr_err)
                    print("[PROMOTION_REGISTRY] skipped")

                # ── Promotion deployment planner (dry-run display) ────────────
                try:
                    from src.analytics.promotion_deployment_planner import (
                        run_promotion_deployment_planner as _pdp_run,
                        format_deployment_report as _pdp_fmt,
                    )
                    from src.paths import (
                        PROMOTION_CANDIDATE_REGISTRY_FILE as _pdp_reg_file,
                        PROMOTION_DEPLOYMENT_PLAN_FILE    as _pdp_out_file,
                    )
                    _pdp_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _pdp_result = _pdp_run(_pdp_reg_file, _pdp_out_file, _pdp_today)
                    _pdp_rpt    = _pdp_fmt(_pdp_result.planned_records, _pdp_result.summary)
                    if _pdp_rpt:
                        print("\n" + _pdp_rpt)
                    if _pdp_result.new_records:
                        logger.info(
                            "[PDP] planner: new=%d planned=%d top=%s",
                            len(_pdp_result.new_records),
                            _pdp_result.summary.planned_count,
                            _pdp_result.summary.highest_priority_candidate or "none",
                        )
                except Exception as _pdp_err:
                    logger.warning("[PDP] promotion_deployment_planner hook failed: %s", _pdp_err)
                    print("[PROMOTION_DEPLOYMENT] skipped")

                # ── Shadow deployment simulator (dry-run display) ─────────────
                try:
                    from src.analytics.shadow_deployment_simulator import (
                        run_shadow_deployment_simulator as _sds_run,
                        format_shadow_deployment_report as _sds_fmt,
                    )
                    from src.paths import (
                        PROMOTION_DEPLOYMENT_PLAN_FILE as _sds_plan_file,
                        SHADOW_DEPLOYMENT_FILE         as _sds_out_file,
                    )
                    _sds_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _sds_result = _sds_run(_sds_plan_file, _sds_out_file, _sds_today)
                    _sds_rpt    = _sds_fmt(_sds_result.deployed_records, _sds_result.summary)
                    if _sds_rpt:
                        print("\n" + _sds_rpt)
                    if _sds_result.new_records:
                        logger.info(
                            "[SDS] simulator: new=%d active=%d",
                            len(_sds_result.new_records),
                            _sds_result.summary.active_shadow_deployments,
                        )
                except Exception as _sds_err:
                    logger.warning("[SDS] shadow_deployment_simulator hook failed: %s", _sds_err)
                    print("[SHADOW_DEPLOYMENT] skipped")

                # ── Promotion impact tracker (dry-run display) ────────────────
                try:
                    from src.analytics.promotion_impact_tracker import (
                        run_promotion_impact_tracker as _pit_run,
                        format_impact_report         as _pit_fmt,
                    )
                    from src.paths import (
                        SHADOW_DEPLOYMENT_FILE          as _pit_shadow_file,
                        HOLD_DURATION_CALIBRATION_FILE  as _pit_hd_file,
                        CAPITAL_CONCENTRATION_SHADOW_FILE as _pit_cc_file,
                        PRIORITY_CALIBRATION_FILE       as _pit_pc_file,
                        SIGNAL_ATTRIBUTION_FILE         as _pit_sa_file,
                        FORWARD_CONTINUATION_OUTCOME_FILE as _pit_fco_file,
                        PROMOTION_IMPACT_TRACKER_FILE   as _pit_out_file,
                    )
                    _pit_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _pit_result = _pit_run(
                        shadow_file=_pit_shadow_file,
                        hold_duration_file=_pit_hd_file,
                        capital_conc_file=_pit_cc_file,
                        priority_cal_file=_pit_pc_file,
                        signal_attr_file=_pit_sa_file,
                        fco_file=_pit_fco_file,
                        output_file=_pit_out_file,
                        today=_pit_today,
                    )
                    _pit_rpt = _pit_fmt(_pit_result.impact_records, _pit_result.summary)
                    if _pit_rpt:
                        print("\n" + _pit_rpt)
                    if _pit_result.new_records:
                        logger.info(
                            "[PIT] tracker: new=%d positive=%d success_rate=%.1f%%",
                            len(_pit_result.new_records),
                            _pit_result.summary.n_positive,
                            _pit_result.summary.success_rate,
                        )
                except Exception as _pit_err:
                    logger.warning("[PIT] promotion_impact_tracker hook failed: %s", _pit_err)
                    print("[PROMOTION_IMPACT] skipped")

                # ── Promotion governance scoreboard (dry-run display) ─────────
                try:
                    from src.analytics.promotion_governance_scoreboard import (
                        run_promotion_governance_scoreboard as _pgs_run,
                        format_scoreboard_report            as _pgs_fmt,
                    )
                    from src.paths import (
                        PROMOTION_READINESS_FILE            as _pgs_readiness,
                        PROMOTION_CANDIDATE_REGISTRY_FILE   as _pgs_registry,
                        PROMOTION_IMPACT_TRACKER_FILE       as _pgs_impact,
                        SHADOW_OUTCOME_FILE                 as _pgs_shadow,
                        EVIDENCE_PROMOTION_FILE             as _pgs_evidence,
                        PROMOTION_GOVERNANCE_SCOREBOARD_FILE as _pgs_out,
                    )
                    _pgs_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _pgs_result = _pgs_run(
                        readiness_file=_pgs_readiness,
                        registry_file=_pgs_registry,
                        impact_file=_pgs_impact,
                        shadow_file=_pgs_shadow,
                        evidence_file=_pgs_evidence,
                        output_file=_pgs_out,
                        today=_pgs_today,
                    )
                    _pgs_rpt = _pgs_fmt(_pgs_result.ranked_records, _pgs_result.summary)
                    if _pgs_rpt:
                        print("\n" + _pgs_rpt)
                    if _pgs_result.new_records:
                        logger.info(
                            "[PGS] scoreboard: new=%d top=%s score=%.1f pf=%d",
                            len(_pgs_result.new_records),
                            _pgs_result.summary.top_candidate or "none",
                            _pgs_result.summary.top_governance_score,
                            _pgs_result.summary.n_promote_first,
                        )
                except Exception as _pgs_err:
                    logger.warning("[PGS] promotion_governance_scoreboard hook failed: %s", _pgs_err)
                    print("[PROMOTION_SCOREBOARD] skipped")

                # ── Promotion rollout manager (dry-run display) ───────────────
                try:
                    from src.analytics.promotion_rollout_manager import (
                        run_promotion_rollout_manager as _prm_run,
                        format_rollout_report         as _prm_fmt,
                    )
                    from src.paths import (
                        PROMOTION_GOVERNANCE_SCOREBOARD_FILE as _prm_sb_file,
                        PROMOTION_CANDIDATE_REGISTRY_FILE    as _prm_reg_file,
                        PROMOTION_IMPACT_TRACKER_FILE        as _prm_impact_file,
                        PROMOTION_ROLLOUT_MANAGER_FILE       as _prm_out_file,
                    )
                    _prm_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _prm_result = _prm_run(
                        scoreboard_file=_prm_sb_file,
                        registry_file=_prm_reg_file,
                        impact_file=_prm_impact_file,
                        output_file=_prm_out_file,
                        today=_prm_today,
                    )
                    _prm_rpt = _prm_fmt(_prm_result.state_records, _prm_result.summary)
                    if _prm_rpt:
                        print("\n" + _prm_rpt)
                    if _prm_result.new_records:
                        logger.info(
                            "[PRM] rollout: new=%d ps=%d mc=%d pr=%d rb=%d top=%s",
                            len(_prm_result.new_records),
                            _prm_result.summary.n_paper_shadow,
                            _prm_result.summary.n_micro_capital,
                            _prm_result.summary.n_production_ready,
                            _prm_result.summary.n_rollback_required,
                            _prm_result.summary.top_rollout_candidate or "none",
                        )
                except Exception as _prm_err:
                    logger.warning("[PRM] promotion_rollout_manager hook failed: %s", _prm_err)
                    print("[PROMOTION_ROLLOUT] skipped")

                # ── Production promotion gate (dry-run display) ───────────────
                try:
                    from src.analytics.production_promotion_gate import (
                        run_production_promotion_gate as _ppg_run,
                        format_gate_report            as _ppg_fmt,
                    )
                    from src.paths import (
                        PROMOTION_ROLLOUT_MANAGER_FILE        as _ppg_rollout_file,
                        PROMOTION_GOVERNANCE_SCOREBOARD_FILE  as _ppg_sb_file,
                        PROMOTION_IMPACT_TRACKER_FILE         as _ppg_impact_file,
                        PRODUCTION_PROMOTION_GATE_FILE        as _ppg_out_file,
                    )
                    _ppg_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _ppg_result = _ppg_run(
                        rollout_file=_ppg_rollout_file,
                        scoreboard_file=_ppg_sb_file,
                        impact_file=_ppg_impact_file,
                        output_file=_ppg_out_file,
                        today=_ppg_today,
                    )
                    _ppg_rpt = _ppg_fmt(_ppg_result)
                    if _ppg_rpt:
                        print("\n" + _ppg_rpt)
                    if not _ppg_result.skipped:
                        logger.info(
                            "[PPG] gate: approved=%d review=%d rejected=%d top=%s",
                            _ppg_result.summary.n_approved,
                            _ppg_result.summary.n_review_required,
                            _ppg_result.summary.n_rejected,
                            _ppg_result.summary.top_candidate or "none",
                        )
                except Exception as _ppg_err:
                    logger.warning("[PPG] production_promotion_gate hook failed: %s", _ppg_err)
                    print("[PRODUCTION_GATE] skipped")

                # ── Active experiment registry routing (dry-run) ──────────────
                try:
                    from src.experiments.active_experiment_registry import (
                        load_registry                       as _exp_load,
                        get_active_experiments              as _exp_active,
                        assign_symbol_to_experiment         as _exp_assign,
                        create_experiments_from_gate_approval as _exp_create,
                        route_symbols_to_experiments        as _exp_route,
                        append_assignment                   as _exp_append_asgn,
                        format_experiment_report            as _exp_fmt,
                    )
                    from src.paths import (
                        ACTIVE_EXPERIMENT_REGISTRY_FILE as _exp_reg_file,
                        EXPERIMENT_ASSIGNMENT_FILE      as _exp_asgn_file,
                    )
                    _exp_today = datetime.now(JST).strftime("%Y-%m-%d")
                    # Promotion integration: PPG APPROVED → auto-create experiments
                    try:
                        _ppg_gate_recs = [
                            {"decision": gr.decision, "candidate": gr.candidate}
                            for gr in _ppg_result.gate_records
                        ]
                        _exp_new = _exp_create(_ppg_gate_recs, _exp_reg_file, _exp_today)
                        for _en in _exp_new:
                            logger.info("[EXP] new experiment: %s feature=%s", _en.experiment_id, _en.feature_name)
                    except Exception:
                        pass
                    _exp_all    = _exp_load(_exp_reg_file)
                    _exp_active = _exp_active(_exp_all)
                    if _exp_active:
                        # Use buy signals for routing if available; fall back to empty list
                        try:
                            _exp_symbols = [str(s.get("symbol", s)) if isinstance(s, dict) else str(s)
                                            for s in (signals or [])]
                        except Exception:
                            _exp_symbols = []
                        _exp_routing = _exp_route(_exp_symbols, _exp_active, _exp_today)
                        for _sym, _asgns in _exp_routing.items():
                            for _asgn in _asgns:
                                _exp_append_asgn(_asgn, _exp_asgn_file)
                                if _asgn.group == "EXPERIMENT":
                                    logger.info("[EXP_TEST] %s → %s", _sym, _asgn.experiment_id)
                                    print(f"[EXP_ASSIGN] {_sym}: EXPERIMENT ({_asgn.experiment_id})")
                                else:
                                    logger.info("[EXP_CONTROL] %s → CONTROL", _sym)
                                    print(f"[EXP_ASSIGN] {_sym}: CONTROL ({_asgn.experiment_id})")
                        _exp_rpt = _exp_fmt(_exp_active, _exp_routing)
                        if _exp_rpt:
                            print("\n" + _exp_rpt)
                        logger.info("[EXP] active_experiments=%d symbols_routed=%d",
                                    len(_exp_active), len(_exp_routing))
                except Exception as _exp_err:
                    logger.warning("[EXP] experiment routing hook failed, defaulting CONTROL: %s", _exp_err)
                    print("[EXP_ASSIGN] all symbols defaulting to CONTROL (registry unavailable)")

                # ── Automatic rollback governance (dry-run) ───────────────────
                try:
                    from src.governance.automatic_rollback import (
                        evaluate_all_active_experiments as _rb_eval_all,
                        DECISION_ROLLED_BACK            as _rb_rolled,
                        DECISION_ROLLBACK_CANDIDATE     as _rb_cand,
                        DECISION_REVIEW_REQUIRED        as _rb_review,
                    )
                    from src.paths import (
                        ACTIVE_EXPERIMENT_REGISTRY_FILE as _rb_reg_file,
                        EXPERIMENT_PERFORMANCE_FILE     as _rb_perf_file,
                        ROLLBACK_GOVERNANCE_FILE        as _rb_out_file,
                    )
                    _rb_today     = datetime.now(JST).strftime("%Y-%m-%d")
                    _rb_decisions = _rb_eval_all(
                        registry_path=_rb_reg_file,
                        performance_path=_rb_perf_file,
                        rollback_path=_rb_out_file,
                        today=_rb_today,
                    )
                    for _rb_d in _rb_decisions:
                        if _rb_d.decision == _rb_rolled:
                            print(f"[RB_APPLIED] {_rb_d.experiment_id} → ROLLED_BACK")
                        elif _rb_d.decision == _rb_cand:
                            print(f"[RB_CANDIDATE] {_rb_d.experiment_id} day={_rb_d.consecutive_candidate_days}")
                        elif _rb_d.decision == _rb_review:
                            print(f"[RB_REVIEW] {_rb_d.experiment_id} triggers={_rb_d.trigger_reasons}")
                    if _rb_decisions:
                        logger.info("[RB] evaluated=%d decisions", len(_rb_decisions))
                except Exception as _rb_err:
                    logger.warning("[RB] automatic_rollback hook failed (fail-safe): %s", _rb_err)
                    print("[RB] skipped (fail-safe)")

                # ── Progressive capital escalation (dry-run) ──────────────────
                try:
                    from src.governance.progressive_capital_escalation import (
                        evaluate_all_experiment_escalations as _esc_eval_all,
                        DECISION_ESCALATE_LEVEL_1 as _esc_lv1,
                        DECISION_ESCALATE_LEVEL_2 as _esc_lv2,
                        DECISION_ESCALATE_LEVEL_3 as _esc_lv3,
                        DECISION_FULL_SCALE        as _esc_full,
                        DECISION_DEESCALATE        as _esc_down,
                    )
                    from src.paths import (
                        ACTIVE_EXPERIMENT_REGISTRY_FILE as _esc_reg_file,
                        EXPERIMENT_PERFORMANCE_FILE     as _esc_perf_file,
                        CAPITAL_ESCALATION_FILE         as _esc_out_file,
                    )
                    _esc_today     = datetime.now(JST).strftime("%Y-%m-%d")
                    _esc_decisions = _esc_eval_all(
                        registry_path=_esc_reg_file,
                        performance_path=_esc_perf_file,
                        escalation_path=_esc_out_file,
                        today=_esc_today,
                    )
                    for _esc_d in _esc_decisions:
                        if _esc_d.decision in (_esc_lv1, _esc_lv2, _esc_lv3, _esc_full):
                            print(f"[ESC_UP] {_esc_d.experiment_id} "
                                  f"L{_esc_d.previous_level}→L{_esc_d.new_level} "
                                  f"alloc={_esc_d.allocation_pct:.1%}")
                        elif _esc_d.decision == _esc_down:
                            print(f"[ESC_DOWN] {_esc_d.experiment_id} "
                                  f"L{_esc_d.previous_level}→L{_esc_d.new_level}")
                    if _esc_decisions:
                        logger.info("[ESC] evaluated=%d decisions", len(_esc_decisions))
                except Exception as _esc_err:
                    logger.warning("[ESC] escalation hook failed (fail-safe): %s", _esc_err)
                    print("[ESC] skipped (fail-safe)")

                # ── Improvement attribution ranking (dry-run) ─────────────────
                try:
                    from src.analytics.improvement_attribution_ranking import (
                        run_improvement_attribution_ranking as _iar_run,
                        format_ranking_report               as _iar_fmt,
                    )
                    from src.paths import (
                        EXPERIMENT_PERFORMANCE_FILE       as _iar_perf_file,
                        ACTIVE_EXPERIMENT_REGISTRY_FILE   as _iar_reg_file,
                        ROLLBACK_GOVERNANCE_FILE          as _iar_rb_file,
                        CAPITAL_ESCALATION_FILE           as _iar_esc_file,
                        IMPROVEMENT_RANKING_FILE          as _iar_out_file,
                    )
                    _iar_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _iar_result = _iar_run(
                        performance_path=_iar_perf_file,
                        registry_path=_iar_reg_file,
                        rollback_path=_iar_rb_file,
                        escalation_path=_iar_esc_file,
                        output_path=_iar_out_file,
                        today=_iar_today,
                    )
                    _iar_rpt = _iar_fmt(_iar_result)
                    if _iar_rpt:
                        print("\n" + _iar_rpt)
                    if _iar_result.ranked:
                        logger.info(
                            "[IAR] improvement_ranking: n=%d top=%s score=%.1f",
                            _iar_result.total_experiments,
                            _iar_result.top_experiment or "none",
                            _iar_result.top_score,
                        )
                except Exception as _iar_err:
                    logger.warning("[IAR] improvement_attribution_ranking hook failed: %s", _iar_err)
                    print("[IMPROVEMENT_RANKING] skipped")

                # ── System health audit (dry-run) ─────────────────────────────
                try:
                    from src.analytics.system_health_audit import (
                        run_system_health_audit as _sha_run,
                        format_health_report    as _sha_fmt,
                    )
                    from src.paths import (
                        EVIDENCE_PROMOTION_FILE          as _sha_ev,
                        SHADOW_RECOMMENDATION_FILE       as _sha_sr,
                        PROMOTION_READINESS_FILE         as _sha_pr,
                        PRODUCTION_PROMOTION_GATE_FILE   as _sha_gate,
                        EXPERIMENT_ASSIGNMENT_FILE       as _sha_assign,
                        CAPITAL_ESCALATION_FILE          as _sha_esc,
                        ROLLBACK_GOVERNANCE_FILE         as _sha_rb,
                        IMPROVEMENT_RANKING_FILE         as _sha_rank,
                        ACTIVE_EXPERIMENT_REGISTRY_FILE  as _sha_reg,
                        EXPERIMENT_PERFORMANCE_FILE      as _sha_perf,
                        SYSTEM_HEALTH_REPORT_FILE        as _sha_out,
                    )
                    _sha_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _sha_report = _sha_run(
                        evidence_file=_sha_ev,
                        shadow_file=_sha_sr,
                        promotion_file=_sha_pr,
                        gate_file=_sha_gate,
                        assignment_file=_sha_assign,
                        escalation_file=_sha_esc,
                        rollback_file=_sha_rb,
                        ranking_file=_sha_rank,
                        registry_file=_sha_reg,
                        performance_file=_sha_perf,
                        output_file=_sha_out,
                        today=_sha_today,
                    )
                    _sha_rpt = _sha_fmt(_sha_report)
                    if _sha_rpt:
                        print("\n" + _sha_rpt)
                    logger.info(
                        "[SHA] health_audit: status=%s checks=%d active=%d",
                        _sha_report.overall_status,
                        len(_sha_report.checks),
                        _sha_report.summary.active_experiments,
                    )
                except Exception as _sha_err:
                    logger.warning("[SHA] system_health_audit hook failed: %s", _sha_err)
                    print("[SYSTEM_HEALTH] skipped")

                # ── Weekly executive review (dry-run) ─────────────────────────
                try:
                    from src.analytics.weekly_executive_review import (
                        run_weekly_executive_review as _wer_run,
                        format_executive_review     as _wer_fmt,
                    )
                    from src.paths import (
                        EVIDENCE_PROMOTION_FILE          as _wer_ev,
                        SHADOW_RECOMMENDATION_FILE       as _wer_sr,
                        PROMOTION_READINESS_FILE         as _wer_pr,
                        PRODUCTION_PROMOTION_GATE_FILE   as _wer_gate,
                        ACTIVE_EXPERIMENT_REGISTRY_FILE  as _wer_reg,
                        CAPITAL_ESCALATION_FILE          as _wer_esc,
                        ROLLBACK_GOVERNANCE_FILE         as _wer_rb,
                        IMPROVEMENT_RANKING_FILE         as _wer_rank,
                        SYSTEM_HEALTH_REPORT_FILE        as _wer_health,
                        EXPERIMENT_PERFORMANCE_FILE      as _wer_perf,
                        WEEKLY_EXECUTIVE_REVIEW_FILE     as _wer_out,
                    )
                    _wer_today  = datetime.now(JST).strftime("%Y-%m-%d")
                    _wer_result = _wer_run(
                        evidence_file=_wer_ev,
                        shadow_file=_wer_sr,
                        promotion_file=_wer_pr,
                        gate_file=_wer_gate,
                        registry_file=_wer_reg,
                        escalation_file=_wer_esc,
                        rollback_file=_wer_rb,
                        ranking_file=_wer_rank,
                        health_report_file=_wer_health,
                        performance_file=_wer_perf,
                        output_file=_wer_out,
                        today=_wer_today,
                    )
                    _wer_rpt = _wer_fmt(_wer_result)
                    if _wer_rpt:
                        print("\n" + _wer_rpt)
                    logger.info(
                        "[WER] weekly_review: health=%s active=%d alerts=%d",
                        _wer_result.system_health.overall_status,
                        _wer_result.experiment_summary.total_active,
                        _wer_result.key_alerts.critical_count
                        + _wer_result.key_alerts.warning_count,
                    )
                except Exception as _wer_err:
                    logger.warning("[WER] weekly_executive_review hook failed: %s", _wer_err)
                    print("[WEEKLY_REVIEW] skipped")

                # ── Predictive candidate ranking (dry-run display) ─────────────
                try:
                    from src.live.predictive_candidate_ranker import (
                        rank_with_predictive as _rank_pred,
                        append_ranking_record as _append_pred_rec,
                        format_predictive_display as _fmt_pred,
                    )
                    _pred_buy_sigs = [
                        _s for _s in result.signals if _s.get("action") == "BUY"
                    ]
                    if _pred_buy_sigs:
                        _pred_ranking = _rank_pred(
                            buy_signals=_pred_buy_sigs,
                            predictive_result=_predictive_result,
                            efficiency_scores=_efficiency_scores,
                            run_ts=datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                            run_id=run_id,
                        )
                        print(_fmt_pred(_pred_ranking))
                        _append_pred_rec(_pred_ranking, PREDICTIVE_CANDIDATE_LOG_FILE)
                        logger.info(
                            "[PRED_RANK] ranked=%d predictive=%s efficiency=%s",
                            _pred_ranking.n_candidates,
                            _pred_ranking.predictive_available,
                            _pred_ranking.efficiency_available,
                        )
                except Exception as _pred_rank_err:
                    logger.warning("[PRED_RANK] dry-run ranking failed (%s)", _pred_rank_err)

                # ── Future Leader shadow observer (dry-run) ───────────────────
                # observation_only=True guaranteed — no order routing.
                # FAIL-CLOSED on log write; computation errors are fail-open.
                _fl_results: list = []
                _fl_mail_section: str = ""
                try:
                    from src.live.future_leader_screener import (
                        run_future_leader_hook as _run_fl,
                        format_future_leader_mail_section as _fmt_fl,
                    )
                    _fl_results = _run_fl(
                        universe               = RSR_UNIVERSE_62,
                        backtest_dataset_dir   = BACKTEST_DATASET_DIR,
                        default_data_version   = DEFAULT_DATA_VERSION or "",
                        cache_dir              = CACHE_DIR,
                        candidates_log_path    = FUTURE_LEADER_CANDIDATES_FILE,
                        reports_dir            = FUTURE_LEADER_REPORTS_DIR,
                        integrity_log_dir      = FUTURE_LEADER_INTEGRITY_DIR,
                        survivability_log_dir  = FUTURE_LEADER_SURVIVABILITY_DIR,
                        effective_capital      = float(_eff_capital),
                        run_id                 = run_id,
                        print_summary          = True,
                    )
                    _fl_mail_section = _fmt_fl(_fl_results) if _fl_results else ""
                    if _fl_results:
                        _obs_fl = [r for r in _fl_results if r.is_governance_observation_candidate]
                        if _obs_fl:
                            logger.info(
                                "[FUTURE_LEADER] observation candidates: %s",
                                [r.symbol for r in _obs_fl],
                            )
                except IOError as _fl_io_err:
                    logger.warning(
                        "[FUTURE_LEADER] log write FAILED (fail-closed): %s", _fl_io_err
                    )
                except Exception as _fl_err:
                    logger.warning(
                        "[FUTURE_LEADER] hook failed (%s) — observation skipped", _fl_err
                    )

                if _fl_mail_section:
                    print("\n" + _fl_mail_section)

                # ── Future Leader survivability materialization (DRY) ─────────
                try:
                    from src.analytics.future_leader_survivability import (
                        run_future_leader_survivability_materialization as _fl_surv,
                    )
                    _fl_surv_count = _fl_surv(
                        candidates_log_path   = FUTURE_LEADER_CANDIDATES_FILE,
                        integrity_log_dir     = FUTURE_LEADER_INTEGRITY_DIR,
                        survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                        backtest_dataset_dir  = BACKTEST_DATASET_DIR,
                        default_data_version  = DEFAULT_DATA_VERSION or "",
                        cache_dir             = CACHE_DIR,
                    )
                    if _fl_surv_count > 0:
                        logger.info(
                            "[FL_SURV] dry-run: materialized %d survivability records",
                            _fl_surv_count,
                        )
                except Exception as _fl_surv_err:
                    logger.warning(
                        "[FL_SURV] dry-run hook failed (fail-open): %s", _fl_surv_err
                    )

                # ── Future Leader failure clustering (DRY) ────────────────────
                try:
                    from src.analytics.future_leader_failure_clustering import (
                        run_future_leader_failure_materialization as _fl_fail,
                    )
                    _fl_fail_count = _fl_fail(
                        candidates_log_path   = FUTURE_LEADER_CANDIDATES_FILE,
                        survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                        failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                        backtest_dataset_dir  = BACKTEST_DATASET_DIR,
                        default_data_version  = DEFAULT_DATA_VERSION or "",
                        cache_dir             = CACHE_DIR,
                    )
                    if _fl_fail_count > 0:
                        logger.info(
                            "[FL_FAIL] dry-run: classified %d failure records",
                            _fl_fail_count,
                        )
                except Exception as _fl_fail_err:
                    logger.warning(
                        "[FL_FAIL] dry-run hook failed (fail-open): %s", _fl_fail_err
                    )

                # ── Future Leader alpha persistence half-life (DRY) ───────────
                try:
                    from src.analytics.future_leader_half_life import (
                        run_future_leader_half_life_hook as _fl_hl,
                    )
                    _fl_hl(
                        survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                        failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                        reports_dir           = FUTURE_LEADER_REPORTS_DIR,
                    )
                except Exception as _fl_hl_err:
                    logger.warning(
                        "[FL_HALF_LIFE] dry-run hook failed (fail-open): %s", _fl_hl_err
                    )

                # ── Future Leader regime segmentation (DRY) ──────────────────
                try:
                    from src.analytics.future_leader_regime import (
                        run_future_leader_regime_hook as _fl_regime,
                    )
                    _fl_regime(
                        survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                        failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                        regime_log_dir        = FUTURE_LEADER_REGIME_DIR,
                        reports_dir           = FUTURE_LEADER_REPORTS_DIR,
                        backtest_dataset_dir  = BACKTEST_DATASET_DIR,
                        default_data_version  = DEFAULT_DATA_VERSION or "",
                        cache_dir             = CACHE_DIR,
                    )
                except Exception as _fl_regime_err:
                    logger.warning(
                        "[FL_REGIME] dry-run hook failed (fail-open): %s", _fl_regime_err
                    )

                try:
                    from src.analytics.future_leader_archetype import (
                        run_future_leader_archetype_hook as _fl_arch,
                    )
                    _fl_arch(
                        failure_log_dir   = FUTURE_LEADER_FAILURE_DIR,
                        regime_log_dir    = FUTURE_LEADER_REGIME_DIR,
                        archetype_log_dir = FUTURE_LEADER_ARCHETYPE_DIR,
                        reports_dir       = FUTURE_LEADER_REPORTS_DIR,
                    )
                    logger.info("[FL_ARCH] dry-run: archetype hook complete")
                except Exception as _fl_arch_err:
                    logger.warning(
                        "[FL_ARCH] dry-run hook failed (fail-open): %s", _fl_arch_err
                    )

                try:
                    from src.analytics.future_leader_transition import (
                        run_future_leader_transition_hook as _fl_trans,
                    )
                    _fl_trans(
                        failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                        archetype_log_dir     = FUTURE_LEADER_ARCHETYPE_DIR,
                        survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                        regime_log_dir        = FUTURE_LEADER_REGIME_DIR,
                        transition_log_dir    = FUTURE_LEADER_TRANSITION_DIR,
                        reports_dir           = FUTURE_LEADER_REPORTS_DIR,
                    )
                    logger.info("[FL_TRANS] dry-run: transition hook complete")
                except Exception as _fl_trans_err:
                    logger.warning(
                        "[FL_TRANS] dry-run hook failed (fail-open): %s", _fl_trans_err
                    )

                # ── AUTO_PROMOTE_SAFE_V2 report ───────────────────────────────
                try:
                    from src.universe.auto_promote_safe_v2 import format_probation_report as _v2_rpt
                    from src.paths import PROBATION_PROMOTIONS_FILE as _v2_pf, PROBATION_OUTCOMES_FILE as _v2_of
                    _v2_section = _v2_rpt(_v2_pf, _v2_of)
                    if _v2_section:
                        print(_v2_section)
                except Exception as _v2_rpt_err:
                    logger.warning("[V2] report failed: %s", _v2_rpt_err)

                # ── AUTO_PROMOTE_EXPLAINABILITY report ────────────────────────
                try:
                    from src.universe.auto_promote_safe_v2 import (
                        format_explainability_report as _v2_exp_rpt,
                        format_rejection_stats       as _v2_stats_rpt,
                    )
                    from src.paths import (
                        PROBATION_REJECTION_FILE  as _v2_rej_f,
                        PROBATION_PROMOTIONS_FILE as _v2_pf2,
                    )
                    _v2_exp = _v2_exp_rpt(_v2_rej_f, _v2_pf2)
                    if _v2_exp:
                        print(_v2_exp)
                    _v2_stats = _v2_stats_rpt(_v2_rej_f)
                    if _v2_stats:
                        print(_v2_stats)
                except Exception as _v2_exp_err:
                    logger.warning("[V2] explainability report failed: %s", _v2_exp_err)

                # ── Entry Timing Intelligence report ──────────────────────────
                try:
                    from src.entry import (
                        EntryTimingResult as _ETResult,
                        format_et_report_section  as _et_rpt_dry,
                        append_et_telemetry       as _et_telem_dry,
                        materialize_et_returns    as _et_mat_dry,
                        CONFIDENCE_HIGH as _ET_HIGH, CONFIDENCE_MEDIUM as _ET_MED,
                        CONFIDENCE_LOW  as _ET_LOW,
                    )
                    from src.paths import ENTRY_TIMING_TELEMETRY_FILE as _et_tpath_dry
                    # Reconstruct lightweight result objects from signal dicts
                    _et_dry_scores: dict = {}
                    for _s_et in result.signals:
                        _et_sc = _s_et.get("entry_timing_score")
                        _et_cf = _s_et.get("entry_timing_confidence")
                        if _et_sc is not None and _et_cf is not None:
                            _sym_et = _s_et.get("symbol", "")
                            _et_dry_scores[_sym_et] = _ETResult(
                                symbol             = _sym_et,
                                score              = float(_et_sc),
                                confidence         = str(_et_cf),
                                action             = str(_s_et.get("entry_timing_action", "NORMAL")),
                                breakout_component = 50.0,
                                pullback_component = 50.0,
                                trend_component    = 50.0,
                                market_component   = 60.0,
                                phase              = str(_s_et.get("entry_timing_phase", "normal")),
                            )
                    # Telemetry write for ALL BUY signal candidates
                    # top_k_symbols は per-symbol RSR/breakout 限定; trend_follow 候補が
                    # 含まれないため result.signals から signal==1 銘柄を直接列挙する
                    _today_str_et = datetime.now(JST).strftime("%Y-%m-%d")
                    _et_sigs_map  = {_s.get("symbol", ""): _s for _s in result.signals}
                    _et_telem_syms = [
                        _s.get("symbol", "") for _s in result.signals
                        if _s.get("signal") == 1 and not _s.get("currently_holding", False)
                        and _s.get("symbol", "")
                    ]
                    for _sym_telem in _et_telem_syms:
                        if _sym_telem in _et_dry_scores:
                            _et_res_telem = _et_dry_scores[_sym_telem]
                        else:
                            _et_sig_ref = _et_sigs_map.get(_sym_telem, {})
                            _et_res_telem = _ETResult(
                                symbol             = _sym_telem,
                                score              = float(_et_sig_ref.get("rsr") or 50.0),
                                confidence         = _ET_MED,
                                action             = "NORMAL",
                                breakout_component = 50.0,
                                pullback_component = 50.0,
                                trend_component    = 50.0,
                                market_component   = 50.0,
                                phase              = "normal",
                            )
                        _act_taken = "ENTERED" if any(
                            o.get("symbol") == _sym_telem and o.get("side") == "BUY"
                            for o in result.orders
                        ) else "WATCHED"
                        _et_telem_dry(
                            _et_res_telem,
                            action_taken   = _act_taken,
                            telemetry_path = _et_tpath_dry,
                            date_str       = _today_str_et,
                        )
                    # Forward return materialization (FAIL_OPEN)
                    try:
                        import pandas as _pd_et_dry
                        def _et_ohlcv_loader_dry(sym: str):
                            _ep = CACHE_DIR / "ohlcv" / f"{sym}.parquet"
                            return _pd_et_dry.read_parquet(_ep) if _ep.exists() else None
                        _et_mat_dry(_et_tpath_dry, _et_ohlcv_loader_dry)
                    except Exception:
                        pass
                    # Print report
                    if _et_dry_scores:
                        _et_section_dry = _et_rpt_dry(_et_dry_scores, _et_tpath_dry)
                        if _et_section_dry:
                            print(_et_section_dry)
                except Exception as _et_rpt_dry_err:
                    logger.warning("[ET] dry report failed: %s", _et_rpt_dry_err)

                # ── Entry Timing Promotion evaluation (DRY) ───────────────────
                try:
                    from src.entry.entry_timing_promotion import (
                        run_entry_timing_promotion    as _et_promo_run_dry,
                        format_et_promotion_section   as _et_promo_fmt_dry,
                    )
                    from src.paths import (
                        ENTRY_TIMING_TELEMETRY_FILE  as _et_tfile_dry,
                        ENTRY_TIMING_PROMOTION_FILE  as _et_pfile_dry,
                        ENTRY_TIMING_HISTORY_FILE    as _et_hfile_dry,
                        ENTRY_TIMING_REPORTS_DIR     as _et_rdir_dry,
                    )
                    _et_rdir_dry.mkdir(parents=True, exist_ok=True)
                    _et_promo_state_dry = _et_promo_run_dry(
                        telemetry_file  = _et_tfile_dry,
                        promotion_file  = _et_pfile_dry,
                        history_file    = _et_hfile_dry,
                        today_str       = datetime.now(JST).strftime("%Y-%m-%d"),
                    )
                    _et_promo_sec_dry = _et_promo_fmt_dry(
                        _et_pfile_dry, datetime.now(JST).strftime("%Y-%m-%d")
                    )
                    if _et_promo_sec_dry:
                        print(_et_promo_sec_dry)
                except Exception as _et_promo_dry_err:
                    logger.warning("[ET_PROMO] dry promotion eval failed: %s", _et_promo_dry_err)

                # ── Position Sizing Intelligence report (DRY) ─────────────────
                try:
                    from src.portfolio.position_sizing_intelligence import (
                        PositionSizingSignal      as _PSSig_dry,
                        append_ps_telemetry       as _ps_telem_fn_dry,
                        compute_ps_kpis           as _ps_kpis_fn_dry,
                        format_ps_report_section  as _ps_rpt_fn_dry,
                    )
                    from src.paths import POSITION_SIZING_TELEMETRY_FILE as _ps_tpath_dry
                    _ps_dry_sigs = []
                    _ps_rsr_dry:  dict = {}
                    _ps_et_dry:   dict = {}
                    for _s_psd in result.signals:
                        _conv_d = _s_psd.get("conviction_score")
                        _vw_d   = _s_psd.get("virtual_weight")
                        if _conv_d is not None and _vw_d is not None and _s_psd.get("signal") == 1:
                            _sym_psd = _s_psd["symbol"]
                            _ps_dry_sigs.append(_PSSig_dry(
                                symbol           = _sym_psd,
                                conviction_score = float(_conv_d),
                                virtual_weight   = float(_vw_d),
                                component_scores = {},
                                reason_codes     = [],
                                computed_at      = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                            ))
                            _ps_rsr_dry[_sym_psd] = _s_psd.get("rsr")
                            _ps_et_dry[_sym_psd]  = _s_psd.get("entry_timing_score")
                    if _ps_dry_sigs:
                        _n_psd = len(_ps_dry_sigs)
                        _eq_w_dry = {sig.symbol: 1.0 / _n_psd for sig in _ps_dry_sigs}
                        _ps_telem_fn_dry(
                            signals              = _ps_dry_sigs,
                            actual_weights       = _eq_w_dry,
                            rsr_scores           = _ps_rsr_dry,
                            entry_timing_scores  = _ps_et_dry,
                            future_leader_scores = {},
                            telemetry_path       = _ps_tpath_dry,
                            date_str             = datetime.now(JST).strftime("%Y-%m-%d"),
                        )
                        _ps_kpis_dry = _ps_kpis_fn_dry(_ps_tpath_dry)
                        _ps_sec_dry  = _ps_rpt_fn_dry(_ps_dry_sigs, _ps_kpis_dry)
                        if _ps_sec_dry:
                            print(_ps_sec_dry)
                except Exception as _ps_dry_err:
                    logger.warning("[PSI] dry report failed: %s", _ps_dry_err)

                # ── Position Sizing Promotion evaluation (DRY) ────────────────
                try:
                    from src.portfolio.position_sizing_promotion import (
                        run_position_sizing_promotion as _psp_run_dry,
                        format_psp_report_section as _psp_fmt_dry,
                    )
                    from src.paths import (
                        POSITION_SIZING_TELEMETRY_FILE as _psp_tfile_dry,
                        POSITION_SIZING_PROMOTION_FILE as _psp_pfile_dry,
                        POSITION_SIZING_HISTORY_FILE as _psp_hfile_dry,
                        POSITION_SIZING_REPORTS_DIR as _psp_rdir_dry,
                        OHLCV_DIR as _psp_ohlcv_dry,
                    )
                    _psp_rdir_dry.mkdir(parents=True, exist_ok=True)
                    _psp_run_dry(
                        telemetry_file=_psp_tfile_dry,
                        promotion_file=_psp_pfile_dry,
                        history_file=_psp_hfile_dry,
                        ohlcv_dir=_psp_ohlcv_dry,
                        today_str=datetime.now(JST).strftime("%Y-%m-%d"),
                    )
                    _psp_sec_dry = _psp_fmt_dry(_psp_pfile_dry, datetime.now(JST).strftime("%Y-%m-%d"))
                    if _psp_sec_dry:
                        print(_psp_sec_dry)
                except Exception as _psp_dry_err:
                    logger.warning("[PSP] dry promotion eval failed: %s", _psp_dry_err)

                # ── Breakout Quality Intelligence report ──────────────────────
                try:
                    from src.analytics.breakout_quality import format_bq_report_section as _bq_rpt_dry
                    from src.paths import BREAKOUT_QUALITY_TELEMETRY_FILE as _bq_rpt_path_dry
                    _bq_section_dry = _bq_rpt_dry(_bq_rpt_path_dry)
                    if _bq_section_dry:
                        print(_bq_section_dry)
                except Exception as _bq_rpt_dry_err:
                    logger.warning("[BQ] dry report failed: %s", _bq_rpt_dry_err)

                # ── Continuation Priority Intelligence report ─────────────────
                try:
                    from src.analytics.continuation_priority import (
                        format_priority_ranking_table as _cp_table_dry,
                        format_cp_kpi_section         as _cp_kpi_dry,
                    )
                    from src.paths import CONTINUATION_PRIORITY_TELEMETRY_FILE as _cp_rpt_path_dry
                    _cp_rank_dry = _cp_table_dry(_cp_results)
                    if _cp_rank_dry:
                        print(_cp_rank_dry)
                    _cp_kpi_str_dry = _cp_kpi_dry(_cp_rpt_path_dry)
                    if _cp_kpi_str_dry:
                        print(_cp_kpi_str_dry)
                except Exception as _cp_rpt_dry_err:
                    logger.warning("[CP] dry report failed: %s", _cp_rpt_dry_err)

                # ── [RC] Quality Replacement Engine — Shadow判定 DRY (Study57/58A) ──
                # FAIL_OPEN: 発注変更なし。DRY/LIVE共通でshadow audit実施。
                _signal_rsr_map: dict[str, float] = {
                    _s["symbol"]: float(_s.get("rsr", 0.0))
                    for _s in result.signals
                    if _s.get("symbol")
                }
                try:
                    # StrategyConfig dataclassにresearch_candidatesがないため直接YAMLをロード
                    import yaml as _qr_yaml_dry
                    from types import SimpleNamespace as _SNS_dry
                    _qr_raw_dry = (_qr_yaml_dry.safe_load(
                        STRATEGY_CONFIG_FILE.read_text(encoding="utf-8")
                    ) or {}).get("research_candidates", {}).get("quality_replacement")
                    _rc_qr_cfg_dry = _SNS_dry(**_qr_raw_dry) if isinstance(_qr_raw_dry, dict) else None
                    if _rc_qr_cfg_dry is not None:
                        from src.research_candidate.quality_replacement import (
                            run_quality_replacement_shadow as _qr_run_dry,
                        )
                        from src.paths import (
                            QUAL_REPLACE_AUDIT_FILE    as _qr_audit_f_dry,
                            QUAL_REPLACE_MISSED_FILE   as _qr_missed_f_dry,
                            QUAL_REPLACE_OUTCOMES_FILE as _qr_outcomes_f_dry,
                        )
                        import json as _json_qr_dry
                        _ps_qr_dry_path = RUNTIME_DIR / "portfolio_state.json"
                        _ps_qr_dry: dict = {}
                        if _ps_qr_dry_path.exists():
                            try:
                                _ps_qr_dry = _json_qr_dry.loads(
                                    _ps_qr_dry_path.read_text(encoding="utf-8")
                                )
                            except Exception:
                                pass
                        from src.paths import CACHE_DIR as _qr_cache_dry
                        _qr_result_dry = _qr_run_dry(
                            today           = datetime.now(JST).strftime("%Y-%m-%d"),
                            run_id          = run_id,
                            mode            = "DRY",
                            signals         = list(result.signals),
                            portfolio_state = _ps_qr_dry,
                            cfg             = _rc_qr_cfg_dry,
                            audit_file      = _qr_audit_f_dry,
                            missed_file     = _qr_missed_f_dry,
                            outcomes_file   = _qr_outcomes_f_dry,
                            ohlcv_cache_dir = _qr_cache_dry / "ohlcv",
                        )
                        if _qr_result_dry.get("swap_ready"):
                            logger.warning(
                                "[QR_SHADOW] DRY SWAP_READY: weakest=%s(QS=%.1f) cand=%s(QS=%.1f)"
                                " — shadow only, no order sent",
                                _qr_result_dry["weakest"]["symbol"],
                                _qr_result_dry["weakest"]["qs"],
                                _qr_result_dry["best_cand"]["symbol"],
                                _qr_result_dry["best_cand"]["qs"],
                            )
                except Exception as _rc_qr_dry_err:
                    logger.warning(
                        "[RC_QUAL_REPLACE] dry shadow hook failed (%s) — continuing",
                        _rc_qr_dry_err,
                    )

                # ── Phase9 QR Shadow Audit (DRY) ─────────────────────────────
                # Observation-only. FAIL_OPEN. No order changes.
                try:
                    from src.analytics.qr_shadow_audit import run_phase9_all as _p9_run_dry
                    from src.paths import (
                        QUAL_REPLACE_AUDIT_FILE  as _p9_audit_dry,
                        QUAL_REPLACE_MISSED_FILE as _p9_missed_dry,
                        QUAL_REPLACE_P9A_FILE    as _p9a_f_dry,
                        QUAL_REPLACE_P9B_FILE    as _p9b_f_dry,
                        QUAL_REPLACE_P9C_FILE    as _p9c_f_dry,
                        QUAL_REPLACE_P9D_FILE    as _p9d_f_dry,
                        QUAL_REPLACE_P9E_DIR     as _p9e_d_dry,
                    )
                    _p9_cache_dry = Path("cache")
                    _p9_result_dry = _p9_run_dry(
                        today        = datetime.now(JST).strftime("%Y-%m-%d"),
                        run_id       = run_id,
                        audit_file   = _p9_audit_dry,
                        missed_file  = _p9_missed_dry,
                        p9a_file     = _p9a_f_dry,
                        p9b_file     = _p9b_f_dry,
                        p9c_file     = _p9c_f_dry,
                        p9d_file     = _p9d_f_dry,
                        p9e_dir      = _p9e_d_dry,
                        ohlcv_cache  = _p9_cache_dry / "ohlcv",
                    )
                    logger.info("[PHASE9_QR] dry audit complete: %s", _p9_result_dry)
                except Exception as _p9_dry_err:
                    logger.warning("[PHASE9_QR] dry hook failed (%s) — continuing", _p9_dry_err)

                _emit_phase("report_generation", "complete", run_id=run_id)
                _emit_phase("persistence", "complete", run_id=run_id)

                # ── Deterministic exit validation (DRY mode) ─────────────────
                _emit_phase("shutdown_cleanup", "start", run_id=run_id)
                _dry_audit = _shutdown_audit(run_id=run_id)
                _non_daemon_ok = _dry_audit.get("non_daemon_count", 1) <= 1  # MainThread only
                _child_ok      = _dry_audit.get("active_child_processes", 0) == 0
                if not _non_daemon_ok:
                    logger.warning(
                        "[EXIT_VALIDATION] non-daemon threads alive at DRY exit: %d — %s",
                        _dry_audit.get("non_daemon_count", 0),
                        _dry_audit.get("non_daemon_names", []),
                    )
                if not _child_ok:
                    logger.warning(
                        "[EXIT_VALIDATION] child processes alive at DRY exit: %s",
                        _dry_audit.get("child_process_pids", []),
                    )
                _emit_phase("final_exit", "start", run_id=run_id,
                            extra={"exit_code": 0, "mode": "dry"})
                return 0

            if not order_objects:
                print("\n発注なし。終了します。")
                return 0

            # ----------------------------------------------------------------
            # 発注前安全チェック（3層）
            # 1. オーダーロック（銘柄単位の当日重複チェック）
            # 2. MAX_SYMBOL_ORDERS / MAX_DAILY_ORDERS（数量上限）
            # 3. ExecutionJournal 重複チェック（クラッシュリカバリー）
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
                # 層3: ExecutionJournal 重複チェック（クラッシュ後のリカバリー）
                if _exec_journal.has_active(run_id, sym, o.side):
                    blocked.append(f"{sym}: ExecutionJournal に記録済み（クラッシュリカバリー）")
                    continue
                # 層2: 件数上限チェック
                sym_count = today_per_sym.get(sym, 0)
                if today_total + len(filtered_orders) >= MAX_DAILY_ORDERS:
                    blocked.append(f"{sym}: 本日の発注上限({MAX_DAILY_ORDERS}件)に到達")
                elif sym_count >= MAX_SYMBOL_ORDERS:
                    blocked.append(
                        f"{sym}: 銘柄別上限({MAX_SYMBOL_ORDERS}件/日)に到達"
                        f"（本日既に{sym_count}件）"
                    )
                else:
                    filtered_orders.append(o)

            if blocked:
                print("\n[安全設計] 以下の注文は除外されました:")
                for msg in blocked:
                    print(f"  ⚠ {msg}")
                if not filtered_orders:
                    print("発注可能な注文がありません。終了します。")
                    if not args.no_save:
                        save_live_logs(run_id, result, [])
                    return 0
                order_objects = filtered_orders

            if today_total > 0:
                print(
                    f"\n[安全設計] 本日の発注履歴: 合計{today_total}件"
                    f" / 上限{MAX_DAILY_ORDERS}件"
                )

            order_dicts = result.orders
            if not args.yes:
                confirmed = confirm_live_orders(order_dicts)
                if not confirmed:
                    print("発注をキャンセルしました。")
                    return 0

            # ── 取引時間チェック: 09:00 未満なら待機 ────────────────────────
            import time as _time
            _now = datetime.now(JST)
            _market_open = _now.replace(hour=9, minute=0, second=5, microsecond=0)
            if _now < _market_open:
                _wait_sec = (_market_open - _now).total_seconds()
                print(f"\n⏰ 取引開始まで {_wait_sec:.0f}秒待機中... (09:00:05 発注予定)")
                _time.sleep(_wait_sec)
                print("  → 待機完了。発注を開始します。")

            # ── Stage: broker_sync (gap stop check via kabu API) ─────────────
            logger.info("ギャップダウンチェック中（board 取得）...")
            _gap_orders = order_objects
            try:
                _gap_orders = _supervisor.run_stage(
                    "broker_sync",
                    lambda: bridge.check_gap_stops(order_objects),
                    timeout_sec=20,
                    retry_budget=0,
                )
            except (StageTimeout, StageError) as _gap_e:
                logger.warning(
                    "broker_sync stage failed, skipping gap check: %s", _gap_e
                )
            order_objects = _gap_orders

            # ExecutionJournal: record pending before broker submission
            for o in order_objects:
                try:
                    _exec_journal.record_pending(
                        run_id, o.symbol, o.side, o.qty,
                        float(getattr(o, "estimated_price", 0.0)),
                    )
                except Exception as _je:
                    logger.warning("[JOURNAL] record_pending failed: %s", _je)

            # ── Leakage analytics: signal lookup + accumulator (Task 2) ─────────
            _leakage_skipped_records: list = []
            _leakage_signal_lookup: dict = {
                _s.get("symbol", ""): _s
                for _s in result.signals if _s.get("symbol")
            }
            _leakage_available_cash = float(
                (result.portfolio_summary or {}).get("available_cash", 0.0)
            )

            # ── Phase 5A: intent audit + exposure-aware sizing ────────────────
            try:
                import time as _time_5a
                from src.allocation import (
                    compute_exposure_aware_size, append_intent_record,
                    AllocationIntentRecord,
                )
                from src.truth import load_capital_truth
                from src.paths import CAPITAL_TRUTH_FILE, ALLOCATION_INTENT_FILE

                # Truth confidence: fail-open (missing file = 1.0)
                _truth_confidence_5a = 1.0
                _truth_regime_5a = "unknown"
                try:
                    _cap_truth_5a = load_capital_truth(CAPITAL_TRUTH_FILE)
                    _truth_confidence_5a = _cap_truth_5a.confidence
                    _truth_regime_5a = (
                        "allowed" if _cap_truth_5a.deployment_allowed else "blocked"
                    )
                except Exception:
                    pass

                _now_sec_5a = _time_5a.time()
                _phase5a_orders = []
                for _o5a in order_objects:
                    _sym5a = _o5a.symbol
                    _req_qty = int(getattr(_o5a, "qty", 1))
                    # ── V2 probation: cap allocation at 0.25x (min 100 shares) ──
                    if _sym5a in _probation_active_symbols:
                        _req_qty = max(100, (int(_req_qty * 0.25) // 100) * 100)
                        logger.info("[V2] probation %s qty→%d (0.25x)", _sym5a, _req_qty)
                    _est_price = float(getattr(_o5a, "estimated_price", 0.0))
                    _eff_score = _efficiency_scores.get(_sym5a, 0.5)
                    # Portfolio concentration proxy: order_value / portfolio_equity
                    _order_val = _req_qty * _est_price
                    _port_eq = float(_eff_capital) if _eff_capital > 0 else 1.0
                    _conc_proxy = round(_order_val / _port_eq, 4) if _port_eq > 0 else 0.0
                    _ts5a = datetime.now(JST).isoformat()

                    _sizing = compute_exposure_aware_size(
                        requested_size=_req_qty,
                        truth_confidence=_truth_confidence_5a,
                        effective_independent_n=_phase5a_effective_n,
                        portfolio_concentration=_conc_proxy,
                        capital_efficiency_score=_eff_score,
                        portfolio_equity=_port_eq,
                        estimated_price=_est_price,
                        state_load_time_sec=_alloc_state_load_time,
                        now_sec=_now_sec_5a,
                    )
                    _intent = AllocationIntentRecord(
                        timestamp=_ts5a,
                        symbol=_sym5a,
                        signal_id=getattr(_o5a, "signal_id", ""),
                        strategy_id="fujiko_v2",
                        requested_size=_req_qty,
                        final_size=_sizing.final_size,
                        capital_efficiency_score=_eff_score,
                        effective_independent_n=_sizing.effective_independent_n,
                        portfolio_concentration=_sizing.portfolio_concentration,
                        concentration_penalty=_sizing.concentration_penalty,
                        truth_confidence=_truth_confidence_5a,
                        truth_regime=_truth_regime_5a,
                        allocation_cap=_sizing.allocation_cap,
                        throttle_applied=_sizing.throttle_applied,
                        throttle_reason=_sizing.throttle_reason,
                        fallback_mode=_sizing.fallback_mode,
                        fallback_reason=_sizing.fallback_reason,
                        stale_state_detected=_sizing.stale_state_detected,
                        allocation_state_age_sec=_sizing.allocation_state_age_sec,
                        execution_allowed=_sizing.execution_allowed,
                    )
                    append_intent_record(_intent, ALLOCATION_INTENT_FILE)

                    if not _sizing.execution_allowed:
                        logger.warning(
                            "[ALLOC] %s blocked by Phase5A: %s",
                            _sym5a, _sizing.fallback_reason or _sizing.throttle_reason,
                        )
                        # ── Skipped trade attribution ────────────────────────
                        try:
                            from src.allocation.skipped_trade_outcomes import (
                                SkippedTradeRecord as _SkipRec,
                                append_skipped_trade as _append_skip,
                                SKIP_TRUTH_CONFIDENCE as _SKIP_TC,
                                SKIP_CONCENTRATION_THROTTLE as _SKIP_CT,
                                SKIP_ALLOC_CAP as _SKIP_AC,
                            )
                            from src.allocation.sizing import TRUTH_CONFIDENCE_GATE as _TC_GATE
                            from src.paths import SKIPPED_TRADE_FILE as _stf
                            if _truth_confidence_5a < _TC_GATE:
                                _skip_reason = _SKIP_TC
                            elif _sizing.throttle_applied:
                                _skip_reason = _SKIP_CT
                            else:
                                _skip_reason = _SKIP_AC
                            _append_skip(
                                _SkipRec(
                                    timestamp=_ts5a,
                                    symbol=_sym5a,
                                    signal_id=getattr(_o5a, "signal_id", ""),
                                    strategy_id="fujiko_v2",
                                    skip_reason=_skip_reason,
                                    requested_size=_req_qty,
                                    alloc_cap=_sizing.allocation_cap,
                                    truth_confidence=_truth_confidence_5a,
                                    concentration_at_skip=_sizing.portfolio_concentration,
                                    estimated_entry_price=_est_price,
                                ),
                                _stf,
                            )
                        except Exception as _skip_err:
                            logger.warning("[SKIPPED_TRADE] record failed (%s)", _skip_err)

                        # ── Alpha leakage analytics: SkippedOpportunityRecord ─────
                        try:
                            from src.analytics.skipped_opportunity_analytics import (
                                SkippedOpportunityRecord as _SOR,
                                append_skipped_opportunity as _app_sor,
                            )
                            _sig_lk = _leakage_signal_lookup.get(_sym5a, {})
                            _sor = _SOR.create(
                                run_id=run_id,
                                timestamp=_ts5a,
                                symbol=_sym5a,
                                strategy_id="fujiko_v2",
                                signal_strength=float(_sig_lk.get("rsr", _eff_score * 100)),
                                predicted_rank=int(_sig_lk.get("rsr_rank", 0)),
                                rejection_reason=_skip_reason,
                                available_cash=_leakage_available_cash,
                                alloc_cap=_sizing.allocation_cap,
                                intended_position_size=_req_qty,
                                sector_state=str(_sig_lk.get("sector", "不明")),
                                concentration_state=float(_sizing.portfolio_concentration),
                                price_at_signal=_est_price,
                            )
                            _leakage_skipped_records.append(_sor)
                            _app_sor(_sor, SKIPPED_OPPORTUNITY_FILE)
                        except Exception as _sor_err:
                            logger.warning("[LEAKAGE] skip record failed (%s)", _sor_err)
                        continue
                    _phase5a_orders.append(_o5a)
                    # ── Allocation outcome record ─────────────────────────────
                    try:
                        from src.allocation.allocation_outcomes import (
                            build_outcome_record as _build_ao,
                            append_outcome_record as _append_ao,
                        )
                        from src.paths import ALLOCATION_OUTCOMES_FILE as _aof
                        _append_ao(
                            _build_ao(
                                timestamp=_ts5a,
                                symbol=_sym5a,
                                signal_id=getattr(_o5a, "signal_id", ""),
                                strategy_id="fujiko_v2",
                                intended_qty=_req_qty,
                                adjusted_qty=_sizing.final_size,
                                throttle_reason=_sizing.throttle_reason,
                                truth_confidence=_truth_confidence_5a,
                                alloc_cap=_sizing.allocation_cap,
                                effective_n=_sizing.effective_independent_n,
                                sector_multiplier=_eff_score,
                            ),
                            _aof,
                        )
                    except Exception as _ao_err:
                        logger.warning("[ALLOC_OUTCOME] record failed (%s)", _ao_err)

                if len(_phase5a_orders) < len(order_objects):
                    _blocked_5a = len(order_objects) - len(_phase5a_orders)
                    print(f"\n[Phase5A] {_blocked_5a}件 を Exposure/Truth フィルタで除外しました。")
                    if not _phase5a_orders:
                        print("[Phase5A] 発注可能な注文がありません。終了します。")
                        if not args.no_save:
                            save_live_logs(run_id, result, [])
                        return 0
                    order_objects = _phase5a_orders
            except Exception as _phase5a_err:
                logger.warning("[ALLOC] Phase5A sizing failed (%s) — proceeding without filter", _phase5a_err)

            _trading_day = datetime.now(JST).strftime("%Y-%m-%d")

            # ── V2 probation: block addon + continuation for probation symbols ─
            if _probation_active_symbols and _winner_confirmations:
                _pre_v2_addon = len(_winner_confirmations)
                _winner_confirmations = [
                    _wc for _wc in _winner_confirmations
                    if getattr(_wc, "symbol", "") not in _probation_active_symbols
                ]
                if len(_winner_confirmations) < _pre_v2_addon:
                    logger.info(
                        "[V2] addon blocked for probation: %d→%d confirmations",
                        _pre_v2_addon, len(_winner_confirmations),
                    )

            # ── Winner add-on order injection ─────────────────────────────────
            # Fail-open: errors leave order_objects unchanged.
            # Add-on orders skip Phase5A sizing (100-share unit is pre-validated).
            try:
                if _winner_confirmations:
                    from src.addon import AddOnExecutionPolicy
                    from src.kabusapi.signal_bridge import OrderInstruction as _OI
                    from src.paths import ADDON_STATE_FILE as _ADDON_ST, ADDON_DECISIONS_FILE as _ADDON_DEC
                    _ao_summary = result.portfolio_summary or {}
                    _ao_dd      = float(_ao_summary.get("current_drawdown", 0.0))
                    # Build held_positions for policy: estimate qty from capital/max_pos/price
                    _ao_held: dict = {}
                    for _aosig in result.signals:
                        if not _aosig.get("currently_holding"):
                            continue
                        _aosym  = _aosig.get("symbol", "")
                        _aoep   = float(_aosig.get("entry_price", 0.0))
                        _aopnl  = float(_aosig.get("unrealized_pnl_pct", 0.0))
                        _aocur  = _aoep * (1.0 + _aopnl) if _aoep > 0 else 0.0
                        _aoeq   = float(_eff_capital) if _eff_capital > 0 else float(CAPITAL)
                        # Qty estimate: capital_per_position / entry_price, rounded to unit
                        _aocap_per = _aoeq / max(1, MAX_POS)
                        _aoqty  = max(100, (int(_aocap_per / max(1.0, _aoep)) // 100) * 100) if _aoep > 0 else 100
                        _ao_held[_aosym] = {
                            "qty": _aoqty,
                            "current_price": _aocur,
                            "avg_daily_volume_yen": 0.0,  # fail-open liquidity gate
                        }
                    _ao_pol = AddOnExecutionPolicy(
                        state_path=_ADDON_ST,
                        decisions_path=_ADDON_DEC,
                    )
                    _ao_result = _ao_pol.run(
                        confirmations=_winner_confirmations,
                        portfolio_equity=float(_eff_capital) if _eff_capital > 0 else float(CAPITAL),
                        portfolio_pnl_pct=_ao_dd,
                        held_positions=_ao_held,
                        today=_trading_day,
                        run_id=run_id,
                    )
                    if _ao_result.addon_orders:
                        _ao_instrs = []
                        for _ao_ord in _ao_result.addon_orders:
                            _ao_s    = _ao_ord.symbol
                            _ao_s4   = _ao_s.split(".")[0] if "." in _ao_s else _ao_s[:4]
                            _ao_sect = next(
                                (_sg.get("sector", "不明") for _sg in result.signals
                                 if _sg.get("symbol") == _ao_s),
                                "不明",
                            )
                            _ao_instrs.append(_OI(
                                symbol=_ao_s,
                                symbol_4digit=_ao_s4,
                                sector=_ao_sect,
                                side="BUY",
                                qty=_ao_ord.qty,
                                order_type="MARKET_OPEN",
                                estimated_price=_ao_ord.estimated_price,
                                estimated_amount=_ao_ord.estimated_cost,
                                reason=_ao_ord.reason,
                                atr20=0.0,
                                strategy_type="fujiko",
                            ))
                        order_objects = list(order_objects) + _ao_instrs
                        logger.info(
                            "[ADDON] %d add-on order(s) injected: %s",
                            len(_ao_instrs), [_i.symbol for _i in _ao_instrs],
                        )
                        print(f"\n📈 アドオン発注 ({len(_ao_instrs)}件):")
                        for _ai in _ao_instrs:
                            print(
                                f"  {_ai.symbol} BUY +{_ai.qty}株"
                                f" @ ¥{_ai.estimated_price:,.0f}"
                                f"  {_ai.reason[:70]}"
                            )
                    if _ao_result.blocked:
                        logger.info("[ADDON] blocked: %s", _ao_result.blocked)
            except Exception as _ao_err:
                logger.warning("[ADDON] add-on hook failed (%s) — continuing without add-ons", _ao_err)

            # ── [RC] D_EQ_SCALE Addon: add when unrealized_gain >= 1×ATR20 ─────
            # Study45 production candidate. One addon per position lifecycle.
            # FAIL_OPEN: leaves order_objects unchanged on error. Default OFF.
            try:
                _rc_eq_cfg = getattr(getattr(cfg, "research_candidates", None), "eq_scale_addon", None)
                if _rc_eq_cfg and getattr(_rc_eq_cfg, "enabled", False):
                    from src.research_candidate.eq_scale_addon import generate_eq_scale_addon_orders
                    from src.kabusapi.signal_bridge import OrderInstruction as _OI_eq
                    from src.paths import EQ_SCALE_ADDON_STATE_FILE as _eq_state_f

                    # Load portfolio_state for ATR and highest_close data
                    _ps_eq: dict = {}
                    try:
                        import json as _json_eq
                        _ps_eq_path = RUNTIME_DIR / "portfolio_state.json"
                        if _ps_eq_path.exists():
                            _ps_eq = _json_eq.loads(_ps_eq_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                    # Estimate available cash: equity minus estimated held value
                    _eq_n_held   = sum(1 for _s in result.signals if _s.get("currently_holding"))
                    _eq_equity   = float(_eff_capital) if _eff_capital > 0 else float(CAPITAL)
                    _eq_held_val = _eq_n_held * (_eq_equity / max(1, int(_rc_max_pos)))
                    _eq_cash     = max(0.0, _eq_equity - _eq_held_val)

                    _eq_orders = generate_eq_scale_addon_orders(
                        held_signals      = list(result.signals),
                        portfolio_state   = _ps_eq,
                        available_cash    = _eq_cash,
                        portfolio_equity  = _eq_equity,
                        max_single_weight = float(MAX_SINGLE_WEIGHT),
                        state_path        = _eq_state_f,
                        today             = _trading_day,
                        run_id            = run_id,
                        atr_mult          = float(getattr(_rc_eq_cfg, "atr_mult",   1.0)),
                        size_frac         = float(getattr(_rc_eq_cfg, "size_frac", 0.25)),
                    )
                    if _eq_orders:
                        _eq_instrs = []
                        for _eq_ord in _eq_orders:
                            _eq_s   = _eq_ord.symbol
                            _eq_s4  = _eq_s.split(".")[0] if "." in _eq_s else _eq_s[:4]
                            _eq_sec = next(
                                (_sg.get("sector", "不明") for _sg in result.signals
                                 if _sg.get("symbol") == _eq_s),
                                "不明",
                            )
                            _eq_instrs.append(_OI_eq(
                                symbol=_eq_s,
                                symbol_4digit=_eq_s4,
                                sector=_eq_sec,
                                side="BUY",
                                qty=_eq_ord.qty,
                                order_type="MARKET_OPEN",
                                estimated_price=_eq_ord.estimated_price,
                                estimated_amount=_eq_ord.estimated_cost,
                                reason=_eq_ord.reason,
                                atr20=0.0,
                                strategy_type="fujiko",
                            ))
                        order_objects = list(order_objects) + _eq_instrs
                        logger.info(
                            "[RC_EQ_SCALE] %d add-on order(s) injected: %s",
                            len(_eq_instrs), [_i.symbol for _i in _eq_instrs],
                        )
                        print(f"\n📈 EQ_SCALE アドオン発注 ({len(_eq_instrs)}件):")
                        for _ei in _eq_instrs:
                            print(
                                f"  {_ei.symbol} BUY +{_ei.qty}株"
                                f" @ ¥{_ei.estimated_price:,.0f}"
                                f"  {_ei.reason[:70]}"
                            )
            except Exception as _rc_eq_err:
                logger.warning("[RC_EQ_SCALE] add-on hook failed (%s) — continuing", _rc_eq_err)

            # ── [RC] Quality Replacement Engine — Shadow判定のみ (Study57/58A ADOPT) ──
            # QUALITY_REPLACEMENT_ENABLED=false 固定。発注変更なし。FAIL_OPEN。
            # 目的: BT/ライブ等価性監査・スワップ条件の日次ログ記録。
            _signal_rsr_map: dict[str, float] = {
                _s["symbol"]: float(_s.get("rsr", 0.0))
                for _s in result.signals
                if _s.get("symbol")
            }
            try:
                # StrategyConfig dataclassにresearch_candidatesがないため直接YAMLをロード
                import yaml as _qr_yaml
                from types import SimpleNamespace as _SNS_qr
                _qr_raw = (_qr_yaml.safe_load(
                    STRATEGY_CONFIG_FILE.read_text(encoding="utf-8")
                ) or {}).get("research_candidates", {}).get("quality_replacement")
                _rc_qr_cfg = _SNS_qr(**_qr_raw) if isinstance(_qr_raw, dict) else None
                if _rc_qr_cfg is not None:
                    from src.research_candidate.quality_replacement import run_quality_replacement_shadow
                    from src.paths import (
                        QUAL_REPLACE_AUDIT_FILE    as _qr_audit_f,
                        QUAL_REPLACE_MISSED_FILE   as _qr_missed_f,
                        QUAL_REPLACE_OUTCOMES_FILE as _qr_outcomes_f,
                    )
                    import json as _json_qr
                    _ps_qr_path = RUNTIME_DIR / "portfolio_state.json"
                    _ps_qr: dict = {}
                    if _ps_qr_path.exists():
                        try:
                            _ps_qr = _json_qr.loads(_ps_qr_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    from src.paths import CACHE_DIR as _qr_cache_root
                    _ohlcv_cache = _qr_cache_root / "ohlcv"
                    _qr_result = run_quality_replacement_shadow(
                        today           = _trading_day,
                        run_id          = run_id,
                        mode            = "LIVE" if not DRY_RUN else "DRY",
                        signals         = list(result.signals),
                        portfolio_state = _ps_qr,
                        cfg             = _rc_qr_cfg,
                        audit_file      = _qr_audit_f,
                        missed_file     = _qr_missed_f,
                        outcomes_file   = _qr_outcomes_f,
                        ohlcv_cache_dir = _ohlcv_cache,
                    )
                    if _qr_result.get("swap_ready"):
                        logger.warning(
                            "[QR_SHADOW] SWAP_READY: weakest=%s(QS=%.1f) cand=%s(QS=%.1f)"
                            " — shadow only, no order sent",
                            _qr_result["weakest"]["symbol"], _qr_result["weakest"]["qs"],
                            _qr_result["best_cand"]["symbol"], _qr_result["best_cand"]["qs"],
                        )
            except Exception as _rc_qr_err:
                logger.warning("[RC_QUAL_REPLACE] shadow hook failed (%s) — continuing", _rc_qr_err)

            # ── Phase9 QR Shadow Audit (LIVE) ─────────────────────────────────
            # Observation-only. FAIL_OPEN. No order changes.
            try:
                from src.analytics.qr_shadow_audit import run_phase9_all as _p9_run
                from src.paths import (
                    QUAL_REPLACE_AUDIT_FILE  as _p9_audit,
                    QUAL_REPLACE_MISSED_FILE as _p9_missed,
                    QUAL_REPLACE_P9A_FILE    as _p9a_f,
                    QUAL_REPLACE_P9B_FILE    as _p9b_f,
                    QUAL_REPLACE_P9C_FILE    as _p9c_f,
                    QUAL_REPLACE_P9D_FILE    as _p9d_f,
                    QUAL_REPLACE_P9E_DIR     as _p9e_d,
                    CACHE_DIR                as _p9_cache_root,
                )
                _p9_ohlcv = _p9_cache_root / "ohlcv"
                _p9_result = _p9_run(
                    today        = _trading_day,
                    run_id       = run_id,
                    audit_file   = _p9_audit,
                    missed_file  = _p9_missed,
                    p9a_file     = _p9a_f,
                    p9b_file     = _p9b_f,
                    p9c_file     = _p9c_f,
                    p9d_file     = _p9d_f,
                    p9e_dir      = _p9e_d,
                    ohlcv_cache  = _p9_ohlcv,
                )
                logger.info("[PHASE9_QR] live audit complete: %s", _p9_result)
            except Exception as _p9_err:
                logger.warning("[PHASE9_QR] live hook failed (%s) — continuing", _p9_err)

            # ── Forward Continuation Outcome Intelligence (LIVE) ──────────────
            # Observation-only cross-telemetry signal analysis. FAIL_OPEN.
            try:
                from src.analytics.forward_continuation_outcome import (
                    run_fco_analysis   as _fco_lv_run,
                    format_fco_report  as _fco_lv_fmt,
                )
                from src.paths import (
                    CONTINUATION_PRIORITY_TELEMETRY_FILE as _fco_lv_cp,
                    BREAKOUT_QUALITY_TELEMETRY_FILE      as _fco_lv_bq,
                    FORWARD_CONTINUATION_OUTCOME_FILE    as _fco_lv_out,
                )
                _fco_lv_result = _fco_lv_run(
                    cp_file=_fco_lv_cp,
                    bq_file=_fco_lv_bq,
                    output_file=_fco_lv_out,
                    today=_trading_day,
                )
                _fco_lv_rpt = _fco_lv_fmt(_fco_lv_result)
                if _fco_lv_rpt:
                    print("\n" + _fco_lv_rpt)
                logger.info(
                    "[FCO] LIVE analysis: n=%d materialized=%d top=%s IR=%.4f",
                    _fco_lv_result.n_records_total,
                    _fco_lv_result.n_materialized,
                    _fco_lv_result.top_signal,
                    _fco_lv_result.top_signal_ir,
                )
            except Exception as _fco_lv_err:
                logger.warning("[FCO] LIVE forward_continuation_outcome hook failed: %s", _fco_lv_err)

            # ── Signal Attribution Intelligence (LIVE) ────────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.signal_attribution import (
                    run_signal_attribution    as _sa_lv_run,
                    format_attribution_report as _sa_lv_fmt,
                )
                from src.paths import (
                    CONTINUATION_PRIORITY_TELEMETRY_FILE as _sa_lv_cp,
                    SLOT_PRESSURE_FILE                   as _sa_lv_sp,
                    SIGNAL_ATTRIBUTION_FILE              as _sa_lv_out,
                )
                _sa_lv_result = _sa_lv_run(
                    cp_file=_sa_lv_cp,
                    sp_file=_sa_lv_sp,
                    output_file=_sa_lv_out,
                    today=_trading_day,
                )
                _sa_lv_rpt = _sa_lv_fmt(_sa_lv_result)
                if _sa_lv_rpt:
                    print("\n" + _sa_lv_rpt)
                logger.info(
                    "[SA] LIVE attribution: n=%d materialized=%d top=%s IR=%.4f",
                    _sa_lv_result.n_records_total,
                    _sa_lv_result.n_materialized,
                    _sa_lv_result.top_signal,
                    _sa_lv_result.top_signal_ir,
                )
            except Exception as _sa_lv_err:
                logger.warning("[SA] LIVE signal_attribution hook failed: %s", _sa_lv_err)

            # ── Capital Concentration Shadow Intelligence (LIVE) ──────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.capital_concentration_shadow import (
                    run_concentration_shadow    as _ccs_lv_run,
                    format_concentration_report as _ccs_lv_fmt,
                )
                from src.paths import (
                    CONTINUATION_PRIORITY_TELEMETRY_FILE  as _ccs_lv_cp,
                    CAPITAL_CONCENTRATION_SHADOW_FILE      as _ccs_lv_out,
                )
                _ccs_lv_result = _ccs_lv_run(
                    cp_file=_ccs_lv_cp,
                    output_file=_ccs_lv_out,
                    today=_trading_day,
                )
                _ccs_lv_rpt = _ccs_lv_fmt(_ccs_lv_result)
                if _ccs_lv_rpt:
                    print("\n" + _ccs_lv_rpt)
                logger.info(
                    "[CCS] LIVE n=%d materialized=%d dates=%d alpha=%.4f pos_rate=%.1f%%",
                    _ccs_lv_result.n_records_total,
                    _ccs_lv_result.n_materialized,
                    _ccs_lv_result.n_dates_analyzed,
                    _ccs_lv_result.mean_concentration_alpha,
                    _ccs_lv_result.positive_alpha_rate * 100,
                )
            except Exception as _ccs_lv_err:
                logger.warning("[CCS] LIVE capital_concentration_shadow hook failed: %s", _ccs_lv_err)

            # ── Priority Calibration Intelligence (LIVE) ──────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.priority_calibration import (
                    run_priority_calibration   as _pci_lv_run,
                    format_calibration_report  as _pci_lv_fmt,
                )
                from src.paths import (
                    CONTINUATION_PRIORITY_TELEMETRY_FILE as _pci_lv_cp,
                    PRIORITY_CALIBRATION_FILE            as _pci_lv_out,
                )
                _pci_lv_result = _pci_lv_run(
                    cp_file=_pci_lv_cp,
                    output_file=_pci_lv_out,
                    today=_trading_day,
                )
                _pci_lv_rpt = _pci_lv_fmt(_pci_lv_result)
                if _pci_lv_rpt:
                    print("\n" + _pci_lv_rpt)
                logger.info(
                    "[PCI] LIVE n=%d mat=%d buckets=%d mono=%.4f calib_err=%.4f",
                    _pci_lv_result.n_records_total,
                    _pci_lv_result.n_materialized,
                    _pci_lv_result.n_buckets_populated,
                    _pci_lv_result.priority_monotonicity,
                    _pci_lv_result.priority_calibration_error,
                )
            except Exception as _pci_lv_err:
                logger.warning("[PCI] LIVE priority_calibration hook failed: %s", _pci_lv_err)

            # ── Hold Duration Calibration Intelligence (LIVE) ─────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.hold_duration_calibration import (
                    run_hold_duration_calibration as _hdc_lv_run,
                    format_duration_report        as _hdc_lv_fmt,
                )
                from src.paths import (
                    CONTINUATION_PRIORITY_TELEMETRY_FILE as _hdc_lv_cp,
                    HOLD_DURATION_CALIBRATION_FILE       as _hdc_lv_out,
                )
                _hdc_lv_result = _hdc_lv_run(
                    cp_file=_hdc_lv_cp,
                    output_file=_hdc_lv_out,
                    today=_trading_day,
                )
                _hdc_lv_rpt = _hdc_lv_fmt(_hdc_lv_result)
                if _hdc_lv_rpt:
                    print("\n" + _hdc_lv_rpt)
                logger.info(
                    "[HDC] LIVE n=%d mat=%d buckets=%d best=%s spread=%.4f sup_delta=%.4f",
                    _hdc_lv_result.n_records_total,
                    _hdc_lv_result.n_materialized,
                    _hdc_lv_result.n_buckets_populated,
                    _hdc_lv_result.best_duration_bucket,
                    _hdc_lv_result.duration_ev_spread,
                    _hdc_lv_result.suppression_return_delta,
                )
            except Exception as _hdc_lv_err:
                logger.warning("[HDC] LIVE hold_duration_calibration hook failed: %s", _hdc_lv_err)

            # ── Evidence Promotion Engine (LIVE) ──────────────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.evidence_promotion import (
                    run_evidence_promotion   as _ep_lv_run,
                    format_promotion_report  as _ep_lv_fmt,
                )
                from src.paths import (
                    HOLD_DURATION_CALIBRATION_FILE       as _ep_lv_hdc,
                    CAPITAL_CONCENTRATION_SHADOW_FILE    as _ep_lv_ccs,
                    PRIORITY_CALIBRATION_FILE            as _ep_lv_pci,
                    SIGNAL_ATTRIBUTION_FILE              as _ep_lv_sa,
                    FORWARD_CONTINUATION_OUTCOME_FILE    as _ep_lv_fco,
                    EVIDENCE_PROMOTION_FILE              as _ep_lv_out,
                )
                _ep_lv_result = _ep_lv_run(
                    hdc_file=_ep_lv_hdc,
                    ccs_file=_ep_lv_ccs,
                    pci_file=_ep_lv_pci,
                    sa_file =_ep_lv_sa,
                    fco_file=_ep_lv_fco,
                    output_file=_ep_lv_out,
                    today=_trading_day,
                )
                _ep_lv_rpt = _ep_lv_fmt(_ep_lv_result)
                if _ep_lv_rpt:
                    print("\n" + _ep_lv_rpt)
                logger.info(
                    "[EP] LIVE evidence_promotion: promotable=%d observe=%d insufficient=%d top=%s",
                    _ep_lv_result.n_promotable,
                    _ep_lv_result.n_observe,
                    _ep_lv_result.n_insufficient,
                    _ep_lv_result.top_candidate or "none",
                )
            except Exception as _ep_lv_err:
                logger.warning("[EP] LIVE evidence_promotion hook failed: %s", _ep_lv_err)

            # ── Shadow Recommendation Engine (LIVE) ───────────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.shadow_recommendation import (
                    run_shadow_recommendation       as _sr_lv_run,
                    format_recommendation_report    as _sr_lv_fmt,
                )
                from src.paths import (
                    EVIDENCE_PROMOTION_FILE    as _sr_lv_ep,
                    SHADOW_RECOMMENDATION_FILE as _sr_lv_out,
                )
                _sr_lv_result = _sr_lv_run(
                    ep_file=_sr_lv_ep,
                    output_file=_sr_lv_out,
                    today=_trading_day,
                )
                _sr_lv_rpt = _sr_lv_fmt(_sr_lv_result)
                if _sr_lv_rpt:
                    print("\n" + _sr_lv_rpt)
                logger.info(
                    "[SR] LIVE shadow_recommendation: ready=%d observe=%d wait=%d top=%s",
                    _sr_lv_result.n_shadow_ready,
                    _sr_lv_result.n_continue_observe,
                    _sr_lv_result.n_wait_data,
                    _sr_lv_result.top_shadow_candidate or "none",
                )
            except Exception as _sr_lv_err:
                logger.warning("[SR] LIVE shadow_recommendation hook failed: %s", _sr_lv_err)

            # ── Shadow Outcome Tracker (LIVE) ─────────────────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.shadow_outcome_tracker import (
                    run_shadow_outcome_tracker  as _sot_lv_run,
                    format_outcome_report       as _sot_lv_fmt,
                )
                from src.paths import (
                    SHADOW_RECOMMENDATION_FILE as _sot_lv_sr,
                    EVIDENCE_PROMOTION_FILE    as _sot_lv_ep,
                    SHADOW_OUTCOME_FILE         as _sot_lv_out,
                )
                _sot_lv_result = _sot_lv_run(
                    sr_file=_sot_lv_sr,
                    ep_file=_sot_lv_ep,
                    output_file=_sot_lv_out,
                    today=_trading_day,
                )
                _sot_lv_rpt = _sot_lv_fmt(_sot_lv_result.summary)
                if _sot_lv_rpt:
                    print("\n" + _sot_lv_rpt)
                if _sot_lv_result.new_records:
                    logger.info(
                        "[SOT] LIVE shadow_outcome: new=%d completed=%d success_rate=%.1f%%",
                        len(_sot_lv_result.new_records),
                        _sot_lv_result.summary.n_completed,
                        _sot_lv_result.summary.success_rate * 100,
                    )
            except Exception as _sot_lv_err:
                logger.warning("[SOT] LIVE shadow_outcome hook failed: %s", _sot_lv_err)
                print("[SHADOW_OUTCOME] skipped")

            # ── Promotion Readiness Engine (LIVE) ─────────────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.promotion_readiness import (
                    run_promotion_readiness  as _pr_lv_run,
                    format_readiness_report  as _pr_lv_fmt,
                )
                from src.paths import (
                    EVIDENCE_PROMOTION_FILE    as _pr_lv_ep,
                    SHADOW_RECOMMENDATION_FILE as _pr_lv_sr,
                    SHADOW_OUTCOME_FILE         as _pr_lv_oc,
                    PROMOTION_READINESS_FILE    as _pr_lv_out,
                )
                _pr_lv_result = _pr_lv_run(
                    ep_file=_pr_lv_ep,
                    sr_file=_pr_lv_sr,
                    outcome_file=_pr_lv_oc,
                    output_file=_pr_lv_out,
                    today=_trading_day,
                )
                _pr_lv_rpt = _pr_lv_fmt(_pr_lv_result.snapshot)
                if _pr_lv_rpt:
                    print("\n" + _pr_lv_rpt)
                if _pr_lv_result.summary.promotion_ready_count > 0:
                    logger.info(
                        "[PR] LIVE promotion_readiness: ready=%d top=%s avg_score=%.1f",
                        _pr_lv_result.summary.promotion_ready_count,
                        _pr_lv_result.summary.top_ready_candidate or "none",
                        _pr_lv_result.summary.avg_readiness_score,
                    )
            except Exception as _pr_lv_err:
                logger.warning("[PR] LIVE promotion_readiness hook failed: %s", _pr_lv_err)
                print("[PROMOTION_READINESS] skipped")

            # ── Promotion Candidate Registry (LIVE) ───────────────────────────
            # Observation-only. FAIL_OPEN.
            try:
                from src.analytics.promotion_candidate_registry import (
                    run_promotion_candidate_registry  as _pcr_lv_run,
                    format_registry_report            as _pcr_lv_fmt,
                )
                from src.paths import (
                    PROMOTION_READINESS_FILE            as _pcr_lv_pr,
                    PROMOTION_CANDIDATE_REGISTRY_FILE   as _pcr_lv_out,
                )
                _pcr_lv_result = _pcr_lv_run(
                    pr_file=_pcr_lv_pr,
                    output_file=_pcr_lv_out,
                    today=_trading_day,
                )
                _pcr_lv_rpt = _pcr_lv_fmt(
                    _pcr_lv_result.active_registry,
                    _pcr_lv_result.summary,
                    _trading_day,
                )
                if _pcr_lv_rpt:
                    print("\n" + _pcr_lv_rpt)
                if _pcr_lv_result.new_records:
                    logger.info(
                        "[PCR] LIVE registry: new=%d active=%d top=%s",
                        len(_pcr_lv_result.new_records),
                        _pcr_lv_result.summary.active_candidates,
                        _pcr_lv_result.summary.top_candidate or "none",
                    )
            except Exception as _pcr_lv_err:
                logger.warning("[PCR] LIVE promotion_registry hook failed: %s", _pcr_lv_err)
                print("[PROMOTION_REGISTRY] skipped")

            # ── Promotion deployment planner (LIVE) ───────────────────────────
            try:
                from src.analytics.promotion_deployment_planner import (
                    run_promotion_deployment_planner as _pdp_lv_run,
                    format_deployment_report as _pdp_lv_fmt,
                )
                from src.paths import (
                    PROMOTION_CANDIDATE_REGISTRY_FILE as _pdp_lv_reg_file,
                    PROMOTION_DEPLOYMENT_PLAN_FILE    as _pdp_lv_out_file,
                )
                _pdp_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _pdp_lv_result = _pdp_lv_run(_pdp_lv_reg_file, _pdp_lv_out_file, _pdp_lv_today)
                _pdp_lv_rpt    = _pdp_lv_fmt(_pdp_lv_result.planned_records, _pdp_lv_result.summary)
                if _pdp_lv_rpt:
                    print("\n" + _pdp_lv_rpt)
                if _pdp_lv_result.new_records:
                    logger.info(
                        "[PDP] LIVE planner: new=%d planned=%d top=%s",
                        len(_pdp_lv_result.new_records),
                        _pdp_lv_result.summary.planned_count,
                        _pdp_lv_result.summary.highest_priority_candidate or "none",
                    )
            except Exception as _pdp_lv_err:
                logger.warning("[PDP] LIVE promotion_deployment_planner hook failed: %s", _pdp_lv_err)
                print("[PROMOTION_DEPLOYMENT] skipped")

            # ── Shadow deployment simulator (LIVE) ────────────────────────────
            try:
                from src.analytics.shadow_deployment_simulator import (
                    run_shadow_deployment_simulator as _sds_lv_run,
                    format_shadow_deployment_report as _sds_lv_fmt,
                )
                from src.paths import (
                    PROMOTION_DEPLOYMENT_PLAN_FILE as _sds_lv_plan_file,
                    SHADOW_DEPLOYMENT_FILE         as _sds_lv_out_file,
                )
                _sds_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _sds_lv_result = _sds_lv_run(_sds_lv_plan_file, _sds_lv_out_file, _sds_lv_today)
                _sds_lv_rpt    = _sds_lv_fmt(_sds_lv_result.deployed_records, _sds_lv_result.summary)
                if _sds_lv_rpt:
                    print("\n" + _sds_lv_rpt)
                if _sds_lv_result.new_records:
                    logger.info(
                        "[SDS] LIVE simulator: new=%d active=%d",
                        len(_sds_lv_result.new_records),
                        _sds_lv_result.summary.active_shadow_deployments,
                    )
            except Exception as _sds_lv_err:
                logger.warning("[SDS] LIVE shadow_deployment_simulator hook failed: %s", _sds_lv_err)
                print("[SHADOW_DEPLOYMENT] skipped")

            # ── Promotion impact tracker (LIVE) ───────────────────────────────
            try:
                from src.analytics.promotion_impact_tracker import (
                    run_promotion_impact_tracker as _pit_lv_run,
                    format_impact_report         as _pit_lv_fmt,
                )
                from src.paths import (
                    SHADOW_DEPLOYMENT_FILE            as _pit_lv_shadow_file,
                    HOLD_DURATION_CALIBRATION_FILE    as _pit_lv_hd_file,
                    CAPITAL_CONCENTRATION_SHADOW_FILE  as _pit_lv_cc_file,
                    PRIORITY_CALIBRATION_FILE         as _pit_lv_pc_file,
                    SIGNAL_ATTRIBUTION_FILE           as _pit_lv_sa_file,
                    FORWARD_CONTINUATION_OUTCOME_FILE as _pit_lv_fco_file,
                    PROMOTION_IMPACT_TRACKER_FILE     as _pit_lv_out_file,
                )
                _pit_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _pit_lv_result = _pit_lv_run(
                    shadow_file=_pit_lv_shadow_file,
                    hold_duration_file=_pit_lv_hd_file,
                    capital_conc_file=_pit_lv_cc_file,
                    priority_cal_file=_pit_lv_pc_file,
                    signal_attr_file=_pit_lv_sa_file,
                    fco_file=_pit_lv_fco_file,
                    output_file=_pit_lv_out_file,
                    today=_pit_lv_today,
                )
                _pit_lv_rpt = _pit_lv_fmt(_pit_lv_result.impact_records, _pit_lv_result.summary)
                if _pit_lv_rpt:
                    print("\n" + _pit_lv_rpt)
                if _pit_lv_result.new_records:
                    logger.info(
                        "[PIT] LIVE tracker: new=%d positive=%d success_rate=%.1f%%",
                        len(_pit_lv_result.new_records),
                        _pit_lv_result.summary.n_positive,
                        _pit_lv_result.summary.success_rate,
                    )
            except Exception as _pit_lv_err:
                logger.warning("[PIT] LIVE promotion_impact_tracker hook failed: %s", _pit_lv_err)
                print("[PROMOTION_IMPACT] skipped")

            # ── Promotion governance scoreboard (LIVE) ────────────────────────
            try:
                from src.analytics.promotion_governance_scoreboard import (
                    run_promotion_governance_scoreboard as _pgs_lv_run,
                    format_scoreboard_report            as _pgs_lv_fmt,
                )
                from src.paths import (
                    PROMOTION_READINESS_FILE             as _pgs_lv_readiness,
                    PROMOTION_CANDIDATE_REGISTRY_FILE    as _pgs_lv_registry,
                    PROMOTION_IMPACT_TRACKER_FILE        as _pgs_lv_impact,
                    SHADOW_OUTCOME_FILE                  as _pgs_lv_shadow,
                    EVIDENCE_PROMOTION_FILE              as _pgs_lv_evidence,
                    PROMOTION_GOVERNANCE_SCOREBOARD_FILE as _pgs_lv_out,
                )
                _pgs_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _pgs_lv_result = _pgs_lv_run(
                    readiness_file=_pgs_lv_readiness,
                    registry_file=_pgs_lv_registry,
                    impact_file=_pgs_lv_impact,
                    shadow_file=_pgs_lv_shadow,
                    evidence_file=_pgs_lv_evidence,
                    output_file=_pgs_lv_out,
                    today=_pgs_lv_today,
                )
                _pgs_lv_rpt = _pgs_lv_fmt(_pgs_lv_result.ranked_records, _pgs_lv_result.summary)
                if _pgs_lv_rpt:
                    print("\n" + _pgs_lv_rpt)
                if _pgs_lv_result.new_records:
                    logger.info(
                        "[PGS] LIVE scoreboard: new=%d top=%s score=%.1f pf=%d",
                        len(_pgs_lv_result.new_records),
                        _pgs_lv_result.summary.top_candidate or "none",
                        _pgs_lv_result.summary.top_governance_score,
                        _pgs_lv_result.summary.n_promote_first,
                    )
            except Exception as _pgs_lv_err:
                logger.warning("[PGS] LIVE promotion_governance_scoreboard hook failed: %s", _pgs_lv_err)
                print("[PROMOTION_SCOREBOARD] skipped")

            # ── Promotion rollout manager (LIVE) ─────────────────────────
            try:
                from src.analytics.promotion_rollout_manager import (
                    run_promotion_rollout_manager as _prm_lv_run,
                    format_rollout_report         as _prm_lv_fmt,
                )
                from src.paths import (
                    PROMOTION_GOVERNANCE_SCOREBOARD_FILE as _prm_lv_sb_file,
                    PROMOTION_CANDIDATE_REGISTRY_FILE    as _prm_lv_reg_file,
                    PROMOTION_IMPACT_TRACKER_FILE        as _prm_lv_impact_file,
                    PROMOTION_ROLLOUT_MANAGER_FILE       as _prm_lv_out_file,
                )
                _prm_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _prm_lv_result = _prm_lv_run(
                    scoreboard_file=_prm_lv_sb_file,
                    registry_file=_prm_lv_reg_file,
                    impact_file=_prm_lv_impact_file,
                    output_file=_prm_lv_out_file,
                    today=_prm_lv_today,
                )
                _prm_lv_rpt = _prm_lv_fmt(_prm_lv_result.state_records, _prm_lv_result.summary)
                if _prm_lv_rpt:
                    print("\n" + _prm_lv_rpt)
                if _prm_lv_result.new_records:
                    logger.info(
                        "[PRM] LIVE rollout: new=%d ps=%d mc=%d pr=%d rb=%d top=%s",
                        len(_prm_lv_result.new_records),
                        _prm_lv_result.summary.n_paper_shadow,
                        _prm_lv_result.summary.n_micro_capital,
                        _prm_lv_result.summary.n_production_ready,
                        _prm_lv_result.summary.n_rollback_required,
                        _prm_lv_result.summary.top_rollout_candidate or "none",
                    )
            except Exception as _prm_lv_err:
                logger.warning("[PRM] LIVE promotion_rollout_manager hook failed: %s", _prm_lv_err)
                print("[PROMOTION_ROLLOUT] skipped")

            # ── Production promotion gate (LIVE) ──────────────────────────
            try:
                from src.analytics.production_promotion_gate import (
                    run_production_promotion_gate as _ppg_lv_run,
                    format_gate_report            as _ppg_lv_fmt,
                )
                from src.paths import (
                    PROMOTION_ROLLOUT_MANAGER_FILE        as _ppg_lv_rollout_file,
                    PROMOTION_GOVERNANCE_SCOREBOARD_FILE  as _ppg_lv_sb_file,
                    PROMOTION_IMPACT_TRACKER_FILE         as _ppg_lv_impact_file,
                    PRODUCTION_PROMOTION_GATE_FILE        as _ppg_lv_out_file,
                )
                _ppg_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _ppg_lv_result = _ppg_lv_run(
                    rollout_file=_ppg_lv_rollout_file,
                    scoreboard_file=_ppg_lv_sb_file,
                    impact_file=_ppg_lv_impact_file,
                    output_file=_ppg_lv_out_file,
                    today=_ppg_lv_today,
                )
                _ppg_lv_rpt = _ppg_lv_fmt(_ppg_lv_result)
                if _ppg_lv_rpt:
                    print("\n" + _ppg_lv_rpt)
                if not _ppg_lv_result.skipped:
                    logger.info(
                        "[PPG] LIVE gate: approved=%d review=%d rejected=%d top=%s",
                        _ppg_lv_result.summary.n_approved,
                        _ppg_lv_result.summary.n_review_required,
                        _ppg_lv_result.summary.n_rejected,
                        _ppg_lv_result.summary.top_candidate or "none",
                    )
            except Exception as _ppg_lv_err:
                logger.warning("[PPG] LIVE production_promotion_gate hook failed: %s", _ppg_lv_err)
                print("[PRODUCTION_GATE] skipped")

            # ── Active experiment registry routing (LIVE) ─────────────────
            try:
                from src.experiments.active_experiment_registry import (
                    load_registry                         as _exp_lv_load,
                    get_active_experiments                as _exp_lv_active,
                    create_experiments_from_gate_approval as _exp_lv_create,
                    route_symbols_to_experiments          as _exp_lv_route,
                    append_assignment                     as _exp_lv_append_asgn,
                    format_experiment_report              as _exp_lv_fmt,
                )
                from src.paths import (
                    ACTIVE_EXPERIMENT_REGISTRY_FILE as _exp_lv_reg_file,
                    EXPERIMENT_ASSIGNMENT_FILE      as _exp_lv_asgn_file,
                )
                _exp_lv_today = datetime.now(JST).strftime("%Y-%m-%d")
                # Promotion integration: PPG APPROVED → auto-create experiments
                try:
                    _ppg_lv_gate_recs = [
                        {"decision": gr.decision, "candidate": gr.candidate}
                        for gr in _ppg_lv_result.gate_records
                    ]
                    _exp_lv_new = _exp_lv_create(_ppg_lv_gate_recs, _exp_lv_reg_file, _exp_lv_today)
                    for _en_lv in _exp_lv_new:
                        logger.info("[EXP] LIVE new experiment: %s feature=%s",
                                    _en_lv.experiment_id, _en_lv.feature_name)
                except Exception:
                    pass
                _exp_lv_all    = _exp_lv_load(_exp_lv_reg_file)
                _exp_lv_active = _exp_lv_active(_exp_lv_all)
                if _exp_lv_active:
                    try:
                        _exp_lv_symbols = [str(s.get("symbol", s)) if isinstance(s, dict) else str(s)
                                           for s in (signals or [])]
                    except Exception:
                        _exp_lv_symbols = []
                    _exp_lv_routing = _exp_lv_route(_exp_lv_symbols, _exp_lv_active, _exp_lv_today)
                    for _sym_lv, _asgns_lv in _exp_lv_routing.items():
                        for _asgn_lv in _asgns_lv:
                            _exp_lv_append_asgn(_asgn_lv, _exp_lv_asgn_file)
                            if _asgn_lv.group == "EXPERIMENT":
                                logger.info("[EXP_TEST] LIVE %s → %s", _sym_lv, _asgn_lv.experiment_id)
                            else:
                                logger.info("[EXP_CONTROL] LIVE %s → CONTROL", _sym_lv)
                    _exp_lv_rpt = _exp_lv_fmt(_exp_lv_active, _exp_lv_routing)
                    if _exp_lv_rpt:
                        print("\n" + _exp_lv_rpt)
                    logger.info("[EXP] LIVE active_experiments=%d symbols_routed=%d",
                                len(_exp_lv_active), len(_exp_lv_routing))
            except Exception as _exp_lv_err:
                logger.warning("[EXP] LIVE experiment routing hook failed, defaulting CONTROL: %s", _exp_lv_err)
                print("[EXP_ASSIGN] LIVE all symbols defaulting to CONTROL (registry unavailable)")

            # ── Automatic rollback governance (LIVE) ──────────────────────
            try:
                from src.governance.automatic_rollback import (
                    evaluate_all_active_experiments as _rb_lv_eval_all,
                    DECISION_ROLLED_BACK            as _rb_lv_rolled,
                    DECISION_ROLLBACK_CANDIDATE     as _rb_lv_cand,
                    DECISION_REVIEW_REQUIRED        as _rb_lv_review,
                )
                from src.paths import (
                    ACTIVE_EXPERIMENT_REGISTRY_FILE as _rb_lv_reg_file,
                    EXPERIMENT_PERFORMANCE_FILE     as _rb_lv_perf_file,
                    ROLLBACK_GOVERNANCE_FILE        as _rb_lv_out_file,
                )
                _rb_lv_today     = datetime.now(JST).strftime("%Y-%m-%d")
                _rb_lv_decisions = _rb_lv_eval_all(
                    registry_path=_rb_lv_reg_file,
                    performance_path=_rb_lv_perf_file,
                    rollback_path=_rb_lv_out_file,
                    today=_rb_lv_today,
                )
                for _rb_lv_d in _rb_lv_decisions:
                    if _rb_lv_d.decision == _rb_lv_rolled:
                        logger.info("[RB_APPLIED] LIVE %s → ROLLED_BACK", _rb_lv_d.experiment_id)
                        print(f"[RB_APPLIED] LIVE {_rb_lv_d.experiment_id} → ROLLED_BACK")
                    elif _rb_lv_d.decision == _rb_lv_cand:
                        logger.info("[RB_CANDIDATE] LIVE %s day=%d",
                                    _rb_lv_d.experiment_id, _rb_lv_d.consecutive_candidate_days)
                    elif _rb_lv_d.decision == _rb_lv_review:
                        logger.info("[RB_REVIEW] LIVE %s triggers=%s",
                                    _rb_lv_d.experiment_id, _rb_lv_d.trigger_reasons)
                if _rb_lv_decisions:
                    logger.info("[RB] LIVE evaluated=%d decisions", len(_rb_lv_decisions))
            except Exception as _rb_lv_err:
                logger.warning("[RB] LIVE automatic_rollback hook failed (fail-safe): %s", _rb_lv_err)
                print("[RB] LIVE skipped (fail-safe)")

            # ── Progressive capital escalation (live) ──────────────────────────
            try:
                from src.governance.progressive_capital_escalation import (
                    evaluate_all_experiment_escalations as _esc_lv_eval_all,
                    DECISION_ESCALATE_LEVEL_1 as _esc_lv_lv1,
                    DECISION_ESCALATE_LEVEL_2 as _esc_lv_lv2,
                    DECISION_ESCALATE_LEVEL_3 as _esc_lv_lv3,
                    DECISION_FULL_SCALE        as _esc_lv_full,
                    DECISION_DEESCALATE        as _esc_lv_down,
                )
                from src.paths import (
                    ACTIVE_EXPERIMENT_REGISTRY_FILE as _esc_lv_reg_file,
                    EXPERIMENT_PERFORMANCE_FILE     as _esc_lv_perf_file,
                    CAPITAL_ESCALATION_FILE         as _esc_lv_out_file,
                )
                _esc_lv_today     = datetime.now(JST).strftime("%Y-%m-%d")
                _esc_lv_decisions = _esc_lv_eval_all(
                    registry_path=_esc_lv_reg_file,
                    performance_path=_esc_lv_perf_file,
                    escalation_path=_esc_lv_out_file,
                    today=_esc_lv_today,
                )
                for _esc_lv_d in _esc_lv_decisions:
                    if _esc_lv_d.decision in (_esc_lv_lv1, _esc_lv_lv2,
                                              _esc_lv_lv3, _esc_lv_full):
                        logger.info("[ESC_UP] LIVE %s L%d→L%d alloc=%.1f%%",
                                    _esc_lv_d.experiment_id,
                                    _esc_lv_d.previous_level, _esc_lv_d.new_level,
                                    _esc_lv_d.allocation_pct * 100)
                        print(f"[ESC_UP] LIVE {_esc_lv_d.experiment_id} "
                              f"L{_esc_lv_d.previous_level}→L{_esc_lv_d.new_level} "
                              f"alloc={_esc_lv_d.allocation_pct:.1%}")
                    elif _esc_lv_d.decision == _esc_lv_down:
                        logger.info("[ESC_DOWN] LIVE %s L%d→L%d",
                                    _esc_lv_d.experiment_id,
                                    _esc_lv_d.previous_level, _esc_lv_d.new_level)
                        print(f"[ESC_DOWN] LIVE {_esc_lv_d.experiment_id} "
                              f"L{_esc_lv_d.previous_level}→L{_esc_lv_d.new_level}")
                if _esc_lv_decisions:
                    logger.info("[ESC] LIVE evaluated=%d decisions", len(_esc_lv_decisions))
            except Exception as _esc_lv_err:
                logger.warning("[ESC] LIVE escalation hook failed (fail-safe): %s", _esc_lv_err)
                print("[ESC] LIVE skipped (fail-safe)")

            # ── Improvement attribution ranking (live) ────────────────────────
            try:
                from src.analytics.improvement_attribution_ranking import (
                    run_improvement_attribution_ranking as _iar_lv_run,
                    format_ranking_report               as _iar_lv_fmt,
                )
                from src.paths import (
                    EXPERIMENT_PERFORMANCE_FILE       as _iar_lv_perf,
                    ACTIVE_EXPERIMENT_REGISTRY_FILE   as _iar_lv_reg,
                    ROLLBACK_GOVERNANCE_FILE          as _iar_lv_rb,
                    CAPITAL_ESCALATION_FILE           as _iar_lv_esc,
                    IMPROVEMENT_RANKING_FILE          as _iar_lv_out,
                )
                _iar_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _iar_lv_result = _iar_lv_run(
                    performance_path=_iar_lv_perf,
                    registry_path=_iar_lv_reg,
                    rollback_path=_iar_lv_rb,
                    escalation_path=_iar_lv_esc,
                    output_path=_iar_lv_out,
                    today=_iar_lv_today,
                )
                _iar_lv_rpt = _iar_lv_fmt(_iar_lv_result)
                if _iar_lv_rpt:
                    print("\n" + _iar_lv_rpt)
                if _iar_lv_result.ranked:
                    logger.info(
                        "[IAR] LIVE improvement_ranking: n=%d top=%s score=%.1f",
                        _iar_lv_result.total_experiments,
                        _iar_lv_result.top_experiment or "none",
                        _iar_lv_result.top_score,
                    )
            except Exception as _iar_lv_err:
                logger.warning("[IAR] LIVE improvement_attribution_ranking hook failed: %s", _iar_lv_err)
                print("[IMPROVEMENT_RANKING] LIVE skipped")

            # ── System health audit (live) ────────────────────────────────────
            try:
                from src.analytics.system_health_audit import (
                    run_system_health_audit as _sha_lv_run,
                    format_health_report    as _sha_lv_fmt,
                )
                from src.paths import (
                    EVIDENCE_PROMOTION_FILE          as _sha_lv_ev,
                    SHADOW_RECOMMENDATION_FILE       as _sha_lv_sr,
                    PROMOTION_READINESS_FILE         as _sha_lv_pr,
                    PRODUCTION_PROMOTION_GATE_FILE   as _sha_lv_gate,
                    EXPERIMENT_ASSIGNMENT_FILE       as _sha_lv_assign,
                    CAPITAL_ESCALATION_FILE          as _sha_lv_esc,
                    ROLLBACK_GOVERNANCE_FILE         as _sha_lv_rb,
                    IMPROVEMENT_RANKING_FILE         as _sha_lv_rank,
                    ACTIVE_EXPERIMENT_REGISTRY_FILE  as _sha_lv_reg,
                    EXPERIMENT_PERFORMANCE_FILE      as _sha_lv_perf,
                    SYSTEM_HEALTH_REPORT_FILE        as _sha_lv_out,
                )
                _sha_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _sha_lv_report = _sha_lv_run(
                    evidence_file=_sha_lv_ev,
                    shadow_file=_sha_lv_sr,
                    promotion_file=_sha_lv_pr,
                    gate_file=_sha_lv_gate,
                    assignment_file=_sha_lv_assign,
                    escalation_file=_sha_lv_esc,
                    rollback_file=_sha_lv_rb,
                    ranking_file=_sha_lv_rank,
                    registry_file=_sha_lv_reg,
                    performance_file=_sha_lv_perf,
                    output_file=_sha_lv_out,
                    today=_sha_lv_today,
                )
                _sha_lv_rpt = _sha_lv_fmt(_sha_lv_report)
                if _sha_lv_rpt:
                    print("\n" + _sha_lv_rpt)
                logger.info(
                    "[SHA] LIVE health_audit: status=%s checks=%d active=%d",
                    _sha_lv_report.overall_status,
                    len(_sha_lv_report.checks),
                    _sha_lv_report.summary.active_experiments,
                )
            except Exception as _sha_lv_err:
                logger.warning("[SHA] LIVE system_health_audit hook failed: %s", _sha_lv_err)
                print("[SYSTEM_HEALTH] LIVE skipped")

            # ── Weekly executive review (live) ────────────────────────────────
            try:
                from src.analytics.weekly_executive_review import (
                    run_weekly_executive_review as _wer_lv_run,
                    format_executive_review     as _wer_lv_fmt,
                )
                from src.paths import (
                    EVIDENCE_PROMOTION_FILE          as _wer_lv_ev,
                    SHADOW_RECOMMENDATION_FILE       as _wer_lv_sr,
                    PROMOTION_READINESS_FILE         as _wer_lv_pr,
                    PRODUCTION_PROMOTION_GATE_FILE   as _wer_lv_gate,
                    ACTIVE_EXPERIMENT_REGISTRY_FILE  as _wer_lv_reg,
                    CAPITAL_ESCALATION_FILE          as _wer_lv_esc,
                    ROLLBACK_GOVERNANCE_FILE         as _wer_lv_rb,
                    IMPROVEMENT_RANKING_FILE         as _wer_lv_rank,
                    SYSTEM_HEALTH_REPORT_FILE        as _wer_lv_health,
                    EXPERIMENT_PERFORMANCE_FILE      as _wer_lv_perf,
                    WEEKLY_EXECUTIVE_REVIEW_FILE     as _wer_lv_out,
                )
                _wer_lv_today  = datetime.now(JST).strftime("%Y-%m-%d")
                _wer_lv_result = _wer_lv_run(
                    evidence_file=_wer_lv_ev,
                    shadow_file=_wer_lv_sr,
                    promotion_file=_wer_lv_pr,
                    gate_file=_wer_lv_gate,
                    registry_file=_wer_lv_reg,
                    escalation_file=_wer_lv_esc,
                    rollback_file=_wer_lv_rb,
                    ranking_file=_wer_lv_rank,
                    health_report_file=_wer_lv_health,
                    performance_file=_wer_lv_perf,
                    output_file=_wer_lv_out,
                    today=_wer_lv_today,
                )
                _wer_lv_rpt = _wer_lv_fmt(_wer_lv_result)
                if _wer_lv_rpt:
                    print("\n" + _wer_lv_rpt)
                logger.info(
                    "[WER] LIVE weekly_review: health=%s active=%d alerts=%d",
                    _wer_lv_result.system_health.overall_status,
                    _wer_lv_result.experiment_summary.total_active,
                    _wer_lv_result.key_alerts.critical_count
                    + _wer_lv_result.key_alerts.warning_count,
                )
            except Exception as _wer_lv_err:
                logger.warning("[WER] LIVE weekly_executive_review hook failed: %s", _wer_lv_err)
                print("[WEEKLY_REVIEW] LIVE skipped")

            # ── 層0: MAX_POSITION_GUARD (fail-closed) ──────────────────────────
            # 2026-07-07 4銘柄同時保有インシデント対応: signal生成側の全ガード
            # （max_positions判定 / Shadow経路 / 4th-slot gate 等）をすり抜けた
            # 場合の最終防波堤。送信直前に Broker 実保有数を再取得し、
            # 実保有 + 送信予定BUY件数 が max_positions を超えるなら RUN 全体を
            # 停止する。broker reality is authoritative（Autonomous Runtime Rules）。
            try:
                from src.kabusapi.signal_bridge import _capacity_check as _guard_capacity_check
                _guard_positions    = bridge._get_current_positions()
                _guard_pending_buys = sum(1 for _o in order_objects if _o.side == "BUY")
                _guard_remaining    = _guard_capacity_check(
                    bridge.max_positions, len(_guard_positions), _guard_pending_buys,
                )
                if _guard_remaining < 0:
                    _guard_projected = len(_guard_positions) + _guard_pending_buys
                    logger.error(
                        "[MAX_POSITION_GUARD] projected=%d (held=%d + pending_buy=%d) > "
                        "max_positions=%d → 発注中止（fail-closed）",
                        _guard_projected, len(_guard_positions), _guard_pending_buys,
                        bridge.max_positions,
                    )
                    print(
                        f"\n[FATAL][MAX_POSITION_GUARD] 実保有{len(_guard_positions)}件 + "
                        f"送信予定BUY{_guard_pending_buys}件 = {_guard_projected}件 > "
                        f"max_positions={bridge.max_positions} → 発注を全停止します。",
                        file=sys.stderr,
                    )
                    return 1
            except Exception as _guard_err:
                logger.error(
                    "[MAX_POSITION_GUARD] Broker実保有数取得失敗のため安全側で発注中止: %s",
                    _guard_err,
                )
                print(f"\n[FATAL][MAX_POSITION_GUARD] {_guard_err}", file=sys.stderr)
                return 1

            # ── Stage: order_execution (process-isolated via BrokerProcessSupervisor) ──
            _intent_journal = IntentJournal()
            _emit_phase("execution", "start", run_id=run_id)
            print("\n発注中（プロセス分離モード）...")
            _proc_sup = BrokerProcessSupervisor()
            try:
                enforce_order_rate_limit()
                send_results = _supervisor.run_stage(
                    "order_execution",
                    lambda: _submit_orders_process_isolated(
                        order_objects,
                        registry     = _inflight_registry,
                        trading_day  = _trading_day,
                        run_id       = run_id,
                        proc_supervisor = _proc_sup,
                        timeout_sec  = 25,  # inner process timeout; outer supervisor=30s
                    ),
                    timeout_sec=30,
                    retry_budget=0,
                )
                # Update ORDER_LOCK_FILE + audit ledger for each result
                from src.live.order_ledger import OrderLedger as _OL
                _ledger = _OL()
                for _r in send_results:
                    if _r.get("success"):
                        mark_ordered(_r["symbol"], _r["side"])
                        _ledger.mark_submitted(_r["symbol"], _r["side"], str(_r.get("order_id", "")))
                    else:
                        _ledger.mark_failed(_r["symbol"], _r["side"], str(_r.get("error", "")))
                record_order_sent()
            except StageTimeout as _ste:
                logger.error("order_execution タイムアウト: %s", _ste)
                print(f"\n[FATAL] 発注タイムアウト: {_ste}", file=sys.stderr)
                # Mark unresolved inflight orders as failed so recovery is clean
                if _inflight_registry is not None:
                    for _u in _inflight_registry.get_unresolved():
                        safe_cleanup_step(
                            f"inflight_fail_{_u.client_order_id[:8]}",
                            _inflight_registry.mark_failed,
                            _u.client_order_id, f"stage_timeout: {_ste}",
                        )
                for o in order_objects:
                    _iid = _exec_journal.make_intent_id(run_id, o.symbol, o.side)
                    _exec_journal.fail(_iid, "pending", error=str(_ste))
                if not args.no_save:
                    save_live_logs(run_id, result, [{"error": str(_ste)}])
                _exit_code = 1
                return _exit_code
            except StageError as _se:
                logger.error("order_execution 失敗: %s", _se)
                print(f"\n[FATAL] 発注失敗: {_se}", file=sys.stderr)
                if not args.no_save:
                    save_live_logs(run_id, result, [{"error": str(_se)}])
                _exit_code = 1
                return _exit_code

            # ExecutionJournal: update to submitted / ack / failed
            for r in send_results:
                _iid = _exec_journal.make_intent_id(
                    run_id, r.get("symbol", ""), r.get("side", "")
                )
                if r.get("success"):
                    _exec_journal.submit(_iid)
                    _exec_journal.ack(_iid, order_id=str(r.get("order_id", "")))
                else:
                    _exec_journal.fail(_iid, "pending", error=str(r.get("error", "")))

            print("\n=== 発注結果 ===")
            for r in send_results:
                status   = "✅ 成功" if r.get("success") else "❌ 失敗"
                order_id = r.get("order_id", r.get("error", ""))
                print(f"  {r['side']} {r['symbol']} {r['qty']}株 → {status}  ({order_id})")

            # ── Lineage: broker_acknowledgment step (Task 1) ──────────────────
            try:
                if _run_lineage is not None:
                    from src.live.deployment_lineage import (
                        append_lineage_step as _app_step2,
                        persist_lineage_entry as _persist_entry,
                        STEP_BROKER_ACKNOWLEDGMENT as _ST_BA,
                    )
                    _n_ack = sum(1 for r in send_results if r.get("success"))
                    _ack_entry = _app_step2(_run_lineage, _ST_BA, {
                        "orders_submitted": len(send_results),
                        "orders_acknowledged": _n_ack,
                        "run_id": run_id,
                    })
                    _persist_entry(_ack_entry, DEPLOYMENT_LINEAGE_FILE)
                    logger.info(
                        "[LINEAGE] broker_acknowledgment: submitted=%d ack=%d run_id=%s",
                        len(send_results), _n_ack, run_id,
                    )
            except Exception as _ack_lin_err:
                logger.warning("[LINEAGE] ack step failed (%s)", _ack_lin_err)

            # ── Alpha leakage: daily summary (Task 2) ─────────────────────────
            try:
                from src.analytics.daily_leakage_summary import (
                    generate_daily_summary as _gen_ls,
                    log_daily_summary as _log_ls,
                    append_daily_summary as _app_ls,
                )
                from src.analytics.deployable_alpha_metrics import (
                    compute_alpha_metrics as _comp_am,
                    append_alpha_metrics as _app_am,
                )
                _ls_date = datetime.now(JST).strftime("%Y-%m-%d")
                _ls_executed = sum(1 for r in send_results if r.get("success"))
                _ls_summary = _gen_ls(
                    date=_ls_date,
                    run_id=run_id,
                    skipped_records=_leakage_skipped_records,
                    executed_count=_ls_executed,
                )
                _log_ls(_ls_summary)
                _app_ls(_ls_summary, DAILY_LEAKAGE_FILE)
                _ls_metrics = _comp_am(
                    skipped_records=_leakage_skipped_records,
                    executed_count=_ls_executed,
                )
                _app_am(_ls_metrics, ALPHA_METRICS_FILE)
            except Exception as _ls_err:
                logger.warning("[LEAKAGE] daily summary failed (%s)", _ls_err)

            # ── Research priority summary (live) ───────────────────────────────
            try:
                from src.analytics.research_priority_summary import run_research_priority_summary as _run_rps_lv
                from src.analytics.daily_leakage_summary import load_daily_summaries as _load_ls_hist_lv
                _rps_lv_date = datetime.now(JST).strftime("%Y-%m-%d")
                _rps_lv_hist = _load_ls_hist_lv(DAILY_LEAKAGE_FILE)
                # Exclude current-day entry from historical (it was just appended)
                _rps_lv_hist = [s for s in _rps_lv_hist if s.date != _rps_lv_date]
                _rps_lv_sum, _rps_lv_path = _run_rps_lv(
                    date=_rps_lv_date,
                    run_id=run_id,
                    is_live=True,
                    summary_jsonl_path=RESEARCH_PRIORITY_FILE,
                    report_dir=RESEARCH_PRIORITY_REPORT_DIR,
                    skipped_records=_leakage_skipped_records,
                    leakage_summary=_ls_summary if "_ls_summary" in dir() else None,
                    historical_leakage=_rps_lv_hist,
                    integrity_result=_integrity_result,
                )
                if _rps_lv_path:
                    logger.info("[RESEARCH_PRI] report: %s", _rps_lv_path)
                    print(f"\n📊 研究優先レポート: {_rps_lv_path}")
                if _rps_lv_sum:
                    print(f"   ボトルネック: {_rps_lv_sum.dominant_bottleneck}"
                          f" | 優先領域: {_rps_lv_sum.primary_research_category}"
                          f" | 信頼度: {_rps_lv_sum.confidence_level}")
            except Exception as _rps_lv_err:
                logger.warning("[RESEARCH_PRI] live summary failed (%s)", _rps_lv_err)

            _emit_phase("execution", "complete", run_id=run_id)

            # ── Stage: reconciliation (portfolio state update) ───────────────
            _emit_phase("reconciliation", "start", run_id=run_id)
            _today_str = datetime.now(JST).strftime("%Y-%m-%d")
            try:
                _supervisor.run_stage(
                    "reconciliation",
                    lambda: bridge.update_state_after_execution(
                        send_results, _today_str,
                        signal_rsr_map=_signal_rsr_map if "_signal_rsr_map" in dir() else None,
                    ),
                    timeout_sec=15,
                    retry_budget=0,
                )
            except (StageTimeout, StageError) as _rec_e:
                logger.warning(
                    "reconciliation stage failed (portfolio state may be stale): %s",
                    _rec_e,
                )
            _emit_phase("reconciliation", "complete", run_id=run_id)

            # ── EVS: Executed vs Skipped Expectancy — LIVE record ─────────────
            # Runs AFTER real send_results are known, so "executed" reflects an
            # actual broker success=True fill, not a pre-send candidate guess.
            try:
                from src.analytics.executed_vs_skipped_expectancy import (
                    build_opportunity_records as _evs_build_lv,
                    append_opportunity        as _evs_append_lv,
                    DEFAULT_STORE_PATH        as _evs_path_lv,
                )
                _evs_recs_lv = _evs_build_lv(
                    signals=result.signals,
                    stage_audit=getattr(bridge, "_last_stage_audit", []),
                    send_results=send_results,
                    run_id=run_id,
                    run_timestamp=_evs_run_ts,
                    mode="LIVE",
                    source_script="run_live_signal.py",
                    capital_available_pct=_evs_cap_avail_pct,
                    portfolio_heat=_evs_heat,
                    market_regime=_evs_regime,
                    max_positions=int(MAX_POS),
                )
                for _evs_rec_lv in _evs_recs_lv:
                    _evs_append_lv(_evs_rec_lv, _evs_path_lv)
                if _evs_recs_lv:
                    logger.info(
                        "[EVS] LIVE recorded %d opportunity records (executed=%d skipped=%d)",
                        len(_evs_recs_lv),
                        sum(1 for r in _evs_recs_lv if r.executed),
                        sum(1 for r in _evs_recs_lv if not r.executed),
                    )
            except Exception as _evs_err_lv:
                logger.warning("[EVS] LIVE build_opportunity_records failed (%s) — continuing", _evs_err_lv)

            # ── Opportunity Capture integrity fix (2026-07-08) — LIVE record ──
            try:
                from src.analytics.skipped_opportunity_analytics import (
                    build_skipped_opportunity_records as _sor_build_lv,
                    append_skipped_opportunity         as _sor_append_lv,
                )
                _sor_recs_lv = _sor_build_lv(
                    signals=result.signals,
                    stage_audit=getattr(bridge, "_last_stage_audit", []),
                    send_results=send_results,
                    run_id=run_id,
                    run_timestamp=_evs_run_ts,
                    mode="LIVE",
                    source_script="run_live_signal.py",
                    available_cash=_evs_cash,
                )
                for _sor_rec_lv in _sor_recs_lv:
                    _sor_append_lv(_sor_rec_lv, SKIPPED_OPPORTUNITY_FILE)
                if _sor_recs_lv:
                    logger.info("[SKIPPED_OPP] LIVE recorded %d record(s)", len(_sor_recs_lv))
            except Exception as _sor_err_lv:
                logger.warning("[SKIPPED_OPP] LIVE build failed (%s) — continuing", _sor_err_lv)

            # ── Stage: capital state update (adaptive growth) ────────────────
            if _cap_state_loaded:
                try:
                    from src.capital import (
                        update_capital_state, compute_quality_score,
                        update_deployment_ramp, evaluate_freeze, update_freeze_state,
                        update_aggression, update_edge_model, build_edge_observation,
                        transition_deployment_state, update_multihorizon,
                        compute_concentration_metrics, get_participation_rate,
                        update_reflexivity, estimate_market_impact_bps,
                        estimate_net_alpha, estimate_gross_alpha_from_signal,
                        compute_post_cost_alpha_retention,
                        build_opportunity_cost_record, update_opportunity_cost_state,
                        append_opportunity_cost_record,
                        AggressionInputs, ExecutionQualityRecord, ReflexivityObservation,
                        compute_drawdown_penalty, compute_regime_score,
                    )

                    _today_str2 = datetime.now(JST).strftime("%Y-%m-%d")
                    ps2 = result.portfolio_summary
                    _actual_eq = float(ps2.get("current_equity", _cap_state.actual_equity))
                    _drawdown  = abs(float(ps2.get("current_drawdown", 0.0)))

                    # -- Execution quality from send_results --
                    _n_orders = len(send_results) or 1
                    _n_success = sum(1 for r in send_results if r.get("success"))
                    _fill_ratio = _n_success / _n_orders
                    _slippage_bps_list = [
                        abs(float(r.get("slippage_pct", 0.0)) * 10000)
                        for r in send_results
                    ]
                    _avg_slippage_bps = sum(_slippage_bps_list) / len(_slippage_bps_list) if _slippage_bps_list else 0.0
                    _reject_rate = 1.0 - _fill_ratio
                    _eq_record = ExecutionQualityRecord(
                        fill_ratio=_fill_ratio,
                        slippage_bps=_avg_slippage_bps,
                        reject_rate=_reject_rate,
                        execution_latency_ms=200.0,
                        partial_fill_ratio=0.0,
                        liquidity_stress_score=0.0,
                    )

                    # -- Update deployment ramp --
                    _new_ramp, _growth_mod = update_deployment_ramp(
                        _ramp_state, _eq_record
                    )

                    # -- Capital freeze check --
                    _is_frozen, _freeze_reason = evaluate_freeze(
                        slippage_bps=_avg_slippage_bps,
                        reject_rate=_reject_rate,
                        fill_ratio=_fill_ratio,
                        liquidity_stress_score=0.0,
                        partial_fill_ratio=0.0,
                    )
                    _new_freeze = update_freeze_state(
                        _freeze_state,
                        slippage_bps=_avg_slippage_bps,
                        reject_rate=_reject_rate,
                        fill_ratio=_fill_ratio,
                        liquidity_stress_score=0.0,
                        partial_fill_ratio=0.0,
                    )

                    # -- Update capital state --
                    _new_cap = update_capital_state(
                        _cap_state, _actual_eq,
                        growth_frozen=_is_frozen or (_new_ramp.mode == "FROZEN"),
                        daily_growth_limit=cfg.capital_scaling.effective_capital_growth_limit_daily,
                    )

                    # -- Signal quality for edge model --
                    _rsr_vals = [s.get("rsr", 0.0) for s in result.signals]
                    _n_sig = len(_rsr_vals) or 1
                    _breadth_75 = sum(1 for r in _rsr_vals if r >= 75) / _n_sig
                    _top_rsr_avg = max(_rsr_vals) if _rsr_vals else 0.0
                    _mom_vals = [s.get("rsr_momentum", 0.0) for s in result.signals]
                    _mom_pos_frac = sum(1 for m in _mom_vals if m > 0) / _n_sig
                    _regime_fav = ps2.get("regime", "bull") != "bear"

                    # -- Update edge model --
                    _edge_obs = build_edge_observation(
                        trading_day=_today_str2,
                        rsr_breadth_75=_breadth_75,
                        top_signal_rsr_avg=_top_rsr_avg,
                        rsr_momentum_positive_fraction=_mom_pos_frac,
                        regime_is_favorable=_regime_fav,
                    )
                    _new_edge = update_edge_model(_edge_model_state, _edge_obs)

                    # -- Update multi-horizon confidence --
                    _exec_quality = compute_quality_score(_eq_record)
                    _signal_quality = _breadth_75
                    _regime_conf = 0.7 if _regime_fav else 0.5
                    _new_mh = update_multihorizon(
                        _multihorizon_conf,
                        execution_observation=_exec_quality,
                        signal_observation=_signal_quality,
                        regime_observation=_regime_conf,
                        strategic_observation=0.65,
                    )

                    # -- Aggression update --
                    _dd_penalty = compute_drawdown_penalty(_drawdown)
                    _regime_score = compute_regime_score(not _regime_fav, regime_confidence=_regime_conf)
                    _agg_inputs = AggressionInputs(
                        edge_score=_new_edge.composite_edge_score,
                        execution_score=_exec_quality,
                        liquidity_score=1.0 - min(1.0, _avg_slippage_bps / 50.0),
                        regime_score=_regime_score,
                        concentration_penalty=1.0,
                        drawdown_penalty=_dd_penalty,
                    )
                    _new_aggression, _agg_growth_limit = update_aggression(
                        _aggression_state, _agg_inputs,
                        base_growth_limit=cfg.capital_scaling.effective_capital_growth_limit_daily,
                    )

                    # -- Deployment state transition --
                    _new_deploy_rec = transition_deployment_state(
                        _deploy_state_rec,
                        aggression_ema=_new_aggression.ema_score,
                        edge_score=_new_edge.composite_edge_score,
                        execution_mode=_new_ramp.mode,
                        drawdown=_drawdown,
                    )

                    # -- Reflexivity: check fill_ratio vs participation --
                    _part_rate = get_participation_rate(_reflexivity_state)
                    _refl_obs = ReflexivityObservation(
                        trading_day=_today_str2,
                        participation_rate=_part_rate,
                        fill_ratio=_fill_ratio,
                        order_value=sum(
                            float(o.get("estimated_amount", 0))
                            for o in result.orders
                        ),
                        adv_yen=1_000_000_000.0,
                    )
                    _new_refl = update_reflexivity(_reflexivity_state, _refl_obs)

                    # -- Opportunity cost --
                    _deployed_val = sum(
                        float(o.get("estimated_amount", 0))
                        for o in result.orders if o.get("side") == "BUY"
                    )
                    _gross_alpha = estimate_gross_alpha_from_signal(
                        rsr=_top_rsr_avg,
                        rsr_momentum=float(_mom_vals[0]) if _mom_vals else 0.0,
                        regime_is_favorable=_regime_fav,
                    )
                    _market_impact_bps = estimate_market_impact_bps(
                        _deployed_val,
                        max(1.0, _deployed_val / (_part_rate or 0.05)),
                    )
                    _alpha_est = estimate_net_alpha(_gross_alpha, _market_impact_bps)
                    _alpha_retention = compute_post_cost_alpha_retention(
                        _gross_alpha, _alpha_est.expected_net_alpha_bps
                    )
                    _opp_cumulative = float(
                        _opp_cost_state.cumulative_cost_bps
                        if _opp_cost_state is not None else 0.0
                    )
                    _opp_rec = build_opportunity_cost_record(
                        trading_day=_today_str2,
                        deployment_state=_new_deploy_rec.state,
                        deployable_capital=_new_cap.deployable_capital,
                        effective_capital=_new_cap.effective_capital,
                        actual_deployed=_deployed_val,
                        expected_net_alpha_bps=_alpha_est.expected_net_alpha_bps,
                        cumulative_cost_bps=_opp_cumulative,
                    )
                    if _opp_cost_state is not None:
                        _opp_cost_state = update_opportunity_cost_state(_opp_cost_state, _opp_rec)
                    try:
                        append_opportunity_cost_record(_opp_rec, OPPORTUNITY_COST_LOG_FILE)
                    except Exception:
                        pass

                    # -- Concentration metrics --
                    _pos_weights = {}
                    _sector_map = {}
                    for _sig in result.signals:
                        if _sig.get("currently_holding"):
                            _sym = _sig.get("symbol", "")
                            _pos_weights[_sym] = float(_sig.get("rsr", 50.0)) / 100.0
                            _sector_map[_sym] = _sig.get("sector", "不明")
                    try:
                        from src.capital import compute_concentration_metrics, save_concentration_metrics
                        _conc = compute_concentration_metrics(
                            _pos_weights, _sector_map,
                            target_n=cfg.portfolio.max_positions,
                        )
                        save_concentration_metrics(_conc, CONCENTRATION_METRICS_FILE)
                        _eff_n = _conc.effective_n
                    except Exception:
                        _eff_n = 0.0

                    # -- Save all states (atomic writes) --
                    save_capital_state(_new_cap, CAPITAL_STATE_FILE)
                    save_deployment_ramp(_new_ramp, DEPLOYMENT_RAMP_STATE_FILE)
                    save_freeze_state(_new_freeze, CAPITAL_FREEZE_STATE_FILE)
                    save_aggression_state(_new_aggression, AGGRESSION_STATE_FILE)
                    save_edge_model(_new_edge, EDGE_MODEL_STATE_FILE)
                    save_deployment_state_record(_new_deploy_rec, DEPLOYMENT_STATE_RECORD_FILE)
                    save_multihorizon(_new_mh, MULTIHORIZON_STATE_FILE)
                    save_reflexivity_state(_new_refl, REFLEXIVITY_STATE_FILE)
                    save_exploration_state(_exploration_state, EXPLORATION_STATE_FILE)

                    # -- Phase 5A: update + persist allocation states --
                    if _alloc_state_loaded and _exposure_state is not None:
                        try:
                            from src.allocation import (
                                save_exposure_state,
                                compute_stability_observation,
                                update_stability_window,
                                save_stability_window,
                                RankSnapshot,
                            )
                            from src.paths import (
                                EXPOSURE_STATE_FILE as _ESF5A,
                                STABILITY_WINDOW_FILE as _SWF5A,
                                ALLOCATOR_STABILITY_FILE as _ASF5A,
                                ALLOCATION_DECISIONS_FILE as _ADF5A,
                            )
                            from src.allocation import append_stability_observation
                            save_exposure_state(_exposure_state, _ESF5A)
                            # Build rank snapshot from signals
                            _rank_snap = RankSnapshot(
                                trading_day=_today_str2,
                                ranked_symbols=[
                                    _s5b.get("symbol", "")
                                    for _s5b in result.signals
                                    if _s5b.get("symbol")
                                ],
                                efficiency_scores=_efficiency_scores,
                            )
                            if _stability_window_state is not None:
                                _stab_obs = compute_stability_observation(
                                    _stability_window_state, _rank_snap
                                )
                                _stability_window_state = update_stability_window(
                                    _stability_window_state, _rank_snap
                                )
                                save_stability_window(_stability_window_state, _SWF5A)
                                append_stability_observation(_stab_obs, _ASF5A)
                            logger.info(
                                "[ALLOC] states saved: exposure=%d symbols",
                                len(_exposure_state.return_buffers),
                            )
                        except Exception as _alloc_save_err:
                            logger.warning(
                                "[ALLOC] state save failed (%s) — states not persisted",
                                _alloc_save_err,
                            )

                    # -- Append CAGR telemetry --
                    _telem = build_telemetry_entry(
                        actual_equity=_actual_eq,
                        effective_capital=_new_cap.effective_capital,
                        deployable_capital=_new_cap.deployable_capital,
                        risk_adjusted_capital=_new_cap.risk_adjusted_capital,
                        capital_growth_rate=_new_cap.capital_growth_rate,
                        deployed_value=_deployed_val,
                        participation_rate=_part_rate,
                        fill_ratio=_fill_ratio,
                        partial_fill_ratio=0.0,
                        slippage_bps=_avg_slippage_bps,
                        reject_rate=_reject_rate,
                        execution_latency_ms=200.0,
                        liquidity_stress_score=0.0,
                        shadow_capacity_metrics={},
                        deployment_ramp_state=_new_ramp.mode,
                        capital_freeze_state=_new_freeze.state,
                        capital_freeze_reason=_new_freeze.freeze_reason,
                        trading_day=_today_str2,
                        opportunity_cost_bps=_opp_rec.opportunity_cost_bps,
                        idle_capital_ratio=_opp_rec.idle_fraction,
                        edge_persistence_score=_new_edge.composite_edge_score,
                        aggressiveness_score=_new_aggression.ema_score,
                        effective_independent_positions=_eff_n,
                        post_cost_alpha_retention=_alpha_retention,
                        deployment_regime_state=_new_deploy_rec.state,
                        multihorizon_aggregate_confidence=_new_mh.aggregate_confidence,
                    )
                    append_telemetry(_telem, CAPITAL_TELEMETRY_FILE)

                    logger.info(
                        "[CAPITAL] post-exec update: effective=¥%s ramp=%s deploy=%s "
                        "aggr_ema=%.3f edge=%.3f mh_agg=%.3f opp_cost=%.2fbps",
                        f"{_new_cap.effective_capital:,.0f}", _new_ramp.mode, _new_deploy_rec.state,
                        _new_aggression.ema_score, _new_edge.composite_edge_score,
                        _new_mh.aggregate_confidence, _opp_rec.opportunity_cost_bps,
                    )
                except Exception as _cap_update_err:
                    logger.warning(
                        "[CAPITAL] post-exec state update failed (%s) — states unchanged",
                        _cap_update_err,
                    )

            # ── Future Leader survivability materialization (LIVE) ───────────
            try:
                from src.analytics.future_leader_survivability import (
                    run_future_leader_survivability_materialization as _fl_surv_live,
                )
                _fl_surv_live_count = _fl_surv_live(
                    candidates_log_path   = FUTURE_LEADER_CANDIDATES_FILE,
                    integrity_log_dir     = FUTURE_LEADER_INTEGRITY_DIR,
                    survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                    backtest_dataset_dir  = BACKTEST_DATASET_DIR,
                    default_data_version  = DEFAULT_DATA_VERSION or "",
                    cache_dir             = CACHE_DIR,
                )
                if _fl_surv_live_count > 0:
                    logger.info(
                        "[FL_SURV] live: materialized %d survivability records",
                        _fl_surv_live_count,
                    )
            except Exception as _fl_surv_live_err:
                logger.warning(
                    "[FL_SURV] live hook failed (fail-open): %s", _fl_surv_live_err
                )

            # ── Future Leader failure clustering (LIVE) ──────────────────────
            try:
                from src.analytics.future_leader_failure_clustering import (
                    run_future_leader_failure_materialization as _fl_fail_live,
                )
                _fl_fail_live_count = _fl_fail_live(
                    candidates_log_path   = FUTURE_LEADER_CANDIDATES_FILE,
                    survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                    failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                    backtest_dataset_dir  = BACKTEST_DATASET_DIR,
                    default_data_version  = DEFAULT_DATA_VERSION or "",
                    cache_dir             = CACHE_DIR,
                )
                if _fl_fail_live_count > 0:
                    logger.info(
                        "[FL_FAIL] live: classified %d failure records",
                        _fl_fail_live_count,
                    )
            except Exception as _fl_fail_live_err:
                logger.warning(
                    "[FL_FAIL] live hook failed (fail-open): %s", _fl_fail_live_err
                )

            # ── Future Leader alpha persistence half-life (LIVE) ─────────────
            try:
                from src.analytics.future_leader_half_life import (
                    run_future_leader_half_life_hook as _fl_hl_live,
                )
                _fl_hl_live(
                    survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                    failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                    reports_dir           = FUTURE_LEADER_REPORTS_DIR,
                )
            except Exception as _fl_hl_live_err:
                logger.warning(
                    "[FL_HALF_LIFE] live hook failed (fail-open): %s", _fl_hl_live_err
                )

            # ── Future Leader regime segmentation (LIVE) ─────────────────────
            try:
                from src.analytics.future_leader_regime import (
                    run_future_leader_regime_hook as _fl_regime_live,
                )
                _fl_regime_live(
                    survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                    failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                    regime_log_dir        = FUTURE_LEADER_REGIME_DIR,
                    reports_dir           = FUTURE_LEADER_REPORTS_DIR,
                    backtest_dataset_dir  = BACKTEST_DATASET_DIR,
                    default_data_version  = DEFAULT_DATA_VERSION or "",
                    cache_dir             = CACHE_DIR,
                )
            except Exception as _fl_regime_live_err:
                logger.warning(
                    "[FL_REGIME] live hook failed (fail-open): %s", _fl_regime_live_err
                )

            try:
                from src.analytics.future_leader_archetype import (
                    run_future_leader_archetype_hook as _fl_arch_live,
                )
                _fl_arch_live(
                    failure_log_dir   = FUTURE_LEADER_FAILURE_DIR,
                    regime_log_dir    = FUTURE_LEADER_REGIME_DIR,
                    archetype_log_dir = FUTURE_LEADER_ARCHETYPE_DIR,
                    reports_dir       = FUTURE_LEADER_REPORTS_DIR,
                )
                logger.info("[FL_ARCH] live: archetype hook complete")
            except Exception as _fl_arch_live_err:
                logger.warning(
                    "[FL_ARCH] live hook failed (fail-open): %s", _fl_arch_live_err
                )

            try:
                from src.analytics.future_leader_transition import (
                    run_future_leader_transition_hook as _fl_trans_live,
                )
                _fl_trans_live(
                    failure_log_dir       = FUTURE_LEADER_FAILURE_DIR,
                    archetype_log_dir     = FUTURE_LEADER_ARCHETYPE_DIR,
                    survivability_log_dir = FUTURE_LEADER_SURVIVABILITY_DIR,
                    regime_log_dir        = FUTURE_LEADER_REGIME_DIR,
                    transition_log_dir    = FUTURE_LEADER_TRANSITION_DIR,
                    reports_dir           = FUTURE_LEADER_REPORTS_DIR,
                )
                logger.info("[FL_TRANS] live: transition hook complete")
            except Exception as _fl_trans_live_err:
                logger.warning(
                    "[FL_TRANS] live hook failed (fail-open): %s", _fl_trans_live_err
                )

            # ── Phase 4B: Micro Live Governance Bridge ────────────────────────
            # Fail-open: telemetry only; no execution mutation.
            _gov_report    = None
            _gov_upstream  = None
            _prev_gov_audit = None
            try:
                from src.deployment.micro_live_governance_bridge import (
                    UpstreamGovernanceState,
                    run_governance_bridge_hook,
                )
                from src.deployment.connectors.broker_interface import LiveExecutionMode
                from src.deployment.governance_audit import load_last_governance_audit
                from src.paths import (
                    GOVERNANCE_AUDIT_DIR,
                )
                _prev_gov_audit = load_last_governance_audit(GOVERNANCE_AUDIT_DIR)
                _gov_upstream = UpstreamGovernanceState(
                    universe_version=str(getattr(cfg, "universe_version", "v1")),
                    deployment_manifest_hash=str(run_id[:16] if run_id else "unknown"),
                    rollback_state="stable",
                    rollback_pressure_score=0.0,
                    eligibility_gate_passed=True,
                    intended_symbols=[],
                    intended_weights={},
                    upstream_hashes={"run_id": str(run_id or "")},
                )
                _gov_report = run_governance_bridge_hook(
                    state_dir=GOVERNANCE_AUDIT_DIR,
                    audit_dir=GOVERNANCE_AUDIT_DIR,
                    upstream=_gov_upstream,
                    broker=None,
                    total_capital=float(_eff_capital),
                    daily_order_count=len(send_results or []),
                    mode=LiveExecutionMode.HUMAN_CONFIRMED,
                )
                logger.info(
                    "[GOV_BRIDGE] date=%s freeze=%s recon=%s hash=%s",
                    _gov_report.evaluation_date,
                    _gov_report.freeze_state.frozen,
                    _gov_report.reconciliation.status,
                    _gov_report.governance_audit.deterministic_hash,
                )
            except Exception as _gov_bridge_err:
                logger.warning("[GOV_BRIDGE] hook failed (fail-open): %s", _gov_bridge_err)

            # ── Phase 4B.1: Governance Decision Diff ─────────────────────────
            # Fail-open: diff telemetry only; no execution mutation.
            try:
                if _gov_report is not None:
                    from src.deployment.governance_audit import (
                        compute_governance_diff,
                        append_governance_diff,
                    )
                    from src.paths import GOVERNANCE_DIFF_DIR
                    _gov_diff = compute_governance_diff(_prev_gov_audit, _gov_report.governance_audit)
                    append_governance_diff(_gov_diff, GOVERNANCE_DIFF_DIR)
                    if _gov_diff.changed:
                        logger.info(
                            "[GOV_DIFF] hash %s→%s added=%d removed=%d freeze_changed=%s",
                            _gov_diff.previous_hash,
                            _gov_diff.current_hash,
                            len(_gov_diff.added_decisions),
                            len(_gov_diff.removed_decisions),
                            _gov_diff.freeze_state_changed,
                        )
            except Exception as _gov_diff_err:
                logger.warning("[GOV_DIFF] hook failed (fail-open): %s", _gov_diff_err)

            # ── Phase 4B.1: Human Confirmation Telemetry ─────────────────────
            # Fail-open: outcome telemetry only; no execution mutation.
            try:
                if _gov_report is not None:
                    from src.deployment.human_confirmation_telemetry import (
                        make_approved_outcome,
                        make_rejected_outcome,
                        append_human_confirmation,
                    )
                    from src.paths import HUMAN_CONFIRMATION_DIR
                    _gov_hash   = _gov_report.governance_audit.deterministic_hash
                    _gov_frozen = _gov_report.freeze_state.frozen
                    _intended   = (_gov_upstream.intended_symbols or []) if _gov_upstream else []
                    _syms_to_log = _intended if _intended else ["_no_intent_"]
                    for _sym in _syms_to_log:
                        if _gov_frozen:
                            _hc = make_rejected_outcome(
                                symbol=_sym,
                                intended_action="BUY",
                                governance_hash=_gov_hash,
                                rejection_reason="governance_frozen",
                            )
                        else:
                            _hc = make_approved_outcome(
                                symbol=_sym,
                                intended_action="BUY",
                                governance_hash=_gov_hash,
                            )
                        append_human_confirmation(_hc, HUMAN_CONFIRMATION_DIR)
                    logger.info(
                        "[HUMAN_CONF] frozen=%s symbols=%d",
                        _gov_frozen if _gov_report else "n/a",
                        len(_syms_to_log),
                    )
            except Exception as _hc_err:
                logger.warning("[HUMAN_CONF] hook failed (fail-open): %s", _hc_err)

            # ── Phase 4B: Early Promotion Telemetry ──────────────────────────
            # Fail-open: observation only; no LIVE_UNIVERSE mutation.
            try:
                from src.analytics.early_promotion_telemetry import (
                    run_early_promotion_telemetry_hook as _ep_hook,
                )
                from src.paths import EARLY_PROMOTION_LOG_DIR
                _fl_cands_raw = []
                try:
                    from src.live.future_leader_screener import load_future_leader_log
                    _today_str = datetime.now(JST).strftime("%Y-%m-%d")
                    _fl_cands_raw = [
                        {
                            "symbol":              r.get("symbol", ""),
                            "future_leader_score": r.get("future_leader_score", 0.0),
                            "rsr_zone":            r.get("rsr_zone", ""),
                            "accum_score":         r.get("accum_score", 0.0),
                            "drift_5d":            r.get("drift_5d", 0.0),
                            "consecutive_days":    r.get("consecutive_days", 0),
                        }
                        for r in load_future_leader_log(
                            log_dir=Path("logs/future_leader_integrity"),
                            date_str=_today_str.replace("-", ""),
                        )
                    ]
                except Exception:
                    _fl_cands_raw = []
                _ep_fired = _ep_hook(
                    future_leader_candidates=_fl_cands_raw,
                    log_dir=EARLY_PROMOTION_LOG_DIR,
                    signal_date=datetime.now(JST).strftime("%Y-%m-%d"),
                )
                if _ep_fired:
                    logger.info(
                        "[EARLY_PROM] %d early promotion candidate(s): %s",
                        len(_ep_fired), [s.symbol for s in _ep_fired],
                    )
            except Exception as _ep_err:
                logger.warning("[EARLY_PROM] hook failed (fail-open): %s", _ep_err)

            # ── Phase 4B: Missed Alpha Enrichment ────────────────────────────
            # Fail-open: analytics only; no execution mutation.
            try:
                from src.analytics.missed_alpha_enrichment import (
                    run_missed_alpha_enrichment_hook as _ma_enrich,
                )
                from src.paths import (
                    MISSED_ALPHA_LOG_DIR,
                    MISSED_ALPHA_ENRICHED_DIR,
                )
                _ma_summary = _ma_enrich(
                    missed_alpha_log_dir=MISSED_ALPHA_LOG_DIR,
                    survivability_log_dir=FUTURE_LEADER_SURVIVABILITY_DIR,
                    enriched_log_dir=MISSED_ALPHA_ENRICHED_DIR,
                    evaluation_date=datetime.now(JST).strftime("%Y-%m-%d"),
                )
                if _ma_summary:
                    logger.info(
                        "[MISSED_ALPHA] total=%d enriched=%d avg_fwd10d=%s high_cost=%d",
                        _ma_summary.total_records,
                        _ma_summary.enriched_count,
                        f"{_ma_summary.avg_fwd_ret_10d:.4f}" if _ma_summary.avg_fwd_ret_10d is not None else "N/A",
                        _ma_summary.high_cost_count,
                    )
            except Exception as _ma_err:
                logger.warning("[MISSED_ALPHA] enrichment hook failed (fail-open): %s", _ma_err)

            # ── Phase 5A: Runtime Orchestrator Checkpoint ─────────────────────
            # Fail-open: lifecycle telemetry only; no execution mutation.
            try:
                from src.runtime.future_leader_runtime_orchestrator import build_orchestrator
                _orc = build_orchestrator()
                _orc_date = datetime.now(JST).strftime("%Y-%m-%d")
                _orc_results = _orc.run_pipeline(evaluation_date=_orc_date)
                _orc_pass = sum(1 for r in _orc_results if r.success or r.skipped_as_idempotent)
                logger.info(
                    "[RUNTIME_ORC] date=%s state=%s stages=%d/%d",
                    _orc_date,
                    _orc.lifecycle_state.value,
                    _orc_pass,
                    len(_orc_results),
                )
            except Exception as _orc_err:
                logger.warning("[RUNTIME_ORC] hook failed (fail-open): %s", _orc_err)

            # ── log save ─────────────────────────────────────────────────────
            _emit_phase("persistence", "start", run_id=run_id)
            result_dict = json.loads(result.to_json())
            result_dict["send_results"] = send_results
            if not args.no_save:
                _exec_ts  = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
                saved_path = Path(args.output_dir) / f"signal_{_exec_ts}_executed.json"
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                saved_path.write_text(
                    json.dumps(result_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"\n💾 発注結果保存: {saved_path}")
                save_live_logs(run_id, result, send_results)
                print(f"📋 ライブログ: logs/live/{run_id}_orders.json")

            # ── AUTO_PROMOTE_SAFE_V2 report (LIVE) ───────────────────────────
            try:
                from src.universe.auto_promote_safe_v2 import format_probation_report as _v2_rpt_lv
                from src.paths import PROBATION_PROMOTIONS_FILE as _v2_pf_lv, PROBATION_OUTCOMES_FILE as _v2_of_lv
                _v2_section_lv = _v2_rpt_lv(_v2_pf_lv, _v2_of_lv)
                if _v2_section_lv:
                    print(_v2_section_lv)
            except Exception as _v2_lv_err:
                logger.warning("[V2] live report failed: %s", _v2_lv_err)

            # ── Entry Timing Intelligence report (LIVE) ───────────────────────
            try:
                from src.entry import (
                    EntryTimingResult as _ETResultLv,
                    format_et_report_section as _et_rpt_lv,
                    append_et_telemetry      as _et_telem_lv,
                    materialize_et_returns   as _et_mat_lv,
                )
                from src.paths import ENTRY_TIMING_TELEMETRY_FILE as _et_tpath_lv
                _et_lv_scores: dict = {}
                for _s_et_lv in result.signals:
                    _et_sc_lv = _s_et_lv.get("entry_timing_score")
                    _et_cf_lv = _s_et_lv.get("entry_timing_confidence")
                    if _et_sc_lv is not None and _et_cf_lv is not None:
                        _sym_et_lv = _s_et_lv.get("symbol", "")
                        _et_lv_scores[_sym_et_lv] = _ETResultLv(
                            symbol             = _sym_et_lv,
                            score              = float(_et_sc_lv),
                            confidence         = str(_et_cf_lv),
                            action             = str(_s_et_lv.get("entry_timing_action", "NORMAL")),
                            breakout_component = 50.0,
                            pullback_component = 50.0,
                            trend_component    = 50.0,
                            market_component   = 60.0,
                            phase              = str(_s_et_lv.get("entry_timing_phase", "normal")),
                        )
                _today_str_et_lv = datetime.now(JST).strftime("%Y-%m-%d")
                for _sym_telem_lv in result.top_k_symbols:
                    if _sym_telem_lv in _et_lv_scores:
                        _act_lv = "ENTERED" if any(
                            o.get("symbol") == _sym_telem_lv and o.get("side") == "BUY"
                            for o in result.orders
                        ) else "WATCHED"
                        _et_telem_lv(
                            _et_lv_scores[_sym_telem_lv],
                            action_taken   = _act_lv,
                            telemetry_path = _et_tpath_lv,
                            date_str       = _today_str_et_lv,
                        )
                try:
                    import pandas as _pd_et_lv
                    def _et_ohlcv_loader_lv(sym: str):
                        _ep_lv = CACHE_DIR / "ohlcv" / f"{sym}.parquet"
                        return _pd_et_lv.read_parquet(_ep_lv) if _ep_lv.exists() else None
                    _et_mat_lv(_et_tpath_lv, _et_ohlcv_loader_lv)
                except Exception:
                    pass
                if _et_lv_scores:
                    _et_section_lv = _et_rpt_lv(_et_lv_scores, _et_tpath_lv)
                    if _et_section_lv:
                        print(_et_section_lv)
            except Exception as _et_rpt_lv_err:
                logger.warning("[ET] live report failed: %s", _et_rpt_lv_err)

            # ── Entry Timing Promotion evaluation (LIVE) ──────────────────────
            try:
                from src.entry.entry_timing_promotion import (
                    run_entry_timing_promotion    as _et_promo_run_lv,
                    format_et_promotion_section   as _et_promo_fmt_lv,
                )
                from src.paths import (
                    ENTRY_TIMING_TELEMETRY_FILE  as _et_tfile_lv,
                    ENTRY_TIMING_PROMOTION_FILE  as _et_pfile_lv,
                    ENTRY_TIMING_HISTORY_FILE    as _et_hfile_lv,
                    ENTRY_TIMING_REPORTS_DIR     as _et_rdir_lv,
                )
                _et_rdir_lv.mkdir(parents=True, exist_ok=True)
                _et_promo_state_lv = _et_promo_run_lv(
                    telemetry_file  = _et_tfile_lv,
                    promotion_file  = _et_pfile_lv,
                    history_file    = _et_hfile_lv,
                    today_str       = datetime.now(JST).strftime("%Y-%m-%d"),
                )
                _et_promo_sec_lv = _et_promo_fmt_lv(
                    _et_pfile_lv, datetime.now(JST).strftime("%Y-%m-%d")
                )
                if _et_promo_sec_lv:
                    print(_et_promo_sec_lv)
            except Exception as _et_promo_lv_err:
                logger.warning("[ET_PROMO] live promotion eval failed: %s", _et_promo_lv_err)

            # ── Position Sizing Intelligence report (LIVE) ────────────────────
            try:
                from src.portfolio.position_sizing_intelligence import (
                    PositionSizingSignal      as _PSSig_lv,
                    append_ps_telemetry       as _ps_telem_fn_lv,
                    compute_ps_kpis           as _ps_kpis_fn_lv,
                    format_ps_report_section  as _ps_rpt_fn_lv,
                )
                from src.paths import POSITION_SIZING_TELEMETRY_FILE as _ps_tpath_lv
                _ps_lv_sigs = []
                _ps_rsr_lv: dict = {}
                _ps_et_lv:  dict = {}
                _now_lv_str = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S+09:00")
                _today_lv_str = datetime.now(JST).strftime("%Y-%m-%d")
                for _s_pslv in result.signals:
                    _conv_lv = _s_pslv.get("conviction_score")
                    _vw_lv   = _s_pslv.get("virtual_weight")
                    if _conv_lv is not None and _vw_lv is not None and _s_pslv.get("signal") == 1:
                        _sym_lv = _s_pslv["symbol"]
                        _ps_lv_sigs.append(_PSSig_lv(
                            symbol           = _sym_lv,
                            conviction_score = float(_conv_lv),
                            virtual_weight   = float(_vw_lv),
                            component_scores = {},
                            reason_codes     = [],
                            computed_at      = _now_lv_str,
                        ))
                        _ps_rsr_lv[_sym_lv] = _s_pslv.get("rsr")
                        _ps_et_lv[_sym_lv]  = _s_pslv.get("entry_timing_score")
                if _ps_lv_sigs:
                    _n_pslv = len(_ps_lv_sigs)
                    _eq_w_lv = {sig.symbol: 1.0 / _n_pslv for sig in _ps_lv_sigs}
                    _ps_telem_fn_lv(
                        signals              = _ps_lv_sigs,
                        actual_weights       = _eq_w_lv,
                        rsr_scores           = _ps_rsr_lv,
                        entry_timing_scores  = _ps_et_lv,
                        future_leader_scores = {},
                        telemetry_path       = _ps_tpath_lv,
                        date_str             = _today_lv_str,
                    )
                    _ps_kpis_lv = _ps_kpis_fn_lv(_ps_tpath_lv)
                    _ps_sec_lv  = _ps_rpt_fn_lv(_ps_lv_sigs, _ps_kpis_lv)
                    if _ps_sec_lv:
                        print(_ps_sec_lv)
            except Exception as _ps_lv_err:
                logger.warning("[PSI] live report failed: %s", _ps_lv_err)

            # ── Position Sizing Promotion evaluation (LIVE) ───────────────────
            try:
                from src.portfolio.position_sizing_promotion import (
                    run_position_sizing_promotion as _psp_run_lv,
                    format_psp_report_section as _psp_fmt_lv,
                )
                from src.paths import (
                    POSITION_SIZING_TELEMETRY_FILE as _psp_tfile_lv,
                    POSITION_SIZING_PROMOTION_FILE as _psp_pfile_lv,
                    POSITION_SIZING_HISTORY_FILE as _psp_hfile_lv,
                    POSITION_SIZING_REPORTS_DIR as _psp_rdir_lv,
                    OHLCV_DIR as _psp_ohlcv_lv,
                )
                _psp_rdir_lv.mkdir(parents=True, exist_ok=True)
                _psp_run_lv(
                    telemetry_file=_psp_tfile_lv,
                    promotion_file=_psp_pfile_lv,
                    history_file=_psp_hfile_lv,
                    ohlcv_dir=_psp_ohlcv_lv,
                    today_str=datetime.now(JST).strftime("%Y-%m-%d"),
                )
                _psp_sec_lv = _psp_fmt_lv(_psp_pfile_lv, datetime.now(JST).strftime("%Y-%m-%d"))
                if _psp_sec_lv:
                    print(_psp_sec_lv)
            except Exception as _psp_lv_err:
                logger.warning("[PSP] live promotion eval failed: %s", _psp_lv_err)

            # ── Breakout Quality Intelligence report (LIVE) ───────────────────
            try:
                from src.analytics.breakout_quality import format_bq_report_section as _bq_rpt_lv
                from src.paths import BREAKOUT_QUALITY_TELEMETRY_FILE as _bq_rpt_path_lv
                _bq_section_lv = _bq_rpt_lv(_bq_rpt_path_lv)
                if _bq_section_lv:
                    print(_bq_section_lv)
            except Exception as _bq_rpt_lv_err:
                logger.warning("[BQ] live report failed: %s", _bq_rpt_lv_err)

            # ── Continuation Priority Intelligence report (LIVE) ──────────────
            try:
                from src.analytics.continuation_priority import (
                    format_priority_ranking_table as _cp_table_lv,
                    format_cp_kpi_section         as _cp_kpi_lv,
                )
                from src.paths import CONTINUATION_PRIORITY_TELEMETRY_FILE as _cp_rpt_path_lv
                _cp_rank_lv = _cp_table_lv(_cp_results)
                if _cp_rank_lv:
                    print(_cp_rank_lv)
                _cp_kpi_str_lv = _cp_kpi_lv(_cp_rpt_path_lv)
                if _cp_kpi_str_lv:
                    print(_cp_kpi_str_lv)
            except Exception as _cp_rpt_lv_err:
                logger.warning("[CP] live report failed: %s", _cp_rpt_lv_err)

            _emit_phase("persistence", "complete", run_id=run_id)

    except Exception as _top_e:
        logger.error("supervisor top-level exception: %s", _top_e, exc_info=True)
        _exit_code = 1

    finally:
        # always stop heartbeat and release lock — atexit also calls release but
        # explicit cleanup here ensures immediate release on early return / crash
        _emit_phase("shutdown_cleanup", "start", run_id=run_id)
        # ── [THREAD_DUMP] 終了直前スレッド状態診断 ──────────────────────────
        for _td_t in threading.enumerate():
            logger.warning(
                "[THREAD_DUMP] name=%s daemon=%s alive=%s",
                _td_t.name,
                _td_t.daemon,
                _td_t.is_alive(),
            )
        safe_cleanup(
            ("stop_heartbeat", _heartbeat.stop if _heartbeat is not None else (lambda: None)),
            ("release_lock",   release_runtime_lock),
        )
        _shutdown_audit(run_id=run_id)
        _emit_phase("final_exit", "complete", run_id=run_id, extra={"exit_code": _exit_code})

    return _exit_code


if __name__ == "__main__":
    sys.exit(main())
