"""
src/run_live_signal_f4_tp50.py
F4 TP50_T15 専用ライブ実行パス（Frozen spec: docs/f4_tp50_t15_spec.md — 変更禁止）。

位置づけ: E5(`src/run_live_signal_simple_e5.py`)・Fujiko(`src/run_live_signal.py`)・
TP30(`src/run_live_signal_f4_tp30.py`)とは完全に分離した、TP50専用の軽量ライブ実行パス。
いずれのスクリプトも一切変更しない。TP30スクリプトはrollback/reference用としてそのまま
本番コードに残置する（削除しない・cutoverは未実行）。

由来（2026-08-16 TP30→TP50移行）: F4 TP Threshold Neighborhood研究
（backtests/f4_tp_threshold_neighborhood/）により、Trail=15%固定のもとTP50が
TP{35,40,45,55}近傍の中でOOS Calmar最高（1.199）・IS Calmarも最高（0.442、
5構成中トップ）と判明。形状分類=A_LOCAL_ROBUST_PLATEAU。FROZEN_TP=TP50_T15として
凍結。本スクリプトはTP30本番実装からの最小差分移行であり、Exit target閾値
（entry×1.30 → entry×1.50）以外は一切変更していない。

固定仕様（結果を見て変更しない・正本は docs/f4_tp50_t15_spec.md）:
  Entry: F4 Previous-FY B×B（DivAnn>0・EPS>0・DivAnn/RAW-CloseT0>=3.0%・
         PayoutRatioAnn<=70%、PERゲートなし）。T0=DiscDate、T0+1=SignalDate、
         T0+2=Entry（ADJUSTED Open）。TP30と完全同一（src/f4_tp50/entry_pipeline.py
         はsrc/f4_tp30/entry_pipeline.pyのbit-for-bit同一ロジック——同じ閾値定数・
         同じbacktest関数チェーンを呼ぶのみ）。
  Exit : 15% high/low-intraday trailingストップ + 50%固定ターゲット（entry×1.50、
         TP30からの唯一の戦略差分）、同日両成立時はtrailing優先。ADJUSTED OHLC基準。
  数量 : 常に100株固定（部分縮小なし）。
  上限 : max_positions等の固定ポジション上限なし。
  優先順位: なし（コード昇順）。
  資金 : ¥3,500,000（frozen spec §5の想定資本規模。実際の発注可否は毎回
         fetch_broker_snapshot() の実口座cashで判定——E5/TP30と同一のBroker-as-Sole-SSOT
         パターン。1口座を共有するため、E5/TP30との合算資金制約は別途の配分設計が必要）。

再利用する共通基盤（複製しない・TP30と完全同一の共有インフラ）:
  - src.run_live_signal._submit_orders_process_isolated（発注ライフサイクル・重複防止込み）
  - src.run_live_signal_simple_e5.handle_order_submission_stage_failure（StageTimeout/
    StageError時の安全なfail-close処理。TP30と同じく汎用のためそのままimportして再利用）
  - src.live.inflight_registry.InflightRegistry（未約定注文の重複防止・crash recovery、
    E5・TP30・Fujikoと同一の共有ファイルruntime/inflight_orders.jsonl）
  - src.live.client_order_id.make_client_order_id（strategy名込みの決定的idempotencyキー）
  - src.execution.dd_engine（compute_drawdown / is_cb_trigger / assess_risk、
    E5・TP30・Fujikoと同一equity_peak/cb_stateを共有する単一circuit breaker）
  - src.portfolio.broker_source.fetch_broker_snapshot（broker実態cash/position取得）
  - src.portfolio.state_store（load_portfolio_state / save_portfolio_state — 唯一の書込経路）
  - src.f4_tp50.entry_pipeline（TP30と同一のPIT signal計算関数チェーンを再利用
    ——decision parityを構造的に保証。TP30固有コードの複製ではなく、TP30と同じ下層関数を
    同じ定数で呼ぶ別モジュール）
  - src.f4_tp50.exit_engine（TP30のexit_engineをTARGET_PCT=0.50のみ変更したフォーク。
    TradingView検証はTP30分をそのまま再利用可——trailing/gap/touch/優先順位の機構は
    無変更。target=50%境界のみTP50固有の追加検証が必要——docs/research/
    2026-08-16_tp50_t15_production_implementation.md参照）

Position/Cash管理: runtime/portfolio_state.json をE5・TP30・Fujikoと共有し、TP50建玉には
position_strategy_types="f4_tp50" でタグ付けする（TP30と同じ既存方針を踏襲。strategy tag
isolationにより他戦略の建玉には一切触れない——tests/test_e5_tp30_tp50_cutover_isolation.py
で3方向の相互非干渉を検証済み）。

Fundamentals鮮度ガード（frozen specの一部ではない・運用安全策・TP30と同一実装）: 開示データ
（DiscDate）が src.f4_tp50.entry_pipeline.MAX_FUNDAMENTALS_STALENESS_BDAYS 営業日より
古い場合、新規BUYを"シグナルなし"として静かに0件扱いにせず、明示的にブロックする
（STALE_FUNDAMENTALS_DATA・fail-closed）。Exit/リスク管理は鮮度ガードの影響を受けず
常に通常通り動作する。

entry_freeze: strategy.yaml の entry_freeze.enabled（E5/TP30と共有するグローバル設定）が
true の間、新規BUYは生成されても必ずブロックされ発注されない（SELL/exitはブロックしない）。

dry-run（デフォルト・--live未指定）: シグナル計算・sizing判定・skip理由の内訳・監査ログ出力
までを行うが、発注・portfolio_state.jsonへの書き込みは一切行わない（読み取り専用）。

使い方:
    python -m src.run_live_signal_f4_tp50                 # dry-run（デフォルト）
    python -m src.run_live_signal_f4_tp50 --live           # 実発注（freeze/stale中はBUY自動ブロック）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.config_loader import load_strategy_config
from src.database import ca_guard
from src.execution import dd_engine
from src.f4_tp50 import exit_engine as ee
from src.f4_tp50.entry_pipeline import (
    TP50LiveData,
    check_fundamentals_freshness,
    compute_today_entry_candidates,
    load_live_data,
)
from src.live.client_order_id import make_client_order_id  # noqa: F401 (used indirectly via _submit_orders_process_isolated)
from src.live.inflight_registry import InflightRegistry
from src.live.safe_cleanup import safe_cleanup_step  # noqa: F401 (re-exported for handle_order_submission_stage_failure reuse)
from src.live.staged_supervisor import StageError, StageTimeout
from src.market_snapshot.universe import to_kabu_symbol
from src.paths import (
    DATABASE_MASTER_DIR,  # noqa: F401 (kept for parity/parametrization; not used directly here)
    INFLIGHT_REGISTRY_FILE,
    LIVE_LOG_DIR,
    RUNTIME_DIR,
    assert_execution_context,
)
from src.portfolio.state_store import load_portfolio_state, save_portfolio_state

logger = logging.getLogger("f4_tp50_live")
_JST = timezone(timedelta(hours=9))

# LIVE_MODE=true 環境でのスクリプト許可リスト検証（E5/TP30と同一パターン）。
assert_execution_context()

STRATEGY_TYPE = "f4_tp50"
FIXED_LOT_SIZE = ee.FIXED_LOT  # = 100

AUDIT_DIR = RUNTIME_DIR / "f4_tp50"
AUDIT_LOG_FILE = AUDIT_DIR / "order_audit_log.jsonl"


def score_replacement_enabled() -> bool:
    """
    SCORE_REPLACEMENT_ENABLED kill switch (docs/f4_tp50_t15_spec.md sec.5.2,
    docs/f4_score_replacement/09_kill_switch.md). Default False at every
    call site with no arguments overridden. Env var takes precedence over
    config when both are set (same precedence pattern as the existing
    ENTRY_FREEZE_ENABLED env override of strategy.yaml's entry_freeze.enabled
    — see src/config_loader.py).

    Reads src/configs/strategy.yaml DIRECTLY (does not extend the shared
    src.config_loader.StrategyConfig dataclass) so this Score Replacement
    extension stays fully contained to F4 TP50's own files — no shared-infra
    schema change that E5/TP30/Fujiko/RSR's config loading would need to
    account for.

    COMPLETELY SEPARATE from QUALITY_REPLACEMENT_ENABLED (the existing
    Study57/58A Quality Replacement Engine for the RSR/Fujiko strategy,
    src/research_candidate/quality_replacement.py, read inside
    src/run_live_signal.py — a different file this module never imports).
    No flag sharing, no state sharing, no import between the two.
    """
    env_val = os.environ.get("SCORE_REPLACEMENT_ENABLED")
    if env_val is not None:
        return env_val.strip().lower() in ("1", "true", "yes", "on")
    try:
        import yaml

        from src.paths import STRATEGY_CONFIG_FILE
        raw = yaml.safe_load(STRATEGY_CONFIG_FILE.read_text(encoding="utf-8")) or {}
        f4 = raw.get("f4_tp50") or {}
        sr = f4.get("score_replacement") or {}
        return bool(sr.get("enabled", False))
    except Exception as exc:
        logger.warning("[F4_TP50] score_replacement config読込失敗・既定Falseを使用: %s", exc)
        return False

# 発注件数に応じたtimeout（E5/TP30と同一ロジック・別定数——TP50もmax_positions無しのため
# 多数のBUYを一度に生成し得る）。
_ORDER_TIMEOUT_BASE_SEC = 15.0
_ORDER_TIMEOUT_PER_ORDER_MARGIN_SEC = 3.0
_ORDER_TIMEOUT_FLOOR_SEC = 30.0


def compute_order_submission_timeout_sec(n_orders: int) -> float:
    if n_orders <= 0:
        return _ORDER_TIMEOUT_FLOOR_SEC
    rate_limit_sec = float(os.environ.get("ORDER_RATE_LIMIT_SEC", "5"))
    computed = _ORDER_TIMEOUT_BASE_SEC + n_orders * (rate_limit_sec + _ORDER_TIMEOUT_PER_ORDER_MARGIN_SEC)
    return max(_ORDER_TIMEOUT_FLOOR_SEC, computed)


@dataclass
class OrderInstruction:
    """_submit_orders_process_isolated が期待する duck-typed 注文オブジェクト（E5/TP30と同一形状）。"""
    symbol: str
    side: str
    qty: int
    strategy_type: str = STRATEGY_TYPE
    estimated_price: float = 0.0
    atr20: float = 0.0
    reason: str = ""
    sector: str = "不明"
    symbol_4digit: str = ""

    def __post_init__(self):
        if not self.symbol_4digit:
            self.symbol_4digit = self.symbol


def _today_jst() -> str:
    return datetime.now(_JST).strftime("%Y-%m-%d")


# ======================================================================
# 0.5 実行結果メール通知（2026-08-18追加、既存src.notifier再利用のみ・新規SMTP実装なし）
#
# 呼び出しは main() の4箇所の return 直前（result_summary/JSON/HTMLレポート確定後）
# のみ——各 return は互いに排他なので1実行につき最大1通が構造的に保証される。
# 例外はこの関数の外へ絶対に伝播させない（TP50本体の成功/失敗判定に一切影響しない）。
# ======================================================================
_TP50_NOTIFY_SUBJECT_SUFFIX = {
    "dry_run": "[TP50][DRY-RUN] Signal report",
    "success": "[TP50][SUCCESS] Live order completed",
    "error": "[TP50][ERROR] Live order failed",
    "warning": "[TP50][WARNING] Live execution blocked",
}


def _classify_tp50_notification(result_summary: dict) -> str:
    """dry_run/error/warning/success のいずれかを判定する（優先順位: dry_run > error > warning > success）。"""
    if not result_summary.get("live"):
        return "dry_run"
    results = result_summary.get("order_submission_results")
    if results and any(not r.get("success") for r in results):
        return "error"
    fr = result_summary.get("fundamentals_freshness") or {}
    ca = result_summary.get("ca_guard") or {}
    blocked = (
        bool(fr.get("is_stale"))
        or bool(ca.get("buy_candidates_blocked_by_ca_pending"))
        or int(result_summary.get("buys_blocked_by_entry_freeze") or 0) > 0
        or (result_summary.get("risk_gate") or {}).get("recommendation") in ("CB_ACTIVE",)
    )
    return "warning" if blocked else "success"


def _format_intended_order_line(o) -> str:
    return f"  {o.side} {o.symbol} (kabu_symbol={o.symbol_4digit}) qty={o.qty} reason={o.reason}"


def _format_submitted_order_line(r: dict) -> str:
    http = f" http_status={r.get('http_status')}" if r.get("http_status") is not None else ""
    return (f"  {r.get('side')} {r.get('symbol')} (kabu_symbol={r.get('symbol_4digit')}) "
            f"qty={r.get('qty')} success={r.get('success')} order_id={r.get('order_id')} "
            f"error={r.get('error')}{http}")


def _build_tp50_notification_body(
    result_summary: dict, buy_orders_intended: list, sell_orders_intended: list,
) -> str:
    fr = result_summary.get("fundamentals_freshness") or {}
    ca = result_summary.get("ca_guard") or {}
    sb = result_summary.get("sizing_breakdown") or {}
    results = result_summary.get("order_submission_results")

    lines = [
        "strategy: f4_tp50 (TP50)",
        f"run_id: {result_summary.get('run_id')}",
        f"signal_date: {result_summary.get('signal_date')}  entry_date: {result_summary.get('entry_date')}",
        f"live: {result_summary.get('live')}",
        "",
        f"fundamentals_freshness: max_disc_date={fr.get('max_disc_date')} "
        f"is_stale={fr.get('is_stale')} reason={fr.get('reason')}",
        f"CA_PENDING: {len(ca.get('ca_pending_codes') or [])}件 "
        f"codes={ca.get('ca_pending_codes')}",
        f"entry_freeze_enabled: {result_summary.get('entry_freeze_enabled')}  "
        f"buys_blocked_by_entry_freeze: {result_summary.get('buys_blocked_by_entry_freeze')}",
        f"risk_gate: {(result_summary.get('risk_gate') or {}).get('recommendation')}",
        "",
        f"intended BUY: {len(buy_orders_intended)}件  intended SELL: {len(sell_orders_intended)}件",
    ]
    if buy_orders_intended or sell_orders_intended:
        lines.append("--- intended orders (internal symbol / kabu 4-digit symbol) ---")
        lines += [_format_intended_order_line(o) for o in sell_orders_intended]
        lines += [_format_intended_order_line(o) for o in buy_orders_intended]

    lines.append("")
    if results is None:
        lines.append("order submission results: NONE — dry-run（発注は一切送信していません）。")
    else:
        n_success = sum(1 for r in results if r.get("success"))
        lines.append(f"order submission results（実発注結果・{len(results)}件、成功{n_success}件）:")
        lines += [_format_submitted_order_line(r) for r in results]

    lines += [
        "",
        "sizing（SIMULATED FUNDING / INTERNAL SIZING — 実注文ではありません。資金充当の内部計算のみ）:",
        f"  cash_start={sb.get('cash_start')} cash_remaining={sb.get('cash_remaining')} "
        f"funded_total={sb.get('funded_total')} capital_exhausted_skip={sb.get('capital_exhausted_skip')}",
    ]

    real_order_count = len(results) if results else 0
    lines.append("")
    lines.append(f"実発注件数（real broker order attempts）: {real_order_count}")
    return "\n".join(lines)


def _send_tp50_notification(
    result_summary: dict, buy_orders_intended: list, sell_orders_intended: list,
) -> None:
    """
    Fire-and-forget（src.notifier既存仕様を尊重）。通知処理内の例外は必ずここで
    握りつぶし、main()の戻り値・scheduler taskの成否には一切影響させない。
    notifier.py自体は変更しない・subject文言はsubject_suffixでのみ調整する
    （notify_*()のsubject prefixは"✅ CHIBA 発注完了"等の固定形式のため、要求された
    "[TP50][SUCCESS] ..."はsubject_suffixとして末尾に付与する — notifier.py本体を
    変更しない制約との両立）。
    """
    try:
        from src.notifier import notify_dry_run, notify_error, notify_success, notify_warning, wait_pending

        kind = _classify_tp50_notification(result_summary)
        body = _build_tp50_notification_body(result_summary, buy_orders_intended, sell_orders_intended)
        suffix = _TP50_NOTIFY_SUBJECT_SUFFIX[kind]

        if kind == "dry_run":
            notify_dry_run(body)
        elif kind == "error":
            notify_error(body, subject_suffix=suffix)
        elif kind == "warning":
            notify_warning(body, subject_suffix=suffix)
        else:
            notify_success(body, subject_suffix=suffix)

        # notify_*() はdaemon threadでfire-and-forget送信のため、ここでブロックして
        # SMTP送信完了を待ってからプロセス終了させる（待たないとthreadごと消え未送信になる）。
        wait_pending(timeout=15.0)

        logger.info("[F4_TP50][NOTIFY] kind=%s run_id=%s", kind, result_summary.get("run_id"))
    except Exception:
        logger.warning(
            "[F4_TP50][NOTIFY] 通知処理で例外発生（TP50本体の結果には影響しません）: %s",
            traceback.format_exc(limit=3),
        )


# ======================================================================
# 1. 既保有TP50建玉のExit評価（trailing優先・gap/touch・ADJUSTED基準）
# ======================================================================
def evaluate_exits(
    data: TP50LiveData,
    broker_positions: dict[str, int],
    strategy_types: dict[str, str],
    entry_dates: dict[str, str],
    entry_prices: dict[str, float],
    as_of: pd.Timestamp,
    ca_pending_codes: set[str] | None = None,
) -> list[dict]:
    """
    broker snapshot上の建玉のうち position_strategy_types=="f4_tp50" タグのものについて、
    exit_engine.evaluate_exit() で本日のExit判定を行う。entry_date/entry_priceは
    portfolio_state.jsonの既存フィールド（position_entry_dates/position_entry_prices）
    から取得する——TP50固有の新しいトップレベルキーは追加しない。

    ca_pending_codes（2026-08-17追加、docs/f4_tp50_t15_spec.md §3.4 CA guard）:
    src.database.ca_guard.get_ca_pending_codes() が返すCA_PENDING銘柄集合。
    該当銘柄はADJUSTED OHLCの有無に関わらず（＝欠損チェックをすり抜けるケースこそが
    本ガードの対象——分割後もデータ自体は存在するが、スケールが不整合なだけ）Exit評価
    そのものを丸ごとスキップする（fail-closed）。デフォルトNoneは「CA_PENDING銘柄なし」
    として扱う（既存呼び出し元・既存isolationテストとの後方互換のため）。
    """
    ca_pending_codes = ca_pending_codes or set()
    # 2026-08-18修正: broker_positionsは"XXXX.T"（kabu Symbol由来）、strategy_types/
    # entry_dates/entry_prices/data.mats_adj列はTP50内部5桁コードでキーされており、
    # 直接一致しない（実ポジション4826.T x100がUNMANAGEDと誤判定された実障害の原因）。
    # src.portfolio.strategy_router.held_qty_by_internal_key() が両者を正規化して
    # 突き合わせ、strategy_types側の内部キー形式で結果を返す — 以後の
    # entry_dates.get(sym)等の既存ルックアップはそのまま機能する。
    from src.portfolio.strategy_router import held_qty_by_internal_key
    tp50_held_qty = held_qty_by_internal_key(broker_positions, strategy_types, STRATEGY_TYPE)
    tp50_symbols = list(tp50_held_qty.keys())

    exits = []
    for sym in sorted(tp50_symbols):  # コード昇順
        if sym in ca_pending_codes:
            logger.warning(
                "[F4_TP50][CA_GUARD] %s はCA_PENDING（株式分割検出・過去Adjustment*系列が"
                "遡及再調整されていない）— Exit評価を全面スキップ（fail-closed）。"
                "手動リカバリ手順: src/database/ca_guard_manual_recovery.py を参照。", sym,
            )
            continue
        entry_date_str = entry_dates.get(sym)
        entry_price = entry_prices.get(sym)
        if entry_date_str is None or entry_price is None:
            logger.warning(
                "[F4_TP50] %s は position_strategy_types=f4_tp50 だが entry_date/entry_price が "
                "state に無い — Exit評価をスキップ（次回commit_broker_snapshot後のreconciliation "
                "待ち。手動介入で建てられたポジションの可能性）。", sym,
            )
            continue
        entry_date = pd.Timestamp(entry_date_str)
        if entry_date >= as_of:
            continue  # 本日エントリーした銘柄は本日Exit評価しない（frozen spec: entry日は判定対象外）

        if sym not in data.mats_adj["high"].columns:
            logger.warning("[F4_TP50] %s はADJUSTED価格行列に存在しない — Exit評価をスキップ", sym)
            continue
        adj_high_series = data.mats_adj["high"][sym]
        adj_open = data.mats_adj["open"][sym].get(as_of)
        adj_low = data.mats_adj["low"][sym].get(as_of)
        adj_high_today = adj_high_series.get(as_of)
        if any(v is None or pd.isna(v) for v in (adj_open, adj_low, adj_high_today)):
            logger.warning("[F4_TP50] %s は本日のADJUSTED OHLCが取得できない — Exit評価をスキップ", sym)
            continue

        highest = ee.compute_highest_since_entry(adj_high_series, entry_date, float(entry_price), as_of)
        decision = ee.evaluate_exit(sym, float(entry_price), highest, float(adj_open), float(adj_high_today), float(adj_low))
        if decision is not None:
            exits.append({
                # 2026-08-18修正: broker_positionsは"XXXX.T"キーのため直接get(sym)不可
                # （sym=内部5桁コード）。tp50_held_qty（内部キー形式）から取得する。
                "code": sym, "qty": int(tp50_held_qty.get(sym, 0)),
                "exit_reason": decision.exit_reason, "exit_fill_price": decision.exit_fill_price,
                "stop_level": decision.stop_level, "target_price": decision.target_price,
                "highest_since_entry": decision.highest_since_entry,
            })
    return exits


# ======================================================================
# 1.5 CA_PENDING銘柄のBUY候補除外（2026-08-17追加、docs/f4_tp50_t15_spec.md §3.4）
# ======================================================================
def filter_ca_pending_candidates(
    candidates: list[dict], ca_pending_codes: set[str], audit_log: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    CA_PENDING銘柄をBUY候補から除外する。新規EntryはADJUSTED Open（当日値）を
    使うため当日値自体は分割後スケールで正しい可能性が高いが、エントリー後の
    highest_since_entryはエントリー"以後"のAdjustment*系列に依存する——エントリー
    時点でCA_PENDINGな銘柄はその後のExit評価がevaluate_exits()側のガードで
    継続的にスキップされ、ポジションが事実上塩漬けになるリスクがある。よって
    新規建てそのものをブロックする方が安全（fail-closed）。
    Returns (kept, blocked). 入力candidatesは変更しない。
    """
    kept, blocked = [], []
    for cand in candidates:
        if cand["code"] in ca_pending_codes:
            blocked.append(cand)
            if audit_log is not None:
                audit_log.append({"date": _today_jst(), "code": cand["code"], "action": "SKIP_CA_PENDING"})
        else:
            kept.append(cand)
    return kept, blocked


# ======================================================================
# 2. 100株固定サイジング（E5/TP30と同一パターン）
# ======================================================================
def evaluate_buy_sizing(
    buy_candidates: list[dict],
    available_cash: float,
    already_held_symbols: set[str],
    unresolved_buy_symbols: set[str],
    audit_log: list[dict],
) -> tuple[list[dict], dict]:
    cash = float(available_cash)
    funded = []
    capital_exhausted_candidates = []  # Score Replacement extension (2026-08-19, docs/f4_tp50_t15_spec.md
                                        # sec.5.2): additive only — every existing skip/funded behavior above
                                        # is byte-identical to before this field was added.
    n_capital_exhausted = 0
    n_already_held = 0
    n_unresolved_duplicate = 0

    for seq, cand in enumerate(buy_candidates, start=1):
        code, px = cand["code"], cand["entry_price_adjusted_open"]
        if code in already_held_symbols:
            n_already_held += 1
            audit_log.append({"date": _today_jst(), "seq": seq, "code": code, "action": "SKIP_ALREADY_HELD"})
            continue
        if code in unresolved_buy_symbols:
            n_unresolved_duplicate += 1
            audit_log.append({"date": _today_jst(), "seq": seq, "code": code, "action": "SKIP_UNRESOLVED_DUPLICATE"})
            continue

        exec_px_est = ee.compute_entry_fill_price(px)
        notional = exec_px_est * FIXED_LOT_SIZE
        commission = ee.compute_commission(notional)
        cost = notional + commission
        if cost > cash:
            n_capital_exhausted += 1
            capital_exhausted_candidates.append({"code": code, "entry_price_adjusted_open": px,
                                                 "estimated_fill_price": exec_px_est, "estimated_cost": cost})
            audit_log.append({
                "date": _today_jst(), "seq": seq, "code": code,
                "action": "SKIP_CAPITAL_EXHAUSTED", "cash_before": cash, "required": cost,
            })
            continue

        cash -= cost
        funded.append({"code": code, "entry_price_adjusted_open": px, "estimated_fill_price": exec_px_est,
                       "estimated_notional": notional, "estimated_cost": cost})
        audit_log.append({
            "date": _today_jst(), "seq": seq, "code": code, "action": "FUNDED",
            "cash_before": cash + cost, "cash_after": cash,
        })

    breakdown = {
        "orderable_signal_total": len(buy_candidates),
        "already_held_skip": n_already_held,
        "unresolved_duplicate_skip": n_unresolved_duplicate,
        "capital_exhausted_skip": n_capital_exhausted,
        "funded_total": len(funded),
        "cash_start": float(available_cash),
        "cash_remaining": cash,
        "capital_exhausted_candidates": capital_exhausted_candidates,
    }
    return funded, breakdown


def evaluate_and_execute_replacements(
    capital_exhausted_candidates: list[dict],
    score_map: dict,
    already_held_symbols: set[str],
    exit_syms: set[str],
    cash_start: float,
    live: bool,
    client,
    registry,
    trading_day: str,
    run_id: str,
    audit_log: list,
    as_of: pd.Timestamp,
) -> tuple[list[dict], list[dict], float]:
    """
    Score Replacement pass (docs/f4_tp50_t15_spec.md sec.5.2,
    docs/f4_score_replacement/05_replacement_logic.md), invoked ONLY for
    candidates that could NOT be funded directly by evaluate_buy_sizing()
    (i.e. this NEVER changes which candidates evaluate_buy_sizing() itself
    funds — it only decides what happens to the leftover, cash-exhausted
    candidates). `capital_exhausted_candidates` arrives already sorted
    score-descending (it is a sub-sequence of the already-sorted
    buy_candidates list, docs/f4_score_replacement/05 sec.2: "candidates are
    processed in Overall Score descending order").

    live=False (dry-run): decisions only, NO order is ever sent — mirrors
      exactly what would happen, using src.f4_tp50.replacement's pure
      decision function, with a SIMULATED sell fill/cash update (never a
      real broker call).
    live=True: each accepted decision is executed for real via
      src.f4_tp50.executor.execute_replacement() (SELL confirmed filled
      before BUY is ever sent).

    Returns (synthetic_results, decision_log, cash_remaining):
      synthetic_results: list of {symbol, side, success, estimated_price,
        order_id, ...} dicts in the SAME schema _submit_orders_process_isolated
        returns, so the existing apply_fill_metadata_updates() can be reused
        unchanged for portfolio_state.json bookkeeping.
      decision_log: full per-candidate audit trail (REPLACE / NO_REPLACEMENT
        + reason), matching docs/f4_score_replacement/11_shadow_plan.md
        sec.3's schema.
      cash_remaining: cash after all executed/simulated replacements.
    """
    from src.f4_tp50 import replacement as repl
    from src.f4_tp50 import replacement_state as repl_state
    from src.f4_tp50 import score as f4_score

    holdings_scores = repl_state.reconcile_with_broker(already_held_symbols)
    holdings = {
        code: {"entry_score": v.get("entry_score"), "entry_date": v.get("entry_date")}
        for code, v in holdings_scores.items() if code in already_held_symbols
    }

    cash = float(cash_start)
    synthetic_results: list[dict] = []
    decision_log: list[dict] = []
    trading_day_str = as_of.strftime("%Y-%m-%d")

    for cand in capital_exhausted_candidates:
        code = cand["code"]
        px = cand["entry_price_adjusted_open"]
        cand_score = f4_score.score_of(score_map, code)
        exec_px_est = ee.compute_entry_fill_price(px)
        notional = exec_px_est * FIXED_LOT_SIZE
        commission = ee.compute_commission(notional)
        buy_cost = notional + commission

        decision = repl.evaluate_replacement(
            cand_score=cand_score, buy_cost=buy_cost, cash=cash,
            holdings=holdings, exit_syms=exit_syms,
            target_open_price=px, target_tradable=True,
            slippage=ee.SLIPPAGE, commission=ee.COMMISSION, min_commission=ee.MIN_COMMISSION,
            fixed_lot=FIXED_LOT_SIZE,
        )
        if decision is None:
            decision_log.append({
                "date": trading_day_str, "candidate_code": code,
                "candidate_score": None if not np.isfinite(cand_score) else round(cand_score, 2),
                "decision": "NO_REPLACEMENT", "reason": "no eligible target / insufficient funding / non-finite score",
                "cash_before": round(cash, 2),
            })
            continue

        old_code = decision["target_code"]
        if not live:
            # Dry-run: simulate the fill (no order sent). Cash and holdings
            # trackers update locally for this run's reporting only — never
            # persisted (replacement_state.record_entry is only called in
            # the live branch below).
            cash = cash + decision["sell_net"] - buy_cost
            del holdings[old_code]
            holdings[code] = {"entry_score": cand_score, "entry_date": trading_day_str}
            decision_log.append({
                "date": trading_day_str, "candidate_code": code,
                "candidate_score": round(cand_score, 2),
                "decision": "REPLACE_SIMULATED", "reason": "dry-run — no order sent",
                "sold_code": old_code, "holding_score": round(decision["holding"]["entry_score"], 2),
                "score_delta": round(cand_score - decision["holding"]["entry_score"], 2),
                "sell_price": round(decision["sell_px"], 2), "sell_net": round(decision["sell_net"], 2),
                "cash_before_sell": round(cash - decision["sell_net"] + buy_cost, 2),
                "cash_after_sell_simulated": round(cash + buy_cost, 2),
                "buy_price": round(exec_px_est, 2), "buy_cost": round(buy_cost, 2),
                "cash_after_buy_simulated": round(cash, 2),
            })
            synthetic_results.append({"symbol": old_code, "side": "SELL", "success": True,
                                      "estimated_price": decision["sell_px"], "order_id": "DRY_RUN_SIMULATED",
                                      "dry_run": True})
            synthetic_results.append({"symbol": code, "side": "BUY", "success": True,
                                      "estimated_price": exec_px_est, "order_id": "DRY_RUN_SIMULATED",
                                      "dry_run": True})
            continue

        # ---------------- live: real SELL -> confirm -> BUY ----------------
        from src.f4_tp50.executor import (
            STATE_SELL_REJECTED, STATE_SELL_TIMEOUT, STATE_SELL_UNFILLED, execute_replacement,
        )
        exec_result = execute_replacement(
            client=client, registry=registry, old_code=old_code, new_code=code,
            trading_day=trading_day, run_id=run_id, audit_log=audit_log,
        )
        decision_log.append({
            "date": trading_day_str, "candidate_code": code,
            "candidate_score": round(cand_score, 2), "decision": exec_result.state,
            "sold_code": old_code, "holding_score": round(decision["holding"]["entry_score"], 2),
            "sell_order_id": exec_result.sell_order_id, "buy_order_id": exec_result.buy_order_id,
            "cash_after_sell": exec_result.cash_after_sell, "detail": exec_result.detail,
            "transitions": exec_result.transitions,
        })
        if exec_result.success:
            cash = exec_result.cash_after_sell - buy_cost
            del holdings[old_code]
            holdings[code] = {"entry_score": cand_score, "entry_date": trading_day_str}
            repl_state.clear_entry(old_code)
            repl_state.record_entry(code, cand_score, trading_day_str, exec_px_est)
            synthetic_results.append({"symbol": old_code, "side": "SELL", "success": True,
                                      "estimated_price": decision["sell_px"], "order_id": exec_result.sell_order_id})
            synthetic_results.append({"symbol": code, "side": "BUY", "success": True,
                                      "estimated_price": exec_px_est, "order_id": exec_result.buy_order_id})
        else:
            # Any non-success terminal state: no position change happened
            # (SELL never filled, or filled but BUY failed — old position
            # was already sold in that specific edge case; see
            # docs/f4_score_replacement/06 sec.3 BUY_REJECTED/BUY_TIMEOUT ->
            # REACQUIRE_OLD). Cash is NOT locally advanced here — the next
            # run's broker snapshot is the source of truth, matching
            # Broker-as-Sole-SSOT (no local cash fabrication after a
            # failure this module cannot fully resolve synchronously).
            if exec_result.state not in (STATE_SELL_REJECTED, STATE_SELL_TIMEOUT, STATE_SELL_UNFILLED):
                logger.critical(
                    "[F4_TP50_REPLACEMENT] SELL filled but BUY did not complete "
                    "(state=%s old=%s new=%s) — position state requires manual "
                    "reconciliation before next run (REACQUIRE_OLD not automatic).",
                    exec_result.state, old_code, code,
                )

    return synthetic_results, decision_log, cash


def apply_fill_metadata_updates(
    results: list[dict],
    entry_dates: dict[str, str],
    entry_prices: dict[str, float],
    strategy_types: dict[str, str],
    as_of: pd.Timestamp,
) -> tuple[dict[str, str], dict[str, float], dict[str, str], bool]:
    """
    Entry-metadata persistence — byte-identical logic to TP30's
    apply_fill_metadata_updates(). Only touches EXISTING portfolio_state.json
    schema fields (position_entry_dates/position_entry_prices/
    position_strategy_types) — no new top-level keys are introduced.

    On a successful BUY fill: records entry_date=as_of and entry_price (only if
    not already present, so a retry/duplicate result never clobbers the true
    original entry). On a successful SELL fill: removes all three fields for that
    symbol. Failed orders (success=False) never touch metadata.

    Returns (new_entry_dates, new_entry_prices, new_strategy_types, changed).
    Pure function — does not call save_portfolio_state() itself.
    """
    entry_dates = dict(entry_dates)
    entry_prices = dict(entry_prices)
    strategy_types = dict(strategy_types)
    changed = False

    for r in results:
        if not r.get("success"):
            continue
        sym = r.get("symbol")
        if r.get("side") == "BUY":
            if sym not in entry_dates:
                entry_dates[sym] = as_of.strftime("%Y-%m-%d")
                changed = True
            if sym not in entry_prices:
                entry_prices[sym] = float(r.get("estimated_price") or 0.0)
                changed = True
            if strategy_types.get(sym) != STRATEGY_TYPE:
                strategy_types[sym] = STRATEGY_TYPE
                changed = True
        elif r.get("side") == "SELL":
            for d in (entry_dates, entry_prices, strategy_types):
                if sym in d:
                    del d[sym]
                    changed = True

    return entry_dates, entry_prices, strategy_types, changed


def _try_generate_reports() -> None:
    """telemetry専用のfail-open処理（E5/TP30と同じ思想）。本体結果には影響させない。"""
    try:
        from src.run_trading_results_monitor import generate_report as generate_trading_results_report
        path = generate_trading_results_report(strategy_ids=[STRATEGY_TYPE])
        print(f"[REPORT] 売買結果モニターを更新しました: {path}")
    except Exception as exc:
        logger.warning("[F4_TP50] 売買結果モニターの生成に失敗（本体結果には影響なし）: %s", exc)


# ======================================================================
# 3. main
# ======================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="F4 TP50_T15 専用ライブ実行パス")
    parser.add_argument("--live", action="store_true", help="実発注する（entry_freeze/stale中はBUY自動ブロック）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info(
        "[F4_TP50_LAUNCHER] strategy=%s mode=%s timestamp=%s executable=%s argv0=%s",
        STRATEGY_TYPE, "live" if args.live else "dry-run",
        datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"), sys.executable, sys.argv[0],
    )

    cfg = load_strategy_config()
    entry_freeze_enabled = bool(cfg.entry_freeze.enabled)
    logger.info(
        "[F4_TP50] entry_freeze_enabled=%s reason=%s live=%s",
        entry_freeze_enabled, cfg.entry_freeze.reason, args.live,
    )

    # Score Replacement kill switch (docs/f4_tp50_t15_spec.md sec.5.2, default
    # False, COMPLETELY SEPARATE from QUALITY_REPLACEMENT_ENABLED).
    sr_enabled = score_replacement_enabled()
    logger.info("[F4_TP50] score_replacement_enabled=%s", sr_enabled)
    print(f"[SCORE_REPLACEMENT] enabled={sr_enabled}"
         + ("" if sr_enabled else "（既定値。コード昇順・従来Frozen path）"))

    run_id = f"f4_tp50_{datetime.now(_JST).strftime('%Y%m%d_%H%M%S')}"
    audit_log: list[dict] = []

    print("[LOAD] 市場データ・PITファンダメンタル読込中（数分かかります）...")
    data = load_live_data()
    as_of = data.calendar[-1]
    print(f"  最新営業日={as_of.date()}  銘柄数={len(data.codes)}")

    # ── Score計算（SCORE_REPLACEMENT_ENABLED=Trueの場合のみ。02/03準拠） ──
    score_map: dict | None = None
    if sr_enabled:
        try:
            from src.f4_tp50 import score as f4_score
            sector_map = f4_score.load_sector_map()
            today_signal_idx = len(data.calendar) - 2  # SignalDate = calendar[-2] (== signal_date below)
            score_map = f4_score.compute_today_score_map(
                data, data.panel_pf, data.sig_b, sector_map, today_signal_idx,
            )
            print(f"[SCORE] 本日のcandidate cross-section score計算完了: {len(score_map)}銘柄")
        except Exception as exc:
            logger.critical(
                "[F4_TP50] Score計算失敗 — このrunはScore Replacementを無効化し"
                "Frozen path(コード昇順)にfail-closedします: %s", exc,
            )
            print(f"[SCORE][FAIL_CLOSED] Score計算失敗のためこのrunはFrozen pathで実行: {exc}")
            score_map = None
            sr_enabled = False

    # ── CA guard: split検出銘柄のCA_PENDING状態を読込（2026-08-17追加） ──
    ca_state = ca_guard.load_ca_state()
    ca_pending_codes = set(ca_guard.get_ca_pending_codes(ca_state))
    print(f"[CA_GUARD] CA_PENDING銘柄数={len(ca_pending_codes)}"
          + (f" codes={sorted(ca_pending_codes)}" if ca_pending_codes else ""))

    freshness = check_fundamentals_freshness(data.sum_df, as_of)
    print(f"[FRESHNESS] {freshness.reason}")
    logger.info(
        "[F4_TP50] fundamentals_freshness max_disc_date=%s staleness_bdays_vs_as_of=%s "
        "staleness_bdays_vs_real_today=%s is_stale=%s",
        freshness.max_disc_date, freshness.staleness_bdays,
        freshness.staleness_bdays_vs_real_today, freshness.is_stale,
    )

    print("[LOAD] portfolio_state.json 読込中（読み取り専用・dry-runでは書き込まない）...")
    state, vr = load_portfolio_state()
    if not vr.ok:
        logger.warning("[F4_TP50] portfolio_state validation warnings: %s", vr.hard_fails + vr.warnings)
    strategy_types: dict = dict(state.get("position_strategy_types", {}))
    entry_dates: dict = dict(state.get("position_entry_dates", {}))
    entry_prices: dict = dict(state.get("position_entry_prices", {}))
    equity_peak = float(state.get("equity_peak", 0.0))
    cb_state = str(state.get("cb_state", "NORMAL"))
    safe_warn_count = int(state.get("safe_warn_count", 0))

    # ── Broker snapshot（cash・position qty の唯一の取得経路） ──
    snapshot = None
    cash_source = "unavailable"
    client = None  # Score Replacement executor reuses this SAME client when --live (defined even
                   # if the snapshot fetch below fails, so later references never NameError in dry-run).
    try:
        from src.kabusapi.client import KabuClient
        from src.portfolio.broker_source import fetch_broker_snapshot
        client = KabuClient()
        snapshot = fetch_broker_snapshot(client)
        cash_source = "broker_live"
    except Exception as exc:
        logger.warning("[F4_TP50] broker snapshot取得失敗: %s", exc)
        if args.live:
            print("[ABORT] --live 指定時にbroker snapshot取得失敗 — EMERGENCY_STOP（api_unreachable=abort）")
            return 1
        cash_source = "unavailable_dry_run_degraded"

    if snapshot is not None:
        available_cash = float(snapshot.cash)
        broker_positions = dict(snapshot.positions)
        from src.portfolio.equity import compute_live_equity
        last_equity = compute_live_equity(
            snapshot=snapshot, mode="dry" if not args.live else "live",
            equity_peak=equity_peak, persist_snapshot=False,
        )
    else:
        from src.paths import CAPITAL_STATE_FILE
        available_cash = 0.0
        broker_positions = {}
        last_equity = equity_peak
        if CAPITAL_STATE_FILE.exists():
            cap_state = json.loads(CAPITAL_STATE_FILE.read_text(encoding="utf-8"))
            available_cash = float(cap_state.get("actual_equity", 0.0))
            last_equity = available_cash
            cash_source = "capital_state_fallback_dry_run_only"
        print(
            f"[DRY_RUN] broker snapshot取得不可のため capital_state.json 参考値を使用: "
            f"¥{available_cash:,.0f}（already_held/Exit評価はbroker position不明のため実施不可・"
            "この実行では0件として扱う）"
        )

    exits = evaluate_exits(data, broker_positions, strategy_types, entry_dates, entry_prices, as_of, ca_pending_codes)
    print(f"[CALC] Exit対象（trailing 15% / target 50%）: {len(exits)}件")

    # 2026-08-18修正: evaluate_exits()と同一理由（broker "XXXX.T" vs 内部5桁コードの
    # キー不一致）。already_held_symbolsはcandidates_raw（内部5桁コード）との比較に
    # 使われるため、内部キー形式で返すheld_qty_by_internal_key()を使う。
    from src.portfolio.strategy_router import held_qty_by_internal_key
    already_held_symbols = set(held_qty_by_internal_key(broker_positions, strategy_types, STRATEGY_TYPE).keys())

    # ── 未約定注文との重複防止（InflightRegistry・E5/TP30/Fujikoと共有ファイル） ──
    registry = InflightRegistry(INFLIGHT_REGISTRY_FILE)
    registry.load()
    unresolved = registry.get_unresolved()
    unresolved_buy_symbols = {
        o.symbol for o in unresolved
        if o.side == "BUY" and o.strategy in (STRATEGY_TYPE, "", "unknown")
    }
    if unresolved_buy_symbols:
        print(f"[GUARD] 未約定BUY注文が既に存在するため重複スキップ対象: {sorted(unresolved_buy_symbols)}")

    # ── Risk Gate / HALT（dd_engine・E5/TP30/Fujikoと同一circuit breaker） ──
    risk = dd_engine.assess_risk(last_equity, equity_peak or last_equity, cb_state, safe_warn_count)
    print(f"[RISK] recommendation={risk['recommendation']} dd={risk['dd']:.2%} message={risk['message']}")
    buy_blocked_by_risk_gate = risk["recommendation"] in ("CB_ACTIVE",)

    candidates_raw, _as_of2, signal_date = compute_today_entry_candidates(
        data, already_held_symbols, score_map=score_map if sr_enabled else None,
    )
    candidates_raw, ca_blocked_candidates = filter_ca_pending_candidates(candidates_raw, ca_pending_codes, audit_log)
    if ca_blocked_candidates:
        print(f"[GUARD] CA_PENDINGによりBUY候補から除外: {[c['code'] for c in ca_blocked_candidates]}")
    _order_desc = "score降順" if sr_enabled else "コード昇順"
    print(f"[CALC] signal_date={signal_date.date()} entry_date={as_of.date()} "
          f"buy_candidates={len(candidates_raw)}件（{_order_desc}）")

    if buy_blocked_by_risk_gate:
        print(f"[GUARD] Risk Gate({risk['recommendation']})によりBUY候補 {len(candidates_raw)}件を全ブロック")
        buy_candidates = []
    elif freshness.is_stale:
        print(f"[GUARD] {freshness.reason} — BUY候補 {len(candidates_raw)}件を全ブロック"
              "（新規シグナル判定を停止。0件を'シグナルなし'として扱わない）")
        buy_candidates = []
    else:
        buy_candidates = candidates_raw

    funded, sizing_breakdown = evaluate_buy_sizing(
        buy_candidates, available_cash or 0.0, already_held_symbols, unresolved_buy_symbols, audit_log,
    )
    print(
        f"[SIZING] orderable={sizing_breakdown['orderable_signal_total']} "
        f"already_held_skip={sizing_breakdown['already_held_skip']} "
        f"unresolved_duplicate_skip={sizing_breakdown['unresolved_duplicate_skip']} "
        f"capital_exhausted_skip={sizing_breakdown['capital_exhausted_skip']} "
        f"funded={sizing_breakdown['funded_total']}"
    )

    # ── Score Replacement pass（capital_exhausted候補のみ対象。SCORE_REPLACEMENT_ENABLED時のみ） ──
    # 既存の共有batch executor（_submit_orders_process_isolated）は一切使わない。
    # F4専用のsrc.f4_tp50.executor.execute_replacement()がSELL確定fill後にのみBUYを送る。
    replacement_synthetic_results: list[dict] = []
    replacement_decision_log: list[dict] = []
    replacement_live = False  # True only if evaluate_and_execute_replacements actually ran real orders —
                              # gates whether replacement_synthetic_results may touch portfolio_state.json below.
    if sr_enabled and score_map is not None and sizing_breakdown["capital_exhausted_candidates"]:
        exit_syms = {e["code"] for e in exits}
        replacement_live = bool(args.live) and not entry_freeze_enabled
        if entry_freeze_enabled:
            print("[SCORE_REPLACEMENT] entry_freeze_enabled=true — Replacement BUYも通常BUYと同様に"
                 "ブロック対象（この実行では判定のみ・実発注なし）")
        replacement_synthetic_results, replacement_decision_log, _repl_cash_after = evaluate_and_execute_replacements(
            capital_exhausted_candidates=sizing_breakdown["capital_exhausted_candidates"],
            score_map=score_map, already_held_symbols=already_held_symbols, exit_syms=exit_syms,
            cash_start=sizing_breakdown["cash_remaining"], live=replacement_live,
            client=client, registry=registry, trading_day=signal_date.strftime("%Y-%m-%d"),
            run_id=run_id, audit_log=audit_log, as_of=as_of,
        )
        n_replace = sum(1 for d in replacement_decision_log if d["decision"] in ("REPLACE_SIMULATED", "BUY_FILLED"))
        print(f"[SCORE_REPLACEMENT] candidates_evaluated={len(replacement_decision_log)} "
             f"replace_decisions={n_replace} mode={'LIVE' if replacement_live else 'SIMULATED'}")

    buy_orders_intended = [
        OrderInstruction(symbol=f["code"], side="BUY", qty=FIXED_LOT_SIZE,
                          estimated_price=f["entry_price_adjusted_open"], reason="F4_TP50_entry_signal",
                          symbol_4digit=to_kabu_symbol(f["code"]))
        for f in funded
    ]
    sell_orders_intended = [
        OrderInstruction(symbol=e["code"], side="SELL", qty=e["qty"],
                          estimated_price=e["exit_fill_price"], reason=e["exit_reason"],
                          symbol_4digit=to_kabu_symbol(e["code"]))
        for e in exits
    ]

    if entry_freeze_enabled:
        print(f"[ENTRY_FREEZE] enabled=true reason={cfg.entry_freeze.reason} — "
              f"BUY {len(buy_orders_intended)}件を全てブロック（発注しない）。SELL/Exitはブロックしない。")
        orders_to_submit = list(sell_orders_intended)
        buy_blocked_count = len(buy_orders_intended)
    else:
        orders_to_submit = list(sell_orders_intended) + list(buy_orders_intended)
        buy_blocked_count = 0

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
        for entry in audit_log:
            entry["run_id"] = run_id
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    result_summary = {
        "run_id": run_id,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "entry_date": as_of.strftime("%Y-%m-%d"),
        "live": args.live,
        "entry_freeze_enabled": entry_freeze_enabled,
        "fundamentals_freshness": {
            "max_disc_date": str(freshness.max_disc_date.date()) if freshness.max_disc_date is not None else None,
            "staleness_bdays_vs_as_of": freshness.staleness_bdays,
            "staleness_bdays_vs_real_today": freshness.staleness_bdays_vs_real_today,
            "is_stale": freshness.is_stale,
            "reason": freshness.reason,
        },
        "cash_source": cash_source,
        "ca_guard": {
            "ca_pending_codes": sorted(ca_pending_codes),
            "buy_candidates_blocked_by_ca_pending": [c["code"] for c in ca_blocked_candidates],
        },
        "risk_gate": risk,
        "exits_intended": len(sell_orders_intended),
        "buys_intended_before_freeze": len(buy_orders_intended),
        "buys_blocked_by_entry_freeze": buy_blocked_count,
        "sizing_breakdown": sizing_breakdown,
        "order_submission_results": None,
        "score_replacement": {
            "enabled": sr_enabled,
            "candidates_evaluated": len(replacement_decision_log),
            "decisions": replacement_decision_log,
        },
    }
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = LIVE_LOG_DIR / f"f4_tp50_{run_id}.json"
    _log_path.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if not args.live:
        print("[DRY_RUN] 発注は送信しません（--live未指定）。portfolio_state.jsonへの書き込みも行いません。")
        print(f"[DRY_RUN] 送信予定だったであろう注文: SELL={len(sell_orders_intended)} "
              f"BUY(freeze/staleness考慮後)={len(orders_to_submit) - len(sell_orders_intended)}")
        print(json.dumps(result_summary, ensure_ascii=False, indent=2, default=str))
        _try_generate_reports()
        _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended)
        return 0

    # ── --live 経路（entry_freeze/staleness中はBUYが事前に除外済み） ──
    if not orders_to_submit:
        print("[LIVE] 送信対象の注文なし。")
        _try_generate_reports()
        _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended)
        return 0

    from src.live.process_supervisor import BrokerProcessSupervisor
    from src.run_live_signal import _submit_orders_process_isolated
    from src.run_live_signal_simple_e5 import handle_order_submission_stage_failure

    proc_supervisor = BrokerProcessSupervisor()
    order_timeout_sec = compute_order_submission_timeout_sec(len(orders_to_submit))
    logger.info("[F4_TP50] order_submission_timeout_sec=%.1f (n_orders=%d)", order_timeout_sec, len(orders_to_submit))
    try:
        results = _submit_orders_process_isolated(
            orders_to_submit, registry, signal_date.strftime("%Y-%m-%d"), run_id, proc_supervisor,
            timeout_sec=int(order_timeout_sec),
        )
    except (StageTimeout, StageError) as _stage_exc:
        _kind, results = handle_order_submission_stage_failure(_stage_exc, orders_to_submit, registry)
        logger.error("[F4_TP50] order_execution %s: %s", _kind, _stage_exc)
        print(f"\n[FATAL] 発注処理異常終了（{_kind}）: {_stage_exc}", file=sys.stderr)
        result_summary["order_submission_results"] = results
        _log_path.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _try_generate_reports()
        _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended)
        return 1

    print(f"[LIVE] 発注結果: {len(results)}件処理")
    for r in results:
        print(f"  {r.get('symbol')} {r.get('side')} qty={r.get('qty')} success={r.get('success')} "
              f"order_id={r.get('order_id')} error={r.get('error')}")
        if r.get("http_status") is not None:
            print(f"    http_status={r.get('http_status')} kabu_error_body={r.get('kabu_error_body')}")

    result_summary["order_submission_results"] = results
    _log_path.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _try_generate_reports()

    # ── entry metadata persistence（既存フィールドのみ更新、新規トップレベルキー無し） ──
    entry_dates, entry_prices, strategy_types, metadata_changed = apply_fill_metadata_updates(
        results, entry_dates, entry_prices, strategy_types, as_of,
    )

    # 2026-08-19 gap fix: NORMAL (non-Replacement) BUY fills must ALSO record
    # their entry score in the sidecar — previously ONLY Replacement-executed
    # buys called replacement_state.record_entry(), meaning a position bought
    # via ordinary Score-priority entry (the common case whenever capital is
    # not the binding constraint) silently had NO recorded holding score and
    # could therefore never become eligible as a future Replacement target
    # (docs/f4_score_replacement/03_score_pit_contract.md sec.4: score must be
    # captured at entry). Only runs when sr_enabled — under the default
    # SCORE_REPLACEMENT_ENABLED=False path this block never executes, so OFF
    # behavior is unaffected.
    if sr_enabled and score_map is not None:
        from src.f4_tp50 import replacement_state as _repl_state
        for r in results:
            if r.get("side") == "BUY" and r.get("success"):
                sym = r.get("symbol")
                sc = f4_score.score_of(score_map, sym)
                if sym and np.isfinite(sc):
                    _repl_state.record_entry(sym, float(sc), as_of.strftime("%Y-%m-%d"),
                                             float(r.get("estimated_price") or 0.0))

    # Score Replacement実行結果（あれば）も同じ既存フィールド更新関数で反映する
    # （src.f4_tp50.executorはportfolio_state.jsonを直接触らない — 唯一の書込経路は
    # save_portfolio_state()のまま）。holding score自体はruntime/f4_tp50/
    # score_replacement_holdings.jsonへ別途永続化済み（evaluate_and_execute_replacements内）。
    if replacement_live and replacement_synthetic_results:
        # replacement_live guards this: when False, replacement_synthetic_results (if any)
        # are DRY_RUN_SIMULATED entries and must NEVER be persisted as if they were real fills.
        entry_dates, entry_prices, strategy_types, repl_changed = apply_fill_metadata_updates(
            replacement_synthetic_results, entry_dates, entry_prices, strategy_types, as_of,
        )
        metadata_changed = metadata_changed or repl_changed

    if metadata_changed:
        state["position_entry_dates"] = entry_dates
        state["position_entry_prices"] = entry_prices
        state["position_strategy_types"] = strategy_types
        save_portfolio_state(state, data_source="f4_tp50_live_fill_metadata")
        print("[STATE] TP50 entry metadata (position_entry_dates/position_entry_prices/"
              "position_strategy_types) を更新しました。")

    _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
