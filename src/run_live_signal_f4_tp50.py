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
import re
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

# CHIBATrading_TP50_Live タスクスケジューラの登録トリガー時刻（scripts/setup_task_scheduler.ps1
# 参照）。通知本文の【EXECUTION】ブロックで実行時刻と比較し、遅延手動実行を検知する表示にのみ
# 使う（発注可否のロジックには一切影響しない — 2026-08-20 9344インシデント対応）。
_TP50_SCHEDULED_TRIGGER_HHMM = "08:49"
_TP50_MANUAL_DELAYED_THRESHOLD_MIN = 20  # スケジュール時刻からこれ以上遅れていれば「手動遅延実行」とみなす

# Signal Freshness Guard（2026-08-20夜 Production Gate監査で追加）: as_of(最新読込済み
# 営業日)が実カレンダー日からこの営業日数を超えて乖離した場合、新規BUYのみblockする
# （Exit/リスク管理は常に継続。check_fundamentals_freshness()のis_stale判定と同一方針・
# 同一のnp.busday_count手法）。WARN閾値は通常の週末/祝日乖離では発火しない水準。
MAX_ASOF_STALENESS_BDAYS_WARN = 2
MAX_ASOF_STALENESS_BDAYS_BLOCK = 4


def compute_asof_staleness_bdays(as_of_date, real_today_date) -> int:
    """np.busday_count(as_of_date, real_today_date) — check_fundamentals_freshness()と
    同一手法。純関数として切り出し、テスト容易性を確保する（2026-08-20 Production Gate監査）。"""
    return int(np.busday_count(as_of_date, real_today_date))


def should_block_buy_for_stale_asof(staleness_bdays: int) -> bool:
    return staleness_bdays > MAX_ASOF_STALENESS_BDAYS_BLOCK


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


def _is_manual_delayed_run(result_summary: dict) -> bool:
    """run_started_atがscheduled_trigger_hhmm(08:49)より_TP50_MANUAL_DELAYED_THRESHOLD_MIN分
    以上遅い場合、Task Schedulerが自動発火しなかった手動遅延実行とみなす（ヒューリスティック。
    Task Scheduler自体からの直接シグナルではなく時刻比較のみ——2026-08-20 9344インシデント
    調査時に実際に起きた事象を検知できるようにするための実用的な近似）。"""
    started_at = result_summary.get("run_started_at")
    scheduled_hhmm = result_summary.get("scheduled_trigger_hhmm")
    if not started_at or not scheduled_hhmm:
        return False
    try:
        started = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        sched_h, sched_m = (int(x) for x in scheduled_hhmm.split(":"))
        scheduled = started.replace(hour=sched_h, minute=sched_m, second=0, microsecond=0)
        return (started - scheduled).total_seconds() / 60.0 > _TP50_MANUAL_DELAYED_THRESHOLD_MIN
    except (ValueError, TypeError):
        return False


def _classify_tp50_notification(result_summary: dict) -> str:
    """dry_run/error/warning/success のいずれかを判定する（優先順位: dry_run > error > warning > success）。
    2026-08-20追加: Scheduler未発火（手動遅延実行）検知時もwarningへ格上げする
    （9344インシデントの背景となった「本来SUCCESSに見えるが実は異常」を隠さないため）。"""
    if not result_summary.get("live"):
        return "dry_run"
    results = result_summary.get("order_submission_results")
    if results and any(not r.get("success") for r in results):
        return "error"
    if _data_integrity_issues(result_summary):
        return "error"
    fr = result_summary.get("fundamentals_freshness") or {}
    ca = result_summary.get("ca_guard") or {}
    blocked = (
        bool(fr.get("is_stale"))
        or bool(ca.get("buy_candidates_blocked_by_ca_pending"))
        or int(result_summary.get("buys_blocked_by_entry_freeze") or 0) > 0
        or (result_summary.get("risk_gate") or {}).get("recommendation") in ("CB_ACTIVE",)
        or _is_manual_delayed_run(result_summary)
        or bool(result_summary.get("metadata_warnings"))
        or bool(result_summary.get("asof_stale_block"))
    )
    return "warning" if blocked else "success"


# ── 表示ヘルパー ──────────────────────────────────────────────────────────
_EXIT_REASON_LABEL = {
    "trailing_gap_open": "T15トレーリングSTOP（寄付gap）",
    "trailing_touch": "T15トレーリングSTOP",
    "target_gap_open": "TP50利確（寄付gap）",
    "target_touch": "TP50利確",
}


def _exit_reason_category(exit_reason: str) -> str:
    if exit_reason in ("trailing_gap_open", "trailing_touch"):
        return "T15_TRAILING_STOP"
    if exit_reason in ("target_gap_open", "target_touch"):
        return "TP50_TARGET"
    return "OTHER"


def _new_board_source(client):
    """保有銘柄一覧のような複数銘柄の連続board取得で共有する
    KabuBoardSource（src.market_snapshot.source、既存の実運用実装・
    tests/market_snapshot/test_source.pyで独立検証済み）を1つ構築する。

    2026-08-22朝 通知監査で発覚: _render_holdings_table()が11銘柄のboard APIを
    銘柄間の待機なしで連続GETしており、実ログでHTTP 429を確認した
    （93440/94500）。この関数を介して常に同一のKabuBoardSourceインスタンスを
    ループ全体で共有することで、レート制限（5件/秒——CLAUDE.md記載の
    kabu API rate_limit=5/s、src/deployment/connectors/kabus_api_adapter.py
    の_RATE_LIMIT_CALLS_PER_SEC=5と同一根拠）・429時の追加backoff・
    有限retry（1回、計2試行）を全呼び出しに適用する。
    新規のスロットリング実装は作らず、既存の実運用クラスをそのまま再利用する
    （market_snapshot/source.pyへの変更は一切行っていない）。"""
    if client is None:
        return None
    from src.market_snapshot.source import KabuBoardSource
    return KabuBoardSource(client=client, rate_limit_per_sec=5.0, max_retries=1)


def _append_board_lookup_failure(
    code5: str, http_status: int | None, exc_type: str | None, exc_message: str | None,
    attempts: int, final_result: str,
) -> None:
    """board API取得失敗の追跡情報を追記専用サイドカーへ永続化する
    （2026-08-22朝の実インシデント再発防止: 標準出力をtailしていたために
    77810/78120の正確な失敗理由が事後確認できなかった。stdout/stderrの
    リダイレクト有無に関わらず必ずファイルへ残す）。レスポンス本文は保存しない
    （kabu APIのboard応答自体に認証情報等は含まれないが、将来の変更で
    混入するリスクを避けるため、例外メッセージも200文字に切り詰める）。"""
    try:
        record = {
            "symbol": code5, "http_status": http_status,
            "exception_type": exc_type, "exception_message": (exc_message or "")[:200],
            "attempts": attempts, "final_result": final_result,
            "recorded_at": datetime.now(_JST).isoformat(),
        }
        path = AUDIT_DIR / "board_lookup_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("[F4_TP50][BOARD_LOOKUP_FAILURE_LOG_FAILED] code=%s", code5)


def _lookup_symbol_display(client, code5: str, board_source=None) -> tuple[str, float | None, bool]:
    """(SymbolName, Price, price_is_live) をkabu API board(実勢気配)から取得する。
    client未指定・API失敗時は (code5, None, False) を返し、呼び出し側は"取得不可"
    表示にfallbackする。

    board_source（KabuBoardSource、_new_board_source()参照）を複数銘柄の
    連続取得元から共有で渡すことで、レート制限・429 backoff・有限retryが
    ループ全体に一貫して適用される。単発呼び出し（board_source省略時）は
    その場で1回分のインスタンスを作る——スロットリング対象がこの1呼び出し
    しかないため実害はない。

    2026-08-22 通知監査で判明: kabu API board応答のCurrentPriceは、その日まだ
    約定が一度も無い銘柄（薄商い銘柄等）では0/Noneになる（board自体の取得や
    気配値(Bid/Ask)は正常）。この場合、raw.PreviousClose（前営業日終値・実データ・
    捏造ではない）にfallbackし、price_is_live=Falseで呼び出し側に「現在値では
    ないこと」を明示させる。"""
    if client is None:
        return code5, None, False
    code4 = to_kabu_symbol(code5)
    source = board_source or _new_board_source(client)
    raw, status, body = source._fetch_one(code4)
    attempts = source._max_retries + 1
    if raw is None:
        _append_board_lookup_failure(
            code5, status, "HTTPError" if status else "Exception", body,
            attempts, "取得不可",
        )
        return code5, None, False
    name = raw.get("SymbolName") or code5
    current_price = raw.get("CurrentPrice")
    if current_price:
        return name, float(current_price), True
    prev_close = raw.get("PreviousClose")
    if prev_close:
        return name, float(prev_close), False
    _append_board_lookup_failure(code5, status, None, None, attempts, "現在値・前日終値ともに取得不可")
    return name, None, False


def _symbol_line(client, code5: str) -> str:
    """通知本文の「銘柄コード 銘柄名」行を作る。名前が取得できない場合はコードの
    重複表示（"93440 93440"のような無意味な行）を避け、コード単独で表示する。"""
    name, _, _ = _lookup_symbol_display(client, code5)
    return code5 if name == code5 else f"{code5} {name}"


def _lookup_actual_fill_price(client, order_id: str | None) -> float | None:
    if client is None or not order_id or order_id == "DRY_RUN_SIMULATED":
        return None
    fill = _fetch_actual_fill_details(client, order_id)
    return fill["avg_price"] if fill else None


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


def _fmt_signed_num(x, decimals: int = 1) -> str:
    if x is None:
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{decimals}f}"


def _fmt_score(score_map, code5: str) -> str:
    if not score_map:
        return "N/A"
    try:
        from src.f4_tp50 import score as f4_score
        v = f4_score.score_of(score_map, code5)
        if v is None or not np.isfinite(v):
            return "N/A"
        return f"{v:.1f}"
    except Exception:
        return "N/A"


def _render_holdings_table(positions: list[dict], client) -> list[str]:
    """保有銘柄一覧を1銘柄1行で描画する: コード/銘柄名/Score/Entry価格/現在値/損益率/数量。
    2026-08-22朝 通知全面仕様化: 現在ポートフォリオ・Previous Known Stateの両方で
    共有する（区別が必要な場合は呼び出し側が見出し・注意書きで明示する）。

    Score: 本日のcandidate cross-section score_map（compute_today_score_map）は
    「今日のBUY候補」だけをスコアする設計であり、既存保有銘柄は原理上ほぼ含まれない
    （2026-08-22朝の実インシデントで11銘柄全件がScore:N/Aになった真因）。
    保有銘柄のScoreはScore Replacementが記録するentry_score
    （runtime/f4_tp50/score_replacement_holdings.json、docs/f4_score_replacement/
    03_score_pit_contract.md sec.4により「Entry時点で固定・以後リフレッシュしない」
    PIT契約）をSource of Truthとする——別のスコア計算ロジックは新設しない。
    このファイルに記録の無い銘柄（Score Replacement Canary稼働開始前にEntryした
    銘柄等）はScore:N/Aのまま——推測しない。

    Entry価格はpositions[i]["entry_price"]（呼び出し側でbroker実約定metadataから
    構築済み——estimated_price/as_of由来の理論値を混ぜてはならない）をそのまま使い、
    ここでは一切の推測を行わない。現在値は毎回board取得を試みる
    （_lookup_symbol_display内で例外時のみ1回限定retry。前日終値fallback時は
    その旨を明記し、生の現在値と混同させない）。"""
    if not positions:
        return ["0銘柄"]
    from src.f4_tp50 import replacement_state as repl_state
    holding_scores = repl_state.load_holding_scores()
    board_source = _new_board_source(client)  # ループ全体で共有 → レート制限を全呼び出しに適用
    price_cache: dict[str, tuple[str, float | None, bool]] = {}  # 同一銘柄の重複board取得を防止
    lines = []
    for p in positions:
        code = p["code"]
        if code not in price_cache:
            price_cache[code] = _lookup_symbol_display(client, code, board_source=board_source)
        name, current_price, price_is_live = price_cache[code]
        name_part = code if name == code else f"{code} {name}"
        h_score = (holding_scores.get(code) or {}).get("entry_score")
        score_part = f"Score:{h_score:.1f}" if isinstance(h_score, (int, float)) else "Score:N/A"
        entry_price = p.get("entry_price")
        entry_part = f"Entry:{_fmt_yen(entry_price) if entry_price is not None else '取得不可'}"
        if current_price is not None:
            current_part = f"現在値:{_fmt_yen(current_price)}" + ("" if price_is_live else "(前日終値)")
        else:
            current_part = "現在値:取得不可"
        if entry_price is None:
            pnl_part = "損益:Entry価格取得不可のため計算不可"
        elif current_price is None:
            pnl_part = "損益:現在値取得不可のため計算不可"
        else:
            pnl_pct = (current_price / entry_price) - 1.0
            pnl_part = f"損益:{_fmt_pct(pnl_pct)}" + ("" if price_is_live else "(前日終値ベース)")
        qty_part = f"数量:{p.get('qty')}株"
        lines.append("  ".join([name_part, score_part, entry_part, current_part, pnl_part, qty_part]))
    return lines


def _order_for_symbol(results: list[dict] | None, symbol: str, side: str) -> dict | None:
    if not results:
        return None
    for r in results:
        if r.get("symbol") == symbol and r.get("side") == side:
            return r
    return None


_RUN_ID_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_\d{6}$")


def _run_summary_date(data: dict) -> str:
    """result_summaryの実行日を"YYYY-MM-DD"で返す。run_started_atを優先し、
    欠落時（2026-08-20 09:36手動遅延run等、run_started_atフィールド追加前の
    旧スキーマ記録）はrun_id（f4_tp50_YYYYMMDD_HHMMSS）から復元する。
    どちらも得られなければ空文字列（呼び出し側は「日付不明」として扱う）。"""
    started = (data.get("run_started_at") or "")[:10]
    if started:
        return started
    m = _RUN_ID_DATE_RE.search(data.get("run_id") or "")
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def _find_previous_live_run(exclude_run_id: str | None = None, before_date: str | None = None) -> dict | None:
    """LIVE_LOG_DIR(logs/live/)から、before_date（"YYYY-MM-DD"）より前の日付の
    最も新しい"live":true のresult_summaryを探す。
    Dry Run通知の【前日の実績】セクションのデータ源（2026-08-20 9344インシデント対応:
    「前日のkabu API実約定結果」を毎朝のDry Runで必ず確認できるようにする）。
    ファイル名 f4_tp50_f4_tp50_YYYYMMDD_HHMMSS.json はrun_id昇順=時系列昇順なので
    降順ソートの先頭が最新。見つからなければNone（初回実行時・記録なし等）。

    2026-08-21朝 通知監査で発見・修正: before_date未指定（従来仕様）では、Live実行
    直後にDry Runを手動実行すると「本日自身のLive run」が「前日の実績」として誤って
    表示される事故が発生した（本日Live 08:49実行→本日Dry Run 09:45実行時、前日の
    実績欄に本日の日付が出た）。before_dateで厳密に「当日より前」の記録のみに限定する。"""
    try:
        candidates = sorted(LIVE_LOG_DIR.glob("f4_tp50_f4_tp50_*.json"), reverse=True)
    except Exception as exc:
        logger.warning("[F4_TP50][PREV_RUN_LOOKUP] ディレクトリ走査失敗: %s", exc)
        return None
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("live") or data.get("run_id") == exclude_run_id:
            continue
        run_date = _run_summary_date(data)
        if before_date and run_date and run_date >= before_date:
            continue
        return data
    return None


def _render_sell_item(e: dict, order: dict | None, client, score_map, mode: str) -> list[str]:
    """mode: "today_preview"(Dry Runの本日の判断・未発注) /
             "actual_result"(Dry Runの前日の実績・broker実約定) /
             "order_slip"(Liveの発注書・約定価格は示さない)"""
    code = e["code"]
    entry_price = e.get("entry_price")
    category = _exit_reason_category(e["exit_reason"])
    reason_label = _EXIT_REASON_LABEL.get(e["exit_reason"], e["exit_reason"])
    order_id = order.get("order_id") if order else None

    lines = [_symbol_line(client, code), f"Score {_fmt_score(score_map, code)}", "", reason_label]
    lines.append(f"Entry       {_fmt_yen(entry_price)}")
    if category == "T15_TRAILING_STOP":
        lines.append(f"最高値      {_fmt_yen(e.get('highest_since_entry'))}")
        lines.append(f"STOP        {_fmt_yen(e.get('stop_level'))}")
    elif category == "TP50_TARGET":
        lines.append(f"Target      {_fmt_yen(e.get('target_price'))}")

    if mode == "actual_result":
        fill = _fetch_actual_fill_details(client, order_id) if client is not None and order_id else None
        fill_price = fill["avg_price"] if fill else None
        lines.append(f"約定価格    {_fmt_yen(fill_price)}")
        lines.append(f"数量        {e.get('qty')}株")
        lines.append("")
        if entry_price and fill_price:
            pnl = (fill_price - entry_price) * e.get("qty", 0)
            pnl_pct = (fill_price - entry_price) / entry_price
            lines.append(f"実現損益    {_fmt_yen(pnl, signed=True)}")
            lines.append(f"損益率      {_fmt_pct(pnl_pct)}")
        lines.append("")
        lines.append(f"注文ID      {order_id or 'N/A'}")
        if fill and fill.get("earliest_execution_timestamp"):
            lines.append(f"約定時刻    {fill['earliest_execution_timestamp'][11:19]}")
    elif mode == "order_slip":
        lines.append(f"判定価格    {_fmt_yen(e.get('exit_fill_price'))}")
        lines.append("")
        lines.append(f"数量        {e.get('qty')}株")
        lines.append("注文        成行SELL")
        lines.append(f"注文ID      {order_id or 'N/A'}")
        if order is not None and not order.get("success"):
            lines.append(f"⚠ 発注失敗  error={order.get('error')}")
    else:  # today_preview
        lines.append(f"判定価格    {_fmt_yen(e.get('exit_fill_price'))}")
        lines.append(f"数量        {e.get('qty')}株")
        lines.append("")
        lines.append("→ LIVEならSELL")
    lines.append("")
    return lines


def _render_buy_item(f: dict, order: dict | None, client, score_map, mode: str) -> list[str]:
    code = f["code"]
    order_id = order.get("order_id") if order else None
    lines = [_symbol_line(client, code), f"Score {_fmt_score(score_map, code)}", "", "新規Entry"]

    if mode == "actual_result":
        fill = _fetch_actual_fill_details(client, order_id) if client is not None and order_id else None
        fill_price = fill["avg_price"] if fill else None
        lines.append(f"約定価格    {_fmt_yen(fill_price)}")
        lines.append(f"数量        {FIXED_LOT_SIZE}株")
        lines.append(f"注文ID      {order_id or 'N/A'}")
        if fill and fill.get("earliest_execution_timestamp"):
            lines.append(f"約定時刻    {fill['earliest_execution_timestamp'][11:19]}")
    elif mode == "order_slip":
        lines.append(f"予定数量    {FIXED_LOT_SIZE}株")
        lines.append(f"基準価格    {_fmt_yen(f.get('estimated_fill_price'))}")
        lines.append("注文        成行BUY")
        lines.append(f"注文ID      {order_id or 'N/A'}")
        if order is not None and not order.get("success"):
            lines.append(f"⚠ 発注失敗  error={order.get('error')}")
    else:  # today_preview
        lines.append(f"予定数量      {FIXED_LOT_SIZE}株")
        lines.append(f"理論Entry価格 {_fmt_yen(f.get('estimated_fill_price'))}")
        lines.append("※BUY候補の価格は実約定価格ではない")
        lines.append("")
        lines.append("→ LIVEならBUY")
    lines.append("")
    return lines


def _render_replacement_item(d: dict, client, mode: str) -> list[str]:
    old_code = d.get("sold_code")
    new_code = d.get("candidate_code")
    old_score = d.get("holding_score")
    new_score = d.get("candidate_score")
    score_delta = d.get("score_delta")
    if score_delta is None and old_score is not None and new_score is not None:
        score_delta = new_score - old_score

    lines = ["SELL", _symbol_line(client, old_code),
             f"Score {old_score if old_score is not None else 'N/A'}"]
    if mode in ("actual_result", "order_slip") and d.get("sell_order_id"):
        lines.append(f"OrderID {d.get('sell_order_id')}")
    lines += ["", "↓", "", "BUY", _symbol_line(client, new_code),
              f"Score {new_score if new_score is not None else 'N/A'}"]
    if mode in ("actual_result", "order_slip") and d.get("buy_order_id"):
        lines.append(f"OrderID {d.get('buy_order_id')}")
    lines += ["", f"Score差     {_fmt_signed_num(score_delta)}"]
    if mode == "today_preview":
        lines.append("→ LIVEなら入替")
    lines.append("")
    return lines


def _build_previous_day_actuals_section(prev: dict, client, score_map) -> list[str]:
    """【前日の実績】: 直近のLIVE実行結果を、broker実約定(kabu API再取得)ベースで表示する。"""
    prev_date = _run_summary_date(prev).replace("-", "/")
    results = prev.get("order_submission_results") or []
    exits_detail = prev.get("exits_detail") or []
    funded_detail = prev.get("funded_detail") or []
    decisions = (prev.get("score_replacement") or {}).get("decisions") or []
    replace_decisions = [d for d in decisions if d.get("decision") in ("REPLACE_SIMULATED", "BUY_FILLED")]

    lines = ["【前日の実績】", prev_date or "(日付不明)", ""]
    lines.append(f"SELL       {len(exits_detail)}件")
    lines.append(f"BUY        {len(funded_detail)}件")
    lines.append(f"REPLACEMENT {len(replace_decisions)}件")
    lines.append("")
    lines.append("■ SELL")
    if not exits_detail:
        lines.append("0件")
    else:
        for e in exits_detail:
            order = _order_for_symbol(results, e["code"], "SELL")
            lines += _render_sell_item(e, order, client, score_map, mode="actual_result")
    lines.append("")
    lines.append("■ BUY")
    if not funded_detail:
        lines.append("0件")
    else:
        for f in funded_detail:
            order = _order_for_symbol(results, f["code"], "BUY")
            lines += _render_buy_item(f, order, client, score_map, mode="actual_result")
    lines.append("")
    lines.append("■ SCORE REPLACEMENT")
    if not replace_decisions:
        lines.append("0件")
    else:
        for d in replace_decisions:
            lines += _render_replacement_item(d, client, mode="actual_result")
    lines.append("")
    return lines


def _build_previous_day_unavailable_section() -> list[str]:
    """前日LIVE実行記録が見つからない場合の明示表示。無言でセクションを省略すると
    「前日データ取得済みだが取引がなかった(0件)」と「前日データそのものが存在しない」
    を区別できない（2026-08-21朝 通知監査: ユーザー指摘「0件とN/Aを区別」）。"""
    return ["【前日の実績】", "N/A（前回LIVE実行記録が見つかりません）", ""]


_CRITICAL_REPORT_FIELDS = ("available_cash", "positions_count", "cash_source")


def _missing_critical_fields(result_summary: dict) -> list[str]:
    """レポートの信頼性を左右する必須フィールドのうち、未取得(None/欠落)のものを返す。
    2026-08-21朝 通知監査で発見: Cash=N/A（available_cashがNone）でもMetadata=OKと
    表示される等、個別フィールドが取得失敗時に「異常」ではなく暗黙にOK/N/A表示へ
    フォールバックしていたため、レポート全体の信頼性を見た目からは判断できなかった。
    このチェックはレポート全体を DRY RUN INVALID として明示するためのゲート。
    注意: 実運用コードではavailable_cash/positions_count/cash_sourceは常に何らかの
    フォールバック値(0.0・"unavailable"等)が入りNoneにはならない
    （2026-08-22朝 実インシデント調査で判明——ここがNoneになるのは不正な
    テストfixtureのみ）。実際のbroker取得失敗を検知するには
    _data_integrity_issues()のcash_source判定を使うこと。"""
    return [f for f in _CRITICAL_REPORT_FIELDS if result_summary.get(f) is None]


def _data_integrity_issues(result_summary: dict) -> list[str]:
    """本日の判断・現在ポートフォリオの信頼性を損なう実運用上の問題を日本語で列挙する。
    空リスト=問題なし。DRY RUN/REPORT INVALID判定および【本日の判断】の
    「判定不能」表示・【現在ポートフォリオ】のPrevious Known Stateフォールバック
    表示、いずれもこの関数を単一の判定源とする（2026-08-22朝 通知監査）。"""
    issues: list[str] = []
    missing = _missing_critical_fields(result_summary)
    if missing:
        issues.append(f"必須フィールド欠落: {', '.join(missing)}")
    cash_source = result_summary.get("cash_source")
    if cash_source is not None and cash_source != "broker_live":
        issues.append(f"Kabu APIから現在ポートフォリオを取得できませんでした（cash_source={cash_source}）")
    if result_summary.get("fundamentals_freshness") is None:
        issues.append("Fundamentalsデータを取得できませんでした")
    return issues


def _build_system_block(result_summary: dict, live: bool) -> list[str]:
    fr = result_summary.get("fundamentals_freshness")
    results = result_summary.get("order_submission_results")
    lines = ["【SYSTEM】", ""]
    if fr is None:
        lines.append("Market Data       取得不可")
        lines.append("Fundamentals      取得不可")
    else:
        lines.append(f"Market Data       {'OK' if fr.get('is_stale') is False else '異常'}")
        lines.append(f"Fundamentals      {'OK' if fr.get('is_stale') is False else ('STALE' if fr.get('is_stale') else '取得不可')}")
    lines.append(f"Kabu API          {'OK' if result_summary.get('cash_source') == 'broker_live' else '異常/未接続'}")
    if results is not None:
        n_fail = sum(1 for r in results if not r.get("success"))
        lines.append(f"Order Submission  {'OK' if n_fail == 0 else f'異常（失敗{n_fail}件）'}")
    lines.append(f"Metadata          {'OK' if not result_summary.get('metadata_warnings') else '異常'}")
    lines.append(f"Scheduler         {'異常（手動遅延実行）' if _is_manual_delayed_run(result_summary) else 'OK'}")
    lines.append("")
    return lines


def _build_warnings_section(result_summary: dict) -> list[str]:
    """スケジューラ未発火・API異常・metadata mismatch等をヘッダ直下に目立たせる。"""
    fr = result_summary.get("fundamentals_freshness") or {}
    risk = result_summary.get("risk_gate") or {}
    results = result_summary.get("order_submission_results")
    warnings: list[str] = []
    if _is_manual_delayed_run(result_summary):
        warnings.append(f"{result_summary.get('scheduled_trigger_hhmm')} 自動実行されず")
        started_hhmm = (result_summary.get("run_started_at") or "")[11:16]
        warnings.append(f"{started_hhmm} 手動遅延実行")
    if result_summary.get("cash_source") not in ("broker_live", None) and result_summary.get("live"):
        warnings.append(f"API接続異常: cash_source={result_summary.get('cash_source')}")
    if fr.get("is_stale"):
        warnings.append(f"ファンダメンタルズ鮮度異常: {fr.get('reason')}")
    if result_summary.get("asof_stale_block"):
        warnings.append(
            f"市場データ鮮度異常: as_ofが実カレンダー日から{result_summary.get('asof_staleness_bdays')}"
            f"営業日乖離 — 新規BUYをブロック（Exit/リスク管理は継続）"
        )
    ca_blocked = (result_summary.get("ca_guard") or {}).get("buy_candidates_blocked_by_ca_pending") or []
    if ca_blocked:
        warnings.append(f"CA_PENDING銘柄のためBUY除外: {ca_blocked}")
    if risk.get("recommendation") in ("CB_ACTIVE",):
        warnings.append(f"Circuit Breaker発動中: {risk.get('message')}")
    if results and any(not r.get("success") for r in results):
        failed = [f"{r.get('symbol')}({r.get('error')})" for r in results if not r.get("success")]
        warnings.append(f"発注失敗: {failed}")
    warnings += list(result_summary.get("metadata_warnings") or [])
    if not warnings:
        return []
    lines = ["【警告】"]
    lines += warnings
    lines.append("")
    return lines


def _build_dry_run_notification_body(result_summary: dict, client, score_map) -> str:
    sep = "━" * 22
    today_date = (result_summary.get("run_started_at") or "")[:10].replace("-", "/")
    today_time = (result_summary.get("run_started_at") or "")[11:16]
    exits_detail = result_summary.get("exits_detail") or []
    funded_detail = result_summary.get("funded_detail") or []
    decisions = (result_summary.get("score_replacement") or {}).get("decisions") or []
    replace_decisions = [d for d in decisions if d.get("decision") in ("REPLACE_SIMULATED", "BUY_FILLED")]
    risk = result_summary.get("risk_gate") or {}

    lines = [sep, "CHIBA F4 TP50", "DAILY DRY RUN", f"{today_date} {today_time}", sep, ""]

    issues = _data_integrity_issues(result_summary)
    if issues:
        lines.append("⚠⚠⚠ DRY RUN INVALID ⚠⚠⚠")
        lines.append("データ取得失敗のため本日の判断は信頼できません:")
        for issue in issues:
            lines.append(f"  ・{issue}")
        lines.append("")

    lines += _build_warnings_section(result_summary)

    prev = _find_previous_live_run(
        exclude_run_id=result_summary.get("run_id"),
        before_date=(result_summary.get("run_started_at") or "")[:10],
    )
    if prev is not None:
        lines += _build_previous_day_actuals_section(prev, client, score_map)
    else:
        lines += _build_previous_day_unavailable_section()

    lines.append(sep)
    lines.append("【本日の判断】")
    lines.append(today_date)
    lines.append(sep)
    lines.append("")
    if issues:
        lines.append("実行結果：INVALID")
        lines.append("")
        lines.append("SELL        判定不能")
        lines.append("BUY         判定不能")
        lines.append("REPLACEMENT 判定不能")
        lines.append("")
        lines.append("※データ取得失敗のため売買判断を採用しません")
    else:
        lines.append(f"SELL       {len(exits_detail)}")
        lines.append(f"BUY        {len(funded_detail)}")
        lines.append(f"REPLACEMENT {len(replace_decisions)}")
        lines.append("")

        lines.append("■ SELL")
        lines.append("")
        if not exits_detail:
            lines.append("0件")
        else:
            for e in exits_detail:
                lines += _render_sell_item(e, None, client, score_map, mode="today_preview")
        lines.append("")
        lines.append("■ BUY")
        lines.append("")
        if not funded_detail:
            lines.append("0件")
        else:
            for f in funded_detail:
                lines += _render_buy_item(f, None, client, score_map, mode="today_preview")
        lines.append("")
        lines.append("■ SCORE REPLACEMENT")
        lines.append("")
        if not replace_decisions:
            lines.append("0件")
        else:
            for d in replace_decisions:
                lines += _render_replacement_item(d, client, mode="today_preview")
    lines.append("")

    lines.append(sep)
    lines.append("【現在ポートフォリオ】")
    lines.append(sep)
    lines.append("")
    if result_summary.get("cash_source") == "broker_live":
        lines.append(f"Cash        {_fmt_yen(result_summary.get('available_cash'))}")
        lines.append(f"評価額      {_fmt_yen(result_summary.get('market_value'))}")
        lines.append(f"総資産      {_fmt_yen(result_summary.get('last_equity'))}")
        lines.append(f"DD          {_fmt_pct(risk.get('dd'))}")
        lines.append(f"保有銘柄    {result_summary.get('positions_count')}件")
        lines.append("")
        lines += _render_holdings_table(result_summary.get("current_holdings") or [], client)
    else:
        lines.append("⚠ Kabu APIから現在の証券口座状態を取得できませんでした")
        lines.append("")
        lines.append("Current Portfolio: UNAVAILABLE")
        lines.append("")
        lines.append("【Previous Known State】")
        lines.append("以下は前回確定状態（参考情報）です。")
        lines.append("現在の実際の口座残高を保証するものではありません。")
        lines.append("")
        prev_positions = result_summary.get("previous_known_positions") or []
        lines += _render_holdings_table(prev_positions, client)
        if prev_positions:
            lines.append(f"合計 {len(prev_positions)}銘柄")
    lines.append("")

    lines.append(sep)
    lines += _build_system_block(result_summary, live=False)

    lines.append(sep)
    lines.append("※本日の判断はDRY RUNです")
    lines.append("※本日の注文は発注していません")
    lines.append(sep)
    return "\n".join(lines)


def _build_live_notification_body(result_summary: dict, client, score_map) -> str:
    sep = "━" * 22
    today_date = (result_summary.get("run_started_at") or "")[:10].replace("-", "/")
    today_time = (result_summary.get("run_started_at") or "")[11:16]
    results = result_summary.get("order_submission_results")
    exits_detail = result_summary.get("exits_detail") or []
    funded_detail = result_summary.get("funded_detail") or []
    decisions = (result_summary.get("score_replacement") or {}).get("decisions") or []
    replace_decisions = [d for d in decisions if d.get("decision") in ("REPLACE_SIMULATED", "BUY_FILLED")]
    kind = _classify_tp50_notification(result_summary)
    status_label = {"success": "SUCCESS", "warning": "WARNING", "error": "ERROR"}.get(kind, kind.upper())

    lines = [sep, "CHIBA F4 TP50", "LIVE ORDER REPORT", f"{today_date} {today_time}", sep, ""]

    issues = _data_integrity_issues(result_summary)
    if issues:
        lines.append("⚠⚠⚠ REPORT INVALID ⚠⚠⚠")
        lines.append("データ取得失敗のためこのレポート内容は信頼できません:")
        for issue in issues:
            lines.append(f"  ・{issue}")
        lines.append("")

    lines += _build_warnings_section(result_summary)

    lines.append("【ORDER RESULT】")
    lines.append("")
    lines.append(f"SELL          {len(exits_detail)}")
    lines.append(f"BUY           {len(funded_detail)}")
    lines.append(f"REPLACEMENT   {len(replace_decisions)}")
    lines.append("")
    lines.append(f"STATUS        {status_label}")
    lines.append("")

    if exits_detail:
        lines.append(sep)
        lines.append("【SELL ORDER】")
        lines.append(sep)
        lines.append("")
        for e in exits_detail:
            order = _order_for_symbol(results, e["code"], "SELL")
            lines += _render_sell_item(e, order, client, score_map, mode="order_slip")

    if funded_detail:
        lines.append(sep)
        lines.append("【BUY ORDER】")
        lines.append(sep)
        lines.append("")
        for f in funded_detail:
            order = _order_for_symbol(results, f["code"], "BUY")
            lines += _render_buy_item(f, order, client, score_map, mode="order_slip")

    if replace_decisions:
        lines.append(sep)
        lines.append("【SCORE REPLACEMENT】")
        lines.append(sep)
        lines.append("")
        for d in replace_decisions:
            lines += _render_replacement_item(d, client, mode="order_slip")

    lines.append(sep)
    lines += _build_system_block(result_summary, live=True)

    lines.append(sep)
    lines.append("※約定価格は注文時点では未確定")
    lines.append("※翌営業日のDRY RUNで実約定結果を確認します")
    lines.append(sep)
    return "\n".join(lines)


def _build_tp50_notification_body(
    result_summary: dict, buy_orders_intended: list, sell_orders_intended: list,
    client=None, score_map=None,
) -> str:
    """
    通知本文の入口。2026-08-20 全面刷新（9344誤売却事故対応）:
    DRY RUNは「前日の実績（broker実約定）」+「本日の判断」の2部構成、
    LIVEは「発注書」（約定価格は翌日のDry Runで確認する設計・注文時点では未確定のため
    本文には出さない）——役割を完全分離する。buy_orders_intended/sell_orders_intendedは
    後方互換のため引数として残すが、本文生成にはresult_summary["exits_detail"]/
    ["funded_detail"]（entry_price/highest/stop/target込みの詳細データ）を使う。
    """
    if bool(result_summary.get("live")):
        return _build_live_notification_body(result_summary, client, score_map)
    return _build_dry_run_notification_body(result_summary, client, score_map)


def _send_tp50_notification(
    result_summary: dict, buy_orders_intended: list, sell_orders_intended: list,
    client=None, score_map=None,
) -> None:
    """
    Fire-and-forget（既存src.notifier再利用のみ・新規SMTP実装なし）。通知処理内の例外は必ず
    ここで握りつぶし、main()の戻り値・scheduler taskの成否には一切影響させない。
    notifier.py自体は変更しない・subject文言はsubject_suffixでのみ調整する
    （notify_*()のsubject prefixは"✅ CHIBA 発注完了"等の固定形式のため、要求された
    "[TP50][SUCCESS] ..."はsubject_suffixとして末尾に付与する — notifier.py本体を
    変更しない制約との両立）。
    """
    try:
        from src.notifier import notify_dry_run, notify_error, notify_success, notify_warning, wait_pending

        kind = _classify_tp50_notification(result_summary)
        body = _build_tp50_notification_body(
            result_summary, buy_orders_intended, sell_orders_intended, client=client, score_map=score_map,
        )
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
                # 2026-08-20 通知監査証跡強化: entry_price(broker実約定ベース、
                # position_entry_pricesから取得済みの値をそのまま伝播)を通知本文の
                # 「Entry」表示に使う。追加フィールドのみで既存consumerへの影響なし。
                "entry_price": float(entry_price),
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


def _fetch_actual_fill_details(client, order_id: str) -> dict | None:
    """
    Queries kabu API GET /orders for the confirmed broker execution of
    order_id — the SOLE Source of Truth for F4 TP50 entry metadata (never
    as_of/estimated_price/signal_date; see apply_fill_metadata_updates()
    docstring and docs/research/2026-08-20_f4_tp50_9344_position_metadata_
    incident_and_audit.md).

    Aggregates ALL RecType=8 (約定明細) detail records for the order — kabu
    API can return a single BUY as multiple partial executions — into:
        filled_qty:             sum of all execution quantities
        avg_price:              quantity-weighted average execution price
        earliest_execution_date: "YYYY-MM-DD" of the first execution
        execution_count:        number of individual fills aggregated

    Returns None if the order isn't found, has no execution detail yet, or
    the API call fails. Never raises — callers must fail closed on None
    rather than substitute an estimate.
    """
    try:
        orders = client.get_orders(only_open=False)
    except Exception as exc:
        logger.warning("[F4_TP50][FILL_LOOKUP] get_orders失敗 order_id=%s: %s", order_id, exc)
        return None
    for o in orders:
        if o.get("ID") != order_id:
            continue
        fills = []
        for d in o.get("Details", []) or []:
            if d.get("RecType") != 8:
                continue
            price = d.get("Price")
            qty = d.get("Qty")
            exec_day = d.get("ExecutionDay")
            if not (price and qty and exec_day) or float(price) <= 0.0 or float(qty) <= 0.0:
                continue
            fills.append({"price": float(price), "qty": float(qty), "execution_day": str(exec_day)})
        if not fills:
            return None
        total_qty = sum(f["qty"] for f in fills)
        weighted_price = sum(f["price"] * f["qty"] for f in fills) / total_qty
        earliest_ts = min(f["execution_day"] for f in fills)
        return {
            "filled_qty": total_qty,
            "avg_price": weighted_price,
            "earliest_execution_date": earliest_ts[:10],
            "earliest_execution_timestamp": earliest_ts,
            "execution_count": len(fills),
        }
    return None


def _validate_fill_sanity(exec_date_str: str, exec_price: float) -> list[str]:
    """
    Lightweight sanity checks on a confirmed broker fill before it is trusted
    as entry metadata. Not a market-data validator — just catches API/parsing
    corruption (e.g. a nonsense future date, a zero/negative price slipping
    through). Returns a list of problem descriptions (empty = OK).
    """
    problems: list[str] = []
    if exec_price is None or exec_price <= 0.0:
        problems.append(f"exec_price<=0: {exec_price}")
    try:
        exec_dt = datetime.strptime(exec_date_str, "%Y-%m-%d").date()
        today = datetime.now(_JST).date()
        if exec_dt > today:
            problems.append(f"exec_date is in the future: {exec_date_str} > {today}")
    except (ValueError, TypeError):
        problems.append(f"unparseable exec_date: {exec_date_str!r}")
    return problems


def _append_entry_fill_audit(
    symbol: str, order_id: str | None, estimated_price: float | None,
    as_of: pd.Timestamp, fill: dict,
) -> None:
    """
    SSOT audit trail (Phase2 design, 2026-08-20 9344インシデント対応):
    theoretical(signal-side: as_of/estimated_price) と actual(broker fill) を
    同一レコード内で明示的に分離して記録する。将来の監査(本インシデントの
    ような「stored値とbroker実約定の突合」)をログの断片から再構築する必要が
    ないようにするための追記専用サイドカー。portfolio_state.jsonのスキーマは
    一切変更しない（既存フィールドはactual値のみを保持し続ける）。
    """
    try:
        record = {
            "symbol": symbol,
            "broker_order_id": order_id,
            "theoretical": {"signal_as_of": as_of.strftime("%Y-%m-%d"), "estimated_price": estimated_price},
            "actual": fill,
            "recorded_at": datetime.now(_JST).isoformat(),
        }
        path = AUDIT_DIR / "entry_fill_audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("[F4_TP50][AUDIT_WRITE_FAILED] symbol=%s: %s", symbol, exc)


def apply_fill_metadata_updates(
    results: list[dict],
    entry_dates: dict[str, str],
    entry_prices: dict[str, float],
    strategy_types: dict[str, str],
    as_of: pd.Timestamp,
    client=None,
    warnings_sink: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, float], dict[str, str], bool]:
    """
    Entry-metadata persistence. Only touches EXISTING portfolio_state.json
    schema fields (position_entry_dates/position_entry_prices/
    position_strategy_types) — no new top-level keys are introduced.

    On a successful BUY fill: records entry_date/entry_price from the actual
    kabu API broker execution (Details[] RecType=8 records for the order's
    order_id, aggregated across partial fills via _fetch_actual_fill_details()),
    NOT from as_of/estimated_price (those are signal-theoretical values
    computed before submission and can diverge from the real fill date/price
    whenever the order executes later than the signal's assumed as_of — see
    2026-08-20 9344 incident: a delayed fill was recorded under the signal's
    stale as_of/estimated_price, which pulled a pre-entry OHLC bar into
    highest_since_entry and caused a spurious trailing-stop SELL the next
    day). estimated_price/as_of are never written into position_entry_dates/
    position_entry_prices under any code path in this function — the broker
    fill (via _fetch_actual_fill_details) is the sole source for both.

    FAIL CLOSED — entry metadata for a symbol is NOT written this run
    (logged as METADATA_FAIL_CLOSED, so the next run retries since the symbol
    still won't be in entry_dates) when any of:
      - the actual fill cannot be confirmed via the broker (API error, order
        not found, no execution detail yet);
      - the order is only PARTIALLY filled (aggregated Details qty < the
        order's requested qty) — a partial fill is not yet a confirmed
        FIXED_LOT_SIZE entry;
      - the confirmed fill fails a basic sanity check (_validate_fill_sanity:
        non-positive price, execution date in the future/unparseable).
    In every case, estimated_price/as_of are never substituted.

    On every successful, fully-confirmed BUY, an audit record pairing the
    theoretical (as_of/estimated_price) and actual (broker fill) values is
    appended to runtime/f4_tp50/entry_fill_audit.jsonl — an additive sidecar,
    not a portfolio_state.json schema change — so theoretical and actual
    values are never conflated and remain independently reconstructable.

    Only writes if not already present, so a retry/duplicate result never
    clobbers the true original entry (this also makes re-processing the same
    order_id across runs idempotent: the second run finds the symbol already
    populated and skips straight to the strategy_type check). On a successful
    SELL fill: removes all three fields for that symbol. Failed orders
    (success=False) never touch metadata.

    warnings_sink: optional list. When provided, every FAIL CLOSED event
    appends a short human-readable string (for surfacing in the notification
    email's 【警告】block — 2026-08-20, so "metadata mismatch/unconfirmed" is
    visible to a human within the run's own notification, not just logs).

    Returns (new_entry_dates, new_entry_prices, new_strategy_types, changed).
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
            if sym not in entry_dates or sym not in entry_prices:
                order_id = r.get("order_id")
                fill = (
                    _fetch_actual_fill_details(client, order_id) if client is not None and order_id else None
                )
                if fill is None:
                    logger.error(
                        "[F4_TP50][METADATA_FAIL_CLOSED] BUY約定確認不能 symbol=%s order_id=%s — "
                        "entry_date/entry_priceを推測値(estimated_price/as_of)で代用せず、"
                        "今回はmetadata未記録（次回run再試行）。",
                        sym, order_id,
                    )
                    if warnings_sink is not None:
                        warnings_sink.append(f"metadata mismatch: {sym} 約定確認不能（order_id={order_id}）")
                else:
                    expected_qty = float(r.get("qty") or 0.0)
                    if expected_qty and fill["filled_qty"] != expected_qty:
                        _fill_relation = "部分約定" if fill["filled_qty"] < expected_qty else "約定数量超過"
                        logger.error(
                            "[F4_TP50][METADATA_FAIL_CLOSED] QTY MISMATCH(%s) symbol=%s order_id=%s "
                            "filled=%s/%s — 数量一致確認までentry metadataを保留（次回run再試行）。",
                            _fill_relation, sym, order_id, fill["filled_qty"], expected_qty,
                        )
                        if warnings_sink is not None:
                            warnings_sink.append(
                                f"metadata mismatch: {sym} {_fill_relation}（{fill['filled_qty']}/{expected_qty}株）"
                            )
                    else:
                        problems = _validate_fill_sanity(fill["earliest_execution_date"], fill["avg_price"])
                        if problems:
                            logger.error(
                                "[F4_TP50][METADATA_FAIL_CLOSED] VALIDATION symbol=%s order_id=%s "
                                "problems=%s — entry metadata未記録。",
                                sym, order_id, problems,
                            )
                            if warnings_sink is not None:
                                warnings_sink.append(f"metadata mismatch: {sym} 約定データ異常（{problems}）")
                        else:
                            if sym not in entry_dates:
                                entry_dates[sym] = fill["earliest_execution_date"]
                                changed = True
                            if sym not in entry_prices:
                                entry_prices[sym] = fill["avg_price"]
                                changed = True
                            _append_entry_fill_audit(sym, order_id, r.get("estimated_price"), as_of, fill)
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

    run_started_at = datetime.now(_JST)
    run_id = f"f4_tp50_{run_started_at.strftime('%Y%m%d_%H%M%S')}"
    audit_log: list[dict] = []

    print("[LOAD] 市場データ・PITファンダメンタル読込中（数分かかります）...")
    data = load_live_data()
    as_of = data.calendar[-1]
    print(f"  最新営業日={as_of.date()}  銘柄数={len(data.codes)}")

    # ── Signal Freshness Guard（2026-08-20 9344インシデント再発防止、同日夜 Production Gate監査で強化）──
    # as_of(=最新読込済み営業日)が実カレンダー日から大きく乖離している場合の二段階ガード。
    # 「前営業日のシグナルを翌営業日に発注する」こと自体は正当なPIT運用のため、通常の
    # 週末/祝日を挟む1-2営業日の乖離では無条件にブロックしない。ただし、market data
    # パイプライン自体が数営業日単位で停止しているような異常乖離は、たとえentry
    # metadataがbroker実約定ベース(Phase5修正、apply_fill_metadata_updates()参照)で
    # 汚染耐性を持つようになった後も、「著しく古い市場データに基づく新規Entry判断」
    # という別種のリスク（価格が実勢から乖離した状態でのBUY）を防げないため、
    # BUYのみblock（Exit/リスク管理は常に通常通り動作させる — freshness.is_staleと
    # 同一方針）。np.busday_countはcheck_fundamentals_freshness()と同一手法。
    _asof_staleness_bdays = compute_asof_staleness_bdays(as_of.date(), datetime.now(_JST).date())
    asof_stale_block = should_block_buy_for_stale_asof(_asof_staleness_bdays)
    if _asof_staleness_bdays > MAX_ASOF_STALENESS_BDAYS_WARN:
        logger.warning(
            "[F4_TP50][SIGNAL_FRESHNESS] as_of(最新営業日)=%s が実カレンダー日=%sから%d営業日乖離"
            "（block閾値=%d営業日、現在%s）。entry metadataはbroker実約定からのみ記録されるため"
            "metadata汚染リスクは無いが、市場データ自体の陳腐化リスクは別途残る。",
            as_of.date(), datetime.now(_JST).date(), _asof_staleness_bdays,
            MAX_ASOF_STALENESS_BDAYS_BLOCK, "BLOCK" if asof_stale_block else "WARN継続",
        )
        print(f"[SIGNAL_FRESHNESS][{'BLOCK' if asof_stale_block else 'WARNING'}] "
              f"as_of={as_of.date()} 実カレンダー日={datetime.now(_JST).date()} "
              f"staleness={_asof_staleness_bdays}営業日"
              + ("（新規BUYをブロック・Exit/リスク管理は継続）" if asof_stale_block
                 else "（発注は継続・entry metadataはbroker実約定ベースのため汚染リスクなし）"))

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
    # 前回確定ポートフォリオ（ローカルportfolio_state.json由来・broker API不要）。
    # Kabu API障害でbroker snapshotが取得できない日でも、直近に記録されたTP50保有
    # 銘柄の一覧だけは提示できる（2026-08-22朝 通知監査: broker snapshot失敗時に
    # 保有銘柄情報が完全に消える問題への対応）。「現在の実際の残高」ではなく
    # 「前回確定した参考情報」であることを通知本文側で必ず明示する。
    previous_known_positions = sorted(
        (
            {"code": code, "entry_date": entry_dates.get(code), "entry_price": entry_prices.get(code),
             "qty": FIXED_LOT_SIZE}
            for code, stype in strategy_types.items() if stype == STRATEGY_TYPE
        ),
        key=lambda p: p["code"],
    )
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
            # 2026-08-20追加: この早期abort経路は従来notificationを一切送っておらず、
            # 「メールが来ない＝異常」を発見する手段が無かった。result_summary構築前の
            # 最小限の情報でも必ず1通送る（notifier例外はここでも握りつぶす）。
            try:
                from src.notifier import notify_error, wait_pending
                notify_error(
                    f"CHIBA F4 TP50 — EMERGENCY_STOP\n\n"
                    f"broker snapshot取得失敗のため--live実行を中断しました。\n"
                    f"run_id: {run_id}\n実行時刻: {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"error: {exc}\n\n"
                    f"kabuステーション起動・ログイン状態・APIパスワードを確認してください。",
                    subject_suffix="[TP50][ERROR] EMERGENCY_STOP api_unreachable",
                )
                wait_pending(timeout=15.0)
            except Exception:
                logger.warning("[F4_TP50][NOTIFY] EMERGENCY_STOP通知失敗: %s", traceback.format_exc(limit=3))
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
    tp50_held_qty = held_qty_by_internal_key(broker_positions, strategy_types, STRATEGY_TYPE)
    already_held_symbols = set(tp50_held_qty.keys())

    # 現在ポートフォリオ（broker snapshot取得成功時のみ・"何を保有しているか"の
    # Source of Truthはbroker実ポジション。entry_price/entry_dateはbroker実約定
    # ベースのローカルmetadata（apply_fill_metadata_updates()参照）——estimated_price
    # やas_of由来の理論値は絶対に使わない（2026-08-22朝 通知全面仕様化: 9344事故の
    # 教訓の再発防止をDaily Report全体に適用）。現在値・Scoreは通知本文生成時に
    # client/score_mapで解決する（board値は実行時点のスナップショットのため
    # ログJSONへは永続化しない）。
    current_holdings = sorted(
        (
            {"code": code, "entry_date": entry_dates.get(code), "entry_price": entry_prices.get(code),
             "qty": qty}
            for code, qty in tp50_held_qty.items()
        ),
        key=lambda p: p["code"],
    )

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
    elif asof_stale_block:
        print(f"[GUARD] Signal Freshness(as_of {_asof_staleness_bdays}営業日乖離 > "
              f"{MAX_ASOF_STALENESS_BDAYS_BLOCK}営業日) — BUY候補 {len(candidates_raw)}件を全ブロック"
              "（市場データが著しく陳腐化・Exit/リスク管理は継続）")
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
        # 2026-08-20 通知監査証跡強化（9344誤売却事故対応）: メール本文生成に必要な
        # 詳細情報。exits/fundedはevaluate_exits()/evaluate_buy_sizing()が返す
        # 追加フィールド込みの生データ（entry_price/stop_level/target_price/
        # highest_since_entry等）をそのまま保持する。
        "exits_detail": exits,
        "funded_detail": funded,
        "available_cash": available_cash,
        "last_equity": last_equity,
        "market_value": last_equity - available_cash,
        "positions_count": len(broker_positions),
        "previous_known_positions": previous_known_positions,
        "current_holdings": current_holdings,
        "run_started_at": run_started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_trigger_hhmm": _TP50_SCHEDULED_TRIGGER_HHMM,
        "asof_staleness_bdays": _asof_staleness_bdays,
        "asof_stale_block": asof_stale_block,
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
        _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended, client=client, score_map=score_map)
        return 0

    # ── --live 経路（entry_freeze/staleness中はBUYが事前に除外済み） ──
    if not orders_to_submit:
        print("[LIVE] 送信対象の注文なし。")
        _try_generate_reports()
        _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended, client=client, score_map=score_map)
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
        _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended, client=client, score_map=score_map)
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
    # metadata_warnings: 通知【警告】ブロックに表示するfail-closedイベントの収集先
    # （2026-08-20、metadata mismatch可視化）。
    metadata_warnings: list[str] = []
    entry_dates, entry_prices, strategy_types, metadata_changed = apply_fill_metadata_updates(
        results, entry_dates, entry_prices, strategy_types, as_of, client=client,
        warnings_sink=metadata_warnings,
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
            replacement_synthetic_results, entry_dates, entry_prices, strategy_types, as_of, client=client,
            warnings_sink=metadata_warnings,
        )
        metadata_changed = metadata_changed or repl_changed

    result_summary["metadata_warnings"] = metadata_warnings

    if metadata_changed:
        state["position_entry_dates"] = entry_dates
        state["position_entry_prices"] = entry_prices
        state["position_strategy_types"] = strategy_types
        save_portfolio_state(state, data_source="f4_tp50_live_fill_metadata")
        print("[STATE] TP50 entry metadata (position_entry_dates/position_entry_prices/"
              "position_strategy_types) を更新しました。")

    _send_tp50_notification(result_summary, buy_orders_intended, sell_orders_intended, client=client, score_map=score_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
