"""
kabusapi/signal_bridge.py
バックテスト戦略 → kabuステーションAPI 発注ブリッジ（Top-k ローテーション対応）

【処理フロー（毎朝 8:30 頃に実行）】
  1. yfinance で前日終値までのデータを取得
  2. RSR 上位 top_k 銘柄を選出（流動性フィルター付き）
  3. kabuステーション API で現在ポジション・余力を取得
  4. CB 状態機械を評価（NORMAL / CB_ACTIVE / RECOVERY）
  5. シグナル生成: top_k 内なら FujikoStrategy、圏外保有なら rank_exit
  6. 時間ストップ: max_hold_days 営業日超過で SELL
  7. 注文生成 → ドライランまたは実発注

【安全設計】
  - CB_ACTIVE 中は新規 BUY を全停止（SELL のみ実行）
  - max_new_positions_per_day = 2（1回の実行で BUY は最大 2 件）
  - order_rate_limit: kabu API に 20 秒/件でレート制限
  - デフォルトはドライラン（--live なしでは発注しない）
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

_RUN_ID = datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def _log_equity_peak_update(
    old_peak: float, new_peak: float, current_equity: float, caller: str, reason: str,
) -> None:
    """equity_peak 更新を必ず記録する（監査用・状態変更なし）。"""
    logger.warning(
        "[EQUITY_PEAK_UPDATE] old_peak=¥%s new_peak=¥%s current_equity=¥%s "
        "caller=%s reason=%s pid=%d run_id=%s timestamp=%s",
        f"{old_peak:,.0f}", f"{new_peak:,.0f}", f"{current_equity:,.0f}",
        caller, reason, os.getpid(), _RUN_ID,
        datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


class EquityPeakInvariantError(RuntimeError):
    """equity_peak 更新が不変条件（Study96 Phase5）に違反した場合に送出する。
    保存前に検知して run() を中断させ、破損値の永続化を防ぐ（fail-closed）。"""


def _commit_equity_peak(
    state:                 dict,
    candidate_value:       float,
    current_equity:        float,
    *,
    caller:                str,
    reason:                str,
    broker_snapshot:       "BrokerSnapshot | None",
    today_str:             str,
    mode:                  str,
    bypass_candidate_gate: bool = False,
) -> float:
    """
    equity_peak の唯一の書き込み経路（EQUITY_PEAK_HARDENING, 2026-07-03 /
    Study96 EquityPeak SSOT Root Cause Audit, 2026-07-17 で不変条件assert +
    多段階再確認を追加。原因推定（入出金等）ロジックは持ち込まない — 現在の
    証券口座equityのみをSSOTとし、異常値は理由を問わず一律に拒否/保留する）。

    _update_cb_state() 以外からの呼び出しは RuntimeError で拒否する
    （sync_positions.py 等の迂回書込みによる無検証 peak 混入事故の再発防止）。

    処理順:
      1. check_broker_consistency() で broker 生値との乖離を検証。
         不整合なら EQUITY_PEAK_REJECT（state は不変）。
      2. 書込み直前の不変条件assert（peak >= equity・new_highはold_peak超のみ）。
      3. bypass_candidate_gate=False かつ 前回peak比 +10% 以上のジャンプなら
         state["candidate_peak"] へステージングし、即時採用しない。
      4. それ以外は state["equity_peak"] を確定書き込みする。

    全分岐で _log_equity_peak_update()（logger監査）と append_peak_audit()
    （durable JSONL監査）を必ず呼ぶ。

    Returns:
        float: この呼び出し後に有効な equity_peak
               （REJECTED/STAGED 時は変更前の値のまま）

    Raises:
        EquityPeakInvariantError: 書込み直前に candidate_value < current_equity
            （peak は定義上 equity 以上でなければならない）が成立した場合。
            state は一切変更されず、呼び出し元（run()）まで伝播して該当run全体を
            中断させることで、不正値の save_portfolio_state() 永続化を防ぐ。
    """
    _frame = sys._getframe(1)
    if _frame.f_code.co_name != "_update_cb_state":
        raise RuntimeError(
            f"EQUITY_PEAK_FORBIDDEN_WRITE: equity_peak を書き込めるのは "
            f"_update_cb_state() のみ。caller={_frame.f_code.co_name} "
            f"({_frame.f_code.co_filename}:{_frame.f_lineno})"
        )

    old_peak = float(state.get("equity_peak", 0.0))

    _audit_common = dict(
        old_peak       = old_peak,
        new_peak       = candidate_value,
        current_equity = current_equity,
        caller         = caller,
        reason         = reason,
        trading_date   = today_str,
        mode           = mode,
        pid            = os.getpid(),
        run_id         = _RUN_ID,
    )

    is_consistent, broker_equity, diag = check_broker_consistency(current_equity, broker_snapshot)
    if not is_consistent:
        _log_equity_peak_update(
            old_peak, candidate_value, current_equity,
            caller=caller, reason="EQUITY_PEAK_REJECT",
        )
        append_peak_audit(action="REJECTED", broker_equity=broker_equity, diag=diag, **_audit_common)
        return old_peak

    # ── Study96 Phase5: 書込み直前の不変条件assert（fail-closed） ──────────
    # peak は定義上 equity 以上でなければならない。違反時は state を一切変更せず
    # 例外を送出し、run() 全体を中断させて save_portfolio_state() への到達を防ぐ。
    if candidate_value < current_equity - 1.0:
        raise EquityPeakInvariantError(
            f"EQUITY_PEAK_INVARIANT_VIOLATION: candidate_value=¥{candidate_value:,.0f} "
            f"< current_equity=¥{current_equity:,.0f} (peak must be >= equity at commit time). "
            f"caller={caller} reason={reason} old_peak=¥{old_peak:,.0f}"
        )

    if bypass_candidate_gate:
        state["equity_peak"] = round(candidate_value, 0)
        _log_equity_peak_update(old_peak, candidate_value, current_equity, caller=caller, reason=reason)
        append_peak_audit(action="CONFIRMED", broker_equity=broker_equity, diag=diag, **_audit_common)
        return candidate_value

    jump_ratio = (candidate_value / old_peak - 1.0) if old_peak > 0 else 0.0
    if jump_ratio >= CANDIDATE_PEAK_JUMP_THRESHOLD:
        state["candidate_peak"] = {
            "value":                   round(candidate_value, 0),
            "staged_date":             today_str,
            "reason":                  reason,
            "current_equity_at_stage": round(current_equity, 0),
        }
        _log_equity_peak_update(
            old_peak, candidate_value, current_equity,
            caller=caller, reason=f"{reason}_STAGED_CANDIDATE",
        )
        append_peak_audit(action="STAGED", broker_equity=broker_equity, diag=diag, **_audit_common)
        return old_peak

    # ── Study96 Phase5: "new_high" は old_peak を上回る場合のみ許可 ────────
    if reason == "new_high" and candidate_value <= old_peak:
        raise EquityPeakInvariantError(
            f"EQUITY_PEAK_INVARIANT_VIOLATION: reason=new_high なのに "
            f"candidate_value=¥{candidate_value:,.0f} <= old_peak=¥{old_peak:,.0f} "
            f"(new_high は old_peak を上回る場合のみ呼び出されるべき)。caller={caller}"
        )

    state["equity_peak"] = round(candidate_value, 0)
    _log_equity_peak_update(old_peak, candidate_value, current_equity, caller=caller, reason=reason)
    append_peak_audit(action="APPLIED", broker_equity=broker_equity, diag=diag, **_audit_common)
    return candidate_value


def _export_rsr_snapshot(date_str: str, rsr_scores: dict) -> None:
    """
    Write daily RSR snapshot to runtime/rsr/YYYY-MM-DD.json (atomic write).
    Called at end of evaluate() so governance can use T-1 RSR data next day.
    FAIL_OPEN: silently logs and returns on any error.
    """
    try:
        from src.paths import RSR_SNAPSHOT_DIR
        RSR_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        out  = RSR_SNAPSHOT_DIR / f"{date_str}.json"
        tmp  = out.with_suffix(".tmp")
        payload = json.dumps(
            {"date": date_str, "scores": {k: round(float(v), 2) for k, v in rsr_scores.items()}},
            ensure_ascii=False,
        )
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(out)
        logger.debug("[RSR_SNAPSHOT] written %s (%d symbols)", out.name, len(rsr_scores))
    except Exception as exc:
        logger.warning("[RSR_SNAPSHOT] export failed: %s", exc)


class AbortError(RuntimeError):
    """Hard-stop error used to abort live execution without fallback."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason

# paths.py をインポート（呼び出し元がすでに sys.path にプロジェクトルートを追加済み）
from src.paths import (
    ALLOW_YFINANCE_NETWORK,
    BACKTEST_DATASET_DIR,
    CACHE_DIR,
    DEFAULT_DATA_VERSION,
    LOGS_DIR,
    ORDER_LOCK_FILE,
    RUNTIME_DIR,
)
from src.execution.dd_engine import compute_drawdown
from src.portfolio.equity import (
    SAFE_WARN_CONFIRM_REQUIRED,
    append_peak_audit,
    assert_broker_equity_invariant,
    check_broker_consistency,
    check_peak_anomaly,
    compute_live_equity,
    detect_cash_event,
    rebuild_equity_peak,
    BrokerEquityInvariantError,
)
from src.portfolio.state_store import (
    BrokerSnapshot,
    SnapshotValidationError,
    commit_broker_snapshot,
    load_portfolio_state,
    log_startup_state_line,
    save_portfolio_state,
    update_portfolio_state_from_broker,
    write_reconciliation_log,
)
from src.portfolio.broker_source import (
    BrokerSnapshotUnavailable,
    fetch_broker_snapshot,
)

# ------------------------------------------------------------------ #
# ポートフォリオ状態・CB 管理定数
# ------------------------------------------------------------------ #
PORTFOLIO_STATE_FILE          = RUNTIME_DIR / "portfolio_state.json"
OHLCV_CACHE_DIR               = CACHE_DIR   / "ohlcv"              # OHLCVキャッシュ（ダウンロード失敗時のフォールバック）
OHLCV_CACHE_MAX_AGE_DAYS      = 5                      # キャッシュ有効期限（営業日換算で約1週間）
SNAPSHOT_MAX_STALE_DAYS       = 10                     # snapshot 有効期限（GW等の連休を考慮）
DATA_HEALTH_MIN_RATIO         = 0.90                   # RSR42データ健全性下限（これ未満でシグナル停止）
CB_DD_TRIGGER                 = 0.15          # DD がここを割ったら CB 発動
CB_COOLDOWN_TRADING_DAYS      = 30            # CB 後のクールダウン（営業日）
RECOVERY_THRESHOLD_RATIO      = 0.98          # peak の何割まで回復したら NORMAL
REENTRY_COOLDOWN_TRADING_DAYS = 5             # 時間ストップ後の再エントリー禁止（営業日）
MIN_DAILY_VALUE_YEN           = 5_000_000_000 # 流動性フィルター（5B 円/日）
ORDER_RATE_LIMIT_PER_MIN      = 3             # kabu API 発注レート（件/分）

# ── equity_peak ハードニング (EQUITY_PEAK_HARDENING, 2026-07-03) ──────────
CANDIDATE_PEAK_JUMP_THRESHOLD = 0.10   # 前回peak比+10%以上はcandidate_peakへステージング
CANDIDATE_RECONFIRM_TOLERANCE = 0.02   # 各営業日再確認時の許容下振れ（2%）
# Study96 EquityPeak SSOT Root Cause Audit (2026-07-17): 2026-07-15の実インシデントで
# +35.9%ジャンプが「翌営業日1回のみの再確認」でCONFIRMEDされてしまったことが判明。
# SAFE_WARN機構が既に採用している「N連続確認」パターン（SAFE_WARN_CONFIRM_REQUIRED=3）を
# candidate_peak確定にも適用し、単発の持続だけでは確定させない（原因推定なし・
# 純粋に営業日をまたいだ持続回数のみで判定する）。
CANDIDATE_PEAK_RECONFIRM_COUNT = 3     # 確定に必要な連続営業日再確認回数

# ── 幽霊ポジション対策 (GHOST_POSITION_FIX, 2026-07-03 → 2026-07-18廃止) ───
# 2026-07-03 実インシデント: 5301.T が実ブローカーで売却済みにも関わらず、
# 「部分補完」ロジックが毎run無条件に portfolio_state から復活させ続け、
# compute_live_equity() の market_value を ¥595,550 過大評価 → 誤 equity_peak
# (¥4,706,291、実際は一度も到達していない) を発生させた。当時はstreak counterで
# 緩和したが、Broker-as-Sole-SSOTリファクタ(2026-07-18)によりstateからの部分補完
# 自体を全廃した（broker応答は失敗/未接続時を除き常に無条件に信頼する）。

# ── mean_rev 反発未発生検出 ──────────────────────────────────────
# エントリー後 MEANREV_FAIL_DAYS 営業日以内に High が
# entry_price × (1 + MEANREV_MIN_BOUNCE) に到達しなかった場合は早期撤退
MEANREV_FAIL_DAYS    = int(os.environ.get("MEANREV_FAIL_DAYS",    "4"))
MEANREV_MIN_BOUNCE   = float(os.environ.get("MEANREV_MIN_BOUNCE", "0.01"))

# ── クラスタ相場 2段階制御 ─────────────────────────────────────
# level1 (density >= 15%): mean_rev 新規 BUY 停止
# level2 (density >= 25%): momentum 偏重（MOM_WEIGHT_ADJ を更に引き上げ）
CLUSTER_LEVEL1_THRESH = float(os.environ.get("CLUSTER_LEVEL1_THRESH", "0.15"))
CLUSTER_LEVEL2_THRESH = float(os.environ.get("CLUSTER_LEVEL2_THRESH", "0.25"))


# ------------------------------------------------------------------ #
# モジュールレベル ユーティリティ
# ------------------------------------------------------------------ #
def select_top_k(
    rsr_latest: pd.Series,
    k: int,
    liquidity: dict[str, float] | None = None,
) -> list[str]:
    """
    RSR ランキング上位 k 銘柄を返す。
    tie-breaker: liquidity_score（高い方が優先）。
    """
    df = pd.DataFrame({"rsr": rsr_latest})
    if liquidity:
        df["liquidity"] = pd.Series(liquidity)
        df = df.sort_values(["rsr", "liquidity"], ascending=[False, False])
    else:
        df = df.sort_values("rsr", ascending=False)
    return df.head(k).index.tolist()


def _trading_days_held(entry_date_str: str, today: pd.Timestamp) -> int:
    """entry_date から today までの営業日数（entry 当日 = 0）。JPX 祝日対応。"""
    try:
        from src.market.jpx_calendar import JPXCalendar
        entry_ts = pd.Timestamp(entry_date_str)
        if today <= entry_ts:
            return 0
        cal = JPXCalendar()
        return max(0, cal.trading_days_between(entry_ts, today) - 1)
    except Exception:
        return 0


def _add_trading_days(start: pd.Timestamp, n: int) -> pd.Timestamp:
    """start から n 営業日後の日付を返す。JPX 祝日対応。"""
    from src.market.jpx_calendar import JPXCalendar
    cal = JPXCalendar()
    return cal.add_trading_days(start, n)


def _compute_data_as_of(universe_raw: dict) -> str:
    """
    universe_raw 全銘柄の最終取得日のうち最も古い日付を data_as_of として返す。

    2026-07-08 RCA: 旧実装は next(iter(universe_raw)) で辞書挿入順の先頭銘柄
    (実質ランダム)のみを見ており、その銘柄がキャッシュ・フォールバック等で
    他銘柄より古いデータのままだと data_as_of が実態と無関係に古く／新しく
    表示され得た。最古日を採用することで「その日実際に使われた最も古い
    データ」を正直に表示する（＝真の鮮度下限）。
    """
    dates = [
        info["df"].index[-1].date()
        for info in universe_raw.values()
        if info.get("df") is not None and not info["df"].empty
    ]
    if not dates:
        return str(datetime.now(JST).date())
    return str(min(dates))


def _capacity_check(effective_max_pos: int, held_count: int, pending_buy_count: int = 0) -> int:
    """
    残りエントリー可能スロット数を返す（0以下ならこれ以上の新規BUYは禁止）。

    通常BUY・Shadow候補・Fallback・TrendFollowなど、発注経路が異なっても
    必ずこの一本の関数で残スロットを計算すること（2026-07-07 4銘柄同時保有
    インシデント: Shadow経路だけがこの計算を経由せず max_positions を突破した）。
    """
    return effective_max_pos - held_count - pending_buy_count


def _load_order_lock() -> dict:
    if not ORDER_LOCK_FILE.exists():
        return {}
    try:
        return json.loads(ORDER_LOCK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def trend_follow_candidates(
    universe_info: dict,
    top_n: int,
    logger: "logging.Logger",
) -> "pd.DataFrame":
    """
    TREND_FOLLOW戦略の候補銘柄を選定する。
    market_trend=True のときのみ呼ばれる。
    """
    rows = []
    for sym, info in universe_info.items():
        df = info.get("df")
        if df is None or df.empty:
            continue
        last = df.iloc[-1]

        # 流動性フィルタ
        turnover = last.get("avg_turnover_20d", float("nan"))
        volume   = last.get("avg_volume_20d",   float("nan"))
        if pd.isna(turnover) or turnover < 500_000_000:
            continue
        if pd.isna(volume) or volume < 200_000:
            continue

        close      = last.get("Close",      float("nan"))
        ma20       = last.get("ma20",        float("nan"))
        ma20_up    = last.get("ma20_up",     float("nan"))
        ma50       = last.get("ma50",        float("nan"))
        rsr_252    = last.get("rsr_252",     float("nan"))

        if pd.isna(close):
            continue

        # Phase1: close > MA20 AND MA20 rising
        phase1 = (not pd.isna(ma20) and close > ma20) and \
                 (not pd.isna(ma20_up) and ma20_up > 0)

        # fallback: close > MA50 AND rsr_252 > 50
        fallback_flag = False
        if not phase1:
            if (not pd.isna(ma50) and close > ma50) and \
               (not pd.isna(rsr_252) and rsr_252 > 50):
                fallback_flag = True
            else:
                continue  # どちらも満たさない場合スキップ

        # rsr_252 フィルタ
        if pd.isna(rsr_252) or rsr_252 <= 10:
            continue

        rows.append({
            "symbol":        sym,
            "sector":        info.get("sector", ""),
            "score":         rsr_252,
            "fallback_flag": fallback_flag,
            "entry_score":   rsr_252,
            "close":         float(close),
        })

    if not rows:
        logger.info("[trend_follow] no candidates after all filters")
        return pd.DataFrame()

    df_cands = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    # TODO: apply_sector_cap 未存在のためスキップ
    # df_cands = apply_sector_cap(df_cands, ...)

    df_cands = df_cands.head(top_n)
    logger.info(
        f"[trend_follow] candidates={len(df_cands)} "
        f"fallback_count={df_cands['fallback_flag'].sum()} "
        f"top={df_cands['symbol'].tolist()}"
    )
    return df_cands


def _apply_split_correction(
    df: "pd.DataFrame", symbol: str, logger: "logging.Logger"
) -> "pd.DataFrame":
    """スケール異常を自動検知し、yfinance splits で遡及補正する。補正不要なら即 return。"""
    import yfinance as yf  # noqa: PLC0415

    price_col = "Close"
    if price_col not in df.columns or len(df) < 2:
        return df

    # 1. スケール一貫性チェック（最優先）
    _r = df[price_col] / df[price_col].shift(1)
    scale_anomaly = bool(((_r > 5) | (_r < 0.2)).any())
    logger.info(
        f"[split_correction] symbol={symbol} scale_anomaly={scale_anomaly}"
    )
    if not scale_anomaly:
        return df

    # 2. yfinance からスプリット情報取得
    splits: "pd.Series" = pd.Series(dtype=float)
    try:
        splits = yf.Ticker(symbol).splits
    except Exception as _e:
        logger.warning(f"[split_correction] {symbol}: splits fetch failed: {_e}")

    if splits is None or len(splits) == 0:
        # ── フォールバック: ratio から自動推定 ──────────────────────────────
        valid_ratios = [2, 3, 4, 5, 10]
        price_s = df[price_col]
        ratio_series = price_s / price_s.shift(1)

        anomalies = ratio_series[(ratio_series > 5) | (ratio_series < 0.2)].dropna()
        if anomalies.empty:
            logger.warning(f"[split_correction] {symbol}: scale_anomaly=True but anomaly point not found, skipping")
            return df

        # 1) 単発補正制御: 最初の異常点のみ使用
        anomaly_idx = anomalies.index[0]
        loc = price_s.index.get_loc(anomaly_idx)
        if loc == 0:
            logger.warning(f"[split_correction] {symbol}: anomaly at first row, skipping")
            return df

        prev_price = float(price_s.iloc[loc - 1])
        curr_price = float(price_s.iloc[loc])
        if curr_price == 0:
            logger.warning(f"[split_correction] {symbol}: curr_price=0, skipping")
            return df

        detected_ratio_raw = prev_price / curr_price

        # 2) 比率スナップ（round禁止 → 最近傍探索）
        detected_ratio_snapped = min(valid_ratios, key=lambda v: abs(v - detected_ratio_raw))
        if detected_ratio_snapped not in valid_ratios:
            logger.warning(
                f"[split_correction] {symbol}: fallback skip: "
                f"detected_ratio_raw={detected_ratio_raw:.3f} snapped={detected_ratio_snapped} not in {valid_ratios}"
            )
            return df

        # 3) 持続性チェック: 翌日も同スケールであること
        if loc + 1 < len(price_s):
            next_price = float(price_s.iloc[loc + 1])
            persist_ratio = abs(next_price / curr_price) if curr_price != 0 else 999
            is_persistent = 0.5 < persist_ratio < 2.0  # 翌日も同スケール帯
        else:
            is_persistent = True  # 最終行なら持続性は不問
        if not is_persistent:
            logger.warning(
                f"[split_correction] {symbol}: fallback skip: not persistent "
                f"(next/curr={persist_ratio:.3f})"
            )
            return df

        # 4) ETF判定
        is_etf = False
        try:
            is_etf = yf.Ticker(symbol).info.get("quoteType", "") == "ETF"
        except Exception:
            pass

        # 5) 出来高チェック
        vol_ratio_val = float("nan")
        is_volume_ok = False
        if "Volume" in df.columns:
            vol_s = df["Volume"].replace(0, float("nan"))
            vol_ratio_s = vol_s / vol_s.shift(1)
            if anomaly_idx in vol_ratio_s.index:
                vol_ratio_val = float(vol_ratio_s.loc[anomaly_idx])
                ratio_inv = 1.0 / detected_ratio_snapped
                vol_check_val = abs(vol_ratio_val * ratio_inv - 1)
                if is_etf:
                    is_volume_ok = vol_check_val < 1.5   # ETF: 緩め
                    vol_mode = "loose"
                else:
                    is_volume_ok = vol_check_val < 0.3   # 非ETF: 厳しめ
                    vol_mode = "strict"
                logger.info(
                    f"[split_correction] {symbol}: vol_ratio={vol_ratio_val:.3f} "
                    f"ratio_inv={ratio_inv:.3f} vol_check={vol_check_val:.3f} "
                    f"mode={vol_mode} volume_ok={is_volume_ok}"
                )
        else:
            is_volume_ok = True  # Volume列なければスキップ不要
            vol_mode = "no_volume"

        logger.info(
            f"[split_correction] {symbol}: fallback summary "
            f"is_etf={is_etf} detected_ratio_raw={detected_ratio_raw:.3f} "
            f"snapped={detected_ratio_snapped} is_persistent={is_persistent} "
            f"vol_ratio={vol_ratio_val:.3f} volume_ok={is_volume_ok}"
        )

        if not is_volume_ok:
            logger.warning(f"[split_correction] {symbol}: fallback skip: volume check failed")
            return df

        # 6) 補正適用（最初の異常点以前のみ）
        mask = df.index < anomaly_idx
        factor = 1.0 / detected_ratio_snapped
        df_c = df.copy()
        ohlc_cols = [c for c in ("Open", "High", "Low", "Close") if c in df_c.columns]
        for col in ohlc_cols:
            df_c.loc[mask, col] = df_c.loc[mask, col] * factor
        # Volume は補正しない

        # 7) 補正後検証
        ratio_post = df_c[price_col] / df_c[price_col].shift(1)
        still_anomaly = bool(((ratio_post > 5) | (ratio_post < 0.2)).any())
        if still_anomaly:
            logger.warning(
                f"[split_correction] {symbol}: post-correction scale anomaly remains, "
                f"reverting to original df"
            )
            return df

        logger.warning(
            f"[split_correction] {symbol}: FALLBACK correction applied "
            f"factor={factor:.4f} rows_affected={int(mask.sum())} "
            f"anomaly_date={anomaly_idx} post_check=OK"
        )
        return df_c

    # 3. 遡及補正（OHLC ÷ ratio_val、Volume × ratio_val）
    df_idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df_c = df.copy()
    ohlc = [c for c in ("Open", "High", "Low", "Close") if c in df_c.columns]
    vol = "Volume" if "Volume" in df_c.columns else None

    for split_date, ratio_val in splits.items():
        if ratio_val <= 0:
            continue
        split_dt = pd.Timestamp(split_date)
        mask = df_idx < split_dt
        if not mask.any():
            continue
        factor = 1.0 / ratio_val
        logger.info(
            f"[split_correction] {symbol}: split_date={split_dt.date()} "
            f"split_ratio={ratio_val} cumulative_factor={factor:.4f} "
            f"rows_affected={int(mask.sum())}"
        )
        for col in ohlc:
            df_c.loc[mask, col] = df_c.loc[mask, col] * factor
        if vol:
            df_c.loc[mask, vol] = df_c.loc[mask, vol] / factor

    # 4. 補正後確認ログ
    _r2 = df_c[price_col] / df_c[price_col].shift(1)
    still = bool(((_r2 > 5) | (_r2 < 0.2)).any())
    logger.info(
        f"[split_correction] {symbol}: correction done "
        f"before_last={df[price_col].iloc[-1]:.2f} "
        f"after_last={df_c[price_col].iloc[-1]:.2f} "
        f"single_scale_guaranteed={not still}"
    )
    return df_c


# ------------------------------------------------------------------ #
# シグナルデータクラス
# ------------------------------------------------------------------ #
@dataclass
class StockSignal:
    """1銘柄分のシグナル情報"""
    symbol:               str
    sector:               str
    signal:               int     # +1=買い / -1=売り / 0=ホールド
    rsr:                  float   # RSR 値（0〜100）
    rsr_rank:             int     # top_k ユニバース内の RSR 順位（1=最高）
    sepa_score:           int     # SEPA 条件数（0〜8）
    rsr_mom:              float   # RSR モメンタム
    hold_days:            int     # 現在の保有営業日数（未保有は 0）
    currently_holding:    bool    # 現在保有中か
    reason:               str     # シグナル理由（ログ用）
    strategy_type:        str   = "fujiko"   # "fujiko" / "mean_rev"
    trailing_stop_price:  float = 0.0        # ギャップダウン検出用（0 = 未保有 or 未計算）
    entry_price:          float = 0.0        # BUY時参考単価（未保有は 0）
    unrealized_pnl_pct:   float = 0.0        # 含み損益率（未保有は 0）
    entry_date_known:     bool  = True       # False = entry_date欠損（2026-07-15追加）。
                                              # hold_days=0との区別用。表示層は entry_date_known=False
                                              # かつ currently_holding=True の場合 "Unknown" を表示すること
                                              # （0dへのフォールバック表示は禁止 — 保有日数SSOT方針）。


# セクター別採用戦略（dynamic_selection.py の結果に基づく）
SECTOR_STRATEGY: dict[str, str] = {
    "海運":    "fujiko",
    "機械":    "fujiko",
    "電機精密": "fujiko",
    "商社":    "fujiko",
    "電機":    "fujiko",
    "ゲーム":  "fujiko",
    "レジャー": "fujiko",
    "食品":    "fujiko",
    # 平均回帰優位
    "ガス":    "mean_rev",
    "鉄鋼":    "mean_rev",
    "銀行":    "mean_rev",
    "保険":    "mean_rev",
    "輸送機器": "mean_rev",
    "化学":    "mean_rev",
    "小売":    "mean_rev",
    # 動的（高い方）
    "サービス":  "dynamic",
    "医薬品":   "dynamic",
    "不動産":   "dynamic",
    "情報通信":  "dynamic",
    "陸運":     "dynamic",
}

MR_PARAMS: dict = dict(
    rsi_period      = 5,
    rsi_entry       = 25.0,
    rsi_exit        = 65.0,
    ma_long         = 200,
    stop_loss_pct   = 0.07,
    max_hold_days   = 10,
    knife_threshold = 0.15,
)


@dataclass
class OrderInstruction:
    """発注指示（JSON 出力 + API 送信の両方に使用）"""
    symbol:           str
    symbol_4digit:    str        # kabustation 用 4 桁コード
    sector:           str
    side:             str        # "BUY" / "SELL"
    qty:              int
    order_type:       str        # "MARKET_OPEN"（寄成）
    estimated_price:  float
    estimated_amount: float
    reason:           str
    atr20:            float = 0.0   # BUY時ATR20（False Breakout診断用）
    strategy_type:    str   = ""    # "fujiko" / "mean_rev"（mean_rev反発検出で参照）


@dataclass
class BridgeResult:
    """シグナルブリッジ実行結果"""
    generated_at:      str
    mode:              str       # "DRY_RUN" / "LIVE"
    data_as_of:        str
    n_universe:        int
    cb_state:          str       # "NORMAL" / "CB_ACTIVE" / "RECOVERY"
    top_k_symbols:     list[str]
    portfolio_summary: dict
    signals:           list[dict]
    orders:            list[dict]
    warnings:          list[str] = field(default_factory=list)
    universe_stats:    dict      = field(default_factory=dict)
    # {live, shadow, rsr_context, tradeable, filtered_price, filtered_risk}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


# ------------------------------------------------------------------ #
# _build_orders() 契約バリデーター
# ------------------------------------------------------------------ #
def _validate_build_orders_contract(result: object) -> None:
    """
    _build_orders() の戻り値が
    (orders, warnings, blocked_alloc_cap, lot_rounded_up, risk_rejected)
    の 5-tuple であることを強制する。
    """
    if not isinstance(result, tuple):
        raise TypeError(
            f"_build_orders() must return a tuple, got {type(result).__name__}"
        )
    if len(result) != 5:
        raise ValueError(
            f"_build_orders() must return exactly 5 values "
            f"(orders, order_warnings, blocked_alloc_cap, lot_rounded_up, risk_rejected), "
            f"got {len(result)}"
        )


# ------------------------------------------------------------------ #
# シグナルブリッジ本体
# ------------------------------------------------------------------ #
class SignalBridge:
    """
    フジコ法 Top-k ローテーション → kabuステーション発注ブリッジ

    Args:
        universe_tickers:         {symbol: sector}（価格フィルター後・実際に売買する銘柄）
        rsr_universe_tickers:     RSR 計算専用ユニバース（24 銘柄固定）
        fujiko_params:            FujikoStrategy パラメータ辞書
        top_k:                    RSR 上位 k 銘柄のみ BUY 対象（default 4）
        max_hold_days:            最大保有営業日数（超えたら時間ストップ）
        max_new_positions_per_day: 1 回の実行で生成する新規 BUY の上限件数
        order_rate_limit_per_min: kabu API 発注レート制限（件/分）
    """

    def __init__(
        self,
        universe_tickers:          dict[str, str],
        fujiko_params:             dict,
        capital:                   float,
        max_positions:             int,
        min_hold_days:             int,
        emergency_exit_pct:        float,
        max_dd_limit:              float = 0.15,
        min_sectors:               int   = 1,
        max_single_weight:         float = 0.25,
        live:                      bool  = False,
        benchmark_ticker:          str   = "1306.T",
        rsr_universe_tickers:      dict[str, str] | None = None,
        top_k:                     int   = 4,
        max_hold_days:             int | None = 60,
        max_new_positions_per_day: int   = 2,
        order_rate_limit_per_min:  int   = ORDER_RATE_LIMIT_PER_MIN,
        portfolio_state_file:      Path | None = None,
        shadow_universe_tickers:   dict[str, str] | None = None,
        shock_exit_mode:           str   = "full_exit",
        regime_sizing:             str   = "none",
        bear_scale:                float = 1.0,
        cfg=None,
        deployable_capital:        float = 0.0,
        entry_freeze_enabled:      bool  = False,
        entry_freeze_reason:       str   = "Research Freeze",
        require_broker:            bool  = True,
    ) -> None:
        # Broker-as-Sole-SSOT (2026-07-18): True の場合、run() は broker snapshot
        # 取得に失敗すると AbortError で即座に停止する（state/OHLCVへのフォールバック
        # を行わない）。DRY/LIVE共通でTrueが既定。False は手動検証・研究用途の
        # 明示的な省略モードのみに使う想定（--allow-no-broker 経由）。
        self.require_broker            = require_broker
        self.universe_tickers          = universe_tickers
        self.shadow_universe_tickers   = shadow_universe_tickers or {}
        self.rsr_universe_tickers      = (
            rsr_universe_tickers if rsr_universe_tickers is not None
            else universe_tickers
        )
        self.fujiko_params             = fujiko_params
        self.capital                   = capital
        self.max_positions             = max_positions
        self.max_dd_limit              = max_dd_limit
        self.min_sectors               = min_sectors
        self.max_single_weight         = max_single_weight
        self.live                      = live
        self.benchmark_ticker          = benchmark_ticker
        self.top_k                     = top_k
        self.max_hold_days             = max_hold_days
        self.min_hold_days             = min_hold_days
        self.emergency_exit_pct        = emergency_exit_pct
        self.max_new_positions_per_day = max_new_positions_per_day
        self.order_rate_interval_sec   = 60.0 / max(1, order_rate_limit_per_min)
        self._state_file               = portfolio_state_file or PORTFOLIO_STATE_FILE
        self.shock_exit_mode           = shock_exit_mode   # "full_exit" | "partial_50" | "composite"
        self.regime_sizing             = regime_sizing      # "none" | "regime_2" | "regime_4"
        self.bear_scale                = bear_scale         # TOPIX MA200下のサイズ係数
        self._cfg                      = cfg                # StrategyConfig（cap feature flag 参照用）
        self.deployable_capital        = deployable_capital  # from cap_state.deployable_capital (Phase 5B.1)
        # Entry Freeze Mode（資産保全・2026-07-17 Study100/101帰結）: True で新規BUY全面停止。
        # SELL/exit・signal generationには一切影響しない。CBとは独立フラグ（_build_ordersでOR結合）。
        self.entry_freeze_enabled       = bool(entry_freeze_enabled)
        self.entry_freeze_reason        = str(entry_freeze_reason)
        self._positions_api_status     = {"ok": False, "source": "virtual", "error": None}
        self._wallet_api_status        = {"ok": False, "source": "virtual", "error": None}
        # pre_trade_risk_check 用キャッシュ（run() 後に execution layer で再利用）
        self._last_current_positions: dict = {}
        # execution metrics: signal generation timestamp (set in run())
        self._last_signal_time: datetime | None = None
        self._last_universe_raw: dict = {}
        # EVS統合用: 直近runのステージ監査（CAPACITY/CAPITAL/RISK/...）
        self._last_stage_audit: list = []

        # min_rsr はエントリー専用（変更禁止）。exit 閾値は rsr_exit で分離。
        # rsr_exit は FujikoStrategy の引数ではないため _fujiko_params_live から除外する。
        self._fujiko_params_live = {k: v for k, v in fujiko_params.items() if k != "rsr_exit"}
        self.rsr_exit_threshold = float(
            fujiko_params.get("rsr_exit", fujiko_params.get("min_rsr", 75.0))
        )

        # gap_stop 用: run() 後に保有ポジションのストップ情報をキャッシュ
        self._last_held_stop_info: dict[str, dict] = {}

        self._client = None
        try:
            from src.kabusapi.client import KabuClient
            self._client = KabuClient()
            self._client.fetch_token()
            if not live:
                logger.info("API 接続成功（読み取り専用）")
        except Exception as e:
            if live:
                raise
            logger.info("API 未接続（ドライランフォールバック）: %s", e)
            self._client = None

    # ------------------------------------------------------------------ #
    # ポートフォリオ状態管理
    # ------------------------------------------------------------------ #
    def _load_portfolio_state(self) -> dict:
        """ポートフォリオ状態ファイルを読み込む。欠落・破損時は hard abort。"""
        if not self._state_file.exists():
            raise AbortError(
                "portfolio_state_missing",
                f"portfolio state file is missing: {self._state_file}",
            )
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception as e:
            raise AbortError(
                "portfolio_state_missing",
                f"portfolio state file is unreadable: {self._state_file}: {e}",
            ) from e

    def _save_portfolio_state(self, state: dict) -> None:
        """portfolio_state.json への書き込みを state_store 経由で行う。
        直接 json.dump を呼ぶことを禁止する唯一の経路。
        """
        _src = "broker_api" if self._positions_api_status.get("source") == "broker" else "internal"
        save_portfolio_state(state, path=self._state_file, data_source=_src)

    def _update_cb_state(
        self,
        state:               dict,
        current_equity:      float,
        today_str:           str,
        broker_snapshot:     "BrokerSnapshot | None" = None,
    ) -> dict:
        """
        サーキットブレーカー状態機械を更新する。

        current_equity は常にbroker直接値（BrokerSnapshotから計算された唯一のequity）。
        合成・補償済みequityを渡すことは禁止する（2026-07-19 Broker-as-Sole-SSOT
        リファクタ: 旧_cb_equityの合成equity方式を廃止）。equity_peak・drawdown・
        candidate_peak再確認等、本メソッド内の判定はすべてこのraw equityのみで行う。

        PendingOrderState機構（決済ラグ補償によるCB発動抑制）は2026-07-19に完全撤去。
        根拠: (1) SOURCE1(state-based)はstale portfolio_stateとの区別ができず誤補償を
        生む構造的欠陥があった（実機検証で5日前に確定売却済みの3銘柄を「決済待ち」と
        誤認し¥1,748,850を誤算出）。(2) SOURCE2(ledger-based)はrun_live_signal.pyが
        OrderLedger.check_and_record()を呼ばないため実質死コードだった。(3) 両インシデント
        の発生前提（run_morning_signal.pyとの並行スケジュール実行）は2026-07-15 SSOT統合
        で構造的に消滅済み。撤去後もCB_ACTIVEには既存の[CB_FAST_RECOVERY]/[CB_AUTO_RESTORE]
        機構があり、誤発動しても翌run（最大1日）で自己修復する。詳細は該当セッションの
        設計レビューを参照。

        broker_snapshot は equity_peak 更新前の整合性チェックに使う
        （EQUITY_PEAK_HARDENING, _commit_equity_peak() 経由）。

        NORMAL    → CB_ACTIVE  : drawdown <= -15%（BUY 全停止）
        NORMAL    → SAFE_WARN  : equity_peak が current_equity の 125% 超（state 破損疑い）
        SAFE_WARN → CB_ACTIVE  : N 連続確認後に昇格
        SAFE_WARN → NORMAL     : peak 比率が正常に戻った場合（peak を補正して継続）
        CB_ACTIVE → RECOVERY   : クールダウン 30 営業日経過
        RECOVERY  → NORMAL     : equity >= peak × 98%

        equity_peak の書込みは全て _commit_equity_peak() 経由（EQUITY_PEAK_HARDENING）。
        REJECTED/STAGED になっても、本メソッド内の CB 状態遷移判定（drawdown/分岐/
        _save_cb_event）は変更しない — peak の永続化可否と当該run内のCB判定は分離する。

        equity_peakの唯一の真実は現在の証券口座equityであり、原因推定（入出金等）は
        一切行わない（Study96 EquityPeak SSOT Root Cause Audit, 2026-07-17）。異常な
        ジャンプは理由を問わず一律にcandidate_peakへ保留し、N連続営業日の持続が
        確認されて初めて確定する（下記candidate_peak再確認ロジック参照）。
        """
        cb_state_before  = state.get("cb_state", "NORMAL")
        cb_state         = cb_state_before
        equity_peak      = float(state.get("equity_peak", self.capital))
        _original_peak   = equity_peak  # anomaly チェック用に保存
        mode             = "live" if self.live else "dry"

        # ── candidate_peak 多段階再確認 (EQUITY_PEAK_HARDENING 2026-07-03 /
        #    Study96 2026-07-17でN連続確認化) ─────────────────────────────
        # 前回runで+10%以上のジャンプとしてステージングされた候補があれば、
        # 翌営業日以降の最初のrunで再確認する。同日中は何もしない。
        # 2026-07-15実インシデント（+35.9%ジャンプが1回の翌営業日確認だけで
        # CONFIRMEDされてしまった）を受け、CANDIDATE_PEAK_RECONFIRM_COUNT回
        # （既定3）連続で基準を満たすまでは確定しない。原因（入出金等）の
        # 推定は行わず、単に「営業日をまたいで何度も持続したか」だけを見る。
        _candidate_holding_this_call = False
        candidate = state.get("candidate_peak")
        if candidate is not None:
            _cand_value    = float(candidate.get("value", 0))
            _staged_date   = candidate.get("staged_date")
            _confirm_count = int(candidate.get("confirm_count", 0))
            _expected_next_td = (
                _add_trading_days(pd.Timestamp(_staged_date), 1).strftime("%Y-%m-%d")
                if _staged_date else None
            )
            if _expected_next_td and today_str >= _expected_next_td:
                if current_equity >= _cand_value * (1 - CANDIDATE_RECONFIRM_TOLERANCE):
                    _confirm_count += 1
                    if _confirm_count >= CANDIDATE_PEAK_RECONFIRM_COUNT:
                        # Study96追記: N回連続確認に到達しても、確定直前に
                        # check_broker_consistency() が改めて実行される
                        # （_commit_equity_peak()内で無条件に先頭実行）。
                        # そこで不整合と判定されればREJECTEDでold_peakが
                        # 返るため、その場合はconfirm_countを維持したまま
                        # 候補を残し、次回runで再度最終確認を試みる
                        # （持続回数の実績を無駄に失わないため）。
                        _pre_call_peak = equity_peak
                        equity_peak = _commit_equity_peak(
                            state, _cand_value, current_equity,
                            caller="_update_cb_state", reason="candidate_peak_confirmed",
                            broker_snapshot=broker_snapshot, today_str=today_str, mode=mode,
                            bypass_candidate_gate=True,
                        )
                        if equity_peak != _pre_call_peak:
                            state["candidate_peak"] = None
                        else:
                            # 最終整合性チェックで見送り。同一current_equityでの
                            # new_high再ステージングを防ぐため holding 扱いにする。
                            _candidate_holding_this_call = True
                            logger.warning(
                                "[EQUITY_PEAK_FINAL_CONSISTENCY_REJECT] confirm_count=%d/%d "
                                "到達も、確定直前のbroker整合性チェックで不整合と判定され見送り。"
                                "confirm_countは維持し次回runで再試行する。",
                                _confirm_count, CANDIDATE_PEAK_RECONFIRM_COUNT,
                            )
                            state["candidate_peak"] = {
                                **candidate,
                                "confirm_count": _confirm_count,
                                "staged_date":   today_str,
                            }
                    else:
                        # まだ規定回数に達していない → 保留を継続（次回チェックは
                        # 今日から数えて翌営業日）。confirm_countのみ進める。
                        # _candidate_holding_this_call=True にして、直後の
                        # new_high フォールスルーが同一equityで即座に別候補を
                        # 再ステージングし confirm_count を上書きしないようにする。
                        _candidate_holding_this_call = True
                        state["candidate_peak"] = {
                            **candidate,
                            "confirm_count": _confirm_count,
                            "staged_date":   today_str,
                        }
                        logger.info(
                            "[EQUITY_PEAK_CANDIDATE_HOLDING] staged=¥%s current=¥%s "
                            "confirm=%d/%d — 保留継続（原因推定なし・持続回数のみで判定）",
                            f"{_cand_value:,.0f}", f"{current_equity:,.0f}",
                            _confirm_count, CANDIDATE_PEAK_RECONFIRM_COUNT,
                        )
                        append_peak_audit(
                            action="HOLDING", old_peak=equity_peak, new_peak=_cand_value,
                            current_equity=current_equity, broker_equity=None,
                            caller="_update_cb_state", reason="candidate_reconfirm_holding",
                            diag=f"confirm_count={_confirm_count}/{CANDIDATE_PEAK_RECONFIRM_COUNT}",
                            trading_date=today_str, mode=mode, pid=os.getpid(), run_id=_RUN_ID,
                        )
                else:
                    logger.warning(
                        "[EQUITY_PEAK_CANDIDATE_DISCARDED] staged=¥%s current=¥%s "
                        "(再確認基準未達) — 候補を破棄",
                        f"{_cand_value:,.0f}", f"{current_equity:,.0f}",
                    )
                    append_peak_audit(
                        action="DISCARDED", old_peak=equity_peak, new_peak=_cand_value,
                        current_equity=current_equity, broker_equity=None,
                        caller="_update_cb_state", reason="candidate_reconfirm_failed",
                        diag=(
                            f"reconfirm floor not met: {current_equity:,.0f} < "
                            f"{_cand_value * (1 - CANDIDATE_RECONFIRM_TOLERANCE):,.0f}"
                        ),
                        trading_date=today_str, mode=mode, pid=os.getpid(), run_id=_RUN_ID,
                    )
                    state["candidate_peak"] = None
            # today_str < _expected_next_td（ステージング当日）→ 候補は据え置き、no-op

        # equity_peak 更新（現在 equity が peak を超えた場合のみ上方修正）。
        # HOLDING中（今回confirm_countを進めただけ）の場合はスキップする —
        # 同一のcurrent_equityで即座に別のSTAGED候補が生成され、進めたばかりの
        # confirm_countが上書きされてしまうのを防ぐ。
        if not _candidate_holding_this_call and current_equity > equity_peak:
            equity_peak = _commit_equity_peak(
                state, current_equity, current_equity,
                caller="_update_cb_state", reason="new_high",
                broker_snapshot=broker_snapshot, today_str=today_str, mode=mode,
            )

        drawdown = compute_drawdown(current_equity, equity_peak)

        # last_equity は commit_broker_snapshot / update_portfolio_state_from_broker が
        # raw equity で設定済み。

        # ── SAFE_WARN チェック（peak 異常検出）────────────────────────────
        safe_warn_count = int(state.get("safe_warn_count", 0))
        is_anomaly, new_safe_warn_count, anomaly_msg = check_peak_anomaly(
            equity_peak, current_equity, safe_warn_count
        )

        # ── 状態遷移 ─────────────────────────────────────────────────────
        if cb_state == "NORMAL":
            if drawdown <= -CB_DD_TRIGGER:
                # PEAK_ANOMALY も同時に検出されている場合: peak を再構築して DD を再評価する。
                # 異常 peak（stale run / settlement lag 起因）による誤 CB 発動を防止する。
                if is_anomaly:
                    rebuilt_peak = rebuild_equity_peak(current_equity)
                    rebuilt_dd   = (current_equity / rebuilt_peak - 1.0) if rebuilt_peak > 0 else drawdown
                    logger.warning(
                        "[CB_FAIL_SAFE] PEAK_ANOMALY 検出: "
                        "original_peak=¥%s rebuilt_peak=¥%s dd_original=%.1f%% dd_rebuilt=%.1f%%",
                        f"{equity_peak:,.0f}", f"{rebuilt_peak:,.0f}",
                        drawdown * 100, rebuilt_dd * 100,
                    )
                    if rebuilt_dd > -CB_DD_TRIGGER:
                        # 再構築 peak では CB 閾値を超えない → peak を補正して CB 不発動
                        equity_peak = _commit_equity_peak(
                            state, rebuilt_peak, current_equity,
                            caller="_update_cb_state", reason="rebuild_compensated",
                            broker_snapshot=broker_snapshot, today_str=today_str, mode=mode,
                        )
                        drawdown             = rebuilt_dd
                        state["safe_warn_count"] = 0
                        logger.warning(
                            "[CB_FAIL_SAFE] PEAK_ANOMALY_COMPENSATED: "
                            "peak を ¥%s に補正。CB 不発動 (DD=%.1f%%)。",
                            f"{rebuilt_peak:,.0f}", rebuilt_dd * 100,
                        )
                        self._save_cb_event(
                            "NORMAL→PEAK_ANOMALY_COMPENSATED", rebuilt_dd, rebuilt_peak, today_str
                        )
                        # fall through to is_anomaly branch below for SAFE_WARN handling
                    else:
                        # 再構築後も CB 閾値を超える → 正規の CB 発動（peak 補正は行う）
                        equity_peak = _commit_equity_peak(
                            state, rebuilt_peak, current_equity,
                            caller="_update_cb_state", reason="rebuild_cb_trigger",
                            broker_snapshot=broker_snapshot, today_str=today_str, mode=mode,
                        )
                        drawdown             = rebuilt_dd
                        logger.warning(
                            "[CB_FAIL_SAFE] rebuilt_peak でも DD=%.1f%% ≤ -15%% → CB 発動",
                            rebuilt_dd * 100,
                        )

                if drawdown <= -CB_DD_TRIGGER:
                    today_ts     = pd.Timestamp(today_str)
                    cooldown_end = _add_trading_days(today_ts, CB_COOLDOWN_TRADING_DAYS)
                    state["cb_state"]             = "CB_ACTIVE"
                    state["cb_cooldown_end_date"] = cooldown_end.strftime("%Y-%m-%d")
                    state["recovery_threshold"]   = round(equity_peak * RECOVERY_THRESHOLD_RATIO, 0)
                    state["safe_warn_count"]      = 0
                    logger.warning(
                        "CB 発動: DD=%.1f%% / peak=¥%s → BUY 停止 / クールダウン終了=%s",
                        drawdown * 100, f"{equity_peak:,.0f}", state["cb_cooldown_end_date"],
                    )
                    self._save_cb_event("NORMAL→CB_ACTIVE", drawdown, equity_peak, today_str)

            elif is_anomaly:
                # peak が current_equity の 125% 超 かつ通常DD未達 → SAFE_WARN
                state["cb_state"]      = "SAFE_WARN"
                state["safe_warn_count"] = new_safe_warn_count
                logger.warning(
                    "SAFE_WARN 開始: %s / BUY は継続 / 残り確認=%d回",
                    anomaly_msg, max(0, SAFE_WARN_CONFIRM_REQUIRED - new_safe_warn_count),
                )
                self._save_cb_event("NORMAL→SAFE_WARN", drawdown, equity_peak, today_str)
            else:
                state["safe_warn_count"] = 0

        elif cb_state == "SAFE_WARN":
            if is_anomaly:
                state["safe_warn_count"] = new_safe_warn_count
                logger.warning(
                    "SAFE_WARN 継続: %s",
                    anomaly_msg,
                )
                if new_safe_warn_count >= SAFE_WARN_CONFIRM_REQUIRED:
                    # N 連続確認 → CB_ACTIVE に昇格
                    today_ts     = pd.Timestamp(today_str)
                    cooldown_end = _add_trading_days(today_ts, CB_COOLDOWN_TRADING_DAYS)
                    state["cb_state"]             = "CB_ACTIVE"
                    state["cb_cooldown_end_date"] = cooldown_end.strftime("%Y-%m-%d")
                    state["recovery_threshold"]   = round(equity_peak * RECOVERY_THRESHOLD_RATIO, 0)
                    state["safe_warn_count"]      = 0
                    logger.warning(
                        "SAFE_WARN → CB_ACTIVE: %d 連続 peak 異常確認 → BUY 停止",
                        SAFE_WARN_CONFIRM_REQUIRED,
                    )
                    self._save_cb_event("SAFE_WARN→CB_ACTIVE", drawdown, equity_peak, today_str)
            else:
                # peak 比率が正常に戻った → peak を現在 equity に補正して NORMAL 復帰
                state["cb_state"]        = "NORMAL"
                state["safe_warn_count"] = 0
                # 破損 peak を補正: 現在 equity か元の capital の大きい方
                corrected_peak = max(current_equity, float(self.capital))
                equity_peak = _commit_equity_peak(
                    state, corrected_peak, current_equity,
                    caller="_update_cb_state", reason="safe_warn_recovery",
                    broker_snapshot=broker_snapshot, today_str=today_str, mode=mode,
                )
                logger.info(
                    "SAFE_WARN → NORMAL: peak 比率が正常化。peak 補正候補=¥%s "
                    "(実際の反映値は equity_peak_audit.jsonl を参照)",
                    f"{corrected_peak:,.0f}",
                )
                self._save_cb_event("SAFE_WARN→NORMAL", drawdown, current_equity, today_str)

        elif cb_state == "CB_ACTIVE":
            # Fast recovery: equity が recovery_threshold を超え かつ DD が健全なら
            # クールダウン終了を待たず即時 NORMAL に戻す。
            # settlement lag 補償後の equity が正常域に戻った場合に CB を自動修復する。
            _recovery_thr = float(
                state.get("recovery_threshold")
                or _original_peak * RECOVERY_THRESHOLD_RATIO
            )
            _dd_with_orig_peak = (_original_peak - current_equity) / _original_peak if _original_peak > 0 else 0.0
            if current_equity >= _recovery_thr and drawdown > -CB_DD_TRIGGER:
                state["cb_state"]             = "NORMAL"
                state["cb_cooldown_end_date"] = None
                state["recovery_threshold"]   = None
                state["safe_warn_count"]      = 0
                logger.warning(
                    "[CB_FAST_RECOVERY] equity=¥%s >= threshold=¥%s / DD=%.1f%% → CB_ACTIVE を NORMAL に修復",
                    f"{current_equity:,.0f}", f"{_recovery_thr:,.0f}", drawdown * 100,
                )
                self._save_cb_event(
                    "CB_ACTIVE→NORMAL_FAST_RECOVERY", drawdown, current_equity, today_str
                )

            # PEAK_ANOMALY が残存しかつ peak rebuilt によって DD が -15% を超えない場合:
            # CB を即時 NORMAL に戻す（誤発動の自動修復）。
            if is_anomaly and state.get("cb_state") == "CB_ACTIVE":
                rebuilt_peak_cb = rebuild_equity_peak(current_equity)
                rebuilt_dd_cb   = (current_equity / rebuilt_peak_cb - 1.0) if rebuilt_peak_cb > 0 else drawdown
                if rebuilt_dd_cb > -CB_DD_TRIGGER:
                    equity_peak = _commit_equity_peak(
                        state, rebuilt_peak_cb, current_equity,
                        caller="_update_cb_state", reason="cb_auto_restore",
                        broker_snapshot=broker_snapshot, today_str=today_str, mode=mode,
                    )
                    state["cb_state"]             = "NORMAL"
                    state["cb_cooldown_end_date"] = None
                    state["recovery_threshold"]   = None
                    state["safe_warn_count"]      = 0
                    logger.warning(
                        "[CB_AUTO_RESTORE] PEAK_ANOMALY 検出: rebuilt_peak=¥%s で DD=%.1f%% > -15%% "
                        "→ CB_ACTIVE を NORMAL に自動修復",
                        f"{rebuilt_peak_cb:,.0f}", rebuilt_dd_cb * 100,
                    )
                    self._save_cb_event(
                        "CB_ACTIVE→NORMAL_AUTO_RESTORED", rebuilt_dd_cb, rebuilt_peak_cb, today_str
                    )
                else:
                    logger.info(
                        "[CB_ACTIVE] PEAK_ANOMALY あり但し rebuilt_peak でも DD=%.1f%% ≤ -15%% → CB 継続",
                        rebuilt_dd_cb * 100,
                    )
            end_date = state.get("cb_cooldown_end_date")
            if end_date and today_str >= end_date and state.get("cb_state") == "CB_ACTIVE":
                state["cb_state"] = "RECOVERY"
                logger.info("CB_ACTIVE → RECOVERY: クールダウン終了（%s）", end_date)
                self._save_cb_event("CB_ACTIVE→RECOVERY", drawdown, current_equity, today_str)

        elif cb_state == "RECOVERY":
            threshold = float(
                state.get("recovery_threshold")
                or equity_peak * RECOVERY_THRESHOLD_RATIO
            )
            if current_equity >= threshold:
                state["cb_state"]             = "NORMAL"
                state["cb_cooldown_end_date"] = None
                state["recovery_threshold"]   = None
                state["safe_warn_count"]      = 0
                logger.info(
                    "RECOVERY → NORMAL: ¥%s >= threshold=¥%s",
                    f"{current_equity:,.0f}", f"{threshold:,.0f}",
                )
                self._save_cb_event("RECOVERY→NORMAL", drawdown, current_equity, today_str)

        cb_state_after = state.get("cb_state", "NORMAL")

        # ── [CB_STATE] 構造化ログ ────────────────────────────────────────
        # current_equityは常にbroker直接値の単一equity（2026-07-19以降、
        # 「補償済み」equityは存在しない）。
        if cb_state_after != cb_state_before:
            # 状態遷移あり: 詳細を WARNING
            if drawdown <= -CB_DD_TRIGGER:
                _reason = "DRAWDOWN_TRIGGER"
            elif cb_state_after == "SAFE_WARN":
                _reason = "PEAK_ANOMALY"
            else:
                _reason = "STATE_TRANSITION"
            logger.warning(
                "[CB_STATE] before=%s after=%s equity=%.0f dd=%.1f%% reason=%s",
                cb_state_before, cb_state_after, current_equity, drawdown * 100, _reason,
            )
        else:
            # 遷移なし: INFO
            logger.info(
                "[CB_STATE] state=%s equity=%.0f dd=%.1f%%",
                cb_state_after, current_equity, drawdown * 100,
            )

        return state

    def _save_cb_event(
        self, transition: str, drawdown: float, equity: float, date: str
    ) -> None:
        """CB イベントを logs/cb_events/ に JSONL 形式で追記する"""
        cb_log_dir = LOGS_DIR / "cb_events"
        cb_log_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "date":       date,
            "transition": transition,
            "drawdown":   round(drawdown, 4),
            "equity":     round(equity, 0),
        }
        log_file = cb_log_dir / f"cb_{date.replace('-', '')}.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # データ取得
    # ------------------------------------------------------------------ #
    # OHLCVキャッシュ管理
    # ------------------------------------------------------------------ #
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """取得成功データをparquetキャッシュに保存する"""
        try:
            OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OHLCV_CACHE_DIR / f"{ticker}.parquet")
        except Exception as e:
            logger.warning("%s キャッシュ保存失敗: %s", ticker, e)

    def _load_from_cache(self, ticker: str, start_date: str) -> pd.DataFrame:
        """
        parquetキャッシュを読み込む。
        OHLCV_CACHE_MAX_AGE_DAYS 以内のキャッシュのみ有効とする。
        """
        cache_path = OHLCV_CACHE_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(cache_path)
            if df.empty:
                return pd.DataFrame()
            age_days = (datetime.now(JST).date() - df.index.max().date()).days
            if age_days > OHLCV_CACHE_MAX_AGE_DAYS:
                logger.warning("%s キャッシュ期限切れ（%d日前）", ticker, age_days)
                return pd.DataFrame()
            df = df[df.index >= pd.Timestamp(start_date)]
            logger.info("%s キャッシュから読み込み（最終: %s, %d日前）",
                        ticker, df.index.max().date(), age_days)
            return df
        except Exception as e:
            logger.warning("%s キャッシュ読み込み失敗: %s", ticker, e)
            return pd.DataFrame()

    def _load_from_snapshot(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """backtest snapshot からOHLCVを読み込む。"""
        if not DEFAULT_DATA_VERSION:
            return pd.DataFrame()
        snap_path = BACKTEST_DATASET_DIR / DEFAULT_DATA_VERSION / f"{ticker}.parquet"
        if not snap_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(snap_path)
            if df.empty:
                return pd.DataFrame()
            if "Adj Close" in df.columns and "Close" in df.columns:
                df = df.copy()
                df["Close"] = df["Adj Close"]
            df = df.loc[
                (df.index >= pd.Timestamp(start_date))
                & (df.index <= pd.Timestamp(end_date))
            ]
            return df.dropna(subset=["Close"])
        except Exception as e:
            logger.warning("%s snapshot読み込み失敗: %s", ticker, e)
            return pd.DataFrame()

    def _retry_single_fetch(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        バッチ取得失敗時の個別再フェッチ。
        sleep jitter でレートリミット回避。
        """
        import random, time
        time.sleep(random.uniform(0.3, 1.2))
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            df.dropna(subset=["Close"], inplace=True)
            return df
        except Exception as e:
            logger.warning("%s 個別フェッチ失敗: %s", ticker, e)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    def _download_data(self, lookback_days: int = 600) -> tuple[dict, pd.Series]:
        """
        ユニバース全銘柄とベンチマークの株価を取得する。

        取得優先順位:
          1. バッチ一括ダウンロード（yfinance）
          2. 個別リトライ（sleep jitter付き）
          3. ローカルparquetキャッシュ（フォールバック）

        RSR欠損耐性:
          - Close系列の欠損を ffill(limit=3) で補完
          - 3日超の欠損銘柄は RSR計算から除外
          - メトリクス: rsr_missing_count / rsr_filled_count / rsr_excluded_count
        """
        import warnings
        warnings.filterwarnings("ignore")

        end_date   = datetime.now(JST).strftime("%Y-%m-%d")
        start_date = (datetime.now(JST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        logger.info("データ取得: %s 〜 %s", start_date, end_date)

        all_tickers = list(
            set(self.rsr_universe_tickers.keys())
            | set(self.universe_tickers.keys())
            | set(self.shadow_universe_tickers.keys())
        ) + [self.benchmark_ticker]

        raw = pd.DataFrame()
        if ALLOW_YFINANCE_NETWORK:
            raw = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
            )

        # ── Step1: snapshot → 当日バッチ(yfinance) → キャッシュはフォールバックのみ ──
        # 2026-07-08 RCA: 旧実装はキャッシュ(最大5日以内なら有効)をバッチ取得結果より
        # 先に採用しており、当日既に取得済みの新鮮なbatchデータを毎回捨てていた
        # (docstring自体が「1.バッチ 2.個別リトライ 3.キャッシュ」の優先順を明記しており矛盾)。
        # data_as_of がT-1のはずが数日分古く表示・使用され続けたのはこれが原因。
        all_universe = {
            **self.rsr_universe_tickers,
            **self.universe_tickers,
            **self.shadow_universe_tickers,
        }
        universe_raw:    dict = {}
        batch_failed:    list[str] = []
        cache_fallbacks: list[str] = []

        for sym, sector in all_universe.items():
            df = self._load_from_snapshot(sym, start_date, end_date)
            if not df.empty and len(df) >= 252:
                snap_age = (datetime.now(JST).date() - df.index[-1].date()).days
                if snap_age <= SNAPSHOT_MAX_STALE_DAYS:
                    universe_raw[sym] = {"df": df, "sector": sector}
                    continue
                logger.info("%s snapshot陳腐化（%d日前）→ ネットワーク/キャッシュ優先", sym, snap_age)

            df = pd.DataFrame()
            from_batch = False
            if ALLOW_YFINANCE_NETWORK and not raw.empty:
                try:
                    df = raw[sym].copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df = df.droplevel(1, axis=1)
                    df.dropna(subset=["Close"], inplace=True)
                    if not df.empty and len(df) >= 252:
                        from_batch = True
                    else:
                        batch_failed.append(sym)
                except (KeyError, TypeError):
                    batch_failed.append(sym)
            else:
                batch_failed.append(sym)

            if from_batch:
                self._save_to_cache(sym, df)
                universe_raw[sym] = {"df": df, "sector": sector}
            # from_batch=False の銘柄は batch_failed に積まれ済み。
            # キャッシュへのフォールバックは Step3 の個別リトライ後に一本化する
            # （ここで先にフォールバックすると Step3 のリトライが常に上書きするだけの
            #  無駄な二重キャッシュ読みになるため行わない）。

        # ── Step3: バッチ失敗分を個別リトライ → それでも駄目ならキャッシュ ──
        if batch_failed and ALLOW_YFINANCE_NETWORK:
            logger.warning("バッチ失敗: %d銘柄 → 個別リトライ+キャッシュ試行", len(batch_failed))
        for sym in batch_failed if ALLOW_YFINANCE_NETWORK else []:
            sector = all_universe[sym]
            df = self._retry_single_fetch(sym, start_date, end_date)
            if not df.empty and len(df) >= 252:
                self._save_to_cache(sym, df)
                universe_raw[sym] = {"df": df, "sector": sector}
            else:
                df_cached = self._load_from_cache(sym, start_date)
                if not df_cached.empty and len(df_cached) >= 252:
                    cache_fallbacks.append(sym)
                    universe_raw[sym] = {"df": df_cached, "sector": sector}
                else:
                    logger.warning("%s フォールバック失敗 → RSR除外", sym)

        if cache_fallbacks:
            logger.info("キャッシュフォールバック: %s", cache_fallbacks)

        # ── Step4: RSR用Close行列を構築し、欠損を ffill(limit=3) で補完 ──
        # RSRはユニバース内順位型なので欠損=別銘柄のランクが歪む
        rsr_syms = [s for s in self.rsr_universe_tickers if s in universe_raw]
        if rsr_syms:
            close_matrix = pd.DataFrame(
                {s: universe_raw[s]["df"]["Close"] for s in rsr_syms}
            )
            # 欠損カウント（補完前）
            missing_before = close_matrix.isna().sum()
            rsr_missing_count = int((missing_before > 0).sum())

            # ffill(limit=3) 補完
            close_filled = close_matrix.ffill(limit=3)
            filled_cells  = (close_matrix.isna() & close_filled.notna())
            rsr_filled_count = int((filled_cells.any()).sum())

            # 最新日が補完後もNaNの銘柄 → RSR計算から除外
            latest_na = close_filled.iloc[-1].isna()
            rsr_excluded_syms = latest_na[latest_na].index.tolist()
            rsr_excluded_count = len(rsr_excluded_syms)

            # 補完済みCloseをuniverse_rawに反映（RSR計算用銘柄のみ）
            for sym in rsr_syms:
                if sym in rsr_excluded_syms:
                    logger.warning("%s RSR欠損>3日 → RSR計算除外", sym)
                    # universe_rawからは除かない（SELLシグナル生成には残す）
                    universe_raw[sym]["rsr_excluded"] = True
                else:
                    universe_raw[sym]["df"]["Close"] = close_filled[sym]

            logger.info(
                "RSR欠損補完: missing=%d filled=%d excluded=%d",
                rsr_missing_count, rsr_filled_count, rsr_excluded_count,
            )
            # diagnostics用に保持
            self._last_data_health = {
                "rsr_missing_count":  rsr_missing_count,
                "rsr_filled_count":   rsr_filled_count,
                "rsr_excluded_count": rsr_excluded_count,
                "cache_fallback_syms": cache_fallbacks,
            }
        else:
            self._last_data_health = {
                "rsr_missing_count": 0, "rsr_filled_count": 0,
                "rsr_excluded_count": 0, "cache_fallback_syms": [],
            }

        # ── Step5: ベンチマーク取得（split-adjusted 優先） ───────────────────
        # snapshot は株式分割後に混在データになるため yfinance auto_adjust=True を最優先
        df_bench = pd.DataFrame()
        if ALLOW_YFINANCE_NETWORK:
            try:
                _adj = yf.download(
                    self.benchmark_ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if isinstance(_adj.columns, pd.MultiIndex):
                    _adj = _adj.droplevel(1, axis=1)
                _adj.dropna(subset=["Close"], inplace=True)
                if not _adj.empty:
                    df_bench = _adj
                    logger.info(
                        "ベンチマーク: yfinance auto_adjust 取得 %d行 last=%.2f",
                        len(df_bench), float(df_bench["Close"].iloc[-1]),
                    )
            except Exception as _e:
                logger.warning("ベンチマーク yfinance auto_adjust 失敗: %s → フォールバック", _e)
        if df_bench.empty:
            df_bench = self._load_from_snapshot(self.benchmark_ticker, start_date, end_date)
        if df_bench.empty:
            df_bench = self._load_from_cache(self.benchmark_ticker, start_date)
        if df_bench.empty and not raw.empty:
            try:
                df_bench = raw[self.benchmark_ticker].copy()
                if isinstance(df_bench.columns, pd.MultiIndex):
                    df_bench = df_bench.droplevel(1, axis=1)
                df_bench.dropna(subset=["Close"], inplace=True)
            except (KeyError, TypeError):
                pass
        if df_bench.empty and ALLOW_YFINANCE_NETWORK:
            df_bench = self._retry_single_fetch(
                self.benchmark_ticker, start_date, end_date
            )
        if df_bench.empty and universe_raw:
            # benchmark不在時はローカル銘柄群の平均終値を代用し、処理継続を優先する
            bench_close = pd.DataFrame(
                {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
            ).mean(axis=1)
            df_bench = pd.DataFrame({"Close": bench_close.dropna()})

        bench_prices_series = _apply_split_correction(df_bench, "1306.T", logger)["Close"]

        logger.info(
            "取得完了: %d / %d 銘柄（バッチ失敗=%d キャッシュ使用=%d）",
            len(universe_raw), len(all_universe) - 1,
            len(batch_failed), len(cache_fallbacks),
        )
        return universe_raw, bench_prices_series

    # ------------------------------------------------------------------ #
    # 現在のポジション取得
    # ------------------------------------------------------------------ #
    def _virtual_available_cash(self, current_positions: dict) -> float:
        n_held = len(current_positions)
        n_free = max(0, self.max_positions - n_held)
        # deployable_capital（実資産ベース）が設定されていればそちらを優先する。
        # 未設定（0.0）の場合は self.capital（有効資本）にフォールバック。
        base = self.deployable_capital if self.deployable_capital > 0 else self.capital
        return base / max(1, self.max_positions) * n_free

    def _get_current_positions(self) -> dict[str, dict]:
        if self._client is None:
            self._positions_api_status = {
                "ok": False,
                "source": "virtual",
                "error": "client_unavailable",
            }
            logger.warning("positions API: skipped (client unavailable, virtual portfolio)")
            return {}

        try:
            raw_positions = self._client.get_positions()
        except Exception as exc:
            self._positions_api_status = {
                "ok": False,
                "source": "broker_error",
                "error": str(exc),
            }
            logger.error("positions API: failed: %s", exc)
            if self.live:
                raise RuntimeError(f"positions API 取得失敗: {exc}") from exc
            return {}

        from src.common.position_normalizer import filter_live_positions
        live_positions = filter_live_positions(raw_positions)

        positions = {}
        for p in live_positions:
            sym_code = p.get("Symbol", "")
            sym = f"{sym_code}.T"
            if sym in self.universe_tickers:
                positions[sym] = {
                    "qty":       p.get("LeavesQty", 0),
                    "avg_price": p.get("Price", 0.0),
                }
        self._positions_api_status = {"ok": True, "source": "broker", "error": None}
        logger.info(
            "positions API: success (%d live positions, %d raw)",
            len(positions), len(raw_positions),
        )
        return positions

    def _recover_entry_price_from_broker(self, symbol: str) -> float:
        """
        send_result に estimated_price が無い場合の最終リカバリ（fail-closed用）。

        Broker の実際のポジション平均取得単価(avg_price)を1回だけ問い合わせる。
        取得できない場合は 0.0 を返す（呼び出し側が entry_metadata_missing として
        監査ログに記録し、0.0を実価格としてportfolio_stateへ書き込むことはしない）。
        例外は握りつぶす — update_state_after_execution() 全体を失敗させない。
        """
        if self._client is None:
            return 0.0
        try:
            positions = self._get_current_positions()
            pos = positions.get(symbol)
            if pos and float(pos.get("avg_price", 0.0)) > 0:
                logger.info(
                    "[ENTRY_PRICE_RECOVERY] %s: broker avg_price=%.2f で復元",
                    symbol, float(pos["avg_price"]),
                )
                return float(pos["avg_price"])
        except Exception as exc:
            logger.warning("[ENTRY_PRICE_RECOVERY] %s: broker avg_price取得失敗: %s", symbol, exc)
        return 0.0

    # ------------------------------------------------------------------ #
    # 資金確認
    # ------------------------------------------------------------------ #
    def _get_available_cash(self, current_positions: dict) -> float | None:
        if self._client is None:
            virtual_cash = self._virtual_available_cash(current_positions)
            self._wallet_api_status = {
                "ok": False,
                "source": "virtual",
                "error": "client_unavailable",
            }
            logger.warning(
                "wallet API: skipped (client unavailable, virtual cash=¥%s)",
                f"{virtual_cash:,.0f}",
            )
            return virtual_cash

        try:
            wallet = self._client.get_wallet_cash()
            if "StockAccountWallet" not in wallet:
                raise RuntimeError("StockAccountWallet missing")
            available_cash = float(wallet["StockAccountWallet"])
        except Exception as exc:
            self._wallet_api_status = {
                "ok": False,
                "source": "broker_error",
                "error": str(exc),
            }
            logger.error("wallet API: failed: %s", exc)
            if self.live:
                raise RuntimeError(f"wallet API 取得失敗: {exc}") from exc
            return None

        self._wallet_api_status = {"ok": True, "source": "broker", "error": None}
        logger.info("wallet API: success (available_cash=¥%s)", f"{available_cash:,.0f}")
        return available_cash

    # ------------------------------------------------------------------ #
    # シグナル生成（Top-k ローテーション + 時間ストップ）
    # ------------------------------------------------------------------ #
    def _generate_all_signals(
        self,
        universe_raw:      dict,
        bench_prices:      pd.Series,
        current_positions: dict,
        portfolio_state:   dict,
    ) -> tuple[list[StockSignal], list[str], dict]:
        """
        全銘柄のシグナルを生成する。

        Returns:
            (signals, top_k_symbols)
        """
        from src.backtest.rsr                     import calc_universe_rsr, calc_rsr_momentum, calc_sepa
        from src.backtest.fujiko_strategy         import FujikoStrategy
        from src.backtest.mean_reversion_strategy import MeanReversionStrategy
        from src.strategy.universe import get_today_active_syms

        today     = pd.Timestamp.now().normalize()
        today_str = today.strftime("%Y-%m-%d")

        # ── RSR 計算（62銘柄コンテキスト: 42 live + 20 shadow）────────────
        # 2026-03-24 バックテスト検証: RSR62 で Calmar +28% / Sharpe +9% 確認済み。
        # rsr_universe_tickers に shadow を含めることで競争母集団が広がり
        # 通過銘柄の「本物の相対強度」が上がる（エントリー数微減・質向上）
        rsr_prices = {
            sym: info["df"]["Close"]
            for sym, info in universe_raw.items()
            if sym in self.rsr_universe_tickers
        }
        rsr_universe = calc_universe_rsr(rsr_prices).ffill(limit=3)
        rsr_latest   = rsr_universe.iloc[-1].fillna(0.0)   # 最新スナップショット

        # RSR cross-sectional percentile (observation-only logging — no decision impact)
        _rsr_pct_raw: dict[str, float] = {}
        _rsr_pct_smooth: dict[str, float] = {}
        try:
            from src.execution.execution_metrics import compute_rsr_percentile
            _rsr_pct_raw, _rsr_pct_smooth = compute_rsr_percentile(rsr_universe)
        except Exception as _rpe:
            logger.debug("rsr_percentile compute skipped: %s", _rpe)

        # ── dyn_rsr42_bear_rs0: 今日の活性銘柄セットを取得 ─────────────
        # RSR42のみで月次スコアリング → Bull Top30 / Bear Top20(rs>0)
        # 空セットの場合はフィルターなし（フォールバック）
        _rsr42_prices = {
            sym: info["df"]["Close"]
            for sym, info in universe_raw.items()
            if sym in self.universe_tickers   # trade universe = RSR42
        }
        _rsr42_df = calc_universe_rsr(_rsr42_prices).ffill(limit=3)
        _bear_cfg = getattr(
            getattr(self._cfg, "risk_controls", None), "bear_universe_filter", None
        ) if self._cfg is not None else None
        _bear_exclude = (
            list(_bear_cfg.excluded_sectors)
            if (_bear_cfg is not None and _bear_cfg.enabled)
            else None
        )
        _dyn_active_syms = get_today_active_syms(
            universe_raw         = {s: universe_raw[s] for s in self.universe_tickers if s in universe_raw},
            topix_close          = bench_prices,
            rsr_df               = _rsr42_df,
            all_syms             = list(self.universe_tickers.keys()),
            end                  = today_str,
            bear_exclude_sectors = _bear_exclude,
            sym_sector_map       = dict(self.universe_tickers) if _bear_exclude else None,
        )
        logger.info(
            "dyn_rsr42_bear_rs0 活性銘柄: %d / %d 銘柄 %s",
            len(_dyn_active_syms), len(self.universe_tickers),
            sorted(_dyn_active_syms)[:10],
        )

        # ── クラスタ相場検出（早期計算 — シグナルループ・Shadow昇格より前に必要）────
        # Step1/2/3 で参照するため rsr_latest 取得直後に計算する
        _pre_ctx_size      = max(1, len(rsr_latest))
        _pre_rsr_gt80      = int((rsr_latest >= 80).sum())
        _pre_density       = _pre_rsr_gt80 / _pre_ctx_size
        # 2段階クラスタレベル
        #   level 0: 通常相場
        #   level 1 (density >= CLUSTER_LEVEL1_THRESH): mean_rev 新規 BUY 停止
        #   level 2 (density >= CLUSTER_LEVEL2_THRESH): momentum 偏重 + Shadow昇格厳格化
        if _pre_density >= CLUSTER_LEVEL2_THRESH:
            _pre_cluster_level = 2
        elif _pre_density >= CLUSTER_LEVEL1_THRESH:
            _pre_cluster_level = 1
        else:
            _pre_cluster_level = 0
        _pre_cluster_mode  = _pre_cluster_level >= 1  # 後方互換（level1以上 = cluster_mode）
        # ↑ 後段の _trend_cluster_mode と同値。ログは後段でまとめて出力する。

        # ── Shadow Universe RSR（監視専用・RSR42母集団と独立して計算）─────
        # RSR計算母集団（42銘柄）は絶対に変更しない。
        # Shadow poolは独自の相対強度順位を持ち、rsr_pass >= 65 の銘柄数を診断ログに記録する。
        _shadow_rsr_pass  = 0
        _shadow_near_bo   = 0
        _shadow_promo_list: list[str] = []
        if self.shadow_universe_tickers:
            _shadow_prices = {
                sym: info["df"]["Close"]
                for sym, info in universe_raw.items()
                if sym in self.shadow_universe_tickers
            }
            if len(_shadow_prices) >= 5:
                try:
                    _shadow_rsr_u      = calc_universe_rsr(_shadow_prices)
                    _shadow_rsr_latest = _shadow_rsr_u.iloc[-1]
                    _shadow_rsr_pass   = int((_shadow_rsr_latest >= 65).sum())
                    for _ssym in _shadow_rsr_latest[_shadow_rsr_latest >= 65].index:
                        if _ssym not in universe_raw:
                            continue
                        try:
                            _sc = universe_raw[_ssym]["df"]["Close"]
                            if len(_sc) >= 21:
                                _sh20 = float(_sc.iloc[-21:-1].max())
                                _sp   = float(_sc.iloc[-1])
                                if _sh20 > 0 and _sp >= _sh20 * 0.92:
                                    _shadow_near_bo += 1
                        except Exception:
                            pass
                    logger.info(
                        "Shadow Universe: pool=%d rsr_pass(>=65)=%d near_breakout=%d",
                        len(_shadow_prices), _shadow_rsr_pass, _shadow_near_bo,
                    )
                    # 昇格候補: Shadow pool内で RSR閾値超え かつ 価格<=¥8,000（1単元≤¥800,000）
                    # cluster level2: RSR>=90・top2（真の強者のみ昇格）
                    # cluster level1: RSR>=80・top4
                    # 通常時:         RSR>=68・top6
                    # スコア = RSR + 0.5 × RSRモメンタム（コツコツ上昇を優遇）
                    if _pre_cluster_level >= 2:
                        _SHADOW_PROMO_RSR   = 90.0
                        _SHADOW_PROMO_LIMIT = 2
                    elif _pre_cluster_level == 1:
                        _SHADOW_PROMO_RSR   = 80.0
                        _SHADOW_PROMO_LIMIT = 4
                    else:
                        _SHADOW_PROMO_RSR   = 68.0
                        _SHADOW_PROMO_LIMIT = 6
                    _SHADOW_PROMO_PRICE = 8_000
                    _shadow_promo_cands: list[tuple[float, str, float]] = []
                    for _sp_sym in _shadow_rsr_latest[_shadow_rsr_latest >= _SHADOW_PROMO_RSR].index:
                        if _sp_sym in universe_raw:
                            try:
                                _sp_price = float(universe_raw[_sp_sym]["df"]["Close"].iloc[-1])
                                if _sp_price <= _SHADOW_PROMO_PRICE:
                                    _sp_rsr   = float(_shadow_rsr_latest[_sp_sym])
                                    # RSRモメンタム（21日+5日加重）を composite score に加算
                                    # mom = 0.7×mom21 + 0.3×mom5 で急騰ピーク銘柄を抑制し
                                    # 「上昇中の初動銘柄」を優遇する
                                    try:
                                        _sp_rsr_series = _shadow_rsr_u[_sp_sym].dropna()
                                        _n = len(_sp_rsr_series)
                                        _mom21 = float(_sp_rsr_series.iloc[-1] - _sp_rsr_series.iloc[-22]) if _n >= 22 else 0.0
                                        _mom5  = float(_sp_rsr_series.iloc[-1] - _sp_rsr_series.iloc[-6])  if _n >= 6  else 0.0
                                        _sp_mom = 0.7 * _mom21 + 0.3 * _mom5
                                    except Exception:
                                        _sp_mom = 0.0
                                    _sp_score = _sp_rsr + 0.5 * _sp_mom
                                    _shadow_promo_cands.append((_sp_score, _sp_sym, round(_sp_price, 0)))
                            except Exception:
                                pass
                    _shadow_promo_cands.sort(reverse=True)
                    _shadow_promo_list = [s for _, s, _ in _shadow_promo_cands[:_SHADOW_PROMO_LIMIT]]
                    if _shadow_promo_list:
                        logger.info(
                            "Shadow昇格候補(%d銘柄): %s",
                            len(_shadow_promo_list), _shadow_promo_list,
                        )
                except Exception as _se:
                    logger.debug("Shadow RSR計算エラー: %s", _se)
                    _shadow_promo_list = []
        else:
            _shadow_promo_list = []

        # MTF Filter 除去済み（2026-06-24, Study33 IS-1.56pp/Study35 OOS両方でProven Negative）。
        # 診断フィールド構造のみ維持（常に空 → 下流の mtf_* 集計は 0 のまま、KeyError なし）。
        _rsr_weekly_map: dict = {}
        _weekly_ma_ok_map: dict = {}

        # Step 2 (観測バイアス): RSRコンテキスト全体の通過数を記録
        # 売買ユニバースが RSR42 の部分集合のため、強い銘柄が価格フィルターで
        # 除外されると diag_rsr_pass が低いまま固定される問題を検出する。
        # rsr_pass_tradeable_ratio = diag_rsr_pass / _rsr_pass_context_total
        # 0.6 以下 → 高 RSR 銘柄の多くが売買不可（価格 or 流動性フィルターで除外）
        _min_rsr_for_ctx = float(self._fujiko_params_live["min_rsr"])
        _rsr_pass_context_total = int((rsr_latest >= _min_rsr_for_ctx).sum())

        # ── RSR集中度（62銘柄コンテキスト全体での分布）─────────────────
        _ctx_size          = max(1, len(rsr_latest))
        _rsr_gt80_context  = int((rsr_latest >= 80).sum())
        _rsr_gt70_context  = int((rsr_latest >= 70).sum())
        _rsr_top_share     = round(_rsr_gt80_context / _ctx_size, 3)
        # 2段階クラスタレベル（_pre_cluster_level と同値。ここでは後段ログ用に再参照）
        _trend_cluster_mode  = _pre_cluster_level >= 1   # 後方互換
        _trend_cluster_level = _pre_cluster_level        # 2段階制御に使用
        if _trend_cluster_level == 2:
            logger.warning(
                "TREND_CLUSTER_MODE level2: RSR80以上=%d/%d (%.1f%%) >= %.0f%% "
                "→ momentum偏重 + mean_rev全停止 + Shadow昇格RSR≥90",
                _rsr_gt80_context, _ctx_size, _rsr_top_share * 100,
                CLUSTER_LEVEL2_THRESH * 100,
            )
        elif _trend_cluster_level == 1:
            logger.warning(
                "TREND_CLUSTER_MODE level1: RSR80以上=%d/%d (%.1f%%) >= %.0f%% "
                "→ mean_rev BUY停止 + Shadow昇格RSR≥80",
                _rsr_gt80_context, _ctx_size, _rsr_top_share * 100,
                CLUSTER_LEVEL1_THRESH * 100,
            )

        logger.info(
            "RSR コンテキスト: %d 銘柄（統一42銘柄）context_pass=%d "
            "RSR80以上=%d(%.1f%%) RSR70以上=%d cluster_level=%d",
            len(rsr_prices), _rsr_pass_context_total,
            _rsr_gt80_context, _rsr_top_share * 100,
            _rsr_gt70_context, _trend_cluster_level,
        )

        # ── 流動性スコア（Volume × Close の 20 日平均） ───────────────
        liquidity: dict[str, float] = {}
        for sym in rsr_prices:
            if sym not in universe_raw:
                continue
            df = universe_raw[sym]["df"]
            if "Volume" not in df.columns:
                continue
            liq = float((df["Close"] * df["Volume"]).tail(20).mean())
            if liq >= MIN_DAILY_VALUE_YEN:
                liquidity[sym] = liq

        logger.info(
            "流動性フィルター通過: %d / %d 銘柄（≥¥%s/日）",
            len(liquidity), len(rsr_prices), f"{MIN_DAILY_VALUE_YEN:,.0f}",
        )

        # ── min_rsr 閾値（research と同じ filter-first アーキテクチャ） ──────
        # top_k は最後にBUY候補をRSR順にソートして絞るために使う（一次フィルターではない）
        min_rsr_threshold = float(self._fujiko_params_live["min_rsr"])

        # RSR 順位マップ（全銘柄、1=最高）
        rsr_rank_map: dict[str, int] = {}
        for sym, rank in rsr_latest.rank(ascending=False).items():
            if pd.isna(rank):
                continue
            rsr_rank_map[sym] = int(rank)

        # ── 再エントリー禁止リスト ──────────────────────────────────
        reentry_blocked: dict[str, str] = portfolio_state.get("reentry_blocked", {})
        active_blocked  = {sym for sym, end in reentry_blocked.items() if today_str < end}
        if active_blocked:
            logger.info("再エントリー禁止銘柄: %s", active_blocked)

        # ── 保有エントリー日マップ ───────────────────────────────────
        pos_entry_dates:    dict[str, str]   = portfolio_state.get("position_entry_dates",    {})
        pos_entry_prices:   dict[str, float] = portfolio_state.get("position_entry_prices",   {})
        pos_entry_atrs:     dict[str, float] = portfolio_state.get("position_entry_atrs",     {})
        pos_highest_closes: dict[str, float] = portfolio_state.get("position_highest_closes", {})

        # ── 診断カウンター ────────────────────────────────────────
        diag_total        = 0   # 非保有・非ブロック銘柄数（BUY候補の母数）
        diag_rsr_pass     = 0   # RSR閾値通過数
        diag_blocked_rsr  = 0   # RSRで弾かれた数
        diag_blocked_bo   = 0   # RSR通過後にBreakout/SEPA/momentumで弾かれた数
        diag_rsr_dist: list[dict] = []   # RSR分布（全非保有銘柄）
        # Step 2: supply ceiling（RSR>70/60/50 の分布）
        diag_rsr_gt70 = 0
        diag_rsr_gt60 = 0
        diag_rsr_gt50 = 0
        # Step 3: Turtle breakout期間比較（戦略変更なし・ログのみ）
        diag_bo15_pass = 0   # RSR通過 かつ 15日ブレイクなら通過したはずの数
        diag_bo10_pass = 0   # RSR通過 かつ 10日ブレイクなら通過したはずの数
        # Step 4: ブレイクアウト直前銘柄（20日高値の2%以内）
        diag_near_breakout   = 0
        # False Breakout診断（entry後5日以内 かつ -2ATR到達）
        diag_failed_breakout  = 0
        # MTFフィルター（実際に適用）: RSR日次>=75 AND 週足RSR>=70 AND 週足MA20
        # mtf_candidates = RSR日次通過数 / mtf_pass = 3条件すべて通過数
        diag_mtf_candidates   = 0   # RSR日次>=75 の銘柄数（MTF対象母数）
        diag_mtf_filtered     = 0   # 週足MA20フィルターで落ちた数（後方互換）
        diag_mtf_wrsr_pass    = 0   # 週足RSR>=70 通過数
        diag_mtf_wma_pass     = 0   # 週足MA20 通過数
        diag_mtf_full_pass    = 0   # 3条件すべて通過数（実エントリー候補）

        # ── composite shock exit: 前日ベンチマーク騰落率を事前計算 ──────
        _bench_ret_prev = 0.0
        _is_shock_day   = False
        if self.shock_exit_mode == "composite":
            try:
                _bench_ret_prev = float(bench_prices.pct_change().iloc[-1])
                _is_shock_day   = _bench_ret_prev <= -0.05
                if _is_shock_day:
                    logger.warning(
                        "COMPOSITE_SHOCK_DAY: bench_ret=%.2f%% → 個別-8%%超ポジションを決済対象に追加",
                        _bench_ret_prev * 100,
                    )
            except Exception:
                pass

        # ── market_trend: TOPIX(1306.T) > MA200 ────────────────────
        _bench_ma200 = bench_prices.rolling(200, min_periods=1).mean()
        market_trend = bool(bench_prices.iloc[-1] > _bench_ma200.iloc[-1])
        logger.info(
            f"[market_trend_debug] bench_last={bench_prices.iloc[-1]:.2f} "
            f"ma200={_bench_ma200.iloc[-1]:.2f} "
            f"market_trend={market_trend} "
            f"bench_len={len(bench_prices)} "
            f"bench_last_date={bench_prices.index[-1] if hasattr(bench_prices.index, '__len__') else 'no_index'}"
        )

        # ── 特徴量エンジニアリング: 必須カラムをuniverse_rawに付与 ──────────
        # rsr_252 / rsr_63 / ma20 / ma50 / ma20_slope / avg_turnover_20d
        # これらはシグナル生成・missing_col チェックに必須だが
        # _download_data では生OHLCV のみ保持 → ここで一括付与する
        logger.info("特徴量エンジニアリング: %d 銘柄に必須カラムを付与", len(universe_raw))

        # --- rsr_63 cross-sectional ranking (横断ランキング) ---
        _returns_63 = {}
        for _s, _i in universe_raw.items():
            _dfc = _i.get("df")
            if _dfc is None or len(_dfc) < 64:
                continue
            _close63 = _dfc["Close"] if "Close" in _dfc.columns else _dfc["close"]
            _returns_63[_s] = float(_close63.iloc[-1] / _close63.iloc[-64] - 1)

        _rank_63 = pd.Series(_returns_63).rank(pct=True)

        for _s, _i in universe_raw.items():
            if _s in _rank_63:
                _i["df"].loc[:, "rsr_63"] = float(_rank_63[_s])
            else:
                _i["df"].loc[:, "rsr_63"] = float("nan")

        for _sym, _info in universe_raw.items():
            _df = _info["df"]
            if _df.empty:
                continue
            _close  = _df["Close"]
            _volume = _df["Volume"] if "Volume" in _df.columns else pd.Series(0, index=_df.index)

            # MA 系
            _ma20       = _close.rolling(20, min_periods=10).mean()
            _ma50       = _close.rolling(50, min_periods=25).mean()
            _ma20_slope = _ma20.diff(5)
            _ma20_up    = (_ma20_slope > 0).astype(float)

            # 流動性: 売買代金20日平均
            _turnover = (_close * _volume).rolling(20, min_periods=10).mean()

            # rsr_252: 62銘柄コンテキストRSR（rsr_universe は calc_composite_return ≒ 252d）
            if _sym in rsr_universe.columns:
                _rsr252_s = rsr_universe[_sym].reindex(_df.index).ffill()
            else:
                _rsr252_s = pd.Series(float("nan"), index=_df.index)

            _df["ma20"]             = _ma20
            _df["ma50"]             = _ma50
            _df["ma20_slope"]       = _ma20_slope
            _df["ma20_up"]          = _ma20_up
            _df["avg_turnover_20d"] = _turnover
            _df["rsr_252"]          = _rsr252_s

        logger.info("特徴量エンジニアリング完了")
        # ─────────────────────────────────────────────────────────────────

        # ── シグナル生成ループ ───────────────────────────────────────
        signals: list[StockSignal] = []

        # ── 必須カラム欠損チェック（銘柄単位除外） ──────────────────────────
        _required_cols = [
            "Close", "Open", "High", "Low", "Volume",
            "rsr_252", "rsr_63", "ma20", "ma50", "ma20_slope",
            "avg_turnover_20d",
        ]
        _universe_before = len(universe_raw)
        _clean_universe: dict = {}
        for _sym, _info in universe_raw.items():
            _df_check = _info.get("df", pd.DataFrame())
            if _df_check.empty:
                logger.warning(f"[missing_col] {_sym}: df is empty, skipping")
                continue
            _missing = [c for c in _required_cols if c not in _df_check.columns]
            if _missing:
                logger.warning(
                    f"[missing_col] {_sym}: missing={_missing}, skipping (count=1)"
                )
                continue
            _clean_universe[_sym] = _info
        _dropped = _universe_before - len(_clean_universe)
        if _dropped:
            logger.warning(
                f"[missing_col] dropped={_dropped} symbols due to missing cols, "
                f"remaining={len(_clean_universe)}"
            )
        universe_raw = _clean_universe
        # ────────────────────────────────────────────────────────────────────
        # Pre-pass: compute for ALL universe_raw symbols so trend_follow signals
        # (which include shadow/non-universe_tickers symbols) also get a value.
        _sym_dist25ma: dict[str, float | None] = {}
        for _s, _i in universe_raw.items():
            _ps = _i["df"]["Close"]
            _ms = _ps.rolling(25).mean()
            _ma = float(_ms.iloc[-1]) if pd.notna(_ms.iloc[-1]) else float(_ps.iloc[-1])
            _sym_dist25ma[_s] = round((float(_ps.iloc[-1]) - _ma) / _ma, 6) if _ma > 0 else None

        for sym, info in universe_raw.items():
            if sym not in self.universe_tickers:
                continue  # RSR 計算専用の銘柄はシグナル生成しない

            df     = info["df"]
            _price_s        = df["Close"]
            _ma25_s         = _price_s.rolling(25).mean()
            ma25            = float(_ma25_s.iloc[-1]) if pd.notna(_ma25_s.iloc[-1]) else float(_price_s.iloc[-1])
            ma25_slope      = bool(_ma25_s.diff(5).iloc[-1] > 0)
            ret_5d          = float(_price_s.pct_change(5).iloc[-1])
            trend_filter    = (float(_price_s.iloc[-1]) > ma25) or ma25_slope
            momentum_filter = ret_5d > 0
            entry_ok        = trend_filter and momentum_filter
            block_buy       = False
            sector = info["sector"]
            rsr    = rsr_universe[sym].ffill(limit=3) if sym in rsr_universe.columns else None

            rsr_now  = float(rsr.iloc[-1]) if rsr is not None and not rsr.empty and pd.notna(rsr.iloc[-1]) else 0.0
            mom      = calc_rsr_momentum(rsr, self.fujiko_params.get("mom_period", 21)) if rsr is not None else None
            mom_now  = float(mom.iloc[-1]) if mom is not None and not mom.empty and pd.notna(mom.iloc[-1]) else 0.0
            sepa_df  = calc_sepa(df, rsr if rsr is not None else pd.Series(50.0, index=df.index))
            sepa_now = int(sepa_df["sepa_score"].iloc[-1])
            rsr_rank = rsr_rank_map.get(sym, 99)

            currently_holding = sym in current_positions
            # entry_date_known=False は「未保有」ではなく「保有中だがentry_date欠損」を示す。
            # 2026-07-15 entry metadata SSOT修正: 表示層が0dへ誤フォールバックしないための判別フラグ。
            entry_date_known = (not currently_holding) or (sym in pos_entry_dates)

            # ── 保有営業日数チェック ─────────────────────────────────
            hold_td      = 0
            is_time_exit = False
            if (currently_holding
                    and sym in pos_entry_dates
                    and self.max_hold_days is not None):
                hold_td      = _trading_days_held(pos_entry_dates[sym], today)
                is_time_exit = hold_td >= self.max_hold_days

            # ── トレーリングストップ（3×ATR20） ────────────────────────────
            # 優先度: 最高（時間ストップより先に判定）
            # 更新タイミング: 毎朝データ取得後に highest_close を更新してから判定
            #   → 当日 open で寄成 SELL を生成する
            is_trailing_stop = False
            _ts_close        = 0.0
            _ts_atr20        = 0.0
            _ts_highest      = 0.0
            _ts_stop_price   = 0.0
            if currently_holding:
                from src.portfolio.volatility_allocator import calc_atr as _calc_atr
                _ts_close  = float(df["Close"].iloc[-1])
                _ts_atr20  = _calc_atr(df, period=20)
                _prev_high = pos_highest_closes.get(sym, _ts_close)
                _ts_highest = max(_prev_high, _ts_close)
                pos_highest_closes[sym] = _ts_highest   # 毎朝 highest_close を更新（先に更新→後で判定）
                if not np.isnan(_ts_atr20) and _ts_atr20 > 0:
                    _ts_stop_price   = _ts_highest - 3.0 * _ts_atr20
                    is_trailing_stop = _ts_close < _ts_stop_price

            # ── FujikoStrategy（min_rsr=75 で研究と同じ閾値フィルター） ────
            fujiko_strat = FujikoStrategy(
                rsr_series       = rsr,
                benchmark_prices = bench_prices,
                **self._fujiko_params_live,
            )
            f_signal = fujiko_strat.generate_signal(df)

            # 平均回帰シグナル
            mr_strat = MeanReversionStrategy(**MR_PARAMS)
            m_signal = mr_strat.generate_signal(df)

            # ── filter-first アーキテクチャ（research と同じ） ───────────
            # 優先順位: トレーリングストップ > 時間ストップ > RSR低下エグジット > mean_rev反発失敗 > 再エントリー禁止 > 戦略シグナル

            # 緊急 exit 判定（含み損 emergency_exit_pct 以下でmin_holdを無視）
            # 対象: 決算ギャップダウン / 市場急落 など
            is_emergency_exit = False
            if (currently_holding
                    and sym in pos_entry_prices
                    and pos_entry_prices[sym] > 0
                    and _ts_close > 0):
                _pnl_pct = (_ts_close - pos_entry_prices[sym]) / pos_entry_prices[sym]
                if _pnl_pct <= self.emergency_exit_pct:
                    is_emergency_exit = True
                    logger.warning(
                        "EMERGENCY_EXIT %s pnl=%.1f%% (threshold=%.1f%%) hold=%d",
                        sym, _pnl_pct * 100, self.emergency_exit_pct * 100, hold_td,
                    )

            # シンプル閾値エグジット（rsr_exit=70）: backtest engine の rsr_exit_thr と同一ロジック
            _simple_rsr_exit = (rsr_now < self.rsr_exit_threshold)
            _ml_tag = f"RSR_THRESHOLD(rsr={rsr_now:.1f}<{self.rsr_exit_threshold:.0f})"
            is_rank_exit = (currently_holding
                            and _simple_rsr_exit
                            and (hold_td >= self.min_hold_days or is_emergency_exit))

            # ── mean_rev 反発未発生検出（早期撤退）────────────────────────
            # エントリー後 MEANREV_FAIL_DAYS 営業日で High が +MEANREV_MIN_BOUNCE 未到達 → SELL
            # 平均回帰戦略のエントリー後に反発が発生しない場合、構造的下落の可能性が高い
            is_meanrev_fail  = False
            _meanrev_bounce  = 0.0
            _meanrev_strategy_type: str | None = None
            if (currently_holding
                    and sym in pos_entry_prices
                    and sym in pos_entry_dates):
                # このシンボルの保有戦略タイプを portfolio_state から取得
                _pos_strategy = portfolio_state.get("position_strategy_types", {}).get(sym, "")
                if _pos_strategy == "mean_rev":
                    _mr_entry_price = float(pos_entry_prices[sym])
                    _mr_entry_date  = pos_entry_dates[sym]
                    try:
                        _entry_ts    = pd.Timestamp(_mr_entry_date)
                        _high_since  = float(df.loc[df.index >= _entry_ts, "High"].max()) if _mr_entry_price > 0 else 0.0
                        _close_now   = float(df["Close"].iloc[-1])
                        _meanrev_bounce = (_high_since - _mr_entry_price) / max(1.0, _mr_entry_price)
                        # 2条件AND: 反発未到達 かつ 現在値がエントリー比-0.5%以下
                        # 後者はギャップダウン後の弱トレンド継続を捕捉する
                        _below_entry = _close_now < _mr_entry_price * 0.995
                        if (hold_td >= MEANREV_FAIL_DAYS
                                and _meanrev_bounce < MEANREV_MIN_BOUNCE
                                and _below_entry):
                            is_meanrev_fail = True
                            _meanrev_strategy_type = "mean_rev"
                    except Exception:
                        pass

            # ── composite shock exit（最高優先度）────────────────────────────
            is_shock_exit = False
            if currently_holding and _is_shock_day:
                try:
                    _sym_ret_prev = float(
                        universe_raw[sym]["df"]["Close"].pct_change().iloc[-1]
                    )
                    if _sym_ret_prev <= -0.08:
                        is_shock_exit = True
                        logger.warning(
                            "COMPOSITE_SHOCK_EXIT %s sym_ret=%.2f%% (bench=%.2f%%)",
                            sym, _sym_ret_prev * 100, _bench_ret_prev * 100,
                        )
                except Exception:
                    pass

            if is_shock_exit:
                signal_int    = -1
                strategy_type = "fujiko"
                reason = (
                    f"SELL[composite_shock]: bench={_bench_ret_prev:.2%}"
                    f" sym={_sym_ret_prev:.2%} ≤ -8%"
                )

            elif is_trailing_stop:
                signal_int    = -1
                strategy_type = "fujiko"
                reason = (
                    f"SELL[トレーリングストップ]: close={_ts_close:.0f}"
                    f" < peak={_ts_highest:.0f} - 3×ATR({_ts_atr20:.0f})"
                    f" → stop={_ts_stop_price:.0f}"
                )
                logger.info("EXIT %s reason=TRAIL_EXIT hold=%d rsr=%.1f", sym, hold_td, rsr_now)

            elif is_time_exit:
                signal_int    = -1
                strategy_type = "fujiko"
                reason = (
                    f"SELL[時間ストップ]: {hold_td}営業日保有"
                    f"（上限{self.max_hold_days}日） RSR={rsr_now:.1f} rank={rsr_rank}"
                )
                logger.info("EXIT %s reason=TIME_STOP hold=%d rsr=%.1f", sym, hold_td, rsr_now)

            elif is_rank_exit:
                signal_int    = -1
                strategy_type = "fujiko"
                reason = (
                    f"SELL[多層RSR]: {_ml_tag} rank={rsr_rank} hold={hold_td}d"
                )
                _exit_reason_tag = "EMERGENCY_EXIT" if is_emergency_exit else "RSR_EXIT_ML"
                logger.info("EXIT %s reason=%s hold=%d rsr=%.1f tag=%s", sym, _exit_reason_tag, hold_td, rsr_now, _ml_tag)

            elif is_meanrev_fail:
                signal_int    = -1
                strategy_type = "mean_rev"
                reason = (
                    f"SELL[meanrev_fail]: {hold_td}日保有"
                    f" bounce={_meanrev_bounce:+.2%}<{MEANREV_MIN_BOUNCE:.1%}"
                    f" close<entry×0.995 → 反発未発生・構造的下落と判断"
                )

            elif sym in active_blocked:
                signal_int    = 0
                strategy_type = "fujiko"
                reason = (
                    f"HOLD[再エントリー禁止〜{reentry_blocked[sym]}]:"
                    f" rank={rsr_rank} RSR={rsr_now:.1f}"
                )

            else:
                # 全銘柄で戦略を実行（FujikoStrategy が RSR≥min_rsr, SEPA, breakout を内部フィルター）
                reason = ""   # 未初期化参照防止（line 1375 の reason.startswith ガード）
                rule = SECTOR_STRATEGY.get(sector, "dynamic")
                if rule == "fujiko":
                    signal_int, strategy_type = f_signal, "fujiko"
                elif rule == "mean_rev":
                    signal_int, strategy_type = m_signal, "mean_rev"
                    # TREND_CLUSTER_MODE中はmean_rev BUYをブロック
                    # 理由: クラスタ相場 = 資金が特定セクターに集中 = 逆張りは構造的弱化に逆らう
                    # 2026-03-16の5411.T(鉄鋼)エントリーがこのパターンで損失
                    block_buy = _pre_cluster_mode and (not market_trend)
                    if signal_int == 1 and block_buy:
                        signal_int    = 0
                        strategy_type = "mean_rev"
                        reason = (
                            f"HOLD[cluster_block]: {sector}(mean_rev) BUYブロック"
                            f" cluster_mode=True rsr80_share={_pre_rsr_gt80}/{_pre_ctx_size}"
                        )
                else:  # dynamic
                    if f_signal == 1:
                        signal_int, strategy_type = 1, "fujiko"
                    elif m_signal == 1:
                        signal_int, strategy_type = 1, "mean_rev"
                    elif f_signal == -1 or m_signal == -1:
                        signal_int    = -1
                        strategy_type = "fujiko" if f_signal == -1 else "mean_rev"
                    else:
                        signal_int, strategy_type = 0, "fujiko"

                # MTF Filter 除去済み（2026-06-24, Proven Negative）。BUYシグナルへのエントリー制限なし。

                strat_label = "フジコ法" if strategy_type == "fujiko" else "平均回帰"
                if signal_int == 1:
                    reason = (
                        f"BUY[{strat_label}]: RSR={rsr_now:.1f} rank={rsr_rank}"
                        f" SEPA={sepa_now} mom={mom_now:+.1f}"
                    )
                elif signal_int == -1:
                    reason = (
                        f"SELL[{strat_label}]: RSR={rsr_now:.1f} mom={mom_now:+.1f}"
                    )
                    if currently_holding:
                        logger.info("EXIT %s reason=STRATEGY_EXIT hold=%d rsr=%.1f", sym, hold_td, rsr_now)
                elif signal_int == 0:
                    reason = (
                        f"HOLD: RSR={rsr_now:.1f} rank={rsr_rank}"
                        f" SEPA={sepa_now} ({strat_label})"
                    )

            # ── 診断カウント（非保有・非ブロック銘柄のみ集計） ────────
            if not currently_holding and not is_time_exit and not is_rank_exit and sym not in active_blocked:
                diag_total += 1
                diag_rsr_dist.append({"symbol": sym, "rsr": round(rsr_now, 1)})

                # Step 2: supply ceiling カウント（RSR分布）
                if rsr_now > 70: diag_rsr_gt70 += 1
                if rsr_now > 60: diag_rsr_gt60 += 1
                if rsr_now > 50: diag_rsr_gt50 += 1

                if rsr_now >= min_rsr_threshold:
                    diag_rsr_pass += 1

                    # Step 2: MTFフィルター診断（後方互換: 週足MA20不通過なら count）
                    # キャッシュ参照（計算不要）
                    if not _weekly_ma_ok_map.get(sym, True):
                        diag_mtf_filtered += 1

                    # Step 4: ブレイク前圧力（RSR通過銘柄全体・signal問わず計測）
                    # 定義: close >= 20日高値 × 0.97（3%以内）
                    # BUYシグナル済み銘柄も含める → 市場全体のブレイク圧力を早期検知
                    try:
                        _close_s   = df["Close"]
                        _price_now = float(_close_s.iloc[-1])
                        if len(_close_s) >= 21:
                            _high_20 = float(_close_s.iloc[-21:-1].max())
                            if _high_20 > 0 and _price_now >= _high_20 * 0.97:
                                diag_near_breakout += 1
                    except Exception:
                        pass

                    if signal_int == 0:
                        diag_blocked_bo += 1   # RSR通過済みなのにBUYにならない = Breakout/SEPA/Mom

                        # Step 3: Turtle期間比較（現行20日 vs 15日 vs 10日）
                        # 戦略には影響しない診断ログのみ
                        try:
                            close_s   = df["Close"]
                            price_now = float(close_s.iloc[-1])
                            if len(close_s) >= 16:
                                high_15 = float(close_s.iloc[-16:-1].max())
                                if price_now > high_15:
                                    diag_bo15_pass += 1   # 15日なら通過したはず
                            if len(close_s) >= 11:
                                high_10 = float(close_s.iloc[-11:-1].max())
                                if price_now > high_10:
                                    diag_bo10_pass += 1   # 10日なら通過したはず
                        except Exception:
                            pass
                else:
                    diag_blocked_rsr += 1

            # ── False Breakout診断（保有中 かつ entry後5営業日以内 かつ -2ATR到達）─
            if currently_holding and not is_time_exit and not is_rank_exit:
                _ep  = pos_entry_prices.get(sym, 0.0)
                _ea  = pos_entry_atrs.get(sym, 0.0)
                if _ep > 0 and _ea > 0 and hold_td <= 5:
                    try:
                        _price_now = float(df["Close"].iloc[-1])
                        if _price_now < _ep - 2.0 * _ea:
                            diag_failed_breakout += 1
                    except Exception:
                        pass

            if signal_int == 1 and not entry_ok:
                signal_int = 0
                reason = (
                    f"entry_filter: trend_ok={trend_filter} mom_ok={momentum_filter}"
                    f" ma25={ma25:.1f} ret5d={ret_5d:.3f}"
                )
            logger.info(
                f"entry_ok={entry_ok} block_buy={block_buy} "
                f"market_trend={market_trend} cluster_active={_pre_cluster_mode} "
                f"symbol={sym} price_vs_ma25={'above' if float(_price_s.iloc[-1])>ma25 else 'below'} "
                f"ma25_slope={ma25_slope} ret_5d={ret_5d:.3f}"
            )
            logger.info(
                f"[signal_out] symbol={sym} sector={sector} "
                f"signal={signal_int} strategy={strategy_type} "
                f"entry_rank=N/A entry_score=N/A"
            )
            _ep_now = pos_entry_prices.get(sym, 0.0)
            _pnl_pct_now = (
                round((_ts_close - _ep_now) / _ep_now, 4)
                if (currently_holding and _ep_now > 0 and _ts_close > 0)
                else 0.0
            )
            signals.append(StockSignal(
                symbol               = sym,
                sector               = sector,
                signal               = signal_int,
                rsr                  = rsr_now,
                rsr_rank             = rsr_rank,
                sepa_score           = sepa_now,
                rsr_mom              = mom_now,
                hold_days            = hold_td,
                currently_holding    = currently_holding,
                reason               = reason,
                strategy_type        = strategy_type,
                trailing_stop_price  = _ts_stop_price,
                entry_price          = _ep_now,
                unrealized_pnl_pct   = _pnl_pct_now,
                entry_date_known     = entry_date_known,
            ))

        # highest_close の更新内容を portfolio_state に書き戻す（_save_portfolio_state で永続化）
        portfolio_state["position_highest_closes"] = pos_highest_closes

        # BUY 候補を RSR+モメンタム複合スコアでソートして top_k 個に絞る
        # composite = RSR + RSR_momentum × MOM_WEIGHT_ADJ [+ Entry Timing boost]
        # 効果: 上昇加速中の銘柄が同RSRの停滞銘柄より優先される（モメンタム相場で有効）
        # MOM_WEIGHT_ADJ=0.3 → "RSRモメンタム1.0→1.3倍"相当
        # level0=0.3（通常）/ level1=0.5（mean_rev停止相場）/ level2=0.7（完全トレンド支配）
        # level2(density>=25%)は市場がトレンド一辺倒になっているため
        # モメンタム加速中の上位1〜3銘柄への集中度を最大化する
        _MOM_WEIGHT_ADJ = 0.3
        if _trend_cluster_level == 1:
            _MOM_WEIGHT_ADJ = 0.5
        elif _trend_cluster_level >= 2:
            _MOM_WEIGHT_ADJ = 0.7

        # ── Entry Timing Engine (補助レイヤー: RSR選択後のタイブレーカー) ──────
        # RSR主権を維持しつつ、エントリー品質が高い銘柄を僅かに優先する
        # boost範囲: (score-50) * 0.06 → score差100点で ±3pt（RSRスケールと比較して小）
        _et_cfg              = self.fujiko_params.get("entry_timing", {})
        _et_enabled          = bool(_et_cfg.get("enabled", True))
        _et_boost_w_default  = float(_et_cfg.get("boost_weight", 0.06))
        _et_auto_apply       = bool(_et_cfg.get("auto_apply_boost", False))
        # auto_apply_boost=true のとき promotion tier の boost_weight を採用（ASK_FIRST）
        if _et_auto_apply:
            try:
                from src.entry.entry_timing_promotion import get_effective_boost_weight as _et_gbw
                from src.paths import ENTRY_TIMING_PROMOTION_FILE as _et_promo_file
                _et_boost_w = _et_gbw(
                    _et_promo_file,
                    fallback_weight=_et_boost_w_default,
                    auto_apply_enabled=True,
                )
            except Exception as _et_gbw_err:
                logger.warning("[ET] get_effective_boost_weight FAIL_OPEN: %s", _et_gbw_err)
                _et_boost_w = _et_boost_w_default
        else:
            _et_boost_w = _et_boost_w_default
        _et_results: dict = {}
        if _et_enabled:
            try:
                from src.entry import (
                    compute_entry_timing_for_candidates as _et_compute,
                    apply_entry_timing_boost as _et_boost,
                )
                _et_cands_raw = [
                    (0.0, s.symbol)
                    for s in signals if s.signal == 1 and not s.currently_holding
                ]
                _sigs_map = {s.symbol: s for s in signals}
                _et_results = _et_compute(
                    buy_eligible        = _et_cands_raw,
                    universe_raw        = universe_raw,
                    rsr_universe        = rsr_universe,
                    signals_map         = _sigs_map,
                    trend_cluster_level = _trend_cluster_level,
                    universe_rsr_latest = rsr_latest,
                    enabled             = _et_enabled,
                )
                logger.info(
                    "[ET] scored %d candidates  HIGH=%d MEDIUM=%d LOW=%d",
                    len(_et_results),
                    sum(1 for r in _et_results.values() if r.confidence == "HIGH"),
                    sum(1 for r in _et_results.values() if r.confidence == "MEDIUM"),
                    sum(1 for r in _et_results.values() if r.confidence == "LOW"),
                )
            except Exception as _et_err:
                logger.warning("[ET] computation FAIL_OPEN: %s", _et_err)
                _et_results = {}

        def _et_adj(sym: str) -> float:
            if not _et_enabled or sym not in _et_results:
                return 0.0
            try:
                from src.entry import apply_entry_timing_boost as _etb
                return _etb(0.0, _et_results[sym], _et_boost_w, _et_enabled)
            except Exception:
                return 0.0

        _buy_eligible_all = sorted(
            [(s.rsr + s.rsr_mom * _MOM_WEIGHT_ADJ + _et_adj(s.symbol), s.symbol)
             for s in signals if s.signal == 1 and not s.currently_holding],
            reverse=True,
        )

        # block_low_confidence: LOW判定シグナルをエントリー候補から除外
        # デフォルト false → 現行挙動と完全互換（ASK_FIRST 対象）
        _et_block_low = bool(_et_cfg.get("block_low_confidence", False))
        if _et_block_low and _et_enabled and _et_results:
            _before_block = len(_buy_eligible_all)
            _buy_eligible_all = [
                (sc, sym) for sc, sym in _buy_eligible_all
                if sym not in _et_results
                or _et_results[sym].confidence != "LOW"
            ]
            _blocked_low = _before_block - len(_buy_eligible_all)
            if _blocked_low:
                logger.info("[ET] block_low_confidence: %d 銘柄除外", _blocked_low)

        _raw_buy_count = sum(1 for s in signals if s.signal == 1)
        _holding_excluded_count = sum(
            1 for s in signals if s.signal == 1 and s.currently_holding
        )
        # dyn_rsr42_bear_rs0 フィルター: 活性銘柄のみ BUY 対象
        # 空セットはフォールバック（フィルターなし: 活性銘柄計算失敗時の安全策）
        if _dyn_active_syms:
            buy_eligible = [
                (sc, sym) for sc, sym in _buy_eligible_all if sym in _dyn_active_syms
            ]
            _dyn_filtered = len(_buy_eligible_all) - len(buy_eligible)
            if _dyn_filtered:
                logger.info(
                    "dyn フィルター: %d → %d 件（%d 件除外）",
                    len(_buy_eligible_all), len(buy_eligible), _dyn_filtered,
                )
            if not buy_eligible:
                if not _buy_eligible_all:
                    # dyn フィルター適用前から候補ゼロ（upstream ボトルネック）
                    # 原因: rsr_pass=%d, blocked_bo=%d, mtf_full_pass=%d, holding_excluded=%d
                    logger.info(
                        "BUY_ZERO upstream: rsr_pass=%d blocked_rsr=%d blocked_bo=%d "
                        "mtf_candidates=%d mtf_full_pass=%d holding_excluded=%d → dyn フィルター到達前にゼロ",
                        diag_rsr_pass, diag_blocked_rsr, diag_blocked_bo,
                        diag_mtf_candidates, diag_mtf_full_pass, _holding_excluded_count,
                    )
                else:
                    # dyn フィルターが全候補を除外した（発生頻度は低い）→ フォールバック
                    logger.warning(
                        "dyn フィルター後 BUY 候補ゼロ → フォールバック（全候補使用）"
                        " upstream=%d dyn_active=%d",
                        len(_buy_eligible_all), len(_dyn_active_syms),
                    )
                    buy_eligible = _buy_eligible_all
        else:
            buy_eligible = _buy_eligible_all
        # 4スロット目エントリーゲート
        # positions < 3: RSR >= 75（通常閾値＝min_rsr_threshold）
        # positions == 3: RSR >= 80（4枚目は質的厳選）
        # positions > 3:  発生しない（max_positions=4, top_k=4）
        _n_holdings = len(current_positions)
        _filtered_top_k: list[str] = []
        for _, _sym in buy_eligible:
            if len(_filtered_top_k) >= self.top_k:
                break
            _sym_rsr = float(rsr_latest.get(_sym, 0.0))
            if _n_holdings == 3:
                if _sym_rsr < 80.0:
                    logger.info(
                        "4slot_gate BLOCKED %s rsr=%.1f < 80.0 (holdings=%d)",
                        _sym, _sym_rsr, _n_holdings,
                    )
                    continue
            # positions < 3: min_rsr_threshold (75) は上流フィルターで保証済み
            _filtered_top_k.append(_sym)
        top_k_syms = _filtered_top_k
        _today_lock = _load_order_lock().get(today_str, {})
        _duplicate_excluded_count = sum(
            1 for sym in top_k_syms if _today_lock.get(sym) == "BUY"
        )
        _new_entries_count = max(0, len(top_k_syms) - _duplicate_excluded_count)
        # avg_tradeable_rsr: 全フィルター通過後のBUY候補のRSR平均（0件なら None）
        _buy_elig_rsrs = [rsr for rsr, _ in buy_eligible]
        _avg_tradeable_rsr = round(float(np.mean(_buy_elig_rsrs)), 1) if _buy_elig_rsrs else None
        logger.info(
            "BUY 候補（RSR順）: %s → top%d = %s",
            [sym for _, sym in buy_eligible], self.top_k, top_k_syms,
        )
        logger.info(
            "BUY_FUNNEL raw_buy=%d topk=%d new_entries=%d",
            _raw_buy_count, len(top_k_syms), _new_entries_count,
        )

        rsr_dist_sorted = sorted(diag_rsr_dist, key=lambda x: x["rsr"], reverse=True)
        diagnostics = {
            "universe_size":      diag_total,
            "rsr42_total":        len(rsr_prices),  # RSR42母集団の実データ取得数（supply_ratio の分母）
            "rsr_pass":           diag_rsr_pass,
            "blocked_rsr":        diag_blocked_rsr,
            "blocked_breakout":   diag_blocked_bo,
            "buy_candidates":     len(buy_eligible),
            "topk_count":         len(top_k_syms),
            "raw_buy_count":      _raw_buy_count,
            "holding_excluded_count": _holding_excluded_count,
            "duplicate_excluded_count": _duplicate_excluded_count,
            "new_entries_count":  _new_entries_count,
            "avg_tradeable_rsr":  _avg_tradeable_rsr,
            "rsr_distribution":   rsr_dist_sorted[:20],
            # Step 2: supply ceiling（live 28銘柄内）
            "rsr_gt80_context":    _rsr_gt80_context,    # 62銘柄中RSR80以上
            "rsr_gt70_context":    _rsr_gt70_context,    # 62銘柄中RSR70以上
            "rsr_top_share":       _rsr_top_share,        # RSR80以上の比率
            "trend_cluster_mode":  _trend_cluster_mode,   # True = クラスタ相場（level>=1）
            "trend_cluster_level": _trend_cluster_level,  # 0=通常 / 1=mean_rev停止 / 2=momentum偏重
            "rsr_gt70":           diag_rsr_gt70,
            "rsr_gt60":           diag_rsr_gt60,
            "rsr_gt50":           diag_rsr_gt50,
            # Step 3: Turtle期間比較（何銘柄が15日/10日で追加通過するか）
            "bo15_extra":         diag_bo15_pass,   # 15日なら追加通過する銘柄数
            "bo10_extra":         diag_bo10_pass,   # 10日なら追加通過する銘柄数
            # Step 4: ブレイクアウト直前（20日高値の2%以内）
            "near_breakout":      diag_near_breakout,
            "shadow_rsr_pass":        _shadow_rsr_pass,
            "shadow_near_bo":         _shadow_near_bo,
            "shadow_promo_candidates": _shadow_promo_list,
            # Shadow RSR62スコア（発注条件チェック用 / shadow_rsr > live_top10_median に使用）
            # RSR62コンテキスト（全62銘柄内での順位）なので live RSR と直接比較可能
            "shadow_rsr62_scores": {
                sym: round(float(rsr_latest[sym]), 1)
                for sym in self.shadow_universe_tickers
                if sym in rsr_latest.index
            },
            # False Breakout: 保有中 かつ entry後5日以内 かつ -2ATR到達した銘柄数
            "failed_breakout":    diag_failed_breakout,
            # MTFフィルター（実適用）: 3条件 pass/候補
            "mtf_filtered":       diag_mtf_filtered,     # 後方互換（週足MA20不通過数）
            "mtf_candidates":     diag_mtf_candidates,   # RSR日次>=75 のBUY候補数
            "mtf_wrsr_pass":      diag_mtf_wrsr_pass,    # 週足RSR>=70 通過数
            "mtf_wma_pass":       diag_mtf_wma_pass,     # 週足MA20 通過数
            "mtf_full_pass":      diag_mtf_full_pass,    # 3条件すべて通過（実エントリー候補）
            # RSR percentile (observation-only: backtest/live gap quantification)
            "rsr_pct_raw":    _rsr_pct_raw,    # {sym: cross-sectional percentile [0,1]}
            "rsr_pct_smooth": _rsr_pct_smooth, # {sym: EMA(3) of percentile over time}
            # Entry Timing Intelligence (observation-only unless block_low_confidence=true)
            "entry_timing_enabled": _et_enabled,
            "entry_timing_scores": {
                sym: {
                    "score":      round(r.score, 1),
                    "confidence": r.confidence,
                    "action":     r.action,
                    "phase":      r.phase,
                    "bq":         r.breakout_component,
                    "pb":         r.pullback_component,
                    "tr":         r.trend_component,
                    "mk":         r.market_component,
                }
                for sym, r in _et_results.items()
            },
            "sym_dist25ma": _sym_dist25ma,
        }

        # Position Sizing Intelligence (観測専用: Phase 1) は TREND_FOLLOW 統合 +
        # Priority dedup 後に評価する（下方の "Position Sizing Intelligence 評価" 参照）。
        # 理由: trend_follow経由のBUY信号は signals に追加されるのが後段のため、
        # ここで評価すると conviction_score=None のまま出力され、
        # position_sizing_telemetry.jsonl が永久に生成されない不具合があった。
        # (RCA: 2026-06-22 PSP n30=0 root cause)

        # ── 構造的ボトルネック診断（シグナルループ後に一括計算） ──────
        # Step 1: RSR Top10 のうち売買不可銘柄数・重み
        #   blocked_leaders_weight > 0.2（20%超）→ ブレイクアウト期待値崩れの可能性
        # Step 2: 売買不可の理由を price / liquidity / risk に分離
        #   実務で最多は blocked_by_price → 戦略変更ではなくポジションサイズ設計の問題
        # Step 3: RSR Top10 のうち売買可能割合（rsr_top10_tradeable_ratio）
        #   < 0.5 → リーダー集中相場（高価格銘柄主導）
        _blocked_leaders_count    = 0
        _blocked_leaders_weight   = 0.0
        _rsr_top10_tradeable_cnt  = 0
        _blocked_by_price         = 0
        _blocked_by_liquidity     = 0
        _blocked_by_risk          = 0
        _blocked_price_rsr_scores: list[float] = []  # blocked_rsr_mean 計算用
        # Step 2（価格分布ログ）・Step 3（距離ログ）・Step 4〜6（追加監視ログ）
        _rsr_top10_median_price   = None   # RSR Top10 の価格中央値
        _rsr_top10_max_price      = None   # RSR Top10 の最高価格
        _high20_distance_median   = None   # median((high20 - close) / high20) RSR通過全銘柄
        _rsr_top10_sector_count   = None   # RSR Top10 に何セクターあるか（相場拡散の先行指標）
        _mid_pressure_count       = 0      # close >= high20 * 0.90（中間圧力銘柄数）
        _mid_pressure_weight      = 0.0    # mid_pressure銘柄のRSRスコア重み（市場エネルギー）
        _near_breakout_count      = 0      # close >= high20 * 0.92（ブレイク直前圧力 / 8%以内 / bo=0.97より広い）
        _near_breakout_weight     = 0.0    # near_breakout銘柄のRSRスコア重み（>=0.25で相場動く）
        try:
            _rsr_top10 = rsr_latest.nlargest(10)
            _rsr_top10_total_score = float(_rsr_top10.sum()) or 1.0
            for _lsym, _lrsr in _rsr_top10.items():
                if _lsym in self.universe_tickers:
                    _rsr_top10_tradeable_cnt += 1
                else:
                    _blocked_leaders_count += 1
                    _blocked_leaders_weight += float(_lrsr)
            _blocked_leaders_weight = round(_blocked_leaders_weight / _rsr_top10_total_score, 3)

            # Step 2: RSR Top10 の価格分布・セクター集中度
            # median_price 上昇 → リーダー高価格化
            # sector_count 増加 → 相場拡散の先行サイン（ブレイク前によく起きる）
            _top10_prices   = []
            _top10_sectors  = set()
            for _lsym in _rsr_top10.index:
                if _lsym in universe_raw:
                    try:
                        _top10_prices.append(float(universe_raw[_lsym]["df"]["Close"].iloc[-1]))
                    except Exception:
                        pass
                # セクター情報は rsr_universe_tickers か universe_tickers から取得
                _sec = self.rsr_universe_tickers.get(_lsym) or self.universe_tickers.get(_lsym)
                if _sec:
                    _top10_sectors.add(_sec)
            if _top10_prices:
                _rsr_top10_median_price = round(float(np.median(_top10_prices)), 0)
                _rsr_top10_max_price    = round(float(np.max(_top10_prices)), 0)
            _rsr_top10_sector_count = len(_top10_sectors) if _top10_sectors else None

            # Step 3: 20日高値距離の中央値（RSR通過銘柄）
            # 0.06 → 0.03 に縮小するとブレイクアウトクラスター到来のサイン
            # mid_pressure_count: close >= high20 * 0.90（中間圧力 = near_breakoutより早い先行指標）
            _distances = []
            for _dsym in rsr_latest[rsr_latest >= _min_rsr_for_ctx].index:
                if _dsym not in universe_raw:
                    continue
                try:
                    _dclose = universe_raw[_dsym]["df"]["Close"]
                    if len(_dclose) >= 21:
                        _dhigh20 = float(_dclose.iloc[-21:-1].max())
                        _dprice  = float(_dclose.iloc[-1])
                        if _dhigh20 > 0:
                            _dist = (_dhigh20 - _dprice) / _dhigh20
                            _distances.append(_dist)
                            # near_breakout: 高値の8%以内（bo=3%より広い初動検知 / high20_distance=16%の現状に合わせて拡張）
                            if _dprice >= _dhigh20 * 0.92:
                                _near_breakout_count  += 1
                                _near_breakout_weight += float(rsr_latest.get(_dsym, 0))
                            # mid_pressure: 高値の10%以内（near_breakout=5%より早段階）
                            if _dprice >= _dhigh20 * 0.90:
                                _mid_pressure_count  += 1
                                _mid_pressure_weight += float(rsr_latest.get(_dsym, 0))
                except Exception:
                    pass
            if _distances:
                _high20_distance_median = round(float(np.median(_distances)), 4)
            # mid_pressure_weight / near_breakout_weight を正規化（RSR通過銘柄の総スコアで割る）
            _rsr_pass_total_score = float(rsr_latest[rsr_latest >= _min_rsr_for_ctx].sum()) or 1.0
            _mid_pressure_weight  = round(_mid_pressure_weight  / _rsr_pass_total_score, 3)
            _near_breakout_weight = round(_near_breakout_weight / _rsr_pass_total_score, 3)

            # blocked_by_{reason}: RSR >= 閾値 の全銘柄を対象に理由を分類
            for _bsym in rsr_latest[rsr_latest >= _min_rsr_for_ctx].index:
                if _bsym in active_blocked:
                    _blocked_by_risk += 1
                elif _bsym not in self.universe_tickers:
                    _blocked_by_price += 1
                    _blocked_price_rsr_scores.append(float(rsr_latest[_bsym]))
                elif liquidity.get(_bsym, 0) < MIN_DAILY_VALUE_YEN:
                    _blocked_by_liquidity += 1
        except Exception as _e:
            logger.debug("blocked_leaders/price_dist/distance 計算エラー: %s", _e)

        _blocked_rsr_mean = round(float(np.mean(_blocked_price_rsr_scores)), 1) if _blocked_price_rsr_scores else None
        diagnostics["blocked_rsr_mean"]        = _blocked_rsr_mean
        diagnostics["blocked_leaders_count"]   = _blocked_leaders_count
        diagnostics["blocked_leaders_weight"]  = _blocked_leaders_weight
        diagnostics["rsr_top10_tradeable_cnt"] = _rsr_top10_tradeable_cnt
        diagnostics["blocked_by_price"]        = _blocked_by_price
        diagnostics["blocked_by_liquidity"]    = _blocked_by_liquidity
        diagnostics["blocked_by_risk"]         = _blocked_by_risk
        diagnostics["rsr_pass_context_total"]  = _rsr_pass_context_total
        diagnostics["rsr_top10_median_price"]  = _rsr_top10_median_price
        diagnostics["rsr_top10_max_price"]     = _rsr_top10_max_price
        diagnostics["rsr_top10_sector_count"]  = _rsr_top10_sector_count
        diagnostics["high20_distance_median"]  = _high20_distance_median
        diagnostics["mid_pressure_count"]      = _mid_pressure_count
        diagnostics["mid_pressure_weight"]     = _mid_pressure_weight
        diagnostics["near_breakout_count"]     = _near_breakout_count
        diagnostics["near_breakout_weight"]    = _near_breakout_weight

        logger.info(
            "DIAG universe=%d rsr_pass=%d blocked_rsr=%d blocked_breakout=%d candidates=%d topk=%d"
            " | leaders: blocked=%d weight=%.1f%% top10_tradeable=%d"
            " | price=%d liq=%d risk=%d",
            diag_total, diag_rsr_pass, diag_blocked_rsr, diag_blocked_bo,
            len(buy_eligible), len(top_k_syms),
            _blocked_leaders_count, _blocked_leaders_weight * 100, _rsr_top10_tradeable_cnt,
            _blocked_by_price, _blocked_by_liquidity, _blocked_by_risk,
        )

        # ── TREND_FOLLOW 統合 ──────────────────────────────────────────────
        if market_trend:
            _top_n = self.max_positions
            _tf_cands = trend_follow_candidates(
                universe_raw,
                top_n=_top_n,
                logger=logger,
            )
            if not _tf_cands.empty:
                _tf_cands = _tf_cands.reset_index(drop=True)
                _tf_cands["entry_rank"] = range(1, len(_tf_cands) + 1)
                for _tf_i, _row in _tf_cands.iterrows():
                    _tf_sym = _row["symbol"]
                    # 保有中銘柄は新規エントリー候補から除外（per-symbol ループで hold 信号生成済み）
                    if _tf_sym in current_positions:
                        logger.info(
                            f"[trend_follow] {_tf_sym} skipped — currently held"
                        )
                        continue
                    _close_val    = _row.get("close", float("nan"))
                    _required_cash = int(_close_val * 100) if pd.notna(_close_val) and _close_val > 0 else None
                    _entry_rank   = int(_row.get("entry_rank", 0))
                    _entry_score  = float(_row.get("score", float("nan")))
                    if _required_cash is not None:
                        logger.info(f"[debug] {_tf_sym} close={_close_val:.1f} required={_required_cash:,}")
                        if _required_cash > 100_000_000:
                            logger.warning(f"[anomaly] {_tf_sym} required_cash too large: {_required_cash:,}")
                    logger.info(
                        f"[entry] {_tf_sym} reason=trend_follow"
                        f" required={_required_cash:,} rank={_entry_rank} score={_entry_score:.2f}"
                    )
                    signals.append(StockSignal(
                        symbol            = _tf_sym,
                        sector            = str(_row["sector"]),
                        signal            = 1,
                        rsr               = float(_row["entry_score"]),
                        rsr_rank          = int(_tf_i) + 1,
                        sepa_score        = 0,
                        rsr_mom           = 0.0,
                        hold_days         = 0,
                        currently_holding = False,
                        reason            = f"trend_follow fallback={_row['fallback_flag']}",
                        strategy_type     = "trend_follow",
                    ))
        # ────────────────────────────────────────────────────────────────────

        # ── Priority 後処理（同一銘柄の重複シグナルを優先度で1つに絞る） ─────
        _priority_map = {"trend_follow": 3, "breakout": 2, "mean_rev": 1, "fujiko": 0}
        _seen: dict[str, tuple[int, int]] = {}
        for _i, _s in enumerate(signals):
            _stype = getattr(_s, "strategy_type", "fujiko")
            _p = _priority_map.get(_stype, 0)
            if _s.symbol not in _seen or _p > _seen[_s.symbol][1]:
                _seen[_s.symbol] = (_i, _p)
        _keep_idx = {idx for idx, _ in _seen.values()}
        signals = [s for i, s in enumerate(signals) if i in _keep_idx]
        # ────────────────────────────────────────────────────────────────────

        # ── Position Sizing Intelligence 評価 (観測専用: Phase 1) ───────────
        # virtual_weight は telemetry のみ。実発注サイズは変更しない。
        # Phase 2 有効化: strategy.yaml → position_sizing.auto_apply: true (ASK_FIRST)
        # TREND_FOLLOW統合 + Priority dedup 後の最終 signals に対して評価する
        # (trend_follow経由BUY候補も対象に含めるため)。
        _ps_cfg     = self.fujiko_params.get("position_sizing", {})
        _ps_enabled = bool(_ps_cfg.get("enabled", True))
        _ps_results: dict = {}
        if _ps_enabled:
            try:
                from src.portfolio.position_sizing_intelligence import (
                    PositionSizingInput      as _PSI_Input,
                    compute_virtual_weights  as _psi_compute,
                )
                _psi_rsr_pct = diagnostics.get("rsr_pct_smooth", {})
                _psi_inputs  = [
                    _PSI_Input(
                        symbol              = s.symbol,
                        rsr                 = float(s.rsr),
                        sepa_score          = int(s.sepa_score),
                        rsr_momentum        = float(s.rsr_mom),
                        rsr_pct_smooth      = _psi_rsr_pct.get(s.symbol),
                        entry_timing_score  = (
                            float(_et_results[s.symbol].score)
                            if s.symbol in _et_results else None
                        ),
                        future_leader_score = None,   # FAIL_OPEN: not yet at signal time
                    )
                    for s in signals
                    if s.signal == 1 and not s.currently_holding
                ]
                _psi_sigs   = _psi_compute(_psi_inputs)
                _ps_results = {sig.symbol: sig for sig in _psi_sigs}
                if _ps_results:
                    logger.info(
                        "[PSI] scored %d BUY candidates  top=%s(%.0f/%.3f)",
                        len(_ps_results),
                        max(_ps_results, key=lambda k: _ps_results[k].conviction_score),
                        max(_ps_results[k].conviction_score for k in _ps_results),
                        max(_ps_results[k].virtual_weight   for k in _ps_results),
                    )
            except Exception as _psi_err:
                logger.warning("[PSI] computation FAIL_OPEN: %s", _psi_err)
                _ps_results = {}
        diagnostics["ps_results"] = _ps_results   # PositionSizingIntelligence results
        # ────────────────────────────────────────────────────────────────────

        # ── Daily RSR snapshot: persist full RSR scores for T+1 governance ──
        # Governance runs BEFORE SignalBridge each day (runtime dependency order),
        # so today's scores are available as yesterday's snapshot tomorrow.
        try:
            _snap_date = pd.Timestamp.now(tz=JST).strftime("%Y-%m-%d")
            _rsr_snap  = {str(sym): float(rsr_latest[sym]) for sym in rsr_latest.index}
            _export_rsr_snapshot(_snap_date, _rsr_snap)
        except Exception as _snap_err:
            logger.warning("[RSR_SNAPSHOT] evaluate export error: %s", _snap_err)

        return signals, top_k_syms, diagnostics

    # ------------------------------------------------------------------ #
    # 注文生成
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Shadow Phase1 注文生成
    # ------------------------------------------------------------------ #
    def _build_shadow_orders(
        self,
        diag:                    dict,
        universe_raw:            dict,
        current_positions:       dict,
        available_cash:          float,
        cb_active:               bool,
        live_orders:             list,
        shadow_virtual_positions: dict,   # {sym: {"entry_price": float, "entry_date": str, "virtual": True}}
        today_str:               str,
        effective_max_pos:       int,
        above_ma200:             "bool | None" = None,
        shadow_slots:            int   = 1,
        shadow_rsr_min:          float = 70.0,
        shadow_rsr_pass_min:     int   = 8,
    ) -> tuple[list, dict, dict, list]:
        """
        Shadow Universe から条件付き"観測専用"エントリーを生成する（Phase1）。

        条件（すべて満たす場合のみ発動）:
          1. CB NORMAL かつ shadow_rsr_pass >= shadow_rsr_pass_min (=8)
          2. remaining_slots = max_positions - (実保有 + 通常BUY件数) > 0
          3. shadow_rsr62 >= shadow_rsr_min (=70.0)
          4. shadow_rsr62 > live_top10_median
          5. pre_trade_risk_check() 通過（symbol_cap/sector_cap/cluster_cap 等、通常BUYと同一基準）
          6. 価格フィルター: price * 100 <= max_alloc_cap
          7. 未保有 かつ live注文と重複なし

        SAFETY FIX (2026-07-07 4銘柄同時保有インシデント):
          このパスは Broker への実発注を一切行わない（observation_only固定）。
          以前は price/cash 条件を満たすと side="SHADOW_BUY" の実発注可能な
          OrderInstruction を生成し、_send_orders() が Side.BUY として実際に
          kabu API へ送信していた（max_positions / pre_trade_risk_check を
          一切経由しないバイパス経路になっていた）。
          将来 Shadow候補を実発注へ昇格する場合は、必ず通常 BUY パイプライン
          （_build_orders → pre_trade_risk_check → capacity_check → _send_orders）
          へ候補として合流させること。Shadow専用の発注経路は禁止。

        Returns: (orders, shadow_metrics, new_virtual_positions, closed_virtual_syms)
                 orders は常に空リスト（観測専用のため）。
        """
        shadow_metrics = {
            "shadow_signal_count":       0,
            "shadow_entry_count":        0,
            "shadow_blocked_by_alloc":   0,
            "shadow_rsr_pass_met":       False,
            "shadow_virtual_entries":    [],   # 今回新規記録した仮想エントリー
            "shadow_virtual_closed":     [],   # 今回決済した仮想エントリー
            "shadow_virtual_open_count": 0,    # 現在オープン中の仮想エントリー数
        }
        orders: list[OrderInstruction] = []
        new_virtual: dict  = {}   # portfolio_state に追加するもの
        closed_syms: list  = []   # portfolio_state から削除するもの

        _shadow_rsr62 = diag.get("shadow_rsr62_scores", {})

        # ── 仮想エントリーの決済チェック（RSR < shadow_rsr_min で自動決済）───
        for sym, vpos in shadow_virtual_positions.items():
            rsr62 = _shadow_rsr62.get(sym, 0.0)
            if rsr62 < shadow_rsr_min:   # RSR低下 → 仮想決済
                entry_price = vpos.get("entry_price", 0.0)
                if entry_price > 0 and sym in universe_raw:
                    try:
                        exit_price = float(universe_raw[sym]["df"]["Close"].iloc[-1])
                        ret        = round((exit_price / entry_price) - 1, 4)
                        logger.info(
                            "SHADOW_VIRTUAL_CLOSE: %s entry=¥%.0f exit=¥%.0f return=%.2f%%",
                            sym, entry_price, exit_price, ret * 100,
                        )
                        shadow_metrics["shadow_virtual_closed"].append({
                            "symbol":      sym,
                            "entry_price": entry_price,
                            "exit_price":  round(exit_price, 0),
                            "return":      ret,
                            "entry_date":  vpos.get("entry_date"),
                            "exit_date":   today_str,
                        })
                    except Exception:
                        pass
                closed_syms.append(sym)

        shadow_metrics["shadow_virtual_open_count"] = (
            len(shadow_virtual_positions) - len(closed_syms)
        )

        # ── 発動条件チェック ───────────────────────────────────────────────
        _srsr_pass = diag.get("shadow_rsr_pass", 0)
        shadow_metrics["shadow_rsr_pass_met"] = _srsr_pass >= shadow_rsr_pass_min
        if cb_active or _srsr_pass < shadow_rsr_pass_min:
            return orders, shadow_metrics, new_virtual, closed_syms

        # ── capacity_check（max_positions ガード）──────────────────────────
        # 2026-07-07 インシデント: このガードが無かったため実保有2+通常BUY1=3で
        # 満枠にもかかわらず Shadow が4件目を独自に発注してしまった。
        _pending_buy_count = sum(1 for o in live_orders if o.side == "BUY")
        _remaining_slots   = _capacity_check(
            effective_max_pos, len(current_positions), _pending_buy_count,
        )
        shadow_metrics["shadow_remaining_slots"] = _remaining_slots
        if _remaining_slots <= 0:
            logger.info(
                "[SHADOW_CAPACITY_GUARD] remaining_slots=%d (max_positions=%d held=%d pending_buy=%d)"
                " → Shadow候補生成スキップ",
                _remaining_slots, effective_max_pos, len(current_positions), _pending_buy_count,
            )
            return orders, shadow_metrics, new_virtual, closed_syms

        # ── live Top10 RSR 中央値 ──────────────────────────────────────────
        _top10_rsrs = [e["rsr"] for e in diag.get("rsr_distribution", [])[:10]]
        if not _top10_rsrs:
            return orders, shadow_metrics, new_virtual, closed_syms
        _live_top10_median = float(np.median(_top10_rsrs))

        # ── RSR62スコアで候補を絞る ────────────────────────────────────────
        _live_order_syms = {o.symbol for o in live_orders}
        _held_syms       = set(current_positions.keys())
        _max_alloc_cap   = self.capital * self.max_single_weight

        _candidates: list[tuple[float, str]] = []
        for sym, rsr62 in _shadow_rsr62.items():
            if rsr62 < shadow_rsr_min:
                continue
            if rsr62 <= _live_top10_median:
                continue
            if sym in _held_syms or sym in _live_order_syms:
                continue
            _candidates.append((rsr62, sym))

        _candidates.sort(reverse=True)
        shadow_metrics["shadow_signal_count"] = len(_candidates)

        # ── 価格フィルター + risk_check + 仮想エントリー記録（最大 shadow_slots 件）
        # SAFETY FIX: ここでは OrderInstruction を作っても `orders` には絶対に
        # append しない。Shadow は観測専用。実発注は通常BUYパイプラインへの
        # 昇格でのみ行う（Shadow専用の発注経路は禁止）。
        _shadow_slot_used = 0
        for rsr62, sym in _candidates:
            if _shadow_slot_used >= shadow_slots:
                break
            if sym not in universe_raw:
                continue

            _df    = universe_raw[sym]["df"]
            _price = float(_df["Close"].iloc[-1])
            _cost  = _price * 100   # 1単元
            _blocked_by_alloc = _cost > _max_alloc_cap
            if _blocked_by_alloc:
                shadow_metrics["shadow_blocked_by_alloc"] += 1
                logger.info(
                    "SHADOW blocked_by_alloc: %s ¥%.0f/単元 > 上限¥%.0f",
                    sym, _cost, _max_alloc_cap,
                )

            # 余力チェック（送信は行わないため資金拘束はしないが、研究データの
            # 現実性のため affordability は引き続き評価する）
            if not _blocked_by_alloc and available_cash < _cost:
                logger.info("SHADOW 余力不足: %s ¥%.0f > 余力¥%.0f", sym, _cost, available_cash)
                continue

            # ATRベースのロットサイズ（1.25%リスク）— risk_check 評価用の想定数量
            _risk_yen = self.capital * 0.0125
            try:
                _atr = float(_df["Close"].diff().abs().rolling(20).mean().iloc[-1])
                _qty_raw = int(_risk_yen / max(_atr, 1.0))
                _qty     = max(100, (_qty_raw // 100) * 100)
            except Exception:
                _qty = 100
            _qty = min(_qty, int(_max_alloc_cap / _price / 100) * 100)
            _qty = max(100, _qty)

            # pre_trade_risk_check: 通常BUYと同一の symbol_cap/sector_cap/cluster_cap
            # 判定を Shadow候補にも適用する（side="BUY" として評価。実発注はしない）。
            _hypothetical_order = OrderInstruction(
                symbol           = sym,
                symbol_4digit    = sym.replace(".T", ""),
                sector           = self.shadow_universe_tickers.get(sym, "不明"),
                side             = "BUY",
                qty              = _qty,
                order_type       = "MARKET_OPEN",
                estimated_price  = _price,
                estimated_amount = _qty * _price,
                reason           = (
                    f"SHADOW_CANDIDATE: RSR62={rsr62:.1f} "
                    f"(>{_live_top10_median:.1f}=live_top10_median) "
                    f"shadow_rsr_pass={_srsr_pass}"
                ),
                atr20            = 0.0,
            )
            if not self.pre_trade_risk_check(_hypothetical_order, current_positions, universe_raw, above_ma200):
                logger.info("SHADOW risk_check不合格のため候補除外: %s", sym)
                continue

            _shadow_slot_used += 1
            if sym not in shadow_virtual_positions and sym not in _held_syms:
                new_virtual[sym] = {
                    "entry_price": round(_price, 0),
                    "entry_date":  today_str,
                    "virtual":     True,
                    "rsr62":       round(rsr62, 1),
                }
                logger.info(
                    "SHADOW_VIRTUAL_ENTRY: %s @ ¥%.0f RSR62=%.1f (observation_only%s)",
                    sym, _price, rsr62, "・blocked_by_alloc" if _blocked_by_alloc else "",
                )
                shadow_metrics["shadow_virtual_entries"].append({
                    "symbol": sym, "entry_price": round(_price, 0), "rsr62": round(rsr62, 1),
                })

        shadow_metrics["shadow_entry_count"] = 0   # 実発注は常にゼロ（observation_only固定）
        if shadow_metrics["shadow_virtual_entries"]:
            logger.info(
                "SHADOW Phase1 (observation_only): %d件 → %s (rsr62条件: >%.1f AND >%.1f)",
                len(shadow_metrics["shadow_virtual_entries"]),
                [e["symbol"] for e in shadow_metrics["shadow_virtual_entries"]],
                shadow_rsr_min, _live_top10_median,
            )
        return orders, shadow_metrics, new_virtual, closed_syms

    # ------------------------------------------------------------------ #
    # 執行前リスクチェック（二重ガード）
    # ------------------------------------------------------------------ #
    def pre_trade_risk_check(
        self,
        order: "OrderInstruction",
        current_positions: "dict | None" = None,
        universe_raw: "dict | None" = None,
        above_ma200: "bool | None" = None,
    ) -> bool:
        """
        執行前リスクチェック。研究層のcapとは独立した二重ガード。
        Returns False → 発注をスキップ（ログに記録）

        sector_cap     : 同一セクター合計ウェイト 25% 上限
        symbol_cap     : 単一銘柄ウェイト 8% 上限
        cluster_cap    : クラスター合計ウェイト 35% 上限（bear時 25%）
        bear_sector_cap: Bear時のセクター上限 18%（TOPIX < MA200）
        """
        if order.side not in ("BUY",):
            return True  # SELL は常に通す

        _positions = current_positions if current_positions is not None else self._last_current_positions
        _raw       = universe_raw       if universe_raw       is not None else self._last_universe_raw

        def _sym_price(sym: str) -> float:
            if sym in _raw:
                try:
                    return float(_raw[sym]["df"]["Close"].iloc[-1])
                except Exception:
                    pass
            return 0.0

        order_weight = order.estimated_amount / max(1.0, self.capital)

        # ── cap 計算（feature flag: dynamic_cap）──────────────────
        # risk_controls.dynamic_cap が False（production default）の場合は固定 cap を使用。
        # True の場合のみボラティリティ連動の動的 cap を計算する。
        _rc = getattr(self._cfg, "risk_controls", None) if hasattr(self, "_cfg") else None
        _use_dynamic_cap = bool(_rc.dynamic_cap) if _rc is not None else False
        _SYM_CAP_MAX      = float(_rc.symbol_cap)       if _rc is not None else 0.08
        _SEC_CAP_MAX      = float(_rc.sector_cap)       if _rc is not None else 0.25
        _CLUSTER_CAP      = float(getattr(_rc, "cluster_cap",      0.35)) if _rc is not None else 0.35
        _BEAR_SEC_CAP     = float(getattr(_rc, "bear_sector_cap",  0.18)) if _rc is not None else 0.18
        _BEAR_CLUSTER_CAP = float(getattr(_rc, "bear_cluster_cap", 0.25)) if _rc is not None else 0.25

        # ── Bear 判定（TOPIX < MA200）──────────────────────────────
        # above_ma200 が渡されていない場合は universe_raw から 1306.T で推定する。
        _is_bear = False
        if above_ma200 is not None:
            _is_bear = not above_ma200
        elif _raw and "1306.T" in _raw:
            try:
                _topix_c = _raw["1306.T"]["df"]["Close"].dropna()
                if len(_topix_c) >= 100:
                    _is_bear = bool(float(_topix_c.iloc[-1]) < float(_topix_c.rolling(200, min_periods=100).mean().iloc[-1]))
            except Exception:
                pass

        if _use_dynamic_cap:
            _VOL_WINDOW  = 20
            _TARGET_VOL  = 0.15
            _SYM_CAP_MIN = 0.02
            _SEC_CAP_MIN = 0.10

            def _dyn_sym_cap(sym: str) -> float:
                try:
                    c = _raw[sym]["df"]["Close"].pct_change().tail(_VOL_WINDOW).dropna()
                    if len(c) >= 5:
                        v = float(c.std()) * (252 ** 0.5)
                        if v > 0:
                            return max(_SYM_CAP_MIN, min(_SYM_CAP_MAX, 0.5 * _TARGET_VOL / v))
                except Exception:
                    pass
                return _SYM_CAP_MAX

            def _dyn_sec_cap(sector: str) -> float:
                try:
                    members = [
                        s for s, sec in self.universe_tickers.items()
                        if sec == sector and s in _raw
                    ]
                    if not members:
                        return _SEC_CAP_MAX
                    import pandas as _pd
                    idx = _pd.concat(
                        [_raw[s]["df"]["Close"] for s in members], axis=1
                    ).mean(axis=1).pct_change().tail(_VOL_WINDOW).dropna()
                    if len(idx) >= 5:
                        v = float(idx.std()) * (252 ** 0.5)
                        if v > 0:
                            return max(_SEC_CAP_MIN, min(_SEC_CAP_MAX, 1.2 * _TARGET_VOL / v))
                except Exception:
                    pass
                return _SEC_CAP_MAX

            dyn_sector_cap = _dyn_sec_cap(order.sector)
            dyn_symbol_cap = _dyn_sym_cap(order.symbol)
        else:
            # dynamic_cap: false → 固定 cap を使用（production default）
            dyn_sector_cap = _SEC_CAP_MAX
            dyn_symbol_cap = _SYM_CAP_MAX

        # bear adaptive: Bear 時は sector/cluster cap をより絞る
        _eff_sec_cap     = _BEAR_SEC_CAP     if _is_bear else _SEC_CAP_MAX
        _eff_cluster_cap = _BEAR_CLUSTER_CAP if _is_bear else _CLUSTER_CAP
        # dynamic_cap が True の場合は更に tight な方を適用
        if _use_dynamic_cap:
            _eff_sec_cap = min(_eff_sec_cap, dyn_sector_cap)

        # ── セクター集中ゲート（max_names_per_sector / max_weight_per_sector）──
        # 既存の sector_cap（25%/18%）より手前で検査する二重ガード。
        # セクター情報は universe_tickers から取得（portfolio_state に sector フィールドなし）。
        _sc_cfg = getattr(_rc, "sector_concentration", None) if _rc is not None else None
        if _sc_cfg is not None and getattr(_sc_cfg, "enabled", True):
            _order_sector = order.sector
            # 現在保有ポジション中の同一セクター銘柄数
            _sector_names = [
                sym for sym in _positions
                if self.universe_tickers.get(sym) == _order_sector
            ]
            _sector_name_count = len(_sector_names)
            if _sector_name_count >= _sc_cfg.max_names_per_sector:
                logger.info(
                    "sector_block %s sector=%s count=%d/%d",
                    order.symbol, _order_sector,
                    _sector_name_count, _sc_cfg.max_names_per_sector,
                )
                return False

            # セクター合計ウェイト（max_weight_per_sector 上限）
            _sc_sector_val = sum(
                pos.get("qty", 0) * (_sym_price(sym) or float(pos.get("avg_price", 0.0)))
                for sym, pos in _positions.items()
                if self.universe_tickers.get(sym) == _order_sector
            )
            _sc_sector_weight = _sc_sector_val / max(1.0, self.capital)
            if _sc_sector_weight + order_weight > _sc_cfg.max_weight_per_sector:
                logger.info(
                    "sector_block %s sector=%s weight=%.3f/%.3f",
                    order.symbol, _order_sector,
                    _sc_sector_weight + order_weight, _sc_cfg.max_weight_per_sector,
                )
                return False

        # ── セクターウェイト ──────────────────────────────────────────
        sector_val = sum(
            pos.get("qty", 0) * (_sym_price(sym) or float(pos.get("avg_price", 0.0)))
            for sym, pos in _positions.items()
            if self.universe_tickers.get(sym) == order.sector
        )
        sector_weight = sector_val / max(1.0, self.capital)

        if sector_weight + order_weight > _eff_sec_cap:
            logger.warning(
                "RISK_CHECK_REJECT sector_cap: %s sector=%s "
                "current=%.3f order=%.3f cap=%.3f(bear=%s)",
                order.symbol, order.sector, sector_weight, order_weight, _eff_sec_cap, _is_bear,
            )
            return False

        # ── 銘柄ウェイト ──────────────────────────────────────────────
        symbol_val = sum(
            pos.get("qty", 0) * (_sym_price(sym) or float(pos.get("avg_price", 0.0)))
            for sym, pos in _positions.items()
            if sym == order.symbol
        )
        symbol_weight = symbol_val / max(1.0, self.capital)

        if symbol_weight + order_weight > dyn_symbol_cap:
            logger.warning(
                "RISK_CHECK_REJECT symbol_cap: %s current=%.3f order=%.3f cap=%.3f(dyn)",
                order.symbol, symbol_weight, order_weight, dyn_symbol_cap,
            )
            return False

        # ── クラスター集中キャップ（cluster_cap / bear_cluster_cap）──────
        try:
            from src.strategy.cluster import CLUSTER_MAP_DEFAULT, get_cluster, calc_cluster_weight
            _cluster = get_cluster(order.symbol, self.universe_tickers, CLUSTER_MAP_DEFAULT)
            if _cluster != "other":
                _pos_weights = {
                    sym: pos.get("qty", 0) * (_sym_price(sym) or float(pos.get("avg_price", 0.0))) / max(1.0, self.capital)
                    for sym, pos in _positions.items()
                }
                _cw = calc_cluster_weight(_pos_weights, self.universe_tickers, CLUSTER_MAP_DEFAULT)
                _cluster_w = _cw.get(_cluster, 0.0)
                if _cluster_w + order_weight > _eff_cluster_cap:
                    logger.warning(
                        "RISK_CHECK_REJECT cluster_cap: %s cluster=%s "
                        "current=%.3f order=%.3f cap=%.3f(bear=%s)",
                        order.symbol, _cluster, _cluster_w, order_weight, _eff_cluster_cap, _is_bear,
                    )
                    return False
        except Exception as _e:
            logger.debug("cluster_cap check skipped: %s", _e)

        return True

    def _build_orders(
        self,
        signals:              list[StockSignal],
        universe_raw:         dict,
        current_positions:    dict,
        available_cash:       float,
        cb_active:            bool,
        today_new_buys:       int = 0,
        effective_max_pos:    int | None = None,
        above_ma200:          bool | None = None,
        audit_sink:           "list | None" = None,
    ) -> tuple[list[OrderInstruction], list[str], int, int]:
        """
        シグナルリストからポートフォリオルールを適用して注文を生成する。

        Args:
            cb_active:      True のとき新規 BUY を全停止
            today_new_buys: 本日すでに実行済みの新規 BUY 件数（スクリプト再実行時に使用）
            audit_sink:     None以外の場合、各BUY候補についてステージごとの
                            PASS/FAIL判定を .append() する（観測専用・戻り値の
                            形は一切変更しない。2026-06-29 EVS RCA follow-up）。

        Returns:
            (orders, order_warnings, blocked_alloc_cap_count, lot_rounded_up_count)
            常に 4 要素タプル。CB 早期リターンも含め全パスが同じ形を返す。
        """
        def _audit(symbol: str, stage: str, passed: bool, reason: str, **extra) -> None:
            if audit_sink is None:
                return
            audit_sink.append({
                "symbol": symbol, "stage": stage, "passed": passed, "reason": reason,
                **extra,
            })
        orders:                    list[OrderInstruction] = []
        warnings:                  list[str]              = []
        blocked_by_alloc_cap_count: int                   = 0
        lot_rounded_up_count:       int                   = 0
        risk_rejected_count:        int                   = 0

        # Entry Freeze Mode（資産保全・2026-07-17）: CBとは独立フラグ。OR結合で
        # 新規BUYを全停止する。SELL処理・シグナル生成には一切影響しない。
        entry_frozen  = self.entry_freeze_enabled
        block_new_buy = cb_active or entry_frozen

        if cb_active:
            warnings.append("サーキットブレーカー発動中: 新規 BUY を全停止（SELL のみ実行）")
            logger.warning(
                "ENTRY BLOCKED BY CB: BUY 全停止中。"
                " BUY シグナルが出ても発注しません。SELL のみ実行します。"
            )
        if entry_frozen:
            warnings.append(f"ENTRY FREEZE MODE 発動中: 新規 BUY を全停止（reason={self.entry_freeze_reason}）")
            logger.warning(
                "ENTRY_FROZEN: 新規BUY全面停止中 reason=%s。"
                " BUY シグナルが出ても発注しません。SELL のみ実行します。",
                self.entry_freeze_reason,
            )

        # --- 1. 売り注文（保有中 かつ -1 シグナル） ---
        for sig in signals:
            if sig.signal == -1 and sig.currently_holding:
                pos       = current_positions[sig.symbol]
                qty       = pos["qty"]
                ref_price = float(universe_raw[sig.symbol]["df"]["Close"].iloc[-1])
                orders.append(OrderInstruction(
                    symbol           = sig.symbol,
                    symbol_4digit    = sig.symbol.replace(".T", ""),
                    sector           = sig.sector,
                    side             = "SELL",
                    qty              = qty,
                    order_type       = "MARKET_OPEN",
                    estimated_price  = ref_price,
                    estimated_amount = qty * ref_price,
                    reason           = sig.reason,
                ))

        if block_new_buy:
            blocked_buys = [s.symbol for s in signals if s.signal == 1 and not s.currently_holding]
            if blocked_buys and cb_active:
                logger.warning(
                    "ENTRY BLOCKED BY CB: 以下 %d 銘柄の BUY をスキップ → %s",
                    len(blocked_buys), blocked_buys,
                )
            if blocked_buys and entry_frozen:
                for _sym in blocked_buys:
                    logger.warning("ENTRY_FROZEN: symbol=%s reason=%s", _sym, self.entry_freeze_reason)
                    _audit(_sym, "ENTRY_FREEZE", False, self.entry_freeze_reason)
            _result = (orders, warnings, blocked_by_alloc_cap_count, lot_rounded_up_count, risk_rejected_count)
            assert len(_result) == 5, f"CB/Freeze-path return shape broken: {len(_result)}"
            return _result  # CB/Freeze 中は SELL のみ。BUY カウンターは 0 のまま

        # 売り後の回収資金を加算
        sell_proceeds  = sum(o.estimated_amount for o in orders if o.side == "SELL")
        total_cash     = available_cash + sell_proceeds

        # --- 2. 買い注文 ---
        n_held_after_sells = len(current_positions) - len(
            [o for o in orders if o.side == "SELL"]
        )

        # RSR 順位で優先度付け（rank が小さいほど優先）
        buy_candidates = sorted(
            [s for s in signals if s.signal == 1 and not s.currently_holding],
            key=lambda s: s.rsr_rank,
        )

        _eff_max_pos  = effective_max_pos if effective_max_pos is not None else self.max_positions
        max_alloc_cap = self.capital * self.max_single_weight

        # ── Adaptive Allocator v2 ─────────────────────────────────────────────
        from src.execution.adaptive_alloc import AdaptiveAllocator
        _rc_b       = getattr(getattr(self, "_cfg", None), "risk_controls", None)
        _sec_cap_b  = float(getattr(_rc_b, "sector_cap",      0.25)) if _rc_b else 0.25
        _bear_sec_b = float(getattr(_rc_b, "bear_sector_cap", 0.18)) if _rc_b else 0.18
        _is_bear_b  = (above_ma200 is False)

        _allocator = AdaptiveAllocator(
            capital          = self.capital,
            sector_cap       = _sec_cap_b,
            bear_sector_cap  = _bear_sec_b,
            is_bear          = _is_bear_b,
        )

        _selling_syms = {o.symbol for o in orders if o.side == "SELL"}

        def _alloc_price(sym: str) -> float:
            if sym in universe_raw:
                try: return float(universe_raw[sym]["df"]["Close"].iloc[-1])
                except Exception: pass
            return 0.0

        _allocator.set_existing_exposure(
            {sym: pos for sym, pos in current_positions.items()
             if sym not in _selling_syms},
            self.universe_tickers,
            _alloc_price,
        )
        # regime_sizing: TOPIX MA200下なら bear_scale を適用してポジションサイズを縮小
        _regime_scale = 1.0
        if self.regime_sizing != "none" and above_ma200 is False:
            _regime_scale = self.bear_scale
            logger.info(
                "REGIME_SCALE: %s(bear) → size_scale=%.2f (MA200下落相場)",
                self.regime_sizing, _regime_scale,
            )
        max_alloc_cap = max_alloc_cap * _regime_scale
        new_buys_this_run = today_new_buys
        # blocked_by_alloc_cap_count / lot_rounded_up_count は関数先頭で 0 初期化済み
        # リーダースロット設計: RSR >= 85 の最高位銘柄に1スロットだけ大きめ配分を許可
        # 大型株主導相場（blocked_leaders_weight>40%）でも高RSR銘柄を取りこぼさないため
        _LEADER_RSR_THRESHOLD = 85.0
        _LEADER_SLOT_WEIGHT   = 0.35     # 70万円/200万円 — 通常上限0.20の約1.75倍
        _leader_slot_used     = False

        # ── Sector exposure trajectory forecast (diagnostics) ─────────────────
        _fcast_inputs: list[tuple[str, str, int, int, float]] = []
        for _fs in buy_candidates:
            if _fs.symbol in universe_raw:
                try:
                    _fp = float(universe_raw[_fs.symbol]["df"]["Close"].iloc[-1])
                    _fq = max(100, (int(max_alloc_cap / max(1.0, _fp)) // 100) * 100)
                    _fcast_inputs.append((_fs.symbol, _fs.sector, _fs.rsr_rank, _fq, _fp))
                except Exception:
                    pass
        _fcast_frames    = _allocator.forecast(_fcast_inputs)
        _fcast_rejected  = sum(1 for f in _fcast_frames if f.rejected)
        _fcast_degraded  = sum(1 for f in _fcast_frames if not f.rejected and f.multiplier < 1.0)
        logger.info(
            "[ALLOCATOR] forecast: candidates=%d expected_rejected=%d expected_degraded=%d "
            "existing_sector_exposure=%s is_bear=%s",
            len(_fcast_frames), _fcast_rejected, _fcast_degraded,
            {k: round(v, 3) for k, v in _allocator.sector_utilization().items()},
            _is_bear_b,
        )

        # ── Deployability pre-ranking ─────────────────────────────────────────
        # Re-rank buy_candidates by deployability_score before the allocation loop.
        # Surfaces candidates that can actually be deployed given current cash,
        # sector caps, and lot-size constraints. Reduces missed_entries and idle
        # capital accumulation caused by high-alpha undeployable symbols ranking first.
        _deploy_cands: list[dict] = []
        for _fs in buy_candidates:
            if _fs.symbol in universe_raw:
                try:
                    _fp = float(universe_raw[_fs.symbol]["df"]["Close"].iloc[-1])
                    _fq = max(100, (int(max_alloc_cap / max(1.0, _fp)) // 100) * 100)
                except Exception:
                    _fp, _fq = 0.0, 0
            else:
                _fp, _fq = 0.0, 0
            _deploy_cands.append({
                "symbol":       _fs.symbol,
                "alpha_score":  float(_fs.rsr),
                "price":        _fp,
                "qty":          _fq,
                "sector":       _fs.sector,
                "_sig":         _fs,           # carry original signal for the loop below
            })

        _ranked_deploy, _deploy_diag = _allocator.pre_rank_candidates(
            candidates         = _deploy_cands,
            available_cash     = total_cash,
            max_alloc          = max_alloc_cap,
            existing_positions = n_held_after_sells,
            max_positions      = _eff_max_pos,
        )
        # Rebuild buy_candidates in deployability order, preserving original StockSignal refs
        buy_candidates = [_rd["_sig"] for _rd in _ranked_deploy]

        # Emit deployability metrics to JSONL log
        try:
            from src.execution.deployability import emit_deployability_metrics
            from src.paths import RUNTIME_DIR
            _deploy_log = RUNTIME_DIR / "deployability_metrics.jsonl"
            emit_deployability_metrics(
                diag           = _deploy_diag,
                capital        = self.capital,
                available_cash = total_cash,
                committed_yen  = 0.0,          # pre-commit; updated in summary below
                log_path       = _deploy_log,
            )
        except Exception as _de:
            logger.warning("[DEPLOYABILITY] metrics emit failed: %s", _de)

        logger.info(
            "[DEPLOYABILITY] pre-rank: total=%d deployable=%d undeployable=%d "
            "alloc_survival=%.3f undeployable_alpha=%d breakdown=%s",
            _deploy_diag.total_candidates, _deploy_diag.deployable_count,
            _deploy_diag.undeployable_count, _deploy_diag.alloc_survival_rate,
            _deploy_diag.undeployable_alpha_count,
            _deploy_diag.rejection_reason_breakdown,
        )

        for i, sig in enumerate(buy_candidates):
            open_slots = _capacity_check(_eff_max_pos, n_held_after_sells)
            if open_slots <= 0:
                warnings.append(
                    f"最大ポジション数({_eff_max_pos})に達したため"
                    f" {sig.symbol} の BUY をスキップ"
                )
                # break以降このcandidate以下は個別評価されないため、
                # 監査上は残り全候補を capacity 不足として記録しておく。
                for _rem in buy_candidates[i:]:
                    _audit(_rem.symbol, "CAPACITY", False, "position_full",
                           rsr=_rem.rsr, rsr_rank=_rem.rsr_rank,
                           held=n_held_after_sells, max_positions=_eff_max_pos)
                break

            # max_new_positions_per_day チェック
            if new_buys_this_run >= self.max_new_positions_per_day:
                warnings.append(
                    f"本日の新規 BUY 上限({self.max_new_positions_per_day}件)に達しました。"
                    f" {sig.symbol} をスキップ"
                )
                for _rem in buy_candidates[i:]:
                    _audit(_rem.symbol, "DAILY_LIMIT", False, "max_new_positions_per_day",
                           rsr=_rem.rsr, rsr_rank=_rem.rsr_rank,
                           new_buys_this_run=new_buys_this_run,
                           max_new_positions_per_day=self.max_new_positions_per_day)
                break
            _audit(sig.symbol, "CAPACITY", True, "slot_available",
                   rsr=sig.rsr, rsr_rank=sig.rsr_rank,
                   held=n_held_after_sells, max_positions=_eff_max_pos)

            _df_buy   = universe_raw[sig.symbol]["df"]
            ref_price = float(_df_buy["Close"].iloc[-1])
            lot_cost  = ref_price * 100  # 1単元（100株）コスト

            # ── ポジションサイジング（existing: 現金÷残候補数の動的分配、4/13仕様）
            # ATR Risk Sizing は除去済み（2026-06-24, Study32 Proven Negative confidence=HIGH）
            # qty_cap（配分上限ベース）との min() は維持（下記）
            n_remaining     = len(buy_candidates) - i
            effective_slots = min(open_slots, n_remaining)
            _fallback_alloc = total_cash / max(1, effective_slots)
            qty_risk = int(min(_fallback_alloc, max_alloc_cap) // lot_cost) * 100

            # 配分上限キャップ（通常: max_single_weight / リーダー: leader_slot_weight）
            _is_leader = (
                sig.rsr >= _LEADER_RSR_THRESHOLD
                and not _leader_slot_used
                and sig.rsr_rank == 1  # RSR最上位のみ
            )
            _effective_alloc_cap = (
                self.capital * _LEADER_SLOT_WEIGHT if _is_leader else max_alloc_cap
            )
            qty_cap = int(_effective_alloc_cap // lot_cost) * 100

            if qty_cap == 0:
                # 1単元コスト > alloc_cap → 資本制約による除外（戦略ではなくサイズ設計の問題）
                blocked_by_alloc_cap_count += 1
                warnings.append(
                    f"{sig.symbol}: 配分上限キャップにより除外"
                    f" (1単元=¥{lot_cost:,.0f} > alloc_cap=¥{_effective_alloc_cap:,.0f})"
                    f" → BUY スキップ"
                )
                logger.info(
                    f"[debug] {sig.symbol} close={lot_cost / 100:.1f} required={lot_cost:,}"
                )
                if lot_cost > 100_000_000:
                    logger.warning(f"[anomaly] {sig.symbol} required_cash too large: {lot_cost:,}")
                logger.info(
                    f"[skip] {sig.symbol} reason=alloc_cap"
                    f" required={lot_cost:,} cap={int(_effective_alloc_cap):,}"
                    f" rank={sig.rsr_rank} score={sig.rsr:.2f}"
                )
                _audit(sig.symbol, "CAPITAL", False, "alloc_cap_exceeded",
                       rsr=sig.rsr, rsr_rank=sig.rsr_rank,
                       lot_cost=lot_cost, alloc_cap=_effective_alloc_cap)
                continue
            _audit(sig.symbol, "CAPITAL", True, "within_alloc_cap",
                   rsr=sig.rsr, rsr_rank=sig.rsr_rank,
                   lot_cost=lot_cost, alloc_cap=_effective_alloc_cap)

            qty = min(qty_risk, qty_cap)

            if qty <= 0:
                warnings.append(
                    f"{sig.symbol}: サイジング結果qty=0"
                    f" (alloc=¥{_fallback_alloc:,.0f} price=¥{ref_price:,.0f})"
                    f" → BUY スキップ"
                )
                _audit(sig.symbol, "SIZING", False, "zero_qty",
                       rsr=sig.rsr, rsr_rank=sig.rsr_rank)
                continue

            # ── Adaptive sector degradation (v2) ─────────────────────────────
            # Pre-degrade qty based on remaining sector exposure BEFORE
            # constructing the order, so pre_trade_risk_check sees the
            # scaled amount and passes the sector weight gate.
            _degrade = _allocator.apply(sig.sector, qty, ref_price)
            if _degrade.reason == "rejected":
                warnings.append(
                    f"{sig.symbol}: セクター集中制限（adaptive）sector={sig.sector}"
                    f" multiplier={_degrade.multiplier:.3f}"
                    f" < {_allocator._reject_threshold:.2f} → BUY スキップ"
                )
                risk_rejected_count += 1
                _allocator.record_rejected()
                _audit(sig.symbol, "SECTOR_CONCENTRATION", False, "sector_concentration_adaptive",
                       rsr=sig.rsr, rsr_rank=sig.rsr_rank, sector=sig.sector,
                       multiplier=_degrade.multiplier)
                continue
            if _degrade.reason == "degraded":
                qty = _degrade.degraded_qty
                _allocator.record_degraded()

            # ATR20 計算（True Range平均）
            _atr20 = 0.0
            try:
                _tr = pd.concat([
                    _df_buy["High"] - _df_buy["Low"],
                    (_df_buy["High"] - _df_buy["Close"].shift()).abs(),
                    (_df_buy["Low"]  - _df_buy["Close"].shift()).abs(),
                ], axis=1).max(axis=1)
                _atr_val = float(_tr.rolling(20).mean().iloc[-1])
                if not np.isnan(_atr_val):
                    _atr20 = _atr_val
            except Exception:
                pass

            _new_order = OrderInstruction(
                symbol           = sig.symbol,
                symbol_4digit    = sig.symbol.replace(".T", ""),
                sector           = sig.sector,
                side             = "BUY",
                qty              = qty,
                order_type       = "MARKET_OPEN",
                estimated_price  = ref_price,
                estimated_amount = qty * ref_price,
                reason           = sig.reason,
                atr20            = _atr20,
                strategy_type    = sig.strategy_type,
            )
            # pre_trade_risk_check: defense-in-depth for symbol_cap / cluster_cap.
            # Sector weight gate should now pass since we pre-degraded above.
            if not self.pre_trade_risk_check(_new_order, current_positions, universe_raw, above_ma200):
                warnings.append(
                    f"{sig.symbol}: pre_trade_risk_check 不合格（symbol/cluster cap 超過）"
                    f" → BUY スキップ"
                )
                risk_rejected_count += 1
                _audit(sig.symbol, "RISK", False, "pre_trade_risk_check_reject",
                       rsr=sig.rsr, rsr_rank=sig.rsr_rank, qty=qty)
                continue
            _audit(sig.symbol, "RISK", True, "risk_check_pass",
                   rsr=sig.rsr, rsr_rank=sig.rsr_rank, qty=qty)
            orders.append(_new_order)
            _audit(sig.symbol, "ORDER_BUILT", True, "order_constructed",
                   rsr=sig.rsr, rsr_rank=sig.rsr_rank, qty=qty,
                   estimated_price=ref_price, estimated_amount=qty * ref_price)
            _allocator.commit(sig.sector, qty, ref_price)

            n_held_after_sells       += 1
            new_buys_this_run        += 1
            total_cash               -= qty * ref_price
            if _is_leader:
                _leader_slot_used = True
                logger.info(
                    "LEADER SLOT: %s RSR=%.1f alloc=¥%.0f (%.0f%% weight)",
                    sig.symbol, sig.rsr, qty * ref_price,
                    (qty * ref_price) / self.capital * 100,
                )

        _buy_orders_out = [o for o in orders if o.side == "BUY"]
        _missed_alloc   = max(0, len(buy_candidates) - len(_buy_orders_out))
        logger.info(
            "[ALLOCATOR] %s",
            _allocator.summary(available_cash=total_cash, missed_entries=_missed_alloc),
        )

        _result = (orders, warnings, blocked_by_alloc_cap_count, lot_rounded_up_count, risk_rejected_count)
        assert len(_result) == 5, f"normal-path return shape broken: {len(_result)}"
        return _result

    # ------------------------------------------------------------------ #
    # 発注実行（live モードのみ）レート制限付き
    # ------------------------------------------------------------------ #
    def _send_orders(self, orders: list[OrderInstruction]) -> list[dict]:
        """
        kabuステーション API に注文を送信する。
        レート制限: ORDER_RATE_LIMIT_PER_MIN 件/分（デフォルト 3件/分 = 20秒/件）
        """
        from src.kabusapi.client import Side, OrderType, Exchange

        now = datetime.now(JST)
        hh  = now.hour
        mm  = now.minute
        t   = hh * 60 + mm   # 分単位

        # ── FrontOrderType 時間帯別選択 ──────────────────────────────────
        # kabu API 仕様:
        #   13 = 寄成（前場）: 09:00 前の注文 → 前場寄り付きで執行
        #   10 = 成行:         前場・後場の取引時間中
        #   14 = 引成（前場）: 前場クローズ直前 (11:25〜11:30)
        #   16 = 引成（後場）: 後場クローズ直前 (15:25〜15:30)
        # 昼休み (11:30〜12:30) / 後場終了後 (15:30〜) は注文送信しない
        # ─────────────────────────────────────────────────────────────────
        _T_AM_OPEN      = 9  * 60        #  9:00
        _T_AM_CLOSE_PRE = 11 * 60 + 25  # 11:25
        _T_AM_CLOSE     = 11 * 60 + 30  # 11:30
        _T_PM_OPEN      = 12 * 60 + 30  # 12:30
        _T_PM_CLOSE_PRE = 15 * 60 + 25  # 15:25
        _T_PM_CLOSE     = 15 * 60 + 30  # 15:30

        if t < _T_AM_OPEN:
            order_type    = OrderType.MARKET_OPEN
            _ot_label     = "寄成前場(13)"
            _skip_orders  = False
        elif t < _T_AM_CLOSE_PRE:                      # 09:00〜11:25
            order_type    = OrderType.MARKET
            _ot_label     = "成行(10)"
            _skip_orders  = False
        elif t < _T_AM_CLOSE:                          # 11:25〜11:30
            order_type    = OrderType.MARKET_CLOSE
            _ot_label     = "引成前場(14)"
            _skip_orders  = False
        elif t < _T_PM_OPEN:                           # 11:30〜12:30 昼休み
            order_type    = OrderType.MARKET            # 値は参照しない
            _ot_label     = "SKIP(昼休み)"
            _skip_orders  = True
        elif t < _T_PM_CLOSE_PRE:                      # 12:30〜15:25
            order_type    = OrderType.MARKET
            _ot_label     = "成行(10)"
            _skip_orders  = False
        elif t < _T_PM_CLOSE:                          # 15:25〜15:30
            order_type    = OrderType.MARKET_CLOSE_PM
            _ot_label     = "引成後場(16)"
            _skip_orders  = False
        else:                                           # 15:30〜
            order_type    = OrderType.MARKET            # 値は参照しない
            _ot_label     = "SKIP(後場終了)"
            _skip_orders  = True

        logger.info(
            "[ORDER_TYPE_DECISION] %02d:%02d -> FrontOrderType=%s",
            hh, mm, _ot_label,
        )

        if _skip_orders:
            logger.warning(
                "[ORDER_TYPE_DECISION] %02d:%02d -> 注文送信をスキップ (%s)",
                hh, mm, _ot_label,
            )
            return []

        from src.execution.execution_metrics import (
            compute_gap_pct, log_execution_event, GAP_SKIP_THRESHOLD,
        )
        _signal_time_iso = (
            self._last_signal_time.strftime("%Y-%m-%dT%H:%M:%S%z")
            if self._last_signal_time is not None else None
        )

        results = []
        for idx, o in enumerate(orders):
            # SAFETY FIX (2026-07-07 incident): side は "BUY"/"SELL" のみ許可。
            # 以前は SHADOW_BUY も Side.BUY にマッピングしており、観測専用のはずの
            # Shadow候補が実際に kabu API へ送信されてしまっていた。
            # 未知の side は fail-closed でこの注文だけスキップする。
            if o.side == "BUY":
                side_code = Side.BUY
            elif o.side == "SELL":
                side_code = Side.SELL
            else:
                logger.error(
                    "[ORDER_SIDE_GUARD] unknown side=%s symbol=%s → 発注スキップ（fail-closed）",
                    o.side, o.symbol,
                )
                results.append({
                    "symbol":              o.symbol,
                    "side":                o.side,
                    "qty":                 o.qty,
                    "estimated_price":     o.estimated_price,
                    "planned_entry_price": o.estimated_price,
                    "actual_entry_price":  None,
                    "slippage_pct":        None,
                    "gap_pct":             None,
                    "fill_status":         "rejected_unknown_side",
                    "order_submit_time":   None,
                    "fill_time":           None,
                    "atr20":               o.atr20,
                    "sector":              o.sector,
                    "reason":              o.reason,
                    "strategy_type":       o.strategy_type,
                    "success":             False,
                    "order_id":            None,
                    "result_code":         None,
                })
                continue

            # レート制限: 2件目以降にインターバルを挿入
            if idx > 0:
                time.sleep(self.order_rate_interval_sec)

            order_submit_time = datetime.now(JST)
            order_submit_iso  = order_submit_time.strftime("%Y-%m-%dT%H:%M:%S%z")

            # ── GAP SKIP FILTER (BUY only) ────────────────────────
            # Fetch board price and skip entry if open gap > GAP_SKIP_THRESHOLD.
            # No board call for SELL: never skip exits due to gap.
            # Does NOT change signal generation; only prevents execution when price
            # has already gapped up beyond acceptable entry price.
            gap_pct      = None
            board_price  = None
            _is_buy      = (o.side == "BUY")

            if _is_buy and self._client is not None and o.estimated_price > 0:
                try:
                    _board = self._client.get_board(o.symbol_4digit)
                    board_price = (
                        _board.ask_price if _board.ask_price > 0
                        else _board.current_price
                    )
                    gap_pct = compute_gap_pct(board_price, o.estimated_price)
                    if gap_pct is not None and gap_pct > GAP_SKIP_THRESHOLD:
                        logger.info(
                            "GAP_SKIP %s gap=+%.2f%% > %.0f%% "
                            "(board=%.0f prev_close=%.0f)",
                            o.symbol, gap_pct * 100, GAP_SKIP_THRESHOLD * 100,
                            board_price, o.estimated_price,
                        )
                        _exec_entry = {
                            "symbol":              o.symbol,
                            "entry_signal_time":   _signal_time_iso,
                            "order_submit_time":   order_submit_iso,
                            "fill_time":           None,
                            "planned_entry_price": o.estimated_price,
                            "actual_entry_price":  None,
                            "slippage_pct":        None,
                            "gap_pct":             round(gap_pct, 4),
                            "fill_status":         "gap_skip",
                        }
                        log_execution_event(_exec_entry)
                        results.append({
                            "symbol":              o.symbol,
                            "side":                o.side,
                            "qty":                 o.qty,
                            "estimated_price":     o.estimated_price,
                            "planned_entry_price": o.estimated_price,
                            "actual_entry_price":  None,
                            "slippage_pct":        None,
                            "gap_pct":             round(gap_pct, 4),
                            "fill_status":         "gap_skip",
                            "order_submit_time":   order_submit_iso,
                            "fill_time":           None,
                            "atr20":               o.atr20,
                            "sector":              o.sector,
                            "reason":              o.reason,
                            "strategy_type":       o.strategy_type,
                            "success":             False,
                            "order_id":            None,
                            "result_code":         None,
                        })
                        continue   # skip this BUY — gap too large
                    time.sleep(self._BOARD_FETCH_INTERVAL)   # rate-limit board calls
                except Exception as _be:
                    logger.debug("board fetch for gap check failed (%s): %s", o.symbol, _be)

            try:
                # BUY / SELL ともに SOR=9 を使う。
                # Exchange.TSE=1 は get_board()（板情報照会）専用で sendorder には使えない。
                # → sendorder で Exchange=1 を指定すると Code=100378「指定された市場でのお取引はお受けできません」
                # （diagnose_sell.py 総当たり検証 2026-04-15 で確定）
                exchange = Exchange.SOR
                qty      = int(o.qty)   # LeavesQty が float で来る場合の安全策
                logger.debug(
                    "ORDER_PAYLOAD side=%s symbol=%s exchange=%s qty=%d qty_type=%s type=%s",
                    o.side, o.symbol_4digit, exchange, qty, type(qty).__name__, order_type,
                )
                result = self._client.send_order(
                    symbol     = o.symbol_4digit,
                    exchange   = exchange,
                    side       = side_code,
                    qty        = qty,
                    order_type = order_type,
                )
                fill_time_iso = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
                _fill_status  = "submitted" if result.success else "failed"
                results.append({
                    "symbol":              o.symbol,
                    "side":                o.side,
                    "qty":                 o.qty,
                    "estimated_price":     o.estimated_price,
                    "planned_entry_price": o.estimated_price,
                    "actual_entry_price":  None,   # confirmed after open fill
                    "slippage_pct":        None,   # confirmed after open fill
                    "gap_pct":             round(gap_pct, 4) if gap_pct is not None else None,
                    "fill_status":         _fill_status,
                    "order_submit_time":   order_submit_iso,
                    "fill_time":           fill_time_iso if result.success else None,
                    "atr20":               o.atr20,
                    "sector":              o.sector,
                    "reason":              o.reason,
                    "strategy_type":       o.strategy_type,
                    "order_id":            result.order_id,
                    "success":             result.success,
                    "result_code":         result.result_code,
                })
                # log execution quality event
                log_execution_event({
                    "symbol":              o.symbol,
                    "entry_signal_time":   _signal_time_iso,
                    "order_submit_time":   order_submit_iso,
                    "fill_time":           fill_time_iso if result.success else None,
                    "planned_entry_price": o.estimated_price,
                    "actual_entry_price":  None,
                    "slippage_pct":        None,
                    "gap_pct":             round(gap_pct, 4) if gap_pct is not None else None,
                    "fill_status":         _fill_status,
                })
                status = "✅ 成功" if result.success else "❌ 失敗"
                logger.info(
                    "%s %s %s %d株 (OrderId: %s)",
                    status, o.side, o.symbol, o.qty, result.order_id,
                )
            except Exception as e:
                logger.error("%s %s 注文送信エラー: %s", o.side, o.symbol, e)
                log_execution_event({
                    "symbol":            o.symbol,
                    "entry_signal_time": _signal_time_iso,
                    "order_submit_time": order_submit_iso,
                    "fill_time":         None,
                    "planned_entry_price": o.estimated_price,
                    "actual_entry_price":  None,
                    "slippage_pct":        None,
                    "gap_pct":             round(gap_pct, 4) if gap_pct is not None else None,
                    "fill_status":         "failed",
                })
                results.append({
                    "symbol":              o.symbol,
                    "side":                o.side,
                    "qty":                 o.qty,
                    "estimated_price":     o.estimated_price,
                    "planned_entry_price": o.estimated_price,
                    "actual_entry_price":  None,
                    "slippage_pct":        None,
                    "gap_pct":             round(gap_pct, 4) if gap_pct is not None else None,
                    "fill_status":         "failed",
                    "order_submit_time":   order_submit_iso,
                    "fill_time":           None,
                    "atr20":               o.atr20,
                    "sector":              o.sector,
                    "reason":              o.reason,
                    "strategy_type":       o.strategy_type,
                    "success":             False,
                    "error":               str(e),
                })

        return results

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ギャップダウンストップ（9:00 AM 直前に board 価格で再評価）
    # ------------------------------------------------------------------ #

    # board 取得間隔（寄り付き直後の API 負荷を抑制）
    _BOARD_FETCH_INTERVAL = 0.15   # 秒

    # ギャップイベントログ（後日の損失分布分析用）
    _GAP_EVENT_LOG = LOGS_DIR / "gap_events.jsonl"

    @staticmethod
    def _resolve_board_price(board) -> Optional[float]:
        """
        board オブジェクトから有効な価格を解決する。

        優先順: current_price → bid_price → None
        日本株では寄り付き未成立時に両方が 0 になるケースがあるため None を返す。
        None の場合は呼び出し元でスキップする（保守的設計: 強制 SELL しない）。
        """
        cp = board.current_price
        bp = board.bid_price
        if cp is not None and cp > 0:
            return float(cp)
        if bp is not None and bp > 0:
            return float(bp)
        return None   # 寄り付き未成立・板なし → 判定スキップ

    def _log_gap_event(self, event: dict) -> None:
        """ギャップイベントを logs/gap_events.jsonl に追記する（研究用）。"""
        try:
            self._GAP_EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with self._GAP_EVENT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("gap_event ログ書き込み失敗（無視）: %s", e)

    def check_gap_stops(
        self,
        order_objects: list[OrderInstruction],
    ) -> list[OrderInstruction]:
        """
        9:00 AM 直後に board 価格を取得し、ギャップダウン SELL を追加する。

        背景:
            trailing stop は前日終値で判定するため、当日の寄り付きギャップを検出できない。
            例: 前日終値=110 > stop=105 (HOLD) → 当日 board=90 → このメソッドで SELL 追加

        安全設計:
            - API 未接続 (dry-run) は無条件スキップ
            - board 取得失敗銘柄は保守的にスキップ（強制 SELL しない）
            - 既に SELL 注文がある銘柄はスキップ（二重 SELL 防止）
            - current_price=0 かつ bid_price=0（寄り付き未成立）→ スキップ
            - board 取得間隔 0.15秒（寄り付き直後の API 負荷制御）

        副作用:
            - ギャップイベント（トリガー有無を問わず）を logs/gap_events.jsonl に記録

        Returns:
            order_objects にギャップダウン SELL を追加したリスト（追加なければそのまま返す）
        """
        if self._client is None:
            logger.info("gap_stop: API未接続 → ギャップダウンチェックをスキップ")
            return order_objects

        held_stop_info = self._last_held_stop_info
        if not held_stop_info:
            logger.info("gap_stop: 保有ポジションなし → スキップ")
            return order_objects

        # 既に SELL 注文がある銘柄はスキップ（二重 SELL 防止）
        already_selling: set[str] = {o.symbol for o in order_objects if o.side == "SELL"}

        today_str  = datetime.now(JST).strftime("%Y-%m-%d")
        gap_orders: list[OrderInstruction] = []

        for sym, info in held_stop_info.items():
            if sym in already_selling:
                logger.info("gap_stop: %s は既に SELL 注文済み → スキップ", sym)
                continue

            stop_price = info["stop_price"]
            if stop_price <= 0:
                continue

            # board 取得（3回リトライ・0.5秒間隔）
            board = None
            sym_4digit = sym.replace(".T", "")
            for attempt in range(3):
                try:
                    board = self._client.get_board(sym_4digit, exchange=1)  # 1=TSE
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.5)
                    else:
                        logger.warning("gap_stop: %s board取得失敗（3回） → スキップ: %s", sym, e)

            # board 取得間隔（寄り付き直後の API 負荷制御）
            time.sleep(self._BOARD_FETCH_INTERVAL)

            if board is None:
                continue

            # 有効価格を解決（寄り付き未成立で両方 0 なら None → スキップ）
            board_price = self._resolve_board_price(board)
            if board_price is None:
                logger.info("gap_stop: %s 価格取得不可（寄り付き未成立？）→ スキップ", sym)
                self._log_gap_event({
                    "date":           today_str,
                    "symbol":         sym,
                    "board_price":    None,
                    "stop_price":     stop_price,
                    "last_close":     info["last_close"],
                    "gap_triggered":  False,
                    "skip_reason":    "price_unavailable",
                })
                continue

            triggered = board_price < stop_price

            # ギャップイベントを記録（トリガー有無を問わず）
            self._log_gap_event({
                "date":          today_str,
                "symbol":        sym,
                "board_price":   round(board_price, 0),
                "stop_price":    round(stop_price, 0),
                "last_close":    round(info["last_close"], 0),
                "gap_triggered": triggered,
                "skip_reason":   None,
            })

            if triggered:
                logger.warning(
                    "gap_stop TRIGGERED: %s board=%.0f < stop=%.0f (last_close=%.0f) → SELL追加",
                    sym, board_price, stop_price, info["last_close"],
                )
                gap_orders.append(OrderInstruction(
                    symbol           = sym,
                    symbol_4digit    = sym_4digit,
                    sector           = info["sector"],
                    side             = "SELL",
                    qty              = info["qty"],
                    order_type       = "MARKET_OPEN",
                    estimated_price  = board_price,
                    estimated_amount = info["qty"] * board_price,
                    reason           = (
                        f"SELL[ギャップダウン]: board={board_price:.0f}"
                        f" < stop={stop_price:.0f}"
                        f" (last_close={info['last_close']:.0f})"
                    ),
                ))
            else:
                logger.info(
                    "gap_stop OK: %s board=%.0f >= stop=%.0f",
                    sym, board_price, stop_price,
                )

        if gap_orders:
            logger.warning(
                "gap_stop: %d銘柄のギャップダウン SELL を追加 → %s",
                len(gap_orders), [o.symbol for o in gap_orders],
            )

        return order_objects + gap_orders

    # 約定後の状態更新（run_live_signal.py から呼ぶ）
    # ------------------------------------------------------------------ #
    def update_state_after_execution(
        self,
        send_results:    list[dict],
        today_str:       str,
        signal_rsr_map:  dict | None = None,
    ) -> None:
        """
        実際の約定確認後に portfolio_state を更新し、トレードログを書く。
        BUY 成功 → entry_date / entry_price 記録 + logs/trades.jsonl に open エントリー
        SELL 成功 → entry_date / entry_price 削除 + PnL 計算して closed エントリー書き込み
        SELL + 時間ストップ → reentry_blocked に追加（5 営業日）
        """
        import json as _json
        from pathlib import Path as _Path
        from datetime import date as _date

        state               = self._load_portfolio_state()
        pos_entry_dates     = state.setdefault("position_entry_dates",    {})
        pos_entry_prices    = state.setdefault("position_entry_prices",   {})
        pos_entry_atrs      = state.setdefault("position_entry_atrs",     {})
        pos_entry_rsrs      = state.setdefault("position_entry_rsrs",     {})
        pos_highest_closes  = state.setdefault("position_highest_closes", {})
        pos_qtys            = state.setdefault("position_qtys",           {})
        reentry_blocked     = state.setdefault("reentry_blocked",         {})
        # Study26: available_cash を position_qtys / position_entry_prices と
        # 同一 state dict 内で同時更新する（単一トランザクション化）。
        # 次回 commit_broker_snapshot() でブローカー真値に上書きされるまでの
        # 暫定値であり、約定見積り(estimated_price)ベースの推定にすぎない。
        avail_cash          = float(state.get("available_cash", 0.0) or 0.0)
        shadow_positions  = state.setdefault("shadow_positions",        {})  # {sym: entry_price} shadow由来
        pos_strategy_types = state.setdefault("position_strategy_types", {})  # {sym: "fujiko"/"mean_rev"}

        # 最新の市場レジームをメトリクスから取得（regime別成績集計用）
        _latest_regime = None
        _metrics_path  = LOGS_DIR / "diagnostics" / "metrics.jsonl"
        if _metrics_path.exists():
            try:
                _lines = [l for l in _metrics_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                if _lines:
                    _latest_regime = _json.loads(_lines[-1]).get("trend_market")
            except Exception:
                pass

        _trades_path = LOGS_DIR / "trades.jsonl"
        _trades_path.parent.mkdir(parents=True, exist_ok=True)

        for r in send_results:
            if not r.get("success"):
                continue
            sym    = r["symbol"]
            side   = r["side"]
            reason = r.get("reason", "")
            qty    = r.get("qty", 0)
            price  = float(r.get("estimated_price", 0.0))
            sector = r.get("sector", "不明")
            amount = qty * price

            if side == "BUY":
                atr20 = float(r.get("atr20", 0.0))
                pos_entry_dates[sym]      = today_str
                pos_qtys[sym]             = int(qty)  # CB_GUARD_COMP SOURCE1用: T+2決済ラグ補償

                # SAFETY FIX (entry-metadata-loss incident): price==0.0 must never
                # be silently written as if it were a real fill price — that is
                # exactly what corrupted 5301.T's entry_price/highest_close and
                # left it with no ATR (no send_result carried estimated_price
                # through the process-isolated path). Fail closed instead:
                # try one authoritative recovery (broker's real avg cost), and
                # if that also fails, record the gap in an audit trail instead
                # of fabricating 0.0.
                if price <= 0:
                    price  = self._recover_entry_price_from_broker(sym)
                    amount = qty * price   # 復元後のprice基準にavail_cash計算をやり直す
                if price <= 0:
                    _missing = state.setdefault("entry_metadata_missing", {})
                    _missing[sym] = {
                        "detected_at": today_str,
                        "qty":         int(qty),
                        "reason":      "estimated_price missing/zero in send_result "
                                       "and broker avg_price recovery failed",
                    }
                    logger.error(
                        "[ENTRY_METADATA_MISSING] %s: BUY成立したがentry_price復元不可。"
                        "position_entry_prices/highest_closes/entry_atrsへの0.0書き込みを"
                        "抑止し、ATR Trailing対象外として監査ログに記録した。手動確認が必要。",
                        sym,
                    )
                else:
                    pos_entry_prices[sym]     = price
                    pos_highest_closes[sym]   = price   # トレーリングストップ: 初期 highest_close = エントリー価格
                    if atr20 > 0:
                        pos_entry_atrs[sym] = atr20
                    state.get("entry_metadata_missing", {}).pop(sym, None)

                # Quality Replacement Engine — entry RSR for rsr_delta computation (Study57/58A)
                _e_rsr = float((signal_rsr_map or {}).get(sym, 0.0))
                if _e_rsr > 0:
                    pos_entry_rsrs[sym] = _e_rsr
                reentry_blocked.pop(sym, None)
                # 戦略タイプを記録（mean_rev反発未発生検出で参照）
                _order_strategy = r.get("strategy_type", "")
                if _order_strategy:
                    pos_strategy_types[sym] = _order_strategy
                avail_cash -= amount
                logger.info("entry_date 記録: %s → %s @ ¥%.0f ATR20=%.0f", sym, today_str, price, atr20)
                _trade = {
                    "date":         today_str,
                    "symbol":       sym,
                    "sector":       sector,
                    "side":         side,   # "BUY"
                    "qty":          qty,
                    "price":        price,
                    "amount":       amount,
                    "atr20":        atr20,
                    "entry_regime": _latest_regime,
                    "reason":       reason,
                }
                with _trades_path.open("a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_trade, ensure_ascii=False) + "\n")

            elif side == "SELL":
                entry_price = pos_entry_prices.pop(sym, None)
                entry_atr   = pos_entry_atrs.pop(sym, None)
                entry_date  = pos_entry_dates.pop(sym, None)
                pos_highest_closes.pop(sym, None)   # トレーリングストップ用 highest_close をクリア
                pos_qtys.pop(sym, None)             # CB_GUARD_COMP SOURCE1用: 売却済みを除去
                pos_strategy_types.pop(sym, None)
                pos_entry_rsrs.pop(sym, None)       # Quality Replacement Engine — entry RSR クリア
                # shadow由来ポジション判定（記録を削除して返り値を取得）
                _shadow_entry_price = shadow_positions.pop(sym, None)
                _is_shadow = _shadow_entry_price is not None
                if not _is_shadow:
                    # shadow ポジションは実 cash を消費していないため SELL でも加算しない。
                    avail_cash += amount

                pnl     = round((price - entry_price) * qty, 0) if entry_price else None
                pnl_pct = round((price / entry_price) - 1, 4)   if entry_price else None

                # shadow_realized_return: shadow由来のSELL時に計算してログ出力
                if _is_shadow and entry_price:
                    _shadow_ret = round((price / entry_price) - 1, 4)
                    logger.info(
                        "SHADOW realized_return: %s entry=¥%.0f exit=¥%.0f return=%.2f%%",
                        sym, entry_price, price, _shadow_ret * 100,
                    )

                hold_days = None
                if entry_date:
                    try:
                        hold_days = (_date.fromisoformat(today_str) - _date.fromisoformat(entry_date)).days
                    except Exception:
                        pass

                # SAFETY FIX (2026-07-07): 以前は「時間ストップ」理由のSELLのみ
                # cooldown対象だった。RSR_EXIT/トレーリングストップ/緊急exit等
                # 損切り全般が対象外だったため、損切り直後の即日再エントリーが
                # 発生していた（5301.T / 6506.T）。全SELL理由を対象にする。
                _risk_cfg = getattr(self._cfg, "risk", None) if getattr(self, "_cfg", None) else None
                _cooldown_days = int(getattr(_risk_cfg, "reentry_cooldown_days", REENTRY_COOLDOWN_TRADING_DAYS))
                today_ts      = pd.Timestamp(today_str)
                block_end     = _add_trading_days(today_ts, _cooldown_days)
                block_end_str = block_end.strftime("%Y-%m-%d")
                reentry_blocked[sym] = block_end_str
                logger.info(
                    "再エントリー禁止: %s 〜 %s（SELL理由=%s / cooldown%d営業日）",
                    sym, block_end_str, reason, _cooldown_days,
                )

                _trade = {
                    "date":                   today_str,
                    "symbol":                 sym,
                    "sector":                 sector,
                    "side":                   "SELL",
                    "qty":                    qty,
                    "price":                  price,
                    "amount":                 amount,
                    "pnl":                    pnl,
                    "pnl_pct":                pnl_pct,
                    "hold_days":              hold_days,
                    "entry_price":            entry_price,
                    "entry_date":             entry_date,
                    "entry_regime":           _latest_regime,
                    "reason":                 reason,
                    "is_shadow":              _is_shadow,
                    "shadow_realized_return": round((price / entry_price) - 1, 4) if (_is_shadow and entry_price) else None,
                }
                with _trades_path.open("a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_trade, ensure_ascii=False) + "\n")

        # position_qtys / position_entry_prices と同一 commit で available_cash も書く
        # （Study26: 部分 commit による state 内部不整合を防止）。
        state["available_cash"] = round(avail_cash, 0)
        self._save_portfolio_state(state)

    # ------------------------------------------------------------------ #
    # フィーチャーエンリッチメント（_download_data 後に呼ぶ）
    # ------------------------------------------------------------------ #
    def _enrich_universe_df(self, universe_raw: dict) -> None:
        """
        universe_raw の各 df に必須フィーチャーカラムを追加する（インプレース）。

        追加カラム:
          rsr_252       - IBD式12ヶ月複合リターンのクロスセクショナルパーセンタイル (0-100)
          rsr_63        - 63日リターンのクロスセクショナルパーセンタイル (0-100)
          ma20          - 20日単純移動平均
          ma50          - 50日単純移動平均
          ma20_slope    - MA20の5日変化量（正=上昇トレンド）
          avg_turnover_20d - 20日平均売買代金（Close × Volume）
          avg_volume_20d   - 20日平均出来高
        """
        if not universe_raw:
            return

        # ── クロスセクショナルRSR計算 ────────────────────────────────────
        prices_dict = {
            sym: info["df"]["Close"]
            for sym, info in universe_raw.items()
            if "Close" in info.get("df", pd.DataFrame()).columns
        }
        if len(prices_dict) < 2:
            logger.warning("[enrich] ユニバース銘柄数不足 (%d) → RSR計算スキップ", len(prices_dict))
            rsr_252_df = pd.DataFrame()
            rsr_63_df  = pd.DataFrame()
        else:
            try:
                # IBD式複合リターン → クロスセクショナルランク
                def _composite(p: pd.Series) -> pd.Series:
                    r1 = p / p.shift(63) - 1
                    r2 = p.shift(63)  / p.shift(126) - 1
                    r3 = p.shift(126) / p.shift(189) - 1
                    r4 = p.shift(189) / p.shift(252) - 1
                    return 0.4 * r1 + 0.2 * r2 + 0.2 * r3 + 0.2 * r4

                comp_df   = pd.DataFrame({s: _composite(p) for s, p in prices_dict.items()})
                rsr_252_df = (comp_df.rank(axis=1, pct=True) * 100).clip(0, 100)

                r63_df    = pd.DataFrame({s: (p / p.shift(63) - 1) for s, p in prices_dict.items()})
                rsr_63_df  = (r63_df.rank(axis=1, pct=True) * 100).clip(0, 100)
            except Exception as exc:
                logger.warning("[enrich] RSR計算エラー: %s", exc)
                rsr_252_df = pd.DataFrame()
                rsr_63_df  = pd.DataFrame()

        # ── 銘柄ごとフィーチャー付与 ────────────────────────────────────
        enriched = 0
        for sym, info in universe_raw.items():
            df = info.get("df")
            if df is None or df.empty:
                continue
            try:
                close  = df["Close"]
                volume = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)

                df["ma20"]            = close.rolling(20, min_periods=1).mean()
                df["ma50"]            = close.rolling(50, min_periods=1).mean()
                df["ma20_slope"]      = df["ma20"].diff(5)
                df["ma20_up"]         = (df["ma20_slope"] > 0).astype(float)
                df["avg_turnover_20d"] = (close * volume).rolling(20, min_periods=1).mean()
                df["avg_volume_20d"]  = volume.rolling(20, min_periods=1).mean()

                if sym in rsr_252_df.columns:
                    df["rsr_252"] = rsr_252_df[sym].reindex(df.index).ffill().fillna(0.0)
                else:
                    df["rsr_252"] = 0.0

                if sym in rsr_63_df.columns:
                    df["rsr_63"] = rsr_63_df[sym].reindex(df.index).ffill().fillna(0.0)
                else:
                    df["rsr_63"] = 0.0

                universe_raw[sym]["df"] = df
                enriched += 1
            except Exception as exc:
                logger.warning("[enrich] %s フィーチャー計算エラー: %s", sym, exc)

        logger.info("[enrich] フィーチャー付与完了: %d / %d 銘柄", enriched, len(universe_raw))

    # ------------------------------------------------------------------ #
    # メイン実行
    # ------------------------------------------------------------------ #
    def run(self) -> tuple[BridgeResult, list[OrderInstruction]]:
        """シグナル生成〜注文生成（〜発注）を一括実行する"""
        now       = datetime.now(JST)
        today_str = now.strftime("%Y-%m-%d")
        self._last_signal_time = now  # execution metrics: signal generation timestamp

        # 1. ポートフォリオ状態読み込み
        portfolio_state = self._load_portfolio_state()

        # 2. データ取得
        logger.info("データ取得中...")
        universe_raw, bench_prices = self._download_data()

        # 2b. フィーチャーエンリッチメント（rsr_252/rsr_63/ma20/ma50/ma20_slope/avg_turnover_20d）
        logger.info("フィーチャー計算中...")
        self._enrich_universe_df(universe_raw)

        data_as_of = _compute_data_as_of(universe_raw)

        # 3. ポジション・余力取得（Broker-as-Sole-SSOT, 2026-07-18）
        # fetch_broker_snapshot() を1回だけ呼び、cash/positions/market_valuesを
        # 唯一の入力として使う。state/OHLCV/ledgerへのフォールバック・部分補完は
        # 一切行わない — broker応答（保有0件を含む）を無条件に信頼する
        # （2026-07-15〜17 equity_peak異常値インシデントの根本原因は複数の
        # 独立フォールバック経路が食い違う値を生成していたことだった）。
        try:
            if self._client is None:
                raise BrokerSnapshotUnavailable("KabuClient is None — broker への接続がありません")
            _broker_snap = fetch_broker_snapshot(self._client)
            self._positions_api_status = {"ok": True, "source": "broker", "error": None}
            self._wallet_api_status    = {"ok": True, "source": "broker", "error": None}
        except BrokerSnapshotUnavailable as _bsu:
            self._positions_api_status = {"ok": False, "source": "broker_error", "error": str(_bsu)}
            self._wallet_api_status    = {"ok": False, "source": "broker_error", "error": str(_bsu)}
            if self.require_broker:
                logger.error("[BROKER_UNAVAILABLE] %s", _bsu)
                raise AbortError("broker_unavailable", str(_bsu)) from _bsu
            logger.warning("API 未接続（require_broker=False の明示的省略モード）: %s", _bsu)
            _broker_snap = None

        if _broker_snap is not None:
            current_positions = {
                sym: {"qty": qty, "avg_price": _broker_snap.avg_costs.get(sym, 0.0)}
                for sym, qty in _broker_snap.positions.items()
                if sym in self.universe_tickers
            }
            available_cash = _broker_snap.cash
        else:
            current_positions = {}
            available_cash    = self._virtual_available_cash({})
        calc_available_cash = available_cash if available_cash is not None else 0.0

        # pre_trade_risk_check キャッシュ更新（execution layer で再利用）
        self._last_current_positions = current_positions
        self._last_universe_raw      = universe_raw

        # 4. 現在 equity → CB 状態更新
        _run_mode = "live" if self.live else "dry"
        _broker_positions_ok = _broker_snap is not None
        _broker_wallet_ok    = _broker_snap is not None

        current_equity = compute_live_equity(
            snapshot=_broker_snap, mode=_run_mode,
            equity_peak=float(portfolio_state.get("equity_peak", self.capital)),
        ) if _broker_snap is not None else calc_available_cash

        # ── 資産計算不変条件チェック (2026-07-19) ─────────────────────────
        # compute_live_equity()の出力を、同一snapshotから独立に再計算した値と
        # 突き合わせる。1円を超える乖離は資産計算経路の再分岐(SSOT違反)を示す
        # 実装バグであり、フェイルクローズで即座に停止する。AbortErrorへ変換し、
        # run_live_signal.py側のEMERGENCY_STOPハンドリングへ統一的に乗せる。
        if _broker_snap is not None:
            try:
                assert_broker_equity_invariant(_broker_snap, current_equity)
            except BrokerEquityInvariantError as _beie:
                raise AbortError("broker_equity_invariant_violation", str(_beie)) from _beie

        # ── 乖離警告 (Phase 3A): last_equity vs current_equity ──────────────
        # 前回保存値と現在推定値の差が 5% 超 or ¥300,000 超なら WARN する。
        _DIVERGE_PCT   = 0.05
        _DIVERGE_ABS   = 300_000
        _last_eq = float(portfolio_state.get("last_equity", current_equity) or current_equity)
        if _last_eq > 0:
            _div_abs = abs(current_equity - _last_eq)
            _div_pct = _div_abs / _last_eq
            if _div_abs >= _DIVERGE_ABS or _div_pct >= _DIVERGE_PCT:
                logger.warning(
                    "[EQUITY_DIVERGENCE] last=¥%s current=¥%s diff=¥%s (%.1f%%) — "
                    "原因: settlement lag / stale state / API 障害の可能性",
                    f"{_last_eq:,.0f}", f"{current_equity:,.0f}",
                    f"{_div_abs:,.0f}", _div_pct * 100,
                )

        # ── cash残差検知 (Study26): cash_delta が market_value_delta で説明できない
        # 残差を detect_cash_event() に配線する。新規 cash は今回フェッチした broker
        # wallet 値（available_cash）のみを使用し、prev は commit 前 portfolio_state
        # から読む（commit_broker_snapshot() がこの後 available_cash を上書きするため
        # ここで読まないと旧値が失われる）。
        # 観測・監査ログ専用（[EQUITY_CASH_RESIDUAL]）。equity_peak判定には使わない
        # （Study96 EquityPeak SSOT Root Cause Audit, 2026-07-17: 原因推定ロジックを
        # peak確定判断へ持ち込まない方針）。
        if _broker_wallet_ok and available_cash is not None:
            _new_cash = float(available_cash)
            _new_mv   = current_equity - _new_cash
            _prev_cash = float(portfolio_state.get("available_cash", _new_cash) or _new_cash)
            _prev_mv   = _last_eq - _prev_cash
            _cash_event = detect_cash_event(
                prev_cash         = _prev_cash,
                new_cash          = _new_cash,
                prev_market_value = _prev_mv,
                new_market_value  = _new_mv,
            )
            if _cash_event is not None:
                logger.warning(
                    "[EQUITY_CASH_RESIDUAL] cash_delta=¥%s market_value_delta=¥%s "
                    "residual=¥%s event_type=%s",
                    f"{_cash_event['delta']:,.0f}",
                    f"{(_new_mv - _prev_mv):,.0f}",
                    f"{_cash_event['unexplained_delta']:,.0f}",
                    _cash_event['event_type'],
                )

        # ── snapshot commit または partial equity 更新 ───────────────────
        if _broker_snap is not None:
            _broker_snap.equity = current_equity
            try:
                commit_broker_snapshot(portfolio_state, _broker_snap)
            except SnapshotValidationError as _sve:
                logger.error("[SNAPSHOT] commit rejected — falling back to equity-only update: %s", _sve)
                update_portfolio_state_from_broker(portfolio_state, current_equity=current_equity)
        else:
            # DRY / API 障害: equity のみ更新 (positions/cash は state から保持)
            update_portfolio_state_from_broker(portfolio_state, current_equity=current_equity)

        # ── 起動時観測性ログ（[STATE] 1行） ─────────────────────────────
        from src.portfolio.state_store import ValidationResult as _VR
        _startup_vr = _VR(ok=True, snapshot_age_seconds=None)
        log_startup_state_line(portfolio_state, _startup_vr)

        # ── リコンシリエーションログ ─────────────────────────────────────
        _broker_eq = current_equity if _broker_positions_ok and _broker_wallet_ok else None
        write_reconciliation_log(
            mode            = _run_mode,
            broker_cash     = available_cash if _broker_wallet_ok else None,
            state_cash      = float(portfolio_state.get("available_cash", 0)),
            broker_equity   = _broker_eq,
            computed_equity = current_equity,
            positions_match = _broker_positions_ok,
        )

        # ── CB状態更新 ──────────────────────────────────────────────────
        # PendingOrderState機構は2026-07-19に完全撤去（_update_cb_state()の
        # docstring参照）。CB判定はraw broker equityのみで行う。
        if self.live:
            portfolio_state = self._update_cb_state(
                portfolio_state, current_equity, today_str,
                broker_snapshot=_broker_snap,
            )
        else:
            # DRY: compute on isolated copy; carry back cb_state for signal gating only.
            # Never mutate persisted keys (equity_peak / cb_cooldown_end_date / recovery_threshold).
            _ps_dry_cb = self._update_cb_state(
                {**portfolio_state},   # shallow copy is sufficient: _update_cb_state writes top-level scalars only
                current_equity, today_str,
                broker_snapshot=_broker_snap,
            )
            portfolio_state["cb_state"] = _ps_dry_cb.get("cb_state", portfolio_state.get("cb_state", "NORMAL"))
            logger.info(
                "[DRY_STATE_GUARD] persistent_state_write_skipped=True"
                " equity_peak_unchanged=¥%.0f equity_peak_would_be=¥%.0f"
                " cb_state=%s",
                float(portfolio_state.get("equity_peak", 0)),
                float(_ps_dry_cb.get("equity_peak", portfolio_state.get("equity_peak", 0))),
                portfolio_state["cb_state"],
            )
        # SAFE_WARN は警告のみ。BUY 停止は CB_ACTIVE / RECOVERY のみ。
        cb_active = portfolio_state["cb_state"] in ("CB_ACTIVE", "RECOVERY")

        # 5. シグナル生成（Top-k + 時間ストップ）
        logger.info("シグナル生成中（%d 銘柄）...", len(self.universe_tickers))
        signals, top_k_syms, _diag = self._generate_all_signals(
            universe_raw, bench_prices, current_positions, portfolio_state
        )
        # PositionSizingIntelligence results: computed inside _generate_all_signals,
        # passed via diagnostics to avoid changing the return tuple signature.
        _ps_results: dict = _diag.get("ps_results", {})

        buy_count  = sum(1 for s in signals if s.signal ==  1)
        sell_count = sum(1 for s in signals if s.signal == -1)
        logger.info(
            "シグナル: BUY=%d / SELL=%d / HOLD=%d",
            buy_count, sell_count, len(signals) - buy_count - sell_count,
        )

        # gap_stop 用キャッシュ: 保有中ポジションのストップ価格を記録
        # check_gap_stops() が 9:00 AM 直後に board を取得して参照する
        self._last_held_stop_info = {
            s.symbol: {
                "qty":        current_positions[s.symbol]["qty"],
                "stop_price": s.trailing_stop_price,
                "last_close": float(universe_raw[s.symbol]["df"]["Close"].iloc[-1]),
                "sector":     s.sector,
            }
            for s in signals
            if (s.currently_holding
                and s.trailing_stop_price > 0
                and s.symbol in current_positions
                and s.symbol in universe_raw)
        }
        if self._last_held_stop_info:
            logger.info(
                "gap_stop cache: %d 銘柄 %s",
                len(self._last_held_stop_info),
                {sym: f"stop={info['stop_price']:.0f}" for sym, info in self._last_held_stop_info.items()},
            )

        # 6. ブレイクアウトクラスター検知 + CapitalDeploymentOS dynamic max positions (Phase 5B.1)
        from src.live.capital_deployment_os import (
            dynamic_max_positions as _dyn_max_pos,
            MAX_POSITIONS_HARD_CAP as _CDOS_CAP,
        )
        _cdos_dep           = self.deployable_capital if self.deployable_capital > 0 else current_equity
        _cdos_dyn_raw       = _dyn_max_pos(_cdos_dep)
        _equity_based_max   = min(_cdos_dyn_raw, self.max_positions)   # PARAMS_LOCKED clamp
        _CLUSTER_THRESHOLD  = 3
        _buy_cands          = [s for s in signals if s.signal == 1 and not s.currently_holding]
        _buy_cands_count    = len(_buy_cands)
        _breakout_cluster   = _buy_cands_count >= _CLUSTER_THRESHOLD
        _effective_max_pos  = _CDOS_CAP if _breakout_cluster else _equity_based_max
        _effective_max_pos  = min(_effective_max_pos, self.max_positions)  # PARAMS_LOCKED クランプ
        if _cdos_dyn_raw > self.max_positions:
            logger.info(
                "CDOS_CLAMPED: deployable=¥%.0f → dyn_raw=%d clamped_to=%d (PARAMS_LOCKED)",
                _cdos_dep, _cdos_dyn_raw, self.max_positions,
            )
        if _breakout_cluster:
            logger.info(
                "CLUSTER DETECTED: BUY candidates=%d >= threshold=%d → effective_max_pos=%d",
                _buy_cands_count, _CLUSTER_THRESHOLD, _effective_max_pos,
            )

        # 6. 注文生成（regime_sizing 用に TOPIX MA200 状態を事前計算）
        _above_ma200_live: bool | None = None
        if self.regime_sizing != "none":
            try:
                _bc = bench_prices.dropna()
                _above_ma200_live = bool(float(_bc.iloc[-1]) >= float(_bc.rolling(200, min_periods=1).mean().iloc[-1]))
            except Exception:
                pass

        _stage_audit_sink: list = []
        _build_result = self._build_orders(
            signals, universe_raw, current_positions, calc_available_cash,
            cb_active=cb_active,
            effective_max_pos=_effective_max_pos,
            above_ma200=_above_ma200_live,
            audit_sink=_stage_audit_sink,
        )
        _validate_build_orders_contract(_build_result)
        orders, order_warnings, _blocked_alloc_cap, _lot_rounded_up, _risk_rejected = _build_result

        # ── LIVE_STAGE_AUDIT: 恒久的なステージ別 PASS/FAIL 監査ログ ─────────
        # 2026-06-29 EVS RCA follow-up: どこでBUY候補が落ちたかを毎run記録する。
        # RSR上位カットオフで top_k に入らなかった候補も RANKING stage として残す。
        try:
            from src.analytics.live_stage_audit import append_stage_audit
            _rank_audit: list = []
            _topk_set = set(top_k_syms)
            for _rsig in signals:
                if _rsig.signal == 1 and not _rsig.currently_holding and _rsig.symbol not in _topk_set:
                    _rank_audit.append({
                        "symbol": _rsig.symbol, "stage": "RANKING", "passed": False,
                        "reason": "below_top_k_cutoff",
                        "rsr": _rsig.rsr, "rsr_rank": _rsig.rsr_rank, "top_k": self.top_k,
                    })
            _full_stage_audit = _rank_audit + _stage_audit_sink
            append_stage_audit(
                today_str=today_str,
                decisions=_full_stage_audit,
            )
            # EVS統合用: run_live_signal.py / run_morning_signal.py がファイル再読込
            # なしで直接参照できるようインスタンス属性にも保持する（観測専用）。
            self._last_stage_audit = _full_stage_audit
        except Exception as _sa_err:
            logger.warning("[LIVE_STAGE_AUDIT] append failed (%s) — continuing", _sa_err)
            self._last_stage_audit = list(_stage_audit_sink)

        # ── [UNIVERSE] structured log ──────────────────────────────────────────
        _n_live        = len(self.universe_tickers)
        _n_shadow      = len(self.shadow_universe_tickers)
        _n_rsr_context = len(self.rsr_universe_tickers)
        _n_tradeable   = sum(1 for s in self.universe_tickers if s in universe_raw)
        _n_filt_price  = _diag.get("blocked_by_price", 0)
        _n_filt_risk   = _risk_rejected
        logger.info(
            "[UNIVERSE] live=%d shadow=%d context=%d tradeable=%d filtered_price=%d filtered_risk=%d",
            _n_live, _n_shadow, _n_rsr_context, _n_tradeable, _n_filt_price, _n_filt_risk,
        )

        # ── [CLUSTER] concentration diagnostics ───────────────────────────────
        _live_buy_orders = [o for o in orders if o.side == "BUY"]
        _sector_exposure: dict[str, int] = {}
        for _o in _live_buy_orders:
            _sector_exposure[_o.sector] = _sector_exposure.get(_o.sector, 0) + 1
        _missed_entries = max(0, _diag.get("buy_candidates", 0) - len(_live_buy_orders))
        logger.info(
            "[CLUSTER] sector_exposure=%s rejected_alloc_cap=%d risk_rejected=%d missed_entries=%d",
            _sector_exposure, _blocked_alloc_cap, _risk_rejected, _missed_entries,
        )

        # 6b. Shadow Phase1 — observation_only（実発注しない。SAFETY FIX 2026-07-07）
        # 条件: CB NORMAL AND remaining_slots>0 AND shadow_rsr_pass>=8
        #       AND rsr62>=70 AND rsr62>live_top10_median AND pre_trade_risk_check通過
        # shadow_orders は常に [] を返す（+= は将来の昇格経路のための構造維持のみ）。
        _cash_after_live_buys = (
            calc_available_cash
            - sum(o.estimated_amount for o in orders if o.side == "BUY")
        )
        _shadow_virtual_positions = portfolio_state.get("shadow_virtual_positions", {})
        shadow_orders, _shadow_metrics, _new_virtual, _closed_virtual = self._build_shadow_orders(
            diag                     = _diag,
            universe_raw             = universe_raw,
            current_positions        = current_positions,
            available_cash           = _cash_after_live_buys,
            cb_active                = cb_active,
            live_orders              = orders,
            shadow_virtual_positions = _shadow_virtual_positions,
            today_str                = today_str,
            effective_max_pos        = _effective_max_pos,
            above_ma200              = _above_ma200_live,
        )
        # 仮想ポジション状態を portfolio_state に反映（決済→削除、新規→追加）
        for sym in _closed_virtual:
            _shadow_virtual_positions.pop(sym, None)
        _shadow_virtual_positions.update(_new_virtual)
        portfolio_state["shadow_virtual_positions"] = _shadow_virtual_positions
        orders = orders + shadow_orders

        # 6c. LIVE_STATE サマリーログ（戦略停止 / 市場悪化 / フィルター過剰 の切り分け用）
        _entries   = [o for o in orders if o.side == "BUY"]
        _missed_breakout_count = max(0, _buy_cands_count - len(_entries))
        _exposure  = 1.0 - calc_available_cash / max(1.0, current_equity)
        logger.info(
            "LIVE_STATE candidates=%d ranked=%d entries=%d positions=%d exposure=%.3f cluster=%s missed=%d",
            _buy_cands_count, len(top_k_syms), len(_entries),
            len(current_positions), _exposure, _breakout_cluster, _missed_breakout_count,
        )

        # 6c. 運用診断メトリクス → logs/diagnostics/metrics.jsonl に日次追記
        import json as _json
        from pathlib import Path as _Path
        _diag_dir  = LOGS_DIR / "diagnostics"
        _diag_dir.mkdir(parents=True, exist_ok=True)
        _diag_path = _diag_dir / "metrics.jsonl"
        # 市場レジーム（TOPIX ETF 1306.T の 200日MA / 50日MA比較）
        _above_ma200    = None
        _bench_vs_ma    = None
        _trend_market   = None
        _trend_strength = None
        try:
            _bench_close = bench_prices.dropna()
            _ma200       = float(_bench_close.rolling(200).mean().iloc[-1])
            _ma50        = float(_bench_close.rolling(50).mean().iloc[-1])
            _bench_last  = float(_bench_close.iloc[-1])
            _above_ma200 = bool(_bench_last > _ma200)
            _bench_vs_ma = round((_bench_last / _ma200 - 1) * 100, 2)  # %
            # トレンドレジーム分類: bull=MA50>MA200, bear=MA50<MA200（デスクロス）
            if _ma50 > _ma200 * 1.005:
                _trend_market = "bull"
            elif _ma50 < _ma200 * 0.995:
                _trend_market = "bear"
            else:
                _trend_market = "neutral"
            # トレンド強度: (MA50 - MA200) / MA200
            # > +0.05 強トレンド / +0.02〜0.05 通常 / ±0.02 横ばい / < -0.02 下落
            _trend_strength = round((_ma50 - _ma200) / _ma200, 4)
        except Exception:
            pass

        # Step 2: 週次シグナル密度（直近5営業日の日別 candidate_count 合計）
        # 1日に複数回実行しても日ごとに1回分のみカウント（当日の最新値を使用）
        _signals_per_week         = None
        _high20_distance_delta_5d = None  # Step 1: distance の5日変化率（-0.04以下でブレイク前兆）
        try:
            if _diag_path.exists():
                _all_lines = [_json.loads(l) for l in _diag_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                # 日別に最新エントリーを取得
                _by_date_cand: dict[str, int]   = {}
                _by_date_dist: dict[str, float] = {}
                for _e in _all_lines:
                    _by_date_cand[_e["date"]] = _e.get("candidate_count", 0)
                    if _e.get("high20_distance_median") is not None:
                        _by_date_dist[_e["date"]] = float(_e["high20_distance_median"])
                # signals_per_week
                _recent_dates = sorted(_by_date_cand.keys())[-5:]
                _signals_per_week = sum(_by_date_cand[d] for d in _recent_dates)
                # distance delta 5d: 今日の distance - 5営業日前の distance
                _dist_today = _by_date_dist.get(today_str) or _diag.get("high20_distance_median")
                _dist_dates_sorted = sorted(_by_date_dist.keys())
                # 今日を除いた過去5営業日前を探す
                _past_dates = [d for d in _dist_dates_sorted if d < today_str]
                if _dist_today is not None and len(_past_dates) >= 1:
                    # 最大5日前のデータ。5日分なければある分で計算
                    _ref_date = _past_dates[-min(5, len(_past_dates))]
                    _dist_ref = _by_date_dist[_ref_date]
                    _high20_distance_delta_5d = round(float(_dist_today) - _dist_ref, 4)
        except Exception:
            pass

        # Step 3: RSR Top10 ランキング安定性（昨日との重複率）+ Top10滞在半減期
        _top10_overlap        = None
        _rsr_leader_half_life = None
        _rsr_leader_hl_slope  = None   # log-linear slope（回転判定補助）
        _rsr_leader_hl_r2     = None   # R²（<0.2 = fit不安定 → half-life無効）
        _top10_today = [e["symbol"] for e in _diag.get("rsr_distribution", [])[:10]]
        try:
            _rsr_dist_path_tmp = _diag_dir / "rsr_distribution.jsonl"
            if _rsr_dist_path_tmp.exists():
                _dist_lines = _rsr_dist_path_tmp.read_text(encoding="utf-8").splitlines()
                # 同日の記録は除いて直近1件を取得（overlap計算）
                _prev_entries = [l for l in _dist_lines if l.strip() and f'"date": "{today_str}"' not in l]
                if _prev_entries:
                    _prev = _json.loads(_prev_entries[-1])
                    _top10_yesterday = [e["symbol"] for e in _prev.get("top20", [])[:10]]
                    _top10_overlap = len(set(_top10_today) & set(_top10_yesterday))

                # Top10滞在半減期: 連続日のretention率から log半減期を計算
                # >20日=強トレンド / 10〜20=通常 / <8=回転相場
                _all_dist = [_json.loads(l) for l in _dist_lines if l.strip()]
                _all_dist.sort(key=lambda x: x.get("date", ""))
                if len(_all_dist) >= 5:
                    import math as _math
                    _retentions = []
                    for _di in range(1, len(_all_dist)):
                        _s_prev = {e["symbol"] for e in _all_dist[_di-1].get("top20", [])[:10]}
                        _s_curr = {e["symbol"] for e in _all_dist[_di  ].get("top20", [])[:10]}
                        if _s_prev:
                            _retentions.append(len(_s_prev & _s_curr) / len(_s_prev))
                    if len(_retentions) >= 3:
                        # EMA平滑化 → log-linear slope → 半減期推定
                        # 算術平均より日次ノイズを30〜40%削減（回転相場の誤判定防止）
                        _s_ret  = pd.Series(_retentions)
                        _ema    = _s_ret.ewm(span=min(10, len(_s_ret)), adjust=False).mean()
                        _y      = np.log(np.maximum(_ema.values, 1e-6))
                        _x      = np.arange(len(_y), dtype=float)
                        _coeffs    = np.polyfit(_x, _y, 1)
                        _slope     = float(_coeffs[0])
                        _intercept = float(_coeffs[1])
                        # R²計算: fit が不安定な場合（R²<0.2）は half-life を無効扱い
                        _y_pred = _slope * _x + _intercept
                        _ss_tot = float(np.sum((_y - np.mean(_y)) ** 2))
                        _ss_res = float(np.sum((_y - _y_pred) ** 2))
                        _r2     = round(1.0 - _ss_res / _ss_tot, 3) if _ss_tot > 1e-10 else 0.0
                        _rsr_leader_hl_slope = round(_slope, 5)
                        _rsr_leader_hl_r2    = _r2
                        if _slope >= 0:
                            _rsr_leader_half_life = 99.0
                        elif _r2 < 0.2:
                            # フィットが不安定（R²<0.2）→ half-life は計算しても信頼できない
                            _rsr_leader_half_life = None  # 4/8判定では hl_latest=0 扱い
                        else:
                            _rsr_leader_half_life = round(-_math.log(2) / _slope, 1)
                    elif _retentions:
                        # データ不足時フォールバック（算術平均）
                        _avg_ret = float(np.mean(_retentions))
                        if 0 < _avg_ret < 1.0:
                            _rsr_leader_half_life = round(-_math.log(2) / _math.log(_avg_ret), 1)
                        elif _avg_ret >= 1.0:
                            _rsr_leader_half_life = 99.0
        except Exception:
            pass

        _metrics   = {
            "date":                    today_str,
            "run_at":                  now.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "universe_size":           _diag["universe_size"],
            "rsr_pass_count":          _diag["rsr_pass"],       # RSRフィルター通過数（supply診断用）
            "candidate_count":         _diag["buy_candidates"], # 最終BUY候補数（全フィルター通過後）
            "raw_buy_count":           _diag.get("raw_buy_count", 0),          # signal==1 総数
            "holding_excluded_count":  _diag.get("holding_excluded_count", 0), # signal==1 だが保有中で新規対象外
            "duplicate_excluded_count": _diag.get("duplicate_excluded_count", 0), # top_k内だが当日BUY済み
            "new_entries_count":       _diag.get("new_entries_count", 0),      # top_k から duplicate を除いた新規候補
            "blocked_by_rsr":          _diag["blocked_rsr"],    # RSR未達でブロック
            "blocked_by_breakout":     _diag["blocked_breakout"],  # Turtleブレイクアウト未達でブロック
            # RSR集中度（62銘柄コンテキスト全体）
            "rsr_gt80_context":        _diag["rsr_gt80_context"],
            "rsr_gt70_context":        _diag["rsr_gt70_context"],
            "rsr_top_share":           _diag["rsr_top_share"],
            "trend_cluster_mode":      _diag["trend_cluster_mode"],
            "trend_cluster_level":     _diag.get("trend_cluster_level", 0),
            # supply ceiling（live 28銘柄内）
            "rsr_gt70_count":          _diag["rsr_gt70"],
            "rsr_gt60_count":          _diag["rsr_gt60"],
            "rsr_gt50_count":          _diag["rsr_gt50"],
            # Turtle期間比較
            "bo15_extra":              _diag["bo15_extra"],
            "bo10_extra":              _diag["bo10_extra"],
            # Step 2: 週次シグナル密度（理想 2〜6/週）
            "signals_per_week":        _signals_per_week,
            # Step 3: RSR Top10 ランキング安定性（理想 overlap 4〜7）
            "top10_overlap":           _top10_overlap,
            # Top10滞在半減期（>12日=リーダー持続 / <8日=回転相場 / None=R²<0.2で信頼不可）
            "rsr_leader_half_life":    _rsr_leader_half_life,
            "rsr_leader_hl_slope":     _rsr_leader_hl_slope,   # log-linear decay slope
            "rsr_leader_hl_r2":        _rsr_leader_hl_r2,      # R²: <0.2=half-life無効
            "topk_count":              _diag["topk_count"],
            "positions":               len(current_positions),
            "exposure":                round(_exposure, 4),
            "cash_ratio":              round(available_cash / max(1.0, current_equity), 4),
            "market_above_ma200":      _above_ma200,
            "topix_vs_ma200_pct":      _bench_vs_ma,
            # Step 1: 市場トレンドレジーム（bull/neutral/bear）+ トレンド強度
            "trend_market":            _trend_market,
            "trend_strength":          _trend_strength,   # (MA50-MA200)/MA200: >0.05=強 / <-0.02=下落
            # Step 2: ブレイクアウト直前銘柄数（20日高値の2%以内）
            "near_breakout_count":     _diag.get("near_breakout", 0),
            # Step 3: RSR分散（Top20のRSRのstd — 高いほどトップ層が際立つ）
            "rsr_dispersion":          round(float(np.std([e["rsr"] for e in _diag.get("rsr_distribution", [])[:20]])), 2) if _diag.get("rsr_distribution") else None,
            # False Breakout診断（entry後5日以内 かつ -2ATR到達）
            "failed_breakout_count":   _diag.get("failed_breakout", 0),
            "failed_breakout_rate":    round(_diag.get("failed_breakout", 0) / max(1, len(current_positions)), 3) if current_positions else 0.0,
            # 構造的ボトルネック診断（Step 1〜3）
            # RSR Top10 売買ブロック（>20% → ブレイクアウト期待値低下リスク）
            "blocked_leaders_count":   _diag.get("blocked_leaders_count", 0),
            "blocked_leaders_weight":  _diag.get("blocked_leaders_weight", 0.0),
            # RSR Top10 のうち売買可能割合（<0.5 → リーダー集中相場）
            "rsr_top10_tradeable_ratio": round(_diag.get("rsr_top10_tradeable_cnt", 0) / 10, 2),
            # 売買不可の理由別カウント（blocked_by_price が最多 → ポジションサイズ設計の問題）
            "blocked_by_price":        _diag.get("blocked_by_price", 0),
            "blocked_by_liquidity":    _diag.get("blocked_by_liquidity", 0),
            "blocked_by_risk":         _diag.get("blocked_by_risk", 0),
            # 発注フェーズで 1単元コスト > alloc_cap によりスキップした件数
            # 資本制約 vs 市場構造の切り分けに使う（戦略の問題ではなくサイズ設計の問題）
            # blocked_by_alloc_cap: 1単元コスト > alloc_cap で発注不能（Step3: blocked_by_lot）
            # blocked_by_lot:       ATRベース <100株 → フォールバックで100株に丸めた件数
            "blocked_by_alloc_cap":    _blocked_alloc_cap,
            "blocked_by_lot":          _lot_rounded_up,
            # Step 1 (観測バイアス): bo_pressure_raw = near_breakout_count の絶対値
            # bo_rate は RSR供給増加で希薄化するが、raw は市場圧力を直接反映する先行指標
            "bo_pressure_raw":           _diag.get("near_breakout", 0),
            # Step 2 (観測バイアス): RSRコンテキスト全体の通過数と売買可能割合
            # 0.6以下 → 強い銘柄が価格/流動性フィルターで除外されている（候補ゼロの構造的原因）
            "rsr_pass_context_total":    _diag.get("rsr_pass_context_total", 0),
            "rsr_pass_tradeable_ratio":  round(_diag["rsr_pass"] / max(1, _diag.get("rsr_pass_context_total", 1)), 3) if _diag.get("rsr_pass_context_total", 0) > 0 else None,
            # Step 2: RSR Top10 価格分布・セクター集中度
            "rsr_top10_median_price":    _diag.get("rsr_top10_median_price"),
            "rsr_top10_max_price":       _diag.get("rsr_top10_max_price"),
            "rsr_top10_sector_count":    _diag.get("rsr_top10_sector_count"),  # 相場拡散前に増加
            # Step 3: 高値距離ログ（距離収縮で breakout cluster 到来）
            "high20_distance_median":    _diag.get("high20_distance_median"),
            # Step 1: 5日距離変化率（-0.04以下でブレイク前兆 / None=データ蓄積中）
            "high20_distance_delta_5d":  _high20_distance_delta_5d,
            # Step 3追加: 中間圧力カウント（高値10%以内 / near_breakout=5%より早い先行指標）
            "mid_pressure_count":        _diag.get("mid_pressure_count", 0),
            # Step 3追加: 中間圧力重み（RSRスコア加重 / countより市場エネルギーを正確に反映）
            "mid_pressure_weight":       _diag.get("mid_pressure_weight", 0.0),
            # Step 2追加: ブレイク直前圧力 count/weight（高値5%以内 / >=3 or weight>=0.25で相場動く）
            "near_breakout_count":       _diag.get("near_breakout_count", 0),
            "near_breakout_weight":      _diag.get("near_breakout_weight", 0.0),
            # クラスター検知: 同日BUYシグナル数（>=3で effective_max_pos=5 に拡張）
            "breakout_cluster_today":    _buy_cands_count,
            "breakout_cluster_fired":    _breakout_cluster,
            "missed_breakout_count":     _missed_breakout_count,
            # 供給上限診断: RSR通過銘柄のうちブレイク直前の割合（>0.25=十分 / <0.2=停滞）
            "breakout_opportunity_rate": round(_diag.get("near_breakout", 0) / max(1, _diag.get("rsr_pass", 1)), 3) if _diag.get("rsr_pass", 0) > 0 else None,
            # MTFフィルター診断（後方互換）
            "mtf_filtered_candidates": _diag.get("mtf_filtered", 0),
            "mtf_filter_rate":         round(_diag.get("mtf_filtered", 0) / max(1, _diag.get("rsr_pass", 1)), 3) if _diag.get("rsr_pass", 0) > 0 else None,
            # MTF実装（Step 3）: pass率ログ
            # mtf_pass_rate = 3条件通過 / 候補数
            # 0.0 → 市場トレンドなし / 0.3以上 → 相場が来ている
            "mtf_candidates":    _diag.get("mtf_candidates", 0),
            "mtf_wrsr_pass":     _diag.get("mtf_wrsr_pass", 0),
            "mtf_wma_pass":      _diag.get("mtf_wma_pass", 0),
            "mtf_full_pass":     _diag.get("mtf_full_pass", 0),
            "mtf_pass_rate":     round(
                _diag.get("mtf_full_pass", 0) / max(1, _diag.get("mtf_candidates", 1)), 3
            ) if _diag.get("mtf_candidates", 0) > 0 else None,
            # Shadow Universe診断（RSR42母集団と独立 / 監視専用・発注なし）
            # shadow_rsr_pass: pool内RSR>=65の銘柄数（>= 3 で中小型株にも動き始めた合図）
            # shadow_near_bo:  shadow pool内でnear_breakout（8%以内）の銘柄数
            # shadow_promo:    RSR>=68 かつ ¥8,000以下の昇格候補銘柄（自動昇格はしない・確認用）
            "shadow_rsr_pass":   _diag.get("shadow_rsr_pass", 0),
            "shadow_near_bo":    _diag.get("shadow_near_bo", 0),
            "shadow_promo":      _diag.get("shadow_promo_candidates", []),
            "shadow_promo_count": len(_diag.get("shadow_promo_candidates", [])),
            # 追加診断ログ
            # avg_tradeable_rsr: BUY候補のRSR平均（全フィルター通過後 / Noneなら候補ゼロ）
            # blocked_rsr_mean:  blocked_by_price銘柄のRSR平均（高いほど"強いが買えない"銘柄が多い）
            "avg_tradeable_rsr": _diag.get("avg_tradeable_rsr"),
            "blocked_rsr_mean":  _diag.get("blocked_rsr_mean"),
            # Step 1: RSR供給量比率（rsr_pass / RSR42母集団数）
            # 0.10=相場停止 / 0.20=初動 / 0.30=トレンド開始
            "rsr_supply_ratio":  round(
                _diag["rsr_pass"] / max(1, _diag.get("rsr42_total", 42)), 3
            ),
            # Step 3: 市場リーダーブロック率（RSR Top10のうち買えない銘柄の割合）
            # 0.3以下=正常 / 0.5超=機会損失 / 0.7=大型株主導でリーダーが全員買えない状態
            "market_leader_block_rate": round(
                _diag.get("blocked_leaders_count", 0) / 10.0, 2
            ),
            # 資本連動パラメータ（資本変更の効果を追跡する）
            # capital増加 → max_allocation増加 → blocked_by_price減少 の連動を可視化
            "capital":             self.capital,
            "max_allocation":      round(self.capital * 0.30, 0),
            "max_position_yen":    round(self.capital * self.max_single_weight, 0),
            "leader_slot_yen":     round(self.capital * 0.35, 0),
            "risk_per_trade_yen":  round(self.capital * 0.0125, 0),
            # 整合性監査フィールド（PHASE A/B 追加）
            "configured_max_positions":     self.max_positions,
            "effective_max_positions":      _effective_max_pos,
            "dynamic_max_positions":        _cdos_dyn_raw,
            "max_new_positions_per_day":    self.max_new_positions_per_day,
            "price_filter_block_count":     _diag.get("blocked_by_price", 0),
            "liquidity_filter_block_count": _diag.get("blocked_by_liquidity", 0),
            # データ健全性メトリクス（OHLCV欠損補完の状態を記録）
            # rsr_missing_count: 1日以上のCloseが欠損していた銘柄数
            # rsr_filled_count:  ffill(limit=3)で補完できた銘柄数
            # rsr_excluded_count: 欠損>3日でRSR計算から除外した銘柄数
            "rsr_missing_count":   getattr(self, "_last_data_health", {}).get("rsr_missing_count",  0),
            "rsr_filled_count":    getattr(self, "_last_data_health", {}).get("rsr_filled_count",   0),
            "rsr_excluded_count":  getattr(self, "_last_data_health", {}).get("rsr_excluded_count", 0),
            "cache_fallback_count": len(getattr(self, "_last_data_health", {}).get("cache_fallback_syms", [])),
            # Shadow Phase1 観測メトリクス
            # shadow_signal_count:     RSR/価格条件を満たした候補数（発注前）
            # shadow_entry_count:      実際に発注した件数（≤ shadow_slots=1）
            # shadow_blocked_by_alloc: 価格上限でブロックされた件数
            # shadow_rsr_pass_met:     発動条件（rsr_pass>=8）を満たしているか
            "shadow_signal_count":         _shadow_metrics.get("shadow_signal_count",         0),
            "shadow_entry_count":          _shadow_metrics.get("shadow_entry_count",          0),
            "shadow_blocked_by_alloc":     _shadow_metrics.get("shadow_blocked_by_alloc",     0),
            "shadow_rsr_pass_met":         _shadow_metrics.get("shadow_rsr_pass_met",         False),
            "shadow_virtual_open_count":   _shadow_metrics.get("shadow_virtual_open_count",   0),
            "shadow_virtual_entries":      _shadow_metrics.get("shadow_virtual_entries",       []),
            "shadow_virtual_closed":       _shadow_metrics.get("shadow_virtual_closed",        []),
        }
        # Step 2: Shadow signal generation trigger (diagnostic record only)
        # rsr_pass >= 8 OR near_breakout >= 3 → shadow買いシグナル生成条件を記録する。
        # ※ LIVE_UNIVERSE_FILE への昇格は run_universe_governance() で完結済み（上流処理）。
        #    shadow_promo_candidates はシグナル生成候補であり、宇宙ファイル変更とは無関係。
        _promo_trigger = (
            _diag["rsr_pass"] >= 8
            or _diag.get("near_breakout_count", 0) >= 3
        )
        _metrics["shadow_promo_triggered"] = _promo_trigger
        if _promo_trigger and _diag.get("shadow_promo_candidates"):
            _trigger_reason = (
                f"rsr_pass={_diag['rsr_pass']}(>=8)"
                if _diag["rsr_pass"] >= 8
                else f"near_breakout={_diag.get('near_breakout_count',0)}(>=3)"
            )
            logger.debug(
                "[SHADOW] signal_gen trigger [%s]: candidates=%s",
                _trigger_reason, _diag.get("shadow_promo_candidates", []),
            )

        with _diag_path.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_metrics, ensure_ascii=False) + "\n")
        logger.info("診断メトリクス保存: %s", _diag_path)

        # RSR分布ログ → logs/diagnostics/rsr_distribution.jsonl
        _rsr_dist_path = _diag_dir / "rsr_distribution.jsonl"
        _rsr_dist_entry = {
            "date":            today_str,
            "run_at":          now.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "min_rsr_threshold": float(self._fujiko_params_live["min_rsr"]),
            "top20":           _diag.get("rsr_distribution", []),
            "threshold_zone":  [  # 閾値±5のゾーンにいる銘柄（最適点特定用）
                e for e in _diag.get("rsr_distribution", [])
                if abs(e["rsr"] - float(self._fujiko_params_live["min_rsr"])) <= 10
            ],
        }
        with _rsr_dist_path.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_rsr_dist_entry, ensure_ascii=False) + "\n")
        logger.info("RSR分布ログ保存: %s", _rsr_dist_path)

        # 7. 結果オブジェクト構築
        equity_peak = float(portfolio_state.get("equity_peak", self.capital))
        result = BridgeResult(
            generated_at   = now.strftime("%Y-%m-%dT%H:%M:%S%z"),
            mode           = "LIVE" if self.live else "DRY_RUN",
            data_as_of     = data_as_of,
            n_universe     = len(universe_raw),
            cb_state       = portfolio_state["cb_state"],
            top_k_symbols  = top_k_syms,
            portfolio_summary = {
                "available_cash":   available_cash,
                "cash_for_calc":    round(calc_available_cash, 0),
                "current_equity":   round(current_equity, 0),
                "equity_peak":      round(equity_peak, 0),
                "current_drawdown": round(compute_drawdown(current_equity, equity_peak), 4),
                "current_positions":   len(current_positions),
                "max_positions":       self.max_positions,
                "equity_based_max_pos": _equity_based_max,   # PARAMS_LOCKED clamp 後の値
                "cdos_dyn_raw":         _cdos_dyn_raw,        # CDOS raw（クランプ前）
                "open_slots":          max(0, _effective_max_pos - len(current_positions)),
                "portfolio_mode":      "actual" if self._positions_api_status["ok"] else "virtual",
                "positions_api":       self._positions_api_status.copy(),
                "wallet_api":          self._wallet_api_status.copy(),
                "cb_state":            portfolio_state["cb_state"],
                "cb_cooldown_end":     portfolio_state.get("cb_cooldown_end_date"),
                "n_live":              _n_live,
                "n_shadow":            _n_shadow,
                "n_rsr_context":       _n_rsr_context,
                "n_tradeable":         _n_tradeable,
            },
            universe_stats = {
                "live":           _n_live,
                "shadow":         _n_shadow,
                "rsr_context":    _n_rsr_context,
                "tradeable":      _n_tradeable,
                "filtered_price": _n_filt_price,
                "filtered_risk":  _n_filt_risk,
            },
            signals = [
                {
                    "symbol":            s.symbol,
                    "sector":            s.sector,
                    "signal":            s.signal,
                    "strategy_type":     s.strategy_type,
                    "rsr":               round(s.rsr, 1),
                    "rsr_rank":          s.rsr_rank,
                    "sepa_score":        s.sepa_score,
                    "rsr_momentum":      round(s.rsr_mom, 2),
                    "hold_days":          s.hold_days,
                    "entry_date_known":   s.entry_date_known,
                    "currently_holding":  s.currently_holding,
                    "reason":             s.reason,
                    "stop_price":         round(s.trailing_stop_price, 0) if s.trailing_stop_price > 0 else 0,
                    "trailing_stop":      round(s.trailing_stop_price, 0) if s.trailing_stop_price > 0 else 0,
                    "entry_price":        round(s.entry_price, 0) if s.entry_price > 0 else 0,
                    "unrealized_pnl_pct": round(s.unrealized_pnl_pct, 4),
                    # execution quality logging (observation-only)
                    "rsr_pct_raw":        _diag.get("rsr_pct_raw",    {}).get(s.symbol),
                    "rsr_pct_smooth":     _diag.get("rsr_pct_smooth", {}).get(s.symbol),
                    # Entry Timing Intelligence (observation-only tiebreaker)
                    "entry_timing_score":      _diag.get("entry_timing_scores", {}).get(s.symbol, {}).get("score"),
                    "entry_timing_confidence": _diag.get("entry_timing_scores", {}).get(s.symbol, {}).get("confidence"),
                    "entry_timing_action":     _diag.get("entry_timing_scores", {}).get(s.symbol, {}).get("action"),
                    "entry_timing_phase":      _diag.get("entry_timing_scores", {}).get(s.symbol, {}).get("phase"),
                    # Position Sizing Intelligence (observation-only Phase 1)
                    "conviction_score":        _ps_results[s.symbol].conviction_score if s.symbol in _ps_results else None,
                    "virtual_weight":          _ps_results[s.symbol].virtual_weight   if s.symbol in _ps_results else None,
                    "distance_25ma_pct":       _diag.get("sym_dist25ma", {}).get(s.symbol),
                    "entry_signal_time":  now.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                for s in sorted(signals, key=lambda s: s.rsr_rank)
            ],
            orders = [
                {
                    "symbol":           o.symbol,
                    "sector":           o.sector,
                    "side":             o.side,
                    "qty":              o.qty,
                    "order_type":       o.order_type,
                    "estimated_price":  o.estimated_price,
                    "estimated_amount": o.estimated_amount,
                    "reason":           o.reason,
                }
                for o in orders
            ],
            warnings = order_warnings,
        )

        # 8. ポートフォリオ状態保存（LIVE のみ。DRY は [DRY_STATE_GUARD] で保存スキップ済み）
        if self.live:
            self._save_portfolio_state(portfolio_state)

        return result, orders

    # ------------------------------------------------------------------ #
    # ブローカー同期 API（position_sync.py から呼ばれる）
    # ------------------------------------------------------------------ #
    def get_broker_positions(self) -> Dict[str, int]:
        """broker API から現在の保有ポジション {symbol: qty} を取得。
        既存の _get_current_positions() を利用して重複実装を避ける。
        API 未接続の場合は空辞書を返す。
        """
        if self._client is None:
            logger.warning("[SYNC] _client is None. Cannot fetch broker positions.")
            return {}
        try:
            raw_positions = self._client.get_positions()
        except Exception as e:
            logger.error("[SYNC] get_positions() failed: %s", e)
            raise

        from src.common.position_normalizer import filter_live_positions
        live_positions = filter_live_positions(raw_positions)
        result: Dict[str, int] = {}
        for p in live_positions:
            sym_code = p.get("Symbol", "")
            qty = int(p.get("LeavesQty", 0) or 0)
            if sym_code:
                sym = f"{sym_code}.T" if not sym_code.endswith(".T") else sym_code
                result[sym] = qty
        return result

    def get_local_positions(self) -> Dict[str, int]:
        """portfolio_state.json の position_entry_dates キーから {symbol: 1} を返す。
        qty は portfolio_state に記録されていないため 1 をプレースホルダーとして返す。
        drift 検知はシンボルセットの一致/不一致で判定する。
        """
        state = self._load_portfolio_state()
        entry_dates: dict = state.get("position_entry_dates", {})
        return {sym: 1 for sym in entry_dates}

    def overwrite_local_positions(self, broker_positions: Dict[str, int]) -> None:
        """portfolio_state.json の保有銘柄セットをブローカーの値で完全上書きする。
        broker_positions にない銘柄は entry_dates / entry_prices / highest_closes から削除し、
        broker_positions にあってローカルにない銘柄は今日の日付・価格 0 で追加する。
        """
        from datetime import datetime as _dt
        today_str = _dt.now(JST).strftime("%Y-%m-%d")

        state = self._load_portfolio_state()
        entry_dates: dict  = state.setdefault("position_entry_dates",    {})
        entry_prices: dict = state.setdefault("position_entry_prices",   {})
        highest_cls: dict  = state.setdefault("position_highest_closes", {})
        entry_atrs: dict   = state.setdefault("position_entry_atrs",     {})

        broker_syms = set(broker_positions.keys())
        local_syms  = set(entry_dates.keys())

        # ローカルにあるがブローカーにない → 削除
        for sym in local_syms - broker_syms:
            entry_dates.pop(sym, None)
            entry_prices.pop(sym, None)
            highest_cls.pop(sym, None)
            entry_atrs.pop(sym, None)
            logger.info("[SYNC] Removed stale local position: %s", sym)

        # ブローカーにあるがローカルにない → 追加（エントリー価格不明のため 0）
        for sym in broker_syms - local_syms:
            entry_dates[sym]  = today_str
            entry_prices[sym] = 0.0
            highest_cls[sym]  = 0.0
            logger.warning(
                "[SYNC] ⚠️  entry_date を本日 %s に設定しました: %s "
                "（実際の建て日が異なる場合は sync_positions.py --entry-date YYYY-MM-DD で修正してください）",
                today_str, sym,
            )

        self._save_portfolio_state(state)
        logger.info("[SYNC] Overwrote local positions: %s", sorted(broker_syms))
