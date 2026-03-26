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
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# ------------------------------------------------------------------ #
# ポートフォリオ状態・CB 管理定数
# ------------------------------------------------------------------ #
PORTFOLIO_STATE_FILE          = Path("runtime/portfolio_state.json")
OHLCV_CACHE_DIR               = Path("cache/ohlcv")   # OHLCVキャッシュ（ダウンロード失敗時のフォールバック）
OHLCV_CACHE_MAX_AGE_DAYS      = 5                      # キャッシュ有効期限（営業日換算で約1週間）
DATA_HEALTH_MIN_RATIO         = 0.90                   # RSR42データ健全性下限（これ未満でシグナル停止）
CB_DD_TRIGGER                 = 0.15          # DD がここを割ったら CB 発動
CB_COOLDOWN_TRADING_DAYS      = 30            # CB 後のクールダウン（営業日）
RECOVERY_THRESHOLD_RATIO      = 0.98          # peak の何割まで回復したら NORMAL
REENTRY_COOLDOWN_TRADING_DAYS = 5             # 時間ストップ後の再エントリー禁止（営業日）
MIN_DAILY_VALUE_YEN           = 5_000_000_000 # 流動性フィルター（5B 円/日）
ORDER_RATE_LIMIT_PER_MIN      = 3             # kabu API 発注レート（件/分）


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
        from market.jpx_calendar import JPXCalendar
        entry_ts = pd.Timestamp(entry_date_str)
        if today <= entry_ts:
            return 0
        cal = JPXCalendar()
        return max(0, cal.trading_days_between(entry_ts, today) - 1)
    except Exception:
        return 0


def _add_trading_days(start: pd.Timestamp, n: int) -> pd.Timestamp:
    """start から n 営業日後の日付を返す。JPX 祝日対応。"""
    from market.jpx_calendar import JPXCalendar
    cal = JPXCalendar()
    return cal.add_trading_days(start, n)


# ------------------------------------------------------------------ #
# シグナルデータクラス
# ------------------------------------------------------------------ #
@dataclass
class StockSignal:
    """1銘柄分のシグナル情報"""
    symbol:            str
    sector:            str
    signal:            int     # +1=買い / -1=売り / 0=ホールド
    rsr:               float   # RSR 値（0〜100）
    rsr_rank:          int     # top_k ユニバース内の RSR 順位（1=最高）
    sepa_score:        int     # SEPA 条件数（0〜8）
    rsr_mom:           float   # RSR モメンタム
    hold_days:         int     # 現在の保有営業日数（未保有は 0）
    currently_holding: bool    # 現在保有中か
    reason:            str     # シグナル理由（ログ用）
    strategy_type:     str = "fujiko"  # "fujiko" / "mean_rev"


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
    atr20:            float = 0.0  # BUY時ATR20（False Breakout診断用）


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

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


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
        capital:                   float = 2_000_000,
        max_positions:             int   = 4,
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
    ) -> None:
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
        self.max_new_positions_per_day = max_new_positions_per_day
        self.order_rate_interval_sec   = 60.0 / max(1, order_rate_limit_per_min)
        self._state_file               = portfolio_state_file or PORTFOLIO_STATE_FILE

        # min_rsr はパラメータをそのまま使う（research と同じ閾値フィルター）
        self._fujiko_params_live = {**fujiko_params}

        self._client = None
        try:
            from kabusapi.client import KabuClient
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
        """ポートフォリオ状態ファイルを読み込む（存在しなければデフォルト値）"""
        default = {
            "cb_state":              "NORMAL",
            "equity_peak":           self.capital,
            "cb_cooldown_end_date":  None,
            "recovery_threshold":    None,
            "position_entry_dates":  {},
            "position_entry_prices": {},   # BUY時の参考単価（PnL計算用）
            "position_entry_atrs":   {},   # BUY時ATR20（False Breakout診断用）
            "reentry_blocked":       {},
            "last_updated":          None,
        }
        if not self._state_file.exists():
            return default
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("状態ファイル読み込みエラー（デフォルト使用）: %s", e)
            return default

    def _save_portfolio_state(self, state: dict) -> None:
        """ポートフォリオ状態をファイルに保存する"""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state["last_updated"] = datetime.now(JST).strftime("%Y-%m-%d")
        self._state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("ポートフォリオ状態保存: %s", self._state_file)

    def _compute_current_equity(
        self,
        current_positions: dict,
        universe_raw:      dict,
        available_cash:    float,
    ) -> float:
        """現在のポートフォリオ価値（現金 + 含み評価額）を推定する"""
        stock_value = 0.0
        for sym, pos in current_positions.items():
            qty = pos.get("qty", 0)
            if sym in universe_raw:
                price = float(universe_raw[sym]["df"]["Close"].iloc[-1])
            else:
                price = float(pos.get("avg_price", 0.0))
            stock_value += qty * price
        return available_cash + stock_value

    def _update_cb_state(
        self,
        state:          dict,
        current_equity: float,
        today_str:      str,
    ) -> dict:
        """
        サーキットブレーカー状態機械を更新する。

        NORMAL → CB_ACTIVE  : drawdown <= -15%（BUY 全停止）
        CB_ACTIVE → RECOVERY: クールダウン 30 営業日経過
        RECOVERY → NORMAL   : equity >= peak × 98%
        """
        cb_state    = state.get("cb_state", "NORMAL")
        equity_peak = float(state.get("equity_peak", self.capital))

        # equity_peak 更新
        if current_equity > equity_peak:
            equity_peak          = current_equity
            state["equity_peak"] = round(equity_peak, 0)

        drawdown = (current_equity - equity_peak) / equity_peak if equity_peak > 0 else 0.0

        if cb_state == "NORMAL":
            if drawdown <= -CB_DD_TRIGGER:
                today_ts     = pd.Timestamp(today_str)
                cooldown_end = _add_trading_days(today_ts, CB_COOLDOWN_TRADING_DAYS)
                state["cb_state"]             = "CB_ACTIVE"
                state["cb_cooldown_end_date"] = cooldown_end.strftime("%Y-%m-%d")
                state["recovery_threshold"]   = round(equity_peak * RECOVERY_THRESHOLD_RATIO, 0)
                logger.warning(
                    "CB 発動: DD=%.1f%% / peak=¥%s → BUY 停止 / クールダウン終了=%s",
                    drawdown * 100, f"{equity_peak:,.0f}", state["cb_cooldown_end_date"],
                )
                self._save_cb_event("NORMAL→CB_ACTIVE", drawdown, equity_peak, today_str)

        elif cb_state == "CB_ACTIVE":
            end_date = state.get("cb_cooldown_end_date")
            if end_date and today_str >= end_date:
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
                logger.info(
                    "RECOVERY → NORMAL: ¥%s >= threshold=¥%s",
                    f"{current_equity:,.0f}", f"{threshold:,.0f}",
                )
                self._save_cb_event("RECOVERY→NORMAL", drawdown, current_equity, today_str)

        logger.info(
            "CB 状態: %s / equity=¥%s / peak=¥%s / DD=%.1f%%",
            state["cb_state"], f"{current_equity:,.0f}", f"{equity_peak:,.0f}", drawdown * 100,
        )
        return state

    def _save_cb_event(
        self, transition: str, drawdown: float, equity: float, date: str
    ) -> None:
        """CB イベントを logs/cb_events/ に JSONL 形式で追記する"""
        cb_log_dir = Path("logs/cb_events")
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

        # ── Step1: バッチ一括ダウンロード ─────────────────────────────────
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="ticker",
        )

        # ── Step2: 銘柄ごとにDFを切り出す → 失敗分は個別リトライ + キャッシュ ──
        all_universe = {
            **self.rsr_universe_tickers,
            **self.universe_tickers,
            **self.shadow_universe_tickers,
        }
        universe_raw:    dict = {}
        batch_failed:    list[str] = []
        cache_fallbacks: list[str] = []

        for sym, sector in all_universe.items():
            df = pd.DataFrame()
            from_batch = False
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

            if from_batch:
                self._save_to_cache(sym, df)
                universe_raw[sym] = {"df": df, "sector": sector}

        # ── Step3: バッチ失敗分を個別リトライ → それでも駄目ならキャッシュ ──
        if batch_failed:
            logger.warning("バッチ失敗: %d銘柄 → 個別リトライ+キャッシュ試行", len(batch_failed))
        for sym in batch_failed:
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

        # ── Step5: ベンチマーク取得 ────────────────────────────────────────
        df_bench = pd.DataFrame()
        try:
            df_bench = raw[self.benchmark_ticker].copy()
            if isinstance(df_bench.columns, pd.MultiIndex):
                df_bench = df_bench.droplevel(1, axis=1)
            df_bench.dropna(subset=["Close"], inplace=True)
        except (KeyError, TypeError):
            pass
        if df_bench.empty:
            df_bench = self._retry_single_fetch(
                self.benchmark_ticker, start_date, end_date
            )

        logger.info(
            "取得完了: %d / %d 銘柄（バッチ失敗=%d キャッシュ使用=%d）",
            len(universe_raw), len(all_universe) - 1,
            len(batch_failed), len(cache_fallbacks),
        )
        return universe_raw, df_bench["Close"]

    # ------------------------------------------------------------------ #
    # 現在のポジション取得
    # ------------------------------------------------------------------ #
    def _get_current_positions(self) -> dict[str, dict]:
        if self._client is None:
            logger.info("API 未接続: ポジションは空として扱います")
            return {}

        raw_positions = self._client.get_positions()
        positions = {}
        for p in raw_positions:
            sym_code = p.get("Symbol", "")
            sym = f"{sym_code}.T"
            if sym in self.universe_tickers:
                positions[sym] = {
                    "qty":       p.get("LeavesQty", 0),
                    "avg_price": p.get("Price", 0.0),
                }
        logger.info("現在のポジション: %d 銘柄", len(positions))
        return positions

    # ------------------------------------------------------------------ #
    # 資金確認
    # ------------------------------------------------------------------ #
    def _get_available_cash(self, current_positions: dict) -> float:
        if self._client is None:
            n_held = len(current_positions)
            n_free = max(0, self.max_positions - n_held)
            return self.capital / self.max_positions * n_free

        wallet = self._client.get_wallet_cash()
        return float(wallet.get("StockAccountWallet", self.capital))

    # ------------------------------------------------------------------ #
    # MTFキャッシュ（朝1回計算 → 日中はファイル参照）
    # ------------------------------------------------------------------ #
    def _build_mtf_cache_for_day(
        self,
        universe_raw: dict,
        today_str: str,
    ) -> dict:
        """
        週足RSR + 週足MA20フィルターを計算してキャッシュに保存する。
        当日キャッシュが存在すれば計算をスキップして読み込む。

        週足データは金曜引けで確定するため日中に変化しない。
        朝1回だけ計算し、日中の複数回実行はキャッシュを参照するだけにする。

        Returns:
            {
              "rsr_weekly":  {sym: float},   # 週足RSRスコア
              "weekly_ma_ok": {sym: bool},   # weekly_close > weekly_MA20
              "mtf_ok":      {sym: bool},    # rsr_weekly>=75 AND weekly_ma_ok
              "from_cache":  bool,
            }
        """
        import json as _j
        _cache_dir  = Path("cache")
        _cache_path = _cache_dir / f"mtf_state_{today_str}.json"

        # ── キャッシュ読み込み ──────────────────────────────────────────
        if _cache_path.exists():
            try:
                _cached = _j.loads(_cache_path.read_text(encoding="utf-8"))
                if _cached.get("date") == today_str:
                    logger.info("MTFキャッシュ読み込み: %s", _cache_path)
                    _rw = _cached.get("rsr_weekly",  {})
                    _mo = _cached.get("weekly_ma_ok", {})
                    return {
                        "rsr_weekly":   _rw,
                        "weekly_ma_ok": _mo,
                        "mtf_ok":       {s: (_rw.get(s, 0) >= 75.0 and _mo.get(s, False))
                                         for s in set(_rw) | set(_mo)},
                        "from_cache":   True,
                    }
            except Exception as _e:
                logger.warning("MTFキャッシュ読み込みエラー → 再計算: %s", _e)

        # ── 週足Close 一括計算 ──────────────────────────────────────────
        _wc_all: dict[str, pd.Series] = {}
        for _sym in universe_raw:
            try:
                _wc = universe_raw[_sym]["df"]["Close"].resample("W-FRI").last().dropna()
                _wc_all[_sym] = _wc
            except Exception:
                pass

        # ── 週足RSR（42銘柄同一母集団）──────────────────────────────────
        _rsr_weekly: dict[str, float] = {}
        try:
            _rsr_syms = {s: _wc_all[s] for s in self.rsr_universe_tickers if s in _wc_all}
            if len(_rsr_syms) >= 2:
                _wc_df  = pd.DataFrame(_rsr_syms)
                _wr3    = _wc_df / _wc_df.shift(13) - 1
                _wr6    = _wc_df.shift(13) / _wc_df.shift(26) - 1
                _wr9    = _wc_df.shift(26) / _wc_df.shift(39) - 1
                _wr12   = _wc_df.shift(39) / _wc_df.shift(52) - 1
                _wcomp  = 0.4 * _wr3 + 0.2 * _wr6 + 0.2 * _wr9 + 0.2 * _wr12
                _wrsr_s = (_wcomp.rank(axis=1, pct=True).iloc[-1] * 100).clip(0, 100)
                _rsr_weekly = {s: round(float(v), 2) for s, v in _wrsr_s.items() if not np.isnan(v)}
        except Exception as _e:
            logger.warning("週足RSR計算エラー: %s", _e)

        # ── 週足MA20フィルター ─────────────────────────────────────────
        _weekly_ma_ok: dict[str, bool] = {}
        for _sym, _wc in _wc_all.items():
            try:
                _wma20 = _wc.rolling(20).mean().dropna()
                if len(_wma20) >= 5:
                    _weekly_ma_ok[_sym] = bool(float(_wc.iloc[-1]) > float(_wma20.iloc[-1]))
                else:
                    _weekly_ma_ok[_sym] = False   # データ不足 = 通過させない
            except Exception:
                _weekly_ma_ok[_sym] = False       # 計算失敗 = HOLD

        # ── キャッシュ書き込み ──────────────────────────────────────────
        try:
            _cache_dir.mkdir(parents=True, exist_ok=True)
            _cache_path.write_text(
                _j.dumps({
                    "date":         today_str,
                    "rsr_weekly":   _rsr_weekly,
                    "weekly_ma_ok": _weekly_ma_ok,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("MTFキャッシュ書き込み: %s", _cache_path)
        except Exception as _e:
            logger.warning("MTFキャッシュ書き込みエラー: %s", _e)

        return {
            "rsr_weekly":   _rsr_weekly,
            "weekly_ma_ok": _weekly_ma_ok,
            "mtf_ok":       {s: (_rsr_weekly.get(s, 0) >= 75.0 and _weekly_ma_ok.get(s, False))
                             for s in set(_rsr_weekly) | set(_weekly_ma_ok)},
            "from_cache":   False,
        }

    # ------------------------------------------------------------------ #
    # シグナル生成（Top-k ローテーション + 時間ストップ）
    # ------------------------------------------------------------------ #
    def _generate_all_signals(
        self,
        universe_raw:      dict,
        bench_prices:      pd.Series,
        current_positions: dict,
        portfolio_state:   dict,
    ) -> tuple[list[StockSignal], list[str]]:
        """
        全銘柄のシグナルを生成する。

        Returns:
            (signals, top_k_symbols)
        """
        from backtest.rsr                     import calc_universe_rsr, calc_rsr_momentum, calc_sepa
        from backtest.fujiko_strategy         import FujikoStrategy
        from backtest.mean_reversion_strategy import MeanReversionStrategy

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
        rsr_universe = calc_universe_rsr(rsr_prices)
        rsr_latest   = rsr_universe.iloc[-1]   # 最新スナップショット

        # ── クラスタ相場検出（早期計算 — シグナルループ・Shadow昇格より前に必要）────
        # Step1/2/3 で参照するため rsr_latest 取得直後に計算する
        _pre_ctx_size      = max(1, len(rsr_latest))
        _pre_rsr_gt80      = int((rsr_latest >= 80).sum())
        _pre_cluster_mode  = (_pre_rsr_gt80 / _pre_ctx_size) >= 0.15  # 15%以上でcluster
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
                    # cluster_mode中はRSR>=90・top2に絞る（クラスタ相場では真の強者のみ昇格）
                    # 通常時: RSR>=68・top6
                    _SHADOW_PROMO_RSR   = 90.0 if _pre_cluster_mode else 68.0
                    _SHADOW_PROMO_LIMIT = 2    if _pre_cluster_mode else 6
                    _SHADOW_PROMO_PRICE = 8_000
                    _shadow_promo_cands: list[str] = []
                    for _sp_sym in _shadow_rsr_latest[_shadow_rsr_latest >= _SHADOW_PROMO_RSR].index:
                        if _sp_sym in universe_raw:
                            try:
                                _sp_price = float(universe_raw[_sp_sym]["df"]["Close"].iloc[-1])
                                if _sp_price <= _SHADOW_PROMO_PRICE:
                                    _sp_rsr = float(_shadow_rsr_latest[_sp_sym])
                                    _shadow_promo_cands.append((_sp_rsr, _sp_sym, round(_sp_price, 0)))
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

        # ── MTFキャッシュ（朝1回計算・日中は再計算しない） ──────────────
        # 週足データは金曜引けで確定 → 日中に変化しない。
        # 当日キャッシュがあれば読み込み専用、なければ計算して保存。
        _mtf_state      = self._build_mtf_cache_for_day(universe_raw, today_str)
        _rsr_weekly_map = _mtf_state["rsr_weekly"]   # {sym: float}
        _weekly_ma_ok_map = _mtf_state["weekly_ma_ok"]  # {sym: bool}
        logger.info(
            "MTFキャッシュ from_cache=%s / 週足RSR対象=%d銘柄",
            _mtf_state["from_cache"], len(_rsr_weekly_map),
        )

        # Step 2 (観測バイアス): RSRコンテキスト全体の通過数を記録
        # 売買ユニバースが RSR42 の部分集合のため、強い銘柄が価格フィルターで
        # 除外されると diag_rsr_pass が低いまま固定される問題を検出する。
        # rsr_pass_tradeable_ratio = diag_rsr_pass / _rsr_pass_context_total
        # 0.6 以下 → 高 RSR 銘柄の多くが売買不可（価格 or 流動性フィルターで除外）
        _min_rsr_for_ctx = self._fujiko_params_live.get("min_rsr", 75.0)
        _rsr_pass_context_total = int((rsr_latest >= _min_rsr_for_ctx).sum())

        # ── RSR集中度（62銘柄コンテキスト全体での分布）─────────────────
        _ctx_size          = max(1, len(rsr_latest))
        _rsr_gt80_context  = int((rsr_latest >= 80).sum())
        _rsr_gt70_context  = int((rsr_latest >= 70).sum())
        _rsr_top_share     = round(_rsr_gt80_context / _ctx_size, 3)
        _trend_cluster_mode = _rsr_top_share > 0.10   # RSR80以上が10%超 = 集中相場
        if _trend_cluster_mode:
            logger.warning(
                "TREND_CLUSTER_MODE: RSR80以上=%d/%d (%.1f%%) > 10%% "
                "→ エントリー銘柄集中リスク。モメンタム相場に注意",
                _rsr_gt80_context, _ctx_size, _rsr_top_share * 100,
            )

        logger.info(
            "RSR コンテキスト: %d 銘柄（統一42銘柄）context_pass=%d "
            "RSR80以上=%d(%.1f%%) RSR70以上=%d cluster=%s",
            len(rsr_prices), _rsr_pass_context_total,
            _rsr_gt80_context, _rsr_top_share * 100,
            _rsr_gt70_context, _trend_cluster_mode,
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
        min_rsr_threshold = self._fujiko_params_live.get("min_rsr", 75.0)

        # RSR 順位マップ（全銘柄、1=最高）
        rsr_rank_map: dict[str, int] = {
            sym: int(rank)
            for sym, rank in rsr_latest.rank(ascending=False).items()
        }

        # ── 再エントリー禁止リスト ──────────────────────────────────
        reentry_blocked: dict[str, str] = portfolio_state.get("reentry_blocked", {})
        active_blocked  = {sym for sym, end in reentry_blocked.items() if today_str < end}
        if active_blocked:
            logger.info("再エントリー禁止銘柄: %s", active_blocked)

        # ── 保有エントリー日マップ ───────────────────────────────────
        pos_entry_dates:  dict[str, str]   = portfolio_state.get("position_entry_dates",  {})
        pos_entry_prices: dict[str, float] = portfolio_state.get("position_entry_prices", {})
        pos_entry_atrs:   dict[str, float] = portfolio_state.get("position_entry_atrs",   {})

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

        # ── シグナル生成ループ ───────────────────────────────────────
        signals: list[StockSignal] = []

        for sym, info in universe_raw.items():
            if sym not in self.universe_tickers:
                continue  # RSR 計算専用の銘柄はシグナル生成しない

            df     = info["df"]
            sector = info["sector"]
            rsr    = rsr_universe[sym] if sym in rsr_universe.columns else None

            rsr_now  = float(rsr.iloc[-1])  if rsr is not None and not rsr.empty  else 0.0
            mom      = calc_rsr_momentum(rsr, self.fujiko_params.get("mom_period", 21)) if rsr is not None else None
            mom_now  = float(mom.iloc[-1])  if mom is not None and not mom.empty  else 0.0
            sepa_df  = calc_sepa(df, rsr if rsr is not None else pd.Series(50.0, index=df.index))
            sepa_now = int(sepa_df["sepa_score"].iloc[-1])
            rsr_rank = rsr_rank_map.get(sym, 99)

            currently_holding = sym in current_positions

            # ── 保有営業日数チェック ─────────────────────────────────
            hold_td      = 0
            is_time_exit = False
            if (currently_holding
                    and sym in pos_entry_dates
                    and self.max_hold_days is not None):
                hold_td      = _trading_days_held(pos_entry_dates[sym], today)
                is_time_exit = hold_td >= self.max_hold_days

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
            # 優先順位: 時間ストップ > RSR低下エグジット > 再エントリー禁止 > 戦略シグナル
            is_rank_exit = currently_holding and rsr_now < min_rsr_threshold

            if is_time_exit:
                signal_int    = -1
                strategy_type = "fujiko"
                reason = (
                    f"SELL[時間ストップ]: {hold_td}営業日保有"
                    f"（上限{self.max_hold_days}日） RSR={rsr_now:.1f} rank={rsr_rank}"
                )

            elif is_rank_exit:
                signal_int    = -1
                strategy_type = "fujiko"
                reason = (
                    f"SELL[RSR低下]: RSR={rsr_now:.1f} < 閾値{min_rsr_threshold:.0f}"
                    f" rank={rsr_rank}"
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
                rule = SECTOR_STRATEGY.get(sector, "dynamic")
                if rule == "fujiko":
                    signal_int, strategy_type = f_signal, "fujiko"
                elif rule == "mean_rev":
                    signal_int, strategy_type = m_signal, "mean_rev"
                    # TREND_CLUSTER_MODE中はmean_rev BUYをブロック
                    # 理由: クラスタ相場 = 資金が特定セクターに集中 = 逆張りは構造的弱化に逆らう
                    # 2026-03-16の5411.T(鉄鋼)エントリーがこのパターンで損失
                    if signal_int == 1 and _pre_cluster_mode:
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

                # ── MTFフィルター（BUYシグナルにのみ適用）────────────────
                # 条件: 週足RSR >= 70 AND 週足終値 > 週足MA20
                # ダマシブレイク -30〜40% 削減効果。SELLには影響しない。
                if signal_int == 1 and rsr_now >= min_rsr_threshold:
                    diag_mtf_candidates += 1
                    # MTFキャッシュを参照（日中再計算なし）
                    _rsr_weekly_now = float(_rsr_weekly_map.get(sym, 0.0))
                    _weekly_ma_ok   = bool(_weekly_ma_ok_map.get(sym, False))
                    _wrsr_ok        = _rsr_weekly_now >= 75.0  # 日次と同じ閾値
                    if _wrsr_ok:
                        diag_mtf_wrsr_pass += 1
                    if _weekly_ma_ok:
                        diag_mtf_wma_pass  += 1
                    if _wrsr_ok and _weekly_ma_ok:
                        diag_mtf_full_pass += 1
                    else:
                        # MTFフィルター不通過 → BUYを抑制
                        signal_int = 0
                        strategy_type = "fujiko"
                        _mtf_reason = []
                        if not _wrsr_ok:
                            _mtf_reason.append(f"weekly_RSR={_rsr_weekly_now:.1f}<70")
                        if not _weekly_ma_ok:
                            _mtf_reason.append("weekly_close<=MA20")
                        reason = (
                            f"HOLD[MTF弱]: {' / '.join(_mtf_reason)}"
                            f" daily_RSR={rsr_now:.1f} rank={rsr_rank}"
                        )

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
                elif signal_int == 0 and not reason.startswith("HOLD[MTF"):
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

            signals.append(StockSignal(
                symbol            = sym,
                sector            = sector,
                signal            = signal_int,
                rsr               = rsr_now,
                rsr_rank          = rsr_rank,
                sepa_score        = sepa_now,
                rsr_mom           = mom_now,
                hold_days         = hold_td,
                currently_holding = currently_holding,
                reason            = reason,
                strategy_type     = strategy_type,
            ))

        # BUY 候補を RSR+モメンタム複合スコアでソートして top_k 個に絞る
        # composite = RSR + RSR_momentum × MOM_WEIGHT_ADJ
        # 効果: 上昇加速中の銘柄が同RSRの停滞銘柄より優先される（モメンタム相場で有効）
        # MOM_WEIGHT_ADJ=0.3 → "RSRモメンタム1.0→1.3倍"相当
        _MOM_WEIGHT_ADJ = 0.3   # trend_cluster_mode 時は 0.4 に引き上げ
        if _trend_cluster_mode:
            _MOM_WEIGHT_ADJ = 0.4
        buy_eligible = sorted(
            [(s.rsr + s.rsr_mom * _MOM_WEIGHT_ADJ, s.symbol)
             for s in signals if s.signal == 1 and not s.currently_holding],
            reverse=True,
        )
        top_k_syms = [sym for _, sym in buy_eligible[:self.top_k]]
        # avg_tradeable_rsr: 全フィルター通過後のBUY候補のRSR平均（0件なら None）
        _buy_elig_rsrs = [rsr for rsr, _ in buy_eligible]
        _avg_tradeable_rsr = round(float(np.mean(_buy_elig_rsrs)), 1) if _buy_elig_rsrs else None
        logger.info(
            "BUY 候補（RSR順）: %s → top%d = %s",
            [sym for _, sym in buy_eligible], self.top_k, top_k_syms,
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
            "avg_tradeable_rsr":  _avg_tradeable_rsr,
            "rsr_distribution":   rsr_dist_sorted[:20],
            # Step 2: supply ceiling（live 28銘柄内）
            "rsr_gt80_context":   _rsr_gt80_context,   # 62銘柄中RSR80以上
            "rsr_gt70_context":   _rsr_gt70_context,   # 62銘柄中RSR70以上
            "rsr_top_share":      _rsr_top_share,       # RSR80以上の比率（>0.10でcluster）
            "trend_cluster_mode": _trend_cluster_mode,  # True = モメンタム集中相場
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
        }

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
        shadow_slots:            int   = 1,
        shadow_rsr_min:          float = 70.0,
        shadow_rsr_pass_min:     int   = 8,
    ) -> tuple[list, dict, dict, list]:
        """
        Shadow Universe から条件付きBUY注文を生成する（Phase1）。

        条件（すべて満たす場合のみ発動）:
          1. CB NORMAL かつ shadow_rsr_pass >= shadow_rsr_pass_min (=8)
          2. shadow_rsr62 >= shadow_rsr_min (=70.0)
          3. shadow_rsr62 > live_top10_median
          4. 価格フィルター: price * 100 <= max_alloc_cap
          5. 未保有 かつ live注文と重複なし

        blocked_by_alloc でも仮想エントリーを記録し、研究データを生成する。
        仮想エントリーの決済: RSR < shadow_rsr_min に低下した時点で自動計算。

        Returns: (orders, shadow_metrics, new_virtual_positions, closed_virtual_syms)
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

        # ── 価格フィルター + 注文生成（最大 shadow_slots 件）─────────────
        for rsr62, sym in _candidates:
            if len(orders) >= shadow_slots:
                break
            if sym not in universe_raw:
                continue

            _df    = universe_raw[sym]["df"]
            _price = float(_df["Close"].iloc[-1])
            _cost  = _price * 100   # 1単元

            # 価格フィルター
            if _cost > _max_alloc_cap:
                shadow_metrics["shadow_blocked_by_alloc"] += 1
                logger.info(
                    "SHADOW blocked_by_alloc: %s ¥%.0f/単元 > 上限¥%.0f",
                    sym, _cost, _max_alloc_cap,
                )
                # 仮想エントリー記録（blockedでも研究データとして記録）
                if sym not in shadow_virtual_positions and sym not in _held_syms:
                    new_virtual[sym] = {
                        "entry_price": round(_price, 0),
                        "entry_date":  today_str,
                        "virtual":     True,
                        "rsr62":       round(rsr62, 1),
                    }
                    logger.info(
                        "SHADOW_VIRTUAL_ENTRY: %s @ ¥%.0f RSR62=%.1f (blocked_by_alloc)",
                        sym, _price, rsr62,
                    )
                    shadow_metrics["shadow_virtual_entries"].append({
                        "symbol": sym, "entry_price": round(_price, 0), "rsr62": round(rsr62, 1),
                    })
                continue

            # 余力チェック
            if available_cash < _cost:
                logger.info("SHADOW 余力不足: %s ¥%.0f > 余力¥%.0f", sym, _cost, available_cash)
                continue

            # ATRベースのロットサイズ（1%リスク）
            _risk_yen = self.capital * 0.01
            try:
                _atr = float(_df["Close"].diff().abs().rolling(20).mean().iloc[-1])
                _qty_raw = int(_risk_yen / max(_atr, 1.0))
                _qty     = max(100, (_qty_raw // 100) * 100)
            except Exception:
                _qty = 100

            _qty = min(_qty, int(_max_alloc_cap / _price / 100) * 100)
            _qty = max(100, _qty)

            orders.append(OrderInstruction(
                symbol           = sym,
                symbol_4digit    = sym.replace(".T", ""),
                sector           = self.shadow_universe_tickers.get(sym, "不明"),
                side             = "SHADOW_BUY",
                qty              = _qty,
                order_type       = "MARKET_OPEN",
                estimated_price  = _price,
                estimated_amount = _qty * _price,
                reason           = (
                    f"SHADOW_BUY: RSR62={rsr62:.1f} "
                    f"(>{_live_top10_median:.1f}=live_top10_median) "
                    f"shadow_rsr_pass={_srsr_pass}"
                ),
                atr20            = 0.0,
            ))

        shadow_metrics["shadow_entry_count"] = len(orders)
        if orders:
            logger.info(
                "SHADOW Phase1: %d件 → %s (rsr62条件: >%.1f AND >%.1f)",
                len(orders), [o.symbol for o in orders],
                shadow_rsr_min, _live_top10_median,
            )
        return orders, shadow_metrics, new_virtual, closed_syms

    def _build_orders(
        self,
        signals:              list[StockSignal],
        universe_raw:         dict,
        current_positions:    dict,
        available_cash:       float,
        cb_active:            bool,
        today_new_buys:       int = 0,
        effective_max_pos:    int | None = None,
    ) -> tuple[list[OrderInstruction], list[str]]:
        """
        シグナルリストからポートフォリオルールを適用して注文を生成する。

        Args:
            cb_active:      True のとき新規 BUY を全停止
            today_new_buys: 本日すでに実行済みの新規 BUY 件数（スクリプト再実行時に使用）
        """
        orders:   list[OrderInstruction] = []
        warnings: list[str]              = []

        if cb_active:
            warnings.append("サーキットブレーカー発動中: 新規 BUY を全停止（SELL のみ実行）")
            logger.warning(
                "ENTRY BLOCKED BY CB: BUY 全停止中。"
                " BUY シグナルが出ても発注しません。SELL のみ実行します。"
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

        if cb_active:
            blocked_buys = [s.symbol for s in signals if s.signal == 1 and not s.currently_holding]
            if blocked_buys:
                logger.warning(
                    "ENTRY BLOCKED BY CB: 以下 %d 銘柄の BUY をスキップ → %s",
                    len(blocked_buys), blocked_buys,
                )
            return orders, warnings, 0  # CB 中は SELL のみ

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

        sector_count: dict[str, int] = {}
        for sym in current_positions:
            if sym not in [o.symbol for o in orders if o.side == "SELL"]:
                sector = self.universe_tickers.get(sym, "不明")
                sector_count[sector] = sector_count.get(sector, 0) + 1

        _eff_max_pos      = effective_max_pos if effective_max_pos is not None else self.max_positions
        max_per_sector    = max(1, _eff_max_pos // max(1, self.min_sectors))
        max_alloc_cap     = self.capital * self.max_single_weight
        new_buys_this_run = today_new_buys
        blocked_by_alloc_cap_count = 0  # 配分上限キャップで qty_cap=0 になった件数
        lot_rounded_up_count       = 0  # ATR計算 <100株 → 最低1単元フォールバックした件数
        # リーダースロット設計: RSR >= 85 の最高位銘柄に1スロットだけ大きめ配分を許可
        # 大型株主導相場（blocked_leaders_weight>40%）でも高RSR銘柄を取りこぼさないため
        _LEADER_RSR_THRESHOLD = 85.0
        _LEADER_SLOT_WEIGHT   = 0.35     # 70万円/200万円 — 通常上限0.20の約1.75倍
        _leader_slot_used     = False

        for i, sig in enumerate(buy_candidates):
            open_slots = _eff_max_pos - n_held_after_sells
            if open_slots <= 0:
                warnings.append(
                    f"最大ポジション数({_eff_max_pos})に達したため"
                    f" {sig.symbol} の BUY をスキップ"
                )
                break

            # max_new_positions_per_day チェック
            if new_buys_this_run >= self.max_new_positions_per_day:
                warnings.append(
                    f"本日の新規 BUY 上限({self.max_new_positions_per_day}件)に達しました。"
                    f" {sig.symbol} をスキップ"
                )
                break

            # セクター分散チェック
            if sector_count.get(sig.sector, 0) >= max_per_sector:
                warnings.append(
                    f"セクター集中制限: {sig.sector} が上限({max_per_sector})に達しているため"
                    f" {sig.symbol} をスキップ"
                )
                continue

            _df_buy   = universe_raw[sig.symbol]["df"]
            ref_price = float(_df_buy["Close"].iloc[-1])
            lot_cost  = ref_price * 100  # 1単元（100株）コスト

            # ── リスクベース・ポジションサイジング（ATR制御 + 配分上限の二重制御）
            # risk_per_trade = capital × 1%（1トレード最大リスク額）
            # qty_risk  = risk_per_trade / ATR20  （ATRベース株数）
            # qty_cap   = alloc_cap / price        （配分上限ベース株数）
            # qty       = min(qty_risk, qty_cap)  → 小さい方を採用
            _risk_pct      = 0.01                              # 1% リスク
            _risk_per_trade = self.capital * _risk_pct        # 200万 × 1% = 2万円
            _atr_for_size  = 0.0
            try:
                _tr = pd.concat([
                    _df_buy["High"] - _df_buy["Low"],
                    (_df_buy["High"] - _df_buy["Close"].shift()).abs(),
                    (_df_buy["Low"]  - _df_buy["Close"].shift()).abs(),
                ], axis=1).max(axis=1)
                _av = float(_tr.rolling(20).mean().iloc[-1])
                if not np.isnan(_av) and _av > 0:
                    _atr_for_size = _av
            except Exception:
                pass

            if _atr_for_size > 0:
                # ATRベース: risk / ATR20 → 単元に丸める → 0になった場合は最低1単元フォールバック
                qty_raw  = int(_risk_per_trade / _atr_for_size)
                qty_risk = (qty_raw // 100) * 100
                if qty_risk == 0:
                    # ATR計算結果 < 100株 → ロット制約 → 最低1単元を試みる
                    # 後段の qty_cap == 0 チェックが「それでも買えない」を保護する
                    qty_risk = 100
                    lot_rounded_up_count += 1
                    logger.debug(
                        "%s: ATRベース qty_raw=%d → 100株フォールバック (ATR=%.0f risk=¥%.0f)",
                        sig.symbol, qty_raw, _atr_for_size, _risk_per_trade,
                    )
            else:
                # ATR取得失敗時は配分上限ベースにフォールバック
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
                continue

            qty = min(qty_risk, qty_cap)

            if qty <= 0:
                warnings.append(
                    f"{sig.symbol}: リスクベースサイジング結果qty=0"
                    f" (ATR={_atr_for_size:,.0f} risk=¥{_risk_per_trade:,.0f} price=¥{ref_price:,.0f})"
                    f" → BUY スキップ"
                )
                continue

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

            orders.append(OrderInstruction(
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
            ))

            sector_count[sig.sector] = sector_count.get(sig.sector, 0) + 1
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

        return orders, warnings, blocked_by_alloc_cap_count, lot_rounded_up_count

    # ------------------------------------------------------------------ #
    # 発注実行（live モードのみ）レート制限付き
    # ------------------------------------------------------------------ #
    def _send_orders(self, orders: list[OrderInstruction]) -> list[dict]:
        """
        kabuステーション API に注文を送信する。
        レート制限: ORDER_RATE_LIMIT_PER_MIN 件/分（デフォルト 3件/分 = 20秒/件）
        """
        from kabusapi.client import Side, OrderType, Exchange

        now         = datetime.now(JST)
        market_hour = now.hour * 60 + now.minute
        order_type  = OrderType.MARKET_OPEN if market_hour < 9 * 60 else OrderType.MARKET
        logger.info(
            "注文タイプ: %s",
            "寄成（MARKET_OPEN）" if market_hour < 9 * 60 else "成行（MARKET）",
        )

        results = []
        for idx, o in enumerate(orders):
            side_code = Side.BUY if o.side in ("BUY", "SHADOW_BUY") else Side.SELL

            # レート制限: 2件目以降にインターバルを挿入
            if idx > 0:
                time.sleep(self.order_rate_interval_sec)

            try:
                result = self._client.send_order(
                    symbol     = o.symbol_4digit,
                    exchange   = Exchange.SOR,
                    side       = side_code,
                    qty        = o.qty,
                    order_type = order_type,
                )
                results.append({
                    "symbol":          o.symbol,
                    "side":            o.side,
                    "qty":             o.qty,
                    "estimated_price": o.estimated_price,
                    "atr20":           o.atr20,
                    "sector":          o.sector,
                    "reason":          o.reason,
                    "order_id":        result.order_id,
                    "success":         result.success,
                    "result_code":     result.result_code,
                })
                status = "✅ 成功" if result.success else "❌ 失敗"
                logger.info(
                    "%s %s %s %d株 (OrderId: %s)",
                    status, o.side, o.symbol, o.qty, result.order_id,
                )
            except Exception as e:
                logger.error("%s %s 注文送信エラー: %s", o.side, o.symbol, e)
                results.append({
                    "symbol":          o.symbol,
                    "side":            o.side,
                    "qty":             o.qty,
                    "estimated_price": o.estimated_price,
                    "atr20":           o.atr20,
                    "sector":          o.sector,
                    "reason":          o.reason,
                    "success":         False,
                    "error":           str(e),
                })

        return results

    # ------------------------------------------------------------------ #
    # 約定後の状態更新（run_live_signal.py から呼ぶ）
    # ------------------------------------------------------------------ #
    def update_state_after_execution(
        self,
        send_results: list[dict],
        today_str:    str,
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

        state             = self._load_portfolio_state()
        pos_entry_dates   = state.setdefault("position_entry_dates",  {})
        pos_entry_prices  = state.setdefault("position_entry_prices", {})
        pos_entry_atrs    = state.setdefault("position_entry_atrs",   {})
        reentry_blocked   = state.setdefault("reentry_blocked",       {})
        shadow_positions  = state.setdefault("shadow_positions",      {})  # {sym: entry_price} shadow由来

        # 最新の市場レジームをメトリクスから取得（regime別成績集計用）
        _latest_regime = None
        _metrics_path  = _Path("logs/diagnostics/metrics.jsonl")
        if _metrics_path.exists():
            try:
                _lines = [l for l in _metrics_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                if _lines:
                    _latest_regime = _json.loads(_lines[-1]).get("trend_market")
            except Exception:
                pass

        _trades_path = _Path("logs/trades.jsonl")
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

            if side in ("BUY", "SHADOW_BUY"):
                atr20 = float(r.get("atr20", 0.0))
                pos_entry_dates[sym]  = today_str
                pos_entry_prices[sym] = price
                if atr20 > 0:
                    pos_entry_atrs[sym] = atr20
                reentry_blocked.pop(sym, None)
                # shadow由来ポジションを記録（SELL時に shadow_realized_return を計算するため）
                if side == "SHADOW_BUY":
                    shadow_positions[sym] = price
                    logger.info("SHADOW entry_date 記録: %s → %s @ ¥%.0f", sym, today_str, price)
                else:
                    logger.info("entry_date 記録: %s → %s @ ¥%.0f ATR20=%.0f", sym, today_str, price, atr20)
                _trade = {
                    "date":         today_str,
                    "symbol":       sym,
                    "sector":       sector,
                    "side":         side,   # "BUY" or "SHADOW_BUY"
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
                # shadow由来ポジション判定（記録を削除して返り値を取得）
                _shadow_entry_price = shadow_positions.pop(sym, None)
                _is_shadow = _shadow_entry_price is not None

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

                if "時間ストップ" in reason:
                    today_ts      = pd.Timestamp(today_str)
                    block_end     = _add_trading_days(today_ts, REENTRY_COOLDOWN_TRADING_DAYS)
                    block_end_str = block_end.strftime("%Y-%m-%d")
                    reentry_blocked[sym] = block_end_str
                    logger.info(
                        "再エントリー禁止: %s 〜 %s（時間ストップ後%d営業日）",
                        sym, block_end_str, REENTRY_COOLDOWN_TRADING_DAYS,
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

        self._save_portfolio_state(state)

    # ------------------------------------------------------------------ #
    # メイン実行
    # ------------------------------------------------------------------ #
    def run(self) -> tuple[BridgeResult, list[OrderInstruction]]:
        """シグナル生成〜注文生成（〜発注）を一括実行する"""
        now       = datetime.now(JST)
        today_str = now.strftime("%Y-%m-%d")

        # 1. ポートフォリオ状態読み込み
        portfolio_state = self._load_portfolio_state()

        # 2. データ取得
        logger.info("データ取得中...")
        universe_raw, bench_prices = self._download_data()
        sample_sym = next(iter(universe_raw))
        data_as_of = str(universe_raw[sample_sym]["df"].index[-1].date())

        # 3. ポジション・余力取得
        current_positions = self._get_current_positions()
        available_cash    = self._get_available_cash(current_positions)

        # 4. 現在 equity → CB 状態更新
        current_equity  = self._compute_current_equity(
            current_positions, universe_raw, available_cash
        )
        portfolio_state = self._update_cb_state(portfolio_state, current_equity, today_str)
        cb_active       = portfolio_state["cb_state"] != "NORMAL"

        # 5. シグナル生成（Top-k + 時間ストップ）
        logger.info("シグナル生成中（%d 銘柄）...", len(self.universe_tickers))
        signals, top_k_syms, _diag = self._generate_all_signals(
            universe_raw, bench_prices, current_positions, portfolio_state
        )

        buy_count  = sum(1 for s in signals if s.signal ==  1)
        sell_count = sum(1 for s in signals if s.signal == -1)
        logger.info(
            "シグナル: BUY=%d / SELL=%d / HOLD=%d",
            buy_count, sell_count, len(signals) - buy_count - sell_count,
        )

        # 6. ブレイクアウトクラスター検知（同日 BUY シグナル数 ≥ 3 → スロット拡張）
        import math as _math
        _CAPITAL_PER_SLOT   = 1_200_000   # 資本連動: 120万/スロット（360万→3, 480万→4, 600万→5）
        _equity_based_max   = max(self.max_positions, _math.floor(current_equity / _CAPITAL_PER_SLOT))
        _CLUSTER_THRESHOLD  = 3
        _buy_cands          = [s for s in signals if s.signal == 1 and not s.currently_holding]
        _buy_cands_count    = len(_buy_cands)
        _breakout_cluster   = _buy_cands_count >= _CLUSTER_THRESHOLD
        _effective_max_pos  = 5 if _breakout_cluster else _equity_based_max
        if _equity_based_max > self.max_positions:
            logger.info(
                "CAPITAL_EXPANSION: equity=¥%.0f → equity_based_max=%d (base=%d)",
                current_equity, _equity_based_max, self.max_positions,
            )
        if _breakout_cluster:
            logger.info(
                "CLUSTER DETECTED: BUY candidates=%d >= threshold=%d → effective_max_pos=%d",
                _buy_cands_count, _CLUSTER_THRESHOLD, _effective_max_pos,
            )

        # 6. 注文生成
        orders, order_warnings, _blocked_alloc_cap, _lot_rounded_up = self._build_orders(
            signals, universe_raw, current_positions, available_cash,
            cb_active=cb_active,
            effective_max_pos=_effective_max_pos,
        )

        # 6b. Shadow Phase1 注文生成（live orders に追加・既存ロジック不変）
        # 条件: CB NORMAL AND shadow_rsr_pass>=8 AND rsr62>=70 AND rsr62>live_top10_median
        # blocked_by_alloc でも仮想エントリーを記録して研究データを生成する
        _cash_after_live_buys = (
            available_cash
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
        )
        # 仮想ポジション状態を portfolio_state に反映（決済→削除、新規→追加）
        for sym in _closed_virtual:
            _shadow_virtual_positions.pop(sym, None)
        _shadow_virtual_positions.update(_new_virtual)
        portfolio_state["shadow_virtual_positions"] = _shadow_virtual_positions
        orders = orders + shadow_orders

        # 6c. LIVE_STATE サマリーログ（戦略停止 / 市場悪化 / フィルター過剰 の切り分け用）
        _entries   = [o for o in orders if o.side in ("BUY", "SHADOW_BUY")]
        _missed_breakout_count = max(0, _buy_cands_count - len(_entries))
        _exposure  = 1.0 - available_cash / max(1.0, current_equity)
        logger.info(
            "LIVE_STATE candidates=%d ranked=%d entries=%d positions=%d exposure=%.3f cluster=%s missed=%d",
            _buy_cands_count, len(top_k_syms), len(_entries),
            len(current_positions), _exposure, _breakout_cluster, _missed_breakout_count,
        )

        # 6c. 運用診断メトリクス → logs/diagnostics/metrics.jsonl に日次追記
        import json as _json
        from pathlib import Path as _Path
        _diag_dir  = _Path("logs/diagnostics")
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
            "blocked_by_rsr":          _diag["blocked_rsr"],    # RSR未達でブロック
            "blocked_by_breakout":     _diag["blocked_breakout"],  # Turtleブレイクアウト未達でブロック
            # RSR集中度（62銘柄コンテキスト全体）
            "rsr_gt80_context":        _diag["rsr_gt80_context"],
            "rsr_gt70_context":        _diag["rsr_gt70_context"],
            "rsr_top_share":           _diag["rsr_top_share"],
            "trend_cluster_mode":      _diag["trend_cluster_mode"],
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
            "risk_per_trade_yen":  round(self.capital * 0.01, 0),
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
        # Step 2: Shadow昇格トリガー判定
        # rsr_pass >= 8 OR near_breakout >= 3 で警告出力（自動昇格は行わない・要ユーザー確認）
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
            logger.warning(
                "⚠ Shadow昇格トリガー発動 [%s]: 候補=%s"
                " → LIVE_UNIVERSE追加を検討してください（要ユーザー確認）",
                _trigger_reason, _diag.get("shadow_promo_candidates", []),
            )
            print(
                f"\n{'='*60}"
                f"\n⚠  Shadow昇格トリガー発動 [{_trigger_reason}]"
                f"\n   昇格候補: {_diag.get('shadow_promo_candidates', [])}"
                f"\n   LIVE_UNIVERSE への追加はユーザー確認後に実施してください。"
                f"\n{'='*60}"
            )

        with _diag_path.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_metrics, ensure_ascii=False) + "\n")
        logger.info("診断メトリクス保存: %s", _diag_path)

        # RSR分布ログ → logs/diagnostics/rsr_distribution.jsonl
        _rsr_dist_path = _diag_dir / "rsr_distribution.jsonl"
        _rsr_dist_entry = {
            "date":            today_str,
            "run_at":          now.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "min_rsr_threshold": self._fujiko_params_live.get("min_rsr", 75.0),
            "top20":           _diag.get("rsr_distribution", []),
            "threshold_zone":  [  # 閾値±5のゾーンにいる銘柄（最適点特定用）
                e for e in _diag.get("rsr_distribution", [])
                if abs(e["rsr"] - self._fujiko_params_live.get("min_rsr", 75.0)) <= 10
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
                "current_equity":   round(current_equity, 0),
                "equity_peak":      round(equity_peak, 0),
                "current_drawdown": round(
                    (current_equity - equity_peak) / max(1.0, equity_peak), 4
                ),
                "current_positions":   len(current_positions),
                "max_positions":       self.max_positions,
                "equity_based_max_pos": _equity_based_max,   # 資本連動上限（360万→3, 480万→4, 600万→5）
                "open_slots":          max(0, _effective_max_pos - len(current_positions)),
                "cb_state":            portfolio_state["cb_state"],
                "cb_cooldown_end":     portfolio_state.get("cb_cooldown_end_date"),
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
                    "hold_days":         s.hold_days,
                    "currently_holding": s.currently_holding,
                    "reason":            s.reason,
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

        # 8. ポートフォリオ状態保存（equity_peak は更新済み）
        self._save_portfolio_state(portfolio_state)

        return result, orders
