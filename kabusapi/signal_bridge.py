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
    ) -> None:
        self.universe_tickers          = universe_tickers
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
    def _download_data(self, lookback_days: int = 600) -> tuple[dict, pd.Series]:
        """ユニバース全銘柄とベンチマークの株価を取得する"""
        import warnings
        warnings.filterwarnings("ignore")

        end_date   = datetime.now(JST).strftime("%Y-%m-%d")
        start_date = (datetime.now(JST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        logger.info("データ取得: %s 〜 %s", start_date, end_date)

        all_tickers = list(
            set(self.rsr_universe_tickers.keys()) | set(self.universe_tickers.keys())
        ) + [self.benchmark_ticker]
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="ticker",
        )

        all_universe = {**self.rsr_universe_tickers, **self.universe_tickers}
        universe_raw = {}
        for sym, sector in all_universe.items():
            try:
                df = raw[sym].copy()
                df.dropna(subset=["Close"], inplace=True)
                if df.empty or len(df) < 252:
                    logger.warning("%s データ不足（%d 日）→ スキップ", sym, len(df))
                    continue
                universe_raw[sym] = {"df": df, "sector": sector}
            except (KeyError, TypeError):
                logger.warning("%s データ取得失敗 → スキップ", sym)

        try:
            df_bench = raw[self.benchmark_ticker].copy()
            df_bench.dropna(subset=["Close"], inplace=True)
        except (KeyError, TypeError):
            df_bench = yf.download(
                self.benchmark_ticker, start=start_date, end=end_date, progress=False
            )
            if isinstance(df_bench.columns, pd.MultiIndex):
                df_bench = df_bench.droplevel(1, axis=1)

        logger.info("取得成功: %d / %d 銘柄", len(universe_raw), len(self.universe_tickers))
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

        # ── RSR 計算（42銘柄統一コンテキスト） ──────────────────────────
        # research / live / backtest で同一の母集団を使うことで
        # RSR percentile の意味を統一する（母集団が変わると別指標になる）
        rsr_prices = {
            sym: info["df"]["Close"]
            for sym, info in universe_raw.items()
            if sym in self.rsr_universe_tickers
        }
        rsr_universe = calc_universe_rsr(rsr_prices)
        rsr_latest   = rsr_universe.iloc[-1]   # 最新スナップショット

        logger.info(
            "RSR コンテキスト: %d 銘柄（統一42銘柄）",
            len(rsr_prices),
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
        pos_entry_dates: dict[str, str] = portfolio_state.get("position_entry_dates", {})

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
        diag_near_breakout = 0

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
                else:
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
                            # Step 4: 20日高値の2%以内（ブレイクアウト直前）
                            if len(close_s) >= 21:
                                high_20 = float(close_s.iloc[-21:-1].max())
                                if high_20 > 0 and (high_20 - price_now) / high_20 < 0.02:
                                    diag_near_breakout += 1
                        except Exception:
                            pass
                else:
                    diag_blocked_rsr += 1

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

        # BUY 候補を RSR 降順でソートして top_k 個に絞る（eligible.sort → eligible[:top_k]）
        buy_eligible = sorted(
            [(s.rsr, s.symbol) for s in signals if s.signal == 1 and not s.currently_holding],
            reverse=True,
        )
        top_k_syms = [sym for _, sym in buy_eligible[:self.top_k]]
        logger.info(
            "BUY 候補（RSR順）: %s → top%d = %s",
            [sym for _, sym in buy_eligible], self.top_k, top_k_syms,
        )

        rsr_dist_sorted = sorted(diag_rsr_dist, key=lambda x: x["rsr"], reverse=True)
        diagnostics = {
            "universe_size":      diag_total,
            "rsr_pass":           diag_rsr_pass,
            "blocked_rsr":        diag_blocked_rsr,
            "blocked_breakout":   diag_blocked_bo,
            "buy_candidates":     len(buy_eligible),
            "topk_count":         len(top_k_syms),
            "rsr_distribution":   rsr_dist_sorted[:20],
            # Step 2: supply ceiling
            "rsr_gt70":           diag_rsr_gt70,
            "rsr_gt60":           diag_rsr_gt60,
            "rsr_gt50":           diag_rsr_gt50,
            # Step 3: Turtle期間比較（何銘柄が15日/10日で追加通過するか）
            "bo15_extra":         diag_bo15_pass,   # 15日なら追加通過する銘柄数
            "bo10_extra":         diag_bo10_pass,   # 10日なら追加通過する銘柄数
            # Step 4: ブレイクアウト直前（20日高値の2%以内）
            "near_breakout":      diag_near_breakout,
        }
        logger.info(
            "DIAG universe=%d rsr_pass=%d blocked_rsr=%d blocked_breakout=%d candidates=%d topk=%d",
            diag_total, diag_rsr_pass, diag_blocked_rsr, diag_blocked_bo,
            len(buy_eligible), len(top_k_syms),
        )

        return signals, top_k_syms, diagnostics

    # ------------------------------------------------------------------ #
    # 注文生成
    # ------------------------------------------------------------------ #
    def _build_orders(
        self,
        signals:           list[StockSignal],
        universe_raw:      dict,
        current_positions: dict,
        available_cash:    float,
        cb_active:         bool,
        today_new_buys:    int = 0,
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
            return orders, warnings  # CB 中は SELL のみ

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

        max_per_sector = max(1, self.max_positions // max(1, self.min_sectors))
        max_alloc_cap  = self.capital * self.max_single_weight
        new_buys_this_run = today_new_buys

        for i, sig in enumerate(buy_candidates):
            open_slots = self.max_positions - n_held_after_sells
            if open_slots <= 0:
                warnings.append(
                    f"最大ポジション数({self.max_positions})に達したため"
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

            n_remaining     = len(buy_candidates) - i
            effective_slots = min(open_slots, n_remaining)
            alloc           = total_cash / max(1, effective_slots)
            alloc           = min(alloc, max_alloc_cap)

            ref_price = float(universe_raw[sig.symbol]["df"]["Close"].iloc[-1])
            lot_cost  = ref_price * 100
            qty       = int(alloc // lot_cost) * 100
            if qty <= 0:
                warnings.append(
                    f"{sig.symbol}: 1単元100株=¥{lot_cost:,.0f} > 配分上限¥{alloc:,.0f}"
                    f" → BUY スキップ"
                )
                continue

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
            ))

            sector_count[sig.sector] = sector_count.get(sig.sector, 0) + 1
            n_held_after_sells       += 1
            new_buys_this_run        += 1
            total_cash               -= qty * ref_price

        return orders, warnings

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
            side_code = Side.BUY if o.side == "BUY" else Side.SELL

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
        reentry_blocked   = state.setdefault("reentry_blocked",       {})

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

            if side == "BUY":
                pos_entry_dates[sym]  = today_str
                pos_entry_prices[sym] = price
                reentry_blocked.pop(sym, None)
                logger.info("entry_date 記録: %s → %s @ ¥%.0f", sym, today_str, price)
                _trade = {
                    "date":         today_str,
                    "symbol":       sym,
                    "sector":       sector,
                    "side":         "BUY",
                    "qty":          qty,
                    "price":        price,
                    "amount":       amount,
                    "entry_regime": _latest_regime,
                    "reason":       reason,
                }
                with _trades_path.open("a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_trade, ensure_ascii=False) + "\n")

            elif side == "SELL":
                entry_price = pos_entry_prices.pop(sym, None)
                entry_date  = pos_entry_dates.pop(sym, None)

                pnl     = round((price - entry_price) * qty, 0) if entry_price else None
                pnl_pct = round((price / entry_price) - 1, 4)   if entry_price else None

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
                    "date":        today_str,
                    "symbol":      sym,
                    "sector":      sector,
                    "side":        "SELL",
                    "qty":         qty,
                    "price":       price,
                    "amount":      amount,
                    "pnl":         pnl,
                    "pnl_pct":     pnl_pct,
                    "hold_days":   hold_days,
                    "entry_price": entry_price,
                    "entry_date":  entry_date,
                    "entry_regime": _latest_regime,
                    "reason":      reason,
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

        # 6. 注文生成
        orders, order_warnings = self._build_orders(
            signals, universe_raw, current_positions, available_cash,
            cb_active=cb_active,
        )

        # 6b. LIVE_STATE サマリーログ（戦略停止 / 市場悪化 / フィルター過剰 の切り分け用）
        _buy_cands = [s for s in signals if s.signal == 1 and not s.currently_holding]
        _entries   = [o for o in orders if o.side == "BUY"]
        _exposure  = 1.0 - available_cash / max(1.0, current_equity)
        logger.info(
            "LIVE_STATE candidates=%d ranked=%d entries=%d positions=%d exposure=%.3f",
            len(_buy_cands), len(top_k_syms), len(_entries),
            len(current_positions), _exposure,
        )

        # 6c. 運用診断メトリクス → logs/diagnostics/metrics.jsonl に日次追記
        import json as _json
        from pathlib import Path as _Path
        _diag_dir  = _Path("logs/diagnostics")
        _diag_dir.mkdir(parents=True, exist_ok=True)
        _diag_path = _diag_dir / "metrics.jsonl"
        # 市場レジーム（TOPIX ETF 1306.T の 200日MA / 50日MA比較）
        _above_ma200  = None
        _bench_vs_ma  = None
        _trend_market = None
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
        except Exception:
            pass

        # Step 2: 週次シグナル密度（直近5営業日の日別 candidate_count 合計）
        # 1日に複数回実行しても日ごとに1回分のみカウント（当日の最新値を使用）
        _signals_per_week = None
        try:
            if _diag_path.exists():
                _all_lines = [_json.loads(l) for l in _diag_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                # 日別に最新エントリーを取得
                _by_date: dict[str, int] = {}
                for _e in _all_lines:
                    _by_date[_e["date"]] = _e.get("candidate_count", 0)
                # 今日を含む直近5営業日
                _today_dt = pd.Timestamp(today_str)
                _recent_dates = sorted(_by_date.keys())[-5:]
                _signals_per_week = sum(_by_date[d] for d in _recent_dates)
        except Exception:
            pass

        # Step 3: RSR Top10 ランキング安定性（昨日との重複率）
        _top10_overlap = None
        _top10_today = [e["symbol"] for e in _diag.get("rsr_distribution", [])[:10]]
        try:
            _rsr_dist_path_tmp = _diag_dir / "rsr_distribution.jsonl"
            if _rsr_dist_path_tmp.exists():
                _dist_lines = _rsr_dist_path_tmp.read_text(encoding="utf-8").splitlines()
                # 同日の記録は除いて直近1件を取得
                _prev_entries = [l for l in _dist_lines if l.strip() and f'"date": "{today_str}"' not in l]
                if _prev_entries:
                    _prev = _json.loads(_prev_entries[-1])
                    _top10_yesterday = [e["symbol"] for e in _prev.get("top20", [])[:10]]
                    _top10_overlap = len(set(_top10_today) & set(_top10_yesterday))
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
            # supply ceiling（RSRは足りているか？）
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
            "topk_count":              _diag["topk_count"],
            "positions":               len(current_positions),
            "exposure":                round(_exposure, 4),
            "cash_ratio":              round(available_cash / max(1.0, current_equity), 4),
            "market_above_ma200":      _above_ma200,
            "topix_vs_ma200_pct":      _bench_vs_ma,
            # Step 1: 市場トレンドレジーム（bull/neutral/bear）
            "trend_market":            _trend_market,
            # Step 2: ブレイクアウト直前銘柄数（20日高値の2%以内）
            "near_breakout_count":     _diag.get("near_breakout", 0),
            # Step 3: RSR分散（Top20のRSRのstd — 高いほどトップ層が際立つ）
            "rsr_dispersion":          round(float(np.std([e["rsr"] for e in _diag.get("rsr_distribution", [])[:20]])), 2) if _diag.get("rsr_distribution") else None,
        }
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
                "current_positions": len(current_positions),
                "max_positions":     self.max_positions,
                "open_slots":        max(0, self.max_positions - len(current_positions)),
                "cb_state":          portfolio_state["cb_state"],
                "cb_cooldown_end":   portfolio_state.get("cb_cooldown_end_date"),
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
