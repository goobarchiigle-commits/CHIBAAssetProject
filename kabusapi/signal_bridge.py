"""
kabusapi/signal_bridge.py
バックテスト戦略 → kabuステーションAPI 発注ブリッジ

【処理フロー（毎朝 8:30 頃に実行）】
  1. yfinance で前日終値までのデータを取得（JST 8:00 以降に実行すること）
  2. ユニバース全銘柄の RSR / SEPA を計算し、generate_signal() を呼ぶ
  3. kabuステーション API で現在の保有ポジションを取得
  4. 売りシグナル → SELL注文 / 買いシグナル → BUY注文 を生成
  5. JSON ファイルに出力（audit trail として常に保存）
  6. --live フラグがある場合のみ kabuステーション API に発注

【安全設計】
  - デフォルトはドライラン（--live なしでは絶対に発注しない）
  - live モードでも確認プロンプトあり（--yes で省略可能）
  - 1回の実行で送れる注文は最大 max_positions 件まで
  - ポートフォリオDD > 15% のときはサーキットブレーカー

【口座開設前の動作確認】
  kabuステーションが起動していなくても --dry-run モードで動作する。
  その場合、現在の保有ポジションは空として扱う。

【CLAUDE.md ルール3 準拠】
  KABU_API_PASSWORD / KABU_TRADE_PASSWORD は .env のみに記述。
  このファイルに認証情報を直接書かないこと。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


# ------------------------------------------------------------------ #
# シグナルデータクラス
# ------------------------------------------------------------------ #
@dataclass
class StockSignal:
    """1銘柄分のシグナル情報"""
    symbol:      str
    sector:      str
    signal:      int          # +1=買い / -1=売り / 0=ホールド
    rsr:         float        # RSR値（0〜100）
    sepa_score:  int          # SEPA条件数（0〜8）
    rsr_mom:     float        # RSRモメンタム
    currently_holding: bool   # 現在保有中か
    reason:      str          # シグナル理由（ログ用）
    strategy_type: str = "fujiko"  # "fujiko" / "mean_rev"


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
    """発注指示（JSON出力 + API送信の両方に使用）"""
    symbol:        str
    symbol_4digit: str        # kabustation用4桁コード（".T"を除去）
    sector:        str
    side:          str        # "BUY" / "SELL"
    qty:           int        # 注文数量（株）
    order_type:    str        # "MARKET_OPEN"（寄成）
    estimated_price: float    # 参考価格（前日終値）
    estimated_amount: float   # 参考約定金額（円）
    reason:        str


@dataclass
class BridgeResult:
    """シグナルブリッジ実行結果"""
    generated_at:   str
    mode:           str       # "DRY_RUN" / "LIVE"
    data_as_of:     str       # データ基準日（前日）
    n_universe:     int
    portfolio_summary: dict
    signals:        list[dict]
    orders:         list[dict]
    warnings:       list[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


# ------------------------------------------------------------------ #
# シグナルブリッジ本体
# ------------------------------------------------------------------ #
class SignalBridge:
    """
    フジコ法シグナル → kabuステーション発注ブリッジ

    Args:
        universe_tickers: {symbol: sector} の辞書
        fujiko_params:    FujikoStrategy のパラメータ辞書
        port_params:      PortfolioEngine 相当のパラメータ
        live:             True のとき API に発注（デフォルト: False）
    """

    def __init__(
        self,
        universe_tickers:  dict[str, str],
        fujiko_params:     dict,
        capital:           float = 2_000_000,
        max_positions:     int   = 6,
        max_dd_limit:      float = 0.15,
        min_sectors:       int   = 3,
        max_single_weight: float = 0.25,   # 1銘柄の最大ウェイト（資本比）
        live:              bool  = False,
        benchmark_ticker:  str   = "1306.T",
    ) -> None:
        self.universe_tickers  = universe_tickers
        self.fujiko_params     = fujiko_params
        self.capital           = capital
        self.max_positions     = max_positions
        self.max_dd_limit      = max_dd_limit
        self.min_sectors       = min_sectors
        self.max_single_weight = max_single_weight
        self.live              = live
        self.benchmark_ticker  = benchmark_ticker

        self._client = None
        # liveモード・ドライランどちらでもAPIクライアントの接続を試みる
        # → kabuステーション起動中なら実際のポジション・余力を取得できる
        # → 未起動の場合はフォールバック（ポジションなし / 資本から推定）
        try:
            from kabusapi.client import KabuClient
            self._client = KabuClient()
            self._client.fetch_token()
            if not live:
                logger.info("API接続成功（読み取り専用: 実ポジション・余力を取得します）")
        except Exception as e:
            if live:
                raise  # liveモードでAPIに繋がらないのは致命的エラー
            logger.info("API未接続（ドライランフォールバック: ポジション空・余力を資本から推定）: %s", e)
            self._client = None

    # ------------------------------------------------------------------ #
    # データ取得
    # ------------------------------------------------------------------ #
    def _download_data(self, lookback_days: int = 600) -> tuple[dict, pd.Series]:
        """
        ユニバース全銘柄と ベンチマークの株価を取得する。

        lookback_days: 今日から遡る日数（RSR計算に252日以上必要）
        """
        import warnings
        warnings.filterwarnings("ignore")

        end_date   = datetime.now(JST).strftime("%Y-%m-%d")
        start_date = (datetime.now(JST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info("データ取得: %s 〜 %s", start_date, end_date)

        # 一括取得でレート制限を回避（個別ループは廃止）
        all_tickers = list(self.universe_tickers.keys()) + [self.benchmark_ticker]
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="ticker",
        )

        universe_raw = {}
        for sym, sector in self.universe_tickers.items():
            try:
                df = raw[sym].copy()
                df.dropna(subset=["Close"], inplace=True)  # Close=NaN行（市場休日・取得不完全）を除外
                if df.empty or len(df) < 252:
                    logger.warning("%s データ不足（%d日）→ スキップ", sym, len(df))
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
        """
        現在の保有ポジションを取得する。

        ライブモード: kabuステーション API から取得
        ドライランモード: 空のdictを返す（ポジションなしとして扱う）

        Returns:
            {symbol_with_T: {"qty": int, "avg_price": float}}
        """
        if self._client is None:
            logger.info("API未接続: ポジションは空として扱います")
            return {}

        raw_positions = self._client.get_positions()
        positions = {}
        for p in raw_positions:
            sym_code = p.get("Symbol", "")
            # kabustation は ".T" なしの4桁コードを返す → ".T" を付加
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
        """
        現物買付余力を取得する。
        ドライランモードでは self.capital から推定する。
        """
        if self._client is None:
            # API未接続: 資本 ÷ max_positions × 空きスロット数 で推定
            n_held = len(current_positions)
            n_free = max(0, self.max_positions - n_held)
            return self.capital / self.max_positions * n_free

        wallet = self._client.get_wallet_cash()
        return float(wallet.get("StockAccountWallet", self.capital))

    # ------------------------------------------------------------------ #
    # シグナル生成
    # ------------------------------------------------------------------ #
    def _generate_all_signals(
        self,
        universe_raw:      dict,
        bench_prices:      pd.Series,
        current_positions: dict,
    ) -> list[StockSignal]:
        """
        全銘柄のシグナルを生成する。
        セクターごとに最適戦略（フジコ法 / 平均回帰 / 動的）を使い分ける。
        """
        from backtest.rsr                     import calc_universe_rsr, calc_rsr_momentum, calc_sepa
        from backtest.fujiko_strategy         import FujikoStrategy
        from backtest.mean_reversion_strategy import MeanReversionStrategy

        # ユニバース内 RSR を計算（フジコ法用）
        universe_prices = {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
        rsr_universe    = calc_universe_rsr(universe_prices)

        signals = []
        for sym, info in universe_raw.items():
            df     = info["df"]
            sector = info["sector"]
            rsr    = rsr_universe[sym] if sym in rsr_universe.columns else None

            # フジコ法シグナル
            fujiko_strat  = FujikoStrategy(
                rsr_series       = rsr,
                benchmark_prices = bench_prices,
                **self.fujiko_params,
            )
            f_signal = fujiko_strat.generate_signal(df)

            # 平均回帰シグナル
            mr_strat = MeanReversionStrategy(**MR_PARAMS)
            m_signal = mr_strat.generate_signal(df)

            # RSR・SEPA 詳細（フジコ法のデバッグ情報として常に収集）
            rsr_now  = float(rsr.iloc[-1])  if rsr is not None and not rsr.empty  else 0.0
            mom      = calc_rsr_momentum(rsr, self.fujiko_params.get("mom_period", 21))
            mom_now  = float(mom.iloc[-1])  if rsr is not None and not mom.empty  else 0.0
            sepa_df  = calc_sepa(df, rsr if rsr is not None else pd.Series(50.0, index=df.index))
            sepa_now = int(sepa_df["sepa_score"].iloc[-1])

            # セクター別戦略ルールで採用シグナルを決定
            rule = SECTOR_STRATEGY.get(sector, "dynamic")

            if rule == "fujiko":
                signal_int    = f_signal
                strategy_type = "fujiko"
            elif rule == "mean_rev":
                signal_int    = m_signal
                strategy_type = "mean_rev"
            else:  # dynamic: BUY があれば優先（フジコ法→平均回帰の順）
                if f_signal == 1:
                    signal_int    = f_signal
                    strategy_type = "fujiko"
                elif m_signal == 1:
                    signal_int    = m_signal
                    strategy_type = "mean_rev"
                elif f_signal == -1 or m_signal == -1:
                    signal_int    = -1
                    strategy_type = "fujiko" if f_signal == -1 else "mean_rev"
                else:
                    signal_int    = 0
                    strategy_type = "fujiko"  # デフォルト表記

            # シグナル理由文字列
            strat_label = "フジコ法" if strategy_type == "fujiko" else "平均回帰"
            if signal_int == 1:
                if strategy_type == "fujiko":
                    reason = f"BUY[{strat_label}]: SEPA={sepa_now}, RSR={rsr_now:.1f}, mom={mom_now:+.1f}"
                else:
                    reason = f"BUY[{strat_label}]: RSI売られ過ぎ+MA200フィルタ通過"
            elif signal_int == -1:
                if strategy_type == "fujiko":
                    reason = f"SELL[{strat_label}]: RSR={rsr_now:.1f}, mom={mom_now:+.1f}"
                else:
                    reason = f"SELL[{strat_label}]: RSI回復 or 損切り or 時間切れ"
            else:
                reason = f"HOLD: SEPA={sepa_now}, RSR={rsr_now:.1f}, mom={mom_now:+.1f} ({strat_label})"

            signals.append(StockSignal(
                symbol            = sym,
                sector            = sector,
                signal            = signal_int,
                rsr               = rsr_now,
                sepa_score        = sepa_now,
                rsr_mom           = mom_now,
                currently_holding = sym in current_positions,
                reason            = reason,
                strategy_type     = strategy_type,
            ))

        return signals

    # ------------------------------------------------------------------ #
    # 注文生成
    # ------------------------------------------------------------------ #
    def _build_orders(
        self,
        signals:           list[StockSignal],
        universe_raw:      dict,
        current_positions: dict,
        available_cash:    float,
    ) -> tuple[list[OrderInstruction], list[str]]:
        """
        シグナルリストからポートフォリオルールを適用して注文を生成する。
        """
        orders:   list[OrderInstruction] = []
        warnings: list[str]              = []

        # --- 1. 売り注文（保有中 かつ -1 シグナル） ---
        for sig in signals:
            if sig.signal == -1 and sig.currently_holding:
                pos       = current_positions[sig.symbol]
                qty       = pos["qty"]
                ref_price = float(universe_raw[sig.symbol]["df"]["Close"].iloc[-1])
                orders.append(OrderInstruction(
                    symbol          = sig.symbol,
                    symbol_4digit   = sig.symbol.replace(".T", ""),
                    sector          = sig.sector,
                    side            = "SELL",
                    qty             = qty,
                    order_type      = "MARKET_OPEN",
                    estimated_price = ref_price,
                    estimated_amount= qty * ref_price,
                    reason          = sig.reason,
                ))

        # 売り後に回収される資金を加算（ドライランでの推定）
        sell_proceeds = sum(o.estimated_amount for o in orders if o.side == "SELL")
        total_cash    = available_cash + sell_proceeds

        # --- 2. 買い注文（+1 シグナル / 未保有 / ポートフォリオ条件通過） ---
        n_held_after_sells = len(current_positions) - len(
            [o for o in orders if o.side == "SELL"]
        )

        # RSR順に並べて優先度付け
        buy_candidates = sorted(
            [s for s in signals if s.signal == 1 and not s.currently_holding],
            key=lambda s: s.rsr,
            reverse=True,
        )

        # セクター保有数を追跡
        sector_count: dict[str, int] = {}
        for sym, pos in current_positions.items():
            if sym not in [o.symbol for o in orders if o.side == "SELL"]:
                sector = self.universe_tickers.get(sym, "不明")
                sector_count[sector] = sector_count.get(sector, 0) + 1

        max_per_sector  = max(1, self.max_positions // max(1, self.min_sectors))
        max_alloc_cap   = self.capital * self.max_single_weight  # 1銘柄の上限額

        for i, sig in enumerate(buy_candidates):
            open_slots = self.max_positions - n_held_after_sells
            if open_slots <= 0:
                warnings.append(
                    f"最大ポジション数({self.max_positions})に達したため "
                    f"{sig.symbol} の BUY をスキップ"
                )
                break

            # セクター分散チェック
            if sector_count.get(sig.sector, 0) >= max_per_sector:
                warnings.append(
                    f"セクター集中制限: {sig.sector} が上限({max_per_sector})に達しているため "
                    f"{sig.symbol} をスキップ"
                )
                continue

            # 【修正】残BUY候補数と空きスロット数の少ない方で割る
            # → BUY候補が1件しかなければ全余剰資金を1銘柄に充当できる
            n_remaining = len(buy_candidates) - i
            effective_slots = min(open_slots, n_remaining)
            alloc = total_cash / max(1, effective_slots)
            alloc = min(alloc, max_alloc_cap)  # max_single_weight 上限を適用

            ref_price = float(universe_raw[sig.symbol]["df"]["Close"].iloc[-1])
            lot_cost  = ref_price * 100         # 1単元(100株)の費用
            qty       = int(alloc // lot_cost) * 100  # 単元株
            if qty <= 0:
                warnings.append(
                    f"{sig.symbol}: 1単元100株=¥{lot_cost:,.0f} > 配分上限¥{alloc:,.0f} "
                    f"→ BUYスキップ（資本{self.capital:,.0f}円では購入不可）"
                )
                continue

            cost = qty * ref_price
            orders.append(OrderInstruction(
                symbol          = sig.symbol,
                symbol_4digit   = sig.symbol.replace(".T", ""),
                sector          = sig.sector,
                side            = "BUY",
                qty             = qty,
                order_type      = "MARKET_OPEN",
                estimated_price = ref_price,
                estimated_amount= cost,
                reason          = sig.reason,
            ))

            sector_count[sig.sector] = sector_count.get(sig.sector, 0) + 1
            n_held_after_sells += 1
            total_cash -= cost

        return orders, warnings

    # ------------------------------------------------------------------ #
    # 発注実行（live モードのみ）
    # ------------------------------------------------------------------ #
    def _send_orders(self, orders: list[OrderInstruction]) -> list[dict]:
        """
        kabuステーション API に注文を送信する。
        live=True かつ confirmed のときのみ呼ばれる。
        """
        from kabusapi.client import Side, OrderType, Exchange
        from datetime import datetime, timezone, timedelta
        JST = timezone(timedelta(hours=9))
        now = datetime.now(JST)
        market_hour = now.hour * 60 + now.minute  # 分単位
        # 09:00未満 → 寄成（前場オークション参加）
        # 09:00以降 → 成行（通常取引時間中）
        if market_hour < 9 * 60:
            order_type = OrderType.MARKET_OPEN  # 寄成
            logger.info("注文タイプ: 寄成（MARKET_OPEN）")
        else:
            order_type = OrderType.MARKET       # 成行
            logger.info("注文タイプ: 成行（MARKET）")

        results = []
        for o in orders:
            side_code  = Side.BUY  if o.side == "BUY"  else Side.SELL

            try:
                result = self._client.send_order(
                    symbol     = o.symbol_4digit,
                    exchange   = Exchange.SOR,  # SOR必須（TSE=1はsendorderで拒否）
                    side       = side_code,
                    qty        = o.qty,
                    order_type = order_type,
                )
                results.append({
                    "symbol":      o.symbol,
                    "side":        o.side,
                    "qty":         o.qty,
                    "order_id":    result.order_id,
                    "success":     result.success,
                    "result_code": result.result_code,
                })
                status = "✅ 成功" if result.success else "❌ 失敗"
                logger.info(
                    "%s %s %s %d株 → %s (OrderId: %s)",
                    status, o.side, o.symbol, o.qty, status, result.order_id,
                )
            except Exception as e:
                logger.error("%s %s 注文送信エラー: %s", o.side, o.symbol, e)
                results.append({
                    "symbol":  o.symbol,
                    "side":    o.side,
                    "qty":     o.qty,
                    "success": False,
                    "error":   str(e),
                })

        return results

    # ------------------------------------------------------------------ #
    # メイン実行
    # ------------------------------------------------------------------ #
    def run(self) -> BridgeResult:
        """
        シグナル生成〜注文生成（〜発注）を一括実行する。

        Returns:
            BridgeResult（JSON出力可能）
        """
        now = datetime.now(JST)

        # 1. データ取得
        logger.info("データ取得中...")
        universe_raw, bench_prices = self._download_data()

        # データ基準日（最後の取引日）
        sample_sym  = next(iter(universe_raw))
        data_as_of  = str(universe_raw[sample_sym]["df"].index[-1].date())

        # 2. 現在のポジション取得
        current_positions = self._get_current_positions()
        available_cash    = self._get_available_cash(current_positions)

        # 3. シグナル生成
        logger.info("シグナル生成中（%d銘柄）...", len(universe_raw))
        signals = self._generate_all_signals(universe_raw, bench_prices, current_positions)

        buy_count  = sum(1 for s in signals if s.signal ==  1)
        sell_count = sum(1 for s in signals if s.signal == -1)
        logger.info(
            "シグナル: BUY=%d / SELL=%d / HOLD=%d",
            buy_count, sell_count, len(signals) - buy_count - sell_count,
        )

        # 4. 注文生成
        orders, order_warnings = self._build_orders(
            signals, universe_raw, current_positions, available_cash
        )

        # 5. 結果オブジェクト構築
        result = BridgeResult(
            generated_at = now.strftime("%Y-%m-%dT%H:%M:%S%z"),
            mode         = "LIVE" if self.live else "DRY_RUN",
            data_as_of   = data_as_of,
            n_universe   = len(universe_raw),
            portfolio_summary = {
                "available_cash":     available_cash,
                "current_positions":  len(current_positions),
                "max_positions":      self.max_positions,
                "open_slots":         max(0, self.max_positions - len(current_positions)),
            },
            signals = [
                {
                    "symbol":           s.symbol,
                    "sector":           s.sector,
                    "signal":           s.signal,
                    "strategy_type":    s.strategy_type,
                    "rsr":              round(s.rsr, 1),
                    "sepa_score":       s.sepa_score,
                    "rsr_momentum":     round(s.rsr_mom, 2),
                    "currently_holding":s.currently_holding,
                    "reason":           s.reason,
                }
                for s in sorted(signals, key=lambda s: s.rsr, reverse=True)
            ],
            orders   = [
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

        return result, orders
