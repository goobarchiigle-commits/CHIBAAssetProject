"""
backtest/portfolio_engine.py
ポートフォリオバックテストエンジン

設計方針（CLAUDE.md 準拠）:
  - 先読みリーク防止: シグナルは当日終値で判定、約定は翌日始値
  - 損切り: バンド下端を終値で割ったら機械的執行（裁量なし）
  - セクター分散: 最低3セクター以上を維持
  - DD制限: ポートフォリオ全体が15%を超えたらサーキットブレーカー発動
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.engine import TradeCost

if TYPE_CHECKING:
    from backtest.strategy import BaseStrategy

logger = logging.getLogger(__name__)

# Carver「Systematic Trading」Table 4 — IDM（分散乗数）ルックアップ
_IDM_TABLE: dict[int, float] = {
    1: 1.00, 2: 1.20, 3: 1.48, 4: 1.56,
    5: 1.70, 6: 1.90, 7: 2.10, 8: 2.20,
}


# ------------------------------------------------------------------ #
# 保有ポジション
# ------------------------------------------------------------------ #
@dataclass
class Position:
    symbol:      str
    sector:      str
    qty:         int
    entry_price: float
    entry_date:  pd.Timestamp

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.qty


# ------------------------------------------------------------------ #
# ユニバース定義ヘルパー
# ------------------------------------------------------------------ #
DEFAULT_UNIVERSE = {
    "7203.T": "輸送機器",   # トヨタ自動車
    "6758.T": "電機",       # ソニーグループ
    "9984.T": "情報通信",   # ソフトバンクグループ
    "8306.T": "銀行",       # 三菱UFJ
    "4502.T": "医薬品",     # 武田薬品工業
    "3382.T": "小売",       # セブン&アイ
}


# ------------------------------------------------------------------ #
# ポートフォリオエンジン本体
# ------------------------------------------------------------------ #
class PortfolioEngine:
    """
    複数銘柄・セクター分散対応バックテストエンジン

    Args:
        universe:      {symbol: {'df': OHLCV DataFrame, 'sector': str}}
        strategy:      BaseStrategy のインスタンス（全銘柄共通）
                       または {symbol: BaseStrategy} の辞書（銘柄ごとに個別設定）
                       ※ユニバース拡張時は辞書形式を使うこと（RSR系列が銘柄依存のため）
        capital:       初期資本（円）
        max_dd_limit:  ポートフォリオ最大DD閾値（デフォルト 0.15 = 15%）
        min_sectors:   最低保有セクター数（デフォルト 3）
        max_positions: 最大保有銘柄数（デフォルト 6）
        cost:          取引コスト設定
    """

    def __init__(
        self,
        universe:          dict,
        strategy,                      # BaseStrategy or dict[str, BaseStrategy]
        capital:           float     = 2_000_000,
        max_dd_limit:      float     = 0.15,
        min_sectors:       int       = 3,
        max_positions:     int       = 6,
        cost:              TradeCost = None,
        vol_target:        float     = 0.0,    # >0 でボラティリティ調整サイジング有効
                                               # 例: 0.02 = ポートフォリオ日次変動2%目標
        vol_window:        int       = 20,     # ボラティリティ計算ウィンドウ（営業日）
        max_single_weight: float     = 0.25,   # 1銘柄の最大ウェイト（資本比）
        benchmark:         pd.Series = None,   # 絶対モメンタムフィルター用ベンチマーク価格系列
                                               # 例: TOPIXの終値Series
        abs_mom_window:    int       = 252,    # 絶対モメンタム判定期間（営業日）デフォルト12ヶ月
        ma_filter_window:  int       = 200,    # 移動平均フィルター期間（営業日）
        use_ma_filter:     bool      = False,  # True: 200MA基準 / False: 12ヶ月リターン基準
        clenow_scores:     pd.DataFrame = None, # Clenowスコア（銘柄を列、日付をインデックス）
                                               # 設定時: エントリー候補をスコア降順でソート
        fip_scores:        pd.DataFrame = None, # FIP比率（銘柄を列、日付をインデックス）
                                               # 設定時: fip_min_ratio 未満の銘柄はエントリー禁止
        fip_min_ratio:     float        = 0.52, # FIPフィルター最低閾値（デフォルト52%）
        use_idm:           bool         = False, # Carver IDM（分散乗数）を適用する
                                                 # True: 実保有数ベースでポジションを拡大
        use_dd_scalar:     bool         = False, # Carver DDスケーラーを適用する
                                                 # DD > max_dd_limit*0.5 でポジションを段階縮小
    ) -> None:
        self.universe          = universe
        # 単一 Strategy → 全銘柄共通の辞書に正規化
        if isinstance(strategy, dict):
            self.strategies = strategy
        else:
            self.strategies = {sym: strategy for sym in universe}
        self.capital           = capital
        self.max_dd_limit      = max_dd_limit
        self.min_sectors       = min_sectors
        self.max_positions     = max_positions
        self.cost              = cost or TradeCost()
        self.vol_target        = vol_target
        self.vol_window        = vol_window
        self.max_single_weight = max_single_weight
        self.benchmark         = benchmark
        self.abs_mom_window    = abs_mom_window
        self.ma_filter_window  = ma_filter_window
        self.use_ma_filter     = use_ma_filter
        self.clenow_scores     = clenow_scores
        self.fip_scores        = fip_scores
        self.fip_min_ratio     = fip_min_ratio
        self.use_idm           = use_idm
        self.use_dd_scalar     = use_dd_scalar

        # 全銘柄共通の取引日
        self.all_dates = self._common_dates()

    # ------------------------------------------------------------------ #
    # 内部ユーティリティ
    # ------------------------------------------------------------------ #
    def _common_dates(self) -> pd.DatetimeIndex:
        """全銘柄に共通する取引日の一覧（少ない方に合わせる）。"""
        dates = None
        for info in self.universe.values():
            idx = info["df"].index
            dates = idx if dates is None else dates.intersection(idx)
        return dates.sort_values()

    def _position_alloc(self) -> float:
        """1ポジションあたりの投資額（均等配分）。"""
        return self.capital / self.max_positions

    def _vol_adjusted_alloc(
        self,
        symbol:   str,
        past_df:  pd.DataFrame,
        n_current: int = None,
    ) -> float:
        """
        ボラティリティ調整済み配分額を計算する。

        use_idm=True の場合は Carver「Systematic Trading」IDM を適用:
          idm = _IDM_TABLE[n_current]
          target_risk_per_slot = vol_target * idm / n_current
          （実保有数ベースで割るため、1銘柄保有時に大きなポジションを取れる）

        use_idm=False（従来）:
          target_risk_per_slot = vol_target / max_positions

        Returns:
            配分額（円）
        """
        if len(past_df) < self.vol_window + 1:
            return self._position_alloc()

        daily_returns = past_df["Close"].pct_change().dropna()
        vol = float(daily_returns.tail(self.vol_window).std())

        if vol <= 0 or np.isnan(vol):
            return self._position_alloc()

        if self.use_idm and n_current is not None and n_current >= 1:
            idm = _IDM_TABLE.get(n_current, 1.48)
            target_risk_per_slot = self.vol_target * idm / n_current
        else:
            target_risk_per_slot = self.vol_target / self.max_positions

        alloc = target_risk_per_slot * self.capital / vol

        # 上限: 資本の max_single_weight（例: 25%）
        alloc = min(alloc, self.capital * self.max_single_weight)
        # 下限: 均等配分の半分（小さすぎるポジションを防止）
        alloc = max(alloc, self._position_alloc() * 0.5)

        return alloc

    def _dd_scalar(self, current_dd: float) -> float:
        """
        Carver「Systematic Trading」DDスケーラー。
        DD が max_dd_limit の 50% を超えた時点からポジションを段階縮小。

        DD < limit×0.5  → スケーラー = 1.0（フルサイズ）
        DD > limit       → スケーラー = 0.0（エントリー禁止）
        中間             → 線形補間
        """
        if not self.use_dd_scalar:
            return 1.0
        half  = self.max_dd_limit * 0.5
        dd_abs = abs(current_dd)
        if dd_abs < half:
            return 1.0
        if dd_abs >= self.max_dd_limit:
            return 0.0
        return 1.0 - (dd_abs - half) / half

    def _market_regime_ok(self, date: pd.Timestamp) -> bool:
        """
        絶対モメンタムフィルター（Antonacci / Clenow 理論準拠）

        benchmark が未設定の場合は常に True（フィルターなし）。

        use_ma_filter=False（デフォルト）: 12ヶ月リターン基準
          - ベンチマーク過去 abs_mom_window 日のリターン > 0 → 市場上昇中 → BUY OK
          - リターン <= 0 → 下げ相場 → 新規エントリー禁止

        use_ma_filter=True: 移動平均基準（Clenow式）
          - ベンチマーク終値 > ma_filter_window 日移動平均 → BUY OK
          - 終値 <= 移動平均 → 新規エントリー禁止

        既存ポジションは保有継続（個別の損切りルールに委ねる）。
        """
        if self.benchmark is None:
            return True

        bench = self.benchmark
        past  = bench.loc[:date]

        if self.use_ma_filter:
            if len(past) < self.ma_filter_window + 1:
                return True
            ma  = float(past.tail(self.ma_filter_window).mean())
            cur = float(past.iloc[-1])
            return cur > ma
        else:
            if len(past) < self.abs_mom_window + 1:
                return True
            ret_12m = float(past.iloc[-1] / past.iloc[-self.abs_mom_window] - 1)
            return ret_12m > 0.0

    def _sector_ok(self, sector: str, positions: dict[str, Position]) -> bool:
        """
        セクター分散ルールのチェック。

        ルール:
          - max_positions // min_sectors = 1セクターあたりの上限
          - 例: max=6, min_sectors=3 → 1セクター最大2銘柄
        """
        if len(positions) >= self.max_positions:
            return False
        max_per_sector = max(1, self.max_positions // self.min_sectors)
        count = sum(1 for p in positions.values() if p.sector == sector)
        return count < max_per_sector

    def _close_position(
        self,
        symbol:    str,
        pos:       Position,
        price:     float,
        reason:    str,
        date:      pd.Timestamp,
        cash:      float,
        trades:    list,
    ) -> float:
        """ポジションを決済してキャッシュを返す。"""
        exec_price   = price * (1 - self.cost.slippage_rate)
        trade_value  = pos.qty * exec_price
        commission   = max(
            trade_value * self.cost.commission_rate,
            self.cost.min_commission,
        )
        pnl  = (exec_price - pos.entry_price) * pos.qty - commission
        cash += trade_value - commission
        trades.append({
            "date":   date,
            "symbol": symbol,
            "sector": pos.sector,
            "side":   f"SELL_{reason}",
            "qty":    pos.qty,
            "price":  exec_price,
            "pnl":    pnl,
        })
        return cash

    # ------------------------------------------------------------------ #
    # バックテスト実行
    # ------------------------------------------------------------------ #
    def run(self) -> "PortfolioResult":
        """
        バックテストを実行する。

        各足の処理順序:
          1. ポートフォリオ時価評価 + DD計算
          2. DD > 15% → サーキットブレーカー（全決済）
          3. 既存ポジションの損切りチェック（バンド下端割れ）
          4. 新規エントリーチェック（セクター分散ルール適用）
          5. 資産を記録
        """
        cash       = self.capital
        positions: dict[str, Position] = {}
        equity_records: list[dict]     = []
        trades:         list[dict]     = []
        peak_equity    = float(self.capital)
        cb_active      = False   # サーキットブレーカー状態

        dates = self.all_dates

        for i, date in enumerate(dates):

            # 1. 時価評価
            mkt_value = sum(
                pos.market_value(self.universe[sym]["df"].loc[date, "Close"])
                for sym, pos in positions.items()
                if date in self.universe[sym]["df"].index
            )
            port_value  = cash + mkt_value
            peak_equity = max(peak_equity, port_value)
            current_dd  = (port_value - peak_equity) / peak_equity

            # セクター保有状況を記録
            sector_held = list({p.sector for p in positions.values()})

            # 2. サーキットブレーカー（DD > max_dd_limit）
            if current_dd < -self.max_dd_limit and not cb_active:
                logger.warning(
                    "%s DD %.1f%% → サーキットブレーカー発動（全ポジション決済）",
                    date.date(), current_dd * 100,
                )
                for sym in list(positions.keys()):
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash  = self._close_position(
                        sym, positions[sym], price, "CIRCUIT_BREAKER", date, cash, trades
                    )
                positions.clear()
                cb_active = True

            # サーキットブレーカー解除（DD が半分以下に回復したら再開）
            if cb_active and current_dd > -self.max_dd_limit * 0.5:
                cb_active = False
                logger.info("%s サーキットブレーカー解除", date.date())

            # 3. 損切りチェック（バンド下端を終値で割ったら機械的執行）
            if not cb_active:
                for sym in list(positions.keys()):
                    df_sym   = self.universe[sym]["df"]
                    past     = df_sym.loc[:date]
                    signal   = self.strategies[sym].generate_signal(past)

                    if signal == -1:   # バンド下端割れ → 機械的損切り
                        price = df_sym.loc[date, "Close"]
                        cash  = self._close_position(
                            sym, positions[sym], price, "STOP", date, cash, trades
                        )
                        del positions[sym]
                        logger.debug("%s %s バンド下端割れ損切り", date.date(), sym)

            # 4. エントリーチェック
            if not cb_active and i + 1 < len(dates):
                next_date = dates[i + 1]

                # 絶対モメンタムフィルター: 市場が下降トレンドなら新規BUY禁止
                regime_ok = self._market_regime_ok(date)

                # Clenowスコアが設定されている場合はスコア降順でソート
                if self.clenow_scores is not None:
                    def _get_score(sym: str) -> float:
                        if sym not in self.clenow_scores.columns:
                            return 0.0
                        past_score = self.clenow_scores[sym].loc[:date]
                        return float(past_score.iloc[-1]) if len(past_score) > 0 else 0.0
                    candidates = sorted(
                        self.universe.items(),
                        key=lambda kv: _get_score(kv[0]),
                        reverse=True,
                    )
                else:
                    candidates = list(self.universe.items())

                for sym, info in candidates:
                    if sym in positions:
                        continue

                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue

                    # 絶対モメンタムフィルター不通過 → 新規エントリーをスキップ
                    if not regime_ok:
                        continue

                    # FIPフィルター: コツコツ型上昇でない銘柄はスキップ
                    if self.fip_scores is not None and sym in self.fip_scores.columns:
                        fip_past = self.fip_scores[sym].loc[:date]
                        if len(fip_past) > 0:
                            fip_val = float(fip_past.iloc[-1])
                            if fip_val < self.fip_min_ratio:
                                continue

                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)

                    if signal != 1:
                        continue

                    sector = info["sector"]
                    if not self._sector_ok(sector, positions):
                        continue   # セクター分散ルール不通過

                    # 翌足始値で約定（先読みリーク防止）
                    exec_price  = df_sym.loc[next_date, "Open"] * (
                        1 + self.cost.slippage_rate
                    )
                    # ボラティリティ調整サイジング（vol_target > 0 の場合）
                    n_current = len(positions) + 1  # この銘柄を加えた後の総数
                    if self.vol_target > 0:
                        alloc = self._vol_adjusted_alloc(sym, past, n_current=n_current)
                    else:
                        alloc = self._position_alloc()
                    # DDスケーラー: DD深化時にポジションを縮小
                    alloc *= self._dd_scalar(current_dd)
                    qty         = int(alloc // (exec_price * 100)) * 100
                    if qty <= 0:
                        continue

                    total_cost = qty * exec_price + max(
                        qty * exec_price * self.cost.commission_rate,
                        self.cost.min_commission,
                    )
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    positions[sym] = Position(
                        symbol      = sym,
                        sector      = sector,
                        qty         = qty,
                        entry_price = exec_price,
                        entry_date  = next_date,
                    )
                    trades.append({
                        "date":   next_date,
                        "symbol": sym,
                        "sector": sector,
                        "side":   "BUY",
                        "qty":    qty,
                        "price":  exec_price,
                        "pnl":    0.0,
                    })

            # 5. 記録
            equity_records.append({
                "date":          date,
                "value":         port_value,
                "dd":            current_dd,
                "n_positions":   len(positions),
                "n_sectors":     len({p.sector for p in positions.values()}),
                "cb_active":     cb_active,
                "sectors_held":  "|".join(sorted(sector_held)),
                "regime_ok":     self._market_regime_ok(date),
            })

        rec_df = pd.DataFrame(equity_records).set_index("date")
        return PortfolioResult(
            equity_curve    = rec_df["value"],
            dd_series       = rec_df["dd"],
            n_positions     = rec_df["n_positions"],
            n_sectors       = rec_df["n_sectors"],
            cb_active       = rec_df["cb_active"],
            regime_ok       = rec_df["regime_ok"],
            trades          = pd.DataFrame(trades),
            initial_capital = self.capital,
            strategy_name   = next(iter(self.strategies.values())).name,
            max_dd_limit    = self.max_dd_limit,
            min_sectors     = self.min_sectors,
        )


# ------------------------------------------------------------------ #
# 結果クラス
# ------------------------------------------------------------------ #
@dataclass
class PortfolioResult:
    equity_curve:    pd.Series
    dd_series:       pd.Series
    n_positions:     pd.Series
    n_sectors:       pd.Series
    cb_active:       pd.Series
    regime_ok:       pd.Series
    trades:          pd.DataFrame
    initial_capital: float
    strategy_name:   str
    max_dd_limit:    float
    min_sectors:     int

    @property
    def total_return(self) -> float:
        return (self.equity_curve.iloc[-1] / self.initial_capital) - 1.0

    @property
    def cagr(self) -> float:
        """年率換算リターン（CAGR）"""
        start = self.equity_curve.index[0]
        end   = self.equity_curve.index[-1]
        years = (end - start).days / 365.25
        if years <= 0:
            return self.total_return
        return (1 + self.total_return) ** (1.0 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        r = self.equity_curve.pct_change().dropna()
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        return float(self.dd_series.min())

    @property
    def n_trades(self) -> int:
        if self.trades.empty:
            return 0
        return int((self.trades["side"].str.startswith("SELL")).sum())

    @property
    def win_rate(self) -> float:
        sells = self.trades[self.trades["side"].str.startswith("SELL")]
        if sells.empty:
            return 0.0
        return float((sells["pnl"] > 0).sum() / len(sells))

    @property
    def n_circuit_breaker(self) -> int:
        sells = self.trades[self.trades["side"] == "SELL_CIRCUIT_BREAKER"]
        return len(sells)

    def summary(self) -> None:
        width = 52
        print("=" * width)
        print(f" 戦略         : {self.strategy_name}")
        print(f" 期間         : {self.equity_curve.index[0].date()}"
              f" ~ {self.equity_curve.index[-1].date()}")
        print(f" 初期資本     : ¥{self.initial_capital:>12,.0f}")
        print(f" 最終資産     : ¥{self.equity_curve.iloc[-1]:>12,.0f}")
        print("-" * width)
        print(f" 総リターン   : {self.total_return * 100:>+8.2f}%")
        print(f" 年率(CAGR)  : {self.cagr * 100:>+8.2f}%")
        print(f" シャープ比   : {self.sharpe_ratio:>10.3f}")
        print(f" 最大DD       : {self.max_drawdown * 100:>+8.2f}%"
              f"  (制限: {self.max_dd_limit * 100:.0f}%)")
        print(f" DD制限クリア : {'YES ✓' if abs(self.max_drawdown) <= self.max_dd_limit else 'NO  ✗'}")
        print("-" * width)
        print(f" 決済回数     : {self.n_trades:>10} 回")
        print(f" 勝率         : {self.win_rate * 100:>9.1f}%")
        print(f" CB発動回数   : {self.n_circuit_breaker:>10} 回")
        print(f" 平均保有銘柄 : {self.n_positions.mean():>9.1f} 銘柄")
        print(f" 平均保有セクター: {self.n_sectors.mean():>6.1f} / {self.min_sectors} 以上")
        regime_pct = self.regime_ok.mean() * 100
        print(f" 市場上昇期間 : {regime_pct:>8.1f}% （絶対モメンタムOK日）")
        print(f" Calmar比     : {self.cagr / abs(self.max_drawdown) if self.max_drawdown != 0 else float('inf'):>9.3f}")
        print("=" * width)

    def plot(self, save_path: str = None) -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import platform

        if save_path:
            matplotlib.use("Agg")
        if platform.system() == "Windows":
            plt.rcParams["font.family"] = "MS Gothic"

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # --- 資産推移 ---
        ax = axes[0]
        ax.plot(
            self.equity_curve.index,
            self.equity_curve.values / 10_000,
            color="royalblue", linewidth=1.5, label=f"資産 ({self.total_return*100:+.1f}%)",
        )
        ax.axhline(
            self.initial_capital / 10_000,
            color="gray", linestyle="--", linewidth=0.8,
        )
        # CB発動日をマーク
        cb_days = self.equity_curve[self.cb_active].index
        if len(cb_days):
            ax.scatter(
                cb_days,
                self.equity_curve[cb_days].values / 10_000,
                color="red", s=40, zorder=5, label="CB発動",
            )
        ax.set_ylabel("資産（万円）")
        ax.set_title(
            f"ポートフォリオバックテスト  {self.strategy_name}\n"
            f"シャープ:{self.sharpe_ratio:.3f}  最大DD:{self.max_drawdown*100:.1f}%"
            f"  勝率:{self.win_rate*100:.0f}%  CB:{self.n_circuit_breaker}回"
        )
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万")
        )

        # --- ドローダウン ---
        ax = axes[1]
        ax.fill_between(
            self.dd_series.index,
            self.dd_series.values * 100,
            0,
            color="tomato", alpha=0.55, label="DD",
        )
        ax.axhline(
            -self.max_dd_limit * 100,
            color="darkred", linestyle="--", linewidth=1.2,
            label=f"DD上限 -{self.max_dd_limit*100:.0f}%",
        )
        ax.set_ylabel("ドローダウン（%）")
        ax.legend(fontsize=9)

        # --- 保有銘柄数 & セクター数 ---
        ax = axes[2]
        ax.fill_between(
            self.n_positions.index,
            self.n_positions.values,
            alpha=0.4, color="steelblue", label="保有銘柄数",
        )
        ax.step(
            self.n_sectors.index,
            self.n_sectors.values,
            color="darkorange", linewidth=1.2, where="post",
            label="保有セクター数",
        )
        ax.axhline(
            self.min_sectors,
            color="darkorange", linestyle="--", linewidth=0.8,
            label=f"最低{self.min_sectors}セクター",
        )
        ax.set_ylabel("銘柄数 / セクター数")
        ax.set_ylim(0, 8)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
