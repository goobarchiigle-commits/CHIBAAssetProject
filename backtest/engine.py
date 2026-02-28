"""
backtest/engine.py
ウォークフォワード対応バックテストエンジン

設計原則（CLAUDE.md 準拠）:
  - 先読みリーク禁止: 各時点では過去データのみ参照
  - データリーク防止: 学習/テスト期間を時系列順に厳密分離
  - スリッページ・手数料を必ずモデルに含める
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.strategy import BaseStrategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 取引コスト設定
# ------------------------------------------------------------------ #
@dataclass
class TradeCost:
    """
    取引コスト設定（auカブコム証券 現物取引 概算）

    実際の手数料は以下を参照:
      https://kabu.com/item/lp/oneclick_lp/commission_c.html
    """
    # 手数料率（往復分）
    commission_rate: float = 0.00055   # 0.055%（ワンクリック株）
    # 最低手数料（片道）
    min_commission:  float = 99.0      # 99円（税込）
    # スリッページ（約定価格のずれ: 0.1% 想定）
    slippage_rate:   float = 0.001


# ------------------------------------------------------------------ #
# 取引記録
# ------------------------------------------------------------------ #
@dataclass
class Trade:
    date:       pd.Timestamp
    symbol:     str
    side:       str             # "BUY" / "SELL"
    qty:        int
    price:      float
    cost:       float           # 手数料 + スリッページ
    pnl:        float = 0.0    # 決済損益（SELLのみ計算）


# ------------------------------------------------------------------ #
# バックテストエンジン
# ------------------------------------------------------------------ #
class BacktestEngine:
    """
    単一銘柄・日足バックテストエンジン

    使い方:
        from backtest.engine import BacktestEngine, TradeCost
        from backtest.sample_ma import GoldenCrossStrategy

        engine = BacktestEngine(
            prices    = df_prices,          # OHLCV DataFrame（DatetimeIndex）
            strategy  = GoldenCrossStrategy(short=25, long=75),
            capital   = 1_000_000,
            cost      = TradeCost(),
        )
        result = engine.run()
        result.summary()
        result.plot()
    """

    def __init__(
        self,
        prices:   pd.DataFrame,
        strategy: "BaseStrategy",
        capital:  float    = 1_000_000,
        cost:     TradeCost = None,
        symbol:   str      = "N/A",
    ) -> None:
        self.prices   = prices.sort_index()
        self.strategy = strategy
        self.capital  = capital
        self.cost     = cost or TradeCost()
        self.symbol   = symbol

    # ------------------------------------------------------------------ #
    # 実行
    # ------------------------------------------------------------------ #
    def run(self) -> "BacktestResult":
        """
        バックテストを実行する。

        先読みリーク防止:
          各ステップ i では prices.iloc[:i+1] のみを strategy に渡す。
          シグナルが返っても実際の約定は次足の始値で行う（execution_price）。
        """
        prices  = self.prices
        cash    = self.capital
        holding = 0        # 保有株数
        avg_cost = 0.0     # 平均取得単価

        equity_curve: list[float] = []
        trades:       list[Trade] = []

        for i in range(len(prices)):
            row = prices.iloc[i]

            # --- 先読みリーク防止: i+1 以降のデータは渡さない ---
            past_data = prices.iloc[: i + 1]
            signal = self.strategy.generate_signal(past_data)
            # signal: +1 = 買い / -1 = 売り / 0 = 何もしない

            # 約定は「次の足の始値」を使用
            if i + 1 < len(prices):
                exec_price_raw = prices.iloc[i + 1]["Open"]
            else:
                exec_price_raw = row["Close"]

            # スリッページ適用
            if signal == 1:
                exec_price = exec_price_raw * (1 + self.cost.slippage_rate)
            elif signal == -1:
                exec_price = exec_price_raw * (1 - self.cost.slippage_rate)
            else:
                exec_price = exec_price_raw

            # --- 買い ---
            if signal == 1 and holding == 0 and cash > 0:
                qty = int(cash // (exec_price * 100)) * 100  # 単元株（100株）
                if qty > 0:
                    trade_value = qty * exec_price
                    commission  = max(
                        trade_value * self.cost.commission_rate,
                        self.cost.min_commission,
                    )
                    total_cost = trade_value + commission
                    if total_cost <= cash:
                        cash     -= total_cost
                        holding   = qty
                        avg_cost  = exec_price
                        trades.append(Trade(
                            date   = prices.index[i + 1] if i + 1 < len(prices) else prices.index[i],
                            symbol = self.symbol,
                            side   = "BUY",
                            qty    = qty,
                            price  = exec_price,
                            cost   = commission,
                        ))

            # --- 売り ---
            elif signal == -1 and holding > 0:
                trade_value = holding * exec_price
                commission  = max(
                    trade_value * self.cost.commission_rate,
                    self.cost.min_commission,
                )
                pnl  = (exec_price - avg_cost) * holding - commission
                cash += trade_value - commission
                trades.append(Trade(
                    date   = prices.index[i + 1] if i + 1 < len(prices) else prices.index[i],
                    symbol = self.symbol,
                    side   = "SELL",
                    qty    = holding,
                    price  = exec_price,
                    cost   = commission,
                    pnl    = pnl,
                ))
                holding  = 0
                avg_cost = 0.0

            # 資産評価額（現金 + 時価）
            market_value = holding * row["Close"]
            equity_curve.append(cash + market_value)

        return BacktestResult(
            equity_curve = pd.Series(equity_curve, index=prices.index),
            trades       = trades,
            initial_capital = self.capital,
            strategy_name   = self.strategy.name,
            symbol          = self.symbol,
        )


# ------------------------------------------------------------------ #
# ウォークフォワード検証
# ------------------------------------------------------------------ #
class WalkForwardValidator:
    """
    ウォークフォワード検証（CLAUDE.md ルール1・2 準拠）

    - 学習期間と検証期間を時系列順に分割
    - 検証データが学習プロセスに混入しない設計

    使い方:
        wfv = WalkForwardValidator(
            prices       = df,
            strategy_cls = GoldenCrossStrategy,
            train_bars   = 252,    # 学習期間（約1年）
            test_bars    = 63,     # 検証期間（約3ヶ月）
            step_bars    = 63,     # 次のウィンドウへのステップ
        )
        results = wfv.run(param_grid={"short": [10, 25], "long": [50, 75, 100]})
        wfv.summary(results)
    """

    def __init__(
        self,
        prices:       pd.DataFrame,
        strategy_cls: type,
        train_bars:   int = 252,
        test_bars:    int = 63,
        step_bars:    int = 63,
        capital:      float = 1_000_000,
        cost:         TradeCost = None,
        symbol:       str = "N/A",
    ) -> None:
        self.prices       = prices.sort_index()
        self.strategy_cls = strategy_cls
        self.train_bars   = train_bars
        self.test_bars    = test_bars
        self.step_bars    = step_bars
        self.capital      = capital
        self.cost         = cost or TradeCost()
        self.symbol       = symbol

    def _best_params(self, train_data: pd.DataFrame, param_grid: dict) -> dict:
        """学習データのみでパラメータ最適化する（テストデータは一切使わない）。"""
        from itertools import product

        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        best_params = {}
        best_sharpe = -np.inf

        for combo in product(*values):
            params   = dict(zip(keys, combo))
            strategy = self.strategy_cls(**params)
            engine   = BacktestEngine(
                prices  = train_data,
                strategy= strategy,
                capital = self.capital,
                cost    = self.cost,
                symbol  = self.symbol,
            )
            result = engine.run()
            sharpe = result.sharpe_ratio
            if sharpe > best_sharpe:
                best_sharpe  = sharpe
                best_params  = params

        return best_params

    def run(self, param_grid: dict) -> list["BacktestResult"]:
        """ウォークフォワード検証を実行し、各検証期間の結果リストを返す。"""
        prices  = self.prices
        results = []
        start   = 0

        while start + self.train_bars + self.test_bars <= len(prices):
            train_end  = start + self.train_bars
            test_end   = train_end + self.test_bars

            train_data = prices.iloc[start:train_end]   # 学習期間
            test_data  = prices.iloc[train_end:test_end]  # 検証期間

            # パラメータ最適化（学習データのみで実施）
            best_params = self._best_params(train_data, param_grid)
            logger.info(
                "WFV window [%s ~ %s] 最適パラメータ: %s",
                train_data.index[0].date(),
                test_data.index[-1].date(),
                best_params,
            )

            # 検証期間で評価
            strategy = self.strategy_cls(**best_params)
            engine   = BacktestEngine(
                prices  = test_data,
                strategy= strategy,
                capital = self.capital,
                cost    = self.cost,
                symbol  = self.symbol,
            )
            result = engine.run()
            result.meta = {
                "train_start": train_data.index[0],
                "train_end":   train_data.index[-1],
                "test_start":  test_data.index[0],
                "test_end":    test_data.index[-1],
                "best_params": best_params,
            }
            results.append(result)
            start += self.step_bars

        return results

    def summary(self, results: list["BacktestResult"]) -> pd.DataFrame:
        """ウォークフォワード検証の結果サマリーを表示する。"""
        rows = []
        for r in results:
            m = r.meta
            rows.append({
                "検証開始":       m["test_start"].date(),
                "検証終了":       m["test_end"].date(),
                "最適パラメータ": str(m["best_params"]),
                "総リターン(%)":  f"{r.total_return * 100:.2f}",
                "シャープ比":     f"{r.sharpe_ratio:.3f}",
                "最大DD(%)":      f"{r.max_drawdown * 100:.2f}",
                "取引回数":       r.num_trades,
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df


# ------------------------------------------------------------------ #
# バックテスト結果
# ------------------------------------------------------------------ #
@dataclass
class BacktestResult:
    equity_curve:    pd.Series
    trades:          list[Trade]
    initial_capital: float
    strategy_name:   str
    symbol:          str
    meta:            dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.initial_capital) - 1.0

    @property
    def sharpe_ratio(self) -> float:
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        peak = self.equity_curve.cummax()
        dd   = (self.equity_curve - peak) / peak
        return float(dd.min())

    @property
    def num_trades(self) -> int:
        return sum(1 for t in self.trades if t.side == "SELL")

    @property
    def win_rate(self) -> float:
        sell_trades = [t for t in self.trades if t.side == "SELL"]
        if not sell_trades:
            return 0.0
        wins = sum(1 for t in sell_trades if t.pnl > 0)
        return wins / len(sell_trades)

    def summary(self) -> None:
        print("=" * 50)
        print(f"戦略         : {self.strategy_name}")
        print(f"銘柄         : {self.symbol}")
        print(f"期間         : {self.equity_curve.index[0].date()} "
              f"〜 {self.equity_curve.index[-1].date()}")
        print(f"初期資本     : ¥{self.initial_capital:>12,.0f}")
        print(f"最終資産     : ¥{self.equity_curve.iloc[-1]:>12,.0f}")
        print(f"総リターン   : {self.total_return * 100:>+.2f}%")
        print(f"シャープ比   : {self.sharpe_ratio:>8.3f}")
        print(f"最大DD       : {self.max_drawdown * 100:>+.2f}%")
        print(f"取引回数     : {self.num_trades:>8} 回")
        print(f"勝率         : {self.win_rate * 100:>8.1f}%")
        print("=" * 50)

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        import platform
        if platform.system() == "Windows":
            plt.rcParams["font.family"] = "MS Gothic"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 資産推移
        ax1.plot(self.equity_curve.index, self.equity_curve.values,
                 color="royalblue", linewidth=1.5)
        ax1.axhline(self.initial_capital, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("資産額 (¥)")
        ax1.set_title(f"{self.strategy_name} — {self.symbol} バックテスト結果")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))

        # ドローダウン
        peak = self.equity_curve.cummax()
        dd   = (self.equity_curve - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color="tomato", alpha=0.5)
        ax2.set_ylabel("ドローダウン (%)")
        ax2.set_xlabel("日付")

        plt.tight_layout()
        plt.show()
