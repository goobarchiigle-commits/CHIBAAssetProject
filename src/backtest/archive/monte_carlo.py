"""
backtest/monte_carlo.py
モンテカルロ・シミュレーション + Risk of Ruin 算出

【手法】
  バックテストで得た取引損益（PnL）の順番をランダムに入れ替え、
  N通りの「仮想的な資産推移」を生成する。

【破産確率（Risk of Ruin）の定義】
  全シミュレーション中で「一度でも資産が初期資本 × ruin_threshold 以下に
  なったことがある」シミュレーションの割合。
  デフォルト閾値 = 50%（200万円 → 100万円以下）

【制約・注意事項】
  - 各トレードが独立という仮定に基づく（実際は市場環境の連続性があるため
    楽観的になりがち。結果は下限目安として解釈すること）
  - トレード回数 < 20回 の場合は分布が不安定。参考値として扱う
  - RoR < 5% を実運用移行の目安とする

【先読みリーク】
  本モジュールはバックテスト結果の事後分析のみ行う。
  将来データへのアクセスは一切ない。
"""

from __future__ import annotations

import sys
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# 結果クラス
# ------------------------------------------------------------------ #
@dataclass
class MonteCarloResult:
    """モンテカルロ・シミュレーション結果"""

    n_simulations:     int
    n_trades:          int
    initial_capital:   float
    ruin_threshold:    float       # 閾値割合（例: 0.5）
    ruin_threshold_val: float      # 閾値金額（円）

    # トレード損益統計
    pnl_mean:   float
    pnl_std:    float
    pnl_median: float
    win_rate:   float

    # シミュレーション結果（shape: n_simulations × n_trades+1）
    equity_matrix:  np.ndarray
    final_equities: np.ndarray   # shape: (n_simulations,)

    # パーセンタイル資産推移（shape: n_trades+1）
    pct5:  np.ndarray
    pct25: np.ndarray
    pct50: np.ndarray
    pct75: np.ndarray
    pct95: np.ndarray

    # 各シミュレーションの「途中最低資産」（RoR分析の本質）
    min_equities: np.ndarray    # shape: (n_simulations,)

    # Risk of Ruin
    risk_of_ruin: float   # 0〜1

    @property
    def total_return(self) -> float:
        """総リターン（絶対PnLのシャッフルなので全シミュレーション共通）"""
        return (self.final_equities[0] / self.initial_capital) - 1.0

    @property
    def worst_min_equity_return(self) -> float:
        """最悪5%ile の途中最低資産リターン"""
        return (np.percentile(self.min_equities, 5) / self.initial_capital) - 1.0

    @property
    def median_min_equity_return(self) -> float:
        """中央値の途中最低資産リターン"""
        return (np.median(self.min_equities) / self.initial_capital) - 1.0

    def summary(self) -> None:
        ror_pct = self.risk_of_ruin * 100
        if ror_pct < 5:
            ror_judge = "✅ 実運用移行OK"
        elif ror_pct < 20:
            ror_judge = "⚠  要注意"
        else:
            ror_judge = "❌ 危険（実運用不可）"

        print("=" * 58)
        print("  モンテカルロ・シミュレーション結果")
        print("=" * 58)
        print(f"  シミュレーション数   : {self.n_simulations:>8,} 回")
        print(f"  取引回数             : {self.n_trades:>8} 回")
        print(f"  初期資本             : ¥{self.initial_capital:>12,.0f}")
        print(f"  破産閾値             : ¥{self.ruin_threshold_val:>12,.0f}"
              f"  （初期資本の {self.ruin_threshold*100:.0f}%）")
        print("-" * 58)
        print(f"  トレード損益 平均    : ¥{self.pnl_mean:>+12,.0f}")
        print(f"  トレード損益 中央値  : ¥{self.pnl_median:>+12,.0f}")
        print(f"  トレード損益 標準偏差: ¥{self.pnl_std:>12,.0f}")
        print(f"  勝率                 :  {self.win_rate*100:>7.1f}%")
        print("-" * 58)
        print(f"  最終資産（全共通）   : ¥{self.final_equities[0]:>12,.0f}"
              f"  ({self.total_return*100:+.1f}%)")
        print(f"  途中最低資産 中央値  : ¥{np.median(self.min_equities):>12,.0f}"
              f"  ({self.median_min_equity_return*100:+.1f}%)")
        print(f"  途中最低資産  5%ile  : ¥{np.percentile(self.min_equities, 5):>12,.0f}"
              f"  ({self.worst_min_equity_return*100:+.1f}%)")
        print("-" * 58)
        print(f"  ▶ 破産確率 (RoR)     :  {ror_pct:>7.1f}%  {ror_judge}")
        print("=" * 58)

        if self.n_trades < 20:
            print(f"\n  ⚠ 警告: 取引回数({self.n_trades}回)が少ないため分布が不安定です。")
            print("          結果は参考値として扱ってください。")


# ------------------------------------------------------------------ #
# シミュレーター本体
# ------------------------------------------------------------------ #
class MonteCarloSimulator:
    """
    モンテカルロ・シミュレーター

    バックテスト結果（BacktestResult）を受け取り、
    トレードの順番をランダムに入れ替えた資産推移を N 通り生成する。

    Args:
        result:          BacktestResult（backtest/engine.py）
        n_simulations:   シミュレーション回数（デフォルト 1,000）
        ruin_threshold:  破産とみなす資産割合（デフォルト 0.5 = 初期資本の50%）
        seed:            乱数シード（再現性確保）
    """

    def __init__(
        self,
        result,
        n_simulations:  int   = 1_000,
        ruin_threshold: float = 0.5,
        seed:           int   = 42,
    ) -> None:
        self.result         = result
        self.n_simulations  = n_simulations
        self.ruin_threshold = ruin_threshold
        self.rng            = np.random.default_rng(seed)

        # SELL取引からPnLを抽出（買い取引のPnLは0なので除外）
        # BacktestResult（Tradeオブジェクトのリスト）と
        # PortfolioResult（DataFrameのtrades）の両方を吸収する
        trades = result.trades
        if isinstance(trades, pd.DataFrame):
            # PortfolioResult: side が "SELL_*" で始まる行のpnlを使用
            sells = trades[trades["side"].str.startswith("SELL")]
            self.pnls = sells["pnl"].values.astype(float)
        else:
            # BacktestResult: Tradeオブジェクトのリスト
            self.pnls = np.array([
                t.pnl for t in trades if t.side == "SELL"
            ])

    def run(self) -> MonteCarloResult:
        """シミュレーションを実行し MonteCarloResult を返す。"""
        pnls            = self.pnls
        n_trades        = len(pnls)
        initial_capital = self.result.initial_capital
        ruin_val        = initial_capital * self.ruin_threshold

        if n_trades == 0:
            raise ValueError(
                "取引回数が0です。バックテストを先に実行してください。"
            )

        # equity_matrix: 行=シミュレーション / 列=取引ステップ(0〜n_trades)
        equity_matrix = np.empty((self.n_simulations, n_trades + 1))
        equity_matrix[:, 0] = initial_capital

        for sim in range(self.n_simulations):
            shuffled = self.rng.permutation(pnls)
            equity_matrix[sim, 1:] = initial_capital + np.cumsum(shuffled)

        # パーセンタイル資産推移
        pct5  = np.percentile(equity_matrix,  5, axis=0)
        pct25 = np.percentile(equity_matrix, 25, axis=0)
        pct50 = np.percentile(equity_matrix, 50, axis=0)
        pct75 = np.percentile(equity_matrix, 75, axis=0)
        pct95 = np.percentile(equity_matrix, 95, axis=0)

        # 各シミュレーションの「途中最低資産」（初期資本から見た最大ドローダウン点）
        min_equities = equity_matrix.min(axis=1)

        # Risk of Ruin: 一度でも破産閾値以下になったシミュレーションの割合
        ruined       = (min_equities <= ruin_val)
        risk_of_ruin = float(ruined.mean())

        return MonteCarloResult(
            n_simulations      = self.n_simulations,
            n_trades           = n_trades,
            initial_capital    = initial_capital,
            ruin_threshold     = self.ruin_threshold,
            ruin_threshold_val = ruin_val,
            pnl_mean           = float(pnls.mean()),
            pnl_std            = float(pnls.std()),
            pnl_median         = float(np.median(pnls)),
            win_rate           = float((pnls > 0).mean()),
            equity_matrix      = equity_matrix,
            final_equities     = equity_matrix[:, -1],
            min_equities       = min_equities,
            pct5               = pct5,
            pct25              = pct25,
            pct50              = pct50,
            pct75              = pct75,
            pct95              = pct95,
            risk_of_ruin       = risk_of_ruin,
        )

    def plot(self, mc: MonteCarloResult, save_path: str = None) -> None:
        """信頼区間付き資産曲線 + 最終資産分布を描画する。"""
        import matplotlib
        import matplotlib.pyplot as plt
        import platform

        if save_path:
            matplotlib.use("Agg")
        if platform.system() == "Windows":
            plt.rcParams["font.family"] = "MS Gothic"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"モンテカルロ・シミュレーション  "
            f"{self.result.strategy_name} — {getattr(self.result, 'symbol', 'ポートフォリオ')}\n"
            f"{mc.n_simulations:,}通りの資産推移  ／  "
            f"破産確率(RoR): {mc.risk_of_ruin*100:.1f}%  "
            f"（破産閾値: 初期資本の{mc.ruin_threshold*100:.0f}%以下）",
            fontsize=11,
        )

        x        = np.arange(mc.n_trades + 1)
        cap_man  = self.result.initial_capital / 10_000
        ruin_man = mc.ruin_threshold_val / 10_000

        # ---- 左パネル: 資産推移の信頼区間 ----
        ax = axes[0]

        # 全シミュレーションを薄く重ね描き（最大 150 本）
        n_show   = min(150, mc.n_simulations)
        idx_show = self.rng.choice(mc.n_simulations, n_show, replace=False)
        for i in idx_show:
            ax.plot(
                x, mc.equity_matrix[i] / 10_000,
                color="steelblue", alpha=0.04, linewidth=0.6,
            )

        ax.fill_between(
            x, mc.pct5 / 10_000, mc.pct95 / 10_000,
            alpha=0.18, color="steelblue", label="5〜95%ile",
        )
        ax.fill_between(
            x, mc.pct25 / 10_000, mc.pct75 / 10_000,
            alpha=0.38, color="steelblue", label="25〜75%ile",
        )
        ax.plot(x, mc.pct50 / 10_000, color="navy", linewidth=2.0, label="中央値")
        ax.axhline(cap_man,  color="gray",   linestyle="--", linewidth=0.8, label="初期資本")
        ax.axhline(ruin_man, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"破産閾値 ({mc.ruin_threshold*100:.0f}%)")

        ax.set_xlabel("取引回数")
        ax.set_ylabel("資産（万円）")
        ax.set_title("資産推移の信頼区間")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"¥{v:.0f}万"))

        # ---- 右パネル: 途中最低資産の分布（Risk of Ruin の本体） ----
        # 「各シミュレーションで最も資産が低かった瞬間」の分布
        # → 最終資産はシャッフル順に依存しないが、最低点は順序依存
        ax = axes[1]
        mins_man = mc.min_equities / 10_000
        med_min  = float(np.median(mins_man))
        p5_min   = float(np.percentile(mins_man, 5))

        n_bins = min(40, max(10, len(np.unique(mins_man))))
        ax.hist(mins_man, bins=n_bins, color="tomato", alpha=0.7, edgecolor="white")
        ax.axvline(cap_man,  color="gray",    linestyle="--", linewidth=0.8, label="初期資本")
        ax.axvline(ruin_man, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"破産閾値 (RoR={mc.risk_of_ruin*100:.1f}%)")
        ax.axvline(med_min,  color="navy",    linewidth=2.0,
                   label=f"中央値 ¥{med_min:.0f}万")
        ax.axvline(p5_min,   color="orange",  linewidth=1.5, linestyle=":",
                   label=f"最悪5%ile ¥{p5_min:.0f}万")

        ax.set_xlabel("途中最低資産（万円）")
        ax.set_ylabel("シミュレーション数")
        ax.set_title("途中最低資産の分布（最悪ケース分析）")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"¥{v:.0f}万"))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"グラフ保存: {save_path}")
        else:
            plt.show()


# ------------------------------------------------------------------ #
# 単体実行スクリプト
#   TOPIX100ポートフォリオ（502回取引）のモンテカルロ検証
#   前回の8035.T単銘柄（16回）より統計的に遥かに信頼性が高い
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    import matplotlib
    matplotlib.use("Agg")

    from backtest.topix100_backtest import (
        main as run_topix100,
        TOPIX100_TICKERS, START, END, BENCHMARK, CAPITAL,
        FUJIKO_PARAMS, PORT_PARAMS,
    )
    from backtest.universe_builder  import download_universe
    from backtest.rsr               import calc_universe_rsr
    from backtest.fujiko_strategy   import FujikoStrategy
    from backtest.portfolio_engine  import PortfolioEngine
    from backtest.engine            import TradeCost
    import yfinance as yf

    print("=" * 60)
    print("  TOPIX100ポートフォリオ × モンテカルロ・シミュレーション")
    print("=" * 60)

    # ---- ポートフォリオバックテスト実行 ----
    print("\n[1/3] ポートフォリオバックテスト実行中...")
    universe_raw = download_universe(TOPIX100_TICKERS, start=START, end=END, verbose=False)
    print(f"  取得成功: {len(universe_raw)} 銘柄")

    df_bench = yf.download(BENCHMARK, start=START, end=END, progress=False)
    df_bench = df_bench.droplevel(1, axis=1)

    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)

    strategies = {
        sym: FujikoStrategy(
            rsr_series       = rsr_universe[sym] if sym in rsr_universe.columns else None,
            benchmark_prices = df_bench["Close"],
            **FUJIKO_PARAMS,
        )
        for sym in universe_raw
    }

    engine = PortfolioEngine(universe=universe_raw, strategy=strategies, **PORT_PARAMS)
    port_result = engine.run()
    port_result.summary()

    # ---- モンテカルロ ----
    print("\n[2/3] モンテカルロ・シミュレーション実行中（1,000回）...")
    sim = MonteCarloSimulator(
        result         = port_result,
        n_simulations  = 1_000,
        ruin_threshold = 0.5,   # 200万円 → 100万円以下を「破産」
        seed           = 42,
    )
    mc = sim.run()
    mc.summary()

    # ---- グラフ保存 ----
    print("\n[3/3] グラフ保存中...")
    os.makedirs("data", exist_ok=True)
    sim.plot(mc, save_path="data/monte_carlo_topix100.png")
