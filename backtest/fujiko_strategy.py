"""
backtest/fujiko_strategy.py
フジコ投資法 ver.4.0 — Python バックテスト実装

【戦略の構造】
  銘柄選定: SEPA 8条件（6条件以上でエース、8条件でキング）
  エントリー: RSRモメンタムがプラスかつ上昇 + タートルズS1ブレイクアウト確認
  エグジット: RSRモメンタムがマイナスかつ下降 または タートルズS1下限割れ

【先読みリーク防止（CLAUDE.md ルール1）】
  engine.py から渡される data は prices.iloc[:i+1] に切り取り済み。
  本クラスでは iloc[-1]（現在足）と過去データのみ参照する。

【参考元】
  株おじさん note「シン・フジコ投資法（ver.4.0）」
  https://note.com/kabu_ojisan/n/nd7688198c814
"""

from __future__ import annotations

import sys
import os
import warnings

import numpy as np
import pandas as pd

from backtest.strategy import BaseStrategy
from backtest.rsr import (
    calc_universe_rsr,
    calc_rsr_vs_benchmark,
    calc_rsr_momentum,
    calc_sepa,
    calc_starc_band,
)


# ------------------------------------------------------------------ #
# フジコ戦略（単一銘柄 / portfolio_engine 対応）
# ------------------------------------------------------------------ #
class FujikoStrategy(BaseStrategy):
    """
    フジコ投資法 ver.4.0 戦略クラス

    Args:
        rsr_series:      事前計算済みRSR時系列（ユニバース内ランク）
                         None の場合はベンチマーク比較を使用
        benchmark_prices: ベンチマーク価格（rsr_series=None 時に使用）
        min_sepa:        エントリー最低SEPA条件数（デフォルト6=エース以上）
        min_rsr:         エントリー最低RSR（デフォルト70）
        mom_period:      RSRモメンタム計算期間（デフォルト21日）
        turtle_entry:    タートルズS1エントリー期間（デフォルト20日）
        turtle_exit:     タートルズS1エグジット期間（デフォルト10日）
        use_turtle_entry: Trueにするとブレイクアウトもエントリー条件に加える
    """

    def __init__(
        self,
        rsr_series:       pd.Series  = None,
        benchmark_prices: pd.Series  = None,
        min_sepa:         int        = 6,
        min_rsr:          float      = 70.0,
        mom_period:       int        = 21,
        turtle_entry:     int        = 20,
        turtle_exit:      int        = 10,
        use_turtle_entry: bool       = True,
    ) -> None:
        self.rsr_series       = rsr_series
        self.benchmark_prices = benchmark_prices
        self.min_sepa         = min_sepa
        self.min_rsr          = min_rsr
        self.mom_period       = mom_period
        self.turtle_entry     = turtle_entry
        self.turtle_exit      = turtle_exit
        self.use_turtle_entry = use_turtle_entry

    @property
    def name(self) -> str:
        return (
            f"Fujiko(sepa>={self.min_sepa}, rsr>={self.min_rsr:.0f}, "
            f"turtle={self.turtle_entry}/{self.turtle_exit})"
        )

    # ------------------------------------------------------------------ #
    # RSR取得（事前計算済み or リアルタイム計算）
    # ------------------------------------------------------------------ #
    def _get_rsr(self, data: pd.DataFrame) -> pd.Series:
        if self.rsr_series is not None:
            # ユニバース内RSR（事前計算済み）を使用
            aligned = self.rsr_series.reindex(data.index)
            return aligned.ffill()
        if self.benchmark_prices is not None:
            bench = self.benchmark_prices.reindex(data.index).ffill()
            return calc_rsr_vs_benchmark(data["Close"], bench)
        # フォールバック: 全て50（ベンチマーク比較なし）
        return pd.Series(50.0, index=data.index)

    # ------------------------------------------------------------------ #
    # シグナル生成
    # ------------------------------------------------------------------ #
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        フジコ投資法シグナルを生成する。

        エントリー（+1）:
          SEPA条件 >= min_sepa
          かつ RSR >= min_rsr
          かつ RSRモメンタム > 0 かつ 上昇中
          かつ（use_turtle_entry=True の場合）20日高値ブレイクアウト

        エグジット（-1）:
          RSRモメンタム < 0 かつ 下降中（フジコ法の「必ず売り」）
          または タートルズS1 10日安値割れ

        Returns:
            +1: 買い / -1: 売り / 0: 何もしない
        """
        min_bars = 252 + self.mom_period + 2
        if len(data) < min_bars:
            return 0

        close = data["Close"]

        # --- RSR + RSRモメンタム ---
        rsr = self._get_rsr(data)
        mom = calc_rsr_momentum(rsr, self.mom_period)

        rsr_now  = rsr.iloc[-1]
        mom_now  = mom.iloc[-1]
        mom_prev = mom.iloc[-2]

        if pd.isna(rsr_now) or pd.isna(mom_now) or pd.isna(mom_prev):
            return 0

        # --- SEPA 8条件 ---
        sepa = calc_sepa(data, rsr)
        sepa_score = sepa["sepa_score"].iloc[-1]

        # --- タートルズS1 ---
        turtle_high = close.iloc[-(self.turtle_entry + 1):-1].max()  # 前日まで
        turtle_low  = close.iloc[-(self.turtle_exit  + 1):-1].min()
        price_now   = close.iloc[-1]

        # ==================== エグジット判定 ====================
        # 優先度: エグジット > エントリー

        # フジコ法の「必ず売り」: RSRモメンタム負かつ下降
        if mom_now < 0 and mom_now < mom_prev:
            return -1

        # タートルズS1 エグジット: 10日安値割れ
        if price_now < turtle_low:
            return -1

        # ==================== エントリー判定 ====================
        # 1. SEPA条件数チェック（銘柄選定）
        if sepa_score < self.min_sepa:
            return 0

        # 2. RSRチェック（相対強度）
        if rsr_now < self.min_rsr:
            return 0

        # 3. RSRモメンタム: プラスかつ上昇
        if not (mom_now > 0 and mom_now > mom_prev):
            return 0

        # 4. タートルズS1 エントリー確認（オプション）
        if self.use_turtle_entry and price_now <= turtle_high:
            return 0

        return 1


# ------------------------------------------------------------------ #
# バックテスト実行スクリプト（単体実行用）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    import matplotlib
    matplotlib.use("Agg")

    import yfinance as yf
    from backtest.engine   import TradeCost
    from backtest.portfolio_engine import PortfolioEngine, PortfolioResult
    from backtest.ma_band_strategy import MABandStrategy

    # ---------------------------------------------------------------- #
    # データ取得
    # ---------------------------------------------------------------- #
    UNIVERSE_DEF = {
        "7203.T": "輸送機器",
        "6758.T": "電機",
        "9984.T": "情報通信",
        "8306.T": "銀行",
        "4502.T": "医薬品",
        "3382.T": "小売",
        "6861.T": "電機精密",   # キーエンス
        "4063.T": "化学",       # 信越化学
        "8035.T": "電機精密",   # 東京エレクトロン
        "9432.T": "情報通信",   # NTT
    }
    BENCHMARK_TICKER = "1306.T"  # TOPIXのETF（ベンチマーク代替）

    print("データ取得中...")
    universe_raw = {}
    for sym, sector in UNIVERSE_DEF.items():
        df = yf.download(sym, start="2018-01-01", end="2024-12-31", progress=False)
        if not df.empty:
            df = df.droplevel(1, axis=1)
            universe_raw[sym] = {"df": df, "sector": sector}
            print(f"  {sym} ({sector}): {len(df)} 日")

    bench_df = yf.download(BENCHMARK_TICKER, start="2018-01-01", end="2024-12-31", progress=False)
    bench_df = bench_df.droplevel(1, axis=1)
    print(f"  ベンチマーク(TOPIX ETF): {len(bench_df)} 日")

    # ---------------------------------------------------------------- #
    # RSR事前計算（ユニバース内ランク）
    # ---------------------------------------------------------------- #
    print("\nRSR計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)

    # ---------------------------------------------------------------- #
    # フジコ戦略（エース以上: SEPA >= 6）
    # ---------------------------------------------------------------- #
    print("\n--- フジコ戦略（エース: SEPA >= 6）---")
    universe_fujiko = {}
    for sym, info in universe_raw.items():
        strat = FujikoStrategy(
            rsr_series       = rsr_universe[sym] if sym in rsr_universe else None,
            benchmark_prices = bench_df["Close"],
            min_sepa         = 6,
            min_rsr          = 65.0,    # ユニバースが小さいので65に緩和
            mom_period       = 21,
            turtle_entry     = 20,
            turtle_exit      = 10,
            use_turtle_entry = True,
        )
        universe_fujiko[sym] = {"df": info["df"], "sector": info["sector"]}

    # 全銘柄に同じRSRを渡すため、PortfolioEngineをカスタム呼び出し
    # （各銘柄のRSRを個別に設定）
    results_by_symbol = {}
    for sym, info in universe_raw.items():
        strat  = FujikoStrategy(
            rsr_series       = rsr_universe[sym] if sym in rsr_universe.columns else None,
            benchmark_prices = bench_df["Close"],
            min_sepa         = 6,
            min_rsr          = 65.0,
            mom_period       = 21,
            turtle_entry     = 20,
            turtle_exit      = 10,
            use_turtle_entry = True,
        )
        # 単一銘柄バックテスト
        from backtest.engine import BacktestEngine
        eng = BacktestEngine(
            prices   = info["df"],
            strategy = strat,
            capital  = 2_000_000,
            cost     = TradeCost(),
            symbol   = sym,
        )
        results_by_symbol[sym] = eng.run()

    # ---------------------------------------------------------------- #
    # 結果表示
    # ---------------------------------------------------------------- #
    print("\n=== フジコ戦略 単一銘柄バックテスト比較 ===")
    print(f"{'銘柄':<10} {'リターン':>8} {'シャープ':>8} {'最大DD':>8} {'取引':>6} {'勝率':>6}")
    print("-" * 52)
    for sym, r in results_by_symbol.items():
        sector = universe_raw[sym]["sector"]
        print(
            f"{sym:<10} {r.total_return*100:>+7.1f}%"
            f" {r.sharpe_ratio:>8.3f}"
            f" {r.max_drawdown*100:>+7.1f}%"
            f" {r.num_trades:>5}回"
            f" {r.win_rate*100:>5.1f}%"
        )

    # ---------------------------------------------------------------- #
    # MAバンド戦略との比較グラフ
    # ---------------------------------------------------------------- #
    print("\nグラフ生成中...")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import platform

    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "MS Gothic"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 全銘柄の資産推移を重ねて表示
    ax = axes[0]
    colors = plt.cm.tab10.colors
    for i, (sym, r) in enumerate(results_by_symbol.items()):
        eq = r.equity_curve / 10_000
        ax.plot(eq.index, eq.values, linewidth=1.2, color=colors[i],
                label=f"{sym} {r.total_return*100:+.1f}%")
    ax.axhline(200, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("資産（万円）")
    ax.set_title("フジコ戦略 銘柄別バックテスト（2018-2024 / 初期資本200万円）")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # RSR推移（代表銘柄）
    ax = axes[1]
    rep_sym = "7203.T"
    rsr_s   = rsr_universe[rep_sym]
    mom_s   = calc_rsr_momentum(rsr_s)
    ax2     = ax.twinx()

    ax.plot(rsr_s.index, rsr_s.values, color="royalblue", linewidth=1.0, label="RSR")
    ax.axhline(70, color="red",   linestyle="--", linewidth=0.8, label="RSR=70")
    ax.axhline(50, color="gray",  linestyle="--", linewidth=0.6)
    ax.set_ylabel("RSR")
    ax.set_ylim(0, 100)

    mom_pos = mom_s.clip(lower=0)
    mom_neg = mom_s.clip(upper=0)
    ax2.bar(mom_s.index, mom_pos.values, color="deeppink",  alpha=0.6, width=1, label="RSRモメンタム(+)")
    ax2.bar(mom_s.index, mom_neg.values, color="steelblue", alpha=0.6, width=1, label="RSRモメンタム(-)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("RSRモメンタム")

    ax.set_title(f"RSR + RSRモメンタム  ({rep_sym})")
    ax.legend(loc="upper left",  fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("data/fujiko_backtest.png", dpi=150, bbox_inches="tight")
    print("グラフ保存: data/fujiko_backtest.png")
