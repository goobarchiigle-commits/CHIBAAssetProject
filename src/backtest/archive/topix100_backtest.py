"""
backtest/topix100_backtest.py
TOPIX100相当ユニバース × フジコ法 ポートフォリオバックテスト

【実行方法】
  python -m backtest.topix100_backtest

【設計】
  - ユニバース74銘柄でRSRを計算 → 10銘柄より意味のあるランキング
  - 銘柄ごとに FujikoStrategy インスタンスを作成（RSR系列は銘柄固有）
  - PortfolioEngine で最大6銘柄・3セクター以上の分散ポートフォリオを運用
  - サーキットブレーカー: ポートフォリオDD > 15% で全決済
"""

from __future__ import annotations

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import yfinance as yf
import pandas as pd

from backtest.universe_builder  import TOPIX100_TICKERS, download_universe
from backtest.rsr               import calc_universe_rsr
from backtest.fujiko_strategy   import FujikoStrategy
from backtest.portfolio_engine  import PortfolioEngine
from backtest.engine            import TradeCost

START     = "2018-01-01"
END       = "2024-12-31"
BENCHMARK = "1306.T"       # TOPIX ETF
CAPITAL   = 2_000_000

# フジコ法パラメータ（8035.T で検証済みの設定）
FUJIKO_PARAMS = dict(
    min_sepa         = 6,
    min_rsr          = 75.0,   # 76銘柄中 top25% ≈ 19銘柄（65は多すぎた）
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 10,
    use_turtle_entry = True,
)

# PortfolioEngine パラメータ
PORT_PARAMS = dict(
    capital       = CAPITAL,
    max_dd_limit  = 0.15,   # 15%超でサーキットブレーカー
    min_sectors   = 3,
    max_positions = 6,
    cost          = TradeCost(),
)


def main() -> None:

    # ---------------------------------------------------------------- #
    # 1. データ取得
    # ---------------------------------------------------------------- #
    print("=" * 60)
    print("  TOPIX100相当ユニバース フジコ法 ポートフォリオバックテスト")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 60)

    print("\n[1/4] ユニバースデータ取得中...")
    universe_raw = download_universe(TOPIX100_TICKERS, start=START, end=END)

    print("\nベンチマーク取得中...")
    df_bench = yf.download(BENCHMARK, start=START, end=END, progress=False)
    df_bench = df_bench.droplevel(1, axis=1)
    print(f"  ベンチマーク(TOPIX ETF): {len(df_bench)} 日")

    # ---------------------------------------------------------------- #
    # 2. ユニバース内RSR計算
    # ---------------------------------------------------------------- #
    print(f"\n[2/4] RSR計算中（{len(universe_raw)}銘柄のユニバース内ランク）...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR計算完了: {rsr_universe.shape[0]}日 × {rsr_universe.shape[1]}銘柄")

    # ---------------------------------------------------------------- #
    # 3. 銘柄ごとに FujikoStrategy インスタンスを作成
    # ---------------------------------------------------------------- #
    print("\n[3/4] 戦略インスタンス作成中...")
    strategies = {}
    for sym in universe_raw:
        rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
        strategies[sym] = FujikoStrategy(
            rsr_series       = rsr_s,
            benchmark_prices = df_bench["Close"],
            **FUJIKO_PARAMS,
        )
    print(f"  {len(strategies)}銘柄分の戦略インスタンスを作成しました")

    # ---------------------------------------------------------------- #
    # 4. ポートフォリオバックテスト実行
    # ---------------------------------------------------------------- #
    print("\n[4/4] ポートフォリオバックテスト実行中（時間がかかります）...")
    engine = PortfolioEngine(
        universe   = universe_raw,
        strategy   = strategies,     # 銘柄ごとのdict
        **PORT_PARAMS,
    )
    result = engine.run()

    # ---------------------------------------------------------------- #
    # 5. 結果表示
    # ---------------------------------------------------------------- #
    print("\n")
    result.summary()

    # ---- 取引内訳（銘柄別）----
    if not result.trades.empty:
        sells = result.trades[result.trades["side"].str.startswith("SELL")].copy()
        if not sells.empty:
            print("\n=== 銘柄別取引サマリー ===")
            grp = sells.groupby("symbol").agg(
                取引回数 = ("pnl", "count"),
                総損益   = ("pnl", "sum"),
                平均損益 = ("pnl", "mean"),
                勝率     = ("pnl", lambda x: (x > 0).mean() * 100),
            ).sort_values("総損益", ascending=False)
            grp["総損益"]   = grp["総損益"].map(lambda v: f"¥{v:+,.0f}")
            grp["平均損益"] = grp["平均損益"].map(lambda v: f"¥{v:+,.0f}")
            grp["勝率"]     = grp["勝率"].map(lambda v: f"{v:.0f}%")
            print(grp.to_string())

        print("\n=== セクター別取引回数 ===")
        sec_grp = sells.groupby("sector")["pnl"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        sec_grp.columns = ["取引回数", "総損益(円)"]
        sec_grp["総損益(円)"] = sec_grp["総損益(円)"].map(lambda v: f"¥{v:+,.0f}")
        print(sec_grp.to_string())

    # ---- 妥当性チェック ----
    print("\n=== バックテスト妥当性チェック（CLAUDE.md）===")
    sharpe_ok = result.sharpe_ratio > 0.5
    dd_ok     = abs(result.max_drawdown) < 0.20
    trade_ok  = result.n_trades >= 5
    leak_warn = result.sharpe_ratio > 3.0

    print(f"  シャープ比 {result.sharpe_ratio:.3f}  {'✅ > 0.5' if sharpe_ok else '❌ <= 0.5'}"
          f"  {'⚠ 先読みリークを強く疑う！' if leak_warn else ''}")
    print(f"  最大DD     {result.max_drawdown*100:.1f}%  {'✅ < 20%' if dd_ok else '❌ >= 20%'}")
    print(f"  取引回数   {result.n_trades}回  {'✅ >= 5回' if trade_ok else '❌ < 5回（統計的に無意味）'}")
    print(f"  Phase 1基準 {'✅ 達成' if (sharpe_ok and dd_ok) else '❌ 未達成'}")

    # ---- グラフ保存 ----
    os.makedirs("data", exist_ok=True)
    print("\nグラフ保存中...")
    result.plot(save_path="data/topix100_portfolio.png")


if __name__ == "__main__":
    main()
