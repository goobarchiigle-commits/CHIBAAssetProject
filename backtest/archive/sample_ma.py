"""
backtest/sample_ma.py
サンプル戦略: ゴールデンクロス / デッドクロス（移動平均クロス）

戦略ロジック:
  - 短期移動平均が長期移動平均を上抜け → 買い（ゴールデンクロス）
  - 短期移動平均が長期移動平均を下抜け → 売り（デッドクロス）

先読みリーク防止の確認:
  - 使用するのは data["Close"] の過去値のみ
  - shift(1) で「前足の値」と比較し、現在足の終値で未来を参照しない
"""

from __future__ import annotations

import pandas as pd

from backtest.strategy import BaseStrategy


class GoldenCrossStrategy(BaseStrategy):
    """
    シンプルな移動平均クロス戦略

    Args:
        short: 短期 MA の期間（例: 25）
        long:  長期 MA の期間（例: 75）
    """

    def __init__(self, short: int = 25, long: int = 75) -> None:
        if short >= long:
            raise ValueError(f"short({short}) は long({long}) より小さくしてください。")
        self.short = short
        self.long  = long

    @property
    def name(self) -> str:
        return f"GoldenCross(short={self.short}, long={self.long})"

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        過去データのみから MA クロスシグナルを生成する。

        先読みリーク防止:
          - data は engine が「現在足まで」に切り取ったもの
          - shift(1) で前足と現在足を比較（未来参照なし）
        """
        if len(data) < self.long + 1:
            return 0  # データ不足

        close = data["Close"]

        # 移動平均を計算（過去データのみ）
        ma_short = close.rolling(self.short).mean()
        ma_long  = close.rolling(self.long).mean()

        # 現在足（-1）と前足（-2）のクロスを判定
        # ※ shift(1) 相当の比較（iloc[-2] vs iloc[-1]）
        prev_cross = ma_short.iloc[-2] - ma_long.iloc[-2]
        curr_cross = ma_short.iloc[-1] - ma_long.iloc[-1]

        if prev_cross < 0 and curr_cross >= 0:
            return 1   # ゴールデンクロス → 買い
        if prev_cross > 0 and curr_cross <= 0:
            return -1  # デッドクロス → 売り
        return 0


# ------------------------------------------------------------------ #
# バックテスト実行スクリプト（単体実行用）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    import os

    # プロジェクトルートを sys.path に追加
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import yfinance as yf
    from backtest.engine import BacktestEngine, TradeCost, WalkForwardValidator

    # --- データ取得（バックテスト用: yfinance で代替）---
    # 本番データは J-Quants API 推奨
    print("データ取得中...")
    SYMBOL = "7203.T"   # トヨタ自動車
    df = yf.download(SYMBOL, start="2020-01-01", end="2024-12-31", progress=False)
    df = df.droplevel(1, axis=1)   # MultiIndex を解除

    if df.empty:
        print("データ取得失敗。インターネット接続を確認してください。")
        sys.exit(1)

    print(f"取得件数: {len(df)} 日")

    # ------------------------------------------------------------------ #
    # シングルバックテスト
    # ------------------------------------------------------------------ #
    print("\n--- シングルバックテスト ---")
    strategy = GoldenCrossStrategy(short=25, long=75)
    engine   = BacktestEngine(
        prices  = df,
        strategy= strategy,
        capital = 1_000_000,
        cost    = TradeCost(),
        symbol  = SYMBOL,
    )
    result = engine.run()
    result.summary()
    result.plot()

    # ------------------------------------------------------------------ #
    # ウォークフォワード検証
    # ------------------------------------------------------------------ #
    print("\n--- ウォークフォワード検証 ---")
    wfv = WalkForwardValidator(
        prices       = df,
        strategy_cls = GoldenCrossStrategy,
        train_bars   = 252,   # 学習: 約1年
        test_bars    = 63,    # 検証: 約3ヶ月
        step_bars    = 63,    # ステップ: 約3ヶ月
        capital      = 1_000_000,
        symbol       = SYMBOL,
    )
    wf_results = wfv.run(
        param_grid={"short": [10, 25, 50], "long": [50, 75, 100, 200]}
    )
    wfv.summary(wf_results)
