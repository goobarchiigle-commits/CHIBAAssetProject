"""
backtest/ma_band_strategy.py
MAバンド戦略（Pine Script "MAバンド" indicator の Python バックテスト版）

【元コードの構造】
  Band1（短期: 9期間）  = MA(high) / MA(close) / MA(low)
  Band2（中期: 21期間） = MA(high) / MA(close) / MA(low)
  Band3（長期: 200期間）= MA(high) / MA(close) / MA(low)

【元コードにはエントリー条件が存在しない】
  Pine Script は indicator() のため描画ツールに過ぎない。
  以下の戦略ロジックはバンド構造から定義した標準的な解釈。

【先読みリーク防止（CLAUDE.md ルール1準拠）】
  - engine.py が data = prices.iloc[:i+1] に切り取って渡してくる
  - 本クラス内では iloc[-1]（現在足）と iloc[-2]（前足）のみ参照
  - high/low は「足確定後」の値を使う（再描画リスク排除）

【定義したエントリーロジック】
  エントリー（買い）:
    - 前足の close ≤ Band2 上端（u2）かつ
      現在足の close ＞ Band2 上端（u2）  ← 中期バンド上端ブレイクアウト
    - かつ現在足の close ＞ Band3 中央線（m3）← 長期トレンド上向きフィルター

  エグジット（売り）:
    - 前足の close ≥ Band1 下端（l1）かつ
      現在足の close ＜ Band1 下端（l1）  ← 短期バンド下端割れで即撤退

  根拠:
    Band2 上端ブレイクアウトは「中期的な上値抵抗を突破した勢い」を表す。
    Band1 下端割れは「短期的な下値支持を失った」最初のサインであり、
    含み損が大きくなる前の早めの利確・損切りとして機能する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.strategy import BaseStrategy


# ------------------------------------------------------------------ #
# MA計算ユーティリティ（Pine Script の ta.sma / ta.ema / ta.wma に対応）
# ------------------------------------------------------------------ #
def _calc_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    """
    Pine Script の calc_ma() 関数と同等の計算を行う。

    Args:
        series:  価格系列（Close / High / Low）
        length:  期間
        ma_type: "SMA" / "EMA" / "WMA"

    Returns:
        MA 系列（系列長は series と同じ）

    先読みリーク確認:
        rolling() / ewm() はすべて現在足以前のデータのみ参照。
        min_periods=length を明示して、データ不足時は NaN を返す。
    """
    if ma_type == "EMA":
        # Pine Script の ta.ema と同じ: alpha = 2 / (length + 1), adjust=False
        return series.ewm(span=length, adjust=False, min_periods=length).mean()

    if ma_type == "WMA":
        # Pine Script の ta.wma: 直近に大きい重み（線形加重）
        weights = np.arange(1, length + 1, dtype=float)
        weights /= weights.sum()
        return series.rolling(length, min_periods=length).apply(
            lambda x: float(np.dot(x, weights)), raw=True
        )

    # デフォルト: SMA
    return series.rolling(length, min_periods=length).mean()


# ------------------------------------------------------------------ #
# 戦略クラス
# ------------------------------------------------------------------ #
class MABandStrategy(BaseStrategy):
    """
    MAバンド戦略

    Args:
        len1:    Band1（短期）の期間  デフォルト 9
        len2:    Band2（中期）の期間  デフォルト 21
        len3:    Band3（長期）の期間  デフォルト 200
        ma_type: MA の種類 "SMA" / "EMA" / "WMA"
    """

    def __init__(
        self,
        len1:    int = 9,
        len2:    int = 21,
        len3:    int = 200,
        ma_type: str = "SMA",
    ) -> None:
        if not (len1 < len2 < len3):
            raise ValueError(
                f"期間は len1({len1}) < len2({len2}) < len3({len3}) である必要があります。"
            )
        self.len1    = len1
        self.len2    = len2
        self.len3    = len3
        self.ma_type = ma_type.upper()

    @property
    def name(self) -> str:
        return f"MABand({self.len1}/{self.len2}/{self.len3}, {self.ma_type})"

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        シグナルを生成する。

        先読みリーク防止:
          - data は engine.py が prices.iloc[:i+1] に切り取ったもの
          - iloc[-1] = 現在足（確定済み終値）
          - iloc[-2] = 前足
          - high/low は確定済みの値のみ使用（再描画リスクなし）
        """
        if len(data) < self.len3 + 2:
            return 0  # ウォームアップ期間は何もしない

        close = data["Close"]
        high  = data["High"]
        low   = data["Low"]

        # --- バンド計算（過去データのみ使用）---
        u2 = _calc_ma(high,  self.len2, self.ma_type)  # Band2 上端
        l1 = _calc_ma(low,   self.len1, self.ma_type)  # Band1 下端
        m3 = _calc_ma(close, self.len3, self.ma_type)  # Band3 中央線

        # NaN チェック
        if any(s.iloc[-1] != s.iloc[-1] or s.iloc[-2] != s.iloc[-2]
               for s in [u2, l1, m3]):
            return 0

        # --- シグナル判定（現在足と前足のみ比較）---
        #  エントリー: Band2上端ブレイクアウト + 長期トレンド上向き
        if close.iloc[-2] <= u2.iloc[-2] and \
           close.iloc[-1] >  u2.iloc[-1] and \
           close.iloc[-1] >  m3.iloc[-1]:
            return 1   # 買い

        # エグジット: Band1下端割れ
        if close.iloc[-2] >= l1.iloc[-2] and \
           close.iloc[-1] <  l1.iloc[-1]:
            return -1  # 売り

        return 0


# ------------------------------------------------------------------ #
# 単体実行スクリプト（python backtest/ma_band_strategy.py で実行）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import yfinance as yf
    from backtest.engine import BacktestEngine, TradeCost, WalkForwardValidator

    # --- データ取得 ---
    print("データ取得中...")
    SYMBOL = "7203.T"   # トヨタ自動車（検証用）
    df = yf.download(SYMBOL, start="2018-01-01", end="2024-12-31", progress=False)
    df = df.droplevel(1, axis=1)

    if df.empty:
        print("データ取得失敗。インターネット接続を確認してください。")
        sys.exit(1)

    print(f"取得件数: {len(df)} 日\n")

    # ================================================================ #
    # シングルバックテスト（デフォルトパラメータ）
    # ================================================================ #
    print("=" * 60)
    print("【シングルバックテスト】デフォルトパラメータ (9/21/200, SMA)")
    print("=" * 60)

    strategy = MABandStrategy(len1=9, len2=21, len3=200, ma_type="SMA")
    engine   = BacktestEngine(
        prices   = df,
        strategy = strategy,
        capital  = 2_000_000,   # 200万円（月20万円目標を念頭）
        cost     = TradeCost(),
        symbol   = SYMBOL,
    )
    result = engine.run()
    result.summary()
    result.plot()

    # ================================================================ #
    # ウォークフォワード検証（パラメータ最適化 + 過学習チェック）
    # ================================================================ #
    print("\n" + "=" * 60)
    print("【ウォークフォワード検証】")
    print("学習: 約1年 / 検証: 約3ヶ月 / ステップ: 約3ヶ月")
    print("=" * 60)

    wfv = WalkForwardValidator(
        prices       = df,
        strategy_cls = MABandStrategy,
        train_bars   = 252,
        test_bars    = 63,
        step_bars    = 63,
        capital      = 2_000_000,
        symbol       = SYMBOL,
    )
    wf_results = wfv.run(
        param_grid={
            "len1":    [5, 9, 14],
            "len2":    [21, 25],
            "len3":    [100, 200],
            "ma_type": ["SMA", "EMA"],
        }
    )
    wfv.summary(wf_results)

    # ================================================================ #
    # MA種類別比較
    # ================================================================ #
    print("\n" + "=" * 60)
    print("【MA種類別比較】")
    print("=" * 60)

    for ma in ["SMA", "EMA", "WMA"]:
        st = MABandStrategy(len1=9, len2=21, len3=200, ma_type=ma)
        eng = BacktestEngine(
            prices   = df,
            strategy = st,
            capital  = 2_000_000,
            cost     = TradeCost(),
            symbol   = SYMBOL,
        )
        r = eng.run()
        print(
            f"{ma:3s}  総リターン: {r.total_return * 100:+6.1f}%  "
            f"シャープ比: {r.sharpe_ratio:5.3f}  "
            f"最大DD: {r.max_drawdown * 100:+5.1f}%  "
            f"取引数: {r.num_trades:3d}回  "
            f"勝率: {r.win_rate * 100:4.1f}%"
        )
