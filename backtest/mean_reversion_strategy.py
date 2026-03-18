"""
backtest/mean_reversion_strategy.py
短期平均回帰戦略（Inside the Black Box 理論準拠）

【戦略の構造】
  エントリー:
    1. 5日RSI < 25（売られ過ぎ）
    2. 終値 > 200日MA（大局上昇トレンド内）
    3. 終値 > 50日MA × 0.85（落下するナイフ回避：-15%以内）

  エグジット:
    A. 5日RSI > 65（回復）
    B. 終値 < エントリー価格 × (1 - stop_loss_pct)（損切り）
    C. max_hold_days 日経過（時間切れ）

【設計根拠（Inside the Black Box, Ch.3）】
  - Mean reversion: 均衡点からの乖離は収束する
  - トレンドフィルター（MA200）: フジコ法と補完関係
    フジコ法 → トレンド相場で利益
    本戦略  → レンジ相場で利益

【先読みリーク防止（CLAUDE.md ルール1）】
  engine.py から渡される data は prices.iloc[:i+1] に切り取り済み。
  本クラスでは iloc[-1]（現在足）と過去データのみ参照する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.strategy import BaseStrategy


def _calc_rsi(close: pd.Series, period: int = 5) -> float:
    """
    RSI を計算して最新値を返す。

    Wilder の EMA 平滑化（標準実装）。
    期間不足の場合は NaN を返す。
    """
    if len(close) < period + 1:
        return float("nan")

    delta = close.diff().dropna()
    if len(delta) < period:
        return float("nan")

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # 最初の平均（単純平均）
    avg_gain = gain.iloc[:period].mean()
    avg_loss = loss.iloc[:period].mean()

    # Wilder の EMA で残りを計算
    for g, l in zip(gain.iloc[period:], loss.iloc[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionStrategy(BaseStrategy):
    """
    短期平均回帰戦略

    Args:
        rsi_period:      RSI 計算期間（デフォルト5日）
        rsi_entry:       エントリー RSI 閾値（デフォルト25）
        rsi_exit:        エグジット RSI 閾値（デフォルト65）
        ma_long:         トレンドフィルター 長期MA（デフォルト200日）
        stop_loss_pct:   損切りライン（デフォルト0.07 = -7%）
        max_hold_days:   最大保有日数（デフォルト10日）
        knife_threshold: 落下ナイフ回避：50日MA から何%下が限界か（デフォルト0.15）
    """

    def __init__(
        self,
        rsi_period:      int   = 5,
        rsi_entry:       float = 25.0,
        rsi_exit:        float = 65.0,
        ma_long:         int   = 200,
        stop_loss_pct:   float = 0.07,
        max_hold_days:   int   = 10,
        knife_threshold: float = 0.15,
    ) -> None:
        self.rsi_period      = rsi_period
        self.rsi_entry       = rsi_entry
        self.rsi_exit        = rsi_exit
        self.ma_long         = ma_long
        self.stop_loss_pct   = stop_loss_pct
        self.max_hold_days   = max_hold_days
        self.knife_threshold = knife_threshold

        # 内部状態
        self._in_position:  bool  = False
        self._entry_price:  float = 0.0
        self._entry_bar:    int   = -1
        self._bar_count:    int   = 0

    @property
    def name(self) -> str:
        return (
            f"MeanRev(RSI{self.rsi_period}<{self.rsi_entry:.0f}, "
            f"exit>{self.rsi_exit:.0f}, MA{self.ma_long}, "
            f"SL={self.stop_loss_pct*100:.0f}%, hold<={self.max_hold_days}d)"
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        シグナル生成。
          +1 : 買い
          -1 : 売り
           0 : 何もしない
        """
        # 状態リセット（新しいティッカーの先頭バー）
        if len(data) <= 1:
            self._in_position = False
            self._entry_price = 0.0
            self._entry_bar   = -1
            self._bar_count   = 0
            return 0

        close = data["Close"]
        current_price = float(close.iloc[-1])
        self._bar_count = len(data) - 1

        # ---- MA 計算 ----
        if len(close) < self.ma_long:
            return 0
        ma200 = float(close.rolling(self.ma_long).mean().iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1])
        if np.isnan(ma200) or np.isnan(ma50):
            return 0

        # ---- RSI 計算 ----
        rsi = _calc_rsi(close, self.rsi_period)
        if np.isnan(rsi):
            return 0

        # ---- エグジット判定（ポジションあり）----
        if self._in_position:
            hold_days  = self._bar_count - self._entry_bar
            stop_price = self._entry_price * (1.0 - self.stop_loss_pct)

            if (
                rsi >= self.rsi_exit            # RSI 回復
                or current_price <= stop_price  # 損切り
                or hold_days >= self.max_hold_days  # 時間切れ
            ):
                self._in_position = False
                return -1

            return 0  # 保有継続

        # ---- エントリー判定（ポジションなし）----
        # 条件1: RSI 売られ過ぎ
        if rsi >= self.rsi_entry:
            return 0
        # 条件2: MA200 より上（大局上昇トレンド）
        if current_price <= ma200:
            return 0
        # 条件3: 落下するナイフ回避（MA50 から -knife_threshold 以内）
        if current_price < ma50 * (1.0 - self.knife_threshold):
            return 0

        # 全条件クリア → エントリー
        self._in_position = True
        self._entry_price = current_price
        self._entry_bar   = self._bar_count
        return 1
