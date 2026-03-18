"""
backtest/reve_lite_strategy.py
REVE Lite JP — Python 移植版

【戦略の構造】
  REVE スコア: 21コンポーネントの複合モメンタムスコア（0〜3.0 スケール）
  エントリー: グレーフラグ（REVE が一時低下後に回復）かつ MA20 > MA200
  エグジット: BacktestEngine の TP（+5%）/ 保有バー数上限（10バー）に委任

【REVE スコアの21コンポーネント】
  MA アライメント(10): Close>MA20/50/100/200, MA20>50, 50>100, 100>200,
                        MA200 傾き(21日), MA50 傾き(10日), MA20 傾き(5日)
  モメンタム      ( 5): 5/10/21/63日リターン+, 20日高値ブレイクアウト
  ボリューム      ( 3): Vol>VolMA20, Vol>VolMA50, Vol>Vol[5]
  レンジポジション( 3): 20日レンジ上半分, 上4分の1, 200日レンジ上半分

【エントリー閾値（デフォルト）】
  watch_threshold  = 1.2  (REVE < 1.2 でグレーフラグ立て)
  orange_threshold = 1.8  (REVE >= 1.8 でエントリー発動)
"""

from __future__ import annotations

import pandas as pd

from backtest.strategy import BaseStrategy


class ReveLiteStrategy(BaseStrategy):
    """
    REVE Lite JP 戦略（Python 移植版）

    Args:
        watch_threshold:  グレーフラグを立てる REVE 閾値（デフォルト 1.2）
        orange_threshold: エントリーを発動する REVE 閾値（デフォルト 1.8）
        min_bars:         最低必要バー数（デフォルト 225）
    """

    def __init__(
        self,
        watch_threshold:  float = 1.2,
        orange_threshold: float = 1.8,
        min_bars:         int   = 225,
    ) -> None:
        self.watch_threshold  = watch_threshold
        self.orange_threshold = orange_threshold
        self.min_bars         = min_bars
        self._flag_gray       = False

    @property
    def name(self) -> str:
        return f"ReveLite(w={self.watch_threshold}, e={self.orange_threshold})"

    # ------------------------------------------------------------------ #
    # REVE スコア計算
    # ------------------------------------------------------------------ #
    def _calc_reve(self, data: pd.DataFrame) -> float:
        """
        REVE スコアを計算する（21コンポーネント、0〜3.0 スケール）。

        min_bars に満たない場合は 0.0 を返す。
        """
        n = len(data)
        if n < self.min_bars:
            return 0.0

        close = data["Close"]
        high  = data["High"]
        low   = data["Low"]
        has_vol = (
            "Volume" in data.columns
            and data["Volume"].iloc[-1] > 0
        )

        # 移動平均
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()
        ma100 = close.rolling(100).mean()
        ma200 = close.rolling(200).mean()

        c    = close.iloc[-1]
        m20  = ma20.iloc[-1]
        m50  = ma50.iloc[-1]
        m100 = ma100.iloc[-1]
        m200 = ma200.iloc[-1]

        score = 0

        # === MA アライメント (10点) ===
        score += int(c > m20)                                       # 1
        score += int(c > m50)                                       # 2
        score += int(c > m100)                                      # 3
        score += int(c > m200)                                      # 4
        score += int(m20 > m50)                                     # 5
        score += int(m50 > m100)                                    # 6
        score += int(m100 > m200)                                   # 7
        score += int(ma200.iloc[-1] > ma200.iloc[-22])              # 8: MA200 傾き(21日)
        score += int(ma50.iloc[-1]  > ma50.iloc[-11])               # 9: MA50  傾き(10日)
        score += int(ma20.iloc[-1]  > ma20.iloc[-6])                # 10: MA20  傾き(5日)

        # === モメンタム (5点) ===
        score += int(c > close.iloc[-6])                            # 11:  5日モメンタム
        score += int(c > close.iloc[-11])                           # 12: 10日モメンタム
        score += int(c > close.iloc[-22])                           # 13: 21日モメンタム
        score += int(c > close.iloc[-64])                           # 14: 63日モメンタム
        # 20日高値ブレイクアウト（前日までの20日高値を超えているか）
        h20_prev = high.rolling(20).max().iloc[-2]
        score += int(c >= h20_prev)                                 # 15

        # === ボリューム (3点) ===
        if has_vol:
            volume = data["Volume"]
            vol    = volume.iloc[-1]
            vma20  = volume.rolling(20).mean().iloc[-1]
            vma50  = volume.rolling(50).mean().iloc[-1]
            score += int(vol > vma20)                               # 16
            score += int(vol > vma50)                               # 17
            score += int(vol > volume.iloc[-6])                     # 18
        else:
            # ボリュームなし → 長期モメンタムで代替
            score += int(c > close.iloc[-126])                      # 16'
            score += int(n >= 253 and c > close.iloc[-252])         # 17'
            score += int(ma20.iloc[-1] > ma20.iloc[-11])            # 18'

        # === レンジポジション (3点) ===
        h20_val = high.rolling(20).max().iloc[-1]
        l20_val = low.rolling(20).min().iloc[-1]
        rng20   = h20_val - l20_val
        if rng20 > 0:
            pos20 = (c - l20_val) / rng20
            score += int(pos20 > 0.50)                              # 19
            score += int(pos20 > 0.75)                              # 20

        h200_val = high.rolling(200).max().iloc[-1]
        l200_val = low.rolling(200).min().iloc[-1]
        rng200   = h200_val - l200_val
        if rng200 > 0:
            score += int((c - l200_val) / rng200 > 0.50)           # 21

        return (score / 21.0) * 3.0

    # ------------------------------------------------------------------ #
    # シグナル生成
    # ------------------------------------------------------------------ #
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        シグナルを生成する。

        Returns:
            +1: エントリー（グレーフラグからの回復 + アップトレンド）
             0: ホールド / 待機
            -1: 未使用（エグジットは engine の TP / 時間制限に委任）
        """
        if len(data) <= 1:
            self._flag_gray = False
            return 0

        if len(data) < self.min_bars:
            return 0

        close   = data["Close"]
        ma20    = close.rolling(20).mean().iloc[-1]
        ma200   = close.rolling(200).mean().iloc[-1]
        uptrend = ma20 > ma200

        reve = self._calc_reve(data)

        # グレーフラグ更新: REVE が閾値を下回りアップトレンド継続中
        if not self._flag_gray and reve < self.watch_threshold and uptrend:
            self._flag_gray = True

        # エントリー: フラグ立ち + REVE 回復 + アップトレンド
        if self._flag_gray and reve >= self.orange_threshold and uptrend:
            self._flag_gray = False
            return 1

        return 0
