"""
backtest/strategy.py
戦略基底クラス

新しい戦略を作るときは BaseStrategy を継承し、
generate_signal() メソッドだけを実装してください。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    戦略基底クラス

    先読みリーク防止の契約:
      generate_signal(data) に渡される data は
      「現在時点までの過去データのみ」です。
      engine.py がこれを保証します。
      サブクラスは data.iloc[-1] を「現在」として扱い、
      data.iloc[-1] より先のデータを参照してはなりません。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """戦略名を返す。"""

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        シグナルを返す。

        Args:
            data: DatetimeIndex を持つ OHLCV DataFrame。
                  末尾が「現在の足」。未来データは含まない。

        Returns:
            +1 : 買いシグナル
            -1 : 売りシグナル
             0 : 何もしない
        """
