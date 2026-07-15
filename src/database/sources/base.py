"""
src/database/sources/base.py
データソース抽象化の最小プロトコル。

過剰な抽象化を避け、3メソッドのみに留める:
  - fetch_ohlcv(start, end): 対応する場合のみ実装（非対応データソースはNotImplementedErrorでよい）
  - fetch_master(kind): companies/classifications等、種別ごとにDataFrameを返す
  - as_of(): このソースの直近取得時刻（metadata記録用）
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class SourceAdapter(Protocol):
    name: str

    def fetch_ohlcv(self, start: str, end: str) -> pd.DataFrame:
        ...

    def fetch_master(self, kind: str) -> pd.DataFrame:
        ...

    def as_of(self) -> datetime:
        ...
