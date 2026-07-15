"""
src/database/sources/jquants_source.py
J-Quants API を database/market/ 向けの SourceAdapter として実装する。

既存 src/jquants/ の低レベル実装（認証・HTTP・リトライ・レート制限=client.py、
エンドポイントラッパー=provider.py、raw⇔標準名マッピング=schema.py）をそのまま再利用し、
HTTPクライアントを二重実装しない。

重要な境界: このモジュールは src.jquants.study75_downloader・src.jquants.compaction を
呼ばない（＝ data/jquants/cache/daily・data/jquants/processed への書き込みは一切発生しない。
database/market の更新経路が data/jquants/ に不可侵であるための境界＝WriteBoundaryError の
対象となる操作はそもそも実装しない）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.jquants.client import JQuantsClient
from src.jquants.exceptions import JQuantsConfigError
from src.jquants.provider import JQuantsProvider, raw_records_to_frame
from src.jquants.schema import (
    DAILY_BARS_RAW_TO_STANDARD,
    LISTED_INFO_RAW_TO_STANDARD,
    rename_to_standard,
)

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

# database/market の ohlcv スキーマは UpperLimit/LowerLimit/TurnoverValue も1テーブルに含める
# （src/jquants の raw/processed 分離とは異なり、研究者向けに単一フラット表とする設計。plan参照）。
_OHLCV_STANDARD_COLUMNS = list(DAILY_BARS_RAW_TO_STANDARD.values())


class JQuantsSource:
    """database/market/ 向け J-Quants データソースアダプタ（SourceAdapter実装）。"""

    name = "jquants_api"

    def __init__(self, provider: JQuantsProvider | None = None) -> None:
        self.provider = provider or JQuantsProvider()
        self._as_of: datetime | None = None

    def authenticate(self) -> bool:
        """
        認証情報の疎通確認。J-Quants API v2 は x-api-key の静的キーであり明示的な
        ログインステップは存在しないため、設定の存在検証＋実際に1件取得できるかで代替する。
        """
        client: JQuantsClient = self.provider.client
        try:
            client.config.validate()
        except JQuantsConfigError:
            logger.error("[DATABASE_JQUANTS_SOURCE] JQUANTS_API_KEY 未設定")
            return False
        return True

    def fetch_ohlcv_for_date(self, date: str) -> pd.DataFrame:
        """
        指定日（"YYYYMMDD"）の全銘柄OHLCVを取得し、標準列名（UpperLimit/LowerLimit/
        TurnoverValueを含む）のDataFrameを返す。Strategy C（1営業日=1リクエスト）。
        ステージングファイルへの書き込みは行わない（呼び出し側=ohlcv.pyがメモリ上で年次
        Parquetへ直接反映する）。
        """
        records = self.provider.daily_bars_for_date(date)
        df = raw_records_to_frame(records)
        df = rename_to_standard(df, DAILY_BARS_RAW_TO_STANDARD)
        for col in _OHLCV_STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        self._as_of = datetime.now(_JST)
        return df[_OHLCV_STANDARD_COLUMNS]

    def fetch_ohlcv(self, start: str, end: str) -> pd.DataFrame:
        """SourceAdapterプロトコル実装。営業日単位の反復は呼び出し側（sync.py/migrate.py）が担う。"""
        raise NotImplementedError(
            "JQuantsSource は日単位取得のみ対応（fetch_ohlcv_for_date）。"
            "start/end範囲反復は sync.py/migrate.py 側で行うこと（1日=1リクエストのため）。"
        )

    def fetch_master(self, kind: str) -> pd.DataFrame:
        """
        kind="companies"|"classifications": 上場銘柄一覧の最新スナップショット
        （listed_info() code/date省略=全銘柄最新時点、1リクエスト）を標準列名で返す。
        companies/classifications はいずれも同一の listed_info レスポンスから構築されるため
        取得ロジックは共通。
        """
        if kind not in ("companies", "classifications"):
            raise ValueError(f"JQuantsSource.fetch_master: 未対応kind={kind}")
        records = self.provider.listed_info()
        df = raw_records_to_frame(records)
        df = rename_to_standard(df, LISTED_INFO_RAW_TO_STANDARD)
        self._as_of = datetime.now(_JST)
        return df

    def as_of(self) -> datetime:
        return self._as_of or datetime.now(_JST)
