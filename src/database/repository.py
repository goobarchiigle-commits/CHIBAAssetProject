"""
src/database/repository.py
消費者（バックテスト・セクター/ETF/RS/ファクター分析・ML）向けの唯一の公開データアクセスAPI。

database/market/ 配下の物理レイアウト（年次パーティション・列名・ファイル配置）はこのクラスの
内部に隠蔽する。消費側コードは `from src.database.repository import MarketDataRepository` のみを
import し、database/market/... のパス文字列を一切書かないこと（CLAUDE.md: no_hardcode_path=true）。

既存バックテスト（Study75/76含む）を将来 database/market 参照へ移行する際は、
data/jquants/processed/{code}.parquet 直読みをこのクラスの get_ohlcv() 呼び出しに
置き換えるだけで済む設計にしている。
"""
from __future__ import annotations

import logging

import pandas as pd

from src.database.ohlcv import available_years, load_yearly_parquet
from src.paths import DATABASE_MASTER_DIR

logger = logging.getLogger(__name__)


def _load_master(table_name: str) -> pd.DataFrame:
    path = DATABASE_MASTER_DIR / f"{table_name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


class MarketDataRepository:
    """database/market/ の読み取り専用アクセサ。"""

    def get_ohlcv(
        self, codes: list[str] | None = None, start: str = "", end: str = "",
    ) -> pd.DataFrame:
        """
        start〜end（"YYYY-MM-DD"、省略時は全期間）のOHLCVを返す。codes省略時は全銘柄。
        必要な年次Parquetのみを読み込んで結合する。
        """
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None
        years = range(start_ts.year, end_ts.year + 1) if start_ts and end_ts else available_years()

        frames = [load_yearly_parquet(y) for y in years]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

        if start_ts is not None:
            df = df.loc[pd.to_datetime(df["Date"]) >= start_ts]
        if end_ts is not None:
            df = df.loc[pd.to_datetime(df["Date"]) <= end_ts]
        if codes is not None:
            df = df.loc[df["Code"].astype(str).isin(codes)]
        return df.reset_index(drop=True)

    def get_universe_asof(self, date: str, universe_name: str = "TSE_ALL") -> list[str]:
        """指定日時点でuniverse_nameに属していた銘柄コード一覧（サバイバーシップフリー）。"""
        universe = _load_master("universe")
        if universe.empty:
            return []
        cutoff = pd.Timestamp(date)
        subset = universe.loc[universe["UniverseName"] == universe_name]
        from_ok = pd.to_datetime(subset["FromDate"]) <= cutoff
        to_ok = subset["ToDate"].isna() | (pd.to_datetime(subset["ToDate"]) >= cutoff)
        return sorted(subset.loc[from_ok & to_ok, "Code"].astype(str).unique().tolist())

    def get_companies(self, codes: list[str] | None = None) -> pd.DataFrame:
        df = _load_master("companies")
        if codes is not None and not df.empty:
            df = df.loc[df["Code"].astype(str).isin(codes)]
        return df.reset_index(drop=True)

    def get_classifications(self, codes: list[str] | None = None) -> pd.DataFrame:
        df = _load_master("classifications")
        if codes is not None and not df.empty:
            df = df.loc[df["Code"].astype(str).isin(codes)]
        return df.reset_index(drop=True)

    def get_indices(self, index_codes: list[str] | None = None) -> pd.DataFrame:
        """指数マスタ（master/indices.parquet）を返す。価格時系列はv1未実装（indexディレクトリ参照）。"""
        df = _load_master("indices")
        if index_codes is not None and not df.empty:
            df = df.loc[df["IndexCode"].astype(str).isin(index_codes)]
        return df.reset_index(drop=True)
