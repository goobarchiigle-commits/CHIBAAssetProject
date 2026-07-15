"""
src/database/ohlcv.py
database/market/ohlcv/{year}.parquet の年次パーティション管理。

責務分離:
  fetch_ohlcv()          : sources層(JQuantsSource)経由でOHLCVを取得（HTTP詳細は知らない）
  split_by_year()         : DataFrameを年ごとに分割
  save_yearly_parquet()    : 1年分をParquet保存（pyarrow, snappy, index=False, Date/Code昇順・重複排除済み）
  load_yearly_parquet()     : 1年分を読み込み
  update_current_year()      : 当年分のみ差分更新（過去年ファイルは一切書き換えない）
"""
from __future__ import annotations

import logging

import pandas as pd

from src.database.dtypes import optimize_dtypes
from src.database.schema import validate_schema
from src.market.jpx_calendar import JPXCalendar
from src.paths import DATABASE_OHLCV_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)

_SORT_KEYS = ["Date", "Code"]
_DEDUPE_KEYS = ["Date", "Code"]


def _sort_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(_SORT_KEYS).reset_index(drop=True)
    df = df.drop_duplicates(subset=_DEDUPE_KEYS, keep="last").reset_index(drop=True)
    return df


def trading_days_in_range(start: str, end: str) -> list[pd.Timestamp]:
    """start〜end（両端含む）の営業日一覧（JPXCalendar基準）。"""
    calendar = JPXCalendar()
    all_days = pd.date_range(start, end, freq="D")
    return [d for d in all_days if calendar.is_trading_day(d)]


def fetch_ohlcv(start: str, end: str, source) -> pd.DataFrame:
    """
    source（JQuantsSource等・fetch_ohlcv_for_date(date:"YYYYMMDD")を持つオブジェクト）を使い、
    start〜end の全営業日分のOHLCVを取得して1つのDataFrameに結合する。
    HTTP・認証・レート制限の詳細は source（sources/jquants_source.py）に委譲する。
    """
    frames: list[pd.DataFrame] = []
    for day in trading_days_in_range(start, end):
        day_str = day.strftime("%Y%m%d")
        df = source.fetch_ohlcv_for_date(day_str)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def split_by_year(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """DataFrameをDate列の年ごとに分割する。"""
    if df.empty:
        return {}
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return {int(year): group.reset_index(drop=True) for year, group in df.groupby(df["Date"].dt.year)}


def _year_path(year: int):
    ensure_database_market_dirs()
    return DATABASE_OHLCV_DIR / f"{year}.parquet"


def save_yearly_parquet(df: pd.DataFrame, year: int) -> None:
    """
    1年分のOHLCVをdatabase/market/ohlcv/{year}.parquetへ保存する。
    保存前にDate/Code昇順ソート・(Date,Code)重複排除・dtype最適化を行う（ユーザー仕様通り）。
    """
    validate_schema(df, "ohlcv")
    df = _sort_and_dedupe(df)
    df = optimize_dtypes(df, "ohlcv")
    path = _year_path(year)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.info("[DATABASE_OHLCV] saved year=%d rows=%d path=%s", year, len(df), path)


def available_years() -> list[int]:
    """database/market/ohlcv/ 配下に存在する年次Parquetの年一覧を返す。"""
    if not DATABASE_OHLCV_DIR.exists():
        return []
    return sorted(int(p.stem) for p in DATABASE_OHLCV_DIR.glob("*.parquet") if p.stem.isdigit())


def load_yearly_parquet(year: int) -> pd.DataFrame:
    """1年分のOHLCVを読み込む。ファイルが存在しなければ空のDataFrame（スキーマ列のみ）を返す。"""
    path = _year_path(year)
    if not path.exists():
        from src.database.schema import OHLCV_SCHEMA
        return pd.DataFrame(columns=OHLCV_SCHEMA.all_columns)
    return pd.read_parquet(path, engine="pyarrow")


def update_current_year(new_df: pd.DataFrame, year: int | None = None) -> None:
    """
    当年分のみを差分更新する。既存の当年Parquetを読み込み、new_dfと結合・重複排除・ソートして
    上書き保存する。year省略時はnew_dfのDate列から自動判定する（複数年混在時はsplit_by_year()を
    使うこと）。過去年ファイルには一切触れない。
    """
    if new_df.empty:
        logger.info("[DATABASE_OHLCV] update_current_year: new_df is empty, no-op")
        return
    new_df = new_df.copy()
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    years_present = new_df["Date"].dt.year.unique()
    if year is None:
        if len(years_present) != 1:
            raise ValueError(
                f"update_current_year: new_dfが複数年にまたがっています({sorted(years_present)})。"
                "split_by_year()で年ごとに分割してから save_yearly_parquet() を使ってください。"
            )
        year = int(years_present[0])

    existing = load_yearly_parquet(year)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    save_yearly_parquet(combined, year)
