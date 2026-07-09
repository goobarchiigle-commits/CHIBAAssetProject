"""
src/jquants/compaction.py
ステージング → 年パーティション raw/processed への再構築。ステージング元は2系統ある:
  1. 銘柄別 cache/staging/{symbol}.parquet（downloader.py・既存の銘柄別エンジン用）→ compact_years()
  2. 日次 cache/daily/day_YYYY-MM-DD.parquet（study75_downloader.py・Strategy C日次エンジン用）
     → compact_years_from_daily()
どちらも出力先（raw/processed の daily_bars_{year}.parquet）とスキーマは同一。

冪等設計: ステージングが正本（差分取得のたびに更新される）。コンパクションは
「今わかっている全データ」から対象年の raw/daily_bars_{year}.parquet と
processed/daily_bars_{year}.parquet を毎回作り直すだけで、追記・部分更新はしない
（Parquetは行単位追記ができないため。年ファイル単位で丸ごと再構築する方が事故がない）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.jquants import cache
from src.jquants.normalize import normalize_to_processed
from src.paths import JQUANTS_DAILY_STAGING_DIR, JQUANTS_PROCESSED_DIR, JQUANTS_RAW_DIR

logger = logging.getLogger(__name__)


def _read_all_staged() -> pd.DataFrame:
    frames = []
    for symbol in cache.list_staged_symbols():
        df = cache.load_staged(symbol)
        if df.empty:
            continue
        if "Code" not in df.columns:
            df = df.copy()
            df["Code"] = symbol.split(".")[0]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined


def raw_year_path(year: int) -> Path:
    return JQUANTS_RAW_DIR / f"daily_bars_{year}.parquet"


def processed_year_path(year: int) -> Path:
    return JQUANTS_PROCESSED_DIR / f"daily_bars_{year}.parquet"


def _write_year_files(year: int, year_df: pd.DataFrame) -> dict[str, int]:
    year_df.to_parquet(raw_year_path(year))
    processed_df = normalize_to_processed(year_df)
    processed_df.to_parquet(processed_year_path(year))
    logger.info(
        "[JQUANTS_COMPACTION] year=%d raw_rows=%d processed_rows=%d",
        year, len(year_df), len(processed_df),
    )
    return {"raw": int(len(year_df)), "processed": int(len(processed_df))}


def compact_years(years: set[int] | None = None) -> dict:
    """
    銘柄別ステージング（cache/staging/{symbol}.parquet）から、指定年（Noneなら全年）の
    raw/processed 年パーティションを再構築する。
    Returns: {"years": [...], "row_counts": {year: {"raw": n, "processed": n}}}
    """
    cache.ensure_dirs()
    JQUANTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_staged = _read_all_staged()
    if all_staged.empty:
        return {"years": [], "row_counts": {}}

    target_years = years if years is not None else set(all_staged["Date"].dt.year.unique())
    row_counts: dict[int, dict[str, int]] = {}

    for year in sorted(target_years):
        year_df = all_staged.loc[all_staged["Date"].dt.year == year].reset_index(drop=True)
        if year_df.empty:
            continue
        row_counts[year] = _write_year_files(year, year_df)

    return {"years": sorted(row_counts.keys()), "row_counts": row_counts}


# ────────────────────────────────────────────────────────────────────────── #
# Strategy C（日次イテレーション・study75_downloader.py用）
# ────────────────────────────────────────────────────────────────────────── #
def _daily_staging_files_for_year(year: int) -> list[Path]:
    return sorted(JQUANTS_DAILY_STAGING_DIR.glob(f"day_{year}-*.parquet"))


def _all_years_in_daily_staging() -> set[int]:
    years: set[int] = set()
    for path in JQUANTS_DAILY_STAGING_DIR.glob("day_*.parquet"):
        # day_YYYY-MM-DD.parquet
        try:
            years.add(int(path.stem.split("_", 1)[1].split("-", 1)[0]))
        except (IndexError, ValueError):
            logger.warning("[JQUANTS_COMPACTION] 年抽出失敗（無視）: %s", path)
    return years


def compact_years_from_daily(years: set[int] | None = None) -> dict:
    """
    日次ステージング（cache/daily/day_YYYY-MM-DD.parquet）から、指定年（Noneなら
    ステージングに存在する全年）の raw/processed 年パーティションを再構築する。
    ファイル名から年を判定して対象年のファイルだけ読み込むため、影響を受けた年以外は
    ディスクI/Oが発生しない（銘柄別ステージングと違い全ファイルを開く必要がない）。
    Returns: {"years": [...], "row_counts": {year: {"raw": n, "processed": n}}}
    """
    JQUANTS_DAILY_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    JQUANTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    target_years = years if years is not None else _all_years_in_daily_staging()
    row_counts: dict[int, dict[str, int]] = {}

    for year in sorted(target_years):
        files = _daily_staging_files_for_year(year)
        if not files:
            continue
        frames = [df for df in (pd.read_parquet(f) for f in files) if not df.empty]
        if not frames:
            continue
        year_df = pd.concat(frames, ignore_index=True)
        year_df["Date"] = pd.to_datetime(year_df["Date"])
        row_counts[year] = _write_year_files(year, year_df)

    return {"years": sorted(row_counts.keys()), "row_counts": row_counts}
