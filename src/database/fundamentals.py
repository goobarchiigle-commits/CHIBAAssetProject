"""
src/database/fundamentals.py
database/market/fundamentals/summary/{year}.parquet の年次パーティション管理
（J-Quants Bulk API fins/summary）。

Study82 Phase D（2026-07-21）当時は銘柄別逐次REST取得で429レート制限に苦しんだが（research_state.md
参照）、Bulk API経由なら1リクエスト=月次全銘柄分で取得できるためその問題を根本回避する。

fins/summaryは日次データ量が小さい（実測 月次90KB程度gzip）ため、分足/Tickと異なり
年次1ファイルにマージする（ohlcv.pyのupdate_current_year方式を踏襲）。

原本CSV.GZは archive/bulk/ へ永久保存する（削除しない）。Parquetはそこからの派生データ。
"""
from __future__ import annotations

import gzip
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.database import bulk_state
from src.database.schema import FUNDAMENTALS_SUMMARY_SCHEMA, validate_schema
from src.jquants.bulk_client import ENDPOINT_FINS_SUMMARY, JQuantsBulkClient
from src.jquants.exceptions import JQuantsAPIError
from src.paths import ARCHIVE_BULK_DIR, DATABASE_FUNDAMENTALS_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

_DEDUPE_KEYS = ["DiscNo"]
_SORT_KEYS = ["DiscDate", "Code"]


def _summary_dir() -> Path:
    ensure_database_market_dirs()
    out_dir = DATABASE_FUNDAMENTALS_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _year_path(year: int) -> Path:
    return _summary_dir() / f"{year}.parquet"


def _parse_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, dtype={"Code": str, "DiscNo": str})


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DiscDate"] = pd.to_datetime(df["DiscDate"])
    df["Code"] = df["Code"].astype(str)
    df["DiscNo"] = df["DiscNo"].astype(str)
    return df


def _load_year(year: int) -> pd.DataFrame:
    path = _year_path(year)
    if not path.exists():
        return pd.DataFrame(columns=FUNDAMENTALS_SUMMARY_SCHEMA.all_columns)
    return pd.read_parquet(path, engine="pyarrow")


def ingest_month(year: int, month: int, *, force: bool = False) -> dict:
    """
    指定月のfins/summaryを取得し、database/market/fundamentals/summary/{year}.parquet へ
    マージ保存する（DiscNoで重複排除・DiscDate/Code昇順ソート）。
    """
    endpoint = ENDPOINT_FINS_SUMMARY
    period = f"{year}{month:02d}"

    if not force and bulk_state.is_ingested(endpoint, period):
        logger.info("[FUNDAMENTALS] period=%s 既に取り込み済み・スキップ", period)
        return {"period": period, "status": "skipped"}

    from src.database.minute_bars import _trading_days_in_month  # 月内営業日計算を再利用

    client = JQuantsBulkClient()
    trading_days = _trading_days_in_month(year, month)
    if not trading_days:
        return {"period": period, "status": "no_trading_days"}

    source = client.resolve_month_source(endpoint, year, month, trading_days)
    frames: list[pd.DataFrame] = []
    downloaded_keys: list[str] = []
    total_bytes = 0
    last_sha256 = ""

    if source["granularity"] == "monthly":
        key = source["key"]
        dest = ARCHIVE_BULK_DIR / key
        size, sha256 = client.download_to_file(key, dest)
        total_bytes, last_sha256 = size, sha256
        downloaded_keys = [key]
        frames.append(_parse_csv_gz(dest))
        status = "ok"
    else:
        for key in source["keys"]:
            dest = ARCHIVE_BULK_DIR / key
            try:
                size, sha256 = client.download_to_file(key, dest)
            except JQuantsAPIError as e:
                logger.warning("[FUNDAMENTALS] key=%s 取得不可（休場・未発行等・スキップ）: %s", key, e)
                continue
            total_bytes += size
            last_sha256 = sha256
            downloaded_keys.append(key)
            frames.append(_parse_csv_gz(dest))
        status = "partial"

    if not frames:
        logger.warning("[FUNDAMENTALS] period=%s データなし", period)
        return {"period": period, "status": "empty"}

    new_df = pd.concat(frames, ignore_index=True)
    new_df = _standardize(new_df)
    validate_schema(new_df, "fundamentals_summary")

    existing = _load_year(year)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(_SORT_KEYS).drop_duplicates(subset=_DEDUPE_KEYS, keep="last").reset_index(drop=True)

    out_path = _year_path(year)
    combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    csv_gz_local_path = (
        str(ARCHIVE_BULK_DIR / downloaded_keys[0]) if len(downloaded_keys) == 1
        else ";".join(str(ARCHIVE_BULK_DIR / k) for k in downloaded_keys)
    )
    row = bulk_state.build_row(
        endpoint=endpoint,
        period=period,
        source_key=";".join(downloaded_keys),
        source_size_bytes=total_bytes,
        source_last_modified="",
        csv_gz_local_path=csv_gz_local_path,
        csv_gz_sha256=last_sha256,
        parquet_local_path=str(out_path),
        parquet_row_count=len(combined),
        parquet_size_bytes=out_path.stat().st_size,
        symbol_count=int(new_df["Code"].nunique()),
        status=status,
    )
    bulk_state.upsert(row)
    logger.info(
        "[FUNDAMENTALS] period=%s status=%s new_rows=%d year_total_rows=%d",
        period, status, len(new_df), len(combined),
    )
    return {"period": period, "status": status, "new_rows": len(new_df), "year_total_rows": len(combined)}


def backfill(start_year: int = 2024, start_month: int = 7) -> list[dict]:
    """start_year/start_month（契約開始月）〜当月まで月単位でバックフィルする。"""
    today = datetime.now(_JST).date()
    results = []
    year, month = start_year, start_month
    while (year, month) <= (today.year, today.month):
        results.append(ingest_month(year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return results


def main() -> int:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="fins/summaryバックフィル（database/market/fundamentals/summary/）")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--month", help="単月のみ実行（YYYYMM形式）")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.month:
        y, m = int(args.month[:4]), int(args.month[4:6])
        print(ingest_month(y, m, force=args.force))
    else:
        for r in backfill(args.start_year, args.start_month):
            print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
