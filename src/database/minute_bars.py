"""
src/database/minute_bars.py
database/market/minute/{year}/{yyyymm}.parquet の月次パーティション管理
（J-Quants Bulk API equities/bars/minute）。

grain: 月次・全銘柄1ファイル（Bulk APIのhistorical月次ファイルと1:1対応）。直近の未ロールアップ月は
日次liveファイルを束ねて同じ月次Parquetへ書き込み status="partial" として記録する（次回実行時に
月次historicalへ切り替わり次第 status="ok" で上書きされる）。

原本CSV.GZは archive/bulk/ へ永久保存する（削除しない）。Parquetはそこからの派生データ。
"""
from __future__ import annotations

import gzip
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.database import bulk_state
from src.database.dtypes import optimize_dtypes
from src.database.schema import validate_schema
from src.jquants.bulk_client import ENDPOINT_EQUITIES_BARS_MINUTE, JQuantsBulkClient
from src.jquants.exceptions import JQuantsAPIError
from src.market.jpx_calendar import JPXCalendar
from src.paths import ARCHIVE_BULK_DIR, DATABASE_MINUTE_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

_RENAME = {"O": "Open", "H": "High", "L": "Low", "C": "Close", "Vo": "Volume", "Va": "TurnoverValue"}
_SORT_KEYS = ["Date", "Time", "Code"]
_DEDUPE_KEYS = ["Date", "Time", "Code"]


def _trading_days_in_month(year: int, month: int) -> list[str]:
    calendar = JPXCalendar()
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    today = pd.Timestamp(datetime.now(_JST).date())
    if end > today:
        end = today
    if start > end:
        return []
    all_days = pd.date_range(start, end, freq="D")
    return [d.strftime("%Y%m%d") for d in all_days if calendar.is_trading_day(d)]


def _month_path(year: int, month: int) -> Path:
    ensure_database_market_dirs()
    out_dir = DATABASE_MINUTE_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{year}{month:02d}.parquet"


def _parse_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, dtype={"Code": str})


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_RENAME)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    return df


def ingest_month(year: int, month: int, *, force: bool = False) -> dict:
    """
    指定月の分足を取得・保存する。月次historicalが利用可能ならそれを使用（status=ok）、
    未ロールアップなら日次liveを束ねて保存する（status=partial・次回再実行で自動昇格）。
    force=True でローカル状態を無視して再取得する。
    """
    endpoint = ENDPOINT_EQUITIES_BARS_MINUTE
    period = f"{year}{month:02d}"

    if not force and bulk_state.is_ingested(endpoint, period):
        logger.info("[MINUTE_BARS] period=%s 既に取り込み済み・スキップ", period)
        return {"period": period, "status": "skipped"}

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
                logger.warning("[MINUTE_BARS] key=%s 取得不可（休場・未発行等・スキップ）: %s", key, e)
                continue
            total_bytes += size
            last_sha256 = sha256
            downloaded_keys.append(key)
            frames.append(_parse_csv_gz(dest))
        status = "partial"  # 月末に月次historicalへロールアップされ次第、再実行でstatus=okへ昇格する

    if not frames:
        logger.warning("[MINUTE_BARS] period=%s データなし", period)
        return {"period": period, "status": "empty"}

    df = pd.concat(frames, ignore_index=True)
    df = _standardize(df)
    df = df.sort_values(_SORT_KEYS).drop_duplicates(subset=_DEDUPE_KEYS, keep="last").reset_index(drop=True)
    df = optimize_dtypes(df, "minute_bars")
    validate_schema(df, "minute_bars")

    out_path = _month_path(year, month)
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)

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
        parquet_row_count=len(df),
        parquet_size_bytes=out_path.stat().st_size,
        symbol_count=int(df["Code"].nunique()),
        status=status,
    )
    bulk_state.upsert(row)
    logger.info(
        "[MINUTE_BARS] period=%s status=%s rows=%d symbols=%d", period, status, len(df), row["symbol_count"],
    )
    return {"period": period, "status": status, "rows": len(df), "symbols": row["symbol_count"]}


def backfill(start_year: int = 2024, start_month: int = 7) -> list[dict]:
    """start_year/start_month（契約開始月・実測で確認済みの値をデフォルトにしている）〜当月まで月単位でバックフィルする。"""
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

    parser = argparse.ArgumentParser(description="分足バックフィル（database/market/minute/）")
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
