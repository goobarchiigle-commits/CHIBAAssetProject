"""
src/database/ticks.py
database/market/tick/{year}/{yyyymm}.parquet の月次パーティション管理（J-Quants Bulk API equities/trades）。

Tick月次ファイルは実測650MB-1.1GB gzip（展開後 数GB規模）となり、当環境（実測RAM合計8GB・
空き2-3GB規模）では一括読み込みが困難。pandas.read_csv(chunksize=...)でチャンク読み込みし、
pyarrow.parquet.ParquetWriterで逐次書き込む（全件を一度にメモリへ保持しない）。

minute_bars.pyと異なり、月内での全体ソート・重複排除は行わない（Bulk API原本は単一の完全な
月次スナップショットでチャンク内はソース順=Code→Time昇順が前提。日次live束ね時も日をまたいだ
重複は原理的に発生しない=Dateが異なるため、チャンク単位の処理で十分）。

原本CSV.GZは archive/bulk/ へ永久保存する（削除しない）。Parquetはそこからの派生データ。
"""
from __future__ import annotations

import gzip
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.database import bulk_state
from src.database.schema import TICK_SCHEMA, validate_schema
from src.jquants.bulk_client import ENDPOINT_EQUITIES_TRADES, JQuantsBulkClient
from src.jquants.exceptions import JQuantsAPIError
from src.paths import ARCHIVE_BULK_DIR, DATABASE_TICK_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

_CHUNK_ROWS = 500_000
_MIN_FREE_DISK_BYTES = 5 * 1024 ** 3  # 5GB未満なら中断（fail-closed）

_STANDARD_COLUMNS = TICK_SCHEMA.all_columns


def _check_disk_space(path: Path) -> None:
    import shutil
    usage = shutil.disk_usage(path.anchor or "C:\\")
    if usage.free < _MIN_FREE_DISK_BYTES:
        raise RuntimeError(
            f"[TICKS] ディスク空き容量不足のため中断: free={usage.free / 1024**3:.1f}GB "
            f"< 閾値{_MIN_FREE_DISK_BYTES / 1024**3:.0f}GB"
        )


def _month_path(year: int, month: int) -> Path:
    ensure_database_market_dirs()
    out_dir = DATABASE_TICK_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{year}{month:02d}.parquet"


def _cast_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df["Time"] = df["Time"].astype(str)
    df["SessionDistinction"] = df.get("SessionDistinction", pd.Series(dtype=str)).astype(str)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").astype("float32")
    df["TradingVolume"] = pd.to_numeric(df["TradingVolume"], errors="coerce").astype("float32")
    df["TransactionId"] = pd.to_numeric(df["TransactionId"], errors="coerce").astype("Int64")
    return df[_STANDARD_COLUMNS]


def _stream_csv_gz_to_parquet(csv_gz_paths: list[Path], out_path: Path) -> tuple[int, int]:
    """
    複数のCSV.GZ（日次束ねの場合は複数、月次の場合は1件）をチャンク単位で読み込み、
    1つのParquetへ逐次追記する。Returns: (総行数, ユニーク銘柄数)。
    """
    _check_disk_space(out_path)
    tmp_path = out_path.with_suffix(".parquet.part")
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    codes: set[str] = set()
    validated = False

    try:
        for csv_gz_path in csv_gz_paths:
            with gzip.open(csv_gz_path, "rt", encoding="utf-8") as f:
                for chunk in pd.read_csv(f, dtype={"Code": str}, chunksize=_CHUNK_ROWS):
                    chunk = _cast_chunk(chunk)
                    if not validated:
                        validate_schema(chunk, "tick")
                        validated = True
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd", compression_level=3)
                    writer.write_table(table)
                    total_rows += len(chunk)
                    codes.update(chunk["Code"].unique().tolist())
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        tmp_path.unlink(missing_ok=True)
        return 0, 0

    tmp_path.replace(out_path)
    return total_rows, len(codes)


def ingest_month(year: int, month: int, *, force: bool = False) -> dict:
    """
    指定月のTickを取得・保存する。月次historicalが利用可能ならそれを使用（status=ok）、
    未ロールアップなら日次liveを束ねて保存する（status=partial・次回再実行で自動昇格）。
    force=True でローカル状態を無視して再取得する。
    """
    endpoint = ENDPOINT_EQUITIES_TRADES
    period = f"{year}{month:02d}"

    if not force and bulk_state.is_ingested(endpoint, period):
        logger.info("[TICKS] period=%s 既に取り込み済み・スキップ", period)
        return {"period": period, "status": "skipped"}

    from src.database.minute_bars import _trading_days_in_month  # 既存の月内営業日計算を再利用

    client = JQuantsBulkClient()
    trading_days = _trading_days_in_month(year, month)
    if not trading_days:
        return {"period": period, "status": "no_trading_days"}

    source = client.resolve_month_source(endpoint, year, month, trading_days)
    downloaded_keys: list[str] = []
    csv_gz_paths: list[Path] = []
    total_bytes = 0
    last_sha256 = ""

    if source["granularity"] == "monthly":
        key = source["key"]
        dest = ARCHIVE_BULK_DIR / key
        size, sha256 = client.download_to_file(key, dest)
        total_bytes, last_sha256 = size, sha256
        downloaded_keys = [key]
        csv_gz_paths = [dest]
        status = "ok"
    else:
        for key in source["keys"]:
            dest = ARCHIVE_BULK_DIR / key
            try:
                size, sha256 = client.download_to_file(key, dest)
            except JQuantsAPIError as e:
                logger.warning("[TICKS] key=%s 取得不可（休場・未発行等・スキップ）: %s", key, e)
                continue
            total_bytes += size
            last_sha256 = sha256
            downloaded_keys.append(key)
            csv_gz_paths.append(dest)
        status = "partial"

    if not csv_gz_paths:
        logger.warning("[TICKS] period=%s データなし", period)
        return {"period": period, "status": "empty"}

    out_path = _month_path(year, month)
    total_rows, symbol_count = _stream_csv_gz_to_parquet(csv_gz_paths, out_path)
    if total_rows == 0:
        return {"period": period, "status": "empty"}

    csv_gz_local_path = (
        str(csv_gz_paths[0]) if len(csv_gz_paths) == 1
        else ";".join(str(p) for p in csv_gz_paths)
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
        parquet_row_count=total_rows,
        parquet_size_bytes=out_path.stat().st_size,
        symbol_count=symbol_count,
        status=status,
    )
    bulk_state.upsert(row)
    logger.info(
        "[TICKS] period=%s status=%s rows=%d symbols=%d", period, status, total_rows, symbol_count,
    )
    return {"period": period, "status": status, "rows": total_rows, "symbols": symbol_count}


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

    parser = argparse.ArgumentParser(description="Tickバックフィル（database/market/tick/）")
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
