"""
src/database/markets_bulk.py
J-Quants Bulk API markets/* 系（信用取引・空売り）4データセットの年次パーティション管理。

対象（全てStandardプランで取得可能・2026-07-23実測確認）:
  margin_interest   : markets/margin-interest（信用取引週末残高）    → margin/margin_interest/{year}.parquet
  margin_alert      : markets/margin-alert（日々公表信用取引残高）  → margin/margin_alert/{year}.parquet
  short_ratio       : markets/short-ratio（業種別空売り比率）        → shortselling/short_ratio/{year}.parquet
  short_sale_report : markets/short-sale-report（空売り残高報告）    → shortselling/short_sale_report/{year}.parquet

4データセットはいずれも実測データ量が小さい（月次gzip 14KB-300KB程度）ため、
fundamentals.pyと同じ「年次1ファイルへマージ」方式を採用する。列構成がデータセットごとに
大きく異なる（日付列名・識別子列名が不統一）ため、schema.py への個別TableSpec登録はせず、
必須列の存在チェックのみ行う軽量な汎用実装とする（fins/summaryのpass-through方針を踏襲）。

原本CSV.GZは archive/bulk/ へ永久保存する（削除しない）。Parquetはそこからの派生データ。
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.database import bulk_state
from src.jquants.bulk_client import (
    ENDPOINT_MARKETS_MARGIN_ALERT,
    ENDPOINT_MARKETS_MARGIN_INTEREST,
    ENDPOINT_MARKETS_SHORT_RATIO,
    ENDPOINT_MARKETS_SHORT_SALE_REPORT,
    JQuantsBulkClient,
)
from src.jquants.exceptions import JQuantsAPIError
from src.paths import ARCHIVE_BULK_DIR, DATABASE_MARGIN_DIR, DATABASE_SHORTSELLING_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class MarketDomain:
    key: str
    endpoint: str
    date_col: str
    dedupe_keys: tuple[str, ...]
    out_dir: Path


MARKET_DOMAINS: dict[str, MarketDomain] = {
    "margin_interest": MarketDomain(
        "margin_interest", ENDPOINT_MARKETS_MARGIN_INTEREST, "Date", ("Date", "Code"),
        DATABASE_MARGIN_DIR / "margin_interest",
    ),
    "margin_alert": MarketDomain(
        "margin_alert", ENDPOINT_MARKETS_MARGIN_ALERT, "PubDate", ("PubDate", "Code"),
        DATABASE_MARGIN_DIR / "margin_alert",
    ),
    "short_ratio": MarketDomain(
        "short_ratio", ENDPOINT_MARKETS_SHORT_RATIO, "Date", ("Date", "S33"),
        DATABASE_SHORTSELLING_DIR / "short_ratio",
    ),
    "short_sale_report": MarketDomain(
        "short_sale_report", ENDPOINT_MARKETS_SHORT_SALE_REPORT, "DiscDate",
        ("DiscDate", "CalcDate", "Code", "SSName"),
        DATABASE_SHORTSELLING_DIR / "short_sale_report",
    ),
}


def _year_path(domain: MarketDomain, year: int) -> Path:
    ensure_database_market_dirs()
    domain.out_dir.mkdir(parents=True, exist_ok=True)
    return domain.out_dir / f"{year}.parquet"


def _parse_csv_gz(path: Path) -> pd.DataFrame:
    # TSEMrgnRegCls等の固定桁コード列は月によって全行数値化/文字列化の推論が割れてArrow変換時に
    # 失敗することがある（実測確認済み）ため、既知のコード系列は明示的にstr固定する。
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, dtype={"Code": str, "S33": str, "TSEMrgnRegCls": str})


def _load_year(domain: MarketDomain, year: int) -> pd.DataFrame:
    path = _year_path(domain, year)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def ingest_month(domain_key: str, year: int, month: int, *, force: bool = False) -> dict:
    """指定月のBulkデータを取得し、対応する年次Parquetへマージ保存する。"""
    domain = MARKET_DOMAINS[domain_key]
    period = f"{year}{month:02d}"

    if not force and bulk_state.is_ingested(domain.endpoint, period):
        logger.info("[MARKETS_BULK] domain=%s period=%s 既に取り込み済み・スキップ", domain_key, period)
        return {"domain": domain_key, "period": period, "status": "skipped"}

    from src.database.minute_bars import _trading_days_in_month  # 月内営業日計算を再利用

    client = JQuantsBulkClient()
    trading_days = _trading_days_in_month(year, month)
    if not trading_days:
        return {"domain": domain_key, "period": period, "status": "no_trading_days"}

    source = client.resolve_month_source(domain.endpoint, year, month, trading_days)
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
                logger.warning("[MARKETS_BULK] domain=%s key=%s 取得不可・スキップ: %s", domain_key, key, e)
                continue
            total_bytes += size
            last_sha256 = sha256
            downloaded_keys.append(key)
            frames.append(_parse_csv_gz(dest))
        status = "partial"

    if not frames:
        logger.warning("[MARKETS_BULK] domain=%s period=%s データなし", domain_key, period)
        return {"domain": domain_key, "period": period, "status": "empty"}

    new_df = pd.concat(frames, ignore_index=True)
    missing = [c for c in domain.dedupe_keys if c not in new_df.columns]
    if missing:
        raise ValueError(f"[MARKETS_BULK] domain={domain_key}: 必須列が欠落: {missing}")
    new_df[domain.date_col] = pd.to_datetime(new_df[domain.date_col])

    # object dtype列（"*"等の抑制記号が数値列に混入するケースを実測確認済み・margin-alertの
    # ShrtOutChg等）はPyArrow変換時に型混在で失敗するため、date_col以外の全object列をstrへ
    # 明示的に統一する（生データをそのまま透過させる方針・NaN化して情報を欠落させない）。
    for col in new_df.columns:
        if col != domain.date_col and new_df[col].dtype == "object":
            new_df[col] = new_df[col].astype(str)

    year_of_data = int(new_df[domain.date_col].dt.year.mode().iloc[0])
    existing = _load_year(domain, year_of_data)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    # 既存年ファイル(existing)とnew_dfのマージでも同じ列が月によって数値/object化しうるため、
    # 保存直前にもう一度objectダブり列をstrへ統一する（"*"混入等の月差異吸収）。
    for col in combined.columns:
        if col != domain.date_col and combined[col].dtype == "object":
            combined[col] = combined[col].astype(str)
    combined = combined.sort_values(domain.date_col).drop_duplicates(
        subset=list(domain.dedupe_keys), keep="last",
    ).reset_index(drop=True)

    out_path = _year_path(domain, year_of_data)
    combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    csv_gz_local_path = (
        str(ARCHIVE_BULK_DIR / downloaded_keys[0]) if len(downloaded_keys) == 1
        else ";".join(str(ARCHIVE_BULK_DIR / k) for k in downloaded_keys)
    )
    row = bulk_state.build_row(
        endpoint=domain.endpoint,
        period=period,
        source_key=";".join(downloaded_keys),
        source_size_bytes=total_bytes,
        source_last_modified="",
        csv_gz_local_path=csv_gz_local_path,
        csv_gz_sha256=last_sha256,
        parquet_local_path=str(out_path),
        parquet_row_count=len(combined),
        parquet_size_bytes=out_path.stat().st_size,
        symbol_count=int(new_df["Code"].nunique()) if "Code" in new_df.columns else 0,
        status=status,
    )
    bulk_state.upsert(row)
    logger.info(
        "[MARKETS_BULK] domain=%s period=%s status=%s new_rows=%d year_total_rows=%d",
        domain_key, period, status, len(new_df), len(combined),
    )
    return {"domain": domain_key, "period": period, "status": status, "new_rows": len(new_df)}


def backfill(domain_key: str, start_year: int = 2016, start_month: int = 7) -> list[dict]:
    """指定domainをstart_year/start_month〜当月まで月単位でバックフィルする。"""
    today = datetime.now(_JST).date()
    results = []
    year, month = start_year, start_month
    while (year, month) <= (today.year, today.month):
        results.append(ingest_month(domain_key, year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return results


def backfill_all(start_year: int = 2016, start_month: int = 7) -> dict[str, list[dict]]:
    return {key: backfill(key, start_year, start_month) for key in MARKET_DOMAINS}


def main() -> int:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="markets/* Bulkバックフィル（margin/shortselling）")
    parser.add_argument("--domain", choices=list(MARKET_DOMAINS) + ["all"], default="all")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--start-month", type=int, default=7)
    args = parser.parse_args()

    if args.domain == "all":
        for key, results in backfill_all(args.start_year, args.start_month).items():
            for r in results:
                print(r)
    else:
        for r in backfill(args.domain, args.start_year, args.start_month):
            print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
