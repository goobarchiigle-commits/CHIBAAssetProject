"""
src/database/migrate.py
data/jquants/processed（既存の成熟したJ-Quants取り込みパイプライン出力）から
database/market/ への一回限りの移行。

このモジュールが data/jquants/ を読む唯一の場所である（migrate_from_jquants_pipeline()実行後、
database/market の更新経路（sync.py）は data/jquants/ に一切依存しない・plan「確定方針8」参照）。
data/jquants/ 配下へは読み取り（pd.read_parquet）のみ行い、書き込みは一切行わない。

再ダウンロードは発生しない: OHLCV本体はゼロAPIコール（既存processed/raw parquetを変換するのみ）。
companies/classifications構築のみ listed_info スナップショット1回 + JPXプライム150 CSV 1回を要する。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.database import metadata as db_metadata
from src.database.indices import build_indices_parquet
from src.database.master import build_classifications_parquet, build_companies_parquet, build_universe_parquet
from src.database.ohlcv import save_yearly_parquet
from src.database.sources.jpx_official import JPXOfficialSource
from src.database.sources.jquants_source import JQuantsSource
from src.jquants.universe import load_universe_events
from src.paths import DATABASE_MASTER_DIR, JQUANTS_PROCESSED_DIR, JQUANTS_RAW_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

_RAW_EXTRA_COLUMNS = {"UL": "UpperLimit", "LL": "LowerLimit", "Va": "TurnoverValue"}


def _migrate_ohlcv_year(year: int) -> int:
    """
    data/jquants/processed/daily_bars_{year}.parquet（13列・正規化済み）に、
    data/jquants/raw/daily_bars_{year}.parquet の UpperLimit/LowerLimit/TurnoverValue を
    (Date, Code) キーで結合してから database/market/ohlcv/{year}.parquet へ保存する。
    Returns: 保存した行数。
    """
    processed_path = JQUANTS_PROCESSED_DIR / f"daily_bars_{year}.parquet"
    if not processed_path.exists():
        logger.warning("[DATABASE_MIGRATE] processed/%s が存在しない・スキップ", processed_path.name)
        return 0
    processed = pd.read_parquet(processed_path, engine="pyarrow")

    raw_path = JQUANTS_RAW_DIR / f"daily_bars_{year}.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path, engine="pyarrow", columns=["Date", "Code", "UL", "LL", "Va"])
        raw = raw.rename(columns=_RAW_EXTRA_COLUMNS)
        raw["Date"] = pd.to_datetime(raw["Date"])
        processed["Date"] = pd.to_datetime(processed["Date"])
        processed = processed.merge(raw, on=["Date", "Code"], how="left")
    else:
        logger.warning("[DATABASE_MIGRATE] raw/%s が存在しない・UL/LL/TurnoverValueなしで移行", raw_path.name)
        for col in _RAW_EXTRA_COLUMNS.values():
            processed[col] = pd.NA

    save_yearly_parquet(processed, year)
    return len(processed)


def _discover_years() -> list[int]:
    if not JQUANTS_PROCESSED_DIR.exists():
        return []
    years = []
    for path in JQUANTS_PROCESSED_DIR.glob("daily_bars_*.parquet"):
        suffix = path.stem.replace("daily_bars_", "")
        if suffix.isdigit():
            years.append(int(suffix))
    return sorted(years)


def migrate_from_jquants_pipeline() -> dict:
    """
    data/jquants/processed（+raw）→ database/market への一回限りの移行を実行する。
    Returns: {"years_migrated": [...], "ohlcv_rows": int, "companies_rows": int,
              "classifications_rows": int, "universe_rows": int}
    """
    started = datetime.now(_JST)
    ensure_database_market_dirs()

    years = _discover_years()
    total_rows = 0
    for year in years:
        rows = _migrate_ohlcv_year(year)
        total_rows += rows
        logger.info("[DATABASE_MIGRATE] ohlcv year=%d rows=%d", year, rows)

    events = load_universe_events()  # data/jquants/metadata/universe_events.parquet（読み取りのみ）

    source = JQuantsSource()
    listed_snapshot = source.fetch_master("companies")  # 1リクエストのみ

    companies = build_companies_parquet(listed_snapshot, events)
    companies.to_parquet(DATABASE_MASTER_DIR / "companies.parquet", engine="pyarrow", index=False)

    try:
        prime150 = JPXOfficialSource().fetch_jpx_prime150_constituents()
    except Exception as e:  # noqa: BLE001 — 外部CSV取得は fail-open（NULLのまま構築を継続）
        logger.warning("[DATABASE_MIGRATE] JPXプライム150取得失敗（fail-open・NULLのまま続行）: %s", e)
        prime150 = None

    classifications = build_classifications_parquet(listed_snapshot, prime150)
    classifications.to_parquet(DATABASE_MASTER_DIR / "classifications.parquet", engine="pyarrow", index=False)

    universe = build_universe_parquet(events)
    universe.to_parquet(DATABASE_MASTER_DIR / "universe.parquet", engine="pyarrow", index=False)

    topix_path = JQUANTS_PROCESSED_DIR / "topix.parquet"
    topix_df = pd.read_parquet(topix_path, engine="pyarrow").reset_index() if topix_path.exists() else None
    indices = build_indices_parquet(topix_df)
    indices.to_parquet(DATABASE_MASTER_DIR / "indices.parquet", engine="pyarrow", index=False)

    finished = datetime.now(_JST)
    db_metadata.write_metadata(run_record={
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "source": "migrate_from_jquants_pipeline",
        "tables_updated": ["ohlcv", "companies", "classifications", "universe", "indices"],
        "rows_added": total_rows,
        "date_range_from": str(min(years)) if years else None,
        "date_range_to": str(max(years)) if years else None,
        "status": "ok",
    })

    result = {
        "years_migrated": years,
        "ohlcv_rows": total_rows,
        "companies_rows": len(companies),
        "classifications_rows": len(classifications),
        "universe_rows": len(universe),
    }
    logger.info("[DATABASE_MIGRATE] 完了: %s", result)
    return result


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(migrate_from_jquants_pipeline())
