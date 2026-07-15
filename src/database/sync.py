"""
src/database/sync.py
database/market/ の日次更新エントリポイント（CLI）。

移行完了後、この経路が database/market の唯一の更新経路になる。data/jquants/（processed・
cache/daily含む）へは一切書き込まない（plan「確定方針8」）。既存の data/jquants 日次cron
（Study75/76用・study75_downloader.py ベース）は本モジュールと無関係に独立稼働を継続する
（無効化・変更はしない）。

日次実行想定: run_morning_signal.py 等と同様、Task Schedulerからの毎日実行に耐える
（未取得営業日のみ処理・当年Parquetのみ更新・冪等）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.database import metadata as db_metadata
from src.database.master import build_classifications_parquet, build_companies_parquet
from src.database.ohlcv import split_by_year, trading_days_in_range, update_current_year
from src.database.sources.jpx_official import JPXOfficialSource
from src.database.sources.jquants_source import JQuantsSource
from src.paths import DATABASE_MASTER_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))
_UNIVERSE_NAME = "TSE_ALL"


def authenticate() -> bool:
    """
    J-Quants API疎通確認。v2は静的x-api-keyのため明示的ログインはなく、設定妥当性検証で代替する。
    """
    source = JQuantsSource()
    ok = source.authenticate()
    if ok:
        logger.info("[DATABASE_SYNC] authenticate: OK")
    else:
        logger.error("[DATABASE_SYNC] authenticate: 失敗（JQUANTS_API_KEY未設定）")
    return ok


def _missing_trading_days() -> list[pd.Timestamp]:
    info = db_metadata.read_metadata()
    coverage_end = info.get("coverage_end")
    today = datetime.now(_JST).date()
    if not coverage_end:
        logger.warning("[DATABASE_SYNC] dataset_info.json未生成・coverage_end不明のため更新対象なし"
                        "（初回は migrate.py を先に実行すること）")
        return []
    start = (pd.Timestamp(coverage_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    if pd.Timestamp(start) > pd.Timestamp(end):
        return []
    return trading_days_in_range(start, end)


def _update_universe_for_day(day: pd.Timestamp, day_codes: set[str]) -> None:
    """
    universe.parquet（database/market自身が持つイベント状態）を当日のCode集合と突き合わせ、
    新規出現→区間オープン、消失→区間クローズ、で更新する（既存 src.jquants.universe の
    day-over-day diffing と同じ考え方をdatabase/market自身の状態に対して適用する）。
    """
    path = DATABASE_MASTER_DIR / "universe.parquet"
    universe = pd.read_parquet(path, engine="pyarrow") if path.exists() else pd.DataFrame(
        columns=["UniverseName", "Code", "FromDate", "ToDate", "IsTradable", "IsDelisted"]
    )
    mask_open = (universe.get("UniverseName") == _UNIVERSE_NAME) & (universe.get("ToDate").isna() if "ToDate" in universe.columns else False)
    open_codes = set(universe.loc[mask_open, "Code"].astype(str)) if not universe.empty else set()

    new_codes = day_codes - open_codes
    removed_codes = open_codes - day_codes

    if new_codes:
        new_rows = pd.DataFrame({
            "UniverseName": _UNIVERSE_NAME, "Code": sorted(new_codes),
            "FromDate": day, "ToDate": pd.NaT, "IsTradable": True, "IsDelisted": False,
        })
        universe = pd.concat([universe, new_rows], ignore_index=True)

    if removed_codes:
        close_mask = mask_open & universe["Code"].astype(str).isin(removed_codes)
        universe.loc[close_mask, "ToDate"] = day
        universe.loc[close_mask, "IsDelisted"] = True

    from src.database.dtypes import optimize_dtypes
    from src.database.schema import validate_schema
    universe = optimize_dtypes(universe, "universe")
    validate_schema(universe, "universe")
    universe.to_parquet(path, engine="pyarrow", index=False)
    logger.info(
        "[DATABASE_SYNC] universe updated day=%s new=%d removed=%d",
        day.strftime("%Y-%m-%d"), len(new_codes), len(removed_codes),
    )


def _refresh_master_snapshot(source: JQuantsSource) -> None:
    """
    companies/classifications はコード単位の低頻度変化データのため、真の差分更新ではなく
    毎回スナップショット全体を再取得・再構築する（1 listed_info リクエスト + 1 CSVで足りるため
    差分ロジックを持つコストに見合わない・意図的な単純化）。
    """
    listed_snapshot = source.fetch_master("companies")
    events_path = DATABASE_MASTER_DIR / "universe.parquet"
    universe = pd.read_parquet(events_path, engine="pyarrow") if events_path.exists() else pd.DataFrame()
    # companies.parquet の ListingDate/DelistingDate は universe.parquet（区間表現）から再構成する。
    if not universe.empty:
        events = pd.DataFrame({
            "event_date": universe["FromDate"], "code": universe["Code"], "event_type": "ADD",
        })
        closed = universe.loc[universe["ToDate"].notna()]
        if not closed.empty:
            events = pd.concat([events, pd.DataFrame({
                "event_date": closed["ToDate"], "code": closed["Code"], "event_type": "REMOVE",
            })], ignore_index=True)
    else:
        events = pd.DataFrame(columns=["event_date", "code", "event_type"])

    companies = build_companies_parquet(listed_snapshot, events)
    companies.to_parquet(DATABASE_MASTER_DIR / "companies.parquet", engine="pyarrow", index=False)

    try:
        prime150 = JPXOfficialSource().fetch_jpx_prime150_constituents()
    except Exception as e:  # noqa: BLE001 — 外部CSVはfail-open
        logger.warning("[DATABASE_SYNC] JPXプライム150取得失敗（fail-open）: %s", e)
        prime150 = None
    classifications = build_classifications_parquet(listed_snapshot, prime150)
    classifications.to_parquet(DATABASE_MASTER_DIR / "classifications.parquet", engine="pyarrow", index=False)


def update_database() -> dict:
    """
    未取得営業日のみを取得し、database/market/ohlcv/{current_year}.parquet・master/*・metadataを
    更新する。data/jquants/ には一切書き込まない。毎日実行して冪等（既取得日は再取得しない）。
    """
    started = datetime.now(_JST)
    ensure_database_market_dirs()
    if not authenticate():
        return {"status": "failed", "reason": "authenticate_failed"}

    days = _missing_trading_days()
    if not days:
        logger.info("[DATABASE_SYNC] 未取得営業日なし（最新）")
        return {"status": "ok", "days_fetched": 0, "rows_added": 0}

    source = JQuantsSource()
    all_new_codes: set[str] = set()
    rows_added = 0
    for day in days:
        day_str = day.strftime("%Y%m%d")
        df = source.fetch_ohlcv_for_date(day_str)
        if df.empty:
            logger.warning("[DATABASE_SYNC] day=%s 応答が空・スキップ", day_str)
            continue
        for year, year_df in split_by_year(df).items():
            update_current_year(year_df, year=year)
        day_codes = set(df["Code"].astype(str))
        _update_universe_for_day(day, day_codes)
        all_new_codes |= day_codes
        rows_added += len(df)

    _refresh_master_snapshot(source)

    finished = datetime.now(_JST)
    db_metadata.write_metadata(run_record={
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "source": "jquants_api",
        "tables_updated": ["ohlcv", "companies", "classifications", "universe"],
        "rows_added": rows_added,
        "date_range_from": days[0].strftime("%Y-%m-%d") if days else None,
        "date_range_to": days[-1].strftime("%Y-%m-%d") if days else None,
        "status": "ok",
    })
    result = {"status": "ok", "days_fetched": len(days), "rows_added": rows_added}
    logger.info("[DATABASE_SYNC] 完了: %s", result)
    return result


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(update_database())
