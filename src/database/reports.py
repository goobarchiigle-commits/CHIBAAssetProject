"""
src/database/reports.py
database/market/metadata/{source_catalog,coverage_report,data_quality}.json の生成。

metadata/ の既存3ファイル（dataset_info.json/schema.json/update_history.parquet）に加え、
「何を取得可能で何が欠けているか」を一目で把握できるようにするための3レポートを追加する。

source_catalog.json : J-Quants全エンドポイントの棚卸し（プラン要件・実装状況・実測確認済み）。
                       2026-07-23実測ベースの静的カタログ（エンドポイント新設時は手動更新）。
coverage_report.json : 現在ローカルに存在するデータの実測カバレッジ（期間・行数・銘柄数）。
data_quality.json    : 簡易品質チェック（営業日カバレッジ欠落・銘柄カバレッジ異常の検出）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.database import bulk_state
from src.database.ohlcv import available_years as ohlcv_available_years
from src.database.ohlcv import load_yearly_parquet as load_ohlcv_year
from src.market.jpx_calendar import JPXCalendar
from src.paths import (
    DATABASE_ETF_DIR,
    DATABASE_FUNDAMENTALS_DIR,
    DATABASE_MARGIN_DIR,
    DATABASE_MASTER_DIR,
    DATABASE_METADATA_DIR,
    DATABASE_SHORTSELLING_DIR,
    ensure_database_market_dirs,
)

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

SOURCE_CATALOG_FILE: Path = DATABASE_METADATA_DIR / "source_catalog.json"
COVERAGE_REPORT_FILE: Path = DATABASE_METADATA_DIR / "coverage_report.json"
DATA_QUALITY_FILE: Path = DATABASE_METADATA_DIR / "data_quality.json"

# ── source_catalog: 2026-07-23実測確認ベース（公式 https://jpx-jquants.com/ja/spec/data-spec 準拠） ──
SOURCE_CATALOG: list[dict] = [
    {"endpoint": "/equities/bars/daily", "name": "株価四本値", "plan": "Free+", "status": "implemented", "local_path": "ohlcv/"},
    {"endpoint": "/equities/master", "name": "上場銘柄一覧", "plan": "Free+", "status": "implemented", "local_path": "master/companies.parquet"},
    {"endpoint": "/equities/bars/minute (add-on)", "name": "株価分足", "plan": "Add-on(契約済)", "status": "implemented", "local_path": "minute/"},
    {"endpoint": "/equities/trades (add-on, bulk-only)", "name": "株価ティック", "plan": "Add-on(契約済)", "status": "implemented", "local_path": "tick/"},
    {"endpoint": "/fins/summary", "name": "財務情報サマリ", "plan": "Free+", "status": "implemented", "local_path": "fundamentals/summary/"},
    {"endpoint": "/fins/details (statements)", "name": "財務諸表(BS/PL/CF)", "plan": "Premium限定", "status": "premium_locked", "local_path": "fundamentals/statements/（空）"},
    {"endpoint": "/fins/dividend", "name": "配当金情報", "plan": "Premium限定", "status": "premium_locked", "local_path": "fundamentals/dividend/（空）"},
    {"endpoint": "/equities/bars/daily/am", "name": "前場四本値", "plan": "Premium限定・直近のみ", "status": "premium_locked", "local_path": "-"},
    {"endpoint": "/equities/earnings-calendar", "name": "決算発表予定日", "plan": "Free+・直近のみ", "status": "not_implemented", "local_path": "-"},
    {"endpoint": "/markets/calendar", "name": "取引カレンダー", "plan": "Free+", "status": "not_implemented", "local_path": "-（src/market/jpx_calendar.pyで代替）"},
    {"endpoint": "/equities/investor-types", "name": "投資部門別情報", "plan": "Light+", "status": "not_implemented", "local_path": "-"},
    {"endpoint": "/indices/bars/daily?code=0000", "name": "TOPIX四本値", "plan": "Light+", "status": "implemented", "local_path": "index/prices/0000.parquet"},
    {"endpoint": "/indices/bars/daily?code=0080-0090", "name": "TOPIX-17業種別指数", "plan": "Standard+", "status": "implemented", "local_path": "index/prices/0080-0090.parquet"},
    {"endpoint": "/indices/bars/daily?code=0070,0075,0500-0504,B507", "name": "Growth250/REIT/市場区分別/JPXプライム150等", "plan": "Standard+", "status": "implemented", "local_path": "index/prices/"},
    {"endpoint": "日経225（Nikkei Inc独自指数）", "name": "日経225", "plan": "N/A", "status": "not_available", "local_path": "- J-Quantsで提供されていない（別ライセンス要）"},
    {"endpoint": "/derivatives/bars/daily/options/225", "name": "日経225オプション四本値", "plan": "Standard+", "status": "not_implemented", "local_path": "-"},
    {"endpoint": "/derivatives/bars/daily/futures,/options", "name": "先物/オプション四本値(全般)", "plan": "Premium限定", "status": "premium_locked", "local_path": "-"},
    {"endpoint": "/markets/margin-interest", "name": "信用取引週末残高", "plan": "Standard+", "status": "implemented", "local_path": "margin/margin_interest/"},
    {"endpoint": "/markets/margin-alert", "name": "日々公表信用取引残高", "plan": "Standard+", "status": "implemented", "local_path": "margin/margin_alert/"},
    {"endpoint": "/markets/short-ratio", "name": "業種別空売り比率", "plan": "Standard+", "status": "implemented", "local_path": "shortselling/short_ratio/"},
    {"endpoint": "/markets/short-sale-report", "name": "空売り残高報告", "plan": "Standard+", "status": "implemented", "local_path": "shortselling/short_sale_report/"},
    {"endpoint": "/markets/breakdown", "name": "売買内訳データ", "plan": "Premium限定", "status": "premium_locked", "local_path": "-"},
    {"endpoint": "/edinet/major-shareholders,cross-shareholdings,large-volume-shareholders", "name": "EDINET系(大株主/政策保有/大量保有報告)", "plan": "Standard+", "status": "not_implemented", "local_path": "-"},
    {"endpoint": "ETF価格（equities/bars/dailyの一部）", "name": "ETF価格", "plan": "Free+（equitiesに包含）", "status": "implemented", "local_path": "etf/prices/"},
    {"endpoint": "ETF構成銘柄（J-Quants未提供）", "name": "ETF構成銘柄", "plan": "N/A", "status": "not_available", "local_path": "- J-Quantsで提供されていない（運用会社PCF等が必要）"},
    {"endpoint": "/td/list,/td/files,/td/bulk", "name": "TDnet適時開示", "plan": "別Add-on（未契約）", "status": "not_contracted", "local_path": "-"},
]


def build_source_catalog() -> list[dict]:
    return SOURCE_CATALOG


def _describe_partitioned_dir(base_dir: Path, date_col_candidates: tuple[str, ...] = ("Date",)) -> dict:
    """base_dir配下の全*.parquet（再帰）を対象に行数・期間・銘柄数を集計する。"""
    if not base_dir.exists():
        return {"exists": False}
    files = sorted(base_dir.rglob("*.parquet"))
    if not files:
        return {"exists": True, "file_count": 0}
    total_rows = 0
    date_min, date_max = None, None
    symbol_count = None
    for f in files:
        try:
            df = pd.read_parquet(f)
        except (OSError, ValueError):
            continue
        total_rows += len(df)
        date_col = next((c for c in date_col_candidates if c in df.columns), None)
        if date_col and not df.empty:
            dmin, dmax = df[date_col].min(), df[date_col].max()
            date_min = dmin if date_min is None else min(date_min, dmin)
            date_max = dmax if date_max is None else max(date_max, dmax)
        if "Code" in df.columns:
            symbol_count = max(symbol_count or 0, df["Code"].nunique())
    return {
        "exists": True,
        "file_count": len(files),
        "total_rows": int(total_rows),
        "date_min": str(date_min) if date_min is not None else None,
        "date_max": str(date_max) if date_max is not None else None,
        "symbol_count": int(symbol_count) if symbol_count is not None else None,
    }


def _describe_from_bulk_state(endpoint: str) -> dict:
    """
    minute/tickのような大容量ドメイン（合計27GB超）はファイルを毎回全件読み直すと非常に遅い。
    ingest時に既に実測済みの bulk_ingest_state.parquet（parquet_row_count/symbol_count）を
    集計元にする（再スキャンしない）。
    """
    state = bulk_state.load_state()
    if state.empty:
        return {"exists": False}
    sub = state[(state["endpoint"] == endpoint) & (state["status"] == "ok")]
    if sub.empty:
        return {"exists": True, "file_count": 0}
    return {
        "exists": True,
        "file_count": int(sub["parquet_local_path"].nunique()),
        "total_rows": int(sub["parquet_row_count"].sum()),
        "period_min": str(sub["period"].min()),
        "period_max": str(sub["period"].max()),
        "symbol_count_max": int(sub["symbol_count"].max()),
        "source": "bulk_ingest_state.parquet（集計・ファイル再読込みなし）",
    }


def build_coverage_report() -> dict:
    """現在ローカルに存在する主要ドメインの実測カバレッジを集計する。"""
    from src.jquants.bulk_client import ENDPOINT_EQUITIES_BARS_MINUTE, ENDPOINT_EQUITIES_TRADES

    return {
        "generated_at": datetime.now(_JST).isoformat(),
        "ohlcv": _describe_partitioned_dir(DATABASE_MASTER_DIR.parent / "ohlcv"),
        "minute": _describe_from_bulk_state(ENDPOINT_EQUITIES_BARS_MINUTE),
        "tick": _describe_from_bulk_state(ENDPOINT_EQUITIES_TRADES),
        "fundamentals_summary": _describe_partitioned_dir(DATABASE_FUNDAMENTALS_DIR / "summary", ("DiscDate",)),
        "margin_interest": _describe_partitioned_dir(DATABASE_MARGIN_DIR / "margin_interest"),
        "margin_alert": _describe_partitioned_dir(DATABASE_MARGIN_DIR / "margin_alert", ("PubDate",)),
        "short_ratio": _describe_partitioned_dir(DATABASE_SHORTSELLING_DIR / "short_ratio"),
        "short_sale_report": _describe_partitioned_dir(DATABASE_SHORTSELLING_DIR / "short_sale_report", ("DiscDate",)),
        "etf_prices": _describe_partitioned_dir(DATABASE_ETF_DIR / "prices"),
        "index_prices": _describe_partitioned_dir(DATABASE_MASTER_DIR.parent / "index" / "prices"),
    }


def _missing_trading_days_for_ohlcv() -> dict:
    """ohlcv/ の営業日カバレッジ欠落を検出する（JPXCalendar基準の全営業日 vs 実収録日）。"""
    years = ohlcv_available_years()
    if not years:
        return {"checked": False}
    calendar = JPXCalendar()
    missing_by_year: dict[str, list[str]] = {}
    for year in years:
        df = load_ohlcv_year(year)
        if df.empty:
            continue
        present_dates = set(pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"))
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        all_days = [d for d in pd.date_range(start, end, freq="D") if calendar.is_trading_day(d)]
        missing = [d.strftime("%Y-%m-%d") for d in all_days if d.strftime("%Y-%m-%d") not in present_dates]
        # データセット収録開始日より前・当年最終収録日より後は「対象外/未到来」であり欠落ではない
        if present_dates:
            first_present, last_present = min(present_dates), max(present_dates)
            missing = [d for d in missing if first_present <= d <= last_present]
        if missing:
            missing_by_year[str(year)] = missing
    return {"checked": True, "missing_trading_days_by_year": missing_by_year}


def build_data_quality_report() -> dict:
    """簡易品質チェック（営業日欠落検出）。fail-openでレポートするのみ（例外化しない）。"""
    return {
        "generated_at": datetime.now(_JST).isoformat(),
        "ohlcv_trading_day_coverage": _missing_trading_days_for_ohlcv(),
        "bulk_ingest_partial_periods": (
            bulk_state.load_state().loc[lambda d: d["status"] != "ok", ["endpoint", "period", "status"]]
            .to_dict(orient="records")
            if not bulk_state.load_state().empty else []
        ),
    }


def write_all_reports() -> None:
    ensure_database_market_dirs()
    SOURCE_CATALOG_FILE.write_text(
        json.dumps(build_source_catalog(), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    COVERAGE_REPORT_FILE.write_text(
        json.dumps(build_coverage_report(), ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    DATA_QUALITY_FILE.write_text(
        json.dumps(build_data_quality_report(), ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    logger.info("[REPORTS] source_catalog.json / coverage_report.json / data_quality.json 更新完了")


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_all_reports()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
