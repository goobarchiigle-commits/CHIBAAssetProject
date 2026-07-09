"""
src/jquants/study75_downloader.py
Study75専用ダウンロードエンジン — Strategy C（日次イテレーション）。

Download Strategy Validation（2026-07-09）の結論に基づく: 銘柄別に1リクエストずつ取得するより
「1営業日=1リクエスト（全銘柄まとめて返る）」の方がリクエスト数が少なく
（約2,442営業日 vs 約4,400-5,000銘柄）、差分更新も「新しい営業日だけ叩く」で済み運用が単純。

既存の銘柄別エンジン（src/jquants/downloader.py: JQuantsDownloader.sync_full_market）は
後方互換のためそのまま維持する。本モジュールは完全に独立した並行実装で、互いに干渉しない
（ステージング領域・マニフェストが別）。出力先（raw/processed の daily_bars_{year}.parquet）と
スキーマは共通のため、Study76互換Adapter（study75_adapter.py）は無改修で両エンジンの成果物を
同じように読める。

ステージング: data/jquants/cache/daily/day_YYYY-MM-DD.parquet（1営業日=1ファイル=1チェックポイント）
マニフェスト: data/jquants/metadata/daily_completed_dates.json（追記専用・完了済み営業日を記録）
             {"2016-07-11": {"row_count": 4437, "parquet_file_size": 123456,
                              "sha256": "...", "completed_at": "..."}, ...}
             row_count/parquet_file_size/sha256 は保存直後のステージングファイル実体から算出する
             （verify.py がこの値と実ファイルを突き合わせて破損・欠落・不整合を検出する）。
再開: マニフェストに記録済みの日はスキップする。中断しても次回実行時に続きから再開する
     （「完了」は保存成功後にのみ記録するため、途中で落ちた日は未完了のまま残り再取得される）。
コンパクション: compaction.compact_years_from_daily() で影響を受けた年のみ再構築する。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.jquants.compaction import compact_years_from_daily
from src.jquants.provider import JQuantsProvider, raw_records_to_frame
from src.market.jpx_calendar import JPXCalendar
from src.paths import JQUANTS_DAILY_STAGING_DIR, JQUANTS_METADATA_DIR

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

DAILY_MANIFEST_FILE: Path = JQUANTS_METADATA_DIR / "daily_completed_dates.json"

# ステージング検証（validate_staged_day）の基準値。
REQUIRED_DAILY_COLUMNS = ["Date", "Code"]
LOW_ROW_COUNT_THRESHOLD = 100  # 通常は約4,400件/日（全銘柄）。これを大きく下回る場合は要確認扱い。


def ensure_dirs() -> None:
    JQUANTS_DAILY_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)


def daily_staging_path(date_str: str) -> Path:
    return JQUANTS_DAILY_STAGING_DIR / f"day_{date_str}.parquet"


# ────────────────────────────────────────────────────────────────────────── #
# 完了済み営業日マニフェスト（追記専用）
# ────────────────────────────────────────────────────────────────────────── #
def load_completed_dates() -> dict[str, dict]:
    if not DAILY_MANIFEST_FILE.exists():
        return {}
    try:
        return json.loads(DAILY_MANIFEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[STUDY75_DOWNLOADER] daily_completed_dates.json 読込失敗（%s）→ 空状態から再開", e)
        return {}


def _save_completed_dates(completed: dict[str, dict]) -> None:
    ensure_dirs()
    DAILY_MANIFEST_FILE.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_completed_date(date_str: str, row_count: int) -> None:
    """
    追記専用: 完了済み営業日を1件記録する（既存日付があれば上書き・削除はしない）。
    parquet_file_size/sha256 は daily_staging_path(date_str) の実ファイルから算出する
    （呼び出し時点でステージング保存が完了している前提。ファイルが無ければ0/空文字を記録する）。
    """
    completed = load_completed_dates()
    path = daily_staging_path(date_str)
    file_size = path.stat().st_size if path.exists() else 0
    file_hash = sha256_of_file(path) if path.exists() else ""
    completed[date_str] = {
        "row_count": int(row_count),
        "parquet_file_size": int(file_size),
        "sha256": file_hash,
        "completed_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _save_completed_dates(completed)


def validate_staged_day(df: pd.DataFrame, date_str: str) -> dict:
    """
    ステージング直後の1営業日データを検証する（例外化せず警告/エラーとしてレポートに記録するのみ）。
    - rows > 0（営業日カレンダーで既に休場日を除外しているため、0行は異常として扱う）
    - 必須列（Date/Code）の存在
    - 行数が LOW_ROW_COUNT_THRESHOLD 未満（通常約4,400件/日）の場合は警告

    Returns: {"date":, "row_count":, "warnings": [...], "errors": [...], "status": "ok"|"warning"|"error"}
    status=="error" の日は呼び出し側（download_full_market_by_day）が完了マークしない
    （次回実行で自動的に再取得対象になる）。
    """
    row_count = len(df)
    warnings: list[str] = []
    errors: list[str] = []

    if row_count == 0:
        errors.append("zero_rows")

    missing_cols = [c for c in REQUIRED_DAILY_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing_required_columns:{','.join(missing_cols)}")

    if 0 < row_count < LOW_ROW_COUNT_THRESHOLD:
        warnings.append(f"suspiciously_low_row_count:{row_count}")

    status = "error" if errors else ("warning" if warnings else "ok")
    return {
        "date": date_str, "row_count": row_count,
        "warnings": warnings, "errors": errors, "status": status,
    }


# ────────────────────────────────────────────────────────────────────────── #
# 1営業日の取得・ステージング
# ────────────────────────────────────────────────────────────────────────── #
def fetch_and_stage_day(provider: JQuantsProvider, date_str: str) -> pd.DataFrame:
    """
    1営業日ぶんの全銘柄データを1リクエストで取得しステージングに保存する。
    date_str は "YYYY-MM-DD"（ファイル名・マニフェストキーの形式）。API呼び出しは
    確認済みフォーマット "YYYYMMDD" に変換する（provider.daily_bars_for_date参照）。
    Returns: 保存したDataFrame（休場日等で空の可能性もある・呼び出し側でも空を許容する）。
    """
    ensure_dirs()
    api_date = date_str.replace("-", "")
    records = provider.daily_bars_for_date(api_date)
    df = raw_records_to_frame(records)
    df.to_parquet(daily_staging_path(date_str))
    return df


def download_full_market_by_day(
    provider: JQuantsProvider,
    start: str,
    end: str,
    force_full: bool = False,
) -> dict:
    """
    start〜end の全営業日を1日1リクエストで取得する（Strategy C）。
    完了済みマニフェストに記録済みの日はスキップする（再開対応）。
    force_full=True で完了済みマニフェストを無視し全営業日を再取得する。

    取得失敗（JQuantsError等）は呼び出し側へそのまま伝播する（fail-closed）。
    途中まで完了した日はマニフェストに残るため、再実行すれば失敗した日から再開する。
    取得はできたがvalidate_staged_day()がerror判定した日（0行・必須列欠落）は完了マークしない
    （次回実行で自動的に再取得対象になる。warning判定は完了マークした上でissuesに記録するのみ）。

    Returns: {"days_processed": int, "days_skipped": int, "days_failed_validation": int,
              "total_rows": int, "years_touched": [...], "compaction": {...},
              "validation_issues": [...]}
    """
    ensure_dirs()
    calendar = JPXCalendar()
    trading_days = [d for d in pd.bdate_range(start, end) if calendar.is_trading_day(d)]

    completed = {} if force_full else load_completed_dates()
    years_touched: set[int] = set()
    days_processed = 0
    days_skipped = 0
    days_failed_validation = 0
    total_rows = 0
    validation_issues: list[dict] = []

    for i, day in enumerate(trading_days, start=1):
        date_str = day.strftime("%Y-%m-%d")
        if not force_full and date_str in completed:
            days_skipped += 1
            continue

        df = fetch_and_stage_day(provider, date_str)
        validation = validate_staged_day(df, date_str)

        if validation["status"] == "error":
            days_failed_validation += 1
            validation_issues.append(validation)
            logger.error(
                "[STUDY75_DOWNLOADER] date=%s 検証エラー=%s → 完了マークせず次回再取得対象とする",
                date_str, validation["errors"],
            )
            continue

        if validation["status"] == "warning":
            validation_issues.append(validation)
            logger.warning("[STUDY75_DOWNLOADER] date=%s 警告=%s", date_str, validation["warnings"])

        append_completed_date(date_str, len(df))
        years_touched.add(int(day.year))
        days_processed += 1
        total_rows += len(df)
        logger.info(
            "[STUDY75_DOWNLOADER] progress=%d/%d date=%s rows=%d",
            i, len(trading_days), date_str, len(df),
        )

    compaction_result = compact_years_from_daily(years_touched) if years_touched else {"years": [], "row_counts": {}}

    return {
        "days_processed": days_processed,
        "days_skipped": days_skipped,
        "days_failed_validation": days_failed_validation,
        "total_rows": total_rows,
        "years_touched": sorted(years_touched),
        "compaction": compaction_result,
        "validation_issues": validation_issues,
    }
