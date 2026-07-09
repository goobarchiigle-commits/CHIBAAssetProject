"""
src/jquants/cache.py
ローカルキャッシュ + Incremental Update 差分判定 + ステージング。

責務分離（2026-07-09改訂・全銘柄年パーティション化に伴う再設計）:
  Parquetは行単位追記ができないため、取得 → ステージング → コンパクション の2段構成にする。
    data/jquants/cache/staging/{symbol}.parquet   銘柄別マージ済み生データ（Dateは列）。
                                                    これ自体が Checkpoint 単位（1銘柄完了=1ファイル確定）。
    data/jquants/metadata/incremental_state.json   銘柄別「取得済み最終日」（差分取得の判定に使用）。
    data/jquants/raw/_audit/{symbol}_{from}_{to}.json  生レスポンスの監査保存（失敗してもfail-open）。
  年パーティション化（raw/daily_bars_{year}.parquet・processed/daily_bars_{year}.parquet）は
  compaction.py がステージングを読み込んで生成する（本モジュールの責務外）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.paths import JQUANTS_CACHE_DIR, JQUANTS_METADATA_DIR, JQUANTS_PROCESSED_DIR, JQUANTS_RAW_DIR, JQUANTS_STAGING_DIR

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
STATE_FILE: Path = JQUANTS_METADATA_DIR / "incremental_state.json"
_AUDIT_DIR: Path = JQUANTS_RAW_DIR / "_audit"


def ensure_dirs() -> None:
    for d in (JQUANTS_CACHE_DIR, JQUANTS_STAGING_DIR, JQUANTS_METADATA_DIR, JQUANTS_PROCESSED_DIR, JQUANTS_RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, dict]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[JQUANTS_CACHE] incremental_state.json 読込失敗（%s）→ 空状態から再構築", e)
        return {}


def save_state(state: dict[str, dict]) -> None:
    ensure_dirs()
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def staging_path(symbol: str) -> Path:
    return JQUANTS_STAGING_DIR / f"{symbol}.parquet"


def compute_fetch_range(
    symbol: str,
    requested_start: str,
    requested_end: str,
    state: dict[str, dict],
) -> tuple[str, str] | None:
    """
    差分取得すべき [start, end] を返す。全区間キャッシュ済みなら None（取得不要）。
    """
    entry = state.get(symbol)
    if not entry or not entry.get("last_date"):
        return (requested_start, requested_end)

    last_date = pd.Timestamp(entry["last_date"])
    req_end = pd.Timestamp(requested_end)
    if last_date >= req_end:
        return None

    next_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return (next_start, requested_end)


def merge_and_store_staging(symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    既存ステージングと new_df（Dateは列・列名リネームなしの生データ）をマージし、
    重複日付は新しい方を優先して保存する。Returns: マージ後の全期間 DataFrame。
    """
    ensure_dirs()
    path = staging_path(symbol)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if "Date" in combined.columns and not combined.empty:
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)

    combined.to_parquet(path)
    return combined


def update_state_entry(state: dict[str, dict], symbol: str, merged_df: pd.DataFrame) -> None:
    if merged_df.empty or "Date" not in merged_df.columns:
        return
    state[symbol] = {
        "last_date":  pd.Timestamp(merged_df["Date"].max()).strftime("%Y-%m-%d"),
        "row_count":  int(len(merged_df)),
        "updated_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def save_raw_response(symbol: str, from_date: str, to_date: str, records: list[dict]) -> Path:
    """生レスポンスを監査目的で保存する（Incremental Updateとは独立・失敗してもfail-openでよい）。"""
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = _AUDIT_DIR / f"{symbol}_{from_date}_{to_date}.json"
    try:
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("[JQUANTS_CACHE] raw response 保存失敗（fail-open）: %s", e)
    return path


def list_staged_symbols() -> list[str]:
    ensure_dirs()
    return sorted(p.stem for p in JQUANTS_STAGING_DIR.glob("*.parquet"))


def load_staged(symbol: str) -> pd.DataFrame:
    path = staging_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
