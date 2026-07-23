"""
src/database/metadata.py
database/market/metadata/{dataset_info.json, schema.json, update_history.parquet,
database_version.json} の管理。

dataset_info.json: 現在状態のスナップショット（毎回上書き）。
schema.json       : src/database/schema.py の定義から生成（手動同期しない）。
update_history.parquet: 実行ごとの追記専用ログ（既存 src/jquants/manifest.py と同じ設計思想）。
database_version.json : スキーマ/API/圧縮方式のバージョン情報（毎回上書き・created_atのみ初回保持）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.database.schema import DATABASE_SCHEMA_VERSION, JQUANTS_API_VERSION, to_schema_json
from src.paths import (
    DATABASE_MARKET_DIR,
    DATABASE_MASTER_DIR,
    DATABASE_METADATA_DIR,
    DATABASE_OHLCV_DIR,
    ensure_database_market_dirs,
)

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

DATASET_INFO_FILE: Path = DATABASE_METADATA_DIR / "dataset_info.json"
SCHEMA_FILE: Path = DATABASE_METADATA_DIR / "schema.json"
UPDATE_HISTORY_FILE: Path = DATABASE_METADATA_DIR / "update_history.parquet"
DATABASE_VERSION_FILE: Path = DATABASE_METADATA_DIR / "database_version.json"

UPDATE_HISTORY_COLUMNS = [
    "run_id", "started_at", "finished_at", "source", "tables_updated",
    "rows_added", "date_range_from", "date_range_to", "status", "git_commit",
]


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[DATABASE_METADATA] git_commit 取得失敗（fail-open）: %s", e)
        return ""


def dataset_hash() -> str:
    """database/market/ 配下（ohlcv/・master/）の全parquetを対象にしたSHA256（catalog照合用）。"""
    entries = []
    for subdir in (DATABASE_OHLCV_DIR, DATABASE_MASTER_DIR):
        if not subdir.exists():
            continue
        for path in sorted(subdir.glob("*.parquet")):
            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as e:
                logger.warning("[DATABASE_METADATA] %s のハッシュ計算失敗（fail-open）: %s", path, e)
                continue
            entries.append(f"{path.relative_to(DATABASE_MARKET_DIR)}:{file_hash}")
    if not entries:
        return ""
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _describe_parquet(path: Path) -> dict:
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    info: dict = {"row_count": int(len(df))}
    if "Date" in df.columns and not df.empty:
        info["date_min"] = pd.Timestamp(df["Date"].min()).strftime("%Y-%m-%d")
        info["date_max"] = pd.Timestamp(df["Date"].max()).strftime("%Y-%m-%d")
    if "Code" in df.columns:
        info["symbol_count"] = int(df["Code"].nunique())
    return info


def build_dataset_info() -> dict:
    """現在の database/market/ 状態を表す dataset_info.json 相当の辞書を構築する。"""
    ohlcv_tables: dict[str, dict] = {}
    if DATABASE_OHLCV_DIR.exists():
        for path in sorted(DATABASE_OHLCV_DIR.glob("*.parquet")):
            ohlcv_tables[path.name] = _describe_parquet(path)

    master_tables: dict[str, dict] = {}
    if DATABASE_MASTER_DIR.exists():
        for path in sorted(DATABASE_MASTER_DIR.glob("*.parquet")):
            master_tables[path.name] = _describe_parquet(path)

    all_dates = [(v.get("date_min"), v.get("date_max")) for v in ohlcv_tables.values() if v.get("date_min")]
    total_rows = sum(v.get("row_count", 0) for v in ohlcv_tables.values())

    return {
        "last_updated": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": [
            {"name": "jquants_api", "version": "v2"},
            {"name": "jpx_official_csv", "version": ""},
        ],
        "coverage_start": min((d[0] for d in all_dates), default=None),
        "coverage_end": max((d[1] for d in all_dates), default=None),
        "ohlcv": {"tables": ohlcv_tables, "total_rows": total_rows},
        "master": {"tables": master_tables},
        "dataset_hash": dataset_hash(),
        "git_commit": git_commit(),
    }


def write_metadata(run_record: dict | None = None) -> dict:
    """
    schema.json・dataset_info.json を最新状態へ上書きし、run_record が渡されれば
    update_history.parquet に1行追記する。Returns: 書き込んだ dataset_info。
    """
    ensure_database_market_dirs()
    SCHEMA_FILE.write_text(json.dumps(to_schema_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_database_version()

    info = build_dataset_info()
    DATASET_INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[DATABASE_METADATA] dataset_info.json 更新: rows=%d", info["ohlcv"]["total_rows"])

    if run_record is not None:
        append_update_history(run_record)
    return info


def read_database_version() -> dict:
    """database_version.json を読み込む。未生成なら空辞書を返す。"""
    if not DATABASE_VERSION_FILE.exists():
        return {}
    try:
        return json.loads(DATABASE_VERSION_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_database_version(*, compression: str = "zstd (minute/tick) + snappy (他)") -> dict:
    """
    database_version.json を最新状態へ上書きする。created_atは初回生成時刻を維持し、
    updated_atのみ毎回更新する（将来ファイル移動・圧縮方式変更・再開時の基準点として使う）。
    """
    ensure_database_market_dirs()
    existing = read_database_version()
    now = datetime.now(_JST).isoformat()
    info = {
        "schema_version": DATABASE_SCHEMA_VERSION,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "jquants_api_version": JQUANTS_API_VERSION,
        "compression": compression,
        "python": platform.python_version(),
        "git_commit": git_commit(),
    }
    DATABASE_VERSION_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[DATABASE_METADATA] database_version.json 更新: schema_version=%s", info["schema_version"])
    return info


def read_metadata() -> dict:
    """dataset_info.json を読み込む。未生成なら空辞書を返す。"""
    if not DATASET_INFO_FILE.exists():
        return {}
    try:
        return json.loads(DATASET_INFO_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[DATABASE_METADATA] dataset_info.json 読込失敗（%s）", e)
        return {}


def load_update_history() -> pd.DataFrame:
    if not UPDATE_HISTORY_FILE.exists():
        return pd.DataFrame(columns=UPDATE_HISTORY_COLUMNS)
    return pd.read_parquet(UPDATE_HISTORY_FILE)


def append_update_history(record: dict) -> None:
    """update_history.parquet に1行追記する（追記専用ランレコード）。"""
    ensure_database_market_dirs()
    row = {
        "run_id": record.get("run_id") or str(uuid.uuid4()),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "source": record.get("source", ""),
        "tables_updated": record.get("tables_updated", []),
        "rows_added": int(record.get("rows_added", 0)),
        "date_range_from": record.get("date_range_from"),
        "date_range_to": record.get("date_range_to"),
        "status": record.get("status", "ok"),
        "git_commit": git_commit(),
    }
    existing = load_update_history()
    combined = pd.concat([existing, pd.DataFrame([row], columns=UPDATE_HISTORY_COLUMNS)], ignore_index=True)
    combined.to_parquet(UPDATE_HISTORY_FILE, engine="pyarrow", index=False)
    logger.info("[DATABASE_METADATA] update_history.parquet 追記: run_id=%s status=%s", row["run_id"], row["status"])
