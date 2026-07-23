"""
src/database/bulk_state.py
database/market/metadata/bulk_ingest_state.parquet の読み書き。

Bulk API（分足/Tick/Fundamentals summary等）取り込みの冪等性・resume・カバレッジ検証・
トレーサビリティ（Parquet→CSV.GZ原本まで1行で追跡）を1テーブルで担う台帳。
1行 = 1(endpoint, period)。period は "YYYYMM"（月次historical）または "YYYYMMDD"（日次live）。

原本CSV.GZは archive/bulk/ に永久保存し、Parquet（database/market/配下）はそこからの
派生データという位置付け（src/jquants/bulk_client.py・archive/README.md参照）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.database.metadata import git_commit
from src.database.schema import DATABASE_SCHEMA_VERSION, JQUANTS_API_VERSION
from src.paths import DATABASE_METADATA_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

BULK_INGEST_STATE_FILE: Path = DATABASE_METADATA_DIR / "bulk_ingest_state.parquet"

BULK_INGEST_STATE_COLUMNS = [
    "endpoint", "period", "source_key", "source_size_bytes", "source_last_modified",
    "csv_gz_local_path", "csv_gz_sha256", "downloaded_at",
    "parquet_local_path", "parquet_row_count", "parquet_size_bytes", "parquet_created_at",
    "symbol_count", "status",
    "database_version", "api_version", "source_provider", "created_with_commit",
]

SOURCE_PROVIDER = "J-Quants"


def load_state() -> pd.DataFrame:
    """bulk_ingest_state.parquet を読み込む。未生成なら空のDataFrame（列のみ）を返す。"""
    if not BULK_INGEST_STATE_FILE.exists():
        return pd.DataFrame(columns=BULK_INGEST_STATE_COLUMNS)
    return pd.read_parquet(BULK_INGEST_STATE_FILE, engine="pyarrow")


def save_state(df: pd.DataFrame) -> None:
    ensure_database_market_dirs()
    df.to_parquet(BULK_INGEST_STATE_FILE, engine="pyarrow", index=False)


def get_row(endpoint: str, period: str) -> dict | None:
    """指定(endpoint, period)の既存状態行を返す（無ければNone）。"""
    state = load_state()
    if state.empty:
        return None
    mask = (state["endpoint"] == endpoint) & (state["period"] == period)
    matched = state.loc[mask]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def is_ingested(endpoint: str, period: str) -> bool:
    """
    冪等性チェック（ネットワーク呼び出し無し・ローカル状態のみで判定）: 指定(endpoint, period)が
    status=okかつ、記録されたCSV.GZ原本・Parquet派生データの両方がローカルに実在すればスキップ可能。
    片方でも手動削除等で欠けていれば False（再取り込みが必要）を返す
    （「揃っているとstateが言っているが実物が無い」を信用しないfail-closed判定）。
    """
    row = get_row(endpoint, period)
    if row is None or row.get("status") != "ok":
        return False
    csv_gz_path = row.get("csv_gz_local_path") or ""
    parquet_path = row.get("parquet_local_path") or ""
    if not csv_gz_path or not parquet_path:
        return False
    return Path(csv_gz_path).exists() and Path(parquet_path).exists()


def build_row(
    *,
    endpoint: str,
    period: str,
    source_key: str,
    source_size_bytes: int,
    source_last_modified: str,
    csv_gz_local_path: str,
    csv_gz_sha256: str,
    downloaded_at: str | None = None,
    parquet_local_path: str = "",
    parquet_row_count: int = 0,
    parquet_size_bytes: int = 0,
    parquet_created_at: str | None = None,
    symbol_count: int = 0,
    status: str = "ok",
) -> dict:
    """トレーサビリティ列（database_version/api_version/source_provider/created_with_commit）を
    自動付与したstate行を構築する。"""
    now = datetime.now(_JST).isoformat()
    return {
        "endpoint": endpoint,
        "period": period,
        "source_key": source_key,
        "source_size_bytes": int(source_size_bytes),
        "source_last_modified": source_last_modified,
        "csv_gz_local_path": csv_gz_local_path,
        "csv_gz_sha256": csv_gz_sha256,
        "downloaded_at": downloaded_at or now,
        "parquet_local_path": parquet_local_path,
        "parquet_row_count": int(parquet_row_count),
        "parquet_size_bytes": int(parquet_size_bytes),
        "parquet_created_at": parquet_created_at or now,
        "symbol_count": int(symbol_count),
        "status": status,
        "database_version": DATABASE_SCHEMA_VERSION,
        "api_version": JQUANTS_API_VERSION,
        "source_provider": SOURCE_PROVIDER,
        "created_with_commit": git_commit(),
    }


def upsert(row: dict) -> None:
    """(endpoint, period)キーで既存行を置き換え、無ければ追加してから保存する。"""
    state = load_state()
    if not state.empty:
        mask = (state["endpoint"] == row["endpoint"]) & (state["period"] == row["period"])
        state = state.loc[~mask]
    state = pd.concat([state, pd.DataFrame([row], columns=BULK_INGEST_STATE_COLUMNS)], ignore_index=True)
    save_state(state)
    logger.info(
        "[BULK_STATE] upsert endpoint=%s period=%s status=%s rows=%s",
        row["endpoint"], row["period"], row["status"], row.get("parquet_row_count"),
    )


def check_coverage(symbol_count: int, expected_universe_codes: set[str], min_ratio: float = 0.5) -> bool:
    """
    取得漏れ検証: 実測symbol_countが期待銘柄数(expected_universe_codes)のmin_ratio未満なら
    Falseを返す（呼び出し側で警告ログを出す想定・fail-openで処理自体は継続する）。
    """
    expected = len(expected_universe_codes)
    if expected == 0:
        return True
    return (symbol_count / expected) >= min_ratio
