"""
src/jquants/catalog.py
data/jquants/metadata/catalog.json — raw/processed 配下の各parquetのメタ情報を1ファイルに集約する。

目的: Study76以降が「まずディスクを総スキャンする」必要をなくす。同期完了後に毎回上書き生成する
（履歴は持たない・現在の正しい状態のみを表す。実行ごとの履歴は manifest.py が担当）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.jquants.manifest import dataset_hash
from src.paths import JQUANTS_METADATA_DIR, JQUANTS_PROCESSED_DIR, JQUANTS_RAW_DIR

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))
CATALOG_FILE: Path = JQUANTS_METADATA_DIR / "catalog.json"


def _describe_parquet(path: Path) -> dict:
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:
        return {"error": str(e)}

    info: dict = {
        "row_count": int(len(df)),
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime, tz=_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if "Date" in df.columns and not df.empty:
        info["date_min"] = pd.Timestamp(df["Date"].min()).strftime("%Y-%m-%d")
        info["date_max"] = pd.Timestamp(df["Date"].max()).strftime("%Y-%m-%d")
    elif isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        info["date_min"] = df.index.min().strftime("%Y-%m-%d")
        info["date_max"] = df.index.max().strftime("%Y-%m-%d")
    if "Code" in df.columns:
        info["symbol_count"] = int(df["Code"].nunique())
    return info


def _daily_bars_totals() -> tuple[int, int]:
    """
    processed/daily_bars_{year}.parquet（正規化フルスキーマ・全銘柄）のみを対象に、
    総レコード数・銘柄実数（年をまたぐ重複を除いた真のユニーク数）を算出する。
    processed/ 配下には universe.parquet・topix.parquet・{symbol}.parquet（Study76互換）も
    混在するため、それらを含めると二重計上になる→ daily_bars_*.parquet のみに限定する。
    Code列だけを読むため全カラムスキャンより軽い。
    """
    total_rows = 0
    all_codes: set[str] = set()
    for path in sorted(JQUANTS_PROCESSED_DIR.glob("daily_bars_*.parquet")):
        try:
            codes = pd.read_parquet(path, columns=["Code"])["Code"]
        except (OSError, ValueError, KeyError) as e:
            logger.warning("[JQUANTS_CATALOG] %s のCode列読込失敗（スキップ）: %s", path, e)
            continue
        total_rows += len(codes)
        all_codes.update(codes.dropna().astype(str).unique())
    return total_rows, len(all_codes)


def build_catalog() -> dict:
    catalog: dict = {
        "generated_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw": {},
        "processed": {},
    }
    for path in sorted(JQUANTS_RAW_DIR.glob("*.parquet")):
        catalog["raw"][path.name] = _describe_parquet(path)
    for path in sorted(JQUANTS_PROCESSED_DIR.glob("*.parquet")):
        catalog["processed"][path.name] = _describe_parquet(path)

    all_dates = [
        (v.get("date_min"), v.get("date_max"))
        for v in {**catalog["raw"], **catalog["processed"]}.values()
        if v.get("date_min")
    ]
    catalog["coverage_start"] = min((d[0] for d in all_dates), default=None)
    catalog["coverage_end"] = max((d[1] for d in all_dates), default=None)

    total_rows, total_symbols = _daily_bars_totals()
    catalog["total_rows"] = total_rows
    catalog["total_symbols"] = total_symbols
    catalog["dataset_hash"] = dataset_hash()

    return catalog


def save_catalog() -> Path:
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    CATALOG_FILE.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[JQUANTS_CATALOG] 更新: %s", CATALOG_FILE)
    return CATALOG_FILE
