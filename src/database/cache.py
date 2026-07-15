"""
src/database/cache.py
database/market/cache/ 専用の分析キャッシュI/O。

既存の cache/（売買システム専用）・data/jquants/cache/（取り込みステージング）とは完全に別物で、
このモジュールだけが database/market/cache/ に書き込む（役割混在を避けるための唯一の書き込み口）。
セクター集計・RS計算等、分析側が繰り返し使う派生DataFrameの保存を想定する。
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src.paths import DATABASE_CACHE_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_SAFE_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_\-]+")


def _key_to_path(key: str):
    ensure_database_market_dirs()
    safe_key = _SAFE_KEY_PATTERN.sub("_", key)
    return DATABASE_CACHE_DIR / f"{safe_key}.parquet"


def cache_put(key: str, df: pd.DataFrame) -> None:
    """派生DataFrameを database/market/cache/{key}.parquet として保存する。"""
    path = _key_to_path(key)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.info("[DATABASE_CACHE] put key=%s rows=%d path=%s", key, len(df), path)


def cache_get(key: str) -> pd.DataFrame | None:
    """key に対応するキャッシュを読み込む。存在しなければ None を返す。"""
    path = _key_to_path(key)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


def cache_clear(key: str) -> None:
    """key に対応するキャッシュファイルを削除する（存在しなければ何もしない）。"""
    path = _key_to_path(key)
    if path.exists():
        path.unlink()
        logger.info("[DATABASE_CACHE] cleared key=%s", key)
