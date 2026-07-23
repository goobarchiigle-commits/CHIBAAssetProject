"""
src/database/bulk_catalog.py
J-Quants Bulk API（/v2/bulk/list）の生応答を database/market/metadata/bulk_catalog/{date}.json
へ保存する。

目的: 「解約時点でJ-Quantsが何を提供していたか」の一次証拠を残す（ユーザー要望・2026-07-23、
J-Quants解約前の最終作業）。src/database/reports.py の SOURCE_CATALOG は人手でまとめた
サマリだが、本モジュールはBulk API自体の生応答（エンドポイント・Key命名・ファイルサイズ）を
そのまま保存する一次資料。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.jquants.bulk_client import JQuantsBulkClient
from src.paths import DATABASE_METADATA_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

BULK_CATALOG_DIR = DATABASE_METADATA_DIR / "bulk_catalog"


def save_bulk_catalog(date: str | None = None) -> Path:
    """
    指定日（省略時は当日）の bulk/list 生応答を保存する。
    Returns: 保存したファイルパス。
    """
    ensure_database_market_dirs()
    BULK_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now(_JST).strftime("%Y%m%d")

    client = JQuantsBulkClient()
    files = client.list_files(date)
    payload = {
        "queried_date": date,
        "retrieved_at": datetime.now(_JST).isoformat(),
        "endpoint": "/v2/bulk/list",
        "file_count": len(files),
        "files": [
            {"key": f.key, "size_bytes": f.size_bytes, "last_modified": f.last_modified}
            for f in files
        ],
    }
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    out_path = BULK_CATALOG_DIR / f"{date_iso}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[BULK_CATALOG] %s 保存完了: file_count=%d", out_path.name, len(files))
    return out_path


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = save_bulk_catalog()
    print(f"bulk catalog written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
