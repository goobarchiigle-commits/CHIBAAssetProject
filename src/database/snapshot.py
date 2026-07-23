"""
src/database/snapshot.py
database/market/metadata/snapshot_{YYYY-MM-DD}.json の生成。

研究DBの「その時点での完全な状態」を1ファイルに固定する。将来ファイルを移動した・
圧縮方式を変更した・数年後に研究を再開した、といった場面で「今と何が違うか」を
客観的に比較できるようにする（ユーザー要望・2026-07-23）。

全Parquet（database/market/配下、cache/除く）についてSHA256を計算する。大容量ファイル
（tick/minute、合計20GB超）はメモリに一括読み込みせずストリーミングでハッシュ計算する
（実行環境がRAM制約下のため必須・一括read_bytes()は使わない）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq

from src.database import metadata as db_metadata
from src.database.reports import build_coverage_report, build_data_quality_report
from src.paths import DATABASE_MARKET_DIR, DATABASE_METADATA_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

_HASH_CHUNK_BYTES = 16 * 1024 * 1024  # 16MB


def sha256_streaming(path: Path) -> str:
    """ファイル全体をメモリに載せずチャンク単位でSHA256を計算する。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _row_count_from_metadata(path: Path) -> int:
    """Parquetのフッタメタデータから行数を取得する（データ本体を読み込まない・高速）。"""
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception as e:  # noqa: BLE001 — 監査対象なので個別失敗はスキップし記録
        logger.warning("[SNAPSHOT] 行数取得失敗 path=%s: %s", path, e)
        return -1


def build_file_manifest() -> list[dict]:
    """database/market/配下（cache/除く）の全ParquetについてSHA256・サイズ・行数を集計する。"""
    manifest = []
    for path in sorted(DATABASE_MARKET_DIR.rglob("*.parquet")):
        if "cache" in path.parts:
            continue
        stat = path.stat()
        manifest.append({
            "path": str(path.relative_to(DATABASE_MARKET_DIR)).replace("\\", "/"),
            "size_bytes": stat.st_size,
            "row_count": _row_count_from_metadata(path),
            "sha256": sha256_streaming(path),
            "mtime": datetime.fromtimestamp(stat.st_mtime, _JST).isoformat(),
        })
        logger.info("[SNAPSHOT] hashed %s (%d bytes)", path.name, stat.st_size)
    return manifest


def build_snapshot() -> dict:
    ensure_database_market_dirs()
    manifest = build_file_manifest()
    total_rows = sum(m["row_count"] for m in manifest if m["row_count"] >= 0)
    total_bytes = sum(m["size_bytes"] for m in manifest)

    quality = build_data_quality_report()
    missing_days = sum(
        len(v) for v in quality["ohlcv_trading_day_coverage"].get("missing_trading_days_by_year", {}).values()
    )
    missing_count = missing_days + len(quality.get("bulk_ingest_partial_periods", []))

    return {
        "generated_at": datetime.now(_JST).isoformat(),
        "database_version": db_metadata.read_database_version() or db_metadata.write_database_version(),
        "api_contract": {
            "provider": "J-Quants",
            "base_plan": "Standard",
            "add_ons": ["株価分足/Tick（equities/bars/minute, equities/trades・契約済み・2026-07-23時点）"],
            "premium_locked": [
                "fins/details(statements)", "fins/dividend", "前場四本値",
                "先物/一部オプション四本値", "markets/breakdown",
            ],
            "separate_addon_not_contracted": ["TDnet適時開示（/td/list,/td/files,/td/bulk）"],
        },
        "totals": {"file_count": len(manifest), "total_rows": total_rows, "total_bytes": total_bytes},
        "coverage": build_coverage_report(),
        "missing_count": missing_count,
        "files": manifest,
    }


def write_snapshot() -> Path:
    snapshot = build_snapshot()
    date_str = datetime.now(_JST).strftime("%Y-%m-%d")
    out_path = DATABASE_METADATA_DIR / f"snapshot_{date_str}.json"
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(
        "[SNAPSHOT] %s 生成完了: files=%d total_rows=%d total_bytes=%d",
        out_path.name, snapshot["totals"]["file_count"], snapshot["totals"]["total_rows"], snapshot["totals"]["total_bytes"],
    )
    return out_path


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = write_snapshot()
    print(f"snapshot written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
