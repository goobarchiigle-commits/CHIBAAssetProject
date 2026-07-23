"""
src/database/verify_database.py
database/market/ 全体の自動整合性監査（件数・スキーマ・欠損・重複・アーカイブ原本ハッシュ照合）。

「将来ファイルを移動した」「圧縮形式を変更した」「数年後に研究を再開した」といった場面で
データベースの完全性を客観的に確認できるようにする（ユーザー要望・2026-07-23）。

実行: python -m src.database.verify_database [--full-archive-check]
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.database import bulk_state
from src.database.exceptions import SchemaValidationError
from src.database.schema import validate_schema
from src.database.snapshot import sha256_streaming
from src.paths import DATABASE_MASTER_DIR

logger = logging.getLogger(__name__)


def verify_bulk_state_files() -> list[dict]:
    """
    bulk_ingest_state.parquetのstatus=ok全行について、csv_gz_local_path/parquet_local_pathが
    実在するか確認する（軽量・ファイルの存在チェックのみ）。
    """
    issues: list[dict] = []
    state = bulk_state.load_state()
    if state.empty:
        return issues
    for _, row in state[state["status"] == "ok"].iterrows():
        for col in ("csv_gz_local_path", "parquet_local_path"):
            path_str = row.get(col) or ""
            for p in str(path_str).split(";"):
                if p and not Path(p).exists():
                    issues.append({
                        "endpoint": row["endpoint"], "period": row["period"],
                        "issue": f"{col}が存在しない", "path": p,
                    })
    return issues


def verify_archive_checksums(*, sample_size: int | None = 20) -> list[dict]:
    """
    archive/bulk/のCSV.GZ原本のSHA256を再計算し、bulk_ingest_state.parquet記録値と照合する。
    sample_size指定時はランダムサンプルのみ検証（全件だとarchive/合計20GB超読むため時間がかかる）。
    sample_size=Noneで全件検証（--full-archive-check）。
    """
    issues: list[dict] = []
    state = bulk_state.load_state()
    if state.empty:
        return issues
    ok_rows = state[(state["status"] == "ok") & (state["csv_gz_sha256"].fillna("") != "")]
    # partial由来の複数ファイル束ね（";"区切り）は個別ハッシュが記録されていないためスキップ
    ok_rows = ok_rows[~ok_rows["csv_gz_local_path"].fillna("").str.contains(";")]
    targets = ok_rows if sample_size is None or len(ok_rows) <= sample_size else ok_rows.sample(
        n=sample_size, random_state=42,
    )
    for _, row in targets.iterrows():
        p = Path(row["csv_gz_local_path"])
        if not p.exists():
            continue  # 存在チェックはverify_bulk_state_files()側の責務
        actual = sha256_streaming(p)
        expected = row["csv_gz_sha256"]
        if actual != expected:
            issues.append({
                "endpoint": row["endpoint"], "period": row["period"], "path": str(p),
                "expected_sha256": expected, "actual_sha256": actual,
            })
    return issues


def verify_schema_conformance() -> list[dict]:
    """
    schema.py登録済みテーブルについて実データがvalidate_schema()を通るか確認する
    （必須列欠落の検出）。ohlcvは容量が大きいため先頭年・最終年のみサンプル検証する。
    """
    issues: list[dict] = []
    from src.database.ohlcv import available_years, load_yearly_parquet

    years = available_years()
    sample_years = sorted(set(years[:1] + years[-1:])) if years else []
    for year in sample_years:
        df = load_yearly_parquet(year)
        try:
            validate_schema(df, "ohlcv")
        except SchemaValidationError as e:
            issues.append({"table": "ohlcv", "year": year, "error": str(e)})

    for table_name, fname in [
        ("companies", "companies.parquet"), ("classifications", "classifications.parquet"),
        ("universe", "universe.parquet"), ("indices", "indices.parquet"),
    ]:
        path = DATABASE_MASTER_DIR / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        try:
            validate_schema(df, table_name)
        except SchemaValidationError as e:
            issues.append({"table": table_name, "error": str(e)})
    return issues


def verify_duplicates() -> list[dict]:
    """master系テーブルの主キー重複を検出する（軽量チェック。minute/tick等はingest時の
    (Date,Time,Code)重複排除を信頼し、ここでは全件再走査しない）。"""
    issues: list[dict] = []
    checks = [
        ("companies.parquet", ["Code"]),
        ("classifications.parquet", ["Code"]),
        ("indices.parquet", ["IndexCode"]),
    ]
    for fname, keys in checks:
        path = DATABASE_MASTER_DIR / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=keys)
        dup_count = int(df.duplicated(subset=keys).sum())
        if dup_count > 0:
            issues.append({"table": fname, "duplicate_rows": dup_count, "keys": keys})
    return issues


def run_full_verification(*, archive_sample_size: int | None = 20) -> dict:
    result: dict = {
        "bulk_state_file_existence": verify_bulk_state_files(),
        "archive_checksum_mismatches": verify_archive_checksums(sample_size=archive_sample_size),
        "schema_conformance": verify_schema_conformance(),
        "duplicates": verify_duplicates(),
    }
    result["pass"] = all(len(v) == 0 for v in result.values() if isinstance(v, list))
    return result


def write_verify_report(result: dict) -> Path:
    import json
    from datetime import datetime, timedelta, timezone
    from src.paths import DATABASE_METADATA_DIR, ensure_database_market_dirs

    ensure_database_market_dirs()
    date_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    out_path = DATABASE_METADATA_DIR / f"verify_report_{date_str}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out_path


def main() -> int:
    import argparse
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="database/market 整合性監査")
    parser.add_argument(
        "--full-archive-check", action="store_true",
        help="archive/bulk/全件のSHA256を再検証する（既定はランダム20件サンプル・時間がかかる）",
    )
    args = parser.parse_args()

    sample_size = None if args.full_archive_check else 20
    result = run_full_verification(archive_sample_size=sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    report_path = write_verify_report(result)
    print(f"report written: {report_path}")
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
