"""
src/jquants/verify.py
Study75日次ステージング（cache/daily/day_YYYY-MM-DD.parquet）の整合性検証。

daily_completed_dates.json（マニフェスト・正本扱い）と実ファイルを突き合わせ、
以下4種の不整合を検出する（例外化せず、レポートdictに記録して返す・CLI --verify から呼ばれる）:
  - 欠落ファイル      : マニフェストには「完了」と記録されているが実ファイルが存在しない
  - 破損Parquet       : ファイルは存在するが読み込みに失敗する
  - 行数不一致        : マニフェストの row_count と実ファイルの行数が異なる
  - ハッシュ不一致    : マニフェストの sha256 と実ファイルの内容ハッシュが異なる（改変・破損の疑い）
"""
from __future__ import annotations

import logging

import pandas as pd

from src.jquants import study75_downloader

logger = logging.getLogger(__name__)


def verify_daily_staging() -> dict:
    """
    daily_completed_dates.json に記録された全営業日について、実ステージングファイルとの
    整合性を検証する。Returns: レポートdict（例外は送出しない）。
    """
    completed = study75_downloader.load_completed_dates()

    missing_files: list[str] = []
    corrupted_files: list[dict] = []
    row_count_mismatches: list[dict] = []
    hash_mismatches: list[dict] = []

    for date_str, meta in sorted(completed.items()):
        path = study75_downloader.daily_staging_path(date_str)

        if not path.exists():
            missing_files.append(date_str)
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001 — 破損原因を問わず検出対象にするため意図的に広く捕捉
            corrupted_files.append({"date": date_str, "error": str(e)})
            continue

        expected_rows = meta.get("row_count")
        if expected_rows is not None and len(df) != expected_rows:
            row_count_mismatches.append({
                "date": date_str, "expected": expected_rows, "actual": int(len(df)),
            })

        expected_hash = meta.get("sha256")
        if expected_hash:
            actual_hash = study75_downloader.sha256_of_file(path)
            if actual_hash != expected_hash:
                hash_mismatches.append({
                    "date": date_str, "expected": expected_hash, "actual": actual_hash,
                })

    total_issues = (
        len(missing_files) + len(corrupted_files)
        + len(row_count_mismatches) + len(hash_mismatches)
    )

    return {
        "checked_dates": len(completed),
        "missing_files": missing_files,
        "corrupted_files": corrupted_files,
        "row_count_mismatches": row_count_mismatches,
        "hash_mismatches": hash_mismatches,
        "status": "ok" if total_issues == 0 else "issues_found",
        "total_issues": total_issues,
    }
