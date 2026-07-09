"""
src/jquants/manifest.py
data/jquants/metadata/manifest.json — 同期実行1回ごとのランレコード（追記専用）。
Study75以降の再現性確認（「このデータセットはどのコミット・いつの実行で作られたか」）に使う。

各レコード:
  download_started, download_finished, first_date, last_date,
  symbol_count, record_count, generator_version, git_commit, dataset_hash
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.paths import JQUANTS_METADATA_DIR, JQUANTS_PROCESSED_DIR

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))
MANIFEST_FILE: Path = JQUANTS_METADATA_DIR / "manifest.json"
GENERATOR_VERSION = "jquants_sync/1.0.0"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[JQUANTS_MANIFEST] git_commit 取得失敗（fail-open）: %s", e)
        return ""


def dataset_hash() -> str:
    """
    processed/ 配下の全parquetを対象に、(相対パス, 内容SHA256) の一覧を再ハッシュする。
    catalog.py も同一関数を再利用する（重複実装を避ける・単一の正本）。
    """
    if not JQUANTS_PROCESSED_DIR.exists():
        return ""
    entries = []
    for path in sorted(JQUANTS_PROCESSED_DIR.glob("*.parquet")):
        try:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as e:
            logger.warning("[JQUANTS_MANIFEST] %s のハッシュ計算失敗（fail-open）: %s", path, e)
            continue
        entries.append(f"{path.name}:{file_hash}")
    if not entries:
        return ""
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def load_manifest() -> list[dict]:
    if not MANIFEST_FILE.exists():
        return []
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[JQUANTS_MANIFEST] manifest.json 読込失敗（%s）→ 空から再開", e)
        return []


def save_manifest(records: list[dict]) -> None:
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def record_run(
    download_started: datetime,
    download_finished: datetime,
    first_date: str | None,
    last_date: str | None,
    symbol_count: int,
    record_count: int,
) -> dict:
    """本ランの記録を作り manifest.json に追記する。Returns: 追記したレコード。"""
    record = {
        "download_started":  download_started.astimezone(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "download_finished": download_finished.astimezone(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "first_date":    first_date,
        "last_date":     last_date,
        "symbol_count":  int(symbol_count),
        "record_count":  int(record_count),
        "generator_version": GENERATOR_VERSION,
        "git_commit":    _git_commit(),
        "dataset_hash":  dataset_hash(),
    }
    records = load_manifest()
    records.append(record)
    save_manifest(records)
    logger.info("[JQUANTS_MANIFEST] ラン記録追加: %s", MANIFEST_FILE)
    return record
