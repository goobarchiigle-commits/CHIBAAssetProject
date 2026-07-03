"""
src/governance/config_snapshot.py
Immutable production config snapshot management.

Rules:
  - Production runs generate a snapshot before execution
  - Snapshots are append-only (never overwrite)
  - Live runs reference snapshot_id, not the live config file
  - Research mode cannot create snapshots
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.governance.mode import RuntimeMode, ModeViolationError

logger = logging.getLogger(__name__)

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR  = REPO_ROOT / "config" / "snapshots"

JST = timezone(timedelta(hours=9))


def create_snapshot(
    config_path: Path,
    snapshot_dir: Path = SNAPSHOT_DIR,
    mode: RuntimeMode | None = None,
) -> str:
    """
    Create an immutable config snapshot. Never overwrites.

    Args:
        config_path:  Source config file (e.g. src/configs/strategy.yaml).
        snapshot_dir: Directory to store snapshots (default: config/snapshots/).
        mode:         Runtime mode; research mode is blocked.

    Returns:
        snapshot_id string (use to reference the snapshot).

    Raises:
        ModeViolationError: if mode is RESEARCH.
        FileNotFoundError:  if config_path does not exist.
        FileExistsError:    should never fire (idempotent via unique timestamp).
    """
    if mode == RuntimeMode.RESEARCH:
        raise ModeViolationError(
            "Research mode cannot create production config snapshots. "
            "Set RUNTIME_MODE=production or RUNTIME_MODE=paper."
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content      = config_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    timestamp    = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    snapshot_id  = f"{config_path.stem}_{timestamp}_{content_hash}"

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snap_file     = snapshot_dir / f"{snapshot_id}{config_path.suffix}"
    manifest_file = snapshot_dir / f"{snapshot_id}.json"

    if snap_file.exists():
        logger.info("[CONFIG_SNAPSHOT] Already exists (idempotent): %s", snapshot_id)
        return snapshot_id

    snap_file.write_bytes(content)
    try:
        source_rel = str(config_path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        source_rel = str(config_path).replace("\\", "/")

    manifest = {
        "snapshot_id":   snapshot_id,
        "source_file":   source_rel,
        "sha256":        hashlib.sha256(content).hexdigest(),
        "created_at":    datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_stem":   config_path.stem,
        "content_hash":  content_hash,
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "[CONFIG_SNAPSHOT] Created: %s (sha256=%s)",
        snapshot_id, content_hash,
    )
    return snapshot_id


def load_snapshot(
    snapshot_id: str,
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> tuple[Path, dict]:
    """
    Return (snapshot_file_path, manifest_dict) for a snapshot_id.

    Raises:
        FileNotFoundError: if snapshot or manifest does not exist.
        RuntimeError:      if sha256 does not match (integrity failure).
    """
    # Find the snapshot file (extension may vary)
    candidates = list(snapshot_dir.glob(f"{snapshot_id}.*"))
    snap_file  = next((f for f in candidates if f.suffix != ".json"), None)
    manifest_f = snapshot_dir / f"{snapshot_id}.json"

    if snap_file is None or not snap_file.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id} in {snapshot_dir}")
    if not manifest_f.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {snapshot_id}.json")

    manifest = json.loads(manifest_f.read_text(encoding="utf-8"))

    # Integrity check
    actual_hash = hashlib.sha256(snap_file.read_bytes()).hexdigest()
    stored_hash = manifest.get("sha256", "")
    if stored_hash and actual_hash != stored_hash:
        raise RuntimeError(
            f"Config snapshot integrity failure: {snapshot_id}\n"
            f"  stored sha256: {stored_hash}\n"
            f"  actual sha256: {actual_hash}\n"
            "Snapshot may have been tampered with."
        )

    logger.info("[CONFIG_SNAPSHOT] Loaded and verified: %s", snapshot_id)
    return snap_file, manifest


def list_snapshots(snapshot_dir: Path = SNAPSHOT_DIR) -> list[dict]:
    """Return all snapshot manifests, sorted by created_at descending."""
    if not snapshot_dir.exists():
        return []
    manifests: list[dict] = []
    for f in snapshot_dir.glob("*.json"):
        try:
            manifests.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(manifests, key=lambda m: m.get("created_at", ""), reverse=True)
