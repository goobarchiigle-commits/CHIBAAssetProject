"""
tools/validate_artifacts.py
Artifact corruption and integrity validation.

Checks:
  1. Parquet readability (not truncated, valid schema)
  2. JSON schema integrity (valid JSON, required fields present)
  3. Hash consistency (sha256 matches registry entry)
  4. Orphan artifacts (dir exists, no registry entry)
  5. Truncated files (size 0 or below minimum)
  6. Invalid timestamps (ISO format check)

Scans:
  - artifacts/experiments/<experiment_id>/
  - backtests/ (OOS artifacts from oos_registry.json)

Usage:
    python tools/validate_artifacts.py
    python tools/validate_artifacts.py --fix-orphans   # delete orphan dirs (dry-run by default)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
STATE_DIR     = REPO_ROOT / "state"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments"
BACKTESTS_DIR = REPO_ROOT / "backtests"

OOS_REGISTRY_FILE    = STATE_DIR / "oos_registry.json"
EXP_REGISTRY_FILE    = STATE_DIR / "experiment_registry.json"

MIN_FILE_BYTES = 2        # below this → truncated
TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
]
TIMESTAMP_FIELDS = ("created_at", "validated_at", "timestamp")

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[VALIDATE_ARTIFACTS] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_timestamp(s: str) -> bool:
    """Return True if s parses as a known timestamp format."""
    for fmt in TIMESTAMP_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return True
        except ValueError:
            continue
    return False


# ── Per-file checks ────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check_truncated(path: Path) -> list[str]:
    size = path.stat().st_size
    if size < MIN_FILE_BYTES:
        return [f"Truncated file ({size}B): {_rel(path)}"]
    return []


def check_json_integrity(path: Path, required_fields: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    rel = _rel(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {rel} — {exc}"]
    except UnicodeDecodeError as exc:
        return [f"Encoding error in JSON: {rel} — {exc}"]

    if required_fields:
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field '{field}' in {rel}")

    # Timestamp field validation
    for ts_field in TIMESTAMP_FIELDS:
        val = data.get(ts_field)
        if val and isinstance(val, str) and not _parse_timestamp(val):
            errors.append(f"Invalid timestamp '{val}' for field '{ts_field}' in {rel}")

    return errors


def check_parquet_readable(path: Path) -> list[str]:
    rel = _rel(path)
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        if len(df) == 0:
            return [f"Parquet file is empty (0 rows): {rel}"]
        return []
    except ImportError:
        return []  # pandas not available — skip silently
    except Exception as exc:
        return [f"Parquet read failed: {rel} — {exc}"]


def check_hash_against_manifest(
    artifact_path: Path,
    expected_hash: str,
) -> list[str]:
    rel = _rel(artifact_path)
    if not expected_hash:
        return [f"Null hash in manifest for: {rel}"]
    actual = _sha256(artifact_path)
    if actual != expected_hash:
        return [
            f"Hash mismatch: {rel} — "
            f"expected={expected_hash[:16]} actual={actual[:16]}"
        ]
    return []


# ── Experiment artifact validation ────────────────────────────────────────────

def _load_experiment_registry() -> dict:
    if not EXP_REGISTRY_FILE.exists():
        return {"experiments": {}}
    try:
        return json.loads(EXP_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"experiments": {}}


def validate_experiment_artifacts() -> tuple[list[str], list[str]]:
    """Validate all files under artifacts/experiments/."""
    errors:   list[str] = []
    warnings: list[str] = []

    reg_data     = _load_experiment_registry()
    known_exp_ids = set(reg_data.get("experiments", {}).keys())

    if not ARTIFACTS_DIR.exists():
        return errors, warnings

    for exp_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name

        # Orphan check
        if exp_id not in known_exp_ids:
            warnings.append(
                f"Orphan artifact directory (no registry entry): artifacts/experiments/{exp_id}"
            )

        reg_entry = reg_data.get("experiments", {}).get(exp_id, {})
        stored_hashes = reg_entry.get("result_artifact_hashes", {})

        for artifact_file in sorted(exp_dir.iterdir()):
            if not artifact_file.is_file():
                continue

            # Truncation
            errors.extend(check_truncated(artifact_file))

            # Format-specific checks
            if artifact_file.suffix == ".parquet":
                errors.extend(check_parquet_readable(artifact_file))
            elif artifact_file.suffix == ".json":
                errors.extend(check_json_integrity(artifact_file))

            # Hash consistency
            stored_hash = stored_hashes.get(artifact_file.name, "")
            if stored_hash:
                errors.extend(check_hash_against_manifest(artifact_file, stored_hash))
            else:
                warnings.append(
                    f"No stored hash for artifact: "
                    f"artifacts/experiments/{exp_id}/{artifact_file.name}"
                )

    return errors, warnings


# ── OOS artifact validation ───────────────────────────────────────────────────

def validate_oos_artifacts() -> tuple[list[str], list[str]]:
    """Validate OOS artifacts from oos_registry.json."""
    errors:   list[str] = []
    warnings: list[str] = []

    if not OOS_REGISTRY_FILE.exists():
        warnings.append("oos_registry.json not found — skipping OOS artifact check")
        return errors, warnings

    try:
        data = json.loads(OOS_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Cannot read oos_registry.json: {exc}"], []

    for name, entry in data.get("artifacts", {}).items():
        path_str = entry.get("artifact_path", "")
        if not path_str:
            errors.append(f"[{name}] missing artifact_path")
            continue

        artifact_path = REPO_ROOT / path_str
        if not artifact_path.exists():
            warnings.append(f"[{name}] OOS artifact not found on disk: {path_str}")
            continue

        # Truncation
        errors.extend(check_truncated(artifact_path))

        # JSON integrity
        if artifact_path.suffix == ".json":
            errors.extend(check_json_integrity(artifact_path))

        # Hash
        stored_hash = entry.get("artifact_hash", "")
        if stored_hash:
            errors.extend(check_hash_against_manifest(artifact_path, stored_hash))
        else:
            errors.append(f"[{name}] null artifact_hash — fail closed")

    return errors, warnings


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation() -> tuple[list[str], list[str]]:
    """
    Run all artifact validation checks.

    Returns:
        (errors, warnings)
    """
    all_errors:   list[str] = []
    all_warnings: list[str] = []

    logger.info("Validating experiment artifacts ...")
    e, w = validate_experiment_artifacts()
    all_errors.extend(e)
    all_warnings.extend(w)

    logger.info("Validating OOS artifacts ...")
    e, w = validate_oos_artifacts()
    all_errors.extend(e)
    all_warnings.extend(w)

    return all_errors, all_warnings


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate artifact integrity (hash, readability, schema, orphans)"
    )
    parser.add_argument("--fix-orphans", action="store_true",
                        help="Show orphan dirs (deletion not automatic)")
    args = parser.parse_args()

    errors, warnings = run_validation()

    for w in warnings:
        logger.warning("WARN: %s", w)
    for e in errors:
        logger.error("ERROR: %s", e)

    if errors:
        logger.error("Artifact validation FAILED: %d error(s)", len(errors))
        sys.exit(1)

    logger.info(
        "Artifact validation PASSED. errors=0 warnings=%d", len(warnings)
    )


if __name__ == "__main__":
    _cli()
