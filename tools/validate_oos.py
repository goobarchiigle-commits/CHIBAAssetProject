"""
tools/validate_oos.py
OOS artifact integrity validator.

Validates:
  - artifact file exists and hash matches stored sha256
  - OOS is not stale (created_at + max_age_days > today)
  - strategy_id is in strategy_registry.json
  - validation_status is 'valid' (not 'pending' or 'invalid')

Exit codes:
  0 — all artifacts pass
  1 — one or more failures

Subcommands:
  (default)  Validate all artifacts
  --init     Compute and store hashes for all artifacts (first-time setup)
  --artifact NAME  Validate a single artifact by name

Usage:
    python tools/validate_oos.py
    python tools/validate_oos.py --init
    python tools/validate_oos.py --artifact fujiko_composite_oos_2025_wf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
STATE_DIR      = REPO_ROOT / "state"

OOS_REGISTRY_FILE      = STATE_DIR / "oos_registry.json"
STRATEGY_REGISTRY_FILE = STATE_DIR / "strategy_registry.json"

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[VALIDATE_OOS] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_oos_registry() -> tuple[dict, dict]:
    """Returns (meta, artifacts) dicts."""
    if not OOS_REGISTRY_FILE.exists():
        raise FileNotFoundError(f"OOS registry not found: {OOS_REGISTRY_FILE}")
    data      = json.loads(OOS_REGISTRY_FILE.read_text(encoding="utf-8"))
    meta      = data.get("_meta", {})
    artifacts = data.get("artifacts", {})
    return meta, artifacts


def _load_strategy_registry() -> dict:
    if not STRATEGY_REGISTRY_FILE.exists():
        return {}
    data = json.loads(STRATEGY_REGISTRY_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _save_oos_registry(meta: dict, artifacts: dict) -> None:
    """Atomic write of OOS registry."""
    data = {"_meta": meta, "artifacts": artifacts}
    tmp  = OOS_REGISTRY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OOS_REGISTRY_FILE)


def validate_artifact(
    name:             str,
    artifact:         dict,
    strategy_registry: dict,
    now:              datetime,
) -> list[str]:
    """
    Validate one OOS artifact. Returns list of error strings (empty = OK).
    """
    errors: list[str] = []

    # 1. Strategy must be in registry
    strategy_id = artifact.get("strategy_id", "")
    if strategy_id not in strategy_registry:
        errors.append(
            f"[{name}] strategy_id='{strategy_id}' not found in strategy_registry.json. "
            "Unknown artifact lineage."
        )

    # 2. Artifact file must exist
    artifact_path_rel = artifact.get("artifact_path", "")
    artifact_path     = REPO_ROOT / artifact_path_rel
    if not artifact_path.exists():
        errors.append(f"[{name}] artifact_path not found: {artifact_path_rel}")
        return errors  # can't check hash if file is missing

    # 3. Hash must be set (fail closed: null = not initialized = fail)
    stored_hash = artifact.get("artifact_hash")
    if stored_hash is None:
        errors.append(
            f"[{name}] artifact_hash is null. "
            "Run 'python tools/validate_oos.py --init' to compute and store hashes."
        )
    else:
        # 4. Hash must match
        actual_hash = _sha256(artifact_path)
        if actual_hash != stored_hash:
            errors.append(
                f"[{name}] HASH MISMATCH for {artifact_path_rel}\n"
                f"  stored: {stored_hash[:32]}...\n"
                f"  actual: {actual_hash[:32]}...\n"
                "Artifact may have been modified after OOS validation."
            )
        else:
            logger.info("[%s] hash OK (%s...)", name, actual_hash[:12])

    # 5. Validation status must be 'valid'
    status = artifact.get("validation_status", "")
    if status != "valid":
        errors.append(
            f"[{name}] validation_status='{status}'. Only 'valid' artifacts are accepted."
        )

    # 6. Staleness check
    created_str  = artifact.get("created_at", "")
    max_age_days = int(artifact.get("max_age_days", 365))
    if created_str:
        try:
            created_dt = datetime.strptime(created_str, "%Y-%m-%d").replace(tzinfo=JST)
            age_days   = (now - created_dt).days
            if age_days > max_age_days:
                errors.append(
                    f"[{name}] STALE OOS: artifact is {age_days}d old "
                    f"(limit: {max_age_days}d, created: {created_str}). "
                    "Re-run OOS validation and update the registry."
                )
            else:
                logger.info(
                    "[%s] freshness OK: %dd old (limit: %dd)",
                    name, age_days, max_age_days,
                )
        except ValueError:
            errors.append(f"[{name}] unparseable created_at: '{created_str}'")

    return errors


def run_validation(artifact_filter: str | None = None) -> tuple[list[str], list[str]]:
    """
    Run all OOS validations.

    Returns:
        (errors, warnings) — errors → exit 1.
    """
    now      = datetime.now(JST)
    meta, artifacts = _load_oos_registry()
    strategy_registry = _load_strategy_registry()

    errors:   list[str] = []
    warnings: list[str] = []

    if not artifacts:
        warnings.append(
            "OOS registry has no artifacts. "
            "Add entries to state/oos_registry.json."
        )
        return errors, warnings

    for name, artifact in artifacts.items():
        if artifact_filter and name != artifact_filter:
            continue
        errs = validate_artifact(name, artifact, strategy_registry, now)
        errors.extend(errs)

    return errors, warnings


def init_hashes() -> None:
    """Compute and store sha256 hashes for all artifacts with null hash."""
    meta, artifacts = _load_oos_registry()
    updated = 0

    for name, artifact in artifacts.items():
        if artifact.get("artifact_hash") is not None:
            logger.info("[%s] hash already set — skipped", name)
            continue

        path_rel = artifact.get("artifact_path", "")
        path     = REPO_ROOT / path_rel
        if not path.exists():
            logger.warning("[%s] artifact_path not found: %s — skipped", name, path_rel)
            continue

        h = _sha256(path)
        artifact["artifact_hash"] = h
        logger.info("[%s] hash computed and stored: %s...", name, h[:16])
        updated += 1

    if updated:
        _save_oos_registry(meta, artifacts)
        logger.info("OOS registry updated: %d artifact(s) initialized.", updated)
    else:
        logger.info("All artifacts already have hashes. Nothing to initialize.")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="OOS artifact integrity validator."
    )
    parser.add_argument("--init",     action="store_true", help="Compute and store hashes")
    parser.add_argument("--artifact", default="",          help="Validate single artifact by name")
    args = parser.parse_args()

    if args.init:
        init_hashes()
        return

    artifact_filter = args.artifact or None
    errors, warnings = run_validation(artifact_filter=artifact_filter)

    for w in warnings:
        logger.warning("WARN: %s", w)
    for e in errors:
        logger.error("FAIL: %s", e)

    if errors:
        logger.error("%d OOS validation failure(s).", len(errors))
        sys.exit(1)

    if warnings:
        logger.info("OOS validation PASSED with %d warning(s).", len(warnings))
    else:
        logger.info("OOS validation PASSED. All artifacts valid.")


if __name__ == "__main__":
    _cli()
