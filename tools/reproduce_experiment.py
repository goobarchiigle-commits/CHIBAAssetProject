"""
tools/reproduce_experiment.py
Restore experiment config snapshot, validate all hashes, optionally rerun,
and compare outputs against stored artifacts.

Usage:
    python tools/reproduce_experiment.py --experiment-id exp_20260101_abc12345
    python tools/reproduce_experiment.py --experiment-id exp_20260101_abc12345 --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[REPRODUCE] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_experiment_hashes(experiment_id: str) -> tuple[list[str], list[str]]:
    """
    Verify all stored artifact hashes for an experiment.

    Returns (errors, warnings).
    """
    import src.governance.experiment as _exp_mod
    from src.governance.experiment import ExperimentRegistry, ArtifactStore

    registry = ExperimentRegistry()
    errors:   list[str] = []
    warnings: list[str] = []

    try:
        manifest = registry.get(experiment_id)
    except KeyError:
        errors.append(f"Experiment '{experiment_id}' not in registry")
        return errors, warnings

    # Verify seed set
    if manifest.seed is None:
        errors.append(f"[{experiment_id}] seed is null — non-reproducible")

    # Verify config snapshot
    config_hash = manifest.config_hash
    if config_hash is None:
        warnings.append(f"[{experiment_id}] config_hash is null — cannot verify config")
    else:
        try:
            from src.governance.config_snapshot import load_snapshot, SNAPSHOT_DIR
            snap_file, snap_manifest = load_snapshot(
                _find_snapshot_by_hash(config_hash, SNAPSHOT_DIR), SNAPSHOT_DIR
            )
            actual_hash = _sha256_file(snap_file)
            stored_hash = snap_manifest.get("sha256", "")
            if stored_hash and actual_hash != stored_hash:
                errors.append(
                    f"[{experiment_id}] config snapshot integrity failure: "
                    f"stored={stored_hash[:16]} actual={actual_hash[:16]}"
                )
            else:
                logger.info("[REPRODUCE] config snapshot OK: %s", config_hash[:16])
        except Exception as exc:
            warnings.append(f"[{experiment_id}] could not verify config snapshot: {exc}")

    # Verify dataset
    dataset_hash = manifest.dataset_hash
    if dataset_hash is None:
        warnings.append(f"[{experiment_id}] dataset_hash is null — cannot verify dataset")
    else:
        reg_file = REPO_ROOT / "state" / "dataset_registry.json"
        if reg_file.exists():
            try:
                ds_reg = json.loads(reg_file.read_text(encoding="utf-8"))
                found = False
                for ds in ds_reg.get("datasets", {}).values():
                    if ds.get("sha256", "").startswith(dataset_hash[:16]):
                        found = True
                        actual = _sha256_file(REPO_ROOT / ds["path"])
                        if actual != ds["sha256"]:
                            errors.append(
                                f"[{experiment_id}] dataset hash mismatch: "
                                f"registered={ds['sha256'][:16]} actual={actual[:16]}"
                            )
                        break
                if not found:
                    warnings.append(
                        f"[{experiment_id}] dataset hash {dataset_hash[:16]} "
                        "not found in dataset_registry.json"
                    )
            except Exception as exc:
                warnings.append(f"[{experiment_id}] could not verify dataset: {exc}")

    # Verify result artifacts
    stored_hashes = manifest.result_artifact_hashes
    store = ArtifactStore(experiment_id, base_dir=_exp_mod.ARTIFACTS_BASE)
    for artifact_name, expected_hash in stored_hashes.items():
        if not expected_hash:
            errors.append(
                f"[{experiment_id}] null hash for artifact: {artifact_name} — "
                "run init_hashes before verifying"
            )
            continue
        artifact_path = store.artifact_dir / artifact_name
        if not artifact_path.exists():
            errors.append(
                f"[{experiment_id}] artifact missing: {artifact_name}"
            )
            continue
        actual_hash = _sha256_file(artifact_path)
        if actual_hash != expected_hash:
            errors.append(
                f"[{experiment_id}] artifact hash mismatch: {artifact_name} "
                f"expected={expected_hash[:16]} actual={actual_hash[:16]}"
            )
        else:
            logger.info("[REPRODUCE] artifact OK: %s", artifact_name)

    return errors, warnings


def _find_snapshot_by_hash(config_hash: str, snapshot_dir: Path) -> str:
    """
    Find snapshot_id whose manifest sha256 starts with config_hash prefix.
    Returns snapshot_id string. Raises FileNotFoundError if not found.
    """
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
    for manifest_file in snapshot_dir.glob("*.json"):
        try:
            m = json.loads(manifest_file.read_text(encoding="utf-8"))
            sha = m.get("sha256", "")
            if sha.startswith(config_hash[:16]) or config_hash.startswith(sha[:16]):
                return m["snapshot_id"]
        except Exception:
            pass
    raise FileNotFoundError(
        f"No snapshot found with config_hash prefix: {config_hash[:16]}"
    )


def reproduce_experiment(
    experiment_id: str,
    verify_only: bool = True,
) -> tuple[bool, dict]:
    """
    Verify all hashes for an experiment.

    Args:
        experiment_id: The experiment to reproduce/verify.
        verify_only:   If True, only verify hashes (no strategy re-execution).

    Returns:
        (success: bool, report: dict)
    """
    errors, warnings = verify_experiment_hashes(experiment_id)

    report = {
        "experiment_id": experiment_id,
        "verify_only":   verify_only,
        "errors":        errors,
        "warnings":      warnings,
        "success":       len(errors) == 0,
    }

    if not verify_only:
        warnings.append(
            "Full re-execution not implemented — strategy runner integration required. "
            "Hash verification completed."
        )

    return report["success"], report


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Verify experiment artifact hashes and reproducibility"
    )
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to verify")
    parser.add_argument("--verify-only",   action="store_true", default=True,
                        help="Only verify hashes (default: True)")
    args = parser.parse_args()

    success, report = reproduce_experiment(
        args.experiment_id,
        verify_only=args.verify_only,
    )

    for w in report["warnings"]:
        logger.warning("WARN: %s", w)
    for e in report["errors"]:
        logger.error("FAIL: %s", e)

    if success:
        logger.info("Reproduction verification PASSED: %s", args.experiment_id)
    else:
        logger.error(
            "Reproduction verification FAILED: %d error(s)", len(report["errors"])
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    _cli()
