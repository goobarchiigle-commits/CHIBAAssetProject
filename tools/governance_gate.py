"""
tools/governance_gate.py
CI governance gate — runs all governance checks and generates GOVERNANCE_STATUS.md.

Runs:
  1. consistency_check       — registry vs files
  2. validate_active_imports — AST import safety
  3. validate_oos             — OOS artifact integrity
  4. state separation         — live/research dir contamination
  5. reproducibility checks   — experiment registry, seed, dataset hashes

Generates:
  docs/GOVERNANCE_STATUS.md

Exit codes:
  0 — all checks pass
  1 — any check failed

Usage:
    python tools/governance_gate.py
    python tools/governance_gate.py --strict
    python tools/governance_gate.py --no-import-check   # skip AST (CI bootstrap)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
STATE_DIR = REPO_ROOT / "state"
DOCS_DIR  = REPO_ROOT / "docs"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_ROOT))

_IMPORT_ERRORS: list[str] = []

def _safe_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError as exc:
        _IMPORT_ERRORS.append(f"Cannot import governance module '{name}': {exc}")
        return None

_cc  = _safe_import("consistency_check")
_vai = _safe_import("validate_active_imports")
_vo  = _safe_import("validate_oos")
_th  = _safe_import("test_hygiene")
_va  = _safe_import("validate_artifacts")

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[GOVERNANCE_GATE] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_registry() -> dict:
    reg_file = STATE_DIR / "strategy_registry.json"
    if not reg_file.exists():
        return {}
    try:
        data = json.loads(reg_file.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return {}


def _load_oos_registry() -> dict:
    oos_file = STATE_DIR / "oos_registry.json"
    if not oos_file.exists():
        return {}
    try:
        data = json.loads(oos_file.read_text(encoding="utf-8"))
        return data.get("artifacts", {})
    except Exception:
        return {}


def _check_state_separation() -> list[str]:
    """Check live/research dir contamination."""
    errors: list[str] = []
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from src.governance.state_guard import (
            assert_no_live_files_in_research_dir,
            assert_no_research_files_in_live_dir,
            LIVE_STATE_DIR,
            RESEARCH_STATE_DIR,
        )
        violations = assert_no_live_files_in_research_dir(LIVE_STATE_DIR, RESEARCH_STATE_DIR)
        for v in violations:
            errors.append(f"Live-only file in research dir: {v}")
        violations = assert_no_research_files_in_live_dir(LIVE_STATE_DIR, RESEARCH_STATE_DIR)
        for v in violations:
            errors.append(f"Research-only file in live dir: {v}")
    except ImportError as exc:
        errors.append(f"Could not load state_guard: {exc}")
    return errors


def _check_reproducibility() -> tuple[list[str], list[str]]:
    """Check experiment registry, dataset registry, and artifact integrity."""
    errors:   list[str] = []
    warnings: list[str] = []

    # Experiment registry must exist
    exp_file = STATE_DIR / "experiment_registry.json"
    if not exp_file.exists():
        errors.append("experiment_registry.json missing — run tools/reproduce_experiment.py")
        return errors, warnings

    try:
        exp_data = json.loads(exp_file.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"experiment_registry.json unreadable: {exc}")
        return errors, warnings

    experiments = exp_data.get("experiments", {})

    # Dataset registry must exist
    ds_file = STATE_DIR / "dataset_registry.json"
    if not ds_file.exists():
        warnings.append("dataset_registry.json missing — run tools/fingerprint_dataset.py")

    # Per-experiment checks
    for eid, entry in experiments.items():
        # Seed must be set
        if entry.get("seed") is None:
            errors.append(f"[{eid}] seed is null — non-reproducible execution")

        # Check artifact hashes are non-null and artifacts exist
        for artifact_name, artifact_hash in entry.get("result_artifact_hashes", {}).items():
            if not artifact_hash:
                errors.append(f"[{eid}] null hash for artifact: {artifact_name}")
                continue
            artifact_path = REPO_ROOT / "artifacts" / "experiments" / eid / artifact_name
            if not artifact_path.exists():
                errors.append(f"[{eid}] artifact missing on disk: {artifact_name}")

        # Dataset hash mismatch check (if both registries present)
        dataset_hash = entry.get("dataset_hash")
        if dataset_hash and ds_file.exists():
            try:
                ds_data = json.loads(ds_file.read_text(encoding="utf-8"))
                matched = any(
                    ds.get("sha256", "") == dataset_hash
                    for ds in ds_data.get("datasets", {}).values()
                )
                if not matched:
                    warnings.append(
                        f"[{eid}] dataset_hash not found in dataset_registry — "
                        "re-fingerprint dataset"
                    )
            except Exception:
                pass

    return errors, warnings


def _count_stale_oos(artifacts: dict) -> int:
    now = datetime.now(JST)
    stale = 0
    for a in artifacts.values():
        created_str  = a.get("created_at", "")
        max_age_days = int(a.get("max_age_days", 365))
        try:
            created_dt = datetime.strptime(created_str, "%Y-%m-%d").replace(tzinfo=JST)
            if (now - created_dt).days > max_age_days:
                stale += 1
        except (ValueError, TypeError):
            pass
    return stale


def _count_orphan_strategies(registry: dict) -> int:
    """Count strategies whose source_file doesn't exist."""
    count = 0
    for entry in registry.values():
        sf = entry.get("source_file", "")
        if sf and not (REPO_ROOT / sf).exists():
            count += 1
    return count


def run_all_checks(
    strict: bool = False,
    skip_import_check: bool = False,
    skip_hygiene: bool = False,
    skip_artifacts: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Run all governance checks. Refuses partial execution on module import failure.

    Returns:
        (errors, warnings)
    """
    all_errors:   list[str] = []
    all_warnings: list[str] = []

    # ── 0. Refuse partial execution ───────────────────────────────────────────
    if _IMPORT_ERRORS:
        for imp_err in _IMPORT_ERRORS:
            all_errors.append(f"[GATE] PARTIAL EXECUTION REFUSED: {imp_err}")
        logger.critical("Governance gate cannot run with missing modules — failing immediately")
        return all_errors, all_warnings

    # ── 1. Consistency check ──────────────────────────────────────────────────
    logger.info("Running consistency_check ...")
    cc_errors, cc_warnings = _cc.run_checks(strict=False)
    for e in cc_errors:
        all_errors.append(f"[CONSISTENCY] {e}")
    for w in cc_warnings:
        all_warnings.append(f"[CONSISTENCY] {w}")

    # ── 2. Import validation ──────────────────────────────────────────────────
    if not skip_import_check:
        logger.info("Running validate_active_imports ...")
        vai_errors, vai_warnings = _vai.run_validation(strict=strict)
        for e in vai_errors:
            all_errors.append(f"[IMPORTS] {e}")
        for w in vai_warnings:
            all_warnings.append(f"[IMPORTS] {w}")
    else:
        logger.info("Import check skipped (--no-import-check)")
        all_warnings.append("[IMPORTS] Import validation skipped — enable for full CI coverage")

    # ── 3. OOS validation ─────────────────────────────────────────────────────
    logger.info("Running validate_oos ...")
    try:
        oos_errors, oos_warnings = _vo.run_validation()
        for e in oos_errors:
            all_errors.append(f"[OOS] {e}")
        for w in oos_warnings:
            all_warnings.append(f"[OOS] {w}")
    except FileNotFoundError as exc:
        all_errors.append(f"[OOS] {exc}")

    # ── 4. State separation ───────────────────────────────────────────────────
    logger.info("Running state separation check ...")
    sep_errors = _check_state_separation()
    for e in sep_errors:
        all_errors.append(f"[STATE] {e}")

    # ── 5. Reproducibility checks ─────────────────────────────────────────────
    logger.info("Running reproducibility checks ...")
    repro_errors, repro_warnings = _check_reproducibility()
    for e in repro_errors:
        all_errors.append(f"[REPRO] {e}")
    for w in repro_warnings:
        all_warnings.append(f"[REPRO] {w}")

    # ── 6. Test hygiene ────────────────────────────────────────────────────────
    if not skip_hygiene and _th is not None:
        logger.info("Running test_hygiene (static checks only) ...")
        hyg_violations, hyg_warnings = _th.run_hygiene_checks(run_pytest=False)
        for v in hyg_violations:
            all_errors.append(f"[HYGIENE] {v}")
        for w in hyg_warnings:
            all_warnings.append(f"[HYGIENE] {w}")

    # ── 7. Artifact integrity ─────────────────────────────────────────────────
    if not skip_artifacts and _va is not None:
        logger.info("Running artifact validation ...")
        art_errors, art_warnings = _va.run_validation()
        for e in art_errors:
            all_errors.append(f"[ARTIFACT] {e}")
        for w in art_warnings:
            all_warnings.append(f"[ARTIFACT] {w}")

    return all_errors, all_warnings


def generate_json_report(
    errors:   list[str],
    warnings: list[str],
) -> dict:
    """
    Generate machine-readable governance report with severity classification
    and signed summary hash.

    The summary_hash covers the sorted findings and metadata — any mutation
    invalidates the hash (tamper detection, not crypto auth).
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from src.governance.severity import findings_from_gate_output, count_by_severity
        findings     = findings_from_gate_output(errors, warnings)
        sev_counts   = count_by_severity(findings)
        findings_dicts = [f.to_dict() for f in findings]
    except ImportError:
        findings_dicts = []
        sev_counts     = {}

    now = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    gate_status = "PASS" if not errors else "FAIL"

    report = {
        "generated_at":   now,
        "gate_status":    gate_status,
        "error_count":    len(errors),
        "warning_count":  len(warnings),
        "severity_counts": sev_counts,
        "findings":       findings_dicts,
        "errors":         errors,
        "warnings":       warnings,
    }

    # Signed summary hash — computed over the report content (without the hash itself)
    payload  = json.dumps(report, sort_keys=True, ensure_ascii=False)
    summary_hash = hashlib.sha256(payload.encode()).hexdigest()
    report["summary_hash"] = summary_hash

    return report


def generate_status_doc(
    errors:   list[str],
    warnings: list[str],
) -> str:
    now      = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    registry = _load_registry()
    artifacts = _load_oos_registry()

    status_by = {}
    for v in registry.values():
        s = v.get("status", "unknown")
        status_by[s] = status_by.get(s, 0) + 1

    active_count      = status_by.get("active", 0)
    experimental_count = status_by.get("experimental", 0)
    deprecated_count  = status_by.get("deprecated", 0)
    archived_count    = status_by.get("archived", 0)
    stale_oos_count   = _count_stale_oos(artifacts)
    orphan_count      = _count_orphan_strategies(registry)
    artifact_count    = len(artifacts)

    live_dir      = REPO_ROOT / "state" / "live"
    research_dir  = REPO_ROOT / "state" / "research"
    sep_status    = "OK" if live_dir.exists() and research_dir.exists() else "MISSING DIRS"

    gate_status = "PASS" if not errors else "FAIL"
    gate_emoji  = "✅" if gate_status == "PASS" else "❌"

    error_block   = "\n".join(f"- {e}" for e in errors)   or "_None_"
    warning_block = "\n".join(f"- {w}" for w in warnings) or "_None_"

    return f"""# GOVERNANCE_STATUS.md — CHIBAAssetProject
> **Generated**: {now}
> Auto-generated by `tools/governance_gate.py`. Do not edit manually.

---

## Gate Result: {gate_emoji} {gate_status}

| Check | Errors | Warnings |
|-------|--------|---------|
| Consistency | {sum(1 for e in errors if '[CONSISTENCY]' in e)} | {sum(1 for w in warnings if '[CONSISTENCY]' in w)} |
| Import validation | {sum(1 for e in errors if '[IMPORTS]' in e)} | {sum(1 for w in warnings if '[IMPORTS]' in w)} |
| OOS integrity | {sum(1 for e in errors if '[OOS]' in e)} | {sum(1 for w in warnings if '[OOS]' in w)} |
| State separation | {sum(1 for e in errors if '[STATE]' in e)} | 0 |
| Reproducibility | {sum(1 for e in errors if '[REPRO]' in e)} | {sum(1 for w in warnings if '[REPRO]' in w)} |
| Test hygiene | {sum(1 for e in errors if '[HYGIENE]' in e)} | {sum(1 for w in warnings if '[HYGIENE]' in w)} |
| Artifact integrity | {sum(1 for e in errors if '[ARTIFACT]' in e)} | {sum(1 for w in warnings if '[ARTIFACT]' in w)} |

---

## Strategy Registry Summary

| Metric | Count |
|--------|-------|
| Active strategies | {active_count} |
| Experimental | {experimental_count} |
| Deprecated | {deprecated_count} |
| Archived | {archived_count} |
| OOS artifacts | {artifact_count} |
| Stale OOS | {stale_oos_count} |
| Orphan source files | {orphan_count} |

---

## Live / Research Separation

- `state/live/` exists: {live_dir.exists()}
- `state/research/` exists: {research_dir.exists()}
- Separation status: **{sep_status}**

---

## Config Snapshots

- Directory: `config/snapshots/`
- Exists: {(REPO_ROOT / "config" / "snapshots").exists()}

---

## Errors

{error_block}

---

## Warnings

{warning_block}

---

## How to Fix

```bash
# Re-run individual checks:
python tools/consistency_check.py
python tools/validate_active_imports.py
python tools/validate_oos.py
python tools/repo_audit.py

# Regenerate this file:
python tools/governance_gate.py
```
"""


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="CI governance gate — runs all checks and generates GOVERNANCE_STATUS.md"
    )
    parser.add_argument("--strict",           action="store_true",
                        help="Also fail on warnings")
    parser.add_argument("--no-import-check",  action="store_true",
                        help="Skip AST import validation")
    parser.add_argument("--no-status-doc",    action="store_true",
                        help="Skip generating GOVERNANCE_STATUS.md")
    parser.add_argument("--no-json-report",   action="store_true",
                        help="Skip generating docs/GOVERNANCE_REPORT.json")
    parser.add_argument("--skip-hygiene",     action="store_true",
                        help="Skip test hygiene static checks")
    parser.add_argument("--skip-artifacts",   action="store_true",
                        help="Skip artifact integrity checks")
    args = parser.parse_args()

    errors, warnings = run_all_checks(
        strict=args.strict,
        skip_import_check=args.no_import_check,
        skip_hygiene=args.skip_hygiene,
        skip_artifacts=args.skip_artifacts,
    )

    # Generate machine-readable JSON report
    if not args.no_json_report:
        json_report = generate_json_report(errors, warnings)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "GOVERNANCE_REPORT.json").write_text(
            json.dumps(json_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Generated docs/GOVERNANCE_REPORT.json (summary_hash=%s)",
            json_report.get("summary_hash", "")[:16],
        )

    # Generate status doc
    if not args.no_status_doc:
        doc = generate_status_doc(errors, warnings)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "GOVERNANCE_STATUS.md").write_text(doc, encoding="utf-8")
        logger.info("Generated docs/GOVERNANCE_STATUS.md")

    # Report
    for w in warnings:
        logger.warning("WARN: %s", w)
    for e in errors:
        logger.error("FAIL: %s", e)

    if errors:
        logger.error(
            "Governance gate FAILED: %d error(s), %d warning(s).",
            len(errors), len(warnings),
        )
        sys.exit(1)

    if args.strict and warnings:
        logger.error("Governance gate FAILED in strict mode: %d warning(s).", len(warnings))
        sys.exit(1)

    logger.info(
        "Governance gate PASSED: errors=0 warnings=%d",
        len(warnings),
    )


if __name__ == "__main__":
    _cli()
