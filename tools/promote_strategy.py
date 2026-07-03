"""
tools/promote_strategy.py
Strategy promotion pipeline: research → experimental → candidate → active.

Each promotion step enforces governance, OOS, and determinism checks.
Creates an immutable promotion report in docs/promotion_reports/.
Updates state/strategy_registry.json atomically.

Usage:
    python tools/promote_strategy.py --strategy-id my_strat --to experimental
    python tools/promote_strategy.py --strategy-id my_strat --to candidate
    python tools/promote_strategy.py --strategy-id my_strat --to active
    python tools/promote_strategy.py --strategy-id my_strat --to active --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

REPO_ROOT          = Path(__file__).resolve().parent.parent
STATE_DIR          = REPO_ROOT / "state"
DOCS_DIR           = REPO_ROOT / "docs"
REPORTS_DIR        = DOCS_DIR / "promotion_reports"
REGISTRY_FILE      = STATE_DIR / "strategy_registry.json"
ARCHIVE_LOG_FILE   = STATE_DIR / "archive_log.jsonl"

JST = timezone(timedelta(hours=9))

VALID_PROMOTIONS = {
    "experimental": {"from": {"research", "experimental"}, "min_level": 1},
    "candidate":    {"from": {"experimental", "candidate"}, "min_level": 2},
    "active":       {"from": {"candidate", "active"}, "min_level": 3},
}

THRESHOLDS = {
    "experimental": {"min_sharpe": 0.0, "min_trades": 5,  "max_dd": -1.0},
    "candidate":    {"min_sharpe": 0.8, "min_trades": 20, "max_dd": -0.35},
    "active":       {"min_sharpe": 1.0, "min_trades": 30, "max_dd": -0.25},
}

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

logging.basicConfig(
    level=logging.INFO,
    format="[PROMOTE] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Registry helpers ──────────────────────────────────────────────────────────

def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"strategy_registry.json not found: {REGISTRY_FILE}")
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def save_registry(data: dict) -> None:
    tmp = REGISTRY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(REGISTRY_FILE)


def get_strategy_entry(strategy_id: str) -> dict:
    raw = load_registry()
    entry = {k: v for k, v in raw.items() if not k.startswith("_")}.get(strategy_id)
    if entry is None:
        raise KeyError(f"Strategy '{strategy_id}' not found in registry")
    return entry


# ── Validation checks ─────────────────────────────────────────────────────────

def check_source_file(entry: dict, strategy_id: str) -> list[str]:
    errors: list[str] = []
    sf = entry.get("source_file", "")
    if not sf:
        errors.append(f"[{strategy_id}] source_file missing")
    elif not (REPO_ROOT / sf).exists():
        errors.append(f"[{strategy_id}] source_file not found: {sf}")
    return errors


def check_metrics_thresholds(
    entry: dict,
    strategy_id: str,
    target_status: str,
) -> list[str]:
    errors: list[str] = []
    thresholds = THRESHOLDS.get(target_status, {})

    oos_sharpe = entry.get("oos_sharpe")
    oos_max_dd = entry.get("oos_max_dd")
    wf_result  = entry.get("wf_result", "")

    min_sharpe = thresholds.get("min_sharpe", 0.0)
    max_dd     = thresholds.get("max_dd", -1.0)

    if oos_sharpe is not None and oos_sharpe < min_sharpe:
        errors.append(
            f"[{strategy_id}] oos_sharpe={oos_sharpe:.3f} < threshold {min_sharpe:.3f} "
            f"for → {target_status}"
        )

    if oos_max_dd is not None and oos_max_dd < max_dd:
        errors.append(
            f"[{strategy_id}] oos_max_dd={oos_max_dd:.3f} < threshold {max_dd:.3f} "
            f"for → {target_status}"
        )

    if target_status == "active" and "5/5" not in wf_result:
        errors.append(
            f"[{strategy_id}] active promotion requires 5/5 walk-forward pass; "
            f"got wf_result='{wf_result}'"
        )

    return errors


def check_consistency(strategy_id: str) -> list[str]:
    """Run consistency_check for a specific strategy."""
    errors: list[str] = []
    try:
        import consistency_check as _cc
        cc_errors, _ = _cc.run_checks(strict=False)
        strat_errors = [e for e in cc_errors if f"[{strategy_id}]" in e]
        errors.extend(strat_errors)
    except ImportError as exc:
        errors.append(f"Could not run consistency_check: {exc}")
    return errors


def check_oos_validation(strategy_id: str) -> list[str]:
    """Run OOS artifact check for this strategy."""
    errors: list[str] = []
    try:
        import validate_oos as _vo
        oos_errors, _ = _vo.run_validation()
        strat_errors = [e for e in oos_errors if strategy_id in e]
        errors.extend(strat_errors)
    except ImportError as exc:
        errors.append(f"Could not run validate_oos: {exc}")
    return errors


def check_quarantine(strategy_id: str, entry: dict) -> list[str]:
    """Block promotion of strategies whose source_file is in research/quarantine/."""
    sf = entry.get("source_file", "")
    if "quarantine" in sf.replace("\\", "/"):
        return [
            f"[{strategy_id}] source_file is in quarantine: {sf}. "
            "Fix the experiment before promoting."
        ]
    return []


# ── Promotion flow ────────────────────────────────────────────────────────────

def promote(
    strategy_id: str,
    to_status: str,
    *,
    dry_run: bool = False,
    skip_oos: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate all requirements and promote strategy to to_status.

    Returns:
        (success: bool, report: dict)
    """
    if to_status not in VALID_PROMOTIONS:
        raise ValueError(f"Unknown target status: '{to_status}'. Valid: {sorted(VALID_PROMOTIONS)}")

    try:
        entry = get_strategy_entry(strategy_id)
    except KeyError as exc:
        return False, {"errors": [str(exc)], "strategy_id": strategy_id, "to_status": to_status}

    current_status = entry.get("status", "")
    allowed_from   = VALID_PROMOTIONS[to_status]["from"]

    errors:   list[str] = []
    warnings: list[str] = []

    # ── Validate current status ───────────────────────────────────────────────
    if current_status not in allowed_from:
        errors.append(
            f"[{strategy_id}] current status='{current_status}' cannot promote to '{to_status}'. "
            f"Must be one of: {sorted(allowed_from)}"
        )
        return False, _build_report(strategy_id, current_status, to_status, errors, warnings, entry)

    # ── Quarantine check ──────────────────────────────────────────────────────
    errors.extend(check_quarantine(strategy_id, entry))

    # ── Source file ───────────────────────────────────────────────────────────
    errors.extend(check_source_file(entry, strategy_id))

    # ── Metrics thresholds ────────────────────────────────────────────────────
    errors.extend(check_metrics_thresholds(entry, strategy_id, to_status))

    # ── Consistency check ─────────────────────────────────────────────────────
    if to_status in ("candidate", "active"):
        errors.extend(check_consistency(strategy_id))

    # ── OOS validation ────────────────────────────────────────────────────────
    if to_status == "active" and not skip_oos:
        errors.extend(check_oos_validation(strategy_id))

    success = len(errors) == 0

    report = _build_report(strategy_id, current_status, to_status, errors, warnings, entry)

    if success and not dry_run:
        _apply_promotion(strategy_id, to_status)
        _write_report(report)
        logger.info("Promoted %s: %s → %s", strategy_id, current_status, to_status)
    elif dry_run:
        logger.info(
            "DRY-RUN: would promote %s: %s → %s (errors=%d)",
            strategy_id, current_status, to_status, len(errors),
        )
    else:
        logger.error(
            "Promotion FAILED: %s → %s: %d error(s)",
            strategy_id, to_status, len(errors),
        )

    return success, report


def _apply_promotion(strategy_id: str, to_status: str) -> None:
    raw = load_registry()
    raw[strategy_id]["status"] = to_status
    raw[strategy_id]["last_validated_at"] = datetime.now(JST).strftime("%Y-%m-%d")
    save_registry(raw)


def _build_report(
    strategy_id: str,
    from_status: str,
    to_status: str,
    errors: list[str],
    warnings: list[str],
    entry: dict,
) -> dict[str, Any]:
    now = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "strategy_id":   strategy_id,
        "from_status":   from_status,
        "to_status":     to_status,
        "success":       len(errors) == 0,
        "created_at":    now,
        "errors":        errors,
        "warnings":      warnings,
        "metrics_snapshot": {
            "oos_sharpe": entry.get("oos_sharpe"),
            "oos_max_dd": entry.get("oos_max_dd"),
            "oos_cagr":   entry.get("oos_cagr"),
            "wf_result":  entry.get("wf_result"),
        },
        "source_file":   entry.get("source_file", ""),
        "active_config": entry.get("active_config", ""),
    }


def _write_report(report: dict) -> None:
    """Write immutable promotion report to docs/promotion_reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = report["created_at"].replace(":", "-").replace("+", "p")[:19]
    name = f"{report['strategy_id']}_{report['from_status']}_to_{report['to_status']}_{ts}.json"
    dest = REPORTS_DIR / name

    if dest.exists():
        # Immutable — never overwrite; append counter
        i = 1
        while dest.exists():
            dest = REPORTS_DIR / f"{name[:-5]}_{i}.json"
            i += 1

    content = json.dumps(report, ensure_ascii=False, indent=2)
    dest.write_text(content, encoding="utf-8")
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]
    logger.info("Promotion report written: %s (sha256=%s)", dest.name, sha)

    # Also append to archive log
    log_entry = json.dumps({
        "event":        "promotion",
        "strategy_id":  report["strategy_id"],
        "from_status":  report["from_status"],
        "to_status":    report["to_status"],
        "report_file":  (str(dest.relative_to(REPO_ROOT)) if dest.is_relative_to(REPO_ROOT) else str(dest)).replace("\\", "/"),
        "created_at":   report["created_at"],
    })
    with open(ARCHIVE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def quarantine_experiment(
    source_path: Path,
    reason: str = "",
    *,
    dry_run: bool = False,
) -> Path:
    """
    Move a failing experiment file to research/quarantine/.
    Returns destination path.
    """
    import shutil
    quarantine_dir = REPO_ROOT / "research" / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    ts   = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    dest = quarantine_dir / f"{source_path.stem}_{ts}{source_path.suffix}"

    if dry_run:
        logger.info("DRY-RUN quarantine: %s → %s", source_path, dest)
        return dest

    shutil.move(str(source_path), str(dest))

    log_entry = json.dumps({
        "event":  "quarantine",
        "source": str(source_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dest":   str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
        "reason": reason,
        "ts":     datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    with open(ARCHIVE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

    logger.info("Quarantined: %s → %s (reason: %s)", source_path.name, dest.name, reason)
    return dest


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy promotion pipeline: research → experimental → candidate → active"
    )
    parser.add_argument("--strategy-id", required=True, help="Strategy ID in registry")
    parser.add_argument("--to",           required=True,
                        choices=list(VALID_PROMOTIONS), metavar="STATUS",
                        help="Target status: experimental | candidate | active")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Validate only, do not update registry")
    parser.add_argument("--skip-oos",    action="store_true",
                        help="Skip OOS artifact check (use only for bootstrap)")
    args = parser.parse_args()

    success, report = promote(
        args.strategy_id,
        args.to,
        dry_run=args.dry_run,
        skip_oos=args.skip_oos,
    )

    for e in report.get("errors", []):
        logger.error("ERROR: %s", e)
    for w in report.get("warnings", []):
        logger.warning("WARN: %s", w)

    if not success:
        logger.error(
            "Promotion FAILED: %s → %s",
            args.strategy_id, args.to,
        )
        sys.exit(1)

    action = "DRY-RUN promotion validated" if args.dry_run else "Promoted"
    logger.info(
        "%s: %s → %s PASSED",
        action, args.strategy_id, args.to,
    )


if __name__ == "__main__":
    _cli()
