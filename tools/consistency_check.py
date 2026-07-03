"""
tools/consistency_check.py
Consistency validator for CHIBAAssetProject governance layer.

Checks:
  1. strategy_registry.json: schema, valid statuses, no duplicate IDs
  2. Each active strategy: source_file exists, active_config exists
  3. Each active strategy: last_validated_at not stale (>180 days)
  4. Each live entrypoint exists
  5. state/repo_inventory.json freshness (warns if >1 day old)
  6. Capital snapshot age (warns if runtime/portfolio_state.json is stale)

Exit codes:
  0 — all checks pass
  1 — one or more checks failed (details logged)

Usage:
    python tools/consistency_check.py
    python tools/consistency_check.py --strict   # also fail on warnings
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
STATE_DIR  = REPO_ROOT / "state"
RUNTIME_DIR = REPO_ROOT / "runtime"

STRATEGY_REGISTRY_FILE = STATE_DIR / "strategy_registry.json"
REPO_INVENTORY_FILE    = STATE_DIR / "repo_inventory.json"
PORTFOLIO_STATE_FILE   = RUNTIME_DIR / "portfolio_state.json"

VALID_STATUSES = frozenset({"active", "candidate", "experimental", "deprecated", "archived"})
STALE_VALIDATED_DAYS_ACTIVE = 180
STALE_VALIDATED_DAYS_EXPERIMENTAL = 365
INVENTORY_FRESH_DAYS = 1
PORTFOLIO_FRESH_DAYS = 2

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[CONSISTENCY_CHECK] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=JST)
            return dt
        except ValueError:
            continue
    return None


def run_checks(strict: bool = False) -> tuple[list[str], list[str]]:
    """
    Run all consistency checks.

    Returns:
        (errors, warnings) — both are lists of message strings.
        errors → exit 1
        warnings → exit 1 only if strict=True
    """
    errors:   list[str] = []
    warnings: list[str] = []
    now = datetime.now(JST)

    # ── 1. Registry file exists ─────────────────────────────────────────────
    if not STRATEGY_REGISTRY_FILE.exists():
        errors.append(f"strategy_registry.json not found: {STRATEGY_REGISTRY_FILE}")
        return errors, warnings

    try:
        raw = json.loads(STRATEGY_REGISTRY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"strategy_registry.json is not valid JSON: {exc}")
        return errors, warnings

    # ── 2. Registry schema ──────────────────────────────────────────────────
    strategies = {k: v for k, v in raw.items() if not k.startswith("_")}

    required_fields = {"status", "source_file", "active_config", "last_validated_at"}
    for name, entry in strategies.items():
        for field in required_fields:
            if field not in entry:
                errors.append(f"[{name}] missing required field: '{field}'")

        status = entry.get("status", "")
        if status not in VALID_STATUSES:
            errors.append(
                f"[{name}] invalid status='{status}'. Valid: {sorted(VALID_STATUSES)}"
            )

    # ── 3. Source files and configs for active/experimental strategies ──────
    for name, entry in strategies.items():
        status = entry.get("status", "")
        if status in ("deprecated", "archived"):
            continue

        source_file = entry.get("source_file", "")
        if source_file:
            abs_src = REPO_ROOT / source_file
            if not abs_src.exists():
                errors.append(
                    f"[{name}] source_file not found: {source_file}"
                )

        active_config = entry.get("active_config", "")
        if active_config:
            abs_cfg = REPO_ROOT / active_config
            if not abs_cfg.exists():
                errors.append(
                    f"[{name}] active_config not found: {active_config}"
                )

        for ep in entry.get("live_entrypoints", []):
            abs_ep = REPO_ROOT / ep
            if not abs_ep.exists():
                errors.append(
                    f"[{name}] live_entrypoint not found: {ep}"
                )

    # ── 4. Staleness of last_validated_at ───────────────────────────────────
    for name, entry in strategies.items():
        status = entry.get("status", "")
        if status in ("deprecated", "archived"):
            continue

        validated_str = entry.get("last_validated_at")
        validated_dt  = _parse_date(validated_str)
        threshold     = (
            STALE_VALIDATED_DAYS_ACTIVE
            if status == "active"
            else STALE_VALIDATED_DAYS_EXPERIMENTAL
        )

        if validated_dt is None:
            if status == "active":
                errors.append(
                    f"[{name}] last_validated_at is missing or unparseable — required for active strategies"
                )
        else:
            age = (now - validated_dt).days
            if age > threshold:
                msg = (
                    f"[{name}] last_validated_at={validated_str} is {age}d old "
                    f"(limit: {threshold}d for status={status})"
                )
                if status == "active":
                    errors.append(msg)
                else:
                    warnings.append(msg)

    # ── 5. No duplicate strategy IDs ────────────────────────────────────────
    seen: set[str] = set()
    for name in strategies:
        if name in seen:
            errors.append(f"Duplicate strategy ID: '{name}'")
        seen.add(name)

    # ── 6. Inventory freshness (warning only) ───────────────────────────────
    if REPO_INVENTORY_FILE.exists():
        try:
            inv = json.loads(REPO_INVENTORY_FILE.read_text(encoding="utf-8"))
            gen_str = inv.get("_meta", {}).get("generated_at", "")
            gen_dt  = _parse_date(gen_str)
            if gen_dt:
                age = (now - gen_dt).days
                if age > INVENTORY_FRESH_DAYS:
                    warnings.append(
                        f"repo_inventory.json is {age}d old. Run: python tools/repo_audit.py"
                    )
        except Exception:
            warnings.append("Could not parse repo_inventory.json metadata")
    else:
        warnings.append(
            "state/repo_inventory.json not found. Run: python tools/repo_audit.py"
        )

    # ── 7. Portfolio state freshness (warning only) ─────────────────────────
    if PORTFOLIO_STATE_FILE.exists():
        try:
            mtime = datetime.fromtimestamp(
                PORTFOLIO_STATE_FILE.stat().st_mtime, tz=JST
            )
            age = (now - mtime).days
            if age > PORTFOLIO_FRESH_DAYS:
                warnings.append(
                    f"portfolio_state.json is {age}d old (last modified {mtime.strftime('%Y-%m-%d')}). "
                    "Sync positions before trading."
                )
        except Exception:
            warnings.append("Could not check portfolio_state.json modification time")

    return errors, warnings


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CHIBAAssetProject governance consistency."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Also exit 1 on warnings"
    )
    args = parser.parse_args()

    errors, warnings = run_checks(strict=args.strict)

    for w in warnings:
        logger.warning("WARN: %s", w)
    for e in errors:
        logger.error("FAIL: %s", e)

    if errors:
        logger.error("%d check(s) FAILED. Fix issues then re-run.", len(errors))
        sys.exit(1)

    if args.strict and warnings:
        logger.error("%d warning(s) in strict mode — treated as failures.", len(warnings))
        sys.exit(1)

    if warnings:
        logger.info("All critical checks PASSED. %d warning(s).", len(warnings))
    else:
        logger.info("All checks PASSED.")


if __name__ == "__main__":
    _cli()
