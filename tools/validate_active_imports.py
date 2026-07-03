"""
tools/validate_active_imports.py
AST-based import validator for production entrypoints.

Rules:
  - Production entrypoints must not import from:
    * src/backtest/archive/     (hard violation)
    * research/                 (hard violation)
    * files registered as deprecated/archived in strategy_registry.json
  - Strategy imports (from src/backtest/, src/strategies/) that are NOT in the
    registry as active are flagged as warnings (errors in --strict mode)

Usage:
    python tools/validate_active_imports.py
    python tools/validate_active_imports.py --target src/run_live_signal.py
    python tools/validate_active_imports.py --strict
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"

REGISTRY_FILE = STATE_DIR / "strategy_registry.json"

# Paths that are never OK to import from in production
FORBIDDEN_PATH_FRAGMENTS: tuple[str, ...] = (
    "src/backtest/archive",
    "backtest/archive",
    "research/",
)

# Paths containing strategy logic (not infrastructure) — checked against registry
STRATEGY_PATH_PREFIXES: tuple[str, ...] = (
    "src/backtest/",
    "src/strategies/",
)

# Infrastructure paths — always OK (no registry check)
INFRA_PATH_PREFIXES: tuple[str, ...] = (
    "src/execution/",
    "src/common/",
    "src/risk/",
    "src/monitoring/",
    "src/market/",
    "src/portfolio/",
    "src/kabusapi/",
    "src/live/",
    "src/governance/",
    "src/diagnostics/",
    "src/research/",
    "src/scripts/",
    "src/tools/",
    "src/utils/",
)

logging.basicConfig(
    level=logging.INFO,
    format="[VALIDATE_IMPORTS] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        logger.warning("strategy_registry.json not found — skipping registry checks")
        return {}
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as exc:
        logger.warning("Could not load registry: %s", exc)
        return {}


def extract_imports(path: Path) -> list[str]:
    """
    Extract all module names from import / from-import statements via AST.
    Returns list of dotted module strings (e.g. 'src.backtest.archive.clenow').
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        logger.warning("AST parse failed for %s: %s", path, exc)
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def module_to_rel_path(module: str) -> str:
    """
    Convert dotted module name to forward-slash relative path.
    e.g. 'src.backtest.archive.clenow' → 'src/backtest/archive/clenow.py'
    """
    return module.replace(".", "/") + ".py"


def classify_import(rel_path: str) -> str:
    """
    Classify an import by its relative path.
    Returns: 'forbidden' | 'strategy' | 'infra' | 'unknown'
    """
    norm = rel_path.replace("\\", "/")
    for frag in FORBIDDEN_PATH_FRAGMENTS:
        if frag in norm:
            return "forbidden"
    for pfx in INFRA_PATH_PREFIXES:
        if norm.startswith(pfx):
            return "infra"
    for pfx in STRATEGY_PATH_PREFIXES:
        if norm.startswith(pfx):
            return "strategy"
    return "unknown"


def validate_file(
    target: Path,
    registry: dict,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Validate imports in a single file.

    Returns:
        (violations, warnings)
        violations → exit 1 always
        warnings   → exit 1 only if strict=True
    """
    active_source_files = {
        v["source_file"].replace("\\", "/")
        for v in registry.values()
        if v.get("status") == "active" and v.get("source_file")
    }
    forbidden_source_files = {
        v["source_file"].replace("\\", "/")
        for v in registry.values()
        if v.get("status") in ("deprecated", "archived") and v.get("source_file")
    }

    violations: list[str] = []
    warnings:   list[str] = []

    modules = extract_imports(target)
    try:
        rel_target = str(target.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rel_target = str(target).replace("\\", "/")

    for module in modules:
        rel_path = module_to_rel_path(module)
        kind     = classify_import(rel_path)

        if kind == "forbidden":
            violations.append(
                f"[{rel_target}] FORBIDDEN import: '{module}' resolves to forbidden path '{rel_path}'"
            )
            logger.error("VIOLATION: %s imports from forbidden path: %s", rel_target, rel_path)
            continue

        if rel_path in forbidden_source_files:
            violations.append(
                f"[{rel_target}] imports DEPRECATED/ARCHIVED strategy: '{module}' → '{rel_path}'"
            )
            logger.error(
                "VIOLATION: %s imports deprecated/archived strategy: %s",
                rel_target, rel_path,
            )
            continue

        if kind == "strategy" and rel_path not in active_source_files:
            msg = (
                f"[{rel_target}] strategy import not in active registry: "
                f"'{module}' → '{rel_path}'"
            )
            warnings.append(msg)
            logger.warning("WARN: %s", msg)

    return violations, warnings


def get_production_targets(registry: dict) -> list[Path]:
    """Get all live_entrypoints from active registry entries."""
    targets: list[Path] = []
    for entry in registry.values():
        if entry.get("status") == "active":
            for ep in entry.get("live_entrypoints", []):
                p = REPO_ROOT / ep
                if p.exists():
                    targets.append(p)
                else:
                    logger.warning("Entrypoint not found: %s", p)
    return targets


def run_validation(
    targets: list[Path] | None = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Run import validation on all targets.

    Returns:
        (all_violations, all_warnings)
    """
    registry = _load_registry()

    if targets is None:
        targets = get_production_targets(registry)

    if not targets:
        logger.info("No production targets found — nothing to validate")
        return [], []

    all_violations: list[str] = []
    all_warnings:   list[str] = []

    for target in targets:
        logger.info("Checking imports: %s", target.relative_to(REPO_ROOT))
        v, w = validate_file(target, registry, strict=strict)
        all_violations.extend(v)
        all_warnings.extend(w)

    return all_violations, all_warnings


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate production entrypoint imports via AST inspection."
    )
    parser.add_argument("--target",  default="", help="Override target file(s), comma-separated")
    parser.add_argument("--strict",  action="store_true", help="Fail on warnings too")
    args = parser.parse_args()

    if args.target:
        targets = [REPO_ROOT / t.strip() for t in args.target.split(",")]
    else:
        targets = None

    violations, warnings = run_validation(targets=targets, strict=args.strict)

    for w in warnings:
        logger.warning("WARN: %s", w)
    for v in violations:
        logger.error("VIOLATION: %s", v)

    if violations:
        logger.error("%d import violation(s) found. Fix before deploying.", len(violations))
        sys.exit(1)

    if args.strict and warnings:
        logger.error("%d warning(s) in strict mode — treated as violations.", len(warnings))
        sys.exit(1)

    logger.info(
        "Import validation PASSED. violations=0 warnings=%d",
        len(warnings),
    )


if __name__ == "__main__":
    _cli()
