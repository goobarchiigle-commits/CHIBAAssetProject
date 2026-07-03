"""
tools/repo_audit.py
Repository inventory scanner for CHIBAAssetProject.

Scans the entire repository, hashes all files, and detects:
  - duplicate filenames (same name, different paths)
  - duplicate content (same sha256)
  - ambiguous names: final, final2, fix, backup, temp, old
  - oversized files (> SIZE_WARN_MB)
  - stale backtest reports (> STALE_REPORT_DAYS old, not in registry)
  - orphan configs (yaml/json in configs/ not referenced by live code)
  - unused strategy .py files (src/backtest/ active, not in registry, not in archive/)

Output: state/repo_inventory.json

Usage:
    python tools/repo_audit.py
    python tools/repo_audit.py --verbose
    python tools/repo_audit.py --output path/to/inventory.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"

SIZE_WARN_MB       = 10
STALE_REPORT_DAYS  = 90

AMBIGUOUS_PATTERNS: list[str] = [
    "final2", "final", "fix", "backup", "temp", "old", "bak", "copy", "orig", "tmp",
]

SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".pytest_cache", "data", ".codex_tmp",
    "node_modules", ".venv", "venv", "env", ".mypy_cache",
})

SKIP_SUFFIXES: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".parquet",
})

STRATEGY_REGISTRY_FILE = STATE_DIR / "strategy_registry.json"

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[REPO_AUDIT] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return path.suffix.lower() in SKIP_SUFFIXES


def _is_ambiguous(name: str) -> tuple[bool, str]:
    p      = Path(name)
    stem   = p.stem.lower()
    suffix = p.suffix.lower().lstrip(".")
    for pat in AMBIGUOUS_PATTERNS:
        if re.search(r"(?:^|[_\-\.])" + re.escape(pat) + r"(?:[_\-\.\d]|$)", stem):
            return True, pat
        if suffix == pat:
            return True, pat
    return False, ""


def _load_registry() -> dict:
    if not STRATEGY_REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(STRATEGY_REGISTRY_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as exc:
        logger.warning("Could not load registry: %s", exc)
        return {}


def scan(repo_root: Path = REPO_ROOT, verbose: bool = False) -> dict:
    """
    Scan the repository. Returns the full inventory dict.
    """
    registry    = _load_registry()
    registry_source_files = {
        str(REPO_ROOT / v["source_file"])
        for v in registry.values()
        if v.get("source_file")
    }
    registry_configs = {
        str(REPO_ROOT / v["active_config"])
        for v in registry.values()
        if v.get("active_config")
    }

    files:             list[dict] = []
    by_name:           defaultdict[str, list[str]] = defaultdict(list)
    by_hash:           defaultdict[str, list[str]] = defaultdict(list)
    issues:            list[dict] = []
    now_ts = datetime.now(JST)

    logger.info("Scanning %s ...", repo_root)
    total = 0

    for root, dirs, filenames in os.walk(repo_root):
        root_path = Path(root)

        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in sorted(filenames):
            fpath = root_path / fname
            rel   = fpath.relative_to(repo_root)

            if _should_skip(fpath):
                continue

            total += 1
            try:
                size_bytes = fpath.stat().st_size
                mtime      = fpath.stat().st_mtime
                mtime_str  = datetime.fromtimestamp(mtime, tz=JST).strftime("%Y-%m-%d")
                fhash      = sha256_file(fpath)
            except PermissionError:
                logger.warning("Permission denied: %s — skipped", fpath)
                continue
            except Exception as exc:
                logger.warning("Error reading %s: %s — skipped", fpath, exc)
                continue

            entry = {
                "path":        str(rel).replace("\\", "/"),
                "abs_path":    str(fpath),
                "size_bytes":  size_bytes,
                "size_mb":     round(size_bytes / 1_048_576, 3),
                "mtime":       mtime_str,
                "sha256":      fhash,
                "in_registry": str(fpath) in registry_source_files
                               or str(fpath) in registry_configs,
            }
            files.append(entry)
            by_name[fname].append(str(rel).replace("\\", "/"))
            by_hash[fhash].append(str(rel).replace("\\", "/"))

            if verbose:
                logger.debug("  %s  [%s MB]  %s", rel, round(size_bytes/1e6,2), fhash[:8])

    logger.info("Scanned %d files", total)

    # ── Issue detection ──────────────────────────────────────────────────────

    # Duplicate filenames
    for name, paths in by_name.items():
        if len(paths) > 1:
            issues.append({
                "type":    "duplicate_filename",
                "name":    name,
                "paths":   paths,
                "message": f"Filename '{name}' exists in {len(paths)} locations",
            })

    # Duplicate content
    for fhash, paths in by_hash.items():
        if len(paths) > 1:
            issues.append({
                "type":    "duplicate_content",
                "hash":    fhash[:16],
                "paths":   paths,
                "message": f"Identical content ({fhash[:8]}...) in {len(paths)} files",
            })

    # Ambiguous names
    for entry in files:
        is_amb, pattern = _is_ambiguous(Path(entry["path"]).name)
        if is_amb:
            issues.append({
                "type":    "ambiguous_name",
                "path":    entry["path"],
                "pattern": pattern,
                "message": f"Ambiguous filename pattern '{pattern}': {entry['path']}",
            })

    # Oversized files
    for entry in files:
        if entry["size_mb"] > SIZE_WARN_MB:
            issues.append({
                "type":    "oversized_file",
                "path":    entry["path"],
                "size_mb": entry["size_mb"],
                "message": f"File exceeds {SIZE_WARN_MB}MB: {entry['path']} ({entry['size_mb']}MB)",
            })

    # Stale backtest reports (backtests/*.json, not in registry, older than STALE_REPORT_DAYS)
    registry_result_refs: set[str] = set()
    for v in registry.values():
        if isinstance(v, dict):
            for key in ("oos_period", "notes"):
                pass  # registry doesn't store backtest result paths directly

    for entry in files:
        path = entry["path"]
        if (
            path.startswith("backtests/")
            and path.endswith(".json")
            and "archive" not in path
        ):
            try:
                mtime_date = datetime.strptime(entry["mtime"], "%Y-%m-%d").replace(tzinfo=JST)
                age_days   = (now_ts - mtime_date).days
                if age_days > STALE_REPORT_DAYS:
                    issues.append({
                        "type":     "stale_report",
                        "path":     path,
                        "age_days": age_days,
                        "message":  f"Report older than {STALE_REPORT_DAYS}d: {path} ({age_days}d old)",
                    })
            except ValueError:
                pass

    # Unused strategy files: src/backtest/*.py, not in archive/, not in registry
    registry_source_rel = {
        v["source_file"].replace("\\", "/")
        for v in registry.values()
        if v.get("source_file")
    }
    for entry in files:
        path = entry["path"]
        if (
            path.startswith("src/backtest/")
            and path.endswith(".py")
            and "/archive/" not in path
            and path != "src/backtest/__init__.py"
            and path not in registry_source_rel
        ):
            issues.append({
                "type":    "unused_strategy_file",
                "path":    path,
                "message": f"Strategy file not in registry and not archived: {path}",
            })

    # Orphan configs: yaml/json in src/configs/ not referenced by registry or known files
    known_config_rel = {
        v["active_config"].replace("\\", "/")
        for v in registry.values()
        if v.get("active_config")
    }
    # Always-valid configs (referenced by paths.py, CLAUDE.md, or known production use)
    always_valid = {
        "src/configs/strategy.yaml",
        "src/configs/universe.yaml",
        "src/configs/rsr_universe_42.csv",
        # Live/shadow universe files — referenced via LIVE_UNIVERSE_FILE / SHADOW_UNIVERSE_FILE in paths.py
        "src/configs/universe/rsr42_trading.json",
        "src/configs/universe/shadow_universe.json",
        # Historical universe snapshots (not orphans — lineage reference)
        "src/configs/universe/2026Q1_rsr42_universe.json",
        "src/configs/universe/2026Q1_temporal24.json",
        # Rolling universe config used by backtest scripts
        "src/configs/rolling_universe.json",
    }
    for entry in files:
        path = entry["path"]
        if (
            path.startswith("src/configs/")
            and (path.endswith(".yaml") or path.endswith(".json"))
            and path not in known_config_rel
            and path not in always_valid
        ):
            issues.append({
                "type":    "orphan_config",
                "path":    path,
                "message": f"Config not referenced by any registry entry: {path}",
            })

    summary = {
        "total_files":              total,
        "duplicate_filename_count": sum(1 for i in issues if i["type"] == "duplicate_filename"),
        "duplicate_content_count":  sum(1 for i in issues if i["type"] == "duplicate_content"),
        "ambiguous_name_count":     sum(1 for i in issues if i["type"] == "ambiguous_name"),
        "oversized_file_count":     sum(1 for i in issues if i["type"] == "oversized_file"),
        "stale_report_count":       sum(1 for i in issues if i["type"] == "stale_report"),
        "unused_strategy_count":    sum(1 for i in issues if i["type"] == "unused_strategy_file"),
        "orphan_config_count":      sum(1 for i in issues if i["type"] == "orphan_config"),
        "total_issues":             len(issues),
    }

    inventory = {
        "_meta": {
            "generated_at": now_ts.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repo_root":    str(repo_root),
            "summary":      summary,
        },
        "files":  files,
        "issues": issues,
    }
    return inventory


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Repository inventory scanner.")
    parser.add_argument("--output",  default="", help="Override output path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output = Path(args.output) if args.output else (STATE_DIR / "repo_inventory.json")
    inv    = scan(verbose=args.verbose)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inv, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    s = inv["_meta"]["summary"]
    logger.info(
        "Inventory written to %s | files=%d issues=%d "
        "(dup_name=%d dup_content=%d ambiguous=%d stale=%d unused_strat=%d orphan_cfg=%d)",
        output,
        s["total_files"],
        s["total_issues"],
        s["duplicate_filename_count"],
        s["duplicate_content_count"],
        s["ambiguous_name_count"],
        s["stale_report_count"],
        s["unused_strategy_count"],
        s["orphan_config_count"],
    )
    if s["total_issues"] > 0:
        logger.warning("%d issue(s) found. Review state/repo_inventory.json for details.", s["total_issues"])
        sys.exit(1)


if __name__ == "__main__":
    _cli()
