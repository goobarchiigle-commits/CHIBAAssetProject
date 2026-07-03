"""
tools/check_performance_regression.py
Performance regression detection against state/performance_baseline.json.

Usage:
    # Record a measurement:
    python tools/check_performance_regression.py --record test_suite_runtime_s=31.5

    # Check all current measurements against baseline:
    python tools/check_performance_regression.py --check

    # Check a specific measurement:
    python tools/check_performance_regression.py --check test_suite_runtime_s=31.5

    # Time an import and record:
    python tools/check_performance_regression.py --time-import src.governance

    # Auto-measure what we can and check:
    python tools/check_performance_regression.py --auto
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT      = Path(__file__).resolve().parent.parent
STATE_DIR      = REPO_ROOT / "state"
BASELINE_FILE  = STATE_DIR / "performance_baseline.json"
TIMING_DIR     = STATE_DIR / "timing"

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[PERF_REGRESSION] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(REPO_ROOT))


def load_baseline() -> dict:
    if not BASELINE_FILE.exists():
        logger.warning("performance_baseline.json not found — creating empty baseline")
        return {"measurements": {}}
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load baseline: %s", exc)
        return {"measurements": {}}


def save_baseline(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BASELINE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(BASELINE_FILE)


def record_measurement(key: str, value: float) -> None:
    """Update or insert a measurement in the baseline."""
    data = load_baseline()
    measurements = data.setdefault("measurements", {})
    if key not in measurements:
        measurements[key] = {
            "value": value,
            "threshold_pct": 20,
            "unit": "unknown",
            "description": f"Auto-recorded: {key}",
        }
        logger.info("New metric recorded: %s = %s", key, value)
    else:
        old = measurements[key].get("value")
        measurements[key]["value"] = value
        logger.info("Updated metric: %s = %s (was %s)", key, value, old)

    data.setdefault("_meta", {})["last_updated"] = datetime.now(JST).strftime("%Y-%m-%d")
    save_baseline(data)


def check_measurement(
    key: str,
    current_value: float,
    measurements: dict,
) -> list[str]:
    """
    Compare current_value against baseline for key.

    Returns list of violation strings (empty = OK).
    """
    entry = measurements.get(key)
    if entry is None:
        logger.warning("No baseline entry for '%s' — skipping", key)
        return []

    baseline = entry.get("value")
    if baseline is None:
        logger.info("Baseline value is null for '%s' — skipping check", key)
        return []

    threshold_pct = float(entry.get("threshold_pct", 20))
    unit          = entry.get("unit", "")
    allowed_max   = baseline * (1 + threshold_pct / 100)

    if current_value > allowed_max:
        pct_over = ((current_value - baseline) / baseline) * 100
        return [
            f"Performance regression: {key} = {current_value:.2f}{unit} "
            f"exceeds baseline {baseline:.2f}{unit} by {pct_over:.1f}% "
            f"(threshold: +{threshold_pct:.0f}%)"
        ]
    logger.info(
        "OK: %s = %.2f%s (baseline: %.2f%s, threshold: +%.0f%%)",
        key, current_value, unit, baseline, unit, threshold_pct,
    )
    return []


def check_all_from_dict(
    current: dict[str, float],
    baseline_data: Optional[dict] = None,
) -> list[str]:
    """Check multiple measurements against baseline."""
    if baseline_data is None:
        baseline_data = load_baseline()
    measurements = baseline_data.get("measurements", {})
    violations: list[str] = []
    for key, value in current.items():
        violations.extend(check_measurement(key, value, measurements))
    return violations


def time_import(module_name: str) -> float:
    """Time importing a module. Returns elapsed ms."""
    sys.path.insert(0, str(REPO_ROOT))
    # Remove from sys.modules to force fresh import
    for k in list(sys.modules.keys()):
        if k == module_name or k.startswith(module_name + "."):
            del sys.modules[k]
    t0 = time.perf_counter()
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        logger.warning("Import of %s failed: %s", module_name, exc)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms


def auto_measure() -> dict[str, float]:
    """
    Measure what we can automatically.
    Returns {key: value} dict.
    """
    measurements: dict[str, float] = {}

    # Import time
    elapsed_ms = time_import("src.governance")
    measurements["import_time_ms"] = round(elapsed_ms, 1)
    logger.info("src.governance import: %.1f ms", elapsed_ms)

    # Timing file check
    timing_file = STATE_DIR / "last_test_runtime.json"
    if timing_file.exists():
        try:
            d = json.loads(timing_file.read_text(encoding="utf-8"))
            if "runtime_s" in d:
                measurements["test_suite_runtime_s"] = float(d["runtime_s"])
        except Exception:
            pass

    return measurements


def save_timing(key: str, value: float) -> None:
    """Persist a timing measurement to state/timing/."""
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "key": key,
        "value": value,
        "timestamp": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = TIMING_DIR / f"{key}.json"
    out.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def run_checks(
    current: dict[str, float] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Run all performance regression checks.

    Returns:
        (violations, warnings)
    """
    violations: list[str] = []
    warnings:   list[str] = []

    baseline_data = load_baseline()
    measurements  = baseline_data.get("measurements", {})

    if not measurements:
        warnings.append("performance_baseline.json has no measurements — run --record first")
        return violations, warnings

    if current is None:
        current = auto_measure()

    if not current:
        warnings.append("No measurements available for comparison")
        return violations, warnings

    violations.extend(check_all_from_dict(current, baseline_data))
    return violations, warnings


def _parse_kv(kv_str: str) -> tuple[str, float]:
    parts = kv_str.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected key=value, got: {kv_str!r}")
    return parts[0].strip(), float(parts[1].strip())


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Performance regression tracking against baseline"
    )
    parser.add_argument("--record",     metavar="KEY=VALUE",
                        help="Record a measurement into the baseline")
    parser.add_argument("--check",      metavar="KEY=VALUE", nargs="?", const="",
                        help="Check measurement(s) against baseline")
    parser.add_argument("--time-import", metavar="MODULE",
                        help="Time importing a module and check against baseline")
    parser.add_argument("--auto",       action="store_true",
                        help="Auto-measure import time and available metrics")
    args = parser.parse_args()

    if args.record:
        key, value = _parse_kv(args.record)
        record_measurement(key, value)
        logger.info("Recorded: %s = %s", key, value)
        return

    if args.time_import:
        elapsed = time_import(args.time_import)
        logger.info("Import time for %s: %.1f ms", args.time_import, elapsed)
        save_timing(f"import_{args.time_import.replace('.', '_')}", elapsed)
        violations = check_all_from_dict(
            {"import_time_ms": elapsed}
        )
        if violations:
            for v in violations:
                logger.error("REGRESSION: %s", v)
            sys.exit(1)
        return

    if args.check == "":
        # --check with no value: auto-measure
        violations, warnings = run_checks()
    elif args.check:
        key, value = _parse_kv(args.check)
        violations, warnings = run_checks({key: value})
    elif args.auto:
        violations, warnings = run_checks()
    else:
        parser.print_help()
        return

    for w in warnings:
        logger.warning("WARN: %s", w)
    for v in violations:
        logger.error("REGRESSION: %s", v)

    if violations:
        logger.error("Performance regression check FAILED: %d violation(s)", len(violations))
        sys.exit(1)

    logger.info("Performance check PASSED. violations=0 warnings=%d", len(warnings))


if __name__ == "__main__":
    _cli()
