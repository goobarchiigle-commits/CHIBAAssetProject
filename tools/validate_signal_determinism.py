"""
tools/validate_signal_determinism.py
Run a callable strategy function N times with identical seed/config
and assert outputs are bit-for-bit identical.

Usage (programmatic):
    from tools.validate_signal_determinism import validate_determinism
    ok, report = validate_determinism(my_strategy_fn, n_runs=5, seed=42)

Usage (CLI):
    python tools/validate_signal_determinism.py --help
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[SIGNAL_DET] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _hash_result(result: Any) -> str:
    """Convert any result to a stable sha256 string."""
    try:
        import pandas as pd
        import numpy as np

        if isinstance(result, pd.DataFrame):
            raw = result.to_json(orient="split", date_format="iso").encode()
        elif isinstance(result, pd.Series):
            raw = result.to_json(date_format="iso").encode()
        elif isinstance(result, np.ndarray):
            raw = result.tobytes()
        elif isinstance(result, (dict, list)):
            raw = json.dumps(result, sort_keys=True, default=str).encode()
        else:
            raw = str(result).encode()
    except Exception:
        raw = str(result).encode()

    return hashlib.sha256(raw).hexdigest()


def validate_determinism(
    strategy_fn: Callable,
    *,
    n_runs: int = 3,
    seed: int = 42,
    fn_args: tuple = (),
    fn_kwargs: dict | None = None,
) -> tuple[bool, dict]:
    """
    Call strategy_fn(*fn_args, **fn_kwargs) n_runs times, each time:
      - resetting the global seed via set_deterministic(seed)
      - collecting the result hash

    Returns:
        (all_identical: bool, report: dict)

    report keys:
        n_runs, seed, hashes, identical, first_hash, diverge_at (if any)
    """
    from src.governance.determinism import set_deterministic, reset_seed

    if fn_kwargs is None:
        fn_kwargs = {}

    hashes: list[str] = []
    errors: list[str] = []

    for i in range(n_runs):
        reset_seed()
        set_deterministic(seed, source="validate_signal_determinism")
        try:
            result = strategy_fn(*fn_args, **fn_kwargs)
            h = _hash_result(result)
            hashes.append(h)
            logger.info("run %d/%d hash=%s", i + 1, n_runs, h[:16])
        except Exception as exc:
            err_msg = f"run {i+1} raised {type(exc).__name__}: {exc}"
            errors.append(err_msg)
            logger.error(err_msg)

    if errors:
        return False, {
            "n_runs":    n_runs,
            "seed":      seed,
            "hashes":    hashes,
            "errors":    errors,
            "identical": False,
        }

    unique = set(hashes)
    identical = len(unique) == 1

    diverge_at: int | None = None
    if not identical:
        for i in range(1, len(hashes)):
            if hashes[i] != hashes[0]:
                diverge_at = i + 1
                break

    report = {
        "n_runs":     n_runs,
        "seed":       seed,
        "hashes":     hashes,
        "identical":  identical,
        "first_hash": hashes[0] if hashes else None,
        "diverge_at": diverge_at,
    }

    if identical:
        logger.info("PASS: %d runs identical (hash=%s)", n_runs, hashes[0][:16])
    else:
        logger.error(
            "FAIL: runs diverged at run %d (expected %s, got %s)",
            diverge_at, hashes[0][:16], hashes[diverge_at - 1][:16],
        )

    return identical, report


def validate_determinism_strict(
    strategy_fn: Callable,
    *,
    n_runs: int = 3,
    seed: int = 42,
    fn_args: tuple = (),
    fn_kwargs: dict | None = None,
) -> None:
    """
    Like validate_determinism but raises AssertionError on failure.
    Useful for pytest integration.
    """
    ok, report = validate_determinism(
        strategy_fn, n_runs=n_runs, seed=seed,
        fn_args=fn_args, fn_kwargs=fn_kwargs,
    )
    if not ok:
        raise AssertionError(
            f"Signal determinism validation failed:\n"
            f"  diverge_at: run {report.get('diverge_at')}\n"
            f"  hashes: {report.get('hashes')}\n"
            f"  errors: {report.get('errors', [])}"
        )


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate signal determinism: run strategy fn N times, assert identical"
    )
    parser.add_argument("--module",   required=True, help="Python module path (dot notation)")
    parser.add_argument("--function", required=True, help="Function name in module")
    parser.add_argument("--n-runs",   type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--seed",     type=int, default=42, help="Seed (default: 42)")
    args = parser.parse_args()

    try:
        mod = importlib.import_module(args.module)
    except ImportError as exc:
        logger.error("Cannot import module %s: %s", args.module, exc)
        sys.exit(1)

    fn = getattr(mod, args.function, None)
    if fn is None or not callable(fn):
        logger.error("Function '%s' not found in %s", args.function, args.module)
        sys.exit(1)

    ok, report = validate_determinism(fn, n_runs=args.n_runs, seed=args.seed)
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
