"""
src/governance/determinism.py
Global deterministic seed control for reproducible execution.

Rules:
  - All production/paper runs must set a seed before executing strategy logic
  - Seed is stored in module-level state; missing seed raises SeedNotSetError
  - scan_for_datetime_now() uses AST to flag strategy files using wall-clock time
  - reset_seed() clears state (testing only)
"""
from __future__ import annotations

import ast
import logging
import os
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SEED_STATE: Optional[int] = None
_SEED_SOURCE: str = "unset"


class SeedNotSetError(RuntimeError):
    """Raised when strategy execution is attempted without a seed."""


def set_deterministic(seed: int, *, source: str = "manual") -> None:
    """
    Set global seed for random, numpy, and pandas.

    Args:
        seed:   Integer seed value.
        source: Label for audit trail (e.g. 'experiment', 'test', 'manual').
    """
    global _SEED_STATE, _SEED_SOURCE

    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    _SEED_STATE = seed
    _SEED_SOURCE = source

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import pandas as pd
        # pandas uses numpy RNG; setting numpy seed covers it
        _ = pd
    except ImportError:
        pass

    logger.info("[DETERMINISM] seed=%d source=%s", seed, source)


def require_seed() -> int:
    """
    Return current seed. Raises SeedNotSetError if unset.

    Call this at strategy execution entry points to enforce seeding.
    """
    if _SEED_STATE is None:
        raise SeedNotSetError(
            "No deterministic seed set. Call set_deterministic(seed) before "
            "executing strategy logic. Use EXPERIMENT_SEED env var or pass "
            "seed= explicitly."
        )
    return _SEED_STATE


def get_seed() -> Optional[int]:
    """Return current seed or None (no error)."""
    return _SEED_STATE


def get_seed_source() -> str:
    """Return source label of current seed."""
    return _SEED_SOURCE


def reset_seed() -> None:
    """
    Clear seed state. Testing only — never call from production code.
    """
    global _SEED_STATE, _SEED_SOURCE
    _SEED_STATE = None
    _SEED_SOURCE = "unset"
    logger.debug("[DETERMINISM] seed reset")


def seed_from_env(env_var: str = "EXPERIMENT_SEED") -> Optional[int]:
    """
    Read seed from environment variable and call set_deterministic().

    Returns seed int if set, None if env var absent.
    Raises ValueError if env var is set but not a valid integer.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except ValueError:
        raise ValueError(f"${env_var} must be an integer, got: {raw!r}")
    set_deterministic(seed, source=f"env:{env_var}")
    return seed


# ── AST scanner ──────────────────────────────────────────────────────────────

class _DatetimeNowVisitor(ast.NodeVisitor):
    """Collect all call sites that look like datetime.now() / datetime.datetime.now()."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []  # (line_no, code_repr)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        code = self._repr_call(node)
        if code:
            self.hits.append((node.lineno, code))
        self.generic_visit(node)

    @staticmethod
    def _repr_call(node: ast.Call) -> str:
        func = node.func
        # datetime.now()
        if isinstance(func, ast.Attribute) and func.attr == "now":
            if isinstance(func.value, ast.Name) and func.value.id == "datetime":
                return "datetime.now()"
            if isinstance(func.value, ast.Attribute):
                # datetime.datetime.now()
                if (func.value.attr == "datetime"
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "datetime"):
                    return "datetime.datetime.now()"
        # datetime.today()
        if isinstance(func, ast.Attribute) and func.attr == "today":
            if isinstance(func.value, ast.Name) and func.value.id == "datetime":
                return "datetime.today()"
        return ""


def scan_for_datetime_now(path: Path) -> list[tuple[int, str]]:
    """
    AST-scan a Python file for datetime.now() / datetime.datetime.now() calls.

    Returns list of (line_number, code_repr) tuples; empty = clean.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        logger.warning("[DETERMINISM] AST parse failed for %s: %s", path, exc)
        return []

    visitor = _DatetimeNowVisitor()
    visitor.visit(tree)
    return visitor.hits


def scan_directory(
    directory: Path,
    glob_pattern: str = "**/*.py",
    exclude_fragments: tuple[str, ...] = ("test_", "_test", "clock.py"),
) -> dict[str, list[tuple[int, str]]]:
    """
    Scan all .py files under directory for datetime.now() usage.

    Returns {rel_path: [(line, repr), ...]} for files with hits.
    Excludes test files and clock.py (which legitimately use wall time).
    """
    results: dict[str, list[tuple[int, str]]] = {}
    for py_file in sorted(directory.glob(glob_pattern)):
        rel = str(py_file.relative_to(directory)).replace("\\", "/")
        if any(frag in rel for frag in exclude_fragments):
            continue
        hits = scan_for_datetime_now(py_file)
        if hits:
            results[rel] = hits
    return results
