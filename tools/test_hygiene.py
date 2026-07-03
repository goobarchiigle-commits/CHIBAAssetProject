"""
tools/test_hygiene.py
Zero-known-failure policy enforcer.

Rules:
  1. xfail markers must have a future expiry date in reason string (YYYY-MM-DD)
  2. Skip count must not exceed threshold (default: 10)
  3. Source/test files must not contain forbidden patterns
  4. All test failures must appear in state/known_failures.json
  5. Known failures with expired expiry date → hard violation
  6. Failures not in known_failures.json → new regression (hard violation)

Usage:
    python tools/test_hygiene.py
    python tools/test_hygiene.py --skip-threshold 20
    python tools/test_hygiene.py --no-run   # skip running pytest, only static checks
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT     = Path(__file__).resolve().parent.parent
STATE_DIR     = REPO_ROOT / "state"
TESTS_DIR     = REPO_ROOT / "tests"
SRC_DIR       = REPO_ROOT / "src"
TOOLS_DIR     = REPO_ROOT / "tools"

KNOWN_FAILURES_FILE = STATE_DIR / "known_failures.json"

DEFAULT_SKIP_THRESHOLD = 10
JST = timezone(timedelta(hours=9))

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bflaky\b", re.IGNORECASE),             "flaky"),
    (re.compile(r"\bintermittent\b", re.IGNORECASE),      "intermittent"),
    (re.compile(r"\bTODO\b"),                             "TODO"),
    (re.compile(r"temporary\s+disable", re.IGNORECASE),   "temporary disable"),
    (re.compile(r"temporari(?:ly)?\s+skip", re.IGNORECASE), "temporarily skip"),
]

logging.basicConfig(
    level=logging.INFO,
    format="[HYGIENE] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Known failures ────────────────────────────────────────────────────────────

def load_known_failures() -> list[dict]:
    if not KNOWN_FAILURES_FILE.exists():
        return []
    try:
        data = json.loads(KNOWN_FAILURES_FILE.read_text(encoding="utf-8"))
        return data.get("known_failures", [])
    except Exception as exc:
        logger.warning("Could not load known_failures.json: %s", exc)
        return []


def check_known_failure_expiry(known_failures: list[dict]) -> list[str]:
    """Flag known failures whose expiry date has passed."""
    violations: list[str] = []
    today = date.today()
    for entry in known_failures:
        expiry_str = entry.get("expiry", "")
        if not expiry_str:
            violations.append(
                f"Known failure missing expiry: {entry.get('test_id', '?')}"
            )
            continue
        try:
            expiry = date.fromisoformat(expiry_str)
        except ValueError:
            violations.append(
                f"Known failure has invalid expiry date '{expiry_str}': {entry.get('test_id')}"
            )
            continue
        if today > expiry:
            violations.append(
                f"Known failure EXPIRED ({expiry_str}): {entry.get('test_id')} — "
                f"Reason: {entry.get('reason', 'no reason')}. Fix or extend expiry."
            )
    return violations


# ── xfail scanner ─────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class _XfailVisitor(ast.NodeVisitor):
    """Find @pytest.mark.xfail decorators and check for expiry date in reason."""

    def __init__(self, source_lines: list[str]) -> None:
        self.violations: list[tuple[int, str]] = []
        self._lines = source_lines

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._check_decorators(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def _check_decorators(self, node: ast.FunctionDef) -> None:
        for deco in node.decorator_list:
            if not self._is_xfail(deco):
                continue
            reason = self._extract_reason(deco)
            if not reason:
                self.violations.append((
                    deco.lineno,
                    f"@pytest.mark.xfail without reason= at line {deco.lineno} "
                    f"(function: {node.name})",
                ))
            elif not _DATE_RE.search(reason):
                self.violations.append((
                    deco.lineno,
                    f"@pytest.mark.xfail reason has no expiry date (YYYY-MM-DD) "
                    f"at line {deco.lineno} (function: {node.name}): {reason!r}",
                ))

    @staticmethod
    def _is_xfail(node: ast.expr) -> bool:
        # @pytest.mark.xfail or @pytest.mark.xfail(...)
        if isinstance(node, ast.Attribute):
            return node.attr == "xfail"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return func.attr == "xfail"
        return False

    @staticmethod
    def _extract_reason(node: ast.expr) -> Optional[str]:
        if not isinstance(node, ast.Call):
            return None
        for kw in node.keywords:
            if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
        return None


def scan_xfail_markers(directory: Path) -> list[str]:
    """Return violations for xfail markers without expiry dates."""
    violations: list[str] = []
    for py_file in sorted(directory.glob("**/*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree   = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _XfailVisitor(source.splitlines())
        visitor.visit(tree)
        for line_no, msg in visitor.violations:
            try:
                rel = py_file.relative_to(REPO_ROOT)
            except ValueError:
                rel = py_file
            violations.append(f"{rel}:{line_no}: {msg}")
    return violations


# ── Forbidden pattern scanner ─────────────────────────────────────────────────

def scan_forbidden_patterns(
    directories: list[Path],
    glob: str = "**/*.py",
) -> list[str]:
    """Return (file:line: pattern) violations for forbidden comment patterns."""
    violations: list[str] = []
    for directory in directories:
        for py_file in sorted(directory.glob(glob)):
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(lines, 1):
                # Only check comment portions
                comment_start = line.find("#")
                if comment_start == -1:
                    # Also check string literals for TODOs
                    comment_start = 0
                    check_text = line
                else:
                    check_text = line[comment_start:]
                for pattern, label in FORBIDDEN_PATTERNS:
                    if pattern.search(check_text):
                        try:
                            rel = py_file.relative_to(REPO_ROOT)
                        except ValueError:
                            rel = py_file
                        violations.append(f"{rel}:{lineno}: forbidden pattern '{label}'")
                        break
    return violations


# ── pytest runner ─────────────────────────────────────────────────────────────

def run_pytest_collect_failures(
    extra_args: list[str] | None = None,
) -> tuple[list[str], int]:
    """
    Run pytest and return (failed_test_ids, total_skip_count).
    Uses --tb=no -q for speed.
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=line", "-q", "--no-header",
    ] + (extra_args or [])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    output = result.stdout + result.stderr

    failed: list[str] = []
    skip_count = 0

    for line in output.splitlines():
        # Parse FAILED lines: "FAILED tests/foo.py::Bar::test_x - ..."
        if line.startswith("FAILED "):
            test_id = line.split(" ")[1].split(" ")[0]
            failed.append(test_id)
        # Parse skip summary: "N skipped"
        m = re.search(r"(\d+)\s+skipped", line)
        if m:
            skip_count += int(m.group(1))

    return failed, skip_count


def check_skip_threshold(skip_count: int, threshold: int) -> list[str]:
    if skip_count > threshold:
        return [
            f"Skip count {skip_count} exceeds threshold {threshold}. "
            "Review skipped tests and remove unnecessary skips."
        ]
    return []


def check_unregistered_failures(
    actual_failures: list[str],
    known_failures: list[dict],
) -> list[str]:
    """Return failures that are NOT in the known_failures list."""
    known_ids = {entry["test_id"] for entry in known_failures}
    unregistered = [f for f in actual_failures if f not in known_ids]
    return [
        f"Unregistered failure (new regression): {f}" for f in unregistered
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_hygiene_checks(
    skip_threshold: int = DEFAULT_SKIP_THRESHOLD,
    run_pytest: bool = True,
    scan_dirs: list[Path] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Run all hygiene checks.

    Returns:
        (violations, warnings)
    """
    violations: list[str] = []
    warnings:   list[str] = []

    known_failures = load_known_failures()

    # ── 1. Known failure expiry ───────────────────────────────────────────────
    exp_violations = check_known_failure_expiry(known_failures)
    violations.extend(exp_violations)

    # ── 2. xfail marker scan ──────────────────────────────────────────────────
    xfail_violations = scan_xfail_markers(TESTS_DIR)
    violations.extend(xfail_violations)

    # ── 3. Forbidden patterns ─────────────────────────────────────────────────
    check_dirs = scan_dirs or [TESTS_DIR, SRC_DIR]
    pattern_violations = scan_forbidden_patterns(check_dirs)
    violations.extend(pattern_violations)

    # ── 4. pytest run ─────────────────────────────────────────────────────────
    if run_pytest:
        actual_failures, skip_count = run_pytest_collect_failures()
        skip_violations = check_skip_threshold(skip_count, skip_threshold)
        violations.extend(skip_violations)
        unregistered = check_unregistered_failures(actual_failures, known_failures)
        violations.extend(unregistered)
        if actual_failures:
            logger.info(
                "pytest found %d failures (%d known, %d unregistered)",
                len(actual_failures),
                len(actual_failures) - len(unregistered),
                len(unregistered),
            )
    else:
        warnings.append("pytest not run (--no-run mode); skip count and failure checks skipped")

    return violations, warnings


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Test hygiene enforcer — zero-known-failure policy"
    )
    parser.add_argument("--skip-threshold", type=int, default=DEFAULT_SKIP_THRESHOLD,
                        help=f"Max allowed skips (default: {DEFAULT_SKIP_THRESHOLD})")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip running pytest (only static checks)")
    args = parser.parse_args()

    violations, warnings = run_hygiene_checks(
        skip_threshold=args.skip_threshold,
        run_pytest=not args.no_run,
    )

    for w in warnings:
        logger.warning("WARN: %s", w)
    for v in violations:
        logger.error("VIOLATION: %s", v)

    if violations:
        logger.error("Hygiene check FAILED: %d violation(s)", len(violations))
        sys.exit(1)

    logger.info("Hygiene check PASSED. warnings=%d", len(warnings))


if __name__ == "__main__":
    _cli()
