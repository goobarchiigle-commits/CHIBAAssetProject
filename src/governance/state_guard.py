"""
src/governance/state_guard.py
Live / research state separation enforcement.

Prevents:
  - shared capital snapshot between live and research
  - research code writing to live state dirs
  - live state files being accessed from research context

State directories:
  state/live/      — production fills, capital, metrics
  state/research/  — research capital, backtest checkpoints
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src.governance.mode import RuntimeMode, ModeViolationError

logger = logging.getLogger(__name__)

REPO_ROOT          = Path(__file__).resolve().parent.parent.parent
LIVE_STATE_DIR     = REPO_ROOT / "state" / "live"
RESEARCH_STATE_DIR = REPO_ROOT / "state" / "research"

# Files that must ONLY exist in live state
LIVE_ONLY_FILES: frozenset[str] = frozenset({
    "capital_snapshot.json",
    "live_fills.jsonl",
    "live_metrics.jsonl",
    "live_checkpoints",
})

# Files that must ONLY exist in research state
RESEARCH_ONLY_FILES: frozenset[str] = frozenset({
    "research_capital.json",
    "research_fills.jsonl",
    "backtest_checkpoints",
})


def get_state_path(filename: str, mode: RuntimeMode) -> Path:
    """
    Return mode-appropriate state file path.

    production  → state/live/filename
    paper       → state/research/filename  (cannot access live-only)
    research    → state/research/filename  (cannot access live-only)

    Raises ModeViolationError on cross-contamination attempts.
    """
    if mode == RuntimeMode.PRODUCTION:
        if filename in RESEARCH_ONLY_FILES:
            raise ModeViolationError(
                f"Production mode cannot access research-only file: '{filename}'"
            )
        path = LIVE_STATE_DIR / filename
    else:
        if filename in LIVE_ONLY_FILES:
            raise ModeViolationError(
                f"{mode.value} mode cannot access live-only file: '{filename}'. "
                "Live state is restricted to production mode."
            )
        path = RESEARCH_STATE_DIR / filename

    logger.debug("[STATE_GUARD] mode=%s file=%s → %s", mode.value, filename, path)
    return path


def assert_no_shared_capital(
    live_dir: Path = LIVE_STATE_DIR,
    research_dir: Path = RESEARCH_STATE_DIR,
) -> None:
    """
    Assert that live and research capital snapshots are distinct files with distinct content.

    Raises ModeViolationError if they share the same path.
    Logs a warning (not error) if content is identical (may be intentional sync).
    """
    live_cap     = live_dir     / "capital_snapshot.json"
    research_cap = research_dir / "research_capital.json"

    if live_dir.resolve() == research_dir.resolve():
        raise ModeViolationError(
            "Live and research state must not share the same directory. "
            f"Both resolve to: {live_dir.resolve()}"
        )

    if live_cap.resolve() == research_cap.resolve():
        raise ModeViolationError(
            "Live and research capital snapshots must not be the same file. "
            f"Both resolve to: {live_cap.resolve()}"
        )

    if live_cap.exists() and research_cap.exists():
        live_hash     = hashlib.sha256(live_cap.read_bytes()).hexdigest()
        research_hash = hashlib.sha256(research_cap.read_bytes()).hexdigest()
        if live_hash == research_hash:
            logger.warning(
                "[STATE_GUARD] WARN: live capital_snapshot and research_capital have "
                "identical content. Verify they are not unintentionally shared."
            )

    logger.debug("[STATE_GUARD] capital separation OK: %s ≠ %s", live_cap, research_cap)


def assert_no_live_files_in_research_dir(
    live_dir: Path = LIVE_STATE_DIR,
    research_dir: Path = RESEARCH_STATE_DIR,
) -> list[str]:
    """
    Return list of live-only filenames found in research directory.
    Empty list = OK.
    """
    violations: list[str] = []
    if not research_dir.exists():
        return violations
    for fname in LIVE_ONLY_FILES:
        if (research_dir / fname).exists():
            violations.append(fname)
            logger.error(
                "[STATE_GUARD] Live-only file found in research dir: %s/%s",
                research_dir, fname,
            )
    return violations


def assert_no_research_files_in_live_dir(
    live_dir: Path = LIVE_STATE_DIR,
    research_dir: Path = RESEARCH_STATE_DIR,
) -> list[str]:
    """
    Return list of research-only filenames found in live directory.
    Empty list = OK.
    """
    violations: list[str] = []
    if not live_dir.exists():
        return violations
    for fname in RESEARCH_ONLY_FILES:
        if (live_dir / fname).exists():
            violations.append(fname)
            logger.error(
                "[STATE_GUARD] Research-only file found in live dir: %s/%s",
                live_dir, fname,
            )
    return violations
