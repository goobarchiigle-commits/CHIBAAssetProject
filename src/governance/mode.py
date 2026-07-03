"""
src/governance/mode.py
Runtime mode enforcement — research / paper / production.

Read from env var RUNTIME_MODE. Default: "research" (most restrictive — fail safe).

Restrictions:
  production  — cannot load experimental strategies; must use config snapshot
  paper       — cannot mutate live capital snapshot; cannot write to state/live/
  research    — cannot write to state/live/ or config/snapshots/
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths that only production/paper modes may write to
_LIVE_STATE_DIRS: frozenset[str] = frozenset({
    "state/live",
    "config/snapshots",
})

# Paths that only production mode may create config snapshots in
_CONFIG_SNAPSHOT_DIRS: frozenset[str] = frozenset({
    "config/snapshots",
})


class RuntimeMode(str, Enum):
    RESEARCH   = "research"
    PAPER      = "paper"
    PRODUCTION = "production"


class ModeViolationError(RuntimeError):
    """Raised when a runtime mode constraint is violated."""


def get_current_mode() -> RuntimeMode:
    """
    Read RUNTIME_MODE from environment. Default: 'research' (fail-safe).
    Raises ValueError on unknown value.
    """
    raw = os.environ.get("RUNTIME_MODE", "research").lower().strip()
    try:
        mode = RuntimeMode(raw)
        logger.debug("[MODE] current_mode=%s", mode.value)
        return mode
    except ValueError:
        raise ValueError(
            f"Invalid RUNTIME_MODE='{raw}'. "
            f"Valid values: {[m.value for m in RuntimeMode]}"
        )


def assert_production_mode() -> None:
    """Raise ModeViolationError if not in production mode."""
    mode = get_current_mode()
    if mode != RuntimeMode.PRODUCTION:
        raise ModeViolationError(
            f"Production mode required. Current mode: '{mode.value}'. "
            "Set RUNTIME_MODE=production in .env to enable."
        )


def assert_can_load_strategy(strategy_status: str, mode: RuntimeMode | None = None) -> None:
    """
    Raise if the current mode cannot load a strategy with the given status.

    production mode: only 'active' strategies
    paper mode:      'active' and 'experimental'
    research mode:   any
    """
    if mode is None:
        mode = get_current_mode()

    if mode == RuntimeMode.PRODUCTION and strategy_status != "active":
        raise ModeViolationError(
            f"Production mode cannot load strategy with status='{strategy_status}'. "
            "Only 'active' strategies are permitted in production."
        )
    if mode == RuntimeMode.PAPER and strategy_status in ("deprecated", "archived"):
        raise ModeViolationError(
            f"Paper mode cannot load strategy with status='{strategy_status}'."
        )


def assert_not_research_writing_live_state(
    mode: RuntimeMode,
    target_path: Path,
) -> None:
    """
    Prevent research mode from writing to live state or config snapshot paths.
    Raises ModeViolationError if violated.
    """
    if mode != RuntimeMode.RESEARCH:
        return
    path_str = str(target_path).replace("\\", "/")
    for live_dir in _LIVE_STATE_DIRS:
        if live_dir in path_str:
            raise ModeViolationError(
                f"Research mode cannot write to live state path: {target_path}\n"
                f"Live state is restricted to production/paper modes."
            )


def assert_not_paper_mutating_capital(
    mode: RuntimeMode,
    target_path: Path,
) -> None:
    """
    Prevent paper mode from mutating production capital snapshot.
    Raises ModeViolationError if violated.
    """
    if mode != RuntimeMode.PAPER:
        return
    path_str = str(target_path).replace("\\", "/")
    if "state/live/capital" in path_str or "config/snapshots" in path_str:
        raise ModeViolationError(
            f"Paper mode cannot mutate production capital or config snapshots: {target_path}"
        )


def assert_research_cannot_read_live_fills(
    mode: RuntimeMode,
    target_path: Path,
) -> None:
    """Prevent research mode from reading live fills (contamination guard)."""
    if mode != RuntimeMode.RESEARCH:
        return
    path_str = str(target_path).replace("\\", "/")
    if "state/live/live_fills" in path_str or "state/live/live_metrics" in path_str:
        raise ModeViolationError(
            f"Research mode cannot read live fills/metrics: {target_path}"
        )
