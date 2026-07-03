"""
src/live/safe_cleanup.py

Isolated per-step cleanup utility for LIVE execution shutdown.

Design:
    - Each cleanup step is called independently; a failure in one step
      must not prevent subsequent steps from running.
    - All exceptions are caught, logged with context, and suppressed.
    - Callers get a summary list of (name, success, error) triples so
      the watchdog log can record which steps failed.

Usage::

    results = safe_cleanup(
        ("stop_heartbeat",    heartbeat.stop),
        ("release_lock",      lock.release),
        ("flush_registry",    lambda: registry.mark_failed(coi, "shutdown")),
    )
    failed = [name for name, ok, _ in results if not ok]
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


CleanupResult = Tuple[str, bool, Optional[str]]


def safe_cleanup_step(name: str, fn: Callable, *args, **kwargs) -> CleanupResult:
    """
    Execute fn(*args, **kwargs) and return (name, success, error_str).
    Never raises.
    """
    try:
        fn(*args, **kwargs)
        logger.debug("[CLEANUP] %s — ok", name)
        return (name, True, None)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[CLEANUP] %s — FAILED: %s", name, msg)
        return (name, False, msg)


def safe_cleanup(*steps: Tuple[str, Callable]) -> List[CleanupResult]:
    """
    Run a sequence of (name, callable) pairs, each isolated from the others.

    Args:
        steps: tuples of (step_name, zero-arg callable)

    Returns:
        List of CleanupResult: (name, success, error_or_None)
    """
    results: List[CleanupResult] = []
    for name, fn in steps:
        results.append(safe_cleanup_step(name, fn))
    return results
