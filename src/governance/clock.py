"""
src/governance/clock.py
Clock isolation for deterministic strategy execution.

Strategy logic must use ClockSource.now() instead of datetime.now().
This allows tests and reproducibility runs to inject a fixed or sequential clock.

Usage:
    from src.governance.clock import get_clock, set_clock, FixedClock

    # In strategy:
    ts = get_clock().now()

    # In tests:
    set_clock(FixedClock(datetime(2024, 1, 15, 9, 0, 0, tzinfo=JST)))
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# ── Abstract base ─────────────────────────────────────────────────────────────

class ClockSource(ABC):
    """Abstract clock — all strategy time access goes through this."""

    @abstractmethod
    def now(self, tz: timezone = JST) -> datetime:
        """Return current datetime."""

    def today(self, tz: timezone = JST) -> datetime:
        """Return current date at midnight."""
        return self.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)


# ── Concrete implementations ──────────────────────────────────────────────────

class ProductionClock(ClockSource):
    """Real wall-clock time. Only for production/paper entrypoints."""

    def now(self, tz: timezone = JST) -> datetime:
        return datetime.now(tz)


class FixedClock(ClockSource):
    """
    Returns the same timestamp every call.
    Use in tests or reproducibility runs to freeze time.
    """

    def __init__(self, fixed_time: datetime) -> None:
        if fixed_time.tzinfo is None:
            fixed_time = fixed_time.replace(tzinfo=JST)
        self._time = fixed_time

    def now(self, tz: timezone = JST) -> datetime:
        return self._time.astimezone(tz)

    def __repr__(self) -> str:
        return f"FixedClock({self._time.isoformat()})"


class SequentialClock(ClockSource):
    """
    Advances by a fixed step each call.
    Useful for testing time-dependent logic with multiple steps.
    """

    def __init__(
        self,
        start: datetime,
        step_seconds: int = 60,
    ) -> None:
        if start.tzinfo is None:
            start = start.replace(tzinfo=JST)
        self._current = start
        self._step = timedelta(seconds=step_seconds)
        self._call_count = 0

    def now(self, tz: timezone = JST) -> datetime:
        result = self._current.astimezone(tz)
        self._current += self._step
        self._call_count += 1
        return result

    @property
    def call_count(self) -> int:
        return self._call_count

    def __repr__(self) -> str:
        return f"SequentialClock(current={self._current.isoformat()}, step={self._step})"


class IteratorClock(ClockSource):
    """
    Yields timestamps from an explicit sequence.
    Raises StopIteration (wrapped as RuntimeError) when exhausted.
    """

    def __init__(self, timestamps: list[datetime]) -> None:
        self._iter: Iterator[datetime] = iter(timestamps)
        self._count = 0

    def now(self, tz: timezone = JST) -> datetime:
        try:
            ts = next(self._iter)
        except StopIteration:
            raise RuntimeError("IteratorClock exhausted — no more timestamps in sequence")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=JST)
        self._count += 1
        return ts.astimezone(tz)

    @property
    def count(self) -> int:
        return self._count


# ── Module-level registry ─────────────────────────────────────────────────────

_ACTIVE_CLOCK: ClockSource = ProductionClock()


def get_clock() -> ClockSource:
    """Return the currently active clock."""
    return _ACTIVE_CLOCK


def set_clock(clock: ClockSource) -> None:
    """
    Replace the active clock.

    Args:
        clock: Any ClockSource implementation.

    Note: Call reset_clock() in test teardown to restore ProductionClock.
    """
    global _ACTIVE_CLOCK
    if not isinstance(clock, ClockSource):
        raise TypeError(f"Expected ClockSource, got {type(clock).__name__}")
    _ACTIVE_CLOCK = clock
    logger.debug("[CLOCK] active clock set to %r", clock)


def reset_clock() -> None:
    """Restore ProductionClock. Call in test teardown."""
    global _ACTIVE_CLOCK
    _ACTIVE_CLOCK = ProductionClock()
    logger.debug("[CLOCK] reset to ProductionClock")


def now(tz: timezone = JST) -> datetime:
    """Convenience wrapper — strategy code calls clock.now() via this."""
    return _ACTIVE_CLOCK.now(tz)
