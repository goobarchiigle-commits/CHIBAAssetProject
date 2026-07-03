"""Monotonic execution clock — nanosecond-precision sequencing, UTC wall-clock for labeling."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# Timestamp sources — order in the execution chain
SOURCE_SIGNAL = "SIGNAL"
SOURCE_DECISION = "DECISION"
SOURCE_SUBMIT = "SUBMIT"
SOURCE_BROKER = "BROKER"
SOURCE_EXCHANGE = "EXCHANGE"
SOURCE_FILL = "FILL"

ALL_SOURCES = (SOURCE_SIGNAL, SOURCE_DECISION, SOURCE_SUBMIT, SOURCE_BROKER, SOURCE_EXCHANGE, SOURCE_FILL)


@dataclass(frozen=True)
class TimestampRecord:
    record_id: str      # "ts_" + SHA256[:12] of (source, monotonic_ns, utc_ns)
    source: str
    monotonic_ns: int   # time.monotonic_ns() — for ordering only
    utc_ns: int         # nanoseconds since Unix epoch — wall-clock label
    utc_iso: str        # human-readable UTC

    def elapsed_ns(self, other: "TimestampRecord") -> int:
        """Nanoseconds elapsed from other → self (monotonic)."""
        return self.monotonic_ns - other.monotonic_ns

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "monotonic_ns": self.monotonic_ns,
            "utc_ns": self.utc_ns,
            "utc_iso": self.utc_iso,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimestampRecord":
        return cls(
            record_id=d["record_id"],
            source=d["source"],
            monotonic_ns=d["monotonic_ns"],
            utc_ns=d["utc_ns"],
            utc_iso=d["utc_iso"],
        )


def _make_record_id(source: str, monotonic_ns: int, utc_ns: int) -> str:
    raw = f"{source}:{monotonic_ns}:{utc_ns}"
    return "ts_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _utc_ns() -> int:
    dt = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return int(delta.total_seconds() * 1_000_000_000)


def _utc_iso(utc_ns: int) -> str:
    seconds = utc_ns / 1_000_000_000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ExecutionClock:
    """Monotonic-ns sequencer with UTC wall-clock labels. Never uses time.time() for ordering."""

    def __init__(self, *, _override_monotonic_ns: Optional[int] = None, _override_utc_ns: Optional[int] = None):
        self._mono_override = _override_monotonic_ns
        self._utc_override = _override_utc_ns

    def now(self, source: str) -> TimestampRecord:
        if source not in ALL_SOURCES:
            raise ValueError(f"Unknown source: {source!r}")
        mono = self._mono_override if self._mono_override is not None else time.monotonic_ns()
        utc = self._utc_override if self._utc_override is not None else _utc_ns()
        record_id = _make_record_id(source, mono, utc)
        return TimestampRecord(
            record_id=record_id,
            source=source,
            monotonic_ns=mono,
            utc_ns=utc,
            utc_iso=_utc_iso(utc),
        )

    def stamp(self, source: str, monotonic_ns: int, utc_ns: int) -> TimestampRecord:
        """Create a TimestampRecord from explicit values (for replay / test injection)."""
        if source not in ALL_SOURCES:
            raise ValueError(f"Unknown source: {source!r}")
        record_id = _make_record_id(source, monotonic_ns, utc_ns)
        return TimestampRecord(
            record_id=record_id,
            source=source,
            monotonic_ns=monotonic_ns,
            utc_ns=utc_ns,
            utc_iso=_utc_iso(utc_ns),
        )
