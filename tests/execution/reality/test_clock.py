"""Tests: clock.py — ExecutionClock, TimestampRecord."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.clock import (
    ExecutionClock, TimestampRecord,
    SOURCE_SIGNAL, SOURCE_DECISION, SOURCE_SUBMIT, SOURCE_BROKER, SOURCE_EXCHANGE, SOURCE_FILL,
    ALL_SOURCES,
)

MONO = 1_000_000_000
UTC  = 1_767_225_600_000_000_000


def make_clock() -> ExecutionClock:
    return ExecutionClock(_override_monotonic_ns=MONO, _override_utc_ns=UTC)


class TestTimestampRecord:
    def test_frozen(self):
        clk = make_clock()
        ts = clk.now(SOURCE_SIGNAL)
        with pytest.raises(Exception):
            ts.source = "CHANGED"  # type: ignore

    def test_record_id_prefix(self):
        clk = make_clock()
        ts = clk.now(SOURCE_SIGNAL)
        assert ts.record_id.startswith("ts_")

    def test_elapsed_ns(self):
        clk = ExecutionClock(_override_monotonic_ns=1_000, _override_utc_ns=UTC)
        ts_a = clk.now(SOURCE_SIGNAL)
        clk2 = ExecutionClock(_override_monotonic_ns=2_000, _override_utc_ns=UTC)
        ts_b = clk2.now(SOURCE_FILL)
        assert ts_b.elapsed_ns(ts_a) == 1_000

    def test_to_dict_from_dict_roundtrip(self):
        clk = make_clock()
        ts = clk.now(SOURCE_SIGNAL)
        restored = TimestampRecord.from_dict(ts.to_dict())
        assert restored == ts

    def test_utc_iso_format(self):
        clk = make_clock()
        ts = clk.now(SOURCE_SIGNAL)
        assert "T" in ts.utc_iso and ts.utc_iso.endswith("Z")

    def test_source_stored(self):
        clk = make_clock()
        for source in ALL_SOURCES:
            ts = clk.now(source)
            assert ts.source == source

    def test_utc_ns_stored(self):
        clk = make_clock()
        ts = clk.now(SOURCE_SIGNAL)
        assert ts.utc_ns == UTC

    def test_monotonic_ns_stored(self):
        clk = make_clock()
        ts = clk.now(SOURCE_SIGNAL)
        assert ts.monotonic_ns == MONO


class TestExecutionClock:
    def test_unknown_source_raises(self):
        clk = ExecutionClock()
        with pytest.raises(ValueError):
            clk.now("UNKNOWN_SOURCE")

    def test_stamp_creates_record(self):
        clk = ExecutionClock()
        ts = clk.stamp(SOURCE_SIGNAL, MONO, UTC)
        assert ts.source == SOURCE_SIGNAL
        assert ts.monotonic_ns == MONO
        assert ts.utc_ns == UTC

    def test_stamp_unknown_source_raises(self):
        clk = ExecutionClock()
        with pytest.raises(ValueError):
            clk.stamp("BAD", MONO, UTC)

    def test_record_id_deterministic(self):
        clk = make_clock()
        ts1 = clk.now(SOURCE_SIGNAL)
        ts2 = clk.now(SOURCE_SIGNAL)
        assert ts1.record_id == ts2.record_id

    def test_different_sources_different_ids(self):
        clk = make_clock()
        ts_sig = clk.now(SOURCE_SIGNAL)
        ts_fill = clk.now(SOURCE_FILL)
        assert ts_sig.record_id != ts_fill.record_id

    def test_all_sources_valid(self):
        clk = make_clock()
        records = [clk.now(s) for s in ALL_SOURCES]
        assert len(records) == 6

    def test_real_clock_now_returns_record(self):
        clk = ExecutionClock()
        ts = clk.now(SOURCE_SIGNAL)
        assert ts.monotonic_ns > 0
        assert ts.utc_ns > 0
