"""Tests: snapshot.py — ExecutionSnapshot, build_snapshot."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.snapshot import (
    ExpectedFill, ActualFill, ExecutionTimestamps, ExecutionSnapshot, build_snapshot,
)
from tests.execution.reality.conftest import (
    make_expected, make_actual, make_timestamps, make_snapshot,
    SESSION, SYM_A, TS, SIGNAL_NS, FILL_NS,
)


class TestExpectedFill:
    def test_frozen(self):
        ef = make_expected()
        with pytest.raises(Exception):
            ef.quantity = 999  # type: ignore

    def test_to_dict_from_dict_roundtrip(self):
        ef = make_expected()
        restored = ExpectedFill.from_dict(ef.to_dict())
        assert restored == ef


class TestActualFill:
    def test_frozen(self):
        af = make_actual()
        with pytest.raises(Exception):
            af.fill_price = 0.0  # type: ignore

    def test_to_dict_from_dict_roundtrip(self):
        af = make_actual()
        restored = ActualFill.from_dict(af.to_dict())
        assert restored == af


class TestExecutionTimestamps:
    def test_total_latency(self):
        ts = make_timestamps()
        assert ts.total_latency_ns() == FILL_NS - SIGNAL_NS

    def test_signal_to_submit(self):
        from tests.execution.reality.conftest import SUBMIT_NS
        ts = make_timestamps()
        assert ts.signal_to_submit_ns() == SUBMIT_NS - SIGNAL_NS

    def test_submit_to_fill(self):
        from tests.execution.reality.conftest import SUBMIT_NS
        ts = make_timestamps()
        assert ts.submit_to_fill_ns() == FILL_NS - SUBMIT_NS

    def test_frozen(self):
        ts = make_timestamps()
        with pytest.raises(Exception):
            ts.signal_ns = 0  # type: ignore

    def test_to_dict_from_dict_roundtrip(self):
        ts = make_timestamps()
        restored = ExecutionTimestamps.from_dict(ts.to_dict())
        assert restored == ts


class TestBuildSnapshot:
    def test_snapshot_id_prefix(self):
        snap = make_snapshot()
        assert snap.snapshot_id.startswith("exsnap_")

    def test_frozen(self):
        snap = make_snapshot()
        with pytest.raises(Exception):
            snap.slippage_bps = 0.0  # type: ignore

    def test_slippage_bps_buy(self):
        expected = make_expected(price=1000.0)
        actual = make_actual(fill_price=1001.0)
        ts = make_timestamps()
        snap = build_snapshot(SESSION, expected, actual, ts, "1.0.0", TS)
        # (1001 - 1000) / 1000 * 10000 = 10 bps
        assert snap.slippage_bps == pytest.approx(10.0, abs=1e-4)

    def test_slippage_bps_sell(self):
        ef = ExpectedFill(symbol=SYM_A, side="SELL", quantity=100,
                          expected_price=1000.0, expected_slippage_bps=3.0, expected_latency_ns=20_000_000)
        af = ActualFill(symbol=SYM_A, side="SELL", filled_quantity=100, fill_price=999.0,
                        fill_latency_ns=20_000_000, broker_latency_ns=8_000_000, rejected=False, partial=False)
        ts = make_timestamps()
        snap = build_snapshot(SESSION, ef, af, ts, "1.0.0", TS)
        # -(999-1000)/1000*10000 = 10 bps (positive for SELL under-fill)
        assert snap.slippage_bps == pytest.approx(10.0, abs=1e-4)

    def test_fill_ratio_full(self):
        snap = make_snapshot()
        assert snap.fill_ratio == pytest.approx(1.0, abs=1e-9)

    def test_fill_ratio_partial(self):
        expected = make_expected(qty=100)
        actual = make_actual(qty=50)
        ts = make_timestamps()
        snap = build_snapshot(SESSION, expected, actual, ts, "1.0.0", TS)
        assert snap.fill_ratio == pytest.approx(0.5, abs=1e-6)

    def test_snapshot_id_deterministic(self):
        s1 = make_snapshot()
        s2 = make_snapshot()
        assert s1.snapshot_id == s2.snapshot_id

    def test_checksum_nonempty(self):
        snap = make_snapshot()
        assert len(snap.checksum) == 16

    def test_to_dict_from_dict_roundtrip(self):
        snap = make_snapshot()
        restored = ExecutionSnapshot.from_dict(snap.to_dict())
        assert restored.snapshot_id == snap.snapshot_id
        assert restored.slippage_bps == snap.slippage_bps
        assert restored.fill_ratio == snap.fill_ratio

    def test_schema_version_stored(self):
        snap = make_snapshot()
        assert snap.schema_version == "1.0.0"

    def test_expected_price_zero_no_crash(self):
        ef = ExpectedFill(symbol=SYM_A, side="BUY", quantity=100,
                          expected_price=0.0, expected_slippage_bps=0.0, expected_latency_ns=0)
        af = make_actual()
        ts = make_timestamps()
        snap = build_snapshot(SESSION, ef, af, ts, "1.0.0", TS)
        assert snap.slippage_bps == 0.0
