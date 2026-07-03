"""Tests: ledger.py — DivergenceLedger, DivergenceRecord."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.ledger import (
    DivergenceLedger, DivergenceRecord,
    DIV_SLIPPAGE_DRIFT, DIV_LATENCY_DRIFT, DIV_SPREAD_WIDENING,
    DIV_REJECT_CLUSTERING, DIV_LIQUIDITY_COLLAPSE, DIV_SIGNAL_TIMING_DRIFT, DIV_QUEUE_DEGRADATION,
    ALL_DIVERGENCE_TYPES,
    SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL,
)
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, TS3, make_ledger


class TestDivergenceRecord:
    def test_frozen(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 5.0, 2.0, TS)
        with pytest.raises(Exception):
            rec.severity = "CHANGED"  # type: ignore

    def test_record_id_prefix(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 5.0, 2.0, TS)
        assert rec.record_id.startswith("div_")

    def test_to_dict_from_dict_roundtrip(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_LATENCY_DRIFT, SEVERITY_MEDIUM, 100.0, 50.0, TS)
        restored = DivergenceRecord.from_dict(rec.to_dict())
        assert restored.record_id == rec.record_id
        assert restored.divergence_type == rec.divergence_type


class TestDivergenceLedger:
    def test_all_divergence_types(self):
        assert len(ALL_DIVERGENCE_TYPES) == 7

    def test_unknown_type_raises(self):
        ledger = make_ledger()
        with pytest.raises(ValueError):
            ledger.record_divergence(SESSION, SYM_A, "BAD_TYPE", SEVERITY_LOW, 1.0, 1.0, TS)

    def test_record_returns_correct_type(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_HIGH, 10.0, 2.0, TS)
        assert rec.divergence_type == DIV_SLIPPAGE_DRIFT

    def test_divergence_ratio_baseline_zero(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 5.0, 0.0, TS)
        assert rec.divergence_ratio == float("inf")

    def test_divergence_ratio_both_zero(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 0.0, 0.0, TS)
        assert rec.divergence_ratio == 0.0

    def test_divergence_ratio_normal(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 4.0, 2.0, TS)
        assert rec.divergence_ratio == pytest.approx(2.0, abs=1e-9)

    def test_all_records_sorted_by_timestamp(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_LATENCY_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS2)
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        records = ledger.all_records()
        ts_list = [r.recorded_at for r in records]
        assert ts_list == sorted(ts_list)

    def test_records_of_type_filters(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        ledger.record_divergence(SESSION, SYM_A, DIV_LATENCY_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS2)
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS3)
        slippage_records = ledger.records_of_type(DIV_SLIPPAGE_DRIFT)
        assert len(slippage_records) == 2
        assert all(r.divergence_type == DIV_SLIPPAGE_DRIFT for r in slippage_records)

    def test_records_for_symbol(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        ledger.record_divergence(SESSION, SYM_B, DIV_LATENCY_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS2)
        assert len(ledger.records_for_symbol(SYM_A)) == 1
        assert len(ledger.records_for_symbol(SYM_B)) == 1

    def test_records_for_session(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        ledger.record_divergence("other_sess", SYM_A, DIV_LATENCY_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS2)
        assert len(ledger.records_for_session(SESSION)) == 1

    def test_divergence_count_total(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        ledger.record_divergence(SESSION, SYM_A, DIV_LATENCY_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS2)
        assert ledger.divergence_count() == 2

    def test_divergence_count_by_type(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        assert ledger.divergence_count(DIV_SLIPPAGE_DRIFT) == 1
        assert ledger.divergence_count(DIV_LATENCY_DRIFT) == 0

    def test_has_critical_false(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_HIGH, 1.0, 1.0, TS)
        assert not ledger.has_critical()

    def test_has_critical_true(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_CRITICAL, 1.0, 1.0, TS)
        assert ledger.has_critical()

    def test_severity_counts(self):
        ledger = make_ledger()
        ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS)
        ledger.record_divergence(SESSION, SYM_A, DIV_LATENCY_DRIFT, SEVERITY_HIGH, 1.0, 1.0, TS2)
        ledger.record_divergence(SESSION, SYM_A, DIV_SPREAD_WIDENING, SEVERITY_LOW, 1.0, 1.0, TS3)
        counts = ledger.severity_counts()
        assert counts[SEVERITY_LOW] == 2
        assert counts[SEVERITY_HIGH] == 1

    def test_metadata_stored(self):
        ledger = make_ledger()
        rec = ledger.record_divergence(SESSION, SYM_A, DIV_SLIPPAGE_DRIFT, SEVERITY_LOW, 1.0, 1.0, TS,
                                       metadata={"key": "val"})
        assert ("key", "val") in rec.metadata

    def test_to_summary_keys(self):
        ledger = make_ledger()
        s = ledger.to_summary()
        assert "total_records" in s and "has_critical" in s and "severity_counts" in s
