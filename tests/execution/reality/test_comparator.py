"""Tests: comparator.py — PaperLiveComparator, ComparisonRecord."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.comparator import (
    PaperLiveComparator, ComparisonRecord,
    MODE_SIMULATED, MODE_PAPER, MODE_LIVE, ALL_MODES,
)
from tests.execution.reality.conftest import SESSION, SYM_A, SYM_B, TS, TS2, make_comparator


def compare(comp, sym=SYM_A, mode_a=MODE_SIMULATED, mode_b=MODE_LIVE,
            slip_a=2.0, slip_b=4.0, fill_a=1.0, fill_b=0.9,
            lat_a=10_000_000, lat_b=20_000_000, ts=TS):
    return comp.compare(SESSION, sym, mode_a, mode_b, slip_a, slip_b, fill_a, fill_b, lat_a, lat_b, ts)


class TestComparisonRecord:
    def test_frozen(self):
        comp = make_comparator()
        rec = compare(comp)
        with pytest.raises(Exception):
            rec.execution_divergence_score = 0.0  # type: ignore

    def test_record_id_prefix(self):
        comp = make_comparator()
        rec = compare(comp)
        assert rec.record_id.startswith("cmp_")

    def test_to_dict_from_dict_roundtrip(self):
        comp = make_comparator()
        rec = compare(comp)
        restored = ComparisonRecord.from_dict(rec.to_dict())
        assert restored.record_id == rec.record_id
        assert restored.execution_divergence_score == pytest.approx(rec.execution_divergence_score, abs=1e-9)


class TestPaperLiveComparator:
    def test_all_modes(self):
        assert len(ALL_MODES) == 3

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            PaperLiveComparator(divergence_alert_threshold=0.0)

    def test_unknown_mode_raises(self):
        comp = make_comparator()
        with pytest.raises(ValueError):
            comp.compare(SESSION, SYM_A, "BAD", MODE_LIVE, 2.0, 4.0, 1.0, 0.9, 10_000_000, 20_000_000, TS)

    def test_divergence_score_between_zero_and_one(self):
        comp = make_comparator()
        rec = compare(comp)
        assert 0.0 <= rec.execution_divergence_score <= 1.0

    def test_realism_degradation_between_zero_and_one(self):
        comp = make_comparator()
        rec = compare(comp)
        assert 0.0 <= rec.realism_degradation_score <= 1.0

    def test_identical_no_divergence(self):
        comp = make_comparator()
        rec = compare(comp, slip_a=5.0, slip_b=5.0, fill_a=1.0, fill_b=1.0, lat_a=10_000_000, lat_b=10_000_000)
        assert rec.execution_divergence_score == pytest.approx(0.0, abs=1e-6)
        assert rec.realism_degradation_score == pytest.approx(0.0, abs=1e-6)

    def test_mean_divergence_score(self):
        comp = make_comparator()
        compare(comp, ts=TS)
        compare(comp, ts=TS2)
        assert comp.mean_divergence_score() >= 0.0

    def test_mean_divergence_empty(self):
        comp = make_comparator()
        assert comp.mean_divergence_score() == 0.0

    def test_has_significant_divergence_false(self):
        comp = PaperLiveComparator(divergence_alert_threshold=0.90)
        compare(comp, slip_a=2.0, slip_b=2.1, fill_a=1.0, fill_b=0.99, lat_a=10_000_000, lat_b=10_000_001)
        assert not comp.has_significant_divergence()

    def test_has_significant_divergence_true(self):
        comp = PaperLiveComparator(divergence_alert_threshold=0.01)
        compare(comp, slip_a=1.0, slip_b=100.0, fill_a=1.0, fill_b=0.0, lat_a=1, lat_b=1_000_000_000)
        assert comp.has_significant_divergence()

    def test_record_count(self):
        comp = make_comparator()
        compare(comp, ts=TS)
        compare(comp, ts=TS2)
        assert comp.record_count() == 2

    def test_all_records_sorted(self):
        comp = make_comparator()
        compare(comp, ts=TS2)
        compare(comp, ts=TS)
        records = comp.all_records()
        ts_list = [r.recorded_at for r in records]
        assert ts_list == sorted(ts_list)

    def test_record_id_deterministic(self):
        comp1 = make_comparator()
        comp2 = make_comparator()
        r1 = compare(comp1)
        r2 = compare(comp2)
        assert r1.record_id == r2.record_id

    def test_to_summary_keys(self):
        comp = make_comparator()
        s = comp.to_summary()
        assert "mean_divergence_score" in s and "has_significant_divergence" in s
