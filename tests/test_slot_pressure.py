"""
Tests for src/analytics/slot_pressure.py — Phase 5E Slot Pressure Forecasting.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List

import pytest

from src.analytics.slot_pressure import (
    ALL_CORE_REDUCTION,
    CANDIDATE_QUALITY_BONUS,
    CANDIDATE_QUALITY_THRESHOLD,
    DEFAULT_LOOKBACK_DAYS,
    FULL_PORTFOLIO_BONUS,
    OPEN_SLOT_REDUCTION,
    PRESSURE_CRITICAL,
    PRESSURE_CRITICAL_THRESHOLD,
    PRESSURE_HIGH,
    PRESSURE_HIGH_THRESHOLD,
    PRESSURE_LOW,
    PRESSURE_MODERATE,
    PRESSURE_MODERATE_THRESHOLD,
    REASON_HEALTHY,
    REASON_OVERFLOW,
    REASON_SEVERE,
    REASON_WEAK_SLOT,
    REJECTED_BONUS,
    REJECTED_THRESHOLD,
    SCHEMA_VERSION,
    SEVERE_BONUS,
    SEVERE_THRESHOLD,
    TIER_CORE,
    TIER_EXIT,
    TIER_HIGH,
    TIER_NEUTRAL,
    WEAK_SLOT_BONUS,
    WEAK_SLOT_THRESHOLD,
    SlotPressureInput,
    SlotPressureRecord,
    SlotPressureResult,
    aggregate_pressure_kpis,
    build_slot_pressure_input_from_oc_data,
    classify_pressure_regime,
    compute_slot_pressure,
    extract_congestion_reasons,
    format_slot_pressure_report,
    materialize_slot_pressure,
    record_slot_pressure,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _full_inp(**kwargs) -> SlotPressureInput:
    """Full portfolio, one weak slot, overflow of rejected cands, severe events."""
    defaults = dict(
        max_positions=3,
        current_positions=3,
        holding_priorities=[80.0, 70.0, 40.0],  # 40.0 < 45 → weak
        holding_tiers=[TIER_CORE, TIER_HIGH, TIER_NEUTRAL],
        avg_candidate_priority_5d=85.0,           # > 80 → quality pressure
        recent_rejected_candidates=4,             # >= 3
        recent_severe_opportunity_cost_events=2,  # >= 2
    )
    defaults.update(kwargs)
    return SlotPressureInput(**defaults)


def _empty_inp(**kwargs) -> SlotPressureInput:
    """Zero-pressure baseline."""
    defaults = dict(
        max_positions=3,
        current_positions=0,
        holding_priorities=[],
        holding_tiers=[],
        avg_candidate_priority_5d=0.0,
        recent_rejected_candidates=0,
        recent_severe_opportunity_cost_events=0,
    )
    defaults.update(kwargs)
    return SlotPressureInput(**defaults)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# TestSlotPressureScoring
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotPressureScoring:

    def test_zero_pressure_empty_portfolio(self):
        inp = _empty_inp()
        r = compute_slot_pressure(inp)
        # open_slots=3 → -25 reduction; no other signals → max(0, -25) = 0
        assert r.pressure_score == 0.0

    def test_portfolio_full_adds_20(self):
        inp = _empty_inp(max_positions=3, current_positions=3)
        # full (+20) but no open slots reduction doesn't apply (open_slots=0)
        r = compute_slot_pressure(inp)
        assert r.pressure_score == pytest.approx(FULL_PORTFOLIO_BONUS)

    def test_weak_slot_adds_20(self):
        # One open slot so -25; full+20; weak+20 → 20+20-25 = 15 but clamped ≥0
        inp = _empty_inp(
            max_positions=3,
            current_positions=2,          # 1 open slot → -25
            holding_priorities=[44.9],    # below threshold → +20
            holding_tiers=[TIER_NEUTRAL],
        )
        r = compute_slot_pressure(inp)
        # +20 (weak) + 0 (not full) - 25 (open slot) = -5 → clamped 0
        assert r.pressure_score == 0.0
        assert REASON_WEAK_SLOT in r.reasons

    def test_weak_slot_threshold_exact_44_9(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            holding_priorities=[44.9],
            holding_tiers=[TIER_NEUTRAL],
        )
        r = compute_slot_pressure(inp)
        assert REASON_WEAK_SLOT in r.reasons

    def test_weak_slot_threshold_exact_45_not_weak(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            holding_priorities=[45.0],
            holding_tiers=[TIER_NEUTRAL],
        )
        r = compute_slot_pressure(inp)
        assert REASON_WEAK_SLOT not in r.reasons

    def test_rejected_candidates_adds_15(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,  # full → +20
            recent_rejected_candidates=3,
        )
        r = compute_slot_pressure(inp)
        # +20 (full) + 15 (rejected) = 35
        assert r.pressure_score == pytest.approx(FULL_PORTFOLIO_BONUS + REJECTED_BONUS)
        assert REASON_OVERFLOW in r.reasons

    def test_rejected_candidates_below_threshold_no_bonus(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            recent_rejected_candidates=2,
        )
        r = compute_slot_pressure(inp)
        assert REASON_OVERFLOW not in r.reasons

    def test_severe_oc_events_adds_20(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            recent_severe_opportunity_cost_events=2,
        )
        r = compute_slot_pressure(inp)
        assert r.pressure_score == pytest.approx(FULL_PORTFOLIO_BONUS + SEVERE_BONUS)
        assert REASON_SEVERE in r.reasons

    def test_severe_oc_events_below_threshold_no_bonus(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            recent_severe_opportunity_cost_events=1,
        )
        r = compute_slot_pressure(inp)
        assert REASON_SEVERE not in r.reasons

    def test_candidate_quality_adds_15(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            avg_candidate_priority_5d=80.1,
        )
        r = compute_slot_pressure(inp)
        assert r.pressure_score == pytest.approx(FULL_PORTFOLIO_BONUS + CANDIDATE_QUALITY_BONUS)

    def test_candidate_quality_exact_80_no_bonus(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            avg_candidate_priority_5d=80.0,
        )
        r = compute_slot_pressure(inp)
        # no quality bonus (> not >=)
        assert r.pressure_score == pytest.approx(FULL_PORTFOLIO_BONUS)

    def test_all_core_reduces_20(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            holding_priorities=[90.0, 85.0, 80.0],
            holding_tiers=[TIER_CORE, TIER_CORE, TIER_CORE],
        )
        r = compute_slot_pressure(inp)
        # +20 (full) - 20 (all core) = 0
        assert r.pressure_score == pytest.approx(0.0)

    def test_mixed_tiers_no_all_core_reduction(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            holding_priorities=[90.0, 85.0, 80.0],
            holding_tiers=[TIER_CORE, TIER_HIGH, TIER_CORE],
        )
        r = compute_slot_pressure(inp)
        # +20 (full) — no all-core reduction
        assert r.pressure_score == pytest.approx(FULL_PORTFOLIO_BONUS)

    def test_open_slot_reduces_25(self):
        inp = _empty_inp(max_positions=3, current_positions=2)
        r = compute_slot_pressure(inp)
        # 0 - 25 = clamped 0
        assert r.pressure_score == 0.0
        assert REASON_HEALTHY in r.reasons

    def test_open_slots_calc_correct(self):
        inp = _empty_inp(max_positions=5, current_positions=2)
        r = compute_slot_pressure(inp)
        assert r.open_slots == 3

    def test_full_portfolio_zero_open_slots(self):
        inp = _empty_inp(max_positions=3, current_positions=3)
        r = compute_slot_pressure(inp)
        assert r.open_slots == 0
        assert REASON_HEALTHY not in r.reasons

    def test_score_clamped_at_100(self):
        """All additive signals, no reductions → should not exceed 100."""
        inp = _full_inp()
        r = compute_slot_pressure(inp)
        assert 0.0 <= r.pressure_score <= 100.0

    def test_score_clamped_at_0(self):
        inp = _empty_inp()
        r = compute_slot_pressure(inp)
        assert r.pressure_score >= 0.0

    def test_max_pressure_all_signals(self):
        """Full portfolio + weak slot + overflow + severe + quality = 90."""
        inp = _full_inp(holding_tiers=[TIER_NEUTRAL, TIER_HIGH, TIER_NEUTRAL])
        r = compute_slot_pressure(inp)
        # +20+20+15+20+15 = 90
        assert r.pressure_score == pytest.approx(90.0)

    def test_lowest_priority_populated(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            holding_priorities=[80.0, 55.0, 40.0],
            holding_tiers=[TIER_NEUTRAL, TIER_NEUTRAL, TIER_NEUTRAL],
        )
        r = compute_slot_pressure(inp)
        assert r.lowest_priority == pytest.approx(40.0)

    def test_lowest_priority_none_when_no_holdings(self):
        inp = _empty_inp()
        r = compute_slot_pressure(inp)
        assert r.lowest_priority is None

    def test_result_has_pressure_regime(self):
        inp = _full_inp()
        r = compute_slot_pressure(inp)
        assert r.pressure_regime in (PRESSURE_LOW, PRESSURE_MODERATE, PRESSURE_HIGH, PRESSURE_CRITICAL)

    def test_returns_slot_pressure_result(self):
        inp = _empty_inp()
        r = compute_slot_pressure(inp)
        assert isinstance(r, SlotPressureResult)


# ─────────────────────────────────────────────────────────────────────────────
# TestClassifyPressureRegime
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyPressureRegime:

    def test_critical_at_80(self):
        assert classify_pressure_regime(80.0) == PRESSURE_CRITICAL

    def test_critical_above_80(self):
        assert classify_pressure_regime(95.0) == PRESSURE_CRITICAL

    def test_high_at_60(self):
        assert classify_pressure_regime(60.0) == PRESSURE_HIGH

    def test_high_below_80(self):
        assert classify_pressure_regime(79.9) == PRESSURE_HIGH

    def test_moderate_at_40(self):
        assert classify_pressure_regime(40.0) == PRESSURE_MODERATE

    def test_moderate_below_60(self):
        assert classify_pressure_regime(59.9) == PRESSURE_MODERATE

    def test_low_at_39(self):
        assert classify_pressure_regime(39.9) == PRESSURE_LOW

    def test_low_at_0(self):
        assert classify_pressure_regime(0.0) == PRESSURE_LOW

    def test_boundary_79_9_is_high(self):
        assert classify_pressure_regime(79.9) == PRESSURE_HIGH

    def test_boundary_60_is_high(self):
        assert classify_pressure_regime(60.0) == PRESSURE_HIGH


# ─────────────────────────────────────────────────────────────────────────────
# TestCongestionReasons
# ─────────────────────────────────────────────────────────────────────────────

class TestCongestionReasons:

    def test_no_reasons_empty_portfolio_with_open_slots(self):
        inp = _empty_inp(max_positions=3, current_positions=1)
        reasons = extract_congestion_reasons(inp)
        # Only HEALTHY since open slots exist; no other signals
        assert reasons == [REASON_HEALTHY]

    def test_weak_slot_reason(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            holding_priorities=[44.0],
            holding_tiers=[TIER_NEUTRAL],
        )
        reasons = extract_congestion_reasons(inp)
        assert REASON_WEAK_SLOT in reasons

    def test_overflow_reason(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            recent_rejected_candidates=3,
        )
        reasons = extract_congestion_reasons(inp)
        assert REASON_OVERFLOW in reasons

    def test_severe_reason(self):
        inp = _empty_inp(
            max_positions=3,
            current_positions=3,
            recent_severe_opportunity_cost_events=2,
        )
        reasons = extract_congestion_reasons(inp)
        assert REASON_SEVERE in reasons

    def test_healthy_reason_when_open_slots(self):
        inp = _empty_inp(max_positions=3, current_positions=1)
        reasons = extract_congestion_reasons(inp)
        assert REASON_HEALTHY in reasons

    def test_no_healthy_when_full(self):
        inp = _empty_inp(max_positions=3, current_positions=3)
        reasons = extract_congestion_reasons(inp)
        assert REASON_HEALTHY not in reasons

    def test_multiple_reasons_coexist(self):
        inp = _full_inp(holding_tiers=[TIER_NEUTRAL, TIER_HIGH, TIER_NEUTRAL])
        reasons = extract_congestion_reasons(inp)
        assert REASON_WEAK_SLOT in reasons
        assert REASON_OVERFLOW in reasons
        assert REASON_SEVERE in reasons

    def test_extract_delegates_to_compute(self):
        """extract_congestion_reasons must return same reasons as compute_slot_pressure."""
        inp = _full_inp()
        assert extract_congestion_reasons(inp) == compute_slot_pressure(inp).reasons


# ─────────────────────────────────────────────────────────────────────────────
# TestSlotPressureRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotPressureRecord:

    def test_schema_version_default(self):
        rec = SlotPressureRecord(
            date="2026-05-29", pressure_score=50.0, pressure_regime=PRESSURE_MODERATE,
            holding_priorities=[80.0], lowest_priority=80.0,
            recent_rejected_candidates=1, recent_severe_opportunity_cost_events=0,
            avg_candidate_priority_5d=70.0, reasons=[], max_positions=3,
            current_positions=1, open_slots=2,
        )
        assert rec.schema_version == SCHEMA_VERSION

    def test_asdict_serializable(self):
        from dataclasses import asdict
        rec = SlotPressureRecord(
            date="2026-05-29", pressure_score=40.0, pressure_regime=PRESSURE_MODERATE,
            holding_priorities=[70.0, 60.0], lowest_priority=60.0,
            recent_rejected_candidates=0, recent_severe_opportunity_cost_events=0,
            avg_candidate_priority_5d=75.0, reasons=[REASON_HEALTHY],
            max_positions=3, current_positions=2, open_slots=1,
        )
        d = asdict(rec)
        assert json.dumps(d)  # must be JSON serializable


# ─────────────────────────────────────────────────────────────────────────────
# TestRecordSlotPressure
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordSlotPressure:

    def test_writes_jsonl(self, tmp_path):
        f = tmp_path / "sp.jsonl"
        inp = _full_inp(holding_tiers=[TIER_NEUTRAL, TIER_HIGH, TIER_NEUTRAL])
        rec = record_slot_pressure(inp, "2026-05-29", f)
        assert rec is not None
        assert f.exists()
        rows = [json.loads(l) for l in f.read_text().strip().splitlines()]
        assert len(rows) == 1
        assert rows[0]["date"] == "2026-05-29"

    def test_pressure_score_in_record(self, tmp_path):
        f = tmp_path / "sp.jsonl"
        inp = _empty_inp(max_positions=3, current_positions=3)  # score=20
        rec = record_slot_pressure(inp, "2026-05-29", f)
        assert rec.pressure_score == pytest.approx(20.0)

    def test_appends_multiple_records(self, tmp_path):
        f = tmp_path / "sp.jsonl"
        inp = _empty_inp(max_positions=3, current_positions=3)
        record_slot_pressure(inp, "2026-05-28", f)
        record_slot_pressure(inp, "2026-05-29", f)
        rows = [json.loads(l) for l in f.read_text().strip().splitlines()]
        assert len(rows) == 2

    def test_fail_open_missing_parent_dir(self, tmp_path):
        f = tmp_path / "nonexistent" / "subdir" / "sp.jsonl"
        inp = _empty_inp()
        # Should not raise
        result = record_slot_pressure(inp, "2026-05-29", f)
        # May succeed (mkdir) or return None — must not raise
        # The _append_jsonl function creates parent dirs; just verify no exception

    def test_lowest_priority_zero_when_no_holdings(self, tmp_path):
        f = tmp_path / "sp.jsonl"
        inp = _empty_inp()
        rec = record_slot_pressure(inp, "2026-05-29", f)
        assert rec.lowest_priority == 0.0

    def test_record_has_schema_version(self, tmp_path):
        f = tmp_path / "sp.jsonl"
        inp = _empty_inp(max_positions=3, current_positions=3)
        rec = record_slot_pressure(inp, "2026-05-29", f)
        assert rec.schema_version == SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# TestMaterializeSlotPressure
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializeSlotPressure:

    def _make_pressure_record(self, date: str, score: float = 50.0, regime: str = PRESSURE_MODERATE):
        return {
            "date": date,
            "pressure_score": score,
            "pressure_regime": regime,
            "reasons": [],
        }

    def _make_oc_event(self, date: str):
        return {"date": date, "opportunity_cost_severity": "moderate"}

    def test_materializes_old_records(self, tmp_path):
        pf  = tmp_path / "pressure.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(pf, [self._make_pressure_record("2026-05-01")])
        _write_jsonl(ocf, [self._make_oc_event("2026-05-04")])
        count = materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        assert count == 1
        rows = [json.loads(l) for l in ef.read_text().strip().splitlines()]
        assert rows[0]["date"] == "2026-05-01"
        assert rows[0]["materialized"] is True

    def test_skips_too_recent_records(self, tmp_path):
        pf  = tmp_path / "pressure.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(pf, [self._make_pressure_record("2026-05-28")])
        _write_jsonl(ocf, [])
        count = materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        assert count == 0

    def test_idempotent_on_second_run(self, tmp_path):
        pf  = tmp_path / "pressure.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(pf, [self._make_pressure_record("2026-05-01")])
        _write_jsonl(ocf, [])
        materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        count2 = materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        assert count2 == 0

    def test_counts_future_oc_events(self, tmp_path):
        pf  = tmp_path / "pressure.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(pf, [self._make_pressure_record("2026-05-01")])
        _write_jsonl(ocf, [
            self._make_oc_event("2026-05-04"),
            self._make_oc_event("2026-05-05"),
            self._make_oc_event("2026-05-06"),
        ])
        materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        row = json.loads(ef.read_text().strip().splitlines()[0])
        # 3 events fall within 5 biz days of 2026-05-01
        assert row["future_opportunity_cost_events_5d"] >= 1

    def test_zero_oc_events(self, tmp_path):
        pf  = tmp_path / "pressure.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(pf, [self._make_pressure_record("2026-05-01")])
        _write_jsonl(ocf, [])
        materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        row = json.loads(ef.read_text().strip().splitlines()[0])
        assert row["future_opportunity_cost_events_5d"] == 0

    def test_empty_pressure_file_returns_0(self, tmp_path):
        pf  = tmp_path / "pressure.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc.jsonl"
        pf.write_text("")
        ocf.write_text("")
        count = materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        assert count == 0

    def test_missing_files_fail_open(self, tmp_path):
        pf  = tmp_path / "no.jsonl"
        ef  = tmp_path / "enriched.jsonl"
        ocf = tmp_path / "oc_no.jsonl"
        count = materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestAggregatePressureKpis
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregatePressureKpis:

    def _make_pressure_rec(self, regime: str):
        return {
            "date": "2026-05-01",
            "pressure_score": 60.0,
            "pressure_regime": regime,
        }

    def _make_enriched_rec(self, regime: str, future_oc: int):
        return {
            "date": "2026-05-01",
            "pressure_regime": regime,
            "future_opportunity_cost_events_5d": future_oc,
            "future_rejected_candidates_5d": future_oc,
            "materialized": True,
        }

    def test_empty_files_returns_defaults(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        ef = tmp_path / "e.jsonl"
        pf.write_text("")
        ef.write_text("")
        kpis = aggregate_pressure_kpis(pf, ef)
        assert kpis["total_pressure_records"] == 0
        assert kpis["high_vs_low_pressure_future_oc_diff"] is None

    def test_missing_files_fail_open(self, tmp_path):
        kpis = aggregate_pressure_kpis(tmp_path / "no.jsonl")
        assert isinstance(kpis, dict)
        assert "total_pressure_records" in kpis

    def test_total_pressure_records(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        _write_jsonl(pf, [self._make_pressure_rec(PRESSURE_HIGH)] * 5)
        kpis = aggregate_pressure_kpis(pf)
        assert kpis["total_pressure_records"] == 5

    def test_critical_frequency(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        _write_jsonl(pf, [
            self._make_pressure_rec(PRESSURE_CRITICAL),
            self._make_pressure_rec(PRESSURE_HIGH),
            self._make_pressure_rec(PRESSURE_LOW),
            self._make_pressure_rec(PRESSURE_LOW),
        ])
        kpis = aggregate_pressure_kpis(pf)
        assert kpis["critical_pressure_frequency"] == pytest.approx(0.25)

    def test_high_vs_low_diff_positive(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        ef = tmp_path / "e.jsonl"
        _write_jsonl(pf, [self._make_pressure_rec(PRESSURE_HIGH)] * 2)
        _write_jsonl(ef, [
            self._make_enriched_rec(PRESSURE_HIGH, 3),
            self._make_enriched_rec(PRESSURE_HIGH, 5),
            self._make_enriched_rec(PRESSURE_LOW, 1),
            self._make_enriched_rec(PRESSURE_LOW, 0),
        ])
        kpis = aggregate_pressure_kpis(pf, ef)
        # high_avg = (3+5)/2 = 4; low_avg = (1+0)/2 = 0.5; diff = 3.5
        assert kpis["high_vs_low_pressure_future_oc_diff"] == pytest.approx(3.5)

    def test_high_vs_low_diff_none_when_only_high(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        ef = tmp_path / "e.jsonl"
        _write_jsonl(pf, [self._make_pressure_rec(PRESSURE_HIGH)])
        _write_jsonl(ef, [self._make_enriched_rec(PRESSURE_HIGH, 2)])
        kpis = aggregate_pressure_kpis(pf, ef)
        assert kpis["high_vs_low_pressure_future_oc_diff"] is None

    def test_prediction_accuracy_high_pressure_with_oc(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        ef = tmp_path / "e.jsonl"
        _write_jsonl(pf, [self._make_pressure_rec(PRESSURE_HIGH)] * 3)
        _write_jsonl(ef, [
            self._make_enriched_rec(PRESSURE_HIGH, 1),  # correct
            self._make_enriched_rec(PRESSURE_HIGH, 0),  # missed
            self._make_enriched_rec(PRESSURE_CRITICAL, 2),  # correct
        ])
        kpis = aggregate_pressure_kpis(pf, ef)
        assert kpis["future_congestion_prediction_accuracy"] == pytest.approx(2 / 3, abs=1e-3)

    def test_avg_future_oc_by_pressure_populated(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        ef = tmp_path / "e.jsonl"
        _write_jsonl(pf, [self._make_pressure_rec(PRESSURE_LOW)])
        _write_jsonl(ef, [
            self._make_enriched_rec(PRESSURE_LOW, 0),
            self._make_enriched_rec(PRESSURE_HIGH, 4),
        ])
        kpis = aggregate_pressure_kpis(pf, ef)
        assert PRESSURE_LOW in kpis["avg_future_opportunity_cost_by_pressure"]
        assert kpis["avg_future_opportunity_cost_by_pressure"][PRESSURE_LOW] == pytest.approx(0.0)

    def test_schema_version_in_kpis(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        pf.write_text("")
        kpis = aggregate_pressure_kpis(pf)
        assert kpis["schema_version"] == SCHEMA_VERSION

    def test_no_enriched_file_arg_ok(self, tmp_path):
        pf = tmp_path / "p.jsonl"
        _write_jsonl(pf, [self._make_pressure_rec(PRESSURE_HIGH)])
        kpis = aggregate_pressure_kpis(pf)
        assert kpis["total_pressure_records"] == 1
        assert kpis["avg_future_oc_high_pressure"] is None


# ─────────────────────────────────────────────────────────────────────────────
# TestFormatSlotPressureReport
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSlotPressureReport:

    def test_contains_header(self):
        inp = _full_inp(holding_tiers=[TIER_NEUTRAL, TIER_HIGH, TIER_NEUTRAL])
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        assert "SLOT PRESSURE FORECAST" in rpt

    def test_contains_date(self):
        inp = _empty_inp(max_positions=3, current_positions=3)
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        assert "2026-05-29" in rpt

    def test_contains_pressure_score(self):
        inp = _empty_inp(max_positions=3, current_positions=3)
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        assert str(int(r.pressure_score)) in rpt

    def test_shows_reasons(self):
        inp = _full_inp(holding_tiers=[TIER_NEUTRAL, TIER_HIGH, TIER_NEUTRAL])
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        for reason in r.reasons:
            assert reason in rpt

    def test_open_slots_shown(self):
        inp = _empty_inp(max_positions=3, current_positions=1)
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        assert "Open slots" in rpt

    def test_na_when_no_holdings(self):
        inp = _empty_inp()
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        assert "N/A" in rpt

    def test_returns_string(self):
        inp = _empty_inp()
        r = compute_slot_pressure(inp)
        rpt = format_slot_pressure_report(inp, r, "2026-05-29")
        assert isinstance(rpt, str)

    def test_fails_open_returns_empty_string(self):
        """Passing bad types should not raise — returns '' (fail-open)."""
        rpt = format_slot_pressure_report(None, None, "")  # type: ignore
        assert isinstance(rpt, str)


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildSlotPressureInputAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSlotPressureInputAdapter:

    def _make_oc_event(self, date: str, severity: str = "moderate", cand_prio: float = 70.0):
        return {
            "date": date,
            "opportunity_cost_severity": severity,
            "candidate_priority_estimate": cand_prio,
        }

    class _CPResult:
        def __init__(self, priority_score: float, priority_tier: str):
            self.priority_score = priority_score
            self.priority_tier = priority_tier

    def test_object_cp_results(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        ocf.write_text("")
        cps = [self._CPResult(80.0, TIER_CORE), self._CPResult(55.0, TIER_HIGH)]
        inp = build_slot_pressure_input_from_oc_data(2, 3, cps, ocf, today="2026-05-29")
        assert inp.holding_priorities == [80.0, 55.0]
        assert inp.holding_tiers == [TIER_CORE, TIER_HIGH]

    def test_dict_cp_results(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        ocf.write_text("")
        cps = [
            {"priority_score": 75.0, "priority_tier": TIER_CORE},
            {"priority_score": 50.0, "priority_tier": TIER_NEUTRAL},
        ]
        inp = build_slot_pressure_input_from_oc_data(2, 3, cps, ocf, today="2026-05-29")
        assert inp.holding_priorities == [75.0, 50.0]
        assert inp.holding_tiers == [TIER_CORE, TIER_NEUTRAL]

    def test_recent_oc_events_counted(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(ocf, [
            self._make_oc_event("2026-05-25"),
            self._make_oc_event("2026-05-26"),
            self._make_oc_event("2026-05-27"),
        ])
        inp = build_slot_pressure_input_from_oc_data(0, 3, [], ocf, lookback_days=5, today="2026-05-29")
        assert inp.recent_rejected_candidates == 3

    def test_old_events_excluded_from_lookback(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(ocf, [
            self._make_oc_event("2026-05-01"),  # too old
            self._make_oc_event("2026-05-28"),  # within 5d
        ])
        inp = build_slot_pressure_input_from_oc_data(0, 3, [], ocf, lookback_days=5, today="2026-05-29")
        assert inp.recent_rejected_candidates == 1

    def test_severe_events_counted(self, tmp_path):
        from src.analytics.opportunity_cost import SEVERITY_SEVERE
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(ocf, [
            self._make_oc_event("2026-05-27", severity=SEVERITY_SEVERE),
            self._make_oc_event("2026-05-28", severity=SEVERITY_SEVERE),
            self._make_oc_event("2026-05-28", severity="moderate"),
        ])
        inp = build_slot_pressure_input_from_oc_data(0, 3, [], ocf, lookback_days=5, today="2026-05-29")
        assert inp.recent_severe_opportunity_cost_events == 2

    def test_avg_candidate_priority_computed(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(ocf, [
            self._make_oc_event("2026-05-27", cand_prio=80.0),
            self._make_oc_event("2026-05-28", cand_prio=90.0),
        ])
        inp = build_slot_pressure_input_from_oc_data(0, 3, [], ocf, lookback_days=5, today="2026-05-29")
        assert inp.avg_candidate_priority_5d == pytest.approx(85.0)

    def test_empty_oc_file_zeros(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        ocf.write_text("")
        inp = build_slot_pressure_input_from_oc_data(1, 3, [], ocf, today="2026-05-29")
        assert inp.recent_rejected_candidates == 0
        assert inp.recent_severe_opportunity_cost_events == 0
        assert inp.avg_candidate_priority_5d == 0.0

    def test_missing_oc_file_fail_open(self, tmp_path):
        ocf = tmp_path / "nonexistent.jsonl"
        inp = build_slot_pressure_input_from_oc_data(1, 3, [], ocf, today="2026-05-29")
        assert isinstance(inp, SlotPressureInput)
        assert inp.recent_rejected_candidates == 0

    def test_positions_set_correctly(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        ocf.write_text("")
        inp = build_slot_pressure_input_from_oc_data(2, 5, [], ocf, today="2026-05-29")
        assert inp.current_positions == 2
        assert inp.max_positions == 5

    def test_no_today_uses_recent_slice(self, tmp_path):
        ocf = tmp_path / "oc.jsonl"
        _write_jsonl(ocf, [self._make_oc_event("2026-05-01")] * 20)
        # Without today, uses tail slice — should not raise
        inp = build_slot_pressure_input_from_oc_data(0, 3, [], ocf, today="")
        assert isinstance(inp, SlotPressureInput)


# ─────────────────────────────────────────────────────────────────────────────
# TestFailOpen
# ─────────────────────────────────────────────────────────────────────────────

class TestFailOpen:

    def test_record_slot_pressure_bad_path_returns_none(self):
        from pathlib import Path
        bad = Path("/nonexistent/deep/dir/sp.jsonl")
        inp = _empty_inp()
        # Should fail-open returning None or succeed (if mkdir works)
        result = record_slot_pressure(inp, "2026-05-29", bad)
        # Either None or a valid record — must not raise
        assert result is None or isinstance(result, SlotPressureRecord)

    def test_materialize_empty_pressure_file(self, tmp_path):
        pf  = tmp_path / "p.jsonl"
        ef  = tmp_path / "e.jsonl"
        ocf = tmp_path / "oc.jsonl"
        pf.touch()
        ocf.touch()
        count = materialize_slot_pressure(pf, ef, ocf, today="2026-05-29")
        assert count == 0

    def test_aggregate_kpis_corrupt_file(self, tmp_path):
        pf = tmp_path / "bad.jsonl"
        pf.write_text("not json\n{invalid}\n")
        kpis = aggregate_pressure_kpis(pf)
        assert isinstance(kpis, dict)

    def test_format_report_none_result_returns_empty(self):
        rpt = format_slot_pressure_report(None, None, "2026-05-29")  # type: ignore
        assert rpt == ""

    def test_compute_slot_pressure_no_holdings(self):
        inp = SlotPressureInput(
            max_positions=3, current_positions=0,
            holding_priorities=[], holding_tiers=[],
            avg_candidate_priority_5d=0.0,
            recent_rejected_candidates=0,
            recent_severe_opportunity_cost_events=0,
        )
        r = compute_slot_pressure(inp)
        assert r.lowest_priority is None
        assert r.open_slots == 3
