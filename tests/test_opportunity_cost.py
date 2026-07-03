"""
tests/test_opportunity_cost.py
Pytest suite for Opportunity Cost Intelligence.

Covers:
  - estimate_candidate_priority: weights, edge cases, normalization
  - classify_priority_gap: boundary conditions (severe/moderate/low)
  - find_lowest_priority_holding: various portfolio states
  - build_opportunity_cost_record: field correctness, schema_version
  - record_opportunity_cost_events: JSONL append, empty inputs, min_gap filter
  - materialize_outcomes: price fetcher injection, skip-if-too-early, dedup
  - aggregate_kpis: empty / partial / complete records
  - format_opportunity_cost_report: structure, date filtering, empty fallback
  - signal dict adapter: make_rejected_candidate_from_signal
  - held summary adapter: make_held_summary_from_cp_result
  - business day arithmetic: _add_business_days, _due_for_materialization
  - fail-open behavior: bad inputs, bad JSONL, missing files
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pytest

from src.analytics.opportunity_cost import (
    RejectedCandidate,
    HeldPositionSummary,
    OpportunityCostRecord,
    MaterializedRecord,
    # Functions
    estimate_candidate_priority,
    classify_priority_gap,
    find_lowest_priority_holding,
    build_opportunity_cost_record,
    record_opportunity_cost_events,
    materialize_outcomes,
    aggregate_kpis,
    format_opportunity_cost_report,
    make_rejected_candidate_from_signal,
    make_held_summary_from_cp_result,
    # Internal helpers (public via module)
    _add_business_days,
    _due_for_materialization,
    _load_jsonl,
    _append_jsonl,
    # Constants
    PRIORITY_GAP_SEVERE,
    PRIORITY_GAP_MODERATE,
    SEVERITY_SEVERE,
    SEVERITY_MODERATE,
    SEVERITY_LOW,
    SCHEMA_VERSION,
    W_COMPRESSION,
    W_BQ,
    W_RSR,
    W_SURVIVABILITY,
)

TODAY = "2026-05-29"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _cand(
    symbol="7011.T",
    cont=0.0,
    bq=80.0,
    comp=75.0,
    surv=0.85,
    rsr=88.0,
) -> RejectedCandidate:
    return RejectedCandidate(
        symbol=symbol,
        continuation_score=cont,
        breakout_quality=bq,
        compression_score=comp,
        survivability=surv,
        rsr=rsr,
    )


def _held(
    symbol="5803.T",
    score=40.0,
    tier="exit_candidate",
    phase="weak_breakout",
) -> HeldPositionSummary:
    return HeldPositionSummary(
        symbol=symbol,
        priority_score=score,
        priority_tier=tier,
        phase=phase,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. estimate_candidate_priority
# ─────────────────────────────────────────────────────────────────────────────

def test_estimate_priority_returns_float():
    cand = _cand()
    p = estimate_candidate_priority(cand)
    assert isinstance(p, float)


def test_estimate_priority_bounded():
    # Extreme values should stay in [0, 100]
    cand = _cand(bq=200.0, comp=200.0, surv=2.0, rsr=200.0)
    p = estimate_candidate_priority(cand)
    assert 0.0 <= p <= 100.0


def test_estimate_priority_lower_bound():
    cand = _cand(bq=0.0, comp=0.0, surv=0.0, rsr=0.0)
    p = estimate_candidate_priority(cand)
    assert p == 0.0


def test_estimate_priority_higher_scores_give_higher_priority():
    cand_hi = _cand(bq=90.0, comp=90.0, surv=0.90, rsr=95.0)
    cand_lo = _cand(bq=20.0, comp=20.0, surv=0.40, rsr=65.0)
    assert estimate_candidate_priority(cand_hi) > estimate_candidate_priority(cand_lo)


def test_estimate_priority_weights_sum_to_1():
    assert abs(W_COMPRESSION + W_BQ + W_RSR + W_SURVIVABILITY - 1.0) < 1e-9


def test_estimate_priority_rsr_below_floor_gives_zero_rsr_component():
    # RSR < 60 → rsr strength = 0
    cand = _cand(bq=0.0, comp=0.0, surv=0.0, rsr=50.0)
    p = estimate_candidate_priority(cand)
    assert p == 0.0


def test_estimate_priority_rsr_at_ceil():
    cand = _cand(bq=0.0, comp=0.0, surv=0.0, rsr=100.0)
    p = estimate_candidate_priority(cand)
    expected = W_RSR * 100.0
    assert abs(p - expected) < 0.1


def test_estimate_priority_reproducible():
    cand = _cand(bq=75.0, comp=60.0, surv=0.80, rsr=90.0)
    assert estimate_candidate_priority(cand) == estimate_candidate_priority(cand)


# ─────────────────────────────────────────────────────────────────────────────
# 2. classify_priority_gap
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_severe():
    assert classify_priority_gap(PRIORITY_GAP_SEVERE) == SEVERITY_SEVERE
    assert classify_priority_gap(PRIORITY_GAP_SEVERE + 10) == SEVERITY_SEVERE
    assert classify_priority_gap(99.9) == SEVERITY_SEVERE


def test_classify_moderate():
    assert classify_priority_gap(PRIORITY_GAP_MODERATE) == SEVERITY_MODERATE
    assert classify_priority_gap(PRIORITY_GAP_MODERATE + 5) == SEVERITY_MODERATE
    assert classify_priority_gap(PRIORITY_GAP_SEVERE - 0.01) == SEVERITY_MODERATE


def test_classify_low():
    assert classify_priority_gap(0.0) == SEVERITY_LOW
    assert classify_priority_gap(-10.0) == SEVERITY_LOW
    assert classify_priority_gap(PRIORITY_GAP_MODERATE - 0.01) == SEVERITY_LOW


def test_classify_boundary_moderate():
    assert classify_priority_gap(20.0) == SEVERITY_MODERATE
    assert classify_priority_gap(19.99) == SEVERITY_LOW


def test_classify_boundary_severe():
    assert classify_priority_gap(40.0) == SEVERITY_SEVERE
    assert classify_priority_gap(39.99) == SEVERITY_MODERATE


# ─────────────────────────────────────────────────────────────────────────────
# 3. find_lowest_priority_holding
# ─────────────────────────────────────────────────────────────────────────────

def test_find_lowest_single():
    h = _held(score=50.0)
    assert find_lowest_priority_holding([h]) == h


def test_find_lowest_multiple():
    h1 = _held("A", score=80.0)
    h2 = _held("B", score=30.0)
    h3 = _held("C", score=55.0)
    result = find_lowest_priority_holding([h1, h2, h3])
    assert result == h2


def test_find_lowest_empty():
    assert find_lowest_priority_holding([]) is None


def test_find_lowest_all_same_score():
    h1 = _held("A", score=50.0)
    h2 = _held("B", score=50.0)
    result = find_lowest_priority_holding([h1, h2])
    assert result is not None
    assert result.priority_score == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. build_opportunity_cost_record
# ─────────────────────────────────────────────────────────────────────────────

def test_build_record_fields():
    cand = _cand(symbol="7011.T", bq=84.1, comp=79.2, surv=0.85, rsr=90.0)
    held = _held(symbol="5803.T", score=41.3, tier="exit_candidate", phase="weak_breakout")
    rec = build_opportunity_cost_record(cand, held, TODAY, portfolio_full=True)

    assert rec.rejected_symbol == "7011.T"
    assert rec.blocked_by_symbol == "5803.T"
    assert rec.blocked_position_tier == "exit_candidate"
    assert rec.blocked_phase == "weak_breakout"
    assert rec.portfolio_full is True
    assert rec.schema_version == SCHEMA_VERSION


def test_build_record_priority_gap_computed():
    cand = _cand(bq=84.1, comp=79.2, surv=0.85, rsr=90.0)
    held = _held(score=41.3)
    rec = build_opportunity_cost_record(cand, held, TODAY, portfolio_full=True)

    expected_cand = estimate_candidate_priority(cand)
    expected_gap  = round(expected_cand - 41.3, 2)
    assert rec.priority_gap == expected_gap
    assert rec.candidate_priority_estimate == expected_cand


def test_build_record_severity():
    cand = _cand(bq=100.0, comp=100.0, surv=0.90, rsr=100.0)
    held = _held(score=10.0)  # gap will be large → severe
    rec = build_opportunity_cost_record(cand, held, TODAY, portfolio_full=True)
    assert rec.opportunity_cost_severity == SEVERITY_SEVERE


def test_build_record_low_severity():
    cand = _cand(bq=50.0, comp=50.0, surv=0.70, rsr=75.0)
    held = _held(score=48.0)  # very small gap
    rec = build_opportunity_cost_record(cand, held, TODAY, portfolio_full=True)
    assert rec.opportunity_cost_severity == SEVERITY_LOW


# ─────────────────────────────────────────────────────────────────────────────
# 5. record_opportunity_cost_events
# ─────────────────────────────────────────────────────────────────────────────

def test_record_events_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        cand = _cand(bq=80.0, comp=75.0, surv=0.85, rsr=90.0)
        held = _held(score=30.0)  # gap will be positive

        recorded = record_opportunity_cost_events(
            rejected_candidates=[cand],
            held_positions=[held],
            portfolio_full=True,
            date=TODAY,
            events_file=f,
        )
        assert len(recorded) == 1
        assert f.exists()
        lines = f.read_text("utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["rejected_symbol"] == "7011.T"
        assert row["blocked_by_symbol"] == "5803.T"
        assert "ts" in row


def test_record_events_empty_when_not_portfolio_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        recorded = record_opportunity_cost_events(
            rejected_candidates=[_cand()],
            held_positions=[_held()],
            portfolio_full=False,
            date=TODAY,
            events_file=f,
        )
        assert recorded == []
        assert not f.exists()


def test_record_events_empty_when_no_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        recorded = record_opportunity_cost_events(
            rejected_candidates=[],
            held_positions=[_held()],
            portfolio_full=True,
            date=TODAY,
            events_file=f,
        )
        assert recorded == []


def test_record_events_empty_when_no_held():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        recorded = record_opportunity_cost_events(
            rejected_candidates=[_cand()],
            held_positions=[],
            portfolio_full=True,
            date=TODAY,
            events_file=f,
        )
        assert recorded == []


def test_record_events_min_gap_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        # Force a tiny gap by making held priority close to candidate priority
        cand = _cand(bq=50.0, comp=50.0, surv=0.70, rsr=75.0)
        held = _held(score=48.0)  # gap will be small
        recorded = record_opportunity_cost_events(
            rejected_candidates=[cand],
            held_positions=[held],
            portfolio_full=True,
            date=TODAY,
            events_file=f,
            min_gap_to_record=50.0,  # very high threshold → nothing recorded
        )
        assert recorded == []


def test_record_events_multiple_candidates():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        cands = [_cand(f"SYM{i}", bq=80.0, comp=75.0, surv=0.85, rsr=90.0) for i in range(3)]
        held = _held(score=30.0)
        recorded = record_opportunity_cost_events(
            rejected_candidates=cands,
            held_positions=[held],
            portfolio_full=True,
            date=TODAY,
            events_file=f,
        )
        # All 3 should be recorded (gap > 0 for all)
        assert len(recorded) == 3
        lines = f.read_text("utf-8").strip().splitlines()
        assert len(lines) == 3


def test_record_events_append_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "events.jsonl"
        cand = _cand(bq=80.0, comp=75.0, surv=0.85, rsr=90.0)
        held = _held(score=30.0)
        # Two separate calls → two lines
        record_opportunity_cost_events([cand], [held], True, TODAY, f)
        record_opportunity_cost_events([cand], [held], True, TODAY, f)
        lines = f.read_text("utf-8").strip().splitlines()
        assert len(lines) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. Business day helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_add_business_days_5_from_monday():
    # 2026-05-25 (Monday) + 5 bd = 2026-06-01 (Monday)
    result = _add_business_days("2026-05-25", 5)
    assert result == "2026-06-01"


def test_add_business_days_skips_weekend():
    # 2026-05-29 (Friday) + 1 bd = 2026-06-01 (Monday)
    result = _add_business_days("2026-05-29", 1)
    assert result == "2026-06-01"


def test_add_business_days_zero():
    result = _add_business_days("2026-05-29", 0)
    assert result == "2026-05-29"


def test_add_business_days_bad_input():
    # Should return the original string, not raise
    result = _add_business_days("invalid", 5)
    assert result == "invalid"


def test_due_for_materialization_true():
    # event 10 days ago → due
    assert _due_for_materialization("2026-05-19", "2026-05-29", buffer_days=8) is True


def test_due_for_materialization_false():
    # event 3 days ago → not due (buffer=8)
    assert _due_for_materialization("2026-05-26", "2026-05-29", buffer_days=8) is False


def test_due_for_materialization_exact_boundary():
    assert _due_for_materialization("2026-05-21", "2026-05-29", buffer_days=8) is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. materialize_outcomes
# ─────────────────────────────────────────────────────────────────────────────

def _make_event(
    date=TODAY,
    rej_sym="7011.T",
    blk_sym="5803.T",
    gap=50.0,
    sev=SEVERITY_SEVERE,
) -> dict:
    return {
        "date": date,
        "rejected_symbol": rej_sym,
        "candidate_priority_estimate": 90.0,
        "candidate_breakout_quality": 80.0,
        "candidate_compression_score": 75.0,
        "blocked_by_symbol": blk_sym,
        "blocked_position_priority": 40.0,
        "blocked_position_tier": "exit_candidate",
        "priority_gap": gap,
        "blocked_phase": "weak_breakout",
        "portfolio_full": True,
        "opportunity_cost_severity": sev,
        "schema_version": SCHEMA_VERSION,
        "ts": "2026-05-29T09:00:00+0900",
    }


def test_materialize_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "materialized.jsonl"

        # Event from 10 days ago (due)
        old_date = "2026-05-19"
        ev = _make_event(date=old_date)
        _append_jsonl(ev, ef)

        prices = {
            "7011.T": {"2026-05-19": 1000.0, "2026-05-26": 1050.0},
            "5803.T": {"2026-05-19": 2000.0, "2026-05-26": 1900.0},
        }

        def fetcher(sym: str, dt: str) -> Optional[float]:
            return prices.get(sym, {}).get(dt)

        count = materialize_outcomes(ef, mf, fetcher, TODAY, buffer_days=8)
        assert count == 1
        assert mf.exists()

        records = _load_jsonl(mf)
        assert len(records) == 1
        r = records[0]
        assert r["materialization_complete"] is True
        assert abs(r["rejected_candidate_5d_return"] - 0.05) < 1e-6
        assert abs(r["blocked_position_5d_return"] - (-0.05)) < 1e-6
        delta = r["rotation_counterfactual_delta"]
        assert delta is not None
        assert abs(delta - 0.10) < 1e-5


def test_materialize_skips_if_too_early():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "materialized.jsonl"

        # Event from yesterday (not due yet)
        ev = _make_event(date="2026-05-28")
        _append_jsonl(ev, ef)

        count = materialize_outcomes(ef, mf, lambda s, d: 1000.0, TODAY, buffer_days=8)
        assert count == 0
        assert not mf.exists()


def test_materialize_deduplicates():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "materialized.jsonl"

        old_date = "2026-05-19"
        ev = _make_event(date=old_date)
        _append_jsonl(ev, ef)

        # First materialization
        count1 = materialize_outcomes(ef, mf, lambda s, d: 1000.0, TODAY, buffer_days=8)
        # Second call → dedup
        count2 = materialize_outcomes(ef, mf, lambda s, d: 1000.0, TODAY, buffer_days=8)
        assert count1 == 1
        assert count2 == 0


def test_materialize_partial_when_price_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "materialized.jsonl"

        old_date = "2026-05-19"
        ev = _make_event(date=old_date)
        _append_jsonl(ev, ef)

        def fetcher(sym: str, dt: str) -> Optional[float]:
            # Only return price for rejected candidate, not blocked
            if sym == "7011.T":
                return 1000.0
            return None

        count = materialize_outcomes(ef, mf, fetcher, TODAY, buffer_days=8)
        assert count == 1
        r = _load_jsonl(mf)[0]
        assert r["materialization_complete"] is False
        assert r["blocked_position_5d_return"] is None
        assert r["rotation_counterfactual_delta"] is None


def test_materialize_empty_events_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "materialized.jsonl"
        count = materialize_outcomes(ef, mf, lambda s, d: 1000.0, TODAY)
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. aggregate_kpis
# ─────────────────────────────────────────────────────────────────────────────

def test_kpis_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "mat.jsonl"
        kpis = aggregate_kpis(f)
        assert kpis["opportunity_cost_event_count"] == 0
        assert kpis["avg_priority_gap"] is None
        assert kpis["severe_event_count"] == 0


def test_kpis_single_complete_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "mat.jsonl"
        rec = {
            "date": TODAY,
            "rejected_symbol": "A",
            "blocked_by_symbol": "B",
            "candidate_priority_estimate": 90.0,
            "candidate_breakout_quality": 80.0,
            "candidate_compression_score": 75.0,
            "blocked_position_priority": 40.0,
            "blocked_position_tier": "exit_candidate",
            "priority_gap": 50.0,
            "blocked_phase": "weak",
            "portfolio_full": True,
            "opportunity_cost_severity": SEVERITY_SEVERE,
            "schema_version": SCHEMA_VERSION,
            "rejected_candidate_5d_return": 0.05,
            "blocked_position_5d_return": -0.03,
            "rotation_counterfactual_delta": 0.08,
            "materialized_date": TODAY,
            "materialization_complete": True,
        }
        _append_jsonl(rec, f)
        kpis = aggregate_kpis(f)

        assert kpis["opportunity_cost_event_count"] == 1
        assert kpis["avg_priority_gap"] == 50.0
        assert kpis["avg_rejected_candidate_5d_return"] == 0.05
        assert kpis["avg_blocked_position_5d_return"] == -0.03
        assert kpis["avg_rotation_counterfactual_delta"] == 0.08
        assert kpis["severe_event_count"] == 1
        assert kpis["complete_materialization_count"] == 1


def test_kpis_averages_multiple_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "mat.jsonl"
        for gap in [20.0, 40.0, 60.0]:
            sev = SEVERITY_SEVERE if gap >= 40 else SEVERITY_MODERATE
            rec = {
                "date": TODAY, "rejected_symbol": "A", "blocked_by_symbol": "B",
                "candidate_priority_estimate": 70.0,
                "candidate_breakout_quality": 50.0,
                "candidate_compression_score": 50.0,
                "blocked_position_priority": 70.0 - gap,
                "blocked_position_tier": "exit_candidate",
                "priority_gap": gap,
                "blocked_phase": "",
                "portfolio_full": True,
                "opportunity_cost_severity": sev,
                "schema_version": SCHEMA_VERSION,
                "rejected_candidate_5d_return": 0.02,
                "blocked_position_5d_return": -0.01,
                "rotation_counterfactual_delta": 0.03,
                "materialized_date": TODAY,
                "materialization_complete": True,
            }
            _append_jsonl(rec, f)

        kpis = aggregate_kpis(f)
        assert kpis["opportunity_cost_event_count"] == 3
        assert abs(kpis["avg_priority_gap"] - 40.0) < 1e-6
        assert kpis["severe_event_count"] == 2
        assert kpis["moderate_event_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. format_opportunity_cost_report
# ─────────────────────────────────────────────────────────────────────────────

def test_report_empty_when_no_events():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        result = format_opportunity_cost_report(ef, date=TODAY)
        assert result == ""


def test_report_contains_key_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        ev = _make_event(rej_sym="7011.T", blk_sym="5803.T", gap=50.1)
        ev["date"] = TODAY
        _append_jsonl(ev, ef)

        report = format_opportunity_cost_report(ef, date=TODAY)
        assert "7011.T" in report
        assert "5803.T" in report
        assert "50.1" in report


def test_report_contains_severity():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        ev = _make_event(gap=50.0, sev=SEVERITY_SEVERE)
        ev["date"] = TODAY
        _append_jsonl(ev, ef)

        report = format_opportunity_cost_report(ef, date=TODAY)
        assert "SEVERE" in report.upper()


def test_report_filters_by_date():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        # Event from yesterday
        ev_old = _make_event(rej_sym="ZZZZ", date="2026-05-28")
        _append_jsonl(ev_old, ef)

        report = format_opportunity_cost_report(ef, date=TODAY)
        assert "ZZZZ" not in report


def test_report_with_kpi_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "mat.jsonl"

        ev = _make_event()
        ev["date"] = TODAY
        _append_jsonl(ev, ef)

        # Add a materialized record
        mat = {
            "date": TODAY, "rejected_symbol": "A", "blocked_by_symbol": "B",
            "candidate_priority_estimate": 90.0, "candidate_breakout_quality": 80.0,
            "candidate_compression_score": 75.0, "blocked_position_priority": 40.0,
            "blocked_position_tier": "exit_candidate", "priority_gap": 50.0,
            "blocked_phase": "weak", "portfolio_full": True,
            "opportunity_cost_severity": SEVERITY_SEVERE, "schema_version": SCHEMA_VERSION,
            "rejected_candidate_5d_return": 0.05, "blocked_position_5d_return": -0.03,
            "rotation_counterfactual_delta": 0.08, "materialized_date": TODAY,
            "materialization_complete": True,
        }
        _append_jsonl(mat, mf)

        report = format_opportunity_cost_report(ef, materialized_file=mf, date=TODAY)
        assert "KPI" in report or "kpi" in report.lower() or "events=" in report


def test_report_fails_open_bad_file():
    # Should return empty string, not raise
    result = format_opportunity_cost_report(
        Path("/nonexistent/path/events.jsonl"), date=TODAY
    )
    assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# 10. Signal adapter
# ─────────────────────────────────────────────────────────────────────────────

def test_make_rejected_candidate_from_signal_basic():
    sig = {
        "symbol": "7011.T",
        "rsr": 88.5,
        "compression_score": 72.0,
        "breakout_quality_score": 81.0,
        "survivability": 0.82,
        "signal": 1,
    }
    cand = make_rejected_candidate_from_signal(sig)
    assert cand.symbol == "7011.T"
    assert cand.rsr == 88.5
    assert cand.compression_score == 72.0
    assert cand.breakout_quality == 81.0
    assert cand.survivability == 0.82


def test_make_rejected_candidate_defaults_on_missing_fields():
    sig = {"symbol": "1234.T"}
    cand = make_rejected_candidate_from_signal(sig)
    assert cand.symbol == "1234.T"
    assert 0.0 <= cand.survivability <= 1.0
    assert 0.0 <= cand.compression_score <= 100.0


def test_make_rejected_candidate_bq_fallback():
    # bq_score as alternate key
    sig = {"symbol": "A", "bq_score": 77.5}
    cand = make_rejected_candidate_from_signal(sig)
    assert cand.breakout_quality == 77.5


# ─────────────────────────────────────────────────────────────────────────────
# 11. Held summary adapter
# ─────────────────────────────────────────────────────────────────────────────

class _MockCPResult:
    def __init__(self, symbol, score, tier):
        self.symbol = symbol
        self.priority_score = score
        self.priority_tier = tier


def test_make_held_summary_from_cp_result():
    sig = {"symbol": "5803.T", "compression_phase": "distribution"}
    cp  = _MockCPResult("5803.T", 41.3, "exit_candidate")
    held = make_held_summary_from_cp_result((sig, cp))
    assert held is not None
    assert held.symbol == "5803.T"
    assert held.priority_score == 41.3
    assert held.priority_tier == "exit_candidate"
    assert held.phase == "distribution"


def test_make_held_summary_returns_none_on_error():
    result = make_held_summary_from_cp_result(("bad_data", None))
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 12. JSONL helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_load_jsonl_missing_file():
    result = _load_jsonl(Path("/nonexistent/file.jsonl"))
    assert result == []


def test_load_jsonl_malformed_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "test.jsonl"
        f.write_text('{"valid": true}\nnot-json\n{"also": true}\n', encoding="utf-8")
        records = _load_jsonl(f)
        assert len(records) == 2


def test_append_jsonl_creates_parent():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "sub" / "dir" / "file.jsonl"
        _append_jsonl({"x": 1}, f)
        assert f.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 13. Schema version
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_version_in_records():
    cand = _cand(bq=80.0, comp=75.0, surv=0.85, rsr=90.0)
    held = _held(score=30.0)
    rec = build_opportunity_cost_record(cand, held, TODAY, portfolio_full=True)
    assert rec.schema_version == SCHEMA_VERSION


def test_schema_version_constant():
    assert SCHEMA_VERSION == "v1"


# ─────────────────────────────────────────────────────────────────────────────
# 14. End-to-end flow
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    """Record → materialize → kpis → report, all in one test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ef = Path(tmpdir) / "events.jsonl"
        mf = Path(tmpdir) / "materialized.jsonl"

        # Record event
        cand = _cand(bq=84.0, comp=79.0, surv=0.85, rsr=90.0)
        held = _held(score=41.0)
        record_opportunity_cost_events([cand], [held], True, TODAY, ef)

        # Check recorded
        events = _load_jsonl(ef)
        assert len(events) == 1
        assert events[0]["rejected_symbol"] == "7011.T"

        # Manually create a materialized record (simulate old event)
        old_event = dict(events[0])
        old_event["date"] = "2026-05-19"  # 10 days ago
        ef2 = Path(tmpdir) / "old_events.jsonl"
        _append_jsonl(old_event, ef2)

        # Materialize
        prices = {
            "7011.T": {"2026-05-19": 1000.0, "2026-05-26": 1060.0},
            "5803.T": {"2026-05-19": 2000.0, "2026-05-26": 1960.0},
        }
        count = materialize_outcomes(
            ef2, mf, lambda s, d: prices.get(s, {}).get(d), TODAY, buffer_days=8
        )
        assert count == 1

        # KPIs
        kpis = aggregate_kpis(mf)
        assert kpis["opportunity_cost_event_count"] == 1
        assert kpis["avg_rotation_counterfactual_delta"] is not None
        assert kpis["avg_rotation_counterfactual_delta"] > 0  # rejected outperformed

        # Report
        report = format_opportunity_cost_report(ef, date=TODAY)
        assert "7011.T" in report
        assert "5803.T" in report
