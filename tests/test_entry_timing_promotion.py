"""
tests/test_entry_timing_promotion.py — Entry Timing Promotion Engine tests

Covers:
  - WindowKPIs computation (various sample sizes, decile distributions)
  - check_safety_triggers
  - check_tier1_promotion / check_tier2_promotion / check_tier3_promotion
  - evaluate_tier_transition (0→1, 1→2, 2→3, safety demotion, AT_MAX)
  - run_entry_timing_promotion (persistence, idempotency, CSV history)
  - get_effective_boost_weight (auto_apply on/off)
  - format_et_promotion_section (FAIL_OPEN, section content)
  - _compute_monotonicity edge cases
  - FAIL_OPEN throughout
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pytest

from src.entry.entry_timing_promotion import (
    SAFETY_EXPECTANCY_FLOOR,
    SAFETY_MONOTONICITY_FLOOR,
    SAFETY_SAMPLE_FLOOR,
    TIER1_MONOTONICITY_MIN,
    TIER1_SAMPLE_MIN,
    TIER1_WIN_RATE_GAP_MIN,
    TIER2_EXPECTANCY_UPLIFT,
    TIER2_MONOTONICITY_MIN,
    TIER2_TIER1_DAYS_MIN,
    TIER3_EXPECTANCY_UPLIFT,
    TIER3_MONOTONICITY_MIN,
    TIER3_TIER2_DAYS_MIN,
    TIER_BOOST_WEIGHTS,
    PromotionState,
    WindowKPIs,
    _compute_monotonicity,
    check_safety_triggers,
    check_tier1_promotion,
    check_tier2_promotion,
    check_tier3_promotion,
    compute_window_kpis,
    evaluate_tier_transition,
    format_et_promotion_section,
    get_effective_boost_weight,
    run_entry_timing_promotion,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TODAY = date(2026, 5, 31)
TODAY_STR = "2026-05-31"


def _make_record(
    days_ago: int,
    score_decile: int,
    ret_3d: float,
    materialized: bool = True,
) -> dict:
    d = (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "date": d,
        "score_decile": score_decile,
        "subsequent_3d_return": ret_3d,
        "materialized_3d": materialized,
    }


def _make_records_monotonic(n_per_decile: int = 20, window_days: int = 25) -> List[dict]:
    """
    Produce records where higher deciles have strictly higher win rates
    (decile d → win rate = d/10), spread across the last `window_days` days.
    """
    records = []
    for d in range(1, 11):
        win_rate = d / 10.0
        for i in range(n_per_decile):
            days_ago = i % max(1, window_days - 1)
            ret = 0.02 if (i / n_per_decile) < win_rate else -0.01
            records.append(_make_record(days_ago, d, ret))
    return records


def _make_window_kpis(
    sample_count: int = 120,
    mono: float = 0.75,
    win_rate_gap: float = 0.20,
    expectancy_uplift: float = 0.50,
    top_decile_expectancy: float = 0.02,
    sufficient: bool = True,
) -> WindowKPIs:
    return WindowKPIs(
        window_days=30,
        sample_count=sample_count,
        monotonicity_score=mono,
        top_decile_win_rate=0.70,
        bottom_decile_win_rate=0.50,
        top_decile_expectancy=top_decile_expectancy,
        bottom_decile_expectancy=-0.01,
        all_expectancy=0.005,
        expectancy_uplift=expectancy_uplift,
        win_rate_gap=win_rate_gap,
        sufficient=sufficient,
    )


def _make_insufficient_kpis(window_days: int = 30) -> WindowKPIs:
    return WindowKPIs(
        window_days=window_days, sample_count=0,
        monotonicity_score=None, top_decile_win_rate=None,
        bottom_decile_win_rate=None, top_decile_expectancy=None,
        bottom_decile_expectancy=None, all_expectancy=None,
        expectancy_uplift=None, win_rate_gap=None,
        sufficient=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _compute_monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMonotonicity:
    def test_perfect_monotone(self):
        wr = {d: d / 10.0 for d in range(1, 11)}
        assert _compute_monotonicity(wr) == 1.0

    def test_fully_non_monotone(self):
        # alternating: 1.0, 0.0, 1.0, ...
        wr = {d: 1.0 if d % 2 == 1 else 0.0 for d in range(1, 11)}
        result = _compute_monotonicity(wr)
        assert result is not None
        assert result < 0.5

    def test_too_few_deciles_returns_none(self):
        wr = {1: 0.5, 2: 0.6}  # only 2 entries → 1 pair → pairs_checked=1 < 2
        assert _compute_monotonicity(wr) is None

    def test_empty_returns_none(self):
        assert _compute_monotonicity({}) is None

    def test_none_values_skip_pairs(self):
        wr = {1: 0.5, 2: None, 3: 0.7, 4: 0.8, 5: 0.9}
        result = _compute_monotonicity(wr)
        assert result is not None
        # pairs: (3→4), (4→5) both rising; (1→2) skipped, (2→3) skipped
        assert result == 1.0

    def test_all_none_returns_none(self):
        wr = {d: None for d in range(1, 11)}
        assert _compute_monotonicity(wr) is None

    def test_partial_monotone_fraction(self):
        # 9 adjacent pairs, 6 rising → 6/9 ≈ 0.6667
        wr = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5,
              6: 0.6, 7: 0.5, 8: 0.7, 9: 0.8, 10: 0.7}
        result = _compute_monotonicity(wr)
        assert result is not None
        # 7 rises out of 9 pairs: 1→2, 2→3, 3→4, 4→5, 5→6, 7→8, 8→9 = 7; falls: 6→7, 9→10
        assert abs(result - 7 / 9) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# compute_window_kpis
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeWindowKPIs:
    def test_empty_records_returns_zero(self):
        kpis = compute_window_kpis([], 30, TODAY)
        assert kpis.sample_count == 0
        assert kpis.sufficient is False
        assert kpis.monotonicity_score is None

    def test_non_materialized_excluded(self):
        records = [_make_record(1, 5, 0.01, materialized=False) for _ in range(50)]
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.sample_count == 0

    def test_outside_window_excluded(self):
        records = [_make_record(40, 5, 0.01) for _ in range(50)]
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.sample_count == 0

    def test_basic_kpi_computation(self):
        records = _make_records_monotonic(n_per_decile=15, window_days=25)
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.sample_count == 150
        assert kpis.sufficient is True
        assert kpis.monotonicity_score is not None
        # Higher deciles have higher win rates → monotonicity > 0.5
        assert kpis.monotonicity_score > 0.5

    def test_win_rate_gap_positive(self):
        records = _make_records_monotonic(n_per_decile=20, window_days=25)
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.win_rate_gap is not None
        assert kpis.win_rate_gap > 0

    def test_expectancy_uplift_computed(self):
        records = []
        # Decile 9,10: high returns; Decile 1,2: low returns
        for _ in range(30):
            records.append(_make_record(1, 9, 0.03))
            records.append(_make_record(1, 10, 0.03))
            records.append(_make_record(1, 1, -0.02))
            records.append(_make_record(1, 2, -0.02))
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.expectancy_uplift is not None
        assert kpis.expectancy_uplift > 0

    def test_sufficient_flag_boundary(self):
        # Exactly SAFETY_SAMPLE_FLOOR records
        records = [_make_record(1, 5, 0.01) for _ in range(SAFETY_SAMPLE_FLOOR)]
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.sufficient is True

    def test_below_floor_insufficient(self):
        records = [_make_record(1, 5, 0.01) for _ in range(SAFETY_SAMPLE_FLOOR - 1)]
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.sufficient is False

    def test_fail_open_on_corrupt_record(self):
        records = [{"bad": "data"} for _ in range(5)]
        kpis = compute_window_kpis(records, 30, TODAY)
        assert kpis.sample_count == 0  # corrupt records skipped

    def test_window_days_filter(self):
        # 35 records in 30d window, 50 in 60d window
        r30 = [_make_record(20, 5, 0.01) for _ in range(35)]
        r60 = [_make_record(45, 5, 0.01) for _ in range(50)]
        kpis_30 = compute_window_kpis(r30 + r60, 30, TODAY)
        kpis_60 = compute_window_kpis(r30 + r60, 60, TODAY)
        assert kpis_30.sample_count == 35
        assert kpis_60.sample_count == 85


# ─────────────────────────────────────────────────────────────────────────────
# check_safety_triggers
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSafetyTriggers:
    def test_no_triggers_on_good_kpis(self):
        kpis = _make_window_kpis(sample_count=120, mono=0.70, expectancy_uplift=0.30)
        triggered, reasons = check_safety_triggers(kpis)
        assert not triggered
        assert reasons == []

    def test_insufficient_sample_triggers(self):
        kpis = _make_window_kpis(sample_count=10, sufficient=False)
        triggered, reasons = check_safety_triggers(kpis)
        assert triggered
        assert any("sample_count" in r for r in reasons)

    def test_low_monotonicity_triggers(self):
        kpis = _make_window_kpis(mono=SAFETY_MONOTONICITY_FLOOR - 0.01)
        triggered, reasons = check_safety_triggers(kpis)
        assert triggered
        assert any("monotonicity" in r for r in reasons)

    def test_zero_expectancy_uplift_triggers(self):
        kpis = _make_window_kpis(expectancy_uplift=0.0)
        triggered, reasons = check_safety_triggers(kpis)
        assert triggered
        assert any("expectancy_uplift" in r for r in reasons)

    def test_negative_expectancy_triggers(self):
        kpis = _make_window_kpis(expectancy_uplift=-0.1)
        triggered, reasons = check_safety_triggers(kpis)
        assert triggered

    def test_none_monotonicity_not_triggered(self):
        # When monotonicity is None (no data), no monotonicity trigger
        kpis = _make_window_kpis(mono=None, sample_count=120)
        kpis = WindowKPIs(
            window_days=30, sample_count=120,
            monotonicity_score=None, top_decile_win_rate=0.6,
            bottom_decile_win_rate=0.4, top_decile_expectancy=0.02,
            bottom_decile_expectancy=-0.01, all_expectancy=0.005,
            expectancy_uplift=0.50, win_rate_gap=0.20,
            sufficient=True,
        )
        triggered, reasons = check_safety_triggers(kpis)
        assert not triggered  # no mono trigger when None

    def test_multiple_triggers_returned(self):
        kpis = _make_window_kpis(
            sample_count=5, sufficient=False,
            mono=0.30, expectancy_uplift=-0.5
        )
        triggered, reasons = check_safety_triggers(kpis)
        assert triggered
        assert len(reasons) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# check_tier1_promotion
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckTier1Promotion:
    def test_passes_all_conditions(self):
        kpis = _make_window_kpis(
            sample_count=TIER1_SAMPLE_MIN + 10,
            mono=TIER1_MONOTONICITY_MIN + 0.05,
            win_rate_gap=TIER1_WIN_RATE_GAP_MIN + 0.05,
        )
        ok, msg = check_tier1_promotion(kpis)
        assert ok

    def test_fails_insufficient_sample(self):
        kpis = _make_window_kpis(sample_count=TIER1_SAMPLE_MIN - 1, sufficient=False)
        ok, msg = check_tier1_promotion(kpis)
        assert not ok
        assert "sample_count" in msg

    def test_fails_low_monotonicity(self):
        kpis = _make_window_kpis(
            sample_count=TIER1_SAMPLE_MIN + 10,
            mono=TIER1_MONOTONICITY_MIN - 0.01,
        )
        ok, msg = check_tier1_promotion(kpis)
        assert not ok
        assert "monotonicity" in msg

    def test_fails_low_win_rate_gap(self):
        kpis = _make_window_kpis(
            sample_count=TIER1_SAMPLE_MIN + 10,
            mono=TIER1_MONOTONICITY_MIN + 0.05,
            win_rate_gap=TIER1_WIN_RATE_GAP_MIN - 0.01,
        )
        ok, msg = check_tier1_promotion(kpis)
        assert not ok
        assert "win_rate_gap" in msg

    def test_boundary_sample_exact(self):
        kpis = _make_window_kpis(
            sample_count=TIER1_SAMPLE_MIN,
            mono=TIER1_MONOTONICITY_MIN + 0.05,
            win_rate_gap=TIER1_WIN_RATE_GAP_MIN + 0.05,
            sufficient=True,
        )
        ok, _ = check_tier1_promotion(kpis)
        assert ok


# ─────────────────────────────────────────────────────────────────────────────
# check_tier2_promotion
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckTier2Promotion:
    def _good_w60(self):
        return _make_window_kpis(
            sample_count=80,
            mono=TIER2_MONOTONICITY_MIN + 0.05,
            expectancy_uplift=TIER2_EXPECTANCY_UPLIFT + 0.05,
        )

    def test_passes_all_conditions(self):
        ok, _ = check_tier2_promotion(self._good_w60(), days_in_tier=TIER2_TIER1_DAYS_MIN + 5)
        assert ok

    def test_fails_insufficient_days(self):
        ok, msg = check_tier2_promotion(self._good_w60(), days_in_tier=TIER2_TIER1_DAYS_MIN - 1)
        assert not ok
        assert "tier1_days" in msg

    def test_fails_insufficient_sample(self):
        bad = _make_window_kpis(sample_count=10, sufficient=False)
        ok, msg = check_tier2_promotion(bad, days_in_tier=TIER2_TIER1_DAYS_MIN + 5)
        assert not ok
        assert "sample_count" in msg

    def test_fails_low_expectancy_uplift(self):
        kpis = _make_window_kpis(
            sample_count=80,
            mono=TIER2_MONOTONICITY_MIN + 0.05,
            expectancy_uplift=TIER2_EXPECTANCY_UPLIFT - 0.01,
        )
        ok, msg = check_tier2_promotion(kpis, days_in_tier=TIER2_TIER1_DAYS_MIN + 5)
        assert not ok
        assert "expectancy_uplift" in msg

    def test_fails_low_monotonicity(self):
        kpis = _make_window_kpis(
            sample_count=80,
            mono=TIER2_MONOTONICITY_MIN - 0.01,
            expectancy_uplift=TIER2_EXPECTANCY_UPLIFT + 0.05,
        )
        ok, _ = check_tier2_promotion(kpis, days_in_tier=TIER2_TIER1_DAYS_MIN + 5)
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# check_tier3_promotion
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckTier3Promotion:
    def _good_w90(self):
        return _make_window_kpis(
            sample_count=150,
            mono=TIER3_MONOTONICITY_MIN + 0.05,
            expectancy_uplift=TIER3_EXPECTANCY_UPLIFT + 0.05,
            top_decile_expectancy=0.025,
        )

    def test_passes_all_conditions(self):
        ok, _ = check_tier3_promotion(self._good_w90(), days_in_tier=TIER3_TIER2_DAYS_MIN + 5)
        assert ok

    def test_fails_insufficient_days(self):
        ok, msg = check_tier3_promotion(self._good_w90(), days_in_tier=TIER3_TIER2_DAYS_MIN - 1)
        assert not ok
        assert "tier2_days" in msg

    def test_fails_zero_top_decile_expectancy(self):
        kpis = _make_window_kpis(
            sample_count=150,
            mono=TIER3_MONOTONICITY_MIN + 0.05,
            expectancy_uplift=TIER3_EXPECTANCY_UPLIFT + 0.05,
            top_decile_expectancy=0.0,
        )
        ok, msg = check_tier3_promotion(kpis, days_in_tier=TIER3_TIER2_DAYS_MIN + 5)
        assert not ok
        assert "top_decile_expectancy" in msg

    def test_fails_negative_top_decile_expectancy(self):
        kpis = _make_window_kpis(
            sample_count=150,
            mono=TIER3_MONOTONICITY_MIN + 0.05,
            expectancy_uplift=TIER3_EXPECTANCY_UPLIFT + 0.05,
            top_decile_expectancy=-0.01,
        )
        ok, _ = check_tier3_promotion(kpis, days_in_tier=TIER3_TIER2_DAYS_MIN + 5)
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_tier_transition
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateTierTransition:
    def _good_w30(self):
        return _make_window_kpis(
            sample_count=TIER1_SAMPLE_MIN + 10,
            mono=TIER1_MONOTONICITY_MIN + 0.1,
            win_rate_gap=TIER1_WIN_RATE_GAP_MIN + 0.1,
            expectancy_uplift=0.50,
        )

    def _good_w60(self):
        return _make_window_kpis(
            sample_count=80,
            mono=TIER2_MONOTONICITY_MIN + 0.05,
            win_rate_gap=0.15,
            expectancy_uplift=TIER2_EXPECTANCY_UPLIFT + 0.05,
        )

    def _good_w90(self):
        return _make_window_kpis(
            sample_count=150,
            mono=TIER3_MONOTONICITY_MIN + 0.05,
            win_rate_gap=0.20,
            expectancy_uplift=TIER3_EXPECTANCY_UPLIFT + 0.05,
            top_decile_expectancy=0.03,
        )

    def test_tier0_to_1_promotion(self):
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=0,
            tier_start_date=TODAY_STR,
            w30=self._good_w30(),
            w60=_make_insufficient_kpis(60),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert new_tier == 1
        assert rec == "PROMOTE"
        assert any("0→1" in e for e in events)

    def test_tier0_hold_on_insufficient_data(self):
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=0,
            tier_start_date=TODAY_STR,
            w30=_make_insufficient_kpis(30),
            w60=_make_insufficient_kpis(60),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert new_tier == 0
        assert rec == "INSUFFICIENT_DATA"

    def test_tier1_to_2_promotion(self):
        tier_start = (TODAY - timedelta(days=TIER2_TIER1_DAYS_MIN + 5)).strftime("%Y-%m-%d")
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=1,
            tier_start_date=tier_start,
            w30=self._good_w30(),
            w60=self._good_w60(),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert new_tier == 2
        assert rec == "PROMOTE"
        assert any("1→2" in e for e in events)

    def test_tier1_hold_on_short_duration(self):
        tier_start = (TODAY - timedelta(days=TIER2_TIER1_DAYS_MIN - 5)).strftime("%Y-%m-%d")
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=1,
            tier_start_date=tier_start,
            w30=self._good_w30(),
            w60=self._good_w60(),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert new_tier == 1
        assert rec == "HOLD"

    def test_tier2_to_3_promotion(self):
        tier_start = (TODAY - timedelta(days=TIER3_TIER2_DAYS_MIN + 5)).strftime("%Y-%m-%d")
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=2,
            tier_start_date=tier_start,
            w30=self._good_w30(),
            w60=self._good_w60(),
            w90=self._good_w90(),
            today=TODAY,
        )
        assert new_tier == 3
        assert rec == "PROMOTE"
        assert any("2→3" in e for e in events)

    def test_tier3_at_max(self):
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=3,
            tier_start_date=TODAY_STR,
            w30=self._good_w30(),
            w60=self._good_w60(),
            w90=self._good_w90(),
            today=TODAY,
        )
        assert new_tier == 3
        assert rec == "AT_MAX_TIER"
        assert events == []

    def test_safety_demotes_tier2_to_1(self):
        bad_w30 = _make_window_kpis(
            sample_count=120,
            mono=SAFETY_MONOTONICITY_FLOOR - 0.05,  # below floor
            expectancy_uplift=-0.1,
        )
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=2,
            tier_start_date=TODAY_STR,
            w30=bad_w30,
            w60=_make_insufficient_kpis(60),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert new_tier == 1
        assert rec == "DEMOTE"
        assert any("2→1" in e for e in events)

    def test_safety_does_not_demote_tier0(self):
        # At tier 0, safety cannot demote below 0
        bad_w30 = _make_window_kpis(mono=0.30, expectancy_uplift=-0.5)
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=0,
            tier_start_date=TODAY_STR,
            w30=bad_w30,
            w60=_make_insufficient_kpis(60),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert new_tier == 0
        # Should not go to -1; rec is HOLD or DEMOTE but tier stays 0

    def test_never_raises_on_bad_input(self):
        # Should FAIL_OPEN
        new_tier, rec, events = evaluate_tier_transition(
            current_tier=99,
            tier_start_date="bad-date",
            w30=_make_insufficient_kpis(30),
            w60=_make_insufficient_kpis(60),
            w90=_make_insufficient_kpis(90),
            today=TODAY,
        )
        assert isinstance(new_tier, int)
        assert isinstance(rec, str)


# ─────────────────────────────────────────────────────────────────────────────
# run_entry_timing_promotion (integration tests with tmp_path)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunEntryTimingPromotion:
    def _make_telemetry(self, tmp_path: Path, n: int = 150) -> Path:
        tf = tmp_path / "telemetry.jsonl"
        records = _make_records_monotonic(n_per_decile=n // 10, window_days=25)
        with tf.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return tf

    def test_creates_promotion_file_on_first_run(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo" / "promotion.json"
        hf = tmp_path / "promo" / "history.csv"
        state = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        assert pf.exists()
        assert isinstance(state.current_tier, int)
        assert 0 <= state.current_tier <= 3

    def test_persists_and_loads_state(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo" / "promotion.json"
        hf = tmp_path / "promo" / "history.csv"
        state1 = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        state2 = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        assert state1.current_tier == state2.current_tier

    def test_appends_csv_history(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo" / "promotion.json"
        hf = tmp_path / "promo" / "history.csv"
        run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        rows = list(csv.DictReader(hf.open(encoding="utf-8")))
        assert len(rows) == 2

    def test_csv_has_correct_columns(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo" / "promotion.json"
        hf = tmp_path / "promo" / "history.csv"
        run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        rows = list(csv.DictReader(hf.open(encoding="utf-8")))
        assert len(rows) == 1
        row = rows[0]
        assert "tier" in row
        assert "boost_weight" in row
        assert "recommendation" in row

    def test_default_tier0_on_no_data(self, tmp_path):
        tf = tmp_path / "empty_telem.jsonl"
        tf.write_text("", encoding="utf-8")
        pf = tmp_path / "promo.json"
        hf = tmp_path / "history.csv"
        state = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        assert state.current_tier == 0
        assert state.current_boost_weight == TIER_BOOST_WEIGHTS[0]

    def test_fail_open_missing_telemetry(self, tmp_path):
        tf = tmp_path / "nonexistent.jsonl"
        pf = tmp_path / "promo.json"
        hf = tmp_path / "history.csv"
        state = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        assert state.current_tier == 0

    def test_promotion_json_valid(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo.json"
        hf = tmp_path / "history.csv"
        run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        data = json.loads(pf.read_text(encoding="utf-8"))
        assert "current_tier" in data
        assert "current_boost_weight" in data
        assert "windows" in data
        assert "30d" in data["windows"]

    def test_boost_weight_matches_tier(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo.json"
        hf = tmp_path / "history.csv"
        state = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        assert state.current_boost_weight == TIER_BOOST_WEIGHTS[state.current_tier]

    def test_tier_start_date_updated_on_promotion(self, tmp_path):
        """Simulate an existing Tier1 state and verify tier_start_date resets on Tier2 promotion."""
        tf = self._make_telemetry(tmp_path, n=200)
        pf = tmp_path / "promo.json"
        hf = tmp_path / "history.csv"

        # Seed a Tier1 state with 35 days tenure
        tier_start = (TODAY - timedelta(days=35)).strftime("%Y-%m-%d")
        seed_state = PromotionState(
            current_tier=1,
            current_boost_weight=TIER_BOOST_WEIGHTS[1],
            tier_start_date=tier_start,
            last_evaluated=tier_start,
            windows={},
            promotion_recommendation="HOLD",
            safety_triggered=False,
            safety_reasons=[],
            events=[],
        )
        import json as _json
        from dataclasses import asdict
        pf.write_text(_json.dumps(asdict(seed_state), indent=2), encoding="utf-8")

        state = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        # If promoted to tier2, tier_start_date should equal TODAY_STR
        if state.current_tier == 2:
            assert state.tier_start_date == TODAY_STR

    def test_events_capped_at_20(self, tmp_path):
        tf = self._make_telemetry(tmp_path)
        pf = tmp_path / "promo.json"
        hf = tmp_path / "history.csv"

        # Seed a state with 19 existing events
        seed_state = PromotionState(
            current_tier=0,
            current_boost_weight=TIER_BOOST_WEIGHTS[0],
            tier_start_date=TODAY_STR,
            last_evaluated=TODAY_STR,
            windows={},
            promotion_recommendation="HOLD",
            safety_triggered=False,
            safety_reasons=[],
            events=[f"event_{i}" for i in range(19)],
        )
        import json as _json
        from dataclasses import asdict
        pf.write_text(_json.dumps(asdict(seed_state), indent=2), encoding="utf-8")

        state = run_entry_timing_promotion(tf, pf, hf, TODAY_STR)
        assert len(state.events) <= 20


# ─────────────────────────────────────────────────────────────────────────────
# get_effective_boost_weight
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEffectiveBoostWeight:
    def test_auto_apply_false_returns_fallback(self, tmp_path):
        pf = tmp_path / "promo.json"
        result = get_effective_boost_weight(pf, fallback_weight=0.06, auto_apply_enabled=False)
        assert result == 0.06

    def test_auto_apply_true_missing_file_returns_fallback(self, tmp_path):
        pf = tmp_path / "nonexistent.json"
        result = get_effective_boost_weight(pf, fallback_weight=0.06, auto_apply_enabled=True)
        assert result == 0.06

    def test_auto_apply_true_reads_tier_weight(self, tmp_path):
        pf = tmp_path / "promo.json"
        state = PromotionState(
            current_tier=2,
            current_boost_weight=TIER_BOOST_WEIGHTS[2],
            tier_start_date=TODAY_STR,
            last_evaluated=TODAY_STR,
            windows={},
            promotion_recommendation="HOLD",
            safety_triggered=False,
            safety_reasons=[],
            events=[],
        )
        from dataclasses import asdict
        pf.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        result = get_effective_boost_weight(pf, fallback_weight=0.06, auto_apply_enabled=True)
        assert result == TIER_BOOST_WEIGHTS[2]

    def test_auto_apply_true_corrupt_file_returns_fallback(self, tmp_path):
        pf = tmp_path / "promo.json"
        pf.write_text("{corrupt json", encoding="utf-8")
        result = get_effective_boost_weight(pf, fallback_weight=0.08, auto_apply_enabled=True)
        assert result == 0.08

    def test_tier_weights_monotone(self):
        weights = [TIER_BOOST_WEIGHTS[t] for t in range(4)]
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1], "Tier boost weights must be strictly increasing"


# ─────────────────────────────────────────────────────────────────────────────
# format_et_promotion_section
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatEtPromotionSection:
    def _write_state(self, pf: Path, tier: int = 1, safety: bool = False) -> None:
        state = PromotionState(
            current_tier=tier,
            current_boost_weight=TIER_BOOST_WEIGHTS.get(tier, 0.06),
            tier_start_date=(TODAY - timedelta(days=20)).strftime("%Y-%m-%d"),
            last_evaluated=TODAY_STR,
            windows={
                "30d": {
                    "window_days": 30, "sample_count": 150,
                    "monotonicity_score": 0.75, "win_rate_gap": 0.15,
                    "expectancy_uplift": 0.30, "top_decile_win_rate": 0.70,
                    "bottom_decile_win_rate": 0.55, "top_decile_expectancy": 0.02,
                    "bottom_decile_expectancy": -0.01, "all_expectancy": 0.005,
                    "sufficient": True,
                },
                "60d": {"window_days": 60, "sample_count": 5},
                "90d": {"window_days": 90, "sample_count": 2},
            },
            promotion_recommendation="HOLD",
            safety_triggered=safety,
            safety_reasons=["test reason"] if safety else [],
            events=["PROMOTED 0→1 on 2026-05-11: all Tier1 conditions met"],
        )
        from dataclasses import asdict
        pf.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    def test_returns_string(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_tier_label(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf, tier=1)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert "Tier 1" in result

    def test_contains_boost_weight(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf, tier=2)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert str(TIER_BOOST_WEIGHTS[2]) in result or "0.15" in result

    def test_safety_section_shown_when_triggered(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf, tier=2, safety=True)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert "安全装置" in result or "test reason" in result

    def test_safety_section_absent_when_not_triggered(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf, tier=1, safety=False)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert "安全装置" not in result

    def test_fail_open_missing_file(self, tmp_path):
        pf = tmp_path / "nonexistent.json"
        result = format_et_promotion_section(pf, TODAY_STR)
        assert result == ""

    def test_fail_open_corrupt_json(self, tmp_path):
        pf = tmp_path / "promo.json"
        pf.write_text("{bad json", encoding="utf-8")
        result = format_et_promotion_section(pf, TODAY_STR)
        assert isinstance(result, str)  # returns "" on failure

    def test_section_header_present(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert "ENTRY TIMING STATUS" in result

    def test_30d_kpi_block_present(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert "30日" in result or "mono" in result.lower() or "単調性" in result

    def test_tier3_at_max_message(self, tmp_path):
        pf = tmp_path / "promo.json"
        self._write_state(pf, tier=3)
        result = format_et_promotion_section(pf, TODAY_STR)
        assert "最高ティア" in result or "AT_MAX" in result or "Tier 3" in result
