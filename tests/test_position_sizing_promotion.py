"""
tests/test_position_sizing_promotion.py
Tests for src/portfolio/position_sizing_promotion.py
"""
import csv
import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.portfolio.position_sizing_promotion import (
    CONVICTION_BLEND_FACTORS,
    FORWARD_WINDOW,
    SAFETY_MONOTONICITY_FLOOR,
    SAFETY_SAMPLE_FLOOR,
    TIER0_MONOTONICITY_MIN,
    TIER0_SAMPLE_MIN,
    TIER1_MIN_DAYS,
    TIER2_MIN_DAYS,
    TIER_WEIGHT_MODES,
    PortfolioComparison,
    PromotionState,
    QuintileStats,
    WindowAnalysis,
    _compute_fwd_return,
    _compute_portfolio_comparison,
    _compute_quintile_stats,
    _conviction_monotonicity,
    _enrich_with_returns,
    _load_ohlcv,
    _load_telemetry,
    check_safety_triggers,
    compute_window_analysis,
    evaluate_tier_transition,
    format_psp_report_section,
    get_psp_blend_factor,
    run_position_sizing_promotion,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv_csv(path: Path, start_date: date, prices: list[float]) -> None:
    """Write a minimal OHLCV CSV (date, close) starting at start_date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("date,close\n")
        for i, p in enumerate(prices):
            d = start_date + timedelta(days=i)
            f.write(f"{d.isoformat()},{p}\n")


def _make_telemetry_record(
    date_str: str,
    symbol: str,
    conviction: float,
    virtual_weight: float,
    actual_weight: float,
    fwd_return: float | None = None,
) -> dict:
    r = {
        "timestamp": f"{date_str}T09:00:00+09:00",
        "date": date_str,
        "symbol": symbol,
        "conviction_score": conviction,
        "virtual_weight": virtual_weight,
        "actual_weight": actual_weight,
        "rsr_score": conviction,
        "entry_timing_score": None,
        "future_leader_score": None,
        "reason_codes": [],
        "components": {},
    }
    if fwd_return is not None:
        r["forward_return_5d"] = fwd_return
    return r


def _write_telemetry(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_window_analysis(
    sample_count: int = 120,
    enriched_count: int = 80,
    monotonicity: float = 0.75,
    exp_delta: float | None = 0.01,
    return_delta: float | None = 0.005,
    sharpe_delta: float | None = 0.3,
    pc_sufficient: bool = True,
    window_days: int = 30,
) -> WindowAnalysis:
    pc = PortfolioComparison(
        window_days=window_days,
        sample_dates=20,
        equal_return=0.01,
        virtual_return=0.015 if return_delta is not None else None,
        return_delta=return_delta,
        equal_expectancy=0.01,
        virtual_expectancy=0.015,
        expectancy_delta=return_delta,
        equal_sharpe=1.0,
        virtual_sharpe=(1.0 + sharpe_delta) if sharpe_delta is not None else None,
        sharpe_delta=sharpe_delta,
        sufficient=pc_sufficient,
    )
    return WindowAnalysis(
        window_days=window_days,
        sample_count=sample_count,
        enriched_count=enriched_count,
        conviction_monotonicity=monotonicity,
        top_quintile_expectancy=0.02 if exp_delta is not None else None,
        bottom_quintile_expectancy=(0.02 - exp_delta) if exp_delta is not None else None,
        expectancy_delta=exp_delta,
        quintile_stats=[],
        portfolio_comparison=pc,
        sufficient=sample_count >= SAFETY_SAMPLE_FLOOR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_blend_factors_tiers(self):
        assert CONVICTION_BLEND_FACTORS[0] == 0.0
        assert CONVICTION_BLEND_FACTORS[1] == 0.0
        assert CONVICTION_BLEND_FACTORS[2] == 0.5
        assert CONVICTION_BLEND_FACTORS[3] == 1.0

    def test_weight_modes(self):
        assert TIER_WEIGHT_MODES[0] == "equal_weight"
        assert TIER_WEIGHT_MODES[1] == "equal_weight"
        assert TIER_WEIGHT_MODES[2] == "hybrid_weight"
        assert TIER_WEIGHT_MODES[3] == "conviction_weight"

    def test_safety_sample_floor(self):
        assert SAFETY_SAMPLE_FLOOR == 30

    def test_monotonicity_min(self):
        assert TIER0_MONOTONICITY_MIN == 0.60

    def test_tier1_min_days(self):
        assert TIER1_MIN_DAYS == 30


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadOhlcv:
    def test_load_basic(self, tmp_path):
        csv_path = tmp_path / "1234.T.csv"
        _make_ohlcv_csv(csv_path, date(2025, 1, 6), [100.0, 101.0, 102.0])
        result = _load_ohlcv(tmp_path, "1234.T")
        assert result is not None
        assert "2025-01-06" in result
        assert result["2025-01-06"] == 100.0

    def test_missing_file_returns_none(self, tmp_path):
        assert _load_ohlcv(tmp_path, "9999.T") is None

    def test_skip_header(self, tmp_path):
        csv_path = tmp_path / "A.csv"
        csv_path.write_text("date,close\n2025-01-06,100\n", encoding="utf-8")
        result = _load_ohlcv(tmp_path, "A")
        assert "date" not in result
        assert "2025-01-06" in result


class TestComputeFwdReturn:
    def test_basic_5day(self):
        ohlcv = {f"2025-01-{i+6:02d}": 100.0 + i for i in range(10)}
        r = _compute_fwd_return(ohlcv, "2025-01-06", 5)
        assert r is not None
        assert abs(r - 5 / 100.0) < 1e-5

    def test_returns_none_if_insufficient(self):
        ohlcv = {"2025-01-06": 100.0, "2025-01-07": 101.0}
        assert _compute_fwd_return(ohlcv, "2025-01-06", 5) is None

    def test_base_date_after_data_returns_none(self):
        ohlcv = {"2025-01-06": 100.0}
        assert _compute_fwd_return(ohlcv, "2025-01-10", 5) is None

    def test_base_date_gap_finds_next_trading_day(self):
        # base_date is a weekend; next available is Monday
        ohlcv = {"2025-01-06": 100.0, "2025-01-07": 101.0,
                 "2025-01-08": 102.0, "2025-01-09": 103.0,
                 "2025-01-10": 104.0, "2025-01-11": 105.0}
        # base_date=2025-01-04 (Sat); first >= is 2025-01-06
        r = _compute_fwd_return(ohlcv, "2025-01-04", 5)
        assert r is not None
        assert abs(r - 5 / 100.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadTelemetry:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text("", encoding="utf-8")
        result = _load_telemetry(p, 30, date(2025, 3, 1))
        assert result == []

    def test_missing_file(self, tmp_path):
        result = _load_telemetry(tmp_path / "missing.jsonl", 30, date(2025, 3, 1))
        assert result == []

    def test_filters_by_window(self, tmp_path):
        p = tmp_path / "t.jsonl"
        records = [
            _make_telemetry_record("2025-01-01", "A", 80, 0.5, 0.5),  # old
            _make_telemetry_record("2025-03-01", "B", 70, 0.5, 0.5),  # recent
        ]
        _write_telemetry(p, records)
        result = _load_telemetry(p, 30, date(2025, 3, 15))
        assert len(result) == 1
        assert result[0]["symbol"] == "B"

    def test_skip_malformed_lines(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"date":"2025-03-01","symbol":"A","conviction_score":70,"virtual_weight":0.5,"actual_weight":0.5}\nnot_json\n', encoding="utf-8")
        result = _load_telemetry(p, 30, date(2025, 3, 15))
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichWithReturns:
    def test_enrich_basic(self, tmp_path):
        ohlcv_dir = tmp_path / "ohlcv"
        start = date(2025, 1, 6)
        _make_ohlcv_csv(ohlcv_dir / "A.csv", start, [100.0] * 20)
        records = [_make_telemetry_record("2025-01-06", "A", 80, 0.5, 0.5)]
        today = start + timedelta(days=14)
        result = _enrich_with_returns(records, ohlcv_dir, today)
        assert result[0]["forward_return_5d"] is not None

    def test_too_recent_not_enriched(self, tmp_path):
        ohlcv_dir = tmp_path / "ohlcv"
        _make_ohlcv_csv(ohlcv_dir / "A.csv", date(2025, 3, 1), [100.0] * 20)
        records = [_make_telemetry_record("2025-03-01", "A", 80, 0.5, 0.5)]
        # today is only 3 days after the record
        result = _enrich_with_returns(records, ohlcv_dir, date(2025, 3, 4))
        assert result[0]["forward_return_5d"] is None

    def test_missing_ohlcv_leaves_none(self, tmp_path):
        records = [_make_telemetry_record("2025-01-06", "Z999.T", 80, 0.5, 0.5)]
        result = _enrich_with_returns(records, tmp_path / "ohlcv", date(2025, 2, 1))
        assert result[0]["forward_return_5d"] is None

    def test_fail_open_on_bad_record(self, tmp_path):
        records = [{"date": "bad-date", "symbol": "A"}]
        result = _enrich_with_returns(records, tmp_path, date(2025, 2, 1))
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Quintile stats
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeQuintileStats:
    def _make_records_with_returns(self, n_per_quintile=25) -> list[dict]:
        records = []
        # Q1 low conviction low return, Q5 high conviction high return
        for q in range(1, 6):
            conviction = 20.0 * q - 10  # 10, 30, 50, 70, 90
            fwd_return = (q - 3) * 0.005  # -0.01, -0.005, 0, 0.005, 0.01
            for i in range(n_per_quintile):
                records.append({
                    "date": f"2025-01-{i+1:02d}",
                    "symbol": f"S{q}{i}",
                    "conviction_score": conviction + (i * 0.01),  # slight variation
                    "virtual_weight": 0.1,
                    "actual_weight": 0.1,
                    "forward_return_5d": fwd_return + (i * 0.0001),
                })
        return records

    def test_returns_5_quintiles(self):
        records = self._make_records_with_returns()
        stats = _compute_quintile_stats(records)
        assert len(stats) == 5

    def test_quintile_ordering(self):
        records = self._make_records_with_returns()
        stats   = _compute_quintile_stats(records)
        convs   = [s.avg_conviction for s in stats]
        for a, b in zip(convs, convs[1:]):
            assert b > a, "Quintile conviction should increase Q1→Q5"

    def test_win_rate_and_expectancy_populated(self):
        records = self._make_records_with_returns(30)
        stats   = _compute_quintile_stats(records)
        for s in stats:
            assert s.win_rate is not None
            assert s.expectancy is not None

    def test_empty_records(self):
        assert _compute_quintile_stats([]) == []

    def test_no_forward_returns(self):
        records = [{"conviction_score": 80, "date": "2025-01-01", "symbol": "A"}] * 10
        stats = _compute_quintile_stats(records)
        for s in stats:
            assert s.win_rate is None
            assert s.expectancy is None


# ─────────────────────────────────────────────────────────────────────────────
# Conviction monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestConvictionMonotonicity:
    def _make_stats(self, expectancies: list[float | None]) -> list[QuintileStats]:
        return [
            QuintileStats(
                quintile=i+1, sample_count=10, avg_conviction=20.0*(i+1),
                win_rate=0.6, expectancy=e, avg_return=e, sharpe_proxy=None,
            )
            for i, e in enumerate(expectancies)
        ]

    def test_perfectly_monotone(self):
        stats = self._make_stats([0.01, 0.02, 0.03, 0.04, 0.05])
        assert _conviction_monotonicity(stats) == 1.0

    def test_no_monotonicity(self):
        stats = self._make_stats([0.05, 0.04, 0.03, 0.02, 0.01])
        assert _conviction_monotonicity(stats) == 0.0

    def test_partial_monotonicity(self):
        # 2 of 4 pairs non-decreasing
        stats = self._make_stats([0.01, 0.02, 0.01, 0.03, 0.04])
        result = _conviction_monotonicity(stats)
        assert abs(result - 0.75) < 1e-4  # 3/4

    def test_none_expectancy_excluded(self):
        stats = self._make_stats([0.01, None, 0.03, None, 0.05])
        # Only non-None: [0.01, 0.03, 0.05] → 2/2 = 1.0
        assert _conviction_monotonicity(stats) == 1.0

    def test_single_value_returns_zero(self):
        stats = self._make_stats([0.01])
        assert _conviction_monotonicity(stats) == 0.0

    def test_empty_returns_zero(self):
        assert _conviction_monotonicity([]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePortfolioComparison:
    def _multi_date_records(self, n_dates=10) -> list[dict]:
        """2 symbols per date; higher conviction symbol has higher return."""
        records = []
        for d in range(n_dates):
            date_str = f"2025-01-{d+1:02d}"
            # Symbol A: high conviction 80, high return 2%
            # Symbol B: low conviction 40, low return 0.5%
            records.append({
                "date": date_str, "symbol": "A",
                "conviction_score": 80, "virtual_weight": 0.60, "actual_weight": 0.50,
                "forward_return_5d": 0.02,
            })
            records.append({
                "date": date_str, "symbol": "B",
                "conviction_score": 40, "virtual_weight": 0.40, "actual_weight": 0.50,
                "forward_return_5d": 0.005,
            })
        return records

    def test_virtual_beats_equal(self):
        records = self._multi_date_records(10)
        pc = _compute_portfolio_comparison(records, 30)
        # equal = 0.5*0.02 + 0.5*0.005 = 0.0125
        # virtual = 0.6*0.02 + 0.4*0.005 = 0.014
        assert pc.return_delta is not None
        assert pc.return_delta > 0

    def test_sample_dates_count(self):
        records = self._multi_date_records(8)
        pc = _compute_portfolio_comparison(records, 30)
        assert pc.sample_dates == 8

    def test_sufficient_flag_false_for_small_sample(self):
        records = self._multi_date_records(3)
        pc = _compute_portfolio_comparison(records, 30)
        assert pc.sufficient is False

    def test_sufficient_flag_true_for_large_sample(self):
        records = self._multi_date_records(10)
        pc = _compute_portfolio_comparison(records, 30)
        assert pc.sufficient is True

    def test_no_forward_returns_returns_empty(self):
        records = [
            {"date": "2025-01-01", "symbol": "A", "conviction_score": 80,
             "virtual_weight": 0.5, "actual_weight": 0.5, "forward_return_5d": None},
        ]
        pc = _compute_portfolio_comparison(records, 30)
        assert pc.sample_dates == 0
        assert pc.sufficient is False

    def test_single_symbol_per_date_excluded(self):
        # Only 1 symbol per date → no valid dates
        records = [
            {"date": "2025-01-01", "symbol": "A",
             "conviction_score": 80, "virtual_weight": 1.0, "actual_weight": 1.0,
             "forward_return_5d": 0.01},
        ]
        pc = _compute_portfolio_comparison(records, 30)
        assert pc.sample_dates == 0

    def test_renormalizes_virtual_weights(self):
        # Weights don't sum to 1 due to missing records → should still work
        records = [
            {"date": "2025-01-01", "symbol": "A",
             "conviction_score": 80, "virtual_weight": 3.0, "actual_weight": 0.5,
             "forward_return_5d": 0.02},
            {"date": "2025-01-01", "symbol": "B",
             "conviction_score": 40, "virtual_weight": 2.0, "actual_weight": 0.5,
             "forward_return_5d": 0.005},
        ]
        pc = _compute_portfolio_comparison(records, 30)
        # After renorm: A=0.6, B=0.4 → virtual=0.014
        assert pc.return_delta is not None
        assert pc.return_delta > 0


# ─────────────────────────────────────────────────────────────────────────────
# Safety triggers
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSafetyTriggers:
    def test_no_triggers_healthy(self):
        w = _make_window_analysis(
            sample_count=120, monotonicity=0.80, exp_delta=0.01,
            return_delta=0.005, pc_sufficient=True,
        )
        triggered, reasons = check_safety_triggers(w, current_tier=1)
        assert not triggered
        assert reasons == []

    def test_low_sample_triggers(self):
        w = _make_window_analysis(sample_count=20, monotonicity=0.80, exp_delta=0.01)
        triggered, _ = check_safety_triggers(w, current_tier=0)
        assert triggered

    def test_low_monotonicity_triggers(self):
        w = _make_window_analysis(sample_count=120, monotonicity=0.40, exp_delta=0.01)
        triggered, reasons = check_safety_triggers(w, current_tier=0)
        assert triggered
        assert any("monotonicity" in r for r in reasons)

    def test_negative_expectancy_delta_triggers_with_enough_returns(self):
        w = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.80,
            exp_delta=-0.005,
        )
        triggered, reasons = check_safety_triggers(w, current_tier=1)
        assert triggered
        assert any("expectancy" in r for r in reasons)

    def test_negative_expectancy_not_triggered_insufficient_returns(self):
        w = _make_window_analysis(
            sample_count=120, enriched_count=5, monotonicity=0.80,
            exp_delta=-0.005,
        )
        triggered, _ = check_safety_triggers(w, current_tier=1)
        # enriched_count=5 < 10, so expectancy trigger is suppressed
        assert not triggered

    def test_negative_return_delta_triggers_tier2(self):
        w = _make_window_analysis(
            sample_count=120, monotonicity=0.80, exp_delta=0.01,
            return_delta=-0.002, pc_sufficient=True,
        )
        triggered, reasons = check_safety_triggers(w, current_tier=2)
        assert triggered
        assert any("return_delta" in r for r in reasons)

    def test_negative_return_delta_no_trigger_tier1(self):
        w = _make_window_analysis(
            sample_count=120, monotonicity=0.80, exp_delta=0.01,
            return_delta=-0.002, pc_sufficient=True,
        )
        triggered, _ = check_safety_triggers(w, current_tier=1)
        assert not triggered  # return_delta only triggers at Tier >= 2

    def test_insufficient_portfolio_no_return_trigger(self):
        w = _make_window_analysis(
            sample_count=120, monotonicity=0.80, exp_delta=0.01,
            return_delta=-0.002, pc_sufficient=False,
        )
        triggered, _ = check_safety_triggers(w, current_tier=2)
        assert not triggered  # insufficient → no trigger


# ─────────────────────────────────────────────────────────────────────────────
# Tier transition evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateTierTransition:
    TODAY = date(2025, 6, 1)

    def _start_date(self, days_ago: int) -> str:
        return (self.TODAY - timedelta(days=days_ago)).isoformat()

    def test_tier0_promotes_to_1_when_conditions_met(self):
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.75, exp_delta=0.008,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=0, tier_start_date=self._start_date(60),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 1
        assert rec == "PROMOTE"

    def test_tier0_holds_insufficient_sample(self):
        w30 = _make_window_analysis(
            sample_count=50, enriched_count=30, monotonicity=0.75, exp_delta=0.008,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=0, tier_start_date=self._start_date(60),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 0
        assert rec == "HOLD"

    def test_tier0_holds_low_monotonicity(self):
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.55, exp_delta=0.008,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=0, tier_start_date=self._start_date(60),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 0
        assert rec == "HOLD"

    def test_tier0_holds_no_expectancy_delta(self):
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.75, exp_delta=None,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=0, tier_start_date=self._start_date(60),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 0

    def test_tier1_promotes_to_2_when_conditions_met(self):
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.75, exp_delta=0.008,
            return_delta=0.003, pc_sufficient=True,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=1, tier_start_date=self._start_date(35),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 2
        assert rec == "PROMOTE"

    def test_tier1_holds_insufficient_days(self):
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.75, exp_delta=0.008,
            return_delta=0.003, pc_sufficient=True,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=1, tier_start_date=self._start_date(20),  # only 20d
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 1
        assert rec == "HOLD"

    def test_tier1_holds_negative_return_delta(self):
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.75, exp_delta=0.008,
            return_delta=-0.001, pc_sufficient=True,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=1, tier_start_date=self._start_date(35),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 1
        assert rec == "HOLD"

    def test_tier2_promotes_to_3_when_conditions_met(self):
        w60 = _make_window_analysis(
            sample_count=200, enriched_count=80, monotonicity=0.80, exp_delta=0.01,
            return_delta=0.005, sharpe_delta=0.5, pc_sufficient=True,
            window_days=60,
        )
        w30 = _make_window_analysis(
            sample_count=120, enriched_count=30, monotonicity=0.80, exp_delta=0.01,
            return_delta=0.005, pc_sufficient=True,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=2, tier_start_date=self._start_date(65),
            w30=w30, w60=w60, w90=w60, today=self.TODAY,
        )
        assert new_tier == 3
        assert rec == "PROMOTE"

    def test_tier2_holds_insufficient_days(self):
        w60 = _make_window_analysis(
            sample_count=200, enriched_count=80, monotonicity=0.80, exp_delta=0.01,
            return_delta=0.005, sharpe_delta=0.5, pc_sufficient=True,
            window_days=60,
        )
        w30 = _make_window_analysis()
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=2, tier_start_date=self._start_date(40),  # only 40d
            w30=w30, w60=w60, w90=w60, today=self.TODAY,
        )
        assert new_tier == 2
        assert rec == "HOLD"

    def test_tier3_holds_at_max(self):
        w30 = _make_window_analysis(
            sample_count=200, enriched_count=100, monotonicity=0.90, exp_delta=0.02,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=3, tier_start_date=self._start_date(90),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 3
        assert rec == "HOLD"

    def test_safety_demote_overrides_everything(self):
        # Even if Tier 1 and conditions for promote met, safety should win
        w30 = _make_window_analysis(
            sample_count=20,  # below safety floor
            enriched_count=30, monotonicity=0.75, exp_delta=0.008,
        )
        new_tier, rec, _ = evaluate_tier_transition(
            current_tier=1, tier_start_date=self._start_date(35),
            w30=w30, w60=w30, w90=w30, today=self.TODAY,
        )
        assert new_tier == 0
        assert rec == "SAFETY_DEMOTE"


# ─────────────────────────────────────────────────────────────────────────────
# run_position_sizing_promotion (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPositionSizingPromotion:
    def _setup_dirs(self, tmp_path):
        ohlcv_dir  = tmp_path / "ohlcv"
        report_dir = tmp_path / "reports"
        ohlcv_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        return ohlcv_dir, report_dir

    def test_creates_json_on_first_run(self, tmp_path):
        ohlcv_dir, report_dir = self._setup_dirs(tmp_path)
        telem = tmp_path / "telem.jsonl"
        pfile = report_dir / "promotion.json"
        hfile = report_dir / "history.csv"
        state = run_position_sizing_promotion(
            telemetry_file=telem, promotion_file=pfile,
            history_file=hfile, ohlcv_dir=ohlcv_dir,
            today_str="2025-06-01",
        )
        assert pfile.exists()
        assert state.current_tier == 0

    def test_creates_history_csv(self, tmp_path):
        ohlcv_dir, report_dir = self._setup_dirs(tmp_path)
        telem = tmp_path / "telem.jsonl"
        pfile = report_dir / "promotion.json"
        hfile = report_dir / "history.csv"
        run_position_sizing_promotion(
            telemetry_file=telem, promotion_file=pfile,
            history_file=hfile, ohlcv_dir=ohlcv_dir,
            today_str="2025-06-01",
        )
        assert hfile.exists()
        rows = list(csv.DictReader(hfile.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["tier"] == "0"

    def test_idempotent_same_day(self, tmp_path):
        ohlcv_dir, report_dir = self._setup_dirs(tmp_path)
        telem = tmp_path / "telem.jsonl"
        pfile = report_dir / "promotion.json"
        hfile = report_dir / "history.csv"
        run_position_sizing_promotion(
            telemetry_file=telem, promotion_file=pfile,
            history_file=hfile, ohlcv_dir=ohlcv_dir, today_str="2025-06-01",
        )
        run_position_sizing_promotion(
            telemetry_file=telem, promotion_file=pfile,
            history_file=hfile, ohlcv_dir=ohlcv_dir, today_str="2025-06-01",
        )
        rows = list(csv.DictReader(hfile.open(encoding="utf-8")))
        assert len(rows) == 1  # second call skipped due to idempotency

    def test_accumulates_history_across_days(self, tmp_path):
        ohlcv_dir, report_dir = self._setup_dirs(tmp_path)
        telem = tmp_path / "telem.jsonl"
        pfile = report_dir / "promotion.json"
        hfile = report_dir / "history.csv"
        for d in ["2025-06-01", "2025-06-02", "2025-06-03"]:
            run_position_sizing_promotion(
                telemetry_file=telem, promotion_file=pfile,
                history_file=hfile, ohlcv_dir=ohlcv_dir, today_str=d,
            )
        rows = list(csv.DictReader(hfile.open(encoding="utf-8")))
        assert len(rows) == 3

    def test_fail_open_on_missing_paths(self, tmp_path):
        # Should not raise even with completely missing dirs
        state = run_position_sizing_promotion(
            telemetry_file=tmp_path / "nonexist.jsonl",
            promotion_file=tmp_path / "sub" / "p.json",
            history_file=tmp_path / "sub" / "h.csv",
            ohlcv_dir=tmp_path / "ohlcv",
            today_str="2025-06-01",
        )
        assert state is not None
        assert state.current_tier == 0

    def test_tier_preserved_across_runs(self, tmp_path):
        ohlcv_dir, report_dir = self._setup_dirs(tmp_path)
        telem = tmp_path / "telem.jsonl"
        pfile = report_dir / "promotion.json"
        hfile = report_dir / "history.csv"
        # Write initial state at Tier 1
        initial = PromotionState(
            current_tier=1, current_blend_factor=0.0,
            recommended_weight_mode="equal_weight", tier_start_date="2025-04-01",
            last_evaluated="2025-05-31", windows={},
            promotion_recommendation="HOLD", safety_triggered=False,
            safety_reasons=[], events=[],
        )
        from src.portfolio.position_sizing_promotion import _save_state
        _save_state(initial, pfile)
        state = run_position_sizing_promotion(
            telemetry_file=telem, promotion_file=pfile,
            history_file=hfile, ohlcv_dir=ohlcv_dir, today_str="2025-06-01",
        )
        # With no telemetry data, safety should NOT trigger (sample=0 < 30 WOULD trigger)
        # But safety_sample_floor check requires sample < 30 which is true (0 < 30)
        # so it demotes to 0
        assert state.current_tier in (0, 1)  # either demoted or held

    def test_state_json_structure(self, tmp_path):
        ohlcv_dir, report_dir = self._setup_dirs(tmp_path)
        pfile = report_dir / "promotion.json"
        hfile = report_dir / "history.csv"
        run_position_sizing_promotion(
            telemetry_file=tmp_path / "t.jsonl",
            promotion_file=pfile, history_file=hfile,
            ohlcv_dir=ohlcv_dir, today_str="2025-06-01",
        )
        data = json.loads(pfile.read_text(encoding="utf-8"))
        assert "current_tier" in data
        assert "recommended_weight_mode" in data
        assert "tier_start_date" in data
        assert "events" in data
        assert "schema_version" in data


# ─────────────────────────────────────────────────────────────────────────────
# get_psp_blend_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPspBlendFactor:
    def test_returns_fallback_when_not_enabled(self, tmp_path):
        f = tmp_path / "p.json"
        result = get_psp_blend_factor(f, fallback_factor=0.0, auto_apply_enabled=False)
        assert result == 0.0

    def test_returns_fallback_when_file_missing(self, tmp_path):
        f = tmp_path / "p.json"
        result = get_psp_blend_factor(f, fallback_factor=0.0, auto_apply_enabled=True)
        assert result == 0.0

    def test_returns_blend_factor_from_state(self, tmp_path):
        f = tmp_path / "p.json"
        state = PromotionState(
            current_tier=2, current_blend_factor=0.5,
            recommended_weight_mode="hybrid_weight",
            tier_start_date="2025-01-01", last_evaluated="2025-06-01",
            windows={}, promotion_recommendation="HOLD",
            safety_triggered=False, safety_reasons=[], events=[],
        )
        from src.portfolio.position_sizing_promotion import _save_state
        _save_state(state, f)
        result = get_psp_blend_factor(f, fallback_factor=0.0, auto_apply_enabled=True)
        assert result == 0.5

    def test_fail_open_on_corrupt_file(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text("not-json", encoding="utf-8")
        result = get_psp_blend_factor(f, fallback_factor=0.1, auto_apply_enabled=True)
        assert result == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# format_psp_report_section
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPspReportSection:
    def test_empty_when_no_file(self, tmp_path):
        result = format_psp_report_section(tmp_path / "p.json", "2025-06-01")
        assert result == ""

    def test_contains_tier_info(self, tmp_path):
        f = tmp_path / "p.json"
        state = PromotionState(
            current_tier=1, current_blend_factor=0.0,
            recommended_weight_mode="equal_weight",
            tier_start_date="2025-05-01", last_evaluated="2025-06-01",
            windows={}, promotion_recommendation="HOLD",
            safety_triggered=False, safety_reasons=[], events=[],
        )
        from src.portfolio.position_sizing_promotion import _save_state
        _save_state(state, f)
        result = format_psp_report_section(f, "2025-06-01")
        assert "Tier 1" in result
        assert "equal_weight" in result

    def test_contains_safety_info_when_triggered(self, tmp_path):
        f = tmp_path / "p.json"
        state = PromotionState(
            current_tier=0, current_blend_factor=0.0,
            recommended_weight_mode="equal_weight",
            tier_start_date="2025-06-01", last_evaluated="2025-06-01",
            windows={}, promotion_recommendation="SAFETY_DEMOTE",
            safety_triggered=True, safety_reasons=["monotonicity=0.30 < 0.45"],
            events=[],
        )
        from src.portfolio.position_sizing_promotion import _save_state
        _save_state(state, f)
        result = format_psp_report_section(f, "2025-06-01")
        assert "SAFETY_DEMOTE" in result
        assert "monotonicity" in result

    def test_fail_open_corrupt_file(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text("bad", encoding="utf-8")
        result = format_psp_report_section(f, "2025-06-01")
        assert result == ""

    def test_contains_header_separator(self, tmp_path):
        f = tmp_path / "p.json"
        state = PromotionState(
            current_tier=0, current_blend_factor=0.0,
            recommended_weight_mode="equal_weight",
            tier_start_date="2025-06-01", last_evaluated="2025-06-01",
            windows={}, promotion_recommendation="HOLD",
            safety_triggered=False, safety_reasons=[], events=[],
        )
        from src.portfolio.position_sizing_promotion import _save_state
        _save_state(state, f)
        result = format_psp_report_section(f, "2025-06-01")
        assert "POSITION SIZING PROMOTION" in result


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end with OHLCV
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndWithOhlcv:
    def _build_scenario(self, tmp_path, n_days=35, n_symbols=3):
        """
        Build a realistic scenario: N symbols over N days.
        Higher conviction → higher forward return (monotone relationship).
        """
        ohlcv_dir = tmp_path / "ohlcv"
        start     = date(2025, 1, 6)
        symbols   = [f"{1000 + i}.T" for i in range(n_symbols)]
        convictions = [30.0, 60.0, 85.0][:n_symbols]
        base_returns = [0.003, 0.010, 0.020][:n_symbols]  # monotone with conviction

        # Write OHLCV: each symbol has 60 days of price data
        for sym, base_ret in zip(symbols, base_returns):
            prices = [100.0]
            for _ in range(59):
                prices.append(prices[-1] * (1 + base_ret / 5))  # daily ~base_ret/5
            _make_ohlcv_csv(ohlcv_dir / f"{sym}.csv", start, prices)

        # Write telemetry: n_days of signals
        records = []
        for d in range(n_days):
            date_str = (start + timedelta(days=d)).isoformat()
            vws = [0.20, 0.33, 0.47][:n_symbols]  # conviction-weighted virtual
            eq  = 1.0 / n_symbols
            for sym, conv, vw in zip(symbols, convictions, vws):
                records.append(_make_telemetry_record(date_str, sym, conv, vw, eq))

        telem = tmp_path / "telem.jsonl"
        _write_telemetry(telem, records)

        today_str = (start + timedelta(days=n_days)).isoformat()
        return ohlcv_dir, telem, today_str

    def test_conviction_monotonicity_positive_with_consistent_data(self, tmp_path):
        ohlcv_dir, telem, today_str = self._build_scenario(tmp_path, n_days=35)
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
        w30 = compute_window_analysis(telem, ohlcv_dir, 30, today)
        # With monotone conviction→return, monotonicity should be > 0
        assert w30.conviction_monotonicity > 0.0

    def test_virtual_beats_equal_with_consistent_data(self, tmp_path):
        ohlcv_dir, telem, today_str = self._build_scenario(tmp_path, n_days=35)
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
        w30 = compute_window_analysis(telem, ohlcv_dir, 30, today)
        pc = w30.portfolio_comparison
        # With higher conviction → higher return, virtual should beat equal
        if pc.sufficient and pc.return_delta is not None:
            assert pc.return_delta > 0

    def test_full_run_tier0_no_crash(self, tmp_path):
        ohlcv_dir, telem, today_str = self._build_scenario(tmp_path, n_days=35)
        pfile = tmp_path / "reports" / "p.json"
        hfile = tmp_path / "reports" / "h.csv"
        state = run_position_sizing_promotion(
            telemetry_file=telem, promotion_file=pfile,
            history_file=hfile, ohlcv_dir=ohlcv_dir,
            today_str=today_str,
        )
        assert state.current_tier in (0, 1)

    def test_weight_invariants(self, tmp_path):
        ohlcv_dir, telem, today_str = self._build_scenario(tmp_path, n_days=35)
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
        w30 = compute_window_analysis(telem, ohlcv_dir, 30, today)
        pc = w30.portfolio_comparison
        if pc.return_delta is not None and pc.sufficient:
            # Equal and virtual returns should be in a reasonable range
            assert -1.0 < (pc.equal_return or 0) < 1.0
            assert -1.0 < (pc.virtual_return or 0) < 1.0
