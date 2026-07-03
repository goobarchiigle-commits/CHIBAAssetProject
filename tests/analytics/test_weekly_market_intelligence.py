"""
tests/analytics/test_weekly_market_intelligence.py

pytest coverage for weekly_market_intelligence module.

Covers:
  - corrupted trade inputs → WeeklyIntelligenceError
  - missing datasets → graceful degradation + insufficiency markers
  - deterministic outputs (same inputs → byte-identical markdown)
  - expectancy calculations
  - regime classification
  - archetype clustering + overextension score
  - atomic report generation (temp file → replace)
  - append-only archive integrity
  - capital efficiency scenario analysis
  - research priority engine (never "none" without metrics)
  - holding period bucket assignment
  - expectancy drift alert levels
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from src.analytics.weekly_market_intelligence import (
    # Engine
    WeeklyMarketIntelligenceEngine,
    WeeklyIntelligenceError,
    # Section analyzers
    analyze_opportunity_capture,
    analyze_archetype_expectancy,
    analyze_regime_transition,
    analyze_capital_efficiency,
    analyze_holding_period_decay,
    analyze_expectancy_drift,
    generate_research_priorities,
    # Report generation
    generate_markdown,
    generate_html,
    append_to_archive,
    load_weekly_archive,
    _atomic_write,
    # Helpers
    _linear_slope,
    _win_rate,
    _expectancy,
    _safe_mean,
    # Constants
    SCHEMA_VERSION,
    MAX_POSITIONS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

WEEK_END = date(2026, 5, 16)
WEEK_START = date(2026, 5, 10)


def _make_trade(symbol="1234.T", side="BUY", date_str="2026-05-12",
                price=2000.0, qty=100, atr20=50.0, pnl_pct=None,
                hold_days=None, entry_price=None, sector="テスト", reason="turtle_entry",
                entry_regime="bull") -> dict:
    t = {
        "date": date_str, "symbol": symbol, "sector": sector,
        "side": side, "qty": qty, "price": price, "amount": price * qty,
        "atr20": atr20, "reason": reason, "entry_regime": entry_regime,
    }
    if pnl_pct is not None:
        t["pnl_pct"] = pnl_pct
    if hold_days is not None:
        t["hold_days"] = hold_days
    if entry_price is not None:
        t["entry_price"] = entry_price
    return t


def _make_metric(date_str="2026-05-12", cash_ratio=0.7, exposure=0.3,
                 positions=1, candidate_count=1, signals_blocked_rsr=5,
                 signals_blocked_breakout=2, buy_candidates=3) -> dict:
    return {
        "date": date_str,
        "run_at": f"{date_str}T09:00:00+0900",
        "cash_ratio": cash_ratio,
        "exposure": exposure,
        "positions": positions,
        "candidate_count": candidate_count,
        "signals_blocked_rsr": signals_blocked_rsr,
        "signals_blocked_breakout": signals_blocked_breakout,
        "buy_candidates": buy_candidates,
        "universe_size": 42,
    }


def _make_p2(date_str="2026-05-12", equity=3_000_000.0, drawdown=0.0,
             breadth_50=0.55, breadth_75=0.30, signal_count=2) -> dict:
    return {
        "date": date_str,
        "equity": equity,
        "drawdown": drawdown,
        "positions": [],
        "breadth_50": breadth_50,
        "breadth_75": breadth_75,
        "signal_count": signal_count,
    }


def _date_range(start: date, end: date) -> List[str]:
    result = []
    cur = start
    while cur <= end:
        result.append(cur.isoformat())
        cur += timedelta(days=1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_linear_slope_ascending(self):
        s = _linear_slope([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(s - 1.0) < 1e-9

    def test_linear_slope_flat(self):
        assert _linear_slope([3.0, 3.0, 3.0]) == 0.0

    def test_linear_slope_insufficient(self):
        assert _linear_slope([1.0]) is None
        assert _linear_slope([]) is None

    def test_win_rate_empty(self):
        assert _win_rate([]) is None

    def test_win_rate_all_wins(self):
        assert _win_rate([0.01, 0.02, 0.03]) == pytest.approx(1.0)

    def test_win_rate_mixed(self):
        assert _win_rate([0.01, -0.01, 0.02, -0.02]) == pytest.approx(0.5)

    def test_expectancy_single(self):
        assert _expectancy([0.05]) is None

    def test_expectancy_multiple(self):
        e = _expectancy([0.04, -0.02, 0.06])
        assert e == pytest.approx(sum([0.04, -0.02, 0.06]) / 3)

    def test_safe_mean_empty(self):
        assert _safe_mean([]) is None

    def test_safe_mean_values(self):
        assert _safe_mean([1.0, 3.0]) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Opportunity Capture
# ─────────────────────────────────────────────────────────────────────────────

class TestOpportunityCapture:
    def _metrics_30d(self) -> List[dict]:
        dates = _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        return [
            _make_metric(d, candidate_count=1, signals_blocked_rsr=5,
                         signals_blocked_breakout=2, buy_candidates=3)
            for d in dates
        ]

    def test_capture_ratio_computed(self):
        result = analyze_opportunity_capture(
            self._metrics_30d(), [], WEEK_START, WEEK_END
        )
        assert result.available
        assert result.opportunity_capture_ratio is not None
        assert 0.0 <= result.opportunity_capture_ratio <= 1.0

    def test_no_metrics_unavailable(self):
        result = analyze_opportunity_capture([], [], WEEK_START, WEEK_END)
        assert not result.available
        assert result.opportunity_capture_ratio is None

    def test_missing_skipped_opportunities_marker(self):
        result = analyze_opportunity_capture(
            self._metrics_30d(), [], WEEK_START, WEEK_END
        )
        assert any("skipped_opportunities" in m for m in result.insufficiency_markers)

    def test_skip_reason_distribution_keys(self):
        result = analyze_opportunity_capture(
            self._metrics_30d(), [], WEEK_START, WEEK_END
        )
        assert "rsr_filter" in result.skip_reason_distribution
        assert "breakout_filter" in result.skip_reason_distribution

    def test_zero_blocked_signals(self):
        metrics = [_make_metric(d, signals_blocked_rsr=0, signals_blocked_breakout=0,
                                buy_candidates=0, candidate_count=0)
                   for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)]
        result = analyze_opportunity_capture(metrics, [], WEEK_START, WEEK_END)
        assert result.opportunity_capture_ratio is None or result.opportunity_capture_ratio == 0.0

    def test_false_negative_rate_bounded(self):
        result = analyze_opportunity_capture(
            self._metrics_30d(), [], WEEK_START, WEEK_END
        )
        if result.false_negative_rate is not None:
            assert 0.0 <= result.false_negative_rate <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Archetype Expectancy
# ─────────────────────────────────────────────────────────────────────────────

class TestArchetypeExpectancy:
    def _buys(self, n=10) -> List[dict]:
        return [
            _make_trade(symbol=f"{1000+i}.T", side="BUY",
                        date_str=(WEEK_END - timedelta(days=30+i)).isoformat(),
                        price=2000.0, atr20=50.0, reason="turtle_entry")
            for i in range(n)
        ]

    def _sells(self, n=5) -> List[dict]:
        return [
            _make_trade(symbol=f"{1000+i}.T", side="SELL",
                        date_str=(WEEK_END - timedelta(days=i)).isoformat(),
                        price=2100.0, pnl_pct=0.05, hold_days=10)
            for i in range(n)
        ]

    def test_no_buys_unavailable(self):
        result = analyze_archetype_expectancy([], WEEK_END)
        assert not result.available
        assert "No BUY trades" in result.insufficiency_markers[0]

    def test_buys_without_sells(self):
        result = analyze_archetype_expectancy(self._buys(), WEEK_END)
        assert result.available
        assert result.total_buys == 10
        assert result.total_sells == 0
        assert result.win_rate is None
        assert any("Only 0 completed sell" in m for m in result.insufficiency_markers)

    def test_atr_pct_computed(self):
        buys = [_make_trade(price=2000.0, atr20=50.0)]  # atr_pct = 50/2000 = 0.025
        result = analyze_archetype_expectancy(buys + self._sells(5), WEEK_END)
        assert result.avg_atr_pct == pytest.approx(0.025, abs=0.001)

    def test_overextension_alert_trigger(self):
        buys = [_make_trade(price=1000.0, atr20=30.0)] * 5  # atr_pct=0.03 > 0.025
        result = analyze_archetype_expectancy(buys + self._sells(5), WEEK_END)
        assert result.overextension_alert

    def test_overextension_no_alert_when_low_atr(self):
        buys = [_make_trade(price=5000.0, atr20=50.0)] * 5  # atr_pct=0.01 < 0.025
        result = analyze_archetype_expectancy(buys + self._sells(5), WEEK_END)
        assert not result.overextension_alert

    def test_sector_distribution_populated(self):
        buys = [
            _make_trade(sector="技術", price=2000.0, atr20=40.0),
            _make_trade(sector="金融", price=2000.0, atr20=40.0),
            _make_trade(sector="技術", price=2000.0, atr20=40.0),
        ]
        result = analyze_archetype_expectancy(buys, WEEK_END)
        assert result.sector_distribution.get("技術") == 2
        assert result.sector_distribution.get("金融") == 1

    def test_clusters_non_empty(self):
        trades = self._buys(6) + self._sells(5)
        result = analyze_archetype_expectancy(trades, WEEK_END)
        assert len(result.clusters) >= 1
        for cl in result.clusters:
            assert cl.overextension_score >= 0.0

    def test_win_rate_correct_with_sells(self):
        buys = self._buys(3)
        sells = [
            _make_trade(side="SELL", pnl_pct=0.05, hold_days=5,
                        date_str=(WEEK_END - timedelta(days=1)).isoformat()),
            _make_trade(side="SELL", pnl_pct=-0.03, hold_days=8,
                        date_str=(WEEK_END - timedelta(days=2)).isoformat()),
            _make_trade(side="SELL", pnl_pct=0.04, hold_days=10,
                        date_str=(WEEK_END - timedelta(days=3)).isoformat()),
            _make_trade(side="SELL", pnl_pct=0.02, hold_days=6,
                        date_str=(WEEK_END - timedelta(days=4)).isoformat()),
            _make_trade(side="SELL", pnl_pct=-0.01, hold_days=4,
                        date_str=(WEEK_END - timedelta(days=5)).isoformat()),
        ]
        result = analyze_archetype_expectancy(buys + sells, WEEK_END)
        assert result.win_rate == pytest.approx(3.0 / 5.0)

    def test_deterministic(self):
        trades = self._buys(5) + self._sells(3)
        r1 = analyze_archetype_expectancy(trades, WEEK_END)
        r2 = analyze_archetype_expectancy(trades, WEEK_END)
        assert r1.avg_atr_pct == r2.avg_atr_pct
        assert r1.win_rate == r2.win_rate
        assert r1.total_buys == r2.total_buys


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Regime Transition
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeTransition:
    def _p2_30d(self, breadth=0.55, dd=0.0, sc=2) -> List[dict]:
        return [
            _make_p2(d, breadth_50=breadth, drawdown=dd, signal_count=sc)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]

    def test_no_data_unavailable(self):
        result = analyze_regime_transition([], [], WEEK_START, WEEK_END)
        assert not result.available
        assert result.regime_state == "unknown"

    def test_trend_persistent_high_breadth(self):
        result = analyze_regime_transition([], self._p2_30d(breadth=0.65), WEEK_START, WEEK_END)
        assert result.available
        assert result.regime_state == "trend_persistent"
        assert result.regime_confidence > 0.5

    def test_momentum_exhaustion_signal_drought(self):
        p2 = self._p2_30d(breadth=0.20, sc=0)
        result = analyze_regime_transition([], p2, WEEK_START, WEEK_END)
        assert result.signal_drought_consecutive >= 5
        assert result.regime_state in ("momentum_exhaustion", "mean_reverting", "high_volatility")

    def test_high_volatility_on_drawdown(self):
        p2 = self._p2_30d(breadth=0.20, dd=-0.12)
        result = analyze_regime_transition([], p2, WEEK_START, WEEK_END)
        assert result.regime_state == "high_volatility"

    def test_instability_score_bounded(self):
        result = analyze_regime_transition([], self._p2_30d(), WEEK_START, WEEK_END)
        assert 0.0 <= result.instability_score <= 1.0

    def test_weekly_history_populated(self):
        result = analyze_regime_transition([], self._p2_30d(), WEEK_START, WEEK_END)
        assert len(result.weekly_regime_history) >= 1
        for wh in result.weekly_regime_history:
            assert "week_end" in wh
            assert "avg_breadth_50" in wh

    def test_breadth_trend_direction(self):
        # Declining breadth over 30d → negative slope
        p2 = [
            _make_p2(d, breadth_50=0.8 - i * 0.02)
            for i, d in enumerate(_date_range(WEEK_END - timedelta(days=29), WEEK_END))
        ]
        result = analyze_regime_transition([], p2, WEEK_START, WEEK_END)
        assert result.breadth_50_trend_30d is not None
        assert result.breadth_50_trend_30d < 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Capital Efficiency
# ─────────────────────────────────────────────────────────────────────────────

class TestCapitalEfficiency:
    def _metrics_30d(self, cash=0.75, exp=0.25, pos=1, rsr_blocked=10) -> List[dict]:
        return [
            _make_metric(d, cash_ratio=cash, exposure=exp,
                         positions=pos, signals_blocked_rsr=rsr_blocked)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]

    def test_no_metrics_unavailable(self):
        result = analyze_capital_efficiency([], WEEK_END)
        assert not result.available

    def test_avg_cash_ratio(self):
        result = analyze_capital_efficiency(self._metrics_30d(cash=0.80), WEEK_END)
        assert result.avg_cash_ratio_30d == pytest.approx(0.80, abs=0.01)

    def test_slot_utilization_bounded(self):
        result = analyze_capital_efficiency(self._metrics_30d(pos=1), WEEK_END)
        assert 0.0 <= result.slot_utilization_30d <= 1.0
        assert result.slot_utilization_30d == pytest.approx(1 / MAX_POSITIONS, abs=0.01)

    def test_scenarios_positive_when_blocked(self):
        result = analyze_capital_efficiency(self._metrics_30d(rsr_blocked=20), WEEK_END)
        assert result.scenario_50pct_capture_gain is not None
        assert result.scenario_100pct_capture_gain is not None
        assert result.scenario_200pct_capture_gain is not None
        assert result.scenario_50pct_capture_gain < result.scenario_100pct_capture_gain
        assert result.scenario_100pct_capture_gain < result.scenario_200pct_capture_gain

    def test_scenarios_zero_when_no_blocked(self):
        metrics = [
            _make_metric(d, signals_blocked_rsr=0, signals_blocked_breakout=0)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]
        result = analyze_capital_efficiency(metrics, WEEK_END)
        assert result.scenario_50pct_capture_gain == pytest.approx(0.0, abs=1e-9)

    def test_fragmentation_score_bounded(self):
        result = analyze_capital_efficiency(self._metrics_30d(), WEEK_END)
        assert 0.0 <= result.capital_fragmentation_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Holding Period Decay
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldingPeriodDecay:
    def _sells(self) -> List[dict]:
        return [
            _make_trade(side="SELL", hold_days=hd, pnl_pct=pnl,
                        date_str=(WEEK_END - timedelta(days=i)).isoformat())
            for i, (hd, pnl) in enumerate([
                (2, 0.03), (5, 0.05), (12, -0.02), (25, 0.04), (50, 0.01)
            ])
        ]

    def test_no_exits_unavailable(self):
        result = analyze_holding_period_decay([], [], WEEK_END)
        assert not result.available
        assert result.completed_trades == 0
        assert len(result.insufficiency_markers) > 0

    def test_bucket_assignment_correct(self):
        sells = [
            _make_trade(side="SELL", hold_days=1, pnl_pct=0.01,
                        date_str=(WEEK_END - timedelta(days=1)).isoformat()),
            _make_trade(side="SELL", hold_days=20, pnl_pct=0.02,
                        date_str=(WEEK_END - timedelta(days=2)).isoformat()),
        ] + self._sells()
        result = analyze_holding_period_decay(sells, [], WEEK_END)
        bucket_labels = {b.label for b in result.buckets}
        assert "1d" in bucket_labels
        assert "20d" in bucket_labels

    def test_insufficient_data_marker(self):
        sells = [_make_trade(side="SELL", hold_days=5, pnl_pct=0.02,
                             date_str=(WEEK_END - timedelta(days=1)).isoformat())]
        result = analyze_holding_period_decay(sells, [], WEEK_END)
        assert any("need ≥5" in m for m in result.insufficiency_markers)

    def test_optimal_bucket_identified(self):
        result = analyze_holding_period_decay(self._sells() * 2, [], WEEK_END)
        if result.optimal_exit_bucket is not None:
            assert result.optimal_exit_bucket in [
                "1d", "3d", "5d", "10d", "20d", "40d", "60d+"
            ]

    def test_avg_hold_days_correct(self):
        sells = [
            _make_trade(side="SELL", hold_days=10, pnl_pct=0.01,
                        date_str=(WEEK_END - timedelta(days=i)).isoformat())
            for i in range(5)
        ]
        result = analyze_holding_period_decay(sells, [], WEEK_END)
        assert result.avg_hold_days == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Expectancy Drift
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectancyDrift:
    def _p2_30d(self, eq_start=3_000_000, dd=-0.01, sc=2) -> List[dict]:
        return [
            _make_p2(d, equity=eq_start - i * 10000, drawdown=dd, signal_count=sc)
            for i, d in enumerate(_date_range(WEEK_END - timedelta(days=29), WEEK_END))
        ]

    def test_no_data_unavailable(self):
        result = analyze_expectancy_drift([], [], WEEK_END)
        assert not result.available

    def test_available_with_p2(self):
        result = analyze_expectancy_drift([], self._p2_30d(), WEEK_END)
        assert result.available

    def test_alert_low_when_stable(self):
        p2 = [
            _make_p2(d, equity=3_000_000, drawdown=0.0, signal_count=3)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]
        result = analyze_expectancy_drift([], p2, WEEK_END)
        assert result.instability_alert_level in ("LOW", "MEDIUM")

    def test_alert_high_on_accelerating_drawdown(self):
        # Drawdown worsening from -0.01 to -0.09 over 30d
        p2 = [
            _make_p2(d, equity=3_000_000, drawdown=-0.01 - i * 0.003, signal_count=1)
            for i, d in enumerate(_date_range(WEEK_END - timedelta(days=29), WEEK_END))
        ]
        result = analyze_expectancy_drift([], p2, WEEK_END)
        assert result.instability_alert_level in ("MEDIUM", "HIGH")
        assert result.drawdown_acceleration is not None
        assert result.drawdown_acceleration < 0

    def test_edge_stability_bounded(self):
        result = analyze_expectancy_drift([], self._p2_30d(), WEEK_END)
        if result.edge_stability_score is not None:
            assert 0.0 <= result.edge_stability_score <= 1.0

    def test_signal_drought_warning(self):
        p2 = [
            _make_p2(d, signal_count=0)
            for d in _date_range(WEEK_END - timedelta(days=6), WEEK_END)
        ] + [
            _make_p2(d, signal_count=3)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END - timedelta(days=7))
        ]
        result = analyze_expectancy_drift([], p2, WEEK_END)
        assert any("Signal count declining" in w for w in result.warnings) or \
               result.signal_count_trend_7d is not None


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Research Priority Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchPriorityEngine:
    def _full_inputs(self):
        from src.analytics.weekly_market_intelligence import (
            OpportunityCapture, ArchetypeExpectancy, RegimeTransition,
            CapitalEfficiency, HoldingPeriodDecay, ExpectancyDrift,
        )
        opp = OpportunityCapture(
            available=True, opportunity_capture_ratio=0.15, false_negative_rate=0.85,
            missed_alpha_score=0.03, skip_reason_distribution={"rsr_filter": 100, "breakout_filter": 20},
            weekly_rsr_blocked=100, weekly_breakout_blocked=20,
            weekly_candidates_executed=10, weekly_candidates_signaled=5,
        )
        arch = ArchetypeExpectancy(
            available=True, total_buys=15, total_sells=2, matched_pairs=2,
            avg_pnl_pct=0.02, win_rate=0.5, realized_expectancy=0.02,
            avg_hold_days=12.0, avg_atr_pct=0.03, overextension_alert=True,
            clusters=[], sector_distribution={}, regime_distribution={"bull": 15},
            insufficiency_markers=["Only 2 completed sell(s)"],
        )
        regime = RegimeTransition(
            available=True, current_regime="bull", regime_confidence=0.7,
            breadth_50_latest=0.55, breadth_75_latest=0.30,
            breadth_50_trend_7d=-0.01, breadth_50_trend_30d=-0.005,
            equity_drawdown_current=-0.05, equity_drawdown_max_30d=-0.08,
            signal_drought_consecutive=0, signal_count_trend_7d=-0.3,
            failed_breakout_proxy=0.1, regime_state="trend_persistent",
            instability_score=0.25, weekly_regime_history=[],
        )
        cap = CapitalEfficiency(
            available=True, avg_cash_ratio_7d=0.80, avg_cash_ratio_30d=0.82,
            avg_exposure_7d=0.20, avg_exposure_30d=0.18,
            slot_utilization_7d=0.33, slot_utilization_30d=0.33,
            idle_cash_ratio=0.82, peak_positions_30d=1,
            total_rsr_blocked_30d=100, total_breakout_blocked_30d=20,
            capital_fragmentation_score=0.4,
            scenario_50pct_capture_gain=0.06,
            scenario_100pct_capture_gain=0.105,
            scenario_200pct_capture_gain=0.165,
        )
        hold = HoldingPeriodDecay(
            available=True, completed_trades=2, avg_hold_days=20.0,
            max_hold_days=30, min_hold_days=10, buckets=[], optimal_exit_bucket="20d",
            insufficiency_markers=["Only 2 completed trades"],
        )
        drift = ExpectancyDrift(
            available=True, equity_trend_7d=-5000.0, equity_trend_30d=-2000.0,
            drawdown_acceleration=-0.02, signal_count_trend_7d=-0.5,
            edge_stability_score=0.6, alpha_decay_score=0.3,
            instability_alert_level="MEDIUM",
            rolling_win_rate_30d=0.5, warnings=["Signal count declining"],
        )
        return opp, arch, regime, cap, hold, drift

    def test_priorities_never_empty(self):
        opp, arch, regime, cap, hold, drift = self._full_inputs()
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        assert len(result) >= 1

    def test_priorities_ranked_sequentially(self):
        opp, arch, regime, cap, hold, drift = self._full_inputs()
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        for i, pr in enumerate(result, start=1):
            assert pr.rank == i

    def test_overextension_generates_priority(self):
        opp, arch, regime, cap, hold, drift = self._full_inputs()
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        cats = {pr.category for pr in result}
        assert "entry_archetype" in cats

    def test_capital_constraint_generates_priority(self):
        opp, arch, regime, cap, hold, drift = self._full_inputs()
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        cats = {pr.category for pr in result}
        assert "capital_efficiency" in cats

    def test_signal_drought_generates_priority(self):
        opp, arch, regime, cap, hold, drift = self._full_inputs()
        regime.signal_drought_consecutive = 7
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        titles = [pr.title for pr in result]
        assert any("枯渇" in t for t in titles)

    def test_all_priorities_have_required_fields(self):
        opp, arch, regime, cap, hold, drift = self._full_inputs()
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        for pr in result:
            assert pr.category
            assert pr.title
            assert pr.confidence in ("HIGH", "MEDIUM", "LOW")
            assert pr.evidence
            assert pr.statistical_justification
            assert pr.suggested_direction

    def test_fallback_priority_when_all_clear(self):
        from src.analytics.weekly_market_intelligence import (
            OpportunityCapture, ArchetypeExpectancy, RegimeTransition,
            CapitalEfficiency, HoldingPeriodDecay, ExpectancyDrift,
        )
        opp = OpportunityCapture(
            available=True, opportunity_capture_ratio=0.9, false_negative_rate=0.1,
            missed_alpha_score=0.0, skip_reason_distribution={},
            weekly_rsr_blocked=0, weekly_breakout_blocked=0,
            weekly_candidates_executed=3, weekly_candidates_signaled=3,
        )
        arch = ArchetypeExpectancy(
            available=True, total_buys=5, total_sells=5, matched_pairs=5,
            avg_pnl_pct=0.03, win_rate=0.8, realized_expectancy=0.03,
            avg_hold_days=5.0, avg_atr_pct=0.015, overextension_alert=False,
            clusters=[], sector_distribution={}, regime_distribution={},
        )
        regime = RegimeTransition(
            available=True, current_regime="bull", regime_confidence=0.85,
            breadth_50_latest=0.65, breadth_75_latest=0.40,
            breadth_50_trend_7d=0.01, breadth_50_trend_30d=0.005,
            equity_drawdown_current=-0.01, equity_drawdown_max_30d=-0.02,
            signal_drought_consecutive=0, signal_count_trend_7d=0.1,
            failed_breakout_proxy=0.0, regime_state="trend_persistent",
            instability_score=0.1, weekly_regime_history=[],
        )
        cap = CapitalEfficiency(
            available=True, avg_cash_ratio_7d=0.50, avg_cash_ratio_30d=0.55,
            avg_exposure_7d=0.45, avg_exposure_30d=0.40,
            slot_utilization_7d=0.67, slot_utilization_30d=0.60,
            idle_cash_ratio=0.55, peak_positions_30d=2,
            total_rsr_blocked_30d=2, total_breakout_blocked_30d=1,
            capital_fragmentation_score=0.05,
            scenario_50pct_capture_gain=0.001,
            scenario_100pct_capture_gain=0.002,
            scenario_200pct_capture_gain=0.003,
        )
        hold = HoldingPeriodDecay(
            available=True, completed_trades=5, avg_hold_days=8.0,
            max_hold_days=15, min_hold_days=3, buckets=[], optimal_exit_bucket="5d",
        )
        drift = ExpectancyDrift(
            available=True, equity_trend_7d=5000.0, equity_trend_30d=2000.0,
            drawdown_acceleration=0.001, signal_count_trend_7d=0.1,
            edge_stability_score=0.85, alpha_decay_score=0.02,
            instability_alert_level="LOW",
            rolling_win_rate_30d=0.8, warnings=[],
        )
        result = generate_research_priorities(opp, arch, regime, cap, hold, drift, [], WEEK_END)
        # Must produce at least one priority (never empty)
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

class TestReportGeneration:
    def _minimal_report(self) -> "WeeklyIntelligenceReport":  # noqa: F821
        from src.analytics.weekly_market_intelligence import (
            WeeklyIntelligenceReport, OpportunityCapture, ArchetypeExpectancy,
            RegimeTransition, CapitalEfficiency, HoldingPeriodDecay, ExpectancyDrift,
            ResearchPriority,
        )
        return WeeklyIntelligenceReport(
            run_id="test_run",
            generated_at="2026-05-16T15:30:00+09:00",
            report_date="2026-05-16",
            week_start="2026-05-10",
            week_end="2026-05-16",
            schema_version=SCHEMA_VERSION,
            opportunity_capture=OpportunityCapture(
                available=False, opportunity_capture_ratio=None, false_negative_rate=None,
                missed_alpha_score=None, skip_reason_distribution={},
                weekly_rsr_blocked=0, weekly_breakout_blocked=0,
                weekly_candidates_executed=0, weekly_candidates_signaled=0,
                insufficiency_markers=["no data"],
            ),
            archetype_expectancy=ArchetypeExpectancy(
                available=False, total_buys=0, total_sells=0, matched_pairs=0,
                avg_pnl_pct=None, win_rate=None, realized_expectancy=None,
                avg_hold_days=None, avg_atr_pct=0.0, overextension_alert=False,
                clusters=[], sector_distribution={}, regime_distribution={},
                insufficiency_markers=["no data"],
            ),
            regime_transition=RegimeTransition(
                available=False, current_regime="unknown", regime_confidence=0.0,
                breadth_50_latest=None, breadth_75_latest=None,
                breadth_50_trend_7d=None, breadth_50_trend_30d=None,
                equity_drawdown_current=None, equity_drawdown_max_30d=None,
                signal_drought_consecutive=0, signal_count_trend_7d=None,
                failed_breakout_proxy=None, regime_state="unknown",
                instability_score=0.0, weekly_regime_history=[],
            ),
            capital_efficiency=CapitalEfficiency(
                available=False, avg_cash_ratio_7d=0.0, avg_cash_ratio_30d=0.0,
                avg_exposure_7d=0.0, avg_exposure_30d=0.0,
                slot_utilization_7d=0.0, slot_utilization_30d=0.0,
                idle_cash_ratio=0.0, peak_positions_30d=0,
                total_rsr_blocked_30d=0, total_breakout_blocked_30d=0,
                capital_fragmentation_score=0.0,
                scenario_50pct_capture_gain=None, scenario_100pct_capture_gain=None,
                scenario_200pct_capture_gain=None,
            ),
            holding_period=HoldingPeriodDecay(
                available=False, completed_trades=0, avg_hold_days=None,
                max_hold_days=None, min_hold_days=None, buckets=[], optimal_exit_bucket=None,
                insufficiency_markers=["no data"],
            ),
            expectancy_drift=ExpectancyDrift(
                available=False, equity_trend_7d=None, equity_trend_30d=None,
                drawdown_acceleration=None, signal_count_trend_7d=None,
                edge_stability_score=None, alpha_decay_score=None,
                instability_alert_level="LOW",
                rolling_win_rate_30d=None, warnings=[],
            ),
            research_priorities=[
                ResearchPriority(
                    rank=1, category="data_collection", title="データ収集フェーズ",
                    confidence="LOW", evidence="no data",
                    affected_symbols=[], statistical_justification="none",
                    suggested_direction="continue",
                )
            ],
            data_availability={"trades.jsonl": False},
            data_quality_warnings=["test warning"],
            chart_paths=[],
        )

    def test_markdown_contains_sections(self):
        report = self._minimal_report()
        md = generate_markdown(report)
        assert "Section 1" in md
        assert "Section 7" in md
        assert "週次市場インテリジェンス" in md

    def test_markdown_deterministic(self):
        report = self._minimal_report()
        md1 = generate_markdown(report)
        md2 = generate_markdown(report)
        assert md1 == md2

    def test_html_contains_title(self):
        report = self._minimal_report()
        md = generate_markdown(report)
        html = generate_html(md, "2026-05-16")
        assert "週次市場インテリジェンス" in html
        assert "<!DOCTYPE html>" in html

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "test.md"
            _atomic_write(path, "hello world")
            assert path.exists()
            assert path.read_text(encoding="utf-8") == "hello world"

    def test_atomic_write_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.md"
            _atomic_write(path, "first")
            _atomic_write(path, "second")
            assert path.read_text(encoding="utf-8") == "second"


# ─────────────────────────────────────────────────────────────────────────────
# Append-only archive integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestArchiveIntegrity:
    def _minimal_report(self):
        from tests.analytics.test_weekly_market_intelligence import TestReportGeneration
        return TestReportGeneration()._minimal_report()

    def test_append_only_grows(self):
        with tempfile.TemporaryDirectory() as td:
            arc = Path(td) / "archive.jsonl"
            r = self._minimal_report()
            append_to_archive(r, arc)
            size1 = arc.stat().st_size
            r.run_id = "second_run"
            append_to_archive(r, arc)
            size2 = arc.stat().st_size
            assert size2 > size1

    def test_archive_records_parseable(self):
        with tempfile.TemporaryDirectory() as td:
            arc = Path(td) / "archive.jsonl"
            r = self._minimal_report()
            append_to_archive(r, arc)
            rows = load_weekly_archive(arc)
            assert len(rows) == 1
            assert rows[0]["run_id"] == r.run_id
            assert rows[0]["schema_version"] == SCHEMA_VERSION

    def test_archive_multiple_weeks(self):
        with tempfile.TemporaryDirectory() as td:
            arc = Path(td) / "archive.jsonl"
            for i in range(3):
                r = self._minimal_report()
                r.run_id = f"run_{i}"
                r.report_date = f"2026-0{5+i}-16"
                append_to_archive(r, arc)
            rows = load_weekly_archive(arc)
            assert len(rows) == 3
            assert {row["run_id"] for row in rows} == {"run_0", "run_1", "run_2"}

    def test_archive_json_sorted_keys(self):
        with tempfile.TemporaryDirectory() as td:
            arc = Path(td) / "archive.jsonl"
            r = self._minimal_report()
            append_to_archive(r, arc)
            raw_line = arc.read_text(encoding="utf-8").strip()
            parsed = json.loads(raw_line)
            # Keys should be sorted (sort_keys=True in _report_to_dict)
            keys = list(parsed.keys())
            assert keys == sorted(keys)


# ─────────────────────────────────────────────────────────────────────────────
# Corrupted inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestCorruptedInputs:
    def test_corrupted_trade_price_raises(self):
        from src.analytics.weekly_market_intelligence import _load_trades
        with tempfile.TemporaryDirectory() as td:
            trades_file = Path(td) / "logs" / "trades.jsonl"
            trades_file.parent.mkdir(parents=True)
            trades_file.write_text(
                json.dumps({"date": "2026-05-12", "symbol": "1234.T",
                            "side": "BUY", "qty": 100, "price": "NOT_A_NUMBER",
                            "amount": 0}) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(WeeklyIntelligenceError):
                _load_trades(Path(td))

    def test_corrupted_jsonl_line_skipped(self):
        from src.analytics.weekly_market_intelligence import _load_jsonl
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.jsonl"
            p.write_text(
                '{"key": "value1"}\n'
                'CORRUPT LINE NOT JSON\n'
                '{"key": "value2"}\n',
                encoding="utf-8",
            )
            rows, ok = _load_jsonl(p)
            assert ok
            assert len(rows) == 2

    def test_missing_file_returns_empty(self):
        from src.analytics.weekly_market_intelligence import _load_jsonl
        rows, ok = _load_jsonl(Path("nonexistent_path_xyz/file.jsonl"))
        assert not ok
        assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# Full engine integration (dry-run, no file I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineIntegration:
    def test_dry_run_no_files_written(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # No data files — all sections degrade gracefully
            engine = WeeklyMarketIntelligenceEngine(
                base_dir=base,
                report_dir=base / "reports",
                archive_file=base / "archive.jsonl",
            )
            report = engine.run(
                reference_date=WEEK_END,
                send_email=False,
                dry_run=True,
            )
            assert report is not None
            assert report.schema_version == SCHEMA_VERSION
            assert len(report.research_priorities) >= 1
            # No files written
            assert not (base / "archive.jsonl").exists()

    def test_engine_with_real_data_produces_report(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            logs = base / "logs" / "diagnostics"
            logs.mkdir(parents=True)
            p2_dir = base / "logs"

            # Write sample data
            trades_file = base / "logs" / "trades.jsonl"
            metrics_file = logs / "metrics.jsonl"
            p2_file = p2_dir / "phase2_live_metrics.jsonl"

            trades = [
                _make_trade(date_str=(WEEK_END - timedelta(days=i)).isoformat())
                for i in range(10)
            ] + [
                _make_trade(side="SELL", pnl_pct=0.03, hold_days=10,
                            date_str=(WEEK_END - timedelta(days=1)).isoformat()),
                _make_trade(side="SELL", pnl_pct=-0.02, hold_days=20,
                            date_str=(WEEK_END - timedelta(days=2)).isoformat()),
            ]
            trades_file.write_text(
                "\n".join(json.dumps(t, ensure_ascii=False) for t in trades),
                encoding="utf-8",
            )

            metrics_rows = [
                _make_metric((WEEK_END - timedelta(days=i)).isoformat())
                for i in range(30)
            ]
            metrics_file.write_text(
                "\n".join(json.dumps(m, ensure_ascii=False) for m in metrics_rows),
                encoding="utf-8",
            )

            p2_rows = [
                _make_p2((WEEK_END - timedelta(days=i)).isoformat())
                for i in range(30)
            ]
            p2_file.write_text(
                "\n".join(json.dumps(p, ensure_ascii=False) for p in p2_rows),
                encoding="utf-8",
            )

            report_dir = base / "reports"
            engine = WeeklyMarketIntelligenceEngine(
                base_dir=base,
                report_dir=report_dir,
                archive_file=report_dir / "archive.jsonl",
            )
            report = engine.run(
                reference_date=WEEK_END,
                send_email=False,
                dry_run=False,
            )

            assert report is not None
            assert report.report_md_path is not None
            assert Path(report.report_md_path).exists()
            assert report.report_json_path is not None
            assert Path(report.report_json_path).exists()
            assert (report_dir / "archive.jsonl").exists()

            # Archive has exactly one entry
            rows = load_weekly_archive(report_dir / "archive.jsonl")
            assert len(rows) == 1
            assert rows[0]["run_id"] == report.run_id

    def test_engine_fail_closed_on_corrupt_trades(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            trades_file = base / "logs" / "trades.jsonl"
            trades_file.parent.mkdir(parents=True)
            trades_file.write_text(
                json.dumps({"date": "2026-05-12", "symbol": "X", "side": "BUY",
                            "qty": 1, "price": "BROKEN", "amount": 0}) + "\n",
                encoding="utf-8",
            )
            engine = WeeklyMarketIntelligenceEngine(
                base_dir=base, report_dir=base / "reports",
            )
            with pytest.raises(WeeklyIntelligenceError):
                engine.run(reference_date=WEEK_END, send_email=False, dry_run=True)

    def test_week_resolution_to_friday(self):
        engine = WeeklyMarketIntelligenceEngine(base_dir=Path("C:/ai-trading"))
        # 2026-05-19 is Tuesday → nearest Friday before = 2026-05-15 (nope, 2026-05-16 is Sat, 2026-05-15 is Fri)
        # Actually: (Tuesday - 4) % 7 = (1 - 4 + 7) % 7 = 4 days back = 2026-05-15 (Mon? no...)
        # Let me verify: 2026-05-19 is Tuesday (weekday=1), days_since_friday=(1-4)%7=4, so 2026-05-15
        ws, we = engine._resolve_week(date(2026, 5, 19))
        assert we.weekday() == 4  # Friday
        assert we == date(2026, 5, 15)
        assert ws == date(2026, 5, 9)
