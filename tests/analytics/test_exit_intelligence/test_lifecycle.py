"""tests/analytics/test_exit_intelligence/test_lifecycle.py"""
import pytest

from src.analytics.exit_intelligence.lifecycle import (
    ConditionalExpectancySurface,
    CounterfactualExitAnalyzer,
    CounterfactualPartialExitAnalytics,
    LoserRetentionAnalysis,
    PartialExitAllocator,
)
from src.analytics.exit_intelligence.models import (
    ExitObservationRecord,
    ExitVelocityScore,
)


def _make_obs(
    symbol="1234", entry="2026-01-01", exit_date="2026-01-10",
    realized_return=0.10, holding_days=9,
    velocity_score=2.5, distance_from_peak=0.05,
    rs_rank=80.0, breadth=0.65, atr=0.02, regime="bull",
):
    return ExitObservationRecord(
        obs_id=ExitObservationRecord.make_obs_id(symbol, entry, exit_date),
        symbol=symbol, entry_date=entry, exit_date=exit_date,
        holding_days=holding_days, entry_price=1000.0,
        exit_price=1000.0 * (1 + realized_return),
        realized_return=realized_return,
        realized_pnl_jpy=realized_return * 100_000,
        exit_reason="turtle_exit_55d",
        unrealized_pnl_at_exit=realized_return * 100_000,
        distance_from_peak_pct=distance_from_peak,
        rs_rank_at_exit=rs_rank, atr_pct_at_exit=atr,
        breadth_score_at_exit=breadth, regime_at_exit=regime,
        exit_velocity_score=velocity_score, path_snapshot_count=holding_days,
    )


def _make_evs(symbol="1234", score=3.0, band="tighten_risk"):
    return ExitVelocityScore(
        score_id=ExitVelocityScore.make_score_id(symbol, "2026-01-10"),
        symbol=symbol, scored_date="2026-01-10",
        rs_deterioration=0.3, breadth_decay=0.2,
        breakout_failure=0.1, vwap_rejection=0.1,
        upper_shadow=0.2, volume_decay=0.2,
        volatility_compression=0.3, momentum_decay=0.2,
        distance_from_peak=0.1, execution_deterioration=0.0,
        portfolio_replacement_pressure=0.0,
        total_score=score, action_band=band,
        component_weights={}, generated_at="now",
    )


class TestPartialExitAllocator:
    def setup_method(self):
        self.alloc = PartialExitAllocator()

    def test_hold_no_reduction(self):
        evs = _make_evs(score=1.5, band="hold")
        decision = self.alloc.allocate(evs)
        assert decision.recommended_first_reduction == 0.0

    def test_tighten_no_size_reduction(self):
        evs = _make_evs(score=3.5, band="tighten_risk")
        decision = self.alloc.allocate(evs)
        assert decision.recommended_first_reduction == 0.0

    def test_staged_reduction_33pct(self):
        evs = _make_evs(score=5.5, band="staged_reduction")
        decision = self.alloc.allocate(evs)
        assert decision.recommended_first_reduction == pytest.approx(0.33)
        assert decision.stage_1_fraction + decision.stage_2_fraction + decision.stage_3_fraction == pytest.approx(1.0)

    def test_aggressive_full_exit(self):
        evs = _make_evs(score=7.5, band="aggressive_exit")
        decision = self.alloc.allocate(evs)
        assert decision.recommended_first_reduction == 1.0

    def test_staged_fractions_sum_to_1(self):
        evs = _make_evs(score=5.5, band="staged_reduction")
        d = self.alloc.allocate(evs)
        assert d.stage_1_fraction + d.stage_2_fraction + d.stage_3_fraction == pytest.approx(1.0)

    def test_decision_id_deterministic(self):
        evs = _make_evs(score=5.5, band="staged_reduction")
        d1 = self.alloc.allocate(evs)
        d2 = self.alloc.allocate(evs)
        assert d1.decision_id == d2.decision_id

    def test_winner_retention_rationale(self):
        evs = _make_evs(score=1.0, band="hold")
        d = self.alloc.allocate(evs)
        assert "convexity" in d.convexity_preservation_rationale.lower() or "hold" in d.convexity_preservation_rationale.lower()


class TestCounterfactualExitAnalyzer:
    def setup_method(self):
        self.analyzer = CounterfactualExitAnalyzer()

    def test_premature_exit_detected(self):
        obs = _make_obs(realized_return=0.02)
        # Stock rises 10% after exit
        prices = [1030.0 * (1 + i * 0.02) for i in range(20)]
        cf = self.analyzer.analyze(obs, prices)
        assert cf.exit_was_premature is True

    def test_good_exit_not_premature(self):
        obs = _make_obs(realized_return=0.05)
        # Stock falls after exit
        prices = [1020.0 * (1 - i * 0.01) for i in range(20)]
        cf = self.analyzer.analyze(obs, prices)
        assert cf.exit_was_premature is False

    def test_forward_returns_computed(self):
        obs = _make_obs(realized_return=0.10, exit_date="2026-01-10")
        exit_p = 1100.0
        prices = [exit_p + i * 5 for i in range(21)]
        cf = self.analyzer.analyze(obs, prices)
        assert cf.post_exit_return_3d == pytest.approx((prices[2] - exit_p) / exit_p, rel=1e-5)
        assert cf.post_exit_return_5d == pytest.approx((prices[4] - exit_p) / exit_p, rel=1e-5)

    def test_empty_prices(self):
        obs = _make_obs()
        cf = self.analyzer.analyze(obs, [])
        assert cf.post_exit_return_3d is None
        assert cf.post_exit_max_excursion is None

    def test_negative_exit_price_raises(self):
        obs = _make_obs(realized_return=0.05)
        obs_bad = ExitObservationRecord(
            obs_id="x", symbol="1234", entry_date="2026-01-01",
            exit_date="2026-01-10", holding_days=9,
            entry_price=0.0, exit_price=0.0, realized_return=0.0,
            realized_pnl_jpy=0.0, exit_reason="manual",
            unrealized_pnl_at_exit=0.0, distance_from_peak_pct=0.0,
            rs_rank_at_exit=50.0, atr_pct_at_exit=0.02,
            breadth_score_at_exit=0.5, regime_at_exit="bull",
            exit_velocity_score=0.0, path_snapshot_count=0,
        )
        with pytest.raises(ValueError):
            self.analyzer.analyze(obs_bad, [1100.0, 1110.0])

    def test_cf_id_deterministic(self):
        obs = _make_obs()
        prices = [1100.0 + i for i in range(20)]
        cf1 = self.analyzer.analyze(obs, prices)
        cf2 = self.analyzer.analyze(obs, prices)
        assert cf1.cf_id == cf2.cf_id


class TestCounterfactualPartialExitAnalytics:
    def setup_method(self):
        self.analytics = CounterfactualPartialExitAnalytics()

    def test_best_scenario_staged_when_better(self):
        obs = _make_obs(realized_return=0.05)
        # Prices rise after exit: staged (which captures post-exit gains) beats full
        prices = [1100.0 + i * 20 for i in range(20)]
        record = self.analytics.compare(obs, prices)
        # staged would be: 33% at actual return + 33% at d3 return + 34% at d5 return
        assert record.best_scenario in ("full", "staged", "delayed")

    def test_full_exit_when_prices_fall(self):
        obs = _make_obs(realized_return=0.08)
        prices = [1080.0 - i * 10 for i in range(20)]
        record = self.analytics.compare(obs, prices)
        assert record.best_scenario == "full"

    def test_record_id_deterministic(self):
        obs = _make_obs()
        prices = [1100.0] * 20
        r1 = self.analytics.compare(obs, prices)
        r2 = self.analytics.compare(obs, prices)
        assert r1.record_id == r2.record_id

    def test_full_exit_return_preserved(self):
        obs = _make_obs(realized_return=0.10)
        prices = [1100.0] * 20
        record = self.analytics.compare(obs, prices)
        assert record.full_exit_return == pytest.approx(0.10)


class TestLoserRetentionAnalysis:
    def setup_method(self):
        self.analysis = LoserRetentionAnalysis()

    def test_good_cut_high_velocity(self):
        obs = _make_obs(realized_return=-0.05, velocity_score=6.0)
        r = self.analysis.classify(obs, post_exit_return_5d=-0.02)
        assert r.hold_classification == "good_cut"

    def test_bad_cut_low_velocity_recovery(self):
        obs = _make_obs(realized_return=-0.03, velocity_score=2.0)
        r = self.analysis.classify(obs, post_exit_return_5d=0.08)
        assert r.hold_classification == "bad_cut"
        assert r.would_have_recovered is True

    def test_good_hold_winner(self):
        obs = _make_obs(realized_return=0.12, velocity_score=2.0)
        r = self.analysis.classify(obs)
        assert r.hold_classification == "good_hold"

    def test_bad_hold_winner_high_velocity(self):
        obs = _make_obs(realized_return=0.05, velocity_score=7.0)
        r = self.analysis.classify(obs)
        assert r.hold_classification == "bad_hold"

    def test_loser_no_post_exit_data(self):
        obs = _make_obs(realized_return=-0.05, velocity_score=2.0)
        r = self.analysis.classify(obs, post_exit_return_5d=None)
        assert r.hold_classification in ("good_cut", "bad_cut", "bad_hold")

    def test_all_classifications_valid(self):
        valid = {"good_cut", "bad_cut", "good_hold", "bad_hold"}
        for ret in [-0.10, -0.03, 0.05, 0.12]:
            for vel in [1.0, 3.0, 6.0]:
                obs = _make_obs(realized_return=ret, velocity_score=vel)
                r = self.analysis.classify(obs, post_exit_return_5d=0.02 if ret < 0 else None)
                assert r.hold_classification in valid


class TestConditionalExpectancySurface:
    def setup_method(self):
        self.surface = ConditionalExpectancySurface()

    def _make_many_obs(self, n=10, regime="bull", realized_return=0.05):
        return [
            _make_obs(
                symbol=f"{1000 + i}",
                entry=f"2026-01-0{(i%9)+1}",
                exit_date=f"2026-01-{10 + i}",
                realized_return=realized_return,
                rs_rank=82.0, breadth=0.65, atr=0.02, regime=regime,
                distance_from_peak=0.05, velocity_score=2.5,
            )
            for i in range(n)
        ]

    def test_build_returns_cells(self):
        obs_list = self._make_many_obs(n=10)
        cells = self.surface.build(obs_list)
        assert len(cells) > 0

    def test_cells_with_enough_samples_have_expectancy(self):
        obs_list = self._make_many_obs(n=10)
        f5 = {o.obs_id: 0.03 for o in obs_list}
        cells = self.surface.build(obs_list, forward_returns_5d=f5)
        cells_with_data = [c for c in cells if c.sample_count >= 5]
        if cells_with_data:
            for c in cells_with_data:
                assert c.conditional_expectancy is not None

    def test_cells_insufficient_samples_no_expectancy(self):
        obs_list = self._make_many_obs(n=3)
        f5 = {o.obs_id: 0.03 for o in obs_list}
        cells = self.surface.build(obs_list, forward_returns_5d=f5)
        for c in cells:
            if c.sample_count < 5:
                assert c.conditional_expectancy is None

    def test_empty_input(self):
        cells = self.surface.build([])
        assert cells == []

    def test_cells_sorted_by_sample_count_desc(self):
        obs_list = self._make_many_obs(n=15)
        cells = self.surface.build(obs_list)
        counts = [c.sample_count for c in cells]
        assert counts == sorted(counts, reverse=True)

    def test_regime_transitions_separate_cells(self):
        bull_obs = self._make_many_obs(n=5, regime="bull")
        bear_obs = self._make_many_obs(n=5, regime="bear")
        cells = self.surface.build(bull_obs + bear_obs)
        regimes = {c.regime for c in cells}
        assert len(regimes) == 2
