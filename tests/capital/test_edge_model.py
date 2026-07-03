"""Tests for src/capital/edge_model.py"""
import pytest
from src.capital.edge_model import (
    EdgeObservation,
    EdgeModelState,
    update_edge_model,
    build_edge_observation,
)


def _obs(day="2026-01-01", breadth=0.5, strength=0.7, momentum=0.6,
         alignment=1.0, alpha_bps=15.0):
    return EdgeObservation(
        trading_day=day,
        rsr_breadth_75=breadth,
        signal_strength=strength,
        momentum_quality=momentum,
        regime_alignment=alignment,
        net_alpha_proxy_bps=alpha_bps,
    )


class TestBuildEdgeObservation:
    def test_signal_strength_normalized(self):
        obs = build_edge_observation(
            trading_day="2026-01-01",
            rsr_breadth_75=0.5,
            top_signal_rsr_avg=80.0,
            rsr_momentum_positive_fraction=0.6,
            regime_is_favorable=True,
        )
        assert obs.signal_strength == pytest.approx(0.80)

    def test_regime_alignment_favorable(self):
        obs = build_edge_observation("2026-01-01", 0.5, 80.0, 0.6, True)
        assert obs.regime_alignment == pytest.approx(1.0)

    def test_regime_alignment_unfavorable(self):
        obs = build_edge_observation("2026-01-01", 0.5, 80.0, 0.6, False)
        assert obs.regime_alignment == pytest.approx(0.6)

    def test_clamps_breadth(self):
        obs = build_edge_observation("2026-01-01", 1.5, 80.0, 0.6, True)
        assert obs.rsr_breadth_75 <= 1.0

    def test_zero_rsr_avg(self):
        obs = build_edge_observation("2026-01-01", 0.5, 0.0, 0.6, True)
        assert obs.signal_strength == pytest.approx(0.0)


class TestUpdateEdgeModel:
    def test_initial_state_single_obs(self):
        state = EdgeModelState()
        obs = _obs()
        new = update_edge_model(state, obs)
        assert new.composite_edge_score >= 0.0
        assert new.composite_edge_score <= 1.0
        assert len(new.observations) == 1

    def test_history_grows(self):
        state = EdgeModelState()
        for i in range(10):
            state = update_edge_model(state, _obs(day=f"2026-01-{i+1:02d}"))
        assert len(state.observations) == 10

    def test_history_capped(self):
        state = EdgeModelState()
        for i in range(250):
            state = update_edge_model(state, _obs(day=f"2026-01-01"))
        # max_history = PERSISTENCE_WINDOW_LONG * 3 = 63*3 = 189
        assert len(state.observations) <= 189

    def test_stable_high_quality_yields_high_score(self):
        state = EdgeModelState()
        for i in range(30):
            state = update_edge_model(state, _obs(
                breadth=0.8, strength=0.85, momentum=0.75, alignment=1.0
            ))
        assert state.composite_edge_score >= 0.50

    def test_volatile_quality_yields_low_score(self):
        state = EdgeModelState()
        for i in range(30):
            q = 0.9 if i % 2 == 0 else 0.1
            state = update_edge_model(state, _obs(breadth=q, strength=q, momentum=q))
        assert state.composite_edge_score < 0.60

    def test_alpha_decay_detected(self):
        state = EdgeModelState()
        # Weak long-term, strong recent
        for _ in range(21):
            state = update_edge_model(state, _obs(breadth=0.2, strength=0.2))
        for _ in range(5):
            state = update_edge_model(state, _obs(breadth=0.9, strength=0.9))
        assert state.alpha_decay_rate > 0.0

    def test_persistence_scores_in_range(self):
        state = EdgeModelState()
        for i in range(70):
            state = update_edge_model(state, _obs())
        assert 0.0 <= state.persistence_score_5d <= 1.0
        assert 0.0 <= state.persistence_score_21d <= 1.0
        assert 0.0 <= state.persistence_score_63d <= 1.0

    def test_deterministic(self):
        state1 = EdgeModelState()
        state2 = EdgeModelState()
        obs = _obs()
        r1 = update_edge_model(state1, obs)
        r2 = update_edge_model(state2, obs)
        assert r1.composite_edge_score == r2.composite_edge_score
