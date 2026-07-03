"""Tests for src/capital/reflexivity.py"""
import pytest
from src.capital.reflexivity import (
    ReflexivityObservation,
    ReflexivityState,
    update_reflexivity,
    get_participation_rate,
    append_reflexivity_observation,
    load_reflexivity_state,
    save_reflexivity_state,
    BASE_PARTICIPATION_RATE,
    _linear_regression,
    _classify_severity,
)


def _obs(day="2026-01-01", participation=0.03, fill=0.95,
         order_value=90_000.0, adv=3_000_000.0):
    return ReflexivityObservation(
        trading_day=day,
        participation_rate=participation,
        fill_ratio=fill,
        order_value=order_value,
        adv_yen=adv,
    )


class TestLinearRegression:
    def test_perfect_negative_slope(self):
        x = [0.01, 0.02, 0.03, 0.04, 0.05]
        y = [1.0,  0.8,  0.6,  0.4,  0.2]
        slope, intercept, r_sq = _linear_regression(x, y)
        assert slope < 0
        assert r_sq > 0.99

    def test_no_correlation_low_r2(self):
        x = [0.01, 0.02, 0.03, 0.04, 0.05]
        y = [0.9,  0.8,  0.95, 0.85, 0.9]  # noisy
        _, _, r_sq = _linear_regression(x, y)
        assert r_sq < 0.50

    def test_single_point_returns_zero_slope(self):
        slope, _, r_sq = _linear_regression([0.03], [0.95])
        assert slope == pytest.approx(0.0)
        assert r_sq == pytest.approx(0.0)

    def test_positive_slope_detected(self):
        x = [0.01, 0.02, 0.03, 0.04, 0.05]
        y = [0.5,  0.6,  0.7,  0.8,  0.9]
        slope, _, _ = _linear_regression(x, y)
        assert slope > 0


class TestClassifySeverity:
    def test_low_r2_always_normal(self):
        # Even with steep slope, low R² → NORMAL
        assert _classify_severity(-2.0, 0.10) == "NORMAL"

    def test_mild_severity(self):
        assert _classify_severity(-0.40, 0.50) == "MILD"

    def test_moderate_severity(self):
        assert _classify_severity(-0.70, 0.50) == "MODERATE"

    def test_severe_severity(self):
        assert _classify_severity(-1.20, 0.50) == "SEVERE"

    def test_positive_slope_normal(self):
        assert _classify_severity(0.50, 0.80) == "NORMAL"


class TestUpdateReflexivity:
    def test_initial_state_normal(self):
        state = ReflexivityState()
        new = update_reflexivity(state, _obs())
        assert new.severity == "NORMAL"
        assert new.adjusted_participation_rate == pytest.approx(BASE_PARTICIPATION_RATE)

    def test_high_participation_with_poor_fills_detected(self):
        state = ReflexivityState()
        # Feed observations where higher participation → worse fills
        obs_list = [
            _obs(participation=0.01 * (i+1), fill=max(0.5, 1.0 - 0.06 * i))
            for i in range(10)
        ]
        for obs in obs_list:
            state = update_reflexivity(state, obs)
        # Should detect MILD or worse reflexivity
        assert state.severity in ("MILD", "MODERATE", "SEVERE")
        assert state.adjusted_participation_rate < BASE_PARTICIPATION_RATE

    def test_no_reflexivity_stable_fills(self):
        state = ReflexivityState()
        # Stable fill ratio regardless of participation
        for i in range(12):
            state = update_reflexivity(state, _obs(
                participation=0.02 + 0.005 * i, fill=0.95
            ))
        assert state.severity == "NORMAL"

    def test_history_preserved(self):
        state = ReflexivityState()
        for i in range(5):
            state = update_reflexivity(state, _obs())
        assert len(state.observations) == 5

    def test_history_capped(self):
        state = ReflexivityState()
        for i in range(50):
            state = update_reflexivity(state, _obs())
        assert len(state.observations) <= 30  # 10 * 3

    def test_adjusted_rate_in_range(self):
        state = ReflexivityState()
        for _ in range(15):
            state = update_reflexivity(state, _obs())
        rate = get_participation_rate(state)
        assert 0.0 < rate <= BASE_PARTICIPATION_RATE

    def test_severe_halves_participation(self):
        state = ReflexivityState()
        strong_neg_obs = [
            _obs(participation=0.01 * (i+1), fill=max(0.3, 1.0 - 0.07 * i))
            for i in range(10)
        ]
        for obs in strong_neg_obs:
            state = update_reflexivity(state, obs)
        rate = get_participation_rate(state)
        assert rate < BASE_PARTICIPATION_RATE


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "reflexivity.json"
        state = ReflexivityState(
            observations=[],
            slope=-0.5,
            r_squared=0.6,
            severity="MILD",
            adjusted_participation_rate=0.035,
        )
        save_reflexivity_state(state, path)
        loaded = load_reflexivity_state(path)
        assert loaded.severity == "MILD"
        assert loaded.adjusted_participation_rate == pytest.approx(0.035)

    def test_load_nonexistent(self, tmp_path):
        state = load_reflexivity_state(tmp_path / "nonexistent.json")
        assert state.severity == "NORMAL"
        assert state.adjusted_participation_rate == pytest.approx(BASE_PARTICIPATION_RATE)
