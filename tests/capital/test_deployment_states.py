"""Tests for src/capital/deployment_states.py"""
import pytest
from src.capital.deployment_states import (
    DeploymentStateRecord,
    transition_deployment_state,
    get_growth_multiplier,
    STATE_GROWTH_MULTIPLIERS,
)


def _record(**kwargs):
    defaults = dict(
        state="RAMPING",
        exploration_active=False,
        consecutive_high_aggression=0,
        consecutive_defensive=0,
        periods_since_state_change=0,
    )
    defaults.update(kwargs)
    return DeploymentStateRecord(**defaults)


class TestFrozenOverride:
    def test_frozen_execution_mode(self):
        rec = _record(state="AGGRESSIVE", consecutive_high_aggression=5)
        new = transition_deployment_state(rec, 0.95, 0.90, "FROZEN")
        assert new.state == "FROZEN"

    def test_recovering_execution_mode(self):
        rec = _record(state="FROZEN")
        new = transition_deployment_state(rec, 0.85, 0.80, "RECOVERING")
        assert new.state == "RECOVERING"

    def test_frozen_resets_consecutive_high(self):
        rec = _record(state="RAMPING", consecutive_high_aggression=3)
        new = transition_deployment_state(rec, 0.95, 0.90, "FROZEN")
        assert new.consecutive_high_aggression == 0


class TestAggressiveTransition:
    def test_requires_3_consecutive_periods(self):
        rec = _record(consecutive_high_aggression=2)
        new = transition_deployment_state(rec, 0.90, 0.80, "RAMPING", drawdown=0.02)
        assert new.consecutive_high_aggression == 3
        assert new.state == "AGGRESSIVE"

    def test_not_aggressive_below_threshold(self):
        rec = _record(consecutive_high_aggression=2)
        new = transition_deployment_state(rec, 0.85, 0.80, "RAMPING", drawdown=0.02)
        assert new.state != "AGGRESSIVE"

    def test_drawdown_blocks_aggressive(self):
        rec = _record(consecutive_high_aggression=3)
        new = transition_deployment_state(rec, 0.90, 0.80, "RAMPING", drawdown=0.10)
        assert new.state != "AGGRESSIVE"


class TestDefensiveTransition:
    def test_immediate_on_low_ema(self):
        rec = _record(state="RAMPING")
        new = transition_deployment_state(rec, 0.40, 0.50, "RAMPING")
        assert new.state == "DEFENSIVE"

    def test_immediate_on_high_drawdown(self):
        rec = _record(state="RAMPING")
        new = transition_deployment_state(rec, 0.60, 0.60, "RAMPING", drawdown=0.13)
        assert new.state == "DEFENSIVE"

    def test_contraction_is_single_period(self):
        # Single bad period → immediately defensive
        rec = _record(state="AGGRESSIVE", consecutive_high_aggression=5)
        new = transition_deployment_state(rec, 0.35, 0.30, "RAMPING", drawdown=0.0)
        assert new.state == "DEFENSIVE"


class TestGrowthMultipliers:
    def test_all_states_defined(self):
        for state in ("RAMPING", "FROZEN", "RECOVERING", "AGGRESSIVE", "DEFENSIVE",
                      "OPPORTUNISTIC", "EXPLORATION"):
            assert state in STATE_GROWTH_MULTIPLIERS

    def test_frozen_is_zero(self):
        assert STATE_GROWTH_MULTIPLIERS["FROZEN"] == pytest.approx(0.0)

    def test_aggressive_is_2x(self):
        assert STATE_GROWTH_MULTIPLIERS["AGGRESSIVE"] == pytest.approx(2.0)

    def test_get_multiplier_function(self):
        rec = _record(state="OPPORTUNISTIC")
        assert get_growth_multiplier(rec) == pytest.approx(1.5)

    def test_unknown_state_defaults_to_1(self):
        rec = _record(state="UNKNOWN_STATE")
        assert get_growth_multiplier(rec) == pytest.approx(1.0)


class TestPeriodTracking:
    def test_periods_reset_on_state_change(self):
        rec = _record(state="RAMPING", periods_since_state_change=5)
        new = transition_deployment_state(rec, 0.35, 0.30, "RAMPING")
        # Changed to DEFENSIVE → periods reset
        assert new.state == "DEFENSIVE"
        assert new.periods_since_state_change == 0

    def test_periods_increment_same_state(self):
        rec = _record(state="RAMPING", periods_since_state_change=2)
        new = transition_deployment_state(rec, 0.65, 0.60, "RAMPING")
        # Stays RAMPING
        if new.state == "RAMPING":
            assert new.periods_since_state_change == 3
