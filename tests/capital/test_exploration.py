"""Tests for src/capital/exploration.py"""
import pytest
from src.capital.exploration import (
    ExplorationState,
    compute_exploration_budget,
    update_exploration_state,
    assess_exploration_allocation,
    record_exploration_pnl,
    EXPLORATION_FRACTION,
    MIN_EXPLORATION_BUDGET,
    MAX_EXPLORATION_POSITIONS,
)


def _state(**kwargs):
    defaults = dict(
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_budget=150_000.0,
        exploration_deployed=0.0,
        exploration_available=150_000.0,
        active_exploration_positions=[],
        cumulative_pnl_bps=0.0,
        exploration_active=True,
    )
    defaults.update(kwargs)
    return ExplorationState(**defaults)


class TestComputeExplorationBudget:
    def test_five_percent_of_capital(self):
        budget = compute_exploration_budget(3_000_000.0)
        assert budget == pytest.approx(150_000.0)

    def test_zero_capital(self):
        assert compute_exploration_budget(0.0) == pytest.approx(0.0)

    def test_negative_capital(self):
        assert compute_exploration_budget(-1000.0) == pytest.approx(0.0)

    def test_fraction_clamped_to_10pct(self):
        budget = compute_exploration_budget(1_000_000.0, exploration_fraction=0.50)
        assert budget == pytest.approx(100_000.0)  # clamped to 10%

    def test_isolation_from_main_capital(self):
        # exploration_budget is ADDITIVE overhead, not subtracted from deployable
        effective = 3_000_000.0
        budget = compute_exploration_budget(effective)
        # Main deployable remains full (minus cash buffer), exploration is separate
        assert budget == pytest.approx(effective * EXPLORATION_FRACTION)


class TestUpdateExplorationState:
    def test_budget_updates_with_equity(self):
        state = ExplorationState()
        new = update_exploration_state(state, effective_capital=4_000_000.0)
        assert new.exploration_budget == pytest.approx(200_000.0)

    def test_deployed_reflects_positions(self):
        state = ExplorationState(exploration_fraction=EXPLORATION_FRACTION,
                                 exploration_budget=150_000.0,
                                 exploration_deployed=0.0,
                                 exploration_available=150_000.0,
                                 active_exploration_positions=["9984.T"],
                                 cumulative_pnl_bps=0.0,
                                 exploration_active=True)
        new = update_exploration_state(state, 3_000_000.0,
                                       active_positions=["9984.T"],
                                       position_values={"9984.T": 80_000.0})
        assert new.exploration_deployed == pytest.approx(80_000.0)
        assert new.exploration_available == pytest.approx(150_000.0 - 80_000.0)


class TestAssessExplorationAllocation:
    def test_approved_within_budget(self):
        state = _state(exploration_available=150_000.0)
        result = assess_exploration_allocation(state, "9984.T", 100_000.0)
        assert result.approved is True
        assert result.allocated_value == pytest.approx(100_000.0)

    def test_rejected_flag_off(self):
        state = _state(exploration_active=False)
        result = assess_exploration_allocation(state, "9984.T", 100_000.0)
        assert result.approved is False
        assert "exploration_active=False" in result.reason

    def test_rejected_insufficient_budget(self):
        state = _state(exploration_budget=30_000.0, exploration_available=30_000.0)
        result = assess_exploration_allocation(state, "9984.T", 30_000.0)
        # budget < MIN_EXPLORATION_BUDGET (50_000) → rejected
        assert result.approved is False

    def test_rejected_exceeds_available(self):
        state = _state(exploration_available=50_000.0)
        result = assess_exploration_allocation(state, "9984.T", 100_000.0)
        assert result.approved is False

    def test_rejected_max_positions_reached(self):
        state = _state(active_exploration_positions=["9984.T"])
        result = assess_exploration_allocation(state, "7203.T", 80_000.0)
        assert result.approved is False

    def test_rejected_already_in_exploration(self):
        state = _state(active_exploration_positions=["9984.T"],
                       exploration_available=200_000.0)
        result = assess_exploration_allocation(state, "9984.T", 50_000.0)
        assert result.approved is False

    def test_budget_remaining_updated(self):
        state = _state(exploration_available=150_000.0)
        result = assess_exploration_allocation(state, "9984.T", 80_000.0)
        assert result.budget_remaining_after == pytest.approx(70_000.0)


class TestRecordExplorationPnl:
    def test_accumulates_pnl(self):
        state = _state(cumulative_pnl_bps=10.0)
        new = record_exploration_pnl(state, 5.0)
        assert new.cumulative_pnl_bps == pytest.approx(15.0)

    def test_negative_pnl(self):
        state = _state(cumulative_pnl_bps=10.0)
        new = record_exploration_pnl(state, -3.0)
        assert new.cumulative_pnl_bps == pytest.approx(7.0)
