"""Tests for capital_state.py"""
import json
import pytest
from pathlib import Path

from src.capital.capital_state import (
    CapitalState,
    compute_effective_capital,
    compute_deployable,
    compute_risk_adjusted,
    update_capital_state,
    load_capital_state,
    save_capital_state,
    DAILY_GROWTH_LIMIT_DEFAULT,
)


class TestComputeEffectiveCapital:
    def test_initial_equity_equals_effective(self):
        new_eff, rate = compute_effective_capital(3_000_000, 3_000_000)
        assert new_eff == 3_000_000
        assert rate == 0.0

    def test_deposit_rate_limited(self):
        # 3M → 4M injection: should only grow by 1.5% per day
        new_eff, rate = compute_effective_capital(4_000_000, 3_000_000, 0.015)
        assert new_eff == pytest.approx(3_000_000 * 1.015)
        assert rate == pytest.approx(0.015, abs=1e-6)

    def test_deposit_below_limit_fully_reflected(self):
        # Small deposit: actual < prev * (1 + limit) → full reflection
        new_eff, rate = compute_effective_capital(3_010_000, 3_000_000, 0.015)
        # 3M * 1.015 = 3,045,000 > 3,010,000 → new_eff = actual
        assert new_eff == 3_010_000

    def test_drawdown_propagates_immediately(self):
        # Loss propagates without rate limit
        new_eff, rate = compute_effective_capital(2_700_000, 3_000_000)
        assert new_eff == 2_700_000
        assert rate == pytest.approx(-0.10, abs=1e-6)

    def test_freeze_prevents_growth(self):
        # growth_limit=0.0 → no growth
        new_eff, rate = compute_effective_capital(4_000_000, 3_000_000, 0.0)
        assert new_eff == 3_000_000
        assert rate == 0.0

    def test_freeze_still_propagates_drawdown(self):
        # Even with growth_limit=0, drawdown passes through
        new_eff, rate = compute_effective_capital(2_500_000, 3_000_000, 0.0)
        assert new_eff == 2_500_000

    def test_zero_actual_equity(self):
        new_eff, rate = compute_effective_capital(0, 3_000_000)
        assert new_eff == 0.0
        assert rate == 0.0

    def test_ramp_reaches_target_over_time(self):
        # Simulate 50-day ramp from 3M to 4M at 1.5%/day
        effective = 3_000_000.0
        actual = 4_000_000.0
        for _ in range(100):
            effective, _ = compute_effective_capital(actual, effective, 0.015)
        assert effective == pytest.approx(actual, rel=0.001)


class TestComputeDeployable:
    def test_10pct_buffer(self):
        result = compute_deployable(3_000_000, 0.10)
        assert result == pytest.approx(2_700_000)

    def test_zero_buffer(self):
        result = compute_deployable(3_000_000, 0.0)
        assert result == 3_000_000

    def test_negative_buffer_clamped(self):
        # Buffer < 0 treated as 0 (clamp to [0,1])
        result = compute_deployable(3_000_000, -0.05)
        assert result == pytest.approx(3_000_000)

    def test_buffer_above_1_clamped(self):
        # Buffer > 1 treated as 1 → deployable = 0
        result = compute_deployable(3_000_000, 1.5)
        assert result == pytest.approx(0.0)


class TestComputeRiskAdjusted:
    def test_all_scalars_one(self):
        result = compute_risk_adjusted(2_700_000, 1.0, 1.0, 1.0)
        assert result == 2_700_000

    def test_vol_scalar(self):
        result = compute_risk_adjusted(2_700_000, 0.5, 1.0, 1.0)
        assert result == pytest.approx(1_350_000)

    def test_all_scalars_reduce(self):
        result = compute_risk_adjusted(2_700_000, 0.8, 0.9, 0.7)
        assert result == pytest.approx(2_700_000 * 0.8 * 0.9 * 0.7, rel=1e-6)

    def test_scalar_above_1_clamped(self):
        # Scalars > 1 should be clamped to 1
        result = compute_risk_adjusted(2_700_000, 1.5, 1.0, 1.0)
        assert result == 2_700_000


class TestUpdateCapitalState:
    def setup_method(self):
        self.initial = CapitalState.initial(3_000_000, cash_buffer=0.10)

    def test_deposit_rate_limited(self):
        updated = update_capital_state(self.initial, 4_000_000)
        assert updated.effective_capital < 4_000_000
        assert updated.effective_capital > 3_000_000
        assert updated.actual_equity == 4_000_000

    def test_drawdown_immediate(self):
        updated = update_capital_state(self.initial, 2_500_000)
        assert updated.effective_capital == pytest.approx(2_500_000)

    def test_frozen_no_growth(self):
        updated = update_capital_state(self.initial, 4_000_000, growth_frozen=True)
        assert updated.effective_capital == pytest.approx(3_000_000)

    def test_frozen_drawdown_still_propagates(self):
        updated = update_capital_state(self.initial, 2_000_000, growth_frozen=True)
        assert updated.effective_capital == pytest.approx(2_000_000)

    def test_risk_adjusted_smaller_than_deployable(self):
        updated = update_capital_state(
            self.initial, 3_000_000,
            volatility_scalar=0.8, liquidity_scalar=0.9, execution_scalar=1.0
        )
        assert updated.risk_adjusted_capital < updated.deployable_capital

    def test_deployable_smaller_than_effective(self):
        updated = update_capital_state(self.initial, 3_000_000)
        assert updated.deployable_capital < updated.effective_capital

    def test_schema_version_preserved(self):
        state = CapitalState.initial(3_000_000)
        updated = update_capital_state(state, 3_100_000)
        assert updated.schema_version == state.schema_version


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        state = CapitalState.initial(3_500_000)
        path = tmp_path / "capital_state.json"
        save_capital_state(state, path)
        loaded = load_capital_state(path)
        assert loaded is not None
        assert loaded.actual_equity == state.actual_equity
        assert loaded.effective_capital == state.effective_capital

    def test_load_missing_returns_none(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        result = load_capital_state(path)
        assert result is None

    def test_load_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        result = load_capital_state(path)
        assert result is None

    def test_atomic_write(self, tmp_path):
        state = CapitalState.initial(3_000_000)
        path = tmp_path / "capital_state.json"
        save_capital_state(state, path)
        # tmp file should not linger
        assert not (tmp_path / "capital_state.tmp").exists()
        assert path.exists()
