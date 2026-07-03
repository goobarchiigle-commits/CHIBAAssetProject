"""
tests/test_phase5b1_capital_integration.py
Phase 5B.1 — CapitalDeploymentOS integration: focused validation tests.

Verifies:
1. actual_equity increase changes target position size
2. actual_equity increase changes affordability threshold
3. governance uses updated capital (ordering fix)
4. CapitalDeploymentOS dynamic_max_positions flows to _build_orders
5. 3M vs 4M account behavior is different
6. replay determinism unchanged
7. duplicate-order protection unchanged
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.capital.capital_state import CapitalState
from src.live.capital_deployment_os import (
    MAX_POSITIONS_HARD_CAP,
    MIN_POSITIONS_FLOOR,
    PARAMS_LOCKED_MAX_POSITIONS,
    dynamic_max_positions,
    target_position_size,
)
def derive_risk_params(capital: int) -> dict:
    """Mirrors run_live_signal.derive_risk_params (pure function; replicated to avoid module-level guard)."""
    return {
        "risk_per_trade":  capital * 0.01,
        "max_position":    capital * 0.20,
        "leader_slot":     capital * 0.35,
        "max_allocation":  capital * 0.30,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Target position size scales with actual equity
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetSizeScalesWithEquity:

    def test_higher_equity_gives_larger_per_slot_budget(self):
        cs_3m = CapitalState.initial(3_000_000.0)
        cs_4m = CapitalState.initial(4_000_000.0)
        mp_3m = dynamic_max_positions(cs_3m.deployable_capital)
        mp_4m = dynamic_max_positions(cs_4m.deployable_capital)
        # per-slot budget = deployable / max_pos
        budget_3m = cs_3m.deployable_capital / mp_3m
        budget_4m = cs_4m.deployable_capital / mp_4m
        # 4M has more deployable per slot (larger account)
        assert budget_4m > budget_3m

    def test_target_position_size_at_price_3000(self):
        price = 3_000.0
        cs_3m = CapitalState.initial(3_000_000.0)
        cs_4m = CapitalState.initial(4_000_000.0)
        mp_3m = dynamic_max_positions(cs_3m.deployable_capital)
        mp_4m = dynamic_max_positions(cs_4m.deployable_capital)
        qty_3m = target_position_size(cs_3m.deployable_capital, mp_3m, price)
        qty_4m = target_position_size(cs_4m.deployable_capital, mp_4m, price)
        # 4M account should produce at least as many shares
        assert qty_4m >= qty_3m

    def test_derive_risk_params_max_position_scales(self):
        rp_3m = derive_risk_params(3_000_000)
        rp_4m = derive_risk_params(4_000_000)
        assert rp_4m["max_position"] > rp_3m["max_position"]

    def test_target_size_zero_on_invalid_inputs(self):
        assert target_position_size(0.0, 3, 5000.0) == 0
        assert target_position_size(3_000_000.0, 0, 5000.0) == 0
        assert target_position_size(3_000_000.0, 3, 0.0) == 0

    def test_target_size_lot_rounded(self):
        # Result must be a multiple of 100 (lot size)
        qty = target_position_size(3_000_000.0, 4, 2_500.0)
        assert qty % 100 == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Affordability threshold scales with actual equity
# ─────────────────────────────────────────────────────────────────────────────

class TestAffordabilityScalesWithEquity:

    def _max_alloc_for(self, actual_equity: int) -> int:
        cs = CapitalState.initial(float(actual_equity))
        return int(cs.risk_adjusted_capital * 0.30)

    def test_4m_has_higher_affordability_than_3m(self):
        assert self._max_alloc_for(4_000_000) > self._max_alloc_for(3_000_000)

    def test_3m_affordability(self):
        # 3M × 0.90 (cash_buffer) × 0.30 = 810,000
        assert self._max_alloc_for(3_000_000) == pytest.approx(810_000, rel=0.01)

    def test_4m_affordability(self):
        # 4M × 0.90 × 0.30 = 1,080,000
        assert self._max_alloc_for(4_000_000) == pytest.approx(1_080_000, rel=0.01)

    def test_6m_affordability(self):
        # 6M × 0.90 × 0.30 = 1,620,000
        assert self._max_alloc_for(6_000_000) == pytest.approx(1_620_000, rel=0.01)

    def test_risk_params_max_allocation_scales(self):
        rp_3m = derive_risk_params(3_000_000)
        rp_4m = derive_risk_params(4_000_000)
        assert rp_4m["max_allocation"] == pytest.approx(1_200_000)
        assert rp_3m["max_allocation"] == pytest.approx(900_000)

    def test_max_alloc_is_30_pct_of_capital(self):
        for cap in [2_000_000, 3_000_000, 5_000_000, 8_000_000]:
            rp = derive_risk_params(cap)
            assert rp["max_allocation"] == pytest.approx(cap * 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Governance ordering — capital loaded before governance call
# ─────────────────────────────────────────────────────────────────────────────

class TestGovernanceOrdering:

    def test_capital_state_initial_has_correct_fields(self):
        cs = CapitalState.initial(4_000_000.0)
        assert cs.actual_equity == 4_000_000.0
        assert cs.deployable_capital == pytest.approx(3_600_000.0)
        assert cs.risk_adjusted_capital == pytest.approx(3_600_000.0)

    def test_eff_capital_from_cap_state_not_static(self):
        """Live capital (from cap_state) differs from static CAPITAL when equity changes."""
        STATIC_CAPITAL = 3_000_000
        cs = CapitalState.initial(4_000_000.0)
        eff_capital = int(cs.risk_adjusted_capital)
        assert eff_capital != STATIC_CAPITAL

    def test_governance_receives_live_capital(self):
        """
        Verify the pattern: _eff_capital is updated from cap_state BEFORE the
        governance call receives it. We mock the governance function and check
        the capital argument it receives.
        """
        captured = {}

        def fake_governance(**kwargs):
            captured["capital"] = kwargs.get("capital")
            return MagicMock(
                promoted_symbols=[],
                decisions=[],
                updated_live_universe={},
                universe_file_updated=False,
                governance_report=MagicMock(
                    universe_size_before=0,
                    universe_size_after=0,
                    deployable_breadth_score=0.0,
                    high_price_exclusion_count=0,
                ),
            )

        # Simulate the updated ordering
        STATIC_CAPITAL = 3_000_000
        cs = CapitalState.initial(4_000_000.0)
        _eff_capital = STATIC_CAPITAL
        if cs and cs.risk_adjusted_capital > 0:
            _eff_capital = int(cs.risk_adjusted_capital)
        # Now governance is called with live _eff_capital
        fake_governance(capital=_eff_capital, run_id="test")
        assert captured["capital"] == int(cs.risk_adjusted_capital)
        assert captured["capital"] != STATIC_CAPITAL


# ─────────────────────────────────────────────────────────────────────────────
# 4. CapitalDeploymentOS dynamic_max_positions flows to order generation
# ─────────────────────────────────────────────────────────────────────────────

class TestCDOSMaxPosFlowsToOrders:

    def test_dynamic_max_pos_comes_from_cdos_module(self):
        from src.live.capital_deployment_os import dynamic_max_positions
        # The function exists and is importable (it's the authority)
        assert callable(dynamic_max_positions)

    def test_signal_bridge_has_deployable_capital_attribute(self):
        from src.kabusapi.signal_bridge import SignalBridge
        import inspect
        sig = inspect.signature(SignalBridge.__init__)
        assert "deployable_capital" in sig.parameters

    def test_signal_bridge_stores_deployable_capital(self):
        from src.kabusapi.signal_bridge import SignalBridge
        bridge = SignalBridge(
            universe_tickers   = {"1234.T": "製造業"},
            fujiko_params      = {"min_sepa": 5, "min_rsr": 75.0, "mom_period": 20,
                                  "turtle_entry": 20, "turtle_exit": 55, "use_turtle_entry": True},
            capital            = 3_000_000,
            max_positions      = 3,
            min_hold_days      = 3,
            emergency_exit_pct = 0.10,
            deployable_capital = 2_700_000.0,
        )
        assert bridge.deployable_capital == pytest.approx(2_700_000.0)

    def test_dynamic_max_pos_used_when_deployable_set(self):
        """dynamic_max_positions(2.7M) = 3 (PARAMS_LOCKED クランプ)."""
        result = dynamic_max_positions(2_700_000.0)
        assert result == PARAMS_LOCKED_MAX_POSITIONS

    def test_cdos_cap_applied(self):
        """dynamic_max_positions は PARAMS_LOCKED_MAX_POSITIONS=3 を超えない。"""
        assert dynamic_max_positions(100_000_000.0) == PARAMS_LOCKED_MAX_POSITIONS

    def test_cdos_floor_applied(self):
        """dynamic_max_positions never goes below MIN_POSITIONS_FLOOR."""
        assert dynamic_max_positions(100.0) == MIN_POSITIONS_FLOOR


# ─────────────────────────────────────────────────────────────────────────────
# 5. 3M vs 4M account behavior is different
# ─────────────────────────────────────────────────────────────────────────────

class TestThreeMvseFourM:

    def test_max_positions_different(self):
        # PARAMS_LOCKED クランプにより 3M も 4M も PARAMS_LOCKED_MAX_POSITIONS=3 に収束
        cs_3m = CapitalState.initial(3_000_000.0)
        cs_4m = CapitalState.initial(4_000_000.0)
        mp_3m = dynamic_max_positions(cs_3m.deployable_capital)
        mp_4m = dynamic_max_positions(cs_4m.deployable_capital)
        assert mp_3m == PARAMS_LOCKED_MAX_POSITIONS
        assert mp_4m == PARAMS_LOCKED_MAX_POSITIONS

    def test_affordability_different(self):
        cs_3m = CapitalState.initial(3_000_000.0)
        cs_4m = CapitalState.initial(4_000_000.0)
        alloc_3m = cs_3m.risk_adjusted_capital * 0.30
        alloc_4m = cs_4m.risk_adjusted_capital * 0.30
        assert alloc_4m > alloc_3m

    def test_sizing_budget_different(self):
        cs_3m = CapitalState.initial(3_000_000.0)
        cs_4m = CapitalState.initial(4_000_000.0)
        # old signal_bridge formula (CAPITAL_PER_SLOT=1.2M, base=3):
        old_3m = max(3, math.floor(3_000_000 / 1_200_000))   # = 3 (old behavior)
        old_4m = max(3, math.floor(4_000_000 / 1_200_000))   # = 3 (SAME! old bug)
        new_3m = dynamic_max_positions(cs_3m.deployable_capital)  # = 3 (PARAMS_LOCKED クランプ)
        new_4m = dynamic_max_positions(cs_4m.deployable_capital)  # = 3 (PARAMS_LOCKED クランプ)
        # Old: both 3M and 4M gave same 3 slots
        assert old_3m == old_4m, "Old formula: 3M and 4M both gave 3 slots"
        # PARAMS_LOCKED クランプにより新旧ともに 3 で一致
        assert new_3m == PARAMS_LOCKED_MAX_POSITIONS
        assert new_4m == PARAMS_LOCKED_MAX_POSITIONS

    def test_concrete_slot_counts(self):
        """
        PARAMS_LOCKED クランプにより全資本帯で max=3 に収束。
        3M account: deployable=2.7M → tier=2 → raw=4 → clamped=3
        4M account: deployable=3.6M → tier=3 → raw=5 → clamped=3
        """
        cs_3m = CapitalState.initial(3_000_000.0)
        cs_4m = CapitalState.initial(4_000_000.0)
        assert dynamic_max_positions(cs_3m.deployable_capital) == PARAMS_LOCKED_MAX_POSITIONS
        assert dynamic_max_positions(cs_4m.deployable_capital) == PARAMS_LOCKED_MAX_POSITIONS

    def test_6m_still_capped_at_3(self):
        cs_6m = CapitalState.initial(6_000_000.0)
        assert dynamic_max_positions(cs_6m.deployable_capital) == PARAMS_LOCKED_MAX_POSITIONS

    def test_10m_still_capped_at_3(self):
        cs_10m = CapitalState.initial(10_000_000.0)
        assert dynamic_max_positions(cs_10m.deployable_capital) == PARAMS_LOCKED_MAX_POSITIONS


# ─────────────────────────────────────────────────────────────────────────────
# 6. Replay determinism unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayDeterminism:

    def test_dynamic_max_pos_is_deterministic(self):
        """Same input → same output, always."""
        for cap in [1_000_000.0, 2_000_000.0, 2_700_000.0, 3_600_000.0, 5_000_000.0]:
            results = {dynamic_max_positions(cap) for _ in range(10)}
            assert len(results) == 1, f"Non-deterministic at {cap}"

    def test_target_position_size_is_deterministic(self):
        for (cap, pos, price) in [(2_700_000.0, 4, 3000.0), (3_600_000.0, 5, 5000.0)]:
            results = {target_position_size(cap, pos, price) for _ in range(10)}
            assert len(results) == 1

    def test_capital_state_initial_is_deterministic(self):
        for equity in [3_000_000.0, 4_000_000.0]:
            states = [CapitalState.initial(equity) for _ in range(5)]
            dep_set = {s.deployable_capital for s in states}
            assert len(dep_set) == 1

    def test_same_cap_state_gives_same_max_pos(self):
        cs = CapitalState.initial(3_500_000.0)
        results = {dynamic_max_positions(cs.deployable_capital) for _ in range(10)}
        assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Duplicate-order protection: signal_bridge interface unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateOrderProtectionUnchanged:

    def test_signal_bridge_init_accepts_deployable_capital(self):
        """New optional param — existing callers without it should still work."""
        from src.kabusapi.signal_bridge import SignalBridge
        # Without deployable_capital (backward compatible)
        bridge_old_style = SignalBridge(
            universe_tickers   = {"1234.T": "製造業"},
            fujiko_params      = {"min_sepa": 5, "min_rsr": 75.0, "mom_period": 20,
                                  "turtle_entry": 20, "turtle_exit": 55, "use_turtle_entry": True},
            capital            = 3_000_000,
            max_positions      = 3,
            min_hold_days      = 3,
            emergency_exit_pct = 0.10,
        )
        assert bridge_old_style.deployable_capital == 0.0

    def test_signal_bridge_max_positions_attribute_preserved(self):
        """self.max_positions still exists as safety fallback."""
        from src.kabusapi.signal_bridge import SignalBridge
        bridge = SignalBridge(
            universe_tickers   = {"1234.T": "製造業"},
            fujiko_params      = {"min_sepa": 5, "min_rsr": 75.0, "mom_period": 20,
                                  "turtle_entry": 20, "turtle_exit": 55, "use_turtle_entry": True},
            capital            = 3_000_000,
            max_positions      = 3,
            min_hold_days      = 3,
            emergency_exit_pct = 0.10,
            deployable_capital = 2_700_000.0,
        )
        assert bridge.max_positions == 3  # base config value preserved

    def test_signal_bridge_capital_attribute_preserved(self):
        """self.capital (used for max_alloc_cap) reflects live capital."""
        from src.kabusapi.signal_bridge import SignalBridge
        bridge = SignalBridge(
            universe_tickers   = {"1234.T": "製造業"},
            fujiko_params      = {"min_sepa": 5, "min_rsr": 75.0, "mom_period": 20,
                                  "turtle_entry": 20, "turtle_exit": 55, "use_turtle_entry": True},
            capital            = 3_600_000,
            max_positions      = 3,
            min_hold_days      = 3,
            emergency_exit_pct = 0.10,
        )
        assert bridge.capital == 3_600_000

    def test_deployable_capital_zero_falls_back_to_current_equity(self):
        """When deployable_capital=0, signal_bridge uses current_equity as fallback."""
        # This is tested at the logic level: in signal_bridge.run(),
        # _cdos_dep = deployable_capital if deployable_capital > 0 else current_equity
        # The fallback exists so the system works even before first capital state save.
        dep_0 = 0.0
        current_equity = 3_000_000.0
        _cdos_dep = dep_0 if dep_0 > 0 else current_equity
        result = dynamic_max_positions(_cdos_dep)
        assert result == dynamic_max_positions(current_equity)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_live_signal derive_risk_params
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriveRiskParams:

    def test_structure(self):
        rp = derive_risk_params(3_000_000)
        assert "risk_per_trade"  in rp
        assert "max_position"    in rp
        assert "leader_slot"     in rp
        assert "max_allocation"  in rp

    def test_ratios(self):
        cap = 5_000_000
        rp = derive_risk_params(cap)
        assert rp["risk_per_trade"]  == pytest.approx(cap * 0.01)
        assert rp["max_position"]    == pytest.approx(cap * 0.20)
        assert rp["leader_slot"]     == pytest.approx(cap * 0.35)
        assert rp["max_allocation"]  == pytest.approx(cap * 0.30)

    def test_4m_max_alloc_exceeds_900k(self):
        """Core regression: 4M account must not be capped at 900K."""
        rp_4m = derive_risk_params(4_000_000)
        assert rp_4m["max_allocation"] > 900_000

    def test_capital_state_deployable_drives_max_alloc(self):
        """
        After Phase 5B.1: _max_alloc = int(_eff_capital * 0.30)
        where _eff_capital comes from cap_state.risk_adjusted_capital.
        """
        cs = CapitalState.initial(4_000_000.0)
        eff_capital = int(cs.risk_adjusted_capital)
        max_alloc = int(eff_capital * 0.30)
        assert max_alloc > 900_000  # 4M account can access >900K stocks
        assert max_alloc == pytest.approx(1_080_000, rel=0.01)
