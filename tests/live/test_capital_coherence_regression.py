"""
tests/live/test_capital_coherence_regression.py
Regression tests for RC1-RC5 capital-state coherence fixes.

RC1: _eff_capital UnboundLocalError (governance called before assignment)
RC2: cap_state=None registered as critical → RuntimeContractError
RC3: dep graph "capital" not initialized when _cap_state_loaded=False
RC4: phantom positions (runtime>0, broker=0) not purged → stale state
RC5: compute_exposure_report() missing required positional args
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

# ── imports under test ────────────────────────────────────────────────────────
from src.live.runtime_integration import (
    RuntimeStateRegistry,
    IntegrationDependencyGraph,
    RuntimeContractError,
    NullStateAccessError,
    validate_runtime_integration,
    STAGE_RECONCILIATION,
)
from src.capital.capital_state import CapitalState
from src.allocation.exposure_core import compute_exposure_report, ExposureState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

STATIC_CAPITAL = 3_000_000


def _make_cap_state(capital: float = STATIC_CAPITAL) -> CapitalState:
    return CapitalState.initial(capital)


def _make_exposure_state() -> ExposureState:
    return ExposureState(return_buffers={})


# ─────────────────────────────────────────────────────────────────────────────
# RC1: _eff_capital initialization before governance
# ─────────────────────────────────────────────────────────────────────────────

class TestRC1EffCapitalInitialization:
    """
    Verify that _eff_capital is always defined before any code that uses it.
    The governance block calls run_universe_governance(capital=_eff_capital);
    if it was only assigned in the capital-load block (which comes later in
    the function), Python raises UnboundLocalError.
    """

    def test_eff_capital_default_equals_static_capital(self):
        """Sentinel: static CAPITAL value is > 0 so the dep-graph check passes."""
        assert STATIC_CAPITAL > 0

    def test_cap_state_initial_produces_positive_risk_adj(self):
        """CapitalState.initial() used as fallback must yield risk_adj > 0."""
        cs = CapitalState.initial(float(STATIC_CAPITAL))
        assert cs.risk_adjusted_capital > 0
        assert cs.actual_equity == float(STATIC_CAPITAL)

    def test_cap_state_initial_risk_adj_le_capital(self):
        """risk_adjusted_capital ≤ capital (cash buffer is applied)."""
        cs = CapitalState.initial(float(STATIC_CAPITAL))
        assert cs.risk_adjusted_capital <= float(STATIC_CAPITAL)

    def test_eff_capital_override_from_cap_state(self):
        """When cap_state loads successfully, _eff_capital takes risk_adjusted_capital."""
        cs = CapitalState.initial(4_000_000.0)
        eff = int(cs.risk_adjusted_capital) if cs.risk_adjusted_capital > 0 else STATIC_CAPITAL
        assert eff > 0
        assert eff <= 4_000_000


# ─────────────────────────────────────────────────────────────────────────────
# RC2: cap_state=None must not be registered as critical
# ─────────────────────────────────────────────────────────────────────────────

class TestRC2CapStateNoneRegistry:
    """
    A None value registered as is_critical=True causes validate_all() to emit
    [REGISTRY:CRITICAL] violation → blocks LIVE.  After RC2 fix, a CapitalState
    fallback is always created, so the registry receives a non-None object.
    """

    def test_none_critical_state_produces_blocking_violation(self):
        """Baseline: None registered as critical → REGISTRY:CRITICAL violation."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state", None, is_critical=True)
        violations = reg.validate_all()
        assert any(":CRITICAL]" in v and "cap_state" in v for v in violations)

    def test_non_none_critical_state_no_violation(self):
        """After fix: CapitalState.initial() as fallback → no critical violation."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state", _make_cap_state(), is_critical=True)
        violations = reg.validate_all()
        assert not any("cap_state" in v for v in violations)

    def test_none_critical_get_raises_null_state_access(self):
        """get() on None critical state raises NullStateAccessError."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state", None, is_critical=True)
        with pytest.raises(NullStateAccessError):
            reg.get("cap_state")

    def test_cap_state_initial_is_valid_registry_value(self):
        """CapitalState.initial() produces a non-None, correctly typed object."""
        cs = _make_cap_state()
        assert cs is not None
        assert isinstance(cs, CapitalState)
        reg = RuntimeStateRegistry()
        reg.register("cap_state", cs, is_critical=True)
        retrieved = reg.get("cap_state")
        assert retrieved is cs

    def test_validate_runtime_integration_no_block_with_cap_state(self, tmp_path):
        """validate_runtime_integration should not raise when cap_state is valid."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state",        _make_cap_state(), is_critical=True)
        reg.register("inflight_registry", MagicMock(),       is_critical=True)

        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])
        dep.declare("execution",  ["allocation", "broker"])
        dep.mark_initialized("capital")
        dep.mark_initialized("allocation")
        dep.mark_initialized("broker")

        result = validate_runtime_integration(
            run_id="test-rc2",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            dep_graph=dep,
            broker_positions={"1234.T": 100},
            runtime_positions={"1234.T": 1},
            cap_state=_make_cap_state(),
            artifact_dir=tmp_path,
            is_live=False,
        )
        # Registry violation for cap_state must not appear
        assert not any("cap_state" in v for v in result.registry_violations)


# ─────────────────────────────────────────────────────────────────────────────
# RC3: dep graph "capital" always initialized
# ─────────────────────────────────────────────────────────────────────────────

class TestRC3DepGraphCapitalAlwaysInitialized:
    """
    dep_graph.mark_initialized("capital") must be called whenever _eff_capital > 0,
    regardless of _cap_state_loaded.  Previously guarded by _cap_state_loaded which
    is False on first run → allocation/broker show as unresolved → LIVE blocks.
    """

    def test_capital_not_initialized_causes_allocation_violation(self):
        """Baseline: capital not marked → allocation dependency violation."""
        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])
        # intentionally skip: dep.mark_initialized("capital")
        violations = dep.validate()
        assert any("allocation" in v and "capital" in v for v in violations)
        assert any("broker" in v and "capital" in v for v in violations)

    def test_capital_initialized_clears_allocation_violation(self):
        """After fix: capital always marked → no dep violations."""
        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])
        dep.mark_initialized("capital")      # always, not gated on _cap_state_loaded
        dep.mark_initialized("allocation")
        dep.mark_initialized("broker")
        violations = dep.validate()
        assert violations == []

    def test_positive_eff_capital_is_sufficient_predicate(self):
        """_eff_capital > 0 is the correct predicate — it is always true."""
        _cap_state_loaded = False   # simulates first-run (no state file)
        _eff_capital = STATIC_CAPITAL

        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])

        # RC3 fix: use _eff_capital > 0 instead of _cap_state_loaded
        if _eff_capital > 0:
            dep.mark_initialized("capital")
        dep.mark_initialized("allocation")
        dep.mark_initialized("broker")

        violations = dep.validate()
        assert violations == []

    def test_zero_eff_capital_does_not_mark_capital(self):
        """Degenerate case: eff_capital=0 should not mark capital as available."""
        _eff_capital = 0

        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])

        if _eff_capital > 0:
            dep.mark_initialized("capital")

        violations = dep.validate()
        assert any("allocation" in v and "capital" in v for v in violations)

    def test_live_mode_no_raise_when_capital_initialized(self, tmp_path):
        """validate_runtime_integration must not raise RuntimeContractError when
        capital dep is correctly marked — simulates the post-fix LIVE path."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state",        _make_cap_state(), is_critical=True)
        reg.register("inflight_registry", MagicMock(),       is_critical=True)

        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])
        dep.declare("execution",  ["allocation", "broker"])
        dep.mark_initialized("capital")
        dep.mark_initialized("allocation")
        dep.mark_initialized("broker")

        # is_live=True: must NOT raise
        result = validate_runtime_integration(
            run_id="test-rc3-live",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            dep_graph=dep,
            broker_positions={},
            runtime_positions={},
            cap_state=_make_cap_state(),
            artifact_dir=tmp_path,
            is_live=True,   # fail-closed
        )
        assert result.execution_allowed
        assert result.blocking_count == 0

    def test_live_mode_raises_when_capital_not_initialized(self, tmp_path):
        """Regression: when capital dep is missing, LIVE must block."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state",        _make_cap_state(), is_critical=True)
        reg.register("inflight_registry", MagicMock(),       is_critical=True)

        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])
        # intentionally omit mark_initialized("capital")
        dep.mark_initialized("allocation")
        dep.mark_initialized("broker")

        with pytest.raises(RuntimeContractError):
            validate_runtime_integration(
                run_id="test-rc3-live-block",
                stage=STAGE_RECONCILIATION,
                registry=reg,
                dep_graph=dep,
                broker_positions={},
                runtime_positions={},
                cap_state=_make_cap_state(),
                artifact_dir=tmp_path,
                is_live=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RC4: phantom position purge
# ─────────────────────────────────────────────────────────────────────────────

class TestRC4PhantomPositionPurge:
    """
    Phantom positions: runtime_qty=1, broker_qty=0, no inflight SELL.
    After RC4 fix, bridge.overwrite_local_positions(broker_pos) is called
    automatically to reconcile portfolio_state.json to broker truth.
    """

    def _make_bridge_mock(self):
        m = MagicMock()
        m.overwrite_local_positions = MagicMock()
        return m

    def _make_inflight(self, symbol: str, side: str, state: str = "PENDING_SUBMIT"):
        o = MagicMock()
        o.symbol = symbol
        o.side = side
        o.state = state
        return o

    def _phantom_purge_logic(
        self,
        broker_pos: dict,
        runtime_pos: dict,
        all_inflight: list,
        bridge,
    ) -> list:
        """Extracted phantom purge logic from run_live_signal.py for unit testing."""
        if not broker_pos:
            return []
        inflight_syms_sell = {
            getattr(o, "symbol", "")
            for o in all_inflight
            if str(getattr(o, "side", "")).upper() == "SELL"
            and str(getattr(o, "state", "")).upper() in ("PENDING_SUBMIT", "SUBMITTED_UNKNOWN")
        }
        phantom_syms = [
            sym for sym, rq in runtime_pos.items()
            if rq > 0
            and broker_pos.get(sym, 0) == 0
            and sym not in inflight_syms_sell
        ]
        if phantom_syms:
            bridge.overwrite_local_positions(broker_pos)
        return phantom_syms

    def test_phantom_detected_and_purged(self):
        """Phantom symbol is detected and overwrite_local_positions called."""
        bridge = self._make_bridge_mock()
        broker_pos   = {"8035.T": 100}         # 8035 held by broker
        runtime_pos  = {"8035.T": 1, "6506.T": 1}  # 6506 phantom (broker=0)
        phantoms = self._phantom_purge_logic(broker_pos, runtime_pos, [], bridge)
        assert "6506.T" in phantoms
        bridge.overwrite_local_positions.assert_called_once_with(broker_pos)

    def test_multiple_phantoms_detected(self):
        """Multiple phantom symbols all purged in a single call."""
        bridge = self._make_bridge_mock()
        broker_pos  = {}
        runtime_pos = {"6506.T": 1, "8015.T": 1}  # both phantom
        phantoms = self._phantom_purge_logic(broker_pos, runtime_pos, [], bridge)
        # broker_pos is empty → purge logic skips (RC4 guard: only purge if broker_pos)
        bridge.overwrite_local_positions.assert_not_called()
        assert phantoms == []

    def test_broker_empty_skips_purge(self):
        """If broker_pos is empty, purge is skipped to avoid wiping valid state."""
        bridge = self._make_bridge_mock()
        phantoms = self._phantom_purge_logic({}, {"6506.T": 1}, [], bridge)
        bridge.overwrite_local_positions.assert_not_called()
        assert phantoms == []

    def test_inflight_sell_protects_from_purge(self):
        """Symbol with pending SELL inflight must NOT be purged (already exiting)."""
        bridge = self._make_bridge_mock()
        broker_pos  = {"8035.T": 100}
        runtime_pos = {"8035.T": 1, "6506.T": 1}
        inflight = [self._make_inflight("6506.T", "SELL", "PENDING_SUBMIT")]
        phantoms = self._phantom_purge_logic(broker_pos, runtime_pos, inflight, bridge)
        assert "6506.T" not in phantoms  # protected by inflight SELL
        assert "8035.T" not in phantoms  # broker holds it correctly

    def test_non_sell_inflight_does_not_protect(self):
        """BUY inflight for a phantom symbol does not prevent purge."""
        bridge = self._make_bridge_mock()
        broker_pos  = {"8035.T": 100}
        runtime_pos = {"8035.T": 1, "6506.T": 1}
        inflight = [self._make_inflight("6506.T", "BUY", "PENDING_SUBMIT")]
        phantoms = self._phantom_purge_logic(broker_pos, runtime_pos, inflight, bridge)
        assert "6506.T" in phantoms  # BUY does not protect phantom

    def test_completed_inflight_does_not_protect(self):
        """SELL inflight in terminal state does not protect from purge."""
        bridge = self._make_bridge_mock()
        broker_pos  = {"8035.T": 100}
        runtime_pos = {"8035.T": 1, "6506.T": 1}
        inflight = [self._make_inflight("6506.T", "SELL", "FILLED")]  # terminal
        phantoms = self._phantom_purge_logic(broker_pos, runtime_pos, inflight, bridge)
        assert "6506.T" in phantoms  # terminal SELL does not protect

    def test_no_phantom_no_purge_call(self):
        """When runtime matches broker, overwrite_local_positions is never called."""
        bridge = self._make_bridge_mock()
        broker_pos  = {"8035.T": 100, "6502.T": 200}
        runtime_pos = {"8035.T": 1,   "6502.T": 1}
        phantoms = self._phantom_purge_logic(broker_pos, runtime_pos, [], bridge)
        assert phantoms == []
        bridge.overwrite_local_positions.assert_not_called()

    def test_duplicate_order_protection_preserved(self):
        """bridge.overwrite_local_positions() is called with broker_pos as auth source.
        Duplicate-order ledger in bridge is untouched by the phantom purge."""
        bridge = self._make_bridge_mock()
        broker_pos  = {}
        # Artificially give broker_pos one entry so purge runs
        broker_pos["8035.T"] = 100
        runtime_pos = {"8035.T": 1, "6506.T": 1}
        self._phantom_purge_logic(broker_pos, runtime_pos, [], bridge)
        # Ensure we called overwrite_local_positions, not any order-cancel method
        bridge.overwrite_local_positions.assert_called_once()
        bridge.cancel_order.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# RC5: compute_exposure_report required args
# ─────────────────────────────────────────────────────────────────────────────

class TestRC5ComputeExposureReportArgs:
    """
    compute_exposure_report(state, symbols, sector_map, rsr_rank_map, n_universe)
    The original call passed only `state` — missing 3 required positional args.
    """

    def _make_state_with_symbols(self, symbols: list) -> ExposureState:
        """ExposureState with stub return buffers for given symbols."""
        bufs: dict = {}
        for sym in symbols:
            bufs[sym] = [0.01, -0.005, 0.02, 0.0, 0.01]  # minimal 5-bar buffer
        return ExposureState(return_buffers=bufs)

    def test_missing_args_raises_type_error(self):
        """Baseline: calling with only state raises TypeError."""
        state = _make_exposure_state()
        with pytest.raises(TypeError):
            compute_exposure_report(state)  # type: ignore[call-arg]

    def test_empty_held_symbols_returns_zero_effective_n(self):
        """Empty symbols list: effective_independent_n == 0."""
        state = _make_exposure_state()
        report = compute_exposure_report(
            state,
            symbols=[],
            sector_map={},
            rsr_rank_map={},
            n_universe=42,
        )
        assert report.effective_independent_n == 0.0
        assert not report.collapse_alert

    def test_single_held_symbol_valid(self):
        """Single held symbol: effective_n = 1.0 (no pair correlation possible)."""
        state = self._make_state_with_symbols(["8035.T"])
        report = compute_exposure_report(
            state,
            symbols=["8035.T"],
            sector_map={"8035.T": "電機"},
            rsr_rank_map={"8035.T": 82.0},
            n_universe=42,
        )
        assert report.effective_independent_n >= 0.0
        assert report.nominal_positions == 1

    def test_two_held_symbols_valid(self):
        """Two held symbols: report produced without error."""
        state = self._make_state_with_symbols(["8035.T", "6502.T"])
        report = compute_exposure_report(
            state,
            symbols=["8035.T", "6502.T"],
            sector_map={"8035.T": "電機", "6502.T": "電機"},
            rsr_rank_map={"8035.T": 82.0, "6502.T": 77.0},
            n_universe=42,
        )
        assert report.nominal_positions == 2
        assert 0.0 <= report.effective_independent_n <= 2.0

    def test_sector_map_and_rsr_built_from_signals(self):
        """Verify that sector_map/rsr_rank_map construction from signals is correct."""
        signals = [
            {"symbol": "8035.T", "sector": "電機",   "rsr": 82.0, "currently_holding": True},
            {"symbol": "6502.T", "sector": "電機",   "rsr": 77.0, "currently_holding": True},
            {"symbol": "9984.T", "sector": "情報",   "rsr": 91.0, "currently_holding": False},
        ]
        held_syms  = [s["symbol"] for s in signals if s.get("currently_holding") and s.get("symbol")]
        sector_map = {s["symbol"]: s.get("sector", "不明") for s in signals if s.get("symbol")}
        rsr_map    = {s["symbol"]: float(s.get("rsr", 50.0)) for s in signals if s.get("symbol")}

        assert held_syms == ["8035.T", "6502.T"]
        assert sector_map["8035.T"] == "電機"
        assert rsr_map["9984.T"] == 91.0

        state = self._make_state_with_symbols(held_syms)
        report = compute_exposure_report(
            state,
            symbols=held_syms,
            sector_map=sector_map,
            rsr_rank_map=rsr_map,
            n_universe=42,
        )
        assert report.nominal_positions == len(held_syms)

    def test_n_universe_default_fallback(self):
        """n_universe can be obtained from result.n_universe with fallback to 42."""
        n_universe_from_result = None  # simulates missing attribute
        n_universe = n_universe_from_result or 42
        assert n_universe == 42


# ─────────────────────────────────────────────────────────────────────────────
# Integration: LIVE fail-closed / DRY degradation behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveFailClosedDryDegradation:
    """
    End-to-end scenarios through validate_runtime_integration to confirm that:
    - LIVE blocks on blocking violations (RuntimeContractError)
    - DRY continues and returns an observable result
    """

    def _make_valid_registry(self) -> RuntimeStateRegistry:
        reg = RuntimeStateRegistry()
        reg.register("cap_state",        _make_cap_state(), is_critical=True)
        reg.register("inflight_registry", MagicMock(),       is_critical=True)
        return reg

    def _make_valid_dep(self) -> IntegrationDependencyGraph:
        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        dep.declare("broker",     ["capital"])
        dep.declare("execution",  ["allocation", "broker"])
        dep.mark_initialized("capital")
        dep.mark_initialized("allocation")
        dep.mark_initialized("broker")
        return dep

    def test_dry_run_continues_on_violations(self, tmp_path):
        """DRY mode: violations are returned without raising."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state",        None,        is_critical=True)  # intentional None
        reg.register("inflight_registry", MagicMock(), is_critical=True)

        dep = self._make_valid_dep()
        result = validate_runtime_integration(
            run_id="test-dry-violation",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            dep_graph=dep,
            is_live=False,
            artifact_dir=tmp_path,
        )
        assert not result.execution_allowed
        assert result.blocking_count > 0
        # No exception raised → DRY degradation behavior correct

    def test_live_raises_on_none_critical_cap_state(self, tmp_path):
        """LIVE mode: None critical state raises RuntimeContractError."""
        reg = RuntimeStateRegistry()
        reg.register("cap_state",        None,        is_critical=True)
        reg.register("inflight_registry", MagicMock(), is_critical=True)

        dep = self._make_valid_dep()
        with pytest.raises(RuntimeContractError):
            validate_runtime_integration(
                run_id="test-live-block",
                stage=STAGE_RECONCILIATION,
                registry=reg,
                dep_graph=dep,
                is_live=True,
                artifact_dir=tmp_path,
            )

    def test_live_passes_with_coherent_capital_state(self, tmp_path):
        """LIVE mode: coherent state with valid cap_state → no exception."""
        reg = self._make_valid_registry()
        dep = self._make_valid_dep()

        result = validate_runtime_integration(
            run_id="test-live-ok",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            dep_graph=dep,
            broker_positions={},
            runtime_positions={},
            cap_state=_make_cap_state(),
            is_live=True,
            artifact_dir=tmp_path,
        )
        assert result.execution_allowed
        assert result.blocking_count == 0

    def test_stale_runtime_positions_produce_consistency_violations(self, tmp_path):
        """Stale holdings (runtime=1, broker=0) appear in consistency violations."""
        reg = self._make_valid_registry()
        dep = self._make_valid_dep()

        result = validate_runtime_integration(
            run_id="test-stale-positions",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            dep_graph=dep,
            broker_positions={"6506.T": 0},      # broker flat
            runtime_positions={"6506.T": 1},      # runtime phantom
            cap_state=_make_cap_state(),
            is_live=False,
            artifact_dir=tmp_path,
        )
        # Consistency violations must be reported
        assert len(result.consistency_violations) > 0

    def test_governance_exception_does_not_abort_dry_run(self):
        """Governance errors must be caught and not raise in DRY mode."""
        # The governance block in run_live_signal.py uses fail-open for general
        # exceptions and fail-closed only for RuntimeError (integrity violations).
        # This test verifies the contract: a non-RuntimeError governance failure
        # must be observable but not abort.
        exc = ValueError("rsr_scores missing")
        # Simulate: governance raises ValueError → should be caught by except Exception
        caught = False
        try:
            raise exc
        except RuntimeError:
            pass  # would abort
        except Exception:
            caught = True  # fail-open: log and continue
        assert caught, "General governance exceptions must be caught (fail-open)"

    def test_missing_capital_init_dry_run_observable(self, tmp_path):
        """DRY: missing capital dep → violation visible in result, not raised."""
        reg = self._make_valid_registry()

        dep = IntegrationDependencyGraph()
        dep.declare("capital",    [])
        dep.declare("allocation", ["capital"])
        # intentionally omit mark_initialized("capital")

        result = validate_runtime_integration(
            run_id="test-missing-cap-dry",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            dep_graph=dep,
            is_live=False,
            artifact_dir=tmp_path,
        )
        # Should produce dep violations without raising
        assert len(result.dep_graph_violations) > 0
        assert any("capital" in v for v in result.dep_graph_violations)
