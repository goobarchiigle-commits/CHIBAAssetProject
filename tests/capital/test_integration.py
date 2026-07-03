"""
Integration tests: end-to-end capital governance pipeline.

Simulates capital injection scenario:
  - Start: 3M actual_equity
  - Day 0: inject 1M (actual_equity → 4M)
  - Over N days: effective_capital ramps to 4M
  - Execute trades with varying quality: verify freeze / unfreeze behavior
  - Verify shadow simulation runs in parallel
  - Verify telemetry accumulates
"""
import json
import pytest
from pathlib import Path

from src.capital.capital_state import (
    CapitalState,
    update_capital_state,
    load_capital_state,
    save_capital_state,
)
from src.capital.deployment_ramp import (
    DeploymentRampState,
    ExecutionQualityRecord,
    update_deployment_ramp,
)
from src.capital.capital_freeze import (
    FreezeState,
    FreezeConfig,
    update_freeze_state,
)
from src.capital.risk_scalars import (
    compute_volatility_scalar,
    compute_execution_scalar,
)
from src.capital.admission_control import filter_universe_by_capital
from src.capital.shadow_simulation import run_shadow_simulation, append_shadow_record
from src.capital.telemetry import build_telemetry_entry, append_telemetry


def _good_exec_record(day: str) -> ExecutionQualityRecord:
    return ExecutionQualityRecord(
        fill_ratio=0.97, partial_fill_ratio=0.03,
        slippage_bps=7.0, reject_rate=0.0,
        execution_latency_ms=100, trading_day=day,
    )


def _poor_exec_record(day: str) -> ExecutionQualityRecord:
    return ExecutionQualityRecord(
        fill_ratio=0.60, partial_fill_ratio=0.45,
        slippage_bps=65.0, reject_rate=0.40,
        execution_latency_ms=1500, trading_day=day,
    )


class TestCapitalRamp:
    def test_1m_injection_ramps_over_time(self):
        """1M injection: effective_capital should reach 4M after sufficient days."""
        state = CapitalState.initial(3_000_000)
        actual = 4_000_000.0

        for day in range(100):
            state = update_capital_state(state, actual)

        assert state.effective_capital == pytest.approx(4_000_000, rel=0.001)

    def test_effective_never_exceeds_actual(self):
        state = CapitalState.initial(3_000_000)
        for _ in range(5):
            state = update_capital_state(state, 4_000_000)
        assert state.effective_capital <= state.actual_equity

    def test_drawdown_reduces_both_effective_and_risk_adjusted(self):
        state = CapitalState.initial(3_000_000)
        # Simulate 10% loss
        state = update_capital_state(state, 2_700_000)
        assert state.effective_capital == pytest.approx(2_700_000)
        assert state.risk_adjusted_capital < 2_700_000

    def test_deposit_during_freeze(self):
        state = CapitalState.initial(3_000_000)
        # Inject 1M but with growth frozen
        state_frozen = update_capital_state(state, 4_000_000, growth_frozen=True)
        assert state_frozen.effective_capital == pytest.approx(3_000_000)
        # Drawdown still propagates
        state_down = update_capital_state(state, 2_500_000, growth_frozen=True)
        assert state_down.effective_capital == pytest.approx(2_500_000)


class TestFreezeRampCycle:
    def test_freeze_suspends_ramp(self):
        """Poor execution → freeze → growth stops → recovery → growth resumes."""
        state = CapitalState.initial(3_000_000)
        ramp = DeploymentRampState(mode="RAMPING")
        freeze = FreezeState()
        actual = 4_000_000.0

        # Day 0: inject 1M
        state = update_capital_state(state, actual)
        initial_eff = state.effective_capital

        # Poor execution → freeze
        ramp, modifier = update_deployment_ramp(ramp, _poor_exec_record("2026-05-16"))
        assert ramp.mode == "FROZEN"
        assert modifier == 0.0

        freeze = update_freeze_state(
            freeze, slippage_bps=65.0, reject_rate=0.40,
            fill_ratio=0.60, liquidity_stress_score=0.20,
        )
        assert freeze.is_frozen

        # Day 1: frozen → effective_capital does not grow
        state_frozen = update_capital_state(
            state, actual, growth_frozen=freeze.is_frozen
        )
        assert state_frozen.effective_capital == pytest.approx(initial_eff, rel=0.001)

    def test_recovery_resumes_growth(self):
        ramp = DeploymentRampState(mode="FROZEN", consecutive_good=0)

        # Feed 5 good records → RECOVERING
        for i in range(5):
            ramp, modifier = update_deployment_ramp(ramp, _good_exec_record(f"2026-05-{17+i:02d}"))
        assert ramp.mode == "RECOVERING"
        assert modifier == 0.5

        # One more good → RAMPING
        ramp, modifier = update_deployment_ramp(ramp, _good_exec_record("2026-05-22"))
        assert ramp.mode == "RAMPING"
        assert modifier == 1.0


class TestAdmissionProgressionWithCapital:
    def test_semiconductor_unlocked_at_higher_capital(self):
        # At 3M: effective_max = min(3M*0.90*0.30, 3M*0.25) = 750K → max ¥7,500/share
        # At 10M: effective_max = min(10M*0.90*0.30, 10M*0.25) = min(2.7M, 2.5M) = 2.5M → max ¥25,000/share
        tickers = {
            "6857.T": "電機精密",  # ¥7,000/share: affordable at 3M
            "8035.T": "電機精密",  # ¥25,000/share: needs ~10M
        }
        prices = {"6857.T": 7_000.0, "8035.T": 25_000.0}

        f3m, r3m = filter_universe_by_capital(tickers, prices, 3_000_000, set())
        f10m, r10m = filter_universe_by_capital(tickers, prices, 10_000_000, set())

        assert "6857.T" in f3m   # ¥7K/share accessible at 3M (¥700K < ¥750K)
        assert "8035.T" not in f3m  # ¥25K/share not accessible at 3M
        assert "8035.T" in f10m  # ¥25K/share accessible at 10M (¥2.5M = ¥2.5M)


class TestShadowSimulationAndTelemetry:
    def test_shadow_and_telemetry_no_live_impact(self, tmp_path):
        shadow_path = tmp_path / "shadow.jsonl"
        telem_path = tmp_path / "telemetry.jsonl"

        signals = [
            {"symbol": "7203.T", "weight": 0.25, "price": 3_000.0, "adv_yen": 10_000_000_000},
            {"symbol": "9984.T", "weight": 0.25, "price": 5_000.0, "adv_yen": 50_000_000_000},
        ]

        # Run shadow simulation
        record = run_shadow_simulation(3_000_000, signals, tiers=[1, 2, 5])
        append_shadow_record(record, shadow_path)

        # Build telemetry entry
        shadow_metrics = {t["tier"]: t for t in record.tiers}
        entry = build_telemetry_entry(
            actual_equity=4_000_000, effective_capital=3_050_000,
            deployable_capital=2_745_000, risk_adjusted_capital=2_470_500,
            capital_growth_rate=0.015, deployed_value=1_800_000,
            participation_rate=0.018, fill_ratio=0.96, partial_fill_ratio=0.04,
            slippage_bps=8.2, reject_rate=0.0, execution_latency_ms=115,
            liquidity_stress_score=0.12, shadow_capacity_metrics=shadow_metrics,
            deployment_ramp_state="RAMPING", capital_freeze_state="NORMAL",
            trading_day="2026-05-16",
        )
        append_telemetry(entry, telem_path)

        # Verify outputs
        shadow_lines = shadow_path.read_text(encoding="utf-8").strip().split("\n")
        telem_lines = telem_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(shadow_lines) == 1
        assert len(telem_lines) == 1

        shadow_data = json.loads(shadow_lines[0])
        telem_data = json.loads(telem_lines[0])

        assert shadow_data["base_capital"] == 3_000_000
        assert len(shadow_data["tiers"]) == 3
        assert telem_data["deployment_ramp_state"] == "RAMPING"
        assert "shadow_capacity_metrics" in telem_data

    def test_telemetry_accumulates_across_days(self, tmp_path):
        telem_path = tmp_path / "telemetry.jsonl"

        for day in range(5):
            entry = build_telemetry_entry(
                actual_equity=3_000_000 + day * 50_000,
                effective_capital=3_000_000 + day * 45_000,
                deployable_capital=2_700_000, risk_adjusted_capital=2_700_000,
                capital_growth_rate=0.015, deployed_value=0,
                participation_rate=0, fill_ratio=1.0, partial_fill_ratio=0,
                slippage_bps=5.0, reject_rate=0, execution_latency_ms=100,
                liquidity_stress_score=0.1, shadow_capacity_metrics={},
                deployment_ramp_state="RAMPING", capital_freeze_state="NORMAL",
                trading_day=f"2026-05-{16+day:02d}",
            )
            append_telemetry(entry, telem_path)

        lines = telem_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_state_persistence_roundtrip(self, tmp_path):
        state = CapitalState.initial(3_000_000)
        state = update_capital_state(state, 4_000_000)  # inject 1M
        path = tmp_path / "capital_state.json"
        save_capital_state(state, path)
        loaded = load_capital_state(path)
        assert loaded.effective_capital == pytest.approx(state.effective_capital, rel=1e-6)
        assert loaded.actual_equity == 4_000_000
