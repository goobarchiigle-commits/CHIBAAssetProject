"""Tests for deployment_ramp.py"""
import pytest

from src.capital.deployment_ramp import (
    DeploymentRampState,
    ExecutionQualityRecord,
    compute_quality_score,
    update_deployment_ramp,
    load_deployment_ramp,
    save_deployment_ramp,
)


def _good_record(day: str = "2026-05-16") -> ExecutionQualityRecord:
    return ExecutionQualityRecord(
        fill_ratio=0.97,
        partial_fill_ratio=0.05,
        slippage_bps=8.0,
        reject_rate=0.0,
        execution_latency_ms=120,
        trading_day=day,
    )


def _poor_record(day: str = "2026-05-16") -> ExecutionQualityRecord:
    return ExecutionQualityRecord(
        fill_ratio=0.55,
        partial_fill_ratio=0.40,
        slippage_bps=80.0,
        reject_rate=0.35,
        execution_latency_ms=1800,
        trading_day=day,
    )


class TestComputeQualityScore:
    def test_perfect_execution(self):
        record = ExecutionQualityRecord(
            fill_ratio=1.0, partial_fill_ratio=0.0,
            slippage_bps=0.0, reject_rate=0.0,
            execution_latency_ms=0.0, trading_day="2026-01-01",
        )
        score = compute_quality_score(record)
        assert score == pytest.approx(1.0)

    def test_high_slippage_lowers_score(self):
        record = ExecutionQualityRecord(
            fill_ratio=1.0, partial_fill_ratio=0.0,
            slippage_bps=100.0, reject_rate=0.0,
            execution_latency_ms=0.0, trading_day="2026-01-01",
        )
        score = compute_quality_score(record)
        # slippage=100bps → slippage_norm=0. Weight=0.30 → max possible ~0.70
        assert score < 0.80

    def test_high_reject_rate_lowers_score(self):
        record = ExecutionQualityRecord(
            fill_ratio=1.0, partial_fill_ratio=0.0,
            slippage_bps=0.0, reject_rate=0.40,
            execution_latency_ms=0.0, trading_day="2026-01-01",
        )
        score = compute_quality_score(record)
        assert score < 0.85

    def test_poor_execution_below_freeze_threshold(self):
        score = compute_quality_score(_poor_record())
        assert score < 0.60

    def test_good_execution_above_ramp_threshold(self):
        score = compute_quality_score(_good_record())
        assert score >= 0.80

    def test_score_in_range(self):
        for rec in [_good_record(), _poor_record()]:
            s = compute_quality_score(rec)
            assert 0.0 <= s <= 1.0


class TestUpdateDeploymentRamp:
    def test_good_exec_stays_ramping(self):
        ramp = DeploymentRampState(mode="RAMPING")
        new_ramp, modifier = update_deployment_ramp(ramp, _good_record())
        assert new_ramp.mode == "RAMPING"
        assert modifier == 1.0

    def test_poor_exec_freezes(self):
        ramp = DeploymentRampState(mode="RAMPING")
        new_ramp, modifier = update_deployment_ramp(ramp, _poor_record())
        assert new_ramp.mode == "FROZEN"
        assert modifier == 0.0

    def test_recovery_requires_n_good_execs(self):
        # Start FROZEN, feed good records, should reach RECOVERING after 5
        ramp = DeploymentRampState(mode="FROZEN", consecutive_good=0)
        for i in range(4):
            ramp, modifier = update_deployment_ramp(ramp, _good_record(), recovery_window=5)
            assert ramp.mode == "FROZEN"  # still frozen
        ramp, modifier = update_deployment_ramp(ramp, _good_record(), recovery_window=5)
        assert ramp.mode == "RECOVERING"
        assert modifier == 0.5

    def test_frozen_poor_resets_consecutive_good(self):
        ramp = DeploymentRampState(mode="FROZEN", consecutive_good=3)
        new_ramp, _ = update_deployment_ramp(ramp, _poor_record())
        assert new_ramp.consecutive_good == 0

    def test_recovering_good_becomes_ramping(self):
        ramp = DeploymentRampState(mode="RECOVERING", consecutive_good=5)
        new_ramp, modifier = update_deployment_ramp(ramp, _good_record())
        assert new_ramp.mode == "RAMPING"
        assert modifier == 1.0

    def test_recovering_poor_refreezes(self):
        ramp = DeploymentRampState(mode="RECOVERING", consecutive_good=5)
        new_ramp, modifier = update_deployment_ramp(ramp, _poor_record())
        assert new_ramp.mode == "FROZEN"
        assert modifier == 0.0

    def test_history_rolling_window(self):
        ramp = DeploymentRampState(mode="RAMPING")
        for i in range(25):  # feed 25 records with window=20
            ramp, _ = update_deployment_ramp(ramp, _good_record(), window=20)
        assert len(ramp.quality_history) <= 20

    def test_history_contains_quality_score(self):
        ramp = DeploymentRampState(mode="RAMPING")
        ramp, _ = update_deployment_ramp(ramp, _good_record())
        assert "quality_score" in ramp.quality_history[-1]


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        ramp = DeploymentRampState(mode="FROZEN", consecutive_good=3)
        path = tmp_path / "ramp.json"
        save_deployment_ramp(ramp, path)
        loaded = load_deployment_ramp(path)
        assert loaded.mode == "FROZEN"
        assert loaded.consecutive_good == 3

    def test_load_missing_returns_ramping(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = load_deployment_ramp(path)
        assert loaded.mode == "RAMPING"

    def test_load_corrupt_returns_ramping(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        loaded = load_deployment_ramp(path)
        assert loaded.mode == "RAMPING"
