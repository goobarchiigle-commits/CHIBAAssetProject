"""Tests for telemetry.py"""
import json
import pytest

from src.capital.telemetry import (
    build_telemetry_entry,
    append_telemetry,
    TELEMETRY_SCHEMA_VERSION,
)


def _entry(**kwargs):
    defaults = dict(
        actual_equity=4_000_000,
        effective_capital=3_150_000,
        deployable_capital=2_835_000,
        risk_adjusted_capital=2_551_500,
        capital_growth_rate=0.015,
        deployed_value=2_000_000,
        participation_rate=0.023,
        fill_ratio=0.95,
        partial_fill_ratio=0.05,
        slippage_bps=8.5,
        reject_rate=0.0,
        execution_latency_ms=120,
        liquidity_stress_score=0.15,
        shadow_capacity_metrics={"1x": {}, "2x": {}, "5x": {}},
        deployment_ramp_state="RAMPING",
        capital_freeze_state="NORMAL",
        trading_day="2026-05-16",
    )
    defaults.update(kwargs)
    return build_telemetry_entry(**defaults)


class TestBuildTelemetryEntry:
    def test_schema_version_set(self):
        entry = _entry()
        assert entry.schema_version == TELEMETRY_SCHEMA_VERSION

    def test_utilization_computed(self):
        entry = _entry(deployed_value=2_000_000, risk_adjusted_capital=2_551_500)
        expected = 2_000_000 / 2_551_500
        assert entry.capital_utilization == pytest.approx(expected, abs=0.001)

    def test_utilization_capped_at_1_5(self):
        # deployed > capital → utilization can exceed 1.0 for diagnostics but capped at 1.5
        entry = _entry(deployed_value=10_000_000, risk_adjusted_capital=2_551_500)
        assert entry.capital_utilization <= 1.5

    def test_all_capital_fields_present(self):
        entry = _entry()
        assert entry.actual_equity == 4_000_000
        assert entry.effective_capital == 3_150_000
        assert entry.deployable_capital == 2_835_000
        assert entry.risk_adjusted_capital == 2_551_500

    def test_freeze_reason_optional(self):
        entry = _entry(capital_freeze_state="NORMAL", capital_freeze_reason=None)
        assert entry.capital_freeze_reason is None

    def test_freeze_reason_populated(self):
        entry = _entry(
            capital_freeze_state="FROZEN",
            capital_freeze_reason="slippage_bps=55",
        )
        assert entry.capital_freeze_reason == "slippage_bps=55"

    def test_trading_day_fallback(self):
        # Not providing trading_day → uses today
        entry = build_telemetry_entry(
            actual_equity=3_000_000, effective_capital=3_000_000,
            deployable_capital=2_700_000, risk_adjusted_capital=2_700_000,
            capital_growth_rate=0.0, deployed_value=0,
            participation_rate=0, fill_ratio=1.0,
            partial_fill_ratio=0, slippage_bps=5.0,
            reject_rate=0, execution_latency_ms=100,
            liquidity_stress_score=0.1, shadow_capacity_metrics={},
            deployment_ramp_state="RAMPING", capital_freeze_state="NORMAL",
        )
        assert entry.trading_day is not None


class TestAppendTelemetry:
    def test_creates_jsonl_file(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        entry = _entry()
        append_telemetry(entry, path)
        assert path.exists()

    def test_append_multiple_lines(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        for _ in range(3):
            append_telemetry(_entry(), path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_valid_json_per_line(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        entry = _entry()
        append_telemetry(entry, path)
        line = path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["schema_version"] == TELEMETRY_SCHEMA_VERSION
        assert "actual_equity" in data
        assert "effective_capital" in data
        assert "shadow_capacity_metrics" in data

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "telemetry.jsonl"
        append_telemetry(_entry(), path)
        assert path.exists()

    def test_append_only_not_overwrite(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        entry1 = _entry(actual_equity=3_000_000)
        entry2 = _entry(actual_equity=4_000_000)
        append_telemetry(entry1, path)
        append_telemetry(entry2, path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["actual_equity"] == 3_000_000
        assert data2["actual_equity"] == 4_000_000
