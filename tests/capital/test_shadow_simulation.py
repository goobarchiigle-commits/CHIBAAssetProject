"""Tests for shadow_simulation.py"""
import json
import pytest

from src.capital.shadow_simulation import (
    ShadowTierResult,
    ShadowSimulationRecord,
    simulate_tier,
    run_shadow_simulation,
    append_shadow_record,
)


def _signals():
    return [
        {"symbol": "7203.T", "weight": 0.25, "price": 3_000.0, "adv_yen": 10_000_000_000},
        {"symbol": "9984.T", "weight": 0.25, "price": 5_000.0, "adv_yen": 50_000_000_000},
        {"symbol": "8306.T", "weight": 0.25, "price": 1_200.0, "adv_yen": 5_000_000_000},
    ]


class TestSimulateTier:
    def test_returns_shadow_tier_result(self):
        result = simulate_tier("1x", 1.0, 4_000_000, _signals())
        assert isinstance(result, ShadowTierResult)
        assert result.tier == "1x"
        assert result.capital == 4_000_000

    def test_higher_capital_more_deployed(self):
        r1x = simulate_tier("1x", 1.0, 4_000_000, _signals())
        r5x = simulate_tier("5x", 5.0, 4_000_000, _signals())
        assert r5x.total_order_value >= r1x.total_order_value

    def test_higher_capital_higher_impact(self):
        r1x = simulate_tier("1x", 1.0, 4_000_000, _signals())
        r5x = simulate_tier("5x", 5.0, 4_000_000, _signals())
        assert r5x.slippage_bps >= r1x.slippage_bps

    def test_fill_ratio_between_0_and_1(self):
        result = simulate_tier("2x", 2.0, 4_000_000, _signals())
        assert 0.0 <= result.fill_ratio_estimate <= 1.0

    def test_empty_signals_returns_zero_values(self):
        result = simulate_tier("1x", 1.0, 4_000_000, [])
        assert result.total_order_value == 0.0
        assert result.deployability_score == 0.0

    def test_zero_price_signal_skipped(self):
        signals = [{"symbol": "BAD.T", "weight": 0.25, "price": 0.0, "adv_yen": 1_000_000}]
        result = simulate_tier("1x", 1.0, 4_000_000, signals)
        assert result.total_order_value == 0.0

    def test_scores_in_valid_range(self):
        result = simulate_tier("1x", 1.0, 4_000_000, _signals())
        assert 0.0 <= result.deployability_score <= 1.0
        assert 0.0 <= result.liquidity_saturation_score <= 1.0
        assert result.participation_rate >= 0.0
        assert result.slippage_bps >= 0.0


class TestRunShadowSimulation:
    def test_returns_all_tiers(self):
        record = run_shadow_simulation(4_000_000, _signals(), tiers=[1, 2, 5])
        assert len(record.tiers) == 3
        tier_labels = {t["tier"] for t in record.tiers}
        assert tier_labels == {"1x", "2x", "5x"}

    def test_base_capital_recorded(self):
        record = run_shadow_simulation(4_000_000, _signals())
        assert record.base_capital == 4_000_000

    def test_trading_day_set(self):
        record = run_shadow_simulation(4_000_000, _signals(), trading_day="2026-05-16")
        assert record.trading_day == "2026-05-16"

    def test_ts_present(self):
        record = run_shadow_simulation(4_000_000, _signals())
        assert record.ts is not None
        assert len(record.ts) > 0

    def test_no_live_path_impact(self):
        # Multiple calls with same inputs must produce same tier results
        # (deterministic, no state mutation)
        r1 = run_shadow_simulation(4_000_000, _signals(), trading_day="2026-05-16")
        r2 = run_shadow_simulation(4_000_000, _signals(), trading_day="2026-05-16")
        for t1, t2 in zip(r1.tiers, r2.tiers):
            assert t1["total_order_value"] == t2["total_order_value"]
            assert t1["slippage_bps"] == t2["slippage_bps"]


class TestAppendShadowRecord:
    def test_appends_jsonl(self, tmp_path):
        path = tmp_path / "shadow.jsonl"
        record = run_shadow_simulation(4_000_000, _signals())
        append_shadow_record(record, path)
        append_shadow_record(record, path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_valid_json_per_line(self, tmp_path):
        path = tmp_path / "shadow.jsonl"
        record = run_shadow_simulation(4_000_000, _signals())
        append_shadow_record(record, path)
        line = path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert "base_capital" in data
        assert "tiers" in data

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "subdir" / "shadow.jsonl"
        record = run_shadow_simulation(4_000_000, _signals())
        append_shadow_record(record, path)
        assert path.exists()
