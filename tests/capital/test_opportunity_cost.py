"""Tests for src/capital/opportunity_cost.py"""
import json
import pytest
from pathlib import Path
from src.capital.opportunity_cost import (
    compute_opportunity_cost_bps,
    build_opportunity_cost_record,
    update_opportunity_cost_state,
    append_opportunity_cost_record,
    load_opportunity_cost_state,
    OpportunityCostState,
)


class TestComputeOpportunityCostBps:
    def test_fully_idle(self):
        cost = compute_opportunity_cost_bps(expected_net_alpha_bps=10.0, idle_fraction=1.0)
        assert cost == pytest.approx(10.0)

    def test_fully_deployed(self):
        cost = compute_opportunity_cost_bps(10.0, 0.0)
        assert cost == pytest.approx(0.0)

    def test_half_idle(self):
        cost = compute_opportunity_cost_bps(10.0, 0.5)
        assert cost == pytest.approx(5.0)

    def test_negative_alpha_yields_zero(self):
        cost = compute_opportunity_cost_bps(-5.0, 1.0)
        assert cost == pytest.approx(0.0)

    def test_idle_fraction_clamped(self):
        cost = compute_opportunity_cost_bps(10.0, 2.0)  # > 1.0
        assert cost == pytest.approx(10.0)  # clamped to 1.0


class TestBuildOpportunityCostRecord:
    def test_fully_deployed_zero_cost(self):
        rec = build_opportunity_cost_record(
            trading_day="2026-01-01",
            deployment_state="RAMPING",
            deployable_capital=2_700_000.0,
            effective_capital=3_000_000.0,
            actual_deployed=2_700_000.0,  # fully deployed
            expected_net_alpha_bps=10.0,
            cumulative_cost_bps=0.0,
        )
        assert rec.opportunity_cost_bps == pytest.approx(0.0, abs=0.5)
        assert "fully deployed" in rec.reason

    def test_frozen_state_full_cost(self):
        rec = build_opportunity_cost_record(
            trading_day="2026-01-01",
            deployment_state="FROZEN",
            deployable_capital=2_700_000.0,
            effective_capital=3_000_000.0,
            actual_deployed=0.0,
            expected_net_alpha_bps=10.0,
            cumulative_cost_bps=0.0,
        )
        assert rec.opportunity_cost_bps > 0.0
        assert rec.idle_fraction > 0.0
        assert "execution quality freeze" in rec.reason

    def test_cumulative_accumulates(self):
        rec = build_opportunity_cost_record(
            "2026-01-02", "FROZEN", 2_700_000.0, 3_000_000.0,
            0.0, 10.0, cumulative_cost_bps=25.0
        )
        assert rec.cumulative_cost_bps > 25.0

    def test_effective_capital_zero_no_crash(self):
        rec = build_opportunity_cost_record(
            "2026-01-01", "FROZEN", 0.0, 0.0, 0.0, 10.0, 0.0
        )
        assert rec.idle_fraction == pytest.approx(0.0)


class TestUpdateOpportunityCostState:
    def test_increments_idle_days(self):
        state = OpportunityCostState(total_idle_days=3)
        rec = build_opportunity_cost_record(
            "2026-01-01", "FROZEN", 2_700_000.0, 3_000_000.0,
            0.0, 10.0, 0.0
        )
        new = update_opportunity_cost_state(state, rec)
        assert new.total_idle_days == 4

    def test_no_idle_day_when_deployed(self):
        state = OpportunityCostState(total_idle_days=3)
        rec = build_opportunity_cost_record(
            "2026-01-01", "RAMPING", 2_700_000.0, 3_000_000.0,
            2_700_000.0, 10.0, 0.0
        )
        new = update_opportunity_cost_state(state, rec)
        assert new.total_idle_days == 3  # no increment when deployed


class TestAppendAndLoad:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "opp_cost.jsonl"
        rec = build_opportunity_cost_record(
            "2026-01-01", "FROZEN", 2_700_000.0, 3_000_000.0,
            0.0, 10.0, 0.0
        )
        append_opportunity_cost_record(rec, path)
        append_opportunity_cost_record(rec, path)

        state = load_opportunity_cost_state(path)
        assert state.total_idle_days == 2  # two FROZEN records (idle)

    def test_load_nonexistent(self, tmp_path):
        state = load_opportunity_cost_state(tmp_path / "nonexistent.jsonl")
        assert state.cumulative_cost_bps == pytest.approx(0.0)

    def test_append_only(self, tmp_path):
        path = tmp_path / "opp_cost.jsonl"
        rec = build_opportunity_cost_record(
            "2026-01-01", "FROZEN", 2_700_000.0, 3_000_000.0,
            0.0, 10.0, 0.0
        )
        append_opportunity_cost_record(rec, path)
        lines_before = len(path.read_text().strip().split("\n"))
        append_opportunity_cost_record(rec, path)
        lines_after = len(path.read_text().strip().split("\n"))
        assert lines_after == lines_before + 1
