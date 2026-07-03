"""Tests for capital_freeze.py"""
import pytest

from src.capital.capital_freeze import (
    FreezeConfig,
    FreezeState,
    evaluate_freeze,
    update_freeze_state,
    load_freeze_state,
    save_freeze_state,
)


def _healthy():
    return dict(
        slippage_bps=8.0,
        reject_rate=0.0,
        fill_ratio=0.97,
        liquidity_stress_score=0.15,
        partial_fill_ratio=0.05,
    )


class TestEvaluateFreeze:
    def test_healthy_metrics_no_freeze(self):
        frozen, reason = evaluate_freeze(**_healthy())
        assert not frozen
        assert reason is None

    def test_high_slippage_triggers(self):
        params = {**_healthy(), "slippage_bps": 55.0}
        frozen, reason = evaluate_freeze(**params)
        assert frozen
        assert "slippage_bps" in reason

    def test_high_reject_rate_triggers(self):
        params = {**_healthy(), "reject_rate": 0.35}
        frozen, reason = evaluate_freeze(**params)
        assert frozen
        assert "reject_rate" in reason

    def test_low_fill_ratio_triggers(self):
        params = {**_healthy(), "fill_ratio": 0.65}
        frozen, reason = evaluate_freeze(**params)
        assert frozen
        assert "fill_ratio" in reason

    def test_high_liquidity_stress_triggers(self):
        params = {**_healthy(), "liquidity_stress_score": 0.75}
        frozen, reason = evaluate_freeze(**params)
        assert frozen
        assert "liquidity_stress" in reason

    def test_high_partial_fill_triggers(self):
        params = {**_healthy(), "partial_fill_ratio": 0.55}
        frozen, reason = evaluate_freeze(**params)
        assert frozen
        assert "partial_fill" in reason

    def test_single_breach_suffices(self):
        # OR logic: only slippage bad, rest healthy
        params = {**_healthy(), "slippage_bps": 60.0}
        frozen, _ = evaluate_freeze(**params)
        assert frozen

    def test_custom_config(self):
        cfg = FreezeConfig(slippage_bps_max=100.0)
        frozen, _ = evaluate_freeze(
            slippage_bps=80.0, reject_rate=0.0,
            fill_ratio=0.95, liquidity_stress_score=0.10, cfg=cfg,
        )
        assert not frozen

    def test_exactly_at_threshold_no_freeze(self):
        # At exact limit: not strictly greater → no freeze
        cfg = FreezeConfig(slippage_bps_max=50.0)
        frozen, _ = evaluate_freeze(
            slippage_bps=50.0, reject_rate=0.0,
            fill_ratio=0.95, liquidity_stress_score=0.10, cfg=cfg,
        )
        assert not frozen


class TestUpdateFreezeState:
    def test_unfrozen_stays_unfrozen_on_healthy(self):
        state = FreezeState()
        new_state = update_freeze_state(state, **_healthy())
        assert not new_state.is_frozen

    def test_freeze_on_trigger(self):
        state = FreezeState()
        params = {**_healthy(), "slippage_bps": 55.0}
        new_state = update_freeze_state(state, **params)
        assert new_state.is_frozen
        assert new_state.trigger_count == 1

    def test_unfreeze_on_recovery(self):
        state = FreezeState(is_frozen=True, trigger_count=2)
        new_state = update_freeze_state(state, **_healthy())
        assert not new_state.is_frozen
        assert new_state.trigger_count == 2  # trigger_count not reset

    def test_trigger_count_increments_each_freeze(self):
        state = FreezeState()
        for _ in range(3):
            params = {**_healthy(), "slippage_bps": 55.0}
            state = update_freeze_state(state, **params)
        assert state.trigger_count == 3


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        state = FreezeState(is_frozen=True, freeze_reason="slippage_bps=55", trigger_count=2)
        path = tmp_path / "freeze.json"
        save_freeze_state(state, path)
        loaded = load_freeze_state(path)
        assert loaded.is_frozen is True
        assert loaded.freeze_reason == "slippage_bps=55"
        assert loaded.trigger_count == 2

    def test_load_missing_returns_unfrozen(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = load_freeze_state(path)
        assert not loaded.is_frozen

    def test_load_corrupt_returns_unfrozen(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid", encoding="utf-8")
        loaded = load_freeze_state(path)
        assert not loaded.is_frozen
