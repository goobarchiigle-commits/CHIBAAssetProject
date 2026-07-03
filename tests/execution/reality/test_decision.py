"""Tests: decision.py — DecisionSnapshot, DeploymentGuard."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.decision import (
    DecisionSnapshot, DeploymentGuard, DeploymentThresholds,
    DECISION_ALLOW, DECISION_REDUCE, DECISION_HALT, ALL_DECISIONS,
)
from tests.execution.reality.conftest import SESSION, TS, TS2, TS3, make_guard


def evaluate(guard, regime="NORMAL", slip=2.0, reject=0.05, decay=0.20, ts=TS):
    return guard.evaluate(SESSION, regime, slip, reject, decay, ts)


class TestDecisionSnapshot:
    def test_frozen(self):
        guard = make_guard()
        snap = evaluate(guard)
        with pytest.raises(Exception):
            snap.decision = "CHANGED"  # type: ignore

    def test_decision_id_prefix(self):
        guard = make_guard()
        snap = evaluate(guard)
        assert snap.decision_id.startswith("dec_")

    def test_context_hash_16_chars(self):
        guard = make_guard()
        snap = evaluate(guard)
        assert len(snap.decision_context_hash) == 16

    def test_is_blocking_halt(self):
        guard = make_guard()
        snap = evaluate(guard, slip=999.0)
        assert snap.is_blocking()

    def test_is_blocking_allow(self):
        guard = make_guard()
        snap = evaluate(guard)
        assert not snap.is_blocking()

    def test_to_dict_from_dict_roundtrip(self):
        guard = make_guard()
        snap = evaluate(guard)
        restored = DecisionSnapshot.from_dict(snap.to_dict())
        assert restored.decision_id == snap.decision_id
        assert restored.decision == snap.decision


class TestDeploymentGuard:
    def test_all_decisions(self):
        assert len(ALL_DECISIONS) == 3

    def test_allow_under_all_thresholds(self):
        guard = make_guard()
        snap = evaluate(guard, slip=5.0, reject=0.05, decay=0.20)
        assert snap.decision == DECISION_ALLOW

    def test_reduce_on_slippage(self):
        guard = DeploymentGuard(DeploymentThresholds(max_slippage_bps=10.0, halt_slippage_bps=50.0))
        snap = evaluate(guard, slip=15.0)
        assert snap.decision == DECISION_REDUCE

    def test_reduce_on_reject_rate(self):
        guard = DeploymentGuard(DeploymentThresholds(max_reject_rate=0.10, halt_reject_rate=0.40))
        snap = evaluate(guard, reject=0.15)
        assert snap.decision == DECISION_REDUCE

    def test_reduce_on_decay(self):
        guard = DeploymentGuard(DeploymentThresholds(max_decay_fraction=0.40, halt_decay_fraction=0.80))
        snap = evaluate(guard, decay=0.50)
        assert snap.decision == DECISION_REDUCE

    def test_halt_on_slippage(self):
        guard = DeploymentGuard(DeploymentThresholds(halt_slippage_bps=20.0))
        snap = evaluate(guard, slip=25.0)
        assert snap.decision == DECISION_HALT

    def test_halt_on_reject_rate(self):
        guard = DeploymentGuard(DeploymentThresholds(halt_reject_rate=0.30))
        snap = evaluate(guard, reject=0.35)
        assert snap.decision == DECISION_HALT

    def test_halt_on_decay(self):
        guard = DeploymentGuard(DeploymentThresholds(halt_decay_fraction=0.70))
        snap = evaluate(guard, decay=0.75)
        assert snap.decision == DECISION_HALT

    def test_halt_on_dislocated_regime(self):
        guard = DeploymentGuard(DeploymentThresholds(halt_on_dislocated=True, halt_on_halted=True))
        snap = evaluate(guard, regime="DISLOCATED")
        assert snap.decision == DECISION_HALT

    def test_halt_on_halted_regime(self):
        guard = make_guard()
        snap = evaluate(guard, regime="HALTED")
        assert snap.decision == DECISION_HALT

    def test_halt_on_liquidity_vacuum(self):
        guard = make_guard()
        snap = evaluate(guard, regime="LIQUIDITY_VACUUM")
        assert snap.decision == DECISION_HALT

    def test_context_hash_deterministic(self):
        g1 = make_guard()
        g2 = make_guard()
        s1 = evaluate(g1, slip=5.0, reject=0.05, decay=0.20)
        s2 = evaluate(g2, slip=5.0, reject=0.05, decay=0.20)
        assert s1.decision_context_hash == s2.decision_context_hash

    def test_context_hash_changes_with_inputs(self):
        guard = make_guard()
        s1 = evaluate(guard, slip=5.0, ts=TS)
        s2 = evaluate(guard, slip=10.0, ts=TS2)
        assert s1.decision_context_hash != s2.decision_context_hash

    def test_history_accumulates(self):
        guard = make_guard()
        evaluate(guard, ts=TS)
        evaluate(guard, ts=TS2)
        assert len(guard.history()) == 2

    def test_current_decision(self):
        guard = make_guard()
        evaluate(guard)
        assert guard.current_decision() == DECISION_ALLOW

    def test_current_decision_none_initially(self):
        guard = make_guard()
        assert guard.current_decision() is None

    def test_is_halted_false(self):
        guard = make_guard()
        evaluate(guard)
        assert not guard.is_halted()

    def test_is_halted_true(self):
        guard = make_guard()
        evaluate(guard, regime="HALTED")
        assert guard.is_halted()

    def test_to_dict_keys(self):
        guard = make_guard()
        d = guard.to_dict()
        assert "thresholds" in d and "current_decision" in d
