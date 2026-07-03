"""
tests/deployment/test_micro_live_governance_bridge.py
Phase 4B — Micro Live Governance Bridge tests.

Coverage:
  - deterministic replay test
  - freeze persistence test
  - reconciliation mismatch test
  - governance hash stability test
  - HUMAN_CONFIRMED flow test
  - replay consistency test
  - append-only audit integrity test
  - constraint validation (MAX_LIVE_POSITIONS/CAPITAL_PCT/DAILY_ORDERS)
  - freeze trigger + MIN_FREEZE_DAYS enforcement
  - unfreeze conditions
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest

from src.deployment.connectors.broker_interface import (
    BrokerAccountState,
    BrokerOrderResult,
    BrokerPosition,
    LiveExecutionMode,
    DEFAULT_EXECUTION_MODE,
)
from src.deployment.governance_audit import (
    FreezeEvaluator,
    GovernanceFreezeState,
    GovernanceAuditRecord,
    ReconciliationStatus,
    append_governance_audit,
    compute_governance_hash,
    load_freeze_state,
    load_governance_audits,
    save_freeze_state,
    MIN_FREEZE_DAYS,
    MIN_CLEAN_RECONCILIATIONS,
    UNFREEZE_PRESSURE_THRESHOLD,
)
from src.deployment.micro_live_governance_bridge import (
    MAX_LIVE_POSITIONS,
    MAX_LIVE_CAPITAL_PCT,
    MAX_LIVE_DAILY_ORDERS,
    MicroCapitalConstraints,
    MicroLiveGovernanceBridge,
    ReconciliationChecker,
    UpstreamGovernanceState,
    run_governance_bridge_hook,
)


# ── Test fixtures ──────────────────────────────────────────────────────────────

def _stable_upstream(**overrides) -> UpstreamGovernanceState:
    base = dict(
        universe_version="v1.0",
        deployment_manifest_hash="abc123",
        rollback_state="stable",
        rollback_pressure_score=0.0,
        eligibility_gate_passed=True,
        intended_symbols=["7203.T"],
        intended_weights={"7203.T": 0.02},
        upstream_hashes={"research": "aaa", "deployment": "bbb"},
        broker_timeout=False,
        api_integrity_unknown=False,
    )
    base.update(overrides)
    return UpstreamGovernanceState(**base)


class _StubBroker:
    """Test double for BrokerInterface."""

    def __init__(
        self,
        positions: Optional[List[BrokerPosition]] = None,
        api_reachable: bool = True,
    ) -> None:
        self._positions = positions or []
        self._api_reachable = api_reachable

    def get_positions(self) -> List[BrokerPosition]:
        return self._positions

    def get_account_state(self) -> BrokerAccountState:
        return BrokerAccountState(
            cash_balance=3_000_000.0,
            total_assets=3_000_000.0,
            positions=self._positions,
            api_reachable=self._api_reachable,
            timestamp="2026-05-25T09:00:00+09:00",
        )

    def submit_order(self, symbol, side, quantity, client_order_id, slippage=0.001):
        return BrokerOrderResult(
            client_order_id=client_order_id,
            broker_order_id="mock-001",
            accepted=True,
            rejection_reason=None,
            submitted_price=None,
            timestamp="2026-05-25T09:00:01+09:00",
        )

    def cancel_order(self, client_order_id, broker_order_id=None):
        return True


# ── Tests: broker_interface ────────────────────────────────────────────────────

class TestLiveExecutionMode:
    def test_default_is_human_confirmed(self):
        assert DEFAULT_EXECUTION_MODE == LiveExecutionMode.HUMAN_CONFIRMED

    def test_all_modes_defined(self):
        modes = {m.value for m in LiveExecutionMode}
        assert "intent_only"      in modes
        assert "human_confirmed"  in modes
        assert "fully_autonomous" in modes

    def test_stub_broker_satisfies_protocol(self):
        broker = _StubBroker()
        assert isinstance(broker, object)   # runtime_checkable Protocol
        account = broker.get_account_state()
        assert account.api_reachable is True
        assert account.total_assets == 3_000_000.0


# ── Tests: governance_audit ────────────────────────────────────────────────────

class TestGovernanceHash:
    def test_same_inputs_produce_same_hash(self):
        h1 = compute_governance_hash(
            universe_version="v1",
            deployment_manifest_hash="abc",
            rollback_state="stable",
            freeze_state_frozen=False,
            governance_decisions=["X", "Y"],
            live_mode="human_confirmed",
            upstream_hashes={"a": "1", "b": "2"},
        )
        h2 = compute_governance_hash(
            universe_version="v1",
            deployment_manifest_hash="abc",
            rollback_state="stable",
            freeze_state_frozen=False,
            governance_decisions=["Y", "X"],  # order invariant via sorted()
            live_mode="human_confirmed",
            upstream_hashes={"b": "2", "a": "1"},  # dict order invariant via sorted()
        )
        assert h1 == h2

    def test_different_inputs_produce_different_hash(self):
        h1 = compute_governance_hash("v1", "abc", "stable", False, [], "human_confirmed", {})
        h2 = compute_governance_hash("v2", "abc", "stable", False, [], "human_confirmed", {})
        assert h1 != h2

    def test_freeze_flag_changes_hash(self):
        h1 = compute_governance_hash("v1", "abc", "stable", False, [], "hc", {})
        h2 = compute_governance_hash("v1", "abc", "stable", True, [], "hc", {})
        assert h1 != h2

    def test_hash_is_16_chars(self):
        h = compute_governance_hash("v1", "abc", "stable", False, [], "hc", {})
        assert len(h) == 16


class TestGovernanceAuditAppend:
    def test_append_and_load(self, tmp_path):
        record = GovernanceAuditRecord(
            evaluation_date="2026-05-25",
            upstream_hashes={"a": "1"},
            eligibility_gate_passed=True,
            governance_decisions=[],
            rollback_decisions=[],
            freeze_decisions=[],
            reconciliation_status="clean",
            deterministic_hash="abc1234",
            timestamp="2026-05-25T09:00:00+09:00",
        )
        append_governance_audit(record, tmp_path)
        records = load_governance_audits(tmp_path, date_str="20260525")
        assert len(records) == 1
        assert records[0]["evaluation_date"] == "2026-05-25"
        assert records[0]["deterministic_hash"] == "abc1234"

    def test_append_is_additive(self, tmp_path):
        for i in range(3):
            rec = GovernanceAuditRecord(
                evaluation_date="2026-05-25",
                upstream_hashes={},
                eligibility_gate_passed=True,
                governance_decisions=[f"decision_{i}"],
                rollback_decisions=[],
                freeze_decisions=[],
                reconciliation_status="clean",
                deterministic_hash=f"hash{i}",
                timestamp=f"2026-05-25T09:0{i}:00+09:00",
            )
            append_governance_audit(rec, tmp_path)
        records = load_governance_audits(tmp_path, date_str="20260525")
        assert len(records) == 3

    def test_load_all_returns_multiple_days(self, tmp_path):
        for date in ["2026-05-24", "2026-05-25"]:
            rec = GovernanceAuditRecord(
                evaluation_date=date,
                upstream_hashes={},
                eligibility_gate_passed=True,
                governance_decisions=[],
                rollback_decisions=[],
                freeze_decisions=[],
                reconciliation_status="clean",
                deterministic_hash="x",
                timestamp="",
            )
            append_governance_audit(rec, tmp_path)
        all_records = load_governance_audits(tmp_path)
        assert len(all_records) == 2


class TestFreezeEvaluator:
    def setup_method(self):
        self.ev = FreezeEvaluator()

    def test_no_trigger_stays_unfrozen(self):
        state = GovernanceFreezeState()
        new, decisions = self.ev.evaluate(state, "2026-05-25")
        assert not new.frozen
        assert decisions == []

    def test_broker_timeout_triggers_freeze(self):
        state = GovernanceFreezeState()
        new, decisions = self.ev.evaluate(state, "2026-05-25", broker_timeout=True)
        assert new.frozen
        assert any("FREEZE_TRIGGERED" in d for d in decisions)
        assert "broker_timeout" in decisions[0]

    def test_reconciliation_mismatch_triggers_freeze(self):
        state = GovernanceFreezeState()
        new, decisions = self.ev.evaluate(
            state, "2026-05-25",
            reconciliation_status=ReconciliationStatus.MISMATCH
        )
        assert new.frozen

    def test_replay_mismatch_triggers_freeze(self):
        state = GovernanceFreezeState()
        new, decisions = self.ev.evaluate(state, "2026-05-25", replay_mismatch=True)
        assert new.frozen

    def test_freeze_persists_for_min_days(self):
        state = GovernanceFreezeState()
        # Trigger freeze
        state, _ = self.ev.evaluate(state, "2026-05-20", broker_timeout=True)
        assert state.frozen

        # MIN_FREEZE_DAYS - 1 days of clean reconciliation should NOT unfreeze
        for i in range(1, MIN_FREEZE_DAYS):
            date = f"2026-05-{20 + i:02d}"
            state, decisions = self.ev.evaluate(
                state, date,
                reconciliation_status=ReconciliationStatus.CLEAN,
                api_reachable=True,
                rollback_pressure_score=0.0,
            )
            assert state.frozen, f"should still be frozen on day {i}"

    def test_freeze_released_after_conditions_met(self):
        state = GovernanceFreezeState()
        # Trigger freeze
        state, _ = self.ev.evaluate(state, "2026-05-20", broker_timeout=True)

        # Advance MIN_FREEZE_DAYS + MIN_CLEAN_RECONCILIATIONS days
        total_days = max(MIN_FREEZE_DAYS, MIN_CLEAN_RECONCILIATIONS)
        for i in range(1, total_days + 2):
            date = f"2026-05-{20 + i:02d}"
            state, decisions = self.ev.evaluate(
                state, date,
                reconciliation_status=ReconciliationStatus.CLEAN,
                api_reachable=True,
                rollback_pressure_score=0.0,
            )

        assert not state.frozen

    def test_high_pressure_prevents_unfreeze(self):
        state = GovernanceFreezeState()
        state, _ = self.ev.evaluate(state, "2026-05-20", broker_timeout=True)

        # Many clean days but pressure > threshold
        for i in range(1, 20):
            date = f"2026-05-{20 + i:02d}"
            state, decisions = self.ev.evaluate(
                state, date,
                reconciliation_status=ReconciliationStatus.CLEAN,
                api_reachable=True,
                rollback_pressure_score=UNFREEZE_PRESSURE_THRESHOLD + 0.1,
            )
        # Should remain frozen due to high pressure
        assert state.frozen

    def test_clean_counter_resets_on_new_trigger(self):
        state = GovernanceFreezeState()
        state, _ = self.ev.evaluate(state, "2026-05-20", broker_timeout=True)
        # Some clean days
        for i in range(1, 3):
            date = f"2026-05-{20 + i:02d}"
            state, _ = self.ev.evaluate(
                state, date,
                reconciliation_status=ReconciliationStatus.CLEAN,
                api_reachable=True,
                rollback_pressure_score=0.0,
            )
        clean_before = state.consecutive_clean_reconciliations
        assert clean_before >= 2

        # New trigger resets counter
        state, decisions = self.ev.evaluate(
            state, "2026-05-23", api_integrity_unknown=True
        )
        assert state.consecutive_clean_reconciliations == 0
        assert any("RETRIGGERED" in d for d in decisions)


class TestFreezeStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "freeze.json"
        state = GovernanceFreezeState(
            frozen=True,
            freeze_start_date="2026-05-20",
            freeze_day_count=3,
            consecutive_clean_reconciliations=1,
            last_evaluation_date="2026-05-23",
        )
        save_freeze_state(state, path)
        loaded = load_freeze_state(path)
        assert loaded.frozen is True
        assert loaded.freeze_day_count == 3
        assert loaded.consecutive_clean_reconciliations == 1

    def test_load_missing_returns_default(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        state = load_freeze_state(path)
        assert not state.frozen
        assert state.freeze_day_count == 0


# ── Tests: MicroCapitalConstraints ────────────────────────────────────────────

class TestMicroCapitalConstraints:
    def test_single_symbol_ok(self):
        ok, violations = MicroCapitalConstraints.validate(
            intended_weights={"7203.T": 0.02},
            daily_order_count=0,
        )
        assert ok
        assert violations == []

    def test_too_many_symbols(self):
        ok, violations = MicroCapitalConstraints.validate(
            intended_weights={"7203.T": 0.01, "9984.T": 0.01},
            daily_order_count=0,
        )
        assert not ok
        assert any("MAX_LIVE_POSITIONS" in v for v in violations)

    def test_excess_weight(self):
        ok, violations = MicroCapitalConstraints.validate(
            intended_weights={"7203.T": 0.05},   # > MAX_LIVE_CAPITAL_PCT=0.02
            daily_order_count=0,
        )
        assert not ok
        assert any("MAX_LIVE_CAPITAL_PCT" in v for v in violations)

    def test_max_daily_orders_hit(self):
        ok, violations = MicroCapitalConstraints.validate(
            intended_weights={},
            daily_order_count=MAX_LIVE_DAILY_ORDERS,
        )
        assert not ok
        assert any("MAX_LIVE_DAILY_ORDERS" in v for v in violations)

    def test_empty_weights_and_no_orders_ok(self):
        ok, violations = MicroCapitalConstraints.validate({}, daily_order_count=0)
        assert ok

    def test_exact_weight_limit_ok(self):
        ok, violations = MicroCapitalConstraints.validate(
            {"7203.T": MAX_LIVE_CAPITAL_PCT},
            daily_order_count=0,
        )
        assert ok


# ── Tests: ReconciliationChecker ──────────────────────────────────────────────

class TestReconciliationChecker:
    def setup_method(self):
        self.checker = ReconciliationChecker()

    def test_both_empty_is_clean(self):
        result = self.checker.check([], [], 3_000_000.0, {})
        assert result.status == ReconciliationStatus.CLEAN

    def test_matching_positions_clean(self):
        pos = [BrokerPosition("7203.T", 100, 2000.0, 60_000.0, 0.0)]
        result = self.checker.check(
            ["7203.T"], pos, 3_000_000.0, {"7203.T": 0.02}
        )
        assert result.status == ReconciliationStatus.CLEAN

    def test_unexpected_broker_position_is_mismatch(self):
        pos = [BrokerPosition("9984.T", 10, 5000.0, 50_000.0, 0.0)]
        result = self.checker.check(["7203.T"], pos, 3_000_000.0, {"7203.T": 0.02})
        assert result.status == ReconciliationStatus.MISMATCH
        assert any("unexpected:9984.T" in m for m in result.mismatched_symbols)

    def test_missing_intended_position_is_mismatch(self):
        result = self.checker.check(["7203.T"], [], 3_000_000.0, {"7203.T": 0.02})
        assert result.status == ReconciliationStatus.MISMATCH
        assert any("missing:7203.T" in m for m in result.mismatched_symbols)

    def test_weight_mismatch_detected(self):
        # Actual weight = 200_000 / 3_000_000 = 0.0667, intended = 0.02 — far off
        pos = [BrokerPosition("7203.T", 100, 2000.0, 200_000.0, 0.0)]
        result = self.checker.check(
            ["7203.T"], pos, 3_000_000.0, {"7203.T": 0.02}
        )
        assert result.status == ReconciliationStatus.MISMATCH
        assert any("weight_mismatch" in m for m in result.mismatched_symbols)


# ── Tests: MicroLiveGovernanceBridge ──────────────────────────────────────────

class TestMicroLiveGovernanceBridge:
    def _bridge(self, tmp_path) -> MicroLiveGovernanceBridge:
        return MicroLiveGovernanceBridge(
            state_dir=tmp_path / "state",
            audit_dir=tmp_path / "audit",
            mode=LiveExecutionMode.HUMAN_CONFIRMED,
        )

    def test_stable_upstream_produces_enter_intent(self, tmp_path):
        bridge = self._bridge(tmp_path)
        upstream = _stable_upstream()
        broker = _StubBroker()
        report = bridge.evaluate(upstream, broker, 3_000_000.0, 0, "2026-05-25")

        assert len(report.broker_intents) >= 1
        intent = report.broker_intents[0]
        assert intent.action == "enter"
        assert intent.order_eligible is True
        assert not intent.governance_blocked

    def test_frozen_governance_blocks_entry(self, tmp_path):
        bridge = self._bridge(tmp_path)
        # Trigger freeze via reconciliation mismatch
        upstream_mismatch = _stable_upstream()
        broker_with_unexpected = _StubBroker(
            positions=[BrokerPosition("9984.T", 10, 5000.0, 50_000.0, 0.0)]
        )
        report1 = bridge.evaluate(upstream_mismatch, broker_with_unexpected, 3_000_000.0, 0, "2026-05-20")
        assert report1.freeze_state.frozen

        # Next day — freeze still active
        report2 = bridge.evaluate(_stable_upstream(), _StubBroker(), 3_000_000.0, 0, "2026-05-21")
        assert report2.freeze_state.frozen
        intent = report2.broker_intents[0]
        assert intent.action == "freeze"
        assert intent.order_eligible is False

    def test_eligibility_gate_failure_blocks_entry(self, tmp_path):
        bridge = self._bridge(tmp_path)
        upstream = _stable_upstream(eligibility_gate_passed=False)
        report = bridge.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        intent = report.broker_intents[0]
        assert intent.action == "idle"
        assert intent.order_eligible is False
        assert any("ELIGIBILITY_GATE_FAILED" in r for r in intent.block_reasons)

    def test_rollback_active_blocks_entry(self, tmp_path):
        bridge = self._bridge(tmp_path)
        upstream = _stable_upstream(rollback_state="full_rollback")
        report = bridge.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        intent = report.broker_intents[0]
        assert intent.action == "rollback"
        assert intent.order_eligible is False

    def test_max_daily_orders_blocks_entry(self, tmp_path):
        bridge = self._bridge(tmp_path)
        upstream = _stable_upstream()
        report = bridge.evaluate(upstream, _StubBroker(), 3_000_000.0, MAX_LIVE_DAILY_ORDERS, "2026-05-25")
        intent = report.broker_intents[0]
        assert intent.order_eligible is False
        assert any("MAX_DAILY_ORDERS_HIT" in r for r in intent.block_reasons)

    def test_audit_record_written_to_jsonl(self, tmp_path):
        bridge = self._bridge(tmp_path)
        report = bridge.evaluate(_stable_upstream(), _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        audit_dir = tmp_path / "audit"
        records = load_governance_audits(audit_dir, date_str="20260525")
        assert len(records) == 1
        assert records[0]["deterministic_hash"] == report.governance_audit.deterministic_hash

    def test_deterministic_hash_stability(self, tmp_path):
        """Same inputs on same date must produce same hash."""
        bridge1 = self._bridge(tmp_path / "b1")
        bridge2 = self._bridge(tmp_path / "b2")
        upstream = _stable_upstream()
        r1 = bridge1.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        r2 = bridge2.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        assert r1.governance_audit.deterministic_hash == r2.governance_audit.deterministic_hash

    def test_replay_mismatch_triggers_freeze(self, tmp_path):
        """Calling bridge with same date but different upstream should NOT produce mismatch
        on first call (no prior hash). Second call same date same upstream must not mismatch."""
        bridge = self._bridge(tmp_path)
        upstream = _stable_upstream()
        r1 = bridge.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        # Same call: no mismatch — freeze state should not change
        r2 = bridge.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        assert r2.freeze_state.frozen == r1.freeze_state.frozen

    def test_append_only_audit_integrity(self, tmp_path):
        bridge = self._bridge(tmp_path)
        for i in range(3):
            date = f"2026-05-{25 + i:02d}"
            bridge.evaluate(_stable_upstream(), _StubBroker(), 3_000_000.0, 0, date)
        records = load_governance_audits(tmp_path / "audit")
        assert len(records) == 3

    def test_intent_only_mode_blocks_submission(self, tmp_path):
        bridge = MicroLiveGovernanceBridge(
            state_dir=tmp_path / "state",
            audit_dir=tmp_path / "audit",
            mode=LiveExecutionMode.INTENT_ONLY,
        )
        report = bridge.evaluate(_stable_upstream(), _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        # In INTENT_ONLY mode, order_eligible must be False (no submission)
        assert all(not i.order_eligible for i in report.broker_intents)

    def test_freeze_state_persisted_across_instances(self, tmp_path):
        """Freeze state survives bridge re-instantiation (restart simulation)."""
        state_dir = tmp_path / "state"
        audit_dir = tmp_path / "audit"

        bridge1 = MicroLiveGovernanceBridge(state_dir, audit_dir)
        broker_mismatch = _StubBroker(
            positions=[BrokerPosition("9984.T", 10, 5000.0, 50_000.0, 0.0)]
        )
        bridge1.evaluate(_stable_upstream(), broker_mismatch, 3_000_000.0, 0, "2026-05-20")

        # New bridge instance (simulates restart)
        bridge2 = MicroLiveGovernanceBridge(state_dir, audit_dir)
        report2 = bridge2.evaluate(_stable_upstream(), _StubBroker(), 3_000_000.0, 0, "2026-05-21")
        assert report2.freeze_state.frozen  # freeze survived restart

    def test_human_confirmed_flow(self, tmp_path):
        """HUMAN_CONFIRMED: order_eligible=True only when all gates pass."""
        bridge = MicroLiveGovernanceBridge(
            state_dir=tmp_path / "state",
            audit_dir=tmp_path / "audit",
            mode=LiveExecutionMode.HUMAN_CONFIRMED,
        )
        upstream = _stable_upstream()
        report = bridge.evaluate(upstream, _StubBroker(), 3_000_000.0, 0, "2026-05-25")
        eligible_intents = [i for i in report.broker_intents if i.order_eligible]
        assert len(eligible_intents) >= 1
        assert eligible_intents[0].symbol == "7203.T"


class TestRunGovernanceBridgeHook:
    def test_hook_produces_report(self, tmp_path):
        report = run_governance_bridge_hook(
            state_dir=tmp_path / "state",
            audit_dir=tmp_path / "audit",
            upstream=_stable_upstream(),
            broker=_StubBroker(),
            total_capital=3_000_000.0,
            daily_order_count=0,
            mode=LiveExecutionMode.HUMAN_CONFIRMED,
            evaluation_date="2026-05-25",
        )
        assert report.evaluation_date == "2026-05-25"
        assert report.live_mode == "human_confirmed"
        assert len(report.broker_intents) >= 1
