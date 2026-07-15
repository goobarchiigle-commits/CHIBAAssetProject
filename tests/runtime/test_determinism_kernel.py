"""
tests/runtime/test_determinism_kernel.py

Deterministic runtime kernel validation tests.

Covers:
  - same inputs -> same replay hash
  - canonical ordering stability (dict insertion-order independent)
  - stale lock recovery (dead PID)
  - invalid mutex schema ignored (non-lock artifact at lock path)
  - freeze violation detection + persistence
  - append-only persistence (JSONL only grows)
  - invariant failure fail-closed (report persisted before raise)
  - deployment always blocked
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.runtime.determinism.replay_hash import ReplayEquivalenceHash, _canonical, _canonical_json
from src.runtime.determinism.freeze_guard import RuntimeFreezeGuard, FreezeViolationError
from src.runtime.determinism.invariant_validator import (
    RuntimeInvariantValidator,
    DeploymentGate,
    InvariantError,
    DEPLOYMENT_GATE_FORCE_BLOCK,
    LIVE_ORDER_SUBMISSION_DISABLED,
)
from src.runtime.determinism.process_lock import ProcessLock, ProcessLockError, LOCK_PATH


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_hash_path(tmp_path):
    return tmp_path / "replay_hashes.jsonl"


@pytest.fixture
def tmp_violations_path(tmp_path):
    return tmp_path / "freeze_violations.jsonl"


@pytest.fixture
def tmp_reports_path(tmp_path):
    return tmp_path / "invariant_reports.jsonl"


@pytest.fixture
def tmp_lock_path(tmp_path):
    return tmp_path / "execution.lock.json"


# ─────────────────────────────────────────────────────────────────────────────
# ReplayEquivalenceHash
# ─────────────────────────────────────────────────────────────────────────────

_INPUTS = dict(
    epoch_manifest={"version": "1", "dataset": "ds_a"},
    orchestration_manifest={"stages": ["a", "b"]},
    dependency_manifest={"deps": {"alpha": "1.0"}},
    governance_snapshot={"policy": "strict"},
    runtime_decision_ledger=[{"ts": 1000, "decision": "buy"}],
)


class TestReplayEquivalenceHash:
    def _reh(self, tmp_path):
        return ReplayEquivalenceHash(output_path=tmp_path / "hashes.jsonl")

    def test_same_inputs_same_hash(self, tmp_path):
        reh = self._reh(tmp_path)
        h1 = reh.compute(**_INPUTS, persist=False)
        h2 = reh.compute(**_INPUTS, persist=False)
        assert h1 == h2

    def test_different_inputs_different_hash(self, tmp_path):
        reh = self._reh(tmp_path)
        h1 = reh.compute(**_INPUTS, persist=False)
        modified = dict(_INPUTS, governance_snapshot={"policy": "relaxed"})
        h2 = reh.compute(**modified, persist=False)
        assert h1 != h2

    def test_canonical_ordering_dict_insertion_order(self, tmp_path):
        reh = self._reh(tmp_path)
        base = dict(_INPUTS)
        reordered = dict(
            runtime_decision_ledger=_INPUTS["runtime_decision_ledger"],
            governance_snapshot=_INPUTS["governance_snapshot"],
            dependency_manifest=_INPUTS["dependency_manifest"],
            orchestration_manifest=_INPUTS["orchestration_manifest"],
            epoch_manifest=_INPUTS["epoch_manifest"],
        )
        h1 = reh.compute(**base, persist=False)
        h2 = reh.compute(**reordered, persist=False)
        assert h1 == h2

    def test_canonical_ordering_nested_dict(self, tmp_path):
        reh = self._reh(tmp_path)
        obj_a = {"b": 2, "a": 1}
        obj_b = {"a": 1, "b": 2}
        assert _canonical_json(obj_a) == _canonical_json(obj_b)

    def test_hash_is_sha256_hex(self, tmp_path):
        reh = self._reh(tmp_path)
        h = reh.compute(**_INPUTS, persist=False)
        assert len(h) == 64
        int(h, 16)  # must be valid hex

    def test_persistence_appends_entry(self, tmp_path):
        reh = self._reh(tmp_path)
        h = reh.compute(**_INPUTS, run_id="run_001", persist=True)
        entries = reh.load_hashes()
        assert len(entries) == 1
        assert entries[0]["hash"] == h
        assert entries[0]["run_id"] == "run_001"

    def test_persistence_grows_monotonically(self, tmp_path):
        reh = self._reh(tmp_path)
        for i in range(3):
            modified = dict(_INPUTS, governance_snapshot={"policy": str(i)})
            reh.compute(**modified, run_id=f"run_{i}", persist=True)
        entries = reh.load_hashes()
        assert len(entries) == 3

    def test_persist_false_no_file_written(self, tmp_path):
        p = tmp_path / "hashes.jsonl"
        reh = ReplayEquivalenceHash(output_path=p)
        reh.compute(**_INPUTS, persist=False)
        assert not p.exists()

    def test_canonical_primitive_types(self):
        assert _canonical(42) == 42
        assert _canonical("x") == "x"
        assert _canonical([3, 1, 2]) == [3, 1, 2]

    def test_canonical_nested_list_in_dict(self):
        obj = {"b": [3, 1], "a": [2]}
        canon = _canonical(obj)
        assert list(canon.keys()) == ["a", "b"]


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeFreezeGuard
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE = {
    "schema_manifests":       {"v": "1"},
    "governance_manifests":   {"policy": "strict"},
    "feature_manifests":      {"features": ["f1", "f2"]},
    "orchestration_manifests": {"stages": ["s1"]},
}


class TestRuntimeFreezeGuard:
    def _guard(self, tmp_path):
        vpath = tmp_path / "violations.jsonl"
        return RuntimeFreezeGuard(baseline=_BASELINE, violations_path=vpath)

    def test_no_violation_on_identical(self, tmp_path):
        guard = self._guard(tmp_path)
        import copy
        guard.check(copy.deepcopy(_BASELINE))  # must not raise

    def test_violation_raises_freeze_violation_error(self, tmp_path):
        guard = self._guard(tmp_path)
        mutated = dict(_BASELINE, schema_manifests={"v": "MUTATED"})
        with pytest.raises(FreezeViolationError) as exc_info:
            guard.check(mutated)
        assert exc_info.value.key == "schema_manifests"

    def test_violation_persisted_before_raise(self, tmp_path):
        vpath = tmp_path / "violations.jsonl"
        guard = RuntimeFreezeGuard(baseline=_BASELINE, violations_path=vpath)
        mutated = dict(_BASELINE, governance_manifests={"policy": "none"})
        with pytest.raises(FreezeViolationError):
            guard.check(mutated)
        violations = guard.load_violations()
        assert len(violations) == 1
        assert violations[0]["key"] == "governance_manifests"
        assert violations[0]["event"] == "freeze_violation"

    def test_only_freeze_sources_checked(self, tmp_path):
        guard = self._guard(tmp_path)
        current = dict(_BASELINE, extra_unrelated_key={"x": 99})
        guard.check(current)  # must not raise — extra keys are ignored

    def test_multiple_violations_each_recorded(self, tmp_path):
        vpath = tmp_path / "v.jsonl"
        guard = RuntimeFreezeGuard(baseline=_BASELINE, violations_path=vpath)
        for key in ["schema_manifests", "feature_manifests"]:
            mutated = dict(_BASELINE, **{key: {"mutated": True}})
            with pytest.raises(FreezeViolationError):
                guard.check(mutated)
        violations = guard.load_violations()
        assert len(violations) == 2

    def test_baseline_deepcopy_isolation(self, tmp_path):
        baseline = {"schema_manifests": {"v": "1"}, "governance_manifests": {}}
        guard = RuntimeFreezeGuard(baseline=baseline, violations_path=tmp_path / "v.jsonl")
        # mutate original — guard's baseline must be unaffected
        baseline["schema_manifests"]["v"] = "TAMPERED"
        current = {"schema_manifests": {"v": "1"}, "governance_manifests": {}}
        guard.check(current)  # must not raise

    def test_feature_manifest_mutation_detected(self, tmp_path):
        guard = self._guard(tmp_path)
        mutated = dict(_BASELINE, feature_manifests={"features": ["f1"]})
        with pytest.raises(FreezeViolationError) as exc_info:
            guard.check(mutated)
        assert exc_info.value.key == "feature_manifests"

    def test_orchestration_manifest_mutation_detected(self, tmp_path):
        guard = self._guard(tmp_path)
        mutated = dict(_BASELINE, orchestration_manifests={"stages": ["s1", "s2"]})
        with pytest.raises(FreezeViolationError) as exc_info:
            guard.check(mutated)
        assert exc_info.value.key == "orchestration_manifests"


# ─────────────────────────────────────────────────────────────────────────────
# DeploymentGate
# ─────────────────────────────────────────────────────────────────────────────

class TestDeploymentGate:
    def test_always_blocked(self):
        gate = DeploymentGate()
        result = gate.evaluate()
        assert result["status"] == "BLOCKED"
        assert result["blocked"] is True

    def test_is_blocked_true(self):
        assert DeploymentGate().is_blocked() is True

    def test_blocked_ignores_args(self):
        gate = DeploymentGate()
        for args, kwargs in [
            ((), {}),
            (("run_001",), {}),
            ((), {"epoch_id": "e1", "allow": True}),
        ]:
            result = gate.evaluate(*args, **kwargs)
            assert result["status"] == "BLOCKED"

    def test_module_level_constants(self):
        assert DEPLOYMENT_GATE_FORCE_BLOCK is True
        assert LIVE_ORDER_SUBMISSION_DISABLED is True

    def test_class_level_constants(self):
        assert DeploymentGate.DEPLOYMENT_GATE_FORCE_BLOCK is True
        assert DeploymentGate.LIVE_ORDER_SUBMISSION_DISABLED is True

    def test_multiple_calls_always_blocked(self):
        gate = DeploymentGate()
        for _ in range(10):
            assert gate.evaluate()["status"] == "BLOCKED"


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeInvariantValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeInvariantValidator:
    def _validator(self, tmp_path):
        return RuntimeInvariantValidator(reports_path=tmp_path / "reports.jsonl")

    # append-only ledger
    def test_ledger_append_only_valid(self, tmp_path):
        v = self._validator(tmp_path)
        v.validate_ledger_append_only(prior_count=5, current_count=7)  # must not raise

    def test_ledger_append_only_equal(self, tmp_path):
        v = self._validator(tmp_path)
        v.validate_ledger_append_only(prior_count=5, current_count=5)  # must not raise

    def test_ledger_append_only_shrink_raises(self, tmp_path):
        v = self._validator(tmp_path)
        with pytest.raises(InvariantError) as exc_info:
            v.validate_ledger_append_only(prior_count=10, current_count=3)
        assert exc_info.value.invariant == "append_only_ledger"

    def test_ledger_violation_report_persisted(self, tmp_path):
        v = self._validator(tmp_path)
        with pytest.raises(InvariantError):
            v.validate_ledger_append_only(prior_count=10, current_count=3, run_id="r1")
        reports = v.load_reports()
        assert len(reports) == 1
        assert reports[0]["invariant"] == "append_only_ledger"
        assert reports[0]["run_id"] == "r1"

    # idempotent rerun
    def test_idempotent_rerun_same_hash_passes(self, tmp_path):
        v = self._validator(tmp_path)
        v.validate_idempotent_rerun("run_x", "abc123", "abc123")  # must not raise

    def test_idempotent_rerun_different_hash_raises(self, tmp_path):
        v = self._validator(tmp_path)
        with pytest.raises(InvariantError) as exc_info:
            v.validate_idempotent_rerun("run_x", "aaa", "bbb")
        assert exc_info.value.invariant == "idempotent_rerun_equality"

    def test_idempotent_rerun_report_persisted(self, tmp_path):
        v = self._validator(tmp_path)
        with pytest.raises(InvariantError):
            v.validate_idempotent_rerun("run_x", "aaa", "bbb")
        reports = v.load_reports()
        assert any(r["invariant"] == "idempotent_rerun_equality" for r in reports)

    # deployment authority
    def test_deployment_authority_blocked_passes(self, tmp_path):
        v = self._validator(tmp_path)
        gate = DeploymentGate()
        v.validate_deployment_authority(gate)  # must not raise

    def test_deployment_authority_unblocked_raises(self, tmp_path):
        v = self._validator(tmp_path)
        bad_gate = mock.MagicMock()
        bad_gate.evaluate.return_value = {"status": "OK", "blocked": False}
        with pytest.raises(InvariantError) as exc_info:
            v.validate_deployment_authority(bad_gate)
        assert exc_info.value.invariant == "deployment_authority_isolation"

    # artifact lineage
    def test_artifact_lineage_all_present(self, tmp_path):
        v = self._validator(tmp_path)
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text("{}")
        f2.write_text("{}")
        v.validate_artifact_lineage([str(f1), str(f2)])  # must not raise

    def test_artifact_lineage_missing_raises(self, tmp_path):
        v = self._validator(tmp_path)
        with pytest.raises(InvariantError) as exc_info:
            v.validate_artifact_lineage([str(tmp_path / "ghost.json")], run_id="r2")
        assert exc_info.value.invariant == "artifact_lineage_presence"
        assert "ghost.json" in str(exc_info.value.detail)

    def test_artifact_lineage_report_persisted(self, tmp_path):
        v = self._validator(tmp_path)
        with pytest.raises(InvariantError):
            v.validate_artifact_lineage([str(tmp_path / "missing.json")], run_id="r3")
        reports = v.load_reports()
        assert any(r["run_id"] == "r3" for r in reports)

    def test_report_append_only(self, tmp_path):
        v = self._validator(tmp_path)
        for i in range(3):
            with pytest.raises(InvariantError):
                v.validate_ledger_append_only(prior_count=10, current_count=i, run_id=f"r{i}")
        reports = v.load_reports()
        assert len(reports) == 3

    def test_fail_closed_report_before_raise(self, tmp_path):
        """Report must be written before InvariantError is raised."""
        rpath = tmp_path / "reports.jsonl"
        v = RuntimeInvariantValidator(reports_path=rpath)
        raised = False
        try:
            v.validate_ledger_append_only(prior_count=5, current_count=0)
        except InvariantError:
            raised = True
        assert raised
        assert rpath.exists()
        assert rpath.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# ProcessLock
# ─────────────────────────────────────────────────────────────────────────────

def _write_lock_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestProcessLock:
    def test_basic_acquire_release(self, tmp_lock_path):
        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        lock.acquire()
        assert lock.is_held()
        lock.release()
        assert not lock.is_held()

    def test_context_manager(self, tmp_lock_path):
        with ProcessLock(lock_path=tmp_lock_path, mode="test",
                         heartbeat_interval=60.0, heartbeat_expiry=120.0) as lock:
            assert lock.is_held()
        assert not lock.is_held()

    def test_held_duration_monotonic(self, tmp_lock_path):
        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        assert lock.held_duration_sec() is None
        lock.acquire()
        dur = lock.held_duration_sec()
        assert dur is not None and dur >= 0
        lock.release()
        assert lock.held_duration_sec() is None

    def test_double_acquire_raises(self, tmp_lock_path):
        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        lock.acquire()
        try:
            with pytest.raises(ProcessLockError):
                lock.acquire()
        finally:
            lock.release()

    def test_stale_lock_recovery(self, tmp_lock_path):
        """Dead PID + expired heartbeat → stale lock recovered; new acquire succeeds."""
        stale = {
            "version":      "2",
            "pid":          999999,         # almost certainly dead
            "heartbeat_ts": time.time() - 300,  # expired
            "created_at":   time.time() - 400,
            "mode":         "old",
            "hostname":     "old-host",
            "boot_id":      "0",
            "command":      "old_cmd",
            "process_create_time": None,
        }
        _write_lock_file(tmp_lock_path, stale)
        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        lock.acquire()
        assert lock.is_held()
        lock.release()

    def test_invalid_mutex_schema_ignored(self, tmp_lock_path):
        """Non-mutex artifact at lock path must NOT block acquisition."""
        # This mimics a non-lock JSON artifact (e.g. order dedup ledger)
        non_mutex = {"order_id": "ORD-001", "symbol": "1234", "side": "BUY"}
        _write_lock_file(tmp_lock_path, non_mutex)

        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        lock.acquire()
        assert lock.is_held()
        lock.release()

    def test_atexit_registered_on_acquire(self, tmp_lock_path):
        """atexit handler must be registered when lock is acquired."""
        registered: list = []
        orig_register = atexit.register

        def capturing_register(fn, *args, **kwargs):
            registered.append(fn)
            return orig_register(fn, *args, **kwargs)

        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        with mock.patch("src.runtime.determinism.process_lock.atexit.register",
                        side_effect=capturing_register):
            lock._atexit_registered = False  # reset so it re-registers
            lock.acquire()
        assert len(registered) >= 1
        lock.release()

    def test_lock_path_constant_not_order_lock(self):
        """LOCK_PATH must NOT point to runtime/order_lock.json."""
        assert "order_lock" not in str(LOCK_PATH)
        assert "execution.lock" in str(LOCK_PATH)

    def test_release_idempotent(self, tmp_lock_path):
        lock = ProcessLock(lock_path=tmp_lock_path, mode="test",
                           heartbeat_interval=60.0, heartbeat_expiry=120.0)
        lock.release()  # not held — must not raise
        lock.acquire()
        lock.release()
        lock.release()  # second release — must not raise

    def test_live_process_blocks_acquisition(self, tmp_lock_path):
        """If live lock exists (same process, fresh heartbeat) second acquire must fail."""
        lock1 = ProcessLock(lock_path=tmp_lock_path, mode="test",
                            heartbeat_interval=60.0, heartbeat_expiry=120.0)
        lock1.acquire()
        try:
            lock2 = ProcessLock(lock_path=tmp_lock_path, mode="test",
                                heartbeat_interval=60.0, heartbeat_expiry=120.0)
            with pytest.raises(ProcessLockError):
                lock2.acquire()
        finally:
            lock1.release()
