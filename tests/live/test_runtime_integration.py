"""
tests/live/test_runtime_integration.py

Tests for src/live/runtime_integration.py:
  - RuntimeContractError hierarchy
  - RuntimeStateRegistry: None-state prevention, type checks, staleness, manifest
  - IntegrationDependencyGraph: dep validation, init ordering, topological sort
  - StagePreconditionValidator: stage ordering safety
  - RuntimeCoherenceValidator: field validation, allocator interface
  - BrokerRuntimeConsistencyValidator: qty drift, orphan positions, attribution
  - Artifact persistence functions: determinism, append-only, JSON validity
  - validate_runtime_integration: fail policy, blocking count, live vs dry-run
  - Replay determinism: same inputs → same violation set
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.live.runtime_integration import (
    # Exceptions
    RuntimeContractError,
    NullStateAccessError,
    DependencyMissingError,
    StagePreconditionError,
    InitializationOrderError,
    InterfaceSignatureMismatchError,
    # Stage constants
    STAGE_API_CHECK,
    STAGE_BOOTSTRAP,
    STAGE_BROKER_SYNC,
    STAGE_RECONCILIATION,
    STAGE_SIGNAL_GENERATION,
    STAGE_MARKET_DATA,
    STAGE_ALLOCATION,
    STAGE_ORDER_EXECUTION,
    STAGE_CLEANUP,
    STAGE_PRECONDITIONS,
    # Components
    RuntimeStateRegistry,
    RegistryEntry,
    IntegrationDependencyGraph,
    DependencyNode,
    StagePreconditionValidator,
    RuntimeCoherenceValidator,
    BrokerRuntimeConsistencyValidator,
    ConsistencyViolation,
    BrokerRuntimeConsistencyReport,
    CONSISTENCY_QTY_DRIFT,
    CONSISTENCY_ORPHAN_RUNTIME,
    CONSISTENCY_ORPHAN_BROKER,
    CONSISTENCY_INTENT_QTY_MISMATCH,
    CSEV_BLOCKING,
    CSEV_WARNING,
    # Artifact functions
    write_dependency_manifest,
    write_initialization_lineage,
    write_state_validation_report,
    write_coherence_diagnostic,
    write_broker_consistency_report,
    write_failure_attribution,
    # Top-level
    validate_runtime_integration,
    IntegrationValidationResult,
    SCHEMA_VERSION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(offset_sec: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_sec)).isoformat()


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


@dataclass
class _FakeCapState:
    actual_equity:          float
    effective_capital:      float
    risk_adjusted_capital:  float


@dataclass
class _FakeExposureState:
    return_buffers: dict


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeContractError hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeContractError:
    def test_base_is_runtime_error(self):
        err = RuntimeContractError("msg", "some_type")
        assert isinstance(err, RuntimeError)
        assert err.violation_type == "some_type"
        assert "msg" in str(err)

    def test_null_state_access_error(self):
        err = NullStateAccessError("my_state")
        assert isinstance(err, RuntimeContractError)
        assert err.violation_type == "null_state_access"
        assert "my_state" in str(err)
        assert err.state_name == "my_state"

    def test_dependency_missing_error(self):
        err = DependencyMissingError("dep_a", "module_b")
        assert isinstance(err, RuntimeContractError)
        assert err.violation_type == "dependency_missing"
        assert err.dependency == "dep_a"
        assert err.required_by == "module_b"

    def test_stage_precondition_error(self):
        err = StagePreconditionError(STAGE_ORDER_EXECUTION, ["reconciliation"])
        assert isinstance(err, RuntimeContractError)
        assert err.violation_type == "stage_precondition_violation"
        assert err.stage == STAGE_ORDER_EXECUTION
        assert "reconciliation" in err.missing_preconditions

    def test_initialization_order_error(self):
        err = InitializationOrderError("execution", ["capital"])
        assert isinstance(err, RuntimeContractError)
        assert err.violation_type == "initialization_order_error"
        assert err.module == "execution"
        assert "capital" in err.unresolved_deps

    def test_interface_signature_mismatch_error(self):
        err = InterfaceSignatureMismatchError("allocator", ["return_buffers"])
        assert isinstance(err, RuntimeContractError)
        assert err.violation_type == "interface_signature_mismatch"
        assert "return_buffers" in err.missing_attrs


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeStateRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeStateRegistry:
    def test_register_and_get_non_none(self):
        r = RuntimeStateRegistry()
        r.register("foo", 42)
        assert r.get("foo") == 42

    def test_get_unregistered_raises(self):
        r = RuntimeStateRegistry()
        with pytest.raises(RuntimeContractError) as exc_info:
            r.get("missing")
        assert exc_info.value.violation_type == "unregistered_state_access"

    def test_critical_none_raises_null_state_access(self):
        r = RuntimeStateRegistry()
        r.register("cap_state", None, is_critical=True)
        with pytest.raises(NullStateAccessError) as exc_info:
            r.get("cap_state")
        assert exc_info.value.state_name == "cap_state"

    def test_non_critical_none_returns_none(self):
        r = RuntimeStateRegistry()
        r.register("optional_state", None, is_critical=False)
        result = r.get("optional_state")
        assert result is None

    def test_validate_all_critical_none_in_violations(self):
        r = RuntimeStateRegistry()
        r.register("cap_state", None, is_critical=True)
        r.register("exp_state", None, is_critical=False)
        violations = r.validate_all()
        critical = [v for v in violations if ":CRITICAL]" in v]
        warn = [v for v in violations if ":WARN]" in v]
        assert any("cap_state" in v for v in critical)
        assert any("exp_state" in v for v in warn)

    def test_validate_all_no_violations(self):
        r = RuntimeStateRegistry()
        r.register("a", 1, is_critical=True)
        r.register("b", "x", is_critical=False)
        assert r.validate_all() == []

    def test_stale_state_warning_in_violations(self):
        r = RuntimeStateRegistry()
        past = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        entry = RegistryEntry(
            name="stale_obj",
            value={"data": "v"},
            registered_at=past,
            expected_type=None,
            is_critical=False,
            ttl_sec=100.0,
        )
        r._entries["stale_obj"] = entry
        violations = r.validate_all(now_sec=time.time())
        assert any("stale_obj" in v and "stale" in v.lower() for v in violations)

    def test_fresh_state_no_stale_violation(self):
        r = RuntimeStateRegistry()
        r.register("fresh", {"x": 1}, is_critical=False, ttl_sec=3600.0)
        violations = r.validate_all(now_sec=time.time())
        assert not any("stale" in v.lower() for v in violations)

    def test_dump_manifest_structure(self):
        r = RuntimeStateRegistry()
        r.register("x", 99, is_critical=True)
        manifest = r.dump_manifest()
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["entry_count"] == 1
        assert "x" in manifest["entries"]
        assert manifest["entries"]["x"]["is_critical"] is True
        assert manifest["entries"]["x"]["value_is_none"] is False

    def test_none_value_in_manifest(self):
        r = RuntimeStateRegistry()
        r.register("missing", None, is_critical=True)
        manifest = r.dump_manifest()
        assert manifest["entries"]["missing"]["value_is_none"] is True

    def test_multiple_entries_manifest(self):
        r = RuntimeStateRegistry()
        for i in range(5):
            r.register(f"state_{i}", i * 10)
        manifest = r.dump_manifest()
        assert manifest["entry_count"] == 5
        assert len(manifest["entries"]) == 5


# ─────────────────────────────────────────────────────────────────────────────
# IntegrationDependencyGraph
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationDependencyGraph:
    def test_declare_and_validate_satisfied(self):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("allocation", ["capital"])
        g.mark_initialized("capital")
        g.mark_initialized("allocation")
        assert g.validate() == []

    def test_undeclared_dep_is_violation(self):
        g = IntegrationDependencyGraph()
        g.declare("execution", ["capital"])
        violations = g.validate()
        assert any("capital" in v and "undeclared" in v for v in violations)

    def test_uninitialized_dep_is_violation(self):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("allocation", ["capital"])
        g.mark_initialized("allocation")
        violations = g.validate()
        assert any("capital" in v and "not initialized" in v for v in violations)

    def test_strict_order_raises_on_unresolved_dep(self):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("allocation", ["capital"])
        with pytest.raises(InitializationOrderError) as exc_info:
            g.mark_initialized("allocation", strict_order=True)
        assert "capital" in exc_info.value.unresolved_deps

    def test_strict_order_ok_when_deps_satisfied(self):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("allocation", ["capital"])
        g.mark_initialized("capital")
        g.mark_initialized("allocation", strict_order=True)
        assert g._nodes["allocation"].initialized is True

    def test_topological_order_respects_deps(self):
        g = IntegrationDependencyGraph()
        g.declare("c", [])
        g.declare("b", ["c"])
        g.declare("a", ["b"])
        order = g.topological_order()
        assert order.index("c") < order.index("b")
        assert order.index("b") < order.index("a")

    def test_topological_order_deterministic(self):
        g = IntegrationDependencyGraph()
        for name in ["z", "a", "m", "b"]:
            g.declare(name, [])
        order1 = g.topological_order()
        order2 = g.topological_order()
        assert order1 == order2

    def test_dump_manifest_has_topological_order(self):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("execution", ["capital"])
        g.mark_initialized("capital")
        manifest = g.dump_manifest()
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert "topological_order" in manifest
        assert "capital" in manifest["topological_order"]
        assert "capital" in manifest["nodes"]

    def test_no_deps_validates_clean(self):
        g = IntegrationDependencyGraph()
        g.declare("standalone", [])
        g.mark_initialized("standalone")
        assert g.validate() == []

    def test_chain_of_three_validates_in_order(self):
        g = IntegrationDependencyGraph()
        g.declare("a", [])
        g.declare("b", ["a"])
        g.declare("c", ["b"])
        g.mark_initialized("a")
        g.mark_initialized("b")
        g.mark_initialized("c")
        assert g.validate() == []


# ─────────────────────────────────────────────────────────────────────────────
# StagePreconditionValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestStagePreconditionValidator:
    def test_no_preconditions_passes(self):
        v = StagePreconditionValidator()
        assert v.validate_preconditions(STAGE_BOOTSTRAP) == []
        assert v.validate_preconditions(STAGE_API_CHECK) == []

    def test_missing_precondition_returns_violation(self):
        v = StagePreconditionValidator()
        missing = v.validate_preconditions(STAGE_BROKER_SYNC)
        assert STAGE_API_CHECK in missing

    def test_satisfied_precondition_passes(self):
        v = StagePreconditionValidator()
        v.record_stage_complete(STAGE_API_CHECK)
        assert v.validate_preconditions(STAGE_BROKER_SYNC) == []

    def test_assert_preconditions_raises_when_missing(self):
        v = StagePreconditionValidator()
        with pytest.raises(StagePreconditionError) as exc_info:
            v.assert_preconditions(STAGE_ORDER_EXECUTION)
        assert STAGE_ALLOCATION in exc_info.value.missing_preconditions

    def test_assert_preconditions_passes_when_complete(self):
        v = StagePreconditionValidator()
        for stage in [STAGE_API_CHECK, STAGE_BOOTSTRAP, STAGE_BROKER_SYNC,
                      STAGE_MARKET_DATA, STAGE_SIGNAL_GENERATION,
                      STAGE_RECONCILIATION, STAGE_ALLOCATION]:
            v.record_stage_complete(stage)
        v.assert_preconditions(STAGE_ORDER_EXECUTION)

    def test_record_stage_complete_timestamp(self):
        v = StagePreconditionValidator()
        v.record_stage_complete(STAGE_BOOTSTRAP, ts="2026-05-18T00:00:00+00:00")
        assert v._completed[STAGE_BOOTSTRAP] == "2026-05-18T00:00:00+00:00"

    def test_dump_manifest_contains_completed(self):
        v = StagePreconditionValidator()
        v.record_stage_complete(STAGE_API_CHECK)
        v.record_stage_complete(STAGE_BOOTSTRAP)
        manifest = v.dump_manifest()
        assert STAGE_API_CHECK in manifest["completed_stages"]
        assert STAGE_BOOTSTRAP in manifest["completed_stages"]
        assert "preconditions" in manifest

    def test_custom_preconditions(self):
        custom = {"stage_x": ["stage_a", "stage_b"], "stage_a": [], "stage_b": []}
        v = StagePreconditionValidator(preconditions=custom)
        v.record_stage_complete("stage_a")
        missing = v.validate_preconditions("stage_x")
        assert "stage_b" in missing
        assert "stage_a" not in missing

    def test_cleanup_requires_order_execution(self):
        v = StagePreconditionValidator()
        missing = v.validate_preconditions(STAGE_CLEANUP)
        assert STAGE_ORDER_EXECUTION in missing


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeCoherenceValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeCoherenceValidator:
    def test_check_none_critical(self):
        cv = RuntimeCoherenceValidator()
        violations = cv.check_none("cap_state", None, critical=True)
        assert any(":CRITICAL]" in v and "cap_state" in v for v in violations)

    def test_check_none_non_critical(self):
        cv = RuntimeCoherenceValidator()
        violations = cv.check_none("opt_state", None, critical=False)
        assert any(":WARN]" in v and "opt_state" in v for v in violations)

    def test_check_none_present_no_violation(self):
        cv = RuntimeCoherenceValidator()
        assert cv.check_none("x", {"v": 1}) == []

    def test_check_type_mismatch_warning(self):
        cv = RuntimeCoherenceValidator()
        violations = cv.check_type("count", "not_an_int", int)
        assert len(violations) == 1
        assert "count" in violations[0]

    def test_check_type_match_no_violation(self):
        cv = RuntimeCoherenceValidator()
        assert cv.check_type("count", 42, int) == []

    def test_check_type_none_value_no_violation(self):
        cv = RuntimeCoherenceValidator()
        assert cv.check_type("x", None, int) == []

    def test_check_required_attrs_missing(self):
        cv = RuntimeCoherenceValidator()

        class Obj:
            actual_equity = 100.0

        violations = cv.check_required_attrs(
            "cap_state", Obj(), ["actual_equity", "missing_field"]
        )
        assert any("missing_field" in v for v in violations)

    def test_check_required_attrs_all_present(self):
        cv = RuntimeCoherenceValidator()

        class Obj:
            actual_equity = 100.0
            effective_capital = 95.0
            risk_adjusted_capital = 90.0

        assert cv.check_required_attrs("cap_state", Obj(), [
            "actual_equity", "effective_capital", "risk_adjusted_capital"
        ]) == []

    def test_validate_cap_state_none_critical(self):
        cv = RuntimeCoherenceValidator()
        violations = cv.validate_cap_state(None)
        assert any(":CRITICAL]" in v for v in violations)

    def test_validate_cap_state_negative_equity_warning(self):
        cv = RuntimeCoherenceValidator()
        cap = _FakeCapState(actual_equity=-1.0, effective_capital=100.0, risk_adjusted_capital=100.0)
        violations = cv.validate_cap_state(cap)
        assert any("actual_equity" in v for v in violations)

    def test_validate_cap_state_valid(self):
        cv = RuntimeCoherenceValidator()
        cap = _FakeCapState(actual_equity=3_000_000.0, effective_capital=3_000_000.0, risk_adjusted_capital=3_000_000.0)
        violations = cv.validate_cap_state(cap)
        assert violations == []

    def test_validate_exposure_state_none_no_critical(self):
        cv = RuntimeCoherenceValidator()
        violations = cv.validate_exposure_state(None)
        assert not any(":CRITICAL]" in v for v in violations)

    def test_validate_allocator_integration_missing_attr(self):
        cv = RuntimeCoherenceValidator()

        class BadAllocator:
            pass

        violations = cv.validate_allocator_integration(BadAllocator(), None, None)
        assert any("return_buffers" in v for v in violations)

    def test_validate_allocator_integration_valid(self):
        cv = RuntimeCoherenceValidator()

        class GoodAllocator:
            return_buffers = {}

        cap = _FakeCapState(3_000_000.0, 3_000_000.0, 3_000_000.0)
        exp = _FakeExposureState(return_buffers={})
        violations = cv.validate_allocator_integration(GoodAllocator(), cap, exp)
        assert not any(":CRITICAL]" in v for v in violations)

    def test_assert_coherent_raises_on_critical(self):
        cv = RuntimeCoherenceValidator()
        violations = ["[COHERENCE:CRITICAL] something broke"]
        with pytest.raises(RuntimeContractError):
            cv.assert_coherent(violations)

    def test_assert_coherent_passes_on_warnings_only(self):
        cv = RuntimeCoherenceValidator()
        violations = ["[COHERENCE:WARN] minor issue"]
        cv.assert_coherent(violations)  # must not raise

    def test_assert_coherent_passes_on_empty(self):
        cv = RuntimeCoherenceValidator()
        cv.assert_coherent([])

    def test_validate_registry_delegates(self):
        cv = RuntimeCoherenceValidator()
        r = RuntimeStateRegistry()
        r.register("state_a", None, is_critical=True)
        violations = cv.validate_registry(r)
        assert any("state_a" in v for v in violations)


# ─────────────────────────────────────────────────────────────────────────────
# BrokerRuntimeConsistencyValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerRuntimeConsistencyValidator:
    def _val(self) -> BrokerRuntimeConsistencyValidator:
        return BrokerRuntimeConsistencyValidator()

    def test_consistent_returns_empty(self):
        v = self._val()
        violations = v.validate({"A": 100, "B": 200}, {"A": 100, "B": 200})
        assert violations == []

    def test_qty_drift_blocking(self):
        v = self._val()
        violations = v.validate({"A": 200}, {"A": 100})
        assert len(violations) == 1
        assert violations[0].violation_type == CONSISTENCY_QTY_DRIFT
        assert violations[0].severity == CSEV_BLOCKING
        assert violations[0].broker_qty == 200
        assert violations[0].runtime_qty == 100

    def test_orphan_runtime_warning(self):
        v = self._val()
        violations = v.validate({}, {"A": 100})
        assert len(violations) == 1
        assert violations[0].violation_type == CONSISTENCY_ORPHAN_RUNTIME
        assert violations[0].severity == CSEV_WARNING

    def test_orphan_broker_blocking(self):
        v = self._val()
        violations = v.validate({"A": 100}, {})
        assert len(violations) == 1
        assert violations[0].violation_type == CONSISTENCY_ORPHAN_BROKER
        assert violations[0].severity == CSEV_BLOCKING

    def test_multiple_violations(self):
        v = self._val()
        violations = v.validate(
            {"A": 100, "C": 50},
            {"A": 200, "B": 100},
        )
        types = {viol.violation_type for viol in violations}
        assert CONSISTENCY_QTY_DRIFT in types
        assert CONSISTENCY_ORPHAN_BROKER in types
        assert CONSISTENCY_ORPHAN_RUNTIME in types

    def test_intent_qty_mismatch_warning(self):
        v = self._val()
        violations = v.validate({"A": 100}, {"A": 100}, intent_map={"A": 200})
        intent_viol = [viol for viol in violations if viol.violation_type == CONSISTENCY_INTENT_QTY_MISMATCH]
        assert len(intent_viol) == 1
        assert intent_viol[0].severity == CSEV_WARNING

    def test_intent_match_no_violation(self):
        v = self._val()
        violations = v.validate({"A": 100}, {"A": 100}, intent_map={"A": 100})
        intent_viol = [viol for viol in violations if viol.violation_type == CONSISTENCY_INTENT_QTY_MISMATCH]
        assert intent_viol == []

    def test_attribution_populated(self):
        v = self._val()
        violations = v.validate({"X": 300}, {"X": 100})
        assert violations[0].attribution["drift"] == 200
        assert violations[0].attribution["broker_authoritative"] is True

    def test_build_report_consistent(self):
        v = self._val()
        violations = v.validate({"A": 100}, {"A": 100})
        report = v.build_report(violations, {"A": 100}, {"A": 100}, run_id="test_run")
        assert report.consistent is True
        assert report.blocking_count == 0
        assert report.run_id == "test_run"
        assert report.schema_version == SCHEMA_VERSION

    def test_build_report_blocking_count(self):
        v = self._val()
        broker = {"A": 200, "C": 50}
        runtime = {"A": 100, "B": 100}
        violations = v.validate(broker, runtime)
        report = v.build_report(violations, broker, runtime, run_id="r1")
        assert report.consistent is False
        assert report.blocking_count >= 1

    def test_report_contains_serialized_violations(self):
        v = self._val()
        broker = {"A": 200}
        runtime = {"A": 100}
        violations = v.validate(broker, runtime)
        report = v.build_report(violations, broker, runtime)
        assert len(report.violations) == 1
        assert report.violations[0]["violation_type"] == CONSISTENCY_QTY_DRIFT

    def test_empty_positions_no_violations(self):
        v = self._val()
        assert v.validate({}, {}) == []

    def test_sorted_symbols_deterministic(self):
        v = self._val()
        broker = {"Z": 100, "A": 200, "M": 150}
        runtime = {"Z": 0, "A": 100, "M": 150}
        violations1 = v.validate(broker, runtime)
        violations2 = v.validate(broker, runtime)
        assert [x.symbol for x in violations1] == [x.symbol for x in violations2]


# ─────────────────────────────────────────────────────────────────────────────
# Artifact persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactPersistence:
    def test_write_dependency_manifest(self, tmp_path: Path):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("execution", ["capital"])
        g.mark_initialized("capital")
        path = tmp_path / "dep_manifests.jsonl"
        write_dependency_manifest(g, path, run_id="test_001")
        records = _load_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["artifact_type"] == "dependency_manifest"
        assert r["run_id"] == "test_001"
        assert r["schema_version"] == SCHEMA_VERSION
        assert "capital" in r["topological_order"]

    def test_write_initialization_lineage(self, tmp_path: Path):
        reg = RuntimeStateRegistry()
        reg.register("cap", 100_000.0, is_critical=True)
        path = tmp_path / "lineage.jsonl"
        write_initialization_lineage(reg, path, run_id="r_42")
        records = _load_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["artifact_type"] == "initialization_lineage"
        assert r["run_id"] == "r_42"
        assert "cap" in r["entries"]

    def test_write_state_validation_report(self, tmp_path: Path):
        path = tmp_path / "reports.jsonl"
        write_state_validation_report(
            ["[CRITICAL] bad", "[WARN] meh"],
            path, run_id="r_1", stage="reconciliation"
        )
        records = _load_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["violation_count"] == 2
        assert r["stage"] == "reconciliation"

    def test_write_coherence_diagnostic(self, tmp_path: Path):
        path = tmp_path / "coherence.jsonl"
        write_coherence_diagnostic(
            ["[COHERENCE:WARN] something"],
            {"cap_state_loaded": True},
            path, run_id="r_2",
        )
        records = _load_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["artifact_type"] == "coherence_diagnostic"
        assert r["stage_state"]["cap_state_loaded"] is True

    def test_write_broker_consistency_report(self, tmp_path: Path):
        from dataclasses import asdict
        report = BrokerRuntimeConsistencyReport(
            timestamp="2026-05-18T00:00:00+00:00",
            run_id="r_3",
            violations=[],
            broker_positions={"A": 100},
            runtime_positions={"A": 100},
            consistent=True,
            blocking_count=0,
        )
        path = tmp_path / "broker_cons.jsonl"
        write_broker_consistency_report(report, path)
        records = _load_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["artifact_type"] == "broker_runtime_consistency_report"
        assert r["run_id"] == "r_3"
        assert r["consistent"] is True

    def test_write_failure_attribution(self, tmp_path: Path):
        err = RuntimeContractError("null cap_state", "null_state_access")
        path = tmp_path / "failures.jsonl"
        write_failure_attribution(err, path, run_id="r_4", stage="allocation")
        records = _load_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["artifact_type"] == "integration_failure_attribution"
        assert r["violation_type"] == "null_state_access"
        assert r["stage"] == "allocation"

    def test_append_only_multiple_writes(self, tmp_path: Path):
        path = tmp_path / "multi.jsonl"
        for i in range(5):
            write_state_validation_report([], path, run_id=f"run_{i}", stage="test")
        records = _load_jsonl(path)
        assert len(records) == 5
        run_ids = [r["run_id"] for r in records]
        assert run_ids == [f"run_{i}" for i in range(5)]

    def test_creates_parent_directory(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "out.jsonl"
        write_state_validation_report([], nested, run_id="r_x", stage="test")
        assert nested.exists()

    def test_valid_json_in_all_lines(self, tmp_path: Path):
        path = tmp_path / "valid.jsonl"
        g = IntegrationDependencyGraph()
        g.declare("x", [])
        for i in range(3):
            write_dependency_manifest(g, path, run_id=f"run_{i}")
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# validate_runtime_integration (top-level)
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateRuntimeIntegration:
    def _default_registry(self, cap_state=None, critical=True) -> RuntimeStateRegistry:
        r = RuntimeStateRegistry()
        r.register("cap_state", cap_state, is_critical=critical)
        r.register("exposure_state", None, is_critical=False)
        return r

    def test_clean_run_execution_allowed(self):
        result = validate_runtime_integration(
            run_id="clean_001",
            stage=STAGE_RECONCILIATION,
            broker_positions={"A": 100},
            runtime_positions={"A": 100},
            is_live=False,
        )
        assert isinstance(result, IntegrationValidationResult)
        assert result.execution_allowed is True
        assert result.blocking_count == 0
        assert result.run_id == "clean_001"
        assert result.schema_version == SCHEMA_VERSION

    def test_broker_drift_blocking_in_result(self):
        result = validate_runtime_integration(
            run_id="drift_001",
            stage=STAGE_RECONCILIATION,
            broker_positions={"A": 200},
            runtime_positions={"A": 100},
            is_live=False,
        )
        assert result.execution_allowed is False
        assert result.blocking_count >= 1
        assert len(result.consistency_violations) > 0

    def test_null_critical_state_blocking(self, tmp_path: Path):
        reg = RuntimeStateRegistry()
        reg.register("cap_state", None, is_critical=True)
        result = validate_runtime_integration(
            run_id="null_001",
            stage=STAGE_ALLOCATION,
            registry=reg,
            is_live=False,
            artifact_dir=tmp_path,
        )
        assert result.execution_allowed is False
        assert len(result.registry_violations) > 0

    def test_live_mode_raises_on_blocking_violation(self):
        with pytest.raises(RuntimeContractError):
            validate_runtime_integration(
                run_id="live_fail_001",
                stage=STAGE_RECONCILIATION,
                broker_positions={"A": 200},
                runtime_positions={"A": 100},
                is_live=True,
            )

    def test_dry_run_does_not_raise_on_blocking(self):
        result = validate_runtime_integration(
            run_id="dr_fail_001",
            stage=STAGE_RECONCILIATION,
            broker_positions={"A": 200},
            runtime_positions={"A": 100},
            is_live=False,
        )
        assert result.execution_allowed is False

    def test_artifacts_written_to_dir(self, tmp_path: Path):
        reg = RuntimeStateRegistry()
        reg.register("cap_state", 100.0, is_critical=True)
        validate_runtime_integration(
            run_id="artifact_001",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            broker_positions={"A": 100},
            runtime_positions={"A": 100},
            is_live=False,
            artifact_dir=tmp_path,
        )
        assert (tmp_path / "initialization_lineage.jsonl").exists()
        assert (tmp_path / "state_validation_reports.jsonl").exists()
        assert (tmp_path / "broker_runtime_consistency_reports.jsonl").exists()
        assert (tmp_path / "runtime_coherence_diagnostics.jsonl").exists()

    def test_dep_graph_manifest_written(self, tmp_path: Path):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.mark_initialized("capital")
        validate_runtime_integration(
            run_id="dep_001",
            stage=STAGE_ALLOCATION,
            dep_graph=g,
            is_live=False,
            artifact_dir=tmp_path,
        )
        assert (tmp_path / "dependency_manifests.jsonl").exists()
        records = _load_jsonl(tmp_path / "dependency_manifests.jsonl")
        assert records[0]["run_id"] == "dep_001"

    def test_failure_attribution_written_on_live_error(self, tmp_path: Path):
        with pytest.raises(RuntimeContractError):
            validate_runtime_integration(
                run_id="fail_attr_001",
                stage=STAGE_RECONCILIATION,
                broker_positions={"X": 500},
                runtime_positions={},
                is_live=True,
                artifact_dir=tmp_path,
            )
        assert (tmp_path / "integration_failure_attribution.jsonl").exists()
        records = _load_jsonl(tmp_path / "integration_failure_attribution.jsonl")
        assert any(r["run_id"] == "fail_attr_001" for r in records)

    def test_all_violations_aggregated(self, tmp_path: Path):
        reg = RuntimeStateRegistry()
        reg.register("cap_state", None, is_critical=True)
        result = validate_runtime_integration(
            run_id="agg_001",
            stage=STAGE_RECONCILIATION,
            registry=reg,
            broker_positions={"A": 200},
            runtime_positions={"A": 100},
            is_live=False,
            artifact_dir=tmp_path,
        )
        assert len(result.all_violations) >= 2

    def test_no_broker_positions_skips_consistency(self):
        result = validate_runtime_integration(
            run_id="no_broker_001",
            stage=STAGE_RECONCILIATION,
            broker_positions=None,
            runtime_positions=None,
            is_live=False,
        )
        assert result.consistency_violations == []

    def test_precondition_violation_in_result(self):
        pre = StagePreconditionValidator()
        result = validate_runtime_integration(
            run_id="pre_001",
            stage=STAGE_ORDER_EXECUTION,
            precondition_val=pre,
            is_live=False,
        )
        assert len(result.precondition_violations) > 0

    def test_result_schema_version(self):
        result = validate_runtime_integration(
            run_id="schema_001",
            stage=STAGE_BOOTSTRAP,
            is_live=False,
        )
        assert result.schema_version == SCHEMA_VERSION

    def test_result_timestamp_is_iso(self):
        result = validate_runtime_integration(
            run_id="ts_001",
            stage=STAGE_BOOTSTRAP,
            is_live=False,
        )
        datetime.fromisoformat(result.timestamp)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Replay determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayDeterminism:
    def _build_state(self):
        g = IntegrationDependencyGraph()
        g.declare("capital", [])
        g.declare("allocation", ["capital"])
        g.mark_initialized("capital")
        g.mark_initialized("allocation")

        r = RuntimeStateRegistry()
        r.register("cap_state", 3_000_000.0, is_critical=True)
        r.register("exp_state", None, is_critical=False)

        pre = StagePreconditionValidator()
        for s in [STAGE_API_CHECK, STAGE_BOOTSTRAP, STAGE_BROKER_SYNC,
                  STAGE_MARKET_DATA, STAGE_SIGNAL_GENERATION,
                  STAGE_RECONCILIATION, STAGE_ALLOCATION]:
            pre.record_stage_complete(s)

        return g, r, pre

    def test_same_inputs_same_violations(self):
        broker = {"A": 100, "B": 200}
        runtime = {"A": 100, "B": 150}

        results = []
        for _ in range(3):
            g, r, pre = self._build_state()
            res = validate_runtime_integration(
                run_id="replay_001",
                stage=STAGE_ORDER_EXECUTION,
                registry=r,
                dep_graph=g,
                precondition_val=pre,
                broker_positions=broker,
                runtime_positions=runtime,
                is_live=False,
            )
            results.append(res)

        violation_sets = [set(r.all_violations) for r in results]
        assert violation_sets[0] == violation_sets[1] == violation_sets[2]

    def test_same_inputs_same_blocking_count(self):
        broker = {"X": 300}
        runtime = {"X": 200}

        blocking_counts = []
        for _ in range(3):
            g, r, pre = self._build_state()
            res = validate_runtime_integration(
                run_id="replay_002",
                stage=STAGE_ORDER_EXECUTION,
                registry=r,
                dep_graph=g,
                precondition_val=pre,
                broker_positions=broker,
                runtime_positions=runtime,
                is_live=False,
            )
            blocking_counts.append(res.blocking_count)

        assert len(set(blocking_counts)) == 1

    def test_dep_graph_topological_order_deterministic(self):
        orders = []
        for _ in range(5):
            g = IntegrationDependencyGraph()
            g.declare("d", [])
            g.declare("c", ["d"])
            g.declare("b", ["c"])
            g.declare("a", ["b"])
            orders.append(g.topological_order())
        assert all(o == orders[0] for o in orders)

    def test_broker_violations_sorted_deterministic(self):
        v = BrokerRuntimeConsistencyValidator()
        broker = {"Z": 100, "A": 200, "M": 150, "B": 50}
        runtime = {"Z": 0, "A": 100, "M": 0, "B": 50}

        symbol_orders = []
        for _ in range(5):
            violations = v.validate(broker, runtime)
            symbol_orders.append([viol.symbol for viol in violations])

        assert all(s == symbol_orders[0] for s in symbol_orders)

    def test_artifact_content_deterministic(self, tmp_path: Path):
        broker = {"A": 100}
        runtime = {"A": 100}

        for i in range(3):
            path_i = tmp_path / f"run_{i}"
            path_i.mkdir()
            validate_runtime_integration(
                run_id="det_001",
                stage=STAGE_RECONCILIATION,
                broker_positions=broker,
                runtime_positions=runtime,
                is_live=False,
                artifact_dir=path_i,
            )

        records = [
            _load_jsonl(tmp_path / f"run_{i}" / "broker_runtime_consistency_reports.jsonl")[0]
            for i in range(3)
        ]
        for r in records:
            assert r["consistent"] is True
            assert r["blocking_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Fail policy correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestFailPolicy:
    def test_analytics_violation_no_raise_dry_run(self):
        reg = RuntimeStateRegistry()
        reg.register("opt_state", None, is_critical=False)
        result = validate_runtime_integration(
            run_id="fp_001",
            stage=STAGE_BOOTSTRAP,
            registry=reg,
            is_live=False,
        )
        assert isinstance(result, IntegrationValidationResult)

    def test_analytics_violation_no_raise_live(self):
        reg = RuntimeStateRegistry()
        reg.register("opt_state", None, is_critical=False)
        result = validate_runtime_integration(
            run_id="fp_002",
            stage=STAGE_BOOTSTRAP,
            registry=reg,
            is_live=True,
        )
        assert isinstance(result, IntegrationValidationResult)

    def test_critical_violation_no_raise_dry_run(self):
        reg = RuntimeStateRegistry()
        reg.register("cap_state", None, is_critical=True)
        result = validate_runtime_integration(
            run_id="fp_003",
            stage=STAGE_ALLOCATION,
            registry=reg,
            is_live=False,
        )
        assert not result.execution_allowed

    def test_critical_violation_raises_live(self):
        reg = RuntimeStateRegistry()
        reg.register("cap_state", None, is_critical=True)
        with pytest.raises(RuntimeContractError):
            validate_runtime_integration(
                run_id="fp_004",
                stage=STAGE_ALLOCATION,
                registry=reg,
                is_live=True,
            )

    def test_blocking_broker_violation_raises_live(self):
        with pytest.raises(RuntimeContractError) as exc_info:
            validate_runtime_integration(
                run_id="fp_005",
                stage=STAGE_ORDER_EXECUTION,
                broker_positions={"UNKNOWN_SYM": 100},
                runtime_positions={},
                is_live=True,
            )
        assert isinstance(exc_info.value, RuntimeContractError)

    def test_warning_only_broker_violation_no_raise_live(self):
        result = validate_runtime_integration(
            run_id="fp_006",
            stage=STAGE_ORDER_EXECUTION,
            broker_positions={},
            runtime_positions={"A": 100},
            is_live=True,
        )
        assert isinstance(result, IntegrationValidationResult)
        orphan = [v for v in result.consistency_violations
                  if v.get("violation_type") == CONSISTENCY_ORPHAN_RUNTIME]
        assert len(orphan) == 1

    def test_execution_allowed_true_on_clean_live(self):
        result = validate_runtime_integration(
            run_id="fp_007",
            stage=STAGE_RECONCILIATION,
            broker_positions={"A": 100},
            runtime_positions={"A": 100},
            is_live=True,
        )
        assert result.execution_allowed is True

    def test_error_has_correct_violation_type(self):
        try:
            validate_runtime_integration(
                run_id="fp_008",
                stage=STAGE_RECONCILIATION,
                broker_positions={"Q": 100},
                runtime_positions={},
                is_live=True,
            )
            pytest.fail("expected RuntimeContractError")
        except RuntimeContractError as e:
            assert e.violation_type in {
                "consistency_violation",
                "runtime_contract_violation",
                "null_state_access",
                "contract_violation",
                CONSISTENCY_ORPHAN_BROKER,
                CONSISTENCY_QTY_DRIFT,
                CONSISTENCY_ORPHAN_RUNTIME,
                CONSISTENCY_INTENT_QTY_MISMATCH,
            }
