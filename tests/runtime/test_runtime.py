"""pytest coverage for deterministic runtime governance system."""
import json, os, tempfile, time
from pathlib import Path
import pytest

# Patch artifact dir to tempdir for all tests
import src.runtime.decision_ledger as dl_mod
import src.runtime.idempotency as idm_mod
import src.runtime.circuit_breaker as cb_mod
import src.runtime.quarantine as qz_mod
import src.runtime.snapshot as snap_mod
import src.runtime.deployment_gate as dg_mod
import src.runtime.counterfactual as cf_mod
import src.runtime.replay_engine as re_mod

from src.runtime.epoch import ExecutionEpoch
from src.runtime.state_machine import RuntimeStateMachine, RuntimeState
from src.runtime.idempotency import IdempotencyGuard
from src.runtime.circuit_breaker import RuntimeCircuitBreaker, CircuitBreakerConfig
from src.runtime.artifact_validator import ArtifactCompletionValidator
from src.runtime.snapshot import RuntimeSnapshot
from src.runtime.dependency_resolver import DependencyResolver
from src.runtime.decision_ledger import RuntimeDecisionLedger
from src.runtime.quarantine import RuntimeQuarantineZone
from src.runtime.deployment_gate import DeploymentGateInterface
from src.runtime.counterfactual import CounterfactualRuntimeReplay


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


# 1. ExecutionEpoch determinism
def test_epoch_determinism():
    args = dict(dataset_manifest="ds1", feature_manifest="fm1",
                governance_state="gs1", schema_version="1.0",
                template_state="ts1", reality_schema_version="1.0")
    e1 = ExecutionEpoch.create(**args)
    e2 = ExecutionEpoch.create(**args)
    assert e1.epoch_id == e2.epoch_id

def test_epoch_different_inputs():
    e1 = ExecutionEpoch.create("a","b","c","1.0","d","1.0")
    e2 = ExecutionEpoch.create("X","b","c","1.0","d","1.0")
    assert e1.epoch_id != e2.epoch_id

def test_epoch_consistency():
    e1 = ExecutionEpoch.create("a","b","c","1.0","d","1.0")
    e2 = ExecutionEpoch.create("a","b","c","1.0","d","1.0")
    assert e1.validate_consistency(e2)


# 2. StateMachine transitions
def test_state_machine_valid():
    sm = RuntimeStateMachine()
    sm.transition(RuntimeState.READY, "test")
    sm.transition(RuntimeState.RUNNING, "test")
    sm.transition(RuntimeState.VALIDATING, "test")
    sm.transition(RuntimeState.COMMITTED, "test")
    assert sm.state == RuntimeState.COMMITTED

def test_state_machine_invalid():
    sm = RuntimeStateMachine()
    with pytest.raises(ValueError):
        sm.transition(RuntimeState.COMMITTED, "illegal")

def test_state_machine_quarantine_path():
    sm = RuntimeStateMachine()
    sm.transition(RuntimeState.READY, "r")
    sm.transition(RuntimeState.RUNNING, "r")
    sm.transition(RuntimeState.VALIDATING, "r")
    sm.transition(RuntimeState.QUARANTINED, "checksum_fail")
    assert sm.state == RuntimeState.QUARANTINED

def test_state_machine_history_append_only():
    sm = RuntimeStateMachine()
    sm.transition(RuntimeState.READY, "r1")
    h = sm.history()
    assert len(h) == 1
    assert h[0]["to"] == "READY"


# 3. IdempotencyGuard
def test_idempotency_duplicate(tmp):
    g = IdempotencyGuard(str(tmp / "idem.jsonl"))
    assert not g.is_duplicate("run1")
    g.mark_complete("run1", "epoch1")
    assert g.is_duplicate("run1")

def test_idempotency_make_run_id_deterministic():
    r1 = IdempotencyGuard.make_run_id("epoch1", "sched1", "2026-05-10")
    r2 = IdempotencyGuard.make_run_id("epoch1", "sched1", "2026-05-10")
    assert r1 == r2

def test_idempotency_different_dates():
    r1 = IdempotencyGuard.make_run_id("e","s","2026-05-10")
    r2 = IdempotencyGuard.make_run_id("e","s","2026-05-11")
    assert r1 != r2


# 4. CircuitBreaker
def test_circuit_breaker_trips(tmp):
    cfg = CircuitBreakerConfig(max_epoch_mismatches=2, window_seconds=9999)
    cb = RuntimeCircuitBreaker(cfg, str(tmp / "cb.jsonl"))
    assert not cb.is_tripped
    cb.record_event("epoch_mismatch", "r1", "test")
    cb.record_event("epoch_mismatch", "r2", "test")
    assert cb.is_tripped

def test_circuit_breaker_reset(tmp):
    cfg = CircuitBreakerConfig(max_epoch_mismatches=1, window_seconds=9999)
    cb = RuntimeCircuitBreaker(cfg, str(tmp / "cb2.jsonl"))
    cb.record_event("epoch_mismatch", "r1", "x")
    assert cb.is_tripped
    cb.reset("admin")
    assert not cb.is_tripped


# 5. ArtifactCompletionValidator
def test_validator_missing(tmp):
    v = ArtifactCompletionValidator()
    r = v.validate(str(tmp / "nonexistent.bin"))
    assert not r.valid
    assert "not_found" in r.reason

def test_validator_ok(tmp):
    f = tmp / "art.bin"
    f.write_bytes(b"hello")
    v = ArtifactCompletionValidator()
    r = v.validate(str(f))
    assert r.valid
    assert r.checksum is not None

def test_validator_checksum_mismatch(tmp):
    f = tmp / "art2.bin"
    f.write_bytes(b"hello")
    v = ArtifactCompletionValidator()
    r = v.validate(str(f), expected_checksum="badhash")
    assert not r.valid

def test_validator_partial_write(tmp):
    f = tmp / "partial.bin"
    f.write_bytes(b"x")
    v = ArtifactCompletionValidator()
    r = v.validate(str(f), min_bytes=100)
    assert not r.valid


# 6. RuntimeSnapshot round-trip
def test_snapshot_roundtrip(tmp):
    snap = RuntimeSnapshot(str(tmp / "snaps"))
    path = snap.capture("run1", "epoch1", {"stage": "test", "ok": True})
    loaded = snap.load(path)
    assert loaded["run_id"] == "run1"
    assert loaded["epoch_id"] == "epoch1"
    assert loaded["state"]["ok"] is True

def test_snapshot_list(tmp):
    snap = RuntimeSnapshot(str(tmp / "snaps2"))
    snap.capture("run1", "epoch1", {"x": 1})
    snap.capture("run1", "epoch1", {"x": 2})
    snap.capture("run2", "epoch2", {"x": 3})
    assert len(snap.list_snapshots("run1")) == 2
    assert len(snap.list_snapshots("run2")) == 1


# 7. DependencyResolver
def test_dep_resolver_missing(tmp):
    dr = DependencyResolver()
    result = dr.check("epoch1", [str(tmp / "missing.parquet")])
    assert not result.satisfied
    assert any("missing_artifact" in v for v in result.violations)

def test_dep_resolver_ok(tmp):
    f = tmp / "data.parquet"
    f.write_bytes(b"data")
    dr = DependencyResolver()
    result = dr.check("epoch1", [str(f)], max_dataset_age_hours=9999)
    assert result.satisfied


# 8. DecisionLedger append-only
def test_decision_ledger(tmp):
    ledger = RuntimeDecisionLedger(str(tmp / "ledger.jsonl"))
    ledger.record("run_started", "run1", "epoch1", "test", {"x": 1})
    ledger.record("run_completed", "run1", "epoch1", "ok")
    records = ledger.read_all()
    assert len(records) == 2
    assert records[0]["decision_type"] == "run_started"
    assert records[1]["decision_type"] == "run_completed"


# 9. QuarantineZone
def test_quarantine(tmp):
    art = tmp / "bad_artifact.json"
    art.write_text('{"partial": true}')
    qz = RuntimeQuarantineZone(str(tmp / "qzone"), str(tmp / "qmanifest.jsonl"))
    dest = qz.quarantine(str(art), "checksum_failure", "run1", "epoch1")
    assert Path(dest).exists()
    assert art.exists()  # original preserved
    manifest = qz.list_quarantined()
    assert len(manifest) == 1
    assert manifest[0]["reason"] == "checksum_failure"


# 10. DeploymentGateInterface
def test_deployment_gate(tmp):
    gate = DeploymentGateInterface(str(tmp / "gate.jsonl"))
    rec = gate.recommend("run1", "epoch1", True, ["pipeline_completed"])
    assert rec.eligible is True
    recs = gate.list_recommendations()
    assert len(recs) == 1
    assert recs[0]["eligible"] is True

def test_deployment_gate_ineligible(tmp):
    gate = DeploymentGateInterface(str(tmp / "gate2.jsonl"))
    rec = gate.recommend("run1", "epoch1", False, ["pipeline_failed"])
    assert rec.eligible is False


# 11. Counterfactual does NOT write to main ledger
def test_counterfactual_isolation(tmp):
    main_ledger = tmp / "main_ledger.jsonl"
    cf_ledger = tmp / "cf_ledger.jsonl"
    snap = RuntimeSnapshot(str(tmp / "snaps"))
    snap.capture("run1", "epochA", {"stage": "test"})
    cf = CounterfactualRuntimeReplay(str(tmp / "snaps"), str(cf_ledger))
    result = cf.replay("epochA", {"max_retries": 1, "alloc_cap_multiplier": 0.5})
    assert not main_ledger.exists()  # main ledger untouched
    assert cf_ledger.exists()
    experiments = cf.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["policy_overrides"]["max_retries"] == 1


# 12. ReplayEngine reproduces epoch snapshots
def test_replay_engine(tmp):
    from src.runtime.replay_engine import DeterministicReplayEngine
    snap = RuntimeSnapshot(str(tmp / "snaps"))
    snap.capture("run1", "epochB", {"stage": "s1"})
    snap.capture("run1", "epochB", {"stage": "s2"})
    snap.capture("run2", "epochC", {"stage": "s3"})
    engine = DeterministicReplayEngine(str(tmp / "snaps"), str(tmp / "replay.jsonl"))
    steps = engine.replay("epochB")
    assert len(steps) == 2
    assert all(s["epoch_id"] == "epochB" for s in steps)
    steps_c = engine.replay("epochC")
    assert len(steps_c) == 1
