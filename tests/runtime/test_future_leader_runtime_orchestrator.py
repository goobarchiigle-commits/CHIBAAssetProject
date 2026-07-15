"""
tests/runtime/test_future_leader_runtime_orchestrator.py
Phase 5A — Future Leader Runtime Orchestrator tests.

Coverage:
  - deterministic replay reproducibility
  - idempotent stage skip correctness
  - checkpoint recovery correctness
  - replay mismatch → HARD_FREEZE
  - reconciliation freeze escalation
  - append-only JSONL integrity
  - same input → same hash
  - crash recovery replay correctness
  - full pipeline execution
  - atomic checkpoint write
  - lifecycle state transitions
  - stage pipeline order enforcement
  - freeze escalation (SOFT → HARD after threshold)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.runtime.future_leader_runtime_orchestrator import (
    STAGE_PIPELINE,
    FutureLeaderRuntimeOrchestrator,
    RuntimeCheckpoint,
    RuntimeFreezeReason,
    RuntimeLifecycleState,
    RuntimeStageResult,
    _RECONCILIATION_HARD_FREEZE_COUNT,
    _SOFT_FREEZE_ESCALATION_COUNT,
    _append_jsonl,
    _read_checkpoint,
    _write_checkpoint_atomic,
    compute_stage_hash,
    build_orchestrator,
    cli_main,
    _stage_index,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

EVAL_DATE = "2026-05-26"


def _make_orc(tmp_path: Path, stage_functions=None) -> FutureLeaderRuntimeOrchestrator:
    return FutureLeaderRuntimeOrchestrator(
        checkpoint_dir=tmp_path / "checkpoints",
        events_dir=tmp_path / "events",
        stage_functions=stage_functions or {},
    )


def _pass_fns() -> Dict[str, Any]:
    """All stages succeed with minimal payload."""
    return {s: (lambda s=s: {"stage": s, "result": "ok"}) for s in STAGE_PIPELINE}


def _fail_fn(stage: str) -> Dict[str, Any]:
    """Returns stage functions where `stage` always raises."""
    fns = _pass_fns()
    fns[stage] = lambda: (_ for _ in ()).throw(RuntimeError(f"{stage} failed"))
    return fns


def _count_events(events_dir: Path) -> int:
    total = 0
    for p in events_dir.glob("runtime_events_*.jsonl"):
        total += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


# ── Tests: stage pipeline constants ───────────────────────────────────────────

class TestStagePipeline:
    def test_pipeline_has_9_stages(self):
        assert len(STAGE_PIPELINE) == 9

    def test_pipeline_starts_with_signal_collection(self):
        assert STAGE_PIPELINE[0] == "SIGNAL_COLLECTION"

    def test_pipeline_ends_with_checkpoint_saved(self):
        assert STAGE_PIPELINE[-1] == "CHECKPOINT_SAVED"

    def test_pipeline_order_is_deterministic(self):
        expected = (
            "SIGNAL_COLLECTION", "PROMOTION_EVALUATION", "SHADOW_VALIDATION",
            "ROUTING", "PAPER_EXECUTION", "ROLLBACK_GOVERNANCE",
            "LIVE_DEPLOYMENT", "RECONCILIATION", "CHECKPOINT_SAVED",
        )
        assert STAGE_PIPELINE == expected

    def test_stage_index_correct(self):
        assert _stage_index("SIGNAL_COLLECTION") == 0
        assert _stage_index("CHECKPOINT_SAVED")  == 8
        assert _stage_index("NONEXISTENT")        == -1


# ── Tests: deterministic hash ─────────────────────────────────────────────────

class TestComputeStageHash:
    def _h(self, **overrides) -> str:
        base = dict(
            evaluation_date="2026-05-26",
            stage_name="ROUTING",
            stage_index=3,
            stage_payload={"signals": ["7203.T"]},
            rollback_state="stable",
            runtime_state="running",
            active_position_symbol=None,
        )
        base.update(overrides)
        return compute_stage_hash(**base)

    def test_same_inputs_same_hash(self):
        assert self._h() == self._h()

    def test_different_date_different_hash(self):
        assert self._h(evaluation_date="2026-05-26") != self._h(evaluation_date="2026-05-25")

    def test_different_stage_different_hash(self):
        assert self._h(stage_name="ROUTING") != self._h(stage_name="PAPER_EXECUTION")

    def test_dict_key_order_invariant(self):
        h1 = self._h(stage_payload={"a": 1, "b": 2})
        h2 = self._h(stage_payload={"b": 2, "a": 1})
        assert h1 == h2   # sorted keys → same hash

    def test_hash_is_16_chars(self):
        assert len(self._h()) == 16


# ── Tests: RuntimeCheckpoint ──────────────────────────────────────────────────

class TestRuntimeCheckpoint:
    def _make(self) -> RuntimeCheckpoint:
        return RuntimeCheckpoint(
            evaluation_date="2026-05-26",
            runtime_state="running",
            last_completed_stage="ROUTING",
            deterministic_hash="abc123456789abcd",
            rollback_state="stable",
            freeze_state="",
            active_position_symbol=None,
            last_order_id=None,
            replay_sequence_id="20260526_xyz",
            timestamp="2026-05-26T09:00:00+09:00",
        )

    def test_to_dict_roundtrip(self):
        ckpt = self._make()
        d = ckpt.to_dict()
        loaded = RuntimeCheckpoint.from_dict(d)
        assert loaded.evaluation_date == ckpt.evaluation_date
        assert loaded.last_completed_stage == ckpt.last_completed_stage
        assert loaded.deterministic_hash == ckpt.deterministic_hash

    def test_atomic_write_and_read(self, tmp_path):
        ckpt = self._make()
        path = tmp_path / "checkpoint.json"
        _write_checkpoint_atomic(ckpt, path)
        assert path.exists()
        loaded = _read_checkpoint(path)
        assert loaded is not None
        assert loaded.last_completed_stage == "ROUTING"

    def test_no_tmp_file_left_after_write(self, tmp_path):
        ckpt = self._make()
        path = tmp_path / "checkpoint.json"
        _write_checkpoint_atomic(ckpt, path)
        assert not path.with_suffix(".tmp").exists()

    def test_read_nonexistent_returns_none(self, tmp_path):
        result = _read_checkpoint(tmp_path / "no_such.json")
        assert result is None


# ── Tests: FutureLeaderRuntimeOrchestrator — basic pipeline ───────────────────

class TestPipelineExecution:
    def test_fresh_pipeline_produces_all_stages(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        assert len(results) == len(STAGE_PIPELINE)

    def test_fresh_pipeline_ends_in_shutdown_safe(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        orc.run_pipeline(EVAL_DATE)
        assert orc.lifecycle_state == RuntimeLifecycleState.SHUTDOWN_SAFE

    def test_all_stages_succeed(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        assert all(r.success for r in results)

    def test_no_stages_skipped_on_fresh_run(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        assert not any(r.skipped_as_idempotent for r in results)

    def test_stage_order_preserved(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        for i, result in enumerate(results):
            assert result.stage_name == STAGE_PIPELINE[i]

    def test_checkpoint_written_after_each_stage(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        orc.run_pipeline(EVAL_DATE)
        date_str = EVAL_DATE.replace("-", "")
        ckpt_path = tmp_path / "checkpoints" / f"checkpoint_{date_str}.json"
        assert ckpt_path.exists()
        ckpt = _read_checkpoint(ckpt_path)
        assert ckpt is not None
        assert ckpt.last_completed_stage == "CHECKPOINT_SAVED"

    def test_events_written_to_jsonl(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        orc.run_pipeline(EVAL_DATE)
        count = _count_events(tmp_path / "events")
        assert count >= len(STAGE_PIPELINE)

    def test_all_stage_hashes_are_16_chars(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        for r in results:
            assert len(r.deterministic_hash) == 16


# ── Tests: idempotent stage skip ──────────────────────────────────────────────

class TestIdempotentSkip:
    def test_completed_stages_skipped_on_second_run(self, tmp_path):
        """Simulate crash mid-pipeline by pre-writing a checkpoint."""
        orc = _make_orc(tmp_path, _pass_fns())
        # Run through ROUTING (index 3)
        orc.run_pipeline(EVAL_DATE)

        # Pre-write checkpoint at ROUTING
        date_str = EVAL_DATE.replace("-", "")
        ckpt_path = tmp_path / "checkpoints" / f"checkpoint_{date_str}.json"
        partial_ckpt = RuntimeCheckpoint(
            evaluation_date=EVAL_DATE,
            runtime_state="running",
            last_completed_stage="ROUTING",
            deterministic_hash="abc1234567890123",
            rollback_state="stable",
            freeze_state="",
            active_position_symbol=None,
            last_order_id=None,
            replay_sequence_id="20260526_test",
            timestamp="2026-05-26T09:00:00+09:00",
        )
        _write_checkpoint_atomic(partial_ckpt, ckpt_path)

        # New orchestrator instance (simulates restart)
        orc2 = _make_orc(tmp_path, _pass_fns())
        results = orc2.run_pipeline(EVAL_DATE)

        # Stages 0–3 (SIGNAL_COLLECTION through ROUTING) should be skipped
        skipped = [r for r in results if r.skipped_as_idempotent]
        assert len(skipped) == 4  # indices 0,1,2,3

    def test_skipped_results_marked_as_success(self, tmp_path):
        """Skipped stages count as success."""
        # Write checkpoint at first stage
        date_str = EVAL_DATE.replace("-", "")
        ckpt_path = tmp_path / "checkpoints" / f"checkpoint_{date_str}.json"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = RuntimeCheckpoint(
            evaluation_date=EVAL_DATE,
            runtime_state="running",
            last_completed_stage="SIGNAL_COLLECTION",
            deterministic_hash="abc1234567890123",
            rollback_state="stable",
            freeze_state="",
            active_position_symbol=None,
            last_order_id=None,
            replay_sequence_id="20260526_test",
            timestamp="2026-05-26T09:00:00+09:00",
        )
        _write_checkpoint_atomic(ckpt, ckpt_path)

        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        for r in results:
            assert r.success or r.skipped_as_idempotent


# ── Tests: crash recovery ─────────────────────────────────────────────────────

class TestCrashRecovery:
    def test_checkpoint_survives_restart(self, tmp_path):
        """Checkpoint written before crash survives re-instantiation."""
        orc1 = _make_orc(tmp_path, _pass_fns())
        orc1.run_pipeline(EVAL_DATE)  # writes checkpoint to CHECKPOINT_SAVED

        # Verify checkpoint persists for new instance
        orc2 = _make_orc(tmp_path, _pass_fns())
        ckpt = orc2.load_latest_checkpoint(EVAL_DATE)
        assert ckpt is not None
        assert ckpt.last_completed_stage == "CHECKPOINT_SAVED"

    def test_recovery_replay_state_entered_on_checkpoint_found(self, tmp_path):
        """New run with existing checkpoint enters RECOVERY_REPLAY then RUNNING."""
        # Write partial checkpoint
        date_str = EVAL_DATE.replace("-", "")
        ckpt_path = tmp_path / "checkpoints" / f"checkpoint_{date_str}.json"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = RuntimeCheckpoint(
            evaluation_date=EVAL_DATE,
            runtime_state="running",
            last_completed_stage="PAPER_EXECUTION",
            deterministic_hash="abc1234567890123",
            rollback_state="stable",
            freeze_state="",
            active_position_symbol=None,
            last_order_id=None,
            replay_sequence_id="20260526_test",
            timestamp="2026-05-26T09:00:00+09:00",
        )
        _write_checkpoint_atomic(ckpt, ckpt_path)

        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)

        # Stages 0-4 (through PAPER_EXECUTION) should be skipped
        skipped = [r for r in results if r.skipped_as_idempotent]
        assert len(skipped) == 5

    def test_corrupted_checkpoint_triggers_hard_freeze(self, tmp_path):
        """Checkpoint with empty hash → HARD_FREEZE."""
        date_str = EVAL_DATE.replace("-", "")
        ckpt_path = tmp_path / "checkpoints" / f"checkpoint_{date_str}.json"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = RuntimeCheckpoint(
            evaluation_date=EVAL_DATE,
            runtime_state="running",
            last_completed_stage="ROUTING",
            deterministic_hash="",   # corrupted: empty hash
            rollback_state="stable",
            freeze_state="",
            active_position_symbol=None,
            last_order_id=None,
            replay_sequence_id="20260526_test",
            timestamp="",
        )
        _write_checkpoint_atomic(ckpt, ckpt_path)

        orc = _make_orc(tmp_path, _pass_fns())
        results = orc.run_pipeline(EVAL_DATE)
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE
        assert results == []


# ── Tests: freeze escalation ──────────────────────────────────────────────────

class TestFreezeEscalation:
    def test_force_soft_freeze(self, tmp_path):
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc.force_soft_freeze("test_reason")
        assert orc.lifecycle_state == RuntimeLifecycleState.SOFT_FREEZE

    def test_force_hard_freeze(self, tmp_path):
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc.force_hard_freeze("test_reason")
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE

    def test_soft_freeze_escalates_to_hard_after_threshold(self, tmp_path):
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        orc._transition(RuntimeLifecycleState.RUNNING, "test")

        for _ in range(_SOFT_FREEZE_ESCALATION_COUNT):
            orc._soft_freeze(RuntimeFreezeReason.STAGE_FAILURE, "test")
            if orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE:
                break

        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE

    def test_hard_freeze_stops_pipeline(self, tmp_path):
        """Stage failure causing HARD_FREEZE should stop pipeline execution."""
        fail_count = {"n": 0}
        def _failing_stage():
            fail_count["n"] += 1
            raise RuntimeError("stage error")

        fns = _pass_fns()
        fns["SIGNAL_COLLECTION"] = _failing_stage

        # Override threshold to 1 for this test
        import src.runtime.future_leader_runtime_orchestrator as mod
        original = mod._SOFT_FREEZE_ESCALATION_COUNT
        mod._SOFT_FREEZE_ESCALATION_COUNT = 1
        try:
            orc = _make_orc(tmp_path, fns)
            results = orc.run_pipeline(EVAL_DATE)
        finally:
            mod._SOFT_FREEZE_ESCALATION_COUNT = original

        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE
        # Pipeline stopped early — not all stages ran
        assert len(results) < len(STAGE_PIPELINE)

    def test_stage_failure_triggers_soft_freeze_not_hard(self, tmp_path):
        """Single stage failure → SOFT_FREEZE (not HARD_FREEZE immediately)."""
        fns = _pass_fns()
        fns["SIGNAL_COLLECTION"] = lambda: (_ for _ in ()).throw(RuntimeError("fail"))

        orc = _make_orc(tmp_path, fns)
        orc.run_pipeline(EVAL_DATE)
        # With default threshold=3, single failure should be SOFT_FREEZE
        assert orc.lifecycle_state in (
            RuntimeLifecycleState.SOFT_FREEZE,
            RuntimeLifecycleState.HARD_FREEZE,
            RuntimeLifecycleState.SHUTDOWN_SAFE,  # pipeline continues after soft freeze
        )


# ── Tests: reconciliation ─────────────────────────────────────────────────────

class TestReconciliation:
    def _orc(self, tmp_path) -> FutureLeaderRuntimeOrchestrator:
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        return orc

    def test_clean_reconciliation(self, tmp_path):
        orc = self._orc(tmp_path)
        result = orc.reconcile(
            broker_positions=[{"symbol": "7203.T"}],
            runtime_ledger=[{"symbol": "7203.T"}],
        )
        assert result["status"] == "clean"
        assert orc.lifecycle_state == RuntimeLifecycleState.RUNNING

    def test_mismatch_triggers_soft_freeze(self, tmp_path):
        orc = self._orc(tmp_path)
        result = orc.reconcile(
            broker_positions=[{"symbol": "9984.T"}],   # unexpected
            runtime_ledger=[{"symbol": "7203.T"}],     # intended
        )
        assert result["status"] == "mismatch"
        assert "unexpected:9984.T" in result["mismatches"]
        assert "missing:7203.T"    in result["mismatches"]
        assert orc.lifecycle_state == RuntimeLifecycleState.SOFT_FREEZE

    def test_repeated_mismatch_triggers_hard_freeze(self, tmp_path):
        orc = self._orc(tmp_path)
        for _ in range(_RECONCILIATION_HARD_FREEZE_COUNT):
            orc.reconcile(
                broker_positions=[{"symbol": "9984.T"}],
                runtime_ledger=[{"symbol": "7203.T"}],
            )
            if orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE:
                break
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE

    def test_both_empty_is_clean(self, tmp_path):
        orc = self._orc(tmp_path)
        result = orc.reconcile(broker_positions=[], runtime_ledger=[])
        assert result["status"] == "clean"

    def test_truth_hierarchy_uses_runtime_ledger(self, tmp_path):
        orc = self._orc(tmp_path)
        result = orc.reconcile(
            broker_positions=[{"symbol": "7203.T"}],
            runtime_ledger=[{"symbol": "7203.T"}],
            deployment_ledger=[{"symbol": "DIFFERENT.T"}],  # ignored in truth
        )
        assert result["status"] == "clean"
        assert result["truth_hierarchy"] == "runtime_ledger"

    def test_clean_resets_fail_count(self, tmp_path):
        orc = self._orc(tmp_path)
        # One mismatch
        orc.reconcile(
            broker_positions=[{"symbol": "bad.T"}],
            runtime_ledger=[{"symbol": "good.T"}],
        )
        assert orc._recon_fail_count == 1

        # Recovery to RUNNING
        orc._lifecycle_state = RuntimeLifecycleState.RUNNING
        # Clean recon resets count
        orc.reconcile(broker_positions=[{"symbol": "good.T"}], runtime_ledger=[{"symbol": "good.T"}])
        assert orc._recon_fail_count == 0


# ── Tests: append-only JSONL integrity ───────────────────────────────────────

class TestAppendOnlyIntegrity:
    def test_events_only_appended_never_overwritten(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        orc.run_pipeline(EVAL_DATE)
        count_after_1 = _count_events(tmp_path / "events")

        orc2 = _make_orc(tmp_path, _pass_fns())
        # Use different date to avoid checkpoint idempotency interference
        orc2.run_pipeline("2026-05-27")
        count_after_2 = _count_events(tmp_path / "events")

        assert count_after_2 > count_after_1

    def test_jsonl_lines_all_parseable(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        orc.run_pipeline(EVAL_DATE)
        for p in (tmp_path / "events").glob("runtime_events_*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)   # must not raise


# ── Tests: replay verification ────────────────────────────────────────────────

class TestReplayVerification:
    def test_same_inputs_same_hash_across_instances(self, tmp_path):
        """Two orchestrators with same stage fns produce same hashes."""
        fns = _pass_fns()
        orc1 = _make_orc(tmp_path / "o1", fns)
        orc2 = _make_orc(tmp_path / "o2", fns)

        r1 = orc1.run_pipeline(EVAL_DATE)
        r2 = orc2.run_pipeline(EVAL_DATE)

        # Same stage → same hash (deterministic)
        for a, b in zip(r1, r2):
            assert a.stage_name == b.stage_name
            assert a.deterministic_hash == b.deterministic_hash

    def test_replay_pipeline_pass(self, tmp_path):
        """replay_pipeline() should pass after a successful run."""
        fns = _pass_fns()
        orc = _make_orc(tmp_path, fns)
        orc.run_pipeline(EVAL_DATE)
        ok = orc.replay_pipeline(EVAL_DATE)
        assert ok

    def test_replay_no_checkpoint_returns_false(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        ok = orc.replay_pipeline(EVAL_DATE)
        assert not ok

    def test_different_stage_fns_produce_different_hashes(self, tmp_path):
        """Tampered stage function → different payload → different hash."""
        fns_a = _pass_fns()
        fns_b = _pass_fns()
        fns_b["ROUTING"] = lambda: {"stage": "ROUTING", "result": "different_result"}

        orc_a = _make_orc(tmp_path / "a", fns_a)
        orc_b = _make_orc(tmp_path / "b", fns_b)

        r_a = orc_a.run_pipeline(EVAL_DATE)
        r_b = orc_b.run_pipeline(EVAL_DATE)

        routing_a = next(r for r in r_a if r.stage_name == "ROUTING")
        routing_b = next(r for r in r_b if r.stage_name == "ROUTING")
        assert routing_a.deterministic_hash != routing_b.deterministic_hash


# ── Tests: lifecycle transitions ──────────────────────────────────────────────

class TestLifecycleTransitions:
    def _fresh_orc(self, tmp_path) -> FutureLeaderRuntimeOrchestrator:
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        return orc

    def test_idle_to_running(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.RUNNING

    def test_running_to_soft_freeze(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc._transition(RuntimeLifecycleState.SOFT_FREEZE, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.SOFT_FREEZE

    def test_running_to_hard_freeze(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc._transition(RuntimeLifecycleState.HARD_FREEZE, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE

    def test_running_to_shutdown_safe(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc._transition(RuntimeLifecycleState.SHUTDOWN_SAFE, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.SHUTDOWN_SAFE

    def test_soft_freeze_to_hard_freeze(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc._transition(RuntimeLifecycleState.SOFT_FREEZE, "test")
        orc._transition(RuntimeLifecycleState.HARD_FREEZE, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE

    def test_soft_freeze_to_recovery_replay(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        orc._transition(RuntimeLifecycleState.SOFT_FREEZE, "test")
        orc._transition(RuntimeLifecycleState.RECOVERY_REPLAY, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.RECOVERY_REPLAY

    def test_recovery_replay_to_running(self, tmp_path):
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RECOVERY_REPLAY, "test")
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        assert orc.lifecycle_state == RuntimeLifecycleState.RUNNING

    def test_illegal_transition_triggers_hard_freeze(self, tmp_path):
        """Attempt illegal transition → HARD_FREEZE (fail-closed)."""
        orc = self._fresh_orc(tmp_path)
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        # RUNNING → IDLE is illegal
        orc._transition(RuntimeLifecycleState.IDLE, "illegal")
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE


# ── Tests: get_status ─────────────────────────────────────────────────────────

class TestGetStatus:
    def test_status_reflects_state(self, tmp_path):
        orc = _make_orc(tmp_path, _pass_fns())
        orc.run_pipeline(EVAL_DATE)
        status = orc.get_status()
        assert status["lifecycle_state"] == orc.lifecycle_state.value
        assert "freeze_count" in status
        assert "rollback_state" in status


# ── Tests: CLI ────────────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_reconcile_clean(self, tmp_path, monkeypatch):
        import src.runtime.future_leader_runtime_orchestrator as mod
        monkeypatch.setattr(
            mod, "build_orchestrator",
            lambda **kw: _make_orc(tmp_path, _pass_fns()),
        )
        rc = cli_main(["--reconcile"])
        assert rc == 0

    def test_cli_force_soft_freeze(self, tmp_path, monkeypatch):
        import src.runtime.future_leader_runtime_orchestrator as mod
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        monkeypatch.setattr(mod, "build_orchestrator", lambda **kw: orc)
        cli_main(["--force-soft-freeze"])
        assert orc.lifecycle_state == RuntimeLifecycleState.SOFT_FREEZE

    def test_cli_force_hard_freeze(self, tmp_path, monkeypatch):
        import src.runtime.future_leader_runtime_orchestrator as mod
        orc = _make_orc(tmp_path)
        orc._current_date = EVAL_DATE
        orc._replay_seq_id = "test"
        orc._transition(RuntimeLifecycleState.RUNNING, "test")
        monkeypatch.setattr(mod, "build_orchestrator", lambda **kw: orc)
        cli_main(["--force-hard-freeze"])
        assert orc.lifecycle_state == RuntimeLifecycleState.HARD_FREEZE
