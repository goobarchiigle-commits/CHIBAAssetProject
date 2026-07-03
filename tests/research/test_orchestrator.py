"""tests/research/test_orchestrator.py — Phase C orchestrator tests.

Covers:
  - ExperimentRegistry: register, get, duplicate rejection, persistence, list
  - ExperimentQueue: enqueue, pending, status transitions, priority ordering
  - Scheduler: deterministic ordering, identical input → identical output
  - Worker: result structure, failure handling
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_exp(strategy_id: str = "turtle", seed: int = 42, ts: str = "20260101"):
    from src.research.contracts.experiment import ExperimentContract
    eid = ExperimentContract.make_id(strategy_id, "ph001", "dh001", seed, timestamp=ts)
    return ExperimentContract(
        experiment_id=eid,
        strategy_id=strategy_id,
        parameter_hash="ph001",
        dataset_hash="dh001",
        code_version="v1",
        seed=seed,
        created_at="2026-01-01T00:00:00Z",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRegistry:
    def test_register_and_get(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "reg.json")
        exp = _make_exp()
        reg.register(exp)
        assert reg.get(exp.experiment_id) == exp

    def test_duplicate_raises(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry, DuplicateExperimentError
        reg = ExperimentRegistry(tmp_path / "reg.json")
        exp = _make_exp()
        reg.register(exp)
        with pytest.raises(DuplicateExperimentError):
            reg.register(exp)

    def test_get_missing_raises(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry, ExperimentNotFoundError
        reg = ExperimentRegistry(tmp_path / "reg.json")
        with pytest.raises(ExperimentNotFoundError):
            reg.get("exp_00000000_nonexist")

    def test_exists(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "reg.json")
        exp = _make_exp()
        assert not reg.exists(exp.experiment_id)
        reg.register(exp)
        assert reg.exists(exp.experiment_id)

    def test_count(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "reg.json")
        assert reg.count() == 0
        reg.register(_make_exp(seed=1))
        reg.register(_make_exp(seed=2))
        assert reg.count() == 2

    def test_persists_to_disk(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        f = tmp_path / "reg.json"
        reg = ExperimentRegistry(f)
        exp = _make_exp()
        reg.register(exp)
        reg2 = ExperimentRegistry(f)
        assert reg2.exists(exp.experiment_id)

    def test_empty_file_not_required(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "does_not_exist.json")
        assert reg.count() == 0

    def test_list_all(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "reg.json")
        e1 = _make_exp(seed=1)
        e2 = _make_exp(seed=2)
        reg.register(e1)
        reg.register(e2)
        ids = {e.experiment_id for e in reg.list_all()}
        assert e1.experiment_id in ids
        assert e2.experiment_id in ids

    def test_list_by_strategy(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "reg.json")
        reg.register(_make_exp(strategy_id="turtle", seed=1))
        reg.register(_make_exp(strategy_id="turtle", seed=2))
        reg.register(_make_exp(strategy_id="momentum", seed=3))
        turtles = reg.list_by_strategy("turtle")
        assert len(turtles) == 2
        assert all(e.strategy_id == "turtle" for e in turtles)

    def test_atomic_write_no_partial(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        f = tmp_path / "reg.json"
        reg = ExperimentRegistry(f)
        reg.register(_make_exp(seed=1))
        assert f.exists()
        content = json.loads(f.read_text())
        assert "experiments" in content

    def test_update_status(self, tmp_path):
        from src.research.orchestrator.registry import ExperimentRegistry
        reg = ExperimentRegistry(tmp_path / "reg.json")
        exp = _make_exp()
        reg.register(exp)
        reg.update_status(exp.experiment_id, "running")
        d = json.loads((tmp_path / "reg.json").read_text())
        assert d["experiments"][exp.experiment_id]["status"] == "running"


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentQueue
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentQueue:
    def test_enqueue_appears_in_pending(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        q = ExperimentQueue(tmp_path / "queue.jsonl")
        q.enqueue("exp_001")
        assert "exp_001" in q.pending()

    def test_mark_running_removes_from_pending(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        q = ExperimentQueue(tmp_path / "queue.jsonl")
        q.enqueue("exp_001")
        q.mark_running("exp_001")
        assert "exp_001" not in q.pending()

    def test_mark_done(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        q = ExperimentQueue(tmp_path / "queue.jsonl")
        q.enqueue("exp_001")
        q.mark_done("exp_001")
        assert q.status("exp_001") == "done"

    def test_mark_failed(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        q = ExperimentQueue(tmp_path / "queue.jsonl")
        q.enqueue("exp_001")
        q.mark_failed("exp_001")
        assert q.status("exp_001") == "failed"
        assert "exp_001" not in q.pending()

    def test_priority_ordering(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        q = ExperimentQueue(tmp_path / "queue.jsonl")
        q.enqueue("low", priority=0)
        q.enqueue("high", priority=10)
        q.enqueue("mid", priority=5)
        pending = q.pending()
        assert pending[0] == "high"
        assert pending[-1] == "low"

    def test_persist_and_reload(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        f = tmp_path / "queue.jsonl"
        q = ExperimentQueue(f)
        q.enqueue("exp_001")
        q.enqueue("exp_002")
        q2 = ExperimentQueue(f)
        assert "exp_001" in q2.pending()
        assert "exp_002" in q2.pending()

    def test_append_only_file(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        f = tmp_path / "queue.jsonl"
        q = ExperimentQueue(f)
        q.enqueue("exp_001")
        q.enqueue("exp_002")
        lines = f.read_text().splitlines()
        assert len(lines) == 2  # one per enqueue, never rewritten

    def test_status_missing_returns_none(self, tmp_path):
        from src.research.orchestrator.queue import ExperimentQueue
        q = ExperimentQueue(tmp_path / "queue.jsonl")
        assert q.status("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestScheduler:
    def _make_exps(self):
        return [
            _make_exp(strategy_id="b", seed=2),
            _make_exp(strategy_id="a", seed=1),
            _make_exp(strategy_id="a", seed=2),
        ]

    def test_deterministic_ordering(self):
        from src.research.orchestrator.scheduler import schedule_experiments
        exps = self._make_exps()
        order1 = schedule_experiments(list(exps))
        order2 = schedule_experiments(list(exps))
        assert [e.experiment_id for e in order1] == [e.experiment_id for e in order2]

    def test_sorted_by_strategy_then_seed(self):
        from src.research.orchestrator.scheduler import schedule_experiments
        exps = self._make_exps()
        ordered = schedule_experiments(exps)
        ids = [(e.strategy_id, e.seed) for e in ordered]
        assert ids == sorted(ids)

    def test_empty_list(self):
        from src.research.orchestrator.scheduler import schedule_experiments
        assert schedule_experiments([]) == []

    def test_batch_bounded_concurrency(self):
        from src.research.orchestrator.scheduler import batch
        exps = [_make_exp(seed=i) for i in range(7)]
        batches = batch(exps, max_concurrency=3)
        assert all(len(b) <= 3 for b in batches)
        total = sum(len(b) for b in batches)
        assert total == len(exps)

    def test_batch_zero_concurrency_raises(self):
        from src.research.orchestrator.scheduler import batch
        with pytest.raises(ValueError):
            batch([], max_concurrency=0)

    def test_batch_deterministic(self):
        from src.research.orchestrator.scheduler import batch
        exps = [_make_exp(seed=i) for i in range(10)]
        b1 = batch(exps, 3)
        b2 = batch(exps, 3)
        assert [[e.experiment_id for e in b] for b in b1] == [[e.experiment_id for e in b] for b in b2]


# ─────────────────────────────────────────────────────────────────────────────
# Worker (unit tests — no actual subprocess launched)
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerResult:
    def test_result_structure(self):
        from src.research.orchestrator.worker import WorkerResult
        wr = WorkerResult(
            experiment_id="exp_001",
            success=True,
            exit_code=0,
            stdout="done",
            stderr="",
            attempt=1,
        )
        assert wr.success
        assert wr.exit_code == 0

    def test_failed_result(self):
        from src.research.orchestrator.worker import WorkerResult
        wr = WorkerResult(
            experiment_id="exp_001",
            success=False,
            exit_code=1,
            stdout="",
            stderr="error",
            attempt=1,
        )
        assert not wr.success

    def test_run_nonexistent_script_returns_failure(self, tmp_path):
        from src.research.orchestrator.worker import run_experiment_isolated
        result = run_experiment_isolated(
            script_path=tmp_path / "nonexistent_script.py",
            experiment_id="exp_001",
            retry_max=1,
        )
        assert result.success is False

    def test_run_failing_script(self, tmp_path):
        from src.research.orchestrator.worker import run_experiment_isolated
        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        result = run_experiment_isolated(script, "exp_001", retry_max=1)
        assert result.success is False
        assert result.exit_code == 1

    def test_run_passing_script(self, tmp_path):
        from src.research.orchestrator.worker import run_experiment_isolated
        script = tmp_path / "ok.py"
        script.write_text("import sys; print('done'); sys.exit(0)")
        result = run_experiment_isolated(script, "exp_001", retry_max=1)
        assert result.success is True
        assert "done" in result.stdout
