"""tests/research/test_batch.py — Batch experimentation layer tests.

Covers:
  ExperimentBatch    — frozen, deterministic ID, serialization
  ParameterSweep     — deterministic combinations, hash stability, count
  ExperimentGenerator — duplicate rejection, deterministic IDs, batch linkage
  BatchRunner        — execution log structure, non-fatal failure isolation
  BatchSummary       — ranking table, rejection report, regime summary
  BatchArtifacts     — append-only, overwrite rejection
  Correlation        — pairwise matrix, near-duplicate flagging, hash stability
  Capacity           — participation rate, penalty, determinism
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_registry(tmp_path):
    from src.research.orchestrator.registry import ExperimentRegistry
    return ExperimentRegistry(tmp_path / "reg.json")


def _small_sweep(seed: int = 42) -> "ParameterSweep":
    from src.research.batch.sweep import ParameterSweep
    return ParameterSweep(
        strategy_id="turtle_v1",
        grid={"holding_period": [3, 5, 10], "entry_threshold": [0.5, 1.0]},
        seed=seed,
    )


def _make_result(exp_id: str = "exp_20260101_aabb0001", sharpe: float = 1.5, trades: int = 20):
    from src.research.contracts.result import ResultContract, RegimeMetrics, StabilityMetrics
    return ResultContract(
        experiment_id=exp_id,
        fold_id=0,
        sharpe=sharpe,
        sortino=2.0,
        turnover=0.3,
        max_drawdown=-0.08,
        exposure=0.6,
        pnl=5000.0,
        hit_rate=0.55,
        tail_risk=-0.02,
        regime_metrics=RegimeMetrics.empty(),
        stability_metrics=StabilityMetrics(
            parameter_stability=0.9,
            oos_robustness=0.8,
            sharpe_decay=0.2,
        ),
        trade_count=trades,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentBatch
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentBatch:
    def _make(self, ts: str = "20260101") -> "ExperimentBatch":
        from src.research.batch.batch import ExperimentBatch
        bid = ExperimentBatch.make_id("turtle_v1", "ps_hash_001", 42, timestamp=ts)
        return ExperimentBatch(
            batch_id=bid,
            strategy_id="turtle_v1",
            description="test batch",
            seed=42,
            created_at="2026-01-01T00:00:00Z",
            experiment_ids=("exp_001", "exp_002"),
            parameter_space_hash="ps_hash_001",
        )

    def test_frozen(self):
        b = self._make()
        with pytest.raises((AttributeError, TypeError)):
            b.seed = 99  # type: ignore

    def test_make_id_deterministic(self):
        from src.research.batch.batch import ExperimentBatch
        id1 = ExperimentBatch.make_id("s", "ph", 42, timestamp="20260101")
        id2 = ExperimentBatch.make_id("s", "ph", 42, timestamp="20260101")
        assert id1 == id2

    def test_make_id_format(self):
        from src.research.batch.batch import ExperimentBatch
        bid = ExperimentBatch.make_id("s", "ph", 42, timestamp="20260101")
        assert bid.startswith("batch_")
        parts = bid.split("_")
        assert parts[1] == "20260101"
        assert len(parts[2]) == 8

    def test_different_seeds_different_id(self):
        from src.research.batch.batch import ExperimentBatch
        id1 = ExperimentBatch.make_id("s", "ph", 1, timestamp="20260101")
        id2 = ExperimentBatch.make_id("s", "ph", 2, timestamp="20260101")
        assert id1 != id2

    def test_to_dict_round_trip(self):
        from src.research.batch.batch import ExperimentBatch
        b = self._make()
        d = b.to_dict()
        restored = ExperimentBatch.from_dict(d)
        assert restored == b

    def test_experiment_count_in_dict(self):
        b = self._make()
        assert b.to_dict()["experiment_count"] == 2

    def test_to_json_stable(self):
        b = self._make()
        j1 = b.to_json()
        j2 = b.to_json()
        assert j1 == j2


# ─────────────────────────────────────────────────────────────────────────────
# ParameterSweep
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterSweep:
    def test_combination_count(self):
        s = _small_sweep()
        assert s.count() == 6  # 3 * 2

    def test_combinations_length_matches_count(self):
        s = _small_sweep()
        assert len(s.combinations()) == s.count()

    def test_deterministic(self):
        s1 = _small_sweep(seed=42)
        s2 = _small_sweep(seed=42)
        assert s1.combinations() == s2.combinations()

    def test_key_order_independent(self):
        from src.research.batch.sweep import ParameterSweep
        s1 = ParameterSweep("s", {"a": [1, 2], "b": [3, 4]}, seed=0)
        s2 = ParameterSweep("s", {"b": [3, 4], "a": [1, 2]}, seed=0)
        assert s1.combinations() == s2.combinations()

    def test_all_combinations_distinct(self):
        s = _small_sweep()
        combos = s.combinations()
        serialized = [json.dumps(c, sort_keys=True) for c in combos]
        assert len(serialized) == len(set(serialized))

    def test_empty_grid_returns_one_empty_combo(self):
        from src.research.batch.sweep import ParameterSweep
        s = ParameterSweep("s", {}, seed=0)
        assert s.combinations() == [{}]
        assert s.count() == 1

    def test_parameter_space_hash_stable(self):
        s = _small_sweep()
        h1 = s.parameter_space_hash()
        h2 = s.parameter_space_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_different_grids_different_hash(self):
        from src.research.batch.sweep import ParameterSweep
        s1 = ParameterSweep("s", {"a": [1, 2]}, seed=0)
        s2 = ParameterSweep("s", {"a": [1, 3]}, seed=0)
        assert s1.parameter_space_hash() != s2.parameter_space_hash()

    def test_subset(self):
        from src.research.batch.sweep import ParameterSweep
        s = ParameterSweep("s", {"a": [1, 2], "b": [3, 4], "c": [5]}, seed=0)
        sub = s.subset(["a", "c"])
        assert set(sub.grid.keys()) == {"a", "c"}

    def test_large_grid_count(self):
        from src.research.batch.sweep import ParameterSweep
        s = ParameterSweep("s", {
            "h": [3, 5, 10, 20],
            "e": [0.5, 1.0, 1.5, 2.0],
            "x": [0.3, 0.5, 0.7],
            "v": [True, False],
            "stop": [1.5, 2.0, 3.0],
        }, seed=0)
        assert s.count() == 4 * 4 * 3 * 2 * 3  # = 288

    def test_single_param(self):
        from src.research.batch.sweep import ParameterSweep
        s = ParameterSweep("s", {"x": [1, 2, 3]}, seed=0)
        combos = s.combinations()
        assert len(combos) == 3
        assert all("x" in c for c in combos)

    def test_predefined_dims(self):
        from src.research.batch.sweep import (
            holding_period_dim, entry_threshold_dim, vol_filter_dim, stop_atr_mult_dim
        )
        assert len(holding_period_dim()) > 0
        assert len(entry_threshold_dim()) > 0
        assert vol_filter_dim() == [True, False]
        assert all(isinstance(x, float) for x in stop_atr_mult_dim())


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentGenerator:
    def test_generates_correct_count(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep()
        reg = _make_registry(tmp_path)
        result = gen.generate(sweep, "dh001", "v1", reg, timestamp="20260101")
        assert len(result.new_experiments) == sweep.count()
        assert len(result.duplicates) == 0

    def test_deterministic_experiment_ids(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep(seed=42)

        reg1 = _make_registry(tmp_path / "r1")
        reg2 = _make_registry(tmp_path / "r2")

        r1 = gen.generate(sweep, "dh001", "v1", reg1, timestamp="20260101")
        r2 = gen.generate(sweep, "dh001", "v1", reg2, timestamp="20260101")

        ids1 = sorted(e.experiment_id for e in r1.new_experiments)
        ids2 = sorted(e.experiment_id for e in r2.new_experiments)
        assert ids1 == ids2

    def test_duplicate_rejection(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep()
        reg = _make_registry(tmp_path)

        r1 = gen.generate(sweep, "dh001", "v1", reg, timestamp="20260101")
        assert len(r1.new_experiments) == sweep.count()

        r2 = gen.generate(sweep, "dh001", "v1", reg, timestamp="20260101")
        assert len(r2.new_experiments) == 0
        assert len(r2.duplicates) == sweep.count()

    def test_partial_duplicate(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        from src.research.batch.sweep import ParameterSweep
        gen = ExperimentGenerator()
        reg = _make_registry(tmp_path)

        sweep_small = ParameterSweep("turtle_v1", {"holding_period": [3]}, seed=42)
        sweep_full  = ParameterSweep("turtle_v1", {"holding_period": [3, 5]}, seed=42)

        gen.generate(sweep_small, "dh001", "v1", reg, timestamp="20260101")
        r2 = gen.generate(sweep_full, "dh001", "v1", reg, timestamp="20260101")

        assert len(r2.new_experiments) == 1
        assert len(r2.duplicates) == 1

    def test_batch_id_deterministic(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep(seed=42)

        r1 = gen.generate(sweep, "dh001", "v1", _make_registry(tmp_path / "r1"), timestamp="20260101")
        r2 = gen.generate(sweep, "dh001", "v1", _make_registry(tmp_path / "r2"), timestamp="20260101")
        assert r1.batch.batch_id == r2.batch.batch_id

    def test_parameter_map_populated(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep()
        reg = _make_registry(tmp_path)
        result = gen.generate(sweep, "dh001", "v1", reg, timestamp="20260101")
        assert len(result.parameter_map) == len(result.new_experiments)
        for eid, params in result.parameter_map.items():
            assert "holding_period" in params
            assert "entry_threshold" in params

    def test_experiments_registered_in_registry(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep()
        reg = _make_registry(tmp_path)
        result = gen.generate(sweep, "dh001", "v1", reg, timestamp="20260101")
        for exp in result.new_experiments:
            assert reg.exists(exp.experiment_id)

    def test_all_experiments_have_correct_strategy(self, tmp_path):
        from src.research.batch.generator import ExperimentGenerator
        gen = ExperimentGenerator()
        sweep = _small_sweep()
        reg = _make_registry(tmp_path)
        result = gen.generate(sweep, "dh001", "v1", reg, timestamp="20260101")
        for exp in result.new_experiments:
            assert exp.strategy_id == "turtle_v1"
            assert exp.seed == 42


# ─────────────────────────────────────────────────────────────────────────────
# BatchRunner
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchRunner:
    def _make_batch_and_exps(self, tmp_path, n: int = 3):
        from src.research.batch.batch import ExperimentBatch
        from src.research.contracts.experiment import ExperimentContract
        experiment_ids = tuple(f"exp_20260101_{i:08d}" for i in range(n))
        batch = ExperimentBatch(
            batch_id="batch_20260101_deadbeef",
            strategy_id="turtle_v1",
            description="test",
            seed=42,
            created_at="2026-01-01T00:00:00Z",
            experiment_ids=experiment_ids,
            parameter_space_hash="ph001",
        )
        exps = [
            ExperimentContract(
                experiment_id=eid,
                strategy_id="turtle_v1",
                parameter_hash=f"ph{i:03d}",
                dataset_hash="dh001",
                code_version="v1",
                seed=42 + i,
                created_at="2026-01-01T00:00:00Z",
            )
            for i, eid in enumerate(experiment_ids)
        ]
        return batch, exps

    def test_execution_log_structure(self, tmp_path):
        from src.research.batch.runner import BatchRunner
        batch, exps = self._make_batch_and_exps(tmp_path)
        runner = BatchRunner(batch, exps, artifacts_base=tmp_path / "artifacts")

        script = tmp_path / "ok.py"
        script.write_text("import sys; sys.exit(0)")
        log = runner.run(script)

        assert log.batch_id == batch.batch_id
        assert isinstance(log.succeeded, list)
        assert isinstance(log.failed, list)
        assert log.total == len(exps)

    def test_non_fatal_failure_continues(self, tmp_path):
        from src.research.batch.runner import BatchRunner
        batch, exps = self._make_batch_and_exps(tmp_path, n=3)
        runner = BatchRunner(batch, exps, artifacts_base=tmp_path / "artifacts")

        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        log = runner.run(script, retry_max=1)

        # All 3 experiments attempted even though they fail
        assert len(log.failed) == 3
        assert log.total == 3

    def test_execution_log_written(self, tmp_path):
        from src.research.batch.runner import BatchRunner
        batch, exps = self._make_batch_and_exps(tmp_path)
        runner = BatchRunner(batch, exps, artifacts_base=tmp_path / "artifacts")

        script = tmp_path / "ok.py"
        script.write_text("import sys; sys.exit(0)")
        runner.run(script)

        log_path = tmp_path / "artifacts" / batch.batch_id / "execution_log.json"
        assert log_path.exists()
        content = json.loads(log_path.read_text())
        assert content["batch_id"] == batch.batch_id

    def test_log_not_overwritten(self, tmp_path):
        from src.research.batch.runner import BatchRunner
        batch, exps = self._make_batch_and_exps(tmp_path)
        runner = BatchRunner(batch, exps, artifacts_base=tmp_path / "artifacts")
        script = tmp_path / "ok.py"
        script.write_text("import sys; sys.exit(0)")

        runner.run(script)
        log_path = tmp_path / "artifacts" / batch.batch_id / "execution_log.json"
        original_content = log_path.read_text()

        runner.run(script)  # second run should not overwrite
        assert log_path.read_text() == original_content

    def test_worker_log_per_experiment(self, tmp_path):
        from src.research.batch.runner import BatchRunner
        batch, exps = self._make_batch_and_exps(tmp_path, n=2)
        runner = BatchRunner(batch, exps, artifacts_base=tmp_path / "artifacts")
        script = tmp_path / "ok.py"
        script.write_text("import sys; sys.exit(0)")
        log = runner.run(script)
        assert len(log.worker_log) == 2
        for entry in log.worker_log:
            assert "experiment_id" in entry
            assert "success" in entry

    def test_deterministic_execution_order(self, tmp_path):
        from src.research.batch.runner import BatchRunner
        from src.research.orchestrator.scheduler import schedule_experiments
        batch, exps = self._make_batch_and_exps(tmp_path, n=4)
        runner = BatchRunner(batch, exps, artifacts_base=tmp_path / "artifacts")
        script = tmp_path / "ok.py"
        script.write_text("import sys; sys.exit(0)")
        log = runner.run(script)

        expected_order = [e.experiment_id for e in schedule_experiments(exps)]
        actual_order = [entry["experiment_id"] for entry in log.worker_log]
        assert actual_order == expected_order


# ─────────────────────────────────────────────────────────────────────────────
# BatchSummary
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchSummary:
    def _make_results(self, n: int = 5) -> list:
        return [
            _make_result(f"exp_20260101_{i:04d}", sharpe=float(i) / 2, trades=20)
            for i in range(1, n + 1)
        ]

    def test_from_results_correct_count(self):
        from src.research.batch.summary import BatchSummary
        results = self._make_results(5)
        summary = BatchSummary.from_results("batch_001", results)
        assert summary.total_experiments == 5

    def test_ranking_table_length(self):
        from src.research.batch.summary import BatchSummary
        results = self._make_results(5)
        summary = BatchSummary.from_results("batch_001", results)
        assert len(summary.ranking_table()) == 5

    def test_ranking_table_deterministic(self):
        from src.research.batch.summary import BatchSummary
        results = self._make_results(5)
        t1 = BatchSummary.from_results("batch_001", results).ranking_table()
        t2 = BatchSummary.from_results("batch_001", results).ranking_table()
        assert t1 == t2

    def test_rejected_experiments_have_none_rank(self):
        from src.research.batch.summary import BatchSummary
        results = [
            _make_result("exp_20260101_0001", sharpe=4.5, trades=20),  # rejected: sharpe > 3
            _make_result("exp_20260101_0002", sharpe=1.5, trades=20),
        ]
        summary = BatchSummary.from_results("batch_001", results)
        table = summary.ranking_table()
        rejected_rows = [r for r in table if r["rejected"]]
        assert all(r["rank"] is None for r in rejected_rows)

    def test_top_n(self):
        from src.research.batch.summary import BatchSummary
        results = self._make_results(10)
        summary = BatchSummary.from_results("batch_001", results)
        top3 = summary.top_n(3)
        assert len(top3) == 3
        assert all(not r.rejected for r in top3)

    def test_rejection_report_structure(self):
        from src.research.batch.summary import BatchSummary
        results = self._make_results(5)
        summary = BatchSummary.from_results("batch_001", results)
        report = summary.rejection_report
        assert "total" in report
        assert "rejected" in report
        assert "passed" in report
        assert "rejection_rate" in report

    def test_rejection_report_counts_correct(self):
        from src.research.batch.summary import BatchSummary
        results = [
            _make_result("exp_20260101_0001", sharpe=4.0),  # rejected
            _make_result("exp_20260101_0002", sharpe=1.0),  # ok
            _make_result("exp_20260101_0003", trades=1),    # rejected: too few trades
        ]
        summary = BatchSummary.from_results("batch_001", results)
        assert summary.rejection_report["rejected"] == 2
        assert summary.rejection_report["passed"] == 1

    def test_to_dict_structure(self):
        from src.research.batch.summary import BatchSummary
        results = self._make_results(3)
        summary = BatchSummary.from_results("batch_001", results)
        d = summary.to_dict()
        assert "batch_id" in d
        assert "total_experiments" in d
        assert "total_ranked" in d
        assert "rejection_report" in d
        assert "top_3" in d


# ─────────────────────────────────────────────────────────────────────────────
# BatchArtifacts
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchArtifacts:
    def test_write_creates_file(self, tmp_path):
        from src.research.batch.batch import BatchArtifacts
        arts = BatchArtifacts("batch_001", base_dir=tmp_path)
        sha = arts.write("metadata.json", b'{"k": "v"}')
        assert arts.exists("metadata.json")
        assert len(sha) == 64

    def test_overwrite_raises(self, tmp_path):
        from src.research.batch.batch import BatchArtifacts, ArtifactExistsError
        arts = BatchArtifacts("batch_001", base_dir=tmp_path)
        arts.write("metadata.json", b"first")
        with pytest.raises(ArtifactExistsError):
            arts.write("metadata.json", b"second")

    def test_write_json_round_trip(self, tmp_path):
        from src.research.batch.batch import BatchArtifacts
        arts = BatchArtifacts("batch_001", base_dir=tmp_path)
        data = {"key": "value", "num": 42}
        arts.write_json("data.json", data)
        loaded = arts.read_json("data.json")
        assert loaded["key"] == "value"
        assert loaded["num"] == 42

    def test_list_artifacts_sorted(self, tmp_path):
        from src.research.batch.batch import BatchArtifacts
        arts = BatchArtifacts("batch_001", base_dir=tmp_path)
        arts.write("z.json", b"{}")
        arts.write("a.json", b"{}")
        assert arts.list_artifacts() == ["a.json", "z.json"]

    def test_write_batch_metadata(self, tmp_path):
        from src.research.batch.batch import BatchArtifacts, ExperimentBatch
        arts = BatchArtifacts("batch_001", base_dir=tmp_path)
        batch = ExperimentBatch(
            batch_id="batch_20260101_abcd1234",
            strategy_id="turtle_v1",
            description="test",
            seed=42,
            created_at="2026-01-01T00:00:00Z",
            experiment_ids=("exp_001",),
            parameter_space_hash="ph001",
        )
        arts.write_batch_metadata(batch)
        loaded = arts.read_json("batch_metadata.json")
        assert loaded["batch_id"] == batch.batch_id

    def test_write_ranking_table(self, tmp_path):
        from src.research.batch.batch import BatchArtifacts
        arts = BatchArtifacts("batch_001", base_dir=tmp_path)
        table = [{"rank": 1, "experiment_id": "exp_001", "score": 0.85}]
        arts.write_ranking_table(table)
        loaded = arts.read_json("ranking_table.json")
        assert loaded[0]["rank"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelation:
    def _returns_map(self, n: int = 100, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        base = rng.normal(0.001, 0.01, n)
        return {
            "exp_001": pd.Series(base + rng.normal(0, 0.002, n)),
            "exp_002": pd.Series(base + rng.normal(0, 0.002, n)),    # highly correlated
            "exp_003": pd.Series(rng.normal(0.001, 0.01, n)),         # independent
        }

    def test_returns_square_matrix(self):
        from src.research.batch.correlation import compute_pairwise_correlation
        rm = self._returns_map()
        corr = compute_pairwise_correlation(rm)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self):
        from src.research.batch.correlation import compute_pairwise_correlation
        rm = self._returns_map()
        corr = compute_pairwise_correlation(rm)
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-9

    def test_symmetric(self):
        from src.research.batch.correlation import compute_pairwise_correlation
        rm = self._returns_map()
        corr = compute_pairwise_correlation(rm)
        for i in corr.columns:
            for j in corr.columns:
                assert abs(corr.loc[i, j] - corr.loc[j, i]) < 1e-9

    def test_deterministic(self):
        from src.research.batch.correlation import compute_pairwise_correlation
        rm = self._returns_map(seed=7)
        c1 = compute_pairwise_correlation(rm)
        c2 = compute_pairwise_correlation(rm)
        assert c1.equals(c2)

    def test_empty_returns_empty_df(self):
        from src.research.batch.correlation import compute_pairwise_correlation
        assert compute_pairwise_correlation({}).empty

    def test_flag_near_duplicates_detects_high_correlation(self):
        from src.research.batch.correlation import compute_pairwise_correlation, flag_near_duplicates
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, 200)
        rm = {
            "a": pd.Series(base),
            "b": pd.Series(base + rng.normal(0, 0.001, 200)),  # ~1.0 corr
            "c": pd.Series(rng.normal(0, 1, 200)),              # independent
        }
        corr = compute_pairwise_correlation(rm)
        pairs = flag_near_duplicates(corr, threshold=0.95)
        ids_in_pairs = {(p[0], p[1]) for p in pairs}
        assert ("a", "b") in ids_in_pairs

    def test_flag_near_duplicates_empty_on_no_high_corr(self):
        from src.research.batch.correlation import compute_pairwise_correlation, flag_near_duplicates
        rng = np.random.default_rng(99)
        rm = {
            "x": pd.Series(rng.normal(0, 1, 100)),
            "y": pd.Series(rng.normal(0, 1, 100)),
        }
        corr = compute_pairwise_correlation(rm)
        pairs = flag_near_duplicates(corr, threshold=0.99)
        assert len(pairs) == 0

    def test_flag_near_duplicates_sorted_desc_by_corr(self):
        from src.research.batch.correlation import compute_pairwise_correlation, flag_near_duplicates
        rm = self._returns_map(seed=5)
        corr = compute_pairwise_correlation(rm)
        pairs = flag_near_duplicates(corr, threshold=0.0)  # catch all pairs
        corrs = [abs(p[2]) for p in pairs]
        assert corrs == sorted(corrs, reverse=True)

    def test_correlation_hash_stable(self):
        from src.research.batch.correlation import compute_pairwise_correlation, correlation_hash
        rm = self._returns_map()
        corr = compute_pairwise_correlation(rm)
        h1 = correlation_hash(corr)
        h2 = correlation_hash(corr)
        assert h1 == h2
        assert len(h1) == 16

    def test_correlation_hash_changes_on_different_data(self):
        from src.research.batch.correlation import compute_pairwise_correlation, correlation_hash
        rm1 = self._returns_map(seed=1)
        rm2 = self._returns_map(seed=2)
        c1 = compute_pairwise_correlation(rm1)
        c2 = compute_pairwise_correlation(rm2)
        assert correlation_hash(c1) != correlation_hash(c2)

    def test_rolling_correlation_stability_finite(self):
        from src.research.batch.correlation import rolling_correlation_stability
        rng = np.random.default_rng(0)
        s1 = pd.Series(rng.normal(0, 1, 100))
        s2 = pd.Series(rng.normal(0, 1, 100))
        val = rolling_correlation_stability(s1, s2, window=30)
        assert not np.isnan(val)
        assert val >= 0

    def test_rolling_correlation_stability_insufficient_data(self):
        from src.research.batch.correlation import rolling_correlation_stability
        s1 = pd.Series([0.01, 0.02, 0.03])
        s2 = pd.Series([0.01, 0.02, 0.03])
        val = rolling_correlation_stability(s1, s2, window=60)
        assert np.isnan(val)

    def test_build_correlation_report_structure(self):
        from src.research.batch.correlation import compute_pairwise_correlation, build_correlation_report
        rm = self._returns_map()
        corr = compute_pairwise_correlation(rm)
        report = build_correlation_report(corr, near_duplicate_threshold=0.5)
        assert "strategy_count" in report
        assert "near_duplicate_count" in report
        assert "near_duplicates" in report
        assert "matrix_hash" in report
        assert report["strategy_count"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Capacity Proxy
# ─────────────────────────────────────────────────────────────────────────────

class TestCapacity:
    def _make_series(self, values: list) -> pd.Series:
        return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="B"))

    def test_compute_capacity_basic(self):
        from src.research.batch.capacity import compute_capacity_proxy
        pos = self._make_series([100_000.0] * 50)
        adv = self._make_series([1_000_000.0] * 50)
        cap = compute_capacity_proxy(pos, adv)
        assert abs(cap.avg_participation_rate - 0.1) < 1e-6

    def test_avg_less_than_max(self):
        from src.research.batch.capacity import compute_capacity_proxy
        pos = self._make_series([100_000.0, 200_000.0, 150_000.0] * 20)
        adv = self._make_series([1_000_000.0] * 60)
        cap = compute_capacity_proxy(pos, adv)
        assert cap.avg_participation_rate <= cap.max_participation_rate

    def test_no_penalty_below_half_limit(self):
        from src.research.batch.capacity import compute_capacity_proxy
        pos = self._make_series([10_000.0] * 50)   # 1% participation
        adv = self._make_series([1_000_000.0] * 50)  # limit = 10%
        cap = compute_capacity_proxy(pos, adv)
        assert cap.ranking_penalty == 0.0

    def test_penalty_at_double_limit(self):
        from src.research.batch.capacity import compute_capacity_proxy
        pos = self._make_series([200_000.0] * 50)   # 20% participation = 2x limit
        adv = self._make_series([1_000_000.0] * 50)
        cap = compute_capacity_proxy(pos, adv, max_acceptable_participation=0.10)
        assert cap.ranking_penalty == pytest.approx(1.0, abs=1e-9)

    def test_penalty_bounded_zero_to_one(self):
        from src.research.batch.capacity import compute_capacity_proxy
        for mult in [0.0, 0.05, 0.10, 0.20, 0.50]:
            pos = self._make_series([mult * 1_000_000.0] * 50)
            adv = self._make_series([1_000_000.0] * 50)
            cap = compute_capacity_proxy(pos, adv)
            assert 0.0 <= cap.ranking_penalty <= 1.0

    def test_empty_series_returns_zero(self):
        from src.research.batch.capacity import compute_capacity_proxy
        cap = compute_capacity_proxy(pd.Series(dtype=float), pd.Series(dtype=float))
        assert cap.avg_participation_rate == 0.0
        assert cap.ranking_penalty == 0.0

    def test_deterministic(self):
        from src.research.batch.capacity import compute_capacity_proxy
        pos = self._make_series([100_000.0, 150_000.0, 80_000.0] * 10)
        adv = self._make_series([1_000_000.0] * 30)
        c1 = compute_capacity_proxy(pos, adv)
        c2 = compute_capacity_proxy(pos, adv)
        assert c1 == c2

    def test_stress_proxy_at_95th_percentile(self):
        from src.research.batch.capacity import compute_capacity_proxy
        # 85 small + 15 large → 95th pct (pos 94-95 in sorted array) falls in the large group
        values = [10_000.0] * 85 + [500_000.0] * 15
        pos = self._make_series(values)
        adv = self._make_series([1_000_000.0] * 100)
        cap = compute_capacity_proxy(pos, adv)
        # avg ≈ 8.5% * 0.01 + 15% * 0.5 = 0.0085 + 0.075 = 0.0835
        # 95th pct = 0.5 (all top 15% are 0.5)
        assert cap.liquidity_stress_proxy > cap.avg_participation_rate

    def test_to_dict_round_trip(self):
        from src.research.batch.capacity import compute_capacity_proxy, CapacityProxy
        pos = self._make_series([100_000.0] * 50)
        adv = self._make_series([1_000_000.0] * 50)
        cap = compute_capacity_proxy(pos, adv)
        d = cap.to_dict()
        restored = CapacityProxy.from_dict(d)
        assert restored == cap

    def test_batch_capacity_proxies_sorted(self):
        from src.research.batch.capacity import batch_capacity_proxies
        pos_map = {
            "exp_z": self._make_series([100_000.0] * 30),
            "exp_a": self._make_series([50_000.0] * 30),
        }
        adv_map = {
            "exp_z": self._make_series([1_000_000.0] * 30),
            "exp_a": self._make_series([1_000_000.0] * 30),
        }
        result = batch_capacity_proxies(pos_map, adv_map)
        assert list(result.keys()) == sorted(result.keys())
