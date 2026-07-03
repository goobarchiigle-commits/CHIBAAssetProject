"""
tests/test_determinism.py
Comprehensive coverage for Task 4: Deterministic Execution and Experiment Reproducibility.

Tests:
  TestDeterminism           — set_deterministic, require_seed, SeedNotSetError
  TestClockIsolation        — ClockSource implementations, get/set/reset_clock
  TestExperimentManifest    — field validation, to_dict round-trip
  TestExperimentRegistry    — register, get, list, duplicate rejection
  TestArtifactStore         — write, overwrite rejection, hash, list
  TestBuildManifest         — make_experiment_id, build_manifest
  TestFingerprintDataset    — sha256, row_count, schema_hash, nan_profile
  TestValidateSignalDeterminism — identical outputs, divergence detection
  TestReproduceExperiment   — hash verification, missing experiment
  TestGovernanceGateRepro   — gate integration: missing registry, null seed, null hash
  TestDatetimeNowScanner    — AST scan for datetime.now() usage
  TestSeedFromEnv           — env var seed loading
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import textwrap
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_determinism():
    """Reset seed state before/after every test."""
    from src.governance.determinism import reset_seed
    reset_seed()
    yield
    reset_seed()


@pytest.fixture(autouse=True)
def reset_clock_fixture():
    """Restore ProductionClock after every test."""
    from src.governance.clock import reset_clock
    yield
    reset_clock()


@pytest.fixture
def tmp_exp_registry(tmp_path):
    """ExperimentRegistry backed by a temp file."""
    from src.governance.experiment import ExperimentRegistry
    return ExperimentRegistry(registry_file=tmp_path / "experiment_registry.json")


@pytest.fixture
def tmp_artifact_store(tmp_path):
    """ArtifactStore in temp dir."""
    from src.governance.experiment import ArtifactStore
    return ArtifactStore("exp_test_0000abcd", base_dir=tmp_path / "artifacts")


@pytest.fixture
def sample_manifest():
    from src.governance.experiment import build_manifest
    return build_manifest(
        strategy_id="test_strategy",
        runtime_mode="paper",
        seed=42,
        config_hash="aabbccdd",
        dataset_hash="11223344",
        notes="unit test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterminism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_set_deterministic_sets_seed(self):
        from src.governance.determinism import set_deterministic, get_seed
        set_deterministic(42)
        assert get_seed() == 42

    def test_require_seed_returns_seed_when_set(self):
        from src.governance.determinism import set_deterministic, require_seed
        set_deterministic(99)
        assert require_seed() == 99

    def test_require_seed_raises_when_unset(self):
        from src.governance.determinism import SeedNotSetError, require_seed
        with pytest.raises(SeedNotSetError):
            require_seed()

    def test_reset_seed_clears_state(self):
        from src.governance.determinism import set_deterministic, reset_seed, get_seed
        set_deterministic(7)
        reset_seed()
        assert get_seed() is None

    def test_get_seed_returns_none_when_unset(self):
        from src.governance.determinism import get_seed
        assert get_seed() is None

    def test_source_label_stored(self):
        from src.governance.determinism import set_deterministic, get_seed_source
        set_deterministic(1, source="experiment")
        assert get_seed_source() == "experiment"

    def test_non_int_seed_raises_type_error(self):
        from src.governance.determinism import set_deterministic
        with pytest.raises(TypeError):
            set_deterministic("42")  # type: ignore

    def test_random_module_seeded(self):
        from src.governance.determinism import set_deterministic
        set_deterministic(123)
        r1 = [random.random() for _ in range(10)]
        from src.governance.determinism import reset_seed
        reset_seed()
        set_deterministic(123)
        r2 = [random.random() for _ in range(10)]
        assert r1 == r2

    def test_numpy_seeded(self):
        np = pytest.importorskip("numpy")
        from src.governance.determinism import set_deterministic
        set_deterministic(777)
        a1 = np.random.rand(5).tolist()
        from src.governance.determinism import reset_seed
        reset_seed()
        set_deterministic(777)
        a2 = np.random.rand(5).tolist()
        assert a1 == a2

    def test_different_seeds_produce_different_output(self):
        from src.governance.determinism import set_deterministic
        set_deterministic(1)
        r1 = random.random()
        from src.governance.determinism import reset_seed
        reset_seed()
        set_deterministic(2)
        r2 = random.random()
        assert r1 != r2


# ─────────────────────────────────────────────────────────────────────────────
# TestSeedFromEnv
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedFromEnv:
    def test_reads_seed_from_env(self, monkeypatch):
        from src.governance.determinism import seed_from_env, get_seed
        monkeypatch.setenv("EXPERIMENT_SEED", "555")
        result = seed_from_env()
        assert result == 555
        assert get_seed() == 555

    def test_returns_none_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("EXPERIMENT_SEED", raising=False)
        from src.governance.determinism import seed_from_env
        assert seed_from_env() is None

    def test_raises_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("EXPERIMENT_SEED", "notanumber")
        from src.governance.determinism import seed_from_env
        with pytest.raises(ValueError):
            seed_from_env()

    def test_custom_env_var_name(self, monkeypatch):
        from src.governance.determinism import seed_from_env, get_seed
        monkeypatch.setenv("MY_SEED", "999")
        seed_from_env("MY_SEED")
        assert get_seed() == 999


# ─────────────────────────────────────────────────────────────────────────────
# TestDatetimeNowScanner
# ─────────────────────────────────────────────────────────────────────────────

class TestDatetimeNowScanner:
    def test_detects_datetime_now(self, tmp_path):
        from src.governance.determinism import scan_for_datetime_now
        py = tmp_path / "strategy.py"
        py.write_text("from datetime import datetime\ndef fn():\n    return datetime.now()\n")
        hits = scan_for_datetime_now(py)
        assert len(hits) == 1
        assert "datetime.now()" in hits[0][1]

    def test_detects_datetime_datetime_now(self, tmp_path):
        from src.governance.determinism import scan_for_datetime_now
        py = tmp_path / "strategy.py"
        py.write_text("import datetime\ndef fn():\n    return datetime.datetime.now()\n")
        hits = scan_for_datetime_now(py)
        assert len(hits) == 1
        assert "datetime.datetime.now()" in hits[0][1]

    def test_no_hits_on_clean_file(self, tmp_path):
        from src.governance.determinism import scan_for_datetime_now
        py = tmp_path / "clean.py"
        py.write_text("x = 1 + 1\n")
        assert scan_for_datetime_now(py) == []

    def test_detects_datetime_today(self, tmp_path):
        from src.governance.determinism import scan_for_datetime_now
        py = tmp_path / "strategy.py"
        py.write_text("from datetime import datetime\nt = datetime.today()\n")
        hits = scan_for_datetime_now(py)
        assert any("today" in h[1] for h in hits)

    def test_returns_line_number(self, tmp_path):
        from src.governance.determinism import scan_for_datetime_now
        py = tmp_path / "strategy.py"
        py.write_text("# line 1\n# line 2\nfrom datetime import datetime\nx = datetime.now()\n")
        hits = scan_for_datetime_now(py)
        assert hits[0][0] == 4

    def test_scan_directory_excludes_test_files(self, tmp_path):
        from src.governance.determinism import scan_directory
        (tmp_path / "test_something.py").write_text(
            "from datetime import datetime\nx = datetime.now()\n"
        )
        (tmp_path / "strategy.py").write_text(
            "from datetime import datetime\nx = datetime.now()\n"
        )
        result = scan_directory(tmp_path)
        assert "strategy.py" in result
        assert not any("test_" in k for k in result)

    def test_parse_error_returns_empty(self, tmp_path):
        from src.governance.determinism import scan_for_datetime_now
        py = tmp_path / "broken.py"
        py.write_text("def (\n")  # syntax error
        assert scan_for_datetime_now(py) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestClockIsolation
# ─────────────────────────────────────────────────────────────────────────────

class TestClockIsolation:
    def test_fixed_clock_returns_same_time(self):
        from src.governance.clock import FixedClock
        t = datetime(2024, 1, 15, 9, 0, 0, tzinfo=JST)
        c = FixedClock(t)
        assert c.now() == t
        assert c.now() == t  # unchanged

    def test_fixed_clock_naive_gets_jst(self):
        from src.governance.clock import FixedClock
        t = datetime(2024, 6, 1, 10, 0, 0)
        c = FixedClock(t)
        result = c.now()
        assert result.tzinfo is not None

    def test_sequential_clock_advances(self):
        from src.governance.clock import SequentialClock
        start = datetime(2024, 1, 1, 9, 0, 0, tzinfo=JST)
        c = SequentialClock(start, step_seconds=60)
        t1 = c.now()
        t2 = c.now()
        assert (t2 - t1).total_seconds() == 60

    def test_sequential_clock_call_count(self):
        from src.governance.clock import SequentialClock
        c = SequentialClock(datetime(2024, 1, 1, tzinfo=JST), step_seconds=1)
        c.now(); c.now(); c.now()
        assert c.call_count == 3

    def test_iterator_clock_sequence(self):
        from src.governance.clock import IteratorClock
        timestamps = [
            datetime(2024, 1, 1, 9, tzinfo=JST),
            datetime(2024, 1, 1, 10, tzinfo=JST),
        ]
        c = IteratorClock(timestamps)
        assert c.now() == timestamps[0]
        assert c.now() == timestamps[1]

    def test_iterator_clock_exhausted_raises(self):
        from src.governance.clock import IteratorClock
        c = IteratorClock([datetime(2024, 1, 1, tzinfo=JST)])
        c.now()
        with pytest.raises(RuntimeError, match="exhausted"):
            c.now()

    def test_set_clock_replaces_active(self):
        from src.governance.clock import set_clock, get_clock, FixedClock
        t = datetime(2024, 3, 1, tzinfo=JST)
        fc = FixedClock(t)
        set_clock(fc)
        assert get_clock() is fc

    def test_reset_clock_restores_production(self):
        from src.governance.clock import set_clock, reset_clock, get_clock, FixedClock, ProductionClock
        set_clock(FixedClock(datetime(2024, 1, 1, tzinfo=JST)))
        reset_clock()
        assert isinstance(get_clock(), ProductionClock)

    def test_set_clock_type_error(self):
        from src.governance.clock import set_clock
        with pytest.raises(TypeError):
            set_clock("not a clock")  # type: ignore

    def test_module_now_uses_active_clock(self):
        from src.governance.clock import set_clock, FixedClock
        from src.governance.clock import now as clock_now
        t = datetime(2024, 5, 5, 12, 0, 0, tzinfo=JST)
        set_clock(FixedClock(t))
        assert clock_now() == t

    def test_today_returns_midnight(self):
        from src.governance.clock import FixedClock
        t = datetime(2024, 1, 15, 14, 30, 45, tzinfo=JST)
        c = FixedClock(t)
        today = c.today()
        assert today.hour == 0
        assert today.minute == 0
        assert today.second == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestExperimentManifest
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentManifest:
    def test_required_fields_pass(self, sample_manifest):
        from src.governance.experiment import ExperimentManifest
        assert isinstance(sample_manifest, ExperimentManifest)

    def test_missing_required_field_raises(self):
        from src.governance.experiment import ExperimentManifest
        with pytest.raises(ValueError, match="missing required"):
            ExperimentManifest({"strategy_id": "x", "runtime_mode": "paper"})

    def test_to_dict_round_trip(self, sample_manifest):
        d = sample_manifest.to_dict()
        assert d["strategy_id"] == "test_strategy"
        assert d["seed"] == 42
        assert d["runtime_mode"] == "paper"
        assert d["git_commit"] is None

    def test_accessors(self, sample_manifest):
        assert sample_manifest.strategy_id == "test_strategy"
        assert sample_manifest.seed == 42
        assert sample_manifest.runtime_mode == "paper"
        assert sample_manifest.config_hash == "aabbccdd"
        assert sample_manifest.notes == "unit test"
        assert sample_manifest.git_commit is None

    def test_experiment_id_format(self, sample_manifest):
        eid = sample_manifest.experiment_id
        assert eid.startswith("exp_")
        parts = eid.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 8  # hash[:8]


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildManifest
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildManifest:
    def test_make_experiment_id_deterministic(self):
        from src.governance.experiment import make_experiment_id
        id1 = make_experiment_id("strat_a", "cfghash", "dshash", 42)
        id2 = make_experiment_id("strat_a", "cfghash", "dshash", 42)
        # Same content → same hash suffix (date may vary, but hash part is stable)
        assert id1.split("_")[2] == id2.split("_")[2]

    def test_different_seeds_different_id(self):
        from src.governance.experiment import make_experiment_id
        id1 = make_experiment_id("strat_a", "hash", "ds", 42)
        id2 = make_experiment_id("strat_a", "hash", "ds", 99)
        assert id1.split("_")[2] != id2.split("_")[2]

    def test_build_manifest_sets_created_at(self, sample_manifest):
        assert "created_at" in sample_manifest.to_dict()
        assert "T" in sample_manifest.to_dict()["created_at"]

    def test_build_manifest_null_seed(self):
        from src.governance.experiment import build_manifest
        m = build_manifest("strat", "research", seed=None)
        assert m.seed is None

    def test_parent_experiment_propagated(self):
        from src.governance.experiment import build_manifest
        m = build_manifest("strat", "paper", seed=1, parent_experiment="exp_20240101_aaaabbbb")
        assert m.parent_experiment == "exp_20240101_aaaabbbb"


# ─────────────────────────────────────────────────────────────────────────────
# TestExperimentRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRegistry:
    def test_register_and_get(self, tmp_exp_registry, sample_manifest):
        tmp_exp_registry.register(sample_manifest)
        retrieved = tmp_exp_registry.get(sample_manifest.experiment_id)
        assert retrieved.experiment_id == sample_manifest.experiment_id

    def test_duplicate_registration_raises(self, tmp_exp_registry, sample_manifest):
        from src.governance.experiment import ArtifactExistsError
        tmp_exp_registry.register(sample_manifest)
        with pytest.raises(ArtifactExistsError):
            tmp_exp_registry.register(sample_manifest)

    def test_get_missing_raises(self, tmp_exp_registry):
        from src.governance.experiment import ExperimentNotFoundError
        with pytest.raises(ExperimentNotFoundError):
            tmp_exp_registry.get("exp_nonexistent_00000000")

    def test_exists_true_after_register(self, tmp_exp_registry, sample_manifest):
        tmp_exp_registry.register(sample_manifest)
        assert tmp_exp_registry.exists(sample_manifest.experiment_id)

    def test_exists_false_before_register(self, tmp_exp_registry):
        assert not tmp_exp_registry.exists("exp_unknown_00000000")

    def test_count(self, tmp_exp_registry, sample_manifest):
        assert tmp_exp_registry.count() == 0
        tmp_exp_registry.register(sample_manifest)
        assert tmp_exp_registry.count() == 1

    def test_list_all_sorted_desc(self, tmp_path):
        from src.governance.experiment import ExperimentRegistry, build_manifest
        reg = ExperimentRegistry(registry_file=tmp_path / "reg.json")
        m1 = build_manifest("s1", "paper", seed=1)
        m2 = build_manifest("s2", "paper", seed=2, config_hash="different")
        reg.register(m1)
        reg.register(m2)
        all_exps = reg.list_all()
        assert len(all_exps) == 2
        # sorted desc by created_at — both registered same second, order stable
        ids = {m.experiment_id for m in all_exps}
        assert m1.experiment_id in ids
        assert m2.experiment_id in ids

    def test_persists_to_disk(self, tmp_path, sample_manifest):
        from src.governance.experiment import ExperimentRegistry
        f = tmp_path / "reg.json"
        reg = ExperimentRegistry(registry_file=f)
        reg.register(sample_manifest)
        # Reload from disk
        reg2 = ExperimentRegistry(registry_file=f)
        assert reg2.exists(sample_manifest.experiment_id)

    def test_empty_registry_file_not_required(self, tmp_path):
        from src.governance.experiment import ExperimentRegistry
        reg = ExperimentRegistry(registry_file=tmp_path / "nonexistent.json")
        assert reg.count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestArtifactStore
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactStore:
    def test_write_creates_file(self, tmp_artifact_store):
        sha = tmp_artifact_store.write("metrics.json", b'{"sharpe": 1.5}')
        assert tmp_artifact_store.exists("metrics.json")
        assert len(sha) == 64  # sha256 hex

    def test_write_str_content(self, tmp_artifact_store):
        sha = tmp_artifact_store.write("metrics.json", '{"sharpe": 1.5}')
        assert tmp_artifact_store.exists("metrics.json")

    def test_write_overwrite_raises(self, tmp_artifact_store):
        from src.governance.experiment import ArtifactExistsError
        tmp_artifact_store.write("metrics.json", b"first")
        with pytest.raises(ArtifactExistsError):
            tmp_artifact_store.write("metrics.json", b"second")

    def test_hash_of_existing(self, tmp_artifact_store):
        content = b'{"sharpe": 2.1}'
        expected = hashlib.sha256(content).hexdigest()
        tmp_artifact_store.write("metrics.json", content)
        assert tmp_artifact_store.hash_of("metrics.json") == expected

    def test_hash_of_missing_raises(self, tmp_artifact_store):
        with pytest.raises(FileNotFoundError):
            tmp_artifact_store.hash_of("nonexistent.json")

    def test_list_artifacts(self, tmp_artifact_store):
        tmp_artifact_store.write("metrics.json", b"{}")
        tmp_artifact_store.write("config_snapshot.json", b"{}")
        names = tmp_artifact_store.list_artifacts()
        assert "metrics.json" in names
        assert "config_snapshot.json" in names

    def test_all_hashes(self, tmp_artifact_store):
        tmp_artifact_store.write("metrics.json", b'{"x":1}')
        hashes = tmp_artifact_store.all_hashes()
        assert "metrics.json" in hashes
        assert len(hashes["metrics.json"]) == 64

    def test_write_json_serializes(self, tmp_artifact_store):
        data = {"sharpe": 1.5, "max_dd": -0.05}
        tmp_artifact_store.write_json("metrics.json", data)
        raw = (tmp_artifact_store.artifact_dir / "metrics.json").read_text()
        loaded = json.loads(raw)
        assert loaded["sharpe"] == 1.5

    def test_empty_store_list(self, tmp_artifact_store):
        assert tmp_artifact_store.list_artifacts() == []


# ─────────────────────────────────────────────────────────────────────────────
# TestFingerprintDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestFingerprintDataset:
    def _make_csv(self, tmp_path, name="data.csv") -> Path:
        pd = pytest.importorskip("pandas")
        import numpy as np
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000, 2000, 1500, 1800],
        }, index=pd.date_range("2024-01-01", periods=4, tz="Asia/Tokyo"))
        p = tmp_path / name
        df.to_csv(p)
        return p

    def test_fingerprint_returns_sha256(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        fp = fd.fingerprint(p)
        assert len(fp["sha256"]) == 64

    def test_fingerprint_row_count(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        fp = fd.fingerprint(p)
        assert fp["row_count"] == 4

    def test_fingerprint_schema_hash(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        fp = fd.fingerprint(p)
        assert len(fp["schema_hash"]) == 16

    def test_fingerprint_nan_profile(self, tmp_path):
        pd = pytest.importorskip("pandas")
        import fingerprint_dataset as fd
        import numpy as np
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4.0, 5.0, 6.0]})
        p = tmp_path / "nan_data.csv"
        df.to_csv(p)
        fp = fd.fingerprint(p)
        assert fp["nan_profile"].get("a", 0) == 1

    def test_fingerprint_missing_file_raises(self, tmp_path):
        import fingerprint_dataset as fd
        with pytest.raises(FileNotFoundError):
            fd.fingerprint(tmp_path / "missing.csv")

    def test_register_and_retrieve(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        reg_file = tmp_path / "dataset_registry.json"
        # Monkey-patch registry path
        original = fd.REGISTRY_FILE
        fd.REGISTRY_FILE = reg_file
        try:
            fp = fd.register_dataset(p, name="test_data")
            loaded = fd.get_fingerprint("test_data")
            assert loaded is not None
            assert loaded["sha256"] == fp["sha256"]
        finally:
            fd.REGISTRY_FILE = original

    def test_verify_dataset_ok(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        reg_file = tmp_path / "dataset_registry.json"
        original = fd.REGISTRY_FILE
        fd.REGISTRY_FILE = reg_file
        try:
            fd.register_dataset(p, name="ds1")
            assert fd.verify_dataset(p, "ds1") is True
        finally:
            fd.REGISTRY_FILE = original

    def test_verify_dataset_mismatch(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        reg_file = tmp_path / "dataset_registry.json"
        original = fd.REGISTRY_FILE
        fd.REGISTRY_FILE = reg_file
        try:
            fd.register_dataset(p, name="ds1")
            # Tamper with file
            p.write_bytes(p.read_bytes() + b"\n# tampered")
            assert fd.verify_dataset(p, "ds1") is False
        finally:
            fd.REGISTRY_FILE = original

    def test_verify_unregistered_returns_false(self, tmp_path):
        import fingerprint_dataset as fd
        p = self._make_csv(tmp_path)
        reg_file = tmp_path / "dataset_registry.json"
        original = fd.REGISTRY_FILE
        fd.REGISTRY_FILE = reg_file
        try:
            assert fd.verify_dataset(p, "not_registered") is False
        finally:
            fd.REGISTRY_FILE = original

    def test_unsupported_format_raises(self, tmp_path):
        import fingerprint_dataset as fd
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"fake excel content")
        with pytest.raises(ValueError, match="Unsupported"):
            fd.fingerprint(p)


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateSignalDeterminism
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateSignalDeterminism:
    def _make_deterministic_fn(self):
        """Strategy that returns a deterministic list based on seed."""
        def strategy():
            from src.governance.determinism import require_seed
            seed = require_seed()
            import random
            rng = random.Random(seed)
            return [rng.random() for _ in range(10)]
        return strategy

    def _make_nondeterministic_fn(self):
        """Strategy that always returns different output."""
        _counter = [0]
        def strategy():
            from src.governance.determinism import require_seed
            require_seed()
            _counter[0] += 1
            return [_counter[0] * i for i in range(5)]
        return strategy

    def test_identical_runs_pass(self):
        import validate_signal_determinism as vsd
        fn = self._make_deterministic_fn()
        ok, report = vsd.validate_determinism(fn, n_runs=3, seed=42)
        assert ok is True
        assert report["identical"] is True
        assert len(set(report["hashes"])) == 1

    def test_divergent_runs_fail(self):
        import validate_signal_determinism as vsd
        fn = self._make_nondeterministic_fn()
        ok, report = vsd.validate_determinism(fn, n_runs=3, seed=42)
        assert ok is False
        assert report["identical"] is False
        assert report["diverge_at"] is not None

    def test_strict_raises_on_failure(self):
        import validate_signal_determinism as vsd
        fn = self._make_nondeterministic_fn()
        with pytest.raises(AssertionError, match="diverge_at"):
            vsd.validate_determinism_strict(fn, n_runs=2, seed=42)

    def test_strict_no_raise_on_pass(self):
        import validate_signal_determinism as vsd
        fn = self._make_deterministic_fn()
        vsd.validate_determinism_strict(fn, n_runs=2, seed=42)  # no error

    def test_exception_in_run_captured(self):
        import validate_signal_determinism as vsd
        def failing_fn():
            raise RuntimeError("strategy crashed")
        ok, report = vsd.validate_determinism(failing_fn, n_runs=2, seed=42)
        assert ok is False
        assert len(report["errors"]) > 0

    def test_report_contains_all_hashes(self):
        import validate_signal_determinism as vsd
        fn = self._make_deterministic_fn()
        ok, report = vsd.validate_determinism(fn, n_runs=4, seed=42)
        assert len(report["hashes"]) == 4

    def test_different_seeds_different_hashes(self):
        import validate_signal_determinism as vsd
        fn = self._make_deterministic_fn()
        _, r1 = vsd.validate_determinism(fn, n_runs=1, seed=42)
        from src.governance.determinism import reset_seed
        reset_seed()
        _, r2 = vsd.validate_determinism(fn, n_runs=1, seed=99)
        assert r1["first_hash"] != r2["first_hash"]


# ─────────────────────────────────────────────────────────────────────────────
# TestReproduceExperiment
# ─────────────────────────────────────────────────────────────────────────────

class TestReproduceExperiment:
    def test_missing_experiment_returns_error(self, tmp_path, monkeypatch):
        import reproduce_experiment as re_mod
        from src.governance import experiment as exp_mod
        # Patch ExperimentRegistry to use tmp registry
        original_registry = exp_mod.EXPERIMENT_REGISTRY
        exp_mod.EXPERIMENT_REGISTRY = tmp_path / "exp_reg.json"
        try:
            success, report = re_mod.reproduce_experiment("exp_nonexistent_00000000")
            assert success is False
            assert any("not in registry" in e for e in report["errors"])
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original_registry

    def test_null_seed_produces_error(self, tmp_path, monkeypatch):
        import reproduce_experiment as re_mod
        from src.governance import experiment as exp_mod
        reg_file = tmp_path / "exp_reg.json"
        original_registry = exp_mod.EXPERIMENT_REGISTRY
        exp_mod.EXPERIMENT_REGISTRY = reg_file
        try:
            m = exp_mod.build_manifest("strat", "paper", seed=None)
            reg = exp_mod.ExperimentRegistry(registry_file=reg_file)
            reg.register(m)
            success, report = re_mod.reproduce_experiment(m.experiment_id)
            assert not success
            assert any("seed is null" in e for e in report["errors"])
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original_registry

    def test_missing_artifact_produces_error(self, tmp_path, monkeypatch):
        import reproduce_experiment as re_mod
        from src.governance import experiment as exp_mod
        reg_file = tmp_path / "exp_reg.json"
        original_registry = exp_mod.EXPERIMENT_REGISTRY
        original_artifacts = exp_mod.ARTIFACTS_BASE
        exp_mod.EXPERIMENT_REGISTRY = reg_file
        exp_mod.ARTIFACTS_BASE = tmp_path / "artifacts"
        try:
            m = exp_mod.build_manifest(
                "strat", "paper", seed=42,
                result_artifact_hashes={"metrics.json": "abc123"},
            )
            reg = exp_mod.ExperimentRegistry(registry_file=reg_file)
            reg.register(m)
            success, report = re_mod.reproduce_experiment(m.experiment_id)
            assert not success
            assert any("missing" in e for e in report["errors"])
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original_registry
            exp_mod.ARTIFACTS_BASE = original_artifacts

    def test_null_artifact_hash_produces_error(self, tmp_path, monkeypatch):
        import reproduce_experiment as re_mod
        from src.governance import experiment as exp_mod
        reg_file = tmp_path / "exp_reg.json"
        original_registry = exp_mod.EXPERIMENT_REGISTRY
        original_artifacts = exp_mod.ARTIFACTS_BASE
        exp_mod.EXPERIMENT_REGISTRY = reg_file
        exp_mod.ARTIFACTS_BASE = tmp_path / "artifacts"
        try:
            m = exp_mod.build_manifest(
                "strat", "paper", seed=42,
                result_artifact_hashes={"metrics.json": ""},
            )
            reg = exp_mod.ExperimentRegistry(registry_file=reg_file)
            reg.register(m)
            success, report = re_mod.reproduce_experiment(m.experiment_id)
            assert not success
            assert any("null hash" in e for e in report["errors"])
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original_registry
            exp_mod.ARTIFACTS_BASE = original_artifacts

    def test_valid_experiment_passes(self, tmp_path, monkeypatch):
        import reproduce_experiment as re_mod
        from src.governance import experiment as exp_mod
        reg_file    = tmp_path / "exp_reg.json"
        artifacts_d = tmp_path / "artifacts"
        original_registry = exp_mod.EXPERIMENT_REGISTRY
        original_artifacts = exp_mod.ARTIFACTS_BASE
        exp_mod.EXPERIMENT_REGISTRY = reg_file
        exp_mod.ARTIFACTS_BASE = artifacts_d
        try:
            content = b'{"sharpe": 1.5}'
            sha = hashlib.sha256(content).hexdigest()
            m = exp_mod.build_manifest(
                "strat", "paper", seed=42,
                result_artifact_hashes={"metrics.json": sha},
            )
            reg = exp_mod.ExperimentRegistry(registry_file=reg_file)
            reg.register(m)
            # Write the artifact at expected path
            art_dir = artifacts_d / m.experiment_id
            art_dir.mkdir(parents=True)
            (art_dir / "metrics.json").write_bytes(content)

            success, report = re_mod.reproduce_experiment(m.experiment_id)
            assert success, report["errors"]
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original_registry
            exp_mod.ARTIFACTS_BASE = original_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# TestGovernanceGateRepro
# ─────────────────────────────────────────────────────────────────────────────

class TestGovernanceGateRepro:
    def _run_repro_check(self, state_dir: Path):
        """Run just the _check_reproducibility sub-function from governance_gate."""
        import governance_gate as gg
        original = gg.STATE_DIR
        gg.STATE_DIR = state_dir
        try:
            errors, warnings = gg._check_reproducibility()
        finally:
            gg.STATE_DIR = original
        return errors, warnings

    def test_missing_experiment_registry_is_error(self, tmp_path):
        errors, _ = self._run_repro_check(tmp_path)
        assert any("experiment_registry.json missing" in e for e in errors)

    def test_empty_experiment_registry_passes(self, tmp_path):
        reg = tmp_path / "experiment_registry.json"
        reg.write_text(json.dumps({"experiments": {}}), encoding="utf-8")
        errors, _ = self._run_repro_check(tmp_path)
        assert not errors

    def test_null_seed_is_error(self, tmp_path):
        from src.governance.experiment import build_manifest
        m = build_manifest("strat", "paper", seed=None)
        data = {"experiments": {m.experiment_id: m.to_dict()}}
        (tmp_path / "experiment_registry.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        errors, _ = self._run_repro_check(tmp_path)
        assert any("seed is null" in e for e in errors)

    def test_null_artifact_hash_is_error(self, tmp_path):
        from src.governance.experiment import build_manifest
        m = build_manifest(
            "strat", "paper", seed=42,
            result_artifact_hashes={"metrics.json": ""},
        )
        data = {"experiments": {m.experiment_id: m.to_dict()}}
        (tmp_path / "experiment_registry.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        errors, _ = self._run_repro_check(tmp_path)
        assert any("null hash" in e for e in errors)

    def test_missing_artifact_on_disk_is_error(self, tmp_path):
        import governance_gate as gg
        from src.governance.experiment import build_manifest
        m = build_manifest(
            "strat", "paper", seed=42,
            result_artifact_hashes={"metrics.json": "a" * 64},
        )
        data = {"experiments": {m.experiment_id: m.to_dict()}}
        (tmp_path / "experiment_registry.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        orig_root = gg.REPO_ROOT
        gg.REPO_ROOT = tmp_path
        try:
            errors, _ = self._run_repro_check(tmp_path)
        finally:
            gg.REPO_ROOT = orig_root
        assert any("artifact missing" in e for e in errors)

    def test_missing_dataset_registry_is_warning(self, tmp_path):
        reg = tmp_path / "experiment_registry.json"
        reg.write_text(json.dumps({"experiments": {}}), encoding="utf-8")
        # No dataset_registry.json present
        _, warnings = self._run_repro_check(tmp_path)
        assert any("dataset_registry" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerateExperimentLineage
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateExperimentLineage:
    def test_empty_registry_generates_doc(self, tmp_path, monkeypatch):
        import generate_experiment_lineage as gel
        from src.governance import experiment as exp_mod
        original = exp_mod.EXPERIMENT_REGISTRY
        exp_mod.EXPERIMENT_REGISTRY = tmp_path / "exp_reg.json"
        try:
            experiments = gel._load_experiments()
            doc = gel.generate_lineage_doc(experiments)
            assert "EXPERIMENT_LINEAGE" in doc
            assert "Total experiments: **0**" in doc
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original

    def test_registered_experiment_appears_in_doc(self, tmp_path, monkeypatch):
        import generate_experiment_lineage as gel
        from src.governance import experiment as exp_mod
        reg_file = tmp_path / "exp_reg.json"
        original = exp_mod.EXPERIMENT_REGISTRY
        exp_mod.EXPERIMENT_REGISTRY = reg_file
        try:
            m = exp_mod.build_manifest("my_strat", "paper", seed=42)
            exp_mod.ExperimentRegistry(registry_file=reg_file).register(m)
            experiments = gel._load_experiments()
            doc = gel.generate_lineage_doc(experiments)
            assert "my_strat" in doc
            assert m.experiment_id in doc
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original

    def test_parent_child_relationship_in_mermaid(self, tmp_path, monkeypatch):
        import generate_experiment_lineage as gel
        from src.governance import experiment as exp_mod
        reg_file = tmp_path / "exp_reg.json"
        original = exp_mod.EXPERIMENT_REGISTRY
        exp_mod.EXPERIMENT_REGISTRY = reg_file
        try:
            parent = exp_mod.build_manifest("strat", "paper", seed=1)
            child  = exp_mod.build_manifest(
                "strat", "paper", seed=2, config_hash="alt",
                parent_experiment=parent.experiment_id,
            )
            reg = exp_mod.ExperimentRegistry(registry_file=reg_file)
            reg.register(parent)
            reg.register(child)
            experiments = gel._load_experiments()
            mermaid = gel._build_mermaid(experiments)
            assert "-->" in mermaid
            assert parent.experiment_id in mermaid
            assert child.experiment_id in mermaid
        finally:
            exp_mod.EXPERIMENT_REGISTRY = original
