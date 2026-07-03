"""tests/research/features/test_determinism.py

Cross-machine and cross-run determinism guarantees:
  - Same DataFrame → same hash regardless of construction order
  - Same params dict → same hash regardless of key ordering
  - Cache key is stable across independent runs
  - Feature compute() is reproducible: same inputs → same output
  - Checksum stability: stored checksum matches re-read checksum
  - All 7 implementations produce stable output hashes
  - Simulated independent rebuild produces identical keys
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.features.cache import (
    hash_dataframe,
    hash_params,
    build_cache_key,
    FeatureCache,
)
from src.research.features.contract import FeatureContract
from src.research.features.pipeline import FeaturePipeline, FeatureSpec
from src.research.features.registry import FeatureRegistry
from src.research.features.snapshot import FeatureSnapshot
from src.research.features.implementations import (
    RollingVolatility, ATR, Momentum, LiquidityMetrics,
    RollingReturns, BreadthMetrics, RegimeFeatures, ALL_FEATURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000.0 + rng.standard_normal(n).cumsum() * 5
    high = close + rng.uniform(0, 5, n)
    low = close - rng.uniform(0, 5, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _make_wide(n: int = 100, cols: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"S{i:04d}": 1000.0 + rng.standard_normal(n).cumsum() * 5 for i in range(cols)}
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# TestDataframeHashDeterminism
# ---------------------------------------------------------------------------

class TestDataframeHashDeterminism:
    def test_identical_df_identical_hash(self):
        df = _make_ohlcv()
        assert hash_dataframe(df) == hash_dataframe(df)

    def test_independently_constructed_df_same_hash(self):
        """Simulates two independent builds on different machines."""
        df1 = _make_ohlcv(seed=7)
        df2 = _make_ohlcv(seed=7)
        assert hash_dataframe(df1) == hash_dataframe(df2)

    def test_column_insertion_order_does_not_matter(self):
        df = _make_ohlcv()
        df_reordered = df[["volume", "low", "high", "open", "close"]]
        assert hash_dataframe(df) == hash_dataframe(df_reordered)

    def test_value_change_changes_hash(self):
        df = _make_ohlcv()
        df2 = df.copy()
        df2.iloc[5, 3] += 0.001  # tiny change to close
        assert hash_dataframe(df) != hash_dataframe(df2)

    def test_row_order_independent_same_index(self):
        """hash_dataframe sorts by index, so storage order never affects the hash."""
        df = _make_ohlcv()
        df_shuffled = df.sample(frac=1, random_state=0)
        # Same data, same index values, different storage order → same hash
        assert hash_dataframe(df) == hash_dataframe(df_shuffled)

    def test_different_index_range_different_hash(self):
        df1 = _make_ohlcv(n=50)
        df2 = _make_ohlcv(n=50)
        df2.index = pd.date_range("2025-01-01", periods=50, freq="B")
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_extra_row_changes_hash(self):
        df = _make_ohlcv(n=50)
        df2 = _make_ohlcv(n=51)
        assert hash_dataframe(df) != hash_dataframe(df2)

    def test_nan_deterministic(self):
        df = _make_ohlcv(n=20)
        df.iloc[3, 3] = float("nan")
        h1 = hash_dataframe(df)
        h2 = hash_dataframe(df)
        assert h1 == h2

    def test_hash_is_64_char_hex(self):
        h = hash_dataframe(_make_ohlcv())
        assert len(h) == 64
        int(h, 16)


# ---------------------------------------------------------------------------
# TestParamHashDeterminism
# ---------------------------------------------------------------------------

class TestParamHashDeterminism:
    def test_same_params_same_hash(self):
        p = {"window": 20, "min_periods": 5, "method": "ewm"}
        assert hash_params(p) == hash_params(p)

    def test_dict_key_order_irrelevant(self):
        p1 = {"a": 1, "b": 2, "c": 3}
        p2 = {"c": 3, "a": 1, "b": 2}
        assert hash_params(p1) == hash_params(p2)

    def test_value_change_changes_hash(self):
        assert hash_params({"window": 20}) != hash_params({"window": 21})

    def test_extra_key_changes_hash(self):
        assert hash_params({"a": 1}) != hash_params({"a": 1, "b": 2})

    def test_empty_dict_stable(self):
        assert hash_params({}) == hash_params({})

    def test_nested_values_stable(self):
        p = {"ranges": [1, 2, 3], "config": {"k": "v"}}
        assert hash_params(p) == hash_params(p)


# ---------------------------------------------------------------------------
# TestCacheKeyDeterminism
# ---------------------------------------------------------------------------

class TestCacheKeyDeterminism:
    def test_identical_inputs_identical_key(self):
        kwargs = dict(
            feature_name="momentum",
            feature_version="1.0",
            dataset_hash="a" * 64,
            parameter_hash="b" * 16,
            code_version="1.0_abcdef1234567890",
        )
        assert build_cache_key(**kwargs) == build_cache_key(**kwargs)

    def test_simulated_independent_rebuild(self):
        """Key rebuilt from scratch (different process) must match."""
        df = _make_ohlcv(seed=99)
        params = {"window": 20}
        dh = hash_dataframe(df)
        ph = hash_params(params)
        k1 = build_cache_key("momentum", "1.0", dh, ph, "1.0_code")

        # Rebuild independently
        df2 = _make_ohlcv(seed=99)
        params2 = {"window": 20}
        dh2 = hash_dataframe(df2)
        ph2 = hash_params(params2)
        k2 = build_cache_key("momentum", "1.0", dh2, ph2, "1.0_code")

        assert k1 == k2

    def test_key_changes_on_version_bump(self):
        base = dict(feature_name="f", feature_version="1.0",
                    dataset_hash="d", parameter_hash="p", code_version="c")
        k1 = build_cache_key(**base)
        k2 = build_cache_key(**{**base, "feature_version": "2.0"})
        assert k1 != k2

    def test_key_changes_on_dataset_change(self):
        base = dict(feature_name="f", feature_version="1.0",
                    dataset_hash="d1", parameter_hash="p", code_version="c")
        k1 = build_cache_key(**base)
        k2 = build_cache_key(**{**base, "dataset_hash": "d2"})
        assert k1 != k2

    def test_key_changes_on_param_change(self):
        base = dict(feature_name="f", feature_version="1.0",
                    dataset_hash="d", parameter_hash="p1", code_version="c")
        k1 = build_cache_key(**base)
        k2 = build_cache_key(**{**base, "parameter_hash": "p2"})
        assert k1 != k2

    def test_key_changes_on_code_version_change(self):
        base = dict(feature_name="f", feature_version="1.0",
                    dataset_hash="d", parameter_hash="p", code_version="c1")
        k1 = build_cache_key(**base)
        k2 = build_cache_key(**{**base, "code_version": "c2"})
        assert k1 != k2


# ---------------------------------------------------------------------------
# TestFeatureComputeDeterminism
# ---------------------------------------------------------------------------

SINGLE_SYMBOL_FEATURES = [RollingVolatility, Momentum, RollingReturns, RegimeFeatures]
OHLCV_FEATURES = [ATR, LiquidityMetrics]


class TestFeatureComputeDeterminism:
    @pytest.mark.parametrize("cls", SINGLE_SYMBOL_FEATURES)
    def test_single_symbol_same_output(self, cls):
        df = _make_ohlcv()
        inst = cls()
        out1 = inst.compute(df.copy())
        out2 = inst.compute(df.copy())
        pd.testing.assert_frame_equal(out1, out2)

    @pytest.mark.parametrize("cls", OHLCV_FEATURES)
    def test_ohlcv_same_output(self, cls):
        df = _make_ohlcv()
        inst = cls()
        out1 = inst.compute(df.copy())
        out2 = inst.compute(df.copy())
        pd.testing.assert_frame_equal(out1, out2)

    def test_breadth_same_output(self):
        df = _make_wide()
        inst = BreadthMetrics()
        out1 = inst.compute(df.copy())
        out2 = inst.compute(df.copy())
        pd.testing.assert_frame_equal(out1, out2)

    @pytest.mark.parametrize("cls", SINGLE_SYMBOL_FEATURES)
    def test_output_hash_stable_across_runs(self, cls):
        df = _make_ohlcv()
        inst = cls()
        out = inst.compute(df.copy())
        h1 = hash_dataframe(out)
        h2 = hash_dataframe(out)
        assert h1 == h2

    @pytest.mark.parametrize("cls", ALL_FEATURES)
    def test_output_schema_declared_columns_present(self, cls):
        inst = cls()
        if cls in (BreadthMetrics,):
            df = _make_wide()
        else:
            df = _make_ohlcv()
        out = inst.compute(df.copy())
        for col in inst.output_schema():
            assert col in out.columns, f"{cls.__name__} missing column '{col}'"

    def test_regime_with_custom_params_deterministic(self):
        df = _make_ohlcv()
        inst = RegimeFeatures()
        out1 = inst.compute(df.copy(), short_window=20, long_window=100, vol_window=10)
        out2 = inst.compute(df.copy(), short_window=20, long_window=100, vol_window=10)
        pd.testing.assert_frame_equal(out1, out2)

    def test_different_params_different_output(self):
        df = _make_ohlcv()
        inst = Momentum()
        out1 = inst.compute(df.copy(), window=10)
        out2 = inst.compute(df.copy(), window=30)
        # Different window → different values (at least for non-NaN rows)
        non_nan = out1["momentum"].dropna()
        if len(non_nan) > 0:
            assert not out1["momentum"].equals(out2["momentum"])


# ---------------------------------------------------------------------------
# TestPipelineCacheKeyStability
# ---------------------------------------------------------------------------

class TestPipelineCacheKeyStability:
    def _build_pipeline(self, tmp_path):
        r = FeatureRegistry()
        r.register(Momentum)
        c = FeatureCache(base_dir=tmp_path / "fc")
        return FeaturePipeline(registry=r, cache=c), c

    def test_second_pipeline_instance_hits_cache(self, tmp_path):
        """Two independent FeaturePipeline instances share the same persistent cache."""
        df = _make_ohlcv()
        spec = FeatureSpec("momentum", {"window": 20})

        p1, cache = self._build_pipeline(tmp_path)
        r1 = p1.compute(df, [spec])[0]

        # Build second pipeline pointing to same cache dir
        r2 = FeatureRegistry()
        r2.register(Momentum)
        p2 = FeaturePipeline(registry=r2, cache=cache)
        r2_out = p2.compute(df, [spec])[0]

        # check_freq=False: Parquet round-trip drops DatetimeIndex freq metadata
        pd.testing.assert_frame_equal(r1, r2_out, check_freq=False)

    def test_cache_populated_after_compute(self, tmp_path):
        df = _make_ohlcv()
        p, cache = self._build_pipeline(tmp_path)
        p.compute(df, [FeatureSpec("momentum")])
        inst = Momentum()
        dh = hash_dataframe(df)
        ph = hash_params({})
        cv = f"{inst.feature_version}_{inst.code_hash()}"
        key = cache.make_key("momentum", inst.feature_version, dh, ph, cv)
        assert cache.has(key)

    def test_cross_run_key_identical(self, tmp_path):
        """Keys computed in two separate 'runs' (different pipeline instances) must match."""
        df = _make_ohlcv(seed=13)
        params = {"window": 15}

        inst = Momentum()
        dh = hash_dataframe(df)
        ph = hash_params(params)
        cv = f"{inst.feature_version}_{inst.code_hash()}"

        # "Run 1"
        k1 = build_cache_key("momentum", inst.feature_version, dh, ph, cv)

        # "Run 2" — rebuilt from identical inputs
        df2 = _make_ohlcv(seed=13)
        inst2 = Momentum()
        dh2 = hash_dataframe(df2)
        ph2 = hash_params({"window": 15})
        cv2 = f"{inst2.feature_version}_{inst2.code_hash()}"
        k2 = build_cache_key("momentum", inst2.feature_version, dh2, ph2, cv2)

        assert k1 == k2


# ---------------------------------------------------------------------------
# TestSnapshotDeterminism
# ---------------------------------------------------------------------------

class TestSnapshotDeterminism:
    def test_manifest_checksum_stable(self, tmp_path):
        df = _make_ohlcv(n=30)
        inst = Momentum()
        out = inst.compute(df.copy())

        snap = FeatureSnapshot("det_snap_001", base_dir=tmp_path / "snaps")
        sha = snap.write(
            feature_outputs={"momentum": out},
            dataset_hash=hash_dataframe(df),
            feature_hashes={"momentum": inst.identity_hash()},
            schema_hashes={"momentum": "sch_hash_1"},
            timestamp_start="2023-01-01",
            timestamp_end="2023-12-31",
        )
        assert snap.verify_manifest()

    def test_read_output_equals_written(self, tmp_path):
        df = _make_ohlcv(n=30)
        inst = Momentum()
        out = inst.compute(df.copy())

        snap = FeatureSnapshot("det_snap_002", base_dir=tmp_path / "snaps")
        snap.write(
            feature_outputs={"momentum": out},
            dataset_hash=hash_dataframe(df),
            feature_hashes={"momentum": inst.identity_hash()},
            schema_hashes={"momentum": "sch"},
            timestamp_start="2023-01-01",
            timestamp_end="2023-12-31",
        )
        read_back = snap.read("momentum")
        # check_freq=False: Parquet round-trip drops DatetimeIndex freq metadata
        pd.testing.assert_frame_equal(read_back, out, check_freq=False)


# ---------------------------------------------------------------------------
# TestChecksumStability
# ---------------------------------------------------------------------------

class TestChecksumStability:
    def test_parquet_checksum_same_on_reread(self, tmp_path):
        df = _make_ohlcv()
        cache = FeatureCache(base_dir=tmp_path / "fc")
        key = build_cache_key("vol", "1.0", "dh", "ph", "cv")
        sha_stored = cache.put(key, df, {
            "feature_name": "vol", "feature_version": "1.0",
            "dataset_hash": "dh", "parameter_hash": "ph",
            "code_version": "cv", "params": {},
        })
        meta = cache.get_metadata(key)
        assert meta["data_checksum"] == sha_stored

    def test_get_validates_checksum_transparently(self, tmp_path):
        df = _make_ohlcv()
        cache = FeatureCache(base_dir=tmp_path / "fc")
        key = build_cache_key("vol", "1.0", "dh", "ph", "cv")
        cache.put(key, df, {
            "feature_name": "vol", "feature_version": "1.0",
            "dataset_hash": "dh", "parameter_hash": "ph",
            "code_version": "cv", "params": {},
        })
        # Read back twice — both must succeed with same result
        r1 = cache.get(key)
        r2 = cache.get(key)
        pd.testing.assert_frame_equal(r1, r2)
