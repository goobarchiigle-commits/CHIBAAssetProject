"""tests/research/features/test_cache.py — FeatureCache and FeatureArtifacts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.features.cache import (
    FeatureCache,
    FeatureArtifacts,
    CacheCollisionError,
    CacheChecksumError,
    hash_dataframe,
    hash_params,
    build_cache_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    return FeatureCache(base_dir=tmp_path / "features")


@pytest.fixture
def artifacts(tmp_path):
    return FeatureArtifacts(base_dir=tmp_path / "raw")


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.RangeIndex(5),
    )


@pytest.fixture
def sample_key():
    return build_cache_key(
        feature_name="rolling_volatility",
        feature_version="1.0",
        dataset_hash="abc123",
        parameter_hash="def456",
        code_version="1.0_cafebabe12345678",
    )


@pytest.fixture
def sample_metadata(sample_key):
    return {
        "feature_name": "rolling_volatility",
        "feature_version": "1.0",
        "dataset_hash": "abc123",
        "parameter_hash": "def456",
        "code_version": "1.0_cafebabe12345678",
        "params": {"window": 20},
    }


# ---------------------------------------------------------------------------
# TestHashDataframe
# ---------------------------------------------------------------------------

class TestHashDataframe:
    def test_same_df_same_hash(self, sample_df):
        assert hash_dataframe(sample_df) == hash_dataframe(sample_df)

    def test_different_values_different_hash(self, sample_df):
        df2 = sample_df.copy()
        df2.iloc[0, 0] = 999.0
        assert hash_dataframe(sample_df) != hash_dataframe(df2)

    def test_column_order_independent(self, sample_df):
        df_with_extra = sample_df.copy()
        df_with_extra["volume"] = 1000.0
        df_reordered = df_with_extra[["volume", "close"]]
        assert hash_dataframe(df_with_extra) == hash_dataframe(df_reordered)

    def test_different_index_different_hash(self, sample_df):
        df2 = sample_df.copy()
        df2.index = pd.RangeIndex(start=10, stop=15)
        assert hash_dataframe(sample_df) != hash_dataframe(df2)

    def test_empty_df_does_not_raise(self):
        df = pd.DataFrame({"close": []})
        h = hash_dataframe(df)
        assert isinstance(h, str) and len(h) == 64

    def test_nan_handled_deterministically(self):
        import numpy as np
        df = pd.DataFrame({"close": [1.0, float("nan"), 3.0]})
        assert hash_dataframe(df) == hash_dataframe(df)

    def test_hash_is_64_char_hex(self, sample_df):
        h = hash_dataframe(sample_df)
        assert len(h) == 64
        int(h, 16)


# ---------------------------------------------------------------------------
# TestHashParams
# ---------------------------------------------------------------------------

class TestHashParams:
    def test_empty_params(self):
        assert hash_params({}) == hash_params({})

    def test_same_params_same_hash(self):
        p = {"window": 20, "min_periods": 5}
        assert hash_params(p) == hash_params(p)

    def test_key_order_independent(self):
        p1 = {"a": 1, "b": 2}
        p2 = {"b": 2, "a": 1}
        assert hash_params(p1) == hash_params(p2)

    def test_different_values_different_hash(self):
        assert hash_params({"window": 20}) != hash_params({"window": 30})

    def test_hash_is_16_char_hex(self):
        h = hash_params({"window": 20})
        assert len(h) == 16
        int(h, 16)


# ---------------------------------------------------------------------------
# TestBuildCacheKey
# ---------------------------------------------------------------------------

class TestBuildCacheKey:
    def test_deterministic(self):
        kwargs = dict(
            feature_name="f",
            feature_version="1.0",
            dataset_hash="d",
            parameter_hash="p",
            code_version="c",
        )
        assert build_cache_key(**kwargs) == build_cache_key(**kwargs)

    def test_differs_on_each_component(self):
        base = dict(feature_name="f", feature_version="1.0", dataset_hash="d",
                    parameter_hash="p", code_version="c")
        key0 = build_cache_key(**base)
        for field, val in [
            ("feature_name", "g"),
            ("feature_version", "2.0"),
            ("dataset_hash", "d2"),
            ("parameter_hash", "p2"),
            ("code_version", "c2"),
        ]:
            modified = {**base, field: val}
            assert build_cache_key(**modified) != key0, f"key should differ when {field} changes"

    def test_key_is_64_char_hex(self, sample_key):
        assert len(sample_key) == 64
        int(sample_key, 16)


# ---------------------------------------------------------------------------
# TestFeatureArtifacts
# ---------------------------------------------------------------------------

class TestFeatureArtifacts:
    def test_write_and_read_bytes(self, artifacts):
        sha = artifacts.write_bytes("test.bin", b"hello world")
        assert artifacts.read_bytes("test.bin") == b"hello world"

    def test_write_returns_sha256(self, artifacts):
        import hashlib
        content = b"deterministic"
        sha = artifacts.write_bytes("f.bin", content)
        assert sha == hashlib.sha256(content).hexdigest()

    def test_collision_raises(self, artifacts):
        artifacts.write_bytes("dup.bin", b"first")
        with pytest.raises(CacheCollisionError):
            artifacts.write_bytes("dup.bin", b"second")

    def test_read_missing_raises(self, artifacts):
        with pytest.raises(FileNotFoundError):
            artifacts.read_bytes("nonexistent.bin")

    def test_exists_true_after_write(self, artifacts):
        artifacts.write_bytes("x.bin", b"x")
        assert artifacts.exists("x.bin")

    def test_exists_false_before_write(self, artifacts):
        assert not artifacts.exists("missing.bin")

    def test_checksum_matches(self, artifacts):
        import hashlib
        content = b"checksum test"
        artifacts.write_bytes("chk.bin", content)
        expected = hashlib.sha256(content).hexdigest()
        assert artifacts.checksum("chk.bin") == expected

    def test_verify_true_on_correct(self, artifacts):
        import hashlib
        content = b"verify me"
        sha = artifacts.write_bytes("v.bin", content)
        assert artifacts.verify("v.bin", sha)

    def test_verify_false_on_wrong_sha(self, artifacts):
        artifacts.write_bytes("v2.bin", b"data")
        assert not artifacts.verify("v2.bin", "0" * 64)

    def test_list_files_sorted(self, artifacts):
        artifacts.write_bytes("b.bin", b"b")
        artifacts.write_bytes("a.bin", b"a")
        assert artifacts.list_files() == ["a.bin", "b.bin"]


# ---------------------------------------------------------------------------
# TestFeatureCache
# ---------------------------------------------------------------------------

class TestFeatureCache:
    def test_miss_on_empty_cache(self, cache, sample_key):
        assert not cache.has(sample_key)

    def test_put_then_has(self, cache, sample_df, sample_key, sample_metadata):
        cache.put(sample_key, sample_df, sample_metadata)
        assert cache.has(sample_key)

    def test_get_after_put_equals_original(self, cache, sample_df, sample_key, sample_metadata):
        cache.put(sample_key, sample_df, sample_metadata)
        result = cache.get(sample_key)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_get_miss_raises(self, cache, sample_key):
        with pytest.raises(KeyError, match="cache miss"):
            cache.get(sample_key)

    def test_collision_on_duplicate_key(self, cache, sample_df, sample_key, sample_metadata):
        cache.put(sample_key, sample_df, sample_metadata)
        with pytest.raises(CacheCollisionError):
            cache.put(sample_key, sample_df, sample_metadata)

    def test_metadata_stored_and_retrievable(self, cache, sample_df, sample_key, sample_metadata):
        cache.put(sample_key, sample_df, sample_metadata)
        meta = cache.get_metadata(sample_key)
        assert meta["feature_name"] == "rolling_volatility"
        assert meta["feature_version"] == "1.0"
        assert "data_checksum" in meta
        assert "cached_at" in meta
        assert "row_count" in meta

    def test_checksum_failure_on_corrupted_parquet(self, cache, sample_df, sample_key, sample_metadata, tmp_path):
        cache.put(sample_key, sample_df, sample_metadata)
        # Corrupt the parquet file directly
        p = cache.base_dir / f"{sample_key}.parquet"
        p.write_bytes(b"corrupted_garbage_data")
        with pytest.raises(CacheChecksumError):
            cache.get(sample_key)

    def test_put_returns_checksum_string(self, cache, sample_df, sample_key, sample_metadata):
        sha = cache.put(sample_key, sample_df, sample_metadata)
        assert isinstance(sha, str) and len(sha) == 64

    def test_different_dfs_different_keys(self, cache, sample_df, sample_metadata):
        df2 = sample_df.copy()
        df2["close"] = df2["close"] + 1.0
        key1 = build_cache_key("f", "1.0", hash_dataframe(sample_df), "p", "c")
        key2 = build_cache_key("f", "1.0", hash_dataframe(df2), "p", "c")
        assert key1 != key2

    def test_multicolumn_df_roundtrip(self, cache):
        df = pd.DataFrame({
            "rolling_vol": [0.01, 0.02, float("nan"), 0.04, 0.05],
            "atr": [1.5, 1.6, 1.7, 1.8, 1.9],
        })
        key = build_cache_key("multi", "1.0", "dhash", "phash", "chash")
        cache.put(key, df, {"feature_name": "multi", "feature_version": "1.0",
                            "dataset_hash": "d", "parameter_hash": "p", "code_version": "c", "params": {}})
        result = cache.get(key)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)
