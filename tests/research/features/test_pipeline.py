"""tests/research/features/test_pipeline.py — FeaturePipeline cache reuse and dedup."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.features.cache import FeatureCache
from src.research.features.contract import FeatureContract
from src.research.features.pipeline import FeaturePipeline, FeatureSpec, FeaturePipelineError
from src.research.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _CountingFeature(FeatureContract):
    """Tracks how many times compute() is called to verify dedup."""
    feature_name = "counting"
    feature_version = "1.0"
    call_count = 0

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, window: int = 5, **_: Any) -> pd.DataFrame:
        _CountingFeature.call_count += 1
        ret = df["close"].pct_change(window)
        return pd.DataFrame({"counting_out": ret}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"counting_out": float}


class _VolFeature(FeatureContract):
    feature_name = "vol"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, window: int = 10, **_: Any) -> pd.DataFrame:
        std = df["close"].rolling(window, min_periods=window).std()
        return pd.DataFrame({"vol": std}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"vol": float}


class _MissingOutputFeature(FeatureContract):
    """Declares output column but doesn't include it — for validation test."""
    feature_name = "bad_output"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return pd.DataFrame({"wrong_col": df["close"]}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"declared_col": float}


@pytest.fixture(autouse=True)
def reset_call_count():
    _CountingFeature.call_count = 0
    yield
    _CountingFeature.call_count = 0


@pytest.fixture
def registry():
    r = FeatureRegistry()
    r.register(_CountingFeature)
    r.register(_VolFeature)
    return r


@pytest.fixture
def cache(tmp_path):
    return FeatureCache(base_dir=tmp_path / "features")


@pytest.fixture
def pipeline(registry, cache):
    return FeaturePipeline(registry=registry, cache=cache)


@pytest.fixture
def df():
    return pd.DataFrame(
        {"close": [float(i) for i in range(50)]},
        index=pd.RangeIndex(50),
    )


# ---------------------------------------------------------------------------
# TestBasicCompute
# ---------------------------------------------------------------------------

class TestBasicCompute:
    def test_single_spec_returns_one_result(self, pipeline, df):
        results = pipeline.compute(df, [FeatureSpec("counting")])
        assert len(results) == 1

    def test_result_is_dataframe(self, pipeline, df):
        results = pipeline.compute(df, [FeatureSpec("counting")])
        assert isinstance(results[0], pd.DataFrame)

    def test_output_has_declared_column(self, pipeline, df):
        results = pipeline.compute(df, [FeatureSpec("counting")])
        assert "counting_out" in results[0].columns

    def test_empty_specs_returns_empty(self, pipeline, df):
        results = pipeline.compute(df, [])
        assert results == []

    def test_multiple_features(self, pipeline, df):
        specs = [FeatureSpec("counting"), FeatureSpec("vol")]
        results = pipeline.compute(df, specs)
        assert len(results) == 2
        assert "counting_out" in results[0].columns
        assert "vol" in results[1].columns

    def test_result_order_matches_spec_order(self, pipeline, df):
        specs = [FeatureSpec("vol"), FeatureSpec("counting")]
        results = pipeline.compute(df, specs)
        assert "vol" in results[0].columns
        assert "counting_out" in results[1].columns


# ---------------------------------------------------------------------------
# TestCacheReuse
# ---------------------------------------------------------------------------

class TestCacheReuse:
    def test_second_run_hits_cache(self, pipeline, df):
        pipeline.compute(df, [FeatureSpec("counting")])
        assert _CountingFeature.call_count == 1
        pipeline.compute(df, [FeatureSpec("counting")])
        assert _CountingFeature.call_count == 1  # no recomputation

    def test_different_params_no_cache_hit(self, pipeline, df):
        pipeline.compute(df, [FeatureSpec("counting", {"window": 5})])
        assert _CountingFeature.call_count == 1
        pipeline.compute(df, [FeatureSpec("counting", {"window": 10})])
        assert _CountingFeature.call_count == 2

    def test_different_df_no_cache_hit(self, pipeline, df):
        pipeline.compute(df, [FeatureSpec("counting")])
        assert _CountingFeature.call_count == 1
        df2 = df.copy()
        df2["close"] = df2["close"] + 99.0
        pipeline.compute(df2, [FeatureSpec("counting")])
        assert _CountingFeature.call_count == 2

    def test_cache_hit_result_equals_computed(self, pipeline, df):
        r1 = pipeline.compute(df, [FeatureSpec("counting")])[0]
        r2 = pipeline.compute(df, [FeatureSpec("counting")])[0]
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestInRunDeduplication
# ---------------------------------------------------------------------------

class TestInRunDeduplication:
    def test_duplicate_spec_in_same_call_computes_once(self, pipeline, df):
        specs = [FeatureSpec("counting"), FeatureSpec("counting")]
        results = pipeline.compute(df, specs)
        assert _CountingFeature.call_count == 1
        assert len(results) == 2
        pd.testing.assert_frame_equal(results[0], results[1])

    def test_three_duplicates_computes_once(self, pipeline, df):
        specs = [FeatureSpec("counting")] * 3
        pipeline.compute(df, specs)
        assert _CountingFeature.call_count == 1


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_input_column_raises(self, tmp_path, df):
        class _NeedsVolume(FeatureContract):
            feature_name = "needs_volume"
            feature_version = "1.0"
            def required_inputs(self): return ["close", "volume"]
            def compute(self, df, **_): return pd.DataFrame({"x": df["close"]}, index=df.index)
            def output_schema(self): return {"x": float}

        r = FeatureRegistry()
        r.register(_NeedsVolume)
        c = FeatureCache(base_dir=tmp_path / "fc")
        p = FeaturePipeline(registry=r, cache=c)
        df_no_vol = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(FeaturePipelineError, match="missing required inputs"):
            p.compute(df_no_vol, [FeatureSpec("needs_volume")])

    def test_missing_output_column_raises(self, tmp_path, df):
        r = FeatureRegistry()
        r.register(_MissingOutputFeature)
        c = FeatureCache(base_dir=tmp_path / "fc2")
        p = FeaturePipeline(registry=r, cache=c)
        with pytest.raises(FeaturePipelineError, match="missing declared output columns"):
            p.compute(df, [FeatureSpec("bad_output")])

    def test_unknown_feature_raises(self, pipeline, df):
        with pytest.raises(KeyError):
            pipeline.compute(df, [FeatureSpec("nonexistent")])
