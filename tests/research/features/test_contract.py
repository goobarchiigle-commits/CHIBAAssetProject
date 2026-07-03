"""tests/research/features/test_contract.py — FeatureContract ABC enforcement."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.features.contract import FeatureContract, FeatureMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _MinimalFeature(FeatureContract):
    feature_name = "test_feature"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return pd.DataFrame({"out": df["close"] * 2}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"out": float}


class _AltFeature(FeatureContract):
    feature_name = "alt_feature"
    feature_version = "2.0"

    def required_inputs(self) -> List[str]:
        return ["close", "volume"]

    def compute(self, df: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return pd.DataFrame({"x": df["close"]}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"x": float}


@pytest.fixture
def feat():
    return _MinimalFeature()


@pytest.fixture
def df():
    return pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.RangeIndex(5),
    )


# ---------------------------------------------------------------------------
# TestAbstractEnforcement
# ---------------------------------------------------------------------------

class TestAbstractEnforcement:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            FeatureContract()  # type: ignore

    def test_missing_required_inputs_raises(self):
        class _Bad(FeatureContract):
            feature_name = "bad"
            feature_version = "1.0"
            def compute(self, df, **_): return df
            def output_schema(self): return {}

        with pytest.raises(TypeError):
            _Bad()

    def test_missing_compute_raises(self):
        class _Bad(FeatureContract):
            feature_name = "bad"
            feature_version = "1.0"
            def required_inputs(self): return []
            def output_schema(self): return {}

        with pytest.raises(TypeError):
            _Bad()

    def test_missing_output_schema_raises(self):
        class _Bad(FeatureContract):
            feature_name = "bad"
            feature_version = "1.0"
            def required_inputs(self): return []
            def compute(self, df, **_): return df

        with pytest.raises(TypeError):
            _Bad()


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_returns_feature_metadata(self, feat):
        meta = feat.metadata()
        assert isinstance(meta, FeatureMetadata)

    def test_metadata_name(self, feat):
        assert feat.metadata().feature_name == "test_feature"

    def test_metadata_version(self, feat):
        assert feat.metadata().feature_version == "1.0"

    def test_metadata_output_columns_sorted(self, feat):
        # Output schema has one column "out"
        assert feat.metadata().output_columns == ("out",)

    def test_metadata_is_frozen(self, feat):
        meta = feat.metadata()
        with pytest.raises((AttributeError, TypeError)):
            meta.feature_name = "hacked"  # type: ignore

    def test_metadata_equality_same_class(self):
        assert _MinimalFeature().metadata() == _MinimalFeature().metadata()

    def test_metadata_inequality_different_class(self):
        assert _MinimalFeature().metadata() != _AltFeature().metadata()


# ---------------------------------------------------------------------------
# TestIdentityHash
# ---------------------------------------------------------------------------

class TestIdentityHash:
    def test_stable_same_instance(self, feat):
        assert feat.identity_hash() == feat.identity_hash()

    def test_stable_across_instances(self):
        h1 = _MinimalFeature().identity_hash()
        h2 = _MinimalFeature().identity_hash()
        assert h1 == h2

    def test_differs_between_features(self):
        assert _MinimalFeature().identity_hash() != _AltFeature().identity_hash()

    def test_hash_is_hex_string(self, feat):
        h = feat.identity_hash()
        assert isinstance(h, str)
        int(h, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# TestCodeHash
# ---------------------------------------------------------------------------

class TestCodeHash:
    def test_stable_same_class(self):
        assert _MinimalFeature().code_hash() == _MinimalFeature().code_hash()

    def test_differs_when_compute_changes(self):
        class _V1(FeatureContract):
            feature_name = "v1"
            feature_version = "1.0"
            def required_inputs(self): return ["close"]
            def compute(self, df, **_): return pd.DataFrame({"x": df["close"]}, index=df.index)
            def output_schema(self): return {"x": float}

        class _V2(FeatureContract):
            feature_name = "v2"
            feature_version = "1.0"
            def required_inputs(self): return ["close"]
            def compute(self, df, **_): return pd.DataFrame({"x": df["close"] * 2}, index=df.index)
            def output_schema(self): return {"x": float}

        assert _V1().code_hash() != _V2().code_hash()

    def test_hash_is_16_char_hex(self, feat):
        h = feat.code_hash()
        assert len(h) == 16
        int(h, 16)


# ---------------------------------------------------------------------------
# TestCompute
# ---------------------------------------------------------------------------

class TestCompute:
    def test_compute_returns_dataframe(self, feat, df):
        out = feat.compute(df)
        assert isinstance(out, pd.DataFrame)

    def test_compute_does_not_mutate_input(self, feat, df):
        original = df.copy()
        feat.compute(df)
        pd.testing.assert_frame_equal(df, original)

    def test_compute_output_has_correct_column(self, feat, df):
        out = feat.compute(df)
        assert "out" in out.columns

    def test_compute_preserves_index(self, feat, df):
        out = feat.compute(df)
        assert list(out.index) == list(df.index)
