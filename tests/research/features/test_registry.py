"""tests/research/features/test_registry.py — FeatureRegistry."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.features.registry import FeatureRegistry, FeatureRegistryError
from src.research.features.contract import FeatureContract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Alpha(FeatureContract):
    feature_name = "alpha"
    feature_version = "1.0"
    def required_inputs(self): return ["close"]
    def compute(self, df, **_): return pd.DataFrame({"a": df["close"]}, index=df.index)
    def output_schema(self): return {"a": float}


class _Beta(FeatureContract):
    feature_name = "beta"
    feature_version = "1.0"
    def required_inputs(self): return ["close", "volume"]
    def compute(self, df, **_): return pd.DataFrame({"b": df["close"]}, index=df.index)
    def output_schema(self): return {"b": float}


class _AlphaV2(FeatureContract):
    """Same name as _Alpha but different version → conflict."""
    feature_name = "alpha"
    feature_version = "2.0"
    def required_inputs(self): return ["close"]
    def compute(self, df, **_): return pd.DataFrame({"a": df["close"]}, index=df.index)
    def output_schema(self): return {"a": float}


class _AlphaConflictSchema(FeatureContract):
    """Same name+version as _Alpha but different output schema → conflict."""
    feature_name = "alpha"
    feature_version = "1.0"
    def required_inputs(self): return ["close"]
    def compute(self, df, **_): return pd.DataFrame({"a_new": df["close"]}, index=df.index)
    def output_schema(self): return {"a_new": float}


@pytest.fixture
def registry():
    return FeatureRegistry()


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_retrieve(self, registry):
        registry.register(_Alpha)
        assert registry.get("alpha") is _Alpha

    def test_instantiate_returns_instance(self, registry):
        registry.register(_Alpha)
        inst = registry.instantiate("alpha")
        assert isinstance(inst, _Alpha)

    def test_multiple_features(self, registry):
        registry.register(_Alpha)
        registry.register(_Beta)
        assert set(registry.all_features()) == {"alpha", "beta"}

    def test_all_features_sorted(self, registry):
        registry.register(_Beta)
        registry.register(_Alpha)
        assert registry.all_features() == ["alpha", "beta"]

    def test_get_unknown_raises(self, registry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get("unknown")

    def test_metadata_returns_correct(self, registry):
        registry.register(_Alpha)
        meta = registry.metadata("alpha")
        assert meta.feature_name == "alpha"
        assert meta.feature_version == "1.0"


# ---------------------------------------------------------------------------
# TestIdempotentRegistration
# ---------------------------------------------------------------------------

class TestIdempotentRegistration:
    def test_same_class_twice_is_idempotent(self, registry):
        registry.register(_Alpha)
        registry.register(_Alpha)  # should not raise
        assert registry.get("alpha") is _Alpha

    def test_all_features_no_duplicate(self, registry):
        registry.register(_Alpha)
        registry.register(_Alpha)
        assert registry.all_features().count("alpha") == 1


# ---------------------------------------------------------------------------
# TestConflictRejection
# ---------------------------------------------------------------------------

class TestConflictRejection:
    def test_different_version_raises(self, registry):
        registry.register(_Alpha)
        with pytest.raises(FeatureRegistryError, match="conflicting"):
            registry.register(_AlphaV2)

    def test_different_schema_same_version_raises(self, registry):
        registry.register(_Alpha)
        with pytest.raises(FeatureRegistryError, match="conflicting"):
            registry.register(_AlphaConflictSchema)


# ---------------------------------------------------------------------------
# TestLock
# ---------------------------------------------------------------------------

class TestLock:
    def test_register_after_lock_raises(self, registry):
        registry.register(_Alpha)
        registry.lock()
        with pytest.raises(FeatureRegistryError, match="locked"):
            registry.register(_Beta)

    def test_is_locked_property(self, registry):
        assert not registry.is_locked
        registry.lock()
        assert registry.is_locked

    def test_get_after_lock_works(self, registry):
        registry.register(_Alpha)
        registry.lock()
        assert registry.get("alpha") is _Alpha


# ---------------------------------------------------------------------------
# TestLineage
# ---------------------------------------------------------------------------

class TestLineage:
    def test_no_deps_returns_empty(self, registry):
        registry.register(_Alpha)
        assert registry.dependencies("alpha") == []

    def test_deps_stored_correctly(self, registry):
        registry.register(_Alpha)
        registry.register(_Beta, depends_on=["alpha"])
        assert registry.dependencies("beta") == ["alpha"]

    def test_deps_returns_copy(self, registry):
        registry.register(_Alpha)
        registry.register(_Beta, depends_on=["alpha"])
        deps = registry.dependencies("beta")
        deps.append("hacked")
        assert registry.dependencies("beta") == ["alpha"]


# ---------------------------------------------------------------------------
# TestManifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_deterministic(self, registry):
        registry.register(_Alpha)
        registry.register(_Beta)
        m1 = registry.to_manifest()
        m2 = registry.to_manifest()
        assert m1 == m2

    def test_manifest_contains_both_features(self, registry):
        registry.register(_Alpha)
        registry.register(_Beta)
        m = registry.to_manifest()
        assert "alpha" in m
        assert "beta" in m

    def test_manifest_hash_stable(self, registry):
        registry.register(_Alpha)
        h1 = registry.manifest_hash()
        h2 = registry.manifest_hash()
        assert h1 == h2

    def test_manifest_hash_changes_with_new_feature(self, registry):
        registry.register(_Alpha)
        h1 = registry.manifest_hash()
        registry.register(_Beta)
        h2 = registry.manifest_hash()
        assert h1 != h2
