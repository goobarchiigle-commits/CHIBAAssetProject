"""tests/research/features/test_snapshot.py — FeatureSnapshot immutability and integrity."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.features.snapshot import FeatureSnapshot
from src.research.features.cache import CacheCollisionError, CacheChecksumError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def snap(tmp_path):
    return FeatureSnapshot("snap_test_001", base_dir=tmp_path / "snapshots")


@pytest.fixture
def feature_outputs():
    return {
        "vol": pd.DataFrame({"rolling_vol": [0.01, 0.02, 0.03]}, index=pd.RangeIndex(3)),
        "mom": pd.DataFrame({"momentum": [0.1, -0.05, 0.07]}, index=pd.RangeIndex(3)),
    }


@pytest.fixture
def write_kwargs():
    return dict(
        dataset_hash="d" * 64,
        feature_hashes={"vol": "fh_vol", "mom": "fh_mom"},
        schema_hashes={"vol": "sh_vol", "mom": "sh_mom"},
        timestamp_start="2024-01-01",
        timestamp_end="2024-12-31",
    )


# ---------------------------------------------------------------------------
# TestExists
# ---------------------------------------------------------------------------

class TestExists:
    def test_not_exists_before_write(self, snap):
        assert not snap.exists()

    def test_exists_after_write(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        assert snap.exists()


# ---------------------------------------------------------------------------
# TestWrite
# ---------------------------------------------------------------------------

class TestWrite:
    def test_write_returns_manifest_sha(self, snap, feature_outputs, write_kwargs):
        sha = snap.write(feature_outputs, **write_kwargs)
        assert isinstance(sha, str) and len(sha) == 64

    def test_overwrite_raises(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        with pytest.raises(CacheCollisionError):
            snap.write(feature_outputs, **write_kwargs)

    def test_manifest_contains_snapshot_id(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        m = snap.manifest()
        assert m["snapshot_id"] == "snap_test_001"

    def test_manifest_contains_dataset_hash(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        m = snap.manifest()
        assert m["dataset_hash"] == "d" * 64

    def test_manifest_contains_timestamps(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        m = snap.manifest()
        assert m["timestamp_start"] == "2024-01-01"
        assert m["timestamp_end"] == "2024-12-31"

    def test_manifest_feature_entries(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        m = snap.manifest()
        assert "vol" in m["features"]
        assert "mom" in m["features"]

    def test_manifest_has_checksum_field(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        m = snap.manifest()
        assert "manifest_checksum" in m

    def test_extra_metadata_stored(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs, extra_metadata={"fold_id": 3})
        m = snap.manifest()
        assert m["extra"]["fold_id"] == 3

    def test_feature_hashes_in_manifest(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        m = snap.manifest()
        assert m["features"]["vol"]["feature_hash"] == "fh_vol"


# ---------------------------------------------------------------------------
# TestRead
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_returns_dataframe(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        df = snap.read("vol")
        assert isinstance(df, pd.DataFrame)

    def test_read_values_correct(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        result = snap.read("vol")
        pd.testing.assert_frame_equal(result, feature_outputs["vol"])

    def test_read_nonexistent_feature_raises(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        with pytest.raises(KeyError, match="not in snapshot"):
            snap.read("nonexistent_feature")

    def test_read_missing_snapshot_raises(self, snap):
        with pytest.raises(FileNotFoundError):
            snap.manifest()

    def test_checksum_failure_on_corrupted_file(self, snap, feature_outputs, write_kwargs, tmp_path):
        snap.write(feature_outputs, **write_kwargs)
        # Corrupt vol.parquet directly
        p = snap.snapshot_dir / "vol.parquet"
        p.write_bytes(b"garbage_data")
        with pytest.raises(CacheChecksumError):
            snap.read("vol")


# ---------------------------------------------------------------------------
# TestFeatureNames
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_feature_names_sorted(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        assert snap.feature_names() == ["mom", "vol"]

    def test_feature_names_matches_written(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        assert set(snap.feature_names()) == set(feature_outputs.keys())


# ---------------------------------------------------------------------------
# TestVerifyManifest
# ---------------------------------------------------------------------------

class TestVerifyManifest:
    def test_valid_manifest_returns_true(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        assert snap.verify_manifest()

    def test_corrupted_manifest_returns_false(self, snap, feature_outputs, write_kwargs):
        snap.write(feature_outputs, **write_kwargs)
        p = snap.snapshot_dir / "manifest.json"
        content = p.read_text(encoding="utf-8")
        p.write_text(content.replace('"snap_test_001"', '"tampered_id"'), encoding="utf-8")
        assert not snap.verify_manifest()
