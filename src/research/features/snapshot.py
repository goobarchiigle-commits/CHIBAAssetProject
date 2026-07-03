"""FeatureSnapshot — immutable, versioned snapshot of a feature output set.

Storage layout under base_dir/{snapshot_id}/:
  manifest.json          — metadata, per-feature hashes, timestamp range, checksum
  {feature_name}.parquet — feature DataFrame (snappy-compressed)

Append-only: write() raises CacheCollisionError if snapshot already exists.
Checksum validation on every read().
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.paths import FEATURE_SNAPSHOT_DIR
from src.research.features.cache import FeatureArtifacts, CacheCollisionError, CacheChecksumError


class FeatureSnapshot:
    """Write and read immutable feature output snapshots.

    A snapshot captures a named, timestamped set of feature DataFrames
    alongside their provenance metadata. Once written, no field or file
    may be changed — attempts raise CacheCollisionError.
    """

    def __init__(self, snapshot_id: str, base_dir: Optional[Path] = None) -> None:
        self.snapshot_id = snapshot_id
        root = Path(base_dir) if base_dir else FEATURE_SNAPSHOT_DIR
        self.snapshot_dir = root / snapshot_id
        self._artifacts = FeatureArtifacts(self.snapshot_dir)

    def exists(self) -> bool:
        return (self.snapshot_dir / "manifest.json").exists()

    def write(
        self,
        feature_outputs: Dict[str, pd.DataFrame],
        dataset_hash: str,
        feature_hashes: Dict[str, str],
        schema_hashes: Dict[str, str],
        timestamp_start: str,
        timestamp_end: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist feature_outputs. Returns manifest_checksum.

        Args:
            feature_outputs: {feature_name: output_df}
            dataset_hash: SHA256 of the source dataset
            feature_hashes: {feature_name: cache_key or identity_hash}
            schema_hashes: {feature_name: hash of output_schema()}
            timestamp_start: ISO8601 start of data covered
            timestamp_end: ISO8601 end of data covered
            extra_metadata: any additional metadata fields

        Raises:
            CacheCollisionError: snapshot already exists
        """
        if self.exists():
            raise CacheCollisionError(f"snapshot already exists: {self.snapshot_id}")

        per_feature: Dict[str, dict] = {}
        for fname in sorted(feature_outputs.keys()):
            df = feature_outputs[fname]
            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", compression="snappy", index=True)
            parquet_bytes = buf.getvalue()
            data_sha = hashlib.sha256(parquet_bytes).hexdigest()
            self._artifacts.write_bytes(f"{fname}.parquet", parquet_bytes)
            per_feature[fname] = {
                "feature_hash": feature_hashes.get(fname, ""),
                "schema_hash": schema_hashes.get(fname, ""),
                "data_checksum": data_sha,
                "row_count": len(df),
                "columns": list(df.columns),
            }

        manifest: Dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "dataset_hash": dataset_hash,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "features": per_feature,
        }
        if extra_metadata:
            manifest["extra"] = extra_metadata

        # Checksum the manifest before writing (sans the checksum field itself)
        manifest_body = json.dumps(manifest, sort_keys=True, indent=2)
        manifest_sha = hashlib.sha256(manifest_body.encode("utf-8")).hexdigest()
        manifest["manifest_checksum"] = manifest_sha

        self._artifacts.write_json("manifest.json", manifest)
        return manifest_sha

    def read(self, feature_name: str) -> pd.DataFrame:
        """Load one feature DataFrame. Validates checksum on read.

        Raises:
            FileNotFoundError: snapshot does not exist
            KeyError: feature not in snapshot
            CacheChecksumError: data file corrupted
        """
        m = self.manifest()
        if feature_name not in m["features"]:
            raise KeyError(f"feature '{feature_name}' not in snapshot '{self.snapshot_id}'")
        expected_sha = m["features"][feature_name]["data_checksum"]
        parquet_bytes = self._artifacts.read_bytes(f"{feature_name}.parquet")
        actual_sha = hashlib.sha256(parquet_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise CacheChecksumError(
                f"snapshot data corrupt: {self.snapshot_id}/{feature_name}: "
                f"expected={expected_sha[:16]}, got={actual_sha[:16]}"
            )
        return pd.read_parquet(io.BytesIO(parquet_bytes))

    def manifest(self) -> dict:
        if not self.exists():
            raise FileNotFoundError(f"snapshot not found: {self.snapshot_id}")
        return self._artifacts.read_json("manifest.json")

    def feature_names(self) -> List[str]:
        return sorted(self.manifest()["features"].keys())

    def verify_manifest(self) -> bool:
        """Re-compute and compare manifest checksum. Returns True if intact."""
        try:
            raw = self._artifacts.read_json("manifest.json")
            stored_sha = raw.pop("manifest_checksum", None)
            if stored_sha is None:
                return False
            recomputed = hashlib.sha256(
                json.dumps(raw, sort_keys=True, indent=2).encode("utf-8")
            ).hexdigest()
            return recomputed == stored_sha
        except (FileNotFoundError, json.JSONDecodeError):
            return False
