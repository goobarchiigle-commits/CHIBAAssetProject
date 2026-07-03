"""FeatureCache — deterministic, append-only cache for feature DataFrames.

Cache key components:
  feature_name, feature_version, dataset_hash, parameter_hash, code_version

Storage layout under base_dir:
  {cache_key}.parquet     — feature DataFrame (snappy-compressed)
  {cache_key}.meta.json   — metadata + data_checksum

Append-only: CacheCollisionError on any attempt to overwrite.
Atomic writes: tmp-then-rename guarantees no partial files.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.paths import FEATURE_CACHE_DIR


class CacheCollisionError(Exception):
    pass


class CacheChecksumError(Exception):
    pass


# ---------------------------------------------------------------------------
# Deterministic hashing utilities
# ---------------------------------------------------------------------------

def hash_dataframe(df: pd.DataFrame) -> str:
    """Cross-machine deterministic SHA256 of DataFrame contents.

    Normalises: sort by index, sort columns alphabetically, represent values
    as JSON strings so float repr is stable across Python/numpy versions.
    """
    hasher = hashlib.sha256()
    df_s = df.sort_index()[sorted(df.columns)]
    # Index as strings (handles DatetimeIndex, RangeIndex, etc.)
    hasher.update(json.dumps(df_s.index.astype(str).tolist()).encode("utf-8"))
    for col in sorted(df_s.columns):
        hasher.update(col.encode("utf-8"))
        vals = df_s[col].tolist()
        # Canonicalise NaN/inf so they serialise deterministically
        canonical = []
        for v in vals:
            if isinstance(v, float):
                if v != v:
                    canonical.append("__nan__")
                elif v == float("inf"):
                    canonical.append("__inf__")
                elif v == float("-inf"):
                    canonical.append("__-inf__")
                else:
                    canonical.append(repr(v))
            else:
                canonical.append(repr(v))
        hasher.update(json.dumps(canonical).encode("utf-8"))
    return hasher.hexdigest()


def hash_params(params: Dict[str, Any]) -> str:
    """Deterministic SHA256[:16] of a parameter dict. Order-independent."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    feature_name: str,
    feature_version: str,
    dataset_hash: str,
    parameter_hash: str,
    code_version: str,
) -> str:
    payload = json.dumps(
        {
            "feature_name": feature_name,
            "feature_version": feature_version,
            "dataset_hash": dataset_hash,
            "parameter_hash": parameter_hash,
            "code_version": code_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FeatureArtifacts — raw append-only byte storage
# ---------------------------------------------------------------------------

class FeatureArtifacts:
    """Raw append-only storage for feature cache files."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else FEATURE_CACHE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, name: str, content: bytes) -> str:
        """Atomic write. Returns sha256. Raises CacheCollisionError if exists."""
        p = self.base_dir / name
        if p.exists():
            raise CacheCollisionError(f"artifact already exists: {p}")
        sha = hashlib.sha256(content).hexdigest()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(p)
        return sha

    def write_json(self, name: str, data: Any) -> str:
        content = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
        return self.write_bytes(name, content)

    def read_bytes(self, name: str) -> bytes:
        p = self.base_dir / name
        if not p.exists():
            raise FileNotFoundError(f"artifact not found: {name}")
        return p.read_bytes()

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_bytes(name).decode("utf-8"))

    def exists(self, name: str) -> bool:
        return (self.base_dir / name).exists()

    def checksum(self, name: str) -> str:
        return hashlib.sha256(self.read_bytes(name)).hexdigest()

    def verify(self, name: str, expected_sha: str) -> bool:
        try:
            return self.checksum(name) == expected_sha
        except FileNotFoundError:
            return False

    def list_files(self) -> list:
        return sorted(p.name for p in self.base_dir.iterdir() if p.is_file())


# ---------------------------------------------------------------------------
# FeatureCache — DataFrame-level cache
# ---------------------------------------------------------------------------

class FeatureCache:
    """Deterministic cache for feature DataFrames.

    Each entry = one .parquet + one .meta.json file under base_dir.
    No entries are ever deleted or modified (append-only).
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._artifacts = FeatureArtifacts(base_dir)

    @property
    def base_dir(self) -> Path:
        return self._artifacts.base_dir

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    def make_key(
        self,
        feature_name: str,
        feature_version: str,
        dataset_hash: str,
        parameter_hash: str,
        code_version: str,
    ) -> str:
        return build_cache_key(
            feature_name, feature_version, dataset_hash, parameter_hash, code_version
        )

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        return (
            self._artifacts.exists(f"{key}.parquet")
            and self._artifacts.exists(f"{key}.meta.json")
        )

    def put(self, key: str, df: pd.DataFrame, metadata: dict) -> str:
        """Store df under key. Returns data_checksum. Raises CacheCollisionError if key exists."""
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", compression="snappy", index=True)
        parquet_bytes = buf.getvalue()
        data_sha = hashlib.sha256(parquet_bytes).hexdigest()
        meta_full: Dict[str, Any] = {
            **metadata,
            "data_checksum": data_sha,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(df),
            "columns": list(df.columns),
        }
        self._artifacts.write_bytes(f"{key}.parquet", parquet_bytes)
        try:
            self._artifacts.write_json(f"{key}.meta.json", meta_full)
        except CacheCollisionError:
            # Parquet already written; meta also exists — consistent state
            raise
        return data_sha

    def get(self, key: str) -> pd.DataFrame:
        """Load cached DataFrame. Raises KeyError on miss, CacheChecksumError on corruption."""
        if not self.has(key):
            raise KeyError(f"cache miss: {key}")
        meta = self._artifacts.read_json(f"{key}.meta.json")
        parquet_bytes = self._artifacts.read_bytes(f"{key}.parquet")
        actual_sha = hashlib.sha256(parquet_bytes).hexdigest()
        if actual_sha != meta["data_checksum"]:
            raise CacheChecksumError(
                f"checksum mismatch for key={key[:16]}…: "
                f"expected={meta['data_checksum'][:16]}, got={actual_sha[:16]}"
            )
        return pd.read_parquet(io.BytesIO(parquet_bytes))

    def get_metadata(self, key: str) -> dict:
        if not self._artifacts.exists(f"{key}.meta.json"):
            raise KeyError(f"cache miss: {key}")
        return self._artifacts.read_json(f"{key}.meta.json")

    # ------------------------------------------------------------------
    # Hashing helpers (delegates to module-level functions)
    # ------------------------------------------------------------------

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        return hash_dataframe(df)

    @staticmethod
    def hash_params(params: Dict[str, Any]) -> str:
        return hash_params(params)
