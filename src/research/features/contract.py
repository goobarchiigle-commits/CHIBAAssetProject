"""FeatureContract — abstract base for all deterministic research features."""
from __future__ import annotations

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class FeatureMetadata:
    feature_name: str
    feature_version: str
    output_columns: tuple  # sorted tuple of output column names


class FeatureContract(ABC):
    """Deterministic, versioned feature.

    Subclass requirements:
    - Set class-level ``feature_name`` and ``feature_version``
    - Implement ``required_inputs()``, ``compute()``, ``output_schema()``
    - ``compute()`` must be pure: same df + params → same output, no hidden state
    - Bump ``feature_version`` whenever ``compute()`` behavior changes
    """

    feature_name: str   # must be set on subclass
    feature_version: str  # bump on any compute() behavioral change

    @abstractmethod
    def required_inputs(self) -> List[str]:
        """DataFrame columns required before compute() is called."""

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Pure, deterministic computation. Must not mutate df."""

    @abstractmethod
    def output_schema(self) -> Dict[str, type]:
        """Map of output column name → Python type. Used for schema validation."""

    # -------------------------------------------------------------------------
    # Derived stable identifiers (not overridable by subclasses)
    # -------------------------------------------------------------------------

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name=self.feature_name,
            feature_version=self.feature_version,
            output_columns=tuple(sorted(self.output_schema().keys())),
        )

    def code_hash(self) -> str:
        """SHA256[:16] of compute() source lines.

        Detects silent behavioral changes when feature_version is not bumped.
        Combined with feature_version in cache keys.
        """
        src = inspect.getsource(self.__class__.compute)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]

    def identity_hash(self) -> str:
        """Stable hash of (feature_name, feature_version). Schema-level identity."""
        payload = json.dumps(
            {"name": self.feature_name, "version": self.feature_version},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
