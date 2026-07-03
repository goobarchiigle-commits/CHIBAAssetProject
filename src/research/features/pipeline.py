"""FeaturePipeline — compute feature specs against a DataFrame with automatic cache reuse.

Workflow per spec:
  1. Build cache key from (feature, version, dataset_hash, params, code_hash)
  2. Check in-run dedup map (same key already computed this call → reuse)
  3. Check persistent cache (has() → get())
  4. Miss → compute → validate → put() → return

Return: List[pd.DataFrame], one per input spec in input order.
Caller zips specs with results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.research.features.cache import FeatureCache
from src.research.features.contract import FeatureContract
from src.research.features.registry import FeatureRegistry


class FeaturePipelineError(Exception):
    pass


@dataclass
class FeatureSpec:
    """Fully-qualified request: which feature, with which params."""
    feature_name: str
    params: Dict[str, Any] = field(default_factory=dict)


class FeaturePipeline:
    """Execute FeatureSpecs against a DataFrame with transparent cache reuse.

    Thread-safety: not thread-safe; create one pipeline per worker.
    In-run dedup: within a single ``compute()`` call, identical cache keys
    are evaluated at most once regardless of how many times they appear.
    """

    def __init__(self, registry: FeatureRegistry, cache: FeatureCache) -> None:
        self._registry = registry
        self._cache = cache

    def compute(
        self,
        df: pd.DataFrame,
        specs: List[FeatureSpec],
    ) -> List[pd.DataFrame]:
        """Return one output DataFrame per spec, preserving input order.

        Cache is populated on first miss; subsequent identical keys reuse
        the cached result (both within this call and across future calls).
        """
        dataset_hash = self._cache.hash_dataframe(df)
        run_cache: Dict[str, pd.DataFrame] = {}  # key → df, in-run dedup
        results: List[pd.DataFrame] = []

        for spec in specs:
            inst = self._registry.instantiate(spec.feature_name)
            param_hash = self._cache.hash_params(spec.params)
            # code_version encodes both explicit version and source fingerprint
            code_version = f"{inst.feature_version}_{inst.code_hash()}"
            key = self._cache.make_key(
                feature_name=spec.feature_name,
                feature_version=inst.feature_version,
                dataset_hash=dataset_hash,
                parameter_hash=param_hash,
                code_version=code_version,
            )

            if key in run_cache:
                results.append(run_cache[key])
                continue

            if self._cache.has(key):
                out_df = self._cache.get(key)
            else:
                self._check_inputs(inst, df)
                out_df = inst.compute(df.copy(), **spec.params)
                self._check_outputs(inst, out_df)
                self._cache.put(
                    key,
                    out_df,
                    {
                        "feature_name": spec.feature_name,
                        "feature_version": inst.feature_version,
                        "dataset_hash": dataset_hash,
                        "parameter_hash": param_hash,
                        "code_version": code_version,
                        "params": spec.params,
                    },
                )

            run_cache[key] = out_df
            results.append(out_df)

        return results

    # ------------------------------------------------------------------
    # Internal validators
    # ------------------------------------------------------------------

    def _check_inputs(self, inst: FeatureContract, df: pd.DataFrame) -> None:
        missing = [c for c in inst.required_inputs() if c not in df.columns]
        if missing:
            raise FeaturePipelineError(
                f"feature '{inst.feature_name}' missing required inputs: {missing}"
            )

    def _check_outputs(self, inst: FeatureContract, out_df: pd.DataFrame) -> None:
        missing = [c for c in inst.output_schema() if c not in out_df.columns]
        if missing:
            raise FeaturePipelineError(
                f"feature '{inst.feature_name}' compute() missing declared output columns: {missing}"
            )
