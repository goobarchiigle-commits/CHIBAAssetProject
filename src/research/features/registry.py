"""FeatureRegistry — deterministic registration with lineage and schema enforcement."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Type

from src.research.features.contract import FeatureContract, FeatureMetadata


class FeatureRegistryError(Exception):
    pass


class FeatureRegistry:
    """Register feature contracts; enforce identity uniqueness and schema consistency.

    Registration is open until ``lock()`` is called. After locking, any attempt
    to register raises FeatureRegistryError. Lock before passing to a pipeline.

    Rules:
    - Same (name, version, output_columns) → idempotent (silent no-op)
    - Same name but different schema/version → FeatureRegistryError
    - Duplicate name+version with different output_columns → FeatureRegistryError
    """

    def __init__(self) -> None:
        self._contracts: Dict[str, Type[FeatureContract]] = {}
        self._metadata: Dict[str, FeatureMetadata] = {}
        self._lineage: Dict[str, List[str]] = {}
        self._locked = False

    def register(
        self,
        contract_cls: Type[FeatureContract],
        depends_on: Optional[List[str]] = None,
    ) -> None:
        if self._locked:
            raise FeatureRegistryError("registry is locked; no further registration allowed")
        inst = contract_cls()
        name = inst.feature_name
        meta = inst.metadata()
        if name in self._contracts:
            existing = self._metadata[name]
            if existing != meta:
                raise FeatureRegistryError(
                    f"conflicting registration for '{name}': "
                    f"existing={existing}, incoming={meta}"
                )
            return  # idempotent re-registration
        self._contracts[name] = contract_cls
        self._metadata[name] = meta
        self._lineage[name] = list(depends_on or [])

    def lock(self) -> None:
        """Freeze registry. Raises on any subsequent register() call."""
        self._locked = True

    @property
    def is_locked(self) -> bool:
        return self._locked

    def get(self, feature_name: str) -> Type[FeatureContract]:
        if feature_name not in self._contracts:
            raise KeyError(f"feature not registered: '{feature_name}'")
        return self._contracts[feature_name]

    def instantiate(self, feature_name: str) -> FeatureContract:
        return self.get(feature_name)()

    def metadata(self, feature_name: str) -> FeatureMetadata:
        if feature_name not in self._metadata:
            raise KeyError(f"feature not registered: '{feature_name}'")
        return self._metadata[feature_name]

    def dependencies(self, feature_name: str) -> List[str]:
        return list(self._lineage.get(feature_name, []))

    def all_features(self) -> List[str]:
        return sorted(self._contracts.keys())

    def to_manifest(self) -> dict:
        """Deterministic JSON-serialisable manifest of all registered features."""
        return {
            name: {
                "version": self._metadata[name].feature_version,
                "output_columns": list(self._metadata[name].output_columns),
                "depends_on": list(self._lineage[name]),
            }
            for name in sorted(self._contracts.keys())
        }

    def manifest_hash(self) -> str:
        """SHA256 of the registry manifest. Changes whenever any registration changes."""
        import hashlib
        payload = json.dumps(self.to_manifest(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
