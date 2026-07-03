"""Experiment registry — file-backed, append-only, crash-safe."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.research.contracts.experiment import ExperimentContract


class RegistryError(Exception):
    pass


class DuplicateExperimentError(RegistryError):
    pass


class ExperimentNotFoundError(RegistryError):
    pass


class ExperimentRegistry:
    """Persistent, append-only registry of ExperimentContracts.

    Atomic writes: JSON written to .tmp then renamed. No partial writes.
    No overwrite: registering a duplicate raises DuplicateExperimentError.
    """

    def __init__(self, registry_file: Path) -> None:
        self.registry_file = Path(registry_file)
        self._data: Dict[str, dict] = {}
        self._load()

    def register(self, exp: ExperimentContract) -> None:
        if exp.experiment_id in self._data:
            raise DuplicateExperimentError(exp.experiment_id)
        self._data[exp.experiment_id] = exp.to_dict()
        self._persist()

    def get(self, experiment_id: str) -> ExperimentContract:
        if experiment_id not in self._data:
            raise ExperimentNotFoundError(experiment_id)
        return ExperimentContract.from_dict(self._data[experiment_id])

    def exists(self, experiment_id: str) -> bool:
        return experiment_id in self._data

    def list_all(self) -> List[ExperimentContract]:
        return [ExperimentContract.from_dict(d) for d in self._data.values()]

    def list_by_strategy(self, strategy_id: str) -> List[ExperimentContract]:
        return [e for e in self.list_all() if e.strategy_id == strategy_id]

    def count(self) -> int:
        return len(self._data)

    def update_status(self, experiment_id: str, status: str) -> None:
        if experiment_id not in self._data:
            raise ExperimentNotFoundError(experiment_id)
        self._data[experiment_id]["status"] = status
        self._persist()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _load(self) -> None:
        if not self.registry_file.exists():
            return
        try:
            raw = json.loads(self.registry_file.read_text(encoding="utf-8"))
            self._data = raw.get("experiments", {})
        except (json.JSONDecodeError, KeyError):
            raise RegistryError(f"corrupt registry file: {self.registry_file}")

    def _persist(self) -> None:
        payload = json.dumps(
            {"experiments": self._data, "count": len(self._data)},
            sort_keys=True,
            indent=2,
        )
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_file.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.registry_file)
