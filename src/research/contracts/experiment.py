"""ExperimentContract — immutable experiment identity and lineage."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class ExperimentContract:
    experiment_id: str
    strategy_id: str
    parameter_hash: str
    dataset_hash: str
    code_version: str
    seed: int
    created_at: str  # ISO8601 UTC

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "parameter_hash": self.parameter_hash,
            "dataset_hash": self.dataset_hash,
            "code_version": self.code_version,
            "seed": self.seed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentContract:
        return cls(
            experiment_id=d["experiment_id"],
            strategy_id=d["strategy_id"],
            parameter_hash=d["parameter_hash"],
            dataset_hash=d["dataset_hash"],
            code_version=d["code_version"],
            seed=int(d["seed"]),
            created_at=d["created_at"],
        )

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def make_id(
        strategy_id: str,
        parameter_hash: str,
        dataset_hash: str,
        seed: int,
        *,
        timestamp: Optional[str] = None,
    ) -> str:
        """Deterministic experiment_id from content. Same inputs → same id."""
        raw = f"{strategy_id}:{parameter_hash}:{dataset_hash}:{seed}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:8]
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"exp_{ts}_{h}"

    @staticmethod
    def hash_parameters(params: dict) -> str:
        payload = json.dumps(params, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
