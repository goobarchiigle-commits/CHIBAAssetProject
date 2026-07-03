"""ExperimentBatch — immutable container for a set of related experiments.

batch_id is deterministic: same (strategy_id, sweep_hash, seed) → same id.
Experiments are linked by experiment_id only; batch is a pure metadata record.
Artifacts stored append-only under artifacts/batches/<batch_id>/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_BASE = Path("artifacts/batches")


@dataclass(frozen=True)
class ExperimentBatch:
    batch_id: str
    strategy_id: str
    description: str
    seed: int
    created_at: str              # ISO8601 UTC
    experiment_ids: Tuple[str, ...]
    parameter_space_hash: str    # hash of the sweep grid

    # ── serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "strategy_id": self.strategy_id,
            "description": self.description,
            "seed": self.seed,
            "created_at": self.created_at,
            "experiment_ids": list(self.experiment_ids),
            "parameter_space_hash": self.parameter_space_hash,
            "experiment_count": len(self.experiment_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentBatch:
        return cls(
            batch_id=d["batch_id"],
            strategy_id=d["strategy_id"],
            description=d["description"],
            seed=int(d["seed"]),
            created_at=d["created_at"],
            experiment_ids=tuple(d["experiment_ids"]),
            parameter_space_hash=d["parameter_space_hash"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    # ── factory ────────────────────────────────────────────────────────────────

    @staticmethod
    def make_id(
        strategy_id: str,
        parameter_space_hash: str,
        seed: int,
        *,
        timestamp: Optional[str] = None,
    ) -> str:
        """Deterministic batch_id. Same inputs → same id."""
        raw = f"{strategy_id}:{parameter_space_hash}:{seed}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:8]
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"batch_{ts}_{h}"


# ── Artifact storage ───────────────────────────────────────────────────────────

class ArtifactExistsError(Exception):
    pass


class BatchArtifacts:
    """Append-only artifact store for a single batch.

    Same rules as WalkForwardArtifacts:
    - No overwrite (raise ArtifactExistsError)
    - Atomic write via temp-then-rename
    - sha256 on every write
    """

    def __init__(self, batch_id: str, base_dir: Optional[Path] = None) -> None:
        self.batch_id = batch_id
        self.artifact_dir = (Path(base_dir) if base_dir else _DEFAULT_BASE) / batch_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: bytes | str) -> str:
        p = self.artifact_dir / name
        if p.exists():
            raise ArtifactExistsError(f"artifact already exists: {p}")
        if isinstance(content, str):
            content = content.encode("utf-8")
        import hashlib as _hl
        sha = _hl.sha256(content).hexdigest()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(p)
        return sha

    def write_json(self, name: str, data: Any) -> str:
        return self.write(name, json.dumps(data, sort_keys=True, indent=2))

    def exists(self, name: str) -> bool:
        return (self.artifact_dir / name).exists()

    def read_json(self, name: str) -> Any:
        return json.loads((self.artifact_dir / name).read_bytes())

    def list_artifacts(self) -> List[str]:
        return sorted(p.name for p in self.artifact_dir.iterdir() if p.is_file())

    def write_batch_metadata(self, batch: ExperimentBatch) -> str:
        return self.write_json("batch_metadata.json", batch.to_dict())

    def write_ranking_table(self, table: List[dict]) -> str:
        return self.write_json("ranking_table.json", table)

    def write_rejection_report(self, report: dict) -> str:
        return self.write_json("rejection_report.json", report)

    def write_regime_summaries(self, summaries: dict) -> str:
        return self.write_json("regime_summaries.json", summaries)

    def write_registry_snapshot(self, snapshot: List[dict]) -> str:
        return self.write_json("experiment_registry_snapshot.json", snapshot)
