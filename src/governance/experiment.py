"""
src/governance/experiment.py
Experiment manifest, registry, and immutable artifact store.

Design:
  - experiment_id = exp_{YYYYMMDD}_{hash[:8]}  (hash of content)
  - Experiments are append-only (no overwrite)
  - Artifacts are written once; second write raises ArtifactExistsError
  - git_commit always null (project is not a git repo)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

REPO_ROOT            = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_REGISTRY  = REPO_ROOT / "state" / "experiment_registry.json"
ARTIFACTS_BASE       = REPO_ROOT / "artifacts" / "experiments"


class ArtifactExistsError(RuntimeError):
    """Raised when attempting to overwrite an immutable artifact."""


class ExperimentNotFoundError(KeyError):
    """Raised when an experiment_id does not exist in the registry."""


# ── ID generation ─────────────────────────────────────────────────────────────

def make_experiment_id(
    strategy_id: str,
    config_hash: Optional[str],
    dataset_hash: Optional[str],
    seed: Optional[int],
) -> str:
    """
    Deterministic experiment ID from content fingerprint.

    Format: exp_{YYYYMMDD}_{hash[:8]}
    Date component uses JST wall clock (OK — ID is for tracing, not execution).
    """
    payload = json.dumps(
        {
            "strategy_id":  strategy_id,
            "config_hash":  config_hash,
            "dataset_hash": dataset_hash,
            "seed":         seed,
        },
        sort_keys=True,
    )
    content_hash = hashlib.sha256(payload.encode()).hexdigest()[:8]
    date_str     = datetime.now(JST).strftime("%Y%m%d")
    return f"exp_{date_str}_{content_hash}"


# ── Manifest ──────────────────────────────────────────────────────────────────

class ExperimentManifest:
    """
    Immutable record of a single experiment run.

    Fields mirror state/experiment_registry.json schema.
    """

    REQUIRED_FIELDS = (
        "experiment_id",
        "strategy_id",
        "runtime_mode",
        "seed",
        "created_at",
    )

    def __init__(self, data: dict[str, Any]) -> None:
        missing = [f for f in self.REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"ExperimentManifest missing required fields: {missing}")
        self._data = dict(data)

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def experiment_id(self) -> str:
        return self._data["experiment_id"]

    @property
    def strategy_id(self) -> str:
        return self._data["strategy_id"]

    @property
    def seed(self) -> Optional[int]:
        return self._data.get("seed")

    @property
    def runtime_mode(self) -> str:
        return self._data["runtime_mode"]

    @property
    def git_commit(self) -> Optional[str]:
        return self._data.get("git_commit")

    @property
    def config_hash(self) -> Optional[str]:
        return self._data.get("config_hash")

    @property
    def dataset_hash(self) -> Optional[str]:
        return self._data.get("dataset_hash")

    @property
    def strategy_hash(self) -> Optional[str]:
        return self._data.get("strategy_hash")

    @property
    def parent_experiment(self) -> Optional[str]:
        return self._data.get("parent_experiment")

    @property
    def notes(self) -> str:
        return self._data.get("notes", "")

    @property
    def result_artifact_hashes(self) -> dict[str, str]:
        return self._data.get("result_artifact_hashes", {})

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return (
            f"ExperimentManifest(id={self.experiment_id}, "
            f"strategy={self.strategy_id}, seed={self.seed})"
        )


# ── Registry ─────────────────────────────────────────────────────────────────

class ExperimentRegistry:
    """
    Append-only experiment registry backed by state/experiment_registry.json.
    """

    def __init__(self, registry_file: Optional[Path] = None) -> None:
        # Late binding: read module-level constant at instantiation time so tests
        # can patch src.governance.experiment.EXPERIMENT_REGISTRY.
        import src.governance.experiment as _self_mod
        self._file = registry_file if registry_file is not None else _self_mod.EXPERIMENT_REGISTRY

    def _load(self) -> dict[str, Any]:
        if not self._file.exists():
            return {"_meta": {"schema_version": "1.0"}, "experiments": {}}
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if "experiments" not in data:
                data["experiments"] = {}
            return data
        except Exception as exc:
            raise RuntimeError(f"Could not load experiment registry: {exc}") from exc

    def _save(self, data: dict[str, Any]) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._file)

    def register(self, manifest: ExperimentManifest) -> None:
        """
        Register a new experiment. Raises if experiment_id already exists.
        """
        data = self._load()
        eid  = manifest.experiment_id
        if eid in data["experiments"]:
            raise ArtifactExistsError(
                f"Experiment '{eid}' already registered. Experiments are immutable."
            )
        data["experiments"][eid] = manifest.to_dict()
        self._save(data)
        logger.info("[EXPERIMENT] registered: %s", eid)

    def get(self, experiment_id: str) -> ExperimentManifest:
        """Retrieve experiment by ID. Raises ExperimentNotFoundError if absent."""
        data = self._load()
        entry = data["experiments"].get(experiment_id)
        if entry is None:
            raise ExperimentNotFoundError(
                f"Experiment '{experiment_id}' not found in registry"
            )
        return ExperimentManifest(entry)

    def list_all(self) -> list[ExperimentManifest]:
        """Return all experiments, sorted by created_at descending."""
        data = self._load()
        manifests = [ExperimentManifest(v) for v in data["experiments"].values()]
        return sorted(
            manifests,
            key=lambda m: m.to_dict().get("created_at", ""),
            reverse=True,
        )

    def exists(self, experiment_id: str) -> bool:
        return experiment_id in self._load()["experiments"]

    def count(self) -> int:
        return len(self._load()["experiments"])


# ── Artifact store ────────────────────────────────────────────────────────────

class ArtifactStore:
    """
    Immutable artifact writer for experiment outputs.

    Directory structure:
        artifacts/experiments/<experiment_id>/
            signals.parquet
            trades.parquet
            metrics.json
            config_snapshot.json
            dataset_fingerprint.json
            runtime_metadata.json
    """

    KNOWN_ARTIFACTS = frozenset({
        "signals.parquet",
        "trades.parquet",
        "metrics.json",
        "config_snapshot.json",
        "dataset_fingerprint.json",
        "runtime_metadata.json",
    })

    def __init__(
        self,
        experiment_id: str,
        base_dir: Path = ARTIFACTS_BASE,
    ) -> None:
        self.experiment_id = experiment_id
        self.artifact_dir  = base_dir / experiment_id

    def _ensure_dir(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, content: bytes | str) -> str:
        """
        Write artifact. Raises ArtifactExistsError if file already exists.

        Returns sha256 hex digest of written content.
        """
        self._ensure_dir()
        dest = self.artifact_dir / filename
        if dest.exists():
            raise ArtifactExistsError(
                f"Artifact '{filename}' already exists for experiment "
                f"'{self.experiment_id}'. Artifacts are immutable."
            )
        if isinstance(content, str):
            raw = content.encode("utf-8")
        else:
            raw = content
        dest.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        logger.info("[ARTIFACT] wrote %s/%s (sha256=%s)", self.experiment_id, filename, sha[:16])
        return sha

    def write_json(self, filename: str, data: Any) -> str:
        """Serialize data as JSON and write."""
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return self.write(filename, content)

    def hash_of(self, filename: str) -> str:
        """Return sha256 of existing artifact. Raises FileNotFoundError if absent."""
        dest = self.artifact_dir / filename
        if not dest.exists():
            raise FileNotFoundError(
                f"Artifact '{filename}' not found for experiment '{self.experiment_id}'"
            )
        return hashlib.sha256(dest.read_bytes()).hexdigest()

    def exists(self, filename: str) -> bool:
        return (self.artifact_dir / filename).exists()

    def list_artifacts(self) -> list[str]:
        if not self.artifact_dir.exists():
            return []
        return sorted(f.name for f in self.artifact_dir.iterdir() if f.is_file())

    def all_hashes(self) -> dict[str, str]:
        """Return {filename: sha256} for all artifacts."""
        return {name: self.hash_of(name) for name in self.list_artifacts()}


# ── Convenience builder ───────────────────────────────────────────────────────

def build_manifest(
    strategy_id: str,
    runtime_mode: str,
    seed: Optional[int],
    config_hash: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    strategy_hash: Optional[str] = None,
    parent_experiment: Optional[str] = None,
    result_artifact_hashes: Optional[dict[str, str]] = None,
    notes: str = "",
) -> ExperimentManifest:
    """Construct ExperimentManifest with auto-generated ID and timestamp."""
    eid = make_experiment_id(strategy_id, config_hash, dataset_hash, seed)
    data: dict[str, Any] = {
        "experiment_id":          eid,
        "strategy_id":            strategy_id,
        "runtime_mode":           runtime_mode,
        "seed":                   seed,
        "git_commit":             None,  # no git in this project
        "config_hash":            config_hash,
        "dataset_hash":           dataset_hash,
        "strategy_hash":          strategy_hash,
        "parent_experiment":      parent_experiment,
        "result_artifact_hashes": result_artifact_hashes or {},
        "notes":                  notes,
        "created_at":             datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return ExperimentManifest(data)
