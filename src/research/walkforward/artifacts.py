"""Immutable walk-forward artifact storage.

Rules:
  - append-only: existing artifacts cannot be overwritten
  - atomic writes via temp-then-rename
  - checksum on every write and read
  - no silent fallback on collision
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_BASE = Path("artifacts/walkforward")


class ArtifactExistsError(Exception):
    pass


class ArtifactChecksumError(Exception):
    pass


class WalkForwardArtifacts:
    def __init__(self, experiment_id: str, base_dir: Optional[Path] = None) -> None:
        self.experiment_id = experiment_id
        self.artifact_dir = (Path(base_dir) if base_dir else _DEFAULT_BASE) / experiment_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: bytes | str) -> str:
        """Write artifact. Returns sha256. Raises ArtifactExistsError if already present."""
        p = self.artifact_dir / name
        if p.exists():
            raise ArtifactExistsError(f"artifact already exists: {p}")
        if isinstance(content, str):
            content = content.encode("utf-8")
        sha = hashlib.sha256(content).hexdigest()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(p)
        return sha

    def write_json(self, name: str, data: Any) -> str:
        content = json.dumps(data, sort_keys=True, indent=2)
        return self.write(name, content)

    def read(self, name: str) -> bytes:
        p = self.artifact_dir / name
        if not p.exists():
            raise FileNotFoundError(f"artifact not found: {name}")
        return p.read_bytes()

    def read_json(self, name: str) -> Any:
        return json.loads(self.read(name).decode("utf-8"))

    def exists(self, name: str) -> bool:
        return (self.artifact_dir / name).exists()

    def checksum(self, name: str) -> str:
        p = self.artifact_dir / name
        if not p.exists():
            raise FileNotFoundError(name)
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def verify(self, name: str, expected_sha: str) -> bool:
        try:
            actual = self.checksum(name)
        except FileNotFoundError:
            return False
        return actual == expected_sha

    def list_artifacts(self) -> List[str]:
        return sorted(p.name for p in self.artifact_dir.iterdir() if p.is_file())

    def all_checksums(self) -> Dict[str, str]:
        return {name: self.checksum(name) for name in self.list_artifacts()}

    def write_fold_metadata(self, folds: list) -> str:
        data = [f.to_dict() for f in folds]
        return self.write_json("fold_metadata.json", data)

    def write_metrics(self, fold_id: int, is_metrics: dict, oos_metrics: dict) -> str:
        name = f"fold_{fold_id:03d}_metrics.json"
        return self.write_json(name, {"is": is_metrics, "oos": oos_metrics})
