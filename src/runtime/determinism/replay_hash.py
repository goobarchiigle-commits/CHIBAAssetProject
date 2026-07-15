"""
src/runtime/determinism/replay_hash.py

ReplayEquivalenceHash: deterministic content fingerprint for replay validation.

Hash = sha256(canonical_json(
    epoch_manifest +
    orchestration_manifest +
    dependency_manifest +
    governance_snapshot +
    runtime_decision_ledger
))

Properties:
- stable ordering: recursive sort_keys before serialization
- canonical JSON: compact separators, ASCII-safe, no float ambiguity
- identical inputs → identical hash (regardless of dict insertion order)
- append-only persistence to artifacts/runtime_validation/replay_hashes.jsonl
- thread-safe append
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

REPLAY_HASHES_PATH: Path = Path("artifacts/runtime_validation/replay_hashes.jsonl")


def _canonical(obj: Any) -> Any:
    """Recursively normalize for stable canonical representation."""
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def _canonical_json(obj: Any) -> str:
    return json.dumps(_canonical(obj), separators=(",", ":"), ensure_ascii=True)


class ReplayEquivalenceHash:
    """
    Compute and persist deterministic replay hash.

    Usage::
        reh = ReplayEquivalenceHash()
        h = reh.compute(epoch_m, orch_m, dep_m, gov_snap, ledger, run_id="r1")
        assert h == reh.compute(epoch_m, orch_m, dep_m, gov_snap, ledger)
    """

    def __init__(self, output_path: Path = REPLAY_HASHES_PATH) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def compute(
        self,
        epoch_manifest:          Any,
        orchestration_manifest:  Any,
        dependency_manifest:     Any,
        governance_snapshot:     Any,
        runtime_decision_ledger: Any,
        run_id:                  str  = "",
        persist:                 bool = True,
    ) -> str:
        payload = _canonical_json({
            "epoch_manifest":          epoch_manifest,
            "orchestration_manifest":  orchestration_manifest,
            "dependency_manifest":     dependency_manifest,
            "governance_snapshot":     governance_snapshot,
            "runtime_decision_ledger": runtime_decision_ledger,
        })
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        if persist:
            self._append(run_id=run_id, digest=digest)

        return digest

    def load_hashes(self) -> list[dict]:
        if not self._path.exists():
            return []
        results: list[dict] = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return results

    def _append(self, run_id: str, digest: str) -> None:
        entry = json.dumps({"ts": time.time(), "run_id": run_id, "hash": digest})
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
