import json, time
from pathlib import Path
from typing import Any, Dict, List

from .snapshot import RuntimeSnapshot
from .state_machine import RuntimeStateMachine, RuntimeState

class DeterministicReplayEngine:
    """Replay historical runtime execution from snapshots. Read-only reconstruction."""
    def __init__(self, snapshot_dir: str = "artifacts/runtime/snapshots",
                 replay_ledger: str = "artifacts/runtime/replay_ledger.jsonl"):
        self._snapshots = RuntimeSnapshot(snapshot_dir)
        self._ledger = Path(replay_ledger)
        self._ledger.parent.mkdir(parents=True, exist_ok=True)

    def replay(self, epoch_id: str) -> List[Dict[str, Any]]:
        snaps = [s for s in self._snapshots.list_snapshots() if s.get("epoch_id") == epoch_id]
        if not snaps:
            return []
        snaps = sorted(snaps, key=lambda s: s.get("ts", 0))
        reconstructed = []
        for snap in snaps:
            reconstructed.append({
                "ts": snap["ts"],
                "run_id": snap["run_id"],
                "epoch_id": snap["epoch_id"],
                "state": snap.get("state", {}),
            })
        entry = {"ts": time.time(), "epoch_id": epoch_id,
                 "steps_replayed": len(reconstructed)}
        with self._ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return reconstructed
