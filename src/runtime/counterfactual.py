import json, time
from pathlib import Path
from typing import Any, Dict, List

from .snapshot import RuntimeSnapshot

class CounterfactualRuntimeReplay:
    """Replay with alternate policy overrides. NEVER writes to the main decision ledger."""
    def __init__(self, snapshot_dir: str = "artifacts/runtime/snapshots",
                 cf_ledger: str = "artifacts/runtime/counterfactual_ledger.jsonl"):
        self._snapshots = RuntimeSnapshot(snapshot_dir)
        self._cf_ledger = Path(cf_ledger)
        self._cf_ledger.parent.mkdir(parents=True, exist_ok=True)

    def replay(self, epoch_id: str, policy_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replay with alternate thresholds. No adaptation. No mutation of main ledger.
        policy_overrides keys: max_retries, alloc_cap_multiplier, circuit_breaker_threshold, etc.
        """
        snaps = [s for s in self._snapshots.list_snapshots() if s.get("epoch_id") == epoch_id]
        snaps = sorted(snaps, key=lambda s: s.get("ts", 0))
        result = {
            "epoch_id": epoch_id,
            "policy_overrides": policy_overrides,
            "steps": len(snaps),
            "counterfactual_outcome": "replay_only",
            "ts": time.time(),
        }
        # Record to counterfactual ledger only — main ledger is never touched
        with self._cf_ledger.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return result

    def list_experiments(self) -> list:
        if not self._cf_ledger.exists(): return []
        with self._cf_ledger.open() as f:
            return [json.loads(l) for l in f if l.strip()]
