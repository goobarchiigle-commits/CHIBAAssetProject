import json, time, hashlib
from pathlib import Path
from typing import Any, Dict

class RuntimeSnapshot:
    def __init__(self, snapshot_dir: str = "artifacts/runtime/snapshots"):
        self._dir = Path(snapshot_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def capture(self, run_id: str, epoch_id: str, state: Dict[str, Any]) -> str:
        ts = int(time.time())
        payload = {"ts": ts, "run_id": run_id, "epoch_id": epoch_id, "state": state}
        content = json.dumps(payload, sort_keys=True, default=str)
        snap_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        path = self._dir / f"{ts}_{run_id}_{snap_id}.json"
        path.write_text(content)
        return str(path)

    def load(self, snapshot_path: str) -> Dict[str, Any]:
        return json.loads(Path(snapshot_path).read_text())

    def list_snapshots(self, run_id: str = None) -> list:
        snaps = [json.loads(f.read_text()) for f in sorted(self._dir.glob("*.json"))]
        if run_id:
            snaps = [s for s in snaps if s.get("run_id") == run_id]
        return snaps
