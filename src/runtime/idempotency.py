import json, time, threading, hashlib
from pathlib import Path

class IdempotencyGuard:
    def __init__(self, manifest_path: str = "artifacts/runtime/idempotency_manifest.jsonl"):
        self._path = Path(manifest_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _completed_ids(self) -> set:
        if not self._path.exists():
            return set()
        with self._path.open("r") as f:
            return {json.loads(l)["run_id"] for l in f if l.strip()}

    def is_duplicate(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._completed_ids()

    def mark_complete(self, run_id: str, epoch_id: str) -> None:
        with self._lock:
            with self._path.open("a") as f:
                f.write(json.dumps({"run_id": run_id, "epoch_id": epoch_id,
                                    "ts": time.time()}) + "\n")

    @staticmethod
    def make_run_id(epoch_id: str, schedule_id: str, date_str: str) -> str:
        return hashlib.sha256(f"{epoch_id}:{schedule_id}:{date_str}".encode()).hexdigest()[:16]
