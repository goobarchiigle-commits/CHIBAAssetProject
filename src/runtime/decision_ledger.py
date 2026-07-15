import json, time, threading
from pathlib import Path
from typing import Any, Dict

class RuntimeDecisionLedger:
    def __init__(self, ledger_path: str = "artifacts/runtime/decision_ledger.jsonl"):
        self._path = Path(ledger_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, decision_type: str, run_id: str, epoch_id: str,
               reason: str, metadata: Dict[str, Any] = None) -> None:
        entry = {"ts": time.time(), "decision_type": decision_type,
                 "run_id": run_id, "epoch_id": epoch_id,
                 "reason": reason, "metadata": metadata or {}}
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def read_all(self) -> list:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
