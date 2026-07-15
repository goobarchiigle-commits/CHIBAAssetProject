import json, time, threading
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class CircuitBreakerConfig:
    max_epoch_mismatches: int = 3
    max_replay_divergences: int = 2
    max_drift_surges: int = 5
    max_validation_failures: int = 5
    window_seconds: float = 3600.0

class RuntimeCircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig = None,
                 manifest_path: str = "artifacts/runtime/circuit_breaker.jsonl"):
        self._config = config or CircuitBreakerConfig()
        self._path = Path(manifest_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def record_event(self, event_type: str, run_id: str, detail: str) -> None:
        with self._lock:
            with self._path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "event_type": event_type,
                                    "run_id": run_id, "detail": detail}) + "\n")
            self._evaluate()

    def _read_window(self) -> List[dict]:
        if not self._path.exists():
            return []
        cutoff = time.time() - self._config.window_seconds
        rows = []
        with self._path.open("r") as f:
            for l in f:
                if l.strip():
                    e = json.loads(l)
                    if e.get("ts", 0) > cutoff:
                        rows.append(e)
        return rows

    def _evaluate(self) -> None:
        events = self._read_window()
        def count(t): return sum(1 for e in events if e.get("event_type") == t)
        reasons = []
        if count("epoch_mismatch") >= self._config.max_epoch_mismatches:
            reasons.append("epoch_mismatch_threshold")
        if count("replay_divergence") >= self._config.max_replay_divergences:
            reasons.append("replay_divergence_threshold")
        if count("drift_surge") >= self._config.max_drift_surges:
            reasons.append("drift_surge_threshold")
        if count("validation_failure") >= self._config.max_validation_failures:
            reasons.append("validation_failure_threshold")
        if reasons:
            self._tripped = True
            with self._path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "event_type": "TRIPPED",
                                    "reasons": reasons}) + "\n")

    def reset(self, authorized_by: str) -> None:
        with self._lock:
            with self._path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "event_type": "RESET",
                                    "authorized_by": authorized_by}) + "\n")
            self._tripped = False
