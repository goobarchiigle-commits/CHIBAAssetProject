import json, time
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RecoveryConfig:
    max_retries: int = 3
    retry_delay_seconds: float = 60.0

class FailureRecoveryManager:
    def __init__(self, config: RecoveryConfig = None,
                 log_path: str = "artifacts/runtime/recovery_log.jsonl"):
        self._config = config or RecoveryConfig()
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def should_retry(self, run_id: str, stage: str) -> bool:
        return self._count_attempts(run_id, stage) < self._config.max_retries

    def _count_attempts(self, run_id: str, stage: str) -> int:
        if not self._path.exists():
            return 0
        with self._path.open() as f:
            return sum(1 for l in f if l.strip() and
                       json.loads(l).get("run_id") == run_id and
                       json.loads(l).get("stage") == stage)

    def record_attempt(self, run_id: str, stage: str, error: str) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "run_id": run_id,
                                "stage": stage, "error": error}) + "\n")

    def record_recovery(self, run_id: str, stage: str) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "run_id": run_id,
                                "stage": stage, "event": "recovered"}) + "\n")
