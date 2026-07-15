import json, time, statistics
from pathlib import Path
from typing import Any, Dict

class RuntimeDriftMonitor:
    def __init__(self, metrics_path: str = "artifacts/runtime/drift_metrics.jsonl"):
        self._path = Path(metrics_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, run_id: str, epoch_id: str, stage: str,
               duration_s: float, cache_misses: int, retries: int) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "run_id": run_id, "epoch_id": epoch_id,
                                "stage": stage, "duration_s": duration_s,
                                "cache_misses": cache_misses, "retries": retries}) + "\n")

    def compute_drift(self, stage: str, window: int = 10) -> Dict[str, Any]:
        if not self._path.exists():
            return {"status": "no_data"}
        with self._path.open() as f:
            rows = [json.loads(l) for l in f if l.strip() and json.loads(l).get("stage") == stage]
        rows = rows[-window:]
        if len(rows) < 2:
            return {"status": "insufficient_data"}
        durations = [r["duration_s"] for r in rows]
        mean = statistics.mean(durations)
        stdev = statistics.stdev(durations)
        return {"stage": stage, "n": len(rows), "mean_duration": mean,
                "stdev_duration": stdev, "drift_detected": stdev > mean * 0.5}
