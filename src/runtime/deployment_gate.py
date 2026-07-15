import json, time
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DeploymentRecommendation:
    eligible: bool
    run_id: str
    epoch_id: str
    reasons: List[str]
    timestamp: float

class DeploymentGateInterface:
    """Research runtime MAY RECOMMEND ONLY. Never authorize deployment directly."""
    def __init__(self, manifest_path: str = "artifacts/runtime/deployment_recommendations.jsonl"):
        self._path = Path(manifest_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def recommend(self, run_id: str, epoch_id: str, eligible: bool,
                  reasons: List[str]) -> DeploymentRecommendation:
        rec = DeploymentRecommendation(eligible=eligible, run_id=run_id,
                                       epoch_id=epoch_id, reasons=reasons,
                                       timestamp=time.time())
        with self._path.open("a") as f:
            f.write(json.dumps({"ts": rec.timestamp, "run_id": run_id,
                                "epoch_id": epoch_id, "eligible": eligible,
                                "reasons": reasons}) + "\n")
        return rec

    def list_recommendations(self) -> list:
        if not self._path.exists(): return []
        with self._path.open() as f:
            return [json.loads(l) for l in f if l.strip()]
