import json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass(frozen=True)
class ScheduleEntry:
    schedule_id: str
    name: str
    cron_expr: str
    pipeline_stages: tuple
    enabled: bool = True
    cooldown_seconds: float = 300.0

CANONICAL_SCHEDULES = [
    ScheduleEntry("sched_morning_dryrun", "Morning DryRun", "41 8 * * 1-5",
                  ("dataset_snapshot", "feature_cache", "signal_generation")),
    ScheduleEntry("sched_morning_live", "Morning Live", "42 8 * * 1-5",
                  ("signal_generation", "deployment_recommendation")),
    ScheduleEntry("sched_nightly_research", "Nightly Research", "0 22 * * 1-5",
                  ("dataset_snapshot","feature_cache","hypothesis_generation",
                   "experiment_queue","walk_forward","survivorship_analysis",
                   "governance_filtering","execution_reality",
                   "deployment_recommendation","report_generation")),
    ScheduleEntry("sched_replay", "On-Demand Replay", "", ("replay",)),
]

class ScheduleRegistry:
    def __init__(self, history_path: str = "artifacts/runtime/schedule_history.jsonl"):
        self._history = Path(history_path)
        self._history.parent.mkdir(parents=True, exist_ok=True)
        self._schedules: Dict[str, ScheduleEntry] = {s.schedule_id: s for s in CANONICAL_SCHEDULES}

    def get(self, schedule_id: str) -> Optional[ScheduleEntry]:
        return self._schedules.get(schedule_id)

    def all_enabled(self) -> List[ScheduleEntry]:
        return [s for s in self._schedules.values() if s.enabled]

    def record_trigger(self, schedule_id: str, run_id: str, reason: str) -> None:
        with self._history.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "schedule_id": schedule_id,
                                "run_id": run_id, "reason": reason}) + "\n")
