"""Experiment queue — append-only, file-backed, crash-safe.

Each entry is a newline-delimited JSON record.
The queue file is never rewritten; entries are only appended.
Status is tracked in-memory and in a separate state file.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class QueueStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ExperimentQueue:
    def __init__(self, queue_file: Path) -> None:
        self.queue_file = Path(queue_file)
        self._entries: List[Dict] = []
        self._status: Dict[str, str] = {}
        self._load()

    def enqueue(self, experiment_id: str, priority: int = 0) -> None:
        entry = {
            "experiment_id": experiment_id,
            "priority": priority,
            "status": QueueStatus.PENDING,
        }
        self._entries.append(entry)
        self._status[experiment_id] = QueueStatus.PENDING
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        with self.queue_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def pending(self) -> List[str]:
        """Deterministic: return pending IDs in priority-desc, insertion order."""
        pending = [e for e in self._entries if self._status.get(e["experiment_id"]) == QueueStatus.PENDING]
        pending.sort(key=lambda e: (-e["priority"], self._entries.index(e)))
        return [e["experiment_id"] for e in pending]

    def mark_running(self, experiment_id: str) -> None:
        self._set_status(experiment_id, QueueStatus.RUNNING)

    def mark_done(self, experiment_id: str) -> None:
        self._set_status(experiment_id, QueueStatus.DONE)

    def mark_failed(self, experiment_id: str) -> None:
        self._set_status(experiment_id, QueueStatus.FAILED)

    def status(self, experiment_id: str) -> Optional[str]:
        return self._status.get(experiment_id)

    def all_entries(self) -> List[Dict]:
        return list(self._entries)

    def _set_status(self, experiment_id: str, status: str) -> None:
        self._status[experiment_id] = status
        for e in self._entries:
            if e["experiment_id"] == experiment_id:
                e["status"] = status
                break

    def _load(self) -> None:
        if not self.queue_file.exists():
            return
        for line in self.queue_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            self._entries.append(entry)
            eid = entry["experiment_id"]
            self._status[eid] = entry.get("status", QueueStatus.PENDING)
