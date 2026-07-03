"""ResearchMemory — long-term research history persistence.

Tracks:
  - historical exploration paths (which hypothesis_ids were explored per family)
  - rejection histories (count and reasons for rejections)
  - survivorship trajectories (time series of survival rates)
  - topology evolution (fragility score history per grammar_hash)

Goal: prevent repeated rediscovery of failed research directions.
All entries are immutable once recorded.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.research.governance import SCHEMA_VERSION
from src.research.analysis.artifacts import utc_now_iso

# ── Memory type constants ──────────────────────────────────────────────────────

MEMORY_EXPLORATION_PATH = "EXPLORATION_PATH"
MEMORY_REJECTION_HISTORY = "REJECTION_HISTORY"
MEMORY_SURVIVORSHIP_TRAJECTORY = "SURVIVORSHIP_TRAJECTORY"
MEMORY_TOPOLOGY_EVOLUTION = "TOPOLOGY_EVOLUTION"

ALL_MEMORY_TYPES: frozenset = frozenset({
    MEMORY_EXPLORATION_PATH,
    MEMORY_REJECTION_HISTORY,
    MEMORY_SURVIVORSHIP_TRAJECTORY,
    MEMORY_TOPOLOGY_EVOLUTION,
})


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryEntry:
    """Immutable record of a single research memory event."""
    entry_id: str           # "mem_" + SHA256[:12]
    memory_id: str
    memory_type: str
    family_id: str          # family or grammar_hash (for topology entries)
    content_hash: str       # 16-char SHA256 of content
    recorded_at: str
    metadata: tuple         # sorted (key, str_value) pairs

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "family_id": self.family_id,
            "content_hash": self.content_hash,
            "recorded_at": self.recorded_at,
            "metadata": list(self.metadata),
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            entry_id=d["entry_id"],
            memory_id=d["memory_id"],
            memory_type=d["memory_type"],
            family_id=d["family_id"],
            content_hash=d["content_hash"],
            recorded_at=d["recorded_at"],
            metadata=tuple(tuple(p) for p in d["metadata"]),
        )


# ── Memory ─────────────────────────────────────────────────────────────────────


class ResearchMemory:
    """Long-term research history store.

    Immutable entries organized by family_id and memory_type.
    Never deletes or modifies entries.
    Deterministic: entry_id is content-addressed.
    """

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        self._entries: Dict[str, MemoryEntry] = {}            # entry_id → entry
        self._family_index: Dict[str, List[str]] = {}         # family_id → [entry_ids]
        self._type_index: Dict[str, List[str]] = {}           # memory_type → [entry_ids]

    # ── Record methods ─────────────────────────────────────────────────────────

    def record_exploration_path(
        self,
        family_id: str,
        hypothesis_ids: List[str],
        timestamp: Optional[str] = None,
    ) -> MemoryEntry:
        """Record which hypothesis IDs were explored for a family."""
        sorted_ids = sorted(hypothesis_ids)
        content = json.dumps({"family_id": family_id, "hypothesis_ids": sorted_ids}, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        metadata = (("hypothesis_count", str(len(sorted_ids))),)
        return self._record(
            MEMORY_EXPLORATION_PATH, family_id, content_hash, metadata, timestamp
        )

    def record_rejection_history(
        self,
        family_id: str,
        rejection_count: int,
        reasons: List[str],
        timestamp: Optional[str] = None,
    ) -> MemoryEntry:
        """Record rejection count and reasons for a family."""
        sorted_reasons = sorted(set(reasons))
        content = json.dumps({
            "family_id": family_id,
            "rejection_count": rejection_count,
            "reasons": sorted_reasons,
        }, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        metadata = (
            ("reason_count", str(len(sorted_reasons))),
            ("rejection_count", str(rejection_count)),
        )
        return self._record(
            MEMORY_REJECTION_HISTORY, family_id, content_hash, metadata, timestamp
        )

    def record_survivorship_trajectory(
        self,
        family_id: str,
        survivorship_rates: List[float],
        timestamp: Optional[str] = None,
    ) -> MemoryEntry:
        """Record a time-series of survival rates for a family."""
        rounded = [round(r, 6) for r in survivorship_rates]
        content = json.dumps({"family_id": family_id, "rates": rounded}, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        latest = round(rounded[-1], 6) if rounded else 0.0
        metadata = (
            ("latest_rate", str(latest)),
            ("step_count", str(len(rounded))),
        )
        return self._record(
            MEMORY_SURVIVORSHIP_TRAJECTORY, family_id, content_hash, metadata, timestamp
        )

    def record_topology_evolution(
        self,
        grammar_hash: str,
        fragility_scores: List[float],
        timestamp: Optional[str] = None,
    ) -> MemoryEntry:
        """Record a fragility score history for a grammar structure."""
        rounded = [round(s, 6) for s in fragility_scores]
        content = json.dumps({"grammar_hash": grammar_hash, "fragility_scores": rounded}, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        latest = round(rounded[-1], 6) if rounded else 0.0
        metadata = (
            ("latest_fragility", str(latest)),
            ("step_count", str(len(rounded))),
        )
        return self._record(
            MEMORY_TOPOLOGY_EVOLUTION, grammar_hash, content_hash, metadata, timestamp
        )

    # ── Query methods ──────────────────────────────────────────────────────────

    def has_explored(self, family_id: str) -> bool:
        """True if any EXPLORATION_PATH entry exists for this family."""
        return any(
            self._entries[eid].memory_type == MEMORY_EXPLORATION_PATH
            for eid in self._family_index.get(family_id, [])
        )

    def is_known_failure(self, family_id: str) -> bool:
        """True if any REJECTION_HISTORY entry exists for this family."""
        return any(
            self._entries[eid].memory_type == MEMORY_REJECTION_HISTORY
            for eid in self._family_index.get(family_id, [])
        )

    def known_failures(self) -> List[str]:
        """Sorted list of family_ids with any REJECTION_HISTORY entry."""
        result = set()
        for eid in self._type_index.get(MEMORY_REJECTION_HISTORY, []):
            result.add(self._entries[eid].family_id)
        return sorted(result)

    def exploration_path(self, family_id: str) -> List[MemoryEntry]:
        """All memory entries for this family, sorted by recorded_at."""
        ids = self._family_index.get(family_id, [])
        return sorted(
            [self._entries[eid] for eid in ids],
            key=lambda e: e.recorded_at,
        )

    def entries_of_type(self, memory_type: str) -> List[MemoryEntry]:
        """All entries of a given memory type, sorted by recorded_at."""
        ids = self._type_index.get(memory_type, [])
        return sorted(
            [self._entries[eid] for eid in ids],
            key=lambda e: e.recorded_at,
        )

    def all_entries(self) -> List[MemoryEntry]:
        """All entries sorted by recorded_at."""
        return sorted(self._entries.values(), key=lambda e: e.recorded_at)

    def memory_hash(self) -> str:
        """Deterministic 16-char hash of current memory state."""
        payload = json.dumps(sorted(self._entries.keys()), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def entry_count(self) -> int:
        return len(self._entries)

    def to_manifest(self) -> dict:
        entries = [e.to_dict() for e in self.all_entries()]
        return {
            "memory_id": self.memory_id,
            "schema_version": SCHEMA_VERSION,
            "entry_count": len(entries),
            "known_failure_count": len(self.known_failures()),
            "memory_hash": self.memory_hash(),
            "entries": entries,
        }

    # ── Private ────────────────────────────────────────────────────────────────

    def _record(
        self,
        memory_type: str,
        family_id: str,
        content_hash: str,
        metadata: tuple,
        timestamp: Optional[str],
    ) -> MemoryEntry:
        recorded_at = timestamp or utc_now_iso()
        entry_id = _entry_id(self.memory_id, family_id, memory_type, recorded_at)
        entry = MemoryEntry(
            entry_id=entry_id,
            memory_id=self.memory_id,
            memory_type=memory_type,
            family_id=family_id,
            content_hash=content_hash,
            recorded_at=recorded_at,
            metadata=metadata,
        )
        self._entries[entry_id] = entry
        self._family_index.setdefault(family_id, []).append(entry_id)
        self._type_index.setdefault(memory_type, []).append(entry_id)
        return entry


# ── Module helpers ─────────────────────────────────────────────────────────────


def _entry_id(memory_id: str, family_id: str, memory_type: str, recorded_at: str) -> str:
    payload = json.dumps({
        "memory_id": memory_id,
        "family_id": family_id,
        "memory_type": memory_type,
        "recorded_at": recorded_at,
    }, sort_keys=True)
    return "mem_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
