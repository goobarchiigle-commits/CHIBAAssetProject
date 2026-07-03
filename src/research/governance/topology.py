"""TopologyFailureRegistry — recurring structural failure pattern tracking.

Tracks:
  - narrow survivorship plateaus
  - cliff-like parameter regions
  - unstable neighborhood persistence
  - pathological turnover gradients
  - unstable regime transitions

Computes:
  - failure recurrence frequency
  - topology fragility persistence
  - collapse propagation signatures

Records are keyed by (grammar_hash, topology_type).
Repeated registration increments recurrence_count.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.research.governance import SCHEMA_VERSION
from src.research.analysis.artifacts import utc_now_iso

# ── Topology failure type constants ────────────────────────────────────────────

TOPOLOGY_NARROW_PLATEAU = "NARROW_PLATEAU"
TOPOLOGY_CLIFF_EDGE = "CLIFF_EDGE"
TOPOLOGY_UNSTABLE_NEIGHBORHOOD = "UNSTABLE_NEIGHBORHOOD"
TOPOLOGY_PATHOLOGICAL_TURNOVER = "PATHOLOGICAL_TURNOVER"
TOPOLOGY_UNSTABLE_REGIME_TRANSITION = "UNSTABLE_REGIME_TRANSITION"

ALL_TOPOLOGY_TYPES: frozenset = frozenset({
    TOPOLOGY_NARROW_PLATEAU,
    TOPOLOGY_CLIFF_EDGE,
    TOPOLOGY_UNSTABLE_NEIGHBORHOOD,
    TOPOLOGY_PATHOLOGICAL_TURNOVER,
    TOPOLOGY_UNSTABLE_REGIME_TRANSITION,
})


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TopologyFailureRecord:
    """Immutable record of a recurring topology failure pattern."""
    record_id: str                          # "topo_" + SHA256[:12]
    grammar_hash: str
    family_id: str                          # most-recently-affected family
    topology_type: str
    fragility_score: float                  # latest fragility reading (0.0–1.0)
    mean_fragility: float                   # running mean across all registrations
    recurrence_count: int
    collapse_propagation_signature: str     # 16-char SHA256 of failure pattern
    first_seen: str
    last_seen: str
    metadata: tuple                         # sorted (key, str_value) pairs

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "grammar_hash": self.grammar_hash,
            "family_id": self.family_id,
            "topology_type": self.topology_type,
            "fragility_score": self.fragility_score,
            "mean_fragility": self.mean_fragility,
            "recurrence_count": self.recurrence_count,
            "collapse_propagation_signature": self.collapse_propagation_signature,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "metadata": list(self.metadata),
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyFailureRecord":
        return cls(
            record_id=d["record_id"],
            grammar_hash=d["grammar_hash"],
            family_id=d["family_id"],
            topology_type=d["topology_type"],
            fragility_score=d["fragility_score"],
            mean_fragility=d["mean_fragility"],
            recurrence_count=d["recurrence_count"],
            collapse_propagation_signature=d["collapse_propagation_signature"],
            first_seen=d["first_seen"],
            last_seen=d["last_seen"],
            metadata=tuple(tuple(p) for p in d["metadata"]),
        )


# ── Registry ───────────────────────────────────────────────────────────────────


class TopologyFailureRegistry:
    """Registry of recurring topology failure patterns.

    Records are keyed by (grammar_hash, topology_type).
    Each registration updates the recurrence_count and mean_fragility.
    """

    def __init__(self) -> None:
        # (grammar_hash, topology_type) → TopologyFailureRecord
        self._records: Dict[Tuple[str, str], TopologyFailureRecord] = {}
        # Running fragility sum for mean computation
        self._fragility_sums: Dict[Tuple[str, str], float] = {}

    def register_topology_failure(
        self,
        grammar_hash: str,
        family_id: str,
        topology_type: str,
        fragility_score: float,
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> TopologyFailureRecord:
        """Register a topology failure occurrence."""
        if topology_type not in ALL_TOPOLOGY_TYPES:
            raise ValueError(
                f"Invalid topology_type: {topology_type!r}. "
                f"Valid: {sorted(ALL_TOPOLOGY_TYPES)}"
            )
        if not (0.0 <= fragility_score <= 1.0):
            raise ValueError(f"fragility_score must be in [0, 1], got {fragility_score}")

        recorded_at = timestamp or utc_now_iso()
        key = (grammar_hash, topology_type)
        existing = self._records.get(key)
        meta_sig = tuple(sorted((k, str(v)) for k, v in (metadata or {}).items()))
        record_id = _topo_record_id(grammar_hash, topology_type)
        sig = _collapse_signature(grammar_hash, topology_type, fragility_score)

        # Update running sum
        prev_sum = self._fragility_sums.get(key, 0.0)
        new_sum = prev_sum + fragility_score
        new_count = (existing.recurrence_count + 1) if existing else 1
        self._fragility_sums[key] = new_sum
        new_mean = round(new_sum / new_count, 6)

        record = TopologyFailureRecord(
            record_id=record_id,
            grammar_hash=grammar_hash,
            family_id=family_id,
            topology_type=topology_type,
            fragility_score=round(fragility_score, 6),
            mean_fragility=new_mean,
            recurrence_count=new_count,
            collapse_propagation_signature=sig,
            first_seen=existing.first_seen if existing else recorded_at,
            last_seen=recorded_at,
            metadata=meta_sig,
        )
        self._records[key] = record
        return record

    def has_topology_failure(
        self,
        grammar_hash: str,
        topology_type: Optional[str] = None,
    ) -> bool:
        """True if grammar_hash has any topology failure (or a specific type)."""
        if topology_type is not None:
            return (grammar_hash, topology_type) in self._records
        return any(k[0] == grammar_hash for k in self._records)

    def recurrence_count(self, grammar_hash: str) -> int:
        """Total recurrence count across all topology types for this grammar_hash."""
        return sum(
            r.recurrence_count
            for (gh, _), r in self._records.items()
            if gh == grammar_hash
        )

    def mean_fragility(self, grammar_hash: str) -> float:
        """Mean fragility score across all topology failure records for this grammar_hash."""
        relevant = [r for (gh, _), r in self._records.items() if gh == grammar_hash]
        if not relevant:
            return 0.0
        total_count = sum(r.recurrence_count for r in relevant)
        total_sum = sum(
            self._fragility_sums.get((grammar_hash, r.topology_type), 0.0)
            for r in relevant
        )
        return round(total_sum / max(total_count, 1), 6)

    def failure_types_for_grammar(self, grammar_hash: str) -> List[str]:
        """Sorted list of topology failure types for this grammar_hash."""
        return sorted(ft for (gh, ft) in self._records if gh == grammar_hash)

    def persistent_failures(self, min_recurrence: int = 2) -> List[TopologyFailureRecord]:
        """Records with recurrence_count >= min_recurrence, sorted by recurrence desc."""
        result = [r for r in self._records.values() if r.recurrence_count >= min_recurrence]
        return sorted(result, key=lambda r: (-r.recurrence_count, r.record_id))

    def all_records(self) -> List[TopologyFailureRecord]:
        """All records sorted by record_id."""
        return sorted(self._records.values(), key=lambda r: r.record_id)

    def record_count(self) -> int:
        return len(self._records)

    def affected_grammar_hashes(self) -> List[str]:
        """Sorted list of all grammar_hashes with any topology failure."""
        return sorted({k[0] for k in self._records})

    def to_manifest(self) -> dict:
        records = [r.to_dict() for r in self.all_records()]
        return {
            "schema_version": SCHEMA_VERSION,
            "record_count": len(records),
            "affected_grammar_count": len(self.affected_grammar_hashes()),
            "records": records,
        }


# ── Module helpers ─────────────────────────────────────────────────────────────


def _topo_record_id(grammar_hash: str, topology_type: str) -> str:
    payload = json.dumps({"grammar_hash": grammar_hash, "topology_type": topology_type}, sort_keys=True)
    return "topo_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _collapse_signature(grammar_hash: str, topology_type: str, fragility_score: float) -> str:
    payload = json.dumps({
        "grammar_hash": grammar_hash,
        "topology_type": topology_type,
        "fragility_score": round(fragility_score, 4),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
