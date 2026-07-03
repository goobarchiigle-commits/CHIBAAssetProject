"""NegativeKnowledgeRegistry — persist failed research outcomes immutably.

Tracks:
  - dead topologies
  - unstable parameter surfaces
  - regime-fragile families
  - excessive turnover structures
  - collapse-prone mutations
  - persistent rejection patterns

All records are content-addressed by (family_id, failure_type).
Repeated registration of the same failure increments the count.
Records are never deleted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.research.governance import SCHEMA_VERSION
from src.research.analysis.artifacts import utc_now_iso

# ── Failure type constants ─────────────────────────────────────────────────────

FAILURE_DEAD_TOPOLOGY = "DEAD_TOPOLOGY"
FAILURE_UNSTABLE_SURFACE = "UNSTABLE_SURFACE"
FAILURE_REGIME_FRAGILE = "REGIME_FRAGILE"
FAILURE_EXCESSIVE_TURNOVER = "EXCESSIVE_TURNOVER"
FAILURE_COLLAPSE_PRONE = "COLLAPSE_PRONE"
FAILURE_PERSISTENT_REJECTION = "PERSISTENT_REJECTION"

ALL_FAILURE_TYPES: frozenset = frozenset({
    FAILURE_DEAD_TOPOLOGY,
    FAILURE_UNSTABLE_SURFACE,
    FAILURE_REGIME_FRAGILE,
    FAILURE_EXCESSIVE_TURNOVER,
    FAILURE_COLLAPSE_PRONE,
    FAILURE_PERSISTENT_REJECTION,
})


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NegativeKnowledgeRecord:
    """Immutable record of a persistent failure pattern for a family."""
    record_id: str              # "nkr_" + SHA256[:12] of (family_id, failure_type)
    family_id: str
    grammar_hash: str
    failure_type: str           # from FAILURE_* constants
    failure_signature: str      # 16-char SHA256 of (family_id, failure_type, grammar_hash)
    failure_count: int
    first_seen: str             # ISO8601 UTC
    last_seen: str              # ISO8601 UTC
    metadata: tuple             # sorted (key, str_value) pairs

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "family_id": self.family_id,
            "grammar_hash": self.grammar_hash,
            "failure_type": self.failure_type,
            "failure_signature": self.failure_signature,
            "failure_count": self.failure_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "metadata": list(self.metadata),
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeKnowledgeRecord":
        return cls(
            record_id=d["record_id"],
            family_id=d["family_id"],
            grammar_hash=d["grammar_hash"],
            failure_type=d["failure_type"],
            failure_signature=d["failure_signature"],
            failure_count=d["failure_count"],
            first_seen=d["first_seen"],
            last_seen=d["last_seen"],
            metadata=tuple(tuple(p) for p in d["metadata"]),
        )


# ── Registry ───────────────────────────────────────────────────────────────────


class NegativeKnowledgeRegistry:
    """Registry of persistent failure patterns.

    When the same (family_id, failure_type) pair is registered again,
    the existing record is replaced with an updated one (higher failure_count,
    updated last_seen). The record_id remains stable for a given pair.
    """

    def __init__(self) -> None:
        # Keyed by (family_id, failure_type) → NegativeKnowledgeRecord
        self._records: Dict[Tuple[str, str], NegativeKnowledgeRecord] = {}

    def register_failure(
        self,
        family_id: str,
        grammar_hash: str,
        failure_type: str,
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> NegativeKnowledgeRecord:
        """Register a failure occurrence. Returns the updated record."""
        if failure_type not in ALL_FAILURE_TYPES:
            raise ValueError(
                f"Invalid failure_type: {failure_type!r}. "
                f"Valid: {sorted(ALL_FAILURE_TYPES)}"
            )
        recorded_at = timestamp or utc_now_iso()
        key = (family_id, failure_type)
        existing = self._records.get(key)

        meta_sig = tuple(sorted((k, str(v)) for k, v in (metadata or {}).items()))
        record_id = _record_id(family_id, failure_type)
        sig = _failure_signature(family_id, failure_type, grammar_hash)

        if existing is None:
            record = NegativeKnowledgeRecord(
                record_id=record_id,
                family_id=family_id,
                grammar_hash=grammar_hash,
                failure_type=failure_type,
                failure_signature=sig,
                failure_count=1,
                first_seen=recorded_at,
                last_seen=recorded_at,
                metadata=meta_sig,
            )
        else:
            record = NegativeKnowledgeRecord(
                record_id=record_id,
                family_id=family_id,
                grammar_hash=existing.grammar_hash,
                failure_type=failure_type,
                failure_signature=sig,
                failure_count=existing.failure_count + 1,
                first_seen=existing.first_seen,
                last_seen=recorded_at,
                metadata=meta_sig,
            )

        self._records[key] = record
        return record

    def has_failure(self, family_id: str, failure_type: Optional[str] = None) -> bool:
        """True if family has any failure (or a specific failure_type)."""
        if failure_type is not None:
            return (family_id, failure_type) in self._records
        return any(k[0] == family_id for k in self._records)

    def failure_count(self, family_id: str) -> int:
        """Total failure occurrences across all failure types for this family."""
        return sum(
            r.failure_count
            for (fid, _), r in self._records.items()
            if fid == family_id
        )

    def failure_types_for_family(self, family_id: str) -> List[str]:
        """Sorted list of failure types recorded for this family."""
        return sorted(ft for (fid, ft) in self._records if fid == family_id)

    def failure_signature(self, family_id: str) -> str:
        """Combined 16-char signature over all failure types for this family."""
        types = self.failure_types_for_family(family_id)
        if not types:
            return ""
        payload = json.dumps({"family_id": family_id, "failure_types": types}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def persistent_failures(self, min_count: int = 3) -> List[NegativeKnowledgeRecord]:
        """Records with failure_count >= min_count, sorted by failure_count desc, then record_id."""
        result = [r for r in self._records.values() if r.failure_count >= min_count]
        return sorted(result, key=lambda r: (-r.failure_count, r.record_id))

    def all_records(self) -> List[NegativeKnowledgeRecord]:
        """All records, sorted by record_id."""
        return sorted(self._records.values(), key=lambda r: r.record_id)

    def record_count(self) -> int:
        return len(self._records)

    def affected_families(self) -> List[str]:
        """Sorted list of all family IDs with any failure."""
        return sorted({k[0] for k in self._records})

    def to_manifest(self) -> dict:
        records = [r.to_dict() for r in self.all_records()]
        return {
            "schema_version": SCHEMA_VERSION,
            "record_count": len(records),
            "affected_family_count": len(self.affected_families()),
            "records": records,
        }


# ── Module helpers ─────────────────────────────────────────────────────────────


def _record_id(family_id: str, failure_type: str) -> str:
    payload = json.dumps({"family_id": family_id, "failure_type": failure_type}, sort_keys=True)
    return "nkr_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _failure_signature(family_id: str, failure_type: str, grammar_hash: str) -> str:
    payload = json.dumps({
        "family_id": family_id,
        "failure_type": failure_type,
        "grammar_hash": grammar_hash,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
