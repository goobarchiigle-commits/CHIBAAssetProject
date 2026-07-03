"""ExplorationLedger — append-only immutable record of exploration activity.

Tracks:
  - explored families and parameter neighborhoods
  - rejected regions (constraint violations)
  - unstable regions (topology instability)
  - collapsed regions (survivorship collapse)

All entries are immutable. The ledger never overwrites or deletes.
Deterministic: same inputs at same timestamp → same entry_id.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.research.governance import SCHEMA_VERSION
from src.research.analysis.artifacts import utc_now_iso

# ── Entry type constants ───────────────────────────────────────────────────────

ENTRY_EXPLORED = "EXPLORED"
ENTRY_REJECTED = "REJECTED"
ENTRY_UNSTABLE = "UNSTABLE"
ENTRY_COLLAPSED = "COLLAPSED"

ALL_ENTRY_TYPES: frozenset = frozenset({
    ENTRY_EXPLORED, ENTRY_REJECTED, ENTRY_UNSTABLE, ENTRY_COLLAPSED,
})


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LedgerEntry:
    """Immutable record of a single exploration event."""
    entry_id: str           # "led_" + SHA256[:12]
    ledger_id: str
    family_id: str
    hypothesis_id: str
    template_id: str
    entry_type: str         # from ENTRY_* constants
    parameter_hash: str
    grammar_hash: str
    taxonomy_class: str
    recorded_at: str        # ISO8601 UTC
    reason: str
    metadata: tuple         # sorted (key, str_value) pairs

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "ledger_id": self.ledger_id,
            "family_id": self.family_id,
            "hypothesis_id": self.hypothesis_id,
            "template_id": self.template_id,
            "entry_type": self.entry_type,
            "parameter_hash": self.parameter_hash,
            "grammar_hash": self.grammar_hash,
            "taxonomy_class": self.taxonomy_class,
            "recorded_at": self.recorded_at,
            "reason": self.reason,
            "metadata": list(self.metadata),
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        return cls(
            entry_id=d["entry_id"],
            ledger_id=d["ledger_id"],
            family_id=d["family_id"],
            hypothesis_id=d["hypothesis_id"],
            template_id=d["template_id"],
            entry_type=d["entry_type"],
            parameter_hash=d["parameter_hash"],
            grammar_hash=d["grammar_hash"],
            taxonomy_class=d["taxonomy_class"],
            recorded_at=d["recorded_at"],
            reason=d["reason"],
            metadata=tuple(tuple(p) for p in d["metadata"]),
        )


# ── Ledger ─────────────────────────────────────────────────────────────────────


class ExplorationLedger:
    """Append-only exploration ledger.

    All exploration activity is recorded as immutable LedgerEntry objects.
    Supports lineage queries: by family, by grammar structure, by type.
    """

    def __init__(self, ledger_id: str) -> None:
        self.ledger_id = ledger_id
        self._entries: Dict[str, LedgerEntry] = {}           # entry_id → entry
        self._family_index: Dict[str, List[str]] = {}        # family_id → [entry_ids]
        self._grammar_index: Dict[str, List[str]] = {}       # grammar_hash → [entry_ids]
        self._type_index: Dict[str, List[str]] = {}          # entry_type → [entry_ids]

    # ── Record methods ─────────────────────────────────────────────────────────

    def record_explored(
        self,
        family_id: str,
        hypothesis_id: str,
        template_id: str,
        parameter_hash: str,
        grammar_hash: str,
        taxonomy_class: str,
        reason: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> LedgerEntry:
        return self._record(
            ENTRY_EXPLORED, family_id, hypothesis_id, template_id,
            parameter_hash, grammar_hash, taxonomy_class, reason, metadata, timestamp,
        )

    def record_rejected(
        self,
        family_id: str,
        hypothesis_id: str,
        template_id: str,
        parameter_hash: str,
        grammar_hash: str,
        taxonomy_class: str,
        reason: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> LedgerEntry:
        return self._record(
            ENTRY_REJECTED, family_id, hypothesis_id, template_id,
            parameter_hash, grammar_hash, taxonomy_class, reason, metadata, timestamp,
        )

    def record_unstable(
        self,
        family_id: str,
        hypothesis_id: str,
        template_id: str,
        parameter_hash: str,
        grammar_hash: str,
        taxonomy_class: str,
        reason: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> LedgerEntry:
        return self._record(
            ENTRY_UNSTABLE, family_id, hypothesis_id, template_id,
            parameter_hash, grammar_hash, taxonomy_class, reason, metadata, timestamp,
        )

    def record_collapsed(
        self,
        family_id: str,
        hypothesis_id: str,
        template_id: str,
        parameter_hash: str,
        grammar_hash: str,
        taxonomy_class: str,
        reason: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> LedgerEntry:
        return self._record(
            ENTRY_COLLAPSED, family_id, hypothesis_id, template_id,
            parameter_hash, grammar_hash, taxonomy_class, reason, metadata, timestamp,
        )

    # ── Query methods ──────────────────────────────────────────────────────────

    def entries_for_family(self, family_id: str) -> List[LedgerEntry]:
        """All entries for a family, sorted by recorded_at."""
        ids = self._family_index.get(family_id, [])
        return sorted([self._entries[i] for i in ids], key=lambda e: e.recorded_at)

    def entries_for_grammar(self, grammar_hash: str) -> List[LedgerEntry]:
        """All entries for a grammar hash, sorted by recorded_at."""
        ids = self._grammar_index.get(grammar_hash, [])
        return sorted([self._entries[i] for i in ids], key=lambda e: e.recorded_at)

    def entries_of_type(self, entry_type: str) -> List[LedgerEntry]:
        """All entries of a given type, sorted by recorded_at."""
        ids = self._type_index.get(entry_type, [])
        return sorted([self._entries[i] for i in ids], key=lambda e: e.recorded_at)

    def explored_families(self) -> Set[str]:
        """Family IDs with at least one EXPLORED entry."""
        return {
            self._entries[i].family_id
            for i in self._type_index.get(ENTRY_EXPLORED, [])
        }

    def rejected_families(self) -> Set[str]:
        """Family IDs with at least one REJECTED entry."""
        return {
            self._entries[i].family_id
            for i in self._type_index.get(ENTRY_REJECTED, [])
        }

    def unstable_families(self) -> Set[str]:
        """Family IDs with at least one UNSTABLE entry."""
        return {
            self._entries[i].family_id
            for i in self._type_index.get(ENTRY_UNSTABLE, [])
        }

    def collapsed_families(self) -> Set[str]:
        """Family IDs with at least one COLLAPSED entry."""
        return {
            self._entries[i].family_id
            for i in self._type_index.get(ENTRY_COLLAPSED, [])
        }

    def is_explored(self, family_id: str) -> bool:
        return family_id in self.explored_families()

    def is_rejected(self, family_id: str) -> bool:
        return family_id in self.rejected_families()

    def exploration_density(self, family_id: str, max_entries: int = 100) -> float:
        """Fraction of exploration budget consumed for this family.

        density = explored_count / max_entries, clamped to [0, 1].
        Returns 0.0 for unexplored families.
        """
        explored = [
            e for e in self.entries_for_family(family_id)
            if e.entry_type == ENTRY_EXPLORED
        ]
        return min(1.0, len(explored) / max(max_entries, 1))

    def all_entries(self) -> List[LedgerEntry]:
        """All entries sorted by recorded_at."""
        return sorted(self._entries.values(), key=lambda e: e.recorded_at)

    def entry_count(self) -> int:
        return len(self._entries)

    def family_count(self) -> int:
        return len(self._family_index)

    def to_manifest(self) -> dict:
        entries = [e.to_dict() for e in self.all_entries()]
        payload = {
            "ledger_id": self.ledger_id,
            "schema_version": SCHEMA_VERSION,
            "entry_count": len(entries),
            "family_count": self.family_count(),
            "explored_family_count": len(self.explored_families()),
            "rejected_family_count": len(self.rejected_families()),
            "entries": entries,
        }
        return payload

    # ── Private ────────────────────────────────────────────────────────────────

    def _record(
        self,
        entry_type: str,
        family_id: str,
        hypothesis_id: str,
        template_id: str,
        parameter_hash: str,
        grammar_hash: str,
        taxonomy_class: str,
        reason: str,
        metadata: Optional[dict],
        timestamp: Optional[str],
    ) -> LedgerEntry:
        if entry_type not in ALL_ENTRY_TYPES:
            raise ValueError(f"Invalid entry_type: {entry_type!r}")
        recorded_at = timestamp or utc_now_iso()
        meta_sig = tuple(sorted((k, str(v)) for k, v in (metadata or {}).items()))
        entry_id = _entry_id(self.ledger_id, family_id, hypothesis_id, entry_type, recorded_at)
        entry = LedgerEntry(
            entry_id=entry_id,
            ledger_id=self.ledger_id,
            family_id=family_id,
            hypothesis_id=hypothesis_id,
            template_id=template_id,
            entry_type=entry_type,
            parameter_hash=parameter_hash,
            grammar_hash=grammar_hash,
            taxonomy_class=taxonomy_class,
            recorded_at=recorded_at,
            reason=reason,
            metadata=meta_sig,
        )
        self._entries[entry_id] = entry
        self._family_index.setdefault(family_id, []).append(entry_id)
        self._grammar_index.setdefault(grammar_hash, []).append(entry_id)
        self._type_index.setdefault(entry_type, []).append(entry_id)
        return entry


# ── Module helpers ─────────────────────────────────────────────────────────────


def _entry_id(ledger_id: str, family_id: str, hypothesis_id: str, entry_type: str, recorded_at: str) -> str:
    payload = json.dumps({
        "ledger_id": ledger_id,
        "family_id": family_id,
        "hypothesis_id": hypothesis_id,
        "entry_type": entry_type,
        "recorded_at": recorded_at,
    }, sort_keys=True)
    return "led_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
