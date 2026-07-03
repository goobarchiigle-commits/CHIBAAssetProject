"""HypothesisEcologyTracker — evolving alpha ecology structure monitoring.

Monitors:
  - family persistence and extinction
  - mutation survivorship
  - topology concentration (Herfindahl on grammar hashes)
  - regime concentration (Herfindahl on taxonomy classes)
  - exploration saturation

Goal: make exploration itself measurable.
All state transitions are recorded immutably (append-only internal lists).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.research.governance import SCHEMA_VERSION
from src.research.analysis.artifacts import compute_checksum, utc_now_iso


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EcologySnapshot:
    """Immutable point-in-time snapshot of the alpha ecology."""
    snapshot_id: str
    taken_at: str
    family_count: int
    active_family_count: int
    extinct_family_count: int
    mutation_survivorship_rate: float   # survived mutations / total mutations
    topology_concentration: float       # HHI on grammar_hash distribution
    regime_concentration: float         # HHI on taxonomy_class distribution
    exploration_saturation: float       # total_hypotheses / total_space_size (0 if unknown)
    total_hypotheses: int
    survived_hypotheses: int
    schema_version: str
    checksum: str

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "taken_at": self.taken_at,
            "family_count": self.family_count,
            "active_family_count": self.active_family_count,
            "extinct_family_count": self.extinct_family_count,
            "mutation_survivorship_rate": self.mutation_survivorship_rate,
            "topology_concentration": self.topology_concentration,
            "regime_concentration": self.regime_concentration,
            "exploration_saturation": self.exploration_saturation,
            "total_hypotheses": self.total_hypotheses,
            "survived_hypotheses": self.survived_hypotheses,
            "schema_version": self.schema_version,
            "checksum": self.checksum,
        }


# ── Tracker ────────────────────────────────────────────────────────────────────


class HypothesisEcologyTracker:
    """Tracks the evolving structure of the alpha hypothesis ecology.

    All recorded events are append-only (never modified or deleted).
    Snapshots are computed on demand from accumulated state.
    """

    def __init__(self) -> None:
        # Family registry: family_id → {taxonomy_class, grammar_hash, is_active}
        self._families: Dict[str, dict] = {}
        # Extinctions: family_id → {reason, timestamp}
        self._extinctions: Dict[str, dict] = {}
        # Mutations: list of {parent_id, child_id, survived, timestamp}
        self._mutations: List[dict] = []
        # Hypotheses: list of {hypothesis_id, survived (bool|None), timestamp}
        self._hypotheses: List[dict] = []
        # Snapshots (immutable, append-only)
        self._snapshots: List[EcologySnapshot] = []

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_family(
        self,
        family_id: str,
        taxonomy_class: str,
        grammar_hash: str,
        is_active: bool = True,
        timestamp: Optional[str] = None,
    ) -> None:
        """Register a family (idempotent for same family_id; updates if re-registered)."""
        self._families[family_id] = {
            "taxonomy_class": taxonomy_class,
            "grammar_hash": grammar_hash,
            "is_active": is_active,
            "registered_at": timestamp or utc_now_iso(),
        }

    def record_extinction(
        self,
        family_id: str,
        reason: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Mark a family as extinct. Updates is_active=False in family registry."""
        recorded_at = timestamp or utc_now_iso()
        self._extinctions[family_id] = {"reason": reason, "timestamp": recorded_at}
        if family_id in self._families:
            self._families[family_id]["is_active"] = False

    def record_mutation(
        self,
        parent_id: str,
        child_id: str,
        survived: bool,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a mutation event and its survivorship outcome."""
        self._mutations.append({
            "parent_id": parent_id,
            "child_id": child_id,
            "survived": survived,
            "timestamp": timestamp or utc_now_iso(),
        })

    def record_hypothesis(
        self,
        hypothesis_id: str,
        survived: Optional[bool] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a hypothesis observation with optional survivorship label."""
        self._hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "survived": survived,
            "timestamp": timestamp or utc_now_iso(),
        })

    # ── Computed metrics ───────────────────────────────────────────────────────

    def topology_concentration(self) -> float:
        """Herfindahl index on grammar_hash distribution of active families."""
        active = [f for f in self._families.values() if f["is_active"]]
        return _herfindahl([f["grammar_hash"] for f in active])

    def regime_concentration(self) -> float:
        """Herfindahl index on taxonomy_class distribution of active families."""
        active = [f for f in self._families.values() if f["is_active"]]
        return _herfindahl([f["taxonomy_class"] for f in active])

    def mutation_survivorship_rate(self) -> float:
        """Fraction of mutations that survived. Returns 1.0 if no mutations."""
        if not self._mutations:
            return 1.0
        survived = sum(1 for m in self._mutations if m["survived"])
        return round(survived / len(self._mutations), 6)

    def exploration_saturation(self, total_space_size: int) -> float:
        """Fraction of hypothesis space explored. 0.0 if total_space_size=0."""
        if total_space_size <= 0:
            return 0.0
        return min(1.0, round(len(self._hypotheses) / total_space_size, 6))

    def active_families(self) -> List[str]:
        """Sorted list of active family IDs."""
        return sorted(fid for fid, f in self._families.items() if f["is_active"])

    def extinct_families(self) -> List[str]:
        """Sorted list of extinct family IDs."""
        return sorted(self._extinctions.keys())

    def family_taxonomy_map(self) -> Dict[str, str]:
        """family_id → taxonomy_class for all families."""
        return {fid: f["taxonomy_class"] for fid, f in self._families.items()}

    # ── Snapshot ───────────────────────────────────────────────────────────────

    def take_snapshot(
        self,
        total_space_size: int = 0,
        timestamp: Optional[str] = None,
    ) -> EcologySnapshot:
        """Compute and store an immutable ecology snapshot."""
        taken_at = timestamp or utc_now_iso()
        total_hyp = len(self._hypotheses)
        survived_hyp = sum(1 for h in self._hypotheses if h["survived"] is True)
        active = self.active_families()
        extinct = self.extinct_families()

        payload = {
            "taken_at": taken_at,
            "family_count": len(self._families),
            "active_family_count": len(active),
            "extinct_family_count": len(extinct),
            "mutation_survivorship_rate": self.mutation_survivorship_rate(),
            "topology_concentration": self.topology_concentration(),
            "regime_concentration": self.regime_concentration(),
            "exploration_saturation": self.exploration_saturation(total_space_size),
            "total_hypotheses": total_hyp,
            "survived_hypotheses": survived_hyp,
        }
        checksum = compute_checksum(payload)
        snap_id = _snapshot_id(taken_at, checksum)

        snapshot = EcologySnapshot(
            snapshot_id=snap_id,
            taken_at=taken_at,
            family_count=len(self._families),
            active_family_count=len(active),
            extinct_family_count=len(extinct),
            mutation_survivorship_rate=self.mutation_survivorship_rate(),
            topology_concentration=self.topology_concentration(),
            regime_concentration=self.regime_concentration(),
            exploration_saturation=self.exploration_saturation(total_space_size),
            total_hypotheses=total_hyp,
            survived_hypotheses=survived_hyp,
            schema_version=SCHEMA_VERSION,
            checksum=checksum,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def snapshots(self) -> List[EcologySnapshot]:
        """All snapshots in order taken (oldest first)."""
        return list(self._snapshots)

    def to_summary(self) -> dict:
        """Serializable summary of current ecology state."""
        snaps = [s.to_dict() for s in self._snapshots]
        return {
            "schema_version": SCHEMA_VERSION,
            "family_count": len(self._families),
            "active_family_count": len(self.active_families()),
            "extinct_family_count": len(self.extinct_families()),
            "mutation_count": len(self._mutations),
            "mutation_survivorship_rate": self.mutation_survivorship_rate(),
            "topology_concentration": self.topology_concentration(),
            "regime_concentration": self.regime_concentration(),
            "total_hypotheses": len(self._hypotheses),
            "snapshot_count": len(snaps),
            "snapshots": snaps,
        }


# ── Module helpers ─────────────────────────────────────────────────────────────


def _herfindahl(labels: List[str]) -> float:
    """Herfindahl-Hirschman index for a list of category labels.

    Returns 0.0 for empty list, 1.0 for a single unique label.
    """
    if not labels:
        return 0.0
    total = len(labels)
    counts: Dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return round(sum((n / total) ** 2 for n in counts.values()), 6)


def _snapshot_id(taken_at: str, checksum: str) -> str:
    payload = json.dumps({"taken_at": taken_at, "checksum": checksum[:8]}, sort_keys=True)
    return "eco_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
