"""ExplorationBudgetController — bounded deterministic research attention allocation.

Allocates exploration budget across families based on priority classification.
All allocation is deterministic: same state → same allocation.

Priority classes (descending priority):
  PERSISTENT_CORE      — enduring alpha structure, highest priority
  UNDEREXPLORED        — no exploration entries yet, needs baseline coverage
  PROMISING            — active but not core, default for non-suppressed families
  STABLE_PLATEAU       — no topology failures, stable structure
  SUPPRESSED           — blocked by failure recurrence or constraint violations

This is NOT adaptive optimization.
It is governed deterministic prioritization.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.research.governance import SCHEMA_VERSION
from src.research.governance.ledger import ExplorationLedger
from src.research.governance.negative import NegativeKnowledgeRegistry
from src.research.governance.persistence import FamilyPersistenceTracker
from src.research.governance.topology import TopologyFailureRegistry
from src.research.analysis.artifacts import utc_now_iso

# ── Priority class constants ───────────────────────────────────────────────────

PRIORITY_PERSISTENT_CORE = "PERSISTENT_CORE"
PRIORITY_UNDEREXPLORED = "UNDEREXPLORED"
PRIORITY_PROMISING = "PROMISING"
PRIORITY_STABLE_PLATEAU = "STABLE_PLATEAU"
PRIORITY_SUPPRESSED = "SUPPRESSED"

# Priority scores for budget allocation (SUPPRESSED = 0)
_PRIORITY_SCORES: Dict[str, int] = {
    PRIORITY_PERSISTENT_CORE: 5,
    PRIORITY_UNDEREXPLORED: 4,
    PRIORITY_PROMISING: 3,
    PRIORITY_STABLE_PLATEAU: 2,
    PRIORITY_SUPPRESSED: 0,
}


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BudgetAllocation:
    """Immutable budget allocation for a single family."""
    allocation_id: str          # "bud_" + SHA256[:12]
    family_id: str
    priority_class: str
    allocated_budget: int       # number of hypotheses to explore (0 if suppressed)
    reason: str
    generated_at: str

    def to_dict(self) -> dict:
        return {
            "allocation_id": self.allocation_id,
            "family_id": self.family_id,
            "priority_class": self.priority_class,
            "allocated_budget": self.allocated_budget,
            "reason": self.reason,
            "generated_at": self.generated_at,
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetAllocation":
        return cls(
            allocation_id=d["allocation_id"],
            family_id=d["family_id"],
            priority_class=d["priority_class"],
            allocated_budget=d["allocated_budget"],
            reason=d["reason"],
            generated_at=d["generated_at"],
        )


# ── Controller ─────────────────────────────────────────────────────────────────


class ExplorationBudgetController:
    """Deterministic bounded exploration budget allocator.

    Args:
        total_budget:          Total hypothesis budget across all families.
        max_per_family:        Maximum budget per family (cap).
        suppression_threshold: Failure count at which a family is suppressed.
        topology_fragility_cap: Mean fragility above which family is suppressed.
    """

    def __init__(
        self,
        total_budget: int = 1000,
        max_per_family: int = 100,
        suppression_threshold: int = 5,
        topology_fragility_cap: float = 0.80,
    ) -> None:
        if total_budget < 0:
            raise ValueError("total_budget must be >= 0")
        if max_per_family < 0:
            raise ValueError("max_per_family must be >= 0")
        if suppression_threshold < 1:
            raise ValueError("suppression_threshold must be >= 1")
        self.total_budget = total_budget
        self.max_per_family = max_per_family
        self.suppression_threshold = suppression_threshold
        self.topology_fragility_cap = topology_fragility_cap

    def classify_family(
        self,
        family_id: str,
        grammar_hash: str,
        ledger: ExplorationLedger,
        negative_registry: NegativeKnowledgeRegistry,
        topology_registry: TopologyFailureRegistry,
        persistence_tracker: FamilyPersistenceTracker,
    ) -> Tuple[str, str]:
        """Classify family into a priority class.

        Returns (priority_class, reason).
        Deterministic: same registry state → same classification.
        """
        # 1. Suppression: failure recurrence
        failure_count = negative_registry.failure_count(family_id)
        if failure_count >= self.suppression_threshold:
            return PRIORITY_SUPPRESSED, f"failure_count={failure_count} >= threshold={self.suppression_threshold}"

        # 2. Suppression: topology fragility
        frag = topology_registry.mean_fragility(grammar_hash)
        if frag > self.topology_fragility_cap:
            return PRIORITY_SUPPRESSED, f"mean_fragility={frag:.4f} > cap={self.topology_fragility_cap}"

        # 3. Persistent core
        if family_id in persistence_tracker.tracked_families():
            try:
                rec = persistence_tracker.compute_persistence(family_id)
                if rec.is_enduring_core:
                    return PRIORITY_PERSISTENT_CORE, (
                        f"persistence_rate={rec.persistence_rate:.4f} "
                        f"mutation_survivorship={rec.mutation_survivorship:.4f}"
                    )
            except KeyError:
                pass

        # 4. Underexplored
        if ledger.exploration_density(family_id) == 0.0:
            return PRIORITY_UNDEREXPLORED, "no exploration entries"

        # 5. Stable plateau: no topology failures
        if not topology_registry.has_topology_failure(grammar_hash):
            return PRIORITY_STABLE_PLATEAU, "no topology failures recorded"

        # 6. Default: promising
        return PRIORITY_PROMISING, "active non-suppressed family"

    def allocate(
        self,
        family_grammar_map: Dict[str, str],
        ledger: ExplorationLedger,
        negative_registry: NegativeKnowledgeRegistry,
        topology_registry: TopologyFailureRegistry,
        persistence_tracker: FamilyPersistenceTracker,
        timestamp: Optional[str] = None,
    ) -> List[BudgetAllocation]:
        """Allocate exploration budget across families.

        Args:
            family_grammar_map: {family_id: grammar_hash} for all families to allocate.

        Returns:
            Sorted list of BudgetAllocation (by family_id for determinism).
        """
        generated_at = timestamp or utc_now_iso()

        # Classify all families
        classifications: List[Tuple[str, str, str]] = []  # (family_id, priority, reason)
        for fid in sorted(family_grammar_map.keys()):
            gh = family_grammar_map[fid]
            priority, reason = self.classify_family(
                fid, gh, ledger, negative_registry, topology_registry, persistence_tracker
            )
            classifications.append((fid, priority, reason))

        # Separate suppressed from allocatable
        non_suppressed = [(fid, p, r) for fid, p, r in classifications if p != PRIORITY_SUPPRESSED]
        suppressed = [(fid, p, r) for fid, p, r in classifications if p == PRIORITY_SUPPRESSED]

        # Compute raw allocations for non-suppressed families
        total_score = sum(_PRIORITY_SCORES[p] for _, p, _ in non_suppressed)
        raw_allocs: Dict[str, int] = {}

        if total_score > 0 and non_suppressed:
            remaining = self.total_budget
            # Sort by (score desc, family_id asc) for stable deterministic ordering
            ordered = sorted(non_suppressed, key=lambda x: (-_PRIORITY_SCORES[x[1]], x[0]))
            for fid, priority, _ in ordered:
                score = _PRIORITY_SCORES[priority]
                raw = int(score / total_score * self.total_budget)
                alloc = min(raw, self.max_per_family, remaining)
                raw_allocs[fid] = alloc
                remaining -= alloc
        else:
            for fid, _, _ in non_suppressed:
                raw_allocs[fid] = 0

        # Build allocation objects
        allocations: List[BudgetAllocation] = []

        for fid, priority, reason in sorted(classifications, key=lambda x: x[0]):
            budget = raw_allocs.get(fid, 0)
            alloc = BudgetAllocation(
                allocation_id=_allocation_id(fid, generated_at),
                family_id=fid,
                priority_class=priority,
                allocated_budget=budget,
                reason=reason,
                generated_at=generated_at,
            )
            allocations.append(alloc)

        return allocations

    def total_allocated(self, allocations: List[BudgetAllocation]) -> int:
        """Sum of allocated budgets."""
        return sum(a.allocated_budget for a in allocations)

    def suppressed_families(self, allocations: List[BudgetAllocation]) -> List[str]:
        """Sorted list of suppressed family IDs."""
        return sorted(a.family_id for a in allocations if a.priority_class == PRIORITY_SUPPRESSED)

    def to_report(self, allocations: List[BudgetAllocation]) -> dict:
        """Serializable allocation report."""
        entries = [a.to_dict() for a in allocations]
        return {
            "schema_version": SCHEMA_VERSION,
            "total_budget": self.total_budget,
            "total_allocated": self.total_allocated(allocations),
            "family_count": len(allocations),
            "suppressed_count": len(self.suppressed_families(allocations)),
            "allocations": entries,
        }

    def to_dict(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "max_per_family": self.max_per_family,
            "suppression_threshold": self.suppression_threshold,
            "topology_fragility_cap": self.topology_fragility_cap,
        }


# ── Module helpers ─────────────────────────────────────────────────────────────


def _allocation_id(family_id: str, generated_at: str) -> str:
    payload = json.dumps({"family_id": family_id, "generated_at": generated_at}, sort_keys=True)
    return "bud_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
