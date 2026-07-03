"""ResearchConstraintEngine — deterministic governance of exploration quality.

Enforces:
  - exploration density limits (prevent over-saturation)
  - family branching limits (prevent explosive branching)
  - topology fragility suppression
  - failure recurrence suppression
  - unstable region suppression

All checks return (is_valid: bool, violations: List[str]).
Fail closed: any violation → exploration blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.research.governance.ledger import ExplorationLedger
from src.research.governance.negative import NegativeKnowledgeRegistry
from src.research.governance.topology import TopologyFailureRegistry


# ── Constraint parameters ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GovernanceConstraints:
    """Configurable governance thresholds."""
    max_exploration_density: float = 0.90       # suppress if density >= this
    max_family_branching: int = 10              # suppress if branch count > this
    max_failure_recurrence: int = 5             # suppress if total failures >= this
    max_topology_mean_fragility: float = 0.80   # suppress if mean_fragility > this
    max_unstable_fraction: float = 0.50         # suppress if UNSTABLE/total > this
    exploration_density_max_entries: int = 100  # denominator for density ratio

    def to_dict(self) -> dict:
        return {
            "max_exploration_density": self.max_exploration_density,
            "max_family_branching": self.max_family_branching,
            "max_failure_recurrence": self.max_failure_recurrence,
            "max_topology_mean_fragility": self.max_topology_mean_fragility,
            "max_unstable_fraction": self.max_unstable_fraction,
            "exploration_density_max_entries": self.exploration_density_max_entries,
        }


# ── Engine ─────────────────────────────────────────────────────────────────────


class ResearchConstraintEngine:
    """Deterministic fail-closed constraint evaluation for exploration governance.

    Usage:
        engine = ResearchConstraintEngine()
        valid, violations = engine.check_all(family_id, grammar_hash, ledger, neg, topo)
        if not valid:
            # block this exploration direction
    """

    def __init__(self, constraints: Optional[GovernanceConstraints] = None) -> None:
        self.constraints = constraints or GovernanceConstraints()

    # ── Individual checks ──────────────────────────────────────────────────────

    def check_exploration_density(
        self,
        family_id: str,
        ledger: ExplorationLedger,
    ) -> Tuple[bool, List[str]]:
        """Block if exploration density >= max_exploration_density."""
        density = ledger.exploration_density(
            family_id, self.constraints.exploration_density_max_entries
        )
        if density >= self.constraints.max_exploration_density:
            return False, [
                f"family {family_id!r}: exploration_density={density:.4f} "
                f">= max={self.constraints.max_exploration_density}"
            ]
        return True, []

    def check_topology_stability(
        self,
        grammar_hash: str,
        topology_registry: TopologyFailureRegistry,
    ) -> Tuple[bool, List[str]]:
        """Block if mean topology fragility exceeds threshold."""
        frag = topology_registry.mean_fragility(grammar_hash)
        if frag > self.constraints.max_topology_mean_fragility:
            return False, [
                f"grammar {grammar_hash!r}: mean_fragility={frag:.4f} "
                f"> max={self.constraints.max_topology_mean_fragility}"
            ]
        return True, []

    def check_failure_recurrence(
        self,
        family_id: str,
        negative_registry: NegativeKnowledgeRegistry,
    ) -> Tuple[bool, List[str]]:
        """Block if total failure count >= max_failure_recurrence."""
        count = negative_registry.failure_count(family_id)
        if count >= self.constraints.max_failure_recurrence:
            return False, [
                f"family {family_id!r}: failure_count={count} "
                f">= max={self.constraints.max_failure_recurrence}"
            ]
        return True, []

    def check_family_branching(
        self,
        family_id: str,
        branch_count: int,
    ) -> Tuple[bool, List[str]]:
        """Block if sub-family branch count exceeds limit."""
        if branch_count > self.constraints.max_family_branching:
            return False, [
                f"family {family_id!r}: branch_count={branch_count} "
                f"> max={self.constraints.max_family_branching}"
            ]
        return True, []

    def check_unstable_fraction(
        self,
        family_id: str,
        ledger: ExplorationLedger,
    ) -> Tuple[bool, List[str]]:
        """Block if fraction of entries marked UNSTABLE exceeds limit."""
        from src.research.governance.ledger import ENTRY_UNSTABLE
        all_entries = ledger.entries_for_family(family_id)
        if not all_entries:
            return True, []
        unstable = sum(1 for e in all_entries if e.entry_type == ENTRY_UNSTABLE)
        fraction = unstable / len(all_entries)
        if fraction > self.constraints.max_unstable_fraction:
            return False, [
                f"family {family_id!r}: unstable_fraction={fraction:.4f} "
                f"> max={self.constraints.max_unstable_fraction}"
            ]
        return True, []

    # ── Composite check ────────────────────────────────────────────────────────

    def check_all(
        self,
        family_id: str,
        grammar_hash: str,
        ledger: ExplorationLedger,
        negative_registry: NegativeKnowledgeRegistry,
        topology_registry: TopologyFailureRegistry,
    ) -> Tuple[bool, List[str]]:
        """Run all constraint checks. Aggregates violations from all checks."""
        all_violations: List[str] = []

        for check_fn, args in [
            (self.check_exploration_density, (family_id, ledger)),
            (self.check_topology_stability, (grammar_hash, topology_registry)),
            (self.check_failure_recurrence, (family_id, negative_registry)),
            (self.check_unstable_fraction, (family_id, ledger)),
        ]:
            _, violations = check_fn(*args)
            all_violations.extend(violations)

        return (len(all_violations) == 0, all_violations)

    def should_suppress(
        self,
        family_id: str,
        grammar_hash: str,
        ledger: ExplorationLedger,
        negative_registry: NegativeKnowledgeRegistry,
        topology_registry: TopologyFailureRegistry,
    ) -> bool:
        """True if this family should be suppressed (any constraint violated)."""
        valid, _ = self.check_all(family_id, grammar_hash, ledger, negative_registry, topology_registry)
        return not valid

    def to_dict(self) -> dict:
        return {"constraints": self.constraints.to_dict()}

    @staticmethod
    def default() -> "ResearchConstraintEngine":
        return ResearchConstraintEngine()
