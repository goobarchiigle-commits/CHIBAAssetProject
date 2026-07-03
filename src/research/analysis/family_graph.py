"""ExperimentFamilyGraph — deterministic family-level survivorship grouping.

Groups experiments into families based on strategy lineage and feature composition.
Tracks family survivorship, persistence, and robust cores.

Grouping is union-find over deterministic similarity criteria:
  1. Same strategy_id (always merged)
  2. Feature Jaccard similarity >= threshold

Family IDs are deterministic hashes of sorted member experiment_ids.
No ML, no embeddings, no randomness.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple

from src.research.analysis.contract import FamilyRecord, SurvivorRecord

# Minimum Jaccard similarity for feature-based grouping
_FEATURE_JACCARD_THRESH = 0.50
# Minimum fraction of members sharing a feature for family signature
_SIGNATURE_MAJORITY = 0.60
# Robust-core thresholds
_ROBUST_SURVIVORSHIP = 0.60
_ROBUST_PERSISTENCE = 0.50


class ExperimentFamilyGraph:
    """Build and analyze family groupings from a population of SurvivorRecords.

    Args:
        feature_jaccard_threshold: minimum Jaccard similarity for feature-based merging.
        signature_majority: minimum fraction of members that must share a feature
                            for it to appear in the family's feature_signature.
    """

    def __init__(
        self,
        feature_jaccard_threshold: float = _FEATURE_JACCARD_THRESH,
        signature_majority: float = _SIGNATURE_MAJORITY,
    ) -> None:
        if not (0.0 <= feature_jaccard_threshold <= 1.0):
            raise ValueError(f"feature_jaccard_threshold must be in [0,1], got {feature_jaccard_threshold}")
        if not (0.0 < signature_majority <= 1.0):
            raise ValueError(f"signature_majority must be in (0,1], got {signature_majority}")
        self.feature_jaccard_threshold = feature_jaccard_threshold
        self.signature_majority = signature_majority

    def build(self, records: List[SurvivorRecord]) -> List[FamilyRecord]:
        """Build family records from a list of SurvivorRecords.

        Args:
            records: any order; internally sorted by experiment_id for stability.

        Returns:
            List[FamilyRecord] sorted by family_id for determinism.
        """
        if not records:
            return []

        sorted_records = sorted(records, key=lambda r: r.experiment_id)
        ids = [r.experiment_id for r in sorted_records]

        # Union-Find over sorted experiment_ids
        parent = {eid: eid for eid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Deterministic tie-break: smaller id becomes root
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        record_map: Dict[str, SurvivorRecord] = {r.experiment_id: r for r in sorted_records}

        # Pass 1: merge by same strategy_id
        strategy_groups: Dict[str, List[str]] = {}
        for r in sorted_records:
            strategy_groups.setdefault(r.strategy_id, []).append(r.experiment_id)
        for members in strategy_groups.values():
            for i in range(1, len(members)):
                union(members[0], members[i])

        # Pass 2: merge by feature Jaccard similarity (only within same strategy_id)
        # This avoids merging unrelated strategies that happen to share features.
        for i, r_a in enumerate(sorted_records):
            for r_b in sorted_records[i + 1:]:
                if r_a.strategy_id != r_b.strategy_id:
                    continue
                j = _jaccard(set(r_a.feature_names), set(r_b.feature_names))
                if j >= self.feature_jaccard_threshold:
                    union(r_a.experiment_id, r_b.experiment_id)

        # Collect groups
        groups: Dict[str, List[str]] = {}
        for eid in sorted(ids):
            root = find(eid)
            groups.setdefault(root, []).append(eid)

        # Build FamilyRecord for each group
        family_records: List[FamilyRecord] = []
        for root in sorted(groups.keys()):
            members = sorted(groups[root])
            family_records.append(self._build_family(members, record_map))

        return sorted(family_records, key=lambda f: f.family_id)

    def _build_family(
        self,
        member_ids: List[str],  # already sorted
        record_map: Dict[str, SurvivorRecord],
    ) -> FamilyRecord:
        members = [record_map[eid] for eid in member_ids]

        # Family ID: hash of sorted member experiment_ids
        family_id = _family_id_hash(member_ids)

        # strategy_id: most common (or first alphabetically for ties)
        strategy_id = _majority_value([r.strategy_id for r in members])

        # Feature signature: features present in >= signature_majority fraction of members
        feature_signature = _feature_signature(members, self.signature_majority)

        # Regime affinity: regimes where mean sharpe across members > 0
        regime_affinity = _regime_affinity(members)

        # Survivorship and persistence
        n_survived = sum(1 for r in members if r.survived)
        family_survivorship = n_survived / len(members)

        surviving_scores = [r.score for r in members if r.survived]
        family_persistence = (
            sum(surviving_scores) / len(surviving_scores)
            if surviving_scores else 0.0
        )

        is_robust = (
            family_survivorship >= _ROBUST_SURVIVORSHIP
            and family_persistence >= _ROBUST_PERSISTENCE
        )

        return FamilyRecord(
            family_id=family_id,
            strategy_id=strategy_id,
            member_experiment_ids=tuple(member_ids),
            feature_signature=feature_signature,
            regime_affinity=regime_affinity,
            family_survivorship=round(family_survivorship, 6),
            family_persistence_score=round(family_persistence, 6),
            is_robust_core=is_robust,
        )

    # ── Family-level analytics ─────────────────────────────────────────────────

    @staticmethod
    def survivorship_distribution(families: List[FamilyRecord]) -> Dict[str, float]:
        """Percentile distribution of family_survivorship across all families.

        Returns sorted dict: {"p10": ..., "p25": ..., "p50": ..., "p75": ..., "p90": ...}
        """
        if not families:
            return {}
        scores = sorted(f.family_survivorship for f in families)
        n = len(scores)

        def percentile(p: float) -> float:
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return round(scores[lo] * (1 - frac) + scores[hi] * frac, 6)

        return {
            "p10": percentile(10),
            "p25": percentile(25),
            "p50": percentile(50),
            "p75": percentile(75),
            "p90": percentile(90),
            "mean": round(sum(scores) / n, 6),
        }

    @staticmethod
    def robust_cores(families: List[FamilyRecord]) -> List[FamilyRecord]:
        """Return families classified as robust cores, sorted by survivorship desc."""
        return sorted(
            [f for f in families if f.is_robust_core],
            key=lambda f: (-f.family_survivorship, f.family_id),
        )

    @staticmethod
    def family_correlation(
        f1: FamilyRecord,
        f2: FamilyRecord,
    ) -> float:
        """Jaccard similarity of member experiment_id sets (structural overlap)."""
        s1 = set(f1.member_experiment_ids)
        s2 = set(f2.member_experiment_ids)
        return _jaccard(s1, s2)


# ── Module helpers ─────────────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _family_id_hash(sorted_member_ids: List[str]) -> str:
    payload = json.dumps(sorted_member_ids, sort_keys=True)
    return "fam_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _majority_value(values: List[str]) -> str:
    """Most frequent value; alphabetically first for ties."""
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts.keys()), key=lambda k: counts[k])


def _feature_signature(
    members: List[SurvivorRecord],
    majority: float,
) -> tuple:
    """Features present in >= majority fraction of members. Sorted tuple."""
    if not members:
        return ()
    all_features: Dict[str, int] = {}
    for r in members:
        for feat in r.feature_names:
            all_features[feat] = all_features.get(feat, 0) + 1
    threshold = majority * len(members)
    return tuple(sorted(f for f, c in all_features.items() if c >= threshold))


def _regime_affinity(members: List[SurvivorRecord]) -> tuple:
    """Sorted tuple of regime names where mean family sharpe > 0."""
    regime_sums: Dict[str, List[float]] = {}
    for r in members:
        for regime, sharpe in r.regime_score_pairs:
            regime_sums.setdefault(regime, []).append(sharpe)
    affinity = []
    for regime in sorted(regime_sums.keys()):
        scores = regime_sums[regime]
        if sum(scores) / len(scores) > 0:
            affinity.append(regime)
    return tuple(affinity)
