"""StructuralFingerprint — deterministic behavioral fingerprints for near-duplicate detection.

Detects structurally equivalent hypotheses before experimentation.
Computes feature overlap signatures from canonical grammar types.

All computations are deterministic: same inputs → same fingerprint, always.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, Set, Tuple

from src.research.templates.canonical import (
    ENTRY_GAP_CONTINUATION,
    ENTRY_GAP_FADE,
    ENTRY_LIQUIDITY_SWEEP,
    ENTRY_MEAN_REVERSION_BOUNCE,
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    ENTRY_VOLATILITY_BREAKOUT,
    FILTER_LIQUIDITY,
    FILTER_MOMENTUM_STRENGTH,
    FILTER_REGIME,
    FILTER_TREND,
    FILTER_VOLATILITY,
)
from src.research.templates.grammar import GrammarExpression

# ── Feature implication map ────────────────────────────────────────────────────
# Each canonical type implies a set of feature dependencies.

_ENTRY_FEATURES: Dict[str, FrozenSet[str]] = {
    ENTRY_MOMENTUM_BREAKOUT: frozenset({"momentum", "returns", "atr", "rsr"}),
    ENTRY_TURTLE_BREAKOUT:   frozenset({"high_channel", "low_channel", "atr", "returns"}),
    ENTRY_MEAN_REVERSION_BOUNCE: frozenset({"rsi", "bollinger", "returns", "atr"}),
    ENTRY_GAP_CONTINUATION:  frozenset({"gap_pct", "atr", "returns"}),
    ENTRY_GAP_FADE:          frozenset({"gap_pct", "atr", "momentum", "returns"}),
    ENTRY_VOLATILITY_BREAKOUT: frozenset({"atr", "bollinger", "volatility", "returns"}),
    ENTRY_LIQUIDITY_SWEEP:   frozenset({"volume", "spread", "atr", "returns"}),
}

_FILTER_FEATURES: Dict[str, FrozenSet[str]] = {
    FILTER_VOLATILITY:         frozenset({"atr", "volatility"}),
    FILTER_LIQUIDITY:          frozenset({"volume", "spread"}),
    FILTER_REGIME:             frozenset({"regime_label", "adx"}),
    FILTER_TREND:              frozenset({"adx", "ma_ratio"}),
    FILTER_MOMENTUM_STRENGTH:  frozenset({"rsr", "momentum"}),
}

# Similarity threshold above which two fingerprints are "near-duplicates"
NEAR_DUPLICATE_THRESHOLD: float = 0.80


@dataclass(frozen=True)
class StructuralFingerprint:
    """Deterministic behavioral fingerprint for a hypothesis.

    structural_equivalence_hash: hash of grammar structure (no params).
        Two hypotheses with the same hash use identical logical components.
    full_grammar_hash: hash of grammar including parameters.
        Identical fingerprints are exact structural+parameter matches.
    feature_overlap_signature: sorted tuple of feature names implied by the grammar.
    """
    structural_equivalence_hash: str   # 16-char hash, structure only
    full_grammar_hash: str             # 16-char hash, structure + params
    feature_overlap_signature: tuple   # sorted tuple of feature name strings

    @staticmethod
    def from_expression(expression: GrammarExpression) -> "StructuralFingerprint":
        """Compute fingerprint from a GrammarExpression."""
        features: Set[str] = set()
        for clause in expression.clauses:
            entry_feats = _ENTRY_FEATURES.get(clause.canonical_type)
            if entry_feats:
                features.update(entry_feats)
            filter_feats = _FILTER_FEATURES.get(clause.canonical_type)
            if filter_feats:
                features.update(filter_feats)

        return StructuralFingerprint(
            structural_equivalence_hash=expression.structural_hash(),
            full_grammar_hash=expression.grammar_hash(),
            feature_overlap_signature=tuple(sorted(features)),
        )

    def signal_overlap_ratio(self, other: "StructuralFingerprint") -> float:
        """Jaccard similarity of feature overlap signatures. [0, 1]."""
        a = set(self.feature_overlap_signature)
        b = set(other.feature_overlap_signature)
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def behavioral_similarity(self, other: "StructuralFingerprint") -> float:
        """Composite similarity: 0.7 * structural_match + 0.3 * feature_overlap.

        Returns 1.0 for identical structural hash.
        Returns 0.0 for completely disjoint structures with no feature overlap.
        """
        structural_match = 1.0 if self.structural_equivalence_hash == other.structural_equivalence_hash else 0.0
        feature_sim = self.signal_overlap_ratio(other)
        return round(0.7 * structural_match + 0.3 * feature_sim, 6)

    def is_near_duplicate(self, other: "StructuralFingerprint") -> bool:
        """True if behavioral_similarity >= NEAR_DUPLICATE_THRESHOLD."""
        return self.behavioral_similarity(other) >= NEAR_DUPLICATE_THRESHOLD

    def is_exact_structural_match(self, other: "StructuralFingerprint") -> bool:
        """True if same structure, regardless of parameters."""
        return self.structural_equivalence_hash == other.structural_equivalence_hash

    def canonical_id(self) -> str:
        """Stable 16-char ID for this fingerprint."""
        payload = json.dumps({
            "structural": self.structural_equivalence_hash,
            "features": list(self.feature_overlap_signature),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "structural_equivalence_hash": self.structural_equivalence_hash,
            "full_grammar_hash": self.full_grammar_hash,
            "feature_overlap_signature": list(self.feature_overlap_signature),
            "canonical_id": self.canonical_id(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralFingerprint":
        return cls(
            structural_equivalence_hash=d["structural_equivalence_hash"],
            full_grammar_hash=d["full_grammar_hash"],
            feature_overlap_signature=tuple(d["feature_overlap_signature"]),
        )
