"""HypothesisSpace — bounded comparable exploration space with lineage tracking.

Tracks:
  - family lineage (hypothesis → template)
  - parameter neighborhoods (grammar_hash → hypothesis IDs)
  - structural similarity (structural fingerprints)
  - mutation ancestry (parent → child chains)

Goal: survivorship analysis operates on structured alpha ecologies,
not isolated experiments.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Set

from src.research.templates import SCHEMA_VERSION
from src.research.templates.fingerprint import StructuralFingerprint
from src.research.templates.grammar import AlphaGrammar
from src.research.templates.template import HypothesisSpec, HypothesisTemplate


class HypothesisSpace:
    """Registry of templates and generated hypotheses with deterministic lineage.

    Maintains:
      - template registry (ordered by template_id)
      - hypothesis registry (ordered by hypothesis_id)
      - grammar index: structural_hash → [hypothesis_ids]
      - family index: family_id → [hypothesis_ids]
      - fingerprint index: for near-duplicate detection
    """

    def __init__(self, space_id: str) -> None:
        self.space_id = space_id
        self._templates: Dict[str, HypothesisTemplate] = {}
        self._hypotheses: Dict[str, HypothesisSpec] = {}
        self._grammar_index: Dict[str, List[str]] = {}    # grammar_hash → [hypothesis_ids]
        self._family_index: Dict[str, List[str]] = {}     # family_id → [hypothesis_ids]
        self._fingerprint_index: Dict[str, StructuralFingerprint] = {}  # hypothesis_id → fingerprint
        self._mutation_ancestry: Dict[str, Optional[str]] = {}  # hypothesis_id → parent_id

    # ── Template management ────────────────────────────────────────────────────

    def add_template(self, template: HypothesisTemplate) -> None:
        """Register a template. Raises ValueError on duplicate template_id."""
        if template.template_id in self._templates:
            raise ValueError(f"Template already registered: {template.template_id}")
        self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> HypothesisTemplate:
        if template_id not in self._templates:
            raise KeyError(f"Template not found: {template_id}")
        return self._templates[template_id]

    def template_ids(self) -> List[str]:
        """Sorted list of registered template IDs."""
        return sorted(self._templates.keys())

    # ── Hypothesis management ──────────────────────────────────────────────────

    def add_hypothesis(self, spec: HypothesisSpec) -> bool:
        """Register a hypothesis. Returns False (no-op) if hypothesis_id already exists.

        Maintains all indices atomically.
        """
        if spec.hypothesis_id in self._hypotheses:
            return False

        self._hypotheses[spec.hypothesis_id] = spec
        self._mutation_ancestry[spec.hypothesis_id] = spec.parent_id

        # Grammar index (by structural hash)
        self._grammar_index.setdefault(spec.grammar_hash, []).append(spec.hypothesis_id)

        # Family index
        self._family_index.setdefault(spec.family_id, []).append(spec.hypothesis_id)

        # Fingerprint index
        grammar_expr = AlphaGrammar.structural_expression(
            entry_type=spec.entry_type,
            exit_type=spec.exit_type,
            filter_types=spec.filter_types,
            holding_model=spec.holding_model,
            sizing_model=spec.sizing_model,
            regime_filter=spec.regime_filter,
        )
        self._fingerprint_index[spec.hypothesis_id] = StructuralFingerprint.from_expression(grammar_expr)

        return True

    def add_all(self, specs: List[HypothesisSpec]) -> Dict[str, int]:
        """Add a list of hypothesis specs. Returns counts: new, duplicate."""
        new_count = 0
        dup_count = 0
        for spec in specs:
            if self.add_hypothesis(spec):
                new_count += 1
            else:
                dup_count += 1
        return {"new": new_count, "duplicate": dup_count}

    def hypothesis_ids(self) -> List[str]:
        """Sorted list of all registered hypothesis IDs."""
        return sorted(self._hypotheses.keys())

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisSpec:
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Hypothesis not found: {hypothesis_id}")
        return self._hypotheses[hypothesis_id]

    # ── Family and structural queries ──────────────────────────────────────────

    def get_family(self, family_id: str) -> List[HypothesisSpec]:
        """All hypotheses in a family, sorted by hypothesis_id."""
        ids = self._family_index.get(family_id, [])
        return [self._hypotheses[hid] for hid in sorted(ids)]

    def family_ids(self) -> List[str]:
        """Sorted list of all family IDs."""
        return sorted(self._family_index.keys())

    def structural_siblings(self, spec: HypothesisSpec) -> List[HypothesisSpec]:
        """All hypotheses with the same grammar_hash (same structure, possibly different params)."""
        ids = self._grammar_index.get(spec.grammar_hash, [])
        return [
            self._hypotheses[hid]
            for hid in sorted(ids)
            if hid != spec.hypothesis_id
        ]

    def near_duplicates(
        self,
        spec: HypothesisSpec,
        threshold: Optional[float] = None,
    ) -> List[HypothesisSpec]:
        """Find hypotheses that are near-duplicates of spec by behavioral similarity.

        Returns sorted by hypothesis_id (deterministic).
        """
        from src.research.templates.fingerprint import NEAR_DUPLICATE_THRESHOLD
        thresh = threshold if threshold is not None else NEAR_DUPLICATE_THRESHOLD
        fp = self._fingerprint_index.get(spec.hypothesis_id)
        if fp is None:
            return []
        results = []
        for hid in sorted(self._fingerprint_index.keys()):
            if hid == spec.hypothesis_id:
                continue
            other_fp = self._fingerprint_index[hid]
            if fp.behavioral_similarity(other_fp) >= thresh:
                results.append(self._hypotheses[hid])
        return results

    def lineage_chain(self, hypothesis_id: str) -> List[str]:
        """Return full ancestry chain from root to this hypothesis (inclusive).

        Order: [root, ..., parent, hypothesis_id]
        """
        chain = [hypothesis_id]
        current = hypothesis_id
        seen: Set[str] = {current}
        while True:
            parent = self._mutation_ancestry.get(current)
            if parent is None or parent in seen:
                break
            seen.add(parent)
            chain.insert(0, parent)
            current = parent
        return chain

    # ── Space-level analytics ──────────────────────────────────────────────────

    def family_distribution(self) -> Dict[str, int]:
        """Family_id → member count. Sorted by family_id."""
        return {fid: len(ids) for fid, ids in sorted(self._family_index.items())}

    def grammar_distribution(self) -> Dict[str, int]:
        """Structural grammar_hash → hypothesis count. Sorted by hash."""
        return {gh: len(ids) for gh, ids in sorted(self._grammar_index.items())}

    def taxonomy_distribution(self) -> Dict[str, int]:
        """Taxonomy class → hypothesis count. Sorted by class name."""
        counts: Dict[str, int] = {}
        for spec in self._hypotheses.values():
            counts[spec.taxonomy_class] = counts.get(spec.taxonomy_class, 0) + 1
        return dict(sorted(counts.items()))

    def space_hash(self) -> str:
        """Deterministic 16-char hash of the current space state."""
        payload = json.dumps({
            "space_id": self.space_id,
            "hypothesis_ids": sorted(self._hypotheses.keys()),
            "template_ids": sorted(self._templates.keys()),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def snapshot(self) -> dict:
        """Serializable snapshot of the space state."""
        return {
            "space_id": self.space_id,
            "schema_version": SCHEMA_VERSION,
            "space_hash": self.space_hash(),
            "template_count": len(self._templates),
            "hypothesis_count": len(self._hypotheses),
            "family_count": len(self._family_index),
            "grammar_family_count": len(self._grammar_index),
            "template_ids": sorted(self._templates.keys()),
            "hypothesis_ids": sorted(self._hypotheses.keys()),
            "family_distribution": self.family_distribution(),
            "grammar_distribution": self.grammar_distribution(),
            "taxonomy_distribution": self.taxonomy_distribution(),
        }

    # ── Size ───────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._hypotheses)
