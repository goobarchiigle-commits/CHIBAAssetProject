"""AlphaGenerator — deterministic alpha variant generation from governed spaces.

Generates HypothesisSpec objects from HypothesisTemplates with:
  - Deterministic ordering (stable by hypothesis_id)
  - Structural duplicate detection via StructuralFingerprint
  - Budget enforcement before generation
  - Append-only lineage tracking

NOT a generic strategy generator. Only generates bounded, canonical hypothesis families.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.research.templates import SCHEMA_VERSION
from src.research.templates.budget import BudgetExceededError, BudgetReport, SearchBudgetController
from src.research.templates.fingerprint import StructuralFingerprint, NEAR_DUPLICATE_THRESHOLD
from src.research.templates.grammar import AlphaGrammar
from src.research.templates.space import HypothesisSpace
from src.research.templates.template import HypothesisSpec, HypothesisTemplate


@dataclass
class GenerationManifest:
    """Immutable record of a generation run."""
    manifest_id: str
    generated_at: str
    schema_version: str
    template_id: str
    total_candidates: int
    accepted: int
    rejected_duplicates: int
    rejected_structural_duplicates: int
    budget_report: BudgetReport
    hypothesis_ids: Tuple[str, ...]        # sorted
    duplicate_ids: Tuple[str, ...]         # exact duplicates skipped
    structural_duplicate_pairs: Tuple[Tuple[str, str], ...]  # (new_id, existing_id)

    def to_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "total_candidates": self.total_candidates,
            "accepted": self.accepted,
            "rejected_duplicates": self.rejected_duplicates,
            "rejected_structural_duplicates": self.rejected_structural_duplicates,
            "budget_report": self.budget_report.to_dict(),
            "hypothesis_ids": list(self.hypothesis_ids),
            "duplicate_ids": list(self.duplicate_ids),
            "structural_duplicate_pairs": [list(p) for p in self.structural_duplicate_pairs],
        }


class AlphaGenerator:
    """Generates governed alpha variants from HypothesisTemplates.

    All generation is:
      - Deterministic: same template + space state → same output
      - Bounded: budget enforced before generation begins
      - Duplicate-safe: exact and structural duplicates flagged
      - Lineage-tracked: hypotheses are linked to their template

    Args:
        budget:   SearchBudgetController for bounds enforcement.
        detect_structural_duplicates: if True, flag near-duplicates in manifest.
        structural_duplicate_threshold: similarity threshold for near-duplicate detection.
    """

    def __init__(
        self,
        budget: Optional[SearchBudgetController] = None,
        detect_structural_duplicates: bool = True,
        structural_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    ) -> None:
        self.budget = budget or SearchBudgetController()
        self.detect_structural_duplicates = detect_structural_duplicates
        self.structural_duplicate_threshold = structural_duplicate_threshold

    def generate_from_template(
        self,
        template: HypothesisTemplate,
        space: HypothesisSpace,
        timestamp: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> GenerationManifest:
        """Generate all hypothesis variants from a template into a HypothesisSpace.

        Pre-flight: budget enforcement (raises BudgetExceededError if over limit).
        Generation: all valid combinations, sorted by hypothesis_id.
        Post-flight: structural duplicate detection.

        Args:
            template: The template to generate from.
            space:    Target HypothesisSpace (modified in place).
            timestamp: ISO8601 timestamp label (uses UTC now if None).
            parent_id: Optional parent hypothesis ID for mutation ancestry.

        Returns:
            GenerationManifest with full accounting of accepted/rejected hypotheses.

        Raises:
            BudgetExceededError: if template exceeds configured budget limits.
        """
        generated_at = timestamp or datetime.now(timezone.utc).isoformat()

        # Pre-flight budget check
        budget_report = self.budget.enforce_template(
            entry_types=template.entry_types,
            exit_types=template.exit_types,
            filter_combos=template.filter_combos,
            holding_models=template.holding_models,
            sizing_models=template.sizing_models,
            regime_filters=template.regime_filters,
            parameter_domains=template.parameter_domains,
            template_id=template.template_id,
        )

        # Generate candidates
        candidates = template.generate_specs(timestamp=generated_at, parent_id=parent_id)

        # Register template in space (idempotent check)
        if template.template_id not in space.template_ids():
            space.add_template(template)

        # Add to space and track duplicates
        accepted_ids: List[str] = []
        duplicate_ids: List[str] = []
        structural_dup_pairs: List[Tuple[str, str]] = []

        # Build fingerprint map for existing space contents (for near-dup detection)
        existing_fingerprints: Dict[str, StructuralFingerprint] = {}
        if self.detect_structural_duplicates:
            for hid in space.hypothesis_ids():
                existing_spec = space.get_hypothesis(hid)
                expr = AlphaGrammar.structural_expression(
                    entry_type=existing_spec.entry_type,
                    exit_type=existing_spec.exit_type,
                    filter_types=existing_spec.filter_types,
                    holding_model=existing_spec.holding_model,
                    sizing_model=existing_spec.sizing_model,
                    regime_filter=existing_spec.regime_filter,
                )
                existing_fingerprints[hid] = StructuralFingerprint.from_expression(expr)

        # Process candidates in hypothesis_id order (already sorted)
        new_fingerprints: Dict[str, StructuralFingerprint] = {}

        for spec in candidates:
            # Exact duplicate check
            if spec.hypothesis_id in space.hypothesis_ids():
                duplicate_ids.append(spec.hypothesis_id)
                continue

            # Structural near-duplicate check against already-accepted in this run
            if self.detect_structural_duplicates:
                expr = AlphaGrammar.structural_expression(
                    entry_type=spec.entry_type,
                    exit_type=spec.exit_type,
                    filter_types=spec.filter_types,
                    holding_model=spec.holding_model,
                    sizing_model=spec.sizing_model,
                    regime_filter=spec.regime_filter,
                )
                new_fp = StructuralFingerprint.from_expression(expr)

                for existing_id, existing_fp in sorted(existing_fingerprints.items()):
                    if new_fp.behavioral_similarity(existing_fp) >= self.structural_duplicate_threshold:
                        structural_dup_pairs.append((spec.hypothesis_id, existing_id))
                        break  # flag first match only

            # Accept hypothesis
            space.add_hypothesis(spec)
            accepted_ids.append(spec.hypothesis_id)

            # Update fingerprint map for subsequent near-dup checks
            if self.detect_structural_duplicates:
                expr = AlphaGrammar.structural_expression(
                    entry_type=spec.entry_type,
                    exit_type=spec.exit_type,
                    filter_types=spec.filter_types,
                    holding_model=spec.holding_model,
                    sizing_model=spec.sizing_model,
                    regime_filter=spec.regime_filter,
                )
                existing_fingerprints[spec.hypothesis_id] = StructuralFingerprint.from_expression(expr)

        manifest_id = _make_manifest_id(template.template_id, generated_at)

        return GenerationManifest(
            manifest_id=manifest_id,
            generated_at=generated_at,
            schema_version=SCHEMA_VERSION,
            template_id=template.template_id,
            total_candidates=len(candidates),
            accepted=len(accepted_ids),
            rejected_duplicates=len(duplicate_ids),
            rejected_structural_duplicates=len(
                {pair[0] for pair in structural_dup_pairs}
            ),
            budget_report=budget_report,
            hypothesis_ids=tuple(sorted(accepted_ids)),
            duplicate_ids=tuple(sorted(duplicate_ids)),
            structural_duplicate_pairs=tuple(sorted(structural_dup_pairs)),
        )

    def generate_all(
        self,
        templates: List[HypothesisTemplate],
        space: HypothesisSpace,
        timestamp: Optional[str] = None,
    ) -> List[GenerationManifest]:
        """Generate from all templates. Enforces space-level budget before any generation.

        Args:
            templates: List of templates (processed in template_id order for determinism).
            space:     Target HypothesisSpace.
            timestamp: Shared timestamp label for this generation run.

        Returns:
            List of GenerationManifests, one per template.

        Raises:
            BudgetExceededError: if total space exceeds budget.
        """
        generated_at = timestamp or datetime.now(timezone.utc).isoformat()

        # Pre-flight: estimate all templates, check space total
        reports: List[BudgetReport] = []
        for template in sorted(templates, key=lambda t: t.template_id):
            report = self.budget.estimate_template_size(
                entry_types=template.entry_types,
                exit_types=template.exit_types,
                filter_combos=template.filter_combos,
                holding_models=template.holding_models,
                sizing_models=template.sizing_models,
                regime_filters=template.regime_filters,
                parameter_domains=template.parameter_domains,
                template_id=template.template_id,
            )
            if not report.within_budget:
                raise BudgetExceededError(
                    f"Template {template.template_id!r} exceeds per-template budget:\n" +
                    "\n".join(f"  - {v}" for v in report.violations)
                )
            reports.append(report)

        self.budget.enforce_space_total(reports)

        # Generate from each template in deterministic order
        manifests: List[GenerationManifest] = []
        for template in sorted(templates, key=lambda t: t.template_id):
            manifest = self.generate_from_template(template, space, timestamp=generated_at)
            manifests.append(manifest)

        return manifests


def _make_manifest_id(template_id: str, generated_at: str) -> str:
    payload = json.dumps({"template_id": template_id, "generated_at": generated_at}, sort_keys=True)
    compact_ts = generated_at.replace("-", "").replace(":", "").replace("T", "").replace("Z", "").replace(".", "")[:20]
    h = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"manifest_{compact_ts}_{h}"
