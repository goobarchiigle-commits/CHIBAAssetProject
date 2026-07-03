"""HypothesisTemplate — bounded research space definition.

A template defines a finite, enumerable hypothesis space via:
  - Canonical domain selections (entry/exit/filter/holding/sizing/regime types)
  - Parameter domains (bounded lists of allowed values per parameter)
  - Constraint hierarchy (validation rules)

Templates are immutable after construction.
template_id is deterministic from template content.
All generated HypothesisSpec objects have deterministic IDs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.research.templates import SCHEMA_VERSION
from src.research.templates.canonical import (
    ALL_ENTRY_TYPES,
    ALL_EXIT_TYPES,
    ALL_FILTER_TYPES,
    ALL_HOLDING_MODELS,
    ALL_REGIME_FILTERS,
    ALL_SIZING_MODELS,
    CanonicalSemantics,
    REGIME_ANY,
)
from src.research.templates.constraints import ConstraintHierarchy
from src.research.templates.grammar import AlphaGrammar
from src.research.templates.taxonomy import AlphaTaxonomy


@dataclass(frozen=True)
class HypothesisSpec:
    """Immutable concrete hypothesis generated from a template.

    hypothesis_id is deterministic: same structural choices + parameters → same id.
    """
    hypothesis_id: str        # "hyp_" + SHA256[:12]
    template_id: str
    family_id: str            # structural family within template
    parent_id: Optional[str]  # mutation ancestry (None for root hypotheses)

    entry_type: str
    exit_type: str
    filter_types: tuple       # sorted tuple of canonical filter type strings
    holding_model: str
    sizing_model: str
    regime_filter: Optional[str]

    parameters: tuple         # sorted (key, str_value) pairs
    parameter_hash: str       # 16-char hash of parameters dict

    grammar_hash: str         # 16-char structural hash (no params)
    grammar_expression: str   # human-readable canonical repr
    taxonomy_class: str       # from AlphaTaxonomy

    created_at: str           # ISO8601 UTC
    schema_version: str

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "template_id": self.template_id,
            "family_id": self.family_id,
            "parent_id": self.parent_id,
            "entry_type": self.entry_type,
            "exit_type": self.exit_type,
            "filter_types": list(self.filter_types),
            "holding_model": self.holding_model,
            "sizing_model": self.sizing_model,
            "regime_filter": self.regime_filter,
            "parameters": list(self.parameters),
            "parameter_hash": self.parameter_hash,
            "grammar_hash": self.grammar_hash,
            "grammar_expression": self.grammar_expression,
            "taxonomy_class": self.taxonomy_class,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisSpec":
        return cls(
            hypothesis_id=d["hypothesis_id"],
            template_id=d["template_id"],
            family_id=d["family_id"],
            parent_id=d.get("parent_id"),
            entry_type=d["entry_type"],
            exit_type=d["exit_type"],
            filter_types=tuple(d["filter_types"]),
            holding_model=d["holding_model"],
            sizing_model=d["sizing_model"],
            regime_filter=d.get("regime_filter"),
            parameters=tuple(tuple(p) for p in d["parameters"]),
            parameter_hash=d["parameter_hash"],
            grammar_hash=d["grammar_hash"],
            grammar_expression=d["grammar_expression"],
            taxonomy_class=d["taxonomy_class"],
            created_at=d["created_at"],
            schema_version=d["schema_version"],
        )

    def parameter_dict(self) -> Dict[str, Any]:
        """Reconstruct parameters as a dict."""
        return dict(self.parameters)


class HypothesisTemplate:
    """Bounded research space definition.

    Defines finite enumerable domains for all hypothesis components.
    Validates against ConstraintHierarchy before generation.

    Args:
        name: Human-readable template name (used in template_id).
        entry_types: Sorted tuple of canonical entry type strings.
        exit_types: Sorted tuple of canonical exit type strings.
        filter_combos: Tuple of sorted tuples representing allowed filter combinations.
                       Include empty tuple () for "no filters" option.
        holding_models: Sorted tuple of canonical holding model strings.
        sizing_models: Sorted tuple of canonical sizing model strings.
        regime_filters: Sorted tuple of regime filter values (include None for "any").
        parameter_domains: Dict of param_name → sorted list of allowed values.
        constraints: ConstraintHierarchy for validation.
    """

    def __init__(
        self,
        name: str,
        entry_types: Tuple[str, ...],
        exit_types: Tuple[str, ...],
        filter_combos: Tuple[Tuple[str, ...], ...],
        holding_models: Tuple[str, ...],
        sizing_models: Tuple[str, ...],
        regime_filters: Tuple[Optional[str], ...],
        parameter_domains: Dict[str, List[Any]],
        constraints: Optional[ConstraintHierarchy] = None,
    ) -> None:
        self.name = name
        self.entry_types = tuple(sorted(entry_types))
        self.exit_types = tuple(sorted(exit_types))
        self.filter_combos = tuple(
            tuple(sorted(c)) for c in sorted(filter_combos, key=lambda c: sorted(c))
        )
        self.holding_models = tuple(sorted(holding_models))
        self.sizing_models = tuple(sorted(sizing_models))
        # Sort regime filters, placing None last for stability
        self.regime_filters = tuple(sorted(regime_filters, key=lambda x: x or ""))
        self.parameter_domains = {
            k: list(v) for k, v in sorted(parameter_domains.items())
        }
        self.constraints = constraints or ConstraintHierarchy.default()

        # Validate canonical types
        CanonicalSemantics.validate_all_types(
            self.entry_types,
            self.exit_types,
            self.filter_combos,
            self.holding_models,
            self.sizing_models,
        )

        # Validate constraints
        valid, violations = self.constraints.check_template_structure(
            self.entry_types,
            self.exit_types,
            self.filter_combos,
            self.holding_models,
            self.sizing_models,
            self.parameter_domains,
        )
        if not valid:
            raise ValueError(
                f"Template {name!r} constraint violations:\n" +
                "\n".join(f"  - {v}" for v in violations)
            )

        # Compute template_id from content
        self.template_id = self._compute_template_id()

    # ── Public interface ───────────────────────────────────────────────────────

    def generate_specs(
        self,
        timestamp: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> List[HypothesisSpec]:
        """Generate all HypothesisSpec objects for this template.

        Returns list sorted by hypothesis_id for determinism.
        Same template + same timestamp → same IDs (timestamp only affects created_at label).
        """
        created_at = timestamp or datetime.now(timezone.utc).isoformat()
        specs: List[HypothesisSpec] = []
        seen_ids: set = set()

        for entry_type in self.entry_types:
            for exit_type in self.exit_types:
                for filter_combo in self.filter_combos:
                    for holding_model in self.holding_models:
                        for sizing_model in self.sizing_models:
                            for regime_filter in self.regime_filters:
                                for params in self._parameter_combinations():
                                    spec = self._build_spec(
                                        entry_type=entry_type,
                                        exit_type=exit_type,
                                        filter_combo=filter_combo,
                                        holding_model=holding_model,
                                        sizing_model=sizing_model,
                                        regime_filter=regime_filter,
                                        params=params,
                                        created_at=created_at,
                                        parent_id=parent_id,
                                    )
                                    if spec.hypothesis_id not in seen_ids:
                                        seen_ids.add(spec.hypothesis_id)
                                        specs.append(spec)

        return sorted(specs, key=lambda s: s.hypothesis_id)

    def estimate_size(self) -> int:
        """Deterministic estimate of total hypothesis count."""
        structural = (
            len(self.entry_types) *
            len(self.exit_types) *
            max(len(self.filter_combos), 1) *
            len(self.holding_models) *
            len(self.sizing_models) *
            max(len(self.regime_filters), 1)
        )
        param_product = 1
        for values in self.parameter_domains.values():
            param_product *= len(values)
        return structural * param_product

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "entry_types": list(self.entry_types),
            "exit_types": list(self.exit_types),
            "filter_combos": [list(c) for c in self.filter_combos],
            "holding_models": list(self.holding_models),
            "sizing_models": list(self.sizing_models),
            "regime_filters": list(self.regime_filters),
            "parameter_domains": {k: list(v) for k, v in self.parameter_domains.items()},
            "estimated_size": self.estimate_size(),
            "schema_version": SCHEMA_VERSION,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_template_id(self) -> str:
        payload = json.dumps({
            "name": self.name,
            "entry_types": list(self.entry_types),
            "exit_types": list(self.exit_types),
            "filter_combos": [list(c) for c in self.filter_combos],
            "holding_models": list(self.holding_models),
            "sizing_models": list(self.sizing_models),
            "regime_filters": list(self.regime_filters),
            "parameter_domains": {k: list(v) for k, v in self.parameter_domains.items()},
        }, sort_keys=True)
        return "tmpl_" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    def _parameter_combinations(self) -> List[Dict[str, Any]]:
        """All cartesian product combinations of parameter domains, sorted by param_hash."""
        if not self.parameter_domains:
            return [{}]
        keys = list(self.parameter_domains.keys())  # already sorted
        value_lists = [self.parameter_domains[k] for k in keys]
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
        combos.sort(key=lambda p: _param_hash(p))
        return combos

    def _build_spec(
        self,
        entry_type: str,
        exit_type: str,
        filter_combo: tuple,
        holding_model: str,
        sizing_model: str,
        regime_filter: Optional[str],
        params: Dict[str, Any],
        created_at: str,
        parent_id: Optional[str],
    ) -> HypothesisSpec:
        filter_types = tuple(sorted(filter_combo))
        param_sig = tuple(sorted((k, str(v)) for k, v in params.items()))
        ph = _param_hash(params)

        # Grammar expression (structural, for hash)
        grammar_expr = AlphaGrammar.structural_expression(
            entry_type=entry_type,
            exit_type=exit_type,
            filter_types=filter_types,
            holding_model=holding_model,
            sizing_model=sizing_model,
            regime_filter=regime_filter,
        )
        grammar_hash = grammar_expr.structural_hash()
        grammar_repr = grammar_expr.structural_repr()

        # Family ID: structural within this template
        family_id = _family_id(self.template_id, entry_type, exit_type, filter_types)

        # Hypothesis ID: full content-addressed
        hypothesis_id = _hypothesis_id(
            self.template_id, entry_type, exit_type, filter_types,
            holding_model, sizing_model, regime_filter, ph,
        )

        # Taxonomy
        taxonomy_class = AlphaTaxonomy.classify(entry_type, filter_types, regime_filter)

        return HypothesisSpec(
            hypothesis_id=hypothesis_id,
            template_id=self.template_id,
            family_id=family_id,
            parent_id=parent_id,
            entry_type=entry_type,
            exit_type=exit_type,
            filter_types=filter_types,
            holding_model=holding_model,
            sizing_model=sizing_model,
            regime_filter=regime_filter,
            parameters=param_sig,
            parameter_hash=ph,
            grammar_hash=grammar_hash,
            grammar_expression=grammar_repr,
            taxonomy_class=taxonomy_class,
            created_at=created_at,
            schema_version=SCHEMA_VERSION,
        )


# ── Module-level helpers ───────────────────────────────────────────────────────

def _param_hash(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _family_id(
    template_id: str,
    entry_type: str,
    exit_type: str,
    filter_types: tuple,
) -> str:
    payload = json.dumps({
        "template_id": template_id,
        "entry_type": entry_type,
        "exit_type": exit_type,
        "filter_types": list(filter_types),
    }, sort_keys=True)
    return "fam_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _hypothesis_id(
    template_id: str,
    entry_type: str,
    exit_type: str,
    filter_types: tuple,
    holding_model: str,
    sizing_model: str,
    regime_filter: Optional[str],
    parameter_hash: str,
) -> str:
    payload = json.dumps({
        "template_id": template_id,
        "entry_type": entry_type,
        "exit_type": exit_type,
        "filter_types": list(filter_types),
        "holding_model": holding_model,
        "sizing_model": sizing_model,
        "regime_filter": regime_filter,
        "parameter_hash": parameter_hash,
    }, sort_keys=True)
    return "hyp_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
