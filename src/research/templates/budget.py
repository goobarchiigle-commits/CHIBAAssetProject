"""SearchBudgetController — prevents combinatorial explosion in hypothesis generation.

Enforces strict bounds on generation spaces before enumeration begins.
Fails closed: any limit exceeded → BudgetExceededError raised.

All checks are deterministic and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


class BudgetExceededError(ValueError):
    """Raised when a hypothesis space exceeds configured budget limits."""


@dataclass(frozen=True)
class BudgetReport:
    """Deterministic report of estimated generation costs."""
    template_id: str
    estimated_variants: int
    structural_combinations: int   # entry × exit × filter_combos × hold × size × regime
    parameter_combinations: int    # product of all parameter domain sizes
    max_parameter_cardinality: int  # largest single parameter domain
    within_budget: bool
    violations: tuple              # sorted tuple of violation strings

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "estimated_variants": self.estimated_variants,
            "structural_combinations": self.structural_combinations,
            "parameter_combinations": self.parameter_combinations,
            "max_parameter_cardinality": self.max_parameter_cardinality,
            "within_budget": self.within_budget,
            "violations": list(self.violations),
        }


class SearchBudgetController:
    """Budget controller for bounded deterministic hypothesis generation.

    Args:
        max_variants_per_template:    Maximum hypothesis variants from a single template.
        max_parameter_cardinality:    Maximum number of values in any one parameter domain.
        max_total_generation_space:   Maximum total variants across all templates in a space.
        max_neighborhood_density:     Maximum neighbors per point in topology analysis.
        max_family_branching_factor:  Maximum sub-families per structural family.
    """

    def __init__(
        self,
        max_variants_per_template: int = 500,
        max_parameter_cardinality: int = 10,
        max_total_generation_space: int = 5_000,
        max_neighborhood_density: int = 20,
        max_family_branching_factor: int = 50,
    ) -> None:
        if max_variants_per_template < 1:
            raise ValueError("max_variants_per_template must be >= 1")
        if max_parameter_cardinality < 1:
            raise ValueError("max_parameter_cardinality must be >= 1")
        if max_total_generation_space < 1:
            raise ValueError("max_total_generation_space must be >= 1")
        if max_neighborhood_density < 1:
            raise ValueError("max_neighborhood_density must be >= 1")
        if max_family_branching_factor < 1:
            raise ValueError("max_family_branching_factor must be >= 1")

        self.max_variants_per_template = max_variants_per_template
        self.max_parameter_cardinality = max_parameter_cardinality
        self.max_total_generation_space = max_total_generation_space
        self.max_neighborhood_density = max_neighborhood_density
        self.max_family_branching_factor = max_family_branching_factor

    def estimate_template_size(
        self,
        entry_types: tuple,
        exit_types: tuple,
        filter_combos: tuple,
        holding_models: tuple,
        sizing_models: tuple,
        regime_filters: tuple,
        parameter_domains: Dict[str, list],
        template_id: str = "",
    ) -> BudgetReport:
        """Estimate generation size for a template.

        Returns BudgetReport. Does NOT raise — caller decides whether to reject.
        """
        structural = (
            len(entry_types) *
            len(exit_types) *
            max(len(filter_combos), 1) *
            len(holding_models) *
            len(sizing_models) *
            max(len(regime_filters), 1)
        )

        param_product = 1
        max_card = 0
        for values in parameter_domains.values():
            n = len(values)
            param_product *= n
            max_card = max(max_card, n)

        total = structural * param_product
        violations: List[str] = []

        if total > self.max_variants_per_template:
            violations.append(
                f"estimated_variants={total} > max_variants_per_template={self.max_variants_per_template}"
            )
        if max_card > self.max_parameter_cardinality:
            violations.append(
                f"max_parameter_cardinality={max_card} > limit={self.max_parameter_cardinality}"
            )

        return BudgetReport(
            template_id=template_id,
            estimated_variants=total,
            structural_combinations=structural,
            parameter_combinations=param_product,
            max_parameter_cardinality=max_card,
            within_budget=len(violations) == 0,
            violations=tuple(sorted(violations)),
        )

    def enforce_template(
        self,
        entry_types: tuple,
        exit_types: tuple,
        filter_combos: tuple,
        holding_models: tuple,
        sizing_models: tuple,
        regime_filters: tuple,
        parameter_domains: Dict[str, list],
        template_id: str = "",
    ) -> BudgetReport:
        """Estimate and raise BudgetExceededError if over budget."""
        report = self.estimate_template_size(
            entry_types, exit_types, filter_combos, holding_models,
            sizing_models, regime_filters, parameter_domains, template_id,
        )
        if not report.within_budget:
            raise BudgetExceededError(
                f"Template {template_id!r} exceeds budget:\n" +
                "\n".join(f"  - {v}" for v in report.violations)
            )
        return report

    def enforce_space_total(
        self,
        reports: List[BudgetReport],
    ) -> None:
        """Raise BudgetExceededError if sum of all template variants exceeds total limit."""
        total = sum(r.estimated_variants for r in reports)
        if total > self.max_total_generation_space:
            raise BudgetExceededError(
                f"Total generation space={total} exceeds max_total_generation_space="
                f"{self.max_total_generation_space}. "
                f"Reduce templates or parameter domains."
            )

    def check_neighborhood_density(self, neighbor_count: int, experiment_id: str = "") -> None:
        """Raise if a single point has too many neighbors."""
        if neighbor_count > self.max_neighborhood_density:
            raise BudgetExceededError(
                f"Experiment {experiment_id!r}: neighbor_count={neighbor_count} > "
                f"max_neighborhood_density={self.max_neighborhood_density}"
            )

    def check_family_branching(self, branch_count: int, family_id: str = "") -> None:
        """Raise if a family has too many sub-families."""
        if branch_count > self.max_family_branching_factor:
            raise BudgetExceededError(
                f"Family {family_id!r}: branch_count={branch_count} > "
                f"max_family_branching_factor={self.max_family_branching_factor}"
            )

    def to_dict(self) -> dict:
        return {
            "max_variants_per_template": self.max_variants_per_template,
            "max_parameter_cardinality": self.max_parameter_cardinality,
            "max_total_generation_space": self.max_total_generation_space,
            "max_neighborhood_density": self.max_neighborhood_density,
            "max_family_branching_factor": self.max_family_branching_factor,
        }
