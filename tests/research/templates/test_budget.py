"""Tests: budget.py — SearchBudgetController and BudgetReport."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.budget import (
    BudgetExceededError,
    BudgetReport,
    SearchBudgetController,
)
from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    EXIT_TURTLE,
    FILTER_VOLATILITY,
    HOLD_SIGNAL_EXIT,
    SIZE_ATR_BASED,
)


def _make_controller(**kwargs) -> SearchBudgetController:
    defaults = dict(
        max_variants_per_template=500,
        max_parameter_cardinality=10,
        max_total_generation_space=5000,
        max_neighborhood_density=20,
        max_family_branching_factor=50,
    )
    defaults.update(kwargs)
    return SearchBudgetController(**defaults)


def _small_estimate(ctrl: SearchBudgetController, **overrides) -> BudgetReport:
    kwargs = dict(
        entry_types=(ENTRY_MOMENTUM_BREAKOUT,),
        exit_types=(EXIT_TURTLE,),
        filter_combos=((), (FILTER_VOLATILITY,)),
        holding_models=(HOLD_SIGNAL_EXIT,),
        sizing_models=(SIZE_ATR_BASED,),
        regime_filters=(None,),
        parameter_domains={"entry_period": [10, 20], "exit_period": [20, 40]},
        template_id="test_tmpl",
    )
    kwargs.update(overrides)
    return ctrl.estimate_template_size(**kwargs)


class TestSearchBudgetController:
    def test_invalid_init_zero_variants(self):
        with pytest.raises(ValueError):
            SearchBudgetController(max_variants_per_template=0)

    def test_invalid_init_zero_cardinality(self):
        with pytest.raises(ValueError):
            SearchBudgetController(max_parameter_cardinality=0)

    def test_small_template_within_budget(self):
        ctrl = _make_controller()
        report = _small_estimate(ctrl)
        # 1 entry × 1 exit × 2 filter_combos × 1 hold × 1 size × 1 regime × 2×2 params = 8
        assert report.within_budget
        assert report.estimated_variants == 8

    def test_structural_combinations_correct(self):
        ctrl = _make_controller()
        report = _small_estimate(ctrl)
        # 1×1×2×1×1×1 = 2
        assert report.structural_combinations == 2

    def test_parameter_combinations_correct(self):
        ctrl = _make_controller()
        report = _small_estimate(ctrl)
        # 2×2 = 4
        assert report.parameter_combinations == 4

    def test_max_parameter_cardinality_correct(self):
        ctrl = _make_controller()
        report = _small_estimate(ctrl)
        assert report.max_parameter_cardinality == 2

    def test_exceed_variants_per_template(self):
        ctrl = _make_controller(max_variants_per_template=5)
        report = _small_estimate(ctrl)
        assert not report.within_budget
        assert any("max_variants_per_template" in v for v in report.violations)

    def test_exceed_cardinality(self):
        ctrl = _make_controller(max_parameter_cardinality=1)
        report = _small_estimate(ctrl)
        assert not report.within_budget
        assert any("max_parameter_cardinality" in v for v in report.violations)

    def test_violations_sorted(self):
        ctrl = _make_controller(max_variants_per_template=1, max_parameter_cardinality=1)
        report = _small_estimate(ctrl)
        assert list(report.violations) == sorted(report.violations)

    def test_enforce_template_within_budget(self):
        ctrl = _make_controller()
        report = ctrl.enforce_template(
            entry_types=(ENTRY_MOMENTUM_BREAKOUT,),
            exit_types=(EXIT_TURTLE,),
            filter_combos=((),),
            holding_models=(HOLD_SIGNAL_EXIT,),
            sizing_models=(SIZE_ATR_BASED,),
            regime_filters=(None,),
            parameter_domains={"entry_period": [10, 20]},
        )
        assert report.within_budget

    def test_enforce_template_raises_on_over_budget(self):
        ctrl = _make_controller(max_variants_per_template=1)
        with pytest.raises(BudgetExceededError):
            ctrl.enforce_template(
                entry_types=(ENTRY_MOMENTUM_BREAKOUT,),
                exit_types=(EXIT_TURTLE,),
                filter_combos=((),),
                holding_models=(HOLD_SIGNAL_EXIT,),
                sizing_models=(SIZE_ATR_BASED,),
                regime_filters=(None,),
                parameter_domains={"entry_period": [10, 20, 30]},
            )

    def test_enforce_space_total_within(self):
        ctrl = _make_controller(max_total_generation_space=100)
        r1 = _small_estimate(ctrl)  # 8 variants
        r2 = _small_estimate(ctrl)  # 8 variants
        ctrl.enforce_space_total([r1, r2])  # 16 total, fine

    def test_enforce_space_total_raises(self):
        ctrl = _make_controller(max_total_generation_space=5)
        r1 = _small_estimate(ctrl)  # 8 variants
        with pytest.raises(BudgetExceededError):
            ctrl.enforce_space_total([r1])

    def test_check_neighborhood_density_ok(self):
        ctrl = _make_controller(max_neighborhood_density=20)
        ctrl.check_neighborhood_density(10)  # should not raise

    def test_check_neighborhood_density_raises(self):
        ctrl = _make_controller(max_neighborhood_density=10)
        with pytest.raises(BudgetExceededError):
            ctrl.check_neighborhood_density(15, "exp_a")

    def test_check_family_branching_ok(self):
        ctrl = _make_controller(max_family_branching_factor=50)
        ctrl.check_family_branching(10)  # should not raise

    def test_check_family_branching_raises(self):
        ctrl = _make_controller(max_family_branching_factor=5)
        with pytest.raises(BudgetExceededError):
            ctrl.check_family_branching(6, "fam_xyz")

    def test_template_id_propagated_to_report(self):
        ctrl = _make_controller()
        report = _small_estimate(ctrl, template_id="my_template")
        assert report.template_id == "my_template"

    def test_budget_report_frozen(self):
        ctrl = _make_controller()
        report = _small_estimate(ctrl)
        with pytest.raises(Exception):
            report.within_budget = False  # type: ignore

    def test_budget_report_to_dict(self):
        ctrl = _make_controller()
        d = _small_estimate(ctrl).to_dict()
        assert "template_id" in d
        assert "estimated_variants" in d
        assert "within_budget" in d
        assert "violations" in d

    def test_to_dict(self):
        ctrl = _make_controller()
        d = ctrl.to_dict()
        assert d["max_variants_per_template"] == 500
        assert d["max_total_generation_space"] == 5000

    def test_no_filter_combos_structural_counts_one(self):
        ctrl = _make_controller()
        report = ctrl.estimate_template_size(
            entry_types=(ENTRY_MOMENTUM_BREAKOUT,),
            exit_types=(EXIT_TURTLE,),
            filter_combos=(),
            holding_models=(HOLD_SIGNAL_EXIT,),
            sizing_models=(SIZE_ATR_BASED,),
            regime_filters=(None,),
            parameter_domains={"entry_period": [10, 20]},
        )
        # max(len(filter_combos), 1) = 1
        assert report.structural_combinations == 1
