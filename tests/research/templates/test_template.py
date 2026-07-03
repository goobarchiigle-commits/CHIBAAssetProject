"""Tests: template.py — HypothesisSpec and HypothesisTemplate."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.template import HypothesisSpec, HypothesisTemplate
from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    EXIT_TURTLE,
    EXIT_MOMENTUM,
    FILTER_VOLATILITY,
    FILTER_LIQUIDITY,
    HOLD_SIGNAL_EXIT,
    HOLD_FIXED_DAYS,
    SIZE_ATR_BASED,
    SIZE_FIXED,
)
from src.research.templates.taxonomy import TAXONOMY_TREND

from tests.research.templates.conftest import make_small_template


class TestHypothesisSpec:
    def _make_spec(self) -> HypothesisSpec:
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        return specs[0]

    def test_spec_frozen(self):
        spec = self._make_spec()
        with pytest.raises(Exception):
            spec.entry_type = "CHANGED"  # type: ignore

    def test_hypothesis_id_prefix(self):
        spec = self._make_spec()
        assert spec.hypothesis_id.startswith("hyp_")

    def test_template_id_prefix(self):
        spec = self._make_spec()
        assert spec.template_id.startswith("tmpl_")

    def test_family_id_prefix(self):
        spec = self._make_spec()
        assert spec.family_id.startswith("fam_")

    def test_parameter_hash_length(self):
        spec = self._make_spec()
        assert len(spec.parameter_hash) == 16

    def test_grammar_hash_length(self):
        spec = self._make_spec()
        assert len(spec.grammar_hash) == 16

    def test_filter_types_sorted(self):
        tmpl = make_small_template(filter_combos=((FILTER_VOLATILITY, FILTER_LIQUIDITY),))
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        for spec in specs:
            assert list(spec.filter_types) == sorted(spec.filter_types)

    def test_parameters_sorted(self):
        spec = self._make_spec()
        keys = [k for k, _ in spec.parameters]
        assert keys == sorted(keys)

    def test_taxonomy_class_nonempty(self):
        spec = self._make_spec()
        assert isinstance(spec.taxonomy_class, str)
        assert len(spec.taxonomy_class) > 0

    def test_momentum_entry_classified_trend(self):
        tmpl = make_small_template(entry_types=(ENTRY_MOMENTUM_BREAKOUT,))
        spec = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")[0]
        assert spec.taxonomy_class == TAXONOMY_TREND

    def test_to_dict_from_dict_roundtrip(self):
        spec = self._make_spec()
        restored = HypothesisSpec.from_dict(spec.to_dict())
        assert restored.hypothesis_id == spec.hypothesis_id
        assert restored.entry_type == spec.entry_type
        assert restored.parameters == spec.parameters

    def test_parameter_dict(self):
        spec = self._make_spec()
        pd = spec.parameter_dict()
        assert isinstance(pd, dict)
        for k, v_str in spec.parameters:
            assert k in pd


class TestHypothesisTemplate:
    def test_template_id_prefix(self):
        tmpl = make_small_template()
        assert tmpl.template_id.startswith("tmpl_")

    def test_template_id_deterministic(self):
        t1 = make_small_template()
        t2 = make_small_template()
        assert t1.template_id == t2.template_id

    def test_template_id_changes_with_name(self):
        t1 = make_small_template(name="alpha")
        t2 = make_small_template(name="beta")
        assert t1.template_id != t2.template_id

    def test_entry_types_sorted(self):
        tmpl = make_small_template(entry_types=(ENTRY_TURTLE_BREAKOUT, ENTRY_MOMENTUM_BREAKOUT))
        assert list(tmpl.entry_types) == sorted(tmpl.entry_types)

    def test_invalid_entry_type_raises(self):
        with pytest.raises(ValueError):
            HypothesisTemplate(
                name="bad",
                entry_types=("COMPLETELY_INVALID",),
                exit_types=(EXIT_TURTLE,),
                filter_combos=((),),
                holding_models=(HOLD_SIGNAL_EXIT,),
                sizing_models=(SIZE_ATR_BASED,),
                regime_filters=(None,),
                parameter_domains={"entry_period": [10, 20]},
            )

    def test_constraint_violation_raises(self):
        with pytest.raises(ValueError):
            HypothesisTemplate(
                name="bad_domains",
                entry_types=(ENTRY_MOMENTUM_BREAKOUT,),
                exit_types=(EXIT_TURTLE,),
                filter_combos=((),),
                holding_models=(HOLD_SIGNAL_EXIT,),
                sizing_models=(SIZE_ATR_BASED,),
                regime_filters=(None,),
                parameter_domains={},  # empty domains → constraint violation
            )

    def test_estimate_size_correct(self):
        tmpl = make_small_template(
            entry_types=(ENTRY_MOMENTUM_BREAKOUT,),
            exit_types=(EXIT_TURTLE,),
            filter_combos=((), (FILTER_VOLATILITY,)),
            holding_models=(HOLD_SIGNAL_EXIT,),
            sizing_models=(SIZE_ATR_BASED,),
            regime_filters=(None,),
            parameter_domains={"entry_period": [10, 20], "exit_period": [20, 40]},
        )
        # 1×1×2×1×1×1×4 = 8
        assert tmpl.estimate_size() == 8

    def test_generate_specs_count_matches_estimate(self):
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        assert len(specs) == tmpl.estimate_size()

    def test_generate_specs_sorted_by_id(self):
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        ids = [s.hypothesis_id for s in specs]
        assert ids == sorted(ids)

    def test_generate_specs_deterministic(self):
        tmpl = make_small_template()
        s1 = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        s2 = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        assert [s.hypothesis_id for s in s1] == [s.hypothesis_id for s in s2]

    def test_generate_specs_ids_unique(self):
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        ids = [s.hypothesis_id for s in specs]
        assert len(ids) == len(set(ids))

    def test_generate_specs_all_from_same_template(self):
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        for s in specs:
            assert s.template_id == tmpl.template_id

    def test_generate_specs_with_parent_id(self):
        tmpl = make_small_template()
        specs = tmpl.generate_specs(
            timestamp="2026-01-01T00:00:00+00:00", parent_id="hyp_parent"
        )
        for s in specs:
            assert s.parent_id == "hyp_parent"

    def test_multiple_entry_types_produce_different_families(self):
        tmpl = make_small_template(
            entry_types=(ENTRY_MOMENTUM_BREAKOUT, ENTRY_TURTLE_BREAKOUT),
            exit_types=(EXIT_TURTLE,),
            filter_combos=((),),
            holding_models=(HOLD_SIGNAL_EXIT,),
            sizing_models=(SIZE_ATR_BASED,),
            regime_filters=(None,),
            parameter_domains={"entry_period": [10, 20]},
        )
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        family_ids = {s.family_id for s in specs}
        assert len(family_ids) == 2  # one per entry type

    def test_to_dict_has_required_keys(self):
        tmpl = make_small_template()
        d = tmpl.to_dict()
        assert "template_id" in d
        assert "name" in d
        assert "entry_types" in d
        assert "estimated_size" in d
        assert "schema_version" in d

    def test_regime_filter_none_produces_specs(self):
        tmpl = make_small_template(regime_filters=(None,))
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        assert all(s.regime_filter is None for s in specs)
