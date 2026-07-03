"""Tests: generator.py — AlphaGenerator generation and manifest."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.generator import AlphaGenerator, GenerationManifest
from src.research.templates.budget import BudgetExceededError, SearchBudgetController
from src.research.templates.space import HypothesisSpace
from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    EXIT_TURTLE,
    FILTER_VOLATILITY,
    HOLD_SIGNAL_EXIT,
    SIZE_ATR_BASED,
)
from tests.research.templates.conftest import make_small_template


def _make_generator(**kwargs) -> AlphaGenerator:
    return AlphaGenerator(**kwargs)


def _make_space(space_id="gen_test") -> HypothesisSpace:
    return HypothesisSpace(space_id)


class TestAlphaGenerator:
    def test_generate_from_template_returns_manifest(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        manifest = gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        assert isinstance(manifest, GenerationManifest)

    def test_manifest_id_nonempty(self):
        gen = _make_generator()
        manifest = gen.generate_from_template(
            make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
        )
        assert len(manifest.manifest_id) > 0

    def test_accepted_equals_template_estimate(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        manifest = gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        assert manifest.accepted == tmpl.estimate_size()
        assert manifest.total_candidates == tmpl.estimate_size()

    def test_no_duplicates_on_fresh_space(self):
        gen = _make_generator()
        manifest = gen.generate_from_template(
            make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
        )
        assert manifest.rejected_duplicates == 0

    def test_all_duplicates_on_repeated_generation(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        m2 = gen.generate_from_template(tmpl, space, timestamp="2026-01-02T00:00:00+00:00")
        assert m2.rejected_duplicates == tmpl.estimate_size()
        assert m2.accepted == 0

    def test_hypothesis_ids_sorted(self):
        gen = _make_generator()
        manifest = gen.generate_from_template(
            make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
        )
        assert list(manifest.hypothesis_ids) == sorted(manifest.hypothesis_ids)

    def test_space_populated(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        assert len(space) == tmpl.estimate_size()

    def test_template_registered_in_space(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        assert tmpl.template_id in space.template_ids()

    def test_budget_exceeded_raises(self):
        gen = _make_generator(budget=SearchBudgetController(max_variants_per_template=1))
        with pytest.raises(BudgetExceededError):
            gen.generate_from_template(
                make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
            )

    def test_manifest_schema_version(self):
        from src.research.templates import SCHEMA_VERSION
        gen = _make_generator()
        manifest = gen.generate_from_template(
            make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
        )
        assert manifest.schema_version == SCHEMA_VERSION

    def test_manifest_template_id_matches(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        manifest = gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        assert manifest.template_id == tmpl.template_id

    def test_manifest_to_dict_keys(self):
        gen = _make_generator()
        manifest = gen.generate_from_template(
            make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
        )
        d = manifest.to_dict()
        assert "manifest_id" in d
        assert "total_candidates" in d
        assert "accepted" in d
        assert "rejected_duplicates" in d
        assert "budget_report" in d
        assert "hypothesis_ids" in d

    def test_generate_all_from_two_templates(self):
        gen = _make_generator()
        t1 = make_small_template(name="tmpl_a")
        t2 = make_small_template(
            name="tmpl_b",
            entry_types=(ENTRY_TURTLE_BREAKOUT,),
            parameter_domains={"entry_period": [5, 10], "exit_period": [20, 40]},
        )
        space = _make_space()
        manifests = gen.generate_all([t1, t2], space, timestamp="2026-01-01T00:00:00+00:00")
        assert len(manifests) == 2
        total_accepted = sum(m.accepted for m in manifests)
        assert total_accepted == len(space)

    def test_generate_all_space_budget_exceeded_raises(self):
        gen = _make_generator(budget=SearchBudgetController(max_total_generation_space=3))
        t1 = make_small_template(name="a")
        t2 = make_small_template(name="b")
        space = _make_space()
        with pytest.raises(BudgetExceededError):
            gen.generate_all([t1, t2], space, timestamp="2026-01-01T00:00:00+00:00")

    def test_generate_all_deterministic_order(self):
        gen = _make_generator()
        t1 = make_small_template(name="tmpl_a")
        t2 = make_small_template(
            name="tmpl_b",
            entry_types=(ENTRY_TURTLE_BREAKOUT,),
            parameter_domains={"entry_period": [5, 10], "exit_period": [20, 40]},
        )
        space1 = _make_space("s1")
        space2 = _make_space("s2")
        m1 = gen.generate_all([t1, t2], space1, timestamp="2026-01-01T00:00:00+00:00")
        m2 = gen.generate_all([t2, t1], space2, timestamp="2026-01-01T00:00:00+00:00")
        # Same hypotheses regardless of input order
        ids1 = sorted(hid for m in m1 for hid in m.hypothesis_ids)
        ids2 = sorted(hid for m in m2 for hid in m.hypothesis_ids)
        assert ids1 == ids2

    def test_detect_structural_duplicates_flag(self):
        # With two templates that produce same structural families
        gen = _make_generator(detect_structural_duplicates=True)
        t1 = make_small_template(name="orig")
        space = _make_space()
        manifest = gen.generate_from_template(t1, space, timestamp="2026-01-01T00:00:00+00:00")
        assert isinstance(manifest.structural_duplicate_pairs, tuple)

    def test_no_structural_dup_detection_when_disabled(self):
        gen = _make_generator(detect_structural_duplicates=False)
        tmpl = make_small_template()
        space = _make_space()
        manifest = gen.generate_from_template(tmpl, space, timestamp="2026-01-01T00:00:00+00:00")
        assert manifest.structural_duplicate_pairs == ()

    def test_budget_report_in_manifest(self):
        from src.research.templates.budget import BudgetReport
        gen = _make_generator()
        manifest = gen.generate_from_template(
            make_small_template(), _make_space(), timestamp="2026-01-01T00:00:00+00:00"
        )
        assert isinstance(manifest.budget_report, BudgetReport)
        assert manifest.budget_report.within_budget

    def test_parent_id_propagated(self):
        gen = _make_generator()
        tmpl = make_small_template()
        space = _make_space()
        manifest = gen.generate_from_template(
            tmpl, space, timestamp="2026-01-01T00:00:00+00:00", parent_id="hyp_parent_xyz"
        )
        for hid in manifest.hypothesis_ids:
            spec = space.get_hypothesis(hid)
            assert spec.parent_id == "hyp_parent_xyz"
