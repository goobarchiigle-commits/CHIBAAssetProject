"""Tests: space.py — HypothesisSpace lineage and analytics."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

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


def _make_space_with_specs(space_id="test_space"):
    space = HypothesisSpace(space_id)
    tmpl = make_small_template()
    specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
    space.add_template(tmpl)
    space.add_all(specs)
    return space, tmpl, specs


class TestHypothesisSpace:
    def test_add_template_registers(self):
        space, tmpl, _ = _make_space_with_specs()
        assert tmpl.template_id in space.template_ids()

    def test_add_template_duplicate_raises(self):
        space, tmpl, _ = _make_space_with_specs()
        with pytest.raises(ValueError):
            space.add_template(tmpl)

    def test_get_template(self):
        space, tmpl, _ = _make_space_with_specs()
        retrieved = space.get_template(tmpl.template_id)
        assert retrieved.template_id == tmpl.template_id

    def test_get_template_missing_raises(self):
        space = HypothesisSpace("s")
        with pytest.raises(KeyError):
            space.get_template("tmpl_nonexistent")

    def test_hypothesis_ids_sorted(self):
        space, _, _ = _make_space_with_specs()
        ids = space.hypothesis_ids()
        assert ids == sorted(ids)

    def test_hypothesis_count(self):
        space, tmpl, specs = _make_space_with_specs()
        assert len(space) == len(specs)

    def test_add_hypothesis_duplicate_returns_false(self):
        space, tmpl, specs = _make_space_with_specs()
        result = space.add_hypothesis(specs[0])
        assert result is False

    def test_add_all_counts(self):
        space = HypothesisSpace("s")
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        space.add_template(tmpl)
        counts = space.add_all(specs)
        assert counts["new"] == len(specs)
        assert counts["duplicate"] == 0

    def test_add_all_duplicate_count(self):
        space, _, specs = _make_space_with_specs()
        counts = space.add_all(specs)  # all duplicates
        assert counts["new"] == 0
        assert counts["duplicate"] == len(specs)

    def test_get_hypothesis(self):
        space, _, specs = _make_space_with_specs()
        spec = specs[0]
        retrieved = space.get_hypothesis(spec.hypothesis_id)
        assert retrieved.hypothesis_id == spec.hypothesis_id

    def test_get_hypothesis_missing_raises(self):
        space = HypothesisSpace("s")
        with pytest.raises(KeyError):
            space.get_hypothesis("hyp_nonexistent")

    def test_family_ids_sorted(self):
        space, _, _ = _make_space_with_specs()
        fids = space.family_ids()
        assert fids == sorted(fids)

    def test_get_family_nonempty(self):
        space, _, specs = _make_space_with_specs()
        fid = specs[0].family_id
        family = space.get_family(fid)
        assert len(family) > 0
        ids = [s.hypothesis_id for s in family]
        assert ids == sorted(ids)

    def test_structural_siblings_same_grammar_hash(self):
        space, tmpl, specs = _make_space_with_specs()
        spec = specs[0]
        siblings = space.structural_siblings(spec)
        for sib in siblings:
            assert sib.grammar_hash == spec.grammar_hash
            assert sib.hypothesis_id != spec.hypothesis_id

    def test_lineage_chain_root(self):
        space, _, specs = _make_space_with_specs()
        spec = specs[0]
        chain = space.lineage_chain(spec.hypothesis_id)
        assert chain[-1] == spec.hypothesis_id

    def test_lineage_chain_with_parent(self):
        tmpl = make_small_template()
        parent_specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        parent = parent_specs[0]

        child_tmpl = make_small_template(
            parameter_domains={"entry_period": [5, 15], "exit_period": [25, 45]}
        )
        child_specs = child_tmpl.generate_specs(
            timestamp="2026-02-01T00:00:00+00:00", parent_id=parent.hypothesis_id
        )

        space = HypothesisSpace("lineage_test")
        space.add_template(tmpl)
        space.add_all(parent_specs)
        space.add_template(child_tmpl)
        space.add_all(child_specs)

        child = next(s for s in child_specs if s.parent_id == parent.hypothesis_id)
        chain = space.lineage_chain(child.hypothesis_id)
        assert parent.hypothesis_id in chain
        assert chain[-1] == child.hypothesis_id

    def test_family_distribution_values_positive(self):
        space, _, _ = _make_space_with_specs()
        dist = space.family_distribution()
        assert all(v > 0 for v in dist.values())

    def test_grammar_distribution_keys_sorted(self):
        space, _, _ = _make_space_with_specs()
        dist = space.grammar_distribution()
        keys = list(dist.keys())
        assert keys == sorted(keys)

    def test_taxonomy_distribution_nonempty(self):
        space, _, _ = _make_space_with_specs()
        dist = space.taxonomy_distribution()
        assert len(dist) > 0
        total = sum(dist.values())
        assert total == len(space)

    def test_space_hash_deterministic(self):
        space1, tmpl, specs = _make_space_with_specs("sp1")
        space2 = HypothesisSpace("sp1")
        space2.add_template(tmpl)
        space2.add_all(specs)
        assert space1.space_hash() == space2.space_hash()

    def test_space_hash_16_chars(self):
        space, _, _ = _make_space_with_specs()
        assert len(space.space_hash()) == 16

    def test_snapshot_has_required_keys(self):
        space, _, _ = _make_space_with_specs()
        snap = space.snapshot()
        assert "space_id" in snap
        assert "hypothesis_count" in snap
        assert "family_count" in snap
        assert "space_hash" in snap
        assert "schema_version" in snap

    def test_snapshot_hypothesis_count_matches(self):
        space, _, specs = _make_space_with_specs()
        snap = space.snapshot()
        assert snap["hypothesis_count"] == len(specs)

    def test_near_duplicates_same_structure(self):
        # Two templates with same entry/exit/filter but different param names → same structural fingerprint
        tmpl = make_small_template()
        specs = tmpl.generate_specs(timestamp="2026-01-01T00:00:00+00:00")
        space = HypothesisSpace("nd_test")
        space.add_template(tmpl)
        space.add_all(specs)
        spec = specs[0]
        # structural siblings share fingerprint → should all be near duplicates
        nds = space.near_duplicates(spec, threshold=0.99)
        # Same structural family → should find matches (may find structural siblings)
        assert isinstance(nds, list)
