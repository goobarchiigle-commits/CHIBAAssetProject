"""Tests: ecology.py — HypothesisEcologyTracker."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.ecology import (
    HypothesisEcologyTracker, EcologySnapshot,
    _herfindahl,
)
from tests.research.governance.conftest import (
    make_ecology, FAM_A, FAM_B, FAM_C, GH_A, GH_B, TS, TS2
)

TAXONOMY_TREND = "trend"
TAXONOMY_BREAKOUT = "breakout"


class TestHerfindahl:
    def test_empty_list(self):
        assert _herfindahl([]) == 0.0

    def test_single_label(self):
        assert _herfindahl(["a", "a", "a"]) == pytest.approx(1.0, abs=1e-9)

    def test_equal_two_labels(self):
        # HHI = 2 * (0.5)^2 = 0.5
        assert _herfindahl(["a", "b"]) == pytest.approx(0.5, abs=1e-9)

    def test_unequal_labels(self):
        # 3 a, 1 b → HHI = (3/4)^2 + (1/4)^2 = 9/16 + 1/16 = 10/16 = 0.625
        result = _herfindahl(["a", "a", "a", "b"])
        assert result == pytest.approx(0.625, abs=1e-4)


class TestHypothesisEcologyTracker:
    def test_record_family(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        assert FAM_A in eco.active_families()

    def test_record_family_idempotent(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS2)  # second registration
        assert len(eco.active_families()) == 1

    def test_record_extinction(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_extinction(FAM_A, "low_survivorship", timestamp=TS2)
        assert FAM_A not in eco.active_families()
        assert FAM_A in eco.extinct_families()

    def test_record_mutation(self):
        eco = make_ecology()
        eco.record_mutation(FAM_A, FAM_B, survived=True, timestamp=TS)
        assert eco.mutation_survivorship_rate() == pytest.approx(1.0, abs=1e-9)

    def test_mutation_survivorship_rate_mixed(self):
        eco = make_ecology()
        eco.record_mutation(FAM_A, FAM_B, survived=True, timestamp=TS)
        eco.record_mutation(FAM_A, FAM_C, survived=False, timestamp=TS)
        assert eco.mutation_survivorship_rate() == pytest.approx(0.5, abs=1e-9)

    def test_mutation_survivorship_rate_no_mutations(self):
        eco = make_ecology()
        assert eco.mutation_survivorship_rate() == 1.0

    def test_record_hypothesis(self):
        eco = make_ecology()
        eco.record_hypothesis("hyp_a", survived=True, timestamp=TS)
        eco.record_hypothesis("hyp_b", survived=False, timestamp=TS)
        # just checks it doesn't fail; state verified in snapshot

    def test_topology_concentration_single_grammar(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_family(FAM_B, TAXONOMY_TREND, GH_A, timestamp=TS)
        # All same grammar → HHI = 1.0
        assert eco.topology_concentration() == pytest.approx(1.0, abs=1e-9)

    def test_topology_concentration_two_grammars_equal(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_family(FAM_B, TAXONOMY_BREAKOUT, GH_B, timestamp=TS)
        # 50/50 → HHI = 0.5
        assert eco.topology_concentration() == pytest.approx(0.5, abs=1e-9)

    def test_topology_concentration_no_families(self):
        eco = make_ecology()
        assert eco.topology_concentration() == 0.0

    def test_regime_concentration_single_taxonomy(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_family(FAM_B, TAXONOMY_TREND, GH_B, timestamp=TS)
        assert eco.regime_concentration() == pytest.approx(1.0, abs=1e-9)

    def test_regime_concentration_two_taxonomies(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_family(FAM_B, TAXONOMY_BREAKOUT, GH_B, timestamp=TS)
        assert eco.regime_concentration() == pytest.approx(0.5, abs=1e-9)

    def test_exploration_saturation(self):
        eco = make_ecology()
        eco.record_hypothesis("h1", timestamp=TS)
        eco.record_hypothesis("h2", timestamp=TS)
        assert eco.exploration_saturation(10) == pytest.approx(0.2, abs=1e-9)

    def test_exploration_saturation_zero_space(self):
        eco = make_ecology()
        assert eco.exploration_saturation(0) == 0.0

    def test_exploration_saturation_capped_at_one(self):
        eco = make_ecology()
        for i in range(20):
            eco.record_hypothesis(f"h{i}", timestamp=TS)
        assert eco.exploration_saturation(10) == pytest.approx(1.0, abs=1e-9)

    def test_active_families_sorted(self):
        eco = make_ecology()
        eco.record_family(FAM_B, TAXONOMY_TREND, GH_B, timestamp=TS)
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        assert eco.active_families() == sorted(eco.active_families())

    def test_extinct_families_sorted(self):
        eco = make_ecology()
        eco.record_family(FAM_B, TAXONOMY_TREND, GH_B, timestamp=TS)
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_extinction(FAM_B, "gone", timestamp=TS2)
        eco.record_extinction(FAM_A, "gone", timestamp=TS2)
        assert eco.extinct_families() == sorted(eco.extinct_families())

    def test_take_snapshot_returns_immutable(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        snap = eco.take_snapshot(total_space_size=100, timestamp=TS)
        assert isinstance(snap, EcologySnapshot)
        with pytest.raises(Exception):
            snap.family_count = 99  # type: ignore

    def test_snapshot_id_prefix(self):
        eco = make_ecology()
        snap = eco.take_snapshot(timestamp=TS)
        assert snap.snapshot_id.startswith("eco_")

    def test_snapshot_values_consistent(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.record_family(FAM_B, TAXONOMY_BREAKOUT, GH_B, timestamp=TS)
        eco.record_extinction(FAM_B, "gone", timestamp=TS2)
        eco.record_mutation(FAM_A, FAM_C, survived=True, timestamp=TS)
        eco.record_hypothesis("h1", survived=True, timestamp=TS)
        snap = eco.take_snapshot(total_space_size=10, timestamp=TS2)
        assert snap.family_count == 2
        assert snap.active_family_count == 1
        assert snap.extinct_family_count == 1
        assert snap.mutation_survivorship_rate == pytest.approx(1.0, abs=1e-6)
        assert snap.total_hypotheses == 1
        assert snap.survived_hypotheses == 1

    def test_snapshot_checksum_nonempty(self):
        eco = make_ecology()
        snap = eco.take_snapshot(timestamp=TS)
        assert len(snap.checksum) > 0

    def test_snapshots_accumulate(self):
        eco = make_ecology()
        eco.take_snapshot(timestamp=TS)
        eco.take_snapshot(timestamp=TS2)
        assert len(eco.snapshots()) == 2

    def test_snapshot_deterministic(self):
        eco1 = make_ecology()
        eco1.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco2 = make_ecology()
        eco2.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        s1 = eco1.take_snapshot(total_space_size=10, timestamp=TS)
        s2 = eco2.take_snapshot(total_space_size=10, timestamp=TS)
        assert s1.checksum == s2.checksum
        assert s1.snapshot_id == s2.snapshot_id

    def test_to_summary_keys(self):
        eco = make_ecology()
        eco.record_family(FAM_A, TAXONOMY_TREND, GH_A, timestamp=TS)
        eco.take_snapshot(timestamp=TS)
        summary = eco.to_summary()
        assert "family_count" in summary
        assert "mutation_survivorship_rate" in summary
        assert "topology_concentration" in summary
        assert "snapshot_count" in summary
