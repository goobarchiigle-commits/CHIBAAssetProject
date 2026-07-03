"""Tests: family_graph.py — ExperimentFamilyGraph."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.family_graph import ExperimentFamilyGraph, _jaccard
from tests.research.analysis.conftest import make_record


class TestExperimentFamilyGraph:
    def test_empty_returns_empty(self):
        assert ExperimentFamilyGraph().build([]) == []

    def test_single_record_one_family(self):
        records = [make_record("exp_a")]
        families = ExperimentFamilyGraph().build(records)
        assert len(families) == 1
        assert "exp_a" in families[0].member_experiment_ids

    def test_same_strategy_merged(self):
        records = [
            make_record("e1", strategy_id="strat_x", feature_names=["f_a"]),
            make_record("e2", strategy_id="strat_x", feature_names=["f_b"]),
        ]
        families = ExperimentFamilyGraph().build(records)
        # Same strategy_id → single family
        assert len(families) == 1
        assert set(families[0].member_experiment_ids) == {"e1", "e2"}

    def test_different_strategies_separate_families(self):
        records = [
            make_record("e1", strategy_id="strat_a", feature_names=["f_a"]),
            make_record("e2", strategy_id="strat_b", feature_names=["f_b"]),
        ]
        families = ExperimentFamilyGraph().build(records)
        assert len(families) == 2

    def test_deterministic_across_record_order(self):
        records = [
            make_record("e1", strategy_id="strat_x", feature_names=["f_a", "f_b"]),
            make_record("e2", strategy_id="strat_x", feature_names=["f_a", "f_c"]),
            make_record("e3", strategy_id="strat_y", feature_names=["f_d"]),
        ]
        graph = ExperimentFamilyGraph()
        f1 = graph.build(records)
        f2 = graph.build(list(reversed(records)))
        assert sorted(fam.family_id for fam in f1) == sorted(fam.family_id for fam in f2)
        for fam_a in f1:
            match = next((f for f in f2 if f.family_id == fam_a.family_id), None)
            assert match is not None
            assert set(fam_a.member_experiment_ids) == set(match.member_experiment_ids)

    def test_family_id_stable_hash(self):
        records = [
            make_record("e1", strategy_id="strat_x"),
            make_record("e2", strategy_id="strat_x"),
        ]
        families = ExperimentFamilyGraph().build(records)
        fid = families[0].family_id
        assert fid.startswith("fam_")
        assert len(fid) == 16  # "fam_" + 12 hex chars

    def test_family_survivorship(self):
        records = [
            make_record("e1", strategy_id="s", survived=True),
            make_record("e2", strategy_id="s", survived=True),
            make_record("e3", strategy_id="s", survived=False),
        ]
        families = ExperimentFamilyGraph().build(records)
        assert len(families) == 1
        assert families[0].family_survivorship == pytest.approx(2 / 3, abs=1e-5)

    def test_robust_core_classification(self):
        # 3/3 survived with high score → should be robust
        records = [
            make_record(f"e{i}", strategy_id="s", survived=True, score=0.8)
            for i in range(3)
        ]
        families = ExperimentFamilyGraph().build(records)
        assert families[0].is_robust_core is True

    def test_non_robust_low_survivorship(self):
        records = [
            make_record(f"e{i}", strategy_id="s", survived=(i == 0), score=0.3)
            for i in range(5)
        ]
        families = ExperimentFamilyGraph().build(records)
        # 1/5 survivorship < 0.60 threshold
        assert families[0].is_robust_core is False

    def test_feature_signature_majority(self):
        records = [
            make_record("e1", strategy_id="s", feature_names=["f_common", "f_rare"]),
            make_record("e2", strategy_id="s", feature_names=["f_common"]),
            make_record("e3", strategy_id="s", feature_names=["f_common"]),
        ]
        families = ExperimentFamilyGraph(signature_majority=0.6).build(records)
        sig = families[0].feature_signature
        assert "f_common" in sig
        # f_rare appears in only 1/3 < 0.6 → not in signature
        assert "f_rare" not in sig

    def test_regime_affinity_positive_regimes_only(self):
        records = [
            make_record("e1", strategy_id="s", regime_scores={"trend_up": 1.0, "panic": -0.5}),
            make_record("e2", strategy_id="s", regime_scores={"trend_up": 0.8, "panic": -0.3}),
        ]
        families = ExperimentFamilyGraph().build(records)
        affinity = families[0].regime_affinity
        assert "trend_up" in affinity
        assert "panic" not in affinity

    def test_survivorship_distribution(self):
        records = [make_record(f"e{i}", strategy_id=f"s{i}", survived=(i % 2 == 0)) for i in range(6)]
        families = ExperimentFamilyGraph().build(records)
        dist = ExperimentFamilyGraph.survivorship_distribution(families)
        assert "p50" in dist
        assert "mean" in dist

    def test_survivorship_distribution_empty(self):
        assert ExperimentFamilyGraph.survivorship_distribution([]) == {}

    def test_family_correlation(self):
        from src.research.analysis.contract import FamilyRecord
        f1 = FamilyRecord(
            family_id="f1", strategy_id="s", member_experiment_ids=("e1", "e2"),
            feature_signature=(), regime_affinity=(), family_survivorship=0.5,
            family_persistence_score=0.5, is_robust_core=False,
        )
        f2 = FamilyRecord(
            family_id="f2", strategy_id="s", member_experiment_ids=("e2", "e3"),
            feature_signature=(), regime_affinity=(), family_survivorship=0.5,
            family_persistence_score=0.5, is_robust_core=False,
        )
        corr = ExperimentFamilyGraph.family_correlation(f1, f2)
        # Jaccard({e1,e2},{e2,e3}) = 1/3
        assert abs(corr - 1 / 3) < 1e-9

    def test_robust_cores_sorted_by_survivorship(self):
        records = [
            make_record("e1", strategy_id="a", survived=True, score=0.9),
            make_record("e2", strategy_id="b", survived=True, score=0.7),
            make_record("e3", strategy_id="c", survived=False, score=0.1),
        ]
        families = ExperimentFamilyGraph().build(records)
        cores = ExperimentFamilyGraph.robust_cores(families)
        if len(cores) > 1:
            survivorships = [f.family_survivorship for f in cores]
            assert survivorships == sorted(survivorships, reverse=True)


class TestJaccard:
    def test_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_both_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_partial_overlap(self):
        assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9
