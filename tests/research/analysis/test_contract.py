"""Tests: contract.py — SurvivorRecord, TopologyResult, immutability, hashing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.contract import (
    SCHEMA_VERSION,
    FamilyRecord,
    FeatureSurvivalStats,
    RegimeResilienceResult,
    SurvivorRecord,
    SurvivorshipLedgerEntry,
    TopologyResult,
)
from tests.research.analysis.conftest import make_record


class TestSurvivorRecord:
    def test_build_sorts_fields(self):
        r = make_record(
            feature_names=["f_vol", "f_mom"],
            regime_scores={"range": 0.1, "trend_up": 1.0},
            rejection_reasons=["z_reason", "a_reason"],
            fold_ids=[2, 0, 1],
        )
        assert r.feature_names == ("f_mom", "f_vol")
        assert r.regime_score_pairs[0][0] == "range"
        assert r.rejection_reasons == ("a_reason", "z_reason")
        assert r.fold_ids == (0, 1, 2)

    def test_frozen_immutability(self):
        r = make_record()
        with pytest.raises(Exception):
            r.score = 9.9  # type: ignore

    def test_canonical_hash_deterministic(self):
        r1 = make_record()
        r2 = make_record()
        assert r1.canonical_hash() == r2.canonical_hash()

    def test_canonical_hash_differs_on_different_data(self):
        r1 = make_record(exp_id="exp_a", score=0.5)
        r2 = make_record(exp_id="exp_b", score=0.8)
        assert r1.canonical_hash() != r2.canonical_hash()

    def test_regime_scores_dict(self):
        r = make_record(regime_scores={"trend_up": 1.2, "range": 0.3})
        d = r.regime_scores()
        assert d["trend_up"] == 1.2
        assert d["range"] == 0.3

    def test_to_dict_from_dict_roundtrip(self):
        r = make_record(
            feature_names=["f_a", "f_b"],
            regime_scores={"trend_up": 1.0, "panic": -0.5},
            fold_ids=[0, 1, 2],
        )
        restored = SurvivorRecord.from_dict(r.to_dict())
        assert restored == r

    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "1.0.0"


class TestTopologyResult:
    def test_to_dict_keys(self):
        t = TopologyResult(
            experiment_id="exp_x",
            topology_fragility_score=0.3,
            plateau_width=0.7,
            neighbor_decay_rate=0.1,
            parameter_elasticity=0.5,
            surface_entropy=1.2,
            neighborhood_survivorship=0.8,
            neighbor_count=4,
        )
        d = t.to_dict()
        assert set(d.keys()) == {
            "experiment_id", "topology_fragility_score", "plateau_width",
            "neighbor_decay_rate", "parameter_elasticity", "surface_entropy",
            "neighborhood_survivorship", "neighbor_count",
        }

    def test_frozen(self):
        t = TopologyResult(
            experiment_id="x", topology_fragility_score=0.5,
            plateau_width=0.5, neighbor_decay_rate=0.1,
            parameter_elasticity=0.1, surface_entropy=0.5,
            neighborhood_survivorship=0.5, neighbor_count=3,
        )
        with pytest.raises(Exception):
            t.neighbor_count = 99  # type: ignore
