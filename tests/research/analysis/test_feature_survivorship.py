"""Tests: feature_survivorship.py — FeatureSurvivorshipAnalyzer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.feature_survivorship import (
    FeatureObservation,
    FeatureSurvivorshipAnalyzer,
)


def _obs(exp_id, features, survived=True, regime="trend_up", family="fam_a"):
    return FeatureObservation(
        experiment_id=exp_id,
        feature_names=list(features),
        survived=survived,
        regime=regime,
        family_id=family,
    )


class TestFeatureSurvivorshipAnalyzer:
    def test_empty_returns_empty(self):
        assert FeatureSurvivorshipAnalyzer().analyze([]) == {}

    def test_basic_survival_rate(self):
        obs = [
            _obs("e1", ["f_mom", "f_vol"], survived=True),
            _obs("e2", ["f_mom"], survived=False),
            _obs("e3", ["f_vol"], survived=True),
        ]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        assert stats["f_mom"].total_occurrences == 2
        assert stats["f_mom"].survived_occurrences == 1
        assert abs(stats["f_mom"].feature_survival_rate - 0.5) < 1e-9
        assert stats["f_vol"].feature_survival_rate == 1.0

    def test_deterministic_across_observation_order(self):
        obs = [
            _obs("e1", ["f_a", "f_b"], survived=True, regime="trend_up", family="fam_1"),
            _obs("e2", ["f_a"], survived=False, regime="range", family="fam_1"),
            _obs("e3", ["f_b", "f_c"], survived=True, regime="trend_up", family="fam_2"),
            _obs("e4", ["f_c"], survived=False, regime="panic", family="fam_2"),
        ]
        ana = FeatureSurvivorshipAnalyzer()
        s1 = ana.analyze(obs)
        s2 = ana.analyze(list(reversed(obs)))
        for feat in s1:
            assert s1[feat].feature_survival_rate == s2[feat].feature_survival_rate
            assert s1[feat].feature_family_frequency == s2[feat].feature_family_frequency
            assert s1[feat].feature_regime_affinity == s2[feat].feature_regime_affinity

    def test_regime_affinity_keys_sorted(self):
        obs = [
            _obs("e1", ["f_x"], survived=True, regime="trend_up"),
            _obs("e2", ["f_x"], survived=False, regime="panic"),
            _obs("e3", ["f_x"], survived=True, regime="range"),
        ]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        affinity = stats["f_x"].feature_regime_affinity
        regimes = [pair[0] for pair in affinity]
        assert regimes == sorted(regimes)

    def test_feature_family_frequency(self):
        obs = [
            _obs("e1", ["f_a"], family="fam_1"),
            _obs("e2", ["f_a"], family="fam_2"),
            _obs("e3", ["f_a"], family="fam_1"),  # same family, shouldn't double-count
        ]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        assert stats["f_a"].feature_family_frequency == 2  # two distinct families

    def test_decay_resistance_stable_feature(self):
        # Feature survives equally in early and late experiments
        obs = [_obs(f"e{i}", ["f_stable"], survived=True) for i in range(10)]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        assert stats["f_stable"].feature_decay_resistance == 1.0

    def test_decay_resistance_declining_feature(self):
        # Survives early, fails late → decay_resistance < 1
        early = [_obs(f"e{i}", ["f_decay"], survived=True) for i in range(5)]
        late = [_obs(f"e{i+5}", ["f_decay"], survived=False) for i in range(5)]
        stats = FeatureSurvivorshipAnalyzer().analyze(early + late)
        assert stats["f_decay"].feature_decay_resistance == 0.0

    def test_rank_by_survival_rate_stable_order(self):
        obs = [
            _obs("e1", ["f_high"], survived=True),
            _obs("e2", ["f_high"], survived=True),
            _obs("e3", ["f_low"], survived=False),
            _obs("e4", ["f_low"], survived=False),
        ]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        ranked = FeatureSurvivorshipAnalyzer.rank_by_survival_rate(stats)
        assert ranked[0].feature_name == "f_high"
        assert ranked[-1].feature_name == "f_low"

    def test_duplicate_experiment_id_raises(self):
        obs = [
            _obs("dup", ["f_x"]),
            _obs("dup", ["f_y"]),
        ]
        with pytest.raises(ValueError, match="Duplicate experiment_id"):
            FeatureSurvivorshipAnalyzer().analyze(obs)

    def test_persistence_score_bounded(self):
        obs = [_obs(f"e{i}", ["f_x"], family=f"fam_{i}") for i in range(5)]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        assert 0.0 <= stats["f_x"].feature_persistence_score <= 1.0

    def test_all_stats_fields_present(self):
        obs = [_obs("e1", ["f_a"], survived=True, regime="range", family="fam_1")]
        stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        s = stats["f_a"]
        assert s.total_occurrences == 1
        assert s.survived_occurrences == 1
        assert s.feature_survival_rate == 1.0
        assert s.feature_regime_affinity == (("range", 1.0),)
