"""Tests: topology.py — ParameterTopologyAnalyzer determinism and correctness."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.topology import (
    ParameterTopologyAnalyzer,
    _fragility,
    _plateau_width,
    _surface_entropy,
)


def _make_experiments():
    """3x2 grid on (entry_period, exit_period). Scores vary deterministically."""
    return [
        ("exp_1", {"entry_period": 10, "exit_period": 20}, 0.80, True),
        ("exp_2", {"entry_period": 10, "exit_period": 40}, 0.50, True),
        ("exp_3", {"entry_period": 20, "exit_period": 20}, 0.40, False),
        ("exp_4", {"entry_period": 20, "exit_period": 40}, 0.45, True),
        ("exp_5", {"entry_period": 30, "exit_period": 20}, 0.30, False),
        ("exp_6", {"entry_period": 30, "exit_period": 40}, 0.35, False),
    ]


class TestParameterTopologyAnalyzer:
    def test_analyze_empty(self):
        ana = ParameterTopologyAnalyzer()
        assert ana.analyze([]) == {}

    def test_analyze_returns_all_ids(self):
        exps = _make_experiments()
        results = ParameterTopologyAnalyzer().analyze(exps)
        assert set(results.keys()) == {e[0] for e in exps}

    def test_deterministic_across_input_orders(self):
        exps = _make_experiments()
        ana = ParameterTopologyAnalyzer()
        r1 = ana.analyze(exps)
        r2 = ana.analyze(list(reversed(exps)))
        for eid in r1:
            assert r1[eid].topology_fragility_score == r2[eid].topology_fragility_score
            assert r1[eid].plateau_width == r2[eid].plateau_width
            assert r1[eid].neighbor_count == r2[eid].neighbor_count

    def test_isolated_experiment_fragility_one(self):
        # Single experiment → no neighbors → fragility=1
        exps = [("exp_solo", {"p": 5}, 0.8, True)]
        results = ParameterTopologyAnalyzer().analyze(exps)
        r = results["exp_solo"]
        assert r.topology_fragility_score == 1.0
        assert r.neighbor_count == 0
        assert r.plateau_width == 0.0

    def test_stable_plateau_low_fragility(self):
        # All experiments have identical scores → perfect plateau
        exps = [
            ("exp_a", {"p": 1}, 0.5, True),
            ("exp_b", {"p": 2}, 0.5, True),
            ("exp_c", {"p": 3}, 0.5, True),
        ]
        results = ParameterTopologyAnalyzer().analyze(exps)
        # Middle experiment has neighbors on both sides with same score
        r = results["exp_b"]
        assert r.topology_fragility_score == 0.0
        assert r.plateau_width == 1.0

    def test_spike_experiment_high_fragility(self):
        # exp_spike has much higher score than neighbors
        exps = [
            ("exp_left", {"p": 1}, 0.1, False),
            ("exp_spike", {"p": 2}, 5.0, True),
            ("exp_right", {"p": 3}, 0.1, False),
        ]
        results = ParameterTopologyAnalyzer().analyze(exps)
        r = results["exp_spike"]
        assert r.topology_fragility_score > 0.8
        assert r.neighborhood_survivorship == 0.0  # both neighbors failed

    def test_neighborhood_survivorship_correct(self):
        exps = [
            ("exp_a", {"p": 1}, 0.5, True),
            ("exp_b", {"p": 2}, 0.4, False),
            ("exp_c", {"p": 3}, 0.6, True),
        ]
        results = ParameterTopologyAnalyzer().analyze(exps)
        # exp_b has neighbors exp_a (survived) and exp_c (survived)
        r = results["exp_b"]
        assert r.neighborhood_survivorship == 1.0

    def test_parameter_elasticity_zero_for_uniform_score(self):
        # Same score everywhere → elasticity = 0
        exps = [
            ("exp_a", {"p": 1, "q": 1}, 0.5, True),
            ("exp_b", {"p": 1, "q": 2}, 0.5, True),
            ("exp_c", {"p": 2, "q": 1}, 0.5, True),
            ("exp_d", {"p": 2, "q": 2}, 0.5, True),
        ]
        results = ParameterTopologyAnalyzer().analyze(exps)
        for r in results.values():
            assert r.parameter_elasticity == 0.0

    def test_duplicate_experiment_id_raises(self):
        exps = [
            ("exp_a", {"p": 1}, 0.5, True),
            ("exp_a", {"p": 2}, 0.6, False),
        ]
        with pytest.raises(ValueError, match="Duplicate experiment_id"):
            ParameterTopologyAnalyzer().analyze(exps)

    def test_invalid_linf_radius_raises(self):
        with pytest.raises(ValueError):
            ParameterTopologyAnalyzer(linf_radius=0)

    def test_non_numeric_params_ignored(self):
        exps = [
            ("exp_a", {"p": 1, "mode": "fast"}, 0.5, True),
            ("exp_b", {"p": 2, "mode": "fast"}, 0.4, True),
        ]
        # Should not crash; non-numeric params are skipped
        results = ParameterTopologyAnalyzer().analyze(exps)
        assert len(results) == 2

    def test_surface_entropy_zero_for_equal_scores(self):
        assert _surface_entropy([0.5, 0.5, 0.5]) == 0.0

    def test_surface_entropy_positive_for_varied_scores(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
        assert _surface_entropy(scores) > 0.0

    def test_fragility_isolated_is_one(self):
        # score = 1.0, mean_neighbor = 0.0 → nearly 1.0 (eps prevents exact 1.0)
        assert _fragility(1.0, 0.0) > 0.999

    def test_fragility_plateau_is_zero(self):
        assert _fragility(0.5, 0.5) == 0.0

    def test_plateau_width_all_within_tolerance(self):
        score = 1.0
        neighbors = [0.95, 1.0, 1.05]  # all within 10%
        assert _plateau_width(score, neighbors, 0.10) == 1.0

    def test_plateau_width_none_within_tolerance(self):
        score = 1.0
        neighbors = [0.5, 0.3, 0.2]  # all > 10% away
        assert _plateau_width(score, neighbors, 0.10) == 0.0

    def test_result_fields_rounded_to_6dp(self):
        exps = _make_experiments()
        results = ParameterTopologyAnalyzer().analyze(exps)
        for r in results.values():
            assert len(str(r.topology_fragility_score).split(".")[-1]) <= 6
