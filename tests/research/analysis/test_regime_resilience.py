"""Tests: regime_resilience.py — RegimeResilienceAnalyzer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.regime_resilience import RegimeResilienceAnalyzer
from tests.research.analysis.conftest import make_record


def _strong_record(exp_id, regimes=("trend_up", "range", "high_vol"), score=0.6):
    regime_scores = {r: 1.0 for r in regimes}
    return make_record(exp_id=exp_id, regime_scores=regime_scores, score=score, survived=True)


def _weak_record(exp_id, good_regime, bad_regimes=("range",)):
    regime_scores = {good_regime: 1.5}
    regime_scores.update({r: -0.5 for r in bad_regimes})
    return make_record(exp_id=exp_id, regime_scores=regime_scores, score=0.3, survived=False)


class TestRegimeResilienceAnalyzer:
    def test_empty_returns_zero_resilience(self):
        result = RegimeResilienceAnalyzer().analyze([])
        assert result.regime_resilience_score == 0.0
        assert result.cross_regime_retention == 0.0
        assert result.single_regime_dependent is True

    def test_all_strong_high_resilience(self):
        records = [_strong_record(f"e{i}") for i in range(5)]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.cross_regime_retention == 1.0
        assert result.regime_resilience_score > 0.5

    def test_deterministic_across_record_order(self):
        records = [
            _strong_record("e1", regimes=("trend_up", "panic")),
            _strong_record("e2", regimes=("range", "high_vol")),
            _weak_record("e3", good_regime="panic", bad_regimes=("trend_up",)),
        ]
        ana = RegimeResilienceAnalyzer()
        r1 = ana.analyze(records)
        r2 = ana.analyze(list(reversed(records)))
        assert r1.cross_regime_retention == r2.cross_regime_retention
        assert r1.regime_resilience_score == r2.regime_resilience_score
        assert r1.transition_pairs == r2.transition_pairs
        assert r1.transition_fragility == r2.transition_fragility

    def test_transition_pairs_sorted(self):
        records = [_strong_record(f"e{i}", regimes=("trend_up", "range", "panic")) for i in range(3)]
        result = RegimeResilienceAnalyzer().analyze(records)
        keys = [(a, b) for (a, b), _ in result.transition_pairs]
        assert keys == sorted(keys)

    def test_panic_only_alpha_detection(self):
        # All positive performance concentrated in panic
        records = [
            make_record(
                f"e{i}",
                regime_scores={"panic": 2.0, "trend_up": -0.5, "range": -0.3},
                score=0.4,
                survived=True,
            )
            for i in range(4)
        ]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.panic_only_alpha is True

    def test_trend_only_alpha_detection(self):
        records = [
            make_record(
                f"e{i}",
                regime_scores={"trend_up": 2.0, "trend_down": 1.5, "range": -0.8},
                score=0.5,
                survived=True,
            )
            for i in range(4)
        ]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.trend_only_alpha is True

    def test_vol_sensitive_detection(self):
        records = [
            make_record(
                f"e{i}",
                regime_scores={"high_vol": 2.5, "low_vol": -1.0},
                score=0.5,
                survived=True,
            )
            for i in range(3)
        ]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.vol_sensitive_alpha is True

    def test_regime_concentration_single_regime(self):
        # Only one regime positive → concentration = 1.0
        records = [
            make_record(f"e{i}", regime_scores={"trend_up": 1.5}, score=0.5, survived=True)
            for i in range(3)
        ]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.regime_concentration == 1.0

    def test_regime_concentration_balanced(self):
        # Equal performance across regimes → concentration near 1/n
        n_regimes = 4
        regime_scores = {f"r{k}": 1.0 for k in range(n_regimes)}
        records = [
            make_record(f"e{i}", regime_scores=regime_scores, score=0.5, survived=True)
            for i in range(4)
        ]
        result = RegimeResilienceAnalyzer().analyze(records)
        # Herfindahl for equal shares: sum((1/n)^2) = 1/n
        assert result.regime_concentration == pytest.approx(1.0 / n_regimes, abs=1e-4)

    def test_recovery_stability_no_panic(self):
        # No panic regime → recovery_stability = 0.0
        records = [_strong_record(f"e{i}", regimes=("trend_up", "range")) for i in range(3)]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.recovery_stability == 0.0

    def test_recovery_stability_good_recovery(self):
        # Positive in panic AND positive in other regime
        records = [
            make_record(f"e{i}", regime_scores={"panic": 1.2, "range": 0.8}, score=0.5, survived=True)
            for i in range(3)
        ]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert result.recovery_stability == 1.0

    def test_result_fields_bounded(self):
        records = [_strong_record(f"e{i}") for i in range(3)]
        result = RegimeResilienceAnalyzer().analyze(records)
        assert 0.0 <= result.cross_regime_retention <= 1.0
        assert 0.0 <= result.regime_resilience_score <= 1.0
        assert 0.0 <= result.regime_concentration <= 1.0
        assert 0.0 <= result.recovery_stability <= 1.0

    def test_to_dict_serializable(self):
        import json
        records = [_strong_record(f"e{i}") for i in range(2)]
        result = RegimeResilienceAnalyzer().analyze(records)
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_min_experiments_filter(self):
        # With min_regime_experiments=5, a transition pair with only 2 records is excluded
        records = [_strong_record(f"e{i}", regimes=("trend_up", "range")) for i in range(2)]
        result = RegimeResilienceAnalyzer(min_regime_experiments=5).analyze(records)
        assert result.transition_pairs == ()
