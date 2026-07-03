"""Tests: regime.py — ExecutionRegimeClassifier, RegimeClassification."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.regime import (
    ExecutionRegimeClassifier, RegimeClassification, RegimeThresholds,
    REGIME_NORMAL, REGIME_THIN, REGIME_STRESSED, REGIME_DISLOCATED,
    REGIME_OPEN_AUCTION, REGIME_CLOSE_AUCTION, REGIME_HALTED, REGIME_LIQUIDITY_VACUUM,
    ALL_REGIMES,
)
from tests.execution.reality.conftest import TS, TS2, make_regime_classifier


class TestRegimeClassification:
    def test_frozen(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        with pytest.raises(Exception):
            rc.regime = "CHANGED"  # type: ignore

    def test_classification_id_prefix(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        assert rc.classification_id.startswith("rgm_")

    def test_to_dict_from_dict_roundtrip(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        restored = RegimeClassification.from_dict(rc.to_dict())
        assert restored.regime == rc.regime
        assert restored.classification_id == rc.classification_id


class TestExecutionRegimeClassifier:
    def test_all_regimes_count(self):
        assert len(ALL_REGIMES) == 8

    def test_normal_conditions(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        assert rc.regime == REGIME_NORMAL

    def test_thin_market_spread(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(thin_spread_bps=30.0))
        rc = clf.classify(35.0, 0.50, 0.05, 2.0, TS)
        assert rc.regime == REGIME_THIN

    def test_thin_market_participation(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(thin_participation=0.20))
        rc = clf.classify(10.0, 0.15, 0.05, 2.0, TS)
        assert rc.regime == REGIME_THIN

    def test_stressed_spread(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(stressed_spread_bps=100.0))
        rc = clf.classify(110.0, 0.50, 0.05, 2.0, TS)
        assert rc.regime == REGIME_STRESSED

    def test_stressed_reject_rate(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(stressed_reject_rate=0.15))
        rc = clf.classify(10.0, 0.50, 0.20, 2.0, TS)
        assert rc.regime == REGIME_STRESSED

    def test_dislocated_spread(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(dislocated_spread_bps=300.0))
        rc = clf.classify(350.0, 0.50, 0.05, 2.0, TS)
        assert rc.regime == REGIME_DISLOCATED

    def test_open_auction(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS, is_open_auction=True)
        assert rc.regime == REGIME_OPEN_AUCTION

    def test_close_auction(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS, is_close_auction=True)
        assert rc.regime == REGIME_CLOSE_AUCTION

    def test_halted_zero_participation(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(halted_participation=0.0, vacuum_participation=0.05))
        rc = clf.classify(10.0, 0.0, 0.05, 2.0, TS)
        assert rc.regime == REGIME_HALTED

    def test_liquidity_vacuum(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(vacuum_participation=0.05))
        rc = clf.classify(10.0, 0.03, 0.05, 2.0, TS)
        assert rc.regime == REGIME_LIQUIDITY_VACUUM

    def test_is_tradeable_normal(self):
        clf = make_regime_classifier()
        rc = clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        assert rc.is_tradeable()

    def test_is_tradeable_thin(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(thin_spread_bps=30.0))
        rc = clf.classify(35.0, 0.50, 0.05, 2.0, TS)
        assert rc.is_tradeable()

    def test_is_not_tradeable_halted(self):
        clf = ExecutionRegimeClassifier(RegimeThresholds(halted_participation=0.0, vacuum_participation=0.05))
        rc = clf.classify(10.0, 0.0, 0.05, 2.0, TS)
        assert not rc.is_tradeable()

    def test_history_accumulates(self):
        clf = make_regime_classifier()
        clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        clf.classify(10.0, 0.50, 0.05, 2.0, TS2)
        assert len(clf.history()) == 2

    def test_current_regime(self):
        clf = make_regime_classifier()
        clf.classify(10.0, 0.50, 0.05, 2.0, TS)
        assert clf.current_regime() == REGIME_NORMAL

    def test_current_regime_none_initially(self):
        clf = make_regime_classifier()
        assert clf.current_regime() is None

    def test_classification_id_deterministic(self):
        clf1 = make_regime_classifier()
        clf2 = make_regime_classifier()
        rc1 = clf1.classify(10.0, 0.50, 0.05, 2.0, TS)
        rc2 = clf2.classify(10.0, 0.50, 0.05, 2.0, TS)
        assert rc1.classification_id == rc2.classification_id

    def test_to_dict_keys(self):
        clf = make_regime_classifier()
        d = clf.to_dict()
        assert "thresholds" in d and "history_count" in d and "current_regime" in d
