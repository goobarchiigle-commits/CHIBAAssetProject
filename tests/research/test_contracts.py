"""tests/research/test_contracts.py — Phase A contract tests.

Covers:
  - ExperimentContract: frozen, deterministic ID, serialization round-trip, stable_hash
  - FoldContract: frozen, serialization round-trip, validate()
  - ResultContract: frozen, serialization round-trip
  - StrategyContract: ABC enforcement
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentContract
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentContract:
    def _make(self, seed: int = 42, strategy_id: str = "turtle_v1") -> "ExperimentContract":
        from src.research.contracts.experiment import ExperimentContract
        eid = ExperimentContract.make_id(strategy_id, "ph00001", "dh00001", seed, timestamp="20260101")
        return ExperimentContract(
            experiment_id=eid,
            strategy_id=strategy_id,
            parameter_hash="ph00001",
            dataset_hash="dh00001",
            code_version="abc123",
            seed=seed,
            created_at="2026-01-01T00:00:00Z",
        )

    def test_frozen(self):
        from src.research.contracts.experiment import ExperimentContract
        exp = self._make()
        with pytest.raises((AttributeError, TypeError)):
            exp.seed = 99  # type: ignore

    def test_make_id_deterministic(self):
        from src.research.contracts.experiment import ExperimentContract
        id1 = ExperimentContract.make_id("s", "ph", "dh", 42, timestamp="20260101")
        id2 = ExperimentContract.make_id("s", "ph", "dh", 42, timestamp="20260101")
        assert id1 == id2

    def test_make_id_format(self):
        from src.research.contracts.experiment import ExperimentContract
        eid = ExperimentContract.make_id("s", "ph", "dh", 42, timestamp="20260101")
        assert eid.startswith("exp_")
        parts = eid.split("_")
        assert parts[1] == "20260101"
        assert len(parts[2]) == 8

    def test_different_seeds_different_id(self):
        from src.research.contracts.experiment import ExperimentContract
        id1 = ExperimentContract.make_id("s", "ph", "dh", 42, timestamp="20260101")
        id2 = ExperimentContract.make_id("s", "ph", "dh", 99, timestamp="20260101")
        assert id1 != id2

    def test_to_dict_round_trip(self):
        exp = self._make()
        from src.research.contracts.experiment import ExperimentContract
        d = exp.to_dict()
        restored = ExperimentContract.from_dict(d)
        assert restored == exp

    def test_stable_hash_deterministic(self):
        exp = self._make()
        assert exp.stable_hash() == exp.stable_hash()
        assert len(exp.stable_hash()) == 16

    def test_stable_hash_changes_on_seed(self):
        from src.research.contracts.experiment import ExperimentContract
        e1 = self._make(seed=1)
        e2 = self._make(seed=2)
        assert e1.stable_hash() != e2.stable_hash()

    def test_hash_parameters_deterministic(self):
        from src.research.contracts.experiment import ExperimentContract
        params = {"lookback": 20, "threshold": 0.5}
        h1 = ExperimentContract.hash_parameters(params)
        h2 = ExperimentContract.hash_parameters(params)
        assert h1 == h2

    def test_hash_parameters_key_order_invariant(self):
        from src.research.contracts.experiment import ExperimentContract
        h1 = ExperimentContract.hash_parameters({"a": 1, "b": 2})
        h2 = ExperimentContract.hash_parameters({"b": 2, "a": 1})
        assert h1 == h2

    def test_to_json_stable(self):
        exp = self._make()
        j1 = exp.to_json()
        j2 = exp.to_json()
        assert j1 == j2
        assert json.loads(j1)["seed"] == 42


# ─────────────────────────────────────────────────────────────────────────────
# FoldContract
# ─────────────────────────────────────────────────────────────────────────────

class TestFoldContract:
    def _make(self, fold_id: int = 0) -> "FoldContract":
        from src.research.contracts.fold import FoldContract, DateRange
        return FoldContract(
            fold_id=fold_id,
            train_range=DateRange(date(2020, 1, 1), date(2022, 1, 1)),
            validation_range=DateRange(date(2022, 1, 15), date(2022, 7, 1)),
            test_range=DateRange(date(2022, 7, 1), date(2023, 1, 1)),
            embargo_range=DateRange(date(2022, 1, 1), date(2022, 1, 15)),
            regime_distribution={"trend_up": 0.4, "range": 0.6},
        )

    def test_frozen(self):
        fold = self._make()
        with pytest.raises((AttributeError, TypeError)):
            fold.fold_id = 99  # type: ignore

    def test_to_dict_round_trip(self):
        from src.research.contracts.fold import FoldContract
        fold = self._make()
        d = fold.to_dict()
        restored = FoldContract.from_dict(d)
        assert restored == fold

    def test_date_range_days(self):
        from src.research.contracts.fold import DateRange
        dr = DateRange(date(2020, 1, 1), date(2020, 1, 11))
        assert dr.days() == 10

    def test_date_range_contains(self):
        from src.research.contracts.fold import DateRange
        dr = DateRange(date(2020, 1, 1), date(2020, 1, 10))
        assert date(2020, 1, 5) in dr
        assert date(2020, 1, 10) not in dr  # exclusive end
        assert date(2019, 12, 31) not in dr

    def test_validate_passes_for_valid_fold(self):
        self._make().validate()  # no exception

    def test_validate_raises_on_overlap(self):
        from src.research.contracts.fold import FoldContract, DateRange
        bad = FoldContract(
            fold_id=0,
            train_range=DateRange(date(2020, 1, 1), date(2022, 6, 1)),
            validation_range=DateRange(date(2022, 1, 1), date(2022, 7, 1)),  # overlaps train end
            test_range=DateRange(date(2022, 7, 1), date(2023, 1, 1)),
            embargo_range=DateRange(date(2022, 1, 1), date(2022, 1, 15)),
            regime_distribution={},
        )
        with pytest.raises(ValueError):
            bad.validate()

    def test_with_regime_distribution(self):
        fold = self._make()
        new_dist = {"panic": 0.1, "trend_up": 0.9}
        updated = fold.with_regime_distribution(new_dist)
        assert updated.regime_distribution == new_dist
        assert updated.fold_id == fold.fold_id
        assert fold.regime_distribution != new_dist  # original unchanged


# ─────────────────────────────────────────────────────────────────────────────
# ResultContract
# ─────────────────────────────────────────────────────────────────────────────

class TestResultContract:
    def _make(self, experiment_id: str = "exp_20260101_aabbccdd") -> "ResultContract":
        from src.research.contracts.result import ResultContract, RegimeMetrics, StabilityMetrics
        return ResultContract(
            experiment_id=experiment_id,
            fold_id=0,
            sharpe=1.5,
            sortino=2.0,
            turnover=0.3,
            max_drawdown=-0.08,
            exposure=0.6,
            pnl=15000.0,
            hit_rate=0.55,
            tail_risk=-0.025,
            regime_metrics=RegimeMetrics(by_regime={"trend_up": {"sharpe": 2.0, "max_drawdown": -0.05, "hit_rate": 0.6, "trade_count": 10}}),
            stability_metrics=StabilityMetrics(parameter_stability=0.9, oos_robustness=0.8, sharpe_decay=0.3),
            trade_count=25,
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.sharpe = 9.9  # type: ignore

    def test_to_dict_round_trip(self):
        from src.research.contracts.result import ResultContract
        r = self._make()
        d = r.to_dict()
        restored = ResultContract.from_dict(d)
        assert restored == r

    def test_to_json_valid(self):
        r = self._make()
        d = json.loads(r.to_json())
        assert d["sharpe"] == 1.5
        assert d["trade_count"] == 25

    def test_regime_metrics_empty(self):
        from src.research.contracts.result import RegimeMetrics
        rm = RegimeMetrics.empty()
        assert rm.by_regime == {}

    def test_stability_metrics_null(self):
        from src.research.contracts.result import StabilityMetrics
        sm = StabilityMetrics.null()
        assert sm.oos_robustness == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# StrategyContract
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyContract:
    def test_cannot_instantiate_abstract(self):
        from src.research.contracts.strategy import StrategyContract
        with pytest.raises(TypeError):
            StrategyContract()  # type: ignore

    def test_concrete_subclass_works(self):
        from src.research.contracts.strategy import StrategyContract

        class MinimalStrategy(StrategyContract):
            def generate_signals(self, data): return []
            def generate_orders(self, signals, portfolio): return []
            def required_features(self): return ["close"]
            def parameter_space(self): return {"lookback": range(10, 60)}

        s = MinimalStrategy()
        assert s.required_features() == ["close"]

    def test_partial_implementation_raises(self):
        from src.research.contracts.strategy import StrategyContract

        class Partial(StrategyContract):
            def generate_signals(self, data): return []
            def generate_orders(self, signals, portfolio): return []
            # missing required_features and parameter_space

        with pytest.raises(TypeError):
            Partial()  # type: ignore
