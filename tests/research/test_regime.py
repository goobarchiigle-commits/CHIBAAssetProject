"""tests/research/test_regime.py — Phase D regime engine tests.

Covers:
  - label_regimes: each timestamp gets exactly one label
  - Determinism: same OHLCV → same labels, always
  - Known patterns → correct labels
  - compute_regime_distribution: fractions sum to 1
  - label_hash: stable, changes on different labels
  - Missing columns → clear error
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _make_ohlcv(n: int = 120, seed: int = 0) -> "pd.DataFrame":
    """Synthetic OHLCV for testing."""
    rng = np.random.default_rng(seed)
    close = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    close = np.maximum(close, 100.0)
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100000, 1000000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestLabelRegimes:
    def test_returns_series_same_length(self):
        from src.research.regime.labeler import label_regimes
        df = _make_ohlcv(100)
        labels = label_regimes(df)
        assert len(labels) == 100

    def test_each_timestamp_has_one_label(self):
        from src.research.regime.labeler import label_regimes
        df = _make_ohlcv(100)
        labels = label_regimes(df)
        assert labels.notna().all()
        assert (labels != "").all()

    def test_index_matches_input(self):
        from src.research.regime.labeler import label_regimes
        df = _make_ohlcv(80)
        labels = label_regimes(df)
        assert (labels.index == df.index).all()

    def test_deterministic_same_input(self):
        from src.research.regime.labeler import label_regimes
        df = _make_ohlcv(100, seed=42)
        l1 = label_regimes(df)
        l2 = label_regimes(df)
        assert (l1 == l2).all()

    def test_all_labels_valid(self):
        from src.research.regime.labeler import label_regimes
        from src.research.regime.rules import ALL_REGIMES
        df = _make_ohlcv(200, seed=7)
        labels = label_regimes(df)
        assert set(labels.unique()).issubset(set(ALL_REGIMES))

    def test_missing_column_raises(self):
        from src.research.regime.labeler import label_regimes
        df = _make_ohlcv(50).drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            label_regimes(df)

    def test_empty_dataframe_returns_empty(self):
        from src.research.regime.labeler import label_regimes
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        labels = label_regimes(df)
        assert len(labels) == 0

    def test_gap_up_detected(self):
        from src.research.regime.labeler import label_regimes
        from src.research.regime.rules import GAP_UP
        df = _make_ohlcv(50, seed=0)
        # Force a large gap up on row 10
        df = df.copy()
        df.iloc[10, df.columns.get_loc("open")] = df.iloc[9]["close"] * 1.05
        labels = label_regimes(df)
        assert labels.iloc[10] == GAP_UP

    def test_gap_down_detected(self):
        from src.research.regime.labeler import label_regimes
        from src.research.regime.rules import GAP_DOWN
        df = _make_ohlcv(50, seed=0)
        df = df.copy()
        df.iloc[10, df.columns.get_loc("open")] = df.iloc[9]["close"] * 0.95
        labels = label_regimes(df)
        assert labels.iloc[10] == GAP_DOWN

    def test_panic_detected_on_crash(self):
        from src.research.regime.labeler import label_regimes
        from src.research.regime.rules import PANIC
        # Construct rows with: no gap (open ≈ prev close), big down return, wide ATR
        # This ensures gap_down doesn't fire (gap < 2%) but panic does.
        n = 100
        close = np.ones(n) * 1000.0
        # Intraday crash: open at prev close, close crashes 5% → no gap but big down day
        for i in range(60, n):
            close[i] = close[i - 1] * 0.95   # -5% per day, well below -3% panic threshold
        open_ = np.concatenate([[1000.0], close[:-1]])  # open = prev close, gap ~ 0%
        high  = open_ * 1.005                # small intraday range up
        low   = close * 0.90                 # very wide bar → high ATR
        volume = np.ones(n) * 500000.0
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
        labels = label_regimes(df)
        assert PANIC in labels.values

    def test_row_zero_is_range(self):
        from src.research.regime.labeler import label_regimes
        from src.research.regime.rules import RANGE
        df = _make_ohlcv(50, seed=0)
        labels = label_regimes(df)
        assert labels.iloc[0] == RANGE  # no previous close available


class TestRegimeDistribution:
    def test_fractions_sum_to_one(self):
        from src.research.regime.labeler import label_regimes, compute_regime_distribution
        df = _make_ohlcv(200, seed=1)
        labels = label_regimes(df)
        dist = compute_regime_distribution(labels)
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_all_keys_are_valid_regimes(self):
        from src.research.regime.labeler import label_regimes, compute_regime_distribution
        from src.research.regime.rules import ALL_REGIMES
        df = _make_ohlcv(200, seed=2)
        labels = label_regimes(df)
        dist = compute_regime_distribution(labels)
        assert set(dist.keys()).issubset(set(ALL_REGIMES))

    def test_empty_series_returns_empty_dict(self):
        from src.research.regime.labeler import compute_regime_distribution
        dist = compute_regime_distribution(pd.Series(dtype=object))
        assert dist == {}

    def test_sorted_keys(self):
        from src.research.regime.labeler import compute_regime_distribution
        import pandas as _pd2
        labels = _pd2.Series(["range", "trend_up", "range", "trend_down", "panic"])
        dist = compute_regime_distribution(labels)
        assert list(dist.keys()) == sorted(dist.keys())


class TestLabelHash:
    def test_deterministic(self):
        from src.research.regime.labeler import label_regimes, label_hash
        df = _make_ohlcv(100, seed=5)
        labels = label_regimes(df)
        h1 = label_hash(labels)
        h2 = label_hash(labels)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_labels_different_hash(self):
        from src.research.regime.labeler import label_regimes, label_hash
        df1 = _make_ohlcv(100, seed=1)
        df2 = _make_ohlcv(100, seed=2)
        l1 = label_regimes(df1)
        l2 = label_regimes(df2)
        if not (l1 == l2).all():
            assert label_hash(l1) != label_hash(l2)
