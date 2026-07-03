"""tests/research/test_evaluation.py — Phase E evaluation tests.

Covers:
  - compute_sharpe / sortino / max_drawdown / turnover / hit_rate / tail_risk
  - compute_regime_metrics: per-regime breakdown
  - check_rejection: each threshold triggers correctly
  - rank_results: deterministic, rejected at bottom
  - Ranking reproducibility
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _make_result(
    experiment_id: str = "exp_20260101_aabb0001",
    fold_id: int = 0,
    sharpe: float = 1.5,
    sortino: float = 2.0,
    turnover: float = 0.3,
    max_drawdown: float = -0.08,
    oos_robustness: float = 0.8,
    sharpe_decay: float = 0.2,
    trade_count: int = 20,
    hit_rate: float = 0.55,
):
    from src.research.contracts.result import ResultContract, RegimeMetrics, StabilityMetrics
    return ResultContract(
        experiment_id=experiment_id,
        fold_id=fold_id,
        sharpe=sharpe,
        sortino=sortino,
        turnover=turnover,
        max_drawdown=max_drawdown,
        exposure=0.6,
        pnl=10000.0,
        hit_rate=hit_rate,
        tail_risk=-0.02,
        regime_metrics=RegimeMetrics.empty(),
        stability_metrics=StabilityMetrics(
            parameter_stability=0.9,
            oos_robustness=oos_robustness,
            sharpe_decay=sharpe_decay,
        ),
        trade_count=trade_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_sharpe
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSharpe:
    def test_positive_sharpe(self):
        from src.research.evaluation.metrics import compute_sharpe
        rng = np.random.default_rng(42)
        r = pd.Series(0.005 + rng.normal(0, 0.01, 252))  # positive mean, varying
        assert compute_sharpe(r) > 0

    def test_zero_returns_sharpe_zero(self):
        from src.research.evaluation.metrics import compute_sharpe
        r = pd.Series([0.0] * 100)
        assert compute_sharpe(r) == 0.0

    def test_fewer_than_two_returns_zero(self):
        from src.research.evaluation.metrics import compute_sharpe
        assert compute_sharpe(pd.Series([0.05])) == 0.0

    def test_deterministic(self):
        from src.research.evaluation.metrics import compute_sharpe
        r = pd.Series([0.01, -0.005, 0.02, 0.003, -0.01] * 10)
        assert compute_sharpe(r) == compute_sharpe(r)


# ─────────────────────────────────────────────────────────────────────────────
# compute_sortino
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSortino:
    def test_positive_returns_inf(self):
        from src.research.evaluation.metrics import compute_sortino
        r = pd.Series([0.01] * 50)
        result = compute_sortino(r)
        assert result == float("inf")

    def test_mixed_returns_positive(self):
        from src.research.evaluation.metrics import compute_sortino
        r = pd.Series([0.01, -0.005, 0.02, -0.001] * 20)
        assert compute_sortino(r) > 0

    def test_fewer_than_two_returns_zero(self):
        from src.research.evaluation.metrics import compute_sortino
        assert compute_sortino(pd.Series([0.05])) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_max_drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_monotone_equity_no_drawdown(self):
        from src.research.evaluation.metrics import compute_max_drawdown
        equity = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert compute_max_drawdown(equity) >= -0.001

    def test_known_drawdown(self):
        from src.research.evaluation.metrics import compute_max_drawdown
        equity = pd.Series([1.0, 1.2, 0.9, 1.1])
        dd = compute_max_drawdown(equity)
        assert dd < -0.20  # 0.9/1.2 - 1 ≈ -25%

    def test_negative_value(self):
        from src.research.evaluation.metrics import compute_max_drawdown
        equity = pd.Series([1.0, 0.8, 0.6, 0.7])
        assert compute_max_drawdown(equity) < 0

    def test_empty_returns_zero(self):
        from src.research.evaluation.metrics import compute_max_drawdown
        assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_hit_rate / tail_risk / turnover
# ─────────────────────────────────────────────────────────────────────────────

class TestOtherMetrics:
    def test_hit_rate_all_positive(self):
        from src.research.evaluation.metrics import compute_hit_rate
        assert compute_hit_rate(pd.Series([0.01, 0.02, 0.03])) == 1.0

    def test_hit_rate_all_negative(self):
        from src.research.evaluation.metrics import compute_hit_rate
        assert compute_hit_rate(pd.Series([-0.01, -0.02])) == 0.0

    def test_hit_rate_empty(self):
        from src.research.evaluation.metrics import compute_hit_rate
        assert compute_hit_rate(pd.Series(dtype=float)) == 0.0

    def test_tail_risk_negative(self):
        from src.research.evaluation.metrics import compute_tail_risk
        r = pd.Series([0.01, -0.05, 0.02, -0.10, 0.03, -0.08])
        assert compute_tail_risk(r) < 0

    def test_turnover_zero_for_static_positions(self):
        from src.research.evaluation.metrics import compute_turnover
        pos = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.5, 0.5, 0.5]})
        assert compute_turnover(pos) == 0.0

    def test_turnover_positive(self):
        from src.research.evaluation.metrics import compute_turnover
        pos = pd.DataFrame({"A": [0.0, 1.0, 0.0], "B": [1.0, 0.0, 1.0]})
        assert compute_turnover(pos) > 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_regime_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeMetrics:
    def test_returns_dict_per_regime(self):
        from src.research.evaluation.metrics import compute_regime_metrics
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005], name="r")
        labels  = pd.Series(["trend_up", "range", "trend_up", "panic", "range"], name="l")
        returns.index = range(5)
        labels.index  = range(5)
        metrics = compute_regime_metrics(returns, labels)
        assert "trend_up" in metrics
        assert "range" in metrics

    def test_each_regime_has_required_keys(self):
        from src.research.evaluation.metrics import compute_regime_metrics
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005])
        labels  = pd.Series(["trend_up", "range", "trend_up", "range", "range"])
        metrics = compute_regime_metrics(returns, labels)
        for regime, m in metrics.items():
            assert "sharpe" in m
            assert "max_drawdown" in m
            assert "hit_rate" in m
            assert "trade_count" in m

    def test_trade_count_correct(self):
        from src.research.evaluation.metrics import compute_regime_metrics
        returns = pd.Series([0.01] * 5 + [-0.01] * 3)
        labels  = pd.Series(["trend_up"] * 5 + ["range"] * 3)
        metrics = compute_regime_metrics(returns, labels)
        assert metrics["trend_up"]["trade_count"] == 5
        assert metrics["range"]["trade_count"] == 3

    def test_deterministic(self):
        from src.research.evaluation.metrics import compute_regime_metrics
        r = pd.Series([0.01, -0.01, 0.02] * 10)
        l = pd.Series(["trend_up", "range", "panic"] * 10)
        m1 = compute_regime_metrics(r, l)
        m2 = compute_regime_metrics(r, l)
        assert m1 == m2


# ─────────────────────────────────────────────────────────────────────────────
# check_rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestRejection:
    def test_clean_result_passes(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result()
        rejected, reasons = check_rejection(r)
        assert not rejected
        assert reasons == []

    def test_sharpe_too_high_rejected(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result(sharpe=3.5)
        rejected, reasons = check_rejection(r)
        assert rejected
        assert any("sharpe" in reason for reason in reasons)

    def test_too_few_trades_rejected(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result(trade_count=2)
        rejected, reasons = check_rejection(r)
        assert rejected
        assert any("trade_count" in reason for reason in reasons)

    def test_excessive_drawdown_rejected(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result(max_drawdown=-0.6)
        rejected, reasons = check_rejection(r)
        assert rejected
        assert any("max_drawdown" in reason for reason in reasons)

    def test_high_turnover_rejected(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result(turnover=6.0)
        rejected, reasons = check_rejection(r)
        assert rejected

    def test_high_sharpe_decay_rejected(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result(sharpe_decay=2.5)
        rejected, reasons = check_rejection(r)
        assert rejected

    def test_multiple_violations_all_reported(self):
        from src.research.evaluation.rejection import check_rejection
        r = _make_result(sharpe=4.0, trade_count=1, max_drawdown=-0.9)
        rejected, reasons = check_rejection(r)
        assert rejected
        assert len(reasons) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# rank_results
# ─────────────────────────────────────────────────────────────────────────────

class TestRanking:
    def test_better_sharpe_ranks_higher(self):
        from src.research.evaluation.ranking import rank_results
        good = _make_result("exp_20260101_0001", sharpe=2.0, oos_robustness=0.9)
        bad  = _make_result("exp_20260101_0002", sharpe=0.5, oos_robustness=0.4)
        ranked = rank_results([bad, good])
        assert ranked[0].result.experiment_id == good.experiment_id

    def test_rejected_at_bottom(self):
        from src.research.evaluation.ranking import rank_results
        good     = _make_result("exp_20260101_0001", sharpe=1.5)
        rejected = _make_result("exp_20260101_0002", sharpe=4.0)  # > MAX_SHARPE
        ranked = rank_results([rejected, good])
        assert ranked[-1].rejected is True
        assert ranked[-1].result.experiment_id == rejected.experiment_id

    def test_deterministic_same_input(self):
        from src.research.evaluation.ranking import rank_results
        results = [
            _make_result(f"exp_20260101_{i:04d}", sharpe=float(i) / 5, oos_robustness=float(i) / 10)
            for i in range(1, 8)
        ]
        r1 = [x.result.experiment_id for x in rank_results(list(results))]
        r2 = [x.result.experiment_id for x in rank_results(list(results))]
        assert r1 == r2

    def test_scores_descending(self):
        from src.research.evaluation.ranking import rank_results
        results = [
            _make_result(f"exp_20260101_{i:04d}", sharpe=float(i), oos_robustness=float(i) / 5)
            for i in range(1, 6)
        ]
        ranked = rank_results(results)
        scores = [r.score for r in ranked if not r.rejected]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        from src.research.evaluation.ranking import rank_results
        assert rank_results([]) == []

    def test_all_rejected(self):
        from src.research.evaluation.ranking import rank_results
        results = [_make_result(f"exp_20260101_{i:04d}", sharpe=4.0) for i in range(3)]
        ranked = rank_results(results)
        assert all(r.rejected for r in ranked)

    def test_rejection_reasons_populated(self):
        from src.research.evaluation.ranking import rank_results
        r = _make_result(sharpe=5.0)
        ranked = rank_results([r])
        assert len(ranked[0].rejection_reasons) > 0

    def test_tie_breaking_by_experiment_id(self):
        from src.research.evaluation.ranking import rank_results
        # Two results with equal sharpe/robustness → tie-break by experiment_id
        r1 = _make_result("exp_20260101_aaaa", sharpe=1.5, oos_robustness=0.8)
        r2 = _make_result("exp_20260101_bbbb", sharpe=1.5, oos_robustness=0.8)
        ranked1 = rank_results([r1, r2])
        ranked2 = rank_results([r2, r1])
        ids1 = [r.result.experiment_id for r in ranked1]
        ids2 = [r.result.experiment_id for r in ranked2]
        assert ids1 == ids2  # same regardless of input order
