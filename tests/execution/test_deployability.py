"""
tests/execution/test_deployability.py
Capital-aware deployability scoring layer tests.

Covers:
  - deployable candidates prioritized over undeployable ones
  - lot feasibility rejection (insufficient cash for 1 lot)
  - capital_fit multiplier computation
  - sector capacity multiplier computation
  - alloc_survival_probability bounded 0..1
  - deterministic ranking (same input → same output)
  - deterministic tie-breaking (symbol alphabetic)
  - max_positions rejection
  - zero/negative price rejection
  - diagnostics aggregate accuracy
  - metrics emission
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.deployability import (
    DeployabilityDiagnostics,
    DeployabilityResult,
    emit_deployability_metrics,
    rank_by_deployability,
    score_candidate,
    LOT_SIZE,
    MIN_SCORE_RATIO,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score(
    symbol="6920.T",
    alpha=1.0,
    price=1000.0,
    qty=200,
    available_cash=3_000_000.0,
    max_alloc=600_000.0,
    sector="電機精密",
    sector_weight=0.0,
    sector_cap=0.25,
    capital=3_000_000.0,
    existing_positions=0,
    max_positions=3,
    reject_threshold=0.25,
) -> DeployabilityResult:
    return score_candidate(
        symbol=symbol,
        alpha_score=alpha,
        price=price,
        qty_requested=qty,
        available_cash=available_cash,
        max_alloc=max_alloc,
        sector=sector,
        sector_weight=sector_weight,
        sector_cap=sector_cap,
        capital=capital,
        existing_positions=existing_positions,
        max_positions=max_positions,
        reject_threshold=reject_threshold,
    )


def _rank(candidates, available_cash=3_000_000.0, capital=3_000_000.0,
          sector_weights=None, sector_cap=0.25, max_alloc=600_000.0,
          existing_positions=0, max_positions=3):
    return rank_by_deployability(
        candidates=candidates,
        available_cash=available_cash,
        capital=capital,
        sector_weights=sector_weights or {},
        sector_cap=sector_cap,
        max_alloc=max_alloc,
        existing_positions=existing_positions,
        max_positions=max_positions,
    )


def _cand(symbol, alpha=1.0, price=1000.0, qty=200, sector="電機精密"):
    return {"symbol": symbol, "alpha_score": alpha, "price": price, "qty": qty, "sector": sector}


# ─────────────────────────────────────────────────────────────────────────────
# TestScoreCandidate
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCandidate:

    def test_fully_deployable_score_bounded_by_alpha(self):
        r = _score(alpha=1.0, price=1000.0, qty=100, available_cash=3_000_000.0)
        assert r.is_deployable
        assert 0.0 < r.deployability_score <= 1.0

    def test_zero_price_not_deployable(self):
        r = _score(price=0.0)
        assert not r.is_deployable
        assert r.deployability_score == 0.0
        assert r.rejection_reason == "price_or_qty_zero"

    def test_zero_qty_not_deployable(self):
        r = _score(qty=0)
        assert not r.is_deployable
        assert r.rejection_reason == "price_or_qty_zero"

    def test_max_positions_reached(self):
        r = _score(existing_positions=3, max_positions=3)
        assert not r.is_deployable
        assert r.rejection_reason == "max_positions_reached"

    def test_insufficient_cash_1_lot(self):
        # price=100_000, need 100*100_000=10_000_000 for 1 lot, cash=5_000_000
        r = _score(price=100_000.0, qty=100, available_cash=5_000_000.0,
                   max_alloc=6_000_000.0, capital=10_000_000.0)
        # 1 lot cost = 100 * 100_000 = 10_000_000 > available_cash=5_000_000 → lot_feasibility=0
        assert not r.is_deployable
        assert "insufficient_cash_for_1_lot" in r.rejection_reason

    def test_sector_cap_full_rejected(self):
        r = _score(sector_weight=0.25, sector_cap=0.25,
                   price=1000.0, qty=100, available_cash=3_000_000.0)
        assert not r.is_deployable
        assert "sector_cap_insufficient" in r.rejection_reason

    def test_sector_partial_capacity(self):
        # remaining = 0.25 - 0.15 = 0.10; requested = 100*1000/3M ≈ 0.033
        # sector_mult = 0.10/0.033 > 1.0 → capped at 1.0
        r = _score(sector_weight=0.15, sector_cap=0.25,
                   price=1000.0, qty=100, available_cash=3_000_000.0)
        assert r.sector_capacity_multiplier == 1.0
        assert r.is_deployable

    def test_alloc_survival_bounded(self):
        r = _score()
        assert 0.0 <= r.alloc_survival_probability <= 1.0

    def test_capital_fit_does_not_exceed_one(self):
        r = _score(qty=100, price=100.0, available_cash=3_000_000.0, max_alloc=600_000.0)
        assert r.capital_fit_multiplier <= 1.0

    def test_high_alpha_preserved_in_score(self):
        r_high = _score(alpha=100.0)
        r_low  = _score(alpha=1.0)
        assert r_high.deployability_score > r_low.deployability_score


# ─────────────────────────────────────────────────────────────────────────────
# TestRankByDeployability
# ─────────────────────────────────────────────────────────────────────────────

class TestRankByDeployability:

    def test_deployable_before_undeployable(self):
        """Low-alpha deployable candidate ranks before high-alpha undeployable."""
        cands = [
            _cand("A", alpha=10.0, price=100_000.0, qty=100),  # 1 lot = 10M > cash
            _cand("B", alpha=1.0,  price=100.0,    qty=100),   # cheap, easily deployable
        ]
        ranked, _ = _rank(cands, available_cash=500_000.0, capital=3_000_000.0)
        assert ranked[0]["symbol"] == "B", "deployable B should rank first"

    def test_same_score_tie_broken_by_alpha_then_symbol(self):
        """Tie in deployability_score → higher alpha wins; same alpha → alphabetic."""
        # Two candidates with same sector=empty + same price → same score
        cands = [
            _cand("Z", alpha=2.0, price=100.0, qty=100),
            _cand("A", alpha=2.0, price=100.0, qty=100),
        ]
        ranked, _ = _rank(cands)
        # Same alpha → alphabetic tie-break → A before Z
        assert ranked[0]["symbol"] == "A"
        assert ranked[1]["symbol"] == "Z"

    def test_deterministic_same_input_same_output(self):
        cands = [_cand(f"{i}.T", alpha=float(i), price=1000.0, qty=100) for i in range(5)]
        r1, _ = _rank(cands)
        r2, _ = _rank(cands)
        assert [c["symbol"] for c in r1] == [c["symbol"] for c in r2]

    def test_deployability_fields_added(self):
        cands = [_cand("6920.T")]
        ranked, _ = _rank(cands)
        c = ranked[0]
        assert "deployability_score" in c
        assert "rejection_reason" in c
        assert "is_deployable" in c
        assert "capital_fit_multiplier" in c
        assert "sector_cap_multiplier" in c
        assert "lot_feasibility_multiplier" in c
        assert "alloc_survival_probability" in c

    def test_empty_candidates_returns_empty(self):
        ranked, diag = _rank([])
        assert ranked == []
        assert diag.total_candidates == 0

    def test_all_undeployable_returns_all_with_zero_score(self):
        cands = [_cand("A", price=0.0), _cand("B", price=0.0)]
        ranked, diag = _rank(cands)
        assert len(ranked) == 2
        assert diag.deployable_count == 0
        assert all(c["deployability_score"] == 0.0 for c in ranked)


# ─────────────────────────────────────────────────────────────────────────────
# TestDeployabilityDiagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestDeployabilityDiagnostics:

    def test_alloc_survival_rate_correct(self):
        cands = [
            _cand("A", price=100.0),   # deployable
            _cand("B", price=100.0),   # deployable
            _cand("C", price=0.0),     # undeployable
        ]
        _, diag = _rank(cands, available_cash=3_000_000.0)
        assert diag.total_candidates == 3
        assert diag.deployable_count == 2
        assert diag.undeployable_count == 1
        assert abs(diag.alloc_survival_rate - 2 / 3) < 1e-6

    def test_rejection_reason_breakdown_populated(self):
        cands = [_cand("A", price=0.0), _cand("B", price=0.0)]
        _, diag = _rank(cands)
        assert sum(diag.rejection_reason_breakdown.values()) == 2

    def test_idle_cash_ratio_zero_when_fully_deployed(self):
        # available_cash == capital → idle_cash_ratio = 0
        _, diag = _rank(
            [_cand("A")],
            available_cash=3_000_000.0,
            capital=3_000_000.0,
        )
        assert diag.idle_cash_ratio == 0.0

    def test_undeployable_alpha_count_only_nonzero_alpha(self):
        cands = [
            _cand("A", alpha=5.0, price=0.0),   # undeployable, has alpha
            _cand("B", alpha=0.0, price=0.0),   # undeployable, no alpha
        ]
        _, diag = _rank(cands)
        assert diag.undeployable_alpha_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestMetricsEmission
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEmission:

    def test_emit_writes_jsonl(self, tmp_path):
        diag = DeployabilityDiagnostics(
            total_candidates=3,
            deployable_count=2,
            undeployable_count=1,
            alloc_survival_rate=2 / 3,
        )
        log_path = tmp_path / "deploy_metrics.jsonl"
        metrics = emit_deployability_metrics(
            diag=diag,
            capital=3_000_000.0,
            available_cash=1_000_000.0,
            committed_yen=500_000.0,
            log_path=log_path,
        )
        assert log_path.exists()
        records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        assert len(records) == 1
        r = records[0]
        assert r["total_candidates"] == 3
        assert r["deployable_count"] == 2
        assert "capital_efficiency" in r
        assert "idle_cash_ratio" in r

    def test_emit_no_log_path_returns_dict(self):
        diag = DeployabilityDiagnostics()
        metrics = emit_deployability_metrics(
            diag=diag, capital=1.0, available_cash=1.0, committed_yen=0.0,
            log_path=None,
        )
        assert isinstance(metrics, dict)
        assert "ts" in metrics

    def test_capital_efficiency_bounded(self, tmp_path):
        diag = DeployabilityDiagnostics()
        metrics = emit_deployability_metrics(
            diag=diag, capital=1_000_000.0, available_cash=1_000_000.0,
            committed_yen=2_000_000.0,   # over-allocated (guard test)
            log_path=None,
        )
        assert metrics["capital_efficiency"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestAdaptiveAllocIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveAllocIntegration:
    """Verify AdaptiveAllocator.pre_rank_candidates() wraps deployability correctly."""

    def test_pre_rank_returns_ranked_and_diag(self):
        from src.execution.adaptive_alloc import AdaptiveAllocator

        alloc = AdaptiveAllocator(capital=3_000_000.0, sector_cap=0.25)
        cands = [
            {"symbol": "A", "alpha_score": 10.0, "price": 100_000.0, "qty": 100, "sector": "電機精密"},
            {"symbol": "B", "alpha_score":  1.0, "price":    100.0, "qty": 100, "sector": "電機精密"},
        ]
        ranked, diag = alloc.pre_rank_candidates(
            candidates=cands,
            available_cash=500_000.0,
            max_alloc=600_000.0,
            existing_positions=0,
            max_positions=3,
        )
        # B (deployable, cheap) should outrank A (undeployable, expensive)
        assert ranked[0]["symbol"] == "B"
        assert diag.total_candidates == 2

    def test_pre_rank_does_not_mutate_sector_weights(self):
        from src.execution.adaptive_alloc import AdaptiveAllocator

        alloc = AdaptiveAllocator(capital=3_000_000.0, sector_cap=0.25)
        alloc._sector_weight["電機精密"] = 0.10
        before = dict(alloc._sector_weight)
        cands = [{"symbol": "A", "alpha_score": 1.0, "price": 1000.0, "qty": 100, "sector": "電機精密"}]
        alloc.pre_rank_candidates(
            candidates=cands, available_cash=3_000_000.0,
            max_alloc=600_000.0, existing_positions=0, max_positions=3,
        )
        assert alloc._sector_weight == before
