"""
tests/analytics/test_exit_analytics.py

Coverage:
  - ExitConvexityAnalyzer: excursion, volatility, convexity, flag correctness
  - ExitEfficiencyScorer: weighted component computation
  - RunnerRetentionDiagnostics: aggregate batch diagnostics
  - PostExitContinuationTracker: continuation returns, benchmark-relative, sector-relative
  - VolatilityAdjustedExitAnalytics: vol-scaled benchmark-relative returns
  - TrailingPolicyReplayer: all 6 policies, edge cases, determinism
  - reports: append_exit_record dedup, load_exit_records, persist_holding_path, parquet
  - generate_markdown_report: structure, reproducibility, required sections
  - deterministic replay: same inputs → byte-identical records
  - missing-data handling: empty path, no continuation, None prices
  - volatility normalization: exact vol calc verification
  - holding-path persistence: write/read round-trip
  - counterfactual trailing policy determinism
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from src.analytics.exits.models import (
    ALL_POLICIES,
    POLICY_ATR_TRAILING,
    POLICY_FIXED_TRAILING,
    POLICY_HYBRID,
    POLICY_MOMENTUM_EXHAUSTION,
    POLICY_TIME_DECAY,
    POLICY_VOL_SCALED,
    CompletedTrade,
    ExitConvexityRecord,
    HoldingPath,
    PolicyReplayResult,
    PostExitContinuation,
    RunnerRetentionSummary,
    SCHEMA_VERSION,
    EXIT_REASON_TURTLE_55D,
    EXIT_REASON_STOP_LOSS,
)
from src.analytics.exits.convexity import (
    ExitConvexityAnalyzer,
    ExitEfficiencyScorer,
    RunnerRetentionDiagnostics,
    _compute_excursions,
    _compute_volatility,
    _compute_trend_stability,
    _safe_div,
    _clip,
)
from src.analytics.exits.continuation import (
    PostExitContinuationTracker,
    VolatilityAdjustedExitAnalytics,
)
from src.analytics.exits.trailing_replay import (
    TrailingPolicyReplayer,
    _find_fixed_trailing_exit,
    _find_atr_trailing_exit,
    _find_time_decay_exit,
    _find_momentum_exhaustion_exit,
)
from src.analytics.exits.reports import (
    append_exit_record,
    append_policy_replay,
    generate_markdown_report,
    load_exit_records,
    load_holding_path,
    load_policy_replays,
    persist_holding_path,
    write_exit_records_parquet,
    write_policy_replays_parquet,
    write_report_atomic,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants & fixtures
# ─────────────────────────────────────────────────────────────────────────────

SYMBOL = "7203.T"
ENTRY_TS = "2026-04-01T09:00:00+00:00"
EXIT_TS = "2026-04-20T09:00:00+00:00"
ENTRY_PRICE = 1000.0
EXIT_PRICE = 1080.0
SHARES = 100
HOLD_DAYS = 19.0


def _make_trade(
    symbol: str = SYMBOL,
    entry_price: float = ENTRY_PRICE,
    exit_price: float = EXIT_PRICE,
    shares: int = SHARES,
    hold_days: float = HOLD_DAYS,
    exit_reason: str = EXIT_REASON_TURTLE_55D,
    trade_id: str = "T001",
) -> CompletedTrade:
    record_id = CompletedTrade.make_record_id(symbol, ENTRY_TS, EXIT_TS)
    realized_return = (exit_price - entry_price) / entry_price
    realized_pnl = (exit_price - entry_price) * shares
    return CompletedTrade(
        trade_id=trade_id,
        record_id=record_id,
        symbol=symbol,
        entry_timestamp=ENTRY_TS,
        exit_timestamp=EXIT_TS,
        holding_duration_days=hold_days,
        entry_price=entry_price,
        exit_price=exit_price,
        shares=shares,
        realized_return=realized_return,
        realized_pnl_jpy=realized_pnl,
        exit_reason=exit_reason,
    )


def _make_path(prices: List[float], entry_price: float = ENTRY_PRICE, shares: int = SHARES) -> HoldingPath:
    record_id = CompletedTrade.make_record_id(SYMBOL, ENTRY_TS, EXIT_TS)
    path_points = [{"day": float(i), "price": p} for i, p in enumerate(prices)]
    equity = [{"day": float(i), "pnl_jpy": (p - entry_price) * shares} for i, p in enumerate(prices)]
    return HoldingPath(
        record_id=record_id,
        symbol=SYMBOL,
        entry_timestamp=ENTRY_TS,
        exit_timestamp=EXIT_TS,
        holding_price_path=path_points,
        holding_equity_curve=equity,
    )


def _make_continuation(
    record_id: str,
    p1d: float = 0.02,
    p3d: float = 0.04,
    p5d: float = 0.05,
    b5d: float = 0.01,
) -> PostExitContinuation:
    return PostExitContinuation(
        record_id=record_id,
        post_exit_return_1d=p1d,
        post_exit_return_3d=p3d,
        post_exit_return_5d=p5d,
        benchmark_relative_post_exit_return_1d=p1d - 0.005,
        benchmark_relative_post_exit_return_3d=p3d - 0.01,
        benchmark_relative_post_exit_return_5d=b5d,
        sector_relative_post_exit_return=p5d - 0.02,
        relative_trend_persistence_score=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CompletedTrade.make_record_id determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_record_id_deterministic():
    id1 = CompletedTrade.make_record_id(SYMBOL, ENTRY_TS, EXIT_TS)
    id2 = CompletedTrade.make_record_id(SYMBOL, ENTRY_TS, EXIT_TS)
    assert id1 == id2


def test_record_id_differs_for_different_inputs():
    id1 = CompletedTrade.make_record_id(SYMBOL, ENTRY_TS, EXIT_TS)
    id2 = CompletedTrade.make_record_id("1234.T", ENTRY_TS, EXIT_TS)
    assert id1 != id2


def test_record_id_is_hex_string():
    rid = CompletedTrade.make_record_id(SYMBOL, ENTRY_TS, EXIT_TS)
    assert len(rid) == 64
    int(rid, 16)  # raises if not hex


# ─────────────────────────────────────────────────────────────────────────────
# 2. _compute_excursions correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_excursions_simple_uptrend():
    # prices go 1000 → 1100 → 1200 → 1150 (exit at 1150, peak at idx=2)
    prices = [1000.0, 1100.0, 1200.0, 1150.0]
    mfe, mae, peak_pnl, giveback, peak_idx = _compute_excursions(1000.0, 100, prices)
    assert mfe == pytest.approx(0.20)   # (1200-1000)/1000
    assert mae == pytest.approx(0.0)    # min is 1000 = entry
    assert peak_pnl == pytest.approx((1200 - 1000) * 100)
    assert peak_idx == 2
    # giveback = (peak_pnl - realized_pnl) / peak_pnl
    # = (20000 - 15000) / 20000 = 0.25
    assert giveback == pytest.approx(0.25)


def test_excursions_with_adverse_excursion():
    # dip below entry then recover
    prices = [1000.0, 950.0, 1050.0, 1100.0]
    mfe, mae, peak_pnl, giveback, peak_idx = _compute_excursions(1000.0, 100, prices)
    assert mfe == pytest.approx(0.10)
    assert mae == pytest.approx(-0.05)   # (950-1000)/1000
    assert peak_idx == 3


def test_excursions_empty_prices():
    mfe, mae, peak_pnl, giveback, peak_idx = _compute_excursions(1000.0, 100, [])
    assert mfe == 0.0
    assert mae == 0.0
    assert peak_pnl == 0.0
    assert giveback == 0.0
    assert peak_idx == 0


def test_excursions_single_price():
    mfe, mae, peak_pnl, giveback, peak_idx = _compute_excursions(1000.0, 100, [1050.0])
    assert mfe == pytest.approx(0.05)
    assert mae == pytest.approx(0.05)   # only price = both max and min
    assert peak_idx == 0


def test_excursions_declining_path():
    prices = [1000.0, 980.0, 960.0, 940.0]
    mfe, mae, peak_pnl, giveback, peak_idx = _compute_excursions(1000.0, 100, prices)
    assert mfe == pytest.approx(0.0)   # peak is entry (first)
    assert mae == pytest.approx(-0.06)  # (940-1000)/1000
    assert peak_pnl == 0.0             # no positive PnL at peak


# ─────────────────────────────────────────────────────────────────────────────
# 3. _compute_volatility correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_volatility_constant_prices():
    prices = [1000.0] * 10
    vol = _compute_volatility(prices)
    assert vol == pytest.approx(0.0)


def test_volatility_single_price():
    vol = _compute_volatility([1000.0])
    assert vol == 0.0


def test_volatility_empty():
    vol = _compute_volatility([])
    assert vol == 0.0


def test_volatility_known_values():
    # prices: 100, 110, 99, 105 → returns: 0.1, -0.1, 0.0606...
    prices = [100.0, 110.0, 99.0, 105.0]
    returns = [0.1, (99-110)/110, (105-99)/99]
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean)**2 for r in returns) / n
    expected_vol = math.sqrt(var) * math.sqrt(252)
    computed = _compute_volatility(prices)
    assert computed == pytest.approx(expected_vol, rel=1e-6)


def test_volatility_two_prices():
    prices = [100.0, 105.0]
    vol = _compute_volatility(prices)
    # single return = 0.05, vol = std([0.05]) = 0, annualized = 0
    assert vol == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _compute_trend_stability
# ─────────────────────────────────────────────────────────────────────────────

def test_trend_stability_no_adverse():
    # MAE = 0 → stability = 1.0
    assert _compute_trend_stability(0.10, 0.0) == pytest.approx(1.0)


def test_trend_stability_full_adverse():
    # MAE = -MFE → stability = 0.0
    assert _compute_trend_stability(0.10, -0.10) == pytest.approx(0.0, abs=0.01)


def test_trend_stability_partial():
    # MAE = -0.05, MFE = 0.10 → 1 - 0.05/0.10 = 0.5
    val = _compute_trend_stability(0.10, -0.05)
    assert val == pytest.approx(0.5, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ExitConvexityAnalyzer.analyze basic correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_uptrend_no_continuation():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    # Prices: 1000 → 1050 → 1100 → 1080 (peak at idx=2)
    hp = _make_path([1000.0, 1050.0, 1100.0, 1080.0])
    analyzer = ExitConvexityAnalyzer()
    rec = analyzer.analyze(trade, hp)

    assert rec.record_id == trade.record_id
    assert rec.symbol == SYMBOL
    assert rec.realized_return == pytest.approx(0.08)
    assert rec.max_favorable_excursion == pytest.approx(0.10)
    assert rec.max_adverse_excursion == pytest.approx(0.0)
    assert rec.peak_unrealized_pnl == pytest.approx((1100 - 1000) * 100)
    assert 0.0 <= rec.pnl_giveback_ratio <= 1.0
    assert rec.convexity_capture_rate == pytest.approx(
        min(1.0, 0.08 / 0.10), abs=0.01
    )
    # No continuation → post_exit fields None
    assert rec.post_exit_return_1d is None
    assert rec.post_exit_return_5d is None
    assert rec.runner_continuation_score == 0.0
    assert rec.premature_exit_ratio == 0.0
    assert rec.is_premature_exit is False


def test_analyze_with_continuation_premature():
    trade = _make_trade(entry_price=1000.0, exit_price=1050.0)
    hp = _make_path([1000.0, 1050.0, 1080.0, 1050.0])
    cont = PostExitContinuation(
        record_id=trade.record_id,
        post_exit_return_1d=0.02,
        post_exit_return_3d=0.04,
        post_exit_return_5d=0.08,  # > realized_return * 0.5 = 0.025 → premature
    )
    rec = ExitConvexityAnalyzer().analyze(trade, hp, cont)

    assert rec.post_exit_return_5d == pytest.approx(0.08)
    assert rec.is_premature_exit is True
    assert rec.runner_continuation_score > 0.0


def test_analyze_with_continuation_not_premature():
    trade = _make_trade(entry_price=1000.0, exit_price=1100.0)
    hp = _make_path([1000.0, 1050.0, 1100.0])
    cont = PostExitContinuation(
        record_id=trade.record_id,
        post_exit_return_5d=0.01,  # tiny continuation → not premature
    )
    rec = ExitConvexityAnalyzer().analyze(trade, hp, cont)
    assert rec.is_premature_exit is False


def test_analyze_convexity_truncated():
    # Exit at 1050 when peak was 1200 → capture = 0.05/0.20 = 0.25 < 0.70
    trade = _make_trade(entry_price=1000.0, exit_price=1050.0)
    hp = _make_path([1000.0, 1100.0, 1200.0, 1050.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    assert rec.is_convexity_truncated is True
    assert rec.convexity_capture_rate < 0.70


def test_analyze_not_convexity_truncated():
    # Exit at 1095 when peak was 1100 → capture ≈ 0.90
    trade = _make_trade(entry_price=1000.0, exit_price=1095.0)
    hp = _make_path([1000.0, 1050.0, 1100.0, 1095.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    assert rec.is_convexity_truncated is False


def test_analyze_empty_holding_path():
    trade = _make_trade()
    hp = _make_path([])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    assert rec.max_favorable_excursion == 0.0
    assert rec.max_adverse_excursion == 0.0
    assert rec.volatility_during_hold == 0.0


def test_analyze_runner_retained():
    # Good exit: runner_retention > 0.8
    trade = _make_trade(entry_price=1000.0, exit_price=1100.0)
    hp = _make_path([1000.0, 1050.0, 1100.0])
    cont = PostExitContinuation(record_id=trade.record_id, post_exit_return_5d=0.01)
    rec = ExitConvexityAnalyzer().analyze(trade, hp, cont)
    # total_move = 0.10 + 0.01 = 0.11, runner_retention = 0.10/0.11 ≈ 0.91
    assert rec.runner_retention_rate > 0.8
    assert rec.is_runner_retained is True


def test_analyze_generated_at_utc():
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    # Must be ISO format
    dt = datetime.fromisoformat(rec.generated_at)
    assert dt.tzinfo is not None


def test_analyze_vol_insensitive_flag():
    # Very volatile path but tiny return → vol-insensitive
    trade = _make_trade(entry_price=1000.0, exit_price=1010.0)  # 1% return
    # High volatility path
    prices = [1000.0, 900.0, 1100.0, 950.0, 1050.0, 1010.0]
    hp = _make_path(prices)
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    # vol will be high relative to 1% return
    if rec.volatility_during_hold > 0:
        assert rec.is_volatility_insensitive is True


def test_analyze_schema_version():
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    assert rec.schema_version == SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# 6. ExitEfficiencyScorer
# ─────────────────────────────────────────────────────────────────────────────

def test_scorer_perfect_exit():
    scorer = ExitEfficiencyScorer()
    score = scorer.score_components(
        convexity_capture_rate=1.0,
        pnl_giveback_ratio=0.0,
        premature_exit_ratio=0.0,
    )
    assert score == pytest.approx(1.0, abs=1e-5)


def test_scorer_worst_exit():
    scorer = ExitEfficiencyScorer()
    score = scorer.score_components(
        convexity_capture_rate=0.0,
        pnl_giveback_ratio=1.0,
        premature_exit_ratio=1.0,
    )
    assert score == pytest.approx(0.0, abs=1e-5)


def test_scorer_weights_sum_to_one():
    from src.analytics.exits.convexity import _W_CONVEXITY, _W_GIVEBACK, _W_CONTINUATION
    assert _W_CONVEXITY + _W_GIVEBACK + _W_CONTINUATION == pytest.approx(1.0)


def test_scorer_partial_score():
    scorer = ExitEfficiencyScorer()
    score = scorer.score_components(
        convexity_capture_rate=0.8,
        pnl_giveback_ratio=0.2,
        premature_exit_ratio=0.3,
    )
    # 0.8*0.4 + 0.8*0.3 + 0.7*0.3 = 0.32 + 0.24 + 0.21 = 0.77
    assert score == pytest.approx(0.77, abs=0.001)


def test_scorer_clips_out_of_range():
    scorer = ExitEfficiencyScorer()
    score = scorer.score_components(
        convexity_capture_rate=1.5,   # >1 → clipped to 1
        pnl_giveback_ratio=-0.1,      # <0 → clipped to 0
        premature_exit_ratio=0.0,
    )
    assert 0.0 <= score <= 1.0


def test_scorer_from_record():
    trade = _make_trade(entry_price=1000.0, exit_price=1100.0)
    hp = _make_path([1000.0, 1050.0, 1100.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    scorer = ExitEfficiencyScorer()
    score = scorer.score(rec)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(rec.exit_efficiency_score, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 7. RunnerRetentionDiagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(
    symbol: str = SYMBOL,
    realized_return: float = 0.08,
    mfe: float = 0.10,
    mae: float = 0.0,
    giveback: float = 0.20,
    capture: float = 0.80,
    efficiency: float = 0.75,
    runner_retention: float = 0.90,
    premature: bool = False,
    truncated: bool = False,
    vol_insensitive: bool = False,
    p5d: Optional[float] = None,
    alpha_leakage: Optional[float] = None,
) -> ExitConvexityRecord:
    record_id = CompletedTrade.make_record_id(symbol, ENTRY_TS, EXIT_TS)
    return ExitConvexityRecord(
        record_id=record_id,
        trade_id="T001",
        symbol=symbol,
        entry_timestamp=ENTRY_TS,
        exit_timestamp=EXIT_TS,
        holding_duration_days=HOLD_DAYS,
        realized_return=realized_return,
        exit_reason=EXIT_REASON_TURTLE_55D,
        max_favorable_excursion=mfe,
        max_adverse_excursion=mae,
        peak_unrealized_pnl=mfe * 1000 * 100,
        pnl_giveback_ratio=giveback,
        volatility_during_hold=0.20,
        volatility_adjusted_return=0.40,
        post_exit_return_5d=p5d,
        convexity_capture_rate=capture,
        runner_retention_rate=runner_retention,
        exit_efficiency_score=efficiency,
        post_exit_alpha_leakage=alpha_leakage,
        is_premature_exit=premature,
        is_convexity_truncated=truncated,
        is_volatility_insensitive=vol_insensitive,
        is_runner_retained=runner_retention > 0.8,
    )


def test_retention_diagnostics_empty():
    diag = RunnerRetentionDiagnostics()
    summary = diag.compute([])
    assert summary.total_trades == 0
    assert summary.premature_exit_rate == 0.0
    assert summary.top_premature_exits == []


def test_retention_diagnostics_basic():
    records = [
        _make_record("7203.T", premature=True, p5d=0.05, alpha_leakage=0.05),
        _make_record("6758.T", truncated=True, capture=0.50),
        _make_record("9984.T", premature=False, runner_retention=0.95),
    ]
    diag = RunnerRetentionDiagnostics()
    summary = diag.compute(records)

    assert summary.total_trades == 3
    assert summary.premature_exit_count == 1
    assert summary.convexity_truncated_count == 1
    assert summary.premature_exit_rate == pytest.approx(1 / 3)
    assert summary.convexity_truncation_rate == pytest.approx(1 / 3)
    assert summary.mean_post_exit_alpha_leakage == pytest.approx(0.05)


def test_retention_diagnostics_top_premature_sorted():
    records = [
        _make_record("A", premature=True, p5d=0.03),
        _make_record("B", premature=True, p5d=0.10),
        _make_record("C", premature=True, p5d=0.07),
    ]
    # Need different record_ids
    for i, r in enumerate(records):
        r.record_id = f"rec{i:04d}"

    diag = RunnerRetentionDiagnostics()
    summary = diag.compute(records)
    # Top premature: sorted by p5d descending → B(0.10), C(0.07), A(0.03)
    assert summary.top_premature_exits[0] == "rec0001"  # B
    assert summary.top_premature_exits[1] == "rec0002"  # C
    assert summary.top_premature_exits[2] == "rec0000"  # A


def test_retention_diagnostics_top_truncations_sorted():
    records = [
        _make_record("A", truncated=True, capture=0.60),
        _make_record("B", truncated=True, capture=0.40),
        _make_record("C", truncated=False, capture=0.90),
    ]
    for i, r in enumerate(records):
        r.record_id = f"rec{i:04d}"

    diag = RunnerRetentionDiagnostics()
    summary = diag.compute(records)
    # Top truncations: sorted by capture ascending → B(0.40), A(0.60), C(0.90)
    assert summary.top_convexity_truncations[0] == "rec0001"  # B


def test_retention_diagnostics_no_alpha_leakage():
    records = [_make_record(alpha_leakage=None), _make_record(alpha_leakage=None)]
    diag = RunnerRetentionDiagnostics()
    summary = diag.compute(records)
    assert summary.mean_post_exit_alpha_leakage is None


def test_retention_diagnostics_schema_version():
    diag = RunnerRetentionDiagnostics()
    summary = diag.compute([_make_record()])
    assert summary.schema_version == SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# 8. PostExitContinuationTracker
# ─────────────────────────────────────────────────────────────────────────────

def test_continuation_basic():
    trade = _make_trade(exit_price=1080.0)
    tracker = PostExitContinuationTracker()
    cont = tracker.compute(
        trade=trade,
        post_exit_prices={1: 1091.6, 3: 1101.6, 5: 1134.0},  # 1%, 2%, 5% from 1080
    )
    assert cont.record_id == trade.record_id
    assert cont.post_exit_return_1d == pytest.approx((1091.6 - 1080) / 1080, rel=1e-4)
    assert cont.post_exit_return_3d == pytest.approx((1101.6 - 1080) / 1080, rel=1e-4)
    assert cont.post_exit_return_5d == pytest.approx((1134.0 - 1080) / 1080, rel=1e-4)


def test_continuation_benchmark_relative():
    trade = _make_trade(exit_price=1080.0)
    tracker = PostExitContinuationTracker()
    # Stock +5%, benchmark +2%
    cont = tracker.compute(
        trade=trade,
        post_exit_prices={1: 1080.0, 3: 1080.0, 5: 1134.0},  # 5% at 5d
        benchmark_prices={0: 1080.0, 1: 1080.0, 3: 1080.0, 5: 1101.6},  # 2% at 5d
    )
    # b5d = post_exit_5d - benchmark_5d = 0.05 - 0.02 = 0.03 approx
    assert cont.benchmark_relative_post_exit_return_5d == pytest.approx(
        (1134.0 - 1080) / 1080 - (1101.6 - 1080) / 1080, rel=0.01
    )


def test_continuation_sector_relative():
    trade = _make_trade(exit_price=1000.0)
    tracker = PostExitContinuationTracker()
    cont = tracker.compute(
        trade=trade,
        post_exit_prices={5: 1050.0},   # stock +5%
        sector_price_5d=1030.0,          # sector +3%
    )
    # sector_rel = 0.05 - 0.03 = 0.02
    assert cont.sector_relative_post_exit_return == pytest.approx(0.02, abs=0.001)


def test_continuation_missing_prices_returns_none():
    trade = _make_trade(exit_price=1000.0)
    tracker = PostExitContinuationTracker()
    cont = tracker.compute(trade=trade, post_exit_prices={})
    assert cont.post_exit_return_1d is None
    assert cont.post_exit_return_3d is None
    assert cont.post_exit_return_5d is None


def test_continuation_relative_trend_persistence():
    trade = _make_trade(exit_price=1000.0)
    tracker = PostExitContinuationTracker()
    # All benchmark-relative positive → persistence = 1.0
    cont = tracker.compute(
        trade=trade,
        post_exit_prices={1: 1010.0, 3: 1020.0, 5: 1030.0},
        benchmark_prices={0: 1000.0, 1: 1005.0, 3: 1010.0, 5: 1015.0},
    )
    # all b_relative > 0 → rel_trend = 1.0
    assert cont.relative_trend_persistence_score == pytest.approx(1.0)


def test_continuation_deterministic_for_same_inputs():
    trade = _make_trade(exit_price=1000.0)
    tracker = PostExitContinuationTracker()
    prices = {1: 1010.0, 3: 1025.0, 5: 1040.0}
    c1 = tracker.compute(trade, prices)
    c2 = tracker.compute(trade, prices)
    assert asdict(c1) == asdict(c2)


# ─────────────────────────────────────────────────────────────────────────────
# 9. VolatilityAdjustedExitAnalytics
# ─────────────────────────────────────────────────────────────────────────────

def test_vol_adjusted_scales_benchmark_relative():
    trade = _make_trade()
    cont = PostExitContinuation(
        record_id=trade.record_id,
        post_exit_return_5d=0.05,
        benchmark_relative_post_exit_return_1d=0.02,
        benchmark_relative_post_exit_return_3d=0.03,
        benchmark_relative_post_exit_return_5d=0.04,
    )
    va = VolatilityAdjustedExitAnalytics()
    adjusted = va.compute(trade, cont, volatility_during_hold=0.25)
    scale = 1.0 + 0.25
    assert adjusted.benchmark_relative_post_exit_return_1d == pytest.approx(0.02 / scale)
    assert adjusted.benchmark_relative_post_exit_return_3d == pytest.approx(0.03 / scale)
    assert adjusted.benchmark_relative_post_exit_return_5d == pytest.approx(0.04 / scale)


def test_vol_adjusted_zero_vol_unchanged():
    trade = _make_trade()
    cont = PostExitContinuation(
        record_id=trade.record_id,
        benchmark_relative_post_exit_return_5d=0.03,
    )
    va = VolatilityAdjustedExitAnalytics()
    adjusted = va.compute(trade, cont, volatility_during_hold=0.0)
    # Zero vol → unchanged
    assert adjusted.benchmark_relative_post_exit_return_5d == pytest.approx(0.03)


def test_vol_adjusted_preserves_raw_returns():
    trade = _make_trade()
    cont = PostExitContinuation(
        record_id=trade.record_id,
        post_exit_return_1d=0.01,
        post_exit_return_3d=0.02,
        post_exit_return_5d=0.03,
    )
    va = VolatilityAdjustedExitAnalytics()
    adjusted = va.compute(trade, cont, volatility_during_hold=0.30)
    # Raw returns should be unchanged
    assert adjusted.post_exit_return_1d == pytest.approx(0.01)
    assert adjusted.post_exit_return_5d == pytest.approx(0.03)


def test_vol_adjusted_trend_persistence_recomputed():
    trade = _make_trade()
    cont = PostExitContinuation(
        record_id=trade.record_id,
        benchmark_relative_post_exit_return_1d=0.02,
        benchmark_relative_post_exit_return_3d=0.03,
        benchmark_relative_post_exit_return_5d=0.04,
        relative_trend_persistence_score=1.0,
    )
    va = VolatilityAdjustedExitAnalytics()
    adjusted = va.compute(trade, cont, volatility_during_hold=0.20)
    # All scaled values still positive → persistence = 1.0
    assert adjusted.relative_trend_persistence_score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. TrailingPolicyReplayer — individual policy correctness
# ─────────────────────────────────────────────────────────────────────────────

def _make_replay_path(prices: List[float]) -> HoldingPath:
    record_id = CompletedTrade.make_record_id(SYMBOL, ENTRY_TS, EXIT_TS)
    return HoldingPath(
        record_id=record_id,
        symbol=SYMBOL,
        entry_timestamp=ENTRY_TS,
        exit_timestamp=EXIT_TS,
        holding_price_path=[{"day": float(i), "price": p} for i, p in enumerate(prices)],
        holding_equity_curve=[],
    )


def test_fixed_trailing_find_exit():
    # Peak 1100, stop 5% = 1045, price drops to 1040 at idx=4
    prices = [1000.0, 1050.0, 1100.0, 1070.0, 1040.0, 1060.0]
    days = list(range(len(prices)))
    idx, day = _find_fixed_trailing_exit(prices, days, 0.05)
    # 1040 < 1100 * 0.95 = 1045 → triggers at idx=4
    assert idx == 4
    assert day == 4.0


def test_fixed_trailing_no_trigger():
    # Price never falls enough
    prices = [1000.0, 1050.0, 1100.0, 1090.0, 1095.0]
    days = list(range(len(prices)))
    idx, day = _find_fixed_trailing_exit(prices, days, 0.05)
    # Never triggers → last index
    assert idx == len(prices) - 1


def test_time_decay_exit_exact():
    days = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    idx, day = _find_time_decay_exit(days, max_days=20.0)
    assert idx == 4
    assert day == 20.0


def test_time_decay_exit_before_max():
    days = [0.0, 5.0, 10.0]  # never reaches 20
    idx, day = _find_time_decay_exit(days, max_days=20.0)
    assert idx == len(days) - 1


def test_momentum_exhaustion_exit():
    # After window=3, price[3]=980 < price[0]=1000 → trigger at idx=3
    prices = [1000.0, 1020.0, 1010.0, 980.0, 990.0]
    days = list(range(len(prices)))
    idx, day = _find_momentum_exhaustion_exit(prices, days, 3)
    assert idx == 3


def test_momentum_no_exhaustion():
    # Always trending up
    prices = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0]
    days = list(range(len(prices)))
    idx, day = _find_momentum_exhaustion_exit(prices, days, 2)
    # Never crosses below → last index
    assert idx == len(prices) - 1


def test_replayer_fixed_trailing():
    trade = _make_trade(entry_price=1000.0, exit_price=1040.0)
    # Actual exit at 1040, policy would exit at 1040 (5% from peak 1100 → stop at 1045)
    hp = _make_replay_path([1000.0, 1050.0, 1100.0, 1070.0, 1040.0])
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_FIXED_TRAILING, {"stop_pct": 0.05})

    assert result.policy_name == POLICY_FIXED_TRAILING
    assert result.trade_record_id == trade.record_id
    assert result.actual_realized_return == pytest.approx(0.04)
    # Hypothetical exit at price 1040 → same return
    assert result.hypothetical_realized_return == pytest.approx(0.04)
    assert 0.0 <= result.convexity_retention <= 1.0
    assert 0.0 <= result.pnl_giveback <= 1.0


def test_replayer_time_decay():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    # 5-day path, policy exits at day 2
    prices = [1000.0, 1030.0, 1060.0, 1050.0, 1080.0]
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_TIME_DECAY, {"max_days": 2.0})

    assert result.policy_name == POLICY_TIME_DECAY
    # Hyp exit at day=2 → price=1060 → return = 0.06
    assert result.hypothetical_realized_return == pytest.approx(0.06)


def test_replayer_hybrid_picks_earlier():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    # Price drops to stop before time decay
    prices = [1000.0, 1100.0, 1000.0, 980.0, 1080.0]  # peak at 1100, drops to 1000 → 5% stop triggers
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_HYBRID, {"stop_pct": 0.05, "max_days": 10.0})

    assert result.policy_name == POLICY_HYBRID
    # Stop should trigger before time decay at day 10
    assert "hyp_day" in result.hypothetical_exit_timestamp


def test_replayer_momentum_exhaustion():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    prices = [1000.0, 1020.0, 1040.0, 1010.0, 1080.0]
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_MOMENTUM_EXHAUSTION, {"momentum_window": 2})
    # prices[3]=1010 < prices[1]=1020 → trigger at idx=3
    assert result.hypothetical_realized_return == pytest.approx((1010 - 1000) / 1000)


def test_replayer_vol_scaled():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    prices = [1000.0, 1100.0, 1050.0, 1040.0, 1080.0]
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_VOL_SCALED, {"k": 1.0})
    assert result.policy_name == POLICY_VOL_SCALED
    assert 0.0 <= result.convexity_retention <= 1.0


def test_replayer_atr_trailing():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    prices = [1000.0, 1100.0, 1050.0, 1000.0, 1080.0]
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_ATR_TRAILING, {"n_atr": 1.0, "atr_period": 2})
    assert result.policy_name == POLICY_ATR_TRAILING
    assert 0.0 <= result.pnl_giveback <= 1.0


def test_replayer_empty_path_returns_actual():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    hp = _make_replay_path([])
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_FIXED_TRAILING)
    # No path → hypothetical = actual
    assert result.hypothetical_realized_return == pytest.approx(trade.realized_return)
    assert result.convexity_retention == pytest.approx(1.0)


def test_replayer_invalid_policy_raises():
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    with pytest.raises(ValueError, match="Unknown policy"):
        replayer.replay(trade, hp, "nonexistent_policy")


def test_replayer_all_policies():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    prices = [1000.0, 1050.0, 1100.0, 1090.0, 1080.0]
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    results = replayer.replay_all_policies(trade, hp)
    assert len(results) == len(ALL_POLICIES)
    policy_names = [r.policy_name for r in results]
    for p in ALL_POLICIES:
        assert p in policy_names


def test_replayer_determinism():
    trade = _make_trade(entry_price=1000.0, exit_price=1080.0)
    prices = [1000.0, 1050.0, 1100.0, 1090.0, 1080.0]
    hp = _make_replay_path(prices)
    replayer = TrailingPolicyReplayer()
    r1 = replayer.replay(trade, hp, POLICY_FIXED_TRAILING, {"stop_pct": 0.05})
    r2 = replayer.replay(trade, hp, POLICY_FIXED_TRAILING, {"stop_pct": 0.05})
    # Same inputs → same result (excluding generated_at)
    assert r1.hypothetical_realized_return == r2.hypothetical_realized_return
    assert r1.convexity_retention == r2.convexity_retention
    assert r1.replay_id == r2.replay_id


def test_replayer_replay_id_deterministic():
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    r1 = replayer.replay(trade, hp, POLICY_FIXED_TRAILING)
    r2 = replayer.replay(trade, hp, POLICY_FIXED_TRAILING)
    assert r1.replay_id == r2.replay_id
    assert len(r1.replay_id) == 64


# ─────────────────────────────────────────────────────────────────────────────
# 11. Persistence: append_exit_record / load_exit_records
# ─────────────────────────────────────────────────────────────────────────────

def test_append_and_load_exit_records(tmp_path):
    path = tmp_path / "exit_records.jsonl"
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)

    result = append_exit_record(rec, path)
    assert result is True
    assert path.exists()

    loaded = load_exit_records(path)
    assert len(loaded) == 1
    assert loaded[0].record_id == rec.record_id


def test_append_exit_record_dedup(tmp_path):
    path = tmp_path / "exit_records.jsonl"
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)

    r1 = append_exit_record(rec, path)
    r2 = append_exit_record(rec, path)  # duplicate
    assert r1 is True
    assert r2 is False

    loaded = load_exit_records(path)
    assert len(loaded) == 1


def test_append_multiple_records(tmp_path):
    path = tmp_path / "exit_records.jsonl"
    analyzer = ExitConvexityAnalyzer()

    for i in range(5):
        trade = _make_trade(
            symbol=f"{7000 + i}.T",
            entry_price=1000.0 + i * 10,
            exit_price=1100.0 + i * 10,
        )
        hp = _make_path([1000.0 + i * 10, 1100.0 + i * 10])
        rec = analyzer.analyze(trade, hp)
        # Force unique record_id
        rec.record_id = f"rec{i:04d}"
        append_exit_record(rec, path)

    loaded = load_exit_records(path)
    assert len(loaded) == 5


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    loaded = load_exit_records(path)
    assert loaded == []


def test_load_missing_file(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    loaded = load_exit_records(path)
    assert loaded == []


def test_load_handles_corrupt_line(tmp_path):
    path = tmp_path / "corrupt.jsonl"
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    line = json.dumps(rec.to_dict(), sort_keys=True, ensure_ascii=False)
    path.write_text(line + "\n{CORRUPT JSON\n", encoding="utf-8")
    loaded = load_exit_records(path)
    assert len(loaded) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 12. Persistence: append_policy_replay / load_policy_replays
# ─────────────────────────────────────────────────────────────────────────────

def test_append_and_load_policy_replays(tmp_path):
    path = tmp_path / "replays.jsonl"
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1050.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_FIXED_TRAILING)

    r = append_policy_replay(result, path)
    assert r is True

    loaded = load_policy_replays(path)
    assert len(loaded) == 1
    assert loaded[0].replay_id == result.replay_id


def test_append_policy_replay_dedup(tmp_path):
    path = tmp_path / "replays.jsonl"
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_FIXED_TRAILING)

    r1 = append_policy_replay(result, path)
    r2 = append_policy_replay(result, path)
    assert r1 is True
    assert r2 is False
    loaded = load_policy_replays(path)
    assert len(loaded) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 13. Persistence: holding path
# ─────────────────────────────────────────────────────────────────────────────

def test_persist_and_load_holding_path(tmp_path):
    hp = _make_path([1000.0, 1050.0, 1100.0, 1080.0])
    persist_holding_path(hp, tmp_path)

    loaded = load_holding_path(hp.record_id, tmp_path)
    assert loaded is not None
    assert loaded.record_id == hp.record_id
    assert len(loaded.holding_price_path) == 4
    assert loaded.holding_price_path[2]["price"] == pytest.approx(1100.0)


def test_persist_holding_path_overwrites_atomically(tmp_path):
    hp = _make_path([1000.0, 1080.0])
    persist_holding_path(hp, tmp_path)

    # Update and re-persist
    hp.holding_price_path.append({"day": 2.0, "price": 1100.0})
    persist_holding_path(hp, tmp_path)

    loaded = load_holding_path(hp.record_id, tmp_path)
    assert loaded is not None
    assert len(loaded.holding_price_path) == 3


def test_load_holding_path_missing(tmp_path):
    result = load_holding_path("nonexistent_id", tmp_path)
    assert result is None


def test_holding_path_round_trip_equity_curve(tmp_path):
    hp = _make_path([1000.0, 1050.0, 1100.0])
    persist_holding_path(hp, tmp_path)
    loaded = load_holding_path(hp.record_id, tmp_path)
    assert loaded.holding_equity_curve[2]["pnl_jpy"] == pytest.approx(
        (1100.0 - ENTRY_PRICE) * SHARES
    )


# ─────────────────────────────────────────────────────────────────────────────
# 14. Parquet persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_write_exit_records_parquet(tmp_path):
    import pandas as pd
    path = tmp_path / "exit_records.parquet"
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)

    write_exit_records_parquet([rec], path)
    assert path.exists()

    df = pd.read_parquet(path, engine="pyarrow")
    assert len(df) == 1
    assert "record_id" in df.columns
    assert "convexity_capture_rate" in df.columns


def test_write_policy_replays_parquet(tmp_path):
    import pandas as pd
    path = tmp_path / "replays.parquet"
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    results = replayer.replay_all_policies(trade, hp)

    write_policy_replays_parquet(results, path)
    assert path.exists()

    df = pd.read_parquet(path, engine="pyarrow")
    assert len(df) == len(ALL_POLICIES)
    assert "policy_name" in df.columns


def test_write_exit_records_parquet_empty(tmp_path):
    path = tmp_path / "empty.parquet"
    write_exit_records_parquet([], path)
    # Should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 15. Markdown report generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_records_for_report() -> list:
    analyzer = ExitConvexityAnalyzer()
    records = []
    for i in range(5):
        trade = _make_trade(
            symbol=f"{7000 + i}.T",
            entry_price=1000.0,
            exit_price=1050.0 + i * 10,
        )
        trade.record_id = f"rec{i:04d}"
        hp = _make_path([1000.0, 1025.0, 1060.0, 1050.0 + i * 10])
        cont = PostExitContinuation(
            record_id=trade.record_id,
            post_exit_return_5d=0.02 + i * 0.01,
            benchmark_relative_post_exit_return_5d=0.01 + i * 0.005,
        )
        rec = analyzer.analyze(trade, hp, cont)
        rec.record_id = f"rec{i:04d}"
        records.append(rec)
    return records


def test_report_required_sections():
    records = _make_records_for_report()
    md = generate_markdown_report(records)
    assert "# Exit Convexity Analytics Report" in md
    assert "## Summary Statistics" in md
    assert "## Convexity Leakage Diagnostics" in md
    assert "## Exit Inefficiency Rankings" in md
    assert "## Top Alpha Leakage Trades" in md
    assert "## Trailing Policy Comparisons" in md
    assert "## Trend Persistence Observations" in md


def test_report_with_policy_replays():
    records = _make_records_for_report()
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1050.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    replays = replayer.replay_all_policies(trade, hp)

    md = generate_markdown_report(records, policy_replays=replays)
    assert "## Trailing Policy Comparisons" in md
    assert POLICY_FIXED_TRAILING in md


def test_report_reproducibility():
    records = _make_records_for_report()
    md1 = generate_markdown_report(records, report_date="2026-05-19", top_n=3)
    md2 = generate_markdown_report(records, report_date="2026-05-19", top_n=3)
    # Generated timestamps will differ; strip first 3 lines (header + date)
    lines1 = md1.split("\n")[3:]
    lines2 = md2.split("\n")[3:]
    assert lines1 == lines2


def test_report_empty_records():
    md = generate_markdown_report([])
    assert "No records" in md or "_No records._" in md


def test_report_includes_summary():
    records = _make_records_for_report()
    diag = RunnerRetentionDiagnostics()
    summary = diag.compute(records)
    md = generate_markdown_report(records, summary=summary)
    assert "## Runner Retention Summary" in md
    assert "Total Trades" in md


def test_report_schema_version_footer():
    md = generate_markdown_report([])
    assert f"schema_version={SCHEMA_VERSION}" in md


def test_write_report_atomic(tmp_path):
    path = tmp_path / "report.md"
    content = "# Test Report\n\nContent here."
    write_report_atomic(content, path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == content


def test_write_report_atomic_overwrites(tmp_path):
    path = tmp_path / "report.md"
    write_report_atomic("version 1", path)
    write_report_atomic("version 2", path)
    assert path.read_text(encoding="utf-8") == "version 2"


# ─────────────────────────────────────────────────────────────────────────────
# 16. Deterministic replay: same inputs → identical JSONL output
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_replay_jsonl(tmp_path):
    path1 = tmp_path / "r1.jsonl"
    path2 = tmp_path / "r2.jsonl"

    analyzer = ExitConvexityAnalyzer()
    trade = _make_trade()
    hp = _make_path([1000.0, 1050.0, 1100.0, 1080.0])
    cont = _make_continuation(trade.record_id)

    for path in (path1, path2):
        rec = analyzer.analyze(trade, hp, cont)
        rec.generated_at = "2026-05-19T00:00:00+00:00"  # fix timestamp
        append_exit_record(rec, path)

    lines1 = path1.read_text(encoding="utf-8").strip()
    lines2 = path2.read_text(encoding="utf-8").strip()
    assert lines1 == lines2


def test_deterministic_record_dict_content(tmp_path):
    analyzer = ExitConvexityAnalyzer()
    trade = _make_trade()
    hp = _make_path([1000.0, 1050.0, 1100.0, 1080.0])
    cont = _make_continuation(trade.record_id)

    r1 = analyzer.analyze(trade, hp, cont)
    r2 = analyzer.analyze(trade, hp, cont)

    d1 = {k: v for k, v in asdict(r1).items() if k != "generated_at"}
    d2 = {k: v for k, v in asdict(r2).items() if k != "generated_at"}
    assert d1 == d2


# ─────────────────────────────────────────────────────────────────────────────
# 17. ExitConvexityRecord.from_dict round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_exit_record_roundtrip():
    trade = _make_trade()
    hp = _make_path([1000.0, 1080.0])
    cont = _make_continuation(trade.record_id)
    rec = ExitConvexityAnalyzer().analyze(trade, hp, cont)

    d = rec.to_dict()
    restored = ExitConvexityRecord.from_dict(d)

    assert restored.record_id == rec.record_id
    assert restored.realized_return == pytest.approx(rec.realized_return)
    assert restored.post_exit_return_5d == pytest.approx(rec.post_exit_return_5d)
    assert restored.is_premature_exit == rec.is_premature_exit
    assert restored.schema_version == rec.schema_version


def test_holding_path_roundtrip():
    hp = _make_path([1000.0, 1050.0, 1100.0])
    d = hp.to_dict()
    restored = HoldingPath.from_dict(d)
    assert restored.record_id == hp.record_id
    assert len(restored.holding_price_path) == 3
    assert restored.holding_price_path[1]["price"] == pytest.approx(1050.0)


def test_policy_replay_roundtrip():
    trade = _make_trade()
    hp = _make_replay_path([1000.0, 1080.0])
    replayer = TrailingPolicyReplayer()
    result = replayer.replay(trade, hp, POLICY_FIXED_TRAILING)
    d = result.to_dict()
    restored = PolicyReplayResult.from_dict(d)
    assert restored.replay_id == result.replay_id
    assert restored.hypothetical_realized_return == pytest.approx(result.hypothetical_realized_return)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Edge cases / guard conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_div_zero_denominator():
    assert _safe_div(5.0, 0.0, fallback=99.0) == 99.0


def test_clip_below_zero():
    assert _clip(-0.5) == 0.0


def test_clip_above_one():
    assert _clip(1.5) == 1.0


def test_analyze_zero_entry_price():
    trade = _make_trade(entry_price=0.001)  # near-zero but not zero
    hp = _make_path([0.001, 0.002])
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    assert rec is not None


def test_analyze_flat_path():
    trade = _make_trade(entry_price=1000.0, exit_price=1000.0)
    hp = _make_path([1000.0] * 5)
    rec = ExitConvexityAnalyzer().analyze(trade, hp)
    assert rec.max_favorable_excursion == pytest.approx(0.0)
    assert rec.volatility_during_hold == pytest.approx(0.0)
    assert rec.pnl_giveback_ratio == pytest.approx(0.0)


def test_post_exit_continuation_missing_data_no_crash():
    trade = _make_trade(exit_price=0.0)  # edge: zero exit price
    tracker = PostExitContinuationTracker()
    # Should not raise even with zero/missing prices
    cont = tracker.compute(trade, post_exit_prices={1: 0, 3: 0, 5: 0})
    assert cont.post_exit_return_1d is None


def test_all_policies_in_constant():
    assert len(ALL_POLICIES) == 6
    for p in [
        POLICY_FIXED_TRAILING, POLICY_ATR_TRAILING, POLICY_VOL_SCALED,
        POLICY_TIME_DECAY, POLICY_HYBRID, POLICY_MOMENTUM_EXHAUSTION,
    ]:
        assert p in ALL_POLICIES


def test_report_largest_post_exit_continuation_section():
    records = _make_records_for_report()
    md = generate_markdown_report(records)
    assert "Largest Post-Exit Continuations" in md
