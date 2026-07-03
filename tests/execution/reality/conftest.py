"""Shared test fixtures for execution reality tests."""
from __future__ import annotations

from src.execution.reality.clock import ExecutionClock, SOURCE_SIGNAL, SOURCE_DECISION, SOURCE_SUBMIT, SOURCE_BROKER, SOURCE_EXCHANGE, SOURCE_FILL
from src.execution.reality.snapshot import ExpectedFill, ActualFill, ExecutionTimestamps, build_snapshot
from src.execution.reality.slippage import SlippageTracker
from src.execution.reality.fill_quality import FillQualityAnalyzer
from src.execution.reality.latency import LatencyDriftMonitor
from src.execution.reality.liquidity import LiquidityRealityMonitor
from src.execution.reality.comparator import PaperLiveComparator
from src.execution.reality.decay import AlphaDecayTracker
from src.execution.reality.decision import DeploymentGuard, DeploymentThresholds
from src.execution.reality.replay import CounterfactualReplay
from src.execution.reality.ledger import DivergenceLedger
from src.execution.reality.regime import ExecutionRegimeClassifier

# Nanosecond epoch constants (2026-01-01 UTC)
SIGNAL_NS   = 1_767_225_600_000_000_000
DECISION_NS = 1_767_225_600_001_000_000
SUBMIT_NS   = 1_767_225_600_002_000_000
BROKER_NS   = 1_767_225_600_010_000_000
EXCHANGE_NS = 1_767_225_600_015_000_000
FILL_NS     = 1_767_225_600_020_000_000

TS  = "2026-01-01T00:00:00.000000Z"
TS2 = "2026-01-01T00:00:01.000000Z"
TS3 = "2026-01-01T00:00:02.000000Z"

SYM_A = "7203"
SYM_B = "6758"
SESSION = "sess_test_001"


def make_timestamps() -> ExecutionTimestamps:
    return ExecutionTimestamps(
        signal_ns=SIGNAL_NS,
        decision_ns=DECISION_NS,
        submit_ns=SUBMIT_NS,
        broker_accept_ns=BROKER_NS,
        exchange_ns=EXCHANGE_NS,
        fill_ns=FILL_NS,
    )


def make_expected(symbol: str = SYM_A, price: float = 1000.0, qty: int = 100) -> ExpectedFill:
    return ExpectedFill(
        symbol=symbol, side="BUY", quantity=qty,
        expected_price=price, expected_slippage_bps=3.0, expected_latency_ns=20_000_000,
    )


def make_actual(symbol: str = SYM_A, fill_price: float = 1001.0, qty: int = 100) -> ActualFill:
    return ActualFill(
        symbol=symbol, side="BUY", filled_quantity=qty,
        fill_price=fill_price, fill_latency_ns=20_000_000, broker_latency_ns=8_000_000,
        rejected=False, partial=False,
    )


def make_snapshot(session_id: str = SESSION, **kwargs):
    expected = make_expected(**{k: v for k, v in kwargs.items() if k in ("symbol", "price", "qty")})
    actual = make_actual(symbol=expected.symbol)
    ts = make_timestamps()
    return build_snapshot(session_id, expected, actual, ts, schema_version="1.0.0", recorded_at=TS)


def make_slippage_tracker() -> SlippageTracker:
    return SlippageTracker()


def make_fill_analyzer() -> FillQualityAnalyzer:
    return FillQualityAnalyzer()


def make_latency_monitor() -> LatencyDriftMonitor:
    return LatencyDriftMonitor(baseline_latency_ns=50_000_000)


def make_liquidity_monitor() -> LiquidityRealityMonitor:
    return LiquidityRealityMonitor()


def make_comparator() -> PaperLiveComparator:
    return PaperLiveComparator()


def make_decay_tracker() -> AlphaDecayTracker:
    return AlphaDecayTracker()


def make_guard() -> DeploymentGuard:
    return DeploymentGuard()


def make_replay() -> CounterfactualReplay:
    return CounterfactualReplay()


def make_ledger() -> DivergenceLedger:
    return DivergenceLedger("test_ledger")


def make_regime_classifier() -> ExecutionRegimeClassifier:
    return ExecutionRegimeClassifier()
