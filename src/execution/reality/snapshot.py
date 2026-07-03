"""Immutable execution observation — captures expected vs. actual fill at nanosecond precision."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from src.execution.reality.clock import TimestampRecord


@dataclass(frozen=True)
class ExpectedFill:
    symbol: str
    side: str               # BUY / SELL
    quantity: int
    expected_price: float
    expected_slippage_bps: float
    expected_latency_ns: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "expected_price": self.expected_price,
            "expected_slippage_bps": self.expected_slippage_bps,
            "expected_latency_ns": self.expected_latency_ns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpectedFill":
        return cls(**d)


@dataclass(frozen=True)
class ActualFill:
    symbol: str
    side: str
    filled_quantity: int
    fill_price: float
    fill_latency_ns: int    # signal_ns → fill_ns
    broker_latency_ns: int  # submit_ns → broker_ns
    rejected: bool
    partial: bool

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "filled_quantity": self.filled_quantity,
            "fill_price": self.fill_price,
            "fill_latency_ns": self.fill_latency_ns,
            "broker_latency_ns": self.broker_latency_ns,
            "rejected": self.rejected,
            "partial": self.partial,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActualFill":
        return cls(**d)


@dataclass(frozen=True)
class ExecutionTimestamps:
    signal_ns: int
    decision_ns: int
    submit_ns: int
    broker_accept_ns: int
    exchange_ns: int
    fill_ns: int

    def total_latency_ns(self) -> int:
        return self.fill_ns - self.signal_ns

    def signal_to_submit_ns(self) -> int:
        return self.submit_ns - self.signal_ns

    def submit_to_fill_ns(self) -> int:
        return self.fill_ns - self.submit_ns

    def to_dict(self) -> dict:
        return {
            "signal_ns": self.signal_ns,
            "decision_ns": self.decision_ns,
            "submit_ns": self.submit_ns,
            "broker_accept_ns": self.broker_accept_ns,
            "exchange_ns": self.exchange_ns,
            "fill_ns": self.fill_ns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionTimestamps":
        return cls(**d)


@dataclass(frozen=True)
class ExecutionSnapshot:
    snapshot_id: str        # "exsnap_" + SHA256[:12]
    session_id: str
    symbol: str
    side: str
    expected: ExpectedFill
    actual: ActualFill
    timestamps: ExecutionTimestamps
    slippage_bps: float     # (fill_price - expected_price) / expected_price * 10000 for BUY
    fill_ratio: float       # filled_quantity / expected_quantity
    schema_version: str
    recorded_at: str        # UTC ISO
    checksum: str

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "side": self.side,
            "expected": self.expected.to_dict(),
            "actual": self.actual.to_dict(),
            "timestamps": self.timestamps.to_dict(),
            "slippage_bps": self.slippage_bps,
            "fill_ratio": self.fill_ratio,
            "schema_version": self.schema_version,
            "recorded_at": self.recorded_at,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionSnapshot":
        return cls(
            snapshot_id=d["snapshot_id"],
            session_id=d["session_id"],
            symbol=d["symbol"],
            side=d["side"],
            expected=ExpectedFill.from_dict(d["expected"]),
            actual=ActualFill.from_dict(d["actual"]),
            timestamps=ExecutionTimestamps.from_dict(d["timestamps"]),
            slippage_bps=d["slippage_bps"],
            fill_ratio=d["fill_ratio"],
            schema_version=d["schema_version"],
            recorded_at=d["recorded_at"],
            checksum=d["checksum"],
        )


def _compute_slippage_bps(expected: ExpectedFill, actual: ActualFill) -> float:
    if expected.expected_price == 0.0:
        return 0.0
    raw = (actual.fill_price - expected.expected_price) / expected.expected_price * 10_000
    if expected.side == "SELL":
        raw = -raw
    return round(raw, 6)


def _compute_fill_ratio(expected: ExpectedFill, actual: ActualFill) -> float:
    if expected.quantity == 0:
        return 0.0
    return min(1.0, actual.filled_quantity / expected.quantity)


def _snapshot_id(session_id: str, symbol: str, signal_ns: int) -> str:
    raw = json.dumps({"session_id": session_id, "symbol": symbol, "signal_ns": signal_ns},
                     sort_keys=True, separators=(",", ":"))
    return "exsnap_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _checksum(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_snapshot(
    session_id: str,
    expected: ExpectedFill,
    actual: ActualFill,
    timestamps: ExecutionTimestamps,
    schema_version: str,
    recorded_at: str,
) -> ExecutionSnapshot:
    snapshot_id = _snapshot_id(session_id, expected.symbol, timestamps.signal_ns)
    slippage_bps = _compute_slippage_bps(expected, actual)
    fill_ratio = _compute_fill_ratio(expected, actual)
    payload = {
        "session_id": session_id,
        "symbol": expected.symbol,
        "side": expected.side,
        "slippage_bps": slippage_bps,
        "fill_ratio": fill_ratio,
        "signal_ns": timestamps.signal_ns,
    }
    checksum = _checksum(payload)
    return ExecutionSnapshot(
        snapshot_id=snapshot_id,
        session_id=session_id,
        symbol=expected.symbol,
        side=expected.side,
        expected=expected,
        actual=actual,
        timestamps=timestamps,
        slippage_bps=slippage_bps,
        fill_ratio=fill_ratio,
        schema_version=schema_version,
        recorded_at=recorded_at,
        checksum=checksum,
    )
