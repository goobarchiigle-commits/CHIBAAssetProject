"""DivergenceLedger — append-only record of execution divergence events."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Divergence types
DIV_SLIPPAGE_DRIFT = "SLIPPAGE_DRIFT"
DIV_LATENCY_DRIFT = "LATENCY_DRIFT"
DIV_SPREAD_WIDENING = "SPREAD_WIDENING"
DIV_REJECT_CLUSTERING = "REJECT_CLUSTERING"
DIV_LIQUIDITY_COLLAPSE = "LIQUIDITY_COLLAPSE"
DIV_SIGNAL_TIMING_DRIFT = "SIGNAL_TIMING_DRIFT"
DIV_QUEUE_DEGRADATION = "QUEUE_DEGRADATION"

ALL_DIVERGENCE_TYPES = (
    DIV_SLIPPAGE_DRIFT,
    DIV_LATENCY_DRIFT,
    DIV_SPREAD_WIDENING,
    DIV_REJECT_CLUSTERING,
    DIV_LIQUIDITY_COLLAPSE,
    DIV_SIGNAL_TIMING_DRIFT,
    DIV_QUEUE_DEGRADATION,
)

SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DivergenceRecord:
    record_id: str          # "div_" + SHA256[:12]
    ledger_id: str
    session_id: str
    symbol: str
    divergence_type: str
    severity: str
    observed_value: float
    baseline_value: float
    divergence_ratio: float # observed / baseline; inf-safe
    recorded_at: str
    metadata: Tuple[Tuple[str, str], ...]

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "ledger_id": self.ledger_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "divergence_type": self.divergence_type,
            "severity": self.severity,
            "observed_value": self.observed_value,
            "baseline_value": self.baseline_value,
            "divergence_ratio": self.divergence_ratio,
            "recorded_at": self.recorded_at,
            "metadata": list(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DivergenceRecord":
        return cls(
            record_id=d["record_id"],
            ledger_id=d["ledger_id"],
            session_id=d["session_id"],
            symbol=d["symbol"],
            divergence_type=d["divergence_type"],
            severity=d["severity"],
            observed_value=d["observed_value"],
            baseline_value=d["baseline_value"],
            divergence_ratio=d["divergence_ratio"],
            recorded_at=d["recorded_at"],
            metadata=tuple(tuple(x) for x in d["metadata"]),
        )


def _record_id(ledger_id: str, session_id: str, symbol: str, divergence_type: str, recorded_at: str) -> str:
    raw = json.dumps(
        {"ledger_id": ledger_id, "session_id": session_id, "symbol": symbol,
         "divergence_type": divergence_type, "recorded_at": recorded_at},
        sort_keys=True, separators=(",", ":")
    )
    return "div_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _divergence_ratio(observed: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if observed == 0.0 else float("inf")
    return observed / baseline


class DivergenceLedger:
    """Append-only ledger of execution divergence events."""

    def __init__(self, ledger_id: str = "default_div_ledger"):
        self._ledger_id = ledger_id
        self._records: List[DivergenceRecord] = []
        self._type_index: Dict[str, List[DivergenceRecord]] = {}
        self._symbol_index: Dict[str, List[DivergenceRecord]] = {}
        self._session_index: Dict[str, List[DivergenceRecord]] = {}

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    def record_divergence(
        self,
        session_id: str,
        symbol: str,
        divergence_type: str,
        severity: str,
        observed_value: float,
        baseline_value: float,
        timestamp: str,
        metadata: Optional[dict] = None,
    ) -> DivergenceRecord:
        if divergence_type not in ALL_DIVERGENCE_TYPES:
            raise ValueError(f"Unknown divergence type: {divergence_type!r}")
        meta_tuple: Tuple[Tuple[str, str], ...] = tuple(
            sorted((k, str(v)) for k, v in (metadata or {}).items())
        )
        record_id = _record_id(self._ledger_id, session_id, symbol, divergence_type, timestamp)
        ratio = _divergence_ratio(observed_value, baseline_value)
        rec = DivergenceRecord(
            record_id=record_id,
            ledger_id=self._ledger_id,
            session_id=session_id,
            symbol=symbol,
            divergence_type=divergence_type,
            severity=severity,
            observed_value=observed_value,
            baseline_value=baseline_value,
            divergence_ratio=ratio,
            recorded_at=timestamp,
            metadata=meta_tuple,
        )
        self._records.append(rec)
        self._type_index.setdefault(divergence_type, []).append(rec)
        self._symbol_index.setdefault(symbol, []).append(rec)
        self._session_index.setdefault(session_id, []).append(rec)
        return rec

    def all_records(self) -> List[DivergenceRecord]:
        return sorted(self._records, key=lambda r: r.recorded_at)

    def records_of_type(self, divergence_type: str) -> List[DivergenceRecord]:
        return sorted(self._type_index.get(divergence_type, []), key=lambda r: r.recorded_at)

    def records_for_symbol(self, symbol: str) -> List[DivergenceRecord]:
        return sorted(self._symbol_index.get(symbol, []), key=lambda r: r.recorded_at)

    def records_for_session(self, session_id: str) -> List[DivergenceRecord]:
        return sorted(self._session_index.get(session_id, []), key=lambda r: r.recorded_at)

    def divergence_count(self, divergence_type: Optional[str] = None) -> int:
        if divergence_type is None:
            return len(self._records)
        return len(self._type_index.get(divergence_type, []))

    def has_critical(self) -> bool:
        return any(r.severity == SEVERITY_CRITICAL for r in self._records)

    def severity_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._records:
            counts[r.severity] = counts.get(r.severity, 0) + 1
        return counts

    def to_summary(self) -> dict:
        return {
            "ledger_id": self._ledger_id,
            "total_records": len(self._records),
            "severity_counts": self.severity_counts(),
            "has_critical": self.has_critical(),
            "divergence_types": sorted(self._type_index.keys()),
        }
