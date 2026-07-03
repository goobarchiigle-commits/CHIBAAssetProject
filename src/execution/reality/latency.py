"""LatencyDriftMonitor — nanosecond-precision latency tracking and drift detection."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LatencyObservation:
    obs_id: str                 # "lat_" + SHA256[:12]
    session_id: str
    symbol: str
    signal_to_order_ns: int     # signal_ns → submit_ns
    order_to_fill_ns: int       # submit_ns → fill_ns
    broker_response_ns: int     # submit_ns → broker_accept_ns
    jitter_ns: int              # |actual_total - expected_total|
    total_latency_ns: int       # signal_ns → fill_ns
    observed_at: str

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "signal_to_order_ns": self.signal_to_order_ns,
            "order_to_fill_ns": self.order_to_fill_ns,
            "broker_response_ns": self.broker_response_ns,
            "jitter_ns": self.jitter_ns,
            "total_latency_ns": self.total_latency_ns,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LatencyObservation":
        return cls(**d)


def _obs_id(session_id: str, symbol: str, observed_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "symbol": symbol, "observed_at": observed_at},
                     sort_keys=True, separators=(",", ":"))
    return "lat_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


class LatencyDriftMonitor:
    """Tracks latency observations; detects drift vs. baseline."""

    def __init__(self, baseline_latency_ns: int = 50_000_000, drift_factor: float = 2.0):
        if baseline_latency_ns <= 0:
            raise ValueError("baseline_latency_ns must be > 0")
        if drift_factor <= 1.0:
            raise ValueError("drift_factor must be > 1.0")
        self._baseline_ns = baseline_latency_ns
        self._drift_factor = drift_factor
        self._observations: List[LatencyObservation] = []
        self._symbol_obs: Dict[str, List[LatencyObservation]] = {}

    def record(
        self,
        session_id: str,
        symbol: str,
        signal_to_order_ns: int,
        order_to_fill_ns: int,
        broker_response_ns: int,
        expected_total_ns: int,
        observed_at: str,
    ) -> LatencyObservation:
        total = signal_to_order_ns + order_to_fill_ns
        jitter = abs(total - expected_total_ns)
        obs_id = _obs_id(session_id, symbol, observed_at)
        obs = LatencyObservation(
            obs_id=obs_id,
            session_id=session_id,
            symbol=symbol,
            signal_to_order_ns=signal_to_order_ns,
            order_to_fill_ns=order_to_fill_ns,
            broker_response_ns=broker_response_ns,
            jitter_ns=jitter,
            total_latency_ns=total,
            observed_at=observed_at,
        )
        self._observations.append(obs)
        self._symbol_obs.setdefault(symbol, []).append(obs)
        return obs

    def _obs(self, symbol: Optional[str]) -> List[LatencyObservation]:
        if symbol is not None:
            return self._symbol_obs.get(symbol, [])
        return self._observations

    def mean_latency_ns(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.total_latency_ns for o in obs) / len(obs)

    def mean_jitter_ns(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.jitter_ns for o in obs) / len(obs)

    def latency_volatility_ns(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if len(obs) < 2:
            return 0.0
        vals = [o.total_latency_ns for o in obs]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(variance)

    def has_latency_drift(self, symbol: Optional[str] = None) -> bool:
        return self.mean_latency_ns(symbol) > self._baseline_ns * self._drift_factor

    def p99_latency_ns(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        vals = sorted(o.total_latency_ns for o in obs)
        idx = max(0, int(math.ceil(0.99 * len(vals))) - 1)
        return float(vals[idx])

    def observation_count(self, symbol: Optional[str] = None) -> int:
        return len(self._obs(symbol))

    def all_observations(self) -> List[LatencyObservation]:
        return sorted(self._observations, key=lambda o: o.observed_at)

    def to_summary(self) -> dict:
        return {
            "observation_count": len(self._observations),
            "mean_latency_ns": self.mean_latency_ns(),
            "mean_jitter_ns": self.mean_jitter_ns(),
            "latency_volatility_ns": self.latency_volatility_ns(),
            "has_latency_drift": self.has_latency_drift(),
            "baseline_latency_ns": self._baseline_ns,
            "drift_factor": self._drift_factor,
        }
