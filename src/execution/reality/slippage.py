"""SlippageTracker — accumulates fill-level slippage observations, detects structural drift."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SlippageObservation:
    obs_id: str             # "slip_" + SHA256[:12]
    session_id: str
    symbol: str
    side: str
    slippage_bps: float
    expected_price: float
    fill_price: float
    observed_at: str

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "side": self.side,
            "slippage_bps": self.slippage_bps,
            "expected_price": self.expected_price,
            "fill_price": self.fill_price,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlippageObservation":
        return cls(**d)


def _obs_id(session_id: str, symbol: str, observed_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "symbol": symbol, "observed_at": observed_at},
                     sort_keys=True, separators=(",", ":"))
    return "slip_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


class SlippageTracker:
    """Accumulates slippage observations; computes mean, tail, volatility, drift flag."""

    def __init__(self, drift_threshold_bps: float = 5.0, tail_percentile: float = 0.95):
        if drift_threshold_bps < 0:
            raise ValueError("drift_threshold_bps must be >= 0")
        if not (0.0 < tail_percentile < 1.0):
            raise ValueError("tail_percentile must be in (0, 1)")
        self._drift_threshold = drift_threshold_bps
        self._tail_percentile = tail_percentile
        self._observations: List[SlippageObservation] = []
        self._symbol_obs: Dict[str, List[SlippageObservation]] = {}

    def record(
        self,
        session_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        fill_price: float,
        observed_at: str,
    ) -> SlippageObservation:
        if expected_price == 0.0:
            slippage_bps = 0.0
        else:
            raw = (fill_price - expected_price) / expected_price * 10_000
            slippage_bps = raw if side == "BUY" else -raw
        obs_id = _obs_id(session_id, symbol, observed_at)
        obs = SlippageObservation(
            obs_id=obs_id,
            session_id=session_id,
            symbol=symbol,
            side=side,
            slippage_bps=slippage_bps,
            expected_price=expected_price,
            fill_price=fill_price,
            observed_at=observed_at,
        )
        self._observations.append(obs)
        self._symbol_obs.setdefault(symbol, []).append(obs)
        return obs

    def _bps_list(self, symbol: Optional[str] = None) -> List[float]:
        if symbol is not None:
            return [o.slippage_bps for o in self._symbol_obs.get(symbol, [])]
        return [o.slippage_bps for o in self._observations]

    def mean_slippage(self, symbol: Optional[str] = None) -> float:
        vals = self._bps_list(symbol)
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def tail_slippage(self, symbol: Optional[str] = None) -> float:
        vals = sorted(self._bps_list(symbol))
        if not vals:
            return 0.0
        idx = max(0, int(math.ceil(self._tail_percentile * len(vals))) - 1)
        return vals[idx]

    def slippage_volatility(self, symbol: Optional[str] = None) -> float:
        vals = self._bps_list(symbol)
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(variance)

    def has_structural_drift(self, symbol: Optional[str] = None) -> bool:
        return self.mean_slippage(symbol) > self._drift_threshold

    def observation_count(self, symbol: Optional[str] = None) -> int:
        if symbol is not None:
            return len(self._symbol_obs.get(symbol, []))
        return len(self._observations)

    def all_observations(self) -> List[SlippageObservation]:
        return sorted(self._observations, key=lambda o: o.observed_at)

    def to_summary(self) -> dict:
        return {
            "observation_count": len(self._observations),
            "mean_slippage_bps": self.mean_slippage(),
            "tail_slippage_bps": self.tail_slippage(),
            "slippage_volatility": self.slippage_volatility(),
            "has_structural_drift": self.has_structural_drift(),
            "drift_threshold_bps": self._drift_threshold,
        }
