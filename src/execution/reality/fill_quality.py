"""FillQualityAnalyzer — tracks fill ratio, partial fills, rejects, reject bursts."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FillObservation:
    obs_id: str             # "fill_" + SHA256[:12]
    session_id: str
    symbol: str
    side: str
    requested_quantity: int
    filled_quantity: int
    fill_ratio: float
    rejected: bool
    partial: bool
    observed_at: str

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "fill_ratio": self.fill_ratio,
            "rejected": self.rejected,
            "partial": self.partial,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FillObservation":
        return cls(**d)


def _obs_id(session_id: str, symbol: str, observed_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "symbol": symbol, "observed_at": observed_at},
                     sort_keys=True, separators=(",", ":"))
    return "fill_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


class FillQualityAnalyzer:
    """Accumulates fill observations; measures quality degradation."""

    def __init__(self, reject_burst_window: int = 5, reject_burst_threshold: float = 0.50):
        if reject_burst_window < 1:
            raise ValueError("reject_burst_window must be >= 1")
        if not (0.0 < reject_burst_threshold <= 1.0):
            raise ValueError("reject_burst_threshold must be in (0, 1]")
        self._burst_window = reject_burst_window
        self._burst_threshold = reject_burst_threshold
        self._observations: List[FillObservation] = []
        self._symbol_obs: Dict[str, List[FillObservation]] = {}

    def record(
        self,
        session_id: str,
        symbol: str,
        side: str,
        requested_quantity: int,
        filled_quantity: int,
        rejected: bool,
        observed_at: str,
    ) -> FillObservation:
        if rejected:
            fill_ratio = 0.0
            filled_quantity = 0
        elif requested_quantity == 0:
            fill_ratio = 0.0
        else:
            fill_ratio = min(1.0, filled_quantity / requested_quantity)
        partial = (not rejected) and (filled_quantity < requested_quantity)
        obs_id = _obs_id(session_id, symbol, observed_at)
        obs = FillObservation(
            obs_id=obs_id,
            session_id=session_id,
            symbol=symbol,
            side=side,
            requested_quantity=requested_quantity,
            filled_quantity=filled_quantity,
            fill_ratio=fill_ratio,
            rejected=rejected,
            partial=partial,
            observed_at=observed_at,
        )
        self._observations.append(obs)
        self._symbol_obs.setdefault(symbol, []).append(obs)
        return obs

    def _obs(self, symbol: Optional[str]) -> List[FillObservation]:
        if symbol is not None:
            return self._symbol_obs.get(symbol, [])
        return self._observations

    def fill_ratio(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 1.0
        return sum(o.fill_ratio for o in obs) / len(obs)

    def partial_fill_rate(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(1 for o in obs if o.partial) / len(obs)

    def reject_rate(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(1 for o in obs if o.rejected) / len(obs)

    def has_reject_burst(self, symbol: Optional[str] = None) -> bool:
        obs = self._obs(symbol)
        if len(obs) < self._burst_window:
            window = obs
        else:
            window = obs[-self._burst_window:]
        if not window:
            return False
        recent_reject_rate = sum(1 for o in window if o.rejected) / len(window)
        return recent_reject_rate >= self._burst_threshold

    def observation_count(self, symbol: Optional[str] = None) -> int:
        return len(self._obs(symbol))

    def all_observations(self) -> List[FillObservation]:
        return sorted(self._observations, key=lambda o: o.observed_at)

    def to_summary(self) -> dict:
        return {
            "observation_count": len(self._observations),
            "fill_ratio": self.fill_ratio(),
            "partial_fill_rate": self.partial_fill_rate(),
            "reject_rate": self.reject_rate(),
            "has_reject_burst": self.has_reject_burst(),
            "burst_window": self._burst_window,
            "burst_threshold": self._burst_threshold,
        }
