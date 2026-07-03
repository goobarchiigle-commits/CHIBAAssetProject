"""LiquidityRealityMonitor — tracks participation rate, ADV usage, spread, fill collapse."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LiquidityObservation:
    obs_id: str             # "liq_" + SHA256[:12]
    session_id: str
    symbol: str
    participation_rate: float   # fraction of volume we consumed
    adv_usage: float            # our volume / average daily volume
    spread_bps: float
    fill_collapse: bool         # True if participation collapsed mid-order
    observed_at: str

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "participation_rate": self.participation_rate,
            "adv_usage": self.adv_usage,
            "spread_bps": self.spread_bps,
            "fill_collapse": self.fill_collapse,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiquidityObservation":
        return cls(**d)


def _obs_id(session_id: str, symbol: str, observed_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "symbol": symbol, "observed_at": observed_at},
                     sort_keys=True, separators=(",", ":"))
    return "liq_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


class LiquidityRealityMonitor:
    """Tracks liquidity reality divergence from research assumptions."""

    def __init__(
        self,
        max_participation_rate: float = 0.25,
        max_adv_usage: float = 0.05,
        collapse_threshold: float = 0.10,
    ):
        if not (0.0 < max_participation_rate <= 1.0):
            raise ValueError("max_participation_rate must be in (0, 1]")
        if not (0.0 < max_adv_usage <= 1.0):
            raise ValueError("max_adv_usage must be in (0, 1]")
        if not (0.0 < collapse_threshold <= 1.0):
            raise ValueError("collapse_threshold must be in (0, 1]")
        self._max_participation = max_participation_rate
        self._max_adv = max_adv_usage
        self._collapse_threshold = collapse_threshold
        self._observations: List[LiquidityObservation] = []
        self._symbol_obs: Dict[str, List[LiquidityObservation]] = {}

    def record(
        self,
        session_id: str,
        symbol: str,
        participation_rate: float,
        adv_usage: float,
        spread_bps: float,
        fill_collapse: bool,
        observed_at: str,
    ) -> LiquidityObservation:
        obs_id = _obs_id(session_id, symbol, observed_at)
        obs = LiquidityObservation(
            obs_id=obs_id,
            session_id=session_id,
            symbol=symbol,
            participation_rate=participation_rate,
            adv_usage=adv_usage,
            spread_bps=spread_bps,
            fill_collapse=fill_collapse,
            observed_at=observed_at,
        )
        self._observations.append(obs)
        self._symbol_obs.setdefault(symbol, []).append(obs)
        return obs

    def _obs(self, symbol: Optional[str]) -> List[LiquidityObservation]:
        if symbol is not None:
            return self._symbol_obs.get(symbol, [])
        return self._observations

    def mean_participation_rate(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.participation_rate for o in obs) / len(obs)

    def mean_adv_usage(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.adv_usage for o in obs) / len(obs)

    def mean_spread_bps(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.spread_bps for o in obs) / len(obs)

    def fill_collapse_rate(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(1 for o in obs if o.fill_collapse) / len(obs)

    def is_over_participating(self, symbol: Optional[str] = None) -> bool:
        return self.mean_participation_rate(symbol) > self._max_participation

    def is_over_adv(self, symbol: Optional[str] = None) -> bool:
        return self.mean_adv_usage(symbol) > self._max_adv

    def has_liquidity_collapse(self, symbol: Optional[str] = None) -> bool:
        return self.fill_collapse_rate(symbol) >= self._collapse_threshold

    def observation_count(self, symbol: Optional[str] = None) -> int:
        return len(self._obs(symbol))

    def all_observations(self) -> List[LiquidityObservation]:
        return sorted(self._observations, key=lambda o: o.observed_at)

    def to_summary(self) -> dict:
        return {
            "observation_count": len(self._observations),
            "mean_participation_rate": self.mean_participation_rate(),
            "mean_adv_usage": self.mean_adv_usage(),
            "mean_spread_bps": self.mean_spread_bps(),
            "fill_collapse_rate": self.fill_collapse_rate(),
            "is_over_participating": self.is_over_participating(),
            "is_over_adv": self.is_over_adv(),
            "has_liquidity_collapse": self.has_liquidity_collapse(),
        }
