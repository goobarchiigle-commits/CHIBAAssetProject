"""AlphaDecayTracker — tracks decay of research alpha signal in live execution."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DECAY_RESEARCH_ALPHA = "RESEARCH_ALPHA_DECAY"
DECAY_EXECUTION_INDUCED = "EXECUTION_INDUCED_DECAY"

ALL_DECAY_TYPES = (DECAY_RESEARCH_ALPHA, DECAY_EXECUTION_INDUCED)


@dataclass(frozen=True)
class AlphaDecayObservation:
    obs_id: str             # "adcy_" + SHA256[:12]
    session_id: str
    symbol: str
    decay_type: str
    research_alpha_bps: float   # expected alpha from research
    realized_alpha_bps: float   # actual alpha captured in live
    decay_bps: float            # research - realized (positive = decay)
    decay_fraction: float       # decay_bps / |research_alpha_bps|, capped [0,1]
    observed_at: str

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "decay_type": self.decay_type,
            "research_alpha_bps": self.research_alpha_bps,
            "realized_alpha_bps": self.realized_alpha_bps,
            "decay_bps": self.decay_bps,
            "decay_fraction": self.decay_fraction,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlphaDecayObservation":
        return cls(**d)


def _obs_id(session_id: str, symbol: str, observed_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "symbol": symbol, "observed_at": observed_at},
                     sort_keys=True, separators=(",", ":"))
    return "adcy_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


class AlphaDecayTracker:
    """Accumulates alpha decay observations; attributes decay to source."""

    def __init__(self, decay_alert_threshold: float = 0.50):
        if not (0.0 <= decay_alert_threshold <= 1.0):
            raise ValueError("decay_alert_threshold must be in [0, 1]")
        self._alert_threshold = decay_alert_threshold
        self._observations: List[AlphaDecayObservation] = []
        self._type_obs: Dict[str, List[AlphaDecayObservation]] = {}
        self._symbol_obs: Dict[str, List[AlphaDecayObservation]] = {}

    def record(
        self,
        session_id: str,
        symbol: str,
        decay_type: str,
        research_alpha_bps: float,
        realized_alpha_bps: float,
        observed_at: str,
    ) -> AlphaDecayObservation:
        if decay_type not in ALL_DECAY_TYPES:
            raise ValueError(f"Unknown decay type: {decay_type!r}")
        decay_bps = research_alpha_bps - realized_alpha_bps
        if research_alpha_bps == 0.0:
            decay_fraction = 0.0
        else:
            decay_fraction = min(1.0, max(0.0, decay_bps / abs(research_alpha_bps)))
        obs_id = _obs_id(session_id, symbol, observed_at)
        obs = AlphaDecayObservation(
            obs_id=obs_id,
            session_id=session_id,
            symbol=symbol,
            decay_type=decay_type,
            research_alpha_bps=research_alpha_bps,
            realized_alpha_bps=realized_alpha_bps,
            decay_bps=decay_bps,
            decay_fraction=decay_fraction,
            observed_at=observed_at,
        )
        self._observations.append(obs)
        self._type_obs.setdefault(decay_type, []).append(obs)
        self._symbol_obs.setdefault(symbol, []).append(obs)
        return obs

    def _obs(self, symbol: Optional[str] = None, decay_type: Optional[str] = None) -> List[AlphaDecayObservation]:
        if symbol is not None:
            return self._symbol_obs.get(symbol, [])
        if decay_type is not None:
            return self._type_obs.get(decay_type, [])
        return self._observations

    def mean_decay_bps(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.decay_bps for o in obs) / len(obs)

    def mean_decay_fraction(self, symbol: Optional[str] = None) -> float:
        obs = self._obs(symbol)
        if not obs:
            return 0.0
        return sum(o.decay_fraction for o in obs) / len(obs)

    def attribute_decay(self) -> Dict[str, float]:
        """Mean decay_bps attributed to each decay type."""
        result: Dict[str, float] = {}
        for decay_type in ALL_DECAY_TYPES:
            obs = self._type_obs.get(decay_type, [])
            if obs:
                result[decay_type] = sum(o.decay_bps for o in obs) / len(obs)
        return result

    def has_significant_decay(self, symbol: Optional[str] = None) -> bool:
        return self.mean_decay_fraction(symbol) >= self._alert_threshold

    def observation_count(self, symbol: Optional[str] = None) -> int:
        return len(self._obs(symbol))

    def all_observations(self) -> List[AlphaDecayObservation]:
        return sorted(self._observations, key=lambda o: o.observed_at)

    def to_summary(self) -> dict:
        return {
            "observation_count": len(self._observations),
            "mean_decay_bps": self.mean_decay_bps(),
            "mean_decay_fraction": self.mean_decay_fraction(),
            "has_significant_decay": self.has_significant_decay(),
            "decay_attribution": self.attribute_decay(),
            "alert_threshold": self._alert_threshold,
        }
