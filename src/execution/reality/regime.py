"""ExecutionRegimeClassifier — classify current execution environment."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

REGIME_NORMAL = "NORMAL"
REGIME_THIN = "THIN"
REGIME_STRESSED = "STRESSED"
REGIME_DISLOCATED = "DISLOCATED"
REGIME_OPEN_AUCTION = "OPEN_AUCTION"
REGIME_CLOSE_AUCTION = "CLOSE_AUCTION"
REGIME_HALTED = "HALTED"
REGIME_LIQUIDITY_VACUUM = "LIQUIDITY_VACUUM"

ALL_REGIMES = (
    REGIME_NORMAL, REGIME_THIN, REGIME_STRESSED, REGIME_DISLOCATED,
    REGIME_OPEN_AUCTION, REGIME_CLOSE_AUCTION, REGIME_HALTED, REGIME_LIQUIDITY_VACUUM,
)


@dataclass(frozen=True)
class RegimeClassification:
    classification_id: str      # "rgm_" + SHA256[:12]
    regime: str
    confidence: float           # [0.0, 1.0]
    reason: str
    spread_bps: float
    participation_rate: float
    reject_rate: float
    slippage_bps: float
    classified_at: str
    schema_version: str

    def is_tradeable(self) -> bool:
        return self.regime in (REGIME_NORMAL, REGIME_THIN)

    def to_dict(self) -> dict:
        return {
            "classification_id": self.classification_id,
            "regime": self.regime,
            "confidence": self.confidence,
            "reason": self.reason,
            "spread_bps": self.spread_bps,
            "participation_rate": self.participation_rate,
            "reject_rate": self.reject_rate,
            "slippage_bps": self.slippage_bps,
            "classified_at": self.classified_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeClassification":
        return cls(**d)


def _classification_id(regime: str, classified_at: str, spread_bps: float) -> str:
    raw = json.dumps({"regime": regime, "classified_at": classified_at, "spread_bps": spread_bps},
                     sort_keys=True, separators=(",", ":"))
    return "rgm_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RegimeThresholds:
    thin_spread_bps: float = 30.0
    stressed_spread_bps: float = 100.0
    dislocated_spread_bps: float = 300.0
    thin_participation: float = 0.20
    stressed_reject_rate: float = 0.15
    vacuum_participation: float = 0.05
    halted_participation: float = 0.0
    auction_reject_rate: float = 0.50

    def to_dict(self) -> dict:
        return {
            "thin_spread_bps": self.thin_spread_bps,
            "stressed_spread_bps": self.stressed_spread_bps,
            "dislocated_spread_bps": self.dislocated_spread_bps,
            "thin_participation": self.thin_participation,
            "stressed_reject_rate": self.stressed_reject_rate,
            "vacuum_participation": self.vacuum_participation,
            "halted_participation": self.halted_participation,
            "auction_reject_rate": self.auction_reject_rate,
        }


class ExecutionRegimeClassifier:
    def __init__(self, thresholds: Optional[RegimeThresholds] = None, schema_version: str = "1.0.0"):
        self._thresholds = thresholds or RegimeThresholds()
        self._schema_version = schema_version
        self._history: list = []

    @property
    def thresholds(self) -> RegimeThresholds:
        return self._thresholds

    def classify(
        self,
        spread_bps: float,
        participation_rate: float,
        reject_rate: float,
        slippage_bps: float,
        classified_at: str,
        *,
        is_open_auction: bool = False,
        is_close_auction: bool = False,
    ) -> RegimeClassification:
        t = self._thresholds
        regime, reason, confidence = self._classify(
            spread_bps, participation_rate, reject_rate, slippage_bps,
            is_open_auction, is_close_auction, t
        )
        classification_id = _classification_id(regime, classified_at, spread_bps)
        rc = RegimeClassification(
            classification_id=classification_id,
            regime=regime,
            confidence=confidence,
            reason=reason,
            spread_bps=spread_bps,
            participation_rate=participation_rate,
            reject_rate=reject_rate,
            slippage_bps=slippage_bps,
            classified_at=classified_at,
            schema_version=self._schema_version,
        )
        self._history.append(rc)
        return rc

    def _classify(
        self, spread_bps, participation_rate, reject_rate, slippage_bps,
        is_open_auction, is_close_auction, t
    ) -> Tuple[str, str, float]:
        if participation_rate <= t.halted_participation and not is_open_auction and not is_close_auction:
            return REGIME_HALTED, "zero_participation", 1.0
        if participation_rate < t.vacuum_participation:
            return REGIME_LIQUIDITY_VACUUM, "near_zero_participation", 0.95
        if is_open_auction:
            return REGIME_OPEN_AUCTION, "open_auction_flag", 1.0
        if is_close_auction:
            return REGIME_CLOSE_AUCTION, "close_auction_flag", 1.0
        if spread_bps >= t.dislocated_spread_bps:
            return REGIME_DISLOCATED, "extreme_spread", 0.95
        if spread_bps >= t.stressed_spread_bps or reject_rate >= t.stressed_reject_rate:
            return REGIME_STRESSED, "stressed_spread_or_reject", 0.85
        if spread_bps >= t.thin_spread_bps or participation_rate < t.thin_participation:
            return REGIME_THIN, "thin_market", 0.80
        return REGIME_NORMAL, "normal_conditions", 0.90

    def history(self) -> list:
        return list(self._history)

    def current_regime(self) -> Optional[str]:
        if not self._history:
            return None
        return self._history[-1].regime

    def to_dict(self) -> dict:
        return {
            "thresholds": self._thresholds.to_dict(),
            "history_count": len(self._history),
            "current_regime": self.current_regime(),
        }
