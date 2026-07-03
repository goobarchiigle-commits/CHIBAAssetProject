"""DecisionSnapshot + DeploymentGuard — immutable decision records; fail-closed gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DECISION_ALLOW = "ALLOW"
DECISION_REDUCE = "REDUCE"
DECISION_HALT = "HALT"

ALL_DECISIONS = (DECISION_ALLOW, DECISION_REDUCE, DECISION_HALT)


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_id: str            # "dec_" + SHA256[:12]
    session_id: str
    decision: str               # ALLOW / REDUCE / HALT
    decision_context_hash: str  # 16-char hash of inputs that drove the decision
    reason: str
    regime: str
    slippage_bps: float
    reject_rate: float
    decay_fraction: float
    decided_at: str
    schema_version: str

    def is_blocking(self) -> bool:
        return self.decision == DECISION_HALT

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "decision": self.decision,
            "decision_context_hash": self.decision_context_hash,
            "reason": self.reason,
            "regime": self.regime,
            "slippage_bps": self.slippage_bps,
            "reject_rate": self.reject_rate,
            "decay_fraction": self.decay_fraction,
            "decided_at": self.decided_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionSnapshot":
        return cls(**d)


@dataclass(frozen=True)
class DeploymentThresholds:
    max_slippage_bps: float = 20.0
    max_reject_rate: float = 0.20
    max_decay_fraction: float = 0.60
    halt_slippage_bps: float = 50.0
    halt_reject_rate: float = 0.40
    halt_decay_fraction: float = 0.80
    halt_on_dislocated: bool = True
    halt_on_halted: bool = True

    def to_dict(self) -> dict:
        return {
            "max_slippage_bps": self.max_slippage_bps,
            "max_reject_rate": self.max_reject_rate,
            "max_decay_fraction": self.max_decay_fraction,
            "halt_slippage_bps": self.halt_slippage_bps,
            "halt_reject_rate": self.halt_reject_rate,
            "halt_decay_fraction": self.halt_decay_fraction,
            "halt_on_dislocated": self.halt_on_dislocated,
            "halt_on_halted": self.halt_on_halted,
        }


def _decision_id(session_id: str, decided_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "decided_at": decided_at},
                     sort_keys=True, separators=(",", ":"))
    return "dec_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _context_hash(
    regime: str, slippage_bps: float, reject_rate: float, decay_fraction: float
) -> str:
    raw = json.dumps(
        {"regime": regime, "slippage_bps": slippage_bps,
         "reject_rate": reject_rate, "decay_fraction": decay_fraction},
        sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DeploymentGuard:
    """Fail-closed gate: HALT unless all thresholds are within bounds."""

    DISLOCATED_REGIMES = {"DISLOCATED", "HALTED", "LIQUIDITY_VACUUM"}

    def __init__(self, thresholds: Optional[DeploymentThresholds] = None, schema_version: str = "1.0.0"):
        self._thresholds = thresholds or DeploymentThresholds()
        self._schema_version = schema_version
        self._history: List[DecisionSnapshot] = []

    @property
    def thresholds(self) -> DeploymentThresholds:
        return self._thresholds

    def evaluate(
        self,
        session_id: str,
        regime: str,
        slippage_bps: float,
        reject_rate: float,
        decay_fraction: float,
        decided_at: str,
    ) -> DecisionSnapshot:
        t = self._thresholds
        decision, reason = self._decide(regime, slippage_bps, reject_rate, decay_fraction, t)
        decision_id = _decision_id(session_id, decided_at)
        context_hash = _context_hash(regime, slippage_bps, reject_rate, decay_fraction)
        snap = DecisionSnapshot(
            decision_id=decision_id,
            session_id=session_id,
            decision=decision,
            decision_context_hash=context_hash,
            reason=reason,
            regime=regime,
            slippage_bps=slippage_bps,
            reject_rate=reject_rate,
            decay_fraction=decay_fraction,
            decided_at=decided_at,
            schema_version=self._schema_version,
        )
        self._history.append(snap)
        return snap

    def _decide(
        self, regime: str, slippage_bps: float, reject_rate: float, decay_fraction: float, t: DeploymentThresholds
    ):
        if t.halt_on_halted and regime in self.DISLOCATED_REGIMES:
            return DECISION_HALT, f"regime={regime}"
        if slippage_bps >= t.halt_slippage_bps:
            return DECISION_HALT, f"slippage={slippage_bps:.1f}bps>={t.halt_slippage_bps:.1f}"
        if reject_rate >= t.halt_reject_rate:
            return DECISION_HALT, f"reject_rate={reject_rate:.2f}>={t.halt_reject_rate:.2f}"
        if decay_fraction >= t.halt_decay_fraction:
            return DECISION_HALT, f"decay_fraction={decay_fraction:.2f}>={t.halt_decay_fraction:.2f}"
        if slippage_bps >= t.max_slippage_bps:
            return DECISION_REDUCE, f"slippage={slippage_bps:.1f}bps>={t.max_slippage_bps:.1f}"
        if reject_rate >= t.max_reject_rate:
            return DECISION_REDUCE, f"reject_rate={reject_rate:.2f}>={t.max_reject_rate:.2f}"
        if decay_fraction >= t.max_decay_fraction:
            return DECISION_REDUCE, f"decay_fraction={decay_fraction:.2f}>={t.max_decay_fraction:.2f}"
        return DECISION_ALLOW, "all_metrics_within_bounds"

    def current_decision(self) -> Optional[str]:
        if not self._history:
            return None
        return self._history[-1].decision

    def is_halted(self) -> bool:
        return self.current_decision() == DECISION_HALT

    def history(self) -> List[DecisionSnapshot]:
        return list(self._history)

    def to_dict(self) -> dict:
        return {
            "thresholds": self._thresholds.to_dict(),
            "history_count": len(self._history),
            "current_decision": self.current_decision(),
        }
