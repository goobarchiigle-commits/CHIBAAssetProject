"""Core immutable data contracts for the survivorship analysis layer."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class SurvivorRecord:
    """Immutable experiment survival record. All collections as sorted tuples."""
    experiment_id: str
    strategy_id: str
    parameter_hash: str
    # sorted tuple of feature name strings
    feature_names: tuple
    # sorted tuple of (regime, sharpe) pairs
    regime_score_pairs: tuple
    score: float
    survived: bool
    # sorted tuple of rejection reason strings
    rejection_reasons: tuple
    # sorted tuple of fold ints
    fold_ids: tuple

    @staticmethod
    def build(
        experiment_id: str,
        strategy_id: str,
        parameter_hash: str,
        feature_names: List[str],
        regime_scores: Dict[str, float],
        score: float,
        survived: bool,
        rejection_reasons: List[str],
        fold_ids: List[int],
    ) -> "SurvivorRecord":
        return SurvivorRecord(
            experiment_id=experiment_id,
            strategy_id=strategy_id,
            parameter_hash=parameter_hash,
            feature_names=tuple(sorted(feature_names)),
            regime_score_pairs=tuple(sorted(regime_scores.items())),
            score=score,
            survived=survived,
            rejection_reasons=tuple(sorted(rejection_reasons)),
            fold_ids=tuple(sorted(fold_ids)),
        )

    def regime_scores(self) -> Dict[str, float]:
        return dict(self.regime_score_pairs)

    def canonical_hash(self) -> str:
        payload = json.dumps({
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "parameter_hash": self.parameter_hash,
            "feature_names": list(self.feature_names),
            "regime_score_pairs": list(self.regime_score_pairs),
            "score": self.score,
            "survived": self.survived,
            "rejection_reasons": list(self.rejection_reasons),
            "fold_ids": list(self.fold_ids),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "parameter_hash": self.parameter_hash,
            "feature_names": list(self.feature_names),
            "regime_score_pairs": [list(p) for p in self.regime_score_pairs],
            "score": self.score,
            "survived": self.survived,
            "rejection_reasons": list(self.rejection_reasons),
            "fold_ids": list(self.fold_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurvivorRecord":
        return cls(
            experiment_id=d["experiment_id"],
            strategy_id=d["strategy_id"],
            parameter_hash=d["parameter_hash"],
            feature_names=tuple(d["feature_names"]),
            regime_score_pairs=tuple(tuple(p) for p in d["regime_score_pairs"]),
            score=float(d["score"]),
            survived=bool(d["survived"]),
            rejection_reasons=tuple(d["rejection_reasons"]),
            fold_ids=tuple(int(f) for f in d["fold_ids"]),
        )


@dataclass(frozen=True)
class TopologyResult:
    experiment_id: str
    topology_fragility_score: float   # 0-1; 1 = isolated spike
    plateau_width: float              # fraction of neighbors within tolerance of score
    neighbor_decay_rate: float        # mean drop from self to neighbors (>=0)
    parameter_elasticity: float       # mean |Δscore| / |Δparam_normalized| per dimension
    surface_entropy: float            # entropy of neighbor score distribution (nats)
    neighborhood_survivorship: float  # fraction of neighbors that survived
    neighbor_count: int

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "topology_fragility_score": self.topology_fragility_score,
            "plateau_width": self.plateau_width,
            "neighbor_decay_rate": self.neighbor_decay_rate,
            "parameter_elasticity": self.parameter_elasticity,
            "surface_entropy": self.surface_entropy,
            "neighborhood_survivorship": self.neighborhood_survivorship,
            "neighbor_count": self.neighbor_count,
        }


@dataclass(frozen=True)
class FeatureSurvivalStats:
    feature_name: str
    total_occurrences: int
    survived_occurrences: int
    feature_survival_rate: float
    feature_persistence_score: float   # 0-1; stability of feature across families
    feature_decay_resistance: float    # 0-1; survival rate sustained vs early
    # sorted tuple of (regime, survival_rate) pairs
    feature_regime_affinity: tuple
    feature_family_frequency: int      # distinct family count

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "total_occurrences": self.total_occurrences,
            "survived_occurrences": self.survived_occurrences,
            "feature_survival_rate": self.feature_survival_rate,
            "feature_persistence_score": self.feature_persistence_score,
            "feature_decay_resistance": self.feature_decay_resistance,
            "feature_regime_affinity": dict(self.feature_regime_affinity),
            "feature_family_frequency": self.feature_family_frequency,
        }


@dataclass(frozen=True)
class RegimeResilienceResult:
    # sorted tuple of ((from_regime, to_regime), retention_rate)
    transition_pairs: tuple
    cross_regime_retention: float
    # sorted tuple of (transition_key, fragility_score)
    transition_fragility: tuple
    regime_concentration: float      # Herfindahl-like; 0=dispersed, 1=all-in-one
    regime_resilience_score: float   # 0-1; higher = more resilient
    recovery_stability: float        # retention across panic→non-panic transitions
    single_regime_dependent: bool
    panic_only_alpha: bool
    trend_only_alpha: bool
    vol_sensitive_alpha: bool

    def to_dict(self) -> dict:
        return {
            "transition_pairs": [list(p) for p in self.transition_pairs],
            "cross_regime_retention": self.cross_regime_retention,
            "transition_fragility": dict(self.transition_fragility),
            "regime_concentration": self.regime_concentration,
            "regime_resilience_score": self.regime_resilience_score,
            "recovery_stability": self.recovery_stability,
            "single_regime_dependent": self.single_regime_dependent,
            "panic_only_alpha": self.panic_only_alpha,
            "trend_only_alpha": self.trend_only_alpha,
            "vol_sensitive_alpha": self.vol_sensitive_alpha,
        }


@dataclass(frozen=True)
class FamilyRecord:
    family_id: str
    strategy_id: str
    member_experiment_ids: tuple   # sorted
    feature_signature: tuple       # sorted feature names common to majority of members
    regime_affinity: tuple         # sorted regimes where mean family sharpe > 0
    family_survivorship: float     # fraction of members that survived
    family_persistence_score: float  # mean score of surviving members
    is_robust_core: bool           # survivorship >= 0.6 AND persistence >= 0.5

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "strategy_id": self.strategy_id,
            "member_experiment_ids": list(self.member_experiment_ids),
            "feature_signature": list(self.feature_signature),
            "regime_affinity": list(self.regime_affinity),
            "family_survivorship": self.family_survivorship,
            "family_persistence_score": self.family_persistence_score,
            "is_robust_core": self.is_robust_core,
        }


@dataclass
class SurvivorshipLedgerEntry:
    ledger_id: str
    generated_at: str   # ISO8601 UTC
    schema_version: str
    experiment_count: int
    survivor_count: int
    family_count: int
    checksum: str
    payload: dict

    def to_dict(self) -> dict:
        return {
            "ledger_id": self.ledger_id,
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
            "experiment_count": self.experiment_count,
            "survivor_count": self.survivor_count,
            "family_count": self.family_count,
            "checksum": self.checksum,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurvivorshipLedgerEntry":
        return cls(
            ledger_id=d["ledger_id"],
            generated_at=d["generated_at"],
            schema_version=d["schema_version"],
            experiment_count=int(d["experiment_count"]),
            survivor_count=int(d["survivor_count"]),
            family_count=int(d["family_count"]),
            checksum=d["checksum"],
            payload=d["payload"],
        )
