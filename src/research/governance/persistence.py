"""FamilyPersistenceTracker — long-term alpha family survival analysis.

Computes:
  - persistence rate (survived / total observations)
  - persistence half-life (geometric decay model)
  - mutation survivorship
  - regime resilience (fraction of observed regimes survived in)
  - family decay trajectories
  - enduring core identification

Goal: identify enduring alpha structures rather than temporary spikes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.research.governance import SCHEMA_VERSION
from src.research.analysis.artifacts import utc_now_iso


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PersistenceRecord:
    """Immutable computed persistence summary for a family."""
    record_id: str              # "pers_" + SHA256[:12] of family_id
    family_id: str
    observation_count: int
    survived_count: int
    persistence_rate: float     # survived_count / observation_count
    half_life_estimate: float   # periods until 50% survival prob; -1.0 = infinite
    mutation_survivorship: float  # survived mutations / total mutations (1.0 if none)
    regime_resilience: float    # distinct survived regimes / distinct observed regimes
    is_enduring_core: bool      # rate>=0.60 AND mutation>=0.50 AND count>=3
    first_seen: str
    last_seen: str

    def to_dict(self) -> dict:
        hl = self.half_life_estimate if self.half_life_estimate != -1.0 else None
        return {
            "record_id": self.record_id,
            "family_id": self.family_id,
            "observation_count": self.observation_count,
            "survived_count": self.survived_count,
            "persistence_rate": self.persistence_rate,
            "half_life_estimate": hl,
            "mutation_survivorship": self.mutation_survivorship,
            "regime_resilience": self.regime_resilience,
            "is_enduring_core": self.is_enduring_core,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersistenceRecord":
        hl = d.get("half_life_estimate")
        return cls(
            record_id=d["record_id"],
            family_id=d["family_id"],
            observation_count=d["observation_count"],
            survived_count=d["survived_count"],
            persistence_rate=d["persistence_rate"],
            half_life_estimate=hl if hl is not None else -1.0,
            mutation_survivorship=d["mutation_survivorship"],
            regime_resilience=d["regime_resilience"],
            is_enduring_core=d["is_enduring_core"],
            first_seen=d["first_seen"],
            last_seen=d["last_seen"],
        )


# ── Tracker ────────────────────────────────────────────────────────────────────


class FamilyPersistenceTracker:
    """Tracks survival observations for alpha families over time.

    Each call to record_observation appends to the family's history.
    Observations are never modified or deleted.
    compute_persistence() is deterministic from accumulated observations.
    """

    def __init__(self) -> None:
        # family_id → list of observation dicts
        self._observations: Dict[str, List[dict]] = {}

    def record_observation(
        self,
        family_id: str,
        survived: bool,
        mutation_id: Optional[str] = None,
        regime: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a survival observation for a family."""
        obs = {
            "survived": bool(survived),
            "mutation_id": mutation_id,
            "regime": regime,
            "timestamp": timestamp or utc_now_iso(),
        }
        self._observations.setdefault(family_id, []).append(obs)

    def compute_persistence(self, family_id: str) -> PersistenceRecord:
        """Compute immutable persistence record from all observations.

        Raises KeyError if no observations exist for family_id.
        """
        obs = self._observations.get(family_id)
        if not obs:
            raise KeyError(f"No observations for family: {family_id!r}")

        observation_count = len(obs)
        survived_count = sum(1 for o in obs if o["survived"])
        rate = survived_count / observation_count

        # Half-life via geometric decay model: P(t) = P0 * p^t → t_½ = ln(2)/(-ln(p))
        if rate >= 1.0 - 1e-9:
            half_life = -1.0  # sentinel for infinity
        elif rate <= 1e-9:
            half_life = 0.0
        else:
            half_life = round(math.log(2) / (-math.log(rate)), 4)

        # Mutation survivorship
        mut_obs = [o for o in obs if o.get("mutation_id") is not None]
        if mut_obs:
            mutation_survivorship = round(
                sum(1 for o in mut_obs if o["survived"]) / len(mut_obs), 6
            )
        else:
            mutation_survivorship = 1.0  # no mutations → no mutation failures

        # Regime resilience
        regime_obs = [o for o in obs if o.get("regime") is not None]
        if regime_obs:
            distinct_regimes = {o["regime"] for o in regime_obs}
            survived_regimes = {o["regime"] for o in regime_obs if o["survived"]}
            regime_resilience = round(len(survived_regimes) / len(distinct_regimes), 6)
        else:
            regime_resilience = 1.0  # no regime observations → no regime failures

        is_enduring = (
            rate >= 0.60
            and mutation_survivorship >= 0.50
            and observation_count >= 3
        )

        timestamps = [o["timestamp"] for o in obs]
        first_seen = min(timestamps)
        last_seen = max(timestamps)

        return PersistenceRecord(
            record_id=_persistence_record_id(family_id),
            family_id=family_id,
            observation_count=observation_count,
            survived_count=survived_count,
            persistence_rate=round(rate, 6),
            half_life_estimate=half_life,
            mutation_survivorship=mutation_survivorship,
            regime_resilience=regime_resilience,
            is_enduring_core=is_enduring,
            first_seen=first_seen,
            last_seen=last_seen,
        )

    def persistence_rate(self, family_id: str) -> float:
        """Quick persistence rate (0.0 if no observations)."""
        obs = self._observations.get(family_id, [])
        if not obs:
            return 0.0
        return round(sum(1 for o in obs if o["survived"]) / len(obs), 6)

    def half_life(self, family_id: str) -> float:
        """Half-life estimate (-1.0=inf). 0.0 if no observations."""
        obs = self._observations.get(family_id, [])
        if not obs:
            return 0.0
        try:
            return self.compute_persistence(family_id).half_life_estimate
        except KeyError:
            return 0.0

    def enduring_cores(self) -> List[str]:
        """Sorted list of family_ids that qualify as enduring cores."""
        result = []
        for fid in sorted(self._observations.keys()):
            try:
                rec = self.compute_persistence(fid)
                if rec.is_enduring_core:
                    result.append(fid)
            except KeyError:
                pass
        return result

    def distinct_regimes(self, family_id: str) -> List[str]:
        """Sorted distinct regimes observed for this family."""
        obs = self._observations.get(family_id, [])
        return sorted({o["regime"] for o in obs if o.get("regime") is not None})

    def decay_trajectories(self) -> Dict[str, List[float]]:
        """family_id → list of running survival rate at each observation step.

        Deterministic: sorted by family_id.
        """
        result: Dict[str, List[float]] = {}
        for fid in sorted(self._observations.keys()):
            obs = self._observations[fid]
            trajectory = []
            for i, o in enumerate(obs, start=1):
                survived_so_far = sum(1 for x in obs[:i] if x["survived"])
                trajectory.append(round(survived_so_far / i, 6))
            result[fid] = trajectory
        return result

    def all_records(self) -> List[PersistenceRecord]:
        """Compute and return persistence records for all families with observations."""
        result = []
        for fid in sorted(self._observations.keys()):
            try:
                result.append(self.compute_persistence(fid))
            except KeyError:
                pass
        return result

    def tracked_families(self) -> List[str]:
        """Sorted list of all family IDs with at least one observation."""
        return sorted(self._observations.keys())

    def to_manifest(self) -> dict:
        records = [r.to_dict() for r in self.all_records()]
        enduring = self.enduring_cores()
        return {
            "schema_version": SCHEMA_VERSION,
            "family_count": len(self._observations),
            "enduring_core_count": len(enduring),
            "enduring_cores": enduring,
            "records": records,
        }


# ── Module helpers ─────────────────────────────────────────────────────────────


def _persistence_record_id(family_id: str) -> str:
    return "pers_" + hashlib.sha256(family_id.encode()).hexdigest()[:12]
