"""
confidence.py — Truth confidence scoring

Scores how much the runtime trusts its own portfolio state.
Score ∈ [0, 1]. Deployment gate: MIN_CONFIDENCE_TO_DEPLOY = 0.70 (capital_truth.py).

Weighted components (sum = 1.0):
  broker   (0.50) — live API reachable? 1.0 = yes, 0.30 = no (journal still valid)
  unknown  (0.25) — 1 − (n_unknown / max(1, n_total_active))
  staleness (0.15) — exponential decay since last reconciliation
  orphan   (0.10) — multiplicative penalty per unrecognized broker position

Decay: STALE_HALF_LIFE_SEC = 300 → at 0s=1.00, 5min=0.50, 30min=0.06, 1hr≈0.003
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

BROKER_WEIGHT:    float = 0.50
UNKNOWN_WEIGHT:   float = 0.25
STALE_WEIGHT:     float = 0.15
ORPHAN_WEIGHT:    float = 0.10

BROKER_UNAVAILABLE_FLOOR: float = 0.30   # broker factor when API unreachable
STALE_HALF_LIFE_SEC:      float = 300.0  # 5 minutes
ORPHAN_PENALTY_PER_POS:   float = 0.30   # per-orphan penalty


@dataclass
class ConfidenceScore:
    """Decomposed truth confidence score."""
    score: float                # composite [0, 1]
    broker_factor: float
    unknown_factor: float
    staleness_factor: float
    orphan_factor: float
    broker_available: bool
    n_unknown: int
    n_total_active: int
    reconcile_age_sec: float
    n_orphans: int
    computed_at: str = ""
    schema_version: int = 1


def score_broker_factor(broker_available: bool) -> float:
    return 1.0 if broker_available else BROKER_UNAVAILABLE_FLOOR


def score_unknown_factor(n_unknown: int, n_total_active: int) -> float:
    if n_total_active <= 0:
        return 1.0
    ratio = n_unknown / n_total_active
    return max(0.0, min(1.0, 1.0 - ratio))


def score_staleness_factor(reconcile_age_sec: float) -> float:
    """Exponential decay: halves every STALE_HALF_LIFE_SEC seconds."""
    if reconcile_age_sec <= 0.0:
        return 1.0
    decay = math.log(2.0) / STALE_HALF_LIFE_SEC
    return math.exp(-decay * max(0.0, reconcile_age_sec))


def score_orphan_factor(n_orphans: int) -> float:
    if n_orphans <= 0:
        return 1.0
    return 1.0 / (1.0 + n_orphans * ORPHAN_PENALTY_PER_POS)


def compute_confidence(
    broker_available: bool,
    n_unknown: int,
    n_total_active: int,
    reconcile_age_sec: float,
    n_orphans: int,
) -> ConfidenceScore:
    """
    Compute composite truth confidence score.

    Args:
        broker_available:  True if kabuステーション API responded this cycle
        n_unknown:         count of UNKNOWN-state execution records
        n_total_active:    count of non-terminal execution records
        reconcile_age_sec: seconds since last successful reconciliation (0 = just now)
        n_orphans:         count of broker positions with no matching intent record
    """
    bf  = score_broker_factor(broker_available)
    uf  = score_unknown_factor(n_unknown, n_total_active)
    sf  = score_staleness_factor(reconcile_age_sec)
    of_ = score_orphan_factor(n_orphans)

    composite = (
        BROKER_WEIGHT  * bf
        + UNKNOWN_WEIGHT * uf
        + STALE_WEIGHT   * sf
        + ORPHAN_WEIGHT  * of_
    )
    composite = round(max(0.0, min(1.0, composite)), 4)

    return ConfidenceScore(
        score             = composite,
        broker_factor     = round(bf, 4),
        unknown_factor    = round(uf, 4),
        staleness_factor  = round(sf, 4),
        orphan_factor     = round(of_, 4),
        broker_available  = broker_available,
        n_unknown         = n_unknown,
        n_total_active    = n_total_active,
        reconcile_age_sec = round(reconcile_age_sec, 1),
        n_orphans         = n_orphans,
        computed_at       = datetime.now(JST).isoformat(),
    )


def confidence_from_truth_state(
    broker_available: bool,
    execution_records,       # list[ExecutionRecord]
    orphan_records,          # list[OrphanRecord]
    reconcile_age_sec: float,
) -> ConfidenceScore:
    """
    Convenience wrapper: derive confidence inputs from execution/orphan record lists.

    n_total_active = records not in terminal states
    n_unknown = records in UNKNOWN state
    """
    from src.truth.execution_truth import TERMINAL_STATES, UNKNOWN_STATES

    n_total_active = sum(1 for r in execution_records if r.state not in TERMINAL_STATES)
    n_unknown      = sum(1 for r in execution_records if r.state in UNKNOWN_STATES)
    n_orphans      = len(orphan_records)

    return compute_confidence(
        broker_available  = broker_available,
        n_unknown         = n_unknown,
        n_total_active    = n_total_active,
        reconcile_age_sec = reconcile_age_sec,
        n_orphans         = n_orphans,
    )
