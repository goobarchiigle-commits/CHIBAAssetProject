"""
capital_truth.py — Capital truth under execution uncertainty

Computes truly deployable capital accounting for uncertain portfolio state.

Three deployable views:
  conservative_deployable  — worst-case: UNKNOWN orders treated as fully consumed
  optimistic_deployable    — best-case: UNKNOWN orders assumed to resolve to zero
  effective_deployable     — confidence-weighted blend:
                              effective = conservative × confidence
                                        + optimistic × (1 - confidence)

Deployment gate:
  confidence ≥ MIN_CONFIDENCE_TO_DEPLOY → use effective_deployable
  confidence <  MIN_CONFIDENCE_TO_DEPLOY → deployment blocked (return 0)

Invariants:
  1. 0 ≤ conservative_deployable ≤ optimistic_deployable
  2. effective_deployable ∈ [conservative, optimistic] depending on confidence
  3. confirmed_exposure + pending_exposure + uncertain_exposure ≤ actual_equity × 1.10
  4. cash_buffer ≤ actual_equity
  5. deployment_allowed is False when confidence < threshold
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

MIN_CONFIDENCE_TO_DEPLOY: float = 0.70   # below this → block deployment
CASH_BUFFER_DEFAULT: float = 0.10        # 10% minimum cash reserve
EXPOSURE_TOLERANCE: float = 1.10        # allow 10% excess (price movements)


@dataclass
class CapitalTruth:
    """
    Authoritative capital state under execution uncertainty.

    safe_deployable  = actual_equity - confirmed - pending - uncertain - orphan - buffer
    effective_deployable = confidence-weighted blend (see module docstring)
    """
    actual_equity: float = 0.0             # from broker wallet API
    confirmed_exposure: float = 0.0        # FILLED position values
    pending_exposure: float = 0.0          # SUBMITTED/ACKED/PARTIAL_FILL remaining values
    uncertain_exposure: float = 0.0        # UNKNOWN order values
    orphan_exposure: float = 0.0           # broker positions w/o intent record

    cash_buffer_amount: float = 0.0        # absolute JPY cash reserve

    conservative_deployable: float = 0.0   # treat uncertain as consumed
    optimistic_deployable: float = 0.0     # treat uncertain as zero
    effective_deployable: float = 0.0      # confidence-weighted blend

    confidence: float = 0.0               # [0, 1] truth confidence
    deployment_allowed: bool = False       # True only if confidence ≥ threshold
    deployment_block_reason: str = ""      # why deployment is blocked (if any)

    n_unknown_orders: int = 0             # count of UNKNOWN orders
    n_pending_orders: int = 0             # count of SUBMITTED/ACKED/PARTIAL_FILL
    n_orphan_positions: int = 0           # count of orphan positions

    computed_at: str = ""
    schema_version: int = 1


def compute_capital_truth(
    actual_equity: float,
    confirmed_exposure: float,
    pending_exposure: float,
    uncertain_exposure: float,
    orphan_exposure: float,
    confidence: float,
    n_unknown_orders: int = 0,
    n_pending_orders: int = 0,
    n_orphan_positions: int = 0,
    cash_buffer_ratio: float = CASH_BUFFER_DEFAULT,
    min_confidence: float = MIN_CONFIDENCE_TO_DEPLOY,
) -> CapitalTruth:
    """
    Compute CapitalTruth from aggregated exposure components.

    Args:
        actual_equity:       total account equity (JPY) from broker API
        confirmed_exposure:  value of confirmed (FILLED) positions
        pending_exposure:    value of pending (SUBMITTED/ACKED) orders
        uncertain_exposure:  value of UNKNOWN orders
        orphan_exposure:     value of unrecognized broker positions
        confidence:          truth confidence score [0, 1]
        cash_buffer_ratio:   fraction of actual_equity to hold as cash reserve
    """
    eq = max(0.0, actual_equity)
    conf = max(0.0, min(1.0, confidence))

    cash_buffer = eq * max(0.0, min(1.0, cash_buffer_ratio))

    # Conservative: block ALL uncertain (worst-case)
    conservative = max(0.0,
        eq
        - confirmed_exposure
        - pending_exposure
        - uncertain_exposure
        - orphan_exposure
        - cash_buffer
    )

    # Optimistic: ignore uncertain (best-case, assumes cancellation)
    optimistic = max(0.0,
        eq
        - confirmed_exposure
        - pending_exposure
        - cash_buffer
        # uncertain and orphan assumed to zero out
    )

    # Conservative ≤ optimistic invariant
    conservative = min(conservative, optimistic)

    # Confidence-weighted effective
    effective = conservative * conf + optimistic * (1.0 - conf)
    effective = max(conservative, min(optimistic, effective))

    # Deployment gate
    allowed = conf >= min_confidence
    block_reason = ""
    if not allowed:
        block_reason = (
            f"confidence={conf:.3f} < threshold={min_confidence:.3f}"
        )
        effective = 0.0
    elif n_unknown_orders > 0:
        # Still allowed but use conservative
        effective = min(effective, conservative)
        block_reason = f"unknown_orders={n_unknown_orders}: using conservative_deployable"

    return CapitalTruth(
        actual_equity           = round(eq, 2),
        confirmed_exposure      = round(confirmed_exposure, 2),
        pending_exposure        = round(pending_exposure, 2),
        uncertain_exposure      = round(uncertain_exposure, 2),
        orphan_exposure         = round(orphan_exposure, 2),
        cash_buffer_amount      = round(cash_buffer, 2),
        conservative_deployable = round(conservative, 2),
        optimistic_deployable   = round(optimistic, 2),
        effective_deployable    = round(max(0.0, effective), 2),
        confidence              = round(conf, 4),
        deployment_allowed      = allowed,
        deployment_block_reason = block_reason,
        n_unknown_orders        = n_unknown_orders,
        n_pending_orders        = n_pending_orders,
        n_orphan_positions      = n_orphan_positions,
        computed_at             = datetime.now(JST).isoformat(),
    )


def build_capital_truth_from_components(
    actual_equity: float,
    execution_records,      # list[ExecutionRecord]
    orphan_records,         # list[OrphanRecord]
    confidence: float,
    cash_buffer_ratio: float = CASH_BUFFER_DEFAULT,
) -> CapitalTruth:
    """
    Build CapitalTruth directly from execution records + orphan records.
    Convenience wrapper over compute_capital_truth.
    """
    from src.truth.execution_truth import (
        compute_exposure_breakdown, UNCERTAIN_STATES, UNKNOWN_STATES,
    )
    breakdown = compute_exposure_breakdown(execution_records)

    orphan_val = sum(
        getattr(o, "estimated_value", 0.0) for o in orphan_records
    )

    return compute_capital_truth(
        actual_equity      = actual_equity,
        confirmed_exposure = breakdown.confirmed_value,
        pending_exposure   = breakdown.pending_value,
        uncertain_exposure = breakdown.uncertain_value,
        orphan_exposure    = orphan_val,
        confidence         = confidence,
        n_unknown_orders   = breakdown.total_uncertain_orders,
        n_pending_orders   = breakdown.total_pending_orders,
        n_orphan_positions = len(orphan_records),
        cash_buffer_ratio  = cash_buffer_ratio,
    )


# ── Invariant checks ──────────────────────────────────────────────────────────

def check_capital_invariants(ct: CapitalTruth) -> list[str]:
    """Returns list of violated invariants (empty = all OK)."""
    violations: list[str] = []

    if ct.conservative_deployable > ct.optimistic_deployable + 1.0:
        violations.append(
            f"conservative_deployable={ct.conservative_deployable:.0f} "
            f"> optimistic_deployable={ct.optimistic_deployable:.0f}"
        )

    if ct.effective_deployable < 0:
        violations.append(f"effective_deployable={ct.effective_deployable:.0f} < 0")

    if ct.effective_deployable > ct.optimistic_deployable + 1.0:
        violations.append(
            f"effective_deployable={ct.effective_deployable:.0f} "
            f"> optimistic_deployable={ct.optimistic_deployable:.0f}"
        )

    total_exposure = (
        ct.confirmed_exposure + ct.pending_exposure
        + ct.uncertain_exposure + ct.orphan_exposure
    )
    if total_exposure > ct.actual_equity * EXPOSURE_TOLERANCE + 1.0:
        violations.append(
            f"total_exposure={total_exposure:.0f} > "
            f"actual_equity×{EXPOSURE_TOLERANCE}={ct.actual_equity * EXPOSURE_TOLERANCE:.0f}"
        )

    if ct.cash_buffer_amount > ct.actual_equity + 1.0:
        violations.append(
            f"cash_buffer={ct.cash_buffer_amount:.0f} > actual_equity={ct.actual_equity:.0f}"
        )

    if not ct.deployment_allowed and ct.effective_deployable > 0.0:
        violations.append(
            f"deployment_allowed=False but effective_deployable={ct.effective_deployable:.0f} > 0"
        )

    if not (0.0 <= ct.confidence <= 1.0):
        violations.append(f"confidence={ct.confidence} out of [0, 1]")

    return violations


# ── Persistence ───────────────────────────────────────────────────────────────

def save_capital_truth(truth: CapitalTruth, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(truth), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_capital_truth(path: Path) -> Optional[CapitalTruth]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = set(CapitalTruth.__dataclass_fields__)
        return CapitalTruth(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("capital_truth load failed (%s)", exc)
        return None
