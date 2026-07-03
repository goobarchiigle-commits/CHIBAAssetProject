"""
invariants.py — Runtime truth invariant checks

Cross-layer assertions over the full truth state.
Returns violation strings (empty list = all invariants satisfied).

Violation severity:
  ERROR  — deployment must be blocked (capital or position incoherence)
  WARN   — anomaly flagged, monitor but may not block

Usage:
  violations = check_all_invariants(truth_state)
  if violations:
      logger.error("INVARIANT VIOLATIONS: %s", violations)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.truth.reconciler import TruthState

logger = logging.getLogger(__name__)


def check_execution_invariants(records) -> list[str]:  # list[ExecutionRecord]
    """
    Per-record invariants:
      - filled_qty ≤ submitted_qty ≤ requested_qty (when submitted_qty > 0)
      - fill_price > 0 iff filled_qty > 0
      - no duplicate intent_ids
      - state is a recognized value
    """
    from src.truth.execution_truth import ALL_VALID_STATES

    violations: list[str] = []
    seen_ids: dict[str, str] = {}

    for rec in records:
        iid = rec.intent_id

        if iid in seen_ids:
            violations.append(
                f"[EXEC] duplicate intent_id={iid} "
                f"(states: {seen_ids[iid]}, {rec.state})"
            )
        seen_ids[iid] = rec.state

        if rec.state not in ALL_VALID_STATES:
            violations.append(
                f"[EXEC] {iid}: unknown state={rec.state!r}"
            )

        if rec.submitted_qty > 0:
            if rec.filled_qty > rec.submitted_qty:
                violations.append(
                    f"[EXEC] {iid}: filled_qty={rec.filled_qty} > "
                    f"submitted_qty={rec.submitted_qty}"
                )
            if rec.submitted_qty > rec.requested_qty:
                violations.append(
                    f"[EXEC] {iid}: submitted_qty={rec.submitted_qty} > "
                    f"requested_qty={rec.requested_qty}"
                )

        if rec.filled_qty > 0 and rec.fill_price <= 0:
            violations.append(
                f"[EXEC] {iid}: filled_qty={rec.filled_qty} > 0 "
                f"but fill_price={rec.fill_price} ≤ 0"
            )
        if rec.filled_qty == 0 and rec.fill_price < 0:
            violations.append(
                f"[EXEC] {iid}: fill_price={rec.fill_price} < 0 with no fills"
            )

        if rec.remaining_qty < 0:
            violations.append(
                f"[EXEC] {iid}: remaining_qty={rec.remaining_qty} < 0"
            )

    return violations


def check_position_invariants(position_truth) -> list[str]:  # PositionTruth
    """
    Per-position invariants:
      - confirmed_lots ≥ 0
      - pending_lots ≥ 0, uncertain_lots ≥ 0
      - avg_cost_price > 0 if confirmed_lots > 0
      - orphans not present in positions (no silent merge)
    """
    violations: list[str] = []
    position_symbols = {p.symbol for p in position_truth.positions}
    orphan_symbols   = {o.symbol for o in position_truth.orphans}

    for pos in position_truth.positions:
        if pos.confirmed_lots < 0:
            violations.append(
                f"[POS] {pos.symbol}: confirmed_lots={pos.confirmed_lots} < 0"
            )
        if pos.pending_lots < 0:
            violations.append(
                f"[POS] {pos.symbol}: pending_lots={pos.pending_lots} < 0"
            )
        if pos.uncertain_lots < 0:
            violations.append(
                f"[POS] {pos.symbol}: uncertain_lots={pos.uncertain_lots} < 0"
            )
        if pos.confirmed_lots > 0 and pos.avg_cost_price <= 0:
            violations.append(
                f"[POS] {pos.symbol}: confirmed_lots={pos.confirmed_lots} > 0 "
                f"but avg_cost_price={pos.avg_cost_price} ≤ 0"
            )
        if not (0.0 <= pos.confidence <= 1.0):
            violations.append(
                f"[POS] {pos.symbol}: confidence={pos.confidence} out of [0, 1]"
            )

    for orph in position_truth.orphans:
        if orph.lots < 0:
            violations.append(
                f"[POS] orphan {orph.symbol}: lots={orph.lots} < 0"
            )
        if orph.symbol in position_symbols:
            violations.append(
                f"[POS] orphan {orph.symbol} silently merged into positions — "
                "invariant violated"
            )

    return violations


def check_capital_layer_invariants(capital_truth) -> list[str]:  # CapitalTruth
    """Delegates to capital_truth.check_capital_invariants with [CAP] prefix."""
    from src.truth.capital_truth import check_capital_invariants
    raw = check_capital_invariants(capital_truth)
    return [f"[CAP] {v}" for v in raw]


def check_cross_layer_invariants(
    records,          # list[ExecutionRecord]
    position_truth,   # PositionTruth
    capital_truth,    # CapitalTruth
    confidence,       # ConfidenceScore
) -> list[str]:
    """
    Cross-layer consistency:
      - total_exposure consistent between execution breakdown and capital truth
      - deployment block when confidence < threshold
      - no deployment if chain invalid (caller should pass flag)
    """
    from src.truth.execution_truth import compute_exposure_breakdown
    from src.truth.capital_truth import MIN_CONFIDENCE_TO_DEPLOY

    violations: list[str] = []
    breakdown = compute_exposure_breakdown(records)

    # Exposure consistency: allow 1.0 JPY rounding tolerance
    tolerance = 1.0

    if abs(breakdown.confirmed_value - capital_truth.confirmed_exposure) > tolerance:
        violations.append(
            f"[CROSS] confirmed_value mismatch: "
            f"breakdown={breakdown.confirmed_value:.0f} "
            f"vs capital_truth={capital_truth.confirmed_exposure:.0f}"
        )
    if abs(breakdown.pending_value - capital_truth.pending_exposure) > tolerance:
        violations.append(
            f"[CROSS] pending_value mismatch: "
            f"breakdown={breakdown.pending_value:.0f} "
            f"vs capital_truth={capital_truth.pending_exposure:.0f}"
        )
    if abs(breakdown.uncertain_value - capital_truth.uncertain_exposure) > tolerance:
        violations.append(
            f"[CROSS] uncertain_value mismatch: "
            f"breakdown={breakdown.uncertain_value:.0f} "
            f"vs capital_truth={capital_truth.uncertain_exposure:.0f}"
        )

    if not capital_truth.deployment_allowed and capital_truth.effective_deployable > 0.0:
        violations.append(
            f"[CROSS] deployment_allowed=False but effective_deployable="
            f"{capital_truth.effective_deployable:.0f} > 0"
        )

    if confidence.score < MIN_CONFIDENCE_TO_DEPLOY and capital_truth.deployment_allowed:
        violations.append(
            f"[CROSS] confidence={confidence.score:.3f} < threshold "
            f"{MIN_CONFIDENCE_TO_DEPLOY:.3f} but deployment_allowed=True"
        )

    return violations


def check_all_invariants(truth_state) -> list[str]:  # TruthState
    """
    Run all invariant checks. Returns all violations across all layers.
    Empty list = runtime state is coherent.
    """
    violations: list[str] = []

    violations.extend(check_execution_invariants(truth_state.execution_records))
    violations.extend(check_position_invariants(truth_state.position_truth))
    violations.extend(check_capital_layer_invariants(truth_state.capital_truth))
    violations.extend(check_cross_layer_invariants(
        truth_state.execution_records,
        truth_state.position_truth,
        truth_state.capital_truth,
        truth_state.confidence,
    ))

    if not truth_state.chain_valid:
        for cv in truth_state.chain_violations:
            violations.append(f"[CHAIN] {cv}")

    return violations


def assert_no_violations(truth_state, raise_on_violation: bool = False) -> None:
    """
    Run all invariant checks. Logs ERROR per violation.
    If raise_on_violation=True, raises RuntimeError on any violation.
    """
    violations = check_all_invariants(truth_state)
    for v in violations:
        logger.error("TRUTH INVARIANT VIOLATION: %s", v)
    if raise_on_violation and violations:
        raise RuntimeError(
            f"Truth invariant violations ({len(violations)}): {violations[0]}"
        )
