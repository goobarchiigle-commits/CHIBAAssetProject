"""
sizing.py — Exposure-aware sizing with concentration throttling

Applies Phase 5A intelligence to order sizing:
  - Truth-layer confidence gate (< 0.70 → fallback mode, final_size=0)
  - Stale state detection (age > STALE_STATE_AGE_SEC → fallback, allow original size)
  - Effective-N-based allocation cap
  - Concentration throttle (can only REDUCE qty, never increase)

All functions are pure. No I/O. Fail-closed on uncertainty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TRUTH_CONFIDENCE_GATE: float = 0.70
STALE_STATE_AGE_SEC: float   = 3600.0    # 1 hour
MAX_CONCENTRATION_PCT: float = 0.25      # mirrors CLAUDE.md max_single_weight
MIN_EFFECTIVE_N_FULL: float  = 2.0       # below this → cap at 80% of requested


@dataclass
class SizingResult:
    """Output of compute_exposure_aware_size. Pure value object."""
    requested_size: int
    final_size: int
    allocation_cap: int
    throttle_applied: bool
    throttle_reason: str
    fallback_mode: bool
    fallback_reason: str
    stale_state_detected: bool
    allocation_state_age_sec: float
    effective_independent_n: float
    portfolio_concentration: float
    concentration_penalty: float
    capital_efficiency_score: float
    execution_allowed: bool


def gate_on_truth_confidence(truth_confidence: float) -> tuple[bool, str]:
    """Returns (gate_passed, reason). False → block execution."""
    if truth_confidence < TRUTH_CONFIDENCE_GATE:
        return False, (
            f"truth_confidence={truth_confidence:.3f} < gate={TRUTH_CONFIDENCE_GATE}"
        )
    return True, ""


def detect_stale_state(state_load_time_sec: float, now_sec: float) -> tuple[bool, float]:
    """Returns (is_stale, age_sec)."""
    if state_load_time_sec <= 0.0:
        return True, STALE_STATE_AGE_SEC
    age = now_sec - state_load_time_sec
    return age > STALE_STATE_AGE_SEC, round(age, 2)


def compute_allocation_cap(
    requested_size: int,
    effective_independent_n: float,
    portfolio_equity: float,
    estimated_price: float,
) -> int:
    """
    Cap from effective-N and max concentration.
    Always <= requested_size.
    """
    if portfolio_equity <= 0 or estimated_price <= 0:
        return requested_size
    max_value = portfolio_equity * MAX_CONCENTRATION_PCT
    cap_from_conc = int(max_value / estimated_price)
    if effective_independent_n < MIN_EFFECTIVE_N_FULL:
        cap_from_n = int(requested_size * 0.80)
    else:
        cap_from_n = requested_size
    return min(requested_size, cap_from_conc, cap_from_n)


def apply_concentration_throttle(
    requested_size: int,
    current_concentration: float,
    max_concentration: float = MAX_CONCENTRATION_PCT,
) -> tuple[int, bool, str]:
    """
    Returns (final_size, throttle_applied, reason).
    Can only REDUCE size, never increase.
    """
    if current_concentration <= max_concentration:
        return requested_size, False, ""
    reduction = max_concentration / max(current_concentration, 1e-9)
    throttled = max(0, int(requested_size * reduction))
    reason = (
        f"concentration={current_concentration:.3f} > max={max_concentration:.3f}; "
        f"reduced {requested_size}→{throttled}"
    )
    return throttled, True, reason


def compute_concentration_penalty(
    current_concentration: float,
    max_concentration: float = MAX_CONCENTRATION_PCT,
) -> float:
    """[0, 1] penalty: 1.0 at or below limit, decreasing above."""
    if current_concentration <= 0:
        return 1.0
    return round(min(1.0, max_concentration / max(current_concentration, 1e-9)), 4)


def compute_exposure_aware_size(
    requested_size: int,
    truth_confidence: float,
    effective_independent_n: float,
    portfolio_concentration: float,
    capital_efficiency_score: float,
    portfolio_equity: float,
    estimated_price: float,
    state_load_time_sec: float,
    now_sec: float,
) -> SizingResult:
    """
    Apply all Phase 5A sizing adjustments. Deterministic.
    Can only REDUCE qty from requested_size, never increase.
    """
    is_stale, age_sec = detect_stale_state(state_load_time_sec, now_sec)
    conc_penalty = compute_concentration_penalty(portfolio_concentration)

    # Gate 1: truth confidence → block execution
    gate_ok, gate_reason = gate_on_truth_confidence(truth_confidence)
    if not gate_ok:
        return SizingResult(
            requested_size=requested_size,
            final_size=0,
            allocation_cap=0,
            throttle_applied=False,
            throttle_reason="",
            fallback_mode=True,
            fallback_reason=gate_reason,
            stale_state_detected=is_stale,
            allocation_state_age_sec=age_sec,
            effective_independent_n=effective_independent_n,
            portfolio_concentration=portfolio_concentration,
            concentration_penalty=conc_penalty,
            capital_efficiency_score=capital_efficiency_score,
            execution_allowed=False,
        )

    # Gate 2: stale state → fallback with original size (record but allow)
    if is_stale:
        return SizingResult(
            requested_size=requested_size,
            final_size=requested_size,
            allocation_cap=requested_size,
            throttle_applied=False,
            throttle_reason="",
            fallback_mode=True,
            fallback_reason=f"stale_state: age={age_sec:.0f}s > {STALE_STATE_AGE_SEC:.0f}s",
            stale_state_detected=True,
            allocation_state_age_sec=age_sec,
            effective_independent_n=effective_independent_n,
            portfolio_concentration=portfolio_concentration,
            concentration_penalty=conc_penalty,
            capital_efficiency_score=capital_efficiency_score,
            execution_allowed=True,
        )

    # Normal path: apply allocation cap then concentration throttle
    alloc_cap = compute_allocation_cap(
        requested_size, effective_independent_n, portfolio_equity, estimated_price,
    )
    capped_size = min(requested_size, alloc_cap)

    throttled_size, throttle_applied, throttle_reason = apply_concentration_throttle(
        capped_size, portfolio_concentration,
    )
    final_size = max(0, throttled_size)

    return SizingResult(
        requested_size=requested_size,
        final_size=final_size,
        allocation_cap=alloc_cap,
        throttle_applied=throttle_applied,
        throttle_reason=throttle_reason,
        fallback_mode=False,
        fallback_reason="",
        stale_state_detected=False,
        allocation_state_age_sec=age_sec,
        effective_independent_n=effective_independent_n,
        portfolio_concentration=portfolio_concentration,
        concentration_penalty=conc_penalty,
        capital_efficiency_score=capital_efficiency_score,
        execution_allowed=final_size > 0,
    )
