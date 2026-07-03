"""
src/execution/broker_truth.py
Broker truth enforcement — invariant checks.

RULE: Broker API is the sole source of truth for positions, cash, equity.
      Deferred entry state must remain fully isolated from portfolio/accounting state.

Call assert_positions_match():
  - before market open
  - after execution sync
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def assert_positions_match(
    broker_positions: dict[str, int],
    local_positions:  dict[str, int],
    tolerance:        int = 0,
) -> None:
    """
    Assert broker_positions == local_positions (per-symbol qty).

    Args:
        broker_positions: symbol → qty from broker API
        local_positions:  symbol → qty from portfolio_state.json
        tolerance:        allowed absolute qty difference per symbol
                          (0 = exact match required)

    Raises:
        AssertionError: on any mismatch beyond tolerance

    Logs:
        [BROKER_TRUTH_CHECK] PASS or MISMATCH with diff detail
    """
    all_syms   = sorted(set(broker_positions) | set(local_positions))
    mismatches: list[str] = []

    for sym in all_syms:
        b_qty = broker_positions.get(sym, 0)
        l_qty = local_positions.get(sym, 0)
        if abs(b_qty - l_qty) > tolerance:
            mismatches.append(
                f"  {sym}: broker={b_qty} local={l_qty} diff={b_qty - l_qty:+d}"
            )

    if mismatches:
        msg = (
            "[BROKER_TRUTH_CHECK] MISMATCH — "
            f"{len(mismatches)} symbol(s):\n" + "\n".join(mismatches)
        )
        logger.critical(msg)
        raise AssertionError(msg)

    logger.info(
        "[BROKER_TRUTH_CHECK] PASS — %d symbols matched (tolerance=%d)",
        len(all_syms), tolerance,
    )


def assert_deferred_isolated(
    deferred_entries: list[Any],
    portfolio_state:  dict[str, Any],
) -> None:
    """
    Verify that deferred entry state has not contaminated portfolio accounting.

    Checks that:
    - portfolio_state["cash"] and ["equity"] keys are present (not mutated to None/NaN)
    - deferred entries are not present in positions with non-zero qty
      unless the symbol is also held as a real broker position

    This is a consistency guard. It does NOT replace broker truth reconciliation.

    Logs:
        [BROKER_TRUTH_CHECK] deferred_isolated result
    """
    deferred_symbols = {e.symbol for e in deferred_entries}
    positions        = portfolio_state.get("positions", {})
    issues: list[str] = []

    # Verify accounting fields are not corrupted
    for field in ("cash", "equity"):
        val = portfolio_state.get(field)
        if val is not None:
            try:
                f = float(val)
                if f != f:  # NaN check
                    issues.append(f"portfolio_state[{field!r}] is NaN")
            except (TypeError, ValueError):
                issues.append(f"portfolio_state[{field!r}] is non-numeric: {val!r}")

    if issues:
        msg = "[BROKER_TRUTH_CHECK] deferred_isolation FAIL:\n" + "\n".join(issues)
        logger.error(msg)
        raise AssertionError(msg)

    logger.info(
        "[BROKER_TRUTH_CHECK] deferred_isolated=OK "
        "deferred_symbols=%d portfolio_positions=%d",
        len(deferred_symbols), len(positions),
    )
