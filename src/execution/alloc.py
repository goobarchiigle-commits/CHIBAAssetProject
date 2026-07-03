"""
src/execution/alloc.py
Fill-aware allocation engine.

RULE:
  After partial fill, allocation MUST use remaining_qty — never original qty.
  Stale allocation (based on original qty after PARTIAL) is forbidden.

Logs [ALLOC_RECALC] for each PARTIAL intent consumed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.execution.intent import ExecutionIntent, IntentStatus

logger = logging.getLogger(__name__)


@dataclass
class FillAwareAlloc:
    """
    Fill-aware allocation state computed from live intents + positions.

    All fields are computed, not stored — recompute on every signal cycle.
    """
    total_exposure: float                    # sum(positions × price) + partial fills
    buying_power:   float                    # cash minus remaining partial fill reserves
    partial_fills:  dict[str, dict] = field(default_factory=dict)
    # symbol → {filled_qty, remaining_qty, exposure_used, exposure_reserved}
    sector_usage:   dict[str, float] = field(default_factory=dict)  # sector → weight (0..1)
    symbol_usage:   dict[str, float] = field(default_factory=dict)  # symbol → weight (0..1)


def compute_fill_aware_alloc(
    intents:       list[ExecutionIntent],
    cash:          float,
    equity:        float,
    market_values: dict[str, float],
    positions:     dict[str, int] | None = None,
    sectors:       dict[str, str]  | None = None,
) -> FillAwareAlloc:
    """
    Compute fill-aware allocation state.

    Consumes remaining_qty and filled_qty from PARTIAL intents.
    Never uses original_qty for reserved-capacity after partial fill.

    Args:
        intents:       Active ExecutionIntents — only PARTIAL ones drive recalc
        cash:          Available broker cash
        equity:        Total equity (cash + all positions at market)
        market_values: symbol → current market price
        positions:     Existing fully-settled positions (symbol → qty)
        sectors:       symbol → sector name (for sector cap tracking)

    Logs:
        [ALLOC_RECALC] symbol=... filled=... remaining=... exposure=...
        [ALLOC_RECALC] total_exposure=... buying_power=... partial_symbols=...
    """
    pos = positions or {}
    sec = sectors   or {}

    # ── Base position exposure ────────────────────────────────────────
    total_exposure  = 0.0
    symbol_usage:    dict[str, float] = {}
    sector_usage:    dict[str, float] = {}

    for sym, qty in pos.items():
        price    = market_values.get(sym, 0.0)
        exposure = int(qty) * price
        total_exposure += exposure
        symbol_usage[sym]             = symbol_usage.get(sym, 0.0)   + exposure
        sector_usage[sec.get(sym, "unknown")] = (
            sector_usage.get(sec.get(sym, "unknown"), 0.0) + exposure
        )

    # ── PARTIAL fills: add filled portion + reserve remaining ─────────
    reserved_cash      = 0.0
    partial_fills_out: dict[str, dict] = {}

    for intent in intents:
        if intent.status != IntentStatus.PARTIAL.value:
            continue

        sym   = intent.symbol
        price = market_values.get(sym, intent.expected_price)

        filled_exposure    = intent.filled_qty    * price
        remaining_exposure = intent.remaining_qty * price
        reserved_cash     += remaining_exposure

        total_exposure += filled_exposure
        symbol_usage[sym]             = symbol_usage.get(sym, 0.0)   + filled_exposure
        sector_usage[sec.get(sym, "unknown")] = (
            sector_usage.get(sec.get(sym, "unknown"), 0.0) + filled_exposure
        )

        partial_fills_out[sym] = {
            "filled_qty":        intent.filled_qty,
            "remaining_qty":     intent.remaining_qty,
            "exposure_used":     filled_exposure,
            "exposure_reserved": remaining_exposure,
        }

        logger.info(
            "[ALLOC_RECALC] symbol=%s filled=%d remaining=%d"
            " exposure_used=%.0f exposure_reserved=%.0f",
            sym, intent.filled_qty, intent.remaining_qty,
            filled_exposure, remaining_exposure,
        )

    buying_power = max(0.0, cash - reserved_cash)

    # Normalise to equity fraction
    if equity > 0:
        symbol_usage_w = {s: v / equity for s, v in symbol_usage.items()}
        sector_usage_w = {s: v / equity for s, v in sector_usage.items()}
    else:
        symbol_usage_w = dict.fromkeys(symbol_usage, 0.0)
        sector_usage_w = dict.fromkeys(sector_usage, 0.0)

    logger.info(
        "[ALLOC_RECALC] total_exposure=%.0f buying_power=%.0f partial_symbols=%d",
        total_exposure, buying_power, len(partial_fills_out),
    )

    return FillAwareAlloc(
        total_exposure = total_exposure,
        buying_power   = buying_power,
        partial_fills  = partial_fills_out,
        sector_usage   = sector_usage_w,
        symbol_usage   = symbol_usage_w,
    )
