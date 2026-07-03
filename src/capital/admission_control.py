"""
admission_control.py — Capital-aware high-price stock admission

Replaces simple price threshold with lot_cost_ratio gate.
As effective_capital grows, higher-priced stocks become accessible.

lot_cost_ratio = (price × lot_size) / effective_capital
Admit if lot_cost_ratio ≤ max_lot_cost_ratio (default 0.30 = 30%)

Capital progression:
  3M → max ¥9,000/share (lot=¥900K ≤ ¥3M×0.30=¥900K)
  4M → max ¥12,000/share
  7M → max ¥21,000/share (Tokyo Electron accessible)
  10M → max ¥30,000/share (most large-caps accessible)

Held positions always admit — SELL signals must not be blocked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOT_SIZE = 100


@dataclass
class AdmissionConfig:
    max_lot_cost_ratio: float = 0.30        # lot_cost / effective_capital upper bound
    cash_safety_buffer: float = 0.10        # cash reserve fraction excluded from deployable
    max_single_position_ratio: float = 0.25 # position-level concentration cap


@dataclass
class AdmissionDecision:
    symbol: str
    admitted: bool
    reason: str
    lot_cost: float
    max_allowed_lot_cost: float
    lot_cost_ratio: float


def assess_admission(
    symbol: str,
    price: float,
    effective_capital: float,
    held: bool = False,
    cfg: Optional[AdmissionConfig] = None,
    lot_size: int = LOT_SIZE,
) -> AdmissionDecision:
    """
    Assess whether a symbol is deployable given current effective_capital.

    Held symbols always admit — enables SELL signal generation.
    Unknown price (price=0) → conservative admit (no block on data absence).
    """
    if cfg is None:
        cfg = AdmissionConfig()

    if held:
        return AdmissionDecision(
            symbol=symbol,
            admitted=True,
            reason="held_position",
            lot_cost=price * lot_size,
            max_allowed_lot_cost=effective_capital * cfg.max_lot_cost_ratio,
            lot_cost_ratio=0.0,
        )

    if price <= 0.0:
        # Unknown price → conservative admit, let signal generation handle it
        return AdmissionDecision(
            symbol=symbol,
            admitted=True,
            reason="price_unknown_conservative_admit",
            lot_cost=0.0,
            max_allowed_lot_cost=0.0,
            lot_cost_ratio=0.0,
        )

    if effective_capital <= 0.0:
        return AdmissionDecision(
            symbol=symbol,
            admitted=False,
            reason="zero_effective_capital",
            lot_cost=price * lot_size,
            max_allowed_lot_cost=0.0,
            lot_cost_ratio=1.0,
        )

    lot_cost = price * lot_size
    deployable = effective_capital * (1.0 - cfg.cash_safety_buffer)
    max_by_ratio = deployable * cfg.max_lot_cost_ratio
    max_by_position = effective_capital * cfg.max_single_position_ratio
    effective_max = min(max_by_ratio, max_by_position)

    lot_cost_ratio = lot_cost / effective_capital

    if lot_cost <= effective_max:
        return AdmissionDecision(
            symbol=symbol,
            admitted=True,
            reason="within_capital_limit",
            lot_cost=lot_cost,
            max_allowed_lot_cost=effective_max,
            lot_cost_ratio=round(lot_cost_ratio, 4),
        )

    reason = (
        f"lot_cost_ratio={lot_cost_ratio:.3f} > max={cfg.max_lot_cost_ratio:.3f} "
        f"(¥{lot_cost:,.0f} > ¥{effective_max:,.0f})"
    )
    logger.debug("Admission REJECTED %s: %s", symbol, reason)
    return AdmissionDecision(
        symbol=symbol,
        admitted=False,
        reason=reason,
        lot_cost=lot_cost,
        max_allowed_lot_cost=effective_max,
        lot_cost_ratio=round(lot_cost_ratio, 4),
    )


def filter_universe_by_capital(
    tickers: dict,
    prices: dict,
    effective_capital: float,
    held_symbols: set,
    cfg: Optional[AdmissionConfig] = None,
) -> tuple[dict, list]:
    """
    Capital-aware universe filter.

    Args:
        tickers:          {symbol: sector}
        prices:           {symbol: float} — latest price per share
        effective_capital: from CapitalState
        held_symbols:     currently held (always admitted)

    Returns:
        (filtered_tickers, rejected_decisions)
    """
    filtered: dict = {}
    rejected: list = []

    for sym, sector in tickers.items():
        price = prices.get(sym)
        is_held = sym in held_symbols

        if price is None:
            filtered[sym] = sector
            continue

        decision = assess_admission(
            sym, float(price), effective_capital, held=is_held, cfg=cfg,
        )
        if decision.admitted:
            filtered[sym] = sector
        else:
            rejected.append(decision)

    return filtered, rejected
