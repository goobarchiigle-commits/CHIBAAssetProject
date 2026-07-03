"""
src/execution/fractional_demand.py
Fractional demand accumulator — INTERNAL advisory tracking only.

Tracks "how much desired exposure failed to execute historically".
This is NOT a virtual holding. It does NOT:
  - modify holdings
  - modify broker exposure
  - simulate ownership
  - consume cash

Execution occurs ONLY after actual broker order fill confirmation.

Example:
    Day1 desired=0.32 lots → accumulated=0.32
    Day2 desired=0.41 lots → accumulated=0.73
    Day3 desired=0.38 lots → accumulated=1.11 → execute 1 actual lot
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LOT_SIZE: int = 100


@dataclass
class AccumulatorState:
    symbol:             str
    accumulated_demand: float = 0.0   # in fractional lots (advisory only)
    execution_count:    int   = 0


class FractionalAccumulator:
    """
    Accumulates fractional lot demand across days.

    INVARIANT: accumulated_demand is observational only.
               It has no effect on cash, positions, or equity.
               Broker positions are the sole truth.
    """

    def __init__(self) -> None:
        self._state: dict[str, AccumulatorState] = {}

    def accumulate(self, symbol: str, desired_lots: float) -> float:
        """
        Add desired_lots to running accumulator.

        Args:
            symbol:       ticker
            desired_lots: fractional lot demand this cycle (must be >= 0)

        Returns:
            new accumulated total (fractional lots)
        """
        if symbol not in self._state:
            self._state[symbol] = AccumulatorState(symbol=symbol)
        demand = max(0.0, desired_lots)
        self._state[symbol].accumulated_demand += demand
        total = self._state[symbol].accumulated_demand
        logger.info(
            "[FRACTIONAL_ACCUM] symbol=%s desired_lots=%.4f accumulated=%.4f",
            symbol, demand, total,
        )
        return total

    def get_accumulated(self, symbol: str) -> float:
        """Return current accumulated demand (fractional lots)."""
        return self._state.get(symbol, AccumulatorState(symbol=symbol)).accumulated_demand

    def check_lot_threshold(self, symbol: str) -> int:
        """
        Return number of executable whole lots if accumulated >= 1 lot.
        Returns 0 if accumulated < 1 lot.
        """
        return int(self.get_accumulated(symbol))

    def consume(self, symbol: str, lots_executed: int) -> float:
        """
        Subtract executed lots from accumulator.
        ONLY call after confirmed broker fill.

        Returns:
            remainder accumulated demand
        """
        if symbol not in self._state:
            return 0.0
        self._state[symbol].accumulated_demand = max(
            0.0, self._state[symbol].accumulated_demand - lots_executed,
        )
        self._state[symbol].execution_count += 1
        remainder = self._state[symbol].accumulated_demand
        logger.info(
            "[FRACTIONAL_ACCUM] symbol=%s lots_executed=%d remainder=%.4f "
            "total_executions=%d",
            symbol, lots_executed, remainder,
            self._state[symbol].execution_count,
        )
        return remainder

    def reset(self, symbol: str) -> None:
        if symbol in self._state:
            del self._state[symbol]

    def total_accumulated(self) -> float:
        return sum(s.accumulated_demand for s in self._state.values())

    def total_execution_count(self) -> int:
        return sum(s.execution_count for s in self._state.values())

    def summary(self) -> dict[str, dict]:
        return {
            sym: {
                "accumulated_demand": round(s.accumulated_demand, 4),
                "execution_count":    s.execution_count,
            }
            for sym, s in self._state.items()
        }
