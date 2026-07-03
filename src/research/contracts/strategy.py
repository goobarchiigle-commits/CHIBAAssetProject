"""StrategyContract — abstract base for all alpha strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class StrategyContract(ABC):
    @abstractmethod
    def generate_signals(self, data: Any) -> Any:
        """Compute signals from feature data. Must be deterministic given seed."""

    @abstractmethod
    def generate_orders(self, signals: Any, portfolio: Any) -> Any:
        """Convert signals to order list given current portfolio state."""

    @abstractmethod
    def required_features(self) -> List[str]:
        """Column names required in the input data."""

    @abstractmethod
    def parameter_space(self) -> Dict[str, Any]:
        """Bounded parameter space for sweep. Each key → range/set of values."""
