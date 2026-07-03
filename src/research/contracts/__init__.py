"""Research contracts — immutable canonical interfaces for alpha experimentation."""
from src.research.contracts.experiment import ExperimentContract
from src.research.contracts.fold import FoldContract, DateRange
from src.research.contracts.result import ResultContract, RegimeMetrics, StabilityMetrics
from src.research.contracts.strategy import StrategyContract

__all__ = [
    "ExperimentContract",
    "FoldContract",
    "DateRange",
    "ResultContract",
    "RegimeMetrics",
    "StabilityMetrics",
    "StrategyContract",
]
