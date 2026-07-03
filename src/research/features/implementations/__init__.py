"""Concrete FeatureContract implementations for alpha research."""
from src.research.features.implementations.volatility import RollingVolatility
from src.research.features.implementations.atr import ATR
from src.research.features.implementations.momentum import Momentum
from src.research.features.implementations.liquidity import LiquidityMetrics
from src.research.features.implementations.returns import RollingReturns
from src.research.features.implementations.breadth import BreadthMetrics
from src.research.features.implementations.regime import RegimeFeatures

ALL_FEATURES = [
    RollingVolatility,
    ATR,
    Momentum,
    LiquidityMetrics,
    RollingReturns,
    BreadthMetrics,
    RegimeFeatures,
]

__all__ = [
    "RollingVolatility",
    "ATR",
    "Momentum",
    "LiquidityMetrics",
    "RollingReturns",
    "BreadthMetrics",
    "RegimeFeatures",
    "ALL_FEATURES",
]
