"""Regime engine — deterministic rule-based regime labeling."""
from src.research.regime.labeler import label_regimes, compute_regime_distribution, label_hash
from src.research.regime.rules import ALL_REGIMES, TREND_UP, TREND_DOWN, RANGE, HIGH_VOL, LOW_VOL, PANIC, GAP_UP, GAP_DOWN

__all__ = [
    "label_regimes",
    "compute_regime_distribution",
    "label_hash",
    "ALL_REGIMES",
    "TREND_UP", "TREND_DOWN", "RANGE",
    "HIGH_VOL", "LOW_VOL",
    "PANIC", "GAP_UP", "GAP_DOWN",
]
