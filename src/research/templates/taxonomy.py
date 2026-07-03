"""AlphaTaxonomy — deterministic classification of alpha hypotheses.

Classifies generated hypotheses into canonical research categories.
Classification is based on canonical entry type and filter structure.
Taxonomy is lineage-compatible: same structure → same class, always.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.research.templates.canonical import (
    ENTRY_GAP_CONTINUATION,
    ENTRY_GAP_FADE,
    ENTRY_LIQUIDITY_SWEEP,
    ENTRY_MEAN_REVERSION_BOUNCE,
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    ENTRY_VOLATILITY_BREAKOUT,
    FILTER_LIQUIDITY,
    FILTER_REGIME,
    FILTER_VOLATILITY,
)

# ── Taxonomy classes ───────────────────────────────────────────────────────────

TAXONOMY_TREND = "trend"
TAXONOMY_MEAN_REVERSION = "mean_reversion"
TAXONOMY_BREAKOUT = "breakout"
TAXONOMY_GAP = "gap"
TAXONOMY_OVERNIGHT = "overnight"
TAXONOMY_INTRADAY = "intraday"
TAXONOMY_VOLATILITY = "volatility"
TAXONOMY_LIQUIDITY_SENSITIVE = "liquidity_sensitive"

ALL_TAXONOMY_CLASSES: frozenset = frozenset({
    TAXONOMY_TREND,
    TAXONOMY_MEAN_REVERSION,
    TAXONOMY_BREAKOUT,
    TAXONOMY_GAP,
    TAXONOMY_OVERNIGHT,
    TAXONOMY_INTRADAY,
    TAXONOMY_VOLATILITY,
    TAXONOMY_LIQUIDITY_SENSITIVE,
})

# Primary classification from entry type
_ENTRY_TAXONOMY: Dict[str, str] = {
    ENTRY_MOMENTUM_BREAKOUT:    TAXONOMY_TREND,
    ENTRY_TURTLE_BREAKOUT:      TAXONOMY_BREAKOUT,
    ENTRY_MEAN_REVERSION_BOUNCE: TAXONOMY_MEAN_REVERSION,
    ENTRY_GAP_CONTINUATION:     TAXONOMY_GAP,
    ENTRY_GAP_FADE:             TAXONOMY_GAP,
    ENTRY_VOLATILITY_BREAKOUT:  TAXONOMY_VOLATILITY,
    ENTRY_LIQUIDITY_SWEEP:      TAXONOMY_LIQUIDITY_SENSITIVE,
}

# Taxonomy descriptors for research interpretation
_TAXONOMY_DESCRIPTIONS: Dict[str, str] = {
    TAXONOMY_TREND:              "Trend-following: momentum persistence across regimes",
    TAXONOMY_MEAN_REVERSION:     "Mean reversion: oscillation around equilibrium",
    TAXONOMY_BREAKOUT:           "Breakout: channel/range expansion entry",
    TAXONOMY_GAP:                "Gap: overnight gap-driven entry signal",
    TAXONOMY_OVERNIGHT:          "Overnight: positions held through market close",
    TAXONOMY_INTRADAY:           "Intraday: positions closed before market close",
    TAXONOMY_VOLATILITY:         "Volatility: ATR/vol-regime driven entry",
    TAXONOMY_LIQUIDITY_SENSITIVE: "Liquidity: volume and spread sensitive entry",
}


class AlphaTaxonomy:
    """Deterministic taxonomy classifier for alpha hypotheses.

    Classification rules (applied in priority order):
      1. Liquidity filter present → LIQUIDITY_SENSITIVE override
      2. Entry type primary classification
    """

    @staticmethod
    def classify(
        entry_type: str,
        filter_types: tuple,
        regime_filter: Optional[str] = None,
    ) -> str:
        """Classify a hypothesis into a canonical taxonomy class.

        Args:
            entry_type:   canonical entry type string
            filter_types: sorted tuple of canonical filter types
            regime_filter: optional regime filter (currently unused in classification)

        Returns:
            Taxonomy class string from ALL_TAXONOMY_CLASSES.
        """
        if FILTER_LIQUIDITY in filter_types:
            return TAXONOMY_LIQUIDITY_SENSITIVE

        base = _ENTRY_TAXONOMY.get(entry_type)
        if base is None:
            raise ValueError(
                f"Unknown entry_type for taxonomy: {entry_type!r}. "
                f"Known: {sorted(_ENTRY_TAXONOMY.keys())}"
            )
        return base

    @staticmethod
    def describe(taxonomy_class: str) -> str:
        """Return human-readable description for a taxonomy class."""
        if taxonomy_class not in ALL_TAXONOMY_CLASSES:
            raise ValueError(f"Unknown taxonomy class: {taxonomy_class!r}")
        return _TAXONOMY_DESCRIPTIONS[taxonomy_class]

    @staticmethod
    def is_valid(taxonomy_class: str) -> bool:
        return taxonomy_class in ALL_TAXONOMY_CLASSES

    @staticmethod
    def primary_from_entry(entry_type: str) -> str:
        """Primary taxonomy class from entry type only (no filter override)."""
        result = _ENTRY_TAXONOMY.get(entry_type)
        if result is None:
            raise ValueError(f"No taxonomy mapping for entry_type: {entry_type!r}")
        return result

    @staticmethod
    def all_classes() -> Tuple[str, ...]:
        """Sorted tuple of all valid taxonomy classes."""
        return tuple(sorted(ALL_TAXONOMY_CLASSES))

    @staticmethod
    def group_by_class(
        hypothesis_ids: list,
        class_map: Dict[str, str],
    ) -> Dict[str, list]:
        """Group hypothesis IDs by taxonomy class. Deterministic: sorted within each group."""
        result: Dict[str, list] = {cls: [] for cls in sorted(ALL_TAXONOMY_CLASSES)}
        for hid in sorted(hypothesis_ids):
            cls = class_map.get(hid, "unknown")
            if cls in result:
                result[cls].append(hid)
        return {k: v for k, v in sorted(result.items()) if v}
