"""CanonicalSemantics — finite enumerable canonical strategy type system.

All strategy components must map to one of the canonical types defined here.
Semantically equivalent strategies map to identical canonical forms.
No arbitrary execution logic is representable outside these types.
"""
from __future__ import annotations

from typing import Dict, Optional

# ── Entry types ────────────────────────────────────────────────────────────────

ENTRY_MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"
ENTRY_TURTLE_BREAKOUT = "TURTLE_BREAKOUT"
ENTRY_MEAN_REVERSION_BOUNCE = "MEAN_REVERSION_BOUNCE"
ENTRY_GAP_CONTINUATION = "GAP_CONTINUATION"
ENTRY_GAP_FADE = "GAP_FADE"
ENTRY_VOLATILITY_BREAKOUT = "VOLATILITY_BREAKOUT"
ENTRY_LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"

ALL_ENTRY_TYPES: frozenset = frozenset({
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    ENTRY_MEAN_REVERSION_BOUNCE,
    ENTRY_GAP_CONTINUATION,
    ENTRY_GAP_FADE,
    ENTRY_VOLATILITY_BREAKOUT,
    ENTRY_LIQUIDITY_SWEEP,
})

# ── Exit types ─────────────────────────────────────────────────────────────────

EXIT_TURTLE = "TURTLE_EXIT"
EXIT_MOMENTUM = "MOMENTUM_EXIT"
EXIT_FIXED_HOLD = "FIXED_HOLD_EXIT"
EXIT_FIXED_STOP = "FIXED_STOP_EXIT"
EXIT_TRAILING_STOP = "TRAILING_STOP_EXIT"
EXIT_SIGNAL_REVERSAL = "SIGNAL_REVERSAL_EXIT"
EXIT_TIME_STOP = "TIME_STOP_EXIT"

ALL_EXIT_TYPES: frozenset = frozenset({
    EXIT_TURTLE,
    EXIT_MOMENTUM,
    EXIT_FIXED_HOLD,
    EXIT_FIXED_STOP,
    EXIT_TRAILING_STOP,
    EXIT_SIGNAL_REVERSAL,
    EXIT_TIME_STOP,
})

# ── Filter types ───────────────────────────────────────────────────────────────

FILTER_VOLATILITY = "VOLATILITY_FILTER"
FILTER_LIQUIDITY = "LIQUIDITY_FILTER"
FILTER_REGIME = "REGIME_FILTER"
FILTER_TREND = "TREND_FILTER"
FILTER_MOMENTUM_STRENGTH = "MOMENTUM_STRENGTH_FILTER"

ALL_FILTER_TYPES: frozenset = frozenset({
    FILTER_VOLATILITY,
    FILTER_LIQUIDITY,
    FILTER_REGIME,
    FILTER_TREND,
    FILTER_MOMENTUM_STRENGTH,
})

# ── Holding models ─────────────────────────────────────────────────────────────

HOLD_FIXED_DAYS = "FIXED_DAYS_HOLD"
HOLD_SIGNAL_EXIT = "SIGNAL_EXIT_HOLD"
HOLD_TURTLE = "TURTLE_HOLD"
HOLD_VOLATILITY_SCALED = "VOLATILITY_SCALED_HOLD"

ALL_HOLDING_MODELS: frozenset = frozenset({
    HOLD_FIXED_DAYS,
    HOLD_SIGNAL_EXIT,
    HOLD_TURTLE,
    HOLD_VOLATILITY_SCALED,
})

# ── Sizing models ──────────────────────────────────────────────────────────────

SIZE_FIXED = "FIXED_SIZE"
SIZE_ATR_BASED = "ATR_BASED_SIZE"
SIZE_VOLATILITY_TARGET = "VOLATILITY_TARGET_SIZE"
SIZE_EQUAL_WEIGHT = "EQUAL_WEIGHT_SIZE"

ALL_SIZING_MODELS: frozenset = frozenset({
    SIZE_FIXED,
    SIZE_ATR_BASED,
    SIZE_VOLATILITY_TARGET,
    SIZE_EQUAL_WEIGHT,
})

# ── Regime filter values ───────────────────────────────────────────────────────

REGIME_ANY = None           # no regime filter (matches all regimes)
REGIME_TREND_ONLY = "TREND_ONLY_REGIME"
REGIME_LOW_VOL_ONLY = "LOW_VOL_ONLY_REGIME"
REGIME_NON_PANIC_ONLY = "NON_PANIC_ONLY_REGIME"
REGIME_EXPANSION_ONLY = "EXPANSION_ONLY_REGIME"

ALL_REGIME_FILTERS: frozenset = frozenset({
    REGIME_ANY,
    REGIME_TREND_ONLY,
    REGIME_LOW_VOL_ONLY,
    REGIME_NON_PANIC_ONLY,
    REGIME_EXPANSION_ONLY,
})

# ── Alias normalization ────────────────────────────────────────────────────────

_ENTRY_ALIASES: Dict[str, str] = {
    "momentum": ENTRY_MOMENTUM_BREAKOUT,
    "mom_breakout": ENTRY_MOMENTUM_BREAKOUT,
    "momentum_breakout": ENTRY_MOMENTUM_BREAKOUT,
    "turtle": ENTRY_TURTLE_BREAKOUT,
    "turtle_breakout": ENTRY_TURTLE_BREAKOUT,
    "breakout": ENTRY_TURTLE_BREAKOUT,
    "mean_reversion": ENTRY_MEAN_REVERSION_BOUNCE,
    "mean_rev": ENTRY_MEAN_REVERSION_BOUNCE,
    "reversion": ENTRY_MEAN_REVERSION_BOUNCE,
    "bounce": ENTRY_MEAN_REVERSION_BOUNCE,
    "gap_continuation": ENTRY_GAP_CONTINUATION,
    "gap_cont": ENTRY_GAP_CONTINUATION,
    "gap_up": ENTRY_GAP_CONTINUATION,
    "gap_fade": ENTRY_GAP_FADE,
    "gap_down": ENTRY_GAP_FADE,
    "volatility_breakout": ENTRY_VOLATILITY_BREAKOUT,
    "vol_breakout": ENTRY_VOLATILITY_BREAKOUT,
    "liquidity_sweep": ENTRY_LIQUIDITY_SWEEP,
    "liq_sweep": ENTRY_LIQUIDITY_SWEEP,
}

_EXIT_ALIASES: Dict[str, str] = {
    "turtle": EXIT_TURTLE,
    "turtle_exit": EXIT_TURTLE,
    "momentum": EXIT_MOMENTUM,
    "momentum_exit": EXIT_MOMENTUM,
    "fixed_hold": EXIT_FIXED_HOLD,
    "hold": EXIT_FIXED_HOLD,
    "fixed_stop": EXIT_FIXED_STOP,
    "stop": EXIT_FIXED_STOP,
    "trailing": EXIT_TRAILING_STOP,
    "trailing_stop": EXIT_TRAILING_STOP,
    "signal_reversal": EXIT_SIGNAL_REVERSAL,
    "reversal": EXIT_SIGNAL_REVERSAL,
    "time_stop": EXIT_TIME_STOP,
    "time": EXIT_TIME_STOP,
}

_FILTER_ALIASES: Dict[str, str] = {
    "volatility": FILTER_VOLATILITY,
    "vol": FILTER_VOLATILITY,
    "volatility_filter": FILTER_VOLATILITY,
    "liquidity": FILTER_LIQUIDITY,
    "liq": FILTER_LIQUIDITY,
    "liquidity_filter": FILTER_LIQUIDITY,
    "regime": FILTER_REGIME,
    "regime_filter": FILTER_REGIME,
    "trend": FILTER_TREND,
    "trend_filter": FILTER_TREND,
    "momentum_strength": FILTER_MOMENTUM_STRENGTH,
    "rsr": FILTER_MOMENTUM_STRENGTH,
    "momentum_filter": FILTER_MOMENTUM_STRENGTH,
}


class CanonicalSemantics:
    """Maps raw strategy descriptions to canonical types.

    Semantically equivalent strategies produce identical canonical forms.
    Raises ValueError for unrecognized types (fail closed).
    """

    @staticmethod
    def normalize_entry(raw: str) -> str:
        upper = raw.upper().strip()
        if upper in ALL_ENTRY_TYPES:
            return upper
        alias = _ENTRY_ALIASES.get(raw.lower().strip())
        if alias:
            return alias
        raise ValueError(f"Unknown entry type: {raw!r}. Valid: {sorted(ALL_ENTRY_TYPES)}")

    @staticmethod
    def normalize_exit(raw: str) -> str:
        upper = raw.upper().strip()
        if upper in ALL_EXIT_TYPES:
            return upper
        alias = _EXIT_ALIASES.get(raw.lower().strip())
        if alias:
            return alias
        raise ValueError(f"Unknown exit type: {raw!r}. Valid: {sorted(ALL_EXIT_TYPES)}")

    @staticmethod
    def normalize_filter(raw: str) -> str:
        upper = raw.upper().strip()
        if upper in ALL_FILTER_TYPES:
            return upper
        alias = _FILTER_ALIASES.get(raw.lower().strip())
        if alias:
            return alias
        raise ValueError(f"Unknown filter type: {raw!r}. Valid: {sorted(ALL_FILTER_TYPES)}")

    @staticmethod
    def normalize_holding_model(raw: str) -> str:
        upper = raw.upper().strip()
        if upper in ALL_HOLDING_MODELS:
            return upper
        raise ValueError(f"Unknown holding model: {raw!r}. Valid: {sorted(ALL_HOLDING_MODELS)}")

    @staticmethod
    def normalize_sizing_model(raw: str) -> str:
        upper = raw.upper().strip()
        if upper in ALL_SIZING_MODELS:
            return upper
        raise ValueError(f"Unknown sizing model: {raw!r}. Valid: {sorted(ALL_SIZING_MODELS)}")

    @staticmethod
    def validate_all_types(
        entry_types: tuple,
        exit_types: tuple,
        filter_combos: tuple,
        holding_models: tuple,
        sizing_models: tuple,
    ) -> None:
        """Validate all types belong to canonical sets. Raises ValueError on unknown."""
        for et in entry_types:
            if et not in ALL_ENTRY_TYPES:
                raise ValueError(f"Invalid entry type: {et!r}")
        for xt in exit_types:
            if xt not in ALL_EXIT_TYPES:
                raise ValueError(f"Invalid exit type: {xt!r}")
        for combo in filter_combos:
            for ft in combo:
                if ft not in ALL_FILTER_TYPES:
                    raise ValueError(f"Invalid filter type: {ft!r}")
        for hm in holding_models:
            if hm not in ALL_HOLDING_MODELS:
                raise ValueError(f"Invalid holding model: {hm!r}")
        for sm in sizing_models:
            if sm not in ALL_SIZING_MODELS:
                raise ValueError(f"Invalid sizing model: {sm!r}")
