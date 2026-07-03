"""Tests: canonical.py — CanonicalSemantics type system."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.canonical import (
    ALL_ENTRY_TYPES,
    ALL_EXIT_TYPES,
    ALL_FILTER_TYPES,
    ALL_HOLDING_MODELS,
    ALL_SIZING_MODELS,
    ALL_REGIME_FILTERS,
    CanonicalSemantics,
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    EXIT_TURTLE,
    FILTER_VOLATILITY,
    HOLD_SIGNAL_EXIT,
    SIZE_ATR_BASED,
    REGIME_ANY,
)


class TestCanonicalTypes:
    def test_all_entry_types_nonempty(self):
        assert len(ALL_ENTRY_TYPES) > 0

    def test_all_exit_types_nonempty(self):
        assert len(ALL_EXIT_TYPES) > 0

    def test_all_filter_types_nonempty(self):
        assert len(ALL_FILTER_TYPES) > 0

    def test_all_holding_models_nonempty(self):
        assert len(ALL_HOLDING_MODELS) > 0

    def test_all_sizing_models_nonempty(self):
        assert len(ALL_SIZING_MODELS) > 0

    def test_regime_any_in_set(self):
        assert REGIME_ANY in ALL_REGIME_FILTERS

    def test_entry_constants_in_set(self):
        assert ENTRY_MOMENTUM_BREAKOUT in ALL_ENTRY_TYPES
        assert ENTRY_TURTLE_BREAKOUT in ALL_ENTRY_TYPES

    def test_exit_constant_in_set(self):
        assert EXIT_TURTLE in ALL_EXIT_TYPES

    def test_filter_constant_in_set(self):
        assert FILTER_VOLATILITY in ALL_FILTER_TYPES


class TestNormalization:
    def test_normalize_entry_exact(self):
        assert CanonicalSemantics.normalize_entry(ENTRY_MOMENTUM_BREAKOUT) == ENTRY_MOMENTUM_BREAKOUT

    def test_normalize_entry_alias_momentum(self):
        assert CanonicalSemantics.normalize_entry("momentum") == ENTRY_MOMENTUM_BREAKOUT

    def test_normalize_entry_alias_turtle(self):
        assert CanonicalSemantics.normalize_entry("turtle") == ENTRY_TURTLE_BREAKOUT

    def test_normalize_entry_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown entry type"):
            CanonicalSemantics.normalize_entry("totally_unknown_xyz")

    def test_normalize_exit_exact(self):
        assert CanonicalSemantics.normalize_exit(EXIT_TURTLE) == EXIT_TURTLE

    def test_normalize_exit_alias(self):
        assert CanonicalSemantics.normalize_exit("turtle") == EXIT_TURTLE

    def test_normalize_exit_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown exit type"):
            CanonicalSemantics.normalize_exit("unknown_exit")

    def test_normalize_filter_exact(self):
        assert CanonicalSemantics.normalize_filter(FILTER_VOLATILITY) == FILTER_VOLATILITY

    def test_normalize_filter_alias_vol(self):
        assert CanonicalSemantics.normalize_filter("vol") == FILTER_VOLATILITY

    def test_normalize_filter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown filter type"):
            CanonicalSemantics.normalize_filter("magic_filter")

    def test_normalize_holding_exact(self):
        assert CanonicalSemantics.normalize_holding_model(HOLD_SIGNAL_EXIT) == HOLD_SIGNAL_EXIT

    def test_normalize_holding_unknown_raises(self):
        with pytest.raises(ValueError):
            CanonicalSemantics.normalize_holding_model("random_hold")

    def test_normalize_sizing_exact(self):
        assert CanonicalSemantics.normalize_sizing_model(SIZE_ATR_BASED) == SIZE_ATR_BASED

    def test_normalize_sizing_unknown_raises(self):
        with pytest.raises(ValueError):
            CanonicalSemantics.normalize_sizing_model("fancy_sizing")

    def test_validate_all_types_valid(self):
        CanonicalSemantics.validate_all_types(
            (ENTRY_MOMENTUM_BREAKOUT,),
            (EXIT_TURTLE,),
            ((), (FILTER_VOLATILITY,)),
            (HOLD_SIGNAL_EXIT,),
            (SIZE_ATR_BASED,),
        )

    def test_validate_all_types_bad_entry_raises(self):
        with pytest.raises(ValueError, match="Invalid entry type"):
            CanonicalSemantics.validate_all_types(
                ("BAD_ENTRY",), (EXIT_TURTLE,), ((),), (HOLD_SIGNAL_EXIT,), (SIZE_ATR_BASED,),
            )

    def test_validate_all_types_bad_filter_raises(self):
        with pytest.raises(ValueError, match="Invalid filter type"):
            CanonicalSemantics.validate_all_types(
                (ENTRY_MOMENTUM_BREAKOUT,), (EXIT_TURTLE,),
                (("UNKNOWN_FILTER",),), (HOLD_SIGNAL_EXIT,), (SIZE_ATR_BASED,),
            )

    def test_normalization_case_insensitive_alias(self):
        result = CanonicalSemantics.normalize_entry("Momentum")
        assert result == ENTRY_MOMENTUM_BREAKOUT

    def test_normalize_entry_deterministic(self):
        a = CanonicalSemantics.normalize_entry("turtle")
        b = CanonicalSemantics.normalize_entry("turtle")
        assert a == b
