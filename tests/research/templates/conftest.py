"""Shared fixtures for template system tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    EXIT_TURTLE,
    EXIT_MOMENTUM,
    FILTER_VOLATILITY,
    FILTER_LIQUIDITY,
    HOLD_SIGNAL_EXIT,
    HOLD_FIXED_DAYS,
    SIZE_ATR_BASED,
    SIZE_FIXED,
)
from src.research.templates.constraints import ConstraintHierarchy
from src.research.templates.template import HypothesisTemplate


def make_small_template(
    name: str = "test_momentum",
    entry_types=None,
    exit_types=None,
    filter_combos=None,
    holding_models=None,
    sizing_models=None,
    regime_filters=None,
    parameter_domains=None,
) -> HypothesisTemplate:
    """Minimal valid template for testing."""
    return HypothesisTemplate(
        name=name,
        entry_types=entry_types or (ENTRY_MOMENTUM_BREAKOUT,),
        exit_types=exit_types or (EXIT_TURTLE,),
        filter_combos=filter_combos or ((), (FILTER_VOLATILITY,)),
        holding_models=holding_models or (HOLD_SIGNAL_EXIT,),
        sizing_models=sizing_models or (SIZE_ATR_BASED,),
        regime_filters=regime_filters or (None,),
        parameter_domains=parameter_domains or {
            "entry_period": [10, 20],
            "exit_period": [20, 40],
        },
    )
