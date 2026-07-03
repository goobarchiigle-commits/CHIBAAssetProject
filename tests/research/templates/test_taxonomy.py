"""Tests: taxonomy.py — AlphaTaxonomy classification."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.taxonomy import (
    AlphaTaxonomy,
    ALL_TAXONOMY_CLASSES,
    TAXONOMY_TREND,
    TAXONOMY_MEAN_REVERSION,
    TAXONOMY_BREAKOUT,
    TAXONOMY_GAP,
    TAXONOMY_VOLATILITY,
    TAXONOMY_LIQUIDITY_SENSITIVE,
)
from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    ENTRY_MEAN_REVERSION_BOUNCE,
    ENTRY_GAP_CONTINUATION,
    ENTRY_GAP_FADE,
    ENTRY_VOLATILITY_BREAKOUT,
    ENTRY_LIQUIDITY_SWEEP,
    FILTER_LIQUIDITY,
    FILTER_VOLATILITY,
)


class TestAlphaTaxonomy:
    def test_momentum_breakout_is_trend(self):
        cls = AlphaTaxonomy.classify(ENTRY_MOMENTUM_BREAKOUT, ())
        assert cls == TAXONOMY_TREND

    def test_turtle_breakout_is_breakout(self):
        cls = AlphaTaxonomy.classify(ENTRY_TURTLE_BREAKOUT, ())
        assert cls == TAXONOMY_BREAKOUT

    def test_mean_reversion_is_mean_reversion(self):
        cls = AlphaTaxonomy.classify(ENTRY_MEAN_REVERSION_BOUNCE, ())
        assert cls == TAXONOMY_MEAN_REVERSION

    def test_gap_continuation_is_gap(self):
        cls = AlphaTaxonomy.classify(ENTRY_GAP_CONTINUATION, ())
        assert cls == TAXONOMY_GAP

    def test_gap_fade_is_gap(self):
        cls = AlphaTaxonomy.classify(ENTRY_GAP_FADE, ())
        assert cls == TAXONOMY_GAP

    def test_volatility_breakout_is_volatility(self):
        cls = AlphaTaxonomy.classify(ENTRY_VOLATILITY_BREAKOUT, ())
        assert cls == TAXONOMY_VOLATILITY

    def test_liquidity_sweep_is_liquidity_sensitive(self):
        cls = AlphaTaxonomy.classify(ENTRY_LIQUIDITY_SWEEP, ())
        assert cls == TAXONOMY_LIQUIDITY_SENSITIVE

    def test_liquidity_filter_overrides_to_liquidity_sensitive(self):
        # momentum + liquidity filter → LIQUIDITY_SENSITIVE
        cls = AlphaTaxonomy.classify(ENTRY_MOMENTUM_BREAKOUT, (FILTER_LIQUIDITY,))
        assert cls == TAXONOMY_LIQUIDITY_SENSITIVE

    def test_volatility_filter_does_not_override(self):
        cls = AlphaTaxonomy.classify(ENTRY_MOMENTUM_BREAKOUT, (FILTER_VOLATILITY,))
        assert cls == TAXONOMY_TREND

    def test_unknown_entry_raises(self):
        with pytest.raises(ValueError):
            AlphaTaxonomy.classify("TOTALLY_UNKNOWN", ())

    def test_primary_from_entry_ignores_filters(self):
        # FILTER_LIQUIDITY should NOT override primary_from_entry
        cls = AlphaTaxonomy.primary_from_entry(ENTRY_MOMENTUM_BREAKOUT)
        assert cls == TAXONOMY_TREND

    def test_primary_from_entry_unknown_raises(self):
        with pytest.raises(ValueError):
            AlphaTaxonomy.primary_from_entry("UNKNOWN_ENTRY")

    def test_is_valid_known_class(self):
        assert AlphaTaxonomy.is_valid(TAXONOMY_TREND)
        assert AlphaTaxonomy.is_valid(TAXONOMY_BREAKOUT)

    def test_is_valid_unknown_class(self):
        assert not AlphaTaxonomy.is_valid("FICTIONAL_CLASS")

    def test_describe_known(self):
        desc = AlphaTaxonomy.describe(TAXONOMY_TREND)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_describe_unknown_raises(self):
        with pytest.raises(ValueError):
            AlphaTaxonomy.describe("UNKNOWN")

    def test_all_classes_sorted(self):
        classes = AlphaTaxonomy.all_classes()
        assert list(classes) == sorted(classes)
        assert len(classes) == len(ALL_TAXONOMY_CLASSES)

    def test_group_by_class(self):
        class_map = {
            "hyp_a": TAXONOMY_TREND,
            "hyp_b": TAXONOMY_BREAKOUT,
            "hyp_c": TAXONOMY_TREND,
        }
        grouped = AlphaTaxonomy.group_by_class(["hyp_a", "hyp_b", "hyp_c"], class_map)
        assert TAXONOMY_TREND in grouped
        assert len(grouped[TAXONOMY_TREND]) == 2
        assert "hyp_b" in grouped[TAXONOMY_BREAKOUT]

    def test_group_by_class_sorted_within_group(self):
        class_map = {"hyp_z": TAXONOMY_TREND, "hyp_a": TAXONOMY_TREND}
        grouped = AlphaTaxonomy.group_by_class(["hyp_z", "hyp_a"], class_map)
        assert grouped[TAXONOMY_TREND] == sorted(grouped[TAXONOMY_TREND])

    def test_group_by_class_empty_classes_omitted(self):
        class_map = {"hyp_x": TAXONOMY_TREND}
        grouped = AlphaTaxonomy.group_by_class(["hyp_x"], class_map)
        # Only TREND should appear (non-empty)
        for cls, members in grouped.items():
            assert len(members) > 0

    def test_regime_filter_does_not_affect_classification(self):
        c1 = AlphaTaxonomy.classify(ENTRY_MOMENTUM_BREAKOUT, (), regime_filter=None)
        c2 = AlphaTaxonomy.classify(ENTRY_MOMENTUM_BREAKOUT, (), regime_filter="TREND_ONLY_REGIME")
        assert c1 == c2
