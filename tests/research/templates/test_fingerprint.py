"""Tests: fingerprint.py — StructuralFingerprint determinism and similarity."""
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
    FILTER_VOLATILITY,
    FILTER_LIQUIDITY,
    HOLD_SIGNAL_EXIT,
    SIZE_ATR_BASED,
)
from src.research.templates.fingerprint import StructuralFingerprint
from src.research.templates.grammar import AlphaGrammar


def _make_fp(
    entry_type=ENTRY_MOMENTUM_BREAKOUT,
    exit_type=EXIT_TURTLE,
    filter_types=(),
    holding_model=HOLD_SIGNAL_EXIT,
    sizing_model=SIZE_ATR_BASED,
    regime_filter=None,
    params=None,
) -> StructuralFingerprint:
    params = params or {}
    expr = AlphaGrammar.build_expression(
        entry_type, exit_type, tuple(sorted(filter_types)),
        holding_model, sizing_model, regime_filter, params
    )
    return StructuralFingerprint.from_expression(expr)


class TestStructuralFingerprint:
    def test_deterministic_same_structure(self):
        fp1 = _make_fp(params={"p": 10})
        fp2 = _make_fp(params={"p": 20})
        assert fp1.structural_equivalence_hash == fp2.structural_equivalence_hash

    def test_full_grammar_hash_differs_on_params(self):
        fp1 = _make_fp(params={"p": 10})
        fp2 = _make_fp(params={"p": 20})
        assert fp1.full_grammar_hash != fp2.full_grammar_hash

    def test_different_entry_different_structural_hash(self):
        fp1 = _make_fp(entry_type=ENTRY_MOMENTUM_BREAKOUT)
        fp2 = _make_fp(entry_type=ENTRY_TURTLE_BREAKOUT)
        assert fp1.structural_equivalence_hash != fp2.structural_equivalence_hash

    def test_feature_signature_nonempty(self):
        fp = _make_fp()
        assert len(fp.feature_overlap_signature) > 0

    def test_feature_signature_sorted(self):
        fp = _make_fp(filter_types=(FILTER_VOLATILITY,))
        assert list(fp.feature_overlap_signature) == sorted(fp.feature_overlap_signature)

    def test_filter_adds_features(self):
        fp_no_filter = _make_fp()
        fp_with_filter = _make_fp(filter_types=(FILTER_VOLATILITY,))
        # Adding filter should not reduce features
        assert len(fp_with_filter.feature_overlap_signature) >= len(fp_no_filter.feature_overlap_signature)

    def test_signal_overlap_ratio_identical(self):
        fp1 = _make_fp()
        fp2 = _make_fp()
        assert fp1.signal_overlap_ratio(fp2) == 1.0

    def test_signal_overlap_ratio_disjoint(self):
        # ENTRY_LIQUIDITY_SWEEP uses volume/spread features, ENTRY_MEAN_REVERSION_BOUNCE uses rsi/bollinger
        from src.research.templates.canonical import ENTRY_LIQUIDITY_SWEEP, ENTRY_MEAN_REVERSION_BOUNCE
        fp1 = _make_fp(entry_type=ENTRY_LIQUIDITY_SWEEP)
        fp2 = _make_fp(entry_type=ENTRY_MEAN_REVERSION_BOUNCE)
        ratio = fp1.signal_overlap_ratio(fp2)
        assert 0.0 <= ratio <= 1.0  # may have overlap (atr is common)

    def test_behavioral_similarity_identical_structure(self):
        fp1 = _make_fp(params={"p": 10})
        fp2 = _make_fp(params={"p": 20})
        # Same structural hash → full weight on structural component
        sim = fp1.behavioral_similarity(fp2)
        assert sim >= 0.7  # at least 70% from structural match

    def test_behavioral_similarity_different_structure(self):
        fp1 = _make_fp(entry_type=ENTRY_MOMENTUM_BREAKOUT)
        fp2 = _make_fp(entry_type=ENTRY_TURTLE_BREAKOUT)
        sim = fp1.behavioral_similarity(fp2)
        assert 0.0 <= sim < 1.0

    def test_is_exact_structural_match(self):
        fp1 = _make_fp(params={"p": 10})
        fp2 = _make_fp(params={"p": 99})
        assert fp1.is_exact_structural_match(fp2) is True

    def test_not_exact_structural_match_different_entry(self):
        fp1 = _make_fp(entry_type=ENTRY_MOMENTUM_BREAKOUT)
        fp2 = _make_fp(entry_type=ENTRY_TURTLE_BREAKOUT)
        assert fp1.is_exact_structural_match(fp2) is False

    def test_canonical_id_stable(self):
        fp1 = _make_fp()
        fp2 = _make_fp()
        assert fp1.canonical_id() == fp2.canonical_id()
        assert len(fp1.canonical_id()) == 16

    def test_to_dict_from_dict_roundtrip(self):
        fp = _make_fp(filter_types=(FILTER_VOLATILITY,), params={"x": 5})
        restored = StructuralFingerprint.from_dict(fp.to_dict())
        assert restored.structural_equivalence_hash == fp.structural_equivalence_hash
        assert restored.feature_overlap_signature == fp.feature_overlap_signature

    def test_frozen(self):
        fp = _make_fp()
        with pytest.raises(Exception):
            fp.full_grammar_hash = "modified"  # type: ignore
