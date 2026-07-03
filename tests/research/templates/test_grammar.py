"""Tests: grammar.py — AlphaGrammar, GrammarClause, GrammarExpression."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.grammar import (
    CLAUSE_ENTRY,
    CLAUSE_EXIT,
    CLAUSE_FILTER,
    CLAUSE_RISK,
    CLAUSE_REGIME,
    AlphaGrammar,
    GrammarClause,
    GrammarExpression,
)
from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    ENTRY_TURTLE_BREAKOUT,
    EXIT_TURTLE,
    FILTER_VOLATILITY,
    FILTER_LIQUIDITY,
    HOLD_SIGNAL_EXIT,
    SIZE_ATR_BASED,
)


class TestGrammarClause:
    def test_build_sorts_params(self):
        clause = GrammarClause.build(CLAUSE_ENTRY, ENTRY_MOMENTUM_BREAKOUT, {"b": 2, "a": 1})
        assert clause.param_signature[0] == ("a", "1")
        assert clause.param_signature[1] == ("b", "2")

    def test_frozen(self):
        clause = GrammarClause.build(CLAUSE_ENTRY, ENTRY_MOMENTUM_BREAKOUT, {"p": 10})
        with pytest.raises(Exception):
            clause.canonical_type = "X"  # type: ignore

    def test_invalid_clause_type_raises(self):
        with pytest.raises(ValueError, match="Invalid clause_type"):
            GrammarClause.build("INVALID_CLAUSE", ENTRY_MOMENTUM_BREAKOUT, {})

    def test_canonical_repr_format(self):
        clause = GrammarClause.build(CLAUSE_ENTRY, ENTRY_MOMENTUM_BREAKOUT, {"period": 20})
        assert "ENTRY:MOMENTUM_BREAKOUT" in clause.canonical_repr()
        assert "period=20" in clause.canonical_repr()

    def test_structural_repr_no_params(self):
        clause = GrammarClause.build(CLAUSE_ENTRY, ENTRY_MOMENTUM_BREAKOUT, {"period": 20})
        assert clause.structural_repr() == "ENTRY:MOMENTUM_BREAKOUT"
        assert "period" not in clause.structural_repr()

    def test_to_dict_from_dict_roundtrip(self):
        clause = GrammarClause.build(CLAUSE_FILTER, FILTER_VOLATILITY, {"atr_mult": 1.5})
        restored = GrammarClause.from_dict(clause.to_dict())
        assert restored == clause

    def test_deterministic_build(self):
        c1 = GrammarClause.build(CLAUSE_ENTRY, ENTRY_MOMENTUM_BREAKOUT, {"p": 1})
        c2 = GrammarClause.build(CLAUSE_ENTRY, ENTRY_MOMENTUM_BREAKOUT, {"p": 1})
        assert c1 == c2


class TestGrammarExpression:
    def _make_expr(self, filter_types=(FILTER_VOLATILITY,), params=None):
        params = params or {"entry_period": 20}
        return AlphaGrammar.build_expression(
            entry_type=ENTRY_MOMENTUM_BREAKOUT,
            exit_type=EXIT_TURTLE,
            filter_types=tuple(sorted(filter_types)),
            holding_model=HOLD_SIGNAL_EXIT,
            sizing_model=SIZE_ATR_BASED,
            regime_filter=None,
            parameters=params,
        )

    def test_entry_clause_first(self):
        expr = self._make_expr()
        assert expr.clauses[0].clause_type == CLAUSE_ENTRY

    def test_exit_clause_after_filters(self):
        expr = self._make_expr()
        clause_types = [c.clause_type for c in expr.clauses]
        entry_idx = clause_types.index(CLAUSE_ENTRY)
        exit_idx = clause_types.index(CLAUSE_EXIT)
        assert exit_idx > entry_idx

    def test_grammar_hash_deterministic(self):
        e1 = self._make_expr()
        e2 = self._make_expr()
        assert e1.grammar_hash() == e2.grammar_hash()

    def test_structural_hash_independent_of_params(self):
        e1 = self._make_expr(params={"entry_period": 10})
        e2 = self._make_expr(params={"entry_period": 20})
        assert e1.structural_hash() == e2.structural_hash()

    def test_grammar_hash_changes_with_params(self):
        e1 = self._make_expr(params={"entry_period": 10})
        e2 = self._make_expr(params={"entry_period": 20})
        assert e1.grammar_hash() != e2.grammar_hash()

    def test_different_entry_types_different_structural_hash(self):
        e1 = AlphaGrammar.build_expression(
            ENTRY_MOMENTUM_BREAKOUT, EXIT_TURTLE, (), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED, None, {}
        )
        e2 = AlphaGrammar.build_expression(
            ENTRY_TURTLE_BREAKOUT, EXIT_TURTLE, (), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED, None, {}
        )
        assert e1.structural_hash() != e2.structural_hash()

    def test_filter_order_invariant(self):
        e1 = AlphaGrammar.build_expression(
            ENTRY_MOMENTUM_BREAKOUT, EXIT_TURTLE,
            (FILTER_VOLATILITY, FILTER_LIQUIDITY), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED, None, {}
        )
        e2 = AlphaGrammar.build_expression(
            ENTRY_MOMENTUM_BREAKOUT, EXIT_TURTLE,
            (FILTER_LIQUIDITY, FILTER_VOLATILITY), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED, None, {}
        )
        assert e1.structural_hash() == e2.structural_hash()

    def test_canonical_repr_nonempty(self):
        expr = self._make_expr()
        assert len(expr.canonical_repr()) > 0

    def test_clause_types_returns_tuple(self):
        expr = self._make_expr()
        ct = expr.clause_types()
        assert isinstance(ct, tuple)
        assert CLAUSE_ENTRY in ct

    def test_to_dict_from_dict_roundtrip(self):
        expr = self._make_expr()
        restored = GrammarExpression.from_dict(expr.to_dict())
        assert restored.structural_hash() == expr.structural_hash()

    def test_structural_expression_no_params(self):
        expr = AlphaGrammar.structural_expression(
            ENTRY_MOMENTUM_BREAKOUT, EXIT_TURTLE, (), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED, None
        )
        for clause in expr.clauses:
            assert clause.param_signature == ()

    def test_regime_clause_present_when_regime_filter_set(self):
        expr = AlphaGrammar.build_expression(
            ENTRY_MOMENTUM_BREAKOUT, EXIT_TURTLE, (), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED,
            "TREND_ONLY_REGIME", {}
        )
        assert CLAUSE_REGIME in expr.clause_types()

    def test_no_regime_clause_when_none(self):
        expr = AlphaGrammar.build_expression(
            ENTRY_MOMENTUM_BREAKOUT, EXIT_TURTLE, (), HOLD_SIGNAL_EXIT, SIZE_ATR_BASED, None, {}
        )
        assert CLAUSE_REGIME not in expr.clause_types()
