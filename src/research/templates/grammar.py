"""AlphaGrammar — deterministic structural grammar for alpha hypotheses.

Grammar components:
    ENTRY, FILTER, EXIT, RISK, REBALANCE, REGIME, LIQUIDITY

Grammar outputs are canonical, hash-stable, and lineage-compatible.
Same structural choices → same hash, regardless of input ordering.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── Clause types ───────────────────────────────────────────────────────────────

CLAUSE_ENTRY = "ENTRY"
CLAUSE_FILTER = "FILTER"
CLAUSE_EXIT = "EXIT"
CLAUSE_RISK = "RISK"
CLAUSE_REBALANCE = "REBALANCE"
CLAUSE_REGIME = "REGIME"
CLAUSE_LIQUIDITY = "LIQUIDITY"

ALL_CLAUSE_TYPES: frozenset = frozenset({
    CLAUSE_ENTRY, CLAUSE_FILTER, CLAUSE_EXIT, CLAUSE_RISK,
    CLAUSE_REBALANCE, CLAUSE_REGIME, CLAUSE_LIQUIDITY,
})

# Canonical ordering for clauses in a grammar expression
_CLAUSE_ORDER: Dict[str, int] = {
    CLAUSE_ENTRY: 0,
    CLAUSE_FILTER: 1,
    CLAUSE_EXIT: 2,
    CLAUSE_RISK: 3,
    CLAUSE_REBALANCE: 4,
    CLAUSE_REGIME: 5,
    CLAUSE_LIQUIDITY: 6,
}


@dataclass(frozen=True)
class GrammarClause:
    """Immutable grammar clause: a single strategy component.

    clause_type: one of ALL_CLAUSE_TYPES
    canonical_type: from CanonicalSemantics
    param_signature: sorted tuple of (key, str(value)) pairs — parameter-stable identity
    """
    clause_type: str
    canonical_type: str
    param_signature: tuple  # sorted (key, str_value) pairs

    @staticmethod
    def build(
        clause_type: str,
        canonical_type: str,
        params: Dict[str, Any],
    ) -> "GrammarClause":
        if clause_type not in ALL_CLAUSE_TYPES:
            raise ValueError(f"Invalid clause_type: {clause_type!r}")
        sig = tuple(sorted((k, str(v)) for k, v in params.items()))
        return GrammarClause(clause_type=clause_type, canonical_type=canonical_type, param_signature=sig)

    def canonical_repr(self) -> str:
        params_str = ",".join(f"{k}={v}" for k, v in self.param_signature)
        return f"{self.clause_type}:{self.canonical_type}({params_str})"

    def structural_repr(self) -> str:
        """Grammar structure without parameter values."""
        return f"{self.clause_type}:{self.canonical_type}"

    def to_dict(self) -> dict:
        return {
            "clause_type": self.clause_type,
            "canonical_type": self.canonical_type,
            "param_signature": list(self.param_signature),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GrammarClause":
        return cls(
            clause_type=d["clause_type"],
            canonical_type=d["canonical_type"],
            param_signature=tuple(tuple(p) for p in d["param_signature"]),
        )

    def _sort_key(self) -> tuple:
        return (_CLAUSE_ORDER.get(self.clause_type, 99), self.canonical_type)


@dataclass(frozen=True)
class GrammarExpression:
    """Immutable ordered grammar expression (full hypothesis structure).

    clauses: sorted tuple of GrammarClauses in canonical order.
    """
    clauses: tuple  # tuple of GrammarClause

    @staticmethod
    def build(clauses: List[GrammarClause]) -> "GrammarExpression":
        """Sort clauses canonically: ENTRY first, FILTER sorted by type, EXIT, RISK, REBALANCE, REGIME, LIQUIDITY."""
        sorted_clauses = sorted(clauses, key=lambda c: (c._sort_key(), c.canonical_type))
        return GrammarExpression(clauses=tuple(sorted_clauses))

    def grammar_hash(self) -> str:
        """16-char SHA256 of full grammar including parameter values."""
        payload = json.dumps(
            [c.to_dict() for c in self.clauses],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def structural_hash(self) -> str:
        """16-char SHA256 of grammar structure WITHOUT parameter values.

        Two expressions with the same structural_hash are semantically equivalent
        modulo parameter choices.
        """
        payload = json.dumps(
            [c.structural_repr() for c in self.clauses],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def canonical_repr(self) -> str:
        """Human-readable canonical representation."""
        return " + ".join(c.canonical_repr() for c in self.clauses)

    def structural_repr(self) -> str:
        return " + ".join(c.structural_repr() for c in self.clauses)

    def clause_types(self) -> Tuple[str, ...]:
        return tuple(c.clause_type for c in self.clauses)

    def canonical_types_for(self, clause_type: str) -> Tuple[str, ...]:
        return tuple(c.canonical_type for c in self.clauses if c.clause_type == clause_type)

    def to_dict(self) -> dict:
        return {
            "clauses": [c.to_dict() for c in self.clauses],
            "grammar_hash": self.grammar_hash(),
            "structural_hash": self.structural_hash(),
            "canonical_repr": self.canonical_repr(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GrammarExpression":
        return cls(clauses=tuple(GrammarClause.from_dict(c) for c in d["clauses"]))


class AlphaGrammar:
    """Factory for building GrammarExpressions from hypothesis components.

    All outputs are deterministic: same inputs → same expression, always.
    """

    @staticmethod
    def build_expression(
        entry_type: str,
        exit_type: str,
        filter_types: tuple,         # sorted tuple of canonical filter type strings
        holding_model: str,
        sizing_model: str,
        regime_filter: Optional[str],
        parameters: Dict[str, Any],  # all hypothesis parameters
    ) -> GrammarExpression:
        """Build a canonical GrammarExpression from hypothesis components.

        Parameters are attached to the ENTRY clause for structural identity.
        """
        clauses: List[GrammarClause] = []

        # ENTRY clause carries all parameters
        clauses.append(GrammarClause.build(CLAUSE_ENTRY, entry_type, parameters))

        # FILTER clauses (one per filter type, no params at grammar level)
        for ft in sorted(filter_types):
            clauses.append(GrammarClause.build(CLAUSE_FILTER, ft, {}))

        # EXIT clause with exit-relevant parameters
        exit_params = {k: v for k, v in parameters.items() if "exit" in k.lower() or "hold" in k.lower()}
        clauses.append(GrammarClause.build(CLAUSE_EXIT, exit_type, exit_params))

        # RISK clause with sizing model
        risk_params = {k: v for k, v in parameters.items() if "atr" in k.lower() or "stop" in k.lower()}
        clauses.append(GrammarClause.build(CLAUSE_RISK, sizing_model, risk_params))

        # REGIME clause (if regime filter specified)
        if regime_filter is not None:
            clauses.append(GrammarClause.build(CLAUSE_REGIME, regime_filter, {}))

        return GrammarExpression.build(clauses)

    @staticmethod
    def structural_expression(
        entry_type: str,
        exit_type: str,
        filter_types: tuple,
        holding_model: str,
        sizing_model: str,
        regime_filter: Optional[str],
    ) -> GrammarExpression:
        """Build structural-only expression (no parameter values).

        Used for structural fingerprinting and family grouping.
        """
        return AlphaGrammar.build_expression(
            entry_type=entry_type,
            exit_type=exit_type,
            filter_types=filter_types,
            holding_model=holding_model,
            sizing_model=sizing_model,
            regime_filter=regime_filter,
            parameters={},
        )
