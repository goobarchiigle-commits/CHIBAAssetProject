"""Tests: constraints.py — ConstraintHierarchy and constraint dataclasses."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.templates.constraints import (
    ConstraintHierarchy,
    HardConstraints,
    ResearchConstraints,
    ExchangeConstraints,
    LiquidityConstraints,
    SurvivorshipConstraints,
)
from src.research.templates.canonical import (
    ENTRY_MOMENTUM_BREAKOUT,
    EXIT_TURTLE,
    FILTER_VOLATILITY,
    HOLD_SIGNAL_EXIT,
    SIZE_ATR_BASED,
)


class TestHardConstraints:
    def test_defaults(self):
        hc = HardConstraints()
        assert hc.min_trades == 5
        assert hc.max_sharpe == 3.0
        assert hc.max_drawdown == 0.5

    def test_frozen(self):
        hc = HardConstraints()
        with pytest.raises(Exception):
            hc.min_trades = 99  # type: ignore

    def test_to_dict_keys(self):
        d = HardConstraints().to_dict()
        assert "min_trades" in d
        assert "max_sharpe" in d
        assert "max_drawdown" in d


class TestResearchConstraints:
    def test_defaults(self):
        rc = ResearchConstraints()
        assert rc.min_parameter_cardinality == 2
        assert rc.max_parameter_cardinality == 10

    def test_frozen(self):
        rc = ResearchConstraints()
        with pytest.raises(Exception):
            rc.min_parameter_cardinality = 1  # type: ignore

    def test_to_dict_keys(self):
        d = ResearchConstraints().to_dict()
        assert "min_parameter_cardinality" in d
        assert "max_parameter_cardinality" in d


class TestExchangeConstraints:
    def test_defaults_claude_md(self):
        ec = ExchangeConstraints()
        assert ec.slippage == 0.001
        assert ec.commission == 0.00055

    def test_frozen(self):
        ec = ExchangeConstraints()
        with pytest.raises(Exception):
            ec.slippage = 0.005  # type: ignore

    def test_to_dict_has_slippage(self):
        d = ExchangeConstraints().to_dict()
        assert "slippage" in d
        assert "commission" in d


class TestConstraintHierarchy:
    def test_default_factory(self):
        ch = ConstraintHierarchy.default()
        assert isinstance(ch.hard, HardConstraints)
        assert isinstance(ch.research, ResearchConstraints)
        assert isinstance(ch.exchange, ExchangeConstraints)
        assert isinstance(ch.liquidity, LiquidityConstraints)
        assert isinstance(ch.survivorship, SurvivorshipConstraints)

    def test_check_parameter_domain_valid(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("entry_period", [5, 10, 20])
        assert valid
        assert violations == []

    def test_check_parameter_domain_too_few(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("entry_period", [10])
        assert not valid
        assert any("cardinality" in v for v in violations)

    def test_check_parameter_domain_too_many(self):
        ch = ConstraintHierarchy(
            research=ResearchConstraints(max_parameter_cardinality=3)
        )
        valid, violations = ch.check_parameter_domain("entry_period", [1, 2, 3, 4])
        assert not valid
        assert any("cardinality" in v for v in violations)

    def test_check_parameter_domain_period_too_small(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("entry_period", [1, 2])
        assert not valid
        assert any(str(1) in v or "min" in v for v in violations)

    def test_check_parameter_domain_period_too_large(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("entry_period", [5, 300])
        assert not valid

    def test_check_parameter_domain_atr_mult_valid(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("atr_mult", [1.0, 2.0])
        assert valid

    def test_check_parameter_domain_atr_mult_too_small(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("atr_mult", [0.1, 2.0])
        assert not valid

    def test_check_parameter_domain_atr_mult_too_large(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_parameter_domain("atr_mult", [1.0, 10.0])
        assert not valid

    def test_check_all_domains_valid(self):
        ch = ConstraintHierarchy()
        domains = {
            "entry_period": [10, 20],
            "exit_period": [20, 40],
        }
        valid, violations = ch.check_all_domains(domains)
        assert valid

    def test_check_all_domains_aggregates_violations(self):
        ch = ConstraintHierarchy()
        domains = {
            "entry_period": [1],  # too few and too small
            "exit_period": [1],   # too few and too small
        }
        valid, violations = ch.check_all_domains(domains)
        assert not valid
        assert len(violations) >= 2

    def test_check_hypothesis_params_valid(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_hypothesis_params({"period": 10, "atr": 2.0})
        assert valid

    def test_check_hypothesis_params_non_primitive(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_hypothesis_params({"param": [1, 2, 3]})
        assert not valid

    def test_check_template_structure_valid(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_template_structure(
            (ENTRY_MOMENTUM_BREAKOUT,),
            (EXIT_TURTLE,),
            ((), (FILTER_VOLATILITY,)),
            (HOLD_SIGNAL_EXIT,),
            (SIZE_ATR_BASED,),
            {"entry_period": [10, 20]},
        )
        assert valid

    def test_check_template_structure_no_entry(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_template_structure(
            (), (EXIT_TURTLE,), ((),), (HOLD_SIGNAL_EXIT,), (SIZE_ATR_BASED,),
            {"entry_period": [10, 20]},
        )
        assert not valid
        assert any("entry_type" in v for v in violations)

    def test_check_template_structure_no_exit(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_template_structure(
            (ENTRY_MOMENTUM_BREAKOUT,), (), ((),), (HOLD_SIGNAL_EXIT,), (SIZE_ATR_BASED,),
            {"entry_period": [10, 20]},
        )
        assert not valid

    def test_check_template_structure_no_params(self):
        ch = ConstraintHierarchy()
        valid, violations = ch.check_template_structure(
            (ENTRY_MOMENTUM_BREAKOUT,), (EXIT_TURTLE,), ((),),
            (HOLD_SIGNAL_EXIT,), (SIZE_ATR_BASED,), {},
        )
        assert not valid
        assert any("parameter_domains" in v for v in violations)

    def test_to_dict_all_levels(self):
        d = ConstraintHierarchy().to_dict()
        assert "hard" in d
        assert "research" in d
        assert "exchange" in d
        assert "liquidity" in d
        assert "survivorship" in d
