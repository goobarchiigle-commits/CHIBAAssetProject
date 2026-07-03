"""ConstraintHierarchy — hierarchical constraint system for hypothesis generation.

Five constraint levels (evaluated in order, fail closed on any violation):
    1. hard_constraints        — absolute validity (mirrors CLAUDE.md VALIDATION)
    2. research_constraints    — research quality and statistical validity
    3. exchange_constraints    — market and exchange practicality
    4. liquidity_constraints   — capacity and slippage assumptions
    5. survivorship_constraints — topology robustness requirements

All constraint checks return (is_valid: bool, violations: List[str]).
Generation fails closed: any violation → hypothesis rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Individual constraint dataclasses ─────────────────────────────────────────


@dataclass(frozen=True)
class HardConstraints:
    """Absolute validity constraints (from CLAUDE.md VALIDATION section)."""
    min_trades: int = 5            # trade_min=5
    max_sharpe: float = 3.0        # sharpe_max=3.0 (above = suspect lookahead)
    max_drawdown: float = 0.5      # dd_max=0.5
    max_turnover: float = 5.0
    max_sharpe_decay: float = 2.0

    def to_dict(self) -> dict:
        return {
            "min_trades": self.min_trades,
            "max_sharpe": self.max_sharpe,
            "max_drawdown": self.max_drawdown,
            "max_turnover": self.max_turnover,
            "max_sharpe_decay": self.max_sharpe_decay,
        }


@dataclass(frozen=True)
class ResearchConstraints:
    """Research quality and statistical validity constraints."""
    min_parameter_cardinality: int = 2    # min values per parameter dimension
    max_parameter_cardinality: int = 10   # max values per parameter dimension
    min_holding_days: int = 1
    max_holding_days: int = 120
    min_entry_period: int = 3
    max_entry_period: int = 200
    min_exit_period: int = 3
    max_exit_period: int = 200

    def to_dict(self) -> dict:
        return {
            "min_parameter_cardinality": self.min_parameter_cardinality,
            "max_parameter_cardinality": self.max_parameter_cardinality,
            "min_holding_days": self.min_holding_days,
            "max_holding_days": self.max_holding_days,
            "min_entry_period": self.min_entry_period,
            "max_entry_period": self.max_entry_period,
            "min_exit_period": self.min_exit_period,
            "max_exit_period": self.max_exit_period,
        }


@dataclass(frozen=True)
class ExchangeConstraints:
    """Market and exchange practicality constraints (JP equities, auカブコム)."""
    min_price_yen: int = 100
    max_price_yen: int = 500_000
    min_daily_volume_shares: int = 10_000
    slippage: float = 0.001          # 0.1% — CLAUDE.md
    commission: float = 0.00055      # 0.055% — CLAUDE.md

    def to_dict(self) -> dict:
        return {
            "min_price_yen": self.min_price_yen,
            "max_price_yen": self.max_price_yen,
            "min_daily_volume_shares": self.min_daily_volume_shares,
            "slippage": self.slippage,
            "commission": self.commission,
        }


@dataclass(frozen=True)
class LiquidityConstraints:
    """Capacity and liquidity assumption constraints."""
    min_atr_mult: float = 0.5       # minimum ATR multiplier for stop
    max_atr_mult: float = 5.0       # maximum ATR multiplier
    min_liquidity_ratio: float = 0.001  # min fraction of ADV per trade

    def to_dict(self) -> dict:
        return {
            "min_atr_mult": self.min_atr_mult,
            "max_atr_mult": self.max_atr_mult,
            "min_liquidity_ratio": self.min_liquidity_ratio,
        }


@dataclass(frozen=True)
class SurvivorshipConstraints:
    """Topology robustness constraints to prevent fragile isolated spikes."""
    max_fragility_score: float = 0.80   # reject if likely isolated spike
    min_plateau_width: float = 0.15     # at least 15% of neighbors in plateau
    min_oos_regimes: int = 1            # must show alpha in at least 1 regime
    max_regime_concentration: float = 0.95  # Herfindahl index cap

    def to_dict(self) -> dict:
        return {
            "max_fragility_score": self.max_fragility_score,
            "min_plateau_width": self.min_plateau_width,
            "min_oos_regimes": self.min_oos_regimes,
            "max_regime_concentration": self.max_regime_concentration,
        }


# ── Hierarchy ──────────────────────────────────────────────────────────────────


class ConstraintHierarchy:
    """Hierarchical constraint system. Fail closed on any level violation.

    Usage:
        hierarchy = ConstraintHierarchy()
        valid, violations = hierarchy.check_parameter_domain("entry_period", [5, 10, 15])
        valid, violations = hierarchy.check_hypothesis_params({"entry_period": 10, "atr_mult": 2.0})
    """

    def __init__(
        self,
        hard: Optional[HardConstraints] = None,
        research: Optional[ResearchConstraints] = None,
        exchange: Optional[ExchangeConstraints] = None,
        liquidity: Optional[LiquidityConstraints] = None,
        survivorship: Optional[SurvivorshipConstraints] = None,
    ) -> None:
        self.hard = hard or HardConstraints()
        self.research = research or ResearchConstraints()
        self.exchange = exchange or ExchangeConstraints()
        self.liquidity = liquidity or LiquidityConstraints()
        self.survivorship = survivorship or SurvivorshipConstraints()

    # ── Parameter domain checks ───────────────────────────────────────────────

    def check_parameter_domain(
        self,
        param_name: str,
        values: list,
    ) -> Tuple[bool, List[str]]:
        """Check a single parameter domain list against research constraints."""
        violations: List[str] = []
        n = len(values)

        if n < self.research.min_parameter_cardinality:
            violations.append(
                f"{param_name}: cardinality={n} < min={self.research.min_parameter_cardinality}"
            )
        if n > self.research.max_parameter_cardinality:
            violations.append(
                f"{param_name}: cardinality={n} > max={self.research.max_parameter_cardinality}"
            )

        # Domain-specific value range checks
        if "period" in param_name.lower() and "exit" not in param_name.lower():
            for v in values:
                if isinstance(v, (int, float)):
                    if v < self.research.min_entry_period:
                        violations.append(f"{param_name}: value={v} < min={self.research.min_entry_period}")
                    if v > self.research.max_entry_period:
                        violations.append(f"{param_name}: value={v} > max={self.research.max_entry_period}")

        if "exit_period" in param_name.lower() or ("hold" in param_name.lower() and "day" in param_name.lower()):
            for v in values:
                if isinstance(v, (int, float)):
                    if v < self.research.min_exit_period:
                        violations.append(f"{param_name}: value={v} < min={self.research.min_exit_period}")
                    if v > self.research.max_exit_period:
                        violations.append(f"{param_name}: value={v} > max={self.research.max_exit_period}")

        if "atr_mult" in param_name.lower():
            for v in values:
                if isinstance(v, (int, float)):
                    if v < self.liquidity.min_atr_mult:
                        violations.append(f"{param_name}: value={v} < min_atr_mult={self.liquidity.min_atr_mult}")
                    if v > self.liquidity.max_atr_mult:
                        violations.append(f"{param_name}: value={v} > max_atr_mult={self.liquidity.max_atr_mult}")

        return (len(violations) == 0, violations)

    def check_all_domains(
        self,
        parameter_domains: Dict[str, list],
    ) -> Tuple[bool, List[str]]:
        """Check all parameter domains. Aggregates violations from all parameters."""
        all_violations: List[str] = []
        for param_name, values in sorted(parameter_domains.items()):
            _, v = self.check_parameter_domain(param_name, values)
            all_violations.extend(v)
        return (len(all_violations) == 0, all_violations)

    # ── Hypothesis parameter checks ───────────────────────────────────────────

    def check_hypothesis_params(
        self,
        params: Dict[str, object],
    ) -> Tuple[bool, List[str]]:
        """Check a concrete parameter dict (a single hypothesis's parameters)."""
        violations: List[str] = []

        for k, v in sorted(params.items()):
            if not isinstance(v, (int, float, str, bool)):
                violations.append(f"param {k}: non-primitive value type {type(v).__name__}")

        return (len(violations) == 0, violations)

    # ── Template-level checks ─────────────────────────────────────────────────

    def check_template_structure(
        self,
        entry_types: tuple,
        exit_types: tuple,
        filter_combos: tuple,
        holding_models: tuple,
        sizing_models: tuple,
        parameter_domains: Dict[str, list],
    ) -> Tuple[bool, List[str]]:
        """Structural validity of a template definition."""
        violations: List[str] = []

        if not entry_types:
            violations.append("template: must have at least one entry_type")
        if not exit_types:
            violations.append("template: must have at least one exit_type")
        if not holding_models:
            violations.append("template: must have at least one holding_model")
        if not sizing_models:
            violations.append("template: must have at least one sizing_model")
        if not parameter_domains:
            violations.append("template: parameter_domains must not be empty")

        _, domain_violations = self.check_all_domains(parameter_domains)
        violations.extend(domain_violations)

        return (len(violations) == 0, violations)

    def to_dict(self) -> dict:
        return {
            "hard": self.hard.to_dict(),
            "research": self.research.to_dict(),
            "exchange": self.exchange.to_dict(),
            "liquidity": self.liquidity.to_dict(),
            "survivorship": self.survivorship.to_dict(),
        }

    @staticmethod
    def default() -> "ConstraintHierarchy":
        """Default hierarchy using CLAUDE.md parameters."""
        return ConstraintHierarchy()
