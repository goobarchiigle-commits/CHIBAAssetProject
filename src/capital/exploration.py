"""
exploration.py — Exploration Budget Layer

5% of effective_capital ring-fenced from production capital.
Isolated ledger: exploration draws NEVER reduce production deployable_capital.

Exploration trades are tagged; their PnL feeds ONLY the edge_model,
not the main capital hierarchy.

Design:
  exploration_budget = effective_capital × EXPLORATION_FRACTION (5%)
  exploration_deployed = sum of active exploration positions (value)
  exploration_available = max(0, exploration_budget - exploration_deployed)

Isolation invariant: production deployable_capital is computed BEFORE
exploration allocation. The exploration budget is additive overhead,
never subtracted from production.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EXPLORATION_FRACTION: float = 0.05    # 5% of effective_capital
MAX_EXPLORATION_POSITIONS: int = 1    # 1 exploration position at a time
MIN_EXPLORATION_BUDGET: float = 50_000.0  # minimum useful budget (¥50K)


@dataclass
class ExplorationState:
    exploration_fraction: float = EXPLORATION_FRACTION
    exploration_budget: float = 0.0      # current budget (updated daily)
    exploration_deployed: float = 0.0    # capital currently in exploration positions
    exploration_available: float = 0.0   # budget - deployed
    active_exploration_positions: list = field(default_factory=list)  # list of symbol strings
    cumulative_pnl_bps: float = 0.0      # lifetime exploration PnL in bps (for edge_model)
    exploration_active: bool = False     # True when exploration flag is set
    schema_version: int = 1


@dataclass
class ExplorationAllocation:
    symbol: str
    allocated_value: float
    budget_remaining_after: float
    approved: bool
    reason: str


def compute_exploration_budget(
    effective_capital: float,
    exploration_fraction: float = EXPLORATION_FRACTION,
) -> float:
    """Returns the exploration budget in JPY."""
    if effective_capital <= 0.0:
        return 0.0
    return round(effective_capital * max(0.0, min(0.10, exploration_fraction)), 2)


def update_exploration_state(
    state: ExplorationState,
    effective_capital: float,
    active_positions: Optional[list[str]] = None,
    position_values: Optional[dict[str, float]] = None,
) -> ExplorationState:
    """
    Recompute exploration budget and availability.

    Args:
        state:             current ExplorationState
        effective_capital: today's effective_capital from capital_state
        active_positions:  list of symbols currently held as exploration
        position_values:   {symbol: market_value} for held positions
    """
    budget = compute_exploration_budget(effective_capital, state.exploration_fraction)

    active = active_positions or []
    vals = position_values or {}
    deployed = sum(vals.get(sym, 0.0) for sym in active)
    available = max(0.0, budget - deployed)

    return ExplorationState(
        exploration_fraction=state.exploration_fraction,
        exploration_budget=round(budget, 2),
        exploration_deployed=round(deployed, 2),
        exploration_available=round(available, 2),
        active_exploration_positions=list(active),
        cumulative_pnl_bps=state.cumulative_pnl_bps,
        exploration_active=state.exploration_active,
    )


def assess_exploration_allocation(
    state: ExplorationState,
    symbol: str,
    requested_value: float,
) -> ExplorationAllocation:
    """
    Approve or reject an exploration trade.

    Approval conditions:
      1. exploration_active flag is set
      2. requested_value ≤ exploration_available
      3. not already holding max exploration positions
      4. exploration_budget ≥ MIN_EXPLORATION_BUDGET
    """
    if not state.exploration_active:
        return ExplorationAllocation(
            symbol=symbol,
            allocated_value=0.0,
            budget_remaining_after=state.exploration_available,
            approved=False,
            reason="exploration_active=False",
        )

    if state.exploration_budget < MIN_EXPLORATION_BUDGET:
        return ExplorationAllocation(
            symbol=symbol,
            allocated_value=0.0,
            budget_remaining_after=state.exploration_available,
            approved=False,
            reason=f"budget={state.exploration_budget:.0f} < min={MIN_EXPLORATION_BUDGET:.0f}",
        )

    if len(state.active_exploration_positions) >= MAX_EXPLORATION_POSITIONS:
        return ExplorationAllocation(
            symbol=symbol,
            allocated_value=0.0,
            budget_remaining_after=state.exploration_available,
            approved=False,
            reason=f"max_positions={MAX_EXPLORATION_POSITIONS} already active",
        )

    if symbol in state.active_exploration_positions:
        return ExplorationAllocation(
            symbol=symbol,
            allocated_value=0.0,
            budget_remaining_after=state.exploration_available,
            approved=False,
            reason=f"{symbol} already in exploration",
        )

    if requested_value > state.exploration_available:
        return ExplorationAllocation(
            symbol=symbol,
            allocated_value=0.0,
            budget_remaining_after=state.exploration_available,
            approved=False,
            reason=f"requested={requested_value:.0f} > available={state.exploration_available:.0f}",
        )

    remaining = state.exploration_available - requested_value
    return ExplorationAllocation(
        symbol=symbol,
        allocated_value=round(requested_value, 2),
        budget_remaining_after=round(remaining, 2),
        approved=True,
        reason=f"approved: budget={state.exploration_budget:.0f} available={state.exploration_available:.0f}",
    )


def record_exploration_pnl(
    state: ExplorationState,
    pnl_bps: float,
) -> ExplorationState:
    """Accumulate exploration PnL into the edge_model feedback signal."""
    return ExplorationState(
        exploration_fraction=state.exploration_fraction,
        exploration_budget=state.exploration_budget,
        exploration_deployed=state.exploration_deployed,
        exploration_available=state.exploration_available,
        active_exploration_positions=list(state.active_exploration_positions),
        cumulative_pnl_bps=round(state.cumulative_pnl_bps + pnl_bps, 4),
        exploration_active=state.exploration_active,
    )


def load_exploration_state(path: Path) -> ExplorationState:
    if not path.exists():
        return ExplorationState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = set(ExplorationState.__dataclass_fields__)
        return ExplorationState(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("exploration_state load failed (%s) — default", exc)
        return ExplorationState()


def save_exploration_state(state: ExplorationState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
