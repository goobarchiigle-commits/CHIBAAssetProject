"""
src/deployment/connectors/broker_interface.py
BrokerInterface Protocol + LiveExecutionMode enum.

Purpose:
  Stable Protocol that MicroCapitalDeployment and governance layers depend on.
  Governance MUST NOT import kabustation API directly.
  Adapter isolation: future broker replacement requires only a new adapter.

Execution modes:
  INTENT_ONLY     — generate broker intents; no submission ever
  HUMAN_CONFIRMED — intents submitted only after explicit human approval
  FULLY_AUTONOMOUS — implemented but forbidden as Phase 4B default

DEFAULT_EXECUTION_MODE = HUMAN_CONFIRMED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable


class LiveExecutionMode(Enum):
    INTENT_ONLY      = "intent_only"
    HUMAN_CONFIRMED  = "human_confirmed"
    FULLY_AUTONOMOUS = "fully_autonomous"


# Phase 4B default — never change to FULLY_AUTONOMOUS without explicit user approval
DEFAULT_EXECUTION_MODE: LiveExecutionMode = LiveExecutionMode.HUMAN_CONFIRMED


@dataclass(frozen=True)
class BrokerPosition:
    symbol:          str
    quantity:        int
    average_cost:    float
    market_value:    float
    unrealized_pnl:  float


@dataclass(frozen=True)
class BrokerAccountState:
    cash_balance:   float
    total_assets:   float
    positions:      List[BrokerPosition]
    api_reachable:  bool
    timestamp:      str


@dataclass(frozen=True)
class BrokerOrderResult:
    client_order_id:   str
    broker_order_id:   Optional[str]
    accepted:          bool
    rejection_reason:  Optional[str]
    submitted_price:   Optional[float]
    timestamp:         str


@runtime_checkable
class BrokerInterface(Protocol):
    """
    Minimal broker abstraction for governance layer.
    Concrete adapters implement this Protocol.
    MicroCapitalDeployment depends only on BrokerInterface — never on adapters directly.
    """

    def get_positions(self) -> List[BrokerPosition]: ...

    def get_account_state(self) -> BrokerAccountState: ...

    def submit_order(
        self,
        symbol:          str,
        side:            str,        # "BUY" | "SELL"
        quantity:        int,
        client_order_id: str,
        slippage:        float = 0.001,
    ) -> BrokerOrderResult: ...

    def cancel_order(
        self,
        client_order_id: str,
        broker_order_id: Optional[str] = None,
    ) -> bool: ...
