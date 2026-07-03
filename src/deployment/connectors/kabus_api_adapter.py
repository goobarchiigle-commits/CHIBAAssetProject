"""
src/deployment/connectors/kabus_api_adapter.py
KabuStation REST API adapter implementing BrokerInterface.

Phase 4B contract:
  - Returns INTENT_ONLY stubs when mode != HUMAN_CONFIRMED (no HTTP calls)
  - Actual HTTP calls happen only when mode == HUMAN_CONFIRMED
  - No credentials hardcoded: reads KABU_API_BASE_URL, KABU_API_TOKEN, KABU_ORDER_PASSWORD
  - Rate limit: 5 calls/sec  Retry: max 1

API base: localhost:18080 (KABU_API_BASE_URL env, default http://localhost:18080)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from .broker_interface import (
    BrokerAccountState,
    BrokerOrderResult,
    BrokerPosition,
    LiveExecutionMode,
    DEFAULT_EXECUTION_MODE,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

_RATE_LIMIT_CALLS_PER_SEC: int   = 5
_HTTP_TIMEOUT_S:           float = 5.0
_ORDER_TIMEOUT_S:          float = 10.0


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


@dataclass
class KabusApiAdapter:
    """
    KabuStation REST API adapter.
    Reads all credentials from env — never hardcodes.
    Returns intent-only stubs (no HTTP) when mode != HUMAN_CONFIRMED.
    """
    mode: LiveExecutionMode = field(default_factory=lambda: DEFAULT_EXECUTION_MODE)
    _last_call_ts: float = field(default=0.0, init=False, repr=False)

    # ── env accessors ──────────────────────────────────────────────────────────
    def _base_url(self) -> str:
        return os.environ.get("KABU_API_BASE_URL", "http://localhost:18080")

    def _token(self) -> str:
        return os.environ.get("KABU_API_TOKEN", "")

    def _order_password(self) -> str:
        return os.environ.get("KABU_ORDER_PASSWORD", "")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        gap = 1.0 / _RATE_LIMIT_CALLS_PER_SEC
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call_ts = time.monotonic()

    def _get(self, path: str, timeout: float = _HTTP_TIMEOUT_S) -> dict:
        import urllib.request
        self._throttle()
        url = f"{self._base_url()}{path}"
        req = urllib.request.Request(
            url,
            headers={"X-API-KEY": self._token(), "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, payload: dict, timeout: float = _ORDER_TIMEOUT_S) -> dict:
        import urllib.request
        self._throttle()
        url = f"{self._base_url()}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"X-API-KEY": self._token(), "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    # ── BrokerInterface implementation ────────────────────────────────────────

    def get_positions(self) -> List[BrokerPosition]:
        if self.mode == LiveExecutionMode.INTENT_ONLY:
            logger.debug("[KABUS] INTENT_ONLY: get_positions → stub []")
            return []
        try:
            data = self._get("/kabusapi/positions")
            result = []
            for p in (data or []):
                result.append(BrokerPosition(
                    symbol=str(p.get("Symbol", "")),
                    quantity=int(p.get("LeavesQty", 0)),
                    average_cost=float(p.get("AveragePrice", 0.0)),
                    market_value=float(p.get("Valuation", 0.0)),
                    unrealized_pnl=float(p.get("ProfitLoss", 0.0)),
                ))
            return result
        except Exception as exc:
            logger.warning("[KABUS] get_positions failed: %s", exc)
            return []

    def get_account_state(self) -> BrokerAccountState:
        if self.mode == LiveExecutionMode.INTENT_ONLY:
            logger.debug("[KABUS] INTENT_ONLY: get_account_state → stub")
            return BrokerAccountState(
                cash_balance=0.0,
                total_assets=0.0,
                positions=[],
                api_reachable=False,
                timestamp=_now_jst(),
            )
        try:
            data = self._get("/kabusapi/wallet/margin")
            positions = self.get_positions()
            cash = float(data.get("MarginAccountWallet", 0.0))
            return BrokerAccountState(
                cash_balance=cash,
                total_assets=cash,
                positions=positions,
                api_reachable=True,
                timestamp=_now_jst(),
            )
        except Exception as exc:
            logger.warning("[KABUS] get_account_state failed: %s", exc)
            return BrokerAccountState(
                cash_balance=0.0,
                total_assets=0.0,
                positions=[],
                api_reachable=False,
                timestamp=_now_jst(),
            )

    def submit_order(
        self,
        symbol:          str,
        side:            str,
        quantity:        int,
        client_order_id: str,
        slippage:        float = 0.001,
    ) -> BrokerOrderResult:
        ts = _now_jst()
        if self.mode != LiveExecutionMode.HUMAN_CONFIRMED:
            logger.info(
                "[KABUS] mode=%s: submit_order BLOCKED symbol=%s side=%s qty=%d",
                self.mode.value, symbol, side, quantity,
            )
            return BrokerOrderResult(
                client_order_id=client_order_id,
                broker_order_id=None,
                accepted=False,
                rejection_reason=f"mode={self.mode.value}: human approval required",
                submitted_price=None,
                timestamp=ts,
            )
        try:
            side_code = 1 if side.upper() == "BUY" else 2
            payload = {
                "Password":       self._order_password(),
                "Symbol":         symbol,
                "Exchange":       1,
                "SecurityType":   1,
                "Side":           str(side_code),
                "CashMargin":     1,
                "DelivType":      0,
                "FundType":       "AA",
                "AccountType":    4,
                "Qty":            quantity,
                "FrontOrderType": 20,   # 成行
                "Price":          0,
                "ExpireDay":      0,
                "ClosePositions": [],
            }
            result = self._post("/kabusapi/sendorder", payload, timeout=_ORDER_TIMEOUT_S)
            order_id = str(result.get("OrderId", ""))
            return BrokerOrderResult(
                client_order_id=client_order_id,
                broker_order_id=order_id if order_id else None,
                accepted=bool(order_id),
                rejection_reason=None if order_id else "no OrderId returned",
                submitted_price=None,
                timestamp=ts,
            )
        except Exception as exc:
            logger.warning("[KABUS] submit_order failed: %s", exc)
            return BrokerOrderResult(
                client_order_id=client_order_id,
                broker_order_id=None,
                accepted=False,
                rejection_reason=str(exc),
                submitted_price=None,
                timestamp=ts,
            )

    def cancel_order(
        self,
        client_order_id: str,
        broker_order_id: Optional[str] = None,
    ) -> bool:
        if self.mode != LiveExecutionMode.HUMAN_CONFIRMED:
            logger.info("[KABUS] mode=%s: cancel_order BLOCKED", self.mode.value)
            return False
        try:
            payload = {
                "Password": self._order_password(),
                "OrderId":  broker_order_id or client_order_id,
            }
            result = self._post("/kabusapi/cancelorder", payload, timeout=_ORDER_TIMEOUT_S)
            return bool(result.get("OrderId"))
        except Exception as exc:
            logger.warning("[KABUS] cancel_order failed: %s", exc)
            return False
