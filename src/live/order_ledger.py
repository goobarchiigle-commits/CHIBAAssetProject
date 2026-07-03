"""
order_ledger.py
当日の発注履歴を管理する。idempotency key で二重発注を防止。
execution_key = f"{trade_date}_{symbol}_{side}"

runtime/execution_ledger.json に永続化。
既存の ORDER_LOCK_FILE（process-lock 用）とは完全に別ファイル。
"""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict

from src.kabusapi.signal_bridge import AbortError
from src.paths import RUNTIME_DIR

logger = logging.getLogger(__name__)

LEDGER_PATH: Path = RUNTIME_DIR / "execution_ledger.json"
MAX_ORDERS_PER_DAY = 10
MAX_ORDER_PER_SYMBOL = 1


class OrderLedger:
    """当日発注記録。重複チェックとカウント管理。"""

    def __init__(self, trade_date: date = None):
        self.trade_date = trade_date or date.today()
        self._ledger: Dict = self._load()

    def _load(self) -> Dict:
        if not LEDGER_PATH.exists():
            return {"date": str(self.trade_date), "orders": {}, "count": 0}
        try:
            with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 当日分のみ保持（古い日付はクリア）
            today_str = str(self.trade_date)
            if data.get("date") != today_str:
                return {"date": today_str, "orders": {}, "count": 0}
            orders = data.get("orders", {})
            active_count = sum(
                1 for order in orders.values()
                if order.get("execution_status") != "shadow"
            )
            data["count"] = active_count
            return data
        except Exception as e:
            logger.error("[LEDGER] Load failed: %s", e)
            return {"date": str(self.trade_date), "orders": {}, "count": 0}

    def _save(self) -> None:
        try:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER_PATH, "w", encoding="utf-8") as f:
                json.dump(self._ledger, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[LEDGER] Save failed: %s", e)

    def execution_key(self, symbol: str, side: str) -> str:
        return f"{self.trade_date}_{symbol}_{side}"

    # pending / shadow / failed 以外（= "sent" など実際に送信済み）のみをカウント対象とする
    _INACTIVE_STATUSES = frozenset({"shadow", "failed"})

    def is_duplicate(self, symbol: str, side: str = "BUY") -> bool:
        """同一 execution_key の live 記録が既にあれば True。shadow / failed は除外。"""
        key = self.execution_key(symbol, side)
        order = self._ledger.get("orders", {}).get(key)
        if not order:
            return False
        return order.get("execution_status") not in self._INACTIVE_STATUSES

    def daily_count(self) -> int:
        orders = self._ledger.get("orders", {})
        return sum(
            1 for order in orders.values()
            if order.get("execution_status") not in self._INACTIVE_STATUSES
        )

    def record_shadow(
        self,
        symbol: str,
        side: str = "BUY",
        qty: int = 0,
        price: float = 0.0,
    ) -> dict:
        """shadow 実行を監査用に記録する。live の重複判定対象には含めない。"""
        return self.check_and_record(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            execution_status="shadow",
        )

    def check_and_record(
        self,
        symbol: str,
        side: str = "BUY",
        qty: int = 0,
        price: float = 0.0,
        execution_status: str = "pending",
    ) -> dict:
        """
        発注可否チェック + 記録。
        発注 API 呼び出し前に必ず呼ぶ（P1-4 原則）。

        Returns:
            {"allowed": True, "execution_key": key}  — 発注可
        """
        is_shadow = execution_status == "shadow"

        # 1日の発注件数チェック
        if not is_shadow and self.daily_count() >= MAX_ORDERS_PER_DAY:
            logger.error("[LEDGER] Blocked %s: daily limit %d", symbol, MAX_ORDERS_PER_DAY)
            raise RuntimeError(f"daily_limit_exceeded: {symbol}")

        # 同一銘柄・同一サイドの重複チェック
        if not is_shadow and self.is_duplicate(symbol, side):
            logger.warning("[LEDGER] Blocked %s %s: duplicate order", symbol, side)
            raise AbortError(
                "duplicate_order_detected",
                f"duplicate order detected: {symbol} {side}",
            )

        # 記録（この時点でロック — 発注 API 呼び出し前）
        key = self.execution_key(symbol, side)
        orders = self._ledger.setdefault("orders", {})
        orders[key] = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "recorded_at": str(self.trade_date),
            "execution_status": execution_status,
        }
        self._ledger["count"] = self.daily_count()
        self._ledger["date"] = str(self.trade_date)
        self._save()

        logger.info(
            "[LEDGER] Recorded %s %s qty=%d status=%s (daily_count=%d)",
            symbol, side, qty, execution_status, self._ledger["count"],
        )
        return {
            "allowed": True,
            "execution_key": key,
            "execution_status": execution_status,
        }

    def mark_submitted(self, symbol: str, side: str, order_id: str = "") -> None:
        """
        pending 状態の注文を submitted に更新する。発注 API 成功後に呼ぶ。

        order_id を記録することで翌日以降の監査証跡として機能する。
        """
        key = self.execution_key(symbol, side)
        orders = self._ledger.get("orders", {})
        if key not in orders:
            logger.warning("[LEDGER] mark_submitted: key not found %s", key)
            return
        orders[key]["execution_status"] = "submitted"
        orders[key]["order_id"] = order_id
        self._ledger["count"] = self.daily_count()
        self._save()
        logger.info("[LEDGER] mark_submitted %s %s order_id=%s", symbol, side, order_id)

    def mark_failed(self, symbol: str, side: str, error: str = "") -> None:
        """
        pending 状態の注文を failed に更新する。発注 API 失敗時に呼ぶ。

        pending のまま残すと翌回の is_duplicate() でブロックされるため、
        失敗が確定した時点で必ず呼ぶこと。
        is_duplicate() は failed を除外するため翌日の duplicate guard を汚染しない。
        """
        key = self.execution_key(symbol, side)
        orders = self._ledger.get("orders", {})
        if key not in orders:
            logger.warning("[LEDGER] mark_failed: key not found %s", key)
            return
        orders[key]["execution_status"] = "failed"
        orders[key]["error"] = error
        self._ledger["count"] = self.daily_count()
        self._save()
        logger.warning("[LEDGER] mark_failed %s %s error=%s", symbol, side, error)
