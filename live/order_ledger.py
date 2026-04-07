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

    def is_duplicate(self, symbol: str, side: str = "BUY") -> bool:
        """同一 execution_key が既に記録されていれば True"""
        key = self.execution_key(symbol, side)
        return key in self._ledger.get("orders", {})

    def daily_count(self) -> int:
        return self._ledger.get("count", 0)

    def check_and_record(
        self,
        symbol: str,
        side: str = "BUY",
        qty: int = 0,
        price: float = 0.0,
    ) -> dict:
        """
        発注可否チェック + 記録。
        発注 API 呼び出し前に必ず呼ぶ（P1-4 原則）。

        Returns:
            {"allowed": True, "execution_key": key}  — 発注可
            {"allowed": False, "reason": ...}         — 発注不可
        """
        # 1日の発注件数チェック
        if self.daily_count() >= MAX_ORDERS_PER_DAY:
            logger.error("[LEDGER] Blocked %s: daily limit %d", symbol, MAX_ORDERS_PER_DAY)
            return {"allowed": False, "reason": "daily_limit_exceeded"}

        # 同一銘柄・同一サイドの重複チェック
        if self.is_duplicate(symbol, side):
            logger.warning("[LEDGER] Blocked %s %s: duplicate order", symbol, side)
            return {"allowed": False, "reason": "duplicate_order"}

        # 記録（この時点でロック — 発注 API 呼び出し前）
        key = self.execution_key(symbol, side)
        orders = self._ledger.setdefault("orders", {})
        orders[key] = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "recorded_at": str(self.trade_date),
        }
        self._ledger["count"] = self._ledger.get("count", 0) + 1
        self._ledger["date"] = str(self.trade_date)
        self._save()

        logger.info(
            "[LEDGER] Recorded %s %s qty=%d (daily_count=%d)",
            symbol, side, qty, self._ledger["count"],
        )
        return {"allowed": True, "execution_key": key}
