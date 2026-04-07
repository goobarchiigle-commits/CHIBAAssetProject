"""
execution_guard.py
発注直前のハードガード。WARNING ではなく実際に止める。
監視ログ（smoke test）とは完全に分離された制御層。
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

MIN_ACTIVE_SYMS = 3          # これを下回ったら発注停止
MAX_ORDERS_PER_DAY = 10      # 1日の最大発注件数


def check_execution_preconditions(
    active_syms: List[str],
    signals: List[dict],       # [{"symbol": "5411.T", "side": "BUY", "qty": 100, ...}]
    current_positions: Dict[str, int],
    daily_order_count: int = 0,
) -> dict:
    """
    発注前に必ず通すハードガード。

    Returns:
        {"status": "OK"}                      — 発注可
        {"status": "BLOCKED", "reason": ...}  — 発注停止
        {"status": "NO_SIGNAL"}               — シグナルなし（正常終了）
    """
    # Guard 1: ユニバースが狭すぎる
    if len(active_syms) < MIN_ACTIVE_SYMS:
        logger.error(
            "[GUARD] BLOCKED: universe_too_narrow "
            "active=%d < min=%d",
            len(active_syms), MIN_ACTIVE_SYMS,
        )
        return {
            "status": "BLOCKED",
            "reason": "universe_too_narrow",
            "detail": f"active_syms={len(active_syms)}, required={MIN_ACTIVE_SYMS}",
        }

    # Guard 2: シグナルなし
    if not signals:
        logger.info("[GUARD] No signals today.")
        return {"status": "NO_SIGNAL"}

    # Guard 3: 1日の発注件数上限
    if daily_order_count >= MAX_ORDERS_PER_DAY:
        logger.error(
            "[GUARD] BLOCKED: daily_order_limit count=%d >= max=%d",
            daily_order_count, MAX_ORDERS_PER_DAY,
        )
        return {
            "status": "BLOCKED",
            "reason": "daily_order_limit_exceeded",
            "detail": f"daily_count={daily_order_count}, max={MAX_ORDERS_PER_DAY}",
        }

    logger.info(
        "[GUARD] OK: active=%d, signals=%d, daily_orders=%d",
        len(active_syms), len(signals), daily_order_count,
    )
    return {"status": "OK"}
