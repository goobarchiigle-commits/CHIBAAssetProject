"""
position_normalizer.py
kabu API /positions レスポンスの正規化レイヤー。

kabu Station は売却済み銘柄を qty=0 のまま返し続ける場合がある。
ビジネスロジック（スロット計算・エクスポージャー・シグナル生成）は
この関数で正規化済みのリストのみを受け取る。

使用箇所:
  - signal_bridge._get_current_positions()
  - signal_bridge.get_broker_positions()
  - scripts/sync_positions.py
  - live/position_sync.py (経由して signal_bridge に委譲)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# kabu API の qty フィールド名。優先順で探索する。
_QTY_KEYS = ("LeavesQty", "LeaveQty", "Qty", "HoldQty")


def _extract_qty(pos: dict) -> float:
    """
    kabu API レスポンス dict から有効 qty を取り出す。
    フィールドが存在しない・None・非数値の場合は 0.0 を返す。
    """
    for key in _QTY_KEYS:
        val = pos.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def filter_live_positions(positions: list[dict]) -> list[dict]:
    """
    qty > 0 のポジションだけを返す。

    Args:
        positions: KabuClient.get_positions() の生レスポンス

    Returns:
        有効保有（qty > 0）のみのリスト。呼び出し元は
        このリスト以外でビジネスロジックを動かしてはならない。

    Examples:
        >>> filter_live_positions([
        ...     {"Symbol": "5401", "LeavesQty": 0},
        ...     {"Symbol": "5411", "LeavesQty": 200},
        ... ])
        [{"Symbol": "5411", "LeavesQty": 200}]
    """
    live: list[dict] = []
    zero: list[str] = []

    for p in positions:
        qty = _extract_qty(p)
        if qty > 0:
            live.append(p)
        else:
            zero.append(p.get("Symbol", "?"))

    if zero:
        logger.info(
            "[POSITION_NORM] qty=0 として除外 (%d件): %s",
            len(zero), zero,
        )

    return live
