"""
src/monitoring/gate.py
発注前ゲート: pre_trade_gate() が False を返したら発注禁止。
SOFT 条件は通過（controller が対応）、HARD 条件のみブロック。
純粋関数。副作用なし。
"""
from __future__ import annotations

from typing import Any

# ── ハード上限（変更禁止） ───────────────────────────────────────────────────────
RISK_HARD_CAP = 5.0    # risk_ratio > this → block
SLIP_HARD_CAP = 0.01   # slip_diff  > this → block


def _gate_legacy(
    state: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[bool, str]:
    """旧シグネチャ互換ゲート: pre_trade_gate(state, metrics)"""
    if state.get("halted"):
        reason = state.get("halt_reason") or "HALT"
        return False, f"HALT: {reason}"

    if metrics.get("status") == "HALT_CORRUPT":
        return False, f"HALT_CORRUPT: {metrics.get('corrupt_reason', 'unknown')}"

    risk_ratio = float(metrics.get("risk_ratio", 0.0))
    slip_diff  = float(metrics.get("slip_diff",  0.0))

    if risk_ratio > RISK_HARD_CAP:
        return False, f"RISK_HARD_CAP超過: risk_ratio={risk_ratio:.3f} > {RISK_HARD_CAP}"

    if slip_diff > SLIP_HARD_CAP:
        return False, f"SLIP_HARD_CAP超過: slip_diff={slip_diff:.5f} > {SLIP_HARD_CAP}"

    return True, "OK"


def _gate_new(
    order: dict[str, Any],
    state: dict[str, Any],
    metrics: dict[str, Any],
    cfg: dict,
) -> tuple[bool, str]:
    """
    Config-driven ゲート: pre_trade_gate(order, state, metrics, cfg)

    チェック順序（上位で False になれば即リターン）:
        1. HALT ラッチ
        2. RISK_HARD_CAP（cfg キーで上書き可、欠損時は無効）
        3. MAX_ORDERS_PER_MIN レートリミット
        4. MAX_SAME_SIDE オーバーエクスポージャー
        5. MAX_PRICE_DEVIATION 価格乖離
    """
    if state.get("halted"):
        return False, "HALT"

    risk_ratio = float(metrics.get("risk_ratio", 0.0))
    if risk_ratio > cfg.get("RISK_HARD_CAP", float("inf")):
        return False, "RISK_HARD"

    if state.get("orders_last_min", 0) > cfg.get("MAX_ORDERS_PER_MIN", float("inf")):
        return False, "RATE_LIMIT"

    if order.get("side") == state.get("last_side"):
        if state.get("same_side_count", 0) > cfg.get("MAX_SAME_SIDE", float("inf")):
            return False, "OVEREXPOSURE"

    mid = float(order.get("mid_price", 0)) + 1e-12
    dev = abs(float(order.get("price", mid)) - mid) / mid
    if dev > cfg.get("MAX_PRICE_DEVIATION", float("inf")):
        return False, "BAD_PRICE"

    return True, "OK"


def pre_trade_gate(
    order_or_state: dict[str, Any],
    state_or_metrics: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    cfg: dict | None = None,
) -> tuple[bool, str]:
    """
    発注前ゲート判定（純粋関数）。2 種のシグネチャを受け付ける。

    旧 (後方互換):
        pre_trade_gate(state, metrics) → (bool, str)

    新 (config-driven):
        pre_trade_gate(order, state, metrics, cfg) → (bool, str)

    Returns:
        (True,  "OK")        発注可
        (False, reason_str)  発注禁止
    """
    if metrics is None:
        # 旧シグネチャ: order_or_state=state, state_or_metrics=metrics
        return _gate_legacy(order_or_state, state_or_metrics or {})
    # 新シグネチャ: order_or_state=order, state_or_metrics=state
    return _gate_new(order_or_state, state_or_metrics or {}, metrics, cfg or {})
