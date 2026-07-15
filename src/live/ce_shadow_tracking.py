"""
src/live/ce_shadow_tracking.py
Capital Efficiency (kNN期待アルファ) Shadow Mode トラッキング（2026-07-15 SSOT統合）。

背景: run_morning_signal.py（2026-07-15廃止・reports/execution_path_ssot_audit_2026-07-14.md
参照）が実発注のBUY数量を実際に調整していたCE機構を、run_live_signal.pyへ
Shadow Modeとして移植したもの。

Tier0固定: 実発注数量（order.qty）には一切影響しない。run_live_signal.py側の
類似機構（Position Sizing Intelligence）が自らTier0/SAFETY_DEMOTEと判定している
現状と整合させる決定（ユーザー承認 2026-07-15）。将来、ce_compare_daily.csvで
OOS有効性が実証された場合のみ、Position Sizing Promotionと同様のTier昇格評価を
別途追加検討する。

DRY/LIVE 両方の実行経路から呼ぶこと（run_morning_signal.pyの原設計がdry/live
分岐前にCE比較ログを記録していたことを踏襲・比較データの欠落を防ぐ）。
FAIL_OPEN — 例外は握りつぶし、呼び出し元の発注処理には一切影響させない。

独立モジュール化の理由: run_live_signal.py は assert_execution_context() に
より未承認スクリプト(pytest等)からのimportをLIVE_MODE=true環境下でブロックする
安全ガードを持つ。本ロジックはそのガードと無関係な純粋関数であるため、
テスト容易性のため独立モジュールとして切り出した。
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


def run_ce_shadow_tracking(
    signals:       list[dict],
    order_objects: list[Any],
    *,
    runtime_dir: "Path | None" = None,
    logs_dir:    "Path | None" = None,
) -> dict[str, dict]:
    """
    Capital Efficiency (kNN期待アルファ) によるBUY注文数量調整を Shadow Mode で
    記録する。order_objects の qty は一切変更しない。

    Args:
        signals:       result.signals（{"symbol", "rsr", ...} のリスト）
        order_objects: OrderInstruction相当（symbol/side/qty/estimated_price属性）
        runtime_dir:   省略時は src.paths.RUNTIME_DIR
        logs_dir:      省略時は src.paths.LOGS_DIR

    Returns:
        {symbol: {ea, confidence, sample_size, regime, actual_qty, shadow_qty}}
        （テスト・ログ表示用。空dictは失敗またはBUY注文0件を意味する）
    """
    ce_meta: dict[str, dict] = {}
    try:
        if runtime_dir is None or logs_dir is None:
            from src.paths import RUNTIME_DIR as _default_runtime, LOGS_DIR as _default_logs
            runtime_dir = runtime_dir or _default_runtime
            logs_dir    = logs_dir or _default_logs

        from src.portfolio.capital_efficiency import CapitalEfficiencyModule
        from src.portfolio.ce_compare_logger import CECompareLogger

        ce_state_file = Path(runtime_dir) / "ce_state.pkl"
        try:
            with open(ce_state_file, "rb") as f:
                ce = pickle.load(f)
            if not isinstance(ce, CapitalEfficiencyModule):
                ce = CapitalEfficiencyModule()
        except Exception:
            ce = CapitalEfficiencyModule()

        ce_logger  = CECompareLogger(runtime_dir=Path(runtime_dir), logs_dir=Path(logs_dir))
        today      = datetime.now(JST).date()
        date_idx   = datetime.now(JST).toordinal()
        ce.on_day_open(date_idx, market_return=0.0)

        filled = ce_logger.try_fill_forward_returns(today)
        if filled:
            ce_logger.flush_daily_csv(today)

        sig_by_sym = {s["symbol"]: s for s in signals}
        for o in order_objects:
            if getattr(o, "side", "") != "BUY":
                continue
            sig   = sig_by_sym.get(o.symbol, {})
            score = float(sig.get("rsr", 75.0))
            ea, confidence, regime = ce.estimate_expected_alpha(date_idx, score, 0.0, 0.9)
            scale       = float(min(1.0, max(0.3, 1.0 + ea * 5.0)))
            base_qty    = int(o.qty)
            shadow_qty  = max(100, (int(base_qty * scale) // 100) * 100)
            logger.info(
                "[CE_SHADOW] %s ea=%.4f conf=%.2f n=%d regime=%s "
                "actual_qty=%d shadow_qty=%d (tier=0 — 発注数量への影響なし)",
                o.symbol, ea, confidence, len(ce._buf), regime, base_qty, shadow_qty,
            )
            ce_logger.record_order(
                date_str    = today.isoformat(),
                symbol      = o.symbol,
                side        = "BUY",
                qty_base    = base_qty,
                qty_ce      = shadow_qty,
                fill_price  = float(getattr(o, "estimated_price", 0.0)),
                ea          = ea,
                confidence  = confidence,
                sample_size = len(ce._buf),
            )
            ce_meta[o.symbol] = {
                "ea": ea, "confidence": confidence, "sample_size": len(ce._buf),
                "regime": regime, "actual_qty": base_qty, "shadow_qty": shadow_qty,
            }
            # Tier0固定: o.qty / order_objects は変更しない（実発注数量は不変）

        with open(ce_state_file, "wb") as f:
            pickle.dump(ce, f)
    except Exception as exc:
        logger.warning(
            "[CE_SHADOW] shadow tracking failed (FAIL_OPEN, 発注処理には無関係): %s", exc,
        )
    return ce_meta
