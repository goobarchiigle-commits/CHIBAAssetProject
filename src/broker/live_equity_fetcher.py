"""
live_equity_fetcher.py — 実口座資産をブローカーAPIから取得する軽量フェッチャー

LIVE / DRY 実行時のみ使用。バックテスト系スクリプトからは呼ばないこと。
FAIL_OPEN: 取得失敗時は None を返す（売買停止しない）。

取得フロー:
  GET /wallet/cash   → StockAccountWallet (現金買付余力)
  GET /positions     → LeavesQty × CurrentPrice (保有株時価)
  actual_equity は src.portfolio.equity.compute_live_equity() に委譲する（SSOT, Study26）。

ログ出力:
  [LIVE_EQUITY] cash=  market_value=  actual_equity=  source=
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.portfolio.equity import compute_live_equity

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


@dataclass
class LiveEquitySnapshot:
    cash: float           # GET /wallet/cash → StockAccountWallet
    market_value: float   # sum(LeavesQty × CurrentPrice) from GET /positions
    actual_equity: float  # cash + market_value
    position_count: int   # 評価対象ポジション数
    source: str           # "broker" | "broker_partial" | "positions_only"
    fetched_at: str       # ISO-8601 JST タイムスタンプ


def fetch_live_equity(client, *, timeout_s: float = 8.0) -> Optional[LiveEquitySnapshot]:
    """
    FAIL_OPEN: 取得失敗時は None を返す。例外を外に投げない。

    両方のAPIが失敗した場合のみ None を返す。
    片方のみ失敗した場合は source="broker_partial" で部分値を返す。

    Args:
        client: fetch_token() 済みの KabuClient インスタンス
        timeout_s: 将来拡張用（現在はクライアント側タイムアウトに依存）

    Returns:
        LiveEquitySnapshot | None
    """
    ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    cash: Optional[float] = None
    market_value: float = 0.0
    position_count: int = 0
    wallet_ok: bool = False
    positions_ok: bool = False

    # ── Step 1: 現金買付余力 ─────────────────────────────────────────────────
    try:
        wallet = client.get_wallet_cash()
        raw_cash = wallet.get("StockAccountWallet")
        if raw_cash is None:
            raise ValueError(f"StockAccountWallet field missing in response: {list(wallet.keys())}")
        cash = float(raw_cash)
        wallet_ok = True
        logger.debug("[LIVE_EQUITY] wallet fetched: StockAccountWallet=¥%s", f"{cash:,.0f}")
    except Exception as exc:
        logger.warning("[LIVE_EQUITY] GET /wallet/cash failed (FAIL_OPEN): %s", exc)

    # ── Step 2: 保有株時価 ────────────────────────────────────────────────────
    _positions_for_equity: dict[str, dict] = {}
    try:
        raw_positions = client.get_positions()
        for pos in (raw_positions or []):
            leaves_qty = float(pos.get("LeavesQty") or 0)
            if leaves_qty <= 0:
                continue
            # CurrentPrice → Price (約定単価) の優先順で取得
            current_price = float(pos.get("CurrentPrice") or 0)
            avg_price     = float(pos.get("Price") or 0)
            price = current_price if current_price > 0 else avg_price
            if price > 0:
                market_value += leaves_qty * price
                position_count += 1
                _sym = pos.get("Symbol") or f"_pos{position_count}"
                _positions_for_equity[_sym] = {"qty": leaves_qty, "avg_price": price}
        positions_ok = True
        logger.debug(
            "[LIVE_EQUITY] positions fetched: count=%d market_value=¥%s",
            position_count, f"{market_value:,.0f}",
        )
    except Exception as exc:
        logger.warning("[LIVE_EQUITY] GET /positions failed (FAIL_OPEN): %s", exc)

    # ── Step 3: 両方失敗 → None ──────────────────────────────────────────────
    if not wallet_ok and not positions_ok:
        logger.warning(
            "[LIVE_EQUITY] cash=N/A market_value=N/A actual_equity=N/A source=unavailable"
            " — both API calls failed, returning None"
        )
        return None

    # ── Step 4: 部分成功の source 決定 ───────────────────────────────────────
    if wallet_ok and positions_ok:
        source = "broker"
    elif positions_ok:
        cash = 0.0
        source = "positions_only"
    else:
        source = "broker_partial"

    # SSOT: cash + market_value はここで直接計算しない。
    # src/portfolio/equity.py compute_live_equity() のみが equity 計算を行う（Study26）。
    actual_equity = compute_live_equity(
        live_cash        = cash or 0.0,
        positions        = _positions_for_equity,
        mode             = "live_equity_fetch",
        persist_snapshot = False,
    )

    snap = LiveEquitySnapshot(
        cash=cash or 0.0,
        market_value=market_value,
        actual_equity=actual_equity,
        position_count=position_count,
        source=source,
        fetched_at=ts,
    )

    logger.info(
        "[LIVE_EQUITY] cash=¥%s market_value=¥%s actual_equity=¥%s source=%s",
        f"{snap.cash:,.0f}",
        f"{snap.market_value:,.0f}",
        f"{snap.actual_equity:,.0f}",
        snap.source,
    )
    return snap
