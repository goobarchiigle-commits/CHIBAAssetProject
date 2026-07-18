"""
src/portfolio/broker_source.py
Broker snapshot 取得の唯一の経路（Broker-as-Sole-SSOT リファクタ, 2026-07-18）。

RULE:
  cash / positions / avg_costs / market_values は必ず fetch_broker_snapshot() 経由で
  取得すること。KabuClient.get_wallet_cash() / get_positions() を他の場所から
  直接呼んで資産計算に使うことを禁止する。

  失敗時は FAIL_OPEN しない。BrokerSnapshotUnavailable を必ず raise する。
  state / OHLCV キャッシュ / 過去snapshot へのフォールバックはここでは行わない
  （呼び出し側が abort するか、明示的に許可されたテスト/研究用途でのみ別経路を使う）。

  market_values は broker `/positions` 応答の CurrentPrice（broker自身が評価した
  現在値）をそのまま使う。ローカル OHLCV キャッシュから再計算しない。
  CurrentPrice が欠損/0 の銘柄のみ、同一応答内の Price（平均取得単価）へ
  フォールバックする（これも broker 由来であり、ローカル計算ではない）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from src.common.position_normalizer import filter_live_positions, _extract_qty
from src.portfolio.state_store import BrokerSnapshot

if TYPE_CHECKING:
    from src.kabusapi.client import KabuClient

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


class BrokerSnapshotUnavailable(RuntimeError):
    """
    broker snapshot 取得の失敗を示す専用例外。

    汎用の Exception/RuntimeError ではなくこの型にすることで、ログ上
    [BROKER_SNAPSHOT_UNAVAILABLE] として即座に「何が原因で止まったか」を
    特定できるようにする。呼び出し側（SignalBridge等）はこれを AbortError で
    ラップして実行を止める。フォールバックせず必ず raise する。
    """


def fetch_broker_snapshot(client: "KabuClient") -> BrokerSnapshot:
    """
    cash・positions・avg_costs・market_values の唯一の取得経路。

    kabu Station API `/wallet/cash` と `/positions` の両方を呼び出し、
    どちらか一方でも失敗した場合は BrokerSnapshotUnavailable を raise する
    （partial success を許さない — commit_broker_snapshot() の
    "FORBIDDEN: partial commit" ルールと同じ方針）。

    equity はここでは計算しない。呼び出し側が
    src.portfolio.equity.compute_live_equity(snapshot=...) で計算すること。

    Returns:
        BrokerSnapshot (equity=0.0 のまま。呼び出し側が埋める)
    """
    if client is None:
        raise BrokerSnapshotUnavailable("KabuClient is None — broker への接続がありません")

    try:
        wallet = client.get_wallet_cash()
        if "StockAccountWallet" not in wallet:
            raise RuntimeError(f"StockAccountWallet field missing in response: {list(wallet.keys())}")
        cash = float(wallet["StockAccountWallet"])
    except BrokerSnapshotUnavailable:
        raise
    except Exception as exc:
        raise BrokerSnapshotUnavailable(f"GET /wallet/cash 失敗: {exc}") from exc

    try:
        raw_positions = client.get_positions()
    except Exception as exc:
        raise BrokerSnapshotUnavailable(f"GET /positions 失敗: {exc}") from exc

    live_positions = filter_live_positions(raw_positions or [])

    positions:     dict[str, int]   = {}
    avg_costs:     dict[str, float] = {}
    market_values: dict[str, float] = {}
    for p in live_positions:
        sym_code = p.get("Symbol", "")
        sym = f"{sym_code}.T"
        qty = int(_extract_qty(p))
        if qty <= 0:
            continue
        avg_price     = float(p.get("Price", 0.0) or 0.0)
        current_price = float(p.get("CurrentPrice", 0.0) or 0.0)
        if current_price <= 0:
            logger.warning(
                "[MARKET_VALUE_FALLBACK_TO_AVG_COST] %s: CurrentPrice欠損/0 → avg_price(¥%.2f)を使用",
                sym, avg_price,
            )
            current_price = avg_price
        positions[sym]     = qty
        avg_costs[sym]     = avg_price
        market_values[sym] = current_price

    ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    snapshot = BrokerSnapshot(
        cash=cash,
        positions=positions,
        avg_costs=avg_costs,
        market_values=market_values,
        equity=0.0,
        ts=ts,
        source="broker",
        api_health={"positions_ok": True, "wallet_ok": True},
    )

    logger.info(
        "[BROKER_SNAPSHOT_VERIFIED] cash=%.0f positions=%d market_value=%.0f ts=%s",
        cash, len(positions), sum(positions[s] * market_values[s] for s in positions), ts,
    )
    _append_snapshot_audit_log(snapshot)
    return snapshot


def _append_snapshot_audit_log(snapshot: BrokerSnapshot) -> None:
    """
    logs/broker_snapshot_audit.jsonl に append-only で監査ログを記録する。

    目的: 「brokerが実際に何を返したか」を後から一意に確定できるようにする。
    equity_snapshots.jsonl と同系統だが、こちらは equity 計算後ではなく
    broker応答の生データ（cash/positions/market_value）そのものを記録する。
    書き込み失敗は記録しない（FAIL_OPEN — 監査ログの失敗で本処理を止めない）。
    """
    import json
    import os
    from pathlib import Path

    base_dir = Path(os.environ.get("AI_TRADING_HOME", str(Path(__file__).resolve().parents[2])))
    audit_file = base_dir / "logs" / "broker_snapshot_audit.jsonl"
    try:
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        market_value = sum(
            snapshot.positions[s] * snapshot.market_values.get(s, 0.0) for s in snapshot.positions
        )
        record = {
            "ts":            snapshot.ts,
            "cash":          round(snapshot.cash, 0),
            "positions":     len(snapshot.positions),
            "market_value":  round(market_value, 0),
            "symbols":       sorted(snapshot.positions.keys()),
        }
        with audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("[BROKER_SNAPSHOT_VERIFIED] audit log write failed (FAIL_OPEN): %s", exc)
