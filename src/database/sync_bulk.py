"""
src/database/sync_bulk.py
Bulk API系（OHLCV/分足/Tick/Fundamentals summary/margin/shortselling）の日次インクリメンタル
更新エントリポイント。J-Quants解約前の最終同期（`sync_all()`実行→`verify_database.py`）にも使う。

当月（進行中の月）は日次liveファイルを束ねて status=partial のまま毎日再構築される。月が変わって
historicalへロールアップされ次第、次回実行時に自動的にstatus=ok（真の月次ファイル）へ切り替わる
（minute_bars.py/ticks.py/fundamentals.py/markets_bulk.py の ingest_month() がその判定を内包して
いるため、本モジュールは「当月と前月をforce無しで毎日再実行するだけ」でよい・force無し実行は
status=ok月に対しては即skipされ実質無コスト）。OHLCV/master系は既存 sync.py の日次差分更新
（未取得営業日のみ）をそのまま呼び出す（二重実装しない）。

Task Scheduler等から毎日実行する想定（run_morning_signal.py等とは独立・database/market/専用）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.database import fundamentals, markets_bulk, minute_bars, ticks
from src.database import sync as ohlcv_sync

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))


def _current_and_previous_month() -> list[tuple[int, int]]:
    today = datetime.now(_JST).date()
    prev_year = today.year if today.month > 1 else today.year - 1
    prev_month = today.month - 1 if today.month > 1 else 12
    return [(prev_year, prev_month), (today.year, today.month)]


def sync_all() -> dict:
    """
    OHLCV/master（既存sync.py）+ 当月・前月の分足/Tick/Fundamentals summary/margin/shortselling
    （Bulk API系）を再実行する（冪等・完了済み月は即skip）。J-Quants解約直前の最終同期に使う想定。
    """
    results: dict = {"ohlcv": None, "minute": [], "tick": [], "fundamentals": [], "markets": {}}

    results["ohlcv"] = ohlcv_sync.update_database()

    for year, month in _current_and_previous_month():
        results["minute"].append(minute_bars.ingest_month(year, month))
        results["tick"].append(ticks.ingest_month(year, month))
        results["fundamentals"].append(fundamentals.ingest_month(year, month))
        for domain_key in markets_bulk.MARKET_DOMAINS:
            results["markets"].setdefault(domain_key, []).append(
                markets_bulk.ingest_month(domain_key, year, month)
            )

    logger.info("[SYNC_BULK] 完了: %s", results)
    return results


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(sync_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
