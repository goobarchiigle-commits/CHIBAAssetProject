"""
src/database/earnings_calendar.py
J-Quants /v2/equities/earnings-calendar（決算発表予定日）の取得・保存。

J-Quants解約前最終差分取得（2026-08-08）で新規実装。Bulk API非対応・REST専用（Free+プラン）。
公式仕様上「翌営業日に決算発表予定の銘柄情報」のみを返す前方参照専用エンドポイントであり、
過去日付の決算発表予定日は提供されない（sync_bulk.pyの他ドメインのような差分バックフィルは
そもそも不可能・取得できるのは実行時点の1スナップショットのみ）。REIT除外。

保存先: database/market/fundamentals/earnings_calendar/{fetched_at:%Y-%m-%d}.parquet
（日付ごとに1ファイル・同日再実行はforceで上書き・翌日分は別ファイルとして蓄積するため
 既存ファイルの上書き・消失リスクなし=ISSUE-001とは無関係な設計）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.jquants.client import JQuantsClient
from src.paths import DATABASE_FUNDAMENTALS_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

EARNINGS_CALENDAR_PATH = "/v2/equities/earnings-calendar"
_OUT_DIR = DATABASE_FUNDAMENTALS_DIR / "earnings_calendar"


def fetch_earnings_calendar(client: "JQuantsClient | None" = None) -> pd.DataFrame:
    """翌営業日決算発表予定銘柄一覧を取得する（列名リネームなし・API応答そのまま）。"""
    client = client or JQuantsClient()
    data = client.get(EARNINGS_CALENDAR_PATH)
    records = data.get("data") or data.get("earnings_calendar") or []
    if not records:
        logger.warning("[EARNINGS_CALENDAR] 応答が空")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str)
    return df


def fetch_and_save(*, force: bool = False) -> dict:
    """
    取得し database/market/fundamentals/earnings_calendar/{today}.parquet へ保存する。
    既存ファイルへの上書きはforce指定時のみ（デフォルトは既存があればスキップ・冪等）。
    """
    ensure_database_market_dirs()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(_JST).strftime("%Y-%m-%d")
    out_path = _OUT_DIR / f"{today}.parquet"

    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        logger.info("[EARNINGS_CALENDAR] %s 既に取得済み・スキップ（rows=%d）", today, len(existing))
        return {"status": "skipped", "date": today, "rows": len(existing), "path": str(out_path)}

    df = fetch_earnings_calendar()
    if df.empty:
        return {"status": "empty", "date": today, "rows": 0, "path": None}

    df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("[EARNINGS_CALENDAR] %s 保存完了 rows=%d cols=%s path=%s", today, len(df), list(df.columns), out_path)
    return {"status": "ok", "date": today, "rows": len(df), "path": str(out_path)}


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = fetch_and_save(force="--force" in sys.argv)
    print(result)
    return 0 if result["status"] in ("ok", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
