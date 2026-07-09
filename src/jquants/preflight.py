"""
src/jquants/preflight.py
Full Download実行前の事前検証（API通信なし・純粋な見積もりとローカルディスクチェックのみ）。

subscription coverage は estimate_subscription_floor()（日付計算のみ）を既定で使う。
真の契約データ提供開始日を確認したい場合は別途 JQuantsProvider.detect_subscription_floor()
（1リクエスト）を呼び、その結果を start として渡すこと（本モジュール自体はAPIを呼ばない）。
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.jquants.provider import estimate_subscription_floor
from src.market.jpx_calendar import JPXCalendar
from src.paths import JQUANTS_DATA_DIR

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

MINIMUM_FREE_GB = 10

# Download Strategy Validation（2026-07-09実測）に基づく経験則。実測値であり保証値ではない
# （実行時の実測値で更新されるべき・--preflightの出力は目安として提示する）。
AVG_SECONDS_PER_REQUEST = 2.0        # 観測レンジ0.4〜3.6秒の中間的見積もり
AVG_ROW_BYTES_COMPRESSED = 50        # Parquet圧縮後の1行あたり概算バイト数（OHLCV系列）
AVG_SYMBOLS_PER_TRADING_DAY = 4400   # 2026-07-08実測（全銘柄日次クエリ・約4,437件）


def check_disk_space(path: Path | None = None) -> dict:
    """指定パス（省略時は data/jquants/）が乗っているドライブの空き容量を確認する。"""
    target = path or JQUANTS_DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024 ** 3)
    return {
        "free_gb": round(free_gb, 2),
        "minimum_required_gb": MINIMUM_FREE_GB,
        "sufficient": free_gb >= MINIMUM_FREE_GB,
    }


def _human_duration(seconds: float) -> str:
    hours = seconds / 3600
    if hours >= 1:
        return f"約{hours:.1f}時間"
    return f"約{seconds / 60:.0f}分"


def estimate_full_download(
    start: str | None = None,
    end: str | None = None,
    base_sleep: float = 0.05,
) -> dict:
    """
    Full Download（Strategy C・日次イテレーション）の見積もりをAPI通信なしで算出する。
    start省略時は estimate_subscription_floor()（推定値）を使う。
    """
    today = datetime.now(_JST).strftime("%Y-%m-%d")
    coverage_start = start or estimate_subscription_floor()
    coverage_end = end or today

    calendar = JPXCalendar()
    trading_days = [d for d in pd.bdate_range(coverage_start, coverage_end) if calendar.is_trading_day(d)]
    trading_day_count = len(trading_days)

    estimated_requests = trading_day_count + 1  # +1 = TOPIX（全期間1リクエストで足りると実測済み）
    estimated_seconds = trading_day_count * (AVG_SECONDS_PER_REQUEST + base_sleep) + AVG_SECONDS_PER_REQUEST
    estimated_rows = trading_day_count * AVG_SYMBOLS_PER_TRADING_DAY
    estimated_disk_bytes = estimated_rows * AVG_ROW_BYTES_COMPRESSED * 2  # raw + processed の概算合計

    years = sorted({int(d.year) for d in trading_days})

    return {
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "coverage_is_estimated": start is None,
        "trading_day_count": trading_day_count,
        "estimated_requests": estimated_requests,
        "estimated_runtime_sec": round(estimated_seconds, 1),
        "estimated_runtime_human": _human_duration(estimated_seconds),
        "estimated_disk_gb": round(estimated_disk_bytes / (1024 ** 3), 2),
        "expected_download_years": years,
    }


def run_preflight(
    start: str | None = None,
    end: str | None = None,
    base_sleep: float = 0.05,
) -> dict:
    """--preflight の本体。見積もり + ディスク空き容量チェックをまとめて返す（API通信なし）。"""
    estimate = estimate_full_download(start, end, base_sleep)
    disk = check_disk_space()
    return {**estimate, "disk": disk}
