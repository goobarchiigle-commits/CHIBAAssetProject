"""
portfolio/holding_period.py
保有期間管理と時間ストップ

JPXCalendar を使って営業日ベースの保有日数を計算し、
MAX_HOLD_DAYS を超えたポジションに時間ストップを適用する。
"""

import pandas as pd
from src.market.jpx_calendar import JPXCalendar

_calendar = JPXCalendar()

MAX_HOLD_DAYS = 60  # 最大保有営業日数（約3ヶ月）


def calc_hold_days(entry_date, today) -> int:
    """
    entry_date から today までの営業日数を返す（両端含む）。

    Args:
        entry_date: 買付日 (str / datetime / Timestamp)
        today:      基準日 (str / datetime / Timestamp)

    Returns:
        int: 営業日数（0以上）
    """
    entry = pd.Timestamp(entry_date)
    end   = pd.Timestamp(today)
    if end < entry:
        return 0
    return _calendar.trading_days_between(entry, end)


def check_time_stop(entry_date, today, max_hold_days: int = MAX_HOLD_DAYS) -> bool:
    """
    時間ストップ判定。保有営業日数が max_hold_days 以上なら True。

    Args:
        entry_date:    買付日
        today:         基準日
        max_hold_days: 最大保有営業日数（デフォルト 60）

    Returns:
        bool: True = 時間ストップ発動（売却すべき）
    """
    hold_days = calc_hold_days(entry_date, today)
    return hold_days >= max_hold_days
