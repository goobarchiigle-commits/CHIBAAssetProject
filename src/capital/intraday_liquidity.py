"""
intraday_liquidity.py — Intraday liquidity regime modeling

ADV-only sizing misses intraday concentration risk.
Opening/closing periods have elevated liquidity; mid-day is thinner.

Tokyo Stock Exchange intraday volume profile (approximate):
  09:00–09:30  opening   ~25% of daily volume in 30 min
  09:30–11:30  morning   ~35% across 120 min
  12:30–14:00  afternoon ~25% across 90 min
  14:00–15:30  closing   ~15% across 90 min
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional


# (start_time, end_time, fraction_of_daily_adv)
_INTRADAY_PROFILE: list[tuple[time, time, float]] = [
    (time(9, 0),  time(9, 30),  0.25),
    (time(9, 30), time(11, 30), 0.35),
    (time(12, 30), time(14, 0), 0.25),
    (time(14, 0),  time(15, 30), 0.15),
]

_DEFAULT_FRACTION = 0.02   # pre-market / lunch / after-hours


def get_interval_liquidity_fraction(now: Optional[datetime] = None) -> float:
    """
    Fraction of ADV expected to trade in the current 30-min interval.

    Scales the window's ADV fraction to a 30-min slice.
    """
    if now is None:
        now = datetime.now()
    t = now.time()

    for start, end, window_fraction in _INTRADAY_PROFILE:
        if start <= t <= end:
            start_min = start.hour * 60 + start.minute
            end_min   = end.hour * 60 + end.minute
            window_minutes = end_min - start_min
            if window_minutes <= 0:
                return _DEFAULT_FRACTION
            return window_fraction * (30.0 / window_minutes)

    return _DEFAULT_FRACTION


def compute_interval_volume_yen(
    adv_yen: float,
    now: Optional[datetime] = None,
) -> float:
    """Expected yen volume in the current 30-minute interval."""
    fraction = get_interval_liquidity_fraction(now)
    return adv_yen * fraction


def max_interval_order_value(
    adv_yen: float,
    participation_rate: float = 0.05,
    now: Optional[datetime] = None,
) -> float:
    """
    Maximum order value (yen) for current interval given participation limit.

    Args:
        adv_yen:           average daily volume in yen
        participation_rate: max fraction of interval volume we can take
    """
    interval_vol = compute_interval_volume_yen(adv_yen, now)
    return interval_vol * participation_rate


def compute_liquidity_stress_score(
    order_value: float,
    adv_yen: float,
    now: Optional[datetime] = None,
) -> float:
    """
    Liquidity stress score [0, 1].

    0.0 = tiny order relative to interval liquidity
    1.0 = order equals or exceeds interval liquidity (saturation)
    """
    if adv_yen <= 0.0:
        return 1.0

    # 100% participation as denominator (absolute interval volume)
    interval_vol = compute_interval_volume_yen(adv_yen, now)
    if interval_vol <= 0.0:
        return 1.0

    stress = min(1.0, order_value / interval_vol)
    return round(stress, 4)
