"""
src/universe/data_freshness_guard.py — Market data freshness enforcement

Checks RSR snapshot recency before universe operations.

LIVE mode: stale data → raises DataStalenessError → caller must abort
DRY  mode: stale data → logs WARNING, returns False, execution continues

Stale definition:
    Latest RSR snapshot date < (today - MAX_STALE_CALENDAR_DAYS)
    Default MAX_STALE_CALENDAR_DAYS = 3 (covers weekend + 1 holiday buffer)

Design:
    - Fail-closed for LIVE: DataStalenessError on stale data
    - Fail-open for DRY: warn and continue
    - No side-effects (read-only)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_STALE_CALENDAR_DAYS: int = 3   # Friday→Monday = 3 calendar days + holiday buffer


class DataStalenessError(RuntimeError):
    """Raised in LIVE mode when RSR data is too old."""


def check_rsr_freshness(
    rsr_snapshot_dir:   Path,
    is_live:            bool,
    max_stale_days:     int = MAX_STALE_CALENDAR_DAYS,
    reference_date:     Optional[date] = None,
) -> bool:
    """
    Verify RSR snapshot is fresh enough.

    Returns:
        True  — data is fresh
        False — data is stale (DRY mode only; LIVE raises DataStalenessError)

    Raises:
        DataStalenessError — when is_live=True and data is stale
    """
    ref = reference_date or date.today()
    result = _evaluate_freshness(rsr_snapshot_dir, ref, max_stale_days)

    if result.is_fresh:
        logger.debug(
            "[FRESHNESS] RSR data fresh: latest=%s age=%dd",
            result.latest_date, result.age_calendar_days,
        )
        return True

    msg = (
        f"RSR data stale: latest={result.latest_date} "
        f"age={result.age_calendar_days}d > max={max_stale_days}d"
        + ("  (no snapshots found)" if result.no_snapshots else "")
    )
    if is_live:
        logger.error("[FRESHNESS] LIVE mode abort: %s", msg)
        raise DataStalenessError(msg)

    logger.warning("[FRESHNESS] DRY mode warning: %s", msg)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────

class _FreshnessResult:
    __slots__ = ("is_fresh", "latest_date", "age_calendar_days", "no_snapshots")

    def __init__(
        self,
        is_fresh:           bool,
        latest_date:        Optional[date],
        age_calendar_days:  int,
        no_snapshots:       bool = False,
    ) -> None:
        self.is_fresh          = is_fresh
        self.latest_date       = latest_date
        self.age_calendar_days = age_calendar_days
        self.no_snapshots      = no_snapshots


def _evaluate_freshness(
    rsr_dir:       Path,
    ref_date:      date,
    max_stale_days: int,
) -> _FreshnessResult:
    if not rsr_dir.exists():
        return _FreshnessResult(False, None, 9999, no_snapshots=True)

    snaps = sorted(rsr_dir.glob("*.json"))
    if not snaps:
        return _FreshnessResult(False, None, 9999, no_snapshots=True)

    try:
        latest_date = date.fromisoformat(snaps[-1].stem)
    except ValueError:
        # Filename not a date (e.g. metadata file) — search backwards
        for snap in reversed(snaps):
            try:
                latest_date = date.fromisoformat(snap.stem)
                break
            except ValueError:
                continue
        else:
            return _FreshnessResult(False, None, 9999, no_snapshots=True)

    age = (ref_date - latest_date).days
    age = max(0, age)
    is_fresh = age <= max_stale_days

    return _FreshnessResult(
        is_fresh=is_fresh,
        latest_date=latest_date,
        age_calendar_days=age,
    )
