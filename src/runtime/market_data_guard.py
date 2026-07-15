"""
market_data_guard.py — Market data staleness and connectivity guard

Detects:
  - Stale quotes: symbol's last_update_time > stale_threshold
  - Stale candles: last candle time > stale_threshold
  - Clock skew: |local_ts - market_ts| > tolerance
  - Partial updates: expected symbols not received

Fail-safe: all checks are pure and non-blocking.
Halt recommendation is advisory only — caller decides whether to act.
Append-only recovery log. Never raises.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

QUOTE_STALE_SEC: float = 120.0
CANDLE_STALE_SEC: float = 300.0
CLOCK_SKEW_TOLERANCE_SEC: float = 30.0
HALT_STALE_FRACTION: float = 0.5


@dataclass
class MarketDataGuardReport:
    """Structured market data health snapshot. Append-only."""
    timestamp: str
    stale_quote_count: int
    stale_quote_symbols: list
    stale_candle_count: int
    stale_candle_symbols: list
    missing_symbol_count: int
    missing_symbols: list
    clock_skew_sec: Optional[float]
    clock_skew_detected: bool
    halt_recommended: bool
    halt_reason: str
    schema_version: int = SCHEMA_VERSION


def check_quote_freshness(
    quote_timestamps: dict[str, float],
    now_sec: float,
    stale_threshold_sec: float = QUOTE_STALE_SEC,
) -> list[str]:
    """Returns sorted list of symbols with stale or missing timestamps."""
    return sorted(
        sym for sym, ts in quote_timestamps.items()
        if ts <= 0 or (now_sec - ts) > stale_threshold_sec
    )


def check_candle_freshness(
    candle_timestamps: dict[str, float],
    now_sec: float,
    stale_threshold_sec: float = CANDLE_STALE_SEC,
) -> list[str]:
    """Returns sorted list of symbols with stale candle timestamps."""
    return sorted(
        sym for sym, ts in candle_timestamps.items()
        if ts <= 0 or (now_sec - ts) > stale_threshold_sec
    )


def detect_clock_skew(
    local_ts_sec: float,
    market_ts_sec: float,
    tolerance_sec: float = CLOCK_SKEW_TOLERANCE_SEC,
) -> Optional[float]:
    """Returns skew in seconds if exceeds tolerance, None if within tolerance."""
    if market_ts_sec <= 0:
        return None
    skew = abs(local_ts_sec - market_ts_sec)
    return round(skew, 2) if skew > tolerance_sec else None


def check_partial_update(
    expected_symbols: list[str],
    received_symbols: list[str],
) -> list[str]:
    """Returns symbols expected but not in received list."""
    received_set = set(received_symbols)
    return sorted(s for s in expected_symbols if s not in received_set)


def run_market_data_guard(
    expected_symbols: list[str],
    quote_timestamps: Optional[dict[str, float]] = None,
    candle_timestamps: Optional[dict[str, float]] = None,
    received_symbols: Optional[list[str]] = None,
    local_ts_sec: Optional[float] = None,
    market_ts_sec: Optional[float] = None,
    now_sec: Optional[float] = None,
    stale_quote_threshold_sec: float = QUOTE_STALE_SEC,
    stale_candle_threshold_sec: float = CANDLE_STALE_SEC,
    timestamp: str = "",
) -> MarketDataGuardReport:
    """
    Full market data guard check. Pure. Deterministic. No I/O.
    Halt is recommended (advisory) if stale fraction > HALT_STALE_FRACTION
    or clock skew exceeds tolerance.
    """
    if now_sec is None:
        now_sec = time.time()
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    stale_quotes = (
        check_quote_freshness(quote_timestamps, now_sec, stale_quote_threshold_sec)
        if quote_timestamps else []
    )
    stale_candles = (
        check_candle_freshness(candle_timestamps, now_sec, stale_candle_threshold_sec)
        if candle_timestamps else []
    )
    missing = (
        check_partial_update(expected_symbols, received_symbols)
        if received_symbols is not None else []
    )

    clock_skew: Optional[float] = None
    clock_skew_detected = False
    if local_ts_sec is not None and market_ts_sec is not None:
        clock_skew = detect_clock_skew(local_ts_sec, market_ts_sec)
        clock_skew_detected = clock_skew is not None

    halt_recommended = False
    halt_reason = ""
    n_expected = max(len(expected_symbols), 1)

    stale_fraction = len(stale_quotes) / n_expected
    if stale_fraction > HALT_STALE_FRACTION:
        halt_recommended = True
        halt_reason = (
            f"stale_quotes={len(stale_quotes)}/{n_expected}"
            f" > threshold={HALT_STALE_FRACTION}"
        )
    elif clock_skew_detected:
        halt_recommended = True
        halt_reason = (
            f"clock_skew={clock_skew:.1f}s"
            f" > tolerance={CLOCK_SKEW_TOLERANCE_SEC}s"
        )

    if halt_recommended:
        logger.warning("[MARKET_DATA_GUARD] halt recommended: %s", halt_reason)

    return MarketDataGuardReport(
        timestamp=timestamp,
        stale_quote_count=len(stale_quotes),
        stale_quote_symbols=stale_quotes,
        stale_candle_count=len(stale_candles),
        stale_candle_symbols=stale_candles,
        missing_symbol_count=len(missing),
        missing_symbols=missing,
        clock_skew_sec=clock_skew,
        clock_skew_detected=clock_skew_detected,
        halt_recommended=halt_recommended,
        halt_reason=halt_reason,
    )


def append_guard_report(report: MarketDataGuardReport, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[MARKET_DATA_GUARD] write failed (%s)", exc)


def load_guard_reports(path: Path) -> list[MarketDataGuardReport]:
    """Load all reports. Corrupted lines skipped."""
    if not path.exists():
        return []
    reports: list[MarketDataGuardReport] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            reports.append(MarketDataGuardReport(
                timestamp=d.get("timestamp", ""),
                stale_quote_count=int(d.get("stale_quote_count", 0)),
                stale_quote_symbols=d.get("stale_quote_symbols", []),
                stale_candle_count=int(d.get("stale_candle_count", 0)),
                stale_candle_symbols=d.get("stale_candle_symbols", []),
                missing_symbol_count=int(d.get("missing_symbol_count", 0)),
                missing_symbols=d.get("missing_symbols", []),
                clock_skew_sec=d.get("clock_skew_sec"),
                clock_skew_detected=bool(d.get("clock_skew_detected", False)),
                halt_recommended=bool(d.get("halt_recommended", False)),
                halt_reason=d.get("halt_reason", ""),
            ))
        except Exception as exc:
            logger.warning("[MARKET_DATA_GUARD] parse error (%s)", exc)
    return reports
