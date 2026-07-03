"""
src/analytics/forward_attribution_validator.py — Forward attribution integrity validator

Fail-closed integrity layer for forward return attribution.
Detects:
  - future bar misalignment (bar date > as_of_date)
  - future leakage (forward return populated before bar existed)
  - timezone inconsistency
  - eval_date on non-trading day
  - holiday gaps within forward window
  - missing future bars (enriched but incomplete)
  - split-adjustment drift
  - duplicate (symbol, eval_date, executed) timestamps
  - incomplete attribution windows

Design:
  - fail-closed: CRITICAL anomalies raise AttributionIntegrityError in strict mode
  - deterministic: same inputs → same replay_hash
  - append-only JSONL anomaly log
  - no external network; calendar is self-contained
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION: int = 1

# ── Severity ──────────────────────────────────────────────────────────────────
SEV_CRITICAL = "CRITICAL"
SEV_WARNING  = "WARNING"
SEV_INFO     = "INFO"

# ── Anomaly types ─────────────────────────────────────────────────────────────
ANOMALY_FUTURE_BAR_MISALIGNMENT = "future_bar_misalignment"
ANOMALY_FUTURE_LEAKAGE          = "future_leakage"
ANOMALY_TIMEZONE_INCONSISTENCY  = "timezone_inconsistency"
ANOMALY_NON_TRADING_DAY         = "non_trading_day"
ANOMALY_HOLIDAY_GAP             = "holiday_gap"
ANOMALY_MISSING_FUTURE_BARS     = "missing_future_bars"
ANOMALY_SPLIT_ADJUSTMENT_DRIFT  = "split_adjustment_drift"
ANOMALY_DUPLICATE_TIMESTAMP     = "duplicate_timestamp"
ANOMALY_INCOMPLETE_WINDOW       = "incomplete_attribution_window"

FORWARD_HORIZONS: List[int] = [1, 3, 5, 10, 20]

# Min calendar days after eval_date before expecting a horizon's value
_HORIZON_MIN_CALENDAR_DAYS: Dict[int, int] = {1: 3, 3: 6, 5: 9, 10: 16, 20: 30}

# Price ratio thresholds that suggest a split event
_SPLIT_HIGH = 1.80   # price jumped 80 %+  → possible reverse split
_SPLIT_LOW  = 0.55   # price dropped 45 %+ → possible forward split

# JST timestamp pattern
_JST_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+09:00|Z)$")


class AttributionIntegrityError(Exception):
    """Raised on CRITICAL attribution anomalies in strict mode."""
    def __init__(self, message: str, anomalies: List["AnomalyRecord"]) -> None:
        super().__init__(message)
        self.anomalies = anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyRecord:
    anomaly_id: str
    anomaly_type: str
    severity: str
    symbol: str
    eval_date: str
    detail: str
    context: Dict[str, Any]
    detected_at: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    schema_version: int
    run_id: str
    generated_at_jst: str
    validation_mode: str        # "strict" | "audit"
    as_of_date: str
    record_count: int
    valid_count: int
    anomaly_count: int
    critical_count: int
    warning_count: int
    info_count: int
    pass_rate: float
    anomalies: List[AnomalyRecord]
    summary_by_type: Dict[str, int]
    integrity_verdict: str      # "PASS" | "FAIL" | "WARN"
    replay_hash: str

    @property
    def passed(self) -> bool:
        return self.integrity_verdict == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Trading calendar
# ─────────────────────────────────────────────────────────────────────────────

class TradingCalendarProvider(ABC):
    @abstractmethod
    def is_trading_day(self, d: date) -> bool: ...

    def nth_trading_day_after(self, start: date, n: int) -> date:
        """Date of the Nth trading day strictly after `start`."""
        d = start
        count = 0
        while count < n:
            d += timedelta(days=1)
            if self.is_trading_day(d):
                count += 1
        return d

    def holidays_in_range(self, start: date, end: date) -> List[date]:
        """Non-weekend, non-trading days in [start, end]."""
        result: List[date] = []
        d = start
        while d <= end:
            if d.weekday() < 5 and not self.is_trading_day(d):
                result.append(d)
            d += timedelta(days=1)
        return result


class SimpleTSECalendar(TradingCalendarProvider):
    """TSE trading calendar approximation: weekends + national holidays."""

    _DEFAULT_HOLIDAYS: Set[str] = {
        # 2024
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08",
        "2024-02-11", "2024-02-12", "2024-02-23",
        "2024-03-20", "2024-04-29",
        "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06",
        "2024-07-15", "2024-08-11", "2024-08-12",
        "2024-09-16", "2024-09-22", "2024-09-23",
        "2024-10-14", "2024-11-03", "2024-11-04", "2024-11-23",
        "2024-12-31",
        # 2025
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-13",
        "2025-02-11", "2025-02-23", "2025-02-24",
        "2025-03-20", "2025-04-29",
        "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06",
        "2025-07-21", "2025-08-11",
        "2025-09-15", "2025-09-23", "2025-10-13",
        "2025-11-03", "2025-11-23", "2025-11-24",
        "2025-12-31",
        # 2026
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-12",
        "2026-02-11", "2026-02-23",
        "2026-03-20", "2026-04-29",
        "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06",
        "2026-07-20", "2026-08-11",
        "2026-09-21", "2026-09-23", "2026-10-12",
        "2026-11-03", "2026-11-23",
        "2026-12-31",
    }

    def __init__(self, extra_holidays: Optional[Set[str]] = None) -> None:
        self._holidays: Set[str] = set(self._DEFAULT_HOLIDAYS)
        if extra_holidays:
            self._holidays |= extra_holidays

    def is_holiday(self, d: date) -> bool:
        return d.strftime("%Y-%m-%d") in self._holidays

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and not self.is_holiday(d)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _make_anomaly(
    anomaly_type: str,
    severity: str,
    symbol: str,
    eval_date: str,
    detail: str,
    context: Dict[str, Any],
) -> AnomalyRecord:
    now_jst = datetime.now(JST).isoformat()
    aid = _sha16(f"{anomaly_type}|{symbol}|{eval_date}|{detail[:80]}")
    return AnomalyRecord(
        anomaly_id=aid,
        anomaly_type=anomaly_type,
        severity=severity,
        symbol=symbol,
        eval_date=eval_date,
        detail=detail,
        context=context,
        detected_at=now_jst,
    )


def _parse_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def _get_attr(snap: Any, attr: str, default: Any = None) -> Any:
    if isinstance(snap, dict):
        return snap.get(attr, default)
    return getattr(snap, attr, default)


def _get_fwd(snap: Any, n: int) -> Optional[float]:
    return _get_attr(snap, f"forward_{n}d")


def _replay_hash(snapshots: List[Any], anomalies: List[AnomalyRecord]) -> str:
    snap_ids = sorted(_get_attr(s, "snapshot_id", "") or "" for s in snapshots)
    by_type: Dict[str, int] = {}
    for a in anomalies:
        by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
    payload = json.dumps(
        {"snap_ids": snap_ids, "anomaly_summary": sorted(by_type.items())},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# ForwardAttributionValidator
# ─────────────────────────────────────────────────────────────────────────────

class ForwardAttributionValidator:
    """
    Validates forward attribution integrity for a batch of entry snapshots.

    Parameters
    ----------
    as_of_date : date
        Reference "today". Forward returns beyond this date are leakage.
    calendar : TradingCalendarProvider, optional
        Defaults to SimpleTSECalendar.
    strict : bool
        If True, raise AttributionIntegrityError on any CRITICAL anomaly.
    run_id : str, optional
        Identifier embedded in the ValidationReport.
    """

    def __init__(
        self,
        as_of_date: date,
        calendar: Optional[TradingCalendarProvider] = None,
        strict: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        self.as_of_date = as_of_date
        self.calendar: TradingCalendarProvider = calendar or SimpleTSECalendar()
        self.strict = strict
        self.run_id = run_id or _sha16(str(as_of_date))[:8]

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        snapshots: List[Any],
        log_path: Optional[Path] = None,
        replay_mode: bool = False,
        replay_report_path: Optional[Path] = None,
    ) -> ValidationReport:
        """
        Validate all snapshots. Returns ValidationReport.
        In strict mode raises AttributionIntegrityError on CRITICAL anomalies.
        In replay_mode verifies the replay_hash matches a stored report.
        """
        all_anomalies: List[AnomalyRecord] = []

        # Global checks (across all snapshots)
        all_anomalies.extend(_check_duplicates(snapshots))
        all_anomalies.extend(_check_split_drift(snapshots))

        # Per-snapshot checks
        for snap in snapshots:
            peer_snaps = [
                s for s in snapshots
                if _get_attr(s, "symbol") == _get_attr(snap, "symbol")
                and s is not snap
            ]
            all_anomalies.extend(self._check_future_leakage(snap))
            all_anomalies.extend(self._check_bar_alignment(snap))
            all_anomalies.extend(self._check_timezone(snap))
            all_anomalies.extend(self._check_eval_date_trading_day(snap))
            all_anomalies.extend(self._check_missing_bars(snap))
            all_anomalies.extend(self._check_incomplete_window(snap))
            all_anomalies.extend(self._check_holiday_gap(snap))

        # Build report
        report = self._build_report(snapshots, all_anomalies)

        # Append to anomaly log
        if log_path and all_anomalies:
            append_anomaly_log(all_anomalies, log_path)

        # Replay verification
        if replay_mode and replay_report_path and replay_report_path.exists():
            self._verify_replay(report, replay_report_path)

        # Fail-closed
        critical = [a for a in all_anomalies if a.severity == SEV_CRITICAL]
        if self.strict and critical:
            raise AttributionIntegrityError(
                f"{len(critical)} CRITICAL attribution anomaly(ies) detected "
                f"(as_of={self.as_of_date}): "
                + "; ".join(a.detail[:60] for a in critical[:3]),
                anomalies=critical,
            )

        return report

    def validate_single(
        self, snap: Any, peer_snaps: Optional[List[Any]] = None
    ) -> List[AnomalyRecord]:
        """Validate one snapshot; returns list of anomalies (does not raise)."""
        anomalies: List[AnomalyRecord] = []
        anomalies.extend(self._check_future_leakage(snap))
        anomalies.extend(self._check_bar_alignment(snap))
        anomalies.extend(self._check_timezone(snap))
        anomalies.extend(self._check_eval_date_trading_day(snap))
        anomalies.extend(self._check_missing_bars(snap))
        anomalies.extend(self._check_incomplete_window(snap))
        anomalies.extend(self._check_holiday_gap(snap))
        if peer_snaps:
            anomalies.extend(_check_split_drift([snap] + peer_snaps, origin_symbol=_get_attr(snap, "symbol")))
        return anomalies

    # ── Per-snapshot checks ───────────────────────────────────────────────────

    def _check_future_leakage(self, snap: Any) -> List[AnomalyRecord]:
        """CRITICAL: any forward_Nd populated when eval_date >= as_of_date."""
        anomalies: List[AnomalyRecord] = []
        eval_d = _parse_date(_get_attr(snap, "eval_date", ""))
        if eval_d is None:
            return anomalies
        if eval_d < self.as_of_date:
            return anomalies  # past eval — bar alignment check handles per-horizon

        # eval_date == today or in future: no forward return can legitimately exist
        for n in FORWARD_HORIZONS:
            val = _get_fwd(snap, n)
            if val is not None:
                anomalies.append(_make_anomaly(
                    anomaly_type=ANOMALY_FUTURE_LEAKAGE,
                    severity=SEV_CRITICAL,
                    symbol=_get_attr(snap, "symbol", "?"),
                    eval_date=str(_get_attr(snap, "eval_date", "")),
                    detail=(
                        f"forward_{n}d={val:.6f} populated but eval_date={eval_d} "
                        f">= as_of_date={self.as_of_date}; future data inaccessible"
                    ),
                    context={
                        "horizon": n,
                        "value": val,
                        "eval_date": str(eval_d),
                        "as_of_date": str(self.as_of_date),
                    },
                ))
        return anomalies

    def _check_bar_alignment(self, snap: Any) -> List[AnomalyRecord]:
        """CRITICAL: forward_Nd populated before the Nth trading bar exists."""
        anomalies: List[AnomalyRecord] = []
        eval_d = _parse_date(_get_attr(snap, "eval_date", ""))
        if eval_d is None or eval_d >= self.as_of_date:
            return anomalies  # handled by _check_future_leakage

        for n in FORWARD_HORIZONS:
            val = _get_fwd(snap, n)
            if val is None:
                continue
            nth_bar = self.calendar.nth_trading_day_after(eval_d, n)
            if nth_bar > self.as_of_date:
                anomalies.append(_make_anomaly(
                    anomaly_type=ANOMALY_FUTURE_BAR_MISALIGNMENT,
                    severity=SEV_CRITICAL,
                    symbol=_get_attr(snap, "symbol", "?"),
                    eval_date=str(_get_attr(snap, "eval_date", "")),
                    detail=(
                        f"forward_{n}d={val:.6f} but {n}th trading bar "
                        f"({nth_bar}) > as_of_date ({self.as_of_date})"
                    ),
                    context={
                        "horizon": n,
                        "bar_date": str(nth_bar),
                        "as_of_date": str(self.as_of_date),
                        "value": val,
                    },
                ))
        return anomalies

    def _check_timezone(self, snap: Any) -> List[AnomalyRecord]:
        """WARNING: timestamp_jst not in JST (+09:00) format."""
        ts = _get_attr(snap, "timestamp_jst", "")
        if not ts:
            return []
        if not _JST_RE.match(str(ts)):
            return [_make_anomaly(
                anomaly_type=ANOMALY_TIMEZONE_INCONSISTENCY,
                severity=SEV_WARNING,
                symbol=_get_attr(snap, "symbol", "?"),
                eval_date=str(_get_attr(snap, "eval_date", "")),
                detail=f"timestamp_jst={ts!r} not in JST (+09:00) format",
                context={"timestamp_jst": ts},
            )]
        return []

    def _check_eval_date_trading_day(self, snap: Any) -> List[AnomalyRecord]:
        """WARNING: eval_date falls on a non-trading day."""
        eval_d = _parse_date(_get_attr(snap, "eval_date", ""))
        if eval_d is None:
            return []
        if not self.calendar.is_trading_day(eval_d):
            kind = "weekend" if eval_d.weekday() >= 5 else "holiday"
            return [_make_anomaly(
                anomaly_type=ANOMALY_NON_TRADING_DAY,
                severity=SEV_WARNING,
                symbol=_get_attr(snap, "symbol", "?"),
                eval_date=str(eval_d),
                detail=f"eval_date={eval_d} is a {kind} (non-trading day)",
                context={"eval_date": str(eval_d), "kind": kind},
            )]
        return []

    def _check_missing_bars(self, snap: Any) -> List[AnomalyRecord]:
        """WARNING: expected forward_Nd to be populated by now, but it's None."""
        anomalies: List[AnomalyRecord] = []
        enrichment = _get_attr(snap, "enrichment_status", "pending")
        if enrichment != "enriched":
            return anomalies

        eval_d = _parse_date(_get_attr(snap, "eval_date", ""))
        if eval_d is None:
            return anomalies

        elapsed = (self.as_of_date - eval_d).days
        for n in FORWARD_HORIZONS:
            min_days = _HORIZON_MIN_CALENDAR_DAYS[n]
            if elapsed < min_days:
                continue
            val = _get_fwd(snap, n)
            if val is None:
                anomalies.append(_make_anomaly(
                    anomaly_type=ANOMALY_MISSING_FUTURE_BARS,
                    severity=SEV_WARNING,
                    symbol=_get_attr(snap, "symbol", "?"),
                    eval_date=str(_get_attr(snap, "eval_date", "")),
                    detail=(
                        f"forward_{n}d is None but enrichment_status=enriched "
                        f"and {elapsed}d have elapsed (min={min_days}d)"
                    ),
                    context={"horizon": n, "elapsed_days": elapsed, "min_days": min_days},
                ))
        return anomalies

    def _check_incomplete_window(self, snap: Any) -> List[AnomalyRecord]:
        """WARNING: enriched snapshot has an inconsistent forward return pattern."""
        enrichment = _get_attr(snap, "enrichment_status", "pending")
        if enrichment != "enriched":
            return []

        # If a longer horizon is filled, shorter horizons must also be filled.
        # e.g. forward_5d set → forward_1d and forward_3d must be set
        populated = {n for n in FORWARD_HORIZONS if _get_fwd(snap, n) is not None}
        if not populated:
            return []

        max_horizon = max(populated)
        anomalies: List[AnomalyRecord] = []
        for n in FORWARD_HORIZONS:
            if n >= max_horizon:
                break
            if n not in populated:
                anomalies.append(_make_anomaly(
                    anomaly_type=ANOMALY_INCOMPLETE_WINDOW,
                    severity=SEV_WARNING,
                    symbol=_get_attr(snap, "symbol", "?"),
                    eval_date=str(_get_attr(snap, "eval_date", "")),
                    detail=(
                        f"forward_{n}d=None but forward_{max_horizon}d is set "
                        f"— incomplete attribution window"
                    ),
                    context={"missing_horizon": n, "max_populated": max_horizon},
                ))
        return anomalies

    def _check_holiday_gap(self, snap: Any) -> List[AnomalyRecord]:
        """INFO: forward window spans a market holiday (verify trading-day alignment)."""
        eval_d = _parse_date(_get_attr(snap, "eval_date", ""))
        if eval_d is None:
            return []

        anomalies: List[AnomalyRecord] = []
        for n in FORWARD_HORIZONS:
            val = _get_fwd(snap, n)
            if val is None:
                continue
            nth_bar = self.calendar.nth_trading_day_after(eval_d, n)
            holidays = self.calendar.holidays_in_range(eval_d + timedelta(days=1), nth_bar)
            if holidays:
                holiday_strs = [str(h) for h in holidays[:3]]
                anomalies.append(_make_anomaly(
                    anomaly_type=ANOMALY_HOLIDAY_GAP,
                    severity=SEV_INFO,
                    symbol=_get_attr(snap, "symbol", "?"),
                    eval_date=str(_get_attr(snap, "eval_date", "")),
                    detail=(
                        f"forward_{n}d window spans {len(holidays)} holiday(s) "
                        f"({', '.join(holiday_strs)}); verify trading-day alignment"
                    ),
                    context={
                        "horizon": n,
                        "holidays": [str(h) for h in holidays],
                        "nth_bar_date": str(nth_bar),
                    },
                ))
        return anomalies

    # ── Report builder ────────────────────────────────────────────────────────

    def _build_report(
        self, snapshots: List[Any], anomalies: List[AnomalyRecord]
    ) -> ValidationReport:
        critical = sum(1 for a in anomalies if a.severity == SEV_CRITICAL)
        warnings = sum(1 for a in anomalies if a.severity == SEV_WARNING)
        infos    = sum(1 for a in anomalies if a.severity == SEV_INFO)

        # A snapshot is "valid" if it has no CRITICAL or WARNING anomalies
        flagged_snap_ids: Set[str] = set()
        for a in anomalies:
            if a.severity in (SEV_CRITICAL, SEV_WARNING):
                # Identify which snapshot triggered this anomaly
                for s in snapshots:
                    if _get_attr(s, "eval_date", "")[:10] == a.eval_date[:10] and \
                       _get_attr(s, "symbol", "") == a.symbol:
                        sid = _get_attr(s, "snapshot_id", "")
                        if sid:
                            flagged_snap_ids.add(sid)
                        break

        total = len(snapshots)
        valid = total - len(flagged_snap_ids)

        by_type: Dict[str, int] = {}
        for a in anomalies:
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1

        if critical > 0:
            verdict = "FAIL"
        elif warnings > 0:
            verdict = "WARN"
        else:
            verdict = "PASS"

        return ValidationReport(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            generated_at_jst=datetime.now(JST).isoformat(),
            validation_mode="strict" if self.strict else "audit",
            as_of_date=str(self.as_of_date),
            record_count=total,
            valid_count=max(0, valid),
            anomaly_count=len(anomalies),
            critical_count=critical,
            warning_count=warnings,
            info_count=infos,
            pass_rate=round(max(0, valid) / total, 4) if total > 0 else 1.0,
            anomalies=anomalies,
            summary_by_type=by_type,
            integrity_verdict=verdict,
            replay_hash=_replay_hash(snapshots, anomalies),
        )

    def _verify_replay(
        self, report: ValidationReport, replay_report_path: Path
    ) -> None:
        """Verify current report hash matches stored replay report."""
        try:
            stored = json.loads(replay_report_path.read_text(encoding="utf-8"))
            stored_hash = stored.get("replay_hash", "")
            if stored_hash and stored_hash != report.replay_hash:
                logger.warning(
                    "Replay hash mismatch: stored=%s current=%s",
                    stored_hash, report.replay_hash,
                )
        except Exception as exc:
            logger.debug("Replay verify failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Global (cross-snapshot) checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_duplicates(snapshots: List[Any]) -> List[AnomalyRecord]:
    """CRITICAL: two snapshots share (symbol, eval_date, executed)."""
    seen: Dict[Tuple[str, str, bool], str] = {}
    anomalies: List[AnomalyRecord] = []
    for snap in snapshots:
        symbol   = str(_get_attr(snap, "symbol", ""))
        eval_date = str(_get_attr(snap, "eval_date", ""))[:10]
        executed = bool(_get_attr(snap, "executed", False))
        key = (symbol, eval_date, executed)
        sid = _get_attr(snap, "snapshot_id", "")
        if key in seen:
            anomalies.append(_make_anomaly(
                anomaly_type=ANOMALY_DUPLICATE_TIMESTAMP,
                severity=SEV_CRITICAL,
                symbol=symbol,
                eval_date=eval_date,
                detail=(
                    f"duplicate (symbol={symbol}, eval_date={eval_date}, "
                    f"executed={executed}): ids={seen[key]!r} and {sid!r}"
                ),
                context={"first_id": seen[key], "duplicate_id": sid, "executed": executed},
            ))
        else:
            seen[key] = sid
    return anomalies


def _check_split_drift(
    snapshots: List[Any],
    origin_symbol: Optional[str] = None,
) -> List[AnomalyRecord]:
    """WARNING: consecutive entry_prices for same symbol imply a split event."""
    anomalies: List[AnomalyRecord] = []

    # Group by symbol
    by_symbol: Dict[str, List[Any]] = {}
    for snap in snapshots:
        sym = str(_get_attr(snap, "symbol", ""))
        if origin_symbol and sym != origin_symbol:
            continue
        by_symbol.setdefault(sym, []).append(snap)

    for sym, snaps in by_symbol.items():
        # Sort by eval_date ascending
        ordered = sorted(
            snaps,
            key=lambda s: str(_get_attr(s, "eval_date", "")),
        )
        for i in range(1, len(ordered)):
            prev_price = float(_get_attr(ordered[i - 1], "entry_price", 0) or 0)
            curr_price = float(_get_attr(ordered[i], "entry_price", 0) or 0)
            if prev_price <= 0 or curr_price <= 0:
                continue
            ratio = curr_price / prev_price
            if ratio > _SPLIT_HIGH or ratio < _SPLIT_LOW:
                prev_date = str(_get_attr(ordered[i - 1], "eval_date", ""))[:10]
                curr_date = str(_get_attr(ordered[i], "eval_date", ""))[:10]
                anomalies.append(_make_anomaly(
                    anomaly_type=ANOMALY_SPLIT_ADJUSTMENT_DRIFT,
                    severity=SEV_WARNING,
                    symbol=sym,
                    eval_date=curr_date,
                    detail=(
                        f"entry_price ratio={ratio:.3f} between "
                        f"{prev_date}({prev_price}) and {curr_date}({curr_price}) "
                        f"— possible unadjusted split"
                    ),
                    context={
                        "prev_date": prev_date,
                        "curr_date": curr_date,
                        "prev_price": prev_price,
                        "curr_price": curr_price,
                        "ratio": round(ratio, 4),
                    },
                ))
    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly log persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_anomaly_log(anomalies: List[AnomalyRecord], log_path: Path) -> None:
    """Append anomalies to a JSONL file. Atomic per-line. Fail-open."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            for a in anomalies:
                f.write(json.dumps(a.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("Anomaly log write failed: %s", exc)


def load_anomaly_log(log_path: Path) -> List[AnomalyRecord]:
    """Load anomaly records from a JSONL file."""
    if not log_path.exists():
        return []
    records: List[AnomalyRecord] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(AnomalyRecord(
                anomaly_id=d.get("anomaly_id", ""),
                anomaly_type=d.get("anomaly_type", ""),
                severity=d.get("severity", ""),
                symbol=d.get("symbol", ""),
                eval_date=d.get("eval_date", ""),
                detail=d.get("detail", ""),
                context=d.get("context", {}),
                detected_at=d.get("detected_at", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("Skipping corrupt anomaly record: %s", exc)
    return records
