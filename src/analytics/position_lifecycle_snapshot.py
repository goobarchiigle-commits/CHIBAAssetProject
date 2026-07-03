"""
src/analytics/position_lifecycle_snapshot.py — Deterministic position lifecycle observation

Captures daily snapshots of every open position for:
  - exit deterioration analysis
  - hold-duration expectancy analysis
  - profit evaporation analysis

Design invariants:
  - append-only JSONL, no mutable overwrite, no retroactive correction
  - atomic crash-safe write (staging file + dedup guard)
  - schema validation fail-closed before persistence
  - deterministic field order (sort_keys=True)
  - zero lookahead: snapshot_date must not exceed today in JST
  - UTF-8, Windows-compatible, no POSIX assumptions
  - read-only analytics — no execution-path mutation
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION: int = 1

DEFAULT_SNAPSHOT_PATH = Path("runtime/analytics/position_lifecycle_snapshots.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# Schema constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_MARKET_REGIMES = frozenset({
    "trend_persistent",
    "trend_weak",
    "range_bound",
    "bear_trend",
    "high_volatility",
    "low_volatility",
    "unknown",
})

_REQUIRED_FIELDS = (
    "snapshot_date",
    "symbol",
    "entry_date",
    "days_held",
    "quantity",
    "entry_price",
    "close_price",
    "unrealized_pnl_pct",
    "max_unrealized_pnl_pct",
    "distance_from_peak_pct",
    "atr_pct",
    "rs_rank",
    "market_regime",
    "breadth_50",
    "breadth_75",
    "sector",
    "position_volatility_rank",
    "portfolio_heat",
    "cash_ratio",
    "snapshot_timestamp_jst",
    "snapshot_id",
    "schema_version",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _today_jst() -> date:
    return datetime.now(JST).date()


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshot — immutable daily record per open position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionLifecycleSnapshot:
    snapshot_date: str              # YYYY-MM-DD
    symbol: str
    entry_date: str                 # YYYY-MM-DD
    days_held: int
    quantity: int
    entry_price: float
    close_price: float
    unrealized_pnl_pct: float       # (close - entry) / entry
    max_unrealized_pnl_pct: float   # peak unrealized since entry
    distance_from_peak_pct: float   # unrealized_pnl_pct - max_unrealized_pnl_pct (<= 0)
    atr_pct: float                  # ATR / close price
    rs_rank: int                    # 1 = strongest
    market_regime: str
    breadth_50: float               # [0, 1]
    breadth_75: float               # [0, 1]
    sector: str
    position_volatility_rank: float # [0, 1] within portfolio
    portfolio_heat: float           # [0, 1] aggregate risk
    cash_ratio: float               # [0, 1]
    snapshot_timestamp_jst: str     # ISO 8601 +09:00
    snapshot_id: str                # sha256(snapshot_date|symbol|entry_date)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_snapshot_id(cls, snapshot_date: str, symbol: str, entry_date: str) -> str:
        return _sha256(f"{snapshot_date}|{symbol}|{entry_date}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PositionLifecycleSnapshot":
        return cls(
            snapshot_date=d["snapshot_date"],
            symbol=d["symbol"],
            entry_date=d["entry_date"],
            days_held=int(d["days_held"]),
            quantity=int(d["quantity"]),
            entry_price=float(d["entry_price"]),
            close_price=float(d["close_price"]),
            unrealized_pnl_pct=float(d["unrealized_pnl_pct"]),
            max_unrealized_pnl_pct=float(d["max_unrealized_pnl_pct"]),
            distance_from_peak_pct=float(d["distance_from_peak_pct"]),
            atr_pct=float(d["atr_pct"]),
            rs_rank=int(d["rs_rank"]),
            market_regime=d["market_regime"],
            breadth_50=float(d["breadth_50"]),
            breadth_75=float(d["breadth_75"]),
            sector=d["sector"],
            position_volatility_rank=float(d["position_volatility_rank"]),
            portfolio_heat=float(d["portfolio_heat"]),
            cash_ratio=float(d["cash_ratio"]),
            snapshot_timestamp_jst=d["snapshot_timestamp_jst"],
            snapshot_id=d["snapshot_id"],
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshotSchemaValidator — fail-closed before persistence
# ─────────────────────────────────────────────────────────────────────────────

class PositionLifecycleSnapshotSchemaError(Exception):
    pass


class PositionLifecycleSnapshotSchemaValidator:
    """
    Validates a PositionLifecycleSnapshot before persistence.
    Raises PositionLifecycleSnapshotSchemaError on any violation.
    Fail-closed: all checks must pass or the record is rejected.
    """

    def validate(self, snap: PositionLifecycleSnapshot) -> None:
        d = snap.to_dict()
        self._check_required_fields(d)
        self._check_snapshot_date(snap.snapshot_date)
        self._check_entry_date(snap.entry_date, snap.snapshot_date)
        self._check_days_held(snap.days_held, snap.entry_date, snap.snapshot_date)
        self._check_positive_int("quantity", snap.quantity)
        self._check_positive_float("entry_price", snap.entry_price)
        self._check_positive_float("close_price", snap.close_price)
        self._check_finite_float("unrealized_pnl_pct", snap.unrealized_pnl_pct)
        self._check_finite_float("max_unrealized_pnl_pct", snap.max_unrealized_pnl_pct)
        self._check_distance_from_peak(snap.distance_from_peak_pct, snap.unrealized_pnl_pct, snap.max_unrealized_pnl_pct)
        self._check_positive_float("atr_pct", snap.atr_pct)
        self._check_positive_int("rs_rank", snap.rs_rank)
        self._check_market_regime(snap.market_regime)
        self._check_unit_float("breadth_50", snap.breadth_50)
        self._check_unit_float("breadth_75", snap.breadth_75)
        self._check_nonempty_str("sector", snap.sector)
        self._check_unit_float("position_volatility_rank", snap.position_volatility_rank)
        self._check_unit_float("portfolio_heat", snap.portfolio_heat)
        self._check_unit_float("cash_ratio", snap.cash_ratio)
        self._check_timestamp_jst(snap.snapshot_timestamp_jst, snap.snapshot_date)
        self._check_snapshot_id(snap.snapshot_id, snap.snapshot_date, snap.symbol, snap.entry_date)
        self._check_no_lookahead(snap.snapshot_date)

    def _check_required_fields(self, d: dict) -> None:
        missing = [f for f in _REQUIRED_FIELDS if f not in d]
        if missing:
            raise PositionLifecycleSnapshotSchemaError(f"missing fields: {missing}")

    def _check_snapshot_date(self, snapshot_date: str) -> None:
        try:
            _parse_date(snapshot_date)
        except (ValueError, TypeError) as exc:
            raise PositionLifecycleSnapshotSchemaError(f"invalid snapshot_date={snapshot_date!r}: {exc}")

    def _check_entry_date(self, entry_date: str, snapshot_date: str) -> None:
        try:
            ed = _parse_date(entry_date)
            sd = _parse_date(snapshot_date)
        except (ValueError, TypeError) as exc:
            raise PositionLifecycleSnapshotSchemaError(f"invalid entry_date={entry_date!r}: {exc}")
        if ed > sd:
            raise PositionLifecycleSnapshotSchemaError(
                f"entry_date {entry_date} > snapshot_date {snapshot_date}: lookahead violation"
            )

    def _check_days_held(self, days_held: int, entry_date: str, snapshot_date: str) -> None:
        if days_held < 0:
            raise PositionLifecycleSnapshotSchemaError(f"days_held={days_held} < 0")
        try:
            expected = (_parse_date(snapshot_date) - _parse_date(entry_date)).days
        except (ValueError, TypeError):
            return
        if days_held != expected:
            raise PositionLifecycleSnapshotSchemaError(
                f"days_held={days_held} != expected {expected} from dates"
            )

    def _check_positive_int(self, field: str, value) -> None:
        if not isinstance(value, int) or value <= 0:
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} must be positive int")

    def _check_positive_float(self, field: str, value) -> None:
        import math
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} not numeric")
        if math.isnan(v) or math.isinf(v) or v <= 0:
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} must be finite positive")

    def _check_finite_float(self, field: str, value) -> None:
        import math
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} not numeric")
        if math.isnan(v) or math.isinf(v):
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} must be finite")

    def _check_distance_from_peak(self, distance: float, unrealized: float, peak: float) -> None:
        import math
        if math.isnan(distance) or math.isinf(distance):
            raise PositionLifecycleSnapshotSchemaError(
                f"distance_from_peak_pct={distance!r} must be finite"
            )
        if distance > 1e-9:
            raise PositionLifecycleSnapshotSchemaError(
                f"distance_from_peak_pct={distance:.6f} > 0: impossible (current > peak)"
            )
        # max_unrealized_pnl_pct must be >= unrealized_pnl_pct (peak >= current)
        if peak < unrealized - 1e-9:
            raise PositionLifecycleSnapshotSchemaError(
                f"max_unrealized_pnl_pct={peak:.6f} < unrealized_pnl_pct={unrealized:.6f}"
            )

    def _check_unit_float(self, field: str, value) -> None:
        import math
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} not numeric")
        if math.isnan(v) or math.isinf(v) or v < 0.0 or v > 1.0:
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} must be in [0, 1]")

    def _check_market_regime(self, regime: str) -> None:
        if not isinstance(regime, str) or not regime.strip():
            raise PositionLifecycleSnapshotSchemaError(f"market_regime={regime!r} must be non-empty str")

    def _check_nonempty_str(self, field: str, value) -> None:
        if not isinstance(value, str) or not value.strip():
            raise PositionLifecycleSnapshotSchemaError(f"{field}={value!r} must be non-empty str")

    def _check_timestamp_jst(self, ts: str, snapshot_date: str) -> None:
        if not isinstance(ts, str) or not ts.endswith("+09:00"):
            raise PositionLifecycleSnapshotSchemaError(
                f"snapshot_timestamp_jst={ts!r} must end with +09:00"
            )
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError) as exc:
            raise PositionLifecycleSnapshotSchemaError(
                f"snapshot_timestamp_jst={ts!r} not valid ISO 8601: {exc}"
            )
        ts_date = dt.date().isoformat()
        if ts_date != snapshot_date:
            raise PositionLifecycleSnapshotSchemaError(
                f"timestamp date {ts_date} != snapshot_date {snapshot_date}"
            )

    def _check_snapshot_id(self, sid: str, snapshot_date: str, symbol: str, entry_date: str) -> None:
        expected = PositionLifecycleSnapshot.make_snapshot_id(snapshot_date, symbol, entry_date)
        if sid != expected:
            raise PositionLifecycleSnapshotSchemaError(
                f"snapshot_id mismatch: got {sid[:12]}… expected {expected[:12]}…"
            )

    def _check_no_lookahead(self, snapshot_date: str) -> None:
        try:
            sd = _parse_date(snapshot_date)
        except (ValueError, TypeError):
            return
        today = _today_jst()
        if sd > today:
            raise PositionLifecycleSnapshotSchemaError(
                f"snapshot_date {snapshot_date} > today {today.isoformat()}: lookahead contamination"
            )


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshotWriter — crash-safe append-only persistence
# ─────────────────────────────────────────────────────────────────────────────

class PositionLifecycleSnapshotWriter:
    """
    Append-only JSONL writer for PositionLifecycleSnapshot records.

    Guarantees:
      - schema validation fail-closed before any write
      - dedup by snapshot_id (one record per position per day)
      - crash-safe: staging file ensures write intent survives crash
      - UTF-8, sort_keys=True, deterministic serialization
      - no mutable overwrite, no retroactive correction
    """

    def __init__(
        self,
        path: Path = DEFAULT_SNAPSHOT_PATH,
        validator: Optional[PositionLifecycleSnapshotSchemaValidator] = None,
    ) -> None:
        self._path = path
        self._validator = validator or PositionLifecycleSnapshotSchemaValidator()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, snap: PositionLifecycleSnapshot) -> bool:
        """
        Validate and append one snapshot.

        Returns True if written, False if duplicate.
        Raises PositionLifecycleSnapshotSchemaError on validation failure (fail-closed).
        """
        self._validator.validate(snap)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._is_duplicate(snap.snapshot_id):
            logger.debug(
                "[LIFECYCLE_SNAP] duplicate snapshot_id=%s symbol=%s date=%s — skipped",
                snap.snapshot_id[:12],
                snap.symbol,
                snap.snapshot_date,
            )
            return False

        line = _canonical_json(snap.to_dict()) + "\n"
        self._crash_safe_append(snap.snapshot_id, line)
        logger.info(
            "[LIFECYCLE_SNAP] written symbol=%s date=%s days_held=%d unrealized=%.4f",
            snap.symbol,
            snap.snapshot_date,
            snap.days_held,
            snap.unrealized_pnl_pct,
        )
        return True

    def write_batch(self, snapshots: List[PositionLifecycleSnapshot]) -> Tuple[int, int]:
        """Write a batch. Returns (written, skipped)."""
        written = skipped = 0
        for snap in snapshots:
            if self.write(snap):
                written += 1
            else:
                skipped += 1
        return written, skipped

    def _is_duplicate(self, snapshot_id: str) -> bool:
        if not self._path.exists():
            return False
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("snapshot_id") == snapshot_id:
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    def _crash_safe_append(self, snapshot_id: str, line: str) -> None:
        """
        Two-phase append: write staging file first, then append to JSONL.
        Staging file presence + dedup guard enables crash recovery.
        """
        stage = self._path.parent / f".pending_{snapshot_id[:12]}.tmp"
        try:
            stage.write_text(line, encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"[LIFECYCLE_SNAP] staging write failed: {exc}") from exc

        try:
            with self._path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            logger.error(
                "[LIFECYCLE_SNAP] JSONL append failed snapshot_id=%s: %s — staging file retained at %s",
                snapshot_id[:12],
                exc,
                stage,
            )
            raise

        try:
            stage.unlink()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshotLoader — replay-compatible read-only access
# ─────────────────────────────────────────────────────────────────────────────

class PositionLifecycleSnapshotLoader:
    """
    Replay-compatible, read-only loader for position lifecycle snapshots.

    All load operations are deterministic for a fixed JSONL state.
    Corrupt lines are skipped with a warning (fail-open for reads).
    """

    def __init__(self, path: Path = DEFAULT_SNAPSHOT_PATH) -> None:
        self._path = path

    def load_all(self) -> List[PositionLifecycleSnapshot]:
        """Load all snapshots in append order (oldest first)."""
        return self._read_snapshots()

    def load_symbol_timeline(self, symbol: str) -> List[PositionLifecycleSnapshot]:
        """All snapshots for one symbol, sorted by snapshot_date ascending."""
        snaps = [s for s in self._read_snapshots() if s.symbol == symbol]
        return sorted(snaps, key=lambda s: s.snapshot_date)

    def load_by_date(self, snapshot_date: str) -> List[PositionLifecycleSnapshot]:
        """All snapshots for a single date."""
        return [s for s in self._read_snapshots() if s.snapshot_date == snapshot_date]

    def load_by_date_range(self, start_date: str, end_date: str) -> List[PositionLifecycleSnapshot]:
        """All snapshots with snapshot_date in [start_date, end_date] inclusive."""
        return [
            s for s in self._read_snapshots()
            if start_date <= s.snapshot_date <= end_date
        ]

    def load_open_positions(self, snapshot_date: str) -> List[PositionLifecycleSnapshot]:
        """All positions open on a given date."""
        return self.load_by_date(snapshot_date)

    def _read_snapshots(self) -> List[PositionLifecycleSnapshot]:
        if not self._path.exists():
            return []
        snaps: List[PositionLifecycleSnapshot] = []
        try:
            text = self._path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("[LIFECYCLE_SNAP] read failed path=%s: %s", self._path, exc)
            return []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                snaps.append(PositionLifecycleSnapshot.from_dict(d))
            except Exception as exc:
                logger.warning(
                    "[LIFECYCLE_SNAP] corrupt line %d skipped: %s", i, exc
                )
        return snaps


# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers — pure functions, no side effects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HoldingDurationAnalytics:
    symbol: str
    total_snapshots: int
    min_days_held: int
    max_days_held: int
    mean_days_held: float
    days_held_distribution: Dict[int, int]   # days_held -> count
    unrealized_pnl_at_exit_proxy: Optional[float]  # last snapshot unrealized_pnl_pct
    peak_unrealized_pnl_pct: float
    final_distance_from_peak_pct: float       # last snapshot's distance


@dataclass
class PeakDeteriorationAnalytics:
    symbol: str
    entry_date: str
    snapshot_dates: List[str]
    unrealized_pnl_series: List[float]
    max_unrealized_pnl_series: List[float]   # running peak, monotone non-decreasing
    distance_from_peak_series: List[float]
    peak_snapshot_date: Optional[str]
    peak_unrealized_pnl_pct: float
    current_unrealized_pnl_pct: float
    deterioration_days: int                  # days since peak
    deterioration_rate_per_day: float        # pct/day decay since peak


def compute_holding_duration_analytics(
    timeline: List[PositionLifecycleSnapshot],
) -> Optional[HoldingDurationAnalytics]:
    """
    Summarise hold-duration statistics from a symbol's snapshot timeline.
    Returns None if timeline is empty.
    Timeline must be for a single symbol.
    """
    if not timeline:
        return None

    symbol = timeline[0].symbol
    days_vals = [s.days_held for s in timeline]
    dist: Dict[int, int] = {}
    for d in days_vals:
        dist[d] = dist.get(d, 0) + 1

    peak = max(s.max_unrealized_pnl_pct for s in timeline)
    last = timeline[-1]
    total = len(timeline)

    return HoldingDurationAnalytics(
        symbol=symbol,
        total_snapshots=total,
        min_days_held=min(days_vals),
        max_days_held=max(days_vals),
        mean_days_held=sum(days_vals) / total,
        days_held_distribution=dist,
        unrealized_pnl_at_exit_proxy=last.unrealized_pnl_pct,
        peak_unrealized_pnl_pct=peak,
        final_distance_from_peak_pct=last.distance_from_peak_pct,
    )


def compute_peak_to_current_deterioration(
    timeline: List[PositionLifecycleSnapshot],
) -> Optional[PeakDeteriorationAnalytics]:
    """
    Compute peak-to-current deterioration curve from a symbol's snapshot timeline.
    Returns None if timeline is empty.
    Timeline must be sorted by snapshot_date ascending (as returned by load_symbol_timeline).
    """
    if not timeline:
        return None

    symbol = timeline[0].symbol
    entry_date = timeline[0].entry_date

    snapshot_dates = [s.snapshot_date for s in timeline]
    unrealized_series = [s.unrealized_pnl_pct for s in timeline]
    distance_series = [s.distance_from_peak_pct for s in timeline]

    # Reconstruct monotone-non-decreasing max_unrealized series
    running_peak = timeline[0].max_unrealized_pnl_pct
    max_series: List[float] = []
    for s in timeline:
        running_peak = max(running_peak, s.unrealized_pnl_pct)
        max_series.append(running_peak)

    global_peak = max(max_series)
    peak_idx = next(
        (i for i, v in enumerate(max_series) if v >= global_peak - 1e-9),
        len(timeline) - 1,
    )
    peak_date: Optional[str] = snapshot_dates[peak_idx] if snapshot_dates else None
    current_unrealized = unrealized_series[-1]

    # Days since peak
    deterioration_days = len(timeline) - 1 - peak_idx

    # Average daily decay rate since peak (fraction / day)
    if deterioration_days > 0:
        total_decay = global_peak - current_unrealized
        rate = total_decay / deterioration_days
    else:
        rate = 0.0

    return PeakDeteriorationAnalytics(
        symbol=symbol,
        entry_date=entry_date,
        snapshot_dates=snapshot_dates,
        unrealized_pnl_series=unrealized_series,
        max_unrealized_pnl_series=max_series,
        distance_from_peak_series=distance_series,
        peak_snapshot_date=peak_date,
        peak_unrealized_pnl_pct=global_peak,
        current_unrealized_pnl_pct=current_unrealized,
        deterioration_days=deterioration_days,
        deterioration_rate_per_day=rate,
    )


def build_snapshot(
    *,
    snapshot_date: str,
    symbol: str,
    entry_date: str,
    quantity: int,
    entry_price: float,
    close_price: float,
    max_unrealized_pnl_pct: float,
    atr_pct: float,
    rs_rank: int,
    market_regime: str,
    breadth_50: float,
    breadth_75: float,
    sector: str,
    position_volatility_rank: float,
    portfolio_heat: float,
    cash_ratio: float,
    snapshot_timestamp_jst: Optional[str] = None,
) -> PositionLifecycleSnapshot:
    """
    Construct a PositionLifecycleSnapshot with derived fields computed automatically.
    snapshot_timestamp_jst defaults to now in JST if not supplied.
    """
    ed = _parse_date(entry_date)
    sd = _parse_date(snapshot_date)
    days_held = (sd - ed).days

    unrealized_pnl_pct = (close_price - entry_price) / entry_price
    distance_from_peak_pct = unrealized_pnl_pct - max_unrealized_pnl_pct

    if snapshot_timestamp_jst is None:
        snapshot_timestamp_jst = datetime.now(JST).isoformat(timespec="seconds")

    snapshot_id = PositionLifecycleSnapshot.make_snapshot_id(snapshot_date, symbol, entry_date)

    return PositionLifecycleSnapshot(
        snapshot_date=snapshot_date,
        symbol=symbol,
        entry_date=entry_date,
        days_held=days_held,
        quantity=quantity,
        entry_price=entry_price,
        close_price=close_price,
        unrealized_pnl_pct=round(unrealized_pnl_pct, 6),
        max_unrealized_pnl_pct=round(max_unrealized_pnl_pct, 6),
        distance_from_peak_pct=round(distance_from_peak_pct, 6),
        atr_pct=atr_pct,
        rs_rank=rs_rank,
        market_regime=market_regime,
        breadth_50=breadth_50,
        breadth_75=breadth_75,
        sector=sector,
        position_volatility_rank=position_volatility_rank,
        portfolio_heat=portfolio_heat,
        cash_ratio=cash_ratio,
        snapshot_timestamp_jst=snapshot_timestamp_jst,
        snapshot_id=snapshot_id,
    )
