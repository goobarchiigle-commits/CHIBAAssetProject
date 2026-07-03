"""
src/analytics/exit_intelligence/observation.py — Observation layer

Append-only JSONL stores for the 4 observation record types.
All writes are atomic (temp+fsync+replace). Immutable records.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    ExitObservationRecord,
    ExitPathSnapshot,
    ExecutionExitObservation,
    PortfolioExitPressureSnapshot,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


def _atomic_append(path: Path, record: dict) -> None:
    """Atomically append one JSON line to path. fsync for durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    # Write to temp file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            # Copy existing content if file exists
            if path.exists():
                fh.write(path.read_text(encoding="utf-8"))
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class ExitObservationStore:
    """Append-only store for ExitObservationRecord."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: ExitObservationRecord) -> None:
        if not record.generated_at:
            record.generated_at = _now_jst()
        _atomic_append(self._path, record.to_dict())
        logger.debug("[EXIT_OBS] appended obs_id=%s symbol=%s", record.obs_id, record.symbol)

    def load_all(self) -> List[ExitObservationRecord]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(ExitObservationRecord.from_dict(d))
            except Exception as exc:
                logger.warning("[EXIT_OBS] parse error: %s", exc)
        return records

    def load_by_symbol(self, symbol: str) -> List[ExitObservationRecord]:
        return [r for r in self.load_all() if r.symbol == symbol]


class ExitPathSnapshotStore:
    """Append-only store for ExitPathSnapshot."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, snapshot: ExitPathSnapshot) -> None:
        if not snapshot.generated_at:
            snapshot.generated_at = _now_jst()
        _atomic_append(self._path, snapshot.to_dict())
        logger.debug("[EXIT_PATH] appended snapshot_id=%s symbol=%s day=%d",
                     snapshot.snapshot_id, snapshot.symbol, snapshot.day_number)

    def load_all(self) -> List[ExitPathSnapshot]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(ExitPathSnapshot.from_dict(d))
            except Exception as exc:
                logger.warning("[EXIT_PATH] parse error: %s", exc)
        return records

    def load_by_symbol(self, symbol: str) -> List[ExitPathSnapshot]:
        return sorted(
            [r for r in self.load_all() if r.symbol == symbol],
            key=lambda s: (s.entry_date, s.day_number),
        )

    def load_open_position(self, symbol: str, entry_date: str) -> List[ExitPathSnapshot]:
        return sorted(
            [r for r in self.load_all()
             if r.symbol == symbol and r.entry_date == entry_date],
            key=lambda s: s.day_number,
        )


class ExecutionExitObservationStore:
    """Append-only store for ExecutionExitObservation."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: ExecutionExitObservation) -> None:
        if not record.generated_at:
            record.generated_at = _now_jst()
        _atomic_append(self._path, record.to_dict())
        logger.debug("[EXEC_EXIT_OBS] appended exec_obs_id=%s symbol=%s",
                     record.exec_obs_id, record.symbol)

    def load_all(self) -> List[ExecutionExitObservation]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(ExecutionExitObservation.from_dict(d))
            except Exception as exc:
                logger.warning("[EXEC_EXIT_OBS] parse error: %s", exc)
        return records


class PortfolioExitPressureStore:
    """Append-only store for PortfolioExitPressureSnapshot."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, snapshot: PortfolioExitPressureSnapshot) -> None:
        if not snapshot.generated_at:
            snapshot.generated_at = _now_jst()
        _atomic_append(self._path, snapshot.to_dict())
        logger.debug("[PORT_PRESSURE] appended pressure_id=%s date=%s",
                     snapshot.pressure_id, snapshot.snapshot_date)

    def load_all(self) -> List[PortfolioExitPressureSnapshot]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(PortfolioExitPressureSnapshot.from_dict(d))
            except Exception as exc:
                logger.warning("[PORT_PRESSURE] parse error: %s", exc)
        return records

    def load_latest(self) -> Optional[PortfolioExitPressureSnapshot]:
        records = self.load_all()
        if not records:
            return None
        return max(records, key=lambda r: r.snapshot_date)


def build_exit_path_snapshot(
    symbol: str,
    entry_date: str,
    snapshot_date: str,
    day_number: int,
    close_price: float,
    entry_price: float,
    peak_price_so_far: float,
    rs_rank: float,
    atr_pct: float,
    breadth_score: float,
    regime: str,
    trend_persistence: float,
    momentum: float,
    momentum_prev: float,
    volume_ratio: float,
    volatility_compression: float,
    upper_shadow_ratio: float,
    exit_signal_count: int,
) -> ExitPathSnapshot:
    """Construct ExitPathSnapshot from raw market fields."""
    unrealized_pnl_fraction = (close_price - entry_price) / entry_price if entry_price > 0 else 0.0
    unrealized_pnl_jpy = unrealized_pnl_fraction  # caller should multiply by position_value

    distance_from_peak = (
        (peak_price_so_far - close_price) / peak_price_so_far
        if peak_price_so_far > 0 else 0.0
    )
    prev_unr = 0.0  # velocity requires prior snapshot; caller can override
    deterioration_velocity = (unrealized_pnl_fraction - prev_unr) / max(day_number, 1)

    return ExitPathSnapshot(
        snapshot_id=ExitPathSnapshot.make_snapshot_id(symbol, entry_date, snapshot_date),
        symbol=symbol,
        entry_date=entry_date,
        snapshot_date=snapshot_date,
        day_number=day_number,
        close_price=close_price,
        entry_price=entry_price,
        unrealized_pnl_fraction=unrealized_pnl_fraction,
        unrealized_pnl_jpy=unrealized_pnl_jpy,
        peak_price_so_far=peak_price_so_far,
        distance_from_peak_pct=distance_from_peak,
        rs_rank=rs_rank,
        atr_pct=atr_pct,
        breadth_score=breadth_score,
        regime=regime,
        trend_persistence=trend_persistence,
        momentum=momentum,
        momentum_prev=momentum_prev,
        volume_ratio=volume_ratio,
        volatility_compression=volatility_compression,
        upper_shadow_ratio=upper_shadow_ratio,
        exit_signal_count=exit_signal_count,
        deterioration_velocity=deterioration_velocity,
    )
