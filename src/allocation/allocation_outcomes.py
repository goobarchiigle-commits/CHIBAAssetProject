"""
allocation_outcomes.py — Append-only allocation outcome tracking

Records the delta between intended and adjusted allocation at order time.
Designed for later enrichment with realized pnl, slippage, holding_days, exit_reason.
Enrichment always produces a new record — never modifies in place.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


@dataclass
class AllocationOutcomeRecord:
    """
    Allocation outcome record. Written at order time; None fields populated later.
    Append-only: enrichment produces a new record, never mutates existing.
    """
    timestamp: str
    symbol: str
    signal_id: str
    strategy_id: str
    intended_qty: int
    adjusted_qty: int
    throttle_reason: str
    truth_confidence: float
    alloc_cap: int
    effective_n: float
    sector_multiplier: float
    realized_pnl: Optional[float]
    realized_slippage: Optional[float]
    holding_days: Optional[int]
    exit_reason: Optional[str]
    schema_version: int = SCHEMA_VERSION


def build_outcome_record(
    timestamp: str,
    symbol: str,
    signal_id: str,
    strategy_id: str,
    intended_qty: int,
    adjusted_qty: int,
    throttle_reason: str,
    truth_confidence: float,
    alloc_cap: int,
    effective_n: float,
    sector_multiplier: float = 1.0,
) -> AllocationOutcomeRecord:
    """Convenience constructor. Enrichable fields default to None."""
    return AllocationOutcomeRecord(
        timestamp=timestamp,
        symbol=symbol,
        signal_id=signal_id,
        strategy_id=strategy_id,
        intended_qty=intended_qty,
        adjusted_qty=adjusted_qty,
        throttle_reason=throttle_reason,
        truth_confidence=truth_confidence,
        alloc_cap=alloc_cap,
        effective_n=effective_n,
        sector_multiplier=sector_multiplier,
        realized_pnl=None,
        realized_slippage=None,
        holding_days=None,
        exit_reason=None,
    )


def enrich_outcome_record(
    record: AllocationOutcomeRecord,
    realized_pnl: Optional[float] = None,
    realized_slippage: Optional[float] = None,
    holding_days: Optional[int] = None,
    exit_reason: Optional[str] = None,
) -> AllocationOutcomeRecord:
    """
    Returns a new enriched record. Original record is unchanged.
    Caller appends the new record to an enriched JSONL; never replaces lines.
    """
    return AllocationOutcomeRecord(
        timestamp=record.timestamp,
        symbol=record.symbol,
        signal_id=record.signal_id,
        strategy_id=record.strategy_id,
        intended_qty=record.intended_qty,
        adjusted_qty=record.adjusted_qty,
        throttle_reason=record.throttle_reason,
        truth_confidence=record.truth_confidence,
        alloc_cap=record.alloc_cap,
        effective_n=record.effective_n,
        sector_multiplier=record.sector_multiplier,
        realized_pnl=realized_pnl if realized_pnl is not None else record.realized_pnl,
        realized_slippage=(
            realized_slippage
            if realized_slippage is not None
            else record.realized_slippage
        ),
        holding_days=holding_days if holding_days is not None else record.holding_days,
        exit_reason=exit_reason if exit_reason is not None else record.exit_reason,
    )


def append_outcome_record(record: AllocationOutcomeRecord, path: Path) -> None:
    """Fail-safe append. Write failure logs WARNING, never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[ALLOC_OUTCOME] write failed (%s) — execution continues", exc)


def load_outcome_records(path: Path) -> list[AllocationOutcomeRecord]:
    """Load all records from JSONL. Corrupted lines are skipped with WARNING."""
    if not path.exists():
        return []
    records: list[AllocationOutcomeRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(AllocationOutcomeRecord(
                timestamp=d.get("timestamp", ""),
                symbol=d.get("symbol", ""),
                signal_id=d.get("signal_id", ""),
                strategy_id=d.get("strategy_id", ""),
                intended_qty=int(d.get("intended_qty", 0)),
                adjusted_qty=int(d.get("adjusted_qty", 0)),
                throttle_reason=d.get("throttle_reason", ""),
                truth_confidence=float(d.get("truth_confidence", 0.0)),
                alloc_cap=int(d.get("alloc_cap", 0)),
                effective_n=float(d.get("effective_n", 0.0)),
                sector_multiplier=float(d.get("sector_multiplier", 1.0)),
                realized_pnl=d.get("realized_pnl"),
                realized_slippage=d.get("realized_slippage"),
                holding_days=d.get("holding_days"),
                exit_reason=d.get("exit_reason"),
            ))
        except Exception as exc:
            logger.warning("[ALLOC_OUTCOME] parse error (%s) — line skipped", exc)
    return records
