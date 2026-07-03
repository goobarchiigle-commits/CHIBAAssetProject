"""
skipped_trade_outcomes.py — Append-only enrichment for allocator-skipped trades

Attribution for trades filtered by Phase 5A allocation layer:
  - alloc_cap: effective-N or concentration cap reduced qty to 0
  - concentration_throttle: portfolio concentration throttle zeroed qty
  - truth_confidence: truth-layer gate rejected execution

Enrichment computes counterfactual outcome (what would have happened).
All enrichment is deterministic and replay-safe (pure function, price list input).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

SKIP_ALLOC_CAP: str = "alloc_cap"
SKIP_CONCENTRATION_THROTTLE: str = "concentration_throttle"
SKIP_TRUTH_CONFIDENCE: str = "truth_confidence"

VALID_SKIP_REASONS: frozenset[str] = frozenset({
    SKIP_ALLOC_CAP,
    SKIP_CONCENTRATION_THROTTLE,
    SKIP_TRUTH_CONFIDENCE,
})


@dataclass
class SkippedTradeRecord:
    """Pre-enrichment record written at skip time. Immutable once written."""
    timestamp: str
    symbol: str
    signal_id: str
    strategy_id: str
    skip_reason: str
    requested_size: int
    alloc_cap: int
    truth_confidence: float
    concentration_at_skip: float
    estimated_entry_price: float
    schema_version: int = SCHEMA_VERSION


@dataclass
class SkippedTradeOutcome:
    """Enriched outcome record. Produced from SkippedTradeRecord + price data."""
    timestamp: str
    symbol: str
    signal_id: str
    strategy_id: str
    skip_reason: str
    requested_size: int
    alloc_cap: int
    truth_confidence: float
    concentration_at_skip: float
    estimated_entry_price: float
    forward_return_1d: Optional[float]
    forward_return_3d: Optional[float]
    forward_return_5d: Optional[float]
    mfe: Optional[float]
    mae: Optional[float]
    enriched_at: Optional[str]
    schema_version: int = SCHEMA_VERSION


def append_skipped_trade(record: SkippedTradeRecord, path: Path) -> None:
    """Fail-safe append. Write failure logs WARNING, never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[SKIPPED_TRADE] write failed (%s) — execution continues", exc)


def append_skipped_outcome(outcome: SkippedTradeOutcome, path: Path) -> None:
    """Fail-safe append. Write failure logs WARNING, never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(outcome), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[SKIPPED_OUTCOME] write failed (%s) — execution continues", exc)


def load_skipped_trades(path: Path) -> list[SkippedTradeRecord]:
    """Load all SkippedTradeRecords from JSONL. Corrupted lines are skipped with WARNING."""
    if not path.exists():
        return []
    records: list[SkippedTradeRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(SkippedTradeRecord(
                timestamp=d.get("timestamp", ""),
                symbol=d.get("symbol", ""),
                signal_id=d.get("signal_id", ""),
                strategy_id=d.get("strategy_id", ""),
                skip_reason=d.get("skip_reason", ""),
                requested_size=int(d.get("requested_size", 0)),
                alloc_cap=int(d.get("alloc_cap", 0)),
                truth_confidence=float(d.get("truth_confidence", 0.0)),
                concentration_at_skip=float(d.get("concentration_at_skip", 0.0)),
                estimated_entry_price=float(d.get("estimated_entry_price", 0.0)),
            ))
        except Exception as exc:
            logger.warning("[SKIPPED_TRADE] parse error (%s) — line skipped", exc)
    return records


def load_skipped_outcomes(path: Path) -> list[SkippedTradeOutcome]:
    """Load all SkippedTradeOutcomes from JSONL. Corrupted lines are skipped with WARNING."""
    if not path.exists():
        return []
    outcomes: list[SkippedTradeOutcome] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            outcomes.append(SkippedTradeOutcome(
                timestamp=d.get("timestamp", ""),
                symbol=d.get("symbol", ""),
                signal_id=d.get("signal_id", ""),
                strategy_id=d.get("strategy_id", ""),
                skip_reason=d.get("skip_reason", ""),
                requested_size=int(d.get("requested_size", 0)),
                alloc_cap=int(d.get("alloc_cap", 0)),
                truth_confidence=float(d.get("truth_confidence", 0.0)),
                concentration_at_skip=float(d.get("concentration_at_skip", 0.0)),
                estimated_entry_price=float(d.get("estimated_entry_price", 0.0)),
                forward_return_1d=d.get("forward_return_1d"),
                forward_return_3d=d.get("forward_return_3d"),
                forward_return_5d=d.get("forward_return_5d"),
                mfe=d.get("mfe"),
                mae=d.get("mae"),
                enriched_at=d.get("enriched_at"),
            ))
        except Exception as exc:
            logger.warning("[SKIPPED_OUTCOME] parse error (%s) — line skipped", exc)
    return outcomes


def enrich_skipped_trade(
    record: SkippedTradeRecord,
    prices: list[float],
    enriched_at: str,
) -> SkippedTradeOutcome:
    """
    Pure deterministic enrichment. No I/O.

    prices: daily closing prices starting from entry day (index 0).
            prices[0] = entry day, prices[1] = day+1, ...

    Returns:
        forward_return_Nd = (prices[N] - entry) / entry
        mfe = max favorable excursion over the full price window (positive fraction)
        mae = max adverse excursion over the full price window (positive fraction)

    If prices is empty or entry price is zero, all metrics are None.
    """
    entry = record.estimated_entry_price

    def _make_empty() -> SkippedTradeOutcome:
        return SkippedTradeOutcome(
            timestamp=record.timestamp,
            symbol=record.symbol,
            signal_id=record.signal_id,
            strategy_id=record.strategy_id,
            skip_reason=record.skip_reason,
            requested_size=record.requested_size,
            alloc_cap=record.alloc_cap,
            truth_confidence=record.truth_confidence,
            concentration_at_skip=record.concentration_at_skip,
            estimated_entry_price=entry,
            forward_return_1d=None,
            forward_return_3d=None,
            forward_return_5d=None,
            mfe=None,
            mae=None,
            enriched_at=enriched_at,
        )

    if not prices or entry <= 0.0:
        return _make_empty()

    def _ret(idx: int) -> Optional[float]:
        if idx < len(prices):
            return round((prices[idx] - entry) / entry, 6)
        return None

    all_rets = [(p - entry) / entry for p in prices]
    mfe = round(max(all_rets), 6) if all_rets else None
    mae = round(abs(min(all_rets)), 6) if all_rets else None

    return SkippedTradeOutcome(
        timestamp=record.timestamp,
        symbol=record.symbol,
        signal_id=record.signal_id,
        strategy_id=record.strategy_id,
        skip_reason=record.skip_reason,
        requested_size=record.requested_size,
        alloc_cap=record.alloc_cap,
        truth_confidence=record.truth_confidence,
        concentration_at_skip=record.concentration_at_skip,
        estimated_entry_price=entry,
        forward_return_1d=_ret(1),
        forward_return_3d=_ret(3),
        forward_return_5d=_ret(5),
        mfe=mfe,
        mae=mae,
        enriched_at=enriched_at,
    )
